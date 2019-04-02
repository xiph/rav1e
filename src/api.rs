// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use arg_enum_proc_macro::ArgEnum;
use bitstream_io::*;
use crate::encoder::*;
use crate::metrics::calculate_frame_psnr;
use crate::partition::*;
use crate::rate::RCState;
use crate::rate::FRAME_NSUBTYPES;
use crate::rate::FRAME_SUBTYPE_I;
use crate::rate::FRAME_SUBTYPE_P;
use crate::scenechange::SceneChangeDetector;
use crate::util::Pixel;

use std::{cmp, fmt, io};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::collections::BTreeSet;
use std::path::PathBuf;

const LOOKAHEAD_FRAMES: u64 = 10;

// TODO: use the num crate?
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Rational {
  pub num: u64,
  pub den: u64
}

impl Rational {
  pub fn new(num: u64, den: u64) -> Self {
    Rational { num, den }
  }
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Point {
  pub x: u16,
  pub y: u16
}


/// Encoder Settings impacting the bitstream produced
#[derive(Clone, Debug)]
pub struct EncoderConfig {
  // output size
  pub width: usize,
  pub height: usize,

  // data format and ancillary color information
  pub bit_depth: usize,
  pub chroma_sampling: ChromaSampling,
  pub chroma_sample_position: ChromaSamplePosition,
  pub pixel_range: PixelRange,
  pub color_description: Option<ColorDescription>,
  pub mastering_display: Option<MasteringDisplay>,
  pub content_light: Option<ContentLight>,

  // encoder configuration
  pub time_base: Rational,
  /// The *minimum* interval between two keyframes
  pub min_key_frame_interval: u64,
  /// The *maximum* interval between two keyframes
  pub max_key_frame_interval: u64,
  pub low_latency: bool,
  pub quantizer: usize,
  pub bitrate: i32,
  pub tune: Tune,
  pub speed_settings: SpeedSettings,
  /// `None` for one-pass encode. `Some(1)` or `Some(2)` for two-pass encoding.
  pub pass: Option<u8>,
  pub show_psnr: bool,
  pub stats_file: Option<PathBuf>,
  pub train_rdo: bool,
}

impl Default for EncoderConfig {
  fn default() -> Self {
    const DEFAULT_SPEED: usize = 5;
    Self::with_speed_preset(DEFAULT_SPEED)
  }
}

impl EncoderConfig {
  pub fn with_speed_preset(speed: usize) -> Self {
    EncoderConfig {
      width: 640,
      height: 480,

      bit_depth: 8,
      chroma_sampling: ChromaSampling::Cs420,
      chroma_sample_position: ChromaSamplePosition::Unknown,
      pixel_range: PixelRange::Unspecified,
      color_description: None,
      mastering_display: None,
      content_light: None,

      time_base: Rational { num: 30, den: 1 },
      min_key_frame_interval: 12,
      max_key_frame_interval: 240,
      low_latency: false,
      quantizer: 100,
      bitrate: 0,
      tune: Tune::default(),
      speed_settings: SpeedSettings::from_preset(speed),
      pass: None,
      show_psnr: false,
      stats_file: None,
      train_rdo: false
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub struct SpeedSettings {
  pub min_block_size: BlockSize,
  pub multiref: bool,
  pub fast_deblock: bool,
  pub reduced_tx_set: bool,
  pub tx_domain_distortion: bool,
  pub tx_domain_rate: bool,
  pub encode_bottomup: bool,
  pub rdo_tx_decision: bool,
  pub prediction_modes: PredictionModesSetting,
  pub include_near_mvs: bool,
  pub no_scene_detection: bool,
  pub diamond_me: bool,
  pub cdef: bool
}

impl Default for SpeedSettings {
  fn default() -> Self {
    SpeedSettings {
      min_block_size: BlockSize::BLOCK_16X16,
      multiref: false,
      fast_deblock: false,
      reduced_tx_set: false,
      tx_domain_distortion: false,
      tx_domain_rate: false,
      encode_bottomup: false,
      rdo_tx_decision: false,
      prediction_modes: PredictionModesSetting::Simple,
      include_near_mvs: false,
      no_scene_detection: false,
      diamond_me: false,
      cdef: false,
    }
  }
}

impl SpeedSettings {
  pub fn from_preset(speed: usize) -> Self {
    SpeedSettings {
      min_block_size: Self::min_block_size_preset(speed),
      multiref: Self::multiref_preset(speed),
      fast_deblock: Self::fast_deblock_preset(speed),
      reduced_tx_set: Self::reduced_tx_set_preset(speed),
      tx_domain_distortion: Self::tx_domain_distortion_preset(speed),
      tx_domain_rate: Self::tx_domain_rate_preset(speed),
      encode_bottomup: Self::encode_bottomup_preset(speed),
      rdo_tx_decision: Self::rdo_tx_decision_preset(speed),
      prediction_modes: Self::prediction_modes_preset(speed),
      include_near_mvs: Self::include_near_mvs_preset(speed),
      no_scene_detection: Self::no_scene_detection_preset(speed),
      diamond_me: Self::diamond_me_preset(speed),
      cdef: Self::cdef_preset(speed),
    }
  }

  /// This preset is set this way because 8x8 with reduced TX set is faster but with equivalent
  /// or better quality compared to 16x16 or 32x32 (to which reduced TX set does not apply).
  fn min_block_size_preset(speed: usize) -> BlockSize {
    if speed == 0 {
      BlockSize::BLOCK_4X4
    } else if speed <= 8 {
      BlockSize::BLOCK_8X8
    } else {
      BlockSize::BLOCK_64X64
    }
  }

  /// Multiref is enabled automatically if low_latency is false,
  /// but if someone is setting low_latency to true manually,
  /// multiref has a large speed penalty with low quality gain.
  /// Because low_latency can be set manually, this setting is conservative.
  fn multiref_preset(speed: usize) -> bool {
    speed <= 1
  }

  fn fast_deblock_preset(speed: usize) -> bool {
    speed >= 8
  }

  fn reduced_tx_set_preset(speed: usize) -> bool {
    speed >= 5
  }

  /// TX domain distortion is always faster, with no significant quality change
  fn tx_domain_distortion_preset(_speed: usize) -> bool {
    true
  }

  fn tx_domain_rate_preset(_speed: usize) -> bool {
    false
  }

  fn encode_bottomup_preset(speed: usize) -> bool {
    speed == 0
  }

  fn rdo_tx_decision_preset(speed: usize) -> bool {
    speed <= 3
  }

  fn prediction_modes_preset(speed: usize) -> PredictionModesSetting {
    if speed <= 1 {
      PredictionModesSetting::ComplexAll
    } else if speed <= 5 {
      PredictionModesSetting::ComplexKeyframes
    } else {
      PredictionModesSetting::Simple
    }
  }

  fn include_near_mvs_preset(speed: usize) -> bool {
    speed <= 2
  }

  fn no_scene_detection_preset(speed: usize) -> bool {
    speed == 10
  }

  /// Currently Diamond ME gives better quality than full search on most videos,
  /// in addition to being faster.
  /// There are a few outliers, such as the Wikipedia test clip.
  ///
  /// TODO: Revisit this setting if full search quality improves in the future.
  fn diamond_me_preset(_speed: usize) -> bool {
    true
  }

  fn cdef_preset(_speed: usize) -> bool {
    true
  }
}

#[allow(dead_code, non_camel_case_types)]
#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub enum FrameType {
  KEY,
  INTER,
  INTRA_ONLY,
  SWITCH
}

impl fmt::Display for FrameType {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    use self::FrameType::*;
    match self {
      KEY => write!(f, "Key frame"),
      INTER => write!(f, "Inter frame"),
      INTRA_ONLY => write!(f, "Intra only frame"),
      SWITCH => write!(f, "Switching frame"),
    }
  }
}

#[derive(Clone, Copy, Debug, PartialOrd, PartialEq)]
pub enum PredictionModesSetting {
  Simple,
  ComplexKeyframes,
  ComplexAll,
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub enum ChromaSampling {
  Cs420,
  Cs422,
  Cs444,
  Cs400,
}

impl Default for ChromaSampling {
  fn default() -> Self {
    ChromaSampling::Cs420
  }
}

impl ChromaSampling {
  // Provides the sampling period in the horizontal and vertical axes.
  pub fn sampling_period(self) -> (usize, usize) {
    use self::ChromaSampling::*;
    match self {
      Cs420 => (2, 2),
      Cs422 => (2, 1),
      Cs444 => (1, 1),
      Cs400 => (2, 2),
    }
  }
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub enum ChromaSamplePosition {
  Unknown,
  Vertical,
  Colocated
}

impl Default for ChromaSamplePosition {
  fn default() -> Self {
    ChromaSamplePosition::Unknown
  }
}

#[derive(ArgEnum, Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub enum PixelRange {
    Unspecified = 0,
    Limited,
    Full,
}

impl Default for PixelRange {
    fn default() -> Self {
        PixelRange::Unspecified
    }
}

#[derive(ArgEnum, Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub enum MatrixCoefficients {
    Identity = 0,
    BT709,
    Unspecified,
    BT470M = 4,
    BT470BG,
    ST170M,
    ST240M,
    YCgCo,
    BT2020NonConstantLuminance,
    BT2020ConstantLuminance,
    ST2085,
    ChromaticityDerivedNonConstantLuminance,
    ChromaticityDerivedConstantLuminance,
    ICtCp,
}

impl Default for MatrixCoefficients {
    fn default() -> Self {
        MatrixCoefficients::Unspecified
    }
}

#[derive(ArgEnum, Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub enum ColorPrimaries {
    BT709 = 1,
    Unspecified,
    BT470M = 4,
    BT470BG,
    ST170M,
    ST240M,
    Film,
    BT2020,
    ST428,
    P3DCI,
    P3Display,
    Tech3213 = 22,
}

impl Default for ColorPrimaries {
    fn default() -> Self {
        ColorPrimaries::Unspecified
    }
}

#[derive(ArgEnum, Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub enum TransferCharacteristics {
    BT1886 = 1,
    Unspecified,
    BT470M = 4,
    BT470BG,
    ST170M,
    ST240M,
    Linear,
    Logarithmic100,
    Logarithmic316,
    XVYCC,
    BT1361E,
    SRGB,
    BT2020Ten,
    BT2020Twelve,
    PerceptualQuantizer,
    ST428,
    HybridLogGamma,
}

impl Default for TransferCharacteristics {
    fn default() -> Self {
        TransferCharacteristics::Unspecified
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ColorDescription {
    pub color_primaries: ColorPrimaries,
    pub transfer_characteristics: TransferCharacteristics,
    pub matrix_coefficients: MatrixCoefficients
}

#[derive(Copy, Clone, Debug)]
pub struct MasteringDisplay {
    pub primaries: [Point; 3],
    pub white_point: Point,
    pub max_luminance: u32,
    pub min_luminance: u32,
}

#[derive(Copy, Clone, Debug)]
pub struct ContentLight {
    pub max_content_light_level: u16,
    pub max_frame_average_light_level: u16,
}

/// Contains all the encoder configuration
#[derive(Clone, Debug)]
pub struct Config {
  pub enc: EncoderConfig,
  /// The number of threads in the threadpool.
  pub threads: usize
}

const MAX_USABLE_THREADS: usize = 4;

impl Config {
  pub fn new_context<T: Pixel>(&self) -> Context<T> {
    // initialize with temporal delimiter
    let packet_data = TEMPORAL_DELIMITER.to_vec();

    let maybe_ac_qi_max = if self.enc.quantizer < 255 {
      Some(self.enc.quantizer as u8)
    } else {
      None
    };

    let threads = if self.threads == 0 {
      rayon::current_num_threads().min(MAX_USABLE_THREADS)
    } else {
      self.threads
    };

    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();

    Context {
      inner: ContextInner {
        frame_count: 0,
        limit: 0,
        idx: 0,
        frames_processed: 0,
        frame_q: BTreeMap::new(),
        frame_invariants: BTreeMap::new(),
        keyframes: BTreeSet::new(),
        packet_data,
        segment_start_idx: 0,
        segment_start_frame: 0,
        keyframe_detector: SceneChangeDetector::new(self.enc.bit_depth),
        config: self.enc.clone(),
        rc_state: RCState::new(
          self.enc.width as i32,
          self.enc.height as i32,
          self.enc.time_base.num as i64,
          self.enc.time_base.den as i64,
          self.enc.bitrate,
          maybe_ac_qi_max,
          self.enc.max_key_frame_interval as i32
        ),
        maybe_prev_log_base_q: None,
        first_pass_data: FirstPassData { frames: Vec::new() },
        pool
      },
      config: self.enc.clone()
    }
  }
}

pub struct ContextInner<T: Pixel> {
  frame_count: u64,
  limit: u64,
  pub(crate) idx: u64,
  frames_processed: u64,
  /// Maps frame *number* to frames
  frame_q: BTreeMap<u64, Option<Arc<Frame<T>>>>, //    packet_q: VecDeque<Packet>
  /// Maps frame *idx* to frame data
  frame_invariants: BTreeMap<u64, FrameInvariants<T>>,
  /// A list of keyframe *numbers* in this encode. Needed so that we don't
  /// need to keep all of the frame_invariants in memory for the whole life of the encode.
  keyframes: BTreeSet<u64>,
  /// A storage space for reordered frames.
  packet_data: Vec<u8>,
  segment_start_idx: u64,
  segment_start_frame: u64,
  keyframe_detector: SceneChangeDetector<T>,
  pub(crate) config: EncoderConfig,
  rc_state: RCState,
  maybe_prev_log_base_q: Option<i64>,
  pub first_pass_data: FirstPassData,
  pool: rayon::ThreadPool,
}

pub struct Context<T: Pixel> {
  inner: ContextInner<T>,
  config: EncoderConfig,
}

#[derive(Clone, Copy, Debug)]
pub enum EncoderStatus {
  /// The encoder needs more data to produce an output Packet--used with frame reordering
  NeedMoreData,
  /// The encoder needs more Frames to analyze lookahead
  NeedMoreFrames,
  /// There are enough Frames queue
  EnoughData,
  ///
  Failure,
  InvalidKey,
  ParseError
}

pub struct Packet<T: Pixel> {
  pub data: Vec<u8>,
  pub rec: Option<Frame<T>>,
  pub number: u64,
  pub frame_type: FrameType,
  /// PSNR for Y, U, and V planes
  pub psnr: Option<(f64, f64, f64)>,
}

impl<T: Pixel> fmt::Display for Packet<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "Frame {} - {} - {} bytes",
      self.number,
      self.frame_type,
      self.data.len()
    )
  }
}

impl<T: Pixel> Context<T> {
  pub fn new_frame(&self) -> Arc<Frame<T>> {
    Arc::new(Frame::new(
      self.config.width,
      self.config.height,
      self.config.chroma_sampling
    ))
  }

  pub fn send_frame<F>(&mut self, frame: F) -> Result<(), EncoderStatus>
  where
    F: Into<Option<Arc<Frame<T>>>>,
    T: Pixel,
  {
    let frame = frame.into();

    if frame.is_none() {
        self.inner.limit = self.inner.frame_count;
    }

    self.inner.send_frame(frame)
  }

  pub fn receive_packet(&mut self) -> Result<Packet<T>, EncoderStatus> {
    self.inner.receive_packet()
  }

  pub fn flush(&mut self) {
    self.send_frame(None).unwrap();
  }

  pub fn container_sequence_header(&mut self) -> Vec<u8> {
    fn sequence_header_inner(seq: &Sequence) -> io::Result<Vec<u8>> {
      let mut buf = Vec::new();

      {
        let mut bw = BitWriter::endian(&mut buf, BigEndian);
        bw.write_bit(true)?; // marker
        bw.write(7, 1)?; // version
        bw.write(3, seq.profile)?;
        bw.write(5, 31)?; // level
        bw.write_bit(false)?; // tier
        bw.write_bit(seq.bit_depth > 8)?; // high_bitdepth
        bw.write_bit(seq.bit_depth == 12)?; // twelve_bit
        bw.write_bit(seq.bit_depth == 1)?; // monochrome
        bw.write_bit(seq.bit_depth == 12)?; // twelve_bit
        bw.write_bit(seq.chroma_sampling != ChromaSampling::Cs444)?; // chroma_subsampling_x
        bw.write_bit(seq.chroma_sampling == ChromaSampling::Cs420)?; // chroma_subsampling_y
        bw.write(2, 0)?; // sample_position
        bw.write(3, 0)?; // reserved
        bw.write_bit(false)?; // initial_presentation_delay_present

        bw.write(4, 0)?; // reserved
      }

      Ok(buf)
    }

    let seq = Sequence::new(&self.config);

    sequence_header_inner(&seq).unwrap()
  }

  pub fn get_first_pass_data(&self) -> &FirstPassData {
    &self.inner.first_pass_data
  }


  // TODO: the methods below should go away

  pub fn get_frame_count(&self) -> u64 {
    self.inner.get_frame_count()
  }

  pub fn set_limit(&mut self, limit: u64) {
    self.inner.set_limit(limit);
  }

  pub fn needs_more_frames(&self, frame_count: u64) -> bool {
    self.inner.needs_more_frames(frame_count)
  }

}


impl<T: Pixel> ContextInner<T> {
  pub fn send_frame<F>(&mut self, frame: F) -> Result<(), EncoderStatus>
  where
    F: Into<Option<Arc<Frame<T>>>>,
    T: Pixel,
  {
    let idx = self.frame_count;
    self.frame_q.insert(idx, frame.into());
    self.frame_count += 1;
    Ok(())
  }

  fn get_frame(&self, frame_number: u64) -> Arc<Frame<T>> {
    // Clones only the arc, so low cost overhead
    self.frame_q.get(&frame_number).as_ref().unwrap().as_ref().unwrap().clone()
  }

  pub fn get_frame_count(&self) -> u64 {
    self.frame_count
  }

  pub fn set_limit(&mut self, limit: u64) {
    self.limit = limit;
  }

  pub(crate) fn needs_more_lookahead(&self) -> bool {
    self.needs_more_frames(self.frame_count) && self.frames_processed + LOOKAHEAD_FRAMES > self.frame_q.keys().last().cloned().unwrap_or(0)
  }

  pub fn needs_more_frames(&self, frame_count: u64) -> bool {
    self.limit == 0 || frame_count < self.limit
  }

  fn next_keyframe(&self) -> u64 {
    let next_detected = self.frame_invariants.values()
      .find(|fi| fi.frame_type == FrameType::KEY && fi.number > self.segment_start_frame)
      .map(|fi| fi.number);
    let next_limit = self.segment_start_frame + self.config.max_key_frame_interval;
    if next_detected.is_none() {
      return next_limit;
    }
    cmp::min(next_detected.unwrap(), next_limit)
  }

  fn set_frame_properties(&mut self, idx: u64) -> bool {
    let (fi, end_of_subgop) = self.build_frame_properties(idx);
    self.frame_invariants.insert(idx, fi);

    end_of_subgop
  }

  fn build_frame_properties(&mut self, idx: u64) -> (FrameInvariants<T>, bool) {
    if idx == 0 {
      let seq = Sequence::new(&self.config);

      // The first frame will always be a key frame
      let fi = FrameInvariants::new_key_frame(
        &FrameInvariants::new(
          self.config.clone(),
          seq
        ),
        0
      );
      return (fi, true);
    }

    let mut fi = self.frame_invariants[&(idx - 1)].clone();

    // FIXME: inter unsupported with 4:2:2 and 4:4:4 chroma sampling
    let chroma_sampling = self.config.chroma_sampling;
    let keyframe_only = chroma_sampling == ChromaSampling::Cs444 ||
      chroma_sampling == ChromaSampling::Cs422;

    // Initially set up the frame as an inter frame.
    // We need to determine what the frame number is before we can
    // look up the frame type. If reordering is enabled, the idx
    // may not match the frame number.
    let idx_in_segment = idx - self.segment_start_idx;
    if idx_in_segment > 0 {
      let next_keyframe = if keyframe_only { self.segment_start_frame + 1 } else { self.next_keyframe() };
      let (fi_temp, end_of_subgop) = FrameInvariants::new_inter_frame(
        &fi,
        self.segment_start_frame,
        idx_in_segment,
        next_keyframe
      );
      fi = fi_temp;
      if !end_of_subgop {
        if !fi.inter_cfg.unwrap().reorder
          || ((idx_in_segment - 1) % fi.inter_cfg.unwrap().group_len == 0
          && fi.number == (next_keyframe - 1))
        {
          self.segment_start_idx = idx;
          self.segment_start_frame = next_keyframe;
          fi.number = next_keyframe;
        } else {
          return (fi, false);
        }
      }
    }

    match self.frame_q.get(&fi.number) {
      Some(Some(_)) => {},
      _ => { return (fi, false); }
    }

    // Now that we know the frame number, look up the correct frame type
    let frame_type = self.determine_frame_type(fi.number);
    if frame_type == FrameType::KEY {
      self.segment_start_idx = idx;
      self.segment_start_frame = fi.number;
      self.keyframes.insert(fi.number);
    }
    fi.frame_type = frame_type;

    let idx_in_segment = idx - self.segment_start_idx;
    if idx_in_segment == 0 {
      fi = FrameInvariants::new_key_frame(&fi, self.segment_start_frame);
    } else {
      let next_keyframe = self.next_keyframe();
      let (fi_temp, end_of_subgop) = FrameInvariants::new_inter_frame(
        &fi,
        self.segment_start_frame,
        idx_in_segment,
        next_keyframe
      );
      fi = fi_temp;
      if !end_of_subgop {
        return (fi, false);
      }
    }
    (fi, true)
  }

  pub fn receive_packet(&mut self) -> Result<Packet<T>, EncoderStatus> {
    if self.limit != 0 && self.frames_processed == self.limit {
      return Err(EncoderStatus::EnoughData);
    }

    if self.needs_more_lookahead() {
      return Err(EncoderStatus::NeedMoreFrames);
    }

    let idx = {
      let mut idx = self.idx;
      while !self.set_frame_properties(idx) {
        self.idx += 1;
        idx = self.idx;
      }

      if !self.needs_more_frames(self.frame_invariants[&idx].number) {
        self.idx += 1;
        return Err(EncoderStatus::EnoughData);
      }
      idx
    };

    let ret = {
      let fi = self.frame_invariants.get_mut(&idx).unwrap();
      if fi.show_existing_frame {
        self.idx += 1;

        let mut fs = FrameState::new(fi);

        // TODO: Record the bits spent here against the original frame for rate
        //  control purposes, or add a new frame subtype?
        let sef_data = encode_show_existing_frame(fi, &mut fs);
        self.packet_data.extend(sef_data);

        let rec = if fi.show_frame { Some(fs.rec) } else { None };
        let fi = fi.clone();
        self.finalize_packet(rec, &fi)
      } else {
        if let Some(f) = self.frame_q.get(&fi.number) {
          self.idx += 1;

          if let Some(frame) = f.clone() {
            let fti = fi.get_frame_subtype();
            let qps =
              self.rc_state.select_qi(self, fti, self.maybe_prev_log_base_q);
            let fi = self.frame_invariants.get_mut(&idx).unwrap();
            fi.set_quantizers(&qps);
            let mut fs = FrameState::new_with_frame(fi, frame.clone());

            // TODO: Trial encoding for first frame of each type.
            let data = self.pool.install(||encode_frame(fi, &mut fs));
            self.maybe_prev_log_base_q = Some(qps.log_base_q);
            // TODO: Add support for dropping frames.
            self.rc_state.update_state(
              (data.len() * 8) as i64,
              fti,
              qps.log_target_q,
              false
            );
            self.packet_data.extend(data);

            fs.rec.pad(fi.width, fi.height);

            // TODO avoid the clone by having rec Arc.
            let rec = if fi.show_frame { Some(fs.rec.clone()) } else { None };

            update_rec_buffer(fi, fs);

            if fi.show_frame {
              let fi = fi.clone();
              self.finalize_packet(rec, &fi)
            } else {
              Err(EncoderStatus::NeedMoreData)
            }
          } else {
            Err(EncoderStatus::NeedMoreData)
          }
        } else {
          Err(EncoderStatus::NeedMoreData)
        }
      }
    };

    if let Ok(ref pkt) = ret {
      self.garbage_collect(pkt.number);
    }

    ret
  }

  fn finalize_packet(&mut self, rec: Option<Frame<T>>, fi: &FrameInvariants<T>) -> Result<Packet<T>, EncoderStatus> {
    let data = self.packet_data.clone();
    self.packet_data.clear();
    if write_temporal_delimiter(&mut self.packet_data).is_err() {
      return Err(EncoderStatus::Failure);
    }

    let mut psnr = None;
    if self.config.show_psnr {
      if let Some(ref rec) = rec {
        let original_frame = self.get_frame(fi.number);
        psnr = Some(calculate_frame_psnr(
          &*original_frame,
          rec,
          fi.sequence.bit_depth
        ));
      }
    }

    if self.config.pass == Some(1) {
      self.first_pass_data.frames.push(FirstPassFrame::from(fi));
    }

    self.frames_processed += 1;
    Ok(Packet {
      data,
      rec,
      number: fi.number,
      frame_type: fi.frame_type,
      psnr
    })
  }

  fn garbage_collect(&mut self, cur_frame: u64) {
    if cur_frame == 0 {
      return;
    }
    for i in 0..cur_frame {
      self.frame_q.remove(&i);
    }
    if self.idx < 2 {
      return;
    }
    for i in 0..(self.idx - 1) {
      self.frame_invariants.remove(&i);
    }
  }

  pub fn flush(&mut self) {
    self.frame_q.insert(self.frame_count, None);
    self.frame_count += 1;
  }

  fn determine_frame_type(&mut self, frame_number: u64) -> FrameType {
    if frame_number == 0 {
      return FrameType::KEY;
    }
    if self.config.speed_settings.no_scene_detection {
      if frame_number % self.config.max_key_frame_interval == 0 {
        return FrameType::KEY;
      } else {
        return FrameType::INTER;
      }
    }

    let prev_keyframe = self.keyframes.iter()
      .rfind(|&&keyframe| keyframe < frame_number)
      .cloned()
      .unwrap_or(0);
    let frame = match self.frame_q.get(&frame_number).cloned() {
      Some(frame) => frame,
      None => { return FrameType::KEY; }
    };
    if let Some(frame) = frame {
      let distance = frame_number - prev_keyframe;
      if distance < self.config.min_key_frame_interval {
        if distance + 1 == self.config.min_key_frame_interval {
          self.keyframe_detector.set_last_frame(frame, frame_number as usize);
        }
        return FrameType::INTER;
      }
      if distance >= self.config.max_key_frame_interval {
        return FrameType::KEY;
      }
      if self.keyframe_detector.detect_scene_change(frame, frame_number as usize) {
        return FrameType::KEY;
      }
    }
    FrameType::INTER
  }

  // Count the number of frames of each subtype in the next
  //  reservoir_frame_delay frames.
  // Returns the number of frames until the last keyframe in the next
  //  reservoir_frame_delay frames, or the end of the interval, whichever
  //  comes first.
  pub(crate) fn guess_frame_subtypes(
    &self, nframes: &mut [i32; FRAME_NSUBTYPES], reservoir_frame_delay: i32
  ) -> i32 {
    // TODO: Ideally this logic should be centralized, but the actual code used
    //  to determine a frame's subtype is spread over many places and
    //  intertwined with mutable state changes that occur when the frame is
    //  actually encoded.
    // So for now we just duplicate it here in stateless fashion.
    for fti in 0..FRAME_NSUBTYPES {
      nframes[fti] = 0;
    }
    let mut prev_keyframe = self.segment_start_idx;
    let mut acc: [i32; FRAME_NSUBTYPES] = [0; FRAME_NSUBTYPES];
    // Updates the frame counts with the accumulated values when we hit a
    //  keyframe.
    fn collect_counts(
      nframes: &mut [i32; FRAME_NSUBTYPES], acc: &mut [i32; FRAME_NSUBTYPES]
    ) {
      for fti in 0..FRAME_NSUBTYPES {
        nframes[fti] += acc[fti];
        acc[fti] = 0;
      }
      acc[FRAME_SUBTYPE_I] += 1;
    }
    for idx in self.idx..(self.idx + reservoir_frame_delay as u64) {
      if let Some(fd) = self.frame_invariants.get(&idx) {
        if fd.frame_type == FrameType::KEY {
          collect_counts(nframes, &mut acc);
          prev_keyframe = idx;
          continue;
        }
      } else if idx == 0
        || idx - prev_keyframe >= self.config.max_key_frame_interval
      {
        collect_counts(nframes, &mut acc);
        prev_keyframe = idx;
        continue;
      }
      // TODO: Implement golden P-frames.
      let mut fti = FRAME_SUBTYPE_P;
      if !self.config.low_latency {
        let pyramid_depth = 2;
        let group_src_len = 1 << pyramid_depth;
        let group_len = group_src_len + pyramid_depth;
        let idx_in_group = (idx - prev_keyframe - 1) % group_len;
        let lvl = if idx_in_group < pyramid_depth {
          idx_in_group
        } else {
          pos_to_lvl(idx_in_group - pyramid_depth + 1, pyramid_depth)
        };
        fti += lvl as usize;
      }
      acc[fti] += 1;
    }
    if prev_keyframe <= self.idx {
      // If there were no keyframes at all, or only the first frame was a
      //  keyframe, the accumulators never flushed and still contain counts for
      //  the entire buffer.
      // In both cases, we return these counts.
      collect_counts(nframes, &mut acc);
      reservoir_frame_delay
    } else {
      // Otherwise, we discard what remains in the accumulators as they contain
      //  the counts from and past the last keyframe.
      (prev_keyframe - self.idx) as i32
    }
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirstPassData {
  frames: Vec<FirstPassFrame>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FirstPassFrame {
  number: u64,
  frame_type: FrameType,
}

impl<T: Pixel> From<&FrameInvariants<T>> for FirstPassFrame {
  fn from(fi: &FrameInvariants<T>) -> FirstPassFrame {
    FirstPassFrame {
      number: fi.number,
      frame_type: fi.frame_type,
    }
  }
}

#[cfg(test)]
mod test {
  use super::*;

  use interpolate_name::interpolate_test;

  fn setup_encoder<T: Pixel>(
    w: usize, h: usize, speed: usize, quantizer: usize, bit_depth: usize,
    chroma_sampling: ChromaSampling, min_keyint: u64, max_keyint: u64,
    bitrate: i32,
    low_latency: bool,
    no_scene_detection: bool,
  ) -> Context<T> {
    assert!(bit_depth == 8 || std::mem::size_of::<T>() > 1);
    let mut enc = EncoderConfig::with_speed_preset(speed);
    enc.quantizer = quantizer;
    enc.min_key_frame_interval = min_keyint;
    enc.max_key_frame_interval = max_keyint;
    enc.low_latency = low_latency;
    enc.width = w;
    enc.height = h;
    enc.bit_depth = bit_depth;
    enc.chroma_sampling = chroma_sampling;
    enc.bitrate = bitrate;
    enc.speed_settings.no_scene_detection = no_scene_detection;

    let cfg = Config { enc, threads: 0 };

    cfg.new_context()
  }

  /*
  fn fill_frame<T: Pixel>(ra: &mut ChaChaRng, frame: &mut Frame<T>) {
    for plane in frame.planes.iter_mut() {
      let stride = plane.cfg.stride;
      for row in plane.data.chunks_mut(stride) {
        for pixel in row {
          let v: u8 = ra.gen();
          *pixel = T::cast_from(v);
        }
      }
    }
  }
  */


  #[interpolate_test(low_latency_no_scene_change, true, true)]
  #[interpolate_test(reorder_no_scene_change, false, true)]
  #[interpolate_test(low_latency_scene_change_detection, true, false)]
  #[interpolate_test(reorder_scene_change_detection, false, false)]
  #[test]
  fn flush(low_lantency: bool, no_scene_detection: bool) {
    let mut ctx = setup_encoder::<u8>(64, 80, 5, 100, 8, ChromaSampling::Cs420, 15, 20, 0, low_lantency, no_scene_detection);
    let limit = 40;

    ctx.set_limit(limit);

    for _ in  0..limit {
      let input = ctx.new_frame();
      let _ = ctx.send_frame(input);
    }

    ctx.flush();

    let mut count = 0;

    'out: for _ in 0..limit {
      loop {
        match ctx.receive_packet() {
          Ok(_) => {
            eprintln!("Packet Received {}/{}", count, limit);
            count += 1;
          },
          Err(EncoderStatus::EnoughData) => {
            eprintln!("{:?}", EncoderStatus::EnoughData);

            break 'out;
          }
          Err(e) => {
            eprintln!("{:?}", e);
            break;
          }
        }
      }
    }

    assert_eq!(limit, count);
  }


  #[interpolate_test(low_latency_no_scene_change, true, true)]
  #[interpolate_test(reorder_no_scene_change, false, true)]
  #[interpolate_test(low_latency_scene_change_detection, true, false)]
  #[interpolate_test(reorder_scene_change_detection, false, false)]
  #[test]
  fn flush_unlimited(low_lantency: bool, no_scene_detection: bool) {
    let mut ctx = setup_encoder::<u8>(64, 80, 5, 100, 8, ChromaSampling::Cs420, 15, 20, 0, low_lantency, no_scene_detection);
    let limit = 40;

    for _ in  0..limit {
      let input = ctx.new_frame();
      let _ = ctx.send_frame(input);
    }

    ctx.flush();

    let mut count = 0;

    'out: for _ in 0..limit {
      loop {
        match ctx.receive_packet() {
          Ok(_) => {
            eprintln!("Packet Received {}/{}", count, limit);
            count += 1;
          },
          Err(EncoderStatus::EnoughData) => {
            eprintln!("{:?}", EncoderStatus::EnoughData);

            break 'out;
          }
          Err(e) => {
            eprintln!("{:?}", e);
            break;
          }
        }
      }
    }

    assert_eq!(limit, count);
  }
}
