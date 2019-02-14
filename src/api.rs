// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use bitstream_io::*;
use crate::encoder::*;
use crate::metrics::calculate_frame_psnr;
use crate::partition::*;
use crate::rate::RCState;
use crate::scenechange::SceneChangeDetector;
use self::EncoderStatus::*;

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
  pub tune: Tune,
  pub speed_settings: SpeedSettings,
  /// `None` for one-pass encode. `Some(1)` or `Some(2)` for two-pass encoding.
  pub pass: Option<u8>,
  pub show_psnr: bool,
  pub stats_file: Option<PathBuf>,
}

impl Default for EncoderConfig {
  fn default() -> Self {
    const DEFAULT_SPEED: usize = 3;
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
      tune: Tune::Psnr,
      speed_settings: SpeedSettings::from_preset(speed),
      pass: None,
      show_psnr: false,
      stats_file: None,
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
  pub encode_bottomup: bool,
  pub rdo_tx_decision: bool,
  pub prediction_modes: PredictionModesSetting,
  pub include_near_mvs: bool,
}

impl SpeedSettings {
  pub fn from_preset(speed: usize) -> Self {
    SpeedSettings {
      min_block_size: Self::min_block_size_preset(speed),
      multiref: Self::multiref_preset(speed),
      fast_deblock: Self::fast_deblock_preset(speed),
      reduced_tx_set: Self::reduced_tx_set_preset(speed),
      tx_domain_distortion: Self::tx_domain_distortion_preset(speed),
      encode_bottomup: Self::encode_bottomup_preset(speed),
      rdo_tx_decision: Self::rdo_tx_decision_preset(speed),
      prediction_modes: Self::prediction_modes_preset(speed),
      include_near_mvs: Self::include_near_mvs_preset(speed),
    }
  }

  fn min_block_size_preset(speed: usize) -> BlockSize {
    if speed <= 1 {
      BlockSize::BLOCK_4X4
    } else if speed <= 2 {
      BlockSize::BLOCK_8X8
    } else if speed <= 3 {
      BlockSize::BLOCK_16X16
    } else if speed <= 4 {
      BlockSize::BLOCK_32X32
    } else {
      BlockSize::BLOCK_64X64
    }
  }

  fn multiref_preset(speed: usize) -> bool {
    speed <= 2
  }

  fn fast_deblock_preset(speed: usize) -> bool {
    speed >= 4
  }

  fn reduced_tx_set_preset(speed: usize) -> bool {
    speed >= 2
  }

  fn tx_domain_distortion_preset(speed: usize) -> bool {
    speed >= 4
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
    } else if speed <= 3 {
      PredictionModesSetting::ComplexKeyframes
    } else {
      PredictionModesSetting::Simple
    }
  }

  fn include_near_mvs_preset(speed: usize) -> bool {
    speed <= 2
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

arg_enum!{
  #[derive(Debug, Clone, Copy, PartialEq)]
  #[repr(C)]
  pub enum PixelRange {
      Unspecified = 0,
      Limited,
      Full,
  }
}

impl Default for PixelRange {
    fn default() -> Self {
        PixelRange::Unspecified
    }
}

arg_enum!{
  #[derive(Debug, Clone, Copy, PartialEq)]
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
}

impl Default for MatrixCoefficients {
    fn default() -> Self {
        MatrixCoefficients::Unspecified
    }
}

arg_enum!{
  #[derive(Debug, Clone, Copy, PartialEq)]
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
}

impl Default for ColorPrimaries {
    fn default() -> Self {
        ColorPrimaries::Unspecified
    }
}

arg_enum!{
  #[derive(Debug, Clone, Copy, PartialEq)]
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

/// Contain all the encoder configuration
#[derive(Clone, Debug)]
pub struct Config {
  pub enc: EncoderConfig
}

impl Config {
  pub fn parse(&mut self, key: &str, value: &str) -> Result<(), EncoderStatus> {
    match key {
      "low_latency" => self.enc.low_latency = value.parse().map_err(|_e| ParseError)?,
      "min_key_frame_interval" => self.enc.min_key_frame_interval = value.parse().map_err(|_e| ParseError)?,
      "key_frame_interval" => self.enc.max_key_frame_interval = value.parse().map_err(|_e| ParseError)?,
      "quantizer" => self.enc.quantizer = value.parse().map_err(|_e| ParseError)?,
      "speed" => self.enc.speed_settings = SpeedSettings::from_preset(value.parse().map_err(|_e| ParseError)?),
      "tune" => self.enc.tune = value.parse().map_err(|_e| ParseError)?,
      _ => return Err(InvalidKey)
    }

    Ok(())
  }

  pub fn new_context(&self) -> Context {
    #[cfg(feature = "aom")]
    unsafe {
      av1_rtcd();
      aom_dsp_rtcd();
    }

    // initialize with temporal delimiter
    let packet_data = TEMPORAL_DELIMITER.to_vec();

    Context {
      frame_count: 0,
      limit: 0,
      idx: 0,
      frames_processed: 0,
      frame_q: BTreeMap::new(),
      frame_data: BTreeMap::new(),
      keyframes: BTreeSet::new(),
      packet_data,
      segment_start_idx: 0,
      segment_start_frame: 0,
      keyframe_detector: SceneChangeDetector::new(self.enc.bit_depth),
      config: self.clone(),
      rc_state: RCState::new(),
      first_pass_data: FirstPassData {
        frames: Vec::new(),
      },
    }
  }
}

pub struct Context {
  //    timebase: Rational,
  frame_count: u64,
  limit: u64,
  idx: u64,
  frames_processed: u64,
  /// Maps frame *number* to frames
  frame_q: BTreeMap<u64, Option<Arc<Frame>>>, //    packet_q: VecDeque<Packet>
  /// Maps frame *idx* to frame data
  frame_data: BTreeMap<u64, FrameInvariants>,
  /// A list of keyframe *numbers* in this encode. Needed so that we don't
  /// need to keep all of the frame_data in memory for the whole life of the encode.
  keyframes: BTreeSet<u64>,
  /// A storage space for reordered frames.
  packet_data: Vec<u8>,
  segment_start_idx: u64,
  segment_start_frame: u64,
  keyframe_detector: SceneChangeDetector,
  pub config: Config,
  rc_state: RCState,
  pub first_pass_data: FirstPassData,
}

#[derive(Clone, Copy, Debug)]
pub enum EncoderStatus {
  /// The encoder needs more Frames to produce an output Packet
  NeedMoreData,
  /// There are enough Frames queue
  EnoughData,
  ///
  Failure,
  InvalidKey,
  ParseError
}

pub struct Packet {
  pub data: Vec<u8>,
  pub rec: Option<Frame>,
  pub number: u64,
  pub frame_type: FrameType,
  /// PSNR for Y, U, and V planes
  pub psnr: Option<(f64, f64, f64)>,
}

impl fmt::Display for Packet {
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

impl Context {
  pub fn new_frame(&self) -> Arc<Frame> {
    Arc::new(Frame::new(
      self.config.enc.width,
      self.config.enc.height,
      self.config.enc.chroma_sampling
    ))
  }

  pub fn send_frame<F>(&mut self, frame: F) -> Result<(), EncoderStatus>
  where
    F: Into<Option<Arc<Frame>>>
  {
    let idx = self.frame_count;
    self.frame_q.insert(idx, frame.into());
    self.frame_count += 1;
    Ok(())
  }

  pub fn get_frame_count(&self) -> u64 {
    self.frame_count
  }

  pub fn set_limit(&mut self, limit: u64) {
    self.limit = limit;
  }

  pub fn needs_more_lookahead(&self) -> bool {
    self.needs_more_frames(self.frame_count) && self.frames_processed + LOOKAHEAD_FRAMES > self.frame_q.keys().last().cloned().unwrap_or(0)
  }

  pub fn needs_more_frames(&self, frame_count: u64) -> bool {
    self.limit == 0 || frame_count < self.limit
  }

  pub fn container_sequence_header(&mut self) -> Vec<u8> {
    fn sequence_header_inner(seq: &Sequence) -> io::Result<Vec<u8>> {
      let mut buf = Vec::new();

      {
        let mut bw = BitWriter::endian(&mut buf, BigEndian);
        bw.write_bit(true)?; // marker
        bw.write(7, 1)?; // version
        bw.write(3, seq.profile)?;
        bw.write(5, 32)?; // level
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

    sequence_header_inner(&self.frame_data[&0].sequence).unwrap()
  }

  fn next_keyframe(&self) -> u64 {
    let next_detected = self.frame_data.values()
      .find(|fi| fi.frame_type == FrameType::KEY && fi.number > self.segment_start_frame)
      .map(|fi| fi.number);
    let next_limit = self.segment_start_frame + self.config.enc.max_key_frame_interval;
    if next_detected.is_none() {
      return next_limit;
    }
    cmp::min(next_detected.unwrap(), next_limit)
  }

  fn set_frame_properties(&mut self, idx: u64) -> bool {
    let (fi, end_of_subgop) = self.build_frame_properties(idx);
    self.frame_data.insert(idx, fi);

    end_of_subgop
  }

  fn build_frame_properties(&mut self, idx: u64) -> (FrameInvariants, bool) {
    if idx == 0 {
      let seq = Sequence::new(&self.config.enc);

      // The first frame will always be a key frame
      let fi = FrameInvariants::new_key_frame(
        &FrameInvariants::new(
          self.config.enc.clone(),
          seq
        ),
        0
      );
      return (fi, true);
    }

    let mut fi = self.frame_data[&(idx - 1)].clone();

    // FIXME: inter unsupported with 4:2:2 and 4:4:4 chroma sampling
    let chroma_sampling = self.config.enc.chroma_sampling;
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

  pub fn receive_packet(&mut self) -> Result<Packet, EncoderStatus> {
    let idx = {
      let mut idx = self.idx;
      while !self.set_frame_properties(idx) {
        self.idx += 1;
        idx = self.idx;
      }

      if !self.needs_more_frames(self.frame_data.get(&idx).unwrap().number) {
        self.idx += 1;
        return Err(EncoderStatus::EnoughData);
      }
      idx
    };

    let ret = {
      let fi = self.frame_data.get_mut(&idx).unwrap();
      if fi.show_existing_frame {
        self.idx += 1;

        let mut fs = FrameState::new(fi);

        let sef_data = encode_frame(fi, &mut fs);
        self.packet_data.extend(sef_data);

        let rec = if fi.show_frame { Some(fs.rec) } else { None };
        let fi = fi.clone();
        self.finalize_packet(&*fs.input, rec, &fi)
      } else {
        if let Some(f) = self.frame_q.get(&fi.number) {
          self.idx += 1;

          if let Some(frame) = f.clone() {
            let fti = fi.get_frame_subtype();
            let qps = self.rc_state.select_qi(fi, fti);
            fi.set_quantizers(&qps);
            let mut fs = FrameState::new_with_frame(fi, frame.clone());

            let data = encode_frame(fi, &mut fs);
            self.packet_data.extend(data);

            fs.rec.pad(fi.width, fi.height);

            // TODO avoid the clone by having rec Arc.
            let rec = if fi.show_frame { Some(fs.rec.clone()) } else { None };

            update_rec_buffer(fi, fs);

            if fi.show_frame {
              let fi = fi.clone();
              self.finalize_packet(&*frame, rec, &fi)
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

  fn finalize_packet(&mut self, original_frame: &Frame, rec: Option<Frame>, fi: &FrameInvariants) -> Result<Packet, EncoderStatus> {
    let data = self.packet_data.clone();
    self.packet_data.clear();
    if write_temporal_delimiter(&mut self.packet_data).is_err() {
      return Err(EncoderStatus::Failure);
    }

    let mut psnr = None;
    if self.config.enc.show_psnr {
      if let Some(ref rec) = rec {
        psnr = Some(calculate_frame_psnr(
          &*original_frame,
          rec,
          fi.sequence.bit_depth
        ));
      }
    }

    if self.config.enc.pass == Some(1) {
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
      self.frame_data.remove(&i);
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
      if distance < self.config.enc.min_key_frame_interval {
        if distance + 1 == self.config.enc.min_key_frame_interval {
          // Run the detector for the current frame, so that it will contain this frame's information
          // to compare against the next frame. We can ignore the results for this frame.
          self.keyframe_detector.detect_scene_change(frame, frame_number as usize);
        }
        return FrameType::INTER;
      }
      if distance >= self.config.enc.max_key_frame_interval {
        return FrameType::KEY;
      }
      if self.keyframe_detector.detect_scene_change(frame, frame_number as usize) {
        return FrameType::KEY;
      }
    }
    FrameType::INTER
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

impl From<&FrameInvariants> for FirstPassFrame {
  fn from(fi: &FrameInvariants) -> FirstPassFrame {
    FirstPassFrame {
      number: fi.number,
      frame_type: fi.frame_type,
    }
  }
}
