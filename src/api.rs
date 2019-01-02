// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use bitstream_io::*;
use encoder::*;
use metrics::calculate_frame_psnr;
use partition::*;
use scenechange::SceneChangeDetector;
use self::EncoderStatus::*;

use std::{cmp, fmt, io};
use std::collections::BTreeMap;
use std::sync::Arc;
use util::Fixed;
use std::collections::BTreeSet;

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

#[derive(Copy, Clone, Debug)]
pub struct EncoderConfig {
  /// The *minimum* interval between two keyframes
  pub min_key_frame_interval: u64,
  /// The *maximum* interval between two keyframes
  pub max_key_frame_interval: u64,
  pub low_latency: bool,
  pub quantizer: usize,
  pub tune: Tune,
  pub color_description: Option<ColorDescription>,
  pub speed_settings: SpeedSettings,
  pub show_psnr: bool,
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
      min_key_frame_interval: 12,
      max_key_frame_interval: 240,
      low_latency: true,
      quantizer: 100,
      tune: Tune::Psnr,
      color_description: None,
      speed_settings: SpeedSettings::from_preset(speed),
      show_psnr: false,
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
    speed >= 1
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

#[derive(Clone, Copy, Debug, PartialOrd, PartialEq)]
pub enum PredictionModesSetting {
  Simple,
  ComplexKeyframes,
  ComplexAll,
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
  #[derive(Debug,Clone,Copy,PartialEq)]
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
  #[derive(Debug,Clone,Copy,PartialEq)]
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

/// Frame-specific information
#[derive(Clone, Copy, Debug)]
pub struct FrameInfo {
  pub width: usize,
  pub height: usize,
  pub bit_depth: usize,
  pub chroma_sampling: ChromaSampling,
  pub chroma_sample_position: ChromaSamplePosition
}

impl Default for FrameInfo {
    fn default() -> FrameInfo {
        FrameInfo {
            width: 640,
            height: 480,
            bit_depth: 8,
            chroma_sampling: Default::default(),
            chroma_sample_position: Default::default()
        }
    }
}

/// Contain all the encoder configuration
#[derive(Clone, Copy, Debug)]
pub struct Config {
  pub frame_info: FrameInfo,
  pub timebase: Rational,
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

    Context {
      frame_count: 0,
      frames_to_be_coded: 0,
      idx: 0,
      frames_processed: 0,
      frame_q: BTreeMap::new(),
      frame_data: BTreeMap::new(),
      keyframes: BTreeSet::new(),
      packet_data: Vec::new(),
      segment_start_idx: 0,
      segment_start_frame: 0,
      keyframe_detector: SceneChangeDetector::new(&self.frame_info),
      config: *self,
    }
  }
}

pub struct Context {
  //    timebase: Rational,
  frame_count: u64,
  frames_to_be_coded: u64,
  idx: u64,
  frames_processed: u64,
  /// Maps frame *number* to frames
  frame_q: BTreeMap<u64, Option<Arc<Frame>>>, //    packet_q: VecDeque<Packet>
  /// Maps frame *idx* to frame data
  frame_data: BTreeMap<u64, FrameInvariants>,
  /// A list of keyframe *numbers* in this encode. Needed so that we don't
  /// need to keep all of the frame_data in memory for the whole life of the encode.
  keyframes: BTreeSet<u64>,
  packet_data: Vec<u8>,
  segment_start_idx: u64,
  segment_start_frame: u64,
  keyframe_detector: SceneChangeDetector,
  config: Config,
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
      self.config.frame_info.width.align_power_of_two(3),
      self.config.frame_info.height.align_power_of_two(3),
      self.config.frame_info.chroma_sampling
    ))
  }

  pub fn send_frame<F>(&mut self, frame: F) -> Result<(), EncoderStatus>
  where
    F: Into<Option<Arc<Frame>>>
  {
    let idx = self.frame_count;
    self.frame_q.insert(idx, frame.into());
    self.frame_count = self.frame_count + 1;
    Ok(())
  }

  pub fn get_frame_count(&self) -> u64 {
    self.frame_count
  }

  pub fn set_frames_to_be_coded(&mut self, frames_to_be_coded: u64) {
    self.frames_to_be_coded = frames_to_be_coded;
  }

  pub fn needs_more_lookahead(&self) -> bool {
    self.needs_more_frames(self.frame_count) && self.frames_processed + LOOKAHEAD_FRAMES > self.frame_q.keys().last().cloned().unwrap_or(0)
  }

  pub fn needs_more_frames(&self, frame_count: u64) -> bool {
    self.frames_to_be_coded == 0 || frame_count < self.frames_to_be_coded
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

  fn set_frame_properties(&mut self, idx: u64) -> Result<(), ()> {
    let props = self.get_frame_properties(idx);
    let result = props.as_ref().map(|_| ()).map_err(|_| ());
    match props {
      Ok(props) | Err(props) => {
        self.frame_data.insert(idx, props);
      }
    }
    result
  }

  fn get_frame_properties(&mut self, idx: u64) -> Result<FrameInvariants, FrameInvariants> {
    if idx == 0 {
      // The first frame will always be a key frame
      let fi = FrameInvariants::new_key_frame(
        &FrameInvariants::new(
          self.config.frame_info.width,
          self.config.frame_info.height,
          self.config.enc,
          Sequence::new(&self.config.frame_info)
        ),
        0
      );
      return Ok(fi);
    }

    let mut fi = self.frame_data[&(idx - 1)].clone();

    // Initially set up the frame as an inter frame.
    // We need to determine what the frame number is before we can
    // look up the frame type. If reordering is enabled, the idx
    // may not match the frame number.
    let idx_in_segment = idx - self.segment_start_idx;
    if idx_in_segment > 0 {
      let next_keyframe = self.next_keyframe();
      let (fi_temp, success) = FrameInvariants::new_inter_frame(
        &fi,
        self.segment_start_frame,
        idx_in_segment,
        next_keyframe
      );
      fi = fi_temp;
      if !success {
        if !fi.inter_cfg.unwrap().reorder
          || ((idx_in_segment - 1) % fi.inter_cfg.unwrap().group_len == 0
          && fi.number == (next_keyframe - 1))
        {
          self.segment_start_idx = idx;
          self.segment_start_frame = next_keyframe;
          fi.number = next_keyframe;
        } else {
          return Err(fi);
        }
      }
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
      let (fi_temp, success) = FrameInvariants::new_inter_frame(
        &fi,
        self.segment_start_frame,
        idx_in_segment,
        next_keyframe
      );
      fi = fi_temp;
      if !success {
        return Err(fi);
      }
    }
    Ok(fi)
  }

  pub fn receive_packet(&mut self) -> Result<Packet, EncoderStatus> {
    let mut idx = self.idx;
    while self.set_frame_properties(idx).is_err() {
      self.idx += 1;
      idx = self.idx;
    }

    if !self.needs_more_frames(self.frame_data.get(&idx).unwrap().number) {
      self.idx += 1;
      return Err(EncoderStatus::EnoughData)
    }

    let fi = self.frame_data.get_mut(&idx).unwrap();
    if fi.show_existing_frame {
      self.idx += 1;

      let mut fs = FrameState::new(fi);

      let data = encode_frame(fi, &mut fs);

      let rec = if fi.show_frame { Some(fs.rec) } else { None };
      let mut psnr = None;
      if self.config.enc.show_psnr {
        if let Some(ref rec) = rec {
          psnr = Some(calculate_frame_psnr(&*fs.input, rec, fi.sequence.bit_depth));
        }
      }

      self.frames_processed += 1;
      Ok(Packet { data, rec, number: fi.number, frame_type: fi.frame_type, psnr })
    } else {
      if let Some(f) = self.frame_q.get(&fi.number) {
        self.idx += 1;

        if let Some(frame) = f {
          let mut fs = FrameState::new_with_frame(fi, frame.clone());

          let data = encode_frame(fi, &mut fs);
          self.packet_data.extend(data);

          fs.rec.pad(fi.width, fi.height);

          // TODO avoid the clone by having rec Arc.
          let rec = if fi.show_frame { Some(fs.rec.clone()) } else { None };

          update_rec_buffer(fi, fs);

          if fi.show_frame {
            let data = self.packet_data.clone();
            self.packet_data = Vec::new();

            let mut psnr = None;
            if self.config.enc.show_psnr {
              if let Some(ref rec) = rec {
                psnr = Some(calculate_frame_psnr(&*frame, rec, fi.sequence.bit_depth));
              }
            }

            self.frames_processed += 1;
            Ok(Packet { data, rec, number: fi.number, frame_type: fi.frame_type, psnr })
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
  }

  pub fn garbage_collect(&mut self, cur_frame: u64) {
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
    self.frame_count = self.frame_count + 1;
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
