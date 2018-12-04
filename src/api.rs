// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use encoder::*;
use partition::*;

use std::cmp;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use scenechange::SceneChangeDetector;
use metrics::calculate_frame_psnr;

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

/// Frame-specific information
#[derive(Clone, Copy, Debug)]
pub struct FrameInfo {
  pub width: usize,
  pub height: usize,
  pub bit_depth: usize,
  pub chroma_sampling: ChromaSampling,
  pub chroma_sample_position: ChromaSamplePosition
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
    use self::EncoderStatus::*;
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
    let fi = FrameInvariants::new(
      self.frame_info.width,
      self.frame_info.height,
      self.enc
    );
    let seq = Sequence::new(&self.frame_info);

    #[cfg(feature = "aom")]
    unsafe {
      av1_rtcd();
      aom_dsp_rtcd();
    }

    Context {
      fi,
      seq,
      frame_count: 0,
      frames_to_be_coded: 0,
      idx: 0,
      frame_q: BTreeMap::new(),
      packet_data: Vec::new(),
      segment_start_idx: 0,
      segment_start_frame: 0,
      frame_types: BTreeMap::new(),
      keyframe_detector: SceneChangeDetector::new(&self.frame_info),
    }
  }
}

pub struct Context {
  fi: FrameInvariants,
  seq: Sequence,
  //    timebase: Rational,
  frame_count: u64,
  frames_to_be_coded: u64,
  idx: u64,
  frame_q: BTreeMap<u64, Option<Arc<Frame>>>, //    packet_q: VecDeque<Packet>
  packet_data: Vec<u8>,
  segment_start_idx: u64,
  segment_start_frame: u64,
  frame_types: BTreeMap<u64, FrameType>,
  keyframe_detector: SceneChangeDetector,
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

#[derive(Debug, Clone, Copy)]
enum FramePropsStatus {
  FramePastSegmentEnd,
}

#[derive(Debug, Clone, Copy)]
struct InterPropsConfig {
  reorder: bool,
  multiref: bool,
  pyramid_depth: u64,
  group_src_len: u64,
  group_len: u64,
  idx_in_group: u64,
  group_idx: u64,
}

impl Context {
  pub fn new_frame(&self) -> Arc<Frame> {
    Arc::new(Frame::new(self.fi.padded_w, self.fi.padded_h))
  }

  pub fn send_frame<F>(&mut self, frame: F) -> Result<(), EncoderStatus>
  where
    F: Into<Option<Arc<Frame>>>
  {
    let idx = self.frame_count;
    self.frame_q.insert(idx, frame.into());
    self.save_frame_type(idx);
    self.frame_count = self.frame_count + 1;
    Ok(())
  }

  pub fn get_frame_count(&self) -> u64 {
    self.frame_count
  }

  pub fn set_frames_to_be_coded(&mut self, frames_to_be_coded: u64) {
    self.frames_to_be_coded = frames_to_be_coded;
  }

  pub fn needs_more_frames(&self, frame_count: u64) -> bool {
    self.frames_to_be_coded == 0 || frame_count < self.frames_to_be_coded
  }

  pub fn container_sequence_header(&mut self) -> Vec<u8> {
    use bitstream_io::*;
    use std::io;
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

    sequence_header_inner(&self.seq).unwrap()
  }

  fn setup_key_frame_properties(&mut self) {
    self.fi.frame_type = FrameType::KEY;
    self.fi.intra_only = true;
    self.fi.order_hint = 0;
    self.fi.refresh_frame_flags = ALL_REF_FRAMES_MASK;
    self.fi.show_frame = true;
    self.fi.show_existing_frame = false;
    self.fi.frame_to_show_map_idx = 0;
    let q_boost = 15;
    self.fi.base_q_idx = (self.fi.config.quantizer.max(1 + q_boost).min(255 + q_boost) - q_boost) as u8;
    self.fi.primary_ref_frame = PRIMARY_REF_NONE;
    self.fi.number = self.segment_start_frame;
    for i in 0..INTER_REFS_PER_FRAME {
      self.fi.ref_frames[i] = 0;
    }
  }

  fn get_inter_props_cfg(&mut self, idx_in_segment: u64) -> InterPropsConfig {
    let reorder = !self.fi.config.low_latency;
    let multiref = reorder || self.fi.config.speed_settings.multiref;

    let pyramid_depth = if reorder { 2 } else { 0 };
    let group_src_len = 1 << pyramid_depth;
    let group_len = group_src_len + if reorder { pyramid_depth } else { 0 };

    let idx_in_group = (idx_in_segment - 1) % group_len;
    let group_idx = (idx_in_segment - 1) / group_len;

    InterPropsConfig {
      reorder,
      multiref,
      pyramid_depth,
      group_src_len,
      group_len,
      idx_in_group,
      group_idx,
    }
  }

  fn next_keyframe(&self) -> u64 {
    let next_detected = self.frame_types.iter()
      .find(|(&i, &ty)| ty == FrameType::KEY && i > self.segment_start_frame)
      .map(|(&i, _)| i);
    let next_limit = self.segment_start_frame + self.fi.config.max_key_frame_interval;
    if next_detected.is_none() {
      return next_limit;
    }
    cmp::min(next_detected.unwrap(), next_limit)
  }

  fn setup_inter_frame_properties(&mut self, cfg: &InterPropsConfig) -> Result<(), FramePropsStatus> {
    self.fi.frame_type = FrameType::INTER;
    self.fi.intra_only = false;

    self.fi.order_hint = (cfg.group_src_len * cfg.group_idx +
      if cfg.reorder && cfg.idx_in_group < cfg.pyramid_depth {
        cfg.group_src_len >> cfg.idx_in_group
      } else {
        cfg.idx_in_group - cfg.pyramid_depth + 1
      }) as u32;
    let number = self.segment_start_frame + self.fi.order_hint as u64;
    if number >= self.next_keyframe() {
      self.fi.show_existing_frame = false;
      self.fi.show_frame = false;
      return Err(FramePropsStatus::FramePastSegmentEnd);
    }

    fn pos_to_lvl(pos: u64, pyramid_depth: u64) -> u64 {
      // Derive level within pyramid for a frame with a given coding order position
      // For example, with a pyramid of depth 2, the 2 least significant bits of the
      // position determine the level:
      // 00 -> 0
      // 01 -> 2
      // 10 -> 1
      // 11 -> 2
      pyramid_depth - (pos | (1 << pyramid_depth)).trailing_zeros() as u64
    }

    let lvl = if !cfg.reorder {
      0
    } else if cfg.idx_in_group < cfg.pyramid_depth {
      cfg.idx_in_group
    } else {
      pos_to_lvl(cfg.idx_in_group - cfg.pyramid_depth + 1, cfg.pyramid_depth)
    };

    // Frames with lvl == 0 are stored in slots 0..4 and frames with higher values
    // of lvl in slots 4..8
    let slot_idx = if lvl == 0 {
      (self.fi.order_hint >> cfg.pyramid_depth) % 4 as u32
    } else {
      3 + lvl as u32
    };
    self.fi.show_frame = !cfg.reorder || cfg.idx_in_group >= cfg.pyramid_depth;
    self.fi.show_existing_frame = self.fi.show_frame && cfg.reorder &&
      (cfg.idx_in_group - cfg.pyramid_depth + 1).count_ones() == 1 &&
      cfg.idx_in_group != cfg.pyramid_depth;
    self.fi.frame_to_show_map_idx = slot_idx;
    self.fi.refresh_frame_flags = if self.fi.show_existing_frame {
      0
    } else {
      1 << slot_idx
    };

    let q_drop = 15 * lvl as usize;
    self.fi.base_q_idx = (self.fi.config.quantizer.min(255 - q_drop) + q_drop) as u8;

    let second_ref_frame = if !cfg.multiref {
      NONE_FRAME
    } else if !cfg.reorder || cfg.idx_in_group == 0 {
      LAST2_FRAME
    } else {
      ALTREF_FRAME
    };
    let ref_in_previous_group = LAST3_FRAME;

    // reuse probability estimates from previous frames only in top level frames
    self.fi.primary_ref_frame = if lvl > 0 { PRIMARY_REF_NONE } else { (ref_in_previous_group - LAST_FRAME) as u32 };

    for i in 0..INTER_REFS_PER_FRAME {
      self.fi.ref_frames[i] = if lvl == 0 {
        if i == second_ref_frame - LAST_FRAME {
          (slot_idx + 4 - 2) as u8 % 4
        } else {
          (slot_idx + 4 - 1) as u8 % 4
        }
      } else {
        if i == second_ref_frame - LAST_FRAME {
          let oh = self.fi.order_hint + (cfg.group_src_len as u32 >> lvl);
          let lvl2 = pos_to_lvl(oh as u64, cfg.pyramid_depth);
          if lvl2 == 0 {
            ((oh >> cfg.pyramid_depth) % 4) as u8
          } else {
            3 + lvl2 as u8
          }
        } else if i == ref_in_previous_group - LAST_FRAME {
          if lvl == 0 {
            (slot_idx + 4 - 1) as u8 % 4
          } else {
            slot_idx as u8
          }
        } else {
          let oh = self.fi.order_hint - (cfg.group_src_len as u32 >> lvl);
          let lvl1 = pos_to_lvl(oh as u64, cfg.pyramid_depth);
          if lvl1 == 0 {
            ((oh >> cfg.pyramid_depth) % 4) as u8
          } else {
            3 + lvl1 as u8
          }
        }
      }
    }

    self.fi.reference_mode = if cfg.multiref && cfg.reorder && cfg.idx_in_group != 0 {
      ReferenceMode::SELECT
    } else {
      ReferenceMode::SINGLE
    };
    self.fi.number = number;
    self.fi.me_range_scale = (cfg.group_src_len >> lvl) as u8;
    Ok(())
  }

  fn frame_properties(&mut self, idx: u64) -> Result<(), FramePropsStatus> {
    if idx == 0 {
      // The first frame will always be a key frame
      self.setup_key_frame_properties();
      return Ok(());
    }

    // Initially set up the frame as an inter frame.
    // We need to determine what the frame number is before we can
    // look up the frame type. If reordering is enabled, the idx
    // may not match the frame number.
    let idx_in_segment = idx - self.segment_start_idx;
    if idx_in_segment > 0 {
      let inter_props = self.get_inter_props_cfg(idx_in_segment);
      if let Err(FramePropsStatus::FramePastSegmentEnd) = self.setup_inter_frame_properties(&inter_props) {
        let start_frame = self.next_keyframe();
        if !inter_props.reorder || ((idx_in_segment - 1) % inter_props.group_len == 0 && self.fi.number == (start_frame - 1)) {
          self.segment_start_idx = idx;
          self.segment_start_frame = start_frame;
          self.fi.number = start_frame;
        } else {
          return Err(FramePropsStatus::FramePastSegmentEnd);
        }
      }
    }

    // Now that we know the frame number, look up the correct frame type
    let frame_type = self.frame_types.get(&self.fi.number).cloned();
    if let Some(frame_type) = frame_type {
      if frame_type == FrameType::KEY {
        self.segment_start_idx = idx;
        self.segment_start_frame = self.fi.number;
      }
      self.fi.frame_type = frame_type;

      let idx_in_segment = idx - self.segment_start_idx;
      if idx_in_segment == 0 {
        self.setup_key_frame_properties();
      } else {
        let inter_props = self.get_inter_props_cfg(idx_in_segment);
        self.setup_inter_frame_properties(&inter_props)?;
      }
    }
    Ok(())
  }

  pub fn receive_packet(&mut self) -> Result<Packet, EncoderStatus> {
    let mut idx = self.idx;
    while self.frame_properties(idx).is_err() {
      self.idx += 1;
      idx = self.idx;
    }

    if !self.needs_more_frames(self.fi.number) {
      self.idx += 1;
      return Err(EncoderStatus::EnoughData)
    }

    if self.fi.show_existing_frame {
      self.idx += 1;

      let mut fs = FrameState::new(&self.fi);

      let data = encode_frame(&mut self.seq, &mut self.fi, &mut fs);

      // TODO avoid the clone by having rec Arc.
      let rec = if self.fi.show_frame { Some(fs.rec.clone()) } else { None };
      let mut psnr = None;
      if self.fi.config.show_psnr {
        if let Some(ref rec) = rec {
          psnr = Some(calculate_frame_psnr(&*fs.input, rec, self.seq.bit_depth));
        }
      }

      Ok(Packet { data, rec, number: self.fi.number, frame_type: self.fi.frame_type, psnr })
    } else {
      if let Some(f) = self.frame_q.remove(&self.fi.number) {
        self.idx += 1;

        if let Some(frame) = f {
          let mut fs = FrameState::new_with_frame(&self.fi, frame.clone());

          let data = encode_frame(&mut self.seq, &mut self.fi, &mut fs);
          self.packet_data.extend(data);

          fs.rec.pad(self.fi.width, self.fi.height);

          // TODO avoid the clone by having rec Arc.
          let rec = if self.fi.show_frame { Some(fs.rec.clone()) } else { None };

          update_rec_buffer(&mut self.fi, fs);

          if self.fi.show_frame {
            let data = self.packet_data.clone();
            self.packet_data = Vec::new();

            let mut psnr = None;
            if self.fi.config.show_psnr {
              if let Some(ref rec) = rec {
                psnr = Some(calculate_frame_psnr(&*frame, rec, self.seq.bit_depth));
              }
            }

            Ok(Packet { data, rec, number: self.fi.number, frame_type: self.fi.frame_type, psnr })
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

  pub fn flush(&mut self) {
    self.frame_q.insert(self.frame_count, None);
    self.frame_count = self.frame_count + 1;
  }

  fn save_frame_type(&mut self, idx: u64) {
    let frame_type = self.determine_frame_type(idx);
    self.frame_types.insert(idx, frame_type);
  }

  fn determine_frame_type(&mut self, idx: u64) -> FrameType {
    if idx == 0 {
      return FrameType::KEY;
    }

    let prev_keyframe = *self.frame_types.iter().rfind(|(_, &ty)| ty == FrameType::KEY).unwrap().0;
    let frame = self.frame_q.get(&idx).cloned().unwrap();
    if let Some(frame) = frame {
      let distance = idx - prev_keyframe;
      if distance < self.fi.config.min_key_frame_interval {
        if distance + 1 == self.fi.config.min_key_frame_interval {
          // Run the detector for the current frame, so that it will contain this frame's information
          // to compare against the next frame. We can ignore the results for this frame.
          self.keyframe_detector.detect_scene_change(frame, idx as usize);
        }
        return FrameType::INTER;
      }
      if distance >= self.fi.config.max_key_frame_interval {
        return FrameType::KEY;
      }
      if self.keyframe_detector.detect_scene_change(frame, idx as usize) {
        return FrameType::KEY;
      }
    }
    FrameType::INTER
  }
}

impl fmt::Display for Context {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "Frame {} - {}", self.fi.number, self.fi.frame_type)
  }
}
