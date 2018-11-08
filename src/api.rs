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

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

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
  pub key_frame_interval: u64,
  pub low_latency: bool,
  pub quantizer: usize,
  pub tune: Tune,
  pub speed_settings: SpeedSettings,
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
      key_frame_interval: 30,
      low_latency: true,
      quantizer: 100,
      tune: Tune::Psnr,
      speed_settings: SpeedSettings::from_preset(speed)
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
  pub chroma_sampling: ChromaSampling
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
      "key_frame_interval" => self.enc.key_frame_interval = value.parse().map_err(|_e| ParseError)?,
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

    Context { fi, seq, frame_count: 0, idx: 0, frame_q: BTreeMap::new(), packet_data: Vec::new() }
  }
}

pub struct Context {
  fi: FrameInvariants,
  seq: Sequence,
  //    timebase: Rational,
  frame_count: u64,
  idx: u64,
  frame_q: BTreeMap<u64, Option<Arc<Frame>>>, //    packet_q: VecDeque<Packet>
  packet_data: Vec<u8>
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
  pub frame_type: FrameType
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
    Arc::new(Frame::new(self.fi.padded_w, self.fi.padded_h))
  }

  pub fn send_frame<F>(&mut self, frame: F) -> Result<(), EncoderStatus>
  where
    F: Into<Option<Arc<Frame>>>
  {
    self.frame_q.insert(self.frame_count, frame.into());
    self.frame_count = self.frame_count + 1;
    Ok(())
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

  pub fn frame_properties(&mut self, idx: u64) -> bool {
    let key_frame_interval: u64 = self.fi.config.key_frame_interval;

    let reorder = !self.fi.config.low_latency;
    let multiref = reorder || self.fi.config.speed_settings.multiref;

    let pyramid_depth = if reorder { 2 } else { 0 };
    let group_src_len = 1 << pyramid_depth;
    let group_len = group_src_len + if reorder { pyramid_depth } else { 0 };
    let segment_len = 1 + (key_frame_interval - 1 + group_src_len - 1) / group_src_len * group_len;

    let idx_in_segment = idx % segment_len;
    let segment_idx = idx / segment_len;

    if idx_in_segment == 0 {
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
      self.fi.number = segment_idx * key_frame_interval;
      for i in 0..INTER_REFS_PER_FRAME {
        self.fi.ref_frames[i] = 0;
      }
    } else {
      let idx_in_group = (idx_in_segment - 1) % group_len;
      let group_idx = (idx_in_segment - 1) / group_len;

      self.fi.frame_type = FrameType::INTER;
      self.fi.intra_only = false;

      self.fi.order_hint = (group_src_len * group_idx +
        if reorder && idx_in_group < pyramid_depth {
          group_src_len >> idx_in_group
        } else {
          idx_in_group - pyramid_depth + 1
        }) as u32;
      if self.fi.order_hint >= key_frame_interval as u32 {
        return false;
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

      let lvl = if !reorder {
        0
      } else if idx_in_group < pyramid_depth {
        idx_in_group
      } else {
        pos_to_lvl(idx_in_group - pyramid_depth + 1, pyramid_depth)
      };

      // Frames with lvl == 0 are stored in slots 0..4 and frames with higher values
      // of lvl in slots 4..8
      let slot_idx = if lvl == 0 {
        (self.fi.order_hint >> pyramid_depth) % 4 as u32
      } else {
        3 + lvl as u32
      };
      self.fi.show_frame = !reorder || idx_in_group >= pyramid_depth;
      self.fi.show_existing_frame = self.fi.show_frame && reorder &&
        (idx_in_group - pyramid_depth + 1).count_ones() == 1 &&
        idx_in_group != pyramid_depth;
      self.fi.frame_to_show_map_idx = slot_idx;
      self.fi.refresh_frame_flags = if self.fi.show_existing_frame {
        0
      } else {
        1 << slot_idx
      };

      let q_drop = 15 * lvl as usize;
      self.fi.base_q_idx = (self.fi.config.quantizer.min(255 - q_drop) + q_drop) as u8;

      let second_ref_frame = if !multiref {
        NONE_FRAME
      } else if !reorder || idx_in_group == 0 {
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
            let oh = self.fi.order_hint + (group_src_len as u32 >> lvl);
            let lvl2 = pos_to_lvl(oh as u64, pyramid_depth);
            if lvl2 == 0 {
              ((oh >> pyramid_depth) % 4) as u8
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
            let oh = self.fi.order_hint - (group_src_len as u32 >> lvl);
            let lvl1 = pos_to_lvl(oh as u64, pyramid_depth);
            if lvl1 == 0 {
              ((oh >> pyramid_depth) % 4) as u8
            } else {
              3 + lvl1 as u8
            }
          }
        }
      }

      self.fi.reference_mode = if multiref && reorder && idx_in_group != 0 {
        ReferenceMode::SELECT
      } else {
        ReferenceMode::SINGLE
      };
      self.fi.number = segment_idx * key_frame_interval + self.fi.order_hint as u64;
      self.fi.me_range_scale = (group_src_len >> lvl) as u8;
    }

    true
  }

  pub fn receive_packet(&mut self) -> Result<Packet, EncoderStatus> {
    let mut idx = self.idx;
    while !self.frame_properties(idx) {
      self.idx = self.idx + 1;
      idx = self.idx;
    }

    if self.fi.show_existing_frame {
      self.idx = self.idx + 1;

      let mut fs = FrameState::new(&self.fi);

      let data = encode_frame(&mut self.seq, &mut self.fi, &mut fs);

      // TODO avoid the clone by having rec Arc.
      let rec = if self.fi.show_frame { Some(fs.rec.clone()) } else { None };

      Ok(Packet { data, rec, number: self.fi.number, frame_type: self.fi.frame_type })
    } else {
      if let Some(f) = self.frame_q.remove(&self.fi.number) {
        self.idx = self.idx + 1;

        if let Some(frame) = f {
          let mut fs = FrameState::new_with_frame(&self.fi, frame);

          let data = encode_frame(&mut self.seq, &mut self.fi, &mut fs);
          self.packet_data.extend(data);

          fs.rec.pad();

          // TODO avoid the clone by having rec Arc.
          let rec = if self.fi.show_frame { Some(fs.rec.clone()) } else { None };

          update_rec_buffer(&mut self.fi, fs);

          if self.fi.show_frame {
            let data = self.packet_data.clone();
            self.packet_data = Vec::new();
            Ok(Packet { data, rec, number: self.fi.number, frame_type: self.fi.frame_type })
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
}

impl fmt::Display for Context {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "Frame {} - {}", self.fi.number, self.fi.frame_type)
  }
}
