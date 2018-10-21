// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use context::CDFContext;
use encoder::*;
use partition::*;

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

// TODO: use the num crate?
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Ratio {
  pub num: usize,
  pub den: usize
}

impl Ratio {
  pub fn new(num: usize, den: usize) -> Self {
    Ratio { num, den }
  }
}

#[derive(Copy, Clone, Debug)]
pub struct EncoderConfig {
  pub quantizer: usize,
  pub speed: usize,
  pub tune: Tune
}

impl Default for EncoderConfig {
  fn default() -> Self {
    EncoderConfig { quantizer: 100, speed: 0, tune: Tune::Psnr }
  }
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
  pub timebase: Ratio,
  pub enc: EncoderConfig
}

impl Config {
  pub fn parse(&mut self, key: &str, value: &str) -> Result<(), EncoderStatus> {
    use self::EncoderStatus::*;
    match key {
        "quantizer" => self.enc.quantizer = value.parse().map_err(|_e| ParseError)?,
        "speed" => self.enc.speed = value.parse().map_err(|_e| ParseError)?,
        "tune" => self.enc.tune = value.parse().map_err(|_e| ParseError)?,
        _ => return Err(InvalidKey)
    }

    Ok(())
  }

  pub fn new_context(&self) -> Context {
    let fi = FrameInvariants::new(
      self.frame_info.width,
      self.frame_info.height,
      self.enc.clone()
    );
    let seq = Sequence::new(&self.frame_info);

    unsafe {
      av1_rtcd();
      aom_dsp_rtcd();
    }

    Context { fi, seq, frame_count: 0, idx: 0, frame_q: BTreeMap::new() }
  }
}

pub struct Context {
  fi: FrameInvariants,
  seq: Sequence,
  //    timebase: Ratio,
  frame_count: u64,
  idx: u64,
  frame_q: BTreeMap<u64, Option<Arc<Frame>>> //    packet_q: VecDeque<Packet>
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
    let key_frame_interval: u64 = 30;

    let reorder = false;
    let multiref = reorder || self.fi.config.speed <= 2;

    let pyramid_depth = if reorder { 1 } else { 0 };
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

      let slot_idx = self.fi.order_hint % REF_FRAMES as u32;
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

      let lvl = if !reorder {
        0
      } else if idx_in_group < pyramid_depth {
        idx_in_group
      } else {
        pyramid_depth - (idx_in_group - pyramid_depth + 1).trailing_zeros() as u64
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

      self.fi.primary_ref_frame = (ref_in_previous_group - LAST_FRAME) as u32;

      assert!(group_src_len <= REF_FRAMES as u64);
      for i in 0..INTER_REFS_PER_FRAME {
        self.fi.ref_frames[i] = (slot_idx as u8 +
        if i == second_ref_frame - LAST_FRAME {
          if lvl == 0 { (REF_FRAMES as u64 - 2) * group_src_len } else { group_src_len >> lvl }
        } else if i == ref_in_previous_group - LAST_FRAME {
          REF_FRAMES as u64 - group_src_len
        } else {
          REF_FRAMES as u64 - (group_src_len >> lvl)
        } as u8) & 7;
      }

      self.fi.number = segment_idx * key_frame_interval + self.fi.order_hint as u64;
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

      let mut fs = FrameState {
        input: Arc::new(Frame::new(self.fi.padded_w, self.fi.padded_h)), // dummy
        rec: Frame::new(self.fi.padded_w, self.fi.padded_h),
        qc: Default::default(),
        cdfs: CDFContext::new(0),
        deblock: Default::default(),
      };

      let data = encode_frame(&mut self.seq, &mut self.fi, &mut fs);

      // TODO avoid the clone by having rec Arc.
      let rec = if self.fi.show_frame { Some(fs.rec.clone()) } else { None };

      Ok(Packet { data, rec, number: self.fi.number, frame_type: self.fi.frame_type })
    } else {
      if let Some(f) = self.frame_q.remove(&self.fi.number) {
        self.idx = self.idx + 1;

        if let Some(frame) = f {
          let mut fs = FrameState {
            input: frame,
            rec: Frame::new(self.fi.padded_w, self.fi.padded_h),
            qc: Default::default(),
            cdfs: CDFContext::new(0),
            deblock: Default::default(),
          };

          let data = encode_frame(&mut self.seq, &mut self.fi, &mut fs);

          fs.rec.pad();

          // TODO avoid the clone by having rec Arc.
          let rec = if self.fi.show_frame { Some(fs.rec.clone()) } else { None };

          update_rec_buffer(&mut self.fi, fs);

          Ok(Packet { data, rec, number: self.fi.number, frame_type: self.fi.frame_type })
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
