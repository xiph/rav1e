// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use encoder::*;
use context::CDFContext;
use partition::*;

use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;

// TODO: use the num crate?
#[derive(Clone, Copy, Debug)]
pub struct Ratio {
  pub num: usize,
  pub den: usize
}

impl Ratio {
  pub fn new(num: usize, den: usize) -> Self {
    Ratio { num, den }
  }
}

/// Here we store all the information we might receive from the cli
#[derive(Clone, Copy, Debug)]
pub struct Config {
  pub width: usize,
  pub height: usize,
  pub bit_depth: usize,
  pub chroma_sampling: ChromaSampling,
  pub timebase: Ratio,
  pub enc: EncoderConfig
}

impl Config {
  pub fn new_context(&self) -> Context {
    let fi = FrameInvariants::new(self.width, self.height, self.enc.clone());
    let seq = Sequence::new(
      self.width,
      self.height,
      self.bit_depth,
      self.chroma_sampling
    );

    unsafe {
        av1_rtcd();
        aom_dsp_rtcd();
    }

    Context { fi, seq, frame_count: 0, idx: 0, frame_q: VecDeque::new() }
  }
}

pub struct Context {
  fi: FrameInvariants,
  seq: Sequence,
  //    timebase: Ratio,
  frame_count: u64,
  idx: u64,
  frame_q: VecDeque<(u64, Option<Arc<Frame>>)> //    packet_q: VecDeque<Packet>
}

#[derive(Clone, Copy, Debug)]
pub enum EncoderStatus {
  /// The encoder needs more Frames to produce an output Packet
  NeedMoreData,
  /// There are enough Frames queue
  EnoughData,
  ///
  Failure
}

pub struct Packet {
  pub data: Vec<u8>,
  pub rec: Option<Frame>,
  pub number: u64,
  pub frame_type: FrameType
}

impl fmt::Display for Packet {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "Frame {} - {} - {} bytes", self.number, self.frame_type, self.data.len())
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
    self.frame_q.push_back((self.frame_count, frame.into()));
    self.frame_count = self.frame_count + 1;
    Ok(())
  }

  pub fn frame_properties(&mut self, idx: u64) {
    let key_frame_interval: u64 = 30;

    let num_hidden_frames_in_segment: u64 = (key_frame_interval + 1) / 4;
    let segment_len = key_frame_interval + num_hidden_frames_in_segment;

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
      let idx_in_group = (idx_in_segment - 1) % 5;
      let group_idx = (idx_in_segment - 1) / 5;

      self.fi.frame_type = FrameType::INTER;
      self.fi.intra_only = false;
      self.fi.order_hint = (4 * group_idx + if idx_in_group == 0 { 4 } else { idx_in_group }) as u32;
      let slot_idx = self.fi.order_hint % REF_FRAMES as u32;
      self.fi.show_frame = idx_in_group != 0;
      self.fi.show_existing_frame = idx_in_group == 4;
      self.fi.frame_to_show_map_idx = slot_idx;
      self.fi.refresh_frame_flags = if self.fi.show_existing_frame {
        0
      } else {
        1 << slot_idx
      };
      let high_quality_frame = idx_in_group % 4 == 0;
      self.fi.base_q_idx = if high_quality_frame {
        self.fi.config.quantizer.max(1).min(255)
      } else {
        let q_drop = 15;
        self.fi.config.quantizer.min(255 - q_drop) + q_drop
      } as u8;
      let first_ref_frame = LAST_FRAME;
      let second_ref_frame = ALTREF_FRAME;

      self.fi.primary_ref_frame = (first_ref_frame - LAST_FRAME) as u32;

      for i in 0..INTER_REFS_PER_FRAME {
        self.fi.ref_frames[i] = if i == second_ref_frame - LAST_FRAME {
          (slot_idx as usize + 3) & 4
        } else {
          (slot_idx as usize + 7) & 4
        };
      }

      if self.fi.order_hint >= key_frame_interval as u32 {
        assert!(idx_in_group == 0);
        self.fi.order_hint = key_frame_interval as u32 - 1;
        self.fi.show_frame = key_frame_interval == 4 * group_idx + 2;
      }
      self.fi.number = segment_idx * key_frame_interval + self.fi.order_hint as u64;
    }
  }

  pub fn receive_packet(&mut self) -> Result<Packet, EncoderStatus> {
    let idx = self.idx;
    self.frame_properties(idx);

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
      let mut j: Option<usize> = None;
      for (i, f) in self.frame_q.iter().enumerate() {
        if f.0 == self.fi.number {
          j = Some(i);
        }
      }
      if let Some(k) = j {
        self.idx = self.idx + 1;

        let f = self.frame_q.remove(k).unwrap();

        if let Some(frame) = f.1 {
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
    self.frame_q.push_back((self.frame_count, None));
    self.frame_count = self.frame_count + 1;
  }
}

impl fmt::Display for Context {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "Frame {} - {}", self.fi.number, self.fi.frame_type)
  }
}
