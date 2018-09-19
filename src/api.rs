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
use partition::LAST_FRAME;

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

    Context { fi, seq, frame_q: VecDeque::new() }
  }
}

pub struct Context {
  fi: FrameInvariants,
  seq: Sequence,
  //    timebase: Ratio,
  frame_q: VecDeque<Option<Arc<Frame>>> //    packet_q: VecDeque<Packet>
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
  pub rec: Frame,
  pub number: usize,
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
    self.frame_q.push_back(frame.into());
    Ok(())
  }

  pub fn receive_packet(&mut self) -> Result<Packet, EncoderStatus> {
    let f = self.frame_q.pop_front().ok_or(EncoderStatus::NeedMoreData)?;
    if let Some(frame) = f {
      let mut fs = FrameState {
        input: frame,
        rec: Frame::new(self.fi.padded_w, self.fi.padded_h),
        qc: Default::default(),
        cdfs: CDFContext::new(0),
        deblock: Default::default(),
      };

      let frame_number_in_segment = self.fi.number % 30;

      self.fi.order_hint = frame_number_in_segment as u32;

      self.fi.frame_type = if frame_number_in_segment == 0 {
        FrameType::KEY
      } else {
        FrameType::INTER
      };

      self.fi.refresh_frame_flags = if self.fi.frame_type == FrameType::KEY {
        ALL_REF_FRAMES_MASK
      } else {
        1
      };

      self.fi.intra_only = self.fi.frame_type == FrameType::KEY
        || self.fi.frame_type == FrameType::INTRA_ONLY;
      // self.fi.use_prev_frame_mvs =
      //  !(self.fi.intra_only || self.fi.error_resilient);

      self.fi.base_q_idx = if self.fi.frame_type == FrameType::KEY {
        let q_boost = 15;
        self.fi.config.quantizer.max(1 + q_boost).min(255 + q_boost) - q_boost
      } else {
        self.fi.config.quantizer.max(1).min(255)
      } as u8;

      self.fi.primary_ref_frame = if self.fi.intra_only || self.fi.error_resilient {
        PRIMARY_REF_NONE
      } else {
        (LAST_FRAME - LAST_FRAME) as u32
      };

      let data = encode_frame(&mut self.seq, &mut self.fi, &mut fs);

      let number = self.fi.number as usize;

      self.fi.number += 1;

      fs.rec.pad();

      // TODO avoid the clone by having rec Arc.
      let rec = fs.rec.clone();

      update_rec_buffer(&mut self.fi, fs);

      Ok(Packet { data, rec, number, frame_type: self.fi.frame_type })
    } else {
      unimplemented!("Flushing not implemented")
    }
  }

  pub fn flush(&mut self) {
    self.frame_q.push_back(None);
  }
}

impl fmt::Display for Context {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "Frame {} - {}", self.fi.number, self.fi.frame_type)
  }
}
