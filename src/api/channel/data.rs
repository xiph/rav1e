// Copyright (c) 2018-2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.
#![allow(missing_docs)]

use crate::api::color::*;
use crate::api::config::EncoderConfig;
use crate::api::context::RcData;
use crate::api::util::*;
use crate::encoder::*;
use crate::frame::*;
use crate::util::Pixel;

use bitstream_io::*;
use crossbeam::channel::*;

use std::io;
use std::sync::Arc;

/// Endpoint to send previous-pass statistics data
pub struct RcDataSender {
  pub(crate) sender: Sender<RcData>,
  pub(crate) limit: u64,
  pub(crate) count: u64,
}

impl RcDataSender {
  pub(crate) fn new(limit: u64, sender: Sender<RcData>) -> RcDataSender {
    Self { sender, limit, count: 0 }
  }
  pub fn try_send(
    &mut self, data: RcData,
  ) -> Result<(), TrySendError<RcData>> {
    if self.limit <= self.count {
      Err(TrySendError::Full(data))
    } else {
      let r = self.sender.try_send(data);
      if r.is_ok() {
        self.count += 1;
      }
      r
    }
  }
  pub fn send(&mut self, data: RcData) -> Result<(), SendError<RcData>> {
    if self.limit <= self.count {
      Err(SendError(data))
    } else {
      let r = self.sender.send(data);
      if r.is_ok() {
        self.count += 1;
      }
      r
    }
  }
  pub fn len(&self) -> usize {
    self.sender.len()
  }
  pub fn is_empty(&self) -> bool {
    self.sender.is_empty()
  }

  // TODO: proxy more methods
}

/// Endpoint to receive current-pass statistics data
pub struct RcDataReceiver(pub(crate) Receiver<RcData>);

impl RcDataReceiver {
  pub fn try_recv(&self) -> Result<RcData, TryRecvError> {
    self.0.try_recv()
  }
  pub fn recv(&self) -> Result<RcData, RecvError> {
    self.0.recv()
  }
  pub fn len(&self) -> usize {
    self.0.len()
  }
  pub fn is_empty(&self) -> bool {
    self.0.is_empty()
  }
  pub fn iter(&self) -> Iter<RcData> {
    self.0.iter()
  }

  pub fn summary_size(&self) -> usize {
    crate::rate::TWOPASS_HEADER_SZ
  }
}

pub type PassDataChannel = (RcDataSender, RcDataReceiver);

pub type FrameInput<T> = (Option<Arc<Frame<T>>>, Option<FrameParameters>);

/// Endpoint to send frames
pub struct FrameSender<T: Pixel> {
  sender: Sender<FrameInput<T>>,
  config: Arc<EncoderConfig>,
  limit: u64,
  count: u64,
}

// Proxy the crossbeam Sender
//
// TODO: enforce the limit
impl<T: Pixel> FrameSender<T> {
  pub(crate) fn new(
    limit: u64, sender: Sender<FrameInput<T>>, config: Arc<EncoderConfig>,
  ) -> FrameSender<T> {
    Self { sender, config, limit, count: 0 }
  }
  pub fn try_send<F: IntoFrame<T>>(
    &mut self, frame: F,
  ) -> Result<(), TrySendError<FrameInput<T>>> {
    if self.limit <= self.count {
      Err(TrySendError::Full(frame.into()))
    } else {
      let r = self.sender.try_send(frame.into());
      if r.is_ok() {
        self.count += 1;
      }
      r
    }
  }

  pub fn send<F: IntoFrame<T>>(
    &mut self, frame: F,
  ) -> Result<(), SendError<FrameInput<T>>> {
    if self.limit <= self.count {
      Err(SendError(frame.into()))
    } else {
      let r = self.sender.send(frame.into());
      if r.is_ok() {
        self.count += 1;
      }
      r
    }
  }
  pub fn len(&self) -> usize {
    self.sender.len()
  }
  pub fn is_empty(&self) -> bool {
    self.sender.is_empty()
  }
  // TODO: proxy more methods
}

// Frame factory
impl<T: Pixel> FrameSender<T> {
  /// Helper to create a new frame with the current encoder configuration
  #[inline]
  pub fn new_frame(&self) -> Frame<T> {
    Frame::new(
      self.config.width,
      self.config.height,
      self.config.chroma_sampling,
    )
  }
}

/// Endpoint to receive packets
pub struct PacketReceiver<T: Pixel> {
  pub(crate) receiver: Receiver<Packet<T>>,
  pub(crate) config: Arc<EncoderConfig>,
}

impl<T: Pixel> PacketReceiver<T> {
  pub fn try_recv(&self) -> Result<Packet<T>, TryRecvError> {
    self.receiver.try_recv()
  }
  pub fn recv(&self) -> Result<Packet<T>, RecvError> {
    self.receiver.recv()
  }
  pub fn len(&self) -> usize {
    self.receiver.len()
  }
  pub fn is_empty(&self) -> bool {
    self.receiver.is_empty()
  }
  pub fn iter(&self) -> Iter<Packet<T>> {
    self.receiver.iter()
  }
}

impl<T: Pixel> PacketReceiver<T> {
  /// Produces a sequence header matching the current encoding context.
  ///
  /// Its format is compatible with the AV1 Matroska and ISOBMFF specification.
  /// Note that the returned header does not include any config OBUs which are
  /// required for some uses. See [the specification].
  ///
  /// [the specification]:
  /// https://aomediacodec.github.io/av1-isobmff/#av1codecconfigurationbox-section
  #[inline]
  pub fn container_sequence_header(&self) -> Vec<u8> {
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
}

/// A channel modeling an encoding process
pub type VideoDataChannel<T> = (FrameSender<T>, PacketReceiver<T>);
