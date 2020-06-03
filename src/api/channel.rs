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
use crate::api::config::*;
use crate::api::context::RcData;
use crate::api::internal::ContextInner;
use crate::api::util::*;

use bitstream_io::*;
use crossbeam::channel::*;

use crate::encoder::*;
use crate::frame::*;
use crate::rate::RCState;
use crate::rayon::{ThreadPool, ThreadPoolBuilder};
use crate::util::Pixel;

use std::io;
use std::sync::Arc;

/// Endpoint to send previous-pass statistics data
pub struct RcDataSender {
  sender: Sender<RcData>,
  limit: u64,
  count: u64,
}

impl RcDataSender {
  fn new(limit: u64, sender: Sender<RcData>) -> RcDataSender {
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
pub struct RcDataReceiver(Receiver<RcData>);

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
  fn new(
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
  receiver: Receiver<Packet<T>>,
  config: Arc<EncoderConfig>,
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

impl Config {
  fn setup<T: Pixel>(
    &self,
  ) -> Result<(ContextInner<T>, Arc<ThreadPool>), InvalidConfig> {
    self.validate()?;
    let inner = self.new_inner()?;

    let pool = if let Some(ref p) = self.pool {
      p.clone()
    } else {
      let pool =
        ThreadPoolBuilder::new().num_threads(self.threads).build().unwrap();
      Arc::new(pool)
    };

    Ok((inner, pool))
  }
}

impl Config {
  /// Create a single pass encoder channel
  ///
  /// Drop the `FrameSender<T>` endpoint to flush the encoder.
  pub fn new_channel<T: Pixel>(
    &self,
  ) -> Result<VideoDataChannel<T>, InvalidConfig> {
    let rc = &self.rate_control;

    if rc.emit_pass_data || rc.summary.is_some() {
      return Err(InvalidConfig::RateControlConfigurationMismatch);
    }

    let (v, _) = self.new_channel_internal()?;

    Ok(v)
  }

  /// Create a first pass encoder channel
  ///
  /// The pass data information is available throguht
  ///
  /// Drop the `FrameSender<T>` endpoint to flush the encoder.
  /// The last buffer in the PassDataReceiver is the summary of the whole
  /// encoding process.
  pub fn new_firstpass_channel<T: Pixel>(
    &self,
  ) -> Result<(VideoDataChannel<T>, RcDataReceiver), InvalidConfig> {
    let rc = &self.rate_control;

    if !rc.emit_pass_data {
      return Err(InvalidConfig::RateControlConfigurationMismatch);
    }
    let (v, (_, r)) = self.new_channel_internal()?;

    Ok((v, r.unwrap()))
  }

  /// Create a second pass encoder channel
  ///
  /// The encoding process require both frames and pass data to progress.
  ///
  /// Drop the `FrameSender<T>` endpoint to flush the encoder.
  pub fn new_secondpass_channel<T: Pixel>(
    &self,
  ) -> Result<(VideoDataChannel<T>, RcDataSender), InvalidConfig> {
    let rc = &self.rate_control;
    if rc.emit_pass_data || rc.summary.is_none() {
      return Err(InvalidConfig::RateControlConfigurationMismatch);
    }

    let (v, (s, _)) = self.new_channel_internal()?;

    Ok((v, s.unwrap()))
  }

  /// Create a multipass encoder channel
  ///
  /// The `PacketReceiver<T>` may block if not enough pass statistics data
  /// are sent through the `PassDataSender` endpoint
  ///
  /// Drop the `FrameSender<T>` endpoint to flush the encoder.
  /// The last buffer in the PassDataReceiver is the summary of the whole
  /// encoding process.
  pub fn new_multipass_channel<T: Pixel>(
    &self,
  ) -> Result<(VideoDataChannel<T>, PassDataChannel), InvalidConfig> {
    let rc = &self.rate_control;
    if rc.summary.is_none() || !rc.emit_pass_data {
      return Err(InvalidConfig::RateControlConfigurationMismatch);
    }

    let (v, (s, r)) = self.new_channel_internal()?;

    Ok((v, (s.unwrap(), r.unwrap())))
  }
}

trait RcFirstPass {
  fn send_pass_data(&mut self, rc_state: &mut RCState);
  fn send_pass_summary(&mut self, rc_state: &mut RCState);
}

trait RcSecondPass {
  fn feed_pass_data<T: Pixel>(
    &mut self, inner: &mut ContextInner<T>,
  ) -> Result<(), ()>;
}

impl RcFirstPass for Sender<RcData> {
  fn send_pass_data(&mut self, rc_state: &mut RCState) {
    if let Some(data) = rc_state.emit_frame_data() {
      let data = data.to_vec().into_boxed_slice();
      self.send(RcData::Frame(data)).unwrap();
    } else {
      unreachable!(
        "The encoder received more frames than its internal
              limit allows"
      );
    }
  }
  fn send_pass_summary(&mut self, rc_state: &mut RCState) {
    let data = rc_state.emit_summary();
    let data = data.to_vec().into_boxed_slice();
    self.send(RcData::Summary(data)).unwrap();
  }
}

impl RcFirstPass for Option<Sender<RcData>> {
  fn send_pass_data(&mut self, rc_state: &mut RCState) {
    match self.as_mut() {
      Some(s) => s.send_pass_data(rc_state),
      None => {}
    }
  }
  fn send_pass_summary(&mut self, rc_state: &mut RCState) {
    match self.as_mut() {
      Some(s) => s.send_pass_summary(rc_state),
      None => {}
    }
  }
}

impl RcSecondPass for Receiver<RcData> {
  fn feed_pass_data<T: Pixel>(
    &mut self, inner: &mut ContextInner<T>,
  ) -> Result<(), ()> {
    while inner.rc_state.twopass_in_frames_needed() > 0
      && !inner.done_processing()
    {
      if let Ok(RcData::Frame(data)) = self.recv() {
        inner
          .rc_state
          .parse_frame_data_packet(data.as_ref())
          .unwrap_or_else(|_| todo!("Error reporting"));
      } else {
        todo!("Error reporting");
      }
    }

    Ok(())
  }
}

impl RcSecondPass for Option<Receiver<RcData>> {
  fn feed_pass_data<T: Pixel>(
    &mut self, inner: &mut ContextInner<T>,
  ) -> Result<(), ()> {
    match self.as_mut() {
      Some(s) => s.feed_pass_data(inner),
      None => Ok(()),
    }
  }
}

impl Config {
  fn new_channel_internal<T: Pixel>(
    &self,
  ) -> Result<
    (VideoDataChannel<T>, (Option<RcDataSender>, Option<RcDataReceiver>)),
    InvalidConfig,
  > {
    // The inner context is already configured to use the summary at this point.
    let (mut inner, pool) = self.setup()?;

    // TODO: make it user-settable
    let input_len = self.enc.rdo_lookahead_frames as usize * 2;

    let (send_frame, receive_frame) = bounded(input_len);
    let (send_packet, receive_packet) = unbounded();

    let rc = &self.rate_control;

    let (mut send_rc_pass1, rc_data_receiver) = if rc.emit_pass_data {
      let (send_rc_pass1, receive_rc_pass1) = unbounded();
      (Some(send_rc_pass1), Some(RcDataReceiver(receive_rc_pass1)))
    } else {
      (None, None)
    };

    let (rc_data_sender, mut receive_rc_pass2, frame_limit) = if rc
      .summary
      .is_some()
    {
      let (frame_limit, pass_limit) =
        rc.summary.as_ref().map(|s| (s.ntus as u64, s.total as u64)).unwrap();

      inner.limit = Some(frame_limit);

      let (send_rc_pass2, receive_rc_pass2) = unbounded();

      (
        Some(RcDataSender::new(pass_limit, send_rc_pass2)),
        Some(receive_rc_pass2),
        frame_limit,
      )
    } else {
      (None, None, std::i32::MAX as u64)
    };

    let config = Arc::new(self.enc);

    let channel = (
      FrameSender::new(frame_limit, send_frame, config.clone()),
      PacketReceiver { receiver: receive_packet, config },
    );

    let pass_channel = (rc_data_sender, rc_data_receiver);

    pool.spawn(move || {
      for f in receive_frame.iter() {
        // info!("frame in {}", inner.frame_count);
        while !inner.needs_more_fi_lookahead() {
          receive_rc_pass2.feed_pass_data(&mut inner).unwrap();
          // needs_more_fi_lookahead() should guard for missing output_frameno
          // already.
          //
          // this call should return either Ok or Err(Encoded)
          let has_pass_data = match inner.receive_packet() {
            Ok(p) => {
              send_packet.send(p).unwrap();
              true
            }
            Err(EncoderStatus::Encoded) => true,
            Err(EncoderStatus::NotReady) => todo!("Error reporting"),
            _ => unreachable!(),
          };
          if has_pass_data {
            send_rc_pass1.send_pass_data(&mut inner.rc_state);
          }
        }

        let (frame, params) = f;
        let _ = inner.send_frame(frame, params); // TODO make sure it cannot fail.
      }

      inner.limit = Some(inner.frame_count);
      let _ = inner.send_frame(None, None);

      loop {
        receive_rc_pass2.feed_pass_data(&mut inner).unwrap();
        let r = inner.receive_packet();
        let has_pass_data = match r {
          Ok(p) => {
            // warn!("Sending out {}", p.input_frameno);
            send_packet.send(p).unwrap();
            true
          }
          Err(EncoderStatus::LimitReached) => break,
          Err(EncoderStatus::Encoded) => true,
          Err(EncoderStatus::NotReady) => todo!("Error reporting"),
          _ => unreachable!(),
        };

        if has_pass_data {
          send_rc_pass1.send_pass_data(&mut inner.rc_state);
        }
      }

      send_rc_pass1.send_pass_summary(&mut inner.rc_state);
    });

    Ok((channel, pass_channel))
  }
}
