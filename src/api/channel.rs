// Copyright (c) 2018-2019, The rav1e contributors. All rights reserved
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
use crate::api::internal::ContextInner;
use crate::api::util::*;

use bitstream_io::*;
use crossbeam::channel::*;

use crate::api::context::RcData;
use crate::encoder::*;
use crate::frame::*;
use crate::rate::RCState;
use crate::rayon::{ThreadPool, ThreadPoolBuilder};
use crate::util::Pixel;

use std::io;
use std::sync::Arc;

/// Endpoint to send previous-pass statistics data
pub struct RcDataSender(Sender<RcData>);

impl RcDataSender {
  pub fn try_send(&self, data: RcData) -> Result<(), TrySendError<RcData>> {
    self.0.try_send(data)
  }

  pub fn send(&self, data: RcData) -> Result<(), SendError<RcData>> {
    self.0.send(data)
  }
  pub fn len(&self) -> usize {
    self.0.len()
  }
  pub fn is_empty(&self) -> bool {
    self.0.is_empty()
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
  enc: Arc<EncoderConfig>,
}

// Proxy the crossbeam Sender
impl<T: Pixel> FrameSender<T> {
  pub fn try_send<F: IntoFrame<T>>(
    &self, frame: F,
  ) -> Result<(), TrySendError<FrameInput<T>>> {
    self.sender.try_send(frame.into())
  }

  pub fn send<F: IntoFrame<T>>(
    &self, frame: F,
  ) -> Result<(), SendError<FrameInput<T>>> {
    self.sender.send(frame.into())
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
    Frame::new(self.enc.width, self.enc.height, self.enc.chroma_sampling)
  }
}

/// Endpoint to receive packets
pub struct PacketReceiver<T: Pixel> {
  receiver: Receiver<Packet<T>>,
  enc: Arc<EncoderConfig>,
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

    let seq = Sequence::new(&self.enc);

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

  /// Create a single pass encoder channel
  ///
  /// Drop the `FrameSender<T>` endpoint to flush the encoder.
  pub fn new_channel<T: Pixel>(
    &self,
  ) -> Result<VideoDataChannel<T>, InvalidConfig> {
    let (mut inner, pool) = self.setup()?;

    let (send_frame, receive_frame) =
      bounded(self.enc.rdo_lookahead_frames as usize * 2); // TODO: user settable
    let (send_packet, receive_packet) = unbounded();

    let enc = Arc::new(self.enc);

    let channel = (
      FrameSender { sender: send_frame, enc: enc.clone() },
      PacketReceiver { receiver: receive_packet, enc },
    );

    pool.spawn(move || {
      for f in receive_frame.iter() {
        // info!("frame in {}", inner.frame_count);
        while !inner.needs_more_fi_lookahead() {
          // needs_more_fi_lookahead() should guard for missing output_frameno
          // already.
          // this call should return either Ok or Err(Encoded)
          if let Ok(p) = inner.receive_packet() {
            // warn!("packet out {}", p.input_frameno);
            send_packet.send(p).unwrap();
          }
        }

        let (frame, params) = f;
        let _ = inner.send_frame(frame, params); // TODO make sure it cannot fail.
      }

      inner.limit = Some(inner.frame_count);
      let _ = inner.send_frame(None, None);

      loop {
        let r = inner.receive_packet();
        match r {
          Ok(p) => {
            // warn!("Sending out {}", p.input_frameno);
            send_packet.send(p).unwrap();
          }
          Err(EncoderStatus::LimitReached) => break,
          Err(EncoderStatus::Encoded) => {}
          _ => unreachable!(),
        }
      }

      // drop(send_packet); // This happens implicitly
    });

    Ok(channel)
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

    let (mut inner, pool) = self.setup()?;

    // TODO: make it user-settable
    let input_len = self.enc.rdo_lookahead_frames as usize * 2;

    let (send_frame, receive_frame) = bounded(input_len);
    let (send_packet, receive_packet) = unbounded();
    let (mut send_rc_pass1, receive_rc_pass1) = unbounded();

    let enc = Arc::new(self.enc);

    let channel = (
      FrameSender { sender: send_frame, enc: enc.clone() },
      PacketReceiver { receiver: receive_packet, enc },
    );

    fn send_pass_data(rc_state: &mut RCState, send_rc: &mut Sender<RcData>) {
      if let Some(data) = rc_state.emit_frame_data() {
        let data = data.to_vec().into_boxed_slice();
        send_rc.send(RcData::Frame(data)).unwrap();
      } else {
        unreachable!(
          "The encoder received more frames than its internal
              limit allows"
        );
      }
    }

    fn send_pass_summary(
      rc_state: &mut RCState, send_rc: &mut Sender<RcData>,
    ) {
      let data = rc_state.emit_summary();
      let data = data.to_vec().into_boxed_slice();
      send_rc.send(RcData::Frame(data)).unwrap();
    }

    pool.spawn(move || {
      for f in receive_frame.iter() {
        // info!("frame in {}", inner.frame_count);
        while !inner.needs_more_fi_lookahead() {
          // needs_more_fi_lookahead() should guard for missing output_frameno
          // already.
          // this call should return either Ok or Err(Encoded)
          let has_pass_data = match inner.receive_packet() {
            Ok(p) => {
              send_packet.send(p).unwrap();
              true
            }
            Err(EncoderStatus::Encoded) => true,
            _ => unreachable!(),
          };

          if has_pass_data {
            send_pass_data(&mut inner.rc_state, &mut send_rc_pass1);
          }
        }

        let (frame, params) = f;
        let _ = inner.send_frame(frame, params); // TODO make sure it cannot fail.
      }

      inner.limit = Some(inner.frame_count);
      let _ = inner.send_frame(None, None);

      loop {
        let r = inner.receive_packet();
        let has_pass_data = match r {
          Ok(p) => {
            // warn!("Sending out {}", p.input_frameno);
            send_packet.send(p).unwrap();
            true
          }
          Err(EncoderStatus::LimitReached) => break,
          Err(EncoderStatus::Encoded) => true,
          _ => unreachable!(),
        };

        if has_pass_data {
          send_pass_data(&mut inner.rc_state, &mut send_rc_pass1);
        }
      }

      send_pass_summary(&mut inner.rc_state, &mut send_rc_pass1);
    });

    Ok((channel, RcDataReceiver(receive_rc_pass1)))
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
    if !rc.summary.is_none() {
      return Err(InvalidConfig::RateControlConfigurationMismatch);
    }

    // The inner context is already configured to use the summary at this point.
    let (mut inner, pool) = self.setup()?;

    // TODO: make it user-settable
    let input_len = self.enc.rdo_lookahead_frames as usize * 2;

    let (send_frame, receive_frame) = bounded(input_len);
    let (send_packet, receive_packet) = unbounded();
    let (send_rc_pass2, mut receive_rc_pass2) = unbounded();

    let enc = Arc::new(self.enc);

    let channel = (
      FrameSender { sender: send_frame, enc: enc.clone() },
      PacketReceiver { receiver: receive_packet, enc },
    );

    fn feed_pass_data<T: Pixel>(
      inner: &mut ContextInner<T>, recv: &mut Receiver<RcData>,
    ) -> Result<(), ()> {
      while inner.rc_state.twopass_in_frames_needed() > 0
        && inner.done_processing()
      {
        if let Ok(RcData::Frame(data)) = recv.recv() {
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

    pool.spawn(move || {
      for f in receive_frame.iter() {
        // info!("frame in {}", inner.frame_count);
        while !inner.needs_more_fi_lookahead() {
          feed_pass_data(&mut inner, &mut receive_rc_pass2).unwrap();
          // needs_more_fi_lookahead() should guard for missing output_frameno
          // already.
          //
          // this call should return either Ok or Err(Encoded)
          match inner.receive_packet() {
            Ok(p) => {
              send_packet.send(p).unwrap();
            }
            Err(EncoderStatus::Encoded) => {}
            Err(EncoderStatus::NotReady) => todo!("Error reporting"),
            _ => unreachable!(),
          };
        }

        let (frame, params) = f;
        let _ = inner.send_frame(frame, params); // TODO make sure it cannot fail.
      }

      inner.limit = Some(inner.frame_count);
      let _ = inner.send_frame(None, None);

      loop {
        feed_pass_data(&mut inner, &mut receive_rc_pass2).unwrap();
        let r = inner.receive_packet();
        match r {
          Ok(p) => {
            // warn!("Sending out {}", p.input_frameno);
            send_packet.send(p).unwrap();
          }
          Err(EncoderStatus::LimitReached) => break,
          Err(EncoderStatus::Encoded) => {}
          _ => unreachable!(),
        };
      }
    });

    Ok((channel, RcDataSender(send_rc_pass2)))
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
    if !rc.summary.is_none() || !rc.emit_pass_data {
      return Err(InvalidConfig::RateControlConfigurationMismatch);
    }

    // The inner context is already configured to use the summary at this point.
    let (mut inner, pool) = self.setup()?;

    // TODO: make it user-settable
    let input_len = self.enc.rdo_lookahead_frames as usize * 2;

    let (send_frame, receive_frame) = bounded(input_len);
    let (send_packet, receive_packet) = unbounded();
    let (mut send_rc_pass1, receive_rc_pass1) = unbounded();
    let (send_rc_pass2, mut receive_rc_pass2) = unbounded();

    let enc = Arc::new(self.enc);

    let channel = (
      FrameSender { sender: send_frame, enc: enc.clone() },
      PacketReceiver { receiver: receive_packet, enc },
    );

    let pass_channel =
      (RcDataSender(send_rc_pass2), RcDataReceiver(receive_rc_pass1));

    fn send_pass_data(rc_state: &mut RCState, send_rc: &mut Sender<RcData>) {
      if let Some(data) = rc_state.emit_frame_data() {
        let data = data.to_vec().into_boxed_slice();
        send_rc.send(RcData::Frame(data)).unwrap();
      } else {
        unreachable!(
          "The encoder received more frames than its internal
              limit allows"
        );
      }
    }

    fn send_pass_summary(
      rc_state: &mut RCState, send_rc: &mut Sender<RcData>,
    ) {
      let data = rc_state.emit_summary();
      let data = data.to_vec().into_boxed_slice();
      send_rc.send(RcData::Frame(data)).unwrap();
    }

    fn feed_pass_data<T: Pixel>(
      inner: &mut ContextInner<T>, recv: &mut Receiver<RcData>,
    ) -> Result<(), ()> {
      while inner.rc_state.twopass_in_frames_needed() > 0
        && inner.done_processing()
      {
        if let Ok(RcData::Frame(data)) = recv.recv() {
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

    pool.spawn(move || {
      for f in receive_frame.iter() {
        // info!("frame in {}", inner.frame_count);
        while !inner.needs_more_fi_lookahead() {
          feed_pass_data(&mut inner, &mut receive_rc_pass2).unwrap();
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
            send_pass_data(&mut inner.rc_state, &mut send_rc_pass1);
          }
        }

        let (frame, params) = f;
        let _ = inner.send_frame(frame, params); // TODO make sure it cannot fail.
      }

      inner.limit = Some(inner.frame_count);
      let _ = inner.send_frame(None, None);

      loop {
        feed_pass_data(&mut inner, &mut receive_rc_pass2).unwrap();
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
          send_pass_data(&mut inner.rc_state, &mut send_rc_pass1);
        }
      }

      send_pass_summary(&mut inner.rc_state, &mut send_rc_pass1);
    });

    Ok((channel, pass_channel))
  }
}
