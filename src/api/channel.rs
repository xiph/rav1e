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

use crate::encoder::*;
use crate::frame::*;
use crate::util::Pixel;

use std::io;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum PassData {
  Summary(Box<[u8]>),
  Frame(Box<[u8]>),
}

impl PassData {
  pub fn is_summary(&self) -> bool {
    match self {
      PassData::Summary(_) => true,
      _ => false,
    }
  }
}

impl AsRef<[u8]> for PassData {
  fn as_ref(&self) -> &[u8] {
    match self {
      PassData::Summary(s) => s.as_ref(),
      PassData::Frame(f) => f.as_ref(),
    }
  }
}

/// Endpoint to send previous-pass statistics data
pub struct PassDataSender(Sender<PassData>);

impl PassDataSender {
  pub fn try_send(
    &self, data: PassData,
  ) -> Result<(), TrySendError<PassData>> {
    self.0.try_send(data)
  }

  pub fn send(&self, data: PassData) -> Result<(), SendError<PassData>> {
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
pub struct PassDataReceiver(Receiver<PassData>);

impl PassDataReceiver {
  pub fn try_recv(&self) -> Result<PassData, TryRecvError> {
    self.0.try_recv()
  }
  pub fn recv(&self) -> Result<PassData, RecvError> {
    self.0.recv()
  }
  pub fn len(&self) -> usize {
    self.0.len()
  }
  pub fn is_empty(&self) -> bool {
    self.0.is_empty()
  }
  pub fn iter(&self) -> Iter<PassData> {
    self.0.iter()
  }
}

pub type PassDataChannel = (PassDataSender, PassDataReceiver);

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
  /// Create a single pass encoder channel
  ///
  /// Drop the `FrameSender<T>` endpoint to flush the encoder.
  pub fn new_channel<T: Pixel>(
    &self,
  ) -> Result<VideoDataChannel<T>, InvalidConfig> {
    let (send_frame, receive_frame) =
      bounded(self.enc.rdo_lookahead_frames as usize * 2); // TODO: user settable
    let (send_packet, receive_packet) = unbounded();

    let mut inner = self.new_inner()?;
    let pool = rayon::ThreadPoolBuilder::new()
      .num_threads(self.threads)
      .build()
      .unwrap();

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

  /// Create an encoder channel with first pass data reporting
  ///
  /// Drop the `Sender<F>` endpoint to flush the encoder.
  /// The last buffer in the Receiver<PassData> is the summary of the whole
  /// encoding process.
  pub fn new_firstpass_channel<T: Pixel>(
    &self,
  ) -> Result<(VideoDataChannel<T>, PassDataReceiver), InvalidConfig> {
    let (video, passdata) = self.new_multipass_channel()?;

    (video, passdata.1)
  }

  /// Create a multipass encoder channel
  ///
  /// To setup a first-pass encode drop the `PassDataSender` before sending the
  /// first Frame.
  ///
  /// The `PacketReceiver<T>` may block if not enough pass statistics data
  /// are sent through the `PassDataSender` endpoint
  ///
  /// Drop the `Sender<F>` endpoint to flush the encoder.
  /// The last buffer in the Receiver<PassData> is the summary of the whole
  /// encoding process.
  pub fn new_multipass_channel<T: Pixel>(
    &self,
  ) -> Result<(VideoDataChannel<T>, PassDataChannel), InvalidConfig> {
    todo!()
  }

  /*
    // TODO: make it user-settable
    let input_len = self.enc.rdo_lookahead_frames as usize * 2;

    let (send_frame, receive_frame) = bounded(input_len);
    let (send_packet, receive_packet) = unbounded();

    let (send_rc_pass1, receive_rc_pass1) = bounded(input_len);
    let (send_rc_pass2, receive_rc_pass2) = unbounded();

    let mut inner = self.new_inner()?;
    let pool = rayon::ThreadPoolBuilder::new()
      .num_threads(self.threads)
      .build()
      .unwrap();


    pool.spawn(move || {
      // Feed in the summary
      let second_pass = receive_rc_pass2
        .recv()
        .map(|data: PassData| {
          // check data.len() vs twopass_bytes_needed()
          let len = inner.rc_state.twopass_in(None).unwrap_or(0);
          println!("Summary {} vs {}", data.as_ref().len(), len);
          inner
            .rc_state
            .twopass_in(Some(data.as_ref()))
            .expect("Faulty pass data");
          true
        })
        .unwrap_or(false);

      // Send the placeholder data
      let first_pass = send_pass_summary(&mut inner, &send_rc_pass1);

      let second_pass_recv =
        if second_pass { Some(receive_rc_pass2) } else { None };

      let first_pass_send =
        if first_pass { Some(send_rc_pass1) } else { None };

      for f in receive_frame.iter() {
        // info!("frame in {}", inner.frame_count);
        while !inner.needs_more_fi_lookahead() {
          // needs_more_fi_lookahead() should guard for missing output_frameno
          // already.
          // this call should return either Ok or Err(Encoded)
          if let Ok(p) = receive_packet_impl(
            &mut inner,
            first_pass_send.as_ref(),
            second_pass_recv.as_ref(),
          ) {
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
        let r = receive_packet_impl(
          &mut inner,
          first_pass_send.as_ref(),
          second_pass_recv.as_ref(),
        );
        match r {
          Ok(p) => {
            // warn!("Sending out {}", p.input_frameno);
            send_packet.send(p).unwrap();
          }
          Err(EncoderStatus::LimitReached) => break,
          Err(EncoderStatus::Encoded) => {}
          Err(e) => panic!("got {:?}", e),
        }
      }

      // Final summary
      let _ = first_pass_send.map(|p| send_pass_summary(&mut inner, &p));

      // drop(send_packet); // This happens implicitly
    });

    Ok((
      (FrameSender(send_frame), PacketReceiver(receive_packet)),
      (PassDataSender(send_rc_pass2), PassDataReceiver(receive_rc_pass1)),
    ))
  }
  */
}
