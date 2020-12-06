// Copyright (c) 2018-2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.
#![allow(missing_docs)]

use crate::api::config::*;
use crate::api::context::RcData;
use crate::api::internal::ContextInner;
use crate::api::util::*;

use crossbeam::channel::*;

use crate::rate::RCState;
use crate::rayon::ThreadPool;
use crate::util::Pixel;

use std::sync::Arc;

mod data;
pub use data::*;

impl Config {
  fn setup<T: Pixel>(
    &self,
  ) -> Result<(ContextInner<T>, Option<Arc<ThreadPool>>), InvalidConfig> {
    self.validate()?;
    let inner = self.new_inner()?;

    let pool = self.new_thread_pool();

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

    let (v, _) = self.new_channel_internal(true)?;

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
    let (v, (_, r)) = self.new_channel_internal(true)?;

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

    let (v, (s, _)) = self.new_channel_internal(true)?;

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

    let (v, (s, r)) = self.new_channel_internal(true)?;

    Ok((v, (s.unwrap(), r.unwrap())))
  }
}

trait RcFirstPass {
  fn store_pass_data(&mut self, rc_state: &mut RCState);
  fn send_pass_data(&mut self);
  fn send_pass_summary(&mut self, rc_state: &mut RCState);
}

trait RcSecondPass {
  fn feed_pass_data<T: Pixel>(
    &mut self, inner: &mut ContextInner<T>,
  ) -> Result<(), ()>;
}

struct PassDataSender {
  sender: Sender<RcData>,
  data: Vec<u8>,
}

impl RcFirstPass for PassDataSender {
  fn store_pass_data(&mut self, rc_state: &mut RCState) {
    if let Some(data) = rc_state.emit_frame_data() {
      self.data.extend(data.iter());
    } else {
      unreachable!(
        "The encoder received more frames than its internal
              limit allows"
      );
    }
  }

  fn send_pass_data(&mut self) {
    let data = self.data.to_vec().into_boxed_slice();
    self.data.clear();
    self.sender.send(RcData::Frame(data)).unwrap();
  }
  fn send_pass_summary(&mut self, rc_state: &mut RCState) {
    let data = rc_state.emit_summary();
    let data = data.to_vec().into_boxed_slice();
    self.sender.send(RcData::Summary(data)).unwrap();
  }
}

impl RcFirstPass for Option<PassDataSender> {
  fn store_pass_data(&mut self, rc_state: &mut RCState) {
    match self.as_mut() {
      Some(s) => s.store_pass_data(rc_state),
      None => {}
    }
  }

  fn send_pass_data(&mut self) {
    match self.as_mut() {
      Some(s) => s.send_pass_data(),
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
    if inner.done_processing() {
      return Ok(());
    }

    while inner.rc_state.twopass_in_frames_needed() > 0 {
      if let Ok(RcData::Frame(data)) = self.recv() {
        inner
          .rc_state
          .parse_frame_data_packet(data.as_ref())
          .unwrap_or_else(|_| todo!("Error reporting"));
      } else {
        todo!("No data reporting");
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
  pub(crate) fn new_channel_internal<T: Pixel>(
    &self, serial_api_compat: bool,
  ) -> Result<
    (VideoDataChannel<T>, (Option<RcDataSender>, Option<RcDataReceiver>)),
    InvalidConfig,
  > {
    // The inner context is already configured to use the summary at this point.
    let (mut inner, pool) = self.setup()?;

    // TODO: make it user-settable
    let input_len = self.enc.rdo_lookahead_frames as usize * 2;

    let (send_frame, receive_frame) =
      if serial_api_compat { unbounded() } else { bounded(input_len) };
    let (send_packet, receive_packet) = unbounded();

    let rc = &self.rate_control;

    let (mut send_rc_pass1, rc_data_receiver) = if rc.emit_pass_data {
      let (send_rc_pass1, receive_rc_pass1) = unbounded();
      (
        Some(PassDataSender { sender: send_rc_pass1, data: Vec::new() }),
        Some(RcDataReceiver(receive_rc_pass1)),
      )
    } else {
      (None, None)
    };

    let (rc_data_sender, mut receive_rc_pass2, frame_limit) = if rc
      .summary
      .is_some()
    {
      // the pass data packets are now lumped together to match the packets emitted.
      let (frame_limit, _pass_limit) =
        rc.summary.as_ref().map(|s| (s.ntus as u64, s.total as u64)).unwrap();

      inner.limit = Some(frame_limit);

      let (send_rc_pass2, receive_rc_pass2) = if serial_api_compat {
        // TODO: derive it from the Encoder configuration
        bounded(input_len)
      } else {
        unbounded()
      };

      (
        Some(RcDataSender::new(frame_limit, send_rc_pass2)),
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

    let run = move || {
      for f in receive_frame.iter() {
        // info!("frame in {}", inner.frame_count);
        while !inner.needs_more_fi_lookahead() {
          receive_rc_pass2.feed_pass_data(&mut inner).unwrap();
          // needs_more_fi_lookahead() should guard for missing output_frameno
          // already.
          //
          // this call should return either Ok or Err(Encoded)
          let pass_data_ready = match inner.receive_packet() {
            Ok(p) => {
              send_packet.send(p).unwrap();
              send_rc_pass1.store_pass_data(&mut inner.rc_state);
              true
            }
            Err(EncoderStatus::Encoded) => {
              send_rc_pass1.store_pass_data(&mut inner.rc_state);
              false
            }
            Err(EncoderStatus::NotReady) => todo!("Error reporting"),
            _ => unreachable!(),
          };
          if pass_data_ready {
            send_rc_pass1.send_pass_data();
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
        let pass_data_ready = match r {
          Ok(p) => {
            send_packet.send(p).unwrap();
            send_rc_pass1.store_pass_data(&mut inner.rc_state);
            true
          }
          Err(EncoderStatus::LimitReached) => break,
          Err(EncoderStatus::Encoded) => {
            send_rc_pass1.store_pass_data(&mut inner.rc_state);
            false
          }
          Err(EncoderStatus::NotReady) => todo!("Error reporting"),
          _ => unreachable!(),
        };

        if pass_data_ready {
          send_rc_pass1.send_pass_data();
        }
      }

      send_rc_pass1.send_pass_summary(&mut inner.rc_state);
    };

    if let Some(pool) = pool {
      pool.spawn(run);
    } else {
      rayon::spawn(run);
    }

    Ok((channel, pass_channel))
  }
}
