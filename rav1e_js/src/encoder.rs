// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(non_snake_case)]

use rav1e::prelude::{Config, Context, EncoderStatus};

use wasm_bindgen::prelude::*;

use crate::utils::construct_js_err;
use crate::{EncoderConfig, Frame, Packet};

/// Wrapper around the encoder context (`rav1e::api::context::Context<u16>`).
///
/// Contains the encoding state.
#[wasm_bindgen]
pub struct Encoder {
  ctx: Context<u16>,
}

#[wasm_bindgen]
impl Encoder {
  #[wasm_bindgen(constructor)]
  pub fn fromEncoderConfig(conf: EncoderConfig) -> Result<Encoder, JsValue> {
    let cfg = Config::new().with_encoder_config(conf.conf);

    match cfg.new_context() {
      Ok(c) => Ok(Encoder { ctx: c }),
      Err(e) => Err(construct_js_err(e, "Invalid EncoderConfig")),
    }
  }

  pub fn default() -> Result<Encoder, JsValue> {
    Self::fromEncoderConfig(EncoderConfig::new())
  }

  pub fn debug(&self) -> String {
    format!("{:?}", self.ctx)
  }

  /// Allocates and returns a new frame.
  pub fn newFrame(&self) -> Frame {
    Frame { f: self.ctx.new_frame() }
  }

  /// Sends the frame for encoding.
  ///
  /// This method adds the frame into the frame queue and runs the first passes of the look-ahead computation.
  pub fn sendFrame(&mut self, frame: &Frame) -> Result<(), JsValue> {
    match self.ctx.send_frame(frame.f.clone()) {
      Ok(_) => Ok(()),
      Err(e) => match e {
        EncoderStatus::EnoughData => Err(construct_js_err(
          e,
          "Unable to append frame to the internal queue",
        )),
        _ => Err(construct_js_err(e, "")),
      },
    }
  }

  /// Flushes the encoder.
  ///
  /// Flushing signals the end of the video. After the encoder has been flushed, no additional frames are accepted.
  pub fn flush(&mut self) {
    self.ctx.flush();
  }

  /// Encodes the next frame and returns the encoded data.
  ///
  /// This method is where the main encoding work is done.
  pub fn receivePacket(&mut self) -> Result<Packet, JsValue> {
    match self.ctx.receive_packet() {
      Ok(packet) => Ok(Packet { p: packet }),
      Err(e) => Err(construct_js_err(e, "")),
    }
  }
}
