// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(non_snake_case)]

use rav1e::prelude::*;
use std::ops::{Deref, DerefMut};
use wasm_bindgen::prelude::*;

use crate::encoder::Encoder;
use crate::utils::construct_js_err;
use crate::{EncoderConfig, Frame, Packet};

/// Contains the encoding state
#[wasm_bindgen]
pub struct FrameEncoder {
  ctx: Context<u8>,
}

impl Encoder for FrameEncoder {
  fn ctx<'a>(&'a self) -> Box<dyn Deref<Target = Context<u8>> + 'a> {
    Box::new(&self.ctx)
  }

  fn ctx_mut<'a>(
    &'a mut self,
  ) -> Box<dyn DerefMut<Target = Context<u8>> + 'a> {
    Box::new(&mut self.ctx)
  }
}

#[wasm_bindgen]
impl FrameEncoder {
  #[wasm_bindgen(constructor)]
  pub fn fromEncoderConfig(
    conf: EncoderConfig,
  ) -> Result<FrameEncoder, JsValue> {
    let cfg = Config::new().with_encoder_config(conf.conf);

    match cfg.new_context() {
      Ok(ctx) => Ok(Self { ctx }),
      Err(e) => Err(construct_js_err(e, "Invalid EncoderConfig")),
    }
  }

  pub fn default() -> Result<FrameEncoder, JsValue> {
    Self::fromEncoderConfig(EncoderConfig::new())
  }

  pub fn debug(&self) -> String {
    self.debug_msg()
  }

  /// Allocates and returns a new frame.
  pub fn newFrame(&self) -> Frame {
    self.new_frame()
  }

  /// Sends the frame for encoding.
  ///
  /// This method adds the frame into the frame queue and runs the first passes of the look-ahead computation.
  pub fn sendFrame(&mut self, frame: &Frame) -> Result<(), JsValue> {
    self.send_frame(frame)
  }

  /// Flushes the encoder.
  ///
  /// Flushing signals the end of the video. After the encoder has been flushed, no additional frames are accepted.
  pub fn flush(&mut self) {
    self.flush_it();
  }

  /// Encodes the next frame and returns the encoded data.
  ///
  /// This method is where the main encoding work is done.
  pub fn receivePacket(&mut self) -> Result<Packet, JsValue> {
    self.receive_packet()
  }
}
