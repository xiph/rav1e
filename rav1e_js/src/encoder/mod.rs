// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use rav1e::prelude::*;
use std::ops::{Deref, DerefMut};
use wasm_bindgen::prelude::*;

use crate::utils::construct_js_err;
use crate::{Frame, Packet};

mod frame_encoder;
pub use frame_encoder::FrameEncoder;

mod video_encoder;
pub use video_encoder::VideoEncoder;

/// Implements basic encoder functionality.
///
/// ## Workaround
/// `wasm_bindgen` doesn't support exporting methods implemented by traits.
/// _(See [rustwasm/wasm-bindgen#2073](https://github.com/rustwasm/wasm-bindgen/issues/2073))_
///
/// The workaround is to wrap the default methods and copy the doc-comments.
pub trait Encoder {
  /// Non-mutable access to encoder context
  fn ctx<'a>(&'a self) -> Box<dyn Deref<Target = Context<u8>> + 'a>;

  /// Mutable access to encoder context
  fn ctx_mut<'a>(&'a mut self)
    -> Box<dyn DerefMut<Target = Context<u8>> + 'a>;

  fn debug_msg(&self) -> String {
    format!("{:?}", **self.ctx())
  }

  /// Allocates and returns a new frame.
  fn new_frame(&self) -> Frame {
    Frame { f: self.ctx().new_frame() }
  }

  /// Sends the frame for encoding.
  ///
  /// This method adds the frame into the frame queue and runs the first passes of the look-ahead computation.
  fn send_frame(&mut self, frame: &Frame) -> Result<(), JsValue> {
    match self.ctx_mut().send_frame(frame.f.clone()) {
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
  fn flush_it(&mut self) {
    self.ctx_mut().flush();
  }

  /// Encodes the next frame and returns the encoded data.
  ///
  /// This method is where the main encoding work is done.
  fn receive_packet(&mut self) -> Result<Packet, JsValue> {
    match self.ctx_mut().receive_packet() {
      Ok(p) => Ok(Packet { p }),
      Err(e) => Err(construct_js_err(e, "")),
    }
  }
}
