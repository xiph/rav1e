// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use rav1e::prelude::Frame as Rav1eFrame;
use wasm_bindgen::prelude::*;

use crate::web::Canvas;

/// Wrapper around `v_frame::frame::Frame<u16>`.
///
/// Represents one video frame.
#[wasm_bindgen]
pub struct Frame {
  pub(crate) f: Rav1eFrame<u16>,
}

#[wasm_bindgen]
impl Frame {
  pub fn debug(&self) -> String {
    format!("{:?}", self.f)
  }
}

#[wasm_bindgen]
pub fn do_sth() -> String {
  let canvas = Canvas::from_id("canvas");
  format!("{:?}", canvas)
}
