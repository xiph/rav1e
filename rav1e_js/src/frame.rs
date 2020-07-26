// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use rav1e;
use wasm_bindgen::prelude::*;
use web_sys::{HtmlCanvasElement, HtmlImageElement};

use crate::web::Canvas;

/// Represents one video frame
#[wasm_bindgen]
pub struct Frame {
  pub(crate) f: rav1e::Frame<u8>,
}

#[wasm_bindgen]
impl Frame {
  pub fn debug(&self) -> String {
    format!("{:?}", self.f)
  }

  /// Create a new `Frame` from the underlying pixel-data of a `HtmlImageElement`.
  pub fn from_img(img: &HtmlImageElement) -> Self {
    let canvas = Canvas::new(img.width(), img.height());
    canvas.draw_image(img);
    Frame { f: canvas.create_frame() }
  }

  pub fn from_canvas(canvas: &HtmlCanvasElement) -> Self {
    let canvas = Canvas::from(canvas);
    Frame { f: canvas.create_frame() }
  }
}
