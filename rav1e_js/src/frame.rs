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

use crate::web;
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

use wasm_bindgen::JsCast;
use web_sys;
use web_sys::{Element, HtmlImageElement};

#[wasm_bindgen]
pub fn do_sth() -> String {
  let img_id = "octocat";
  let img = web::document()
    .get_element_by_id(img_id)
    .unwrap()
    .dyn_into::<HtmlImageElement>()
    .map_err(|e: Element| {
      panic!("Err while casting document.getElementById(\"{}\") to HtmlImageElement: {:?}", img_id, e)
    })
    .unwrap();

  let canvas = Canvas::new(img.width(), img.height());
  canvas.draw_image(&img);
  let data = canvas.data_i444();

  let mut ret = String::new();
  for i in data.iter() {
    ret.push_str(&*format!("{:?}\n\n", i));
  }
  ret
}
