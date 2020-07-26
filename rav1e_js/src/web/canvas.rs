// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use wasm_bindgen::JsCast;
use web_sys;
use web_sys::{CanvasRenderingContext2d, Element, HtmlCanvasElement};

use crate::web;

#[derive(Debug)]
pub struct Canvas {
  html: HtmlCanvasElement,
  context: CanvasRenderingContext2d,
}

impl Canvas {
  pub fn new(width: u32, height: u32) -> Self {
    let html = web::document()
      .create_element("canvas")
      .unwrap()
      .dyn_into::<HtmlCanvasElement>()
      .map_err(|e: Element| {
        panic!("Err while casting document.createElement(\"canvas\") to HtmlCanvasElement: {:?}", e)
      })
      .unwrap();
    html.set_width(width);
    html.set_height(height);

    let context = Self::create_context(&html);
    Self { html, context }
  }

  pub fn from_id(id: &str) -> Self {
    let html: HtmlCanvasElement = web::document()
      .get_element_by_id(id)
      .unwrap()
      .dyn_into::<HtmlCanvasElement>()
      .map_err(|e: Element| {
        panic!("Err while casting document.getElementById(\"{}\") to HtmlCanvasElement: {:?}", id, e)
      })
      .unwrap();

    let context = Self::create_context(&html);
    Canvas { html, context }
  }

  fn create_context(html: &HtmlCanvasElement) -> CanvasRenderingContext2d {
    html
      .get_context("2d")
      .unwrap()
      .unwrap()
      .dyn_into::<CanvasRenderingContext2d>()
      .unwrap()
  }
}
