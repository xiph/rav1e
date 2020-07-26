// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use dcp::{ColorSpace, ErrorKind, ImageFormat, PixelFormat};
use dcv_color_primitives as dcp;
use std::convert::TryInto;
use wasm_bindgen::JsCast;
use web_sys;
use web_sys::{
  CanvasRenderingContext2d, Element, HtmlCanvasElement, HtmlImageElement,
};

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

  fn create_context(html: &HtmlCanvasElement) -> CanvasRenderingContext2d {
    html
      .get_context("2d")
      .unwrap()
      .unwrap()
      .dyn_into::<CanvasRenderingContext2d>()
      .unwrap()
  }

  /// Draw an `HtmlImageElement` onto the canvas
  ///
  /// ## Read more
  /// * [`CanvasRenderingContext2D.drawImage()`](https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/drawImage)
  pub fn draw_image(&self, img: &HtmlImageElement) {
    // The x- and y-axis coordinates in the destination canvas at which to place the top-left
    // corner of the source image.
    let dx = 0.0;
    let dy = 0.0;

    self.context.draw_image_with_html_image_element(img, dx, dy).unwrap();
  }

  /// Get `CanvasRenderingContext2d.ImageData.data` (`RGBA`)
  ///
  /// Represents the underlying pixel data of the canvas.  
  /// Data is stored as a `Vec<u8>` in the RGBA order, with integer values between 0 and 255 (inclusive).
  ///
  /// ## Panics
  /// * `IndexSizeError`
  ///   > Thrown if either sw or sh are zero.
  /// * `SecurityError`
  ///   > The canvas contains or may contain pixels which were loaded from an origin other than the one from which the document itself was loaded. To avoid SecurityError being thrown in this situation, configure CORS to allow the source image to be used in this way. See Allowing cross-origin use of images and canvas.
  ///
  /// ## Read more
  /// * [`CanvasRenderingContext2D.getImageData()`](https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/getImageData)
  /// * [`ImageData`](https://developer.mozilla.org/en-US/docs/Web/API/ImageData)
  /// * [`ImageData.data`](https://developer.mozilla.org/en-US/docs/Web/API/ImageData/data)
  fn data_rgba(&self) -> Vec<u8> {
    let data = self
      .context
      .get_image_data(
        0.0,
        0.0,
        self.html.width() as f64,
        self.html.height() as f64,
      )
      .unwrap()
      .data()
      .0;

    // check if data is complete
    let len = data.len();
    assert!(len % 4 == 0, "Invalid ImageData: `ImageData.data().len() == {}`, which is not dividable by 4 (RGBA)", len);

    data
  }

  /// Get `CanvasRenderingContext2d.ImageData.data` (`RGBA`), but converted to `ARGB`
  fn data_argb(&self) -> Vec<u8> {
    let data = self.data_rgba();

    chunk_data(data)
      .into_iter()
      .map(|mut i| {
        i.rotate_right(1);
        i.to_vec()
      })
      .flatten()
      .collect()
  }

  /// Get `CanvasRenderingContext2d.ImageData.data` (`RGBA`), but converted to `YCbCr` (`I444`, `Bt709`)
  pub fn data_i444(&self) -> [Vec<u8>; 3] {
    let data = self.data_argb();

    // ImageFormats
    let src_format = ImageFormat {
      pixel_format: PixelFormat::Argb,
      color_space: ColorSpace::Lrgb,
      num_planes: 1,
    };

    let dst_format = ImageFormat {
      pixel_format: PixelFormat::I444,
      color_space: ColorSpace::Bt709,
      num_planes: 3,
    };

    // Buffer
    let src_buffers = &[data.as_slice()];

    let mut buffer_vec =
      create_buffer(self.html.width(), self.html.height(), &dst_format);
    let (buffer_vec_0, buffer_vec_1) = buffer_vec.split_at_mut(1);
    let (buffer_vec_1, buffer_vec_2) = buffer_vec_1.split_at_mut(1);
    let dst_buffers = &mut [
      buffer_vec_0[0].as_mut_slice(),
      buffer_vec_1[0].as_mut_slice(),
      buffer_vec_2[0].as_mut_slice(),
    ];

    dcp::initialize();
    match dcp::convert_image(
      self.html.width(),
      self.html.height(),
      &src_format,
      None,
      src_buffers,
      &dst_format,
      None,
      dst_buffers,
    ) {
      Ok(()) => {}
      Err(e) => match e {
        ErrorKind::NotInitialized => panic!("NotInitialized: {}", e),
        ErrorKind::InvalidValue => panic!("InvalidValue: {}", e),
        ErrorKind::InvalidOperation => panic!("InvalidOperation: {}", e),
        ErrorKind::NotEnoughData => panic!("NotEnoughData: {}", e),
      },
    };
    buffer_vec
  }
}

/// Chunk data per pixel
///
/// Data is stored as a `Vec` of `[u8; 4]` arrays. Each array represents one pixel.
fn chunk_data(data: Vec<u8>) -> Vec<[u8; 4]> {
  data
      .chunks(4)
      .map(|i| -> [u8; 4] {
        i.try_into().expect(
          "Invalid ImageData: Couldn't split ImageData into chunks of 4 (e.g. RGBA)",
        )
      })
      .collect()
}

/// Create a pixel buffer to store converted data
fn create_buffer(
  width: u32, height: u32, format: &ImageFormat,
) -> [Vec<u8>; 3] {
  dcp::initialize();

  assert!(format.num_planes == 3, "ImageFormat.num_planes isn't 3");

  let buffers_size = &mut [0; 3];
  dcp::get_buffers_size(width, height, format, None, buffers_size).unwrap();

  let mut dst_buffers = [Vec::new(), Vec::new(), Vec::new()];
  for (i, &dest_len) in buffers_size.iter().enumerate() {
    let mut dst_vec: Vec<u8> = Vec::with_capacity(dest_len);
    dst_vec.resize(dest_len, 0);
    dst_buffers[i] = dst_vec;
  }
  dst_buffers
}
