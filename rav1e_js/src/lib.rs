// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use wasm_bindgen::prelude::*;

pub mod encoder;
pub mod encoder_config;
pub mod frame;
pub mod packet;
pub mod utils;

pub use encoder::Encoder;
pub use encoder_config::EncoderConfig;
pub use frame::Frame;
pub use packet::Packet;

/// Runs on module import
#[wasm_bindgen(start)]
pub fn main_js() {
  utils::set_panic_hook();
}
