// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! Test suite for node.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn simple_encoding() {
  rav1e_js::simple_encoding_without_alert(30);
}
