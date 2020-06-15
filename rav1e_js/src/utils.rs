// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use wasm_bindgen::prelude::*;

/// When the `console_error_panic_hook` feature is enabled, we can call the
/// `set_panic_hook` function at least once during initialization, and then
/// we will get better error messages if our code ever panics.
///
/// For more details see
/// https://github.com/rustwasm/console_error_panic_hook#readme
pub fn set_panic_hook() {
  #[cfg(feature = "console_error_panic_hook")]
  console_error_panic_hook::set_once();
}

/// A macro to provide `println!(..)`-style syntax for `console.log` logging.
#[macro_export]
macro_rules! log {
	( $( $t:tt )* ) => {
		web_sys::console::log_1(&format!( $( $t )* ).into());
	}
}

pub fn construct_js_err(err: impl std::fmt::Display, msg: &str) -> JsValue {
  let mut e = format!("{}", err);
  if msg != "" {
    e.push_str(": ");
    e.push_str(msg);
  }
  JsValue::from(e)
}
