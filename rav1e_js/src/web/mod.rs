// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use web_sys::{Document, Window};

mod canvas;
pub use canvas::Canvas;

/// The `Document` interface represents any web page loaded in the browser and serves
/// as an entry point into the web page's content, which is the DOM tree.
///
/// https://developer.mozilla.org/en-US/docs/Web/API/Document
pub fn document() -> Document {
  window().document().expect("Couldn't find DOM `document` in `window`!")
}

/// The `Window` interface represents a window containing a DOM document.
///
/// https://developer.mozilla.org/en-US/docs/Web/API/Window
pub fn window() -> Window {
  web_sys::window().expect("Couldn't obtain `window` from global context!")
}
