// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

pub mod frame;
pub mod math;
pub mod pixel;
pub mod plane;

mod serialize {
  cfg_if::cfg_if! {
     if #[cfg(feature="serialize")] {
       pub use serde::*;
      } else {
        pub use noop_proc_macro::{Deserialize, Serialize};
     }
  }
}

mod wasm_bindgen {
  cfg_if::cfg_if! {
    if #[cfg(feature="wasm")] {
      pub use wasm_bindgen::prelude::*;
    } else {
      pub use noop_proc_macro::wasm_bindgen;
    }
  }
}

pub mod prelude {
  pub use crate::math::*;
  pub use crate::pixel::*;
}
