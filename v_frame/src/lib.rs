// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![deny(bare_trait_objects)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_ptr_alignment)]
#![allow(clippy::cognitive_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::verbose_bit_mask)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::many_single_char_names)]
// Performance lints
#![warn(clippy::linkedlist)]
#![warn(clippy::missing_const_for_fn)]
#![warn(clippy::mutex_integer)]
#![warn(clippy::suboptimal_flops)]
// Correctness lints
#![warn(clippy::expl_impl_clone_on_copy)]
#![warn(clippy::mem_forget)]
#![warn(clippy::path_buf_push_overwrite)]
// Clarity/formatting lints
#![warn(clippy::map_flatten)]
#![warn(clippy::mut_mut)]
#![warn(clippy::needless_borrow)]
#![warn(clippy::needless_continue)]
#![warn(clippy::range_plus_one)]
// Documentation lints
#![warn(clippy::doc_markdown)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::undocumented_unsafe_blocks)]

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
