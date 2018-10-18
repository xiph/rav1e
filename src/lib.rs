// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(safe_extern_statics)]
#![cfg_attr(feature = "cargo-clippy", allow(collapsible_if))]

extern crate bitstream_io;
extern crate backtrace;
extern crate libc;
extern crate rand;

extern crate num_traits;

pub mod ec;
pub mod partition;
pub mod plane;
pub mod context;
pub mod transform;
pub mod quantize;
pub mod predict;
pub mod rdo;
#[macro_use]
pub mod util;
pub mod entropymode;
pub mod token_cdfs;
pub mod deblock;
pub mod cdef;
pub mod encoder;
pub mod me;
pub mod scan_order;

mod api;

pub use api::*;
pub use encoder::*;

// #[cfg(test)]
#[cfg(all(test, feature="decode_test"))]
mod test_encode_decode;


