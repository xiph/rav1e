// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(safe_extern_statics)]

#[macro_use]
extern crate serde_derive;
extern crate bincode;

#[cfg(all(test, feature = "decode_test_dav1d"))]
extern crate dav1d_sys;

#[cfg(test)]
extern crate interpolate_name;

#[cfg(test)]
#[macro_use]
extern crate pretty_assertions;

pub mod ec;
pub mod partition;
pub mod plane;
pub mod predict;
pub mod quantize;
pub mod rdo;
pub mod rdo_tables;
pub mod transform;
#[macro_use]
pub mod util;
pub mod cdef;
pub mod context;
pub mod deblock;
pub mod encoder;
pub mod entropymode;
pub mod lrf;
pub mod mc;
pub mod me;
pub mod metrics;
pub mod rate;
pub mod scan_order;
pub mod scenechange;
pub mod segmentation;
pub mod tiling;
pub mod token_cdfs;

mod api;
mod frame;
mod header;

pub use crate::api::*;
pub use crate::encoder::*;
pub use crate::header::*;
pub use crate::util::{CastFromPrimitive, Pixel};

pub use crate::frame::Frame;

#[cfg(all(test, any(feature = "decode_test", feature = "decode_test_dav1d")))]
mod test_encode_decode;

#[cfg(all(test, feature = "decode_test"))]
mod test_encode_decode_aom;

#[cfg(all(test, feature = "decode_test_dav1d"))]
mod test_encode_decode_dav1d;
