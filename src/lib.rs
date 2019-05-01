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

#[cfg(all(test, feature="decode_test_dav1d"))]
extern crate dav1d_sys;

#[cfg(test)]
extern crate interpolate_name;

#[cfg(test)]
#[macro_use]
extern crate pretty_assertions;

pub(crate) mod ec;
pub(crate) mod partition;
pub(crate) mod plane;
pub(crate) mod transform;
pub(crate) mod quantize;
pub(crate) mod predict;
pub(crate) mod rdo;
pub(crate) mod rdo_tables;
#[macro_use]
pub(crate) mod util;
pub(crate) mod context;
pub(crate) mod entropymode;
pub(crate) mod token_cdfs;
pub(crate) mod deblock;
pub(crate) mod segmentation;
pub(crate) mod cdef;
pub(crate) mod lrf;
pub(crate) mod encoder;
pub(crate) mod mc;
pub(crate) mod me;
pub(crate) mod metrics;
pub(crate) mod scan_order;
pub(crate) mod scenechange;
pub(crate) mod rate;
pub(crate) mod tiling;

mod api;
mod header;
mod frame;

pub(crate) use crate::encoder::*;

pub use crate::api::*;
pub use crate::util::{CastFromPrimitive, Pixel};

pub use crate::encoder::Tune;
pub use crate::frame::Frame;
pub use crate::partition::BlockSize;

#[cfg(all(test, any(feature="decode_test", feature="decode_test_dav1d")))]
mod test_encode_decode;

#[cfg(all(test, feature="decode_test"))]
mod test_encode_decode_aom;

#[cfg(all(test, feature="decode_test_dav1d"))]
mod test_encode_decode_dav1d;

