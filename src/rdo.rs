// Copyright (c) 2001-2016, Alliance for Open Media. All rights reserved
// Copyright (c) 2017, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(non_camel_case_types)]

use plane::*;
use partition::PredictionMode;
//use std::io::prelude::*;

#[derive(Copy,Clone)]
pub struct RDOOutput {
    pub rd_cost: u64,
    pub pred_mode: PredictionMode,
}

// Sum of Squared Error for a wxh block
pub fn sse_wxh(src1: &PlaneSlice, src2: &PlaneSlice, w: usize, h: usize) -> u64 {
    let mut sse: u64 = 0;
    for j in 0..h {
        for i in 0..w {
            let dist = (src1.p(i, j) as i16 - src2.p(i, j) as i16) as i64;
            sse += (dist * dist) as u64;
        }
    }
    sse
}
