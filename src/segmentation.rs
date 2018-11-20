// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(safe_extern_statics)]

use context::*;
use FrameInvariants;
use FrameState;

pub fn segmentation_optimize(_fi: &FrameInvariants, fs: &mut FrameState) {
    fs.segmentation.enabled = false;
    fs.segmentation.update_data = false;
    fs.segmentation.update_map = false;

    fs.segmentation.features[0][SegLvl::SEG_LVL_ALT_Q as usize] = false;
    fs.segmentation.data[0][SegLvl::SEG_LVL_ALT_Q as usize] = 0;

    /* Figure out parameters */
    fs.segmentation.preskip = false;
    fs.segmentation.last_active_segid = 0;
    if fs.segmentation.enabled {
        for i in 0..8 {
            for j in 0..SegLvl::SEG_LVL_MAX as usize {
                if fs.segmentation.features[i][j] {
                    fs.segmentation.last_active_segid = i as u8;
                    if j >= SegLvl::SEG_LVL_REF_FRAME as usize {
                        fs.segmentation.preskip = true;
                    }
                }
            }
        }
    }
}
