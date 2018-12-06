// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(safe_extern_statics)]

pub const RESTORE_NONE: u8 = 0;
pub const RESTORE_SWITCHABLE: u8 = 1;
pub const RESTORE_WIENER: u8 = 2;
pub const RESTORE_SGRPROJ: u8 = 3;

pub const WIENER_TAPS_MIN: [i8; 3] = [ -5, -23, -17 ];
pub const WIENER_TAPS_MID: [i8; 3] = [ 3, -7, 15 ];
pub const WIENER_TAPS_MAX: [i8; 3] = [ 10, 8, 46 ];

pub const SGR_XQD_MIN: [i8; 2] = [ -96, -32 ];
pub const SGR_XQD_MID: [i8; 2] = [ -32, 31 ];
pub const SGR_XQD_MAX: [i8; 2] = [ 31, 95 ];
