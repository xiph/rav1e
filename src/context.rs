//
// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(safe_extern_statics)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![cfg_attr(feature = "cargo-clippy", allow(cast_lossless))]
#![cfg_attr(feature = "cargo-clippy", allow(unnecessary_mut_passed))]
#![cfg_attr(feature = "cargo-clippy", allow(needless_range_loop))]
#![cfg_attr(feature = "cargo-clippy", allow(collapsible_if))]

use ec::Writer;
use partition::BlockSize::*;
use partition::PredictionMode::*;
use partition::TxSize::*;
use partition::TxType::*;
use partition::*;
use plane::*;
use util::clamp;
use util::msb;
use std::*;

use self::REF_CONTEXTS;
use self::SINGLE_REFS;

pub const PLANES: usize = 3;

const PARTITION_PLOFFSET: usize = 4;
const PARTITION_BLOCK_SIZES: usize = 4 + 1;
const PARTITION_CONTEXTS_PRIMARY: usize = PARTITION_BLOCK_SIZES * PARTITION_PLOFFSET;
const PARTITION_CONTEXTS: usize = PARTITION_CONTEXTS_PRIMARY;
pub const PARTITION_TYPES: usize = 4;

pub const MI_SIZE_LOG2: usize = 2;
pub const MI_SIZE: usize = (1 << MI_SIZE_LOG2);
const MAX_MIB_SIZE_LOG2: usize = (MAX_SB_SIZE_LOG2 - MI_SIZE_LOG2);
pub const MAX_MIB_SIZE: usize = (1 << MAX_MIB_SIZE_LOG2);
pub const MAX_MIB_MASK: usize = (MAX_MIB_SIZE - 1);

const MAX_SB_SIZE_LOG2: usize = 6;
const MAX_SB_SIZE: usize = (1 << MAX_SB_SIZE_LOG2);
const MAX_SB_SQUARE: usize = (MAX_SB_SIZE * MAX_SB_SIZE);

pub const MAX_TX_SIZE: usize = 32;
const MAX_TX_SQUARE: usize = MAX_TX_SIZE * MAX_TX_SIZE;

pub const INTRA_MODES: usize = 13;
const UV_INTRA_MODES: usize = 14;

const CFL_JOINT_SIGNS: usize = 8;
const CFL_ALPHA_CONTEXTS: usize = 6;
const CFL_ALPHABET_SIZE: usize = 16;

const BLOCK_SIZE_GROUPS: usize = 4;
const MAX_ANGLE_DELTA: usize = 3;
const DIRECTIONAL_MODES: usize = 8;
const KF_MODE_CONTEXTS: usize = 5;

const EXT_PARTITION_TYPES: usize = 10;
const TX_SIZES: usize = 4;
const TX_SETS: usize = 9;
const TX_SETS_INTRA: usize = 3;
const TX_SETS_INTER: usize = 4;

const MAX_REF_MV_STACK_SIZE: usize = 8;
pub const REF_CAT_LEVEL: u32 = 640;

pub const FRAME_LF_COUNT: usize = 4;
pub const MAX_LOOP_FILTER: usize = 63;
const DELTA_LF_SMALL: u32 = 3;
const DELTA_LF_PROBS: usize = DELTA_LF_SMALL as usize;

// Number of transform types in each set type
static num_tx_set: [usize; TX_SETS] =
  [1, 2, 5, 7, 7, 10, 12, 16, 16];
pub static av1_tx_used: [[usize; TX_TYPES]; TX_SETS] = [
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
  [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
  [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
  [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
];

// Maps set types above to the indices used for intra
static tx_set_index_intra: [i8; TX_SETS] =
  [0, -1, 2, -1, 1, -1, -1, -1, -16];
// Maps set types above to the indices used for inter
static tx_set_index_inter: [i8; TX_SETS] =
  [0, 3, -1, -1, -1, -1, 2, -1, 1];

static av1_tx_ind: [[usize; TX_TYPES]; TX_SETS] = [
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [1, 3, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [1, 5, 6, 4, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0],
  [1, 5, 6, 4, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0],
  [1, 2, 3, 6, 4, 5, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0],
  [3, 4, 5, 8, 6, 7, 9, 10, 11, 0, 1, 2, 0, 0, 0, 0],
  [7, 8, 9, 12, 10, 11, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6],
  [7, 8, 9, 12, 10, 11, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6]
];

static ss_size_lookup: [[[BlockSize; 2]; 2]; BlockSize::BLOCK_SIZES_ALL] = [
  //  ss_x == 0    ss_x == 0        ss_x == 1      ss_x == 1
  //  ss_y == 0    ss_y == 1        ss_y == 0      ss_y == 1
  [  [ BLOCK_4X4, BLOCK_4X4 ], [BLOCK_4X4, BLOCK_4X4 ] ],
  [  [ BLOCK_4X8, BLOCK_4X4 ], [BLOCK_4X4, BLOCK_4X4 ] ],
  [  [ BLOCK_8X4, BLOCK_4X4 ], [BLOCK_4X4, BLOCK_4X4 ] ],
  [  [ BLOCK_8X8, BLOCK_8X4 ], [BLOCK_4X8, BLOCK_4X4 ] ],
  [  [ BLOCK_8X16, BLOCK_8X8 ], [BLOCK_4X16, BLOCK_4X8 ] ],
  [  [ BLOCK_16X8, BLOCK_16X4 ], [BLOCK_8X8, BLOCK_8X4 ] ],
  [  [ BLOCK_16X16, BLOCK_16X8 ], [BLOCK_8X16, BLOCK_8X8 ] ],
  [  [ BLOCK_16X32, BLOCK_16X16 ], [BLOCK_8X32, BLOCK_8X16 ] ],
  [  [ BLOCK_32X16, BLOCK_32X8 ], [BLOCK_16X16, BLOCK_16X8 ] ],
  [  [ BLOCK_32X32, BLOCK_32X16 ], [BLOCK_16X32, BLOCK_16X16 ] ],
  [  [ BLOCK_32X64, BLOCK_32X32 ], [BLOCK_16X64, BLOCK_16X32 ] ],
  [  [ BLOCK_64X32, BLOCK_64X16 ], [BLOCK_32X32, BLOCK_32X16 ] ],
  [  [ BLOCK_64X64, BLOCK_64X32 ], [BLOCK_32X64, BLOCK_32X32 ] ],
  [  [ BLOCK_64X128, BLOCK_64X64 ], [ BLOCK_32X128, BLOCK_32X64 ] ],
  [  [ BLOCK_128X64, BLOCK_128X32 ], [ BLOCK_64X64, BLOCK_64X32 ] ],
  [  [ BLOCK_128X128, BLOCK_128X64 ], [ BLOCK_64X128, BLOCK_64X64 ] ],
  [  [ BLOCK_4X16, BLOCK_4X8 ], [BLOCK_4X16, BLOCK_4X8 ] ],
  [  [ BLOCK_16X4, BLOCK_16X4 ], [BLOCK_8X4, BLOCK_8X4 ] ],
  [  [ BLOCK_8X32, BLOCK_8X16 ], [BLOCK_INVALID, BLOCK_4X16 ] ],
  [  [ BLOCK_32X8, BLOCK_INVALID ], [BLOCK_16X8, BLOCK_16X4 ] ],
  [  [ BLOCK_16X64, BLOCK_16X32 ], [BLOCK_INVALID, BLOCK_8X32 ] ],
  [  [ BLOCK_64X16, BLOCK_INVALID ], [BLOCK_32X16, BLOCK_32X8 ] ],
  [  [ BLOCK_32X128, BLOCK_32X64 ], [ BLOCK_INVALID, BLOCK_16X64 ] ],
  [  [ BLOCK_128X32, BLOCK_INVALID ], [ BLOCK_64X32, BLOCK_64X16 ] ],
];

pub fn get_plane_block_size(bsize: BlockSize, subsampling_x: usize, subsampling_y: usize)
    -> BlockSize {
  ss_size_lookup[bsize as usize][subsampling_x][subsampling_y]
}

// Generates 4 bit field in which each bit set to 1 represents
// a blocksize partition  1111 means we split 64x64, 32x32, 16x16
// and 8x8.  1000 means we just split the 64x64 to 32x32
static partition_context_lookup: [[u8; 2]; BlockSize::BLOCK_SIZES_ALL] = [
  [ 31, 31 ],  // 4X4   - {0b11111, 0b11111}
  [ 31, 30 ],  // 4X8   - {0b11111, 0b11110}
  [ 30, 31 ],  // 8X4   - {0b11110, 0b11111}
  [ 30, 30 ],  // 8X8   - {0b11110, 0b11110}
  [ 30, 28 ],  // 8X16  - {0b11110, 0b11100}
  [ 28, 30 ],  // 16X8  - {0b11100, 0b11110}
  [ 28, 28 ],  // 16X16 - {0b11100, 0b11100}
  [ 28, 24 ],  // 16X32 - {0b11100, 0b11000}
  [ 24, 28 ],  // 32X16 - {0b11000, 0b11100}
  [ 24, 24 ],  // 32X32 - {0b11000, 0b11000}
  [ 24, 16 ],  // 32X64 - {0b11000, 0b10000}
  [ 16, 24 ],  // 64X32 - {0b10000, 0b11000}
  [ 16, 16 ],  // 64X64 - {0b10000, 0b10000}
  [ 16, 0 ],   // 64X128- {0b10000, 0b00000}
  [ 0, 16 ],   // 128X64- {0b00000, 0b10000}
  [ 0, 0 ],    // 128X128-{0b00000, 0b00000}
  [ 31, 28 ],  // 4X16  - {0b11111, 0b11100}
  [ 28, 31 ],  // 16X4  - {0b11100, 0b11111}
  [ 30, 24 ],  // 8X32  - {0b11110, 0b11000}
  [ 24, 30 ],  // 32X8  - {0b11000, 0b11110}
  [ 28, 16 ],  // 16X64 - {0b11100, 0b10000}
  [ 16, 28 ],  // 64X16 - {0b10000, 0b11100}
  [ 24, 0 ],   // 32X128- {0b11000, 0b00000}
  [ 0, 24 ],   // 128X32- {0b00000, 0b11000}
];

static size_group_lookup: [u8; BlockSize::BLOCK_SIZES_ALL] = [
  0, 0,
  0, 1,
  1, 1,
  2, 2,
  2, 3,
  3, 3,
  3, 3, 3, 3, 0,
  0, 1,
  1, 2,
  2, 3, 3
];

static num_pels_log2_lookup: [u8; BlockSize::BLOCK_SIZES_ALL] = [
  4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 6, 6, 8, 8, 10, 10, 12, 12];

pub static subsize_lookup: [[BlockSize; BlockSize::BLOCK_SIZES_ALL]; EXT_PARTITION_TYPES] =
[
  [     // PARTITION_NONE
    //                            4X4
                                  BLOCK_4X4,
    // 4X8,        8X4,           8X8
    BLOCK_4X8,     BLOCK_8X4,     BLOCK_8X8,
    // 8X16,       16X8,          16X16
    BLOCK_8X16,    BLOCK_16X8,    BLOCK_16X16,
    // 16X32,      32X16,         32X32
    BLOCK_16X32,   BLOCK_32X16,   BLOCK_32X32,
    // 32X64,      64X32,         64X64
    BLOCK_32X64,   BLOCK_64X32,   BLOCK_64X64,
    // 64x128,     128x64,        128x128
    BLOCK_64X128,  BLOCK_128X64,  BLOCK_128X128,
    // 4X16,       16X4,          8X32
    BLOCK_4X16,    BLOCK_16X4,    BLOCK_8X32,
    // 32X8,       16X64,         64X16
    BLOCK_32X8,    BLOCK_16X64,   BLOCK_64X16,
    // 32x128,     128x32
    BLOCK_32X128,  BLOCK_128X32
  ], [  // PARTITION_HORZ
    //                            4X4
                                  BLOCK_INVALID,
    // 4X8,        8X4,           8X8
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_8X4,
    // 8X16,       16X8,          16X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_16X8,
    // 16X32,      32X16,         32X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_32X16,
    // 32X64,      64X32,         64X64
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_64X32,
    // 64x128,     128x64,        128x128
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_128X64,
    // 4X16,       16X4,          8X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
    // 32X8,       16X64,         64X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
    // 32x128,     128x32
    BLOCK_INVALID, BLOCK_INVALID
  ], [  // PARTITION_VERT
    //                            4X4
                                  BLOCK_INVALID,
    // 4X8,        8X4,           8X8
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_4X8,
    // 8X16,       16X8,          16X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_8X16,
    // 16X32,      32X16,         32X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_16X32,
    // 32X64,      64X32,         64X64
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_32X64,
    // 64x128,     128x64,        128x128
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_64X128,
    // 4X16,       16X4,          8X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
    // 32X8,       16X64,         64X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
    // 32x128,     128x32
    BLOCK_INVALID, BLOCK_INVALID
  ], [  // PARTITION_SPLIT
    //                            4X4
                                  BLOCK_INVALID,
    // 4X8,        8X4,           8X8
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_4X4,
    // 8X16,       16X8,          16X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_8X8,
    // 16X32,      32X16,         32X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_16X16,
    // 32X64,      64X32,         64X64
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_32X32,
    // 64x128,     128x64,        128x128
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_64X64,
    // 4X16,       16X4,          8X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
    // 32X8,       16X64,         64X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
    // 32x128,     128x32
    BLOCK_INVALID, BLOCK_INVALID
  ], [  // PARTITION_HORZ_A
    //                            4X4
                                  BLOCK_INVALID,
    // 4X8,        8X4,           8X8
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_8X4,
    // 8X16,       16X8,          16X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_16X8,
    // 16X32,      32X16,         32X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_32X16,
    // 32X64,      64X32,         64X64
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_64X32,
    // 64x128,     128x64,        128x128
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_128X64,
    // 4X16,       16X4,          8X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
    // 32X8,       16X64,         64X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
    // 32x128,     128x32
    BLOCK_INVALID, BLOCK_INVALID
  ], [  // PARTITION_HORZ_B
    //                            4X4
                                  BLOCK_INVALID,
    // 4X8,        8X4,           8X8
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_8X4,
    // 8X16,       16X8,          16X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_16X8,
    // 16X32,      32X16,         32X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_32X16,
    // 32X64,      64X32,         64X64
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_64X32,
    // 64x128,     128x64,        128x128
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_128X64,
    // 4X16,       16X4,          8X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
    // 32X8,       16X64,         64X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
    // 32x128,     128x32
    BLOCK_INVALID, BLOCK_INVALID
  ], [  // PARTITION_VERT_A
    //                            4X4
                                  BLOCK_INVALID,
    // 4X8,        8X4,           8X8
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_4X8,
    // 8X16,       16X8,          16X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_8X16,
    // 16X32,      32X16,         32X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_16X32,
    // 32X64,      64X32,         64X64
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_32X64,
    // 64x128,     128x64,        128x128
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_64X128,
    // 4X16,       16X4,          8X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
    // 32X8,       16X64,         64X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
    // 32x128,     128x32
    BLOCK_INVALID, BLOCK_INVALID
  ], [  // PARTITION_VERT_B
    //                            4X4
                                  BLOCK_INVALID,
    // 4X8,        8X4,           8X8
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_4X8,
    // 8X16,       16X8,          16X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_8X16,
    // 16X32,      32X16,         32X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_16X32,
    // 32X64,      64X32,         64X64
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_32X64,
    // 64x128,     128x64,        128x128
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_64X128,
    // 4X16,       16X4,          8X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
    // 32X8,       16X64,         64X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
    // 32x128,     128x32
    BLOCK_INVALID, BLOCK_INVALID
  ], [  // PARTITION_HORZ_4
    //                            4X4
                                  BLOCK_INVALID,
    // 4X8,        8X4,           8X8
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
    // 8X16,       16X8,          16X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_16X4,
    // 16X32,      32X16,         32X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_32X8,
    // 32X64,      64X32,         64X64
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_64X16,
    // 64x128,     128x64,        128x128
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_128X32,
    // 4X16,       16X4,          8X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
    // 32X8,       16X64,         64X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
    // 32x128,     128x32
    BLOCK_INVALID, BLOCK_INVALID
  ], [  // PARTITION_VERT_4
    //                            4X4
                                  BLOCK_INVALID,
    // 4X8,        8X4,           8X8
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
    // 8X16,       16X8,          16X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_4X16,
    // 16X32,      32X16,         32X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_8X32,
    // 32X64,      64X32,         64X64
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_16X64,
    // 64x128,     128x64,        128x128
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_32X128,
    // 4X16,       16X4,          8X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
    // 32X8,       16X64,         64X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
    // 32x128,     128x32
    BLOCK_INVALID, BLOCK_INVALID
  ]
];

const PLANE_TYPES: usize = 2;
const REF_TYPES: usize = 2;
const SKIP_CONTEXTS: usize = 3;
const INTRA_INTER_CONTEXTS: usize = 4;
const DRL_MODE_CONTEXTS: usize = 3;

// Level Map
const TXB_SKIP_CONTEXTS: usize =  13;

const EOB_COEF_CONTEXTS: usize =  9;

const SIG_COEF_CONTEXTS_2D: usize =  26;
const SIG_COEF_CONTEXTS_1D: usize =  16;
const SIG_COEF_CONTEXTS_EOB: usize =  4;
const SIG_COEF_CONTEXTS: usize = SIG_COEF_CONTEXTS_2D + SIG_COEF_CONTEXTS_1D;

const COEFF_BASE_CONTEXTS: usize = SIG_COEF_CONTEXTS;
const DC_SIGN_CONTEXTS: usize =  3;

const BR_TMP_OFFSET: usize =  12;
const BR_REF_CAT: usize =  4;
const LEVEL_CONTEXTS: usize =  21;

const NUM_BASE_LEVELS: usize =  2;

const BR_CDF_SIZE: usize = 4;
const COEFF_BASE_RANGE: usize = 4 * (BR_CDF_SIZE - 1);

const COEFF_CONTEXT_BITS: usize = 6;
const COEFF_CONTEXT_MASK: usize = (1 << COEFF_CONTEXT_BITS) - 1;
const MAX_BASE_BR_RANGE: usize = COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1;

const BASE_CONTEXT_POSITION_NUM: usize = 12;

// Pad 4 extra columns to remove horizontal availability check.
const TX_PAD_HOR_LOG2: usize = 2;
const TX_PAD_HOR: usize = 4;
// Pad 6 extra rows (2 on top and 4 on bottom) to remove vertical availability
// check.
const TX_PAD_TOP: usize = 2;
const TX_PAD_BOTTOM: usize = 4;
const TX_PAD_VER: usize = (TX_PAD_TOP + TX_PAD_BOTTOM);
// Pad 16 extra bytes to avoid reading overflow in SIMD optimization.
const TX_PAD_END: usize = 16;
const TX_PAD_2D: usize =
  ((MAX_TX_SIZE + TX_PAD_HOR) * (MAX_TX_SIZE + TX_PAD_VER) + TX_PAD_END);

const TX_CLASSES: usize = 3;

#[derive(Copy, Clone, PartialEq)]
pub enum TxClass {
  TX_CLASS_2D = 0,
  TX_CLASS_HORIZ = 1,
  TX_CLASS_VERT = 2
}

use context::TxClass::*;

static tx_type_to_class: [TxClass; TX_TYPES] = [
  TX_CLASS_2D,    // DCT_DCT
  TX_CLASS_2D,    // ADST_DCT
  TX_CLASS_2D,    // DCT_ADST
  TX_CLASS_2D,    // ADST_ADST
  TX_CLASS_2D,    // FLIPADST_DCT
  TX_CLASS_2D,    // DCT_FLIPADST
  TX_CLASS_2D,    // FLIPADST_FLIPADST
  TX_CLASS_2D,    // ADST_FLIPADST
  TX_CLASS_2D,    // FLIPADST_ADST
  TX_CLASS_2D,    // IDTX
  TX_CLASS_VERT,  // V_DCT
  TX_CLASS_HORIZ, // H_DCT
  TX_CLASS_VERT,  // V_ADST
  TX_CLASS_HORIZ, // H_ADST
  TX_CLASS_VERT,  // V_FLIPADST
  TX_CLASS_HORIZ  // H_FLIPADST
];

static eob_to_pos_small: [u8; 33] = [
    0, 1, 2,                                        // 0-2
    3, 3,                                           // 3-4
    4, 4, 4, 4,                                     // 5-8
    5, 5, 5, 5, 5, 5, 5, 5,                         // 9-16
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6  // 17-32
];

static eob_to_pos_large: [u8; 17] = [
    6,                               // place holder
    7,                               // 33-64
    8,  8,                           // 65-128
    9,  9,  9,  9,                   // 129-256
    10, 10, 10, 10, 10, 10, 10, 10,  // 257-512
    11                               // 513-
];


static k_eob_group_start: [u16; 12] = [ 0, 1, 2, 3, 5, 9,
                                        17, 33, 65, 129, 257, 513 ];
static k_eob_offset_bits: [u16; 12] = [ 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ];

fn clip_max3(x: u8) -> u8 {
  if x > 3 {
    3
  } else {
    x
  }
}

// The ctx offset table when TX is TX_CLASS_2D.
// TX col and row indices are clamped to 4

#[cfg_attr(rustfmt, rustfmt_skip)]
static av1_nz_map_ctx_offset: [[[i8; 5]; 5]; TxSize::TX_SIZES_ALL] = [
  // TX_4X4
  [
    [ 0,  1,  6,  6, 0],
    [ 1,  6,  6, 21, 0],
    [ 6,  6, 21, 21, 0],
    [ 6, 21, 21, 21, 0],
    [ 0,  0,  0,  0, 0]
  ],
  // TX_8X8
  [
    [ 0,  1,  6,  6, 21],
    [ 1,  6,  6, 21, 21],
    [ 6,  6, 21, 21, 21],
    [ 6, 21, 21, 21, 21],
    [21, 21, 21, 21, 21]
  ],
  // TX_16X16
  [
    [ 0,  1,  6,  6, 21],
    [ 1,  6,  6, 21, 21],
    [ 6,  6, 21, 21, 21],
    [ 6, 21, 21, 21, 21],
    [21, 21, 21, 21, 21]
  ],
  // TX_32X32
  [
    [ 0,  1,  6,  6, 21],
    [ 1,  6,  6, 21, 21],
    [ 6,  6, 21, 21, 21],
    [ 6, 21, 21, 21, 21],
    [21, 21, 21, 21, 21]
  ],
  // TX_64X64
  [
    [ 0,  1,  6,  6, 21],
    [ 1,  6,  6, 21, 21],
    [ 6,  6, 21, 21, 21],
    [ 6, 21, 21, 21, 21],
    [21, 21, 21, 21, 21]
  ],
  // TX_4X8
  [
    [ 0, 11, 11, 11, 0],
    [11, 11, 11, 11, 0],
    [ 6,  6, 21, 21, 0],
    [ 6, 21, 21, 21, 0],
    [21, 21, 21, 21, 0]
  ],
  // TX_8X4
  [
    [ 0, 16,  6,  6, 21],
    [16, 16,  6, 21, 21],
    [16, 16, 21, 21, 21],
    [16, 16, 21, 21, 21],
    [ 0,  0,  0,  0, 0]
  ],
  // TX_8X16
  [
    [ 0, 11, 11, 11, 11],
    [11, 11, 11, 11, 11],
    [ 6,  6, 21, 21, 21],
    [ 6, 21, 21, 21, 21],
    [21, 21, 21, 21, 21]
  ],
  // TX_16X8
  [
    [ 0, 16,  6,  6, 21],
    [16, 16,  6, 21, 21],
    [16, 16, 21, 21, 21],
    [16, 16, 21, 21, 21],
    [16, 16, 21, 21, 21]
  ],
  // TX_16X32
  [
    [ 0, 11, 11, 11, 11],
    [11, 11, 11, 11, 11],
    [ 6,  6, 21, 21, 21],
    [ 6, 21, 21, 21, 21],
    [21, 21, 21, 21, 21]
  ],
  // TX_32X16
  [
    [ 0, 16,  6,  6, 21],
    [16, 16,  6, 21, 21],
    [16, 16, 21, 21, 21],
    [16, 16, 21, 21, 21],
    [16, 16, 21, 21, 21]
  ],
  // TX_32X64
  [
    [ 0, 11, 11, 11, 11],
    [11, 11, 11, 11, 11],
    [ 6,  6, 21, 21, 21],
    [ 6, 21, 21, 21, 21],
    [21, 21, 21, 21, 21]
  ],
  // TX_64X32
  [
    [ 0, 16,  6,  6, 21],
    [16, 16,  6, 21, 21],
    [16, 16, 21, 21, 21],
    [16, 16, 21, 21, 21],
    [16, 16, 21, 21, 21]
  ],
  // TX_4X16
  [
    [ 0, 11, 11, 11, 0],
    [11, 11, 11, 11, 0],
    [ 6,  6, 21, 21, 0],
    [ 6, 21, 21, 21, 0],
    [21, 21, 21, 21, 0]
  ],
  // TX_16X4
  [
    [ 0, 16,  6,  6, 21],
    [16, 16,  6, 21, 21],
    [16, 16, 21, 21, 21],
    [16, 16, 21, 21, 21],
    [ 0,  0,  0,  0, 0]
  ],
  // TX_8X32
  [
    [ 0, 11, 11, 11, 11],
    [11, 11, 11, 11, 11],
    [ 6,  6, 21, 21, 21],
    [ 6, 21, 21, 21, 21],
    [21, 21, 21, 21, 21]
  ],
  // TX_32X8
  [
    [ 0, 16,  6,  6, 21],
    [16, 16,  6, 21, 21],
    [16, 16, 21, 21, 21],
    [16, 16, 21, 21, 21],
    [16, 16, 21, 21, 21]
  ],
  // TX_16X64
  [
    [ 0, 11, 11, 11, 11],
    [11, 11, 11, 11, 11],
    [ 6,  6, 21, 21, 21],
    [ 6, 21, 21, 21, 21],
    [21, 21, 21, 21, 21]
  ],
  // TX_64X16
  [
    [ 0, 16,  6,  6, 21],
    [16, 16,  6, 21, 21],
    [16, 16, 21, 21, 21],
    [16, 16, 21, 21, 21],
    [16, 16, 21, 21, 21]
  ]
];

const NZ_MAP_CTX_0: usize = SIG_COEF_CONTEXTS_2D;
const NZ_MAP_CTX_5: usize = (NZ_MAP_CTX_0 + 5);
const NZ_MAP_CTX_10: usize = (NZ_MAP_CTX_0 + 10);

static nz_map_ctx_offset_1d: [usize; 32] = [
  NZ_MAP_CTX_0,  NZ_MAP_CTX_5,  NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10,
  NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10,
  NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10,
  NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10,
  NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10,
  NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10,
  NZ_MAP_CTX_10, NZ_MAP_CTX_10 ];

const CONTEXT_MAG_POSITION_NUM: usize = 3;

static mag_ref_offset_with_txclass: [[[usize; 2]; CONTEXT_MAG_POSITION_NUM]; 3] = [
  [ [ 0, 1 ], [ 1, 0 ], [ 1, 1 ] ],
  [ [ 0, 1 ], [ 1, 0 ], [ 0, 2 ] ],
  [ [ 0, 1 ], [ 1, 0 ], [ 2, 0 ] ] ];

// End of Level Map

pub fn has_chroma(
  bo: &BlockOffset, bsize: BlockSize, subsampling_x: usize,
  subsampling_y: usize
) -> bool {
  let bw = bsize.width_mi();
  let bh = bsize.height_mi();

  ((bo.x & 0x01) == 1 || (bw & 0x01) == 0 || subsampling_x == 0)
    && ((bo.y & 0x01) == 1 || (bh & 0x01) == 0 || subsampling_y == 0)
}

pub fn get_tx_set(
  tx_size: TxSize, is_inter: bool, use_reduced_set: bool
) -> TxSet {
  let tx_size_sqr_up = tx_size.sqr_up();
  let tx_size_sqr = tx_size.sqr();
  if tx_size_sqr > TxSize::TX_32X32 {
    TxSet::TX_SET_DCTONLY
  } else if tx_size_sqr_up == TxSize::TX_32X32 {
    if is_inter {
      TxSet::TX_SET_DCT_IDTX
    } else {
      TxSet::TX_SET_DCTONLY
    }
  } else if use_reduced_set {
    if is_inter {
      TxSet::TX_SET_DCT_IDTX
    } else {
      TxSet::TX_SET_DTT4_IDTX
    }
  } else if is_inter {
    if tx_size_sqr == TxSize::TX_16X16 {
      TxSet::TX_SET_DTT9_IDTX_1DDCT
    } else {
      TxSet::TX_SET_ALL16
    }
  } else {
    if tx_size_sqr == TxSize::TX_16X16 {
      TxSet::TX_SET_DTT4_IDTX
    } else {
      TxSet::TX_SET_DTT4_IDTX_1DDCT
    }
  }
}

fn get_tx_set_index(
  tx_size: TxSize, is_inter: bool, use_reduced_set: bool
) -> i8 {
  let set_type = get_tx_set(tx_size, is_inter, use_reduced_set);

  if is_inter {
    tx_set_index_inter[set_type as usize]
  } else {
    tx_set_index_intra[set_type as usize]
  }
}

static intra_mode_to_tx_type_context: [TxType; INTRA_MODES] = [
  DCT_DCT,   // DC
  ADST_DCT,  // V
  DCT_ADST,  // H
  DCT_DCT,   // D45
  ADST_ADST, // D135
  ADST_DCT,  // D117
  DCT_ADST,  // D153
  DCT_ADST,  // D207
  ADST_DCT,  // D63
  ADST_ADST, // SMOOTH
  ADST_DCT,  // SMOOTH_V
  DCT_ADST,  // SMOOTH_H
  ADST_ADST, // PAETH
];

static uv2y: [PredictionMode; UV_INTRA_MODES] = [
  DC_PRED,       // UV_DC_PRED
  V_PRED,        // UV_V_PRED
  H_PRED,        // UV_H_PRED
  D45_PRED,      // UV_D45_PRED
  D135_PRED,     // UV_D135_PRED
  D117_PRED,     // UV_D117_PRED
  D153_PRED,     // UV_D153_PRED
  D207_PRED,     // UV_D207_PRED
  D63_PRED,      // UV_D63_PRED
  SMOOTH_PRED,   // UV_SMOOTH_PRED
  SMOOTH_V_PRED, // UV_SMOOTH_V_PRED
  SMOOTH_H_PRED, // UV_SMOOTH_H_PRED
  PAETH_PRED,    // UV_PAETH_PRED
  DC_PRED        // CFL_PRED
];

pub fn y_intra_mode_to_tx_type_context(pred: PredictionMode) -> TxType {
  intra_mode_to_tx_type_context[pred as usize]
}

pub fn uv_intra_mode_to_tx_type_context(pred: PredictionMode) -> TxType {
  intra_mode_to_tx_type_context[uv2y[pred as usize] as usize]
}

#[derive(Clone,Copy)]
#[repr(C)]
pub struct NMVComponent {
  classes_cdf: [u16; MV_CLASSES + 1],
  class0_fp_cdf: [[u16; MV_FP_SIZE + 1]; CLASS0_SIZE],
  fp_cdf: [u16; MV_FP_SIZE + 1],
  sign_cdf: [u16; 2 + 1],
  class0_hp_cdf: [u16; 2 + 1],
  hp_cdf: [u16; 2 + 1],
  class0_cdf: [u16; CLASS0_SIZE + 1],
  bits_cdf: [[u16; 2 + 1]; MV_OFFSET_BITS],
}

#[derive(Clone,Copy)]
#[repr(C)]
pub struct NMVContext {
  joints_cdf: [u16; MV_JOINTS + 1],
  comps: [NMVComponent; 2],
}

extern "C" {
  static default_partition_cdf:
    [[u16; EXT_PARTITION_TYPES + 1]; PARTITION_CONTEXTS];
  static default_kf_y_mode_cdf:
    [[[u16; INTRA_MODES + 1]; KF_MODE_CONTEXTS]; KF_MODE_CONTEXTS];
  static default_if_y_mode_cdf: [[u16; INTRA_MODES + 1]; BLOCK_SIZE_GROUPS];
  static default_uv_mode_cdf: [[[u16; UV_INTRA_MODES + 1]; INTRA_MODES]; 2];
  static default_cfl_sign_cdf: [u16; CFL_JOINT_SIGNS + 1];
  static default_cfl_alpha_cdf: [[u16; CFL_ALPHABET_SIZE + 1]; CFL_ALPHA_CONTEXTS];
  static default_newmv_cdf: [[u16; 2 + 1]; NEWMV_MODE_CONTEXTS];
  static default_zeromv_cdf: [[u16; 2 + 1]; GLOBALMV_MODE_CONTEXTS];
  static default_refmv_cdf: [[u16; 2 + 1]; REFMV_MODE_CONTEXTS];
  static default_intra_ext_tx_cdf:
    [[[[u16; TX_TYPES + 1]; INTRA_MODES]; TX_SIZES]; TX_SETS_INTRA];
  static default_inter_ext_tx_cdf:
    [[[u16; TX_TYPES + 1]; TX_SIZES]; TX_SETS_INTER];
  static default_skip_cdfs: [[u16; 3]; SKIP_CONTEXTS];
  static default_intra_inter_cdf: [[u16; 3]; INTRA_INTER_CONTEXTS];
  static default_angle_delta_cdf:
    [[u16; 2 * MAX_ANGLE_DELTA + 1 + 1]; DIRECTIONAL_MODES];
  static default_filter_intra_cdfs: [[u16; 3]; BlockSize::BLOCK_SIZES_ALL];

  static default_single_ref_cdf: [[[u16; 2 + 1]; SINGLE_REFS - 1]; REF_CONTEXTS];
  static av1_scan_orders: [[SCAN_ORDER; TX_TYPES]; TxSize::TX_SIZES_ALL];
  static default_delta_lf_multi_cdf: [[u16; DELTA_LF_PROBS + 1 + 1]; FRAME_LF_COUNT];
  static default_delta_lf_cdf: [u16; DELTA_LF_PROBS + 1 + 1];

  // lv_map
  static av1_default_txb_skip_cdfs:
    [[[[u16; 3]; TXB_SKIP_CONTEXTS]; TxSize::TX_SIZES]; 4];
  static av1_default_dc_sign_cdfs:
    [[[[u16; 3]; DC_SIGN_CONTEXTS]; PLANE_TYPES]; 4];
  static av1_default_eob_extra_cdfs:
    [[[[[u16; 3]; EOB_COEF_CONTEXTS]; PLANE_TYPES]; TxSize::TX_SIZES]; 4];

  static av1_default_eob_multi16_cdfs: [[[[u16; 5 + 1]; 2]; PLANE_TYPES]; 4];
  static av1_default_eob_multi32_cdfs: [[[[u16; 6 + 1]; 2]; PLANE_TYPES]; 4];
  static av1_default_eob_multi64_cdfs: [[[[u16; 7 + 1]; 2]; PLANE_TYPES]; 4];
  static av1_default_eob_multi128_cdfs: [[[[u16; 8 + 1]; 2]; PLANE_TYPES]; 4];
  static av1_default_eob_multi256_cdfs: [[[[u16; 9 + 1]; 2]; PLANE_TYPES]; 4];
  static av1_default_eob_multi512_cdfs: [[[[u16; 10 + 1]; 2]; PLANE_TYPES]; 4];
  static av1_default_eob_multi1024_cdfs: [[[[u16; 11 + 1]; 2]; PLANE_TYPES]; 4];

  static av1_default_coeff_base_eob_multi_cdfs:
    [[[[[u16; 3 + 1]; SIG_COEF_CONTEXTS_EOB]; PLANE_TYPES]; TxSize::TX_SIZES]; 4];
  static av1_default_coeff_base_multi_cdfs:
    [[[[[u16; 4 + 1]; SIG_COEF_CONTEXTS]; PLANE_TYPES]; TxSize::TX_SIZES]; 4];
  static av1_default_coeff_lps_multi_cdfs: [[[[[u16; BR_CDF_SIZE + 1];
    LEVEL_CONTEXTS]; PLANE_TYPES];
    TxSize::TX_SIZES]; 4];

  static default_nmv_context: NMVContext;
  static default_drl_cdf:[[u16; 2 + 1]; DRL_MODE_CONTEXTS];
}

#[repr(C)]
pub struct SCAN_ORDER {
  // FIXME: don't hardcode sizes
  pub scan: &'static [u16; 32 * 32],
  pub iscan: &'static [u16; 32 * 32],
  pub neighbors: &'static [u16; ((32 * 32) + 1) * 2]
}

#[derive(Clone)]
pub struct CandidateMV {
  pub this_mv: MotionVector,
  pub comp_mv: MotionVector,
  pub weight: u32
}

#[derive(Clone,Copy)]
pub struct CDFContext {
  partition_cdf: [[u16; EXT_PARTITION_TYPES + 1]; PARTITION_CONTEXTS],
  kf_y_cdf: [[[u16; INTRA_MODES + 1]; KF_MODE_CONTEXTS]; KF_MODE_CONTEXTS],
  y_mode_cdf: [[u16; INTRA_MODES + 1]; BLOCK_SIZE_GROUPS],
  uv_mode_cdf: [[[u16; UV_INTRA_MODES + 1]; INTRA_MODES]; 2],
  cfl_sign_cdf: [u16; CFL_JOINT_SIGNS + 1],
  cfl_alpha_cdf: [[u16; CFL_ALPHABET_SIZE + 1]; CFL_ALPHA_CONTEXTS],
  newmv_cdf: [[u16; 2 + 1]; NEWMV_MODE_CONTEXTS],
  zeromv_cdf: [[u16; 2 + 1]; GLOBALMV_MODE_CONTEXTS],
  refmv_cdf: [[u16; 2 + 1]; REFMV_MODE_CONTEXTS],
  intra_tx_cdf:
    [[[[u16; TX_TYPES + 1]; INTRA_MODES]; TX_SIZES]; TX_SETS_INTRA],
  inter_tx_cdf: [[[u16; TX_TYPES + 1]; TX_SIZES]; TX_SETS_INTER],
  skip_cdfs: [[u16; 3]; SKIP_CONTEXTS],
  intra_inter_cdfs: [[u16; 3]; INTRA_INTER_CONTEXTS],
  angle_delta_cdf: [[u16; 2 * MAX_ANGLE_DELTA + 1 + 1]; DIRECTIONAL_MODES],
  filter_intra_cdfs: [[u16; 3]; BlockSize::BLOCK_SIZES_ALL],
  single_ref_cdfs: [[[u16; 2 + 1]; SINGLE_REFS - 1]; REF_CONTEXTS],
  drl_cdfs: [[u16; 2 + 1]; DRL_MODE_CONTEXTS],
  nmv_context: NMVContext,
  deblock_delta_multi_cdf: [[u16; DELTA_LF_PROBS + 1 + 1]; FRAME_LF_COUNT],
  deblock_delta_cdf: [u16; DELTA_LF_PROBS + 1 + 1],

  // lv_map
  txb_skip_cdf: [[[u16; 3]; TXB_SKIP_CONTEXTS]; TxSize::TX_SIZES],
  dc_sign_cdf: [[[u16; 3]; DC_SIGN_CONTEXTS]; PLANE_TYPES],
  eob_extra_cdf:
    [[[[u16; 3]; EOB_COEF_CONTEXTS]; PLANE_TYPES]; TxSize::TX_SIZES],

  eob_flag_cdf16: [[[u16; 5 + 1]; 2]; PLANE_TYPES],
  eob_flag_cdf32: [[[u16; 6 + 1]; 2]; PLANE_TYPES],
  eob_flag_cdf64: [[[u16; 7 + 1]; 2]; PLANE_TYPES],
  eob_flag_cdf128: [[[u16; 8 + 1]; 2]; PLANE_TYPES],
  eob_flag_cdf256: [[[u16; 9 + 1]; 2]; PLANE_TYPES],
  eob_flag_cdf512: [[[u16; 10 + 1]; 2]; PLANE_TYPES],
  eob_flag_cdf1024: [[[u16; 11 + 1]; 2]; PLANE_TYPES],

  coeff_base_eob_cdf:
    [[[[u16; 3 + 1]; SIG_COEF_CONTEXTS_EOB]; PLANE_TYPES]; TxSize::TX_SIZES],
  coeff_base_cdf:
    [[[[u16; 4 + 1]; SIG_COEF_CONTEXTS]; PLANE_TYPES]; TxSize::TX_SIZES],
  coeff_br_cdf: [[[[u16; BR_CDF_SIZE + 1]; LEVEL_CONTEXTS]; PLANE_TYPES];
    TxSize::TX_SIZES]
}

impl CDFContext {
    pub fn new(quantizer: u8) -> CDFContext {
    let qctx = match quantizer {
      0..=20 => 0,
      21..=60 => 1,
      61..=120 => 2,
      _ => 3
    };
    CDFContext {
      partition_cdf: default_partition_cdf,
      kf_y_cdf: default_kf_y_mode_cdf,
      y_mode_cdf: default_if_y_mode_cdf,
      uv_mode_cdf: default_uv_mode_cdf,
      cfl_sign_cdf: default_cfl_sign_cdf,
      cfl_alpha_cdf: default_cfl_alpha_cdf,
      newmv_cdf: default_newmv_cdf,
      zeromv_cdf: default_zeromv_cdf,
      refmv_cdf: default_refmv_cdf,
      intra_tx_cdf: default_intra_ext_tx_cdf,
      inter_tx_cdf: default_inter_ext_tx_cdf,
      skip_cdfs: default_skip_cdfs,
      intra_inter_cdfs: default_intra_inter_cdf,
      angle_delta_cdf: default_angle_delta_cdf,
      filter_intra_cdfs: default_filter_intra_cdfs,
      single_ref_cdfs: default_single_ref_cdf,
      drl_cdfs: default_drl_cdf,
      nmv_context: default_nmv_context,
      deblock_delta_multi_cdf: default_delta_lf_multi_cdf,
      deblock_delta_cdf: default_delta_lf_cdf,

      // lv_map
      txb_skip_cdf: av1_default_txb_skip_cdfs[qctx],
      dc_sign_cdf: av1_default_dc_sign_cdfs[qctx],
      eob_extra_cdf: av1_default_eob_extra_cdfs[qctx],

      eob_flag_cdf16: av1_default_eob_multi16_cdfs[qctx],
      eob_flag_cdf32: av1_default_eob_multi32_cdfs[qctx],
      eob_flag_cdf64: av1_default_eob_multi64_cdfs[qctx],
      eob_flag_cdf128: av1_default_eob_multi128_cdfs[qctx],
      eob_flag_cdf256: av1_default_eob_multi256_cdfs[qctx],
      eob_flag_cdf512: av1_default_eob_multi512_cdfs[qctx],
      eob_flag_cdf1024: av1_default_eob_multi1024_cdfs[qctx],

      coeff_base_eob_cdf: av1_default_coeff_base_eob_multi_cdfs[qctx],
      coeff_base_cdf: av1_default_coeff_base_multi_cdfs[qctx],
      coeff_br_cdf: av1_default_coeff_lps_multi_cdfs[qctx]
    }
  }

  pub fn reset_counts(&mut self) {
    macro_rules! reset_1d {
      ($field:expr) => (let r = $field.last_mut().unwrap(); *r = 0;)
    }
    macro_rules! reset_2d {
      ($field:expr) => (for mut x in $field.iter_mut() { reset_1d!(x); })
    }
    macro_rules! reset_3d {
      ($field:expr) => (for mut x in $field.iter_mut() { reset_2d!(x); })
    }
    macro_rules! reset_4d {
      ($field:expr) => (for mut x in $field.iter_mut() { reset_3d!(x); })
    }

    for i in 0..4 { self.partition_cdf[i][4] = 0; }
    for i in 4..16 { self.partition_cdf[i][10] = 0; }
    for i in 16..20 { self.partition_cdf[i][8] = 0; }

    reset_3d!(self.kf_y_cdf);
    reset_2d!(self.y_mode_cdf);

    for i in 0..INTRA_MODES {
      self.uv_mode_cdf[0][i][UV_INTRA_MODES - 1] = 0;
      self.uv_mode_cdf[1][i][UV_INTRA_MODES] = 0;
    }
    reset_1d!(self.cfl_sign_cdf);
    reset_2d!(self.cfl_alpha_cdf);
    reset_2d!(self.newmv_cdf);
    reset_2d!(self.zeromv_cdf);
    reset_2d!(self.refmv_cdf);

    for i in 0..TX_SIZES {
      for j in 0..INTRA_MODES {
        self.intra_tx_cdf[1][i][j][7] = 0;
        self.intra_tx_cdf[2][i][j][5] = 0;
      }
      self.inter_tx_cdf[1][i][16] = 0;
      self.inter_tx_cdf[2][i][12] = 0;
      self.inter_tx_cdf[3][i][2] = 0;
    }

    reset_2d!(self.skip_cdfs);
    reset_2d!(self.intra_inter_cdfs);
    reset_2d!(self.angle_delta_cdf);
    reset_2d!(self.filter_intra_cdfs);
    reset_3d!(self.single_ref_cdfs);
    reset_2d!(self.drl_cdfs);
    reset_2d!(self.deblock_delta_multi_cdf);
    reset_1d!(self.deblock_delta_cdf);

    reset_1d!(self.nmv_context.joints_cdf);
    for i in 0..2 {
      reset_1d!(self.nmv_context.comps[i].classes_cdf);
      reset_2d!(self.nmv_context.comps[i].class0_fp_cdf);
      reset_1d!(self.nmv_context.comps[i].fp_cdf);
      reset_1d!(self.nmv_context.comps[i].sign_cdf);
      reset_1d!(self.nmv_context.comps[i].class0_hp_cdf);
      reset_1d!(self.nmv_context.comps[i].hp_cdf);
      reset_1d!(self.nmv_context.comps[i].class0_cdf);
      reset_2d!(self.nmv_context.comps[i].bits_cdf);
    }

    // lv_map
    reset_3d!(self.txb_skip_cdf);
    reset_3d!(self.dc_sign_cdf);
    reset_4d!(self.eob_extra_cdf);

    reset_3d!(self.eob_flag_cdf16);
    reset_3d!(self.eob_flag_cdf32);
    reset_3d!(self.eob_flag_cdf64);
    reset_3d!(self.eob_flag_cdf128);
    reset_3d!(self.eob_flag_cdf256);
    reset_3d!(self.eob_flag_cdf512);
    reset_3d!(self.eob_flag_cdf1024);

    reset_4d!(self.coeff_base_eob_cdf);
    reset_4d!(self.coeff_base_cdf);
    reset_4d!(self.coeff_br_cdf);
  }

  pub fn build_map(&self) -> Vec<(&'static str, usize, usize)> {
    use std::mem::size_of_val;

    let partition_cdf_start =
      self.partition_cdf.first().unwrap().as_ptr() as usize;
    let partition_cdf_end =
      partition_cdf_start + size_of_val(&self.partition_cdf);
    let kf_y_cdf_start = self.kf_y_cdf.first().unwrap().as_ptr() as usize;
    let kf_y_cdf_end = kf_y_cdf_start + size_of_val(&self.kf_y_cdf);
    let y_mode_cdf_start = self.y_mode_cdf.first().unwrap().as_ptr() as usize;
    let y_mode_cdf_end = y_mode_cdf_start + size_of_val(&self.y_mode_cdf);
    let uv_mode_cdf_start =
      self.uv_mode_cdf.first().unwrap().as_ptr() as usize;
    let uv_mode_cdf_end = uv_mode_cdf_start + size_of_val(&self.uv_mode_cdf);
    let cfl_sign_cdf_start = self.cfl_sign_cdf.as_ptr() as usize;
    let cfl_sign_cdf_end = cfl_sign_cdf_start + size_of_val(&self.cfl_sign_cdf);
    let cfl_alpha_cdf_start =
      self.cfl_alpha_cdf.first().unwrap().as_ptr() as usize;
    let cfl_alpha_cdf_end =
      cfl_alpha_cdf_start + size_of_val(&self.cfl_alpha_cdf);
    let intra_tx_cdf_start =
      self.intra_tx_cdf.first().unwrap().as_ptr() as usize;
    let intra_tx_cdf_end =
      intra_tx_cdf_start + size_of_val(&self.intra_tx_cdf);
    let inter_tx_cdf_start =
      self.inter_tx_cdf.first().unwrap().as_ptr() as usize;
    let inter_tx_cdf_end =
      inter_tx_cdf_start + size_of_val(&self.inter_tx_cdf);
    let skip_cdfs_start = self.skip_cdfs.first().unwrap().as_ptr() as usize;
    let skip_cdfs_end = skip_cdfs_start + size_of_val(&self.skip_cdfs);
    let intra_inter_cdfs_start =
      self.intra_inter_cdfs.first().unwrap().as_ptr() as usize;
    let intra_inter_cdfs_end =
      intra_inter_cdfs_start + size_of_val(&self.intra_inter_cdfs);
    let angle_delta_cdf_start =
      self.angle_delta_cdf.first().unwrap().as_ptr() as usize;
    let angle_delta_cdf_end =
      angle_delta_cdf_start + size_of_val(&self.angle_delta_cdf);
    let filter_intra_cdfs_start =
      self.filter_intra_cdfs.first().unwrap().as_ptr() as usize;
    let filter_intra_cdfs_end =
      filter_intra_cdfs_start + size_of_val(&self.filter_intra_cdfs);
    let deblock_delta_multi_cdf_start =
      self.deblock_delta_multi_cdf.first().unwrap().as_ptr() as usize;
    let deblock_delta_multi_cdf_end =
      deblock_delta_multi_cdf_start + size_of_val(&self.deblock_delta_multi_cdf);
    let deblock_delta_cdf_start =
      self.deblock_delta_cdf.as_ptr() as usize;
    let deblock_delta_cdf_end =
      deblock_delta_cdf_start + size_of_val(&self.deblock_delta_cdf);

    let txb_skip_cdf_start =
      self.txb_skip_cdf.first().unwrap().as_ptr() as usize;
    let txb_skip_cdf_end =
      txb_skip_cdf_start + size_of_val(&self.txb_skip_cdf);
    let dc_sign_cdf_start =
      self.dc_sign_cdf.first().unwrap().as_ptr() as usize;
    let dc_sign_cdf_end = dc_sign_cdf_start + size_of_val(&self.dc_sign_cdf);
    let eob_extra_cdf_start =
      self.eob_extra_cdf.first().unwrap().as_ptr() as usize;
    let eob_extra_cdf_end =
      eob_extra_cdf_start + size_of_val(&self.eob_extra_cdf);
    let eob_flag_cdf16_start =
      self.eob_flag_cdf16.first().unwrap().as_ptr() as usize;
    let eob_flag_cdf16_end =
      eob_flag_cdf16_start + size_of_val(&self.eob_flag_cdf16);
    let eob_flag_cdf32_start =
      self.eob_flag_cdf32.first().unwrap().as_ptr() as usize;
    let eob_flag_cdf32_end =
      eob_flag_cdf32_start + size_of_val(&self.eob_flag_cdf32);
    let eob_flag_cdf64_start =
      self.eob_flag_cdf64.first().unwrap().as_ptr() as usize;
    let eob_flag_cdf64_end =
      eob_flag_cdf64_start + size_of_val(&self.eob_flag_cdf64);
    let eob_flag_cdf128_start =
      self.eob_flag_cdf128.first().unwrap().as_ptr() as usize;
    let eob_flag_cdf128_end =
      eob_flag_cdf128_start + size_of_val(&self.eob_flag_cdf128);
    let eob_flag_cdf256_start =
      self.eob_flag_cdf256.first().unwrap().as_ptr() as usize;
    let eob_flag_cdf256_end =
      eob_flag_cdf256_start + size_of_val(&self.eob_flag_cdf256);
    let eob_flag_cdf512_start =
      self.eob_flag_cdf512.first().unwrap().as_ptr() as usize;
    let eob_flag_cdf512_end =
      eob_flag_cdf512_start + size_of_val(&self.eob_flag_cdf512);
    let eob_flag_cdf1024_start =
      self.eob_flag_cdf1024.first().unwrap().as_ptr() as usize;
    let eob_flag_cdf1024_end =
      eob_flag_cdf1024_start + size_of_val(&self.eob_flag_cdf1024);
    let coeff_base_eob_cdf_start =
      self.coeff_base_eob_cdf.first().unwrap().as_ptr() as usize;
    let coeff_base_eob_cdf_end =
      coeff_base_eob_cdf_start + size_of_val(&self.coeff_base_eob_cdf);
    let coeff_base_cdf_start =
      self.coeff_base_cdf.first().unwrap().as_ptr() as usize;
    let coeff_base_cdf_end =
      coeff_base_cdf_start + size_of_val(&self.coeff_base_cdf);
    let coeff_br_cdf_start =
      self.coeff_br_cdf.first().unwrap().as_ptr() as usize;
    let coeff_br_cdf_end =
      coeff_br_cdf_start + size_of_val(&self.coeff_br_cdf);

    vec![
      ("partition_cdf", partition_cdf_start, partition_cdf_end),
      ("kf_y_cdf", kf_y_cdf_start, kf_y_cdf_end),
      ("y_mode_cdf", y_mode_cdf_start, y_mode_cdf_end),
      ("uv_mode_cdf", uv_mode_cdf_start, uv_mode_cdf_end),
      ("cfl_sign_cdf", cfl_sign_cdf_start, cfl_sign_cdf_end),
      ("cfl_alpha_cdf", cfl_alpha_cdf_start, cfl_alpha_cdf_end),
      ("intra_tx_cdf", intra_tx_cdf_start, intra_tx_cdf_end),
      ("inter_tx_cdf", inter_tx_cdf_start, inter_tx_cdf_end),
      ("skip_cdfs", skip_cdfs_start, skip_cdfs_end),
      ("intra_inter_cdfs", intra_inter_cdfs_start, intra_inter_cdfs_end),
      ("angle_delta_cdf", angle_delta_cdf_start, angle_delta_cdf_end),
      ("filter_intra_cdfs", filter_intra_cdfs_start, filter_intra_cdfs_end),
      ("deblock_delta_multi_cdf", deblock_delta_multi_cdf_start, deblock_delta_multi_cdf_end),
      ("deblock_delta_cdf", deblock_delta_cdf_start, deblock_delta_cdf_end),
      ("txb_skip_cdf", txb_skip_cdf_start, txb_skip_cdf_end),
      ("dc_sign_cdf", dc_sign_cdf_start, dc_sign_cdf_end),
      ("eob_extra_cdf", eob_extra_cdf_start, eob_extra_cdf_end),
      ("eob_flag_cdf16", eob_flag_cdf16_start, eob_flag_cdf16_end),
      ("eob_flag_cdf32", eob_flag_cdf32_start, eob_flag_cdf32_end),
      ("eob_flag_cdf64", eob_flag_cdf64_start, eob_flag_cdf64_end),
      ("eob_flag_cdf128", eob_flag_cdf128_start, eob_flag_cdf128_end),
      ("eob_flag_cdf256", eob_flag_cdf256_start, eob_flag_cdf256_end),
      ("eob_flag_cdf512", eob_flag_cdf512_start, eob_flag_cdf512_end),
      ("eob_flag_cdf1024", eob_flag_cdf1024_start, eob_flag_cdf1024_end),
      ("coeff_base_eob_cdf", coeff_base_eob_cdf_start, coeff_base_eob_cdf_end),
      ("coeff_base_cdf", coeff_base_cdf_start, coeff_base_cdf_end),
      ("coeff_br_cdf", coeff_br_cdf_start, coeff_br_cdf_end),
    ]
  }
}

impl fmt::Debug for CDFContext {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "CDFContext contains too many numbers to print :-(")
  }
}

#[cfg(test)]
mod test {
  #[test]
  fn cdf_map() {
    use super::*;

    let cdf = CDFContext::new(8);
    let cdf_map = FieldMap {
      map: cdf.build_map()
    };
    let f = &cdf.partition_cdf[2];
    cdf_map.lookup(f.as_ptr() as usize);
  }

  use super::CFLSign;
  use super::CFLSign::*;

  static cfl_alpha_signs: [[CFLSign; 2]; 8] = [
    [ CFL_SIGN_ZERO, CFL_SIGN_NEG ],
    [ CFL_SIGN_ZERO, CFL_SIGN_POS ],
    [ CFL_SIGN_NEG, CFL_SIGN_ZERO ],
    [ CFL_SIGN_NEG, CFL_SIGN_NEG ],
    [ CFL_SIGN_NEG, CFL_SIGN_POS ],
    [ CFL_SIGN_POS, CFL_SIGN_ZERO ],
    [ CFL_SIGN_POS, CFL_SIGN_NEG ],
    [ CFL_SIGN_POS, CFL_SIGN_POS ]
  ];

  static cfl_context: [[usize; 8]; 2] = [
    [ 0, 0, 0, 1, 2, 3, 4, 5 ],
    [ 0, 3, 0, 1, 4, 0, 2, 5 ]
  ];

  #[test]
  fn cfl_joint_sign() {
    use super::*;

    let mut cfl = CFLParams::new();
    for (joint_sign, &signs) in cfl_alpha_signs.iter().enumerate() {
      cfl.sign = signs;
      assert!(cfl.joint_sign() as usize == joint_sign);
      for uv in 0..2 {
        if signs[uv] != CFL_SIGN_ZERO {
          assert!(cfl.context(uv) == cfl_context[uv][joint_sign]);
        }
      }
    }
  }
}

const SUPERBLOCK_TO_PLANE_SHIFT: usize = MAX_SB_SIZE_LOG2;
const SUPERBLOCK_TO_BLOCK_SHIFT: usize = MAX_MIB_SIZE_LOG2;
pub const BLOCK_TO_PLANE_SHIFT: usize = MI_SIZE_LOG2;
pub const LOCAL_BLOCK_MASK: usize = (1 << SUPERBLOCK_TO_BLOCK_SHIFT) - 1;

/// Absolute offset in superblocks inside a plane, where a superblock is defined
/// to be an N*N square where N = (1 << SUPERBLOCK_TO_PLANE_SHIFT).
#[derive(Clone)]
pub struct SuperBlockOffset {
  pub x: usize,
  pub y: usize
}

impl SuperBlockOffset {
  /// Offset of a block inside the current superblock.
  pub fn block_offset(&self, block_x: usize, block_y: usize) -> BlockOffset {
    BlockOffset {
      x: (self.x << SUPERBLOCK_TO_BLOCK_SHIFT) + block_x,
      y: (self.y << SUPERBLOCK_TO_BLOCK_SHIFT) + block_y
    }
  }

  /// Offset of the top-left pixel of this block.
  pub fn plane_offset(&self, plane: &PlaneConfig) -> PlaneOffset {
    PlaneOffset {
      x: (self.x as isize) << (SUPERBLOCK_TO_PLANE_SHIFT - plane.xdec),
      y: (self.y as isize) << (SUPERBLOCK_TO_PLANE_SHIFT - plane.ydec)
    }
  }
}

/// Absolute offset in blocks inside a plane, where a block is defined
/// to be an N*N square where N = (1 << BLOCK_TO_PLANE_SHIFT).
#[derive(Clone)]
pub struct BlockOffset {
  pub x: usize,
  pub y: usize
}

impl BlockOffset {
  /// Offset of the superblock in which this block is located.
  pub fn sb_offset(&self) -> SuperBlockOffset {
    SuperBlockOffset {
      x: self.x >> SUPERBLOCK_TO_BLOCK_SHIFT,
      y: self.y >> SUPERBLOCK_TO_BLOCK_SHIFT
    }
  }

  /// Offset of the top-left pixel of this block.
  pub fn plane_offset(&self, plane: &PlaneConfig) -> PlaneOffset {
    let po = self.sb_offset().plane_offset(plane);

    let x_offset = self.x & LOCAL_BLOCK_MASK;
    let y_offset = self.y & LOCAL_BLOCK_MASK;

    PlaneOffset {
        x: po.x + (x_offset as isize >> plane.xdec << BLOCK_TO_PLANE_SHIFT),
        y: po.y + (y_offset as isize >> plane.ydec << BLOCK_TO_PLANE_SHIFT)
    }
  }

  pub fn y_in_sb(&self) -> usize {
    self.y % MAX_MIB_SIZE
  }

  pub fn with_offset(&self, col_offset: isize, row_offset: isize) -> BlockOffset {
    let x = self.x as isize + col_offset;
    let y = self.y as isize + row_offset;

    BlockOffset {
      x: x as usize,
      y: y as usize
    }
  }
}

#[derive(Copy, Clone)]
pub struct Block {
  pub mode: PredictionMode,
  pub partition: PartitionType,
  pub skip: bool,
  pub ref_frames: [usize; 2],
  pub mv: [MotionVector; 2],
  pub neighbors_ref_counts: [usize; TOTAL_REFS_PER_FRAME],
  pub cdef_index: u8,
  pub n4_w: usize, /* block width in the unit of mode_info */
  pub n4_h: usize, /* block height in the unit of mode_info */
  pub tx_w: usize, /* transform width in the unit of mode_info */
  pub tx_h: usize, /* transform height in the unit of mode_info */
  pub is_sec_rect: bool,
  // The block-level deblock_deltas are left-shifted by
  // fi.deblock.block_delta_shift and added to the frame-configured
  // deltas
  pub deblock_deltas: [i8; FRAME_LF_COUNT]
}

impl Block {
  pub fn default() -> Block {
    Block {
      mode: PredictionMode::DC_PRED,
      partition: PartitionType::PARTITION_NONE,
      skip: false,
      ref_frames: [INTRA_FRAME; 2],
      mv: [ MotionVector { row:0, col: 0 }; 2],
      neighbors_ref_counts: [0; TOTAL_REFS_PER_FRAME],
      cdef_index: 0,
      n4_w: BLOCK_64X64.width_mi(),
      n4_h: BLOCK_64X64.height_mi(),
      tx_w: TX_64X64.width_mi(),
      tx_h: TX_64X64.height_mi(),
      is_sec_rect: false,
      deblock_deltas: [0, 0, 0, 0]
    }
  }
  pub fn is_inter(&self) -> bool {
    self.mode >= PredictionMode::NEARESTMV
  }
  pub fn has_second_ref(&self) -> bool {
    self.ref_frames[1] > INTRA_FRAME
  }
}

pub struct TXB_CTX {
  pub txb_skip_ctx: usize,
  pub dc_sign_ctx: usize
}

#[derive(Clone, Default)]
pub struct BlockContext {
  pub cols: usize,
  pub rows: usize,
  pub cdef_coded: bool,
  pub code_deltas: bool,
  above_partition_context: Vec<u8>,
  left_partition_context: [u8; MAX_MIB_SIZE],
  above_coeff_context: [Vec<u8>; PLANES],
  left_coeff_context: [[u8; MAX_MIB_SIZE]; PLANES],
  blocks: Vec<Vec<Block>>
}

impl BlockContext {
  pub fn new(cols: usize, rows: usize) -> BlockContext {
    // Align power of two
    let aligned_cols = (cols + ((1 << MAX_MIB_SIZE_LOG2) - 1))
      & !((1 << MAX_MIB_SIZE_LOG2) - 1);
    BlockContext {
      cols,
      rows,
      cdef_coded: false,
      code_deltas: false,
      above_partition_context: vec![0; aligned_cols],
      left_partition_context: [0; MAX_MIB_SIZE],
      above_coeff_context: [
        vec![0; cols << (MI_SIZE_LOG2 - TxSize::smallest_width_log2())],
        vec![0; cols << (MI_SIZE_LOG2 - TxSize::smallest_width_log2())],
        vec![0; cols << (MI_SIZE_LOG2 - TxSize::smallest_width_log2())]
      ],
      left_coeff_context: [[0; MAX_MIB_SIZE]; PLANES],
      blocks: vec![vec![Block::default(); cols]; rows]
    }
  }

  pub fn checkpoint(&mut self) -> BlockContext {
    BlockContext {
      cols: self.cols,
      rows: self.rows,
      cdef_coded: self.cdef_coded,
      code_deltas: self.code_deltas,
      above_partition_context: self.above_partition_context.clone(),
      left_partition_context: self.left_partition_context,
      above_coeff_context: self.above_coeff_context.clone(),
      left_coeff_context: self.left_coeff_context,
      blocks: vec![vec![Block::default(); 0]; 0]
    }
  }

  pub fn rollback(&mut self, checkpoint: &BlockContext) {
    self.cols = checkpoint.cols;
    self.rows = checkpoint.rows;
    self.cdef_coded = checkpoint.cdef_coded;
    self.above_partition_context = checkpoint.above_partition_context.clone();
    self.left_partition_context = checkpoint.left_partition_context;
    self.above_coeff_context = checkpoint.above_coeff_context.clone();
    self.left_coeff_context = checkpoint.left_coeff_context;
  }

  pub fn at_mut(&mut self, bo: &BlockOffset) -> &mut Block {
    &mut self.blocks[bo.y][bo.x]
  }

  pub fn at(&self, bo: &BlockOffset) -> &Block {
    &self.blocks[bo.y][bo.x]
  }

  pub fn above_of(&mut self, bo: &BlockOffset) -> Block {
    if bo.y > 0 {
      self.blocks[bo.y - 1][bo.x]
    } else {
      Block::default()
    }
  }

  pub fn left_of(&mut self, bo: &BlockOffset) -> Block {
    if bo.x > 0 {
      self.blocks[bo.y][bo.x - 1]
    } else {
      Block::default()
    }
  }

  pub fn for_each<F>(&mut self, bo: &BlockOffset, bsize: BlockSize, f: F) -> ()
  where
    F: Fn(&mut Block) -> ()
  {
    let bw = bsize.width_mi();
    let bh = bsize.height_mi();
    for y in 0..bh {
      for x in 0..bw {
        f(&mut self.blocks[bo.y + y as usize][bo.x + x as usize]);
      }
    }
  }

  pub fn set_dc_sign(&mut self, cul_level: &mut u32, dc_val: i32) {
    if dc_val < 0 {
      *cul_level |= 1 << COEFF_CONTEXT_BITS;
    } else if dc_val > 0 {
      *cul_level += 2 << COEFF_CONTEXT_BITS;
    }
  }

  fn set_coeff_context(
    &mut self, plane: usize, bo: &BlockOffset, tx_size: TxSize, xdec: usize,
    ydec: usize, value: u8
  ) {
    for bx in 0..tx_size.width_mi() {
      self.above_coeff_context[plane][(bo.x >> xdec) + bx] = value;
    }
    let bo_y = bo.y_in_sb();
    for by in 0..tx_size.height_mi() {
      self.left_coeff_context[plane][(bo_y >> ydec) + by] = value;
    }
  }

  fn reset_left_coeff_context(&mut self, plane: usize) {
    for c in &mut self.left_coeff_context[plane] {
      *c = 0;
    }
  }

  fn reset_left_partition_context(&mut self) {
    for c in &mut self.left_partition_context {
      *c = 0;
    }
  }
  //TODO(anyone): Add reset_left_tx_context() here then call it in reset_left_contexts()

  pub fn reset_skip_context(
    &mut self, bo: &BlockOffset, bsize: BlockSize, xdec: usize, ydec: usize
  ) {
    const num_planes: usize = 3;
    let nplanes = if bsize >= BLOCK_8X8 {
      3
    } else {
      1 + (num_planes - 1) * has_chroma(bo, bsize, xdec, ydec) as usize
    };

    for plane in 0..nplanes {
      let xdec2 = if plane == 0 {
        0
      } else {
        xdec
      };
      let ydec2 = if plane == 0 {
        0
      } else {
        ydec
      };

      let plane_bsize = if plane == 0 {
        bsize
      } else {
        get_plane_block_size(bsize, xdec2, ydec2)
      };
      let bw = plane_bsize.width_mi();
      let bh = plane_bsize.height_mi();

      for bx in 0..bw {
        self.above_coeff_context[plane][(bo.x >> xdec2) + bx] = 0;
      }

      let bo_y = bo.y_in_sb();
      for by in 0..bh {
        self.left_coeff_context[plane][(bo_y >> ydec2) + by] = 0;
      }
    }
  }

  pub fn reset_left_contexts(&mut self) {
    for p in 0..3 {
      BlockContext::reset_left_coeff_context(self, p);
    }
    BlockContext::reset_left_partition_context(self);

    //TODO(anyone): Call reset_left_tx_context() here.
  }

  pub fn set_mode(
    &mut self, bo: &BlockOffset, bsize: BlockSize, mode: PredictionMode
  ) {
    self.for_each(bo, bsize, |block| block.mode = mode);
  }

  pub fn set_block_size(&mut self, bo: &BlockOffset, bsize: BlockSize) {
    let n4_w = BlockSize::MI_SIZE_WIDE[bsize as usize];
    let n4_h = BlockSize::MI_SIZE_HIGH[bsize as usize];
    self.for_each(bo, bsize, |block| { block.n4_w = n4_w; block.n4_h = n4_h } );
  }

  pub fn set_tx_size(&mut self, bo: &BlockOffset, txsize: TxSize) {
    let tx_w = txsize.width_mi();
    let tx_h = txsize.height_mi();
    self.for_each(bo, txsize.block_size(), |block| { block.tx_w = tx_w; block.tx_h = tx_h } );
  }

  pub fn get_mode(&mut self, bo: &BlockOffset) -> PredictionMode {
    self.blocks[bo.y][bo.x].mode
  }

  fn partition_plane_context(
    &self, bo: &BlockOffset, bsize: BlockSize
  ) -> usize {
    // TODO: this should be way simpler without sub8x8
    let above_ctx = self.above_partition_context[bo.x];
    let left_ctx = self.left_partition_context[bo.y_in_sb()];
    let bsl = bsize.width_log2() - BLOCK_8X8.width_log2();
    let above = (above_ctx >> bsl) & 1;
    let left = (left_ctx >> bsl) & 1;

    assert!(bsize.is_sqr());

    (left * 2 + above) as usize + bsl as usize * PARTITION_PLOFFSET
  }

  pub fn update_partition_context(
    &mut self, bo: &BlockOffset, subsize: BlockSize, bsize: BlockSize
  ) {
    #[allow(dead_code)]
    let bw = bsize.width_mi();
    let bh = bsize.height_mi();

    let above_ctx =
      &mut self.above_partition_context[bo.x..bo.x + bw as usize];
    let left_ctx = &mut self.left_partition_context
      [bo.y_in_sb()..bo.y_in_sb() + bh as usize];

    // update the partition context at the end notes. set partition bits
    // of block sizes larger than the current one to be one, and partition
    // bits of smaller block sizes to be zero.
    for i in 0..bw {
      above_ctx[i as usize] = partition_context_lookup[subsize as usize][0];
    }

    for i in 0..bh {
      left_ctx[i as usize] = partition_context_lookup[subsize as usize][1];
    }
  }

  fn skip_context(&mut self, bo: &BlockOffset) -> usize {
    let above_skip = if bo.y > 0 {
      self.above_of(bo).skip as usize
    } else {
      0
    };
    let left_skip = if bo.x > 0 {
      self.left_of(bo).skip as usize
    } else {
      0
    };
    above_skip + left_skip
  }

  pub fn set_skip(&mut self, bo: &BlockOffset, bsize: BlockSize, skip: bool) {
    self.for_each(bo, bsize, |block| block.skip = skip);
  }

  pub fn set_ref_frame(&mut self, bo: &BlockOffset, bsize: BlockSize, r: usize) {
    let bw = bsize.width_mi();
    let bh = bsize.height_mi();

    for y in 0..bh {
      for x in 0..bw {
        self.blocks[bo.y + y as usize][bo.x + x as usize].ref_frames[0] = r;
      }
    }
  }

  pub fn set_motion_vector(&mut self, bo: &BlockOffset, bsize: BlockSize, mv: MotionVector) {
    let bw = bsize.width_mi();
    let bh = bsize.height_mi();

    for y in 0..bh {
      for x in 0..bw {
        self.blocks[bo.y + y as usize][bo.x + x as usize].mv[0] = mv;
      }
    }
  }

  pub fn set_cdef(&mut self, sbo: &SuperBlockOffset, cdef_index: u8) {
    let bo = sbo.block_offset(0, 0);
    // Checkme: Is 16 still the right block unit for 128x128 superblocks?
    let bw = cmp::min (bo.x + MAX_MIB_SIZE, self.blocks[bo.y as usize].len());
    let bh = cmp::min (bo.y + MAX_MIB_SIZE, self.blocks.len());
    for y in bo.y..bh {
      for x in bo.x..bw {
        self.blocks[y as usize][x as usize].cdef_index = cdef_index;
      }
    }
  }

  // The mode info data structure has a one element border above and to the
  // left of the entries corresponding to real macroblocks.
  // The prediction flags in these dummy entries are initialized to 0.
  // 0 - inter/inter, inter/--, --/inter, --/--
  // 1 - intra/inter, inter/intra
  // 2 - intra/--, --/intra
  // 3 - intra/intra
  pub fn intra_inter_context(&mut self, bo: &BlockOffset) -> usize {
    let has_above = bo.y > 0;
    let has_left = bo.x > 0;

    match (has_above, has_left) {
      (true, true) => {
        let above_intra = !self.above_of(bo).is_inter();
        let left_intra = !self.left_of(bo).is_inter();
        if above_intra && left_intra {
          3
        } else {
          (above_intra || left_intra) as usize
        }
      }
      (true, _) | (_, true) =>
        2 * if has_above {
          !self.above_of(bo).is_inter() as usize
        } else {
          !self.left_of(bo).is_inter() as usize
        },
      (_, _) => 0
    }
  }

  pub fn get_txb_ctx(
    &mut self, plane_bsize: BlockSize, tx_size: TxSize, plane: usize,
    bo: &BlockOffset, xdec: usize, ydec: usize
  ) -> TXB_CTX {
    let mut txb_ctx = TXB_CTX {
      txb_skip_ctx: 0,
      dc_sign_ctx: 0
    };
    const MAX_TX_SIZE_UNIT: usize = 16;
    const signs: [i8; 3] = [0, -1, 1];
    const dc_sign_contexts: [usize; 4 * MAX_TX_SIZE_UNIT + 1] = [
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    ];
    let mut dc_sign: i16 = 0;
    let txb_w_unit = tx_size.width_mi();
    let txb_h_unit = tx_size.height_mi();

    // Decide txb_ctx.dc_sign_ctx
    for k in 0..txb_w_unit {
      let sign = self.above_coeff_context[plane][(bo.x >> xdec) + k]
        >> COEFF_CONTEXT_BITS;
      assert!(sign <= 2);
      dc_sign += signs[sign as usize] as i16;
    }

    for k in 0..txb_h_unit {
      let sign = self.left_coeff_context[plane][(bo.y_in_sb() >> ydec) + k]
        >> COEFF_CONTEXT_BITS;
      assert!(sign <= 2);
      dc_sign += signs[sign as usize] as i16;
    }

    txb_ctx.dc_sign_ctx =
      dc_sign_contexts[(dc_sign + 2 * MAX_TX_SIZE_UNIT as i16) as usize];

    // Decide txb_ctx.txb_skip_ctx
    if plane == 0 {
      if plane_bsize == tx_size.block_size() {
        txb_ctx.txb_skip_ctx = 0;
      } else {
        // This is the algorithm to generate table skip_contexts[min][max].
        //    if (!max)
        //      txb_skip_ctx = 1;
        //    else if (!min)
        //      txb_skip_ctx = 2 + (max > 3);
        //    else if (max <= 3)
        //      txb_skip_ctx = 4;
        //    else if (min <= 3)
        //      txb_skip_ctx = 5;
        //    else
        //      txb_skip_ctx = 6;
        const skip_contexts: [[u8; 5]; 5] = [
          [1, 2, 2, 2, 3],
          [1, 4, 4, 4, 5],
          [1, 4, 4, 4, 5],
          [1, 4, 4, 4, 5],
          [1, 4, 4, 4, 6]
        ];
        let mut top: u8 = 0;
        let mut left: u8 = 0;

        for k in 0..txb_w_unit {
          top |= self.above_coeff_context[0][(bo.x >> xdec) + k];
        }
        top &= COEFF_CONTEXT_MASK as u8;

        for k in 0..txb_h_unit {
          left |= self.left_coeff_context[0][(bo.y_in_sb() >> ydec) + k];
        }
        left &= COEFF_CONTEXT_MASK as u8;

        let max = cmp::min(top | left, 4);
        let min = cmp::min(cmp::min(top, left), 4);
        txb_ctx.txb_skip_ctx =
          skip_contexts[min as usize][max as usize] as usize;
      }
    } else {
      let mut top: u8 = 0;
      let mut left: u8 = 0;

      for k in 0..txb_w_unit {
        top |= self.above_coeff_context[plane][(bo.x >> xdec) + k];
      }
      for k in 0..txb_h_unit {
        left |= self.left_coeff_context[plane][(bo.y_in_sb() >> ydec) + k];
      }
      let ctx_base = (top != 0) as usize + (left != 0) as usize;
      let ctx_offset = if num_pels_log2_lookup[plane_bsize as usize]
        > num_pels_log2_lookup[tx_size.block_size() as usize]
      {
        10
      } else {
        7
      };
      txb_ctx.txb_skip_ctx = ctx_base + ctx_offset;
    }

    txb_ctx
  }
}

#[derive(Copy, Clone, PartialEq)]
pub enum CFLSign {
  CFL_SIGN_ZERO = 0,
  CFL_SIGN_NEG = 1,
  CFL_SIGN_POS = 2
}

impl CFLSign {
  pub fn from_alpha(a: i16) -> CFLSign {
    [ CFL_SIGN_NEG, CFL_SIGN_ZERO, CFL_SIGN_POS ][(a.signum() + 1) as usize]
  }
}

use context::CFLSign::*;
const CFL_SIGNS: usize = 3;
static cfl_sign_value: [i16; CFL_SIGNS] = [ 0, -1, 1 ];

#[derive(Copy, Clone)]
pub struct CFLParams {
  sign: [CFLSign; 2],
  scale: [u8; 2]
}

impl CFLParams {
  pub fn new() -> CFLParams {
    CFLParams {
      sign: [CFL_SIGN_NEG, CFL_SIGN_ZERO],
      scale: [1, 0]
    }
  }
  pub fn joint_sign(&self) -> u32 {
    assert!(self.sign[0] != CFL_SIGN_ZERO || self.sign[1] != CFL_SIGN_ZERO);
    (self.sign[0] as u32) * (CFL_SIGNS as u32) + (self.sign[1] as u32) - 1
  }
  pub fn context(&self, uv: usize) -> usize {
    assert!(self.sign[uv] != CFL_SIGN_ZERO);
    (self.sign[uv] as usize - 1) * CFL_SIGNS + (self.sign[1 - uv] as usize)
  }
  pub fn index(&self, uv: usize) -> u32 {
    assert!(self.sign[uv] != CFL_SIGN_ZERO && self.scale[uv] != 0);
    (self.scale[uv] - 1) as u32
  }
  pub fn alpha(&self, uv: usize) -> i16 {
    cfl_sign_value[self.sign[uv] as usize] * (self.scale[uv] as i16)
  }
  pub fn from_alpha(u: i16, v: i16) -> CFLParams {
    CFLParams {
      sign: [ CFLSign::from_alpha(u), CFLSign::from_alpha(v) ],
      scale: [ u.abs() as u8, v.abs() as u8 ]
    }
  }
}

#[derive(Debug, Default)]
struct FieldMap {
  map: Vec<(&'static str, usize, usize)>
}

impl FieldMap {
  /// Print the field the address belong to
  fn lookup(&self, addr: usize) {
    for (name, start, end) in &self.map {
      // eprintln!("{} {} {} val {}", name, start, end, addr);
      if addr >= *start && addr < *end {
        eprintln!(" CDF {}", name);
        eprintln!("");
        return;
      }
    }

    eprintln!("  CDF address not found {}", addr);
  }
}

macro_rules! symbol_with_update {
  ($self:ident, $w:ident, $s:expr, $cdf:expr) => {
    $w.symbol_with_update($s, $cdf);
    #[cfg(debug)] {
      if let Some(map) = $self.fc_map.as_ref() {
        map.lookup($cdf.as_ptr() as usize);
      }
    }
  };
}

#[derive(Clone)]
pub struct ContextWriterCheckpoint {
  pub fc: CDFContext,
  pub bc: BlockContext
}

#[derive(Clone)]
pub struct ContextWriter {
  pub bc: BlockContext,
  pub fc: CDFContext,
  #[cfg(debug)]
  fc_map: Option<FieldMap> // For debugging purposes
}

impl ContextWriter {
  pub fn new(fc: CDFContext, bc: BlockContext) -> Self {
    #[allow(unused_mut)]
    let mut cw = ContextWriter {
      fc,
      bc,
      #[cfg(debug)]
      fc_map: Default::default()
    };
    #[cfg(debug)] {
      if std::env::var_os("RAV1E_DEBUG").is_some() {
        cw.fc_map = Some(FieldMap {
          map: cw.fc.build_map()
        });
      }
    }

    cw
  }

  fn cdf_element_prob(cdf: &[u16], element: usize) -> u16 {
    (if element > 0 {
      cdf[element - 1]
    } else {
      32768
    }) - cdf[element]
  }

  fn partition_gather_horz_alike(
    out: &mut [u16; 2], cdf_in: &[u16], _bsize: BlockSize
  ) {
    out[0] = 32768;
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_HORZ as usize
    );
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_SPLIT as usize
    );
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_HORZ_A as usize
    );
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_HORZ_B as usize
    );
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_VERT_A as usize
    );
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_HORZ_4 as usize
    );
    out[0] = 32768 - out[0];
    out[1] = 0;
  }

  fn partition_gather_vert_alike(
    out: &mut [u16; 2], cdf_in: &[u16], _bsize: BlockSize
  ) {
    out[0] = 32768;
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_VERT as usize
    );
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_SPLIT as usize
    );
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_HORZ_A as usize
    );
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_VERT_A as usize
    );
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_VERT_B as usize
    );
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_VERT_4 as usize
    );
    out[0] = 32768 - out[0];
    out[1] = 0;
  }

  pub fn write_partition(
    &mut self, w: &mut dyn Writer, bo: &BlockOffset, p: PartitionType, bsize: BlockSize
  ) {
    assert!(bsize >= BlockSize::BLOCK_8X8 );
    let hbs = bsize.width_mi() / 2;
    let has_cols = (bo.x + hbs) < self.bc.cols;
    let has_rows = (bo.y + hbs) < self.bc.rows;
    let ctx = self.bc.partition_plane_context(&bo, bsize);
    assert!(ctx < PARTITION_CONTEXTS);
    let partition_cdf = if bsize <= BlockSize::BLOCK_8X8 {
      &mut self.fc.partition_cdf[ctx][..PARTITION_TYPES+1]
    } else {
      &mut self.fc.partition_cdf[ctx]
    };

    if !has_rows && !has_cols {
      return;
    }

    if has_rows && has_cols {
      symbol_with_update!(self, w, p as u32, partition_cdf);
    } else if !has_rows && has_cols {
      assert!(bsize > BlockSize::BLOCK_8X8);
      let mut cdf = [0u16; 2];
      ContextWriter::partition_gather_vert_alike(
        &mut cdf,
        partition_cdf,
        bsize
      );
      w.symbol((p == PartitionType::PARTITION_SPLIT) as u32, &cdf);
    } else {
      assert!(bsize > BlockSize::BLOCK_8X8);
      let mut cdf = [0u16; 2];
      ContextWriter::partition_gather_horz_alike(
        &mut cdf,
        partition_cdf,
        bsize
      );
      w.symbol((p == PartitionType::PARTITION_SPLIT) as u32, &cdf);
    }
  }
  pub fn write_intra_mode_kf(
    &mut self, w: &mut dyn Writer, bo: &BlockOffset, mode: PredictionMode
  ) {
    static intra_mode_context: [usize; INTRA_MODES] =
      [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];
    let above_mode = self.bc.above_of(bo).mode as usize;
    let left_mode = self.bc.left_of(bo).mode as usize;
    let above_ctx = intra_mode_context[above_mode];
    let left_ctx = intra_mode_context[left_mode];
    let cdf = &mut self.fc.kf_y_cdf[above_ctx][left_ctx];
    symbol_with_update!(self, w, mode as u32, cdf);
  }
  pub fn write_intra_mode(&mut self, w: &mut dyn Writer, bsize: BlockSize, mode: PredictionMode) {
    let cdf =
      &mut self.fc.y_mode_cdf[size_group_lookup[bsize as usize] as usize];
    symbol_with_update!(self, w, mode as u32, cdf);
  }
  pub fn write_intra_uv_mode(
    &mut self, w: &mut dyn Writer, uv_mode: PredictionMode, y_mode: PredictionMode, bs: BlockSize
  ) {
    let cdf =
      &mut self.fc.uv_mode_cdf[bs.cfl_allowed() as usize][y_mode as usize];
    if bs.cfl_allowed() {
      symbol_with_update!(self, w, uv_mode as u32, cdf);
    } else {
      symbol_with_update!(self, w, uv_mode as u32, &mut cdf[..UV_INTRA_MODES]);
    }
  }
  pub fn write_cfl_alphas(&mut self, w: &mut dyn Writer, cfl: CFLParams) {
    symbol_with_update!(self, w, cfl.joint_sign(), &mut self.fc.cfl_sign_cdf);
    for uv in 0..2 {
      if cfl.sign[uv] != CFL_SIGN_ZERO {
        symbol_with_update!(self, w, cfl.index(uv), &mut self.fc.cfl_alpha_cdf[cfl.context(uv)]);
      }
    }
  }
  pub fn write_angle_delta(&mut self, w: &mut dyn Writer, angle: i8, mode: PredictionMode) {
    symbol_with_update!(
      self,
      w,
      (angle + MAX_ANGLE_DELTA as i8) as u32,
      &mut self.fc.angle_delta_cdf
        [mode as usize - PredictionMode::V_PRED as usize]
    );
  }
  pub fn write_use_filter_intra(&mut self, w: &mut dyn Writer, enable: bool, block_size: BlockSize) {
    symbol_with_update!(self, w, enable as u32, &mut self.fc.filter_intra_cdfs[block_size as usize]);
  }

  fn get_mvref_ref_frames(&mut self, ref_frame: usize) -> ([usize; 2], usize) {
    let ref_frame_map: [[usize; 2]; TOTAL_COMP_REFS] = [
      [ LAST_FRAME,  BWDREF_FRAME  ], [ LAST2_FRAME,  BWDREF_FRAME  ],
      [ LAST3_FRAME, BWDREF_FRAME  ], [ GOLDEN_FRAME, BWDREF_FRAME  ],
      [ LAST_FRAME,  ALTREF2_FRAME ], [ LAST2_FRAME,  ALTREF2_FRAME ],
      [ LAST3_FRAME, ALTREF2_FRAME ], [ GOLDEN_FRAME, ALTREF2_FRAME ],
      [ LAST_FRAME,  ALTREF_FRAME  ], [ LAST2_FRAME,  ALTREF_FRAME  ],
      [ LAST3_FRAME, ALTREF_FRAME  ], [ GOLDEN_FRAME, ALTREF_FRAME  ],
      [ LAST_FRAME,  LAST2_FRAME   ], [ LAST_FRAME,   LAST3_FRAME   ],
      [ LAST_FRAME,  GOLDEN_FRAME  ], [ BWDREF_FRAME, ALTREF_FRAME  ],

      // NOTE: Following reference frame pairs are not supported to be explicitly
      //       signalled, but they are possibly chosen by the use of skip_mode,
      //       which may use the most recent one-sided reference frame pair.
      [ LAST2_FRAME, LAST3_FRAME   ], [  LAST2_FRAME, GOLDEN_FRAME  ],
      [ LAST3_FRAME, GOLDEN_FRAME  ], [ BWDREF_FRAME, ALTREF2_FRAME ],
      [ ALTREF2_FRAME, ALTREF_FRAME ]
    ];

    if ref_frame >= REF_FRAMES {
      return ([ ref_frame_map[ref_frame - REF_FRAMES][0],
               ref_frame_map[ref_frame - REF_FRAMES][1] ], 2)
    } else {
      return ([ ref_frame, 0 ], 1)
    }
  }

  fn find_valid_row_offs(&mut self, row_offset: isize, mi_row: usize, mi_rows: usize) -> isize {
    if /* !tile->tg_horz_boundary */ true {
      cmp::min(cmp::max(row_offset, -(mi_row as isize)), (mi_rows - mi_row - 1) as isize)
    } else {
      0
      /* TODO: for tiling */
    }
  }

  fn has_tr(&mut self, bo: &BlockOffset, bsize: BlockSize, is_sec_rect: bool) -> bool {
    let sb_mi_size = BlockSize::MI_SIZE_WIDE[BLOCK_64X64 as usize]; /* Assume 64x64 for now */
    let mask_row = bo.y & LOCAL_BLOCK_MASK;
    let mask_col = bo.x & LOCAL_BLOCK_MASK;
    let target_n4_w = bsize.width_mi();
    let target_n4_h = bsize.height_mi();

    let mut bs = target_n4_w.max(target_n4_h);

    if bs > BlockSize::MI_SIZE_WIDE[BLOCK_64X64 as usize] {
      return false;
    }

    let mut has_tr = !((mask_row & bs) != 0 && (mask_col & bs) != 0);

    /* TODO: assert its a power of two */

    while bs < sb_mi_size {
      if (mask_col & bs) != 0 {
        if (mask_col & (2 * bs) != 0) && (mask_row & (2 * bs) != 0) {
          has_tr = false;
          break;
        }
      } else {
        break;
      }
      bs <<= 1;
    }

    /* The left hand of two vertical rectangles always has a top right (as the
     * block above will have been decoded) */
    if (target_n4_w < target_n4_h) && !is_sec_rect {
      has_tr = true;
    }

    /* The bottom of two horizontal rectangles never has a top right (as the block
     * to the right won't have been decoded) */
    if (target_n4_w > target_n4_h) && is_sec_rect {
      has_tr = false;
    }

    /* The bottom left square of a Vertical A (in the old format) does
     * not have a top right as it is decoded before the right hand
     * rectangle of the partition */
/*
    if blk.partition == PartitionType::PARTITION_VERT_A {
      if blk.n4_w == blk.n4_h {
        if (mask_row & bs) != 0 {
          has_tr = false;
        }
      }
    }
*/

    has_tr
  }

  fn find_valid_col_offs(&mut self, col_offset: isize, mi_col: usize) -> isize {
    cmp::max(col_offset, -(mi_col as isize))
  }

  fn find_matching_mv(&self, blk: &Block, mv_stack: &mut Vec<CandidateMV>, weight: u32) -> bool {
    for mut mv_cand in mv_stack {
      if blk.mv[0].row == mv_cand.this_mv.row && blk.mv[0].col == mv_cand.this_mv.col {
        mv_cand.weight += weight;
        return true;
      }
    }
    false
  }

  fn add_ref_mv_candidate(&self, ref_frame: usize, blk: &Block, mv_stack: &mut Vec<CandidateMV>,
                          weight: u32, newmv_count: &mut usize) -> bool {
    if !blk.is_inter() { /* For intrabc */
      return false;
    }

    if blk.ref_frames[0] == ref_frame {
      let found_match = self.find_matching_mv(blk, mv_stack, weight);

      if !found_match && mv_stack.len() < MAX_REF_MV_STACK_SIZE {
        let mv_cand = CandidateMV {
          this_mv: blk.mv[0],
          comp_mv: blk.mv[1],
          weight: weight
        };

        mv_stack.push(mv_cand);
      }

      if blk.mode == PredictionMode::NEWMV {
        *newmv_count += 1;
      }

      true
    } else {
      false
    }
  }

  fn scan_row_mbmi(&mut self, bo: &BlockOffset, row_offset: isize, max_row_offs: isize,
                   processed_rows: &mut isize, ref_frame: usize,
                   mv_stack: &mut Vec<CandidateMV>, newmv_count: &mut usize, bsize: BlockSize) -> bool {
    let bc = &self.bc;
    let target_n4_w = bsize.width_mi();

    let end_mi = cmp::min(cmp::min(target_n4_w, bc.cols - bo.x),
                          BlockSize::MI_SIZE_WIDE[BLOCK_64X64 as usize]);
    let n4_w_8 = BlockSize::MI_SIZE_WIDE[BLOCK_8X8 as usize];
    let n4_w_16 = BlockSize::MI_SIZE_WIDE[BLOCK_16X16 as usize];
    let mut col_offset = 0;

    if row_offset.abs() > 1 {
      col_offset = 1;
      if ((bo.x & 0x01) != 0) && (target_n4_w < n4_w_8) {
        col_offset -= 1;
      }
    }

    let use_step_16 = target_n4_w >= 16;

    let mut found_match = false;

    let mut i = 0;
    while i < end_mi {
      let cand = bc.at(&bo.with_offset(col_offset + i as isize, row_offset));

      let n4_w = cand.n4_w;
      let mut len = cmp::min(target_n4_w, n4_w);
      if use_step_16 {
        len = cmp::max(n4_w_16, len);
      } else if row_offset.abs() > 1 {
        len = cmp::max(len, n4_w_8);
      }

      let mut weight = 2 as u32;
      if target_n4_w >= n4_w_8 && target_n4_w <= n4_w {
        let inc = cmp::min(-max_row_offs + row_offset + 1, cand.n4_h as isize);
        assert!(inc >= 0);
        weight = cmp::max(weight, inc as u32);
        *processed_rows = (inc as isize) - row_offset - 1;
      }

      if self.add_ref_mv_candidate(ref_frame, cand, mv_stack, len as u32 * weight, newmv_count) {
        found_match = true;
      }

      i += len;
    }

    found_match
  }

  fn scan_col_mbmi(&mut self, bo: &BlockOffset, col_offset: isize, max_col_offs: isize,
                   processed_cols: &mut isize, ref_frame: usize,
                   mv_stack: &mut Vec<CandidateMV>, newmv_count: &mut usize, bsize: BlockSize) -> bool {
    let bc = &self.bc;

    let target_n4_h = bsize.height_mi();

    let end_mi = cmp::min(cmp::min(target_n4_h, bc.rows - bo.y),
                          BlockSize::MI_SIZE_HIGH[BLOCK_64X64 as usize]);
    let n4_h_8 = BlockSize::MI_SIZE_HIGH[BLOCK_8X8 as usize];
    let n4_h_16 = BlockSize::MI_SIZE_HIGH[BLOCK_16X16 as usize];
    let mut row_offset = 0;

    if col_offset.abs() > 1 {
      row_offset = 1;
      if ((bo.y & 0x01) != 0) && (target_n4_h < n4_h_8) {
        row_offset -= 1;
      }
    }

    let use_step_16 = target_n4_h >= 16;

    let mut found_match = false;

    let mut i = 0;
    while i < end_mi {
      let cand = bc.at(&bo.with_offset(col_offset, row_offset + i as isize));
      let n4_h = cand.n4_h;
      let mut len = cmp::min(target_n4_h, n4_h);
      if use_step_16 {
        len = cmp::max(n4_h_16, len);
      } else if col_offset.abs() > 1 {
        len = cmp::max(len, n4_h_8);
      }

      let mut weight = 2 as u32;
      if target_n4_h >= n4_h_8 && target_n4_h <= n4_h {
        let inc = cmp::min(-max_col_offs + col_offset + 1, cand.n4_w as isize);
        assert!(inc >= 0);
        weight = cmp::max(weight, inc as u32);
        *processed_cols = (inc as isize) - col_offset - 1;
      }

      if self.add_ref_mv_candidate(ref_frame, cand, mv_stack, len as u32 * weight, newmv_count) {
        found_match = true;
      }

      i += len;
    }

    found_match
  }

  fn scan_blk_mbmi(&mut self, bo: &BlockOffset, ref_frame: usize,
                   mv_stack: &mut Vec<CandidateMV>, newmv_count: &mut usize) -> bool {
    if bo.x >= self.bc.cols || bo.y >= self.bc.rows {
      return false;
    }

    let weight = 2 * BLOCK_8X8.width_mi() as u32;
    /* Always assume its within a tile, probably wrong */
    self.add_ref_mv_candidate(ref_frame, self.bc.at(bo), mv_stack, weight, newmv_count)
  }

  fn add_offset(&mut self, mv_stack: &mut Vec<CandidateMV>) {
    for mut cand_mv in mv_stack {
      cand_mv.weight += REF_CAT_LEVEL;
    }
  }

  fn setup_mvref_list(&mut self, bo: &BlockOffset, ref_frame: usize, mv_stack: &mut Vec<CandidateMV>,
                      bsize: BlockSize, is_sec_rect: bool) -> usize {
    let (_rf, _rf_num) = self.get_mvref_ref_frames(INTRA_FRAME);

    let target_n4_h = bsize.height_mi();
    let target_n4_w = bsize.width_mi();

    let mut max_row_offs = 0 as isize;
    let row_adj = (target_n4_h < BlockSize::MI_SIZE_HIGH[BLOCK_8X8 as usize]) && (bo.y & 0x01) != 0x0;

    let mut max_col_offs = 0 as isize;
    let col_adj = (target_n4_w < BlockSize::MI_SIZE_WIDE[BLOCK_8X8 as usize]) && (bo.x & 0x01) != 0x0;

    let mut processed_rows = 0 as isize;
    let mut processed_cols = 0 as isize;

    let up_avail = bo.y > 0;
    let left_avail = bo.x > 0;

    if up_avail {
      max_row_offs = -2 * MVREF_ROW_COLS as isize + row_adj as isize;

      // limit max offset for small blocks
      if target_n4_h < BlockSize::MI_SIZE_HIGH[BLOCK_8X8 as usize] {
        max_row_offs = -2 * 2 + row_adj as isize;
      }

      let rows = self.bc.rows;
      max_row_offs = self.find_valid_row_offs(max_row_offs, bo.y, rows);
    }

    if left_avail {
      max_col_offs = -2 * MVREF_ROW_COLS as isize + col_adj as isize;

      // limit max offset for small blocks
      if target_n4_w < BlockSize::MI_SIZE_WIDE[BLOCK_8X8 as usize] {
        max_col_offs = -2 * 2 + col_adj as isize;
      }

      max_col_offs = self.find_valid_col_offs(max_col_offs, bo.x);
    }

    let mut row_match = false;
    let mut col_match = false;
    let mut newmv_count: usize = 0;

    if max_row_offs.abs() >= 1 {
      let found_match = self.scan_row_mbmi(bo, -1, max_row_offs, &mut processed_rows, ref_frame, mv_stack,
                                           &mut newmv_count, bsize);
      row_match |= found_match;
    }
    if max_col_offs.abs() >= 1 {
      let found_match = self.scan_col_mbmi(bo, -1, max_col_offs, &mut processed_cols, ref_frame, mv_stack,
                                           &mut newmv_count, bsize);
      col_match |= found_match;
    }
    if self.has_tr(bo, bsize, is_sec_rect) {
      let found_match = self.scan_blk_mbmi(&bo.with_offset(target_n4_w as isize, -1), ref_frame, mv_stack,
                                           &mut newmv_count);
      row_match |= found_match;
    }

    let nearest_match = if row_match { 1 } else { 0 } + if col_match { 1 } else { 0 };

    self.add_offset(mv_stack);

    /* Scan the second outer area. */
    let mut far_newmv_count: usize = 0; // won't be used

    let found_match = self.scan_blk_mbmi(&bo.with_offset(-1, -1), ref_frame, mv_stack, &mut far_newmv_count);
    row_match |= found_match;

    for idx in 2..MVREF_ROW_COLS+1 {
      let row_offset = -2 * idx as isize + 1 + row_adj as isize;
      let col_offset = -2 * idx as isize + 1 + col_adj as isize;

      if row_offset.abs() <= max_row_offs.abs() && row_offset.abs() > processed_rows {
        let found_match = self.scan_row_mbmi(bo, row_offset, max_row_offs, &mut processed_rows, ref_frame, mv_stack,
                                             &mut far_newmv_count, bsize);
        row_match |= found_match;
      }

      if col_offset.abs() <= max_col_offs.abs() && col_offset.abs() > processed_cols {
        let found_match = self.scan_col_mbmi(bo, col_offset, max_col_offs, &mut processed_cols, ref_frame, mv_stack,
                                             &mut far_newmv_count, bsize);
        col_match |= found_match;
      }
    }

    let total_match = if row_match { 1 } else { 0 } + if col_match { 1 } else { 0 };

    assert!(total_match >= nearest_match);

    let mode_context = match nearest_match {
                         0 =>  cmp::min(total_match, 1) + (total_match << REFMV_OFFSET) ,
                         1 =>  3 - cmp::min(newmv_count, 1) + ((2 + total_match) << REFMV_OFFSET) ,
                         _ =>  5 - cmp::min(newmv_count, 1) + (5 << REFMV_OFFSET)
                       };

    // println!("{} {} {} {} {}", bo.x, bo.y, nearest_match, total_match, mode_context);

    /* TODO: Find nearest match and assign nearest and near mvs */

    // Sort MV stack according to weight
    mv_stack.sort_by(|a, b| b.weight.cmp(&a.weight));

    /* TODO: Handle single reference frame extension */

    // clamp mvs
    for mv in mv_stack {
      let blk_w = bsize.width();
      let blk_h = bsize.height();
      let border_w = 128 + blk_w as isize * 8;
      let border_h = 128 + blk_h as isize * 8;
      let x_min = -(bo.x as isize) * (8 * MI_SIZE) as isize - border_w;
      let x_max = (self.bc.cols - bo.x - blk_w / MI_SIZE) as isize * (8 * MI_SIZE) as isize + border_w;
      let y_min = -(bo.y as isize) * (8 * MI_SIZE) as isize - border_h;
      let y_max = (self.bc.rows - bo.y - blk_h / MI_SIZE) as isize * (8 * MI_SIZE) as isize + border_h;
      mv.this_mv.row = (mv.this_mv.row as isize).max(y_min).min(y_max) as i16;
      mv.this_mv.col = (mv.this_mv.col as isize).max(x_min).min(x_max) as i16;
    }

    mode_context
  }

  pub fn find_mvrefs(&mut self, bo: &BlockOffset, ref_frame: usize,
                     mv_stack: &mut Vec<CandidateMV>, bsize: BlockSize, is_sec_rect: bool) -> usize {
    if ref_frame < REF_FRAMES {
      if ref_frame != INTRA_FRAME {
        /* TODO: convert global mv to an mv here */
      } else {
        /* TODO: set the global mv ref to invalid here */
      }
    }

    if ref_frame != INTRA_FRAME {
      /* TODO: Set zeromv ref to the converted global motion vector */
    } else {
      /* TODO: Set the zeromv ref to 0 */
    }

    let mode_context = self.setup_mvref_list(bo, ref_frame, mv_stack, bsize, is_sec_rect);
    mode_context
  }

  pub fn fill_neighbours_ref_counts(&mut self, bo: &BlockOffset) {
      let mut ref_counts = [0; TOTAL_REFS_PER_FRAME];

      let above_b = self.bc.above_of(bo);
      let left_b = self.bc.left_of(bo);

      if bo.y > 0 && above_b.is_inter() {
        ref_counts[above_b.ref_frames[0] as usize] += 1;
        if above_b.has_second_ref() {
          ref_counts[above_b.ref_frames[1] as usize] += 1;
        }
      }

      if bo.x > 0 && left_b.is_inter() {
        ref_counts[left_b.ref_frames[0] as usize] += 1;
        if left_b.has_second_ref() {
          ref_counts[left_b.ref_frames[1] as usize] += 1;
        }
      }
      self.bc.at_mut(bo).neighbors_ref_counts = ref_counts;
  }

  fn get_ref_frame_ctx_b0(&mut self, bo: &BlockOffset) -> usize {
    let ref_counts = self.bc.at(bo).neighbors_ref_counts;

    let fwd_cnt = ref_counts[LAST_FRAME] + ref_counts[LAST2_FRAME] +
                  ref_counts[LAST3_FRAME] + ref_counts[GOLDEN_FRAME];

    let bwd_cnt = ref_counts[BWDREF_FRAME] + ref_counts[ALTREF2_FRAME] +
                  ref_counts[ALTREF_FRAME];

    if fwd_cnt == bwd_cnt {
      return 1;
    } else if fwd_cnt < bwd_cnt {
      return 0;
    } else {
      return 2;
    }
  }

  fn get_pred_ctx_brfarf2_or_arf(&mut self, bo: &BlockOffset) -> usize {
    let ref_counts = self.bc.at(bo).neighbors_ref_counts;

    let brfarf2_count = ref_counts[BWDREF_FRAME] +
                        ref_counts[ALTREF2_FRAME];

    let arf_count = ref_counts[ALTREF_FRAME];

    if brfarf2_count == arf_count {
      return 1;
    } else if brfarf2_count < arf_count {
      return 0;
    } else {
      return 2;
    }
  }

  fn get_pred_ctx_ll2_or_l3gld(&mut self, bo: &BlockOffset) -> usize {
    let ref_counts = self.bc.at(bo).neighbors_ref_counts;

    let l_l2_count = ref_counts[LAST_FRAME] +
                        ref_counts[LAST2_FRAME];

    let l3_gold_count = ref_counts[LAST3_FRAME] +
                        ref_counts[GOLDEN_FRAME];

    if l_l2_count == l3_gold_count {
      return 1;
    } else if l_l2_count < l3_gold_count {
      return 0;
    } else {
      return 2;
    }
  }

  fn get_pred_ctx_last_or_last2(&mut self, bo: &BlockOffset) -> usize {
    let ref_counts = self.bc.at(bo).neighbors_ref_counts;

    let l_count = ref_counts[LAST_FRAME];

    let l2_count = ref_counts[LAST2_FRAME];

    if l_count == l2_count {
      return 1;
    } else if l_count < l2_count {
      return 0;
    } else {
      return 2;
    }
  }

  fn get_pred_ctx_last3_or_gold(&mut self, bo: &BlockOffset) -> usize {
    let ref_counts = self.bc.at(bo).neighbors_ref_counts;

    let l3_count = ref_counts[LAST3_FRAME];

    let gold_count = ref_counts[GOLDEN_FRAME];

    if l3_count == gold_count {
      return 1;
    } else if l3_count < gold_count {
      return 0;
    } else {
      return 2;
    }
  }

  fn get_pred_ctx_brf_or_arf2(&mut self, bo: &BlockOffset) -> usize {
    let ref_counts = self.bc.at(bo).neighbors_ref_counts;

    let brf_count = ref_counts[BWDREF_FRAME];

    let arf2_count = ref_counts[ALTREF2_FRAME];

    if brf_count == arf2_count {
      return 1;
    } else if brf_count < arf2_count {
      return 0;
    } else {
      return 2;
    }
  }

  pub fn write_ref_frames(&mut self, w: &mut dyn Writer, bo: &BlockOffset) {
    let rf = self.bc.at(bo).ref_frames;
    assert!(rf[0] == LAST_FRAME);

    /* TODO: Handle multiple references */

    let b0_ctx = self.get_ref_frame_ctx_b0(bo);
    let b0 = rf[0] <= ALTREF_FRAME && rf[0] >= BWDREF_FRAME;

    symbol_with_update!(self, w, b0 as u32, &mut self.fc.single_ref_cdfs[b0_ctx][0]);
    if b0 {
      let b1_ctx = self.get_pred_ctx_brfarf2_or_arf(bo);
      let b1 = rf[0] == ALTREF_FRAME;

      symbol_with_update!(self, w, b1 as u32, &mut self.fc.single_ref_cdfs[b1_ctx][1]);
      if !b1 {
        let b5_ctx = self.get_pred_ctx_brf_or_arf2(bo);
        let b5 = rf[0] == ALTREF2_FRAME;

        symbol_with_update!(self, w, b5 as u32, &mut self.fc.single_ref_cdfs[b5_ctx][5]);
      }
    } else {
      let b2_ctx = self.get_pred_ctx_ll2_or_l3gld(bo);
      let b2 = rf[0] == LAST3_FRAME || rf[0] == GOLDEN_FRAME;

      symbol_with_update!(self, w, b2 as u32, &mut self.fc.single_ref_cdfs[b2_ctx][2]);
      if !b2 {
        let b3_ctx = self.get_pred_ctx_last_or_last2(bo);
        let b3 = rf[0] != LAST_FRAME;

        symbol_with_update!(self, w, b3 as u32, &mut self.fc.single_ref_cdfs[b3_ctx][3]);
      } else {
        let b4_ctx = self.get_pred_ctx_last3_or_gold(bo);
        let b4 = rf[0] != LAST3_FRAME;

        symbol_with_update!(self, w, b4 as u32, &mut self.fc.single_ref_cdfs[b4_ctx][4]);
      }
    }
  }

  pub fn write_inter_mode(&mut self, w: &mut dyn Writer, mode: PredictionMode, ctx: usize) {
    let newmv_ctx = ctx & NEWMV_CTX_MASK;
    symbol_with_update!(self, w, (mode != PredictionMode::NEWMV) as u32, &mut self.fc.newmv_cdf[newmv_ctx]);
    if mode != PredictionMode::NEWMV {
      let zeromv_ctx = (ctx >> GLOBALMV_OFFSET) & GLOBALMV_CTX_MASK;
      symbol_with_update!(self, w, (mode != PredictionMode::GLOBALMV) as u32, &mut self.fc.zeromv_cdf[zeromv_ctx]);
      if mode != PredictionMode::GLOBALMV {
        let refmv_ctx = (ctx >> REFMV_OFFSET) & REFMV_CTX_MASK;
        symbol_with_update!(self, w, (mode != PredictionMode::NEARESTMV) as u32, &mut self.fc.refmv_cdf[refmv_ctx]);
      }
    }
  }

  pub fn write_drl_mode(&mut self, w: &mut dyn Writer, drl_mode: bool, ctx: usize) {
    symbol_with_update!(self, w, drl_mode as u32, &mut self.fc.drl_cdfs[ctx]);
  }

  pub fn write_mv(&mut self, w: &mut dyn Writer,
                  mv: &MotionVector, ref_mv: &MotionVector,
                  mv_precision: MvSubpelPrecision) {
    let diff = MotionVector { row: mv.row - ref_mv.row, col: mv.col - ref_mv.col };
    let j: MvJointType = av1_get_mv_joint(&diff);

    w.symbol_with_update(j as u32, &mut self.fc.nmv_context.joints_cdf);

    if mv_joint_vertical(j) {
      encode_mv_component(w, diff.row as i32, &mut self.fc.nmv_context.comps[0], mv_precision);
    }
    if mv_joint_horizontal(j) {
      encode_mv_component(w, diff.col as i32, &mut self.fc.nmv_context.comps[1], mv_precision);
    }
  }

  pub fn write_tx_type(
    &mut self, w: &mut dyn Writer, tx_size: TxSize, tx_type: TxType, y_mode: PredictionMode,
    is_inter: bool, use_reduced_tx_set: bool
  ) {
    let square_tx_size = tx_size.sqr();
    let tx_set =
      get_tx_set(tx_size, is_inter, use_reduced_tx_set);
    let num_tx_types = num_tx_set[tx_set as usize];

    if num_tx_types > 1 {
      let tx_set_index = get_tx_set_index(tx_size, is_inter, use_reduced_tx_set);
      assert!(tx_set_index > 0);
      assert!(av1_tx_used[tx_set as usize][tx_type as usize] != 0);

      if is_inter {
        symbol_with_update!(
          self,
          w,
          av1_tx_ind[tx_set as usize][tx_type as usize] as u32,
          &mut self.fc.inter_tx_cdf[tx_set_index as usize]
            [square_tx_size as usize]
            [..num_tx_set[tx_set as usize] + 1]
        );
      } else {
        let intra_dir = y_mode;
        // TODO: Once use_filter_intra is enabled,
        // intra_dir =
        // fimode_to_intradir[mbmi->filter_intra_mode_info.filter_intra_mode];

        symbol_with_update!(
          self,
          w,
          av1_tx_ind[tx_set as usize][tx_type as usize] as u32,
          &mut self.fc.intra_tx_cdf[tx_set_index as usize]
            [square_tx_size as usize][intra_dir as usize]
            [..num_tx_set[tx_set as usize] + 1]
        );
      }
    }
  }
  pub fn write_skip(&mut self, w: &mut dyn Writer, bo: &BlockOffset, skip: bool) {
    let ctx = self.bc.skip_context(bo);
    symbol_with_update!(self, w, skip as u32, &mut self.fc.skip_cdfs[ctx]);
  }

  pub fn write_cdef(&mut self, w: &mut dyn Writer, strength_index: u8, bits: u8) {
    w.literal(bits, strength_index as u32);
  }

  pub fn write_block_deblock_deltas(&mut self, w: &mut dyn Writer,
                                    bo: &BlockOffset, multi: bool) {
      let block = self.bc.at(bo);
      let deltas = if multi { FRAME_LF_COUNT + PLANES - 3 } else { 1 };
      for i in 0..deltas {
          let delta = block.deblock_deltas[i];
          let abs:u32 = delta.abs() as u32;

          if multi {
              symbol_with_update!(self, w, cmp::min(abs, DELTA_LF_SMALL),
                                  &mut self.fc.deblock_delta_multi_cdf[i]);
          } else {
              symbol_with_update!(self, w, cmp::min(abs, DELTA_LF_SMALL),
                                  &mut self.fc.deblock_delta_cdf);
          };
          if abs >= DELTA_LF_SMALL {
              let bits = msb(abs as i32 - 1) as u32;
              w.literal(3, bits - 1);
              w.literal(bits as u8, abs - (1<<bits) - 1);
          }
          if abs > 0 {
              w.bool(delta < 0, 16384);
          }
      }
  }

  pub fn write_is_inter(&mut self, w: &mut dyn Writer, bo: &BlockOffset, is_inter: bool) {
    let ctx = self.bc.intra_inter_context(bo);
    symbol_with_update!(self, w, is_inter as u32, &mut self.fc.intra_inter_cdfs[ctx]);
  }

  pub fn get_txsize_entropy_ctx(&mut self, tx_size: TxSize) -> usize {
    (tx_size.sqr() as usize + tx_size.sqr() as usize + 1) >> 1
  }

  pub fn txb_init_levels(
    &mut self, coeffs: &[i32], width: usize, height: usize,
    levels_buf: &mut [u8]
  ) {
    let mut offset = TX_PAD_TOP * (width + TX_PAD_HOR);

    for y in 0..height {
      for x in 0..width {
        levels_buf[offset] = clamp(coeffs[y * width + x].abs(), 0, 127) as u8;
        offset += 1;
      }
      offset += TX_PAD_HOR;
    }
  }

  pub fn av1_get_adjusted_tx_size(&mut self, tx_size: TxSize) -> TxSize {
    // TODO: Enable below commented out block if TX64X64 is enabled.
/*
      if tx_size == TX_64X64 || tx_size == TX_64X32 || tx_size == TX_32X64 {
        return TX_32X32
      }
      if (tx_size == TX_16X64) {
        return TX_16X32
      }
      if (tx_size == TX_64X16) {
        return TX_32X16
      }
*/
    tx_size
  }

  pub fn get_txb_bwl(&mut self, tx_size: TxSize) -> usize {
    self.av1_get_adjusted_tx_size(tx_size).width_log2()
  }

  pub fn get_eob_pos_token(&mut self, eob: usize, extra: &mut u32) -> u32 {
    let t = if eob < 33 {
      eob_to_pos_small[eob] as u32
    } else {
      let e = cmp::min((eob - 1) >> 5, 16);
      eob_to_pos_large[e as usize] as u32
    };
    assert!(eob as i32 >= k_eob_group_start[t as usize] as i32);
    *extra = eob as u32 - k_eob_group_start[t as usize] as u32;

    t
  }

  pub fn get_nz_mag(
    &mut self, levels: &[u8], bwl: usize, tx_class: TxClass
  ) -> usize {
    // May version.
    // Note: AOMMIN(level, 3) is useless for decoder since level < 3.
    let mut mag = clip_max3(levels[1]); // { 0, 1 }
    mag += clip_max3(levels[(1 << bwl) + TX_PAD_HOR]); // { 1, 0 }

    if tx_class == TX_CLASS_2D {
      mag += clip_max3(levels[(1 << bwl) + TX_PAD_HOR + 1]); // { 1, 1 }
      mag += clip_max3(levels[2]); // { 0, 2 }
      mag += clip_max3(levels[(2 << bwl) + (2 << TX_PAD_HOR_LOG2)]); // { 2, 0 }
    } else if tx_class == TX_CLASS_VERT {
      mag += clip_max3(levels[(2 << bwl) + (2 << TX_PAD_HOR_LOG2)]); // { 2, 0 }
      mag += clip_max3(levels[(3 << bwl) + (3 << TX_PAD_HOR_LOG2)]); // { 3, 0 }
      mag += clip_max3(levels[(4 << bwl) + (4 << TX_PAD_HOR_LOG2)]); // { 4, 0 }
    } else {
      mag += clip_max3(levels[2]); // { 0, 2 }
      mag += clip_max3(levels[3]); // { 0, 3 }
      mag += clip_max3(levels[4]); // { 0, 4 }
    }

    mag as usize
  }

  pub fn get_nz_map_ctx_from_stats(
    &mut self,
    stats: usize,
    coeff_idx: usize, // raster order
    bwl: usize,
    tx_size: TxSize,
    tx_class: TxClass
  ) -> usize {
    if (tx_class as u32 | coeff_idx as u32) == 0 {
      return 0;
    };
    let row = coeff_idx >> bwl;
    let col = coeff_idx - (row << bwl);
    let mut ctx = (stats + 1) >> 1;
    ctx = cmp::min(ctx, 4);

    match tx_class {
      TX_CLASS_2D => {
        // This is the algorithm to generate table av1_nz_map_ctx_offset[].
        // const int width = tx_size_wide[tx_size];
        // const int height = tx_size_high[tx_size];
        // if (width < height) {
        //   if (row < 2) return 11 + ctx;
        // } else if (width > height) {
        //   if (col < 2) return 16 + ctx;
        // }
        // if (row + col < 2) return ctx + 1;
        // if (row + col < 4) return 5 + ctx + 1;
        // return 21 + ctx;
        return ctx + av1_nz_map_ctx_offset[tx_size as usize][cmp::min(row, 4)][cmp::min(col, 4)] as usize;
      }
      TX_CLASS_HORIZ => {
        let row = coeff_idx >> bwl;
        let col = coeff_idx - (row << bwl);
        ctx + nz_map_ctx_offset_1d[col as usize]
      }
      TX_CLASS_VERT => {
        let row = coeff_idx >> bwl;
        ctx + nz_map_ctx_offset_1d[row]
      }
    }
  }

  pub fn get_nz_map_ctx(
    &mut self, levels: &[u8], coeff_idx: usize, bwl: usize, height: usize,
    scan_idx: usize, is_eob: bool, tx_size: TxSize, tx_class: TxClass
  ) -> usize {
    if is_eob {
      if scan_idx == 0 {
        return 0;
      }
      if scan_idx <= (height << bwl) / 8 {
        return 1;
      }
      if scan_idx <= (height << bwl) / 4 {
        return 2;
      }
      return 3;
    }
    let padded_idx = coeff_idx + ((coeff_idx >> bwl) << TX_PAD_HOR_LOG2);
    let stats = self.get_nz_mag(&levels[padded_idx..], bwl, tx_class);

    self.get_nz_map_ctx_from_stats(stats, coeff_idx, bwl, tx_size, tx_class)
  }

  pub fn get_nz_map_contexts(
    &mut self, levels: &mut [u8], scan: &[u16; 32*32], eob: u16,
    tx_size: TxSize, tx_class: TxClass, coeff_contexts: &mut [i8]
  ) {
    // TODO: If TX_64X64 is enabled, use av1_get_adjusted_tx_size()
    let bwl = tx_size.width_log2();
    let height = tx_size.height();
    for i in 0..eob {
      let pos = scan[i as usize];
      coeff_contexts[pos as usize] = self.get_nz_map_ctx(
        levels,
        pos as usize,
        bwl,
        height,
        i as usize,
        i == eob - 1,
        tx_size,
        tx_class
      ) as i8;
    }
  }

  pub fn get_br_ctx(
    &mut self,
    levels: &[u8],
    c: usize, // raster order
    bwl: usize,
    tx_class: TxClass
  ) -> usize {
    let row: usize = c >> bwl;
    let col: usize = c - (row << bwl);
    let stride: usize = (1 << bwl) + TX_PAD_HOR;
    let pos: usize = row * stride + col;
    let mut mag: usize = levels[pos + 1] as usize;

    mag += levels[pos + stride] as usize;

    match tx_class {
      TX_CLASS_2D => {
        mag += levels[pos + stride + 1] as usize;
        mag = cmp::min((mag + 1) >> 1, 6);
        if c == 0 {
          return mag;
        }
        if (row < 2) && (col < 2) {
          return mag + 7;
        }
      }
      TX_CLASS_HORIZ => {
        mag += levels[pos + 2] as usize;
        mag = cmp::min((mag + 1) >> 1, 6);
        if c == 0 {
          return mag;
        }
        if col == 0 {
          return mag + 7;
        }
      }
      TX_CLASS_VERT => {
        mag += levels[pos + (stride << 1)] as usize;
        mag = cmp::min((mag + 1) >> 1, 6);
        if c == 0 {
          return mag;
        }
        if row == 0 {
          return mag + 7;
        }
      }
    }

    mag + 14
  }

  pub fn get_level_mag_with_txclass(
    &mut self, levels: &[u8], stride: usize, row: usize, col: usize,
    mag: &mut [usize], tx_class: TxClass
  ) {
    for idx in 0..CONTEXT_MAG_POSITION_NUM {
      let ref_row =
        row + mag_ref_offset_with_txclass[tx_class as usize][idx][0];
      let ref_col =
        col + mag_ref_offset_with_txclass[tx_class as usize][idx][1];
      let pos = ref_row * stride + ref_col;
      mag[idx] = levels[pos] as usize;
    }
  }

  pub fn write_coeffs_lv_map(
    &mut self, w: &mut dyn Writer, plane: usize, bo: &BlockOffset, coeffs_in: &[i32],
    pred_mode: PredictionMode,
    tx_size: TxSize, tx_type: TxType, plane_bsize: BlockSize, xdec: usize,
    ydec: usize, use_reduced_tx_set: bool
  ) -> bool {
    let is_inter = pred_mode >= PredictionMode::NEARESTMV;
    //assert!(!is_inter);
    // Note: Both intra and inter mode uses inter scan order. Surprised?
    let scan_order =
      &av1_scan_orders[tx_size as usize][tx_type as usize];
    let scan = scan_order.scan;
    let mut coeffs_storage = [0 as i32; 32 * 32];
    let coeffs = &mut coeffs_storage[..tx_size.area()];
    let mut cul_level = 0 as u32;

    for i in 0..tx_size.area() {
      coeffs[i] = coeffs_in[scan[i] as usize];
      cul_level += coeffs[i].abs() as u32;
    }

    let mut eob = 0;

    if cul_level != 0 {
      for (i, v) in coeffs.iter().enumerate() {
        if *v != 0 {
          eob = i + 1;
        }
      }
    }

    let txs_ctx = self.get_txsize_entropy_ctx(tx_size);
    let txb_ctx =
      self.bc.get_txb_ctx(plane_bsize, tx_size, plane, bo, xdec, ydec);

    {
      let cdf = &mut self.fc.txb_skip_cdf[txs_ctx][txb_ctx.txb_skip_ctx];
      symbol_with_update!(self, w, (eob == 0) as u32, cdf);
    }

    if eob == 0 {
      self.bc.set_coeff_context(plane, bo, tx_size, xdec, ydec, 0);
      return false;
    }

    let mut levels_buf = [0 as u8; TX_PAD_2D];

    self.txb_init_levels(
      coeffs_in,
      tx_size.width(),
      tx_size.height(),
      &mut levels_buf
    );

    let tx_class = tx_type_to_class[tx_type as usize];
    let plane_type = if plane == 0 {
      0
    } else {
      1
    } as usize;

    // Signal tx_type for luma plane only
    if plane == 0 {
      self.write_tx_type(
        w,
        tx_size,
        tx_type,
        pred_mode,
        is_inter,
        use_reduced_tx_set
      );
    }

    // Encode EOB
    let mut eob_extra = 0 as u32;
    let eob_pt = self.get_eob_pos_token(eob, &mut eob_extra);
    let eob_multi_size: usize = tx_size.area_log2() - 4;
    let eob_multi_ctx: usize = if tx_class == TX_CLASS_2D {
      0
    } else {
      1
    };

    match eob_multi_size {
      0 => {
        symbol_with_update!(
          self,
          w,
          eob_pt - 1,
          &mut self.fc.eob_flag_cdf16[plane_type][eob_multi_ctx]
        );
      }
      1 => {
        symbol_with_update!(
          self,
          w,
          eob_pt - 1,
          &mut self.fc.eob_flag_cdf32[plane_type][eob_multi_ctx]
        );
      }
      2 => {
        symbol_with_update!(
          self,
          w,
          eob_pt - 1,
          &mut self.fc.eob_flag_cdf64[plane_type][eob_multi_ctx]
        );
      }
      3 => {
        symbol_with_update!(
          self,
          w,
          eob_pt - 1,
          &mut self.fc.eob_flag_cdf128[plane_type][eob_multi_ctx]
        );
      }
      4 => {
        symbol_with_update!(
          self,
          w,
          eob_pt - 1,
          &mut self.fc.eob_flag_cdf256[plane_type][eob_multi_ctx]
        );
      }
      5 => {
        symbol_with_update!(
          self,
          w,
          eob_pt - 1,
          &mut self.fc.eob_flag_cdf512[plane_type][eob_multi_ctx]
        );
      }
      _ => {
        symbol_with_update!(
          self,
          w,
          eob_pt - 1,
          &mut self.fc.eob_flag_cdf1024[plane_type][eob_multi_ctx]
        );
      }
    };

    let eob_offset_bits = k_eob_offset_bits[eob_pt as usize];

    if eob_offset_bits > 0 {
      let mut eob_shift = eob_offset_bits - 1;
      let mut bit = if (eob_extra & (1 << eob_shift)) != 0 {
        1
      } else {
        0
      } as u32;
      symbol_with_update!(
        self,
        w,
        bit,
        &mut self.fc.eob_extra_cdf[txs_ctx][plane_type][(eob_pt - 3) as usize]
      );
      for i in 1..eob_offset_bits {
        eob_shift = eob_offset_bits as u16 - 1 - i as u16;
        bit = if (eob_extra & (1 << eob_shift)) != 0 {
          1
        } else {
          0
        };
        w.bit(bit as u16);
      }
    }

    let mut coeff_contexts = [0 as i8; MAX_TX_SQUARE];
    let levels =
      &mut levels_buf[TX_PAD_TOP * (tx_size.width() + TX_PAD_HOR)..];

    self.get_nz_map_contexts(
      levels,
      scan,
      eob as u16,
      tx_size,
      tx_class,
      &mut coeff_contexts
    );

    let bwl = self.get_txb_bwl(tx_size);

    for c in (0..eob).rev() {
      let pos = scan[c];
      let coeff_ctx = coeff_contexts[pos as usize];
      let v = coeffs_in[pos as usize];
      let level: u32 = v.abs() as u32;

      if c == eob - 1 {
        symbol_with_update!(
          self,
          w,
          (cmp::min(level, 3) - 1) as u32,
          &mut self.fc.coeff_base_eob_cdf[txs_ctx][plane_type]
            [coeff_ctx as usize]
        );
      } else {
        symbol_with_update!(
          self,
          w,
          (cmp::min(level, 3)) as u32,
          &mut self.fc.coeff_base_cdf[txs_ctx][plane_type][coeff_ctx as usize]
        );
      }

      if level > NUM_BASE_LEVELS as u32 {
        let pos = scan[c as usize];
        let v = coeffs_in[pos as usize];
        let level = v.abs() as u16;

        if level <= NUM_BASE_LEVELS as u16 {
          continue;
        }

        let base_range = level - 1 - NUM_BASE_LEVELS as u16;
        let br_ctx = self.get_br_ctx(levels, pos as usize, bwl, tx_class);
        let mut idx = 0;

        loop {
          if idx >= COEFF_BASE_RANGE {
            break;
          }
          let k = cmp::min(base_range - idx as u16, BR_CDF_SIZE as u16 - 1);
          symbol_with_update!(
            self,
            w,
            k as u32,
            &mut self.fc.coeff_br_cdf
              [cmp::min(txs_ctx, TxSize::TX_32X32 as usize)][plane_type]
              [br_ctx]
          );
          if k < BR_CDF_SIZE as u16 - 1 {
            break;
          }
          idx += BR_CDF_SIZE - 1;
        }
      }
    }

    // Loop to code all signs in the transform block,
    // starting with the sign of DC (if applicable)
    for c in 0..eob {
      let v = coeffs_in[scan[c] as usize];
      let level = v.abs() as u32;
      let sign = if v < 0 {
        1
      } else {
        0
      };

      if level == 0 {
        continue;
      }

      if c == 0 {
        symbol_with_update!(
          self,
          w,
          sign,
          &mut self.fc.dc_sign_cdf[plane_type][txb_ctx.dc_sign_ctx]
        );
      } else {
        w.bit(sign as u16);
      }
      // save extra golomb codes for separate loop
      if level > (COEFF_BASE_RANGE + NUM_BASE_LEVELS) as u32 {
        let pos = scan[c];
        w.write_golomb(
          coeffs_in[pos as usize].abs() as u16
            - COEFF_BASE_RANGE as u16
            - 1
            - NUM_BASE_LEVELS as u16
        );
      }
    }

    cul_level = cmp::min(COEFF_CONTEXT_MASK as u32, cul_level);

    self.bc.set_dc_sign(&mut cul_level, coeffs[0]);

    self.bc.set_coeff_context(plane, bo, tx_size, xdec, ydec, cul_level as u8);
    true
  }

  pub fn checkpoint(&mut self) -> ContextWriterCheckpoint {
    ContextWriterCheckpoint {
      fc: self.fc.clone(),
      bc: self.bc.checkpoint()
    }
  }

  pub fn rollback(&mut self, checkpoint: &ContextWriterCheckpoint) {
    self.fc = checkpoint.fc;
    self.bc.rollback(&checkpoint.bc);
    #[cfg(debug)] {
      if self.fc_map.is_some() {
        self.fc_map = Some(FieldMap {
          map: self.fc.build_map()
        });
      }
    }
  }
}

/* Symbols for coding magnitude class of nonzero components */
const MV_CLASSES:usize = 11;

// MV Class Types
const MV_CLASS_0: usize = 0;   /* (0, 2]     integer pel */
const MV_CLASS_1: usize = 1;   /* (2, 4]     integer pel */
const MV_CLASS_2: usize = 2;   /* (4, 8]     integer pel */
const MV_CLASS_3: usize = 3;   /* (8, 16]    integer pel */
const MV_CLASS_4: usize = 4;   /* (16, 32]   integer pel */
const MV_CLASS_5: usize = 5;   /* (32, 64]   integer pel */
const MV_CLASS_6: usize = 6;   /* (64, 128]  integer pel */
const MV_CLASS_7: usize = 7;   /* (128, 256] integer pel */
const MV_CLASS_8: usize = 8;   /* (256, 512] integer pel */
const MV_CLASS_9: usize = 9;   /* (512, 1024] integer pel */
const MV_CLASS_10: usize = 10; /* (1024,2048] integer pel */

const CLASS0_BITS: usize = 1; /* bits at integer precision for class 0 */
const CLASS0_SIZE: usize = (1 << CLASS0_BITS);
const MV_OFFSET_BITS: usize = (MV_CLASSES + CLASS0_BITS - 2);
const MV_BITS_CONTEXTS: usize = 6;
const MV_FP_SIZE: usize = 4;

const MV_MAX_BITS: usize = (MV_CLASSES + CLASS0_BITS + 2);
const MV_MAX: usize = ((1 << MV_MAX_BITS) - 1);
const MV_VALS: usize = ((MV_MAX << 1) + 1);

const MV_IN_USE_BITS: usize = 14;
const MV_UPP: i32 = (1 << MV_IN_USE_BITS);
const MV_LOW: i32 = (-(1 << MV_IN_USE_BITS));


#[inline(always)]
pub fn av1_get_mv_joint(mv: &MotionVector) -> MvJointType {
  if mv.row == 0 {
    if mv.col == 0 { MvJointType::MV_JOINT_ZERO } else { MvJointType::MV_JOINT_HNZVZ }
  } else {
    if mv.col == 0 { MvJointType::MV_JOINT_HZVNZ } else { MvJointType::MV_JOINT_HNZVNZ }
  }
}
#[inline(always)]
pub fn mv_joint_vertical(joint_type: MvJointType) -> bool {
  joint_type == MvJointType::MV_JOINT_HZVNZ || joint_type == MvJointType::MV_JOINT_HNZVNZ
}
#[inline(always)]
pub fn mv_joint_horizontal(joint_type: MvJointType ) -> bool {
  joint_type == MvJointType::MV_JOINT_HNZVZ || joint_type == MvJointType::MV_JOINT_HNZVNZ
}
#[inline(always)]
pub fn mv_class_base(mv_class: usize) -> u32 {
  if mv_class != MV_CLASS_0 {
    (CLASS0_SIZE << (mv_class as usize + 2)) as u32 }
  else { 0 }
}
#[inline(always)]
// If n != 0, returns the floor of log base 2 of n. If n == 0, returns 0.
pub fn log_in_base_2(n: u32) -> u8 {
  31 - cmp::min(31, n.leading_zeros() as u8)
}
#[inline(always)]
pub fn get_mv_class(z: u32, offset: &mut u32) -> usize {
  let c =
    if z >= CLASS0_SIZE as u32 * 4096 { MV_CLASS_10 }
    else { log_in_base_2(z >> 3) as usize };

  *offset = z - mv_class_base(c);
  c
}

pub fn encode_mv_component(w: &mut Writer, comp: i32, 
  mvcomp: &mut NMVComponent, precision: MvSubpelPrecision) {
  assert!(comp != 0);
  let mut offset: u32 = 0;
  let sign: u32 = if comp < 0 { 1 } else { 0 };
  let mag: u32 = if sign == 1 { -comp as u32 } else { comp as u32 };
  let mv_class = get_mv_class(mag - 1, &mut offset);
  let d = offset >> 3;         // int mv data
  let fr = (offset >> 1) & 3;  // fractional mv data
  let hp = offset & 1;         // high precision mv data

  // Sign
  w.symbol_with_update(sign, &mut mvcomp.sign_cdf);

  // Class
  w.symbol_with_update(mv_class as u32, &mut mvcomp.classes_cdf);

  // Integer bits
  if mv_class == MV_CLASS_0 {
    w.symbol_with_update(d, &mut mvcomp.class0_cdf);
  } else {
    let n = mv_class + CLASS0_BITS - 1;  // number of bits
    for i in 0..n {
      w.symbol_with_update((d >> i) & 1, &mut mvcomp.bits_cdf[i]);
    }
  }
  // Fractional bits
  if precision > MvSubpelPrecision::MV_SUBPEL_NONE {
    w.symbol_with_update(
        fr,
        if mv_class == MV_CLASS_0 { &mut mvcomp.class0_fp_cdf[d as usize] }
        else { &mut mvcomp.fp_cdf });
  }

  // High precision bit
  if precision > MvSubpelPrecision::MV_SUBPEL_LOW_PRECISION {
    w.symbol_with_update(
        hp, 
        if mv_class == MV_CLASS_0 { &mut mvcomp.class0_hp_cdf }
        else { &mut mvcomp.hp_cdf});
  }
}

