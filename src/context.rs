// Copyright (c) 2017-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(non_upper_case_globals)]
#![allow(dead_code)]
#![allow(non_camel_case_types)]

use crate::color::ChromaSampling;
use crate::ec::{Writer, OD_BITRES};
use crate::encoder::FrameInvariants;
use crate::entropymode::*;
use crate::frame::*;
use crate::header::ReferenceMode;
use crate::lrf::*;
use crate::mc::MotionVector;
use crate::partition::BlockSize::*;
use crate::partition::RefType::*;
use crate::partition::*;
use crate::predict::PredictionMode;
use crate::predict::PredictionMode::*;
use crate::scan_order::*;
use crate::tiling::*;
use crate::token_cdfs::*;
use crate::transform::TxSize::*;
use crate::transform::TxType::*;
use crate::transform::*;
use crate::util::*;

use arrayvec::*;
use std::default::Default;
use std::ops::{Add, Index, IndexMut};
use std::*;

pub const MAX_PLANES: usize = 3;

const PARTITION_PLOFFSET: usize = 4;
const PARTITION_BLOCK_SIZES: usize = 4 + 1;
const PARTITION_CONTEXTS_PRIMARY: usize =
  PARTITION_BLOCK_SIZES * PARTITION_PLOFFSET;
pub const PARTITION_CONTEXTS: usize = PARTITION_CONTEXTS_PRIMARY;
pub const PARTITION_TYPES: usize = 4;

pub const MI_SIZE_LOG2: usize = 2;
pub const MI_SIZE: usize = 1 << MI_SIZE_LOG2;
pub const MAX_MIB_SIZE_LOG2: usize = MAX_SB_SIZE_LOG2 - MI_SIZE_LOG2;
pub const MIB_SIZE_LOG2: usize = SB_SIZE_LOG2 - MI_SIZE_LOG2;
pub const MIB_SIZE: usize = 1 << MIB_SIZE_LOG2;
pub const MIB_MASK: usize = MIB_SIZE - 1;

const MAX_SB_SIZE_LOG2: usize = 7;
const SB_SIZE_LOG2: usize = 6;
pub const SB_SIZE: usize = 1 << SB_SIZE_LOG2;
const SB_SQUARE: usize = SB_SIZE * SB_SIZE;

pub const MAX_TX_SIZE: usize = 64;

pub const MAX_CODED_TX_SIZE: usize = 32;
const MAX_CODED_TX_SQUARE: usize = MAX_CODED_TX_SIZE * MAX_CODED_TX_SIZE;

pub const INTRA_MODES: usize = 13;
pub const UV_INTRA_MODES: usize = 14;

pub const CFL_JOINT_SIGNS: usize = 8;
pub const CFL_ALPHA_CONTEXTS: usize = 6;
pub const CFL_ALPHABET_SIZE: usize = 16;
pub const SKIP_MODE_CONTEXTS: usize = 3;
pub const COMP_INDEX_CONTEXTS: usize = 6;
pub const COMP_GROUP_IDX_CONTEXTS: usize = 6;

pub const BLOCK_SIZE_GROUPS: usize = 4;
pub const MAX_ANGLE_DELTA: usize = 3;
pub const DIRECTIONAL_MODES: usize = 8;
pub const KF_MODE_CONTEXTS: usize = 5;

pub const EXT_PARTITION_TYPES: usize = 10;

pub const TX_SIZE_SQR_CONTEXTS: usize = 4; // Coded tx_size <= 32x32, so is the # of CDF contexts from tx sizes

pub const TX_SETS: usize = 6;
pub const TX_SETS_INTRA: usize = 3;
pub const TX_SETS_INTER: usize = 4;

pub const TXFM_PARTITION_CONTEXTS: usize =
  (TxSize::TX_SIZES - TxSize::TX_8X8 as usize) * 6 - 3;

const MAX_VARTX_DEPTH: usize = 2;

const MAX_REF_MV_STACK_SIZE: usize = 8;
pub const REF_CAT_LEVEL: u32 = 640;

pub const FRAME_LF_COUNT: usize = 4;
pub const MAX_LOOP_FILTER: usize = 63;
const DELTA_LF_SMALL: u32 = 3;
pub const DELTA_LF_PROBS: usize = DELTA_LF_SMALL as usize;

const DELTA_Q_SMALL: u32 = 3;
pub const DELTA_Q_PROBS: usize = DELTA_Q_SMALL as usize;

// Number of transform types in each set type
static num_tx_set: [usize; TX_SETS] = [1, 2, 5, 7, 12, 16];
pub static av1_tx_used: [[usize; TX_TYPES]; TX_SETS] = [
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
  [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
  [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
];

// Maps set types above to the indices used for intra
static tx_set_index_intra: [i8; TX_SETS] = [0, -1, 2, 1, -1, -1];
// Maps set types above to the indices used for inter
static tx_set_index_inter: [i8; TX_SETS] = [0, 3, -1, -1, 2, 1];

static av1_tx_ind: [[usize; TX_TYPES]; TX_SETS] = [
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [1, 3, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [1, 5, 6, 4, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0],
  [3, 4, 5, 8, 6, 7, 9, 10, 11, 0, 1, 2, 0, 0, 0, 0],
  [7, 8, 9, 12, 10, 11, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6],
];

pub static max_txsize_rect_lookup: [TxSize; BlockSize::BLOCK_SIZES_ALL] = [
  TX_4X4,   // 4x4
  TX_4X8,   // 4x8
  TX_8X4,   // 8x4
  TX_8X8,   // 8x8
  TX_8X16,  // 8x16
  TX_16X8,  // 16x8
  TX_16X16, // 16x16
  TX_16X32, // 16x32
  TX_32X16, // 32x16
  TX_32X32, // 32x32
  TX_32X64, // 32x64
  TX_64X32, // 64x32
  TX_64X64, // 64x64
  TX_64X64, // 64x128
  TX_64X64, // 128x64
  TX_64X64, // 128x128
  TX_4X16,  // 4x16
  TX_16X4,  // 16x4
  TX_8X32,  // 8x32
  TX_32X8,  // 32x8
  TX_16X64, // 16x64
  TX_64X16, // 64x16
];

pub static sub_tx_size_map: [TxSize; TxSize::TX_SIZES_ALL] = [
  TX_4X4,   // TX_4X4
  TX_4X4,   // TX_8X8
  TX_8X8,   // TX_16X16
  TX_16X16, // TX_32X32
  TX_32X32, // TX_64X64
  TX_4X4,   // TX_4X8
  TX_4X4,   // TX_8X4
  TX_8X8,   // TX_8X16
  TX_8X8,   // TX_16X8
  TX_16X16, // TX_16X32
  TX_16X16, // TX_32X16
  TX_32X32, // TX_32X64
  TX_32X32, // TX_64X32
  TX_4X8,   // TX_4X16
  TX_8X4,   // TX_16X4
  TX_8X16,  // TX_8X32
  TX_16X8,  // TX_32X8
  TX_16X32, // TX_16X64
  TX_32X16, // TX_64X16
];

// Generates 4 bit field in which each bit set to 1 represents
// a blocksize partition  1111 means we split 64x64, 32x32, 16x16
// and 8x8.  1000 means we just split the 64x64 to 32x32
static partition_context_lookup: [[u8; 2]; BlockSize::BLOCK_SIZES_ALL] = [
  [31, 31], // 4X4   - {0b11111, 0b11111}
  [31, 30], // 4X8   - {0b11111, 0b11110}
  [30, 31], // 8X4   - {0b11110, 0b11111}
  [30, 30], // 8X8   - {0b11110, 0b11110}
  [30, 28], // 8X16  - {0b11110, 0b11100}
  [28, 30], // 16X8  - {0b11100, 0b11110}
  [28, 28], // 16X16 - {0b11100, 0b11100}
  [28, 24], // 16X32 - {0b11100, 0b11000}
  [24, 28], // 32X16 - {0b11000, 0b11100}
  [24, 24], // 32X32 - {0b11000, 0b11000}
  [24, 16], // 32X64 - {0b11000, 0b10000}
  [16, 24], // 64X32 - {0b10000, 0b11000}
  [16, 16], // 64X64 - {0b10000, 0b10000}
  [16, 0],  // 64X128- {0b10000, 0b00000}
  [0, 16],  // 128X64- {0b00000, 0b10000}
  [0, 0],   // 128X128-{0b00000, 0b00000}
  [31, 28], // 4X16  - {0b11111, 0b11100}
  [28, 31], // 16X4  - {0b11100, 0b11111}
  [30, 24], // 8X32  - {0b11110, 0b11000}
  [24, 30], // 32X8  - {0b11000, 0b11110}
  [28, 16], // 16X64 - {0b11100, 0b10000}
  [16, 28], // 64X16 - {0b10000, 0b11100}
];

static size_group_lookup: [u8; BlockSize::BLOCK_SIZES_ALL] =
  [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 2, 2];

static num_pels_log2_lookup: [u8; BlockSize::BLOCK_SIZES_ALL] =
  [4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 6, 6, 8, 8, 10, 10];

pub const PLANE_TYPES: usize = 2;
const REF_TYPES: usize = 2;
pub const SKIP_CONTEXTS: usize = 3;
pub const INTRA_INTER_CONTEXTS: usize = 4;
pub const INTER_MODE_CONTEXTS: usize = 8;
pub const DRL_MODE_CONTEXTS: usize = 3;
pub const COMP_INTER_CONTEXTS: usize = 5;
pub const COMP_REF_TYPE_CONTEXTS: usize = 5;
pub const UNI_COMP_REF_CONTEXTS: usize = 3;

// Level Map
pub const TXB_SKIP_CONTEXTS: usize = 13;

pub const EOB_COEF_CONTEXTS: usize = 9;

const SIG_COEF_CONTEXTS_2D: usize = 26;
const SIG_COEF_CONTEXTS_1D: usize = 16;
pub const SIG_COEF_CONTEXTS_EOB: usize = 4;
pub const SIG_COEF_CONTEXTS: usize =
  SIG_COEF_CONTEXTS_2D + SIG_COEF_CONTEXTS_1D;

const COEFF_BASE_CONTEXTS: usize = SIG_COEF_CONTEXTS;
pub const DC_SIGN_CONTEXTS: usize = 3;

const BR_TMP_OFFSET: usize = 12;
const BR_REF_CAT: usize = 4;
pub const LEVEL_CONTEXTS: usize = 21;

pub const NUM_BASE_LEVELS: usize = 2;

pub const BR_CDF_SIZE: usize = 4;
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
const TX_PAD_VER: usize = TX_PAD_TOP + TX_PAD_BOTTOM;
// Pad 16 extra bytes to avoid reading overflow in SIMD optimization.
const TX_PAD_END: usize = 16;
const TX_PAD_2D: usize = (MAX_CODED_TX_SIZE + TX_PAD_HOR)
  * (MAX_CODED_TX_SIZE + TX_PAD_VER)
  + TX_PAD_END;

const TX_CLASSES: usize = 3;

#[derive(Copy, Clone, PartialEq)]
pub enum TxClass {
  TX_CLASS_2D = 0,
  TX_CLASS_HORIZ = 1,
  TX_CLASS_VERT = 2,
}

#[derive(Copy, Clone, PartialEq)]
pub enum SegLvl {
  SEG_LVL_ALT_Q = 0,      /* Use alternate Quantizer .... */
  SEG_LVL_ALT_LF_Y_V = 1, /* Use alternate loop filter value on y plane vertical */
  SEG_LVL_ALT_LF_Y_H = 2, /* Use alternate loop filter value on y plane horizontal */
  SEG_LVL_ALT_LF_U = 3,   /* Use alternate loop filter value on u plane */
  SEG_LVL_ALT_LF_V = 4,   /* Use alternate loop filter value on v plane */
  SEG_LVL_REF_FRAME = 5,  /* Optional Segment reference frame */
  SEG_LVL_SKIP = 6,       /* Optional Segment (0,0) + skip mode */
  SEG_LVL_GLOBALMV = 7,
  SEG_LVL_MAX = 8,
}

pub const seg_feature_bits: [u32; SegLvl::SEG_LVL_MAX as usize] =
  [8, 6, 6, 6, 6, 3, 0, 0];

pub const seg_feature_is_signed: [bool; SegLvl::SEG_LVL_MAX as usize] =
  [true, true, true, true, true, false, false, false];

use crate::context::TxClass::*;

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
  TX_CLASS_HORIZ, // H_FLIPADST
];

static eob_to_pos_small: [u8; 33] = [
  0, 1, 2, // 0-2
  3, 3, // 3-4
  4, 4, 4, 4, // 5-8
  5, 5, 5, 5, 5, 5, 5, 5, // 9-16
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, // 17-32
];

static eob_to_pos_large: [u8; 17] = [
  6, // place holder
  7, // 33-64
  8, 8, // 65-128
  9, 9, 9, 9, // 129-256
  10, 10, 10, 10, 10, 10, 10, 10, // 257-512
  11, // 513-
];

static k_eob_group_start: [u16; 12] =
  [0, 1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513];
static k_eob_offset_bits: [u16; 12] = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

// The ctx offset table when TX is TX_CLASS_2D.
// TX col and row indices are clamped to 4

#[rustfmt::skip]
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
const NZ_MAP_CTX_5: usize = NZ_MAP_CTX_0 + 5;
const NZ_MAP_CTX_10: usize = NZ_MAP_CTX_0 + 10;

static nz_map_ctx_offset_1d: [usize; 32] = [
  NZ_MAP_CTX_0,
  NZ_MAP_CTX_5,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
  NZ_MAP_CTX_10,
];

const CONTEXT_MAG_POSITION_NUM: usize = 3;

static mag_ref_offset_with_txclass: [[[usize; 2]; CONTEXT_MAG_POSITION_NUM];
  3] = [
  [[0, 1], [1, 0], [1, 1]],
  [[0, 1], [1, 0], [0, 2]],
  [[0, 1], [1, 0], [2, 0]],
];

// End of Level Map

pub fn has_chroma(
  bo: TileBlockOffset, bsize: BlockSize, subsampling_x: usize,
  subsampling_y: usize, chroma_sampling: ChromaSampling,
) -> bool {
  if chroma_sampling == ChromaSampling::Cs400 {
    return false;
  };

  let bw = bsize.width_mi();
  let bh = bsize.height_mi();

  ((bo.0.x & 0x01) == 1 || (bw & 0x01) == 0 || subsampling_x == 0)
    && ((bo.0.y & 0x01) == 1 || (bh & 0x01) == 0 || subsampling_y == 0)
}

pub fn get_tx_set(
  tx_size: TxSize, is_inter: bool, use_reduced_set: bool,
) -> TxSet {
  let tx_size_sqr_up = tx_size.sqr_up();
  let tx_size_sqr = tx_size.sqr();

  if tx_size_sqr_up.block_size() > BlockSize::BLOCK_32X32 {
    return TxSet::TX_SET_DCTONLY;
  }

  if is_inter {
    if use_reduced_set || tx_size_sqr_up == TxSize::TX_32X32 {
      TxSet::TX_SET_INTER_3
    } else if tx_size_sqr == TxSize::TX_16X16 {
      TxSet::TX_SET_INTER_2
    } else {
      TxSet::TX_SET_INTER_1
    }
  } else if tx_size_sqr_up == TxSize::TX_32X32 {
    TxSet::TX_SET_DCTONLY
  } else if use_reduced_set || tx_size_sqr == TxSize::TX_16X16 {
    TxSet::TX_SET_INTRA_2
  } else {
    TxSet::TX_SET_INTRA_1
  }
}

fn get_tx_set_index(
  tx_size: TxSize, is_inter: bool, use_reduced_set: bool,
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
  ADST_DCT,  // D113
  DCT_ADST,  // D157
  DCT_ADST,  // D203
  ADST_DCT,  // D67
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
  D113_PRED,     // UV_D113_PRED
  D157_PRED,     // UV_D157_PRED
  D203_PRED,     // UV_D203_PRED
  D67_PRED,      // UV_D67_PRED
  SMOOTH_PRED,   // UV_SMOOTH_PRED
  SMOOTH_V_PRED, // UV_SMOOTH_V_PRED
  SMOOTH_H_PRED, // UV_SMOOTH_H_PRED
  PAETH_PRED,    // UV_PAETH_PRED
  DC_PRED,       // CFL_PRED
];

pub fn uv_intra_mode_to_tx_type_context(pred: PredictionMode) -> TxType {
  intra_mode_to_tx_type_context[uv2y[pred as usize] as usize]
}

#[derive(Clone, Copy)]
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

#[derive(Clone, Copy)]
pub struct NMVContext {
  joints_cdf: [u16; MV_JOINTS + 1],
  comps: [NMVComponent; 2],
}

// lv_map
static default_nmv_context: NMVContext = {
  NMVContext {
    joints_cdf: cdf!(4096, 11264, 19328),
    comps: [
      NMVComponent {
        classes_cdf: cdf!(
          28672, 30976, 31858, 32320, 32551, 32656, 32740, 32757, 32762, 32767
        ),
        class0_fp_cdf: [cdf!(16384, 24576, 26624), cdf!(12288, 21248, 24128)],
        fp_cdf: cdf!(8192, 17408, 21248),
        sign_cdf: cdf!(128 * 128),
        class0_hp_cdf: cdf!(160 * 128),
        hp_cdf: cdf!(128 * 128),
        class0_cdf: cdf!(216 * 128),
        bits_cdf: [
          cdf!(128 * 136),
          cdf!(128 * 140),
          cdf!(128 * 148),
          cdf!(128 * 160),
          cdf!(128 * 176),
          cdf!(128 * 192),
          cdf!(128 * 224),
          cdf!(128 * 234),
          cdf!(128 * 234),
          cdf!(128 * 240),
        ],
      },
      NMVComponent {
        classes_cdf: cdf!(
          28672, 30976, 31858, 32320, 32551, 32656, 32740, 32757, 32762, 32767
        ),
        class0_fp_cdf: [cdf!(16384, 24576, 26624), cdf!(12288, 21248, 24128)],
        fp_cdf: cdf!(8192, 17408, 21248),
        sign_cdf: cdf!(128 * 128),
        class0_hp_cdf: cdf!(160 * 128),
        hp_cdf: cdf!(128 * 128),
        class0_cdf: cdf!(216 * 128),
        bits_cdf: [
          cdf!(128 * 136),
          cdf!(128 * 140),
          cdf!(128 * 148),
          cdf!(128 * 160),
          cdf!(128 * 176),
          cdf!(128 * 192),
          cdf!(128 * 224),
          cdf!(128 * 234),
          cdf!(128 * 234),
          cdf!(128 * 240),
        ],
      },
    ],
  }
};

#[derive(Clone)]
pub struct CandidateMV {
  pub this_mv: MotionVector,
  pub comp_mv: MotionVector,
  pub weight: u32,
}

#[derive(Clone, Copy)]
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
  intra_tx_cdf: [[[[u16; TX_TYPES + 1]; INTRA_MODES]; TX_SIZE_SQR_CONTEXTS];
    TX_SETS_INTRA],
  inter_tx_cdf: [[[u16; TX_TYPES + 1]; TX_SIZE_SQR_CONTEXTS]; TX_SETS_INTER],
  tx_size_cdf: [[[u16; MAX_TX_DEPTH + 1 + 1]; TX_SIZE_CONTEXTS]; MAX_TX_CATS],
  txfm_partition_cdf: [[u16; 2 + 1]; TXFM_PARTITION_CONTEXTS],
  skip_cdfs: [[u16; 3]; SKIP_CONTEXTS],
  intra_inter_cdfs: [[u16; 3]; INTRA_INTER_CONTEXTS],
  angle_delta_cdf: [[u16; 2 * MAX_ANGLE_DELTA + 1 + 1]; DIRECTIONAL_MODES],
  filter_intra_cdfs: [[u16; 3]; BlockSize::BLOCK_SIZES_ALL],
  palette_y_mode_cdfs:
    [[[u16; 3]; PALETTE_Y_MODE_CONTEXTS]; PALETTE_BSIZE_CTXS],
  palette_uv_mode_cdfs: [[u16; 3]; PALETTE_UV_MODE_CONTEXTS],
  comp_mode_cdf: [[u16; 3]; COMP_INTER_CONTEXTS],
  comp_ref_type_cdf: [[u16; 3]; COMP_REF_TYPE_CONTEXTS],
  comp_ref_cdf: [[[u16; 3]; FWD_REFS - 1]; REF_CONTEXTS],
  comp_bwd_ref_cdf: [[[u16; 3]; BWD_REFS - 1]; REF_CONTEXTS],
  single_ref_cdfs: [[[u16; 2 + 1]; SINGLE_REFS - 1]; REF_CONTEXTS],
  drl_cdfs: [[u16; 2 + 1]; DRL_MODE_CONTEXTS],
  compound_mode_cdf: [[u16; INTER_COMPOUND_MODES + 1]; INTER_MODE_CONTEXTS],
  nmv_context: NMVContext,
  deblock_delta_multi_cdf: [[u16; DELTA_LF_PROBS + 1 + 1]; FRAME_LF_COUNT],
  deblock_delta_cdf: [u16; DELTA_LF_PROBS + 1 + 1],
  spatial_segmentation_cdfs: [[u16; 8 + 1]; 3],
  lrf_switchable_cdf: [u16; 3 + 1],
  lrf_sgrproj_cdf: [u16; 2 + 1],
  lrf_wiener_cdf: [u16; 2 + 1],

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
    TxSize::TX_SIZES],
}

impl CDFContext {
  pub fn new(quantizer: u8) -> CDFContext {
    let qctx = match quantizer {
      0..=20 => 0,
      21..=60 => 1,
      61..=120 => 2,
      _ => 3,
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
      tx_size_cdf: default_tx_size_cdf,
      txfm_partition_cdf: default_txfm_partition_cdf,
      skip_cdfs: default_skip_cdfs,
      intra_inter_cdfs: default_intra_inter_cdf,
      angle_delta_cdf: default_angle_delta_cdf,
      filter_intra_cdfs: default_filter_intra_cdfs,
      palette_y_mode_cdfs: default_palette_y_mode_cdfs,
      palette_uv_mode_cdfs: default_palette_uv_mode_cdfs,
      comp_mode_cdf: default_comp_mode_cdf,
      comp_ref_type_cdf: default_comp_ref_type_cdf,
      comp_ref_cdf: default_comp_ref_cdf,
      comp_bwd_ref_cdf: default_comp_bwdref_cdf,
      single_ref_cdfs: default_single_ref_cdf,
      drl_cdfs: default_drl_cdf,
      compound_mode_cdf: default_compound_mode_cdf,
      nmv_context: default_nmv_context,
      deblock_delta_multi_cdf: default_delta_lf_multi_cdf,
      deblock_delta_cdf: default_delta_lf_cdf,
      spatial_segmentation_cdfs: default_spatial_pred_seg_tree_cdf,
      lrf_switchable_cdf: default_switchable_restore_cdf,
      lrf_sgrproj_cdf: default_sgrproj_restore_cdf,
      lrf_wiener_cdf: default_wiener_restore_cdf,

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
      coeff_br_cdf: av1_default_coeff_lps_multi_cdfs[qctx],
    }
  }

  pub fn reset_counts(&mut self) {
    macro_rules! reset_1d {
      ($field:expr) => {
        let r = $field.last_mut().unwrap();
        *r = 0;
      };
    }
    macro_rules! reset_2d {
      ($field:expr) => {
        for x in $field.iter_mut() {
          reset_1d!(x);
        }
      };
    }
    macro_rules! reset_3d {
      ($field:expr) => {
        for x in $field.iter_mut() {
          reset_2d!(x);
        }
      };
    }
    macro_rules! reset_4d {
      ($field:expr) => {
        for x in $field.iter_mut() {
          reset_3d!(x);
        }
      };
    }

    for i in 0..4 {
      self.partition_cdf[i][4] = 0;
    }
    for i in 4..16 {
      self.partition_cdf[i][10] = 0;
    }
    for i in 16..20 {
      self.partition_cdf[i][8] = 0;
    }

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

    for i in 0..TX_SIZE_SQR_CONTEXTS {
      for j in 0..INTRA_MODES {
        self.intra_tx_cdf[1][i][j][7] = 0;
        self.intra_tx_cdf[2][i][j][5] = 0;
      }
      self.inter_tx_cdf[1][i][16] = 0;
      self.inter_tx_cdf[2][i][12] = 0;
      self.inter_tx_cdf[3][i][2] = 0;
    }

    for i in 0..TX_SIZE_CONTEXTS {
      self.tx_size_cdf[0][i][MAX_TX_DEPTH] = 0;
    }
    reset_2d!(self.tx_size_cdf[1]);
    reset_2d!(self.tx_size_cdf[2]);
    reset_2d!(self.tx_size_cdf[3]);

    for i in 0..TXFM_PARTITION_CONTEXTS {
      self.txfm_partition_cdf[i][2] = 0;
    }

    reset_2d!(self.skip_cdfs);
    reset_2d!(self.intra_inter_cdfs);
    reset_2d!(self.angle_delta_cdf);
    reset_2d!(self.filter_intra_cdfs);
    reset_3d!(self.palette_y_mode_cdfs);
    reset_2d!(self.palette_uv_mode_cdfs);
    reset_2d!(self.comp_mode_cdf);
    reset_2d!(self.comp_ref_type_cdf);
    reset_3d!(self.comp_ref_cdf);
    reset_3d!(self.comp_bwd_ref_cdf);
    reset_3d!(self.single_ref_cdfs);
    reset_2d!(self.drl_cdfs);
    reset_2d!(self.compound_mode_cdf);
    reset_2d!(self.deblock_delta_multi_cdf);
    reset_1d!(self.deblock_delta_cdf);
    reset_2d!(self.spatial_segmentation_cdfs);
    reset_1d!(self.lrf_switchable_cdf);
    reset_1d!(self.lrf_sgrproj_cdf);
    reset_1d!(self.lrf_wiener_cdf);

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
    let cfl_sign_cdf_end =
      cfl_sign_cdf_start + size_of_val(&self.cfl_sign_cdf);
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
    let palette_y_mode_cdfs_start =
      self.palette_y_mode_cdfs.first().unwrap().as_ptr() as usize;
    let palette_y_mode_cdfs_end =
      palette_y_mode_cdfs_start + size_of_val(&self.palette_y_mode_cdfs);
    let palette_uv_mode_cdfs_start =
      self.palette_uv_mode_cdfs.first().unwrap().as_ptr() as usize;
    let palette_uv_mode_cdfs_end =
      palette_uv_mode_cdfs_start + size_of_val(&self.palette_uv_mode_cdfs);
    let comp_mode_cdf_start =
      self.comp_mode_cdf.first().unwrap().as_ptr() as usize;
    let comp_mode_cdf_end =
      comp_mode_cdf_start + size_of_val(&self.comp_mode_cdf);
    let comp_ref_type_cdf_start =
      self.comp_ref_type_cdf.first().unwrap().as_ptr() as usize;
    let comp_ref_type_cdf_end =
      comp_ref_type_cdf_start + size_of_val(&self.comp_ref_type_cdf);
    let comp_ref_cdf_start =
      self.comp_ref_cdf.first().unwrap().as_ptr() as usize;
    let comp_ref_cdf_end =
      comp_ref_cdf_start + size_of_val(&self.comp_ref_cdf);
    let comp_bwd_ref_cdf_start =
      self.comp_bwd_ref_cdf.first().unwrap().as_ptr() as usize;
    let comp_bwd_ref_cdf_end =
      comp_bwd_ref_cdf_start + size_of_val(&self.comp_bwd_ref_cdf);
    let deblock_delta_multi_cdf_start =
      self.deblock_delta_multi_cdf.first().unwrap().as_ptr() as usize;
    let deblock_delta_multi_cdf_end = deblock_delta_multi_cdf_start
      + size_of_val(&self.deblock_delta_multi_cdf);
    let deblock_delta_cdf_start = self.deblock_delta_cdf.as_ptr() as usize;
    let deblock_delta_cdf_end =
      deblock_delta_cdf_start + size_of_val(&self.deblock_delta_cdf);
    let spatial_segmentation_cdfs_start =
      self.spatial_segmentation_cdfs.first().unwrap().as_ptr() as usize;
    let spatial_segmentation_cdfs_end = spatial_segmentation_cdfs_start
      + size_of_val(&self.spatial_segmentation_cdfs);
    let lrf_switchable_cdf_start = self.lrf_switchable_cdf.as_ptr() as usize;
    let lrf_switchable_cdf_end =
      lrf_switchable_cdf_start + size_of_val(&self.lrf_switchable_cdf);
    let lrf_sgrproj_cdf_start = self.lrf_sgrproj_cdf.as_ptr() as usize;
    let lrf_sgrproj_cdf_end =
      lrf_sgrproj_cdf_start + size_of_val(&self.lrf_sgrproj_cdf);
    let lrf_wiener_cdf_start = self.lrf_wiener_cdf.as_ptr() as usize;
    let lrf_wiener_cdf_end =
      lrf_wiener_cdf_start + size_of_val(&self.lrf_wiener_cdf);

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
      (
        "palette_y_mode_cdfs",
        palette_y_mode_cdfs_start,
        palette_y_mode_cdfs_end,
      ),
      (
        "palette_uv_mode_cdfs",
        palette_uv_mode_cdfs_start,
        palette_uv_mode_cdfs_end,
      ),
      ("comp_mode_cdf", comp_mode_cdf_start, comp_mode_cdf_end),
      ("comp_ref_type_cdf", comp_ref_type_cdf_start, comp_ref_type_cdf_end),
      ("comp_ref_cdf", comp_ref_cdf_start, comp_ref_cdf_end),
      ("comp_bwd_ref_cdf", comp_bwd_ref_cdf_start, comp_bwd_ref_cdf_end),
      (
        "deblock_delta_multi_cdf",
        deblock_delta_multi_cdf_start,
        deblock_delta_multi_cdf_end,
      ),
      ("deblock_delta_cdf", deblock_delta_cdf_start, deblock_delta_cdf_end),
      (
        "spatial_segmentation_cdfs",
        spatial_segmentation_cdfs_start,
        spatial_segmentation_cdfs_end,
      ),
      ("lrf_switchable_cdf", lrf_switchable_cdf_start, lrf_switchable_cdf_end),
      ("lrf_sgrproj_cdf", lrf_sgrproj_cdf_start, lrf_sgrproj_cdf_end),
      ("lrf_wiener_cdf", lrf_wiener_cdf_start, lrf_wiener_cdf_end),
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
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "CDFContext contains too many numbers to print :-(")
  }
}

#[cfg(test)]
mod test {
  #[test]
  fn cdf_map() {
    use super::*;

    let cdf = CDFContext::new(8);
    let cdf_map = FieldMap { map: cdf.build_map() };
    let f = &cdf.partition_cdf[2];
    cdf_map.lookup(f.as_ptr() as usize);
  }

  use super::CFLSign;
  use super::CFLSign::*;

  static cfl_alpha_signs: [[CFLSign; 2]; 8] = [
    [CFL_SIGN_ZERO, CFL_SIGN_NEG],
    [CFL_SIGN_ZERO, CFL_SIGN_POS],
    [CFL_SIGN_NEG, CFL_SIGN_ZERO],
    [CFL_SIGN_NEG, CFL_SIGN_NEG],
    [CFL_SIGN_NEG, CFL_SIGN_POS],
    [CFL_SIGN_POS, CFL_SIGN_ZERO],
    [CFL_SIGN_POS, CFL_SIGN_NEG],
    [CFL_SIGN_POS, CFL_SIGN_POS],
  ];

  static cfl_context: [[usize; 8]; 2] =
    [[0, 0, 0, 1, 2, 3, 4, 5], [0, 3, 0, 1, 4, 0, 2, 5]];

  #[test]
  fn cfl_joint_sign() {
    use super::*;

    let mut cfl = CFLParams::default();
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

pub const SUPERBLOCK_TO_PLANE_SHIFT: usize = SB_SIZE_LOG2;
pub const SUPERBLOCK_TO_BLOCK_SHIFT: usize = MIB_SIZE_LOG2;
pub const BLOCK_TO_PLANE_SHIFT: usize = MI_SIZE_LOG2;
pub const IMPORTANCE_BLOCK_TO_BLOCK_SHIFT: usize = 1;
pub const LOCAL_BLOCK_MASK: usize = (1 << SUPERBLOCK_TO_BLOCK_SHIFT) - 1;

/// Absolute offset in superblocks, where a superblock is defined
/// to be an N*N square where N = (1 << SUPERBLOCK_TO_PLANE_SHIFT).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SuperBlockOffset {
  pub x: usize,
  pub y: usize,
}

/// Absolute offset in superblocks inside a plane, where a superblock is defined
/// to be an N*N square where N = (1 << SUPERBLOCK_TO_PLANE_SHIFT).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneSuperBlockOffset(pub SuperBlockOffset);

/// Absolute offset in superblocks inside a tile, where a superblock is defined
/// to be an N*N square where N = (1 << SUPERBLOCK_TO_PLANE_SHIFT).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileSuperBlockOffset(pub SuperBlockOffset);

impl SuperBlockOffset {
  /// Offset of a block inside the current superblock.
  const fn block_offset(self, block_x: usize, block_y: usize) -> BlockOffset {
    BlockOffset {
      x: (self.x << SUPERBLOCK_TO_BLOCK_SHIFT) + block_x,
      y: (self.y << SUPERBLOCK_TO_BLOCK_SHIFT) + block_y,
    }
  }

  /// Offset of the top-left pixel of this block.
  const fn plane_offset(self, plane: &PlaneConfig) -> PlaneOffset {
    PlaneOffset {
      x: (self.x as isize) << (SUPERBLOCK_TO_PLANE_SHIFT - plane.xdec),
      y: (self.y as isize) << (SUPERBLOCK_TO_PLANE_SHIFT - plane.ydec),
    }
  }
}

impl Add for SuperBlockOffset {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    Self { x: self.x + rhs.x, y: self.y + rhs.y }
  }
}

impl PlaneSuperBlockOffset {
  /// Offset of a block inside the current superblock.
  #[inline]
  pub const fn block_offset(
    self, block_x: usize, block_y: usize,
  ) -> PlaneBlockOffset {
    PlaneBlockOffset(self.0.block_offset(block_x, block_y))
  }

  /// Offset of the top-left pixel of this block.
  #[inline]
  pub const fn plane_offset(self, plane: &PlaneConfig) -> PlaneOffset {
    self.0.plane_offset(plane)
  }
}

impl Add for PlaneSuperBlockOffset {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    PlaneSuperBlockOffset(self.0 + rhs.0)
  }
}

impl TileSuperBlockOffset {
  /// Offset of a block inside the current superblock.
  #[inline]
  pub const fn block_offset(
    self, block_x: usize, block_y: usize,
  ) -> TileBlockOffset {
    TileBlockOffset(self.0.block_offset(block_x, block_y))
  }

  /// Offset of the top-left pixel of this block.
  #[inline]
  pub const fn plane_offset(self, plane: &PlaneConfig) -> PlaneOffset {
    self.0.plane_offset(plane)
  }
}

impl Add for TileSuperBlockOffset {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    TileSuperBlockOffset(self.0 + rhs.0)
  }
}

/// Absolute offset in blocks, where a block is defined
/// to be an N*N square where N = (1 << BLOCK_TO_PLANE_SHIFT).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockOffset {
  pub x: usize,
  pub y: usize,
}

/// Absolute offset in blocks inside a plane, where a block is defined
/// to be an N*N square where N = (1 << BLOCK_TO_PLANE_SHIFT).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneBlockOffset(pub BlockOffset);

/// Absolute offset in blocks inside a tile, where a block is defined
/// to be an N*N square where N = (1 << BLOCK_TO_PLANE_SHIFT).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TileBlockOffset(pub BlockOffset);

impl BlockOffset {
  /// Offset of the superblock in which this block is located.
  const fn sb_offset(self) -> SuperBlockOffset {
    SuperBlockOffset {
      x: self.x >> SUPERBLOCK_TO_BLOCK_SHIFT,
      y: self.y >> SUPERBLOCK_TO_BLOCK_SHIFT,
    }
  }

  /// Offset of the top-left pixel of this block.
  const fn plane_offset(self, plane: &PlaneConfig) -> PlaneOffset {
    PlaneOffset {
      x: (self.x >> plane.xdec << BLOCK_TO_PLANE_SHIFT) as isize,
      y: (self.y >> plane.ydec << BLOCK_TO_PLANE_SHIFT) as isize,
    }
  }

  /// Convert to plane offset without decimation.
  #[inline]
  const fn to_luma_plane_offset(self) -> PlaneOffset {
    PlaneOffset {
      x: (self.x as isize) << BLOCK_TO_PLANE_SHIFT,
      y: (self.y as isize) << BLOCK_TO_PLANE_SHIFT,
    }
  }

  const fn y_in_sb(self) -> usize {
    self.y % MIB_SIZE
  }

  fn with_offset(self, col_offset: isize, row_offset: isize) -> BlockOffset {
    let x = self.x as isize + col_offset;
    let y = self.y as isize + row_offset;
    debug_assert!(x >= 0);
    debug_assert!(y >= 0);

    BlockOffset { x: x as usize, y: y as usize }
  }
}

impl PlaneBlockOffset {
  /// Offset of the superblock in which this block is located.
  #[inline]
  pub const fn sb_offset(self) -> PlaneSuperBlockOffset {
    PlaneSuperBlockOffset(self.0.sb_offset())
  }

  /// Offset of the top-left pixel of this block.
  #[inline]
  pub const fn plane_offset(self, plane: &PlaneConfig) -> PlaneOffset {
    self.0.plane_offset(plane)
  }

  /// Convert to plane offset without decimation.
  #[inline]
  pub const fn to_luma_plane_offset(self) -> PlaneOffset {
    self.0.to_luma_plane_offset()
  }

  #[inline]
  pub const fn y_in_sb(self) -> usize {
    self.0.y_in_sb()
  }

  #[inline]
  pub fn with_offset(
    self, col_offset: isize, row_offset: isize,
  ) -> PlaneBlockOffset {
    Self(self.0.with_offset(col_offset, row_offset))
  }
}

impl TileBlockOffset {
  /// Offset of the superblock in which this block is located.
  #[inline]
  pub const fn sb_offset(self) -> TileSuperBlockOffset {
    TileSuperBlockOffset(self.0.sb_offset())
  }

  /// Offset of the top-left pixel of this block.
  #[inline]
  pub const fn plane_offset(self, plane: &PlaneConfig) -> PlaneOffset {
    self.0.plane_offset(plane)
  }

  /// Convert to plane offset without decimation.
  #[inline]
  pub const fn to_luma_plane_offset(self) -> PlaneOffset {
    self.0.to_luma_plane_offset()
  }

  #[inline]
  pub const fn y_in_sb(self) -> usize {
    self.0.y_in_sb()
  }

  #[inline]
  pub fn with_offset(
    self, col_offset: isize, row_offset: isize,
  ) -> TileBlockOffset {
    Self(self.0.with_offset(col_offset, row_offset))
  }
}

#[derive(Copy, Clone)]
pub struct Block {
  pub mode: PredictionMode,
  pub partition: PartitionType,
  pub skip: bool,
  pub ref_frames: [RefType; 2],
  pub mv: [MotionVector; 2],
  // note: indexes are reflist index, NOT the same as libaom
  pub neighbors_ref_counts: [usize; INTER_REFS_PER_FRAME],
  pub cdef_index: u8,
  pub bsize: BlockSize,
  pub n4_w: usize, /* block width in the unit of mode_info */
  pub n4_h: usize, /* block height in the unit of mode_info */
  pub txsize: TxSize,
  // The block-level deblock_deltas are left-shifted by
  // fi.deblock.block_delta_shift and added to the frame-configured
  // deltas
  pub deblock_deltas: [i8; FRAME_LF_COUNT],
  pub segmentation_idx: u8,
}

impl Block {
  pub fn is_inter(&self) -> bool {
    self.mode >= PredictionMode::NEARESTMV
  }
  pub fn has_second_ref(&self) -> bool {
    self.ref_frames[1] != INTRA_FRAME && self.ref_frames[1] != NONE_FRAME
  }
}

impl Default for Block {
  fn default() -> Block {
    Block {
      mode: PredictionMode::DC_PRED,
      partition: PartitionType::PARTITION_NONE,
      skip: false,
      ref_frames: [INTRA_FRAME; 2],
      mv: [MotionVector::default(); 2],
      neighbors_ref_counts: [0; INTER_REFS_PER_FRAME],
      cdef_index: 0,
      bsize: BLOCK_64X64,
      n4_w: BLOCK_64X64.width_mi(),
      n4_h: BLOCK_64X64.height_mi(),
      txsize: TX_64X64,
      deblock_deltas: [0, 0, 0, 0],
      segmentation_idx: 0,
    }
  }
}

pub struct TXB_CTX {
  pub txb_skip_ctx: usize,
  pub dc_sign_ctx: usize,
}

#[derive(Clone)]
pub struct FrameBlocks {
  blocks: Box<[Block]>,
  pub cols: usize,
  pub rows: usize,
}

impl FrameBlocks {
  pub fn new(cols: usize, rows: usize) -> Self {
    Self {
      blocks: vec![Block::default(); cols * rows].into_boxed_slice(),
      cols,
      rows,
    }
  }

  #[inline(always)]
  pub fn as_tile_blocks(&self) -> TileBlocks<'_> {
    TileBlocks::new(self, 0, 0, self.cols, self.rows)
  }

  #[inline(always)]
  pub fn as_tile_blocks_mut(&mut self) -> TileBlocksMut<'_> {
    TileBlocksMut::new(self, 0, 0, self.cols, self.rows)
  }
}

impl Index<usize> for FrameBlocks {
  type Output = [Block];
  #[inline]
  fn index(&self, index: usize) -> &Self::Output {
    &self.blocks[index * self.cols..(index + 1) * self.cols]
  }
}

impl IndexMut<usize> for FrameBlocks {
  #[inline]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    &mut self.blocks[index * self.cols..(index + 1) * self.cols]
  }
}

// for convenience, also index by BlockOffset

impl Index<PlaneBlockOffset> for FrameBlocks {
  type Output = Block;
  #[inline]
  fn index(&self, bo: PlaneBlockOffset) -> &Self::Output {
    &self[bo.0.y][bo.0.x]
  }
}

impl IndexMut<PlaneBlockOffset> for FrameBlocks {
  #[inline]
  fn index_mut(&mut self, bo: PlaneBlockOffset) -> &mut Self::Output {
    &mut self[bo.0.y][bo.0.x]
  }
}

// partition contexts are at 8x8 granularity, as it is not possible to
// split 4x4 blocks any further than that
const PARTITION_CONTEXT_GRANULARITY: usize = 8;
const PARTITION_CONTEXT_MAX_WIDTH: usize =
  MAX_TILE_WIDTH / PARTITION_CONTEXT_GRANULARITY;

const COEFF_CONTEXT_MAX_WIDTH: usize = MAX_TILE_WIDTH / MI_SIZE;

#[derive(Clone)]
pub struct BlockContextCheckpoint {
  cdef_coded: bool,
  above_partition_context: [u8; PARTITION_CONTEXT_MAX_WIDTH],
  // left context is also at 8x8 granularity
  left_partition_context: [u8; MIB_SIZE >> 1],
  above_tx_context: [u8; COEFF_CONTEXT_MAX_WIDTH],
  left_tx_context: [u8; MIB_SIZE],
  above_coeff_context: [[u8; COEFF_CONTEXT_MAX_WIDTH]; MAX_PLANES],
  left_coeff_context: [[u8; MIB_SIZE]; MAX_PLANES],
}

pub struct BlockContext<'a> {
  pub cdef_coded: bool,
  pub code_deltas: bool,
  pub update_seg: bool,
  pub preskip_segid: bool,
  above_partition_context: [u8; PARTITION_CONTEXT_MAX_WIDTH],
  left_partition_context: [u8; MIB_SIZE >> 1],
  above_tx_context: [u8; COEFF_CONTEXT_MAX_WIDTH],
  left_tx_context: [u8; MIB_SIZE],
  above_coeff_context: [[u8; COEFF_CONTEXT_MAX_WIDTH]; MAX_PLANES],
  left_coeff_context: [[u8; MIB_SIZE]; MAX_PLANES],
  pub blocks: &'a mut TileBlocksMut<'a>,
}

impl<'a> BlockContext<'a> {
  pub fn new(blocks: &'a mut TileBlocksMut<'a>) -> Self {
    BlockContext {
      cdef_coded: false,
      code_deltas: false,
      update_seg: false,
      preskip_segid: false,
      above_partition_context: [0; PARTITION_CONTEXT_MAX_WIDTH],
      left_partition_context: [0; MIB_SIZE >> 1],
      above_tx_context: [0; COEFF_CONTEXT_MAX_WIDTH],
      left_tx_context: [0; MIB_SIZE],
      above_coeff_context: [
        [0; COEFF_CONTEXT_MAX_WIDTH],
        [0; COEFF_CONTEXT_MAX_WIDTH],
        [0; COEFF_CONTEXT_MAX_WIDTH],
      ],
      left_coeff_context: [[0; MIB_SIZE]; MAX_PLANES],
      blocks,
    }
  }

  pub const fn checkpoint(&self) -> BlockContextCheckpoint {
    BlockContextCheckpoint {
      cdef_coded: self.cdef_coded,
      above_partition_context: self.above_partition_context,
      left_partition_context: self.left_partition_context,
      above_tx_context: self.above_tx_context,
      left_tx_context: self.left_tx_context,
      above_coeff_context: self.above_coeff_context,
      left_coeff_context: self.left_coeff_context,
    }
  }

  pub fn rollback(&mut self, checkpoint: &BlockContextCheckpoint) {
    self.cdef_coded = checkpoint.cdef_coded;
    self.above_partition_context = checkpoint.above_partition_context;
    self.left_partition_context = checkpoint.left_partition_context;
    self.above_tx_context = checkpoint.above_tx_context;
    self.left_tx_context = checkpoint.left_tx_context;
    self.above_coeff_context = checkpoint.above_coeff_context;
    self.left_coeff_context = checkpoint.left_coeff_context;
  }

  pub fn set_dc_sign(cul_level: &mut u32, dc_val: i32) {
    if dc_val < 0 {
      *cul_level |= 1 << COEFF_CONTEXT_BITS;
    } else if dc_val > 0 {
      *cul_level += 2 << COEFF_CONTEXT_BITS;
    }
  }

  fn set_coeff_context(
    &mut self, plane: usize, bo: TileBlockOffset, tx_size: TxSize,
    xdec: usize, ydec: usize, value: u8,
  ) {
    for above in &mut self.above_coeff_context[plane][(bo.0.x >> xdec)..]
      [..tx_size.width_mi()]
    {
      *above = value;
    }
    let bo_y = bo.y_in_sb();
    for left in &mut self.left_coeff_context[plane][(bo_y >> ydec)..]
      [..tx_size.height_mi()]
    {
      *left = value;
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

  pub fn update_tx_size_context(
    &mut self, bo: TileBlockOffset, bsize: BlockSize, tx_size: TxSize,
    skip: bool,
  ) {
    let n4_w = bsize.width_mi();
    let n4_h = bsize.height_mi();

    let (tx_w, tx_h) = if skip {
      (n4_w as u8, n4_h as u8)
    } else {
      (tx_size.width() as u8, tx_size.height() as u8)
    };

    let above_ctx = &mut self.above_tx_context[bo.0.x..bo.0.x + n4_w as usize];
    let left_ctx =
      &mut self.left_tx_context[bo.y_in_sb()..bo.y_in_sb() + n4_h as usize];

    for v in above_ctx[0..n4_w].iter_mut() {
      *v = tx_w;
    }

    for v in left_ctx[0..n4_h].iter_mut() {
      *v = tx_h;
    }
  }

  fn reset_left_tx_context(&mut self) {
    for c in &mut self.left_tx_context {
      *c = 0;
    }
  }

  pub fn reset_skip_context(
    &mut self, bo: TileBlockOffset, bsize: BlockSize, xdec: usize,
    ydec: usize, cs: ChromaSampling,
  ) {
    let num_planes = if cs == ChromaSampling::Cs400 { 1 } else { 3 };
    let nplanes = if bsize >= BLOCK_8X8 {
      num_planes
    } else {
      1 + (num_planes - 1) * has_chroma(bo, bsize, xdec, ydec, cs) as usize
    };

    for plane in 0..nplanes {
      let xdec2 = if plane == 0 { 0 } else { xdec };
      let ydec2 = if plane == 0 { 0 } else { ydec };

      let plane_bsize =
        if plane == 0 { bsize } else { bsize.subsampled_size(xdec2, ydec2) };
      let bw = plane_bsize.width_mi();
      let bh = plane_bsize.height_mi();

      for above in
        &mut self.above_coeff_context[plane][(bo.0.x >> xdec2)..][..bw]
      {
        *above = 0;
      }

      let bo_y = bo.y_in_sb();
      for left in &mut self.left_coeff_context[plane][(bo_y >> ydec2)..][..bh]
      {
        *left = 0;
      }
    }
  }

  pub fn reset_left_contexts(&mut self, planes: usize) {
    for p in 0..planes {
      BlockContext::reset_left_coeff_context(self, p);
    }
    BlockContext::reset_left_partition_context(self);

    BlockContext::reset_left_tx_context(self);
  }

  fn partition_plane_context(
    &self, bo: TileBlockOffset, bsize: BlockSize,
  ) -> usize {
    // TODO: this should be way simpler without sub8x8
    let above_ctx = self.above_partition_context[bo.0.x >> 1];
    let left_ctx = self.left_partition_context[bo.y_in_sb() >> 1];
    let bsl = bsize.width_log2() - BLOCK_8X8.width_log2();
    let above = (above_ctx >> bsl) & 1;
    let left = (left_ctx >> bsl) & 1;

    assert!(bsize.is_sqr());

    (left * 2 + above) as usize + bsl as usize * PARTITION_PLOFFSET
  }

  pub fn update_partition_context(
    &mut self, bo: TileBlockOffset, subsize: BlockSize, bsize: BlockSize,
  ) {
    #[allow(dead_code)]
    assert!(bsize.is_sqr());

    let bw = bsize.width_mi();
    let bh = bsize.height_mi();

    let above_ctx = &mut self.above_partition_context
      [bo.0.x >> 1..(bo.0.x + bw) >> 1 as usize];
    let left_ctx = &mut self.left_partition_context
      [bo.y_in_sb() >> 1..(bo.y_in_sb() + bh) >> 1 as usize];

    // update the partition context at the end notes. set partition bits
    // of block sizes larger than the current one to be one, and partition
    // bits of smaller block sizes to be zero.
    for above in &mut above_ctx[..bw >> 1] {
      *above = partition_context_lookup[subsize as usize][0];
    }

    for left in &mut left_ctx[..bh >> 1] {
      *left = partition_context_lookup[subsize as usize][1];
    }
  }

  fn skip_context(&self, bo: TileBlockOffset) -> usize {
    let above_skip = bo.0.y > 0 && self.blocks.above_of(bo).skip;
    let left_skip = bo.0.x > 0 && self.blocks.left_of(bo).skip;
    above_skip as usize + left_skip as usize
  }

  // The mode info data structure has a one element border above and to the
  // left of the entries corresponding to real macroblocks.
  // The prediction flags in these dummy entries are initialized to 0.
  // 0 - inter/inter, inter/--, --/inter, --/--
  // 1 - intra/inter, inter/intra
  // 2 - intra/--, --/intra
  // 3 - intra/intra
  pub fn intra_inter_context(&self, bo: TileBlockOffset) -> usize {
    let has_above = bo.0.y > 0;
    let has_left = bo.0.x > 0;

    match (has_above, has_left) {
      (true, true) => {
        let above_intra = !self.blocks.above_of(bo).is_inter();
        let left_intra = !self.blocks.left_of(bo).is_inter();
        if above_intra && left_intra {
          3
        } else {
          (above_intra || left_intra) as usize
        }
      }
      (true, false) => {
        if self.blocks.above_of(bo).is_inter() {
          0
        } else {
          2
        }
      }
      (false, true) => {
        if self.blocks.left_of(bo).is_inter() {
          0
        } else {
          2
        }
      }
      _ => 0,
    }
  }

  pub fn get_txb_ctx(
    &self, plane_bsize: BlockSize, tx_size: TxSize, plane: usize,
    bo: TileBlockOffset, xdec: usize, ydec: usize,
  ) -> TXB_CTX {
    let mut txb_ctx = TXB_CTX { txb_skip_ctx: 0, dc_sign_ctx: 0 };
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

    let above_ctxs =
      &self.above_coeff_context[plane][(bo.0.x >> xdec)..][..txb_w_unit];
    let left_ctxs =
      &self.left_coeff_context[plane][(bo.y_in_sb() >> ydec)..][..txb_h_unit];

    // Decide txb_ctx.dc_sign_ctx
    for &ctx in above_ctxs {
      let sign = ctx >> COEFF_CONTEXT_BITS;
      dc_sign += signs[sign as usize] as i16;
    }

    for &ctx in left_ctxs {
      let sign = ctx >> COEFF_CONTEXT_BITS;
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
          [1, 4, 4, 4, 6],
        ];

        let top: u8 = above_ctxs.iter().fold(0, |acc, ctx| acc | *ctx)
          & COEFF_CONTEXT_MASK as u8;

        let left: u8 = left_ctxs.iter().fold(0, |acc, ctx| acc | *ctx)
          & COEFF_CONTEXT_MASK as u8;

        let max = cmp::min(top | left, 4);
        let min = cmp::min(cmp::min(top, left), 4);
        txb_ctx.txb_skip_ctx =
          skip_contexts[min as usize][max as usize] as usize;
      }
    } else {
      let top: u8 = above_ctxs.iter().fold(0, |acc, ctx| acc | *ctx);
      let left: u8 = left_ctxs.iter().fold(0, |acc, ctx| acc | *ctx);
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

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum CFLSign {
  CFL_SIGN_ZERO = 0,
  CFL_SIGN_NEG = 1,
  CFL_SIGN_POS = 2,
}

impl CFLSign {
  pub fn from_alpha(a: i16) -> CFLSign {
    [CFL_SIGN_NEG, CFL_SIGN_ZERO, CFL_SIGN_POS][(a.signum() + 1) as usize]
  }
}

use crate::context::CFLSign::*;
const CFL_SIGNS: usize = 3;
static cfl_sign_value: [i16; CFL_SIGNS] = [0, -1, 1];

#[derive(Copy, Clone, Debug)]
pub struct CFLParams {
  sign: [CFLSign; 2],
  scale: [u8; 2],
}

impl Default for CFLParams {
  fn default() -> Self {
    Self { sign: [CFL_SIGN_NEG, CFL_SIGN_ZERO], scale: [1, 0] }
  }
}

impl CFLParams {
  pub fn joint_sign(self) -> u32 {
    assert!(self.sign[0] != CFL_SIGN_ZERO || self.sign[1] != CFL_SIGN_ZERO);
    (self.sign[0] as u32) * (CFL_SIGNS as u32) + (self.sign[1] as u32) - 1
  }
  pub fn context(self, uv: usize) -> usize {
    assert!(self.sign[uv] != CFL_SIGN_ZERO);
    (self.sign[uv] as usize - 1) * CFL_SIGNS + (self.sign[1 - uv] as usize)
  }
  pub fn index(self, uv: usize) -> u32 {
    assert!(self.sign[uv] != CFL_SIGN_ZERO && self.scale[uv] != 0);
    (self.scale[uv] - 1) as u32
  }
  pub fn alpha(self, uv: usize) -> i16 {
    cfl_sign_value[self.sign[uv] as usize] * (self.scale[uv] as i16)
  }
  pub fn from_alpha(u: i16, v: i16) -> CFLParams {
    CFLParams {
      sign: [CFLSign::from_alpha(u), CFLSign::from_alpha(v)],
      scale: [u.abs() as u8, v.abs() as u8],
    }
  }
}

#[derive(Debug, Default)]
struct FieldMap {
  map: Vec<(&'static str, usize, usize)>,
}

impl FieldMap {
  /// Print the field the address belong to
  fn lookup(&self, addr: usize) {
    for (name, start, end) in &self.map {
      if addr >= *start && addr < *end {
        println!(" CDF {}", name);
        println!();
        return;
      }
    }

    println!("  CDF address not found {}", addr);
  }
}

macro_rules! symbol_with_update {
  ($self:ident, $w:ident, $s:expr, $cdf:expr) => {
    $w.symbol_with_update($s, $cdf);
    #[cfg(feature = "desync_finder")]
    {
      let cdf: &[_] = $cdf;
      if let Some(map) = $self.fc_map.as_ref() {
        map.lookup(cdf.as_ptr() as usize);
      }
    }
  };
}

pub fn av1_get_coded_tx_size(tx_size: TxSize) -> TxSize {
  match tx_size {
    TX_64X64 | TX_64X32 | TX_32X64 => TX_32X32,
    TX_16X64 => TX_16X32,
    TX_64X16 => TX_32X16,
    _ => tx_size,
  }
}

#[derive(Clone)]
pub struct ContextWriterCheckpoint {
  pub fc: CDFContext,
  pub bc: BlockContextCheckpoint,
}

pub struct ContextWriter<'a> {
  pub bc: BlockContext<'a>,
  pub fc: &'a mut CDFContext,
  #[cfg(feature = "desync_finder")]
  fc_map: Option<FieldMap>, // For debugging purposes
}

impl<'a> ContextWriter<'a> {
  #[allow(clippy::let_and_return)]
  pub fn new(fc: &'a mut CDFContext, bc: BlockContext<'a>) -> Self {
    #[allow(unused_mut)]
    let mut cw = ContextWriter {
      fc,
      bc,
      #[cfg(feature = "desync_finder")]
      fc_map: Default::default(),
    };
    #[cfg(feature = "desync_finder")]
    {
      if std::env::var_os("RAV1E_DEBUG").is_some() {
        cw.fc_map = Some(FieldMap { map: cw.fc.build_map() });
      }
    }

    cw
  }

  fn cdf_element_prob(cdf: &[u16], element: usize) -> u16 {
    (if element > 0 { cdf[element - 1] } else { 32768 }) - cdf[element]
  }

  fn partition_gather_horz_alike(
    out: &mut [u16; 2], cdf_in: &[u16], _bsize: BlockSize,
  ) {
    out[0] = 32768;
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_HORZ as usize,
    );
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_SPLIT as usize,
    );
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_HORZ_A as usize,
    );
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_HORZ_B as usize,
    );
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_VERT_A as usize,
    );
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_HORZ_4 as usize,
    );
    out[0] = 32768 - out[0];
    out[1] = 0;
  }

  fn partition_gather_vert_alike(
    out: &mut [u16; 2], cdf_in: &[u16], _bsize: BlockSize,
  ) {
    out[0] = 32768;
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_VERT as usize,
    );
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_SPLIT as usize,
    );
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_HORZ_A as usize,
    );
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_VERT_A as usize,
    );
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_VERT_B as usize,
    );
    out[0] -= ContextWriter::cdf_element_prob(
      cdf_in,
      PartitionType::PARTITION_VERT_4 as usize,
    );
    out[0] = 32768 - out[0];
    out[1] = 0;
  }

  pub fn write_partition(
    &mut self, w: &mut impl Writer, bo: TileBlockOffset, p: PartitionType,
    bsize: BlockSize,
  ) {
    debug_assert!(bsize.is_sqr());
    assert!(bsize >= BlockSize::BLOCK_8X8);
    let hbs = bsize.width_mi() / 2;
    let has_cols = (bo.0.x + hbs) < self.bc.blocks.cols();
    let has_rows = (bo.0.y + hbs) < self.bc.blocks.rows();
    let ctx = self.bc.partition_plane_context(bo, bsize);
    assert!(ctx < PARTITION_CONTEXTS);
    let partition_cdf = if bsize <= BlockSize::BLOCK_8X8 {
      &mut self.fc.partition_cdf[ctx][..=PARTITION_TYPES]
    } else {
      &mut self.fc.partition_cdf[ctx]
    };

    if !has_rows && !has_cols {
      return;
    }

    if has_rows && has_cols {
      symbol_with_update!(self, w, p as u32, partition_cdf);
    } else if !has_rows && has_cols {
      assert!(
        p == PartitionType::PARTITION_SPLIT
          || p == PartitionType::PARTITION_HORZ
      );
      assert!(bsize > BlockSize::BLOCK_8X8);
      let mut cdf = [0u16; 2];
      ContextWriter::partition_gather_vert_alike(
        &mut cdf,
        partition_cdf,
        bsize,
      );
      w.symbol((p == PartitionType::PARTITION_SPLIT) as u32, &cdf);
    } else {
      assert!(
        p == PartitionType::PARTITION_SPLIT
          || p == PartitionType::PARTITION_VERT
      );
      assert!(bsize > BlockSize::BLOCK_8X8);
      let mut cdf = [0u16; 2];
      ContextWriter::partition_gather_horz_alike(
        &mut cdf,
        partition_cdf,
        bsize,
      );
      w.symbol((p == PartitionType::PARTITION_SPLIT) as u32, &cdf);
    }
  }

  fn get_tx_size_context(
    &self, bo: TileBlockOffset, bsize: BlockSize,
  ) -> usize {
    let max_tx_size = max_txsize_rect_lookup[bsize as usize];
    let max_tx_wide = max_tx_size.width();
    let max_tx_high = max_tx_size.height();
    let has_above = bo.0.y > 0;
    let has_left = bo.0.x > 0;
    let mut above = self.bc.above_tx_context[bo.0.x] >= max_tx_wide as u8;
    let mut left = self.bc.left_tx_context[bo.y_in_sb()] >= max_tx_high as u8;

    if has_above {
      let above_blk = self.bc.blocks.above_of(bo);
      if above_blk.is_inter() {
        above = (above_blk.n4_w << MI_SIZE_LOG2) >= max_tx_wide;
      };
    }
    if has_left {
      let left_blk = self.bc.blocks.left_of(bo);
      if left_blk.is_inter() {
        left = (left_blk.n4_h << MI_SIZE_LOG2) >= max_tx_high;
      };
    }
    if has_above && has_left {
      return above as usize + left as usize;
    };
    if has_above {
      return above as usize;
    };
    if has_left {
      return left as usize;
    };
    0
  }

  pub fn write_tx_size_intra(
    &mut self, w: &mut dyn Writer, bo: TileBlockOffset, bsize: BlockSize,
    tx_size: TxSize,
  ) {
    fn tx_size_to_depth(tx_size: TxSize, bsize: BlockSize) -> usize {
      let mut ctx_size = max_txsize_rect_lookup[bsize as usize];
      let mut depth: usize = 0;
      while tx_size != ctx_size {
        depth += 1;
        ctx_size = sub_tx_size_map[ctx_size as usize];
        debug_assert!(depth <= MAX_TX_DEPTH);
      }
      depth
    }
    fn bsize_to_max_depth(bsize: BlockSize) -> usize {
      let mut tx_size: TxSize = max_txsize_rect_lookup[bsize as usize];
      let mut depth = 0;
      while depth < MAX_TX_DEPTH && tx_size != TX_4X4 {
        depth += 1;
        tx_size = sub_tx_size_map[tx_size as usize];
        debug_assert!(depth <= MAX_TX_DEPTH);
      }
      depth
    }
    fn bsize_to_tx_size_cat(bsize: BlockSize) -> usize {
      let mut tx_size: TxSize = max_txsize_rect_lookup[bsize as usize];
      debug_assert!(tx_size != TX_4X4);
      let mut depth = 0;
      while tx_size != TX_4X4 {
        depth += 1;
        tx_size = sub_tx_size_map[tx_size as usize];
      }
      debug_assert!(depth <= MAX_TX_CATS);

      depth - 1
    }

    debug_assert!(!self.bc.blocks[bo].is_inter());
    debug_assert!(bsize > BlockSize::BLOCK_4X4);

    let tx_size_ctx = self.get_tx_size_context(bo, bsize);
    let depth = tx_size_to_depth(tx_size, bsize);

    let max_depths = bsize_to_max_depth(bsize);
    let tx_size_cat = bsize_to_tx_size_cat(bsize);

    debug_assert!(depth <= max_depths);
    debug_assert!(!tx_size.is_rect() || bsize.is_rect_tx_allowed());

    symbol_with_update!(
      self,
      w,
      depth as u32,
      &mut self.fc.tx_size_cdf[tx_size_cat][tx_size_ctx][..=max_depths + 1]
    );
  }

  // Based on https://aomediacodec.github.io/av1-spec/#cdf-selection-process
  // Used to decide the cdf (context) for txfm_split
  fn get_above_tx_width(
    &self, bo: TileBlockOffset, _bsize: BlockSize, _tx_size: TxSize,
    first_tx: bool,
  ) -> usize {
    let has_above = bo.0.y > 0;
    if first_tx {
      if !has_above {
        return 64;
      }
      let above_blk = self.bc.blocks.above_of(bo);
      if above_blk.skip && above_blk.is_inter() {
        return above_blk.bsize.width();
      }
    }
    self.bc.above_tx_context[bo.0.x] as usize
  }

  fn get_left_tx_height(
    &self, bo: TileBlockOffset, _bsize: BlockSize, _tx_size: TxSize,
    first_tx: bool,
  ) -> usize {
    let has_left = bo.0.x > 0;
    if first_tx {
      if !has_left {
        return 64;
      }
      let left_blk = self.bc.blocks.left_of(bo);
      if left_blk.skip && left_blk.is_inter() {
        return left_blk.bsize.height();
      }
    }
    self.bc.left_tx_context[bo.y_in_sb()] as usize
  }

  fn txfm_partition_context(
    &self, bo: TileBlockOffset, bsize: BlockSize, tx_size: TxSize, tbx: usize,
    tby: usize,
  ) -> usize {
    debug_assert!(tx_size > TX_4X4);
    debug_assert!(bsize > BlockSize::BLOCK_4X4);

    // TODO: from 2nd level partition, must know whether the tx block is the topmost(or leftmost) within a partition
    let above = (self.get_above_tx_width(bo, bsize, tx_size, tby == 0)
      < tx_size.width()) as usize;
    let left = (self.get_left_tx_height(bo, bsize, tx_size, tbx == 0)
      < tx_size.height()) as usize;

    let max_tx_size: TxSize = bsize.tx_size().sqr_up();
    let category: usize = (tx_size.sqr_up() != max_tx_size) as usize
      + (TxSize::TX_SIZES as usize - 1 - max_tx_size as usize) * 2;

    debug_assert!(category < TXFM_PARTITION_CONTEXTS);

    category * 3 + above + left
  }

  pub fn write_tx_size_inter(
    &mut self, w: &mut dyn Writer, bo: TileBlockOffset, bsize: BlockSize,
    tx_size: TxSize, txfm_split: bool, tbx: usize, tby: usize, depth: usize,
  ) {
    debug_assert!(self.bc.blocks[bo].is_inter());
    debug_assert!(bsize > BlockSize::BLOCK_4X4);
    debug_assert!(!tx_size.is_rect() || bsize.is_rect_tx_allowed());

    if tx_size != TX_4X4 && depth < MAX_VARTX_DEPTH {
      let ctx = self.txfm_partition_context(bo, bsize, tx_size, tbx, tby);

      symbol_with_update!(
        self,
        w,
        txfm_split as u32,
        &mut self.fc.txfm_partition_cdf[ctx]
      );
    } else {
      debug_assert!(!txfm_split);
    }

    if !txfm_split {
      self.bc.update_tx_size_context(bo, tx_size.block_size(), tx_size, false);
    } else {
      // if txfm_split == true, split one level only
      let split_tx_size = sub_tx_size_map[tx_size as usize];
      let bw = bsize.width_mi() / split_tx_size.width_mi();
      let bh = bsize.height_mi() / split_tx_size.height_mi();

      for by in 0..bh {
        for bx in 0..bw {
          let tx_bo = TileBlockOffset(BlockOffset {
            x: bo.0.x + bx * split_tx_size.width_mi(),
            y: bo.0.y + by * split_tx_size.height_mi(),
          });
          self.write_tx_size_inter(
            w,
            tx_bo,
            bsize,
            split_tx_size,
            false,
            bx,
            by,
            depth + 1,
          );
        }
      }
    }
  }

  pub fn get_cdf_intra_mode_kf(
    &self, bo: TileBlockOffset,
  ) -> &[u16; INTRA_MODES + 1] {
    static intra_mode_context: [usize; INTRA_MODES] =
      [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];
    let above_mode = if bo.0.y > 0 {
      self.bc.blocks.above_of(bo).mode
    } else {
      PredictionMode::DC_PRED
    };
    let left_mode = if bo.0.x > 0 {
      self.bc.blocks.left_of(bo).mode
    } else {
      PredictionMode::DC_PRED
    };
    let above_ctx = intra_mode_context[above_mode as usize];
    let left_ctx = intra_mode_context[left_mode as usize];
    &self.fc.kf_y_cdf[above_ctx][left_ctx]
  }

  pub fn write_intra_mode_kf(
    &mut self, w: &mut dyn Writer, bo: TileBlockOffset, mode: PredictionMode,
  ) {
    static intra_mode_context: [usize; INTRA_MODES] =
      [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];
    let above_mode = if bo.0.y > 0 {
      self.bc.blocks.above_of(bo).mode
    } else {
      PredictionMode::DC_PRED
    };
    let left_mode = if bo.0.x > 0 {
      self.bc.blocks.left_of(bo).mode
    } else {
      PredictionMode::DC_PRED
    };
    let above_ctx = intra_mode_context[above_mode as usize];
    let left_ctx = intra_mode_context[left_mode as usize];
    let cdf = &mut self.fc.kf_y_cdf[above_ctx][left_ctx];
    symbol_with_update!(self, w, mode as u32, cdf);
  }

  pub fn get_cdf_intra_mode(
    &self, bsize: BlockSize,
  ) -> &[u16; INTRA_MODES + 1] {
    &self.fc.y_mode_cdf[size_group_lookup[bsize as usize] as usize]
  }

  pub fn write_intra_mode(
    &mut self, w: &mut dyn Writer, bsize: BlockSize, mode: PredictionMode,
  ) {
    let cdf =
      &mut self.fc.y_mode_cdf[size_group_lookup[bsize as usize] as usize];
    symbol_with_update!(self, w, mode as u32, cdf);
  }

  pub fn write_intra_uv_mode(
    &mut self, w: &mut dyn Writer, uv_mode: PredictionMode,
    y_mode: PredictionMode, bs: BlockSize,
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
        symbol_with_update!(
          self,
          w,
          cfl.index(uv),
          &mut self.fc.cfl_alpha_cdf[cfl.context(uv)]
        );
      }
    }
  }

  pub fn write_angle_delta(
    &mut self, w: &mut dyn Writer, angle: i8, mode: PredictionMode,
  ) {
    symbol_with_update!(
      self,
      w,
      (angle + MAX_ANGLE_DELTA as i8) as u32,
      &mut self.fc.angle_delta_cdf
        [mode as usize - PredictionMode::V_PRED as usize]
    );
  }

  pub fn write_use_filter_intra(
    &mut self, w: &mut dyn Writer, enable: bool, block_size: BlockSize,
  ) {
    symbol_with_update!(
      self,
      w,
      enable as u32,
      &mut self.fc.filter_intra_cdfs[block_size as usize]
    );
  }

  pub fn write_use_palette_mode(
    &mut self, w: &mut dyn Writer, enable: bool, bsize: BlockSize,
    bo: TileBlockOffset, luma_mode: PredictionMode,
    chroma_mode: PredictionMode, xdec: usize, ydec: usize, cs: ChromaSampling,
  ) {
    if enable {
      unimplemented!(); // TODO
    }

    let (ctx_luma, ctx_chroma) = (0, 0); // TODO: increase based on surrounding block info

    if luma_mode == PredictionMode::DC_PRED {
      let bsize_ctx = bsize.width_mi_log2() + bsize.height_mi_log2() - 2;
      symbol_with_update!(
        self,
        w,
        enable as u32,
        &mut self.fc.palette_y_mode_cdfs[bsize_ctx][ctx_luma]
      );
    }

    if has_chroma(bo, bsize, xdec, ydec, cs)
      && chroma_mode == PredictionMode::DC_PRED
    {
      symbol_with_update!(
        self,
        w,
        enable as u32,
        &mut self.fc.palette_uv_mode_cdfs[ctx_chroma]
      );
    }
  }

  fn find_valid_row_offs(
    row_offset: isize, mi_row: usize, mi_rows: usize,
  ) -> isize {
    cmp::min(
      cmp::max(row_offset, -(mi_row as isize)),
      (mi_rows - mi_row - 1) as isize,
    )
  }

  fn find_valid_col_offs(
    col_offset: isize, mi_col: usize, mi_cols: usize,
  ) -> isize {
    cmp::min(
      cmp::max(col_offset, -(mi_col as isize)),
      (mi_cols - mi_col - 1) as isize,
    )
  }

  fn find_matching_mv(
    mv: MotionVector, mv_stack: &mut ArrayVec<[CandidateMV; 9]>,
  ) -> bool {
    for mv_cand in mv_stack {
      if mv.row == mv_cand.this_mv.row && mv.col == mv_cand.this_mv.col {
        return true;
      }
    }
    false
  }

  fn find_matching_mv_and_update_weight(
    mv: MotionVector, mv_stack: &mut ArrayVec<[CandidateMV; 9]>, weight: u32,
  ) -> bool {
    for mut mv_cand in mv_stack {
      if mv.row == mv_cand.this_mv.row && mv.col == mv_cand.this_mv.col {
        mv_cand.weight += weight;
        return true;
      }
    }
    false
  }

  fn find_matching_comp_mv_and_update_weight(
    mvs: [MotionVector; 2], mv_stack: &mut ArrayVec<[CandidateMV; 9]>,
    weight: u32,
  ) -> bool {
    for mv_cand in mv_stack {
      if mvs[0].row == mv_cand.this_mv.row
        && mvs[0].col == mv_cand.this_mv.col
        && mvs[1].row == mv_cand.comp_mv.row
        && mvs[1].col == mv_cand.comp_mv.col
      {
        mv_cand.weight += weight;
        return true;
      }
    }
    false
  }

  fn add_ref_mv_candidate(
    ref_frames: [RefType; 2], blk: &Block,
    mv_stack: &mut ArrayVec<[CandidateMV; 9]>, weight: u32,
    newmv_count: &mut usize, is_compound: bool,
  ) -> bool {
    if !blk.is_inter() {
      /* For intrabc */
      false
    } else if is_compound {
      if blk.ref_frames[0] == ref_frames[0]
        && blk.ref_frames[1] == ref_frames[1]
      {
        let found_match = Self::find_matching_comp_mv_and_update_weight(
          blk.mv, mv_stack, weight,
        );

        if !found_match && mv_stack.len() < MAX_REF_MV_STACK_SIZE {
          let mv_cand =
            CandidateMV { this_mv: blk.mv[0], comp_mv: blk.mv[1], weight };

          mv_stack.push(mv_cand);
        }

        if blk.mode == PredictionMode::NEW_NEWMV
          || blk.mode == PredictionMode::NEAREST_NEWMV
          || blk.mode == PredictionMode::NEW_NEARESTMV
          || blk.mode == PredictionMode::NEAR_NEWMV
          || blk.mode == PredictionMode::NEW_NEARMV
        {
          *newmv_count += 1;
        }

        true
      } else {
        false
      }
    } else {
      let mut found = false;
      for i in 0..2 {
        if blk.ref_frames[i] == ref_frames[0] {
          let found_match = Self::find_matching_mv_and_update_weight(
            blk.mv[i], mv_stack, weight,
          );

          if !found_match && mv_stack.len() < MAX_REF_MV_STACK_SIZE {
            let mv_cand = CandidateMV {
              this_mv: blk.mv[i],
              comp_mv: MotionVector::default(),
              weight,
            };

            mv_stack.push(mv_cand);
          }

          if blk.mode == PredictionMode::NEW_NEWMV
            || blk.mode == PredictionMode::NEAREST_NEWMV
            || blk.mode == PredictionMode::NEW_NEARESTMV
            || blk.mode == PredictionMode::NEAR_NEWMV
            || blk.mode == PredictionMode::NEW_NEARMV
            || blk.mode == PredictionMode::NEWMV
          {
            *newmv_count += 1;
          }

          found = true;
        }
      }
      found
    }
  }

  fn add_extra_mv_candidate<T: Pixel>(
    blk: &Block, ref_frames: [RefType; 2],
    mv_stack: &mut ArrayVec<[CandidateMV; 9]>, fi: &FrameInvariants<T>,
    is_compound: bool, ref_id_count: &mut [usize; 2],
    ref_id_mvs: &mut [[MotionVector; 2]; 2], ref_diff_count: &mut [usize; 2],
    ref_diff_mvs: &mut [[MotionVector; 2]; 2],
  ) {
    if is_compound {
      for cand_list in 0..2 {
        let cand_ref = blk.ref_frames[cand_list];
        if cand_ref != INTRA_FRAME && cand_ref != NONE_FRAME {
          for list in 0..2 {
            let mut cand_mv = blk.mv[cand_list];
            if cand_ref == ref_frames[list] && ref_id_count[list] < 2 {
              ref_id_mvs[list][ref_id_count[list]] = cand_mv;
              ref_id_count[list] += 1;
            } else if ref_diff_count[list] < 2 {
              if fi.ref_frame_sign_bias[cand_ref.to_index()]
                != fi.ref_frame_sign_bias[ref_frames[list].to_index()]
              {
                cand_mv.row = -cand_mv.row;
                cand_mv.col = -cand_mv.col;
              }
              ref_diff_mvs[list][ref_diff_count[list]] = cand_mv;
              ref_diff_count[list] += 1;
            }
          }
        }
      }
    } else {
      for cand_list in 0..2 {
        let cand_ref = blk.ref_frames[cand_list];
        if cand_ref != INTRA_FRAME && cand_ref != NONE_FRAME {
          let mut mv = blk.mv[cand_list];
          if fi.ref_frame_sign_bias[cand_ref.to_index()]
            != fi.ref_frame_sign_bias[ref_frames[0].to_index()]
          {
            mv.row = -mv.row;
            mv.col = -mv.col;
          }

          if !Self::find_matching_mv(mv, mv_stack) {
            let mv_cand = CandidateMV {
              this_mv: mv,
              comp_mv: MotionVector::default(),
              weight: 2,
            };
            mv_stack.push(mv_cand);
          }
        }
      }
    }
  }

  fn scan_row_mbmi(
    &self, bo: TileBlockOffset, row_offset: isize, max_row_offs: isize,
    processed_rows: &mut isize, ref_frames: [RefType; 2],
    mv_stack: &mut ArrayVec<[CandidateMV; 9]>, newmv_count: &mut usize,
    bsize: BlockSize, is_compound: bool,
  ) -> bool {
    let bc = &self.bc;
    let target_n4_w = bsize.width_mi();

    let end_mi = cmp::min(
      cmp::min(target_n4_w, bc.blocks.cols() - bo.0.x),
      BLOCK_64X64.width_mi(),
    );
    let n4_w_8 = BLOCK_8X8.width_mi();
    let n4_w_16 = BLOCK_16X16.width_mi();
    let mut col_offset = 0;

    if row_offset.abs() > 1 {
      col_offset = 1;
      if ((bo.0.x & 0x01) != 0) && (target_n4_w < n4_w_8) {
        col_offset -= 1;
      }
    }

    let use_step_16 = target_n4_w >= 16;

    let mut found_match = false;

    let mut i = 0;
    while i < end_mi {
      let cand =
        &bc.blocks[bo.with_offset(col_offset + i as isize, row_offset)];

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

      if Self::add_ref_mv_candidate(
        ref_frames,
        cand,
        mv_stack,
        len as u32 * weight,
        newmv_count,
        is_compound,
      ) {
        found_match = true;
      }

      i += len;
    }

    found_match
  }

  fn scan_col_mbmi(
    &self, bo: TileBlockOffset, col_offset: isize, max_col_offs: isize,
    processed_cols: &mut isize, ref_frames: [RefType; 2],
    mv_stack: &mut ArrayVec<[CandidateMV; 9]>, newmv_count: &mut usize,
    bsize: BlockSize, is_compound: bool,
  ) -> bool {
    let bc = &self.bc;

    let target_n4_h = bsize.height_mi();

    let end_mi = cmp::min(
      cmp::min(target_n4_h, bc.blocks.rows() - bo.0.y),
      BLOCK_64X64.height_mi(),
    );
    let n4_h_8 = BLOCK_8X8.height_mi();
    let n4_h_16 = BLOCK_16X16.height_mi();
    let mut row_offset = 0;

    if col_offset.abs() > 1 {
      row_offset = 1;
      if ((bo.0.y & 0x01) != 0) && (target_n4_h < n4_h_8) {
        row_offset -= 1;
      }
    }

    let use_step_16 = target_n4_h >= 16;

    let mut found_match = false;

    let mut i = 0;
    while i < end_mi {
      let cand =
        &bc.blocks[bo.with_offset(col_offset, row_offset + i as isize)];
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

      if Self::add_ref_mv_candidate(
        ref_frames,
        cand,
        mv_stack,
        len as u32 * weight,
        newmv_count,
        is_compound,
      ) {
        found_match = true;
      }

      i += len;
    }

    found_match
  }

  fn scan_blk_mbmi(
    &self, bo: TileBlockOffset, ref_frames: [RefType; 2],
    mv_stack: &mut ArrayVec<[CandidateMV; 9]>, newmv_count: &mut usize,
    is_compound: bool,
  ) -> bool {
    if bo.0.x >= self.bc.blocks.cols() || bo.0.y >= self.bc.blocks.rows() {
      return false;
    }

    let weight = 2 * BLOCK_8X8.width_mi() as u32;
    /* Always assume its within a tile, probably wrong */
    Self::add_ref_mv_candidate(
      ref_frames,
      &self.bc.blocks[bo],
      mv_stack,
      weight,
      newmv_count,
      is_compound,
    )
  }

  fn add_offset(mv_stack: &mut ArrayVec<[CandidateMV; 9]>) {
    for mut cand_mv in mv_stack {
      cand_mv.weight += REF_CAT_LEVEL;
    }
  }

  fn setup_mvref_list<T: Pixel>(
    &self, bo: TileBlockOffset, ref_frames: [RefType; 2],
    mv_stack: &mut ArrayVec<[CandidateMV; 9]>, bsize: BlockSize,
    fi: &FrameInvariants<T>, is_compound: bool,
  ) -> usize {
    let (_rf, _rf_num) = (INTRA_FRAME, 1);

    let target_n4_h = bsize.height_mi();
    let target_n4_w = bsize.width_mi();

    let mut max_row_offs = 0 as isize;
    let row_adj =
      (target_n4_h < BLOCK_8X8.height_mi()) && (bo.0.y & 0x01) != 0x0;

    let mut max_col_offs = 0 as isize;
    let col_adj =
      (target_n4_w < BLOCK_8X8.width_mi()) && (bo.0.x & 0x01) != 0x0;

    let mut processed_rows = 0 as isize;
    let mut processed_cols = 0 as isize;

    let up_avail = bo.0.y > 0;
    let left_avail = bo.0.x > 0;

    if up_avail {
      max_row_offs = -2 * MVREF_ROW_COLS as isize + row_adj as isize;

      // limit max offset for small blocks
      if target_n4_h < BLOCK_8X8.height_mi() {
        max_row_offs = -2 * 2 + row_adj as isize;
      }

      let rows = self.bc.blocks.rows();
      max_row_offs = Self::find_valid_row_offs(max_row_offs, bo.0.y, rows);
    }

    if left_avail {
      max_col_offs = -2 * MVREF_ROW_COLS as isize + col_adj as isize;

      // limit max offset for small blocks
      if target_n4_w < BLOCK_8X8.width_mi() {
        max_col_offs = -2 * 2 + col_adj as isize;
      }

      let cols = self.bc.blocks.cols();
      max_col_offs = Self::find_valid_col_offs(max_col_offs, bo.0.x, cols);
    }

    let mut row_match = false;
    let mut col_match = false;
    let mut newmv_count: usize = 0;

    if max_row_offs.abs() >= 1 {
      let found_match = self.scan_row_mbmi(
        bo,
        -1,
        max_row_offs,
        &mut processed_rows,
        ref_frames,
        mv_stack,
        &mut newmv_count,
        bsize,
        is_compound,
      );
      row_match |= found_match;
    }
    if max_col_offs.abs() >= 1 {
      let found_match = self.scan_col_mbmi(
        bo,
        -1,
        max_col_offs,
        &mut processed_cols,
        ref_frames,
        mv_stack,
        &mut newmv_count,
        bsize,
        is_compound,
      );
      col_match |= found_match;
    }
    if has_tr(bo, bsize) && bo.0.y > 0 {
      let found_match = self.scan_blk_mbmi(
        bo.with_offset(target_n4_w as isize, -1),
        ref_frames,
        mv_stack,
        &mut newmv_count,
        is_compound,
      );
      row_match |= found_match;
    }

    let nearest_match =
      if row_match { 1 } else { 0 } + if col_match { 1 } else { 0 };

    Self::add_offset(mv_stack);

    /* Scan the second outer area. */
    let mut far_newmv_count: usize = 0; // won't be used

    let found_match = bo.0.x > 0
      && bo.0.y > 0
      && self.scan_blk_mbmi(
        bo.with_offset(-1, -1),
        ref_frames,
        mv_stack,
        &mut far_newmv_count,
        is_compound,
      );
    row_match |= found_match;

    for idx in 2..=MVREF_ROW_COLS {
      let row_offset = -2 * idx as isize + 1 + row_adj as isize;
      let col_offset = -2 * idx as isize + 1 + col_adj as isize;

      if row_offset.abs() <= max_row_offs.abs()
        && row_offset.abs() > processed_rows
      {
        let found_match = self.scan_row_mbmi(
          bo,
          row_offset,
          max_row_offs,
          &mut processed_rows,
          ref_frames,
          mv_stack,
          &mut far_newmv_count,
          bsize,
          is_compound,
        );
        row_match |= found_match;
      }

      if col_offset.abs() <= max_col_offs.abs()
        && col_offset.abs() > processed_cols
      {
        let found_match = self.scan_col_mbmi(
          bo,
          col_offset,
          max_col_offs,
          &mut processed_cols,
          ref_frames,
          mv_stack,
          &mut far_newmv_count,
          bsize,
          is_compound,
        );
        col_match |= found_match;
      }
    }

    let total_match =
      if row_match { 1 } else { 0 } + if col_match { 1 } else { 0 };

    assert!(total_match >= nearest_match);

    // mode_context contains both newmv_context and refmv_context, where newmv_context
    // lies in the REF_MVOFFSET least significant bits
    let mode_context = match nearest_match {
      0 => cmp::min(total_match, 1) + (total_match << REFMV_OFFSET),
      1 => 3 - cmp::min(newmv_count, 1) + ((2 + total_match) << REFMV_OFFSET),
      _ => 5 - cmp::min(newmv_count, 1) + (5 << REFMV_OFFSET),
    };

    /* TODO: Find nearest match and assign nearest and near mvs */

    // 7.10.2.11 Sort MV stack according to weight
    mv_stack.sort_by(|a, b| b.weight.cmp(&a.weight));

    if mv_stack.len() < 2 {
      // 7.10.2.12 Extra search process

      let w4 = bsize.width_mi().min(16).min(self.bc.blocks.cols() - bo.0.x);
      let h4 = bsize.height_mi().min(16).min(self.bc.blocks.rows() - bo.0.y);
      let num4x4 = w4.min(h4);

      let passes =
        if up_avail { 0 } else { 1 }..if left_avail { 2 } else { 1 };

      let mut ref_id_count = [0 as usize; 2];
      let mut ref_diff_count = [0 as usize; 2];
      let mut ref_id_mvs = [[MotionVector::default(); 2]; 2];
      let mut ref_diff_mvs = [[MotionVector::default(); 2]; 2];

      for pass in passes {
        let mut idx = 0;
        while idx < num4x4 && mv_stack.len() < 2 {
          let rbo = if pass == 0 {
            bo.with_offset(idx as isize, -1)
          } else {
            bo.with_offset(-1, idx as isize)
          };

          let blk = &self.bc.blocks[rbo];
          Self::add_extra_mv_candidate(
            blk,
            ref_frames,
            mv_stack,
            fi,
            is_compound,
            &mut ref_id_count,
            &mut ref_id_mvs,
            &mut ref_diff_count,
            &mut ref_diff_mvs,
          );

          idx += if pass == 0 { blk.n4_w } else { blk.n4_h };
        }
      }

      if is_compound {
        let mut combined_mvs = [[MotionVector::default(); 2]; 2];

        for list in 0..2 {
          let mut comp_count = 0;
          for idx in 0..ref_id_count[list] {
            combined_mvs[comp_count][list] = ref_id_mvs[list][idx];
            comp_count += 1;
          }
          for idx in 0..ref_diff_count[list] {
            if comp_count < 2 {
              combined_mvs[comp_count][list] = ref_diff_mvs[list][idx];
              comp_count += 1;
            }
          }
        }

        if mv_stack.len() == 1 {
          let mv_cand = if combined_mvs[0][0].row == mv_stack[0].this_mv.row
            && combined_mvs[0][0].col == mv_stack[0].this_mv.col
            && combined_mvs[0][1].row == mv_stack[0].comp_mv.row
            && combined_mvs[0][1].col == mv_stack[0].comp_mv.col
          {
            CandidateMV {
              this_mv: combined_mvs[1][0],
              comp_mv: combined_mvs[1][1],
              weight: 2,
            }
          } else {
            CandidateMV {
              this_mv: combined_mvs[0][0],
              comp_mv: combined_mvs[0][1],
              weight: 2,
            }
          };
          mv_stack.push(mv_cand);
        } else {
          for idx in 0..2 {
            let mv_cand = CandidateMV {
              this_mv: combined_mvs[idx][0],
              comp_mv: combined_mvs[idx][1],
              weight: 2,
            };
            mv_stack.push(mv_cand);
          }
        }

        assert!(mv_stack.len() == 2);
      }
    }

    /* TODO: Handle single reference frame extension */

    let frame_bo = PlaneBlockOffset(BlockOffset {
      x: self.bc.blocks.x() + bo.0.x,
      y: self.bc.blocks.y() + bo.0.y,
    });
    // clamp mvs
    for mv in mv_stack {
      let blk_w = bsize.width();
      let blk_h = bsize.height();
      let border_w = 128 + blk_w as isize * 8;
      let border_h = 128 + blk_h as isize * 8;
      let mvx_min =
        -(frame_bo.0.x as isize) * (8 * MI_SIZE) as isize - border_w;
      let mvx_max = (self.bc.blocks.frame_cols()
        - frame_bo.0.x
        - blk_w / MI_SIZE) as isize
        * (8 * MI_SIZE) as isize
        + border_w;
      let mvy_min =
        -(frame_bo.0.y as isize) * (8 * MI_SIZE) as isize - border_h;
      let mvy_max = (self.bc.blocks.frame_rows()
        - frame_bo.0.y
        - blk_h / MI_SIZE) as isize
        * (8 * MI_SIZE) as isize
        + border_h;
      mv.this_mv.row =
        (mv.this_mv.row as isize).max(mvy_min).min(mvy_max) as i16;
      mv.this_mv.col =
        (mv.this_mv.col as isize).max(mvx_min).min(mvx_max) as i16;
      mv.comp_mv.row =
        (mv.comp_mv.row as isize).max(mvy_min).min(mvy_max) as i16;
      mv.comp_mv.col =
        (mv.comp_mv.col as isize).max(mvx_min).min(mvx_max) as i16;
    }

    mode_context
  }

  pub fn find_mvrefs<T: Pixel>(
    &self, bo: TileBlockOffset, ref_frames: [RefType; 2],
    mv_stack: &mut ArrayVec<[CandidateMV; 9]>, bsize: BlockSize,
    fi: &FrameInvariants<T>, is_compound: bool,
  ) -> usize {
    assert!(ref_frames[0] != NONE_FRAME);
    if ref_frames[0] != NONE_FRAME {
      // TODO: If ref_frames[0] != INTRA_FRAME, convert global mv to an mv;
      // otherwise, set the global mv ref to invalid.
    }

    if ref_frames[0] != INTRA_FRAME {
      /* TODO: Set zeromv ref to the converted global motion vector */
    } else {
      /* TODO: Set the zeromv ref to 0 */
      return 0;
    }

    self.setup_mvref_list(bo, ref_frames, mv_stack, bsize, fi, is_compound)
  }

  pub fn fill_neighbours_ref_counts(&mut self, bo: TileBlockOffset) {
    let mut ref_counts = [0; INTER_REFS_PER_FRAME];

    if bo.0.y > 0 {
      let above_b = self.bc.blocks.above_of(bo);
      if above_b.is_inter() {
        ref_counts[above_b.ref_frames[0].to_index()] += 1;
        if above_b.has_second_ref() {
          ref_counts[above_b.ref_frames[1].to_index()] += 1;
        }
      }
    }

    if bo.0.x > 0 {
      let left_b = self.bc.blocks.left_of(bo);
      if left_b.is_inter() {
        ref_counts[left_b.ref_frames[0].to_index()] += 1;
        if left_b.has_second_ref() {
          ref_counts[left_b.ref_frames[1].to_index()] += 1;
        }
      }
    }
    self.bc.blocks[bo].neighbors_ref_counts = ref_counts;
  }

  fn ref_count_ctx(counts0: usize, counts1: usize) -> usize {
    if counts0 < counts1 {
      0
    } else if counts0 == counts1 {
      1
    } else {
      2
    }
  }

  fn get_ref_frame_ctx_b0(&self, bo: TileBlockOffset) -> usize {
    let ref_counts = self.bc.blocks[bo].neighbors_ref_counts;

    let fwd_cnt = ref_counts[LAST_FRAME.to_index()]
      + ref_counts[LAST2_FRAME.to_index()]
      + ref_counts[LAST3_FRAME.to_index()]
      + ref_counts[GOLDEN_FRAME.to_index()];

    let bwd_cnt = ref_counts[BWDREF_FRAME.to_index()]
      + ref_counts[ALTREF2_FRAME.to_index()]
      + ref_counts[ALTREF_FRAME.to_index()];

    ContextWriter::ref_count_ctx(fwd_cnt, bwd_cnt)
  }

  fn get_pred_ctx_brfarf2_or_arf(&self, bo: TileBlockOffset) -> usize {
    let ref_counts = self.bc.blocks[bo].neighbors_ref_counts;

    let brfarf2_count = ref_counts[BWDREF_FRAME.to_index()]
      + ref_counts[ALTREF2_FRAME.to_index()];
    let arf_count = ref_counts[ALTREF_FRAME.to_index()];

    ContextWriter::ref_count_ctx(brfarf2_count, arf_count)
  }

  fn get_pred_ctx_ll2_or_l3gld(&self, bo: TileBlockOffset) -> usize {
    let ref_counts = self.bc.blocks[bo].neighbors_ref_counts;

    let l_l2_count =
      ref_counts[LAST_FRAME.to_index()] + ref_counts[LAST2_FRAME.to_index()];
    let l3_gold_count =
      ref_counts[LAST3_FRAME.to_index()] + ref_counts[GOLDEN_FRAME.to_index()];

    ContextWriter::ref_count_ctx(l_l2_count, l3_gold_count)
  }

  fn get_pred_ctx_last_or_last2(&self, bo: TileBlockOffset) -> usize {
    let ref_counts = self.bc.blocks[bo].neighbors_ref_counts;

    let l_count = ref_counts[LAST_FRAME.to_index()];
    let l2_count = ref_counts[LAST2_FRAME.to_index()];

    ContextWriter::ref_count_ctx(l_count, l2_count)
  }

  fn get_pred_ctx_last3_or_gold(&self, bo: TileBlockOffset) -> usize {
    let ref_counts = self.bc.blocks[bo].neighbors_ref_counts;

    let l3_count = ref_counts[LAST3_FRAME.to_index()];
    let gold_count = ref_counts[GOLDEN_FRAME.to_index()];

    ContextWriter::ref_count_ctx(l3_count, gold_count)
  }

  fn get_pred_ctx_brf_or_arf2(&self, bo: TileBlockOffset) -> usize {
    let ref_counts = self.bc.blocks[bo].neighbors_ref_counts;

    let brf_count = ref_counts[BWDREF_FRAME.to_index()];
    let arf2_count = ref_counts[ALTREF2_FRAME.to_index()];

    ContextWriter::ref_count_ctx(brf_count, arf2_count)
  }

  fn get_comp_mode_ctx(&self, bo: TileBlockOffset) -> usize {
    let avail_left = bo.0.x > 0;
    let avail_up = bo.0.y > 0;
    let (left0, left1) = if avail_left {
      let bo_left = bo.with_offset(-1, 0);
      let ref_frames = &self.bc.blocks[bo_left].ref_frames;
      (ref_frames[0], ref_frames[1])
    } else {
      (INTRA_FRAME, NONE_FRAME)
    };
    let (above0, above1) = if avail_up {
      let bo_up = bo.with_offset(0, -1);
      let ref_frames = &self.bc.blocks[bo_up].ref_frames;
      (ref_frames[0], ref_frames[1])
    } else {
      (INTRA_FRAME, NONE_FRAME)
    };
    let left_single = left1 == NONE_FRAME;
    let above_single = above1 == NONE_FRAME;
    let left_intra = left0 == INTRA_FRAME;
    let above_intra = above0 == INTRA_FRAME;
    let left_backward = left0.is_bwd_ref();
    let above_backward = above0.is_bwd_ref();

    if avail_left && avail_up {
      if above_single && left_single {
        (above_backward ^ left_backward) as usize
      } else if above_single {
        2 + (above_backward || above_intra) as usize
      } else if left_single {
        2 + (left_backward || left_intra) as usize
      } else {
        4
      }
    } else if avail_up {
      if above_single {
        above_backward as usize
      } else {
        3
      }
    } else if avail_left {
      if left_single {
        left_backward as usize
      } else {
        3
      }
    } else {
      1
    }
  }

  fn get_comp_ref_type_ctx(&self, bo: TileBlockOffset) -> usize {
    fn is_samedir_ref_pair(ref0: RefType, ref1: RefType) -> bool {
      (ref0.is_bwd_ref() && ref0 != NONE_FRAME)
        == (ref1.is_bwd_ref() && ref1 != NONE_FRAME)
    }

    let avail_left = bo.0.x > 0;
    let avail_up = bo.0.y > 0;
    let (left0, left1) = if avail_left {
      let bo_left = bo.with_offset(-1, 0);
      let ref_frames = &self.bc.blocks[bo_left].ref_frames;
      (ref_frames[0], ref_frames[1])
    } else {
      (INTRA_FRAME, NONE_FRAME)
    };
    let (above0, above1) = if avail_up {
      let bo_up = bo.with_offset(0, -1);
      let ref_frames = &self.bc.blocks[bo_up].ref_frames;
      (ref_frames[0], ref_frames[1])
    } else {
      (INTRA_FRAME, NONE_FRAME)
    };
    let left_single = left1 == NONE_FRAME;
    let above_single = above1 == NONE_FRAME;
    let left_intra = left0 == INTRA_FRAME;
    let above_intra = above0 == INTRA_FRAME;
    let above_comp_inter = avail_up && !above_intra && !above_single;
    let left_comp_inter = avail_left && !left_intra && !left_single;
    let above_uni_comp =
      above_comp_inter && is_samedir_ref_pair(above0, above1);
    let left_uni_comp = left_comp_inter && is_samedir_ref_pair(left0, left1);

    if avail_up && !above_intra && avail_left && !left_intra {
      let samedir = is_samedir_ref_pair(above0, left0) as usize;

      if !above_comp_inter && !left_comp_inter {
        1 + 2 * samedir
      } else if !above_comp_inter {
        if !left_uni_comp {
          1
        } else {
          3 + samedir
        }
      } else if !left_comp_inter {
        if !above_uni_comp {
          1
        } else {
          3 + samedir
        }
      } else if !above_uni_comp && !left_uni_comp {
        0
      } else if !above_uni_comp || !left_uni_comp {
        2
      } else {
        3 + ((above0 == BWDREF_FRAME) == (left0 == BWDREF_FRAME)) as usize
      }
    } else if avail_up && avail_left {
      if above_comp_inter {
        1 + 2 * above_uni_comp as usize
      } else if left_comp_inter {
        1 + 2 * left_uni_comp as usize
      } else {
        2
      }
    } else if above_comp_inter {
      4 * above_uni_comp as usize
    } else if left_comp_inter {
      4 * left_uni_comp as usize
    } else {
      2
    }
  }

  pub fn write_ref_frames<T: Pixel>(
    &mut self, w: &mut dyn Writer, fi: &FrameInvariants<T>,
    bo: TileBlockOffset,
  ) {
    let rf = self.bc.blocks[bo].ref_frames;
    let sz = self.bc.blocks[bo].n4_w.min(self.bc.blocks[bo].n4_h);

    /* TODO: Handle multiple references */
    let comp_mode = self.bc.blocks[bo].has_second_ref();

    if fi.reference_mode != ReferenceMode::SINGLE && sz >= 2 {
      let ctx = self.get_comp_mode_ctx(bo);
      symbol_with_update!(
        self,
        w,
        comp_mode as u32,
        &mut self.fc.comp_mode_cdf[ctx]
      );
    } else {
      assert!(!comp_mode);
    }

    if comp_mode {
      let comp_ref_type = 1 as u32; // bidir
      let ctx = self.get_comp_ref_type_ctx(bo);
      symbol_with_update!(
        self,
        w,
        comp_ref_type,
        &mut self.fc.comp_ref_type_cdf[ctx]
      );

      if comp_ref_type == 0 {
        unimplemented!();
      } else {
        let compref = rf[0] == GOLDEN_FRAME || rf[0] == LAST3_FRAME;
        let ctx = self.get_pred_ctx_ll2_or_l3gld(bo);
        symbol_with_update!(
          self,
          w,
          compref as u32,
          &mut self.fc.comp_ref_cdf[ctx][0]
        );
        if !compref {
          let compref_p1 = rf[0] == LAST2_FRAME;
          let ctx = self.get_pred_ctx_last_or_last2(bo);
          symbol_with_update!(
            self,
            w,
            compref_p1 as u32,
            &mut self.fc.comp_ref_cdf[ctx][1]
          );
        } else {
          let compref_p2 = rf[0] == GOLDEN_FRAME;
          let ctx = self.get_pred_ctx_last3_or_gold(bo);
          symbol_with_update!(
            self,
            w,
            compref_p2 as u32,
            &mut self.fc.comp_ref_cdf[ctx][2]
          );
        }
        let comp_bwdref = rf[1] == ALTREF_FRAME;
        let ctx = self.get_pred_ctx_brfarf2_or_arf(bo);
        symbol_with_update!(
          self,
          w,
          comp_bwdref as u32,
          &mut self.fc.comp_bwd_ref_cdf[ctx][0]
        );
        if !comp_bwdref {
          let comp_bwdref_p1 = rf[1] == ALTREF2_FRAME;
          let ctx = self.get_pred_ctx_brf_or_arf2(bo);
          symbol_with_update!(
            self,
            w,
            comp_bwdref_p1 as u32,
            &mut self.fc.comp_bwd_ref_cdf[ctx][1]
          );
        }
      }
    } else {
      let b0_ctx = self.get_ref_frame_ctx_b0(bo);
      let b0 = rf[0] != NONE_FRAME && rf[0].is_bwd_ref();

      symbol_with_update!(
        self,
        w,
        b0 as u32,
        &mut self.fc.single_ref_cdfs[b0_ctx][0]
      );
      if b0 {
        let b1_ctx = self.get_pred_ctx_brfarf2_or_arf(bo);
        let b1 = rf[0] == ALTREF_FRAME;

        symbol_with_update!(
          self,
          w,
          b1 as u32,
          &mut self.fc.single_ref_cdfs[b1_ctx][1]
        );
        if !b1 {
          let b5_ctx = self.get_pred_ctx_brf_or_arf2(bo);
          let b5 = rf[0] == ALTREF2_FRAME;

          symbol_with_update!(
            self,
            w,
            b5 as u32,
            &mut self.fc.single_ref_cdfs[b5_ctx][5]
          );
        }
      } else {
        let b2_ctx = self.get_pred_ctx_ll2_or_l3gld(bo);
        let b2 = rf[0] == LAST3_FRAME || rf[0] == GOLDEN_FRAME;

        symbol_with_update!(
          self,
          w,
          b2 as u32,
          &mut self.fc.single_ref_cdfs[b2_ctx][2]
        );
        if !b2 {
          let b3_ctx = self.get_pred_ctx_last_or_last2(bo);
          let b3 = rf[0] != LAST_FRAME;

          symbol_with_update!(
            self,
            w,
            b3 as u32,
            &mut self.fc.single_ref_cdfs[b3_ctx][3]
          );
        } else {
          let b4_ctx = self.get_pred_ctx_last3_or_gold(bo);
          let b4 = rf[0] != LAST3_FRAME;

          symbol_with_update!(
            self,
            w,
            b4 as u32,
            &mut self.fc.single_ref_cdfs[b4_ctx][4]
          );
        }
      }
    }
  }

  pub fn write_compound_mode(
    &mut self, w: &mut dyn Writer, mode: PredictionMode, ctx: usize,
  ) {
    let newmv_ctx = ctx & NEWMV_CTX_MASK;
    let refmv_ctx = (ctx >> REFMV_OFFSET) & REFMV_CTX_MASK;

    let ctx = if refmv_ctx < 2 {
      newmv_ctx.min(1)
    } else if refmv_ctx < 4 {
      (newmv_ctx + 1).min(4)
    } else {
      (newmv_ctx.max(1) + 3).min(7)
    };

    assert!(mode >= PredictionMode::NEAREST_NEARESTMV);
    let val = mode as u32 - PredictionMode::NEAREST_NEARESTMV as u32;
    symbol_with_update!(self, w, val, &mut self.fc.compound_mode_cdf[ctx]);
  }

  pub fn write_inter_mode(
    &mut self, w: &mut dyn Writer, mode: PredictionMode, ctx: usize,
  ) {
    let newmv_ctx = ctx & NEWMV_CTX_MASK;
    symbol_with_update!(
      self,
      w,
      (mode != PredictionMode::NEWMV) as u32,
      &mut self.fc.newmv_cdf[newmv_ctx]
    );
    if mode != PredictionMode::NEWMV {
      let zeromv_ctx = (ctx >> GLOBALMV_OFFSET) & GLOBALMV_CTX_MASK;
      symbol_with_update!(
        self,
        w,
        (mode != PredictionMode::GLOBALMV) as u32,
        &mut self.fc.zeromv_cdf[zeromv_ctx]
      );
      if mode != PredictionMode::GLOBALMV {
        let refmv_ctx = (ctx >> REFMV_OFFSET) & REFMV_CTX_MASK;
        symbol_with_update!(
          self,
          w,
          (mode != PredictionMode::NEARESTMV) as u32,
          &mut self.fc.refmv_cdf[refmv_ctx]
        );
      }
    }
  }

  pub fn write_drl_mode(
    &mut self, w: &mut dyn Writer, drl_mode: bool, ctx: usize,
  ) {
    symbol_with_update!(self, w, drl_mode as u32, &mut self.fc.drl_cdfs[ctx]);
  }

  pub fn write_mv(
    &mut self, w: &mut dyn Writer, mv: MotionVector, ref_mv: MotionVector,
    mv_precision: MvSubpelPrecision,
  ) {
    let diff =
      MotionVector { row: mv.row - ref_mv.row, col: mv.col - ref_mv.col };
    let j: MvJointType = av1_get_mv_joint(diff);

    w.symbol_with_update(j as u32, &mut self.fc.nmv_context.joints_cdf);

    if mv_joint_vertical(j) {
      encode_mv_component(
        w,
        diff.row as i32,
        &mut self.fc.nmv_context.comps[0],
        mv_precision,
      );
    }
    if mv_joint_horizontal(j) {
      encode_mv_component(
        w,
        diff.col as i32,
        &mut self.fc.nmv_context.comps[1],
        mv_precision,
      );
    }
  }

  pub fn write_tx_type(
    &mut self, w: &mut dyn Writer, tx_size: TxSize, tx_type: TxType,
    y_mode: PredictionMode, is_inter: bool, use_reduced_tx_set: bool,
  ) {
    let square_tx_size = tx_size.sqr();
    let tx_set = get_tx_set(tx_size, is_inter, use_reduced_tx_set);
    let num_tx_types = num_tx_set[tx_set as usize];

    if num_tx_types > 1 {
      let tx_set_index =
        get_tx_set_index(tx_size, is_inter, use_reduced_tx_set);
      assert!(tx_set_index > 0);
      assert!(av1_tx_used[tx_set as usize][tx_type as usize] != 0);

      if is_inter {
        symbol_with_update!(
          self,
          w,
          av1_tx_ind[tx_set as usize][tx_type as usize] as u32,
          &mut self.fc.inter_tx_cdf[tx_set_index as usize]
            [square_tx_size as usize][..=num_tx_set[tx_set as usize]]
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
            [..=num_tx_set[tx_set as usize]]
        );
      }
    }
  }
  pub fn write_skip(
    &mut self, w: &mut dyn Writer, bo: TileBlockOffset, skip: bool,
  ) {
    let ctx = self.bc.skip_context(bo);
    symbol_with_update!(self, w, skip as u32, &mut self.fc.skip_cdfs[ctx]);
  }

  fn get_segment_pred(&self, bo: TileBlockOffset) -> (u8, u8) {
    let mut prev_ul = -1;
    let mut prev_u = -1;
    let mut prev_l = -1;
    if bo.0.x > 0 && bo.0.y > 0 {
      prev_ul = self.bc.blocks.above_left_of(bo).segmentation_idx as i8;
    }
    if bo.0.y > 0 {
      prev_u = self.bc.blocks.above_of(bo).segmentation_idx as i8;
    }
    if bo.0.x > 0 {
      prev_l = self.bc.blocks.left_of(bo).segmentation_idx as i8;
    }

    /* Pick CDF index based on number of matching/out-of-bounds segment IDs. */
    let cdf_index: u8;
    if prev_ul < 0 || prev_u < 0 || prev_l < 0 {
      /* Edge case */
      cdf_index = 0;
    } else if (prev_ul == prev_u) && (prev_ul == prev_l) {
      cdf_index = 2;
    } else if (prev_ul == prev_u) || (prev_ul == prev_l) || (prev_u == prev_l)
    {
      cdf_index = 1;
    } else {
      cdf_index = 0;
    }

    /* If 2 or more are identical returns that as predictor, otherwise prev_l. */
    let r: i8;
    if prev_u == -1 {
      /* edge case */
      r = if prev_l == -1 { 0 } else { prev_l };
    } else if prev_l == -1 {
      /* edge case */
      r = prev_u;
    } else {
      r = if prev_ul == prev_u { prev_u } else { prev_l };
    }
    (r as u8, cdf_index)
  }

  fn neg_interleave(x: i32, r: i32, max: i32) -> i32 {
    assert!(x < max);
    if r == 0 {
      return x;
    } else if r >= (max - 1) {
      return -x + max - 1;
    }
    let diff = x - r;
    if 2 * r < max {
      if diff.abs() <= r {
        if diff > 0 {
          return (diff << 1) - 1;
        } else {
          return (-diff) << 1;
        }
      }
      x
    } else {
      if diff.abs() < (max - r) {
        if diff > 0 {
          return (diff << 1) - 1;
        } else {
          return (-diff) << 1;
        }
      }
      (max - x) - 1
    }
  }

  pub fn write_segmentation(
    &mut self, w: &mut dyn Writer, bo: TileBlockOffset, bsize: BlockSize,
    skip: bool, last_active_segid: u8,
  ) {
    let (pred, cdf_index) = self.get_segment_pred(bo);
    if skip {
      self.bc.blocks.set_segmentation_idx(bo, bsize, pred);
      return;
    }
    let seg_idx = self.bc.blocks[bo].segmentation_idx;
    let coded_id = Self::neg_interleave(
      seg_idx as i32,
      pred as i32,
      (last_active_segid + 1) as i32,
    );
    symbol_with_update!(
      self,
      w,
      coded_id as u32,
      &mut self.fc.spatial_segmentation_cdfs[cdf_index as usize]
    );
  }

  // rather than test writing and rolling back the cdf, we just count Q8 bits using the current cdf
  pub fn count_lrf_switchable(
    &self, w: &dyn Writer, rs: &TileRestorationState,
    filter: RestorationFilter, pli: usize,
  ) -> u32 {
    let nsym = &self.fc.lrf_switchable_cdf.len() - 1;
    match filter {
      RestorationFilter::None => {
        w.symbol_bits(0, &self.fc.lrf_switchable_cdf[..nsym])
      }
      RestorationFilter::Wiener { .. } => {
        unreachable!() // for now, not permanently
      }
      RestorationFilter::Sgrproj { set, xqd } => {
        // Does *not* use 'RESTORE_SGRPROJ' but rather just '2'
        let rp = &rs.planes[pli];
        let mut bits = w.symbol_bits(2, &self.fc.lrf_switchable_cdf[..nsym])
          + ((SGRPROJ_PARAMS_BITS as u32) << OD_BITRES);
        for i in 0..2 {
          let s = SGRPROJ_PARAMS_S[set as usize][i];
          let min = SGRPROJ_XQD_MIN[i] as i32;
          let max = SGRPROJ_XQD_MAX[i] as i32;
          if s > 0 {
            bits += w.count_signed_subexp_with_ref(
              xqd[i] as i32,
              min,
              max + 1,
              SGRPROJ_PRJ_SUBEXP_K,
              rp.sgrproj_ref[i] as i32,
            );
          }
        }
        bits
      }
    }
  }

  pub fn write_lrf<T: Pixel>(
    &mut self, w: &mut dyn Writer, fi: &FrameInvariants<T>,
    rs: &mut TileRestorationStateMut, sbo: TileSuperBlockOffset, pli: usize,
  ) {
    if !fi.allow_intrabc {
      // TODO: also disallow if lossless
      let rp = &mut rs.planes[pli];
      if let Some(filter) = rp.restoration_unit(sbo, true).map(|ru| ru.filter)
      {
        match filter {
          RestorationFilter::None => match rp.rp_cfg.lrf_type {
            RESTORE_WIENER => {
              symbol_with_update!(self, w, 0, &mut self.fc.lrf_wiener_cdf);
            }
            RESTORE_SGRPROJ => {
              symbol_with_update!(self, w, 0, &mut self.fc.lrf_sgrproj_cdf);
            }
            RESTORE_SWITCHABLE => {
              symbol_with_update!(self, w, 0, &mut self.fc.lrf_switchable_cdf);
            }
            RESTORE_NONE => {}
            _ => unreachable!(),
          },
          RestorationFilter::Sgrproj { set, xqd } => {
            match rp.rp_cfg.lrf_type {
              RESTORE_SGRPROJ => {
                symbol_with_update!(self, w, 1, &mut self.fc.lrf_sgrproj_cdf);
              }
              RESTORE_SWITCHABLE => {
                // Does *not* write 'RESTORE_SGRPROJ'
                symbol_with_update!(
                  self,
                  w,
                  2,
                  &mut self.fc.lrf_switchable_cdf
                );
              }
              _ => unreachable!(),
            }
            w.literal(SGRPROJ_PARAMS_BITS, set as u32);
            for i in 0..2 {
              let s = SGRPROJ_PARAMS_S[set as usize][i];
              let min = SGRPROJ_XQD_MIN[i] as i32;
              let max = SGRPROJ_XQD_MAX[i] as i32;
              if s > 0 {
                w.write_signed_subexp_with_ref(
                  xqd[i] as i32,
                  min,
                  max + 1,
                  SGRPROJ_PRJ_SUBEXP_K,
                  rp.sgrproj_ref[i] as i32,
                );
                rp.sgrproj_ref[i] = xqd[i];
              } else {
                // Nothing written, just update the reference
                if i == 0 {
                  assert!(xqd[i] == 0);
                  rp.sgrproj_ref[0] = 0;
                } else {
                  rp.sgrproj_ref[1] = 95; // LOL at spec.  The result is always 95.
                }
              }
            }
          }
          RestorationFilter::Wiener { coeffs } => {
            match rp.rp_cfg.lrf_type {
              RESTORE_WIENER => {
                symbol_with_update!(self, w, 1, &mut self.fc.lrf_wiener_cdf);
              }
              RESTORE_SWITCHABLE => {
                // Does *not* write 'RESTORE_WIENER'
                symbol_with_update!(
                  self,
                  w,
                  1,
                  &mut self.fc.lrf_switchable_cdf
                );
              }
              _ => unreachable!(),
            }
            for pass in 0..2 {
              let first_coeff = if pli == 0 {
                0
              } else {
                assert!(coeffs[pass][0] == 0);
                1
              };
              for i in first_coeff..3 {
                let min = WIENER_TAPS_MIN[i] as i32;
                let max = WIENER_TAPS_MAX[i] as i32;
                w.write_signed_subexp_with_ref(
                  coeffs[pass][i] as i32,
                  min,
                  max + 1,
                  (i + 1) as u8,
                  rp.wiener_ref[pass][i] as i32,
                );
                rp.wiener_ref[pass][i] = coeffs[pass][i];
              }
            }
          }
        }
      }
    }
  }

  pub fn write_cdef(
    &mut self, w: &mut dyn Writer, strength_index: u8, bits: u8,
  ) {
    w.literal(bits, strength_index as u32);
  }

  pub fn write_block_deblock_deltas(
    &mut self, w: &mut dyn Writer, bo: TileBlockOffset, multi: bool,
    planes: usize,
  ) {
    fn write_block_delta(w: &mut dyn Writer, cdf: &mut [u16], delta: i8) {
      let abs = delta.abs() as u32;

      w.symbol_with_update(cmp::min(abs, DELTA_LF_SMALL), cdf);

      if abs >= DELTA_LF_SMALL {
        let bits = msb(abs as i32 - 1) as u32;
        w.literal(3, bits - 1);
        w.literal(bits as u8, abs - (1 << bits) - 1);
      }
      if abs > 0 {
        w.bool(delta < 0, 16384);
      }
    }

    let block = &self.bc.blocks[bo];
    if multi {
      let deltas_count = FRAME_LF_COUNT + planes - 3;
      let deltas = &block.deblock_deltas[..deltas_count];
      let cdfs = &mut self.fc.deblock_delta_multi_cdf[..deltas_count];

      for (&delta, cdf) in deltas.iter().zip(cdfs.iter_mut()) {
        write_block_delta(w, cdf, delta);
      }
    } else {
      let delta = block.deblock_deltas[0];
      let cdf = &mut self.fc.deblock_delta_cdf;
      write_block_delta(w, cdf, delta);
    }
  }

  pub fn write_is_inter(
    &mut self, w: &mut dyn Writer, bo: TileBlockOffset, is_inter: bool,
  ) {
    let ctx = self.bc.intra_inter_context(bo);
    symbol_with_update!(
      self,
      w,
      is_inter as u32,
      &mut self.fc.intra_inter_cdfs[ctx]
    );
  }

  fn get_txsize_entropy_ctx(tx_size: TxSize) -> usize {
    (tx_size.sqr() as usize + tx_size.sqr_up() as usize + 1) >> 1
  }

  fn txb_init_levels<T: Coefficient>(
    &self, coeffs: &[T], height: usize, levels: &mut [u8],
    levels_stride: usize,
  ) {
    // Coefficients and levels are transposed from how they work in the spec
    for (coeffs_col, levels_col) in
      coeffs.chunks(height).zip(levels.chunks_mut(levels_stride))
    {
      for (coeff, level) in coeffs_col.iter().zip(levels_col.iter_mut()) {
        *level = clamp(coeff.abs(), T::cast_from(0), T::cast_from(127)).as_();
      }
    }
  }

  // Since the coefficients and levels are transposed in relation to how they
  // work in the spec, use the log of block height in our calculations instead
  // of block width.
  fn get_txb_bhl(tx_size: TxSize) -> usize {
    av1_get_coded_tx_size(tx_size).height_log2()
  }

  fn get_eob_pos_token(eob: usize, extra: &mut u32) -> u32 {
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

  fn get_nz_mag(levels: &[u8], bhl: usize, tx_class: TxClass) -> usize {
    // Levels are transposed from how they work in the spec

    // May version.
    // Note: AOMMIN(level, 3) is useless for decoder since level < 3.
    let mut mag = cmp::min(3, levels[1]); // { 1, 0 }
    mag += cmp::min(3, levels[(1 << bhl) + TX_PAD_HOR]); // { 0, 1 }

    if tx_class == TX_CLASS_2D {
      mag += cmp::min(3, levels[(1 << bhl) + TX_PAD_HOR + 1]); // { 1, 1 }
      mag += cmp::min(3, levels[2]); // { 2, 0 }
      mag += cmp::min(3, levels[(2 << bhl) + (2 << TX_PAD_HOR_LOG2)]); // { 0, 2 }
    } else if tx_class == TX_CLASS_VERT {
      mag += cmp::min(3, levels[2]); // { 2, 0 }
      mag += cmp::min(3, levels[3]); // { 3, 0 }
      mag += cmp::min(3, levels[4]); // { 4, 0 }
    } else {
      mag += cmp::min(3, levels[(2 << bhl) + (2 << TX_PAD_HOR_LOG2)]); // { 0, 2 }
      mag += cmp::min(3, levels[(3 << bhl) + (3 << TX_PAD_HOR_LOG2)]); // { 0, 3 }
      mag += cmp::min(3, levels[(4 << bhl) + (4 << TX_PAD_HOR_LOG2)]); // { 0, 4 }
    }

    mag as usize
  }

  fn get_nz_map_ctx_from_stats(
    stats: usize,
    coeff_idx: usize, // raster order
    bhl: usize,
    tx_size: TxSize,
    tx_class: TxClass,
  ) -> usize {
    if (tx_class as u32 | coeff_idx as u32) == 0 {
      return 0;
    };

    // Coefficients are transposed from how they work in the spec
    let col: usize = coeff_idx >> bhl;
    let row: usize = coeff_idx - (col << bhl);

    let ctx = ((stats + 1) >> 1).min(4);

    ctx
      + match tx_class {
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
          av1_nz_map_ctx_offset[tx_size as usize][cmp::min(row, 4)]
            [cmp::min(col, 4)] as usize
        }
        TX_CLASS_HORIZ => nz_map_ctx_offset_1d[col],
        TX_CLASS_VERT => nz_map_ctx_offset_1d[row],
      }
  }

  fn get_nz_map_ctx(
    levels: &[u8], coeff_idx: usize, bhl: usize, area: usize, scan_idx: usize,
    is_eob: bool, tx_size: TxSize, tx_class: TxClass,
  ) -> usize {
    if is_eob {
      if scan_idx == 0 {
        return 0;
      }
      if scan_idx <= area / 8 {
        return 1;
      }
      if scan_idx <= area / 4 {
        return 2;
      }
      return 3;
    }

    // Levels are transposed from how they work in the spec
    let padded_idx = coeff_idx + ((coeff_idx >> bhl) << TX_PAD_HOR_LOG2);
    let stats = Self::get_nz_mag(&levels[padded_idx..], bhl, tx_class);

    Self::get_nz_map_ctx_from_stats(stats, coeff_idx, bhl, tx_size, tx_class)
  }

  fn get_nz_map_contexts(
    &self, levels: &mut [u8], scan: &[u16], eob: u16, tx_size: TxSize,
    tx_class: TxClass, coeff_contexts: &mut [i8],
  ) {
    let bhl = Self::get_txb_bhl(tx_size);
    let area = av1_get_coded_tx_size(tx_size).area();
    for i in 0..eob {
      let pos = scan[i as usize];
      coeff_contexts[pos as usize] = Self::get_nz_map_ctx(
        levels,
        pos as usize,
        bhl,
        area,
        i as usize,
        i == eob - 1,
        tx_size,
        tx_class,
      ) as i8;
    }
  }

  fn get_br_ctx(
    levels: &[u8],
    coeff_idx: usize, // raster order
    bhl: usize,
    tx_class: TxClass,
  ) -> usize {
    // Coefficients and levels are transposed from how they work in the spec
    let col: usize = coeff_idx >> bhl;
    let row: usize = coeff_idx - (col << bhl);
    let stride: usize = (1 << bhl) + TX_PAD_HOR;
    let pos: usize = col * stride + row;
    let mut mag: usize = (levels[pos + 1] + levels[pos + stride]) as usize;

    match tx_class {
      TX_CLASS_2D => {
        mag += levels[pos + stride + 1] as usize;
        mag = cmp::min((mag + 1) >> 1, 6);
        if coeff_idx == 0 {
          return mag;
        }
        if (row < 2) && (col < 2) {
          return mag + 7;
        }
      }
      TX_CLASS_HORIZ => {
        mag += levels[pos + (stride << 1)] as usize;
        mag = cmp::min((mag + 1) >> 1, 6);
        if coeff_idx == 0 {
          return mag;
        }
        if col == 0 {
          return mag + 7;
        }
      }
      TX_CLASS_VERT => {
        mag += levels[pos + 2] as usize;
        mag = cmp::min((mag + 1) >> 1, 6);
        if coeff_idx == 0 {
          return mag;
        }
        if row == 0 {
          return mag + 7;
        }
      }
    }

    mag + 14
  }

  pub fn write_coeffs_lv_map<T: Coefficient>(
    &mut self, w: &mut dyn Writer, plane: usize, bo: TileBlockOffset,
    coeffs_in: &[T], eob: usize, pred_mode: PredictionMode, tx_size: TxSize,
    tx_type: TxType, plane_bsize: BlockSize, xdec: usize, ydec: usize,
    use_reduced_tx_set: bool,
  ) -> bool {
    let is_inter = pred_mode >= PredictionMode::NEARESTMV;
    //assert!(!is_inter);
    // Note: Both intra and inter mode uses inter scan order. Surprised?
    let scan: &[u16] =
      &av1_scan_orders[tx_size as usize][tx_type as usize].scan[..eob];
    let height = av1_get_coded_tx_size(tx_size).height();

    // Create a slice with coeffs in scan order
    let mut coeffs_storage: Aligned<ArrayVec<[T; 32 * 32]>> =
      Aligned::new(ArrayVec::new());
    let coeffs = &mut coeffs_storage.data;
    coeffs.extend(scan.iter().map(|&scan_idx| coeffs_in[scan_idx as usize]));

    let mut cul_level = coeffs.iter().map(|c| u32::cast_from(c.abs())).sum();

    let txs_ctx = Self::get_txsize_entropy_ctx(tx_size);
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

    let mut levels_buf = [0u8; TX_PAD_2D];
    let levels: &mut [u8] =
      &mut levels_buf[TX_PAD_TOP * (height + TX_PAD_HOR)..];

    self.txb_init_levels(coeffs_in, height, levels, height + TX_PAD_HOR);

    let tx_class = tx_type_to_class[tx_type as usize];
    let plane_type = if plane == 0 { 0 } else { 1 } as usize;

    // Signal tx_type for luma plane only
    if plane == 0 {
      self.write_tx_type(
        w,
        tx_size,
        tx_type,
        pred_mode,
        is_inter,
        use_reduced_tx_set,
      );
    }

    // Encode EOB
    let mut eob_extra = 0 as u32;
    let eob_pt = Self::get_eob_pos_token(eob, &mut eob_extra);
    let eob_multi_size: usize = tx_size.area_log2() - 4;
    let eob_multi_ctx: usize = if tx_class == TX_CLASS_2D { 0 } else { 1 };

    symbol_with_update!(
      self,
      w,
      eob_pt - 1,
      match eob_multi_size {
        0 => &mut self.fc.eob_flag_cdf16[plane_type][eob_multi_ctx],
        1 => &mut self.fc.eob_flag_cdf32[plane_type][eob_multi_ctx],
        2 => &mut self.fc.eob_flag_cdf64[plane_type][eob_multi_ctx],
        3 => &mut self.fc.eob_flag_cdf128[plane_type][eob_multi_ctx],
        4 => &mut self.fc.eob_flag_cdf256[plane_type][eob_multi_ctx],
        5 => &mut self.fc.eob_flag_cdf512[plane_type][eob_multi_ctx],
        _ => &mut self.fc.eob_flag_cdf1024[plane_type][eob_multi_ctx],
      }
    );

    let eob_offset_bits = k_eob_offset_bits[eob_pt as usize];

    if eob_offset_bits > 0 {
      let mut eob_shift = eob_offset_bits - 1;
      let mut bit: u32 =
        if (eob_extra & (1 << eob_shift)) != 0 { 1 } else { 0 };
      symbol_with_update!(
        self,
        w,
        bit,
        &mut self.fc.eob_extra_cdf[txs_ctx][plane_type][(eob_pt - 3) as usize]
      );
      for i in 1..eob_offset_bits {
        eob_shift = eob_offset_bits as u16 - 1 - i as u16;
        bit = if (eob_extra & (1 << eob_shift)) != 0 { 1 } else { 0 };
        w.bit(bit as u16);
      }
    }

    let mut coeff_contexts: Aligned<[i8; MAX_CODED_TX_SQUARE]> =
      Aligned::uninitialized();

    self.get_nz_map_contexts(
      levels,
      scan,
      eob as u16,
      tx_size,
      tx_class,
      &mut coeff_contexts.data,
    );

    let bhl = Self::get_txb_bhl(tx_size);

    for (c, (&pos, &v)) in scan.iter().zip(coeffs.iter()).enumerate().rev() {
      let pos = pos as usize;
      let coeff_ctx = coeff_contexts.data[pos];
      let level = v.abs();

      if c == eob - 1 {
        symbol_with_update!(
          self,
          w,
          (cmp::min(u32::cast_from(level), 3) - 1) as u32,
          &mut self.fc.coeff_base_eob_cdf[txs_ctx][plane_type]
            [coeff_ctx as usize]
        );
      } else {
        symbol_with_update!(
          self,
          w,
          (cmp::min(u32::cast_from(level), 3)) as u32,
          &mut self.fc.coeff_base_cdf[txs_ctx][plane_type][coeff_ctx as usize]
        );
      }

      if level > T::cast_from(NUM_BASE_LEVELS) {
        let base_range = level - T::cast_from(1 + NUM_BASE_LEVELS);
        let br_ctx = Self::get_br_ctx(levels, pos, bhl, tx_class);
        let mut idx: T = T::cast_from(0);

        loop {
          if idx >= T::cast_from(COEFF_BASE_RANGE) {
            break;
          }
          let k = cmp::min(base_range - idx, T::cast_from(BR_CDF_SIZE - 1));
          symbol_with_update!(
            self,
            w,
            u32::cast_from(k),
            &mut self.fc.coeff_br_cdf
              [cmp::min(txs_ctx, TxSize::TX_32X32 as usize)][plane_type]
              [br_ctx]
          );
          if k < T::cast_from(BR_CDF_SIZE - 1) {
            break;
          }
          idx += T::cast_from(BR_CDF_SIZE - 1);
        }
      }
    }

    // Loop to code all signs in the transform block,
    // starting with the sign of DC (if applicable)
    for (c, &v) in coeffs.iter().enumerate() {
      if v == T::cast_from(0) {
        continue;
      }

      let level = v.abs();
      let sign = if v < T::cast_from(0) { 1 } else { 0 };
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
      if level > T::cast_from(COEFF_BASE_RANGE + NUM_BASE_LEVELS) {
        w.write_golomb(u32::cast_from(
          level - T::cast_from(COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1),
        ));
      }
    }

    cul_level = cmp::min(COEFF_CONTEXT_MASK as u32, cul_level);

    BlockContext::set_dc_sign(&mut cul_level, i32::cast_from(coeffs[0]));

    self.bc.set_coeff_context(plane, bo, tx_size, xdec, ydec, cul_level as u8);
    true
  }

  pub const fn checkpoint(&self) -> ContextWriterCheckpoint {
    ContextWriterCheckpoint { fc: *self.fc, bc: self.bc.checkpoint() }
  }

  pub fn rollback(&mut self, checkpoint: &ContextWriterCheckpoint) {
    *self.fc = checkpoint.fc;
    self.bc.rollback(&checkpoint.bc);
    #[cfg(feature = "desync_finder")]
    {
      if self.fc_map.is_some() {
        self.fc_map = Some(FieldMap { map: self.fc.build_map() });
      }
    }
  }
}

/* Symbols for coding magnitude class of nonzero components */
const MV_CLASSES: usize = 11;

// MV Class Types
const MV_CLASS_0: usize = 0; /* (0, 2]     integer pel */
const MV_CLASS_1: usize = 1; /* (2, 4]     integer pel */
const MV_CLASS_2: usize = 2; /* (4, 8]     integer pel */
const MV_CLASS_3: usize = 3; /* (8, 16]    integer pel */
const MV_CLASS_4: usize = 4; /* (16, 32]   integer pel */
const MV_CLASS_5: usize = 5; /* (32, 64]   integer pel */
const MV_CLASS_6: usize = 6; /* (64, 128]  integer pel */
const MV_CLASS_7: usize = 7; /* (128, 256] integer pel */
const MV_CLASS_8: usize = 8; /* (256, 512] integer pel */
const MV_CLASS_9: usize = 9; /* (512, 1024] integer pel */
const MV_CLASS_10: usize = 10; /* (1024,2048] integer pel */

const CLASS0_BITS: usize = 1; /* bits at integer precision for class 0 */
const CLASS0_SIZE: usize = 1 << CLASS0_BITS;
const MV_OFFSET_BITS: usize = MV_CLASSES + CLASS0_BITS - 2;
const MV_BITS_CONTEXTS: usize = 6;
const MV_FP_SIZE: usize = 4;

const MV_MAX_BITS: usize = MV_CLASSES + CLASS0_BITS + 2;
const MV_MAX: usize = (1 << MV_MAX_BITS) - 1;
const MV_VALS: usize = (MV_MAX << 1) + 1;

const MV_IN_USE_BITS: usize = 14;
const MV_UPP: i32 = 1 << MV_IN_USE_BITS;
const MV_LOW: i32 = -(1 << MV_IN_USE_BITS);

#[inline(always)]
pub fn av1_get_mv_joint(mv: MotionVector) -> MvJointType {
  if mv.row == 0 {
    return if mv.col == 0 {
      MvJointType::MV_JOINT_ZERO
    } else {
      MvJointType::MV_JOINT_HNZVZ
    };
  }

  if mv.col == 0 {
    MvJointType::MV_JOINT_HZVNZ
  } else {
    MvJointType::MV_JOINT_HNZVNZ
  }
}
#[inline(always)]
pub fn mv_joint_vertical(joint_type: MvJointType) -> bool {
  joint_type == MvJointType::MV_JOINT_HZVNZ
    || joint_type == MvJointType::MV_JOINT_HNZVNZ
}
#[inline(always)]
pub fn mv_joint_horizontal(joint_type: MvJointType) -> bool {
  joint_type == MvJointType::MV_JOINT_HNZVZ
    || joint_type == MvJointType::MV_JOINT_HNZVNZ
}
#[inline(always)]
pub fn mv_class_base(mv_class: usize) -> u32 {
  if mv_class != MV_CLASS_0 {
    (CLASS0_SIZE << (mv_class as usize + 2)) as u32
  } else {
    0
  }
}
#[inline(always)]
// If n != 0, returns the floor of log base 2 of n. If n == 0, returns 0.
pub fn log_in_base_2(n: u32) -> u8 {
  31 - cmp::min(31, n.leading_zeros() as u8)
}
#[inline(always)]
pub fn get_mv_class(z: u32, offset: &mut u32) -> usize {
  let c = if z >= CLASS0_SIZE as u32 * 4096 {
    MV_CLASS_10
  } else {
    log_in_base_2(z >> 3) as usize
  };

  *offset = z - mv_class_base(c);
  c
}

pub fn encode_mv_component(
  w: &mut dyn Writer, comp: i32, mvcomp: &mut NMVComponent,
  precision: MvSubpelPrecision,
) {
  assert!(comp != 0);
  let mut offset: u32 = 0;
  let sign: u32 = if comp < 0 { 1 } else { 0 };
  let mag: u32 = if sign == 1 { -comp as u32 } else { comp as u32 };
  let mv_class = get_mv_class(mag - 1, &mut offset);
  let d = offset >> 3; // int mv data
  let fr = (offset >> 1) & 3; // fractional mv data
  let hp = offset & 1; // high precision mv data

  // Sign
  w.symbol_with_update(sign, &mut mvcomp.sign_cdf);

  // Class
  w.symbol_with_update(mv_class as u32, &mut mvcomp.classes_cdf);

  // Integer bits
  if mv_class == MV_CLASS_0 {
    w.symbol_with_update(d, &mut mvcomp.class0_cdf);
  } else {
    let n = mv_class + CLASS0_BITS - 1; // number of bits
    for i in 0..n {
      w.symbol_with_update((d >> i) & 1, &mut mvcomp.bits_cdf[i]);
    }
  }
  // Fractional bits
  if precision > MvSubpelPrecision::MV_SUBPEL_NONE {
    w.symbol_with_update(
      fr,
      if mv_class == MV_CLASS_0 {
        &mut mvcomp.class0_fp_cdf[d as usize]
      } else {
        &mut mvcomp.fp_cdf
      },
    );
  }

  // High precision bit
  if precision > MvSubpelPrecision::MV_SUBPEL_LOW_PRECISION {
    w.symbol_with_update(
      hp,
      if mv_class == MV_CLASS_0 {
        &mut mvcomp.class0_hp_cdf
      } else {
        &mut mvcomp.hp_cdf
      },
    );
  }
}
