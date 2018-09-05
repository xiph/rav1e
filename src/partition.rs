// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(non_camel_case_types)]
#![allow(dead_code)]

use self::BlockSize::*;
use self::TxSize::*;
use encoder::FrameInvariants;

pub const NONE_FRAME: isize = -1;
pub const INTRA_FRAME: usize = 0;
pub const LAST_FRAME: usize = 1;
pub const LAST2_FRAME: usize = 2;
pub const LAST3_FRAME: usize = 3;
pub const GOLDEN_FRAME: usize = 4;
pub const BWDREF_FRAME: usize = 5;
pub const ALTREF2_FRAME: usize = 6;
pub const ALTREF_FRAME: usize = 7;

pub const LAST_LAST2_FRAMES: usize = 0;      // { LAST_FRAME, LAST2_FRAME }
pub const LAST_LAST3_FRAMES: usize = 1;      // { LAST_FRAME, LAST3_FRAME }
pub const LAST_GOLDEN_FRAMES: usize = 2;     // { LAST_FRAME, GOLDEN_FRAME }
pub const BWDREF_ALTREF_FRAMES: usize = 3;   // { BWDREF_FRAME, ALTREF_FRAME }
pub const LAST2_LAST3_FRAMES: usize = 4;     // { LAST2_FRAME, LAST3_FRAME }
pub const LAST2_GOLDEN_FRAMES: usize = 5;    // { LAST2_FRAME, GOLDEN_FRAME }
pub const LAST3_GOLDEN_FRAMES: usize = 6;    // { LAST3_FRAME, GOLDEN_FRAME }
pub const BWDREF_ALTREF2_FRAMES: usize = 7;  // { BWDREF_FRAME, ALTREF2_FRAME }
pub const ALTREF2_ALTREF_FRAMES: usize = 8;  // { ALTREF2_FRAME, ALTREF_FRAME }
pub const TOTAL_UNIDIR_COMP_REFS: usize = 9;

// NOTE: UNIDIR_COMP_REFS is the number of uni-directional reference pairs
//       that are explicitly signaled.
pub const UNIDIR_COMP_REFS: usize = BWDREF_ALTREF_FRAMES + 1;

pub const FWD_REFS: usize = GOLDEN_FRAME - LAST_FRAME + 1;
pub const BWD_REFS: usize = ALTREF_FRAME - BWDREF_FRAME + 1;
pub const SINGLE_REFS: usize = FWD_REFS + BWD_REFS;
pub const TOTAL_REFS_PER_FRAME: usize = ALTREF_FRAME - INTRA_FRAME + 1;
pub const INTER_REFS_PER_FRAME: usize = ALTREF_FRAME - LAST_FRAME + 1;
pub const TOTAL_COMP_REFS: usize = FWD_REFS * BWD_REFS + TOTAL_UNIDIR_COMP_REFS;

pub const REF_FRAMES_LOG2: usize = 3;
pub const REF_FRAMES: usize = 1 << REF_FRAMES_LOG2;

pub const REF_CONTEXTS: usize = 3;
pub const MVREF_ROW_COLS: usize = 3;

#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub enum PartitionType {
  PARTITION_NONE,
  PARTITION_HORZ,
  PARTITION_VERT,
  PARTITION_SPLIT,
  PARTITION_HORZ_A,  // HORZ split and the top partition is split again
  PARTITION_HORZ_B,  // HORZ split and the bottom partition is split again
  PARTITION_VERT_A,  // VERT split and the left partition is split again
  PARTITION_VERT_B,  // VERT split and the right partition is split again
  PARTITION_HORZ_4,  // 4:1 horizontal partition
  PARTITION_VERT_4,  // 4:1 vertical partition
  PARTITION_INVALID
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Ord, Eq)]
pub enum BlockSize {
  BLOCK_4X4,
  BLOCK_4X8,
  BLOCK_8X4,
  BLOCK_8X8,
  BLOCK_8X16,
  BLOCK_16X8,
  BLOCK_16X16,
  BLOCK_16X32,
  BLOCK_32X16,
  BLOCK_32X32,
  BLOCK_32X64,
  BLOCK_64X32,
  BLOCK_64X64,
  BLOCK_64X128,
  BLOCK_128X64,
  BLOCK_128X128,
  BLOCK_4X16,
  BLOCK_16X4,
  BLOCK_8X32,
  BLOCK_32X8,
  BLOCK_16X64,
  BLOCK_64X16,
  BLOCK_32X128,
  BLOCK_128X32,
  BLOCK_INVALID
}

impl BlockSize {
  pub const BLOCK_SIZES_ALL: usize = 24;

  const BLOCK_SIZE_WIDTH_LOG2: [usize; BlockSize::BLOCK_SIZES_ALL] =
    [2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 2, 4, 3, 5, 4, 6, 5, 7];

  const BLOCK_SIZE_HEIGHT_LOG2: [usize; BlockSize::BLOCK_SIZES_ALL] =
    [2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 4, 2, 5, 3, 6, 4, 7, 5];

  pub const MI_SIZE_WIDE: [usize; BlockSize::BLOCK_SIZES_ALL] =
    [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16, 8, 32];

  pub const MI_SIZE_HIGH: [usize; BlockSize::BLOCK_SIZES_ALL] =
    [1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4, 32, 8];

  pub fn cfl_allowed(self) -> bool {
    // TODO: fix me when enabling EXT_PARTITION_TYPES
    self <= BlockSize::BLOCK_32X32
  }

  pub fn width(self) -> usize {
    1 << BlockSize::BLOCK_SIZE_WIDTH_LOG2[self as usize]
  }

  pub fn width_log2(self) -> usize {
    BlockSize::BLOCK_SIZE_WIDTH_LOG2[self as usize]
  }

  pub fn width_mi(self) -> usize {
    self.width() >> MI_SIZE_LOG2
  }

  pub fn height(self) -> usize {
    1 << BlockSize::BLOCK_SIZE_HEIGHT_LOG2[self as usize]
  }

  pub fn height_log2(self) -> usize {
    BlockSize::BLOCK_SIZE_HEIGHT_LOG2[self as usize]
  }

  pub fn height_mi(self) -> usize {
    self.height() >> MI_SIZE_LOG2
  }

  pub fn is_sqr(self) -> bool {
    self.width_log2() == self.height_log2()
  }

  pub fn is_sub8x8(self) -> bool {
    self.width_log2().min(self.height_log2()) < 3
  }
}

/// Transform Size
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
#[repr(C)]
pub enum TxSize {
  TX_4X4,
  TX_8X8,
  TX_16X16,
  TX_32X32,
  TX_64X64,
  TX_4X8,
  TX_8X4,
  TX_8X16,
  TX_16X8,
  TX_16X32,
  TX_32X16,
  TX_32X64,
  TX_64X32,
  TX_4X16,
  TX_16X4,
  TX_8X32,
  TX_32X8,
  TX_16X64,
  TX_64X16
}

impl TxSize {
  /// Number of square transform sizes
  pub const TX_SIZES: usize = 5;

  /// Number of transform sizes (including non-square sizes)
  pub const TX_SIZES_ALL: usize = 14 + 5;

  pub fn width(self) -> usize {
    1 << self.width_log2()
  }

  pub fn width_log2(self) -> usize {
    const TX_SIZE_WIDTH_LOG2: [usize; TxSize::TX_SIZES_ALL] =
      [2, 3, 4, 5, 6, 2, 3, 3, 4, 4, 5, 5, 6, 2, 4, 3, 5, 4, 6];
    TX_SIZE_WIDTH_LOG2[self as usize]
  }

  pub fn smallest_width_log2() -> usize {
    TX_4X4.width_log2()
  }

  pub fn height(self) -> usize {
    1 << self.height_log2()
  }

  pub fn height_log2(self) -> usize {
    const TX_SIZE_HEIGHT_LOG2: [usize; TxSize::TX_SIZES_ALL] =
      [2, 3, 4, 5, 6, 3, 2, 4, 3, 5, 4, 6, 5, 4, 2, 5, 3, 6, 4];
    TX_SIZE_HEIGHT_LOG2[self as usize]
  }

  pub fn width_mi(self) -> usize {
    self.width() >> MI_SIZE_LOG2
  }

  pub fn area(self) -> usize {
    1 << self.area_log2()
  }

  pub fn area_log2(self) -> usize {
    self.width_log2() + self.height_log2()
  }

  pub fn height_mi(self) -> usize {
    self.height() >> MI_SIZE_LOG2
  }

  pub fn block_size(self) -> BlockSize {
    const TX_SIZE_TO_BLOCK_SIZE: [BlockSize; TxSize::TX_SIZES_ALL] = [
      BLOCK_4X4,   // TX_4X4
      BLOCK_8X8,   // TX_8X8
      BLOCK_16X16, // TX_16X16
      BLOCK_32X32, // TX_32X32
      BLOCK_64X64,
      BLOCK_4X8,   // TX_4X8
      BLOCK_8X4,   // TX_8X4
      BLOCK_8X16,  // TX_8X16
      BLOCK_16X8,  // TX_16X8
      BLOCK_16X32, // TX_16X32
      BLOCK_32X16, // TX_32X16
      BLOCK_32X64,
      BLOCK_64X32,
      BLOCK_4X16, // TX_4X16
      BLOCK_16X4, // TX_16X4
      BLOCK_8X32, // TX_8X32
      BLOCK_32X8, // TX_32X8
      BLOCK_16X64,
      BLOCK_64X16
    ];
    TX_SIZE_TO_BLOCK_SIZE[self as usize]
  }

  pub fn sqr(self) -> TxSize {
    #[cfg_attr(rustfmt, rustfmt_skip)]
        const TX_SIZE_SQR: [TxSize; TxSize::TX_SIZES_ALL] = [
            TX_4X4,
            TX_8X8,
            TX_16X16,
            TX_32X32,
            TX_64X64,
            TX_4X4,
            TX_4X4,
            TX_8X8,
            TX_8X8,
            TX_16X16,
            TX_16X16,
            TX_32X32,
            TX_32X32,
            TX_4X4,
            TX_4X4,
            TX_8X8,
            TX_8X8,
            TX_16X16,
            TX_16X16
        ];
    TX_SIZE_SQR[self as usize]
  }

  pub fn sqr_up(self) -> TxSize {
    #[cfg_attr(rustfmt, rustfmt_skip)]
        const TX_SIZE_SQR_UP: [TxSize; TxSize::TX_SIZES_ALL] = [
            TX_4X4,
            TX_8X8,
            TX_16X16,
            TX_32X32,
            TX_64X64,
            TX_8X8,
            TX_8X8,
            TX_16X16,
            TX_16X16,
            TX_32X32,
            TX_32X32,
            TX_64X64,
            TX_64X64,
            TX_16X16,
            TX_16X16,
            TX_32X32,
            TX_32X32,
            TX_64X64,
            TX_64X64
        ];
    TX_SIZE_SQR_UP[self as usize]
  }
}

pub const TX_TYPES: usize = 16;

#[derive(Copy, Clone, PartialEq, PartialOrd)]
#[repr(C)]
pub enum TxType {
  DCT_DCT = 0,   // DCT  in both horizontal and vertical
  ADST_DCT = 1,  // ADST in vertical, DCT in horizontal
  DCT_ADST = 2,  // DCT  in vertical, ADST in horizontal
  ADST_ADST = 3, // ADST in both directions
  FLIPADST_DCT = 4,
  DCT_FLIPADST = 5,
  FLIPADST_FLIPADST = 6,
  ADST_FLIPADST = 7,
  FLIPADST_ADST = 8,
  IDTX = 9,
  V_DCT = 10,
  H_DCT = 11,
  V_ADST = 12,
  H_ADST = 13,
  V_FLIPADST = 14,
  H_FLIPADST = 15
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum PredictionMode {
  DC_PRED,     // Average of above and left pixels
  V_PRED,      // Vertical
  H_PRED,      // Horizontal
  D45_PRED,    // Directional 45  deg = round(arctan(1/1) * 180/pi)
  D135_PRED,   // Directional 135 deg = 180 - 45
  D117_PRED,   // Directional 117 deg = 180 - 63
  D153_PRED,   // Directional 153 deg = 180 - 27
  D207_PRED,   // Directional 207 deg = 180 + 27
  D63_PRED,    // Directional 63  deg = round(arctan(2/1) * 180/pi)
  SMOOTH_PRED, // Combination of horizontal and vertical interpolation
  SMOOTH_V_PRED,
  SMOOTH_H_PRED,
  PAETH_PRED,
  UV_CFL_PRED,
  NEARESTMV,
  NEARMV,
  GLOBALMV,
  NEWMV,
  // Compound ref compound modes
  NEAREST_NEARESTMV,
  NEAR_NEARMV,
  NEAREST_NEWMV,
  NEW_NEARESTMV,
  NEAR_NEWMV,
  NEW_NEARMV,
  GLOBAL_GLOBALMV,
  NEW_NEWMV
}

#[derive(Copy, Clone)]
pub struct MotionVector {
  pub row: i16,
  pub col: i16,
}

pub const NEWMV_MODE_CONTEXTS: usize = 7;
pub const GLOBALMV_MODE_CONTEXTS: usize = 2;
pub const REFMV_MODE_CONTEXTS: usize = 9;

pub const REFMV_OFFSET: usize = 4;
pub const GLOBALMV_OFFSET: usize = 3;
pub const NEWMV_CTX_MASK: usize = ((1 << GLOBALMV_OFFSET) - 1);
pub const GLOBALMV_CTX_MASK: usize = ((1 << (REFMV_OFFSET - GLOBALMV_OFFSET)) - 1);
pub const REFMV_CTX_MASK: usize = ((1 << (8 - REFMV_OFFSET)) - 1);

pub static RAV1E_PARTITION_TYPES: &'static [PartitionType] =
  &[PartitionType::PARTITION_NONE, PartitionType::PARTITION_SPLIT];

pub static RAV1E_TX_TYPES: &'static [TxType] = &[
  TxType::DCT_DCT,
  TxType::ADST_DCT,
  TxType::DCT_ADST,
  TxType::ADST_ADST,
  TxType::IDTX,
  TxType::V_DCT,
  TxType::H_DCT
];

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum GlobalMVMode {
  IDENTITY = 0,    // identity transformation, 0-parameter
  TRANSLATION = 1, // translational motion 2-parameter
  ROTZOOM = 2,     // simplified affine with rotation + zoom only, 4-parameter
  AFFINE = 3       // affine, 6-parameter
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum MvSubpelPrecision {
  MV_SUBPEL_NONE = -1,
  MV_SUBPEL_LOW_PRECISION = 0,
  MV_SUBPEL_HIGH_PRECISION,
}

const SUBPEL_FILTERS: [[[i32; 8]; 16]; 6] = [
  [
    [ 0, 0, 0, 128, 0, 0, 0, 0 ],      [ 0, 2, -6, 126, 8, -2, 0, 0 ],
    [ 0, 2, -10, 122, 18, -4, 0, 0 ],  [ 0, 2, -12, 116, 28, -8, 2, 0 ],
    [ 0, 2, -14, 110, 38, -10, 2, 0 ], [ 0, 2, -14, 102, 48, -12, 2, 0 ],
    [ 0, 2, -16, 94, 58, -12, 2, 0 ],  [ 0, 2, -14, 84, 66, -12, 2, 0 ],
    [ 0, 2, -14, 76, 76, -14, 2, 0 ],  [ 0, 2, -12, 66, 84, -14, 2, 0 ],
    [ 0, 2, -12, 58, 94, -16, 2, 0 ],  [ 0, 2, -12, 48, 102, -14, 2, 0 ],
    [ 0, 2, -10, 38, 110, -14, 2, 0 ], [ 0, 2, -8, 28, 116, -12, 2, 0 ],
    [ 0, 0, -4, 18, 122, -10, 2, 0 ],  [ 0, 0, -2, 8, 126, -6, 2, 0 ]
  ],
  [
    [ 0, 0, 0, 128, 0, 0, 0, 0 ],     [ 0, 2, 28, 62, 34, 2, 0, 0 ],
    [ 0, 0, 26, 62, 36, 4, 0, 0 ],    [ 0, 0, 22, 62, 40, 4, 0, 0 ],
    [ 0, 0, 20, 60, 42, 6, 0, 0 ],    [ 0, 0, 18, 58, 44, 8, 0, 0 ],
    [ 0, 0, 16, 56, 46, 10, 0, 0 ],   [ 0, -2, 16, 54, 48, 12, 0, 0 ],
    [ 0, -2, 14, 52, 52, 14, -2, 0 ], [ 0, 0, 12, 48, 54, 16, -2, 0 ],
    [ 0, 0, 10, 46, 56, 16, 0, 0 ],   [ 0, 0, 8, 44, 58, 18, 0, 0 ],
    [ 0, 0, 6, 42, 60, 20, 0, 0 ],    [ 0, 0, 4, 40, 62, 22, 0, 0 ],
    [ 0, 0, 4, 36, 62, 26, 0, 0 ],    [ 0, 0, 2, 34, 62, 28, 2, 0 ]
  ],
  [
    [ 0, 0, 0, 128, 0, 0, 0, 0 ],         [ -2, 2, -6, 126, 8, -2, 2, 0 ],
    [ -2, 6, -12, 124, 16, -6, 4, -2 ],   [ -2, 8, -18, 120, 26, -10, 6, -2 ],
    [ -4, 10, -22, 116, 38, -14, 6, -2 ], [ -4, 10, -22, 108, 48, -18, 8, -2 ],
    [ -4, 10, -24, 100, 60, -20, 8, -2 ], [ -4, 10, -24, 90, 70, -22, 10, -2 ],
    [ -4, 12, -24, 80, 80, -24, 12, -4 ], [ -2, 10, -22, 70, 90, -24, 10, -4 ],
    [ -2, 8, -20, 60, 100, -24, 10, -4 ], [ -2, 8, -18, 48, 108, -22, 10, -4 ],
    [ -2, 6, -14, 38, 116, -22, 10, -4 ], [ -2, 6, -10, 26, 120, -18, 8, -2 ],
    [ -2, 4, -6, 16, 124, -12, 6, -2 ],   [ 0, 2, -2, 8, 126, -6, 2, -2 ]
  ],
  [
    [ 0, 0, 0, 128, 0, 0, 0, 0 ],  [ 0, 0, 0, 120, 8, 0, 0, 0 ],
    [ 0, 0, 0, 112, 16, 0, 0, 0 ], [ 0, 0, 0, 104, 24, 0, 0, 0 ],
    [ 0, 0, 0, 96, 32, 0, 0, 0 ],  [ 0, 0, 0, 88, 40, 0, 0, 0 ],
    [ 0, 0, 0, 80, 48, 0, 0, 0 ],  [ 0, 0, 0, 72, 56, 0, 0, 0 ],
    [ 0, 0, 0, 64, 64, 0, 0, 0 ],  [ 0, 0, 0, 56, 72, 0, 0, 0 ],
    [ 0, 0, 0, 48, 80, 0, 0, 0 ],  [ 0, 0, 0, 40, 88, 0, 0, 0 ],
    [ 0, 0, 0, 32, 96, 0, 0, 0 ],  [ 0, 0, 0, 24, 104, 0, 0, 0 ],
    [ 0, 0, 0, 16, 112, 0, 0, 0 ], [ 0, 0, 0, 8, 120, 0, 0, 0 ]
  ],
  [
    [ 0, 0, 0, 128, 0, 0, 0, 0 ],     [ 0, 0, -4, 126, 8, -2, 0, 0 ],
    [ 0, 0, -8, 122, 18, -4, 0, 0 ],  [ 0, 0, -10, 116, 28, -6, 0, 0 ],
    [ 0, 0, -12, 110, 38, -8, 0, 0 ], [ 0, 0, -12, 102, 48, -10, 0, 0 ],
    [ 0, 0, -14, 94, 58, -10, 0, 0 ], [ 0, 0, -12, 84, 66, -10, 0, 0 ],
    [ 0, 0, -12, 76, 76, -12, 0, 0 ], [ 0, 0, -10, 66, 84, -12, 0, 0 ],
    [ 0, 0, -10, 58, 94, -14, 0, 0 ], [ 0, 0, -10, 48, 102, -12, 0, 0 ],
    [ 0, 0, -8, 38, 110, -12, 0, 0 ], [ 0, 0, -6, 28, 116, -10, 0, 0 ],
    [ 0, 0, -4, 18, 122, -8, 0, 0 ],  [ 0, 0, -2, 8, 126, -4, 0, 0 ]
  ],
  [
    [ 0, 0, 0, 128, 0, 0, 0, 0 ],   [ 0, 0, 30, 62, 34, 2, 0, 0 ],
    [ 0, 0, 26, 62, 36, 4, 0, 0 ],  [ 0, 0, 22, 62, 40, 4, 0, 0 ],
    [ 0, 0, 20, 60, 42, 6, 0, 0 ],  [ 0, 0, 18, 58, 44, 8, 0, 0 ],
    [ 0, 0, 16, 56, 46, 10, 0, 0 ], [ 0, 0, 14, 54, 48, 12, 0, 0 ],
    [ 0, 0, 12, 52, 52, 12, 0, 0 ], [ 0, 0, 12, 48, 54, 14, 0, 0 ],
    [ 0, 0, 10, 46, 56, 16, 0, 0 ], [ 0, 0, 8, 44, 58, 18, 0, 0 ],
    [ 0, 0, 6, 42, 60, 20, 0, 0 ],  [ 0, 0, 4, 40, 62, 22, 0, 0 ],
    [ 0, 0, 4, 36, 62, 26, 0, 0 ],  [ 0, 0, 2, 34, 62, 30, 0, 0 ]
  ]
];

/* Symbols for coding which components are zero jointly */
pub const MV_JOINTS:usize = 4;

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum MvJointType {
  MV_JOINT_ZERO = 0,   /* Zero vector */
  MV_JOINT_HNZVZ = 1,  /* Vert zero, hor nonzero */
  MV_JOINT_HZVNZ = 2,  /* Hor zero, vert nonzero */
  MV_JOINT_HNZVNZ = 3, /* Both components nonzero */
}

use context::*;
use plane::*;
use predict::*;

impl PredictionMode {
  pub fn predict_intra<'a>(
    self, dst: &'a mut PlaneMutSlice<'a>, tx_size: TxSize, bit_depth: usize,
    ac: &[i16], alpha: i16
  ) {
    assert!(self.is_intra());

    match tx_size {
      TxSize::TX_4X4 =>
        self.predict_intra_inner::<Block4x4>(dst, bit_depth, ac, alpha),
      TxSize::TX_8X8 =>
        self.predict_intra_inner::<Block8x8>(dst, bit_depth, ac, alpha),
      TxSize::TX_16X16 =>
        self.predict_intra_inner::<Block16x16>(dst, bit_depth, ac, alpha),
      TxSize::TX_32X32 =>
        self.predict_intra_inner::<Block32x32>(dst, bit_depth, ac, alpha),
      _ => unimplemented!()
    }
  }

  #[inline(always)]
  fn predict_intra_inner<'a, B: Intra>(
    self, dst: &'a mut PlaneMutSlice<'a>, bit_depth: usize,
    ac: &[i16], alpha: i16
  ) {
    // above and left arrays include above-left sample
    // above array includes above-right samples
    // left array includes below-left samples
    let bd = bit_depth;
    let base = 128 << (bd - 8);

    let above = &mut [(base - 1) as u16; 2 * MAX_TX_SIZE + 1][..B::W + B::H + 1];
    let left = &mut [(base + 1) as u16; 2 * MAX_TX_SIZE + 1][..B::H + B::W + 1];

    let stride = dst.plane.cfg.stride;
    let x = dst.x;
    let y = dst.y;

    if y != 0 {
      if self != PredictionMode::H_PRED {
        above[1..B::W + 1].copy_from_slice(&dst.go_up(1).as_slice()[..B::W]);
      } else if self == PredictionMode::H_PRED && x == 0 {
        for i in 0..B::W {
          above[i + 1] = dst.go_up(1).p(0, 0);
        }
      }
    }

    if x != 0 {
      if self != PredictionMode::V_PRED {
        let left_slice = dst.go_left(1);
        for i in 0..B::H {
          left[i + 1] = left_slice.p(0, i);
        }
      } else if self == PredictionMode::V_PRED && y == 0 {
        for i in 0..B::H {
          left[i + 1] = dst.go_left(1).p(0, 0);
        }
      }
    }

    if self == PredictionMode::PAETH_PRED && x != 0 && y != 0 {
      above[0] = dst.go_up(1).go_left(1).p(0, 0);
    }

    if self == PredictionMode::SMOOTH_H_PRED ||
      self == PredictionMode::SMOOTH_V_PRED ||
      self == PredictionMode::SMOOTH_PRED ||
      self == PredictionMode::PAETH_PRED {
      if x == 0 && y != 0 {
        for i in 0..B::H {
          left[i + 1] = dst.go_up(1).p(0, 0);
        }
      }
      if x != 0 && y == 0 {
        for i in 0..B::W {
          above[i + 1] = dst.go_left(1).p(0, 0);
        }
      }
    }

    if self == PredictionMode::PAETH_PRED {
      if x == 0 && y != 0 {
        above[0] = dst.go_up(1).p(0, 0);
      }
      if x != 0 && y == 0 {
        above[0] = dst.go_left(1).p(0, 0);
      }
      if x == 0 && y == 0 {
        above[0] = base;
        left[0] = base;
      }
    }

    let slice = dst.as_mut_slice();
    let above_slice = &above[1..B::W + 1];
    let left_slice = &left[1..B::H + 1];

    match self {
      PredictionMode::DC_PRED | PredictionMode::UV_CFL_PRED => match (x, y) {
        (0, 0) => B::pred_dc_128(slice, stride, bit_depth),
        (_, 0) => B::pred_dc_left(slice, stride, above_slice, left_slice, bit_depth),
        (0, _) => B::pred_dc_top(slice, stride, above_slice, left_slice, bit_depth),
        _ => B::pred_dc(slice, stride, above_slice, left_slice)
      },
      PredictionMode::H_PRED => match (x, y) {
        (0, 0) => B::pred_h(slice, stride, left_slice),
        (0, _) => B::pred_h(slice, stride, above_slice),
        (_, _) => B::pred_h(slice, stride, left_slice)
      },
      PredictionMode::V_PRED => match (x, y) {
        (0, 0) => B::pred_v(slice, stride, above_slice),
        (_, 0) => B::pred_v(slice, stride, left_slice),
        (_, _) => B::pred_v(slice, stride, above_slice)
      },
      PredictionMode::PAETH_PRED =>
        B::pred_paeth(slice, stride, above_slice, left_slice, above[0]),
      PredictionMode::SMOOTH_PRED =>
        B::pred_smooth(slice, stride, above_slice, left_slice),
      PredictionMode::SMOOTH_H_PRED =>
        B::pred_smooth_h(slice, stride, above_slice, left_slice),
      PredictionMode::SMOOTH_V_PRED =>
        B::pred_smooth_v(slice, stride, above_slice, left_slice),
      _ => unimplemented!()
    }
    if self == PredictionMode::UV_CFL_PRED {
      B::pred_cfl(slice, stride, &ac, alpha, bit_depth);
    }
  }

  pub fn is_intra(self) -> bool {
    return self < PredictionMode::NEARESTMV;
  }

  pub fn is_cfl(self) -> bool {
    self == PredictionMode::UV_CFL_PRED
  }

  pub fn is_directional(self) -> bool {
    self >= PredictionMode::V_PRED && self <= PredictionMode::D63_PRED
  }

  pub fn predict_inter<'a>(self, fi: &FrameInvariants, p: usize, po: &PlaneOffset,
                           dst: &'a mut PlaneMutSlice<'a>, width: usize, height: usize,
                           ref_frame: usize, mv: &MotionVector, bit_depth: usize) {
    assert!(!self.is_intra());
    assert!(ref_frame == LAST_FRAME);

    match fi.rec_buffer.frames[fi.ref_frames[ref_frame - LAST_FRAME]] {
      Some(ref rec) => {
        let rec_cfg = &rec.frame.planes[p].cfg;
        let shift_row = 3 + rec_cfg.ydec;
        let shift_col = 3 + rec_cfg.xdec;
        let row_offset = mv.row as i32 >> shift_row;
        let col_offset = mv.col as i32 >> shift_col;
        let row_frac = (mv.row as i32 - (row_offset << shift_row)) << (4 - shift_row);
        let col_frac = (mv.col as i32 - (col_offset << shift_col)) << (4 - shift_col);
        let ref_stride = rec_cfg.stride;

        let stride = dst.plane.cfg.stride;
        let slice = dst.as_mut_slice();

        let max_sample_val = ((1 << bit_depth) - 1) as i32;
        let y_filter_idx = if height <= 4 { 4 } else { 0 };
        let x_filter_idx = if width <= 4 { 4 } else { 0 };

        match (col_frac, row_frac) {
        (0,0) => {
          let qo = PlaneOffset { x: po.x + col_offset as isize, y: po.y + row_offset as isize };
          let ps = rec.frame.planes[p].slice(&qo);
          let s = ps.as_slice_clamped();
          for r in 0..height {
            for c in 0..width {
              let output_index = r * stride + c;
              slice[output_index] = s[r * ref_stride + c];
            }
          }
        }
        (0,_) => {
          let qo = PlaneOffset { x: po.x + col_offset as isize, y: po.y + row_offset as isize - 3 };
          let ps = rec.frame.planes[p].slice(&qo);
          let s = ps.as_slice_clamped();
          for r in 0..height {
            for c in 0..width {
              let mut sum: i32 = 0;
              for k in 0..8 {
                sum += s[(r + k) * ref_stride + c] as i32 * SUBPEL_FILTERS[y_filter_idx][row_frac as usize][k];
              }
              let output_index = r * stride + c;
              let val = ((sum + 64) >> 7).max(0).min(max_sample_val);
              slice[output_index] = val as u16;
            }
          }
        }
        (_,0) => {
          let qo = PlaneOffset { x: po.x + col_offset as isize - 3, y: po.y + row_offset as isize };
          let ps = rec.frame.planes[p].slice(&qo);
          let s = ps.as_slice_clamped();
          for r in 0..height {
            for c in 0..width {
            let mut sum: i32 = 0;
            for k in 0..8 {
              sum += s[r * ref_stride + (c + k)] as i32 * SUBPEL_FILTERS[x_filter_idx][col_frac as usize][k];
            }
            let output_index = r * stride + c;
            let val = ((((sum + 4) >> 3) + 8) >> 4).max(0).min(max_sample_val);
            slice[output_index] = val as u16;
            }
          }
        }
        (_,_) => {
          let mut intermediate = [0 as i16; 8 * (128 + 7)];

          let qo = PlaneOffset { x: po.x + col_offset as isize - 3, y: po.y + row_offset as isize - 3 };
          let ps = rec.frame.planes[p].slice(&qo);
          let s = ps.as_slice_clamped();
          for cg in (0..width).step_by(8) {
            for r in 0..height+7 {
              for c in cg..(cg+8).min(width) {
                let mut sum: i32 = 0;
                for k in 0..8 {
                  sum += s[r * ref_stride + (c + k)] as i32 * SUBPEL_FILTERS[x_filter_idx][col_frac as usize][k];
                }
                let val = (sum + 4) >> 3;
                intermediate[8 * r + (c - cg)] = val as i16;
              }
            }

            for r in 0..height {
              for c in cg..(cg+8).min(width) {
                let mut sum: i32 = 0;
                for k in 0..8 {
                  sum += intermediate[8 * (r + k) + c - cg] as i32 * SUBPEL_FILTERS[y_filter_idx][row_frac as usize][k];
                }
                let output_index = r * stride + c;
                let val = ((sum + 1024) >> 11).max(0).min(max_sample_val);
                slice[output_index] = val as u16;
              }
            }
          }
        }
      }
      },
      None => (),
    }

  }
}

#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub enum TxSet {
  // DCT only
  TX_SET_DCTONLY,
  // DCT + Identity only
  TX_SET_DCT_IDTX,
  // Discrete Trig transforms w/o flip (4) + Identity (1)
  TX_SET_DTT4_IDTX,
  // Discrete Trig transforms w/o flip (4) + Identity (1) + 1D Hor/vert DCT (2)
  // for 16x16 only
  TX_SET_DTT4_IDTX_1DDCT_16X16,
  // Discrete Trig transforms w/o flip (4) + Identity (1) + 1D Hor/vert DCT (2)
  TX_SET_DTT4_IDTX_1DDCT,
  // Discrete Trig transforms w/ flip (9) + Identity (1)
  TX_SET_DTT9_IDTX,
  // Discrete Trig transforms w/ flip (9) + Identity (1) + 1D Hor/Ver DCT (2)
  TX_SET_DTT9_IDTX_1DDCT,
  // Discrete Trig transforms w/ flip (9) + Identity (1) + 1D Hor/Ver (6)
  // for 16x16 only
  TX_SET_ALL16_16X16,
  // Discrete Trig transforms w/ flip (9) + Identity (1) + 1D Hor/Ver (6)
  TX_SET_ALL16
}

pub fn get_subsize(bsize: BlockSize, partition: PartitionType) -> BlockSize {
  subsize_lookup[partition as usize][bsize as usize]
}
