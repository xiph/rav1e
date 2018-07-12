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

use BlockSize::*;
use TxSize::*;

#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub enum PartitionType {
  PARTITION_NONE,
  PARTITION_HORZ,
  PARTITION_VERT,
  PARTITION_SPLIT,
  PARTITION_INVALID
}

#[derive(Copy, Clone, PartialEq, PartialOrd)]
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
  BLOCK_4X16,
  BLOCK_16X4,
  BLOCK_8X32,
  BLOCK_32X8,
  BLOCK_16X64,
  BLOCK_64X16,
  BLOCK_INVALID
}

impl BlockSize {
  pub const BLOCK_SIZES_ALL: usize = 19;

  const BLOCK_SIZE_WIDTH_LOG2: [usize; BlockSize::BLOCK_SIZES_ALL] =
    [2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 2, 4, 3, 5, 4, 6];

  const BLOCK_SIZE_HEIGHT_LOG2: [usize; BlockSize::BLOCK_SIZES_ALL] =
    [2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 4, 2, 5, 3, 6, 4];

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
}

/// Transform Size
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
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
  pub const TX_SIZES: usize = 4;

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

#[derive(Copy, Clone, PartialEq)]
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
  NEARESTMV,
  NEARMV,
  ZEROMV,
  NEWMV,
  // Compound ref compound modes
  NEAREST_NEARESTMV,
  NEAR_NEARMV,
  NEAREST_NEWMV,
  NEW_NEARESTMV,
  NEAR_NEWMV,
  NEW_NEARMV,
  ZERO_ZEROMV,
  NEW_NEWMV
}

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

use context::*;
use plane::*;
use predict::*;

impl PredictionMode {
  pub fn predict<'a>(&self, dst: &'a mut PlaneMutSlice<'a>, tx_size: TxSize) {
    match tx_size {
      TxSize::TX_4X4 => self.predict_inner::<Block4x4>(dst),
      TxSize::TX_8X8 => self.predict_inner::<Block8x8>(dst),
      TxSize::TX_16X16 => self.predict_inner::<Block16x16>(dst),
      TxSize::TX_32X32 => self.predict_inner::<Block32x32>(dst),
      _ => unimplemented!()
    }
  }

  #[inline(always)]
  fn predict_inner<'a, B: Intra>(&self, dst: &'a mut PlaneMutSlice<'a>) {
    // above and left arrays include above-left sample
    // above array includes above-right samples
    // left array includes below-left samples
    let above = &mut [127u16; 2 * MAX_TX_SIZE + 1][..B::W + B::H + 1];
    let left = &mut [129u16; 2 * MAX_TX_SIZE + 1][..B::H + B::W + 1];

    let stride = dst.plane.cfg.stride;
    let x = dst.x;
    let y = dst.y;

    if self != &PredictionMode::H_PRED && y != 0 {
      above[1..B::W + 1].copy_from_slice(&dst.go_up(1).as_slice()[..B::W]);
    }

    if self != &PredictionMode::V_PRED && x != 0 {
      let left_slice = dst.go_left(1);
      for i in 0..B::H {
        left[i + 1] = left_slice.p(0, i);
      }
    }

    if self == &PredictionMode::PAETH_PRED && x != 0 && y != 0 {
      above[0] = dst.go_up(1).go_left(1).p(0, 0);
      left[0] = above[0];
    }

    let slice = dst.as_mut_slice();
    let above_slice = &above[1..B::W + 1];
    let left_slice = &left[1..B::H + 1];

    match *self {
      PredictionMode::DC_PRED => match (x, y) {
        (0, 0) => B::pred_dc_128(slice, stride),
        (_, 0) => B::pred_dc_left(slice, stride, above_slice, left_slice),
        (0, _) => B::pred_dc_top(slice, stride, above_slice, left_slice),
        _ => B::pred_dc(slice, stride, above_slice, left_slice)
      },
      PredictionMode::H_PRED => B::pred_h(slice, stride, left_slice),
      PredictionMode::V_PRED => B::pred_v(slice, stride, above_slice),
      PredictionMode::PAETH_PRED =>
        B::pred_paeth(slice, stride, above_slice, left_slice, above[0]),
      PredictionMode::SMOOTH_PRED =>
        B::pred_smooth(slice, stride, above_slice, left_slice, 8),
      PredictionMode::SMOOTH_H_PRED =>
        B::pred_smooth_h(slice, stride, above_slice, left_slice, 8),
      PredictionMode::SMOOTH_V_PRED =>
        B::pred_smooth_v(slice, stride, above_slice, left_slice, 8),
      _ => unimplemented!()
    }
  }

  pub fn is_directional(self) -> bool {
    self >= PredictionMode::V_PRED && self <= PredictionMode::D63_PRED
  }
}

#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub enum TxSetType {
  // DCT only
  EXT_TX_SET_DCTONLY,
  // DCT + Identity only
  EXT_TX_SET_DCT_IDTX,
  // Discrete Trig transforms w/o flip (4) + Identity (1)
  EXT_TX_SET_DTT4_IDTX,
  // Discrete Trig transforms w/o flip (4) + Identity (1) + 1D Hor/vert DCT (2)
  // for 16x16 only
  EXT_TX_SET_DTT4_IDTX_1DDCT_16X16,
  // Discrete Trig transforms w/o flip (4) + Identity (1) + 1D Hor/vert DCT (2)
  EXT_TX_SET_DTT4_IDTX_1DDCT,
  // Discrete Trig transforms w/ flip (9) + Identity (1)
  EXT_TX_SET_DTT9_IDTX,
  // Discrete Trig transforms w/ flip (9) + Identity (1) + 1D Hor/Ver DCT (2)
  EXT_TX_SET_DTT9_IDTX_1DDCT,
  // Discrete Trig transforms w/ flip (9) + Identity (1) + 1D Hor/Ver (6)
  // for 16x16 only
  EXT_TX_SET_ALL16_16X16,
  // Discrete Trig transforms w/ flip (9) + Identity (1) + 1D Hor/Ver (6)
  EXT_TX_SET_ALL16
}

pub fn get_subsize(bsize: BlockSize, partition: PartitionType) -> BlockSize {
  subsize_lookup[partition as usize][bsize as usize]
}
