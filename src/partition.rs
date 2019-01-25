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
use context::*;
use encoder::{ChromaSampling, FrameInvariants};
use mc::*;
use plane::*;
use predict::*;
use util::*;

pub const NONE_FRAME: usize = 8;
pub const INTRA_FRAME: usize = 0;
pub const LAST_FRAME: usize = 1;
pub const LAST2_FRAME: usize = 2;
pub const LAST3_FRAME: usize = 3;
pub const GOLDEN_FRAME: usize = 4;
pub const BWDREF_FRAME: usize = 5;
pub const ALTREF2_FRAME: usize = 6;
pub const ALTREF_FRAME: usize = 7;

pub const LAST_LAST2_FRAMES: usize = 0; // { LAST_FRAME, LAST2_FRAME }
pub const LAST_LAST3_FRAMES: usize = 1; // { LAST_FRAME, LAST3_FRAME }
pub const LAST_GOLDEN_FRAMES: usize = 2; // { LAST_FRAME, GOLDEN_FRAME }
pub const BWDREF_ALTREF_FRAMES: usize = 3; // { BWDREF_FRAME, ALTREF_FRAME }
pub const LAST2_LAST3_FRAMES: usize = 4; // { LAST2_FRAME, LAST3_FRAME }
pub const LAST2_GOLDEN_FRAMES: usize = 5; // { LAST2_FRAME, GOLDEN_FRAME }
pub const LAST3_GOLDEN_FRAMES: usize = 6; // { LAST3_FRAME, GOLDEN_FRAME }
pub const BWDREF_ALTREF2_FRAMES: usize = 7; // { BWDREF_FRAME, ALTREF2_FRAME }
pub const ALTREF2_ALTREF_FRAMES: usize = 8; // { ALTREF2_FRAME, ALTREF_FRAME }
pub const TOTAL_UNIDIR_COMP_REFS: usize = 9;

// NOTE: UNIDIR_COMP_REFS is the number of uni-directional reference pairs
//       that are explicitly signaled.
pub const UNIDIR_COMP_REFS: usize = BWDREF_ALTREF_FRAMES + 1;

pub const FWD_REFS: usize = GOLDEN_FRAME - LAST_FRAME + 1;
pub const BWD_REFS: usize = ALTREF_FRAME - BWDREF_FRAME + 1;
pub const SINGLE_REFS: usize = FWD_REFS + BWD_REFS;
pub const TOTAL_REFS_PER_FRAME: usize = ALTREF_FRAME - INTRA_FRAME + 1;
pub const INTER_REFS_PER_FRAME: usize = ALTREF_FRAME - LAST_FRAME + 1;
pub const TOTAL_COMP_REFS: usize =
  FWD_REFS * BWD_REFS + TOTAL_UNIDIR_COMP_REFS;

pub const REF_FRAMES_LOG2: usize = 3;
pub const REF_FRAMES: usize = 1 << REF_FRAMES_LOG2;

pub const REF_CONTEXTS: usize = 3;
pub const MVREF_ROW_COLS: usize = 3;

#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub enum PartitionType {
  PARTITION_NONE,
  PARTITION_HORZ,
  PARTITION_VERT,
  PARTITION_SPLIT,
  PARTITION_HORZ_A, // HORZ split and the top partition is split again
  PARTITION_HORZ_B, // HORZ split and the bottom partition is split again
  PARTITION_VERT_A, // VERT split and the left partition is split again
  PARTITION_VERT_B, // VERT split and the right partition is split again
  PARTITION_HORZ_4, // 4:1 horizontal partition
  PARTITION_VERT_4, // 4:1 vertical partition
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
  BLOCK_INVALID
}

impl BlockSize {
  pub const BLOCK_SIZES_ALL: usize = 22;

  const BLOCK_SIZE_WIDTH_LOG2: [usize; BlockSize::BLOCK_SIZES_ALL] =
    [2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 2, 4, 3, 5, 4, 6];

  const BLOCK_SIZE_HEIGHT_LOG2: [usize; BlockSize::BLOCK_SIZES_ALL] =
    [2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 4, 2, 5, 3, 6, 4];

  pub fn from_width_and_height(w: usize, h: usize) -> BlockSize {
    match (w, h) {
      (4, 4) => BLOCK_4X4,
      (4, 8) => BLOCK_4X8,
      (8, 4) => BLOCK_8X4,
      (8, 8) => BLOCK_8X8,
      (8, 16) => BLOCK_8X16,
      (16, 8) => BLOCK_16X8,
      (16, 16) => BLOCK_16X16,
      (16, 32) => BLOCK_16X32,
      (32, 16) => BLOCK_32X16,
      (32, 32) => BLOCK_32X32,
      (32, 64) => BLOCK_32X64,
      (64, 32) => BLOCK_64X32,
      (64, 64) => BLOCK_64X64,
      (64, 128) => BLOCK_64X128,
      (128, 64) => BLOCK_128X64,
      (128, 128) => BLOCK_128X128,
      (4, 16) => BLOCK_4X16,
      (16, 4) => BLOCK_16X4,
      (8, 32) => BLOCK_8X32,
      (32, 8) => BLOCK_32X8,
      (16, 64) => BLOCK_16X64,
      (64, 16) => BLOCK_64X16,
      _ => unreachable!()
    }
  }

  pub fn cfl_allowed(self) -> bool {
    // TODO: fix me when enabling EXT_PARTITION_TYPES
    self <= BlockSize::BLOCK_32X32
  }

  pub fn width(self) -> usize {
    1 << self.width_log2()
  }

  pub fn width_log2(self) -> usize {
    BlockSize::BLOCK_SIZE_WIDTH_LOG2[self as usize]
  }

  pub fn width_mi(self) -> usize {
    self.width() >> MI_SIZE_LOG2
  }

  pub fn height(self) -> usize {
    1 << self.height_log2()
  }

  pub fn height_log2(self) -> usize {
    BlockSize::BLOCK_SIZE_HEIGHT_LOG2[self as usize]
  }

  pub fn height_mi(self) -> usize {
    self.height() >> MI_SIZE_LOG2
  }

  pub fn tx_size(self) -> TxSize {
    match self {
      BLOCK_4X4 => TX_4X4,
      BLOCK_4X8 => TX_4X8,
      BLOCK_8X4 => TX_8X4,
      BLOCK_8X8 => TX_8X8,
      BLOCK_8X16 => TX_8X16,
      BLOCK_16X8 => TX_16X8,
      BLOCK_16X16 => TX_16X16,
      BLOCK_16X32 => TX_16X32,
      BLOCK_32X16 => TX_32X16,
      BLOCK_32X32 => TX_32X32,
      BLOCK_32X64 => TX_32X64,
      BLOCK_64X32 => TX_64X32,
      BLOCK_4X16 => TX_4X16,
      BLOCK_16X4 => TX_16X4,
      BLOCK_8X32 => TX_8X32,
      BLOCK_32X8 => TX_32X8,
      BLOCK_16X64 => TX_16X64,
      BLOCK_64X16 => TX_64X16,
      BLOCK_INVALID => unreachable!(),
      _ => TX_64X64
    }
  }

  pub fn largest_uv_tx_size(self, chroma_sampling: ChromaSampling) -> TxSize {
    match chroma_sampling {
      ChromaSampling::Cs444 => match self {
        BLOCK_64X64 | BLOCK_64X32 | BLOCK_32X64 |
        BLOCK_128X128 | BLOCK_128X64 | BLOCK_64X128 => TX_32X32,
        BLOCK_64X16 => TX_32X16,
        BLOCK_16X64 => TX_16X32,
        _ => self.tx_size()
      },
      ChromaSampling::Cs422 => match self {
        BLOCK_4X4 | BLOCK_8X4 => TX_4X4,
        BLOCK_8X8 => TX_4X8,
        BLOCK_16X8 => TX_8X8,
        BLOCK_16X16 => TX_8X16,
        BLOCK_32X16 => TX_16X16,
        BLOCK_32X32 => TX_16X32,
        BLOCK_64X32 | BLOCK_64X64 |
        BLOCK_128X64 | BLOCK_128X128 => TX_32X32,
        BLOCK_16X4 => TX_8X4,
        BLOCK_32X8 => TX_16X8,
        BLOCK_64X16 => TX_32X16,
        _ => unreachable!() // vertical splits are illegal in 4:2:2
      },
      ChromaSampling::Cs420 => match self {
        BLOCK_4X4 | BLOCK_8X4 | BLOCK_4X8 | BLOCK_8X8 => TX_4X4,
        BLOCK_8X16 | BLOCK_4X16 => TX_4X8,
        BLOCK_16X8 | BLOCK_16X4 => TX_8X4,
        BLOCK_16X16 => TX_8X8,
        BLOCK_16X32 => TX_8X16,
        BLOCK_32X16 => TX_16X8,
        BLOCK_32X32 => TX_16X16,
        BLOCK_32X64 => TX_16X32,
        BLOCK_64X32 => TX_32X16,
        BLOCK_8X32 => TX_4X16,
        BLOCK_32X8 => TX_16X4,
        BLOCK_16X64 => TX_8X32,
        BLOCK_64X16 => TX_32X8,
        _ => TX_32X32
      }
    }
  }

  pub fn is_sqr(self) -> bool {
    self.width_log2() == self.height_log2()
  }

  pub fn is_sub8x8(self) -> bool {
    self.width_log2().min(self.height_log2()) < 3
  }

  pub fn sub8x8_offset(self) -> (isize, isize) {
    let offset_x: isize = if self.width_log2() == 2 { -1 } else { 0 };
    let offset_y: isize = if self.height_log2() == 2 { -1 } else { 0 };

    (offset_x, offset_y)
  }

  pub fn greater_than(self, other: BlockSize) -> bool {
    (self.width() > other.width() && self.height() >= other.height()) ||
    (self.width() >= other.width() && self.height() > other.height())
  }

  pub fn gte(self, other: BlockSize) -> bool {
    self.greater_than(other) ||
    (self.width() == other.width() && self.height() == other.height())
  }

  #[cfg_attr(rustfmt, rustfmt_skip)]
  const SUBSIZE_LOOKUP: [[BlockSize; BlockSize::BLOCK_SIZES_ALL];
    EXT_PARTITION_TYPES] = [
    // PARTITION_NONE
    [
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
      BLOCK_32X8,    BLOCK_16X64,   BLOCK_64X16
    ],
    // PARTITION_HORZ
    [
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
      BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID
    ],
    // PARTITION_VERT
    [
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
      BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID
    ],
    // PARTITION_SPLIT
    [
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
      BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID
    ],
    // PARTITION_HORZ_A
    [
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
    ],
    // PARTITION_HORZ_B
    [
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
      BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID
    ],
    // PARTITION_VERT_A
    [
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
      BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID
    ],
    // PARTITION_VERT_B
    [
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
      BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID
    ],
    // PARTITION_HORZ_4
    [
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
      BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
      // 4X16,       16X4,          8X32
      BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
      // 32X8,       16X64,         64X16
      BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID
    ],
    // PARTITION_VERT_4
    [
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
      BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
      // 4X16,       16X4,          8X32
      BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
      // 32X8,       16X64,         64X16
      BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID
    ]
  ];

  pub fn subsize(self, partition: PartitionType) -> BlockSize {
    BlockSize::SUBSIZE_LOOKUP[partition as usize][self as usize]
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
    match self {
      TX_4X4 | TX_4X8 | TX_4X16 => 2,
      TX_8X8 | TX_8X4 | TX_8X16 | TX_8X32 => 3,
      TX_16X16 | TX_16X8 | TX_16X32 | TX_16X4 | TX_16X64 => 4,
      TX_32X32 | TX_32X16 | TX_32X64 | TX_32X8 => 5,
      TX_64X64 | TX_64X32 | TX_64X16 => 6
    }
  }

  pub fn width_index(self) -> usize {
    self.width_log2() - TX_4X4.width_log2()
  }

  pub fn height(self) -> usize {
    1 << self.height_log2()
  }

  pub fn height_log2(self) -> usize {
    match self {
      TX_4X4 | TX_8X4 | TX_16X4 => 2,
      TX_8X8 | TX_4X8 | TX_16X8 | TX_32X8 => 3,
      TX_16X16 | TX_8X16 | TX_32X16 | TX_4X16 | TX_64X16 => 4,
      TX_32X32 | TX_16X32 | TX_64X32 | TX_8X32 => 5,
      TX_64X64 | TX_32X64 | TX_16X64 => 6
    }
  }

  pub fn height_index(self) -> usize {
    self.height_log2() - TX_4X4.height_log2()
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
    match self {
      TX_4X4 => BLOCK_4X4,
      TX_8X8 => BLOCK_8X8,
      TX_16X16 => BLOCK_16X16,
      TX_32X32 => BLOCK_32X32,
      TX_64X64 => BLOCK_64X64,
      TX_4X8 => BLOCK_4X8,
      TX_8X4 => BLOCK_8X4,
      TX_8X16 => BLOCK_8X16,
      TX_16X8 => BLOCK_16X8,
      TX_16X32 => BLOCK_16X32,
      TX_32X16 => BLOCK_32X16,
      TX_32X64 => BLOCK_32X64,
      TX_64X32 => BLOCK_64X32,
      TX_4X16 => BLOCK_4X16,
      TX_16X4 => BLOCK_16X4,
      TX_8X32 => BLOCK_8X32,
      TX_32X8 => BLOCK_32X8,
      TX_16X64 => BLOCK_16X64,
      TX_64X16 => BLOCK_64X16
    }
  }

  pub fn sqr(self) -> TxSize {
    match self {
      TX_4X4 | TX_4X8 | TX_8X4 | TX_4X16 | TX_16X4 => TX_4X4,
      TX_8X8 | TX_8X16 | TX_16X8 | TX_8X32 | TX_32X8 => TX_8X8,
      TX_16X16 | TX_16X32 | TX_32X16 | TX_16X64 | TX_64X16 => TX_16X16,
      TX_32X32 | TX_32X64 | TX_64X32 => TX_32X32,
      TX_64X64 => TX_64X64
    }
  }

  pub fn sqr_up(self) -> TxSize {
    match self {
      TX_4X4 => TX_4X4,
      TX_8X8 | TX_4X8 | TX_8X4 => TX_8X8,
      TX_16X16 | TX_8X16 | TX_16X8 | TX_4X16 | TX_16X4 => TX_16X16,
      TX_32X32 | TX_16X32 | TX_32X16 | TX_8X32 | TX_32X8 => TX_32X32,
      TX_64X64 | TX_32X64 | TX_64X32 | TX_16X64 | TX_64X16 => TX_64X64
    }
  }

  pub fn by_dims(w: usize, h: usize) -> TxSize {
    match (w, h) {
      (4, 4) => TX_4X4,
      (8, 8) => TX_8X8,
      (16, 16) => TX_16X16,
      (32, 32) => TX_32X32,
      (64, 64) => TX_64X64,
      (4, 8) => TX_4X8,
      (8, 4) => TX_8X4,
      (8, 16) => TX_8X16,
      (16, 8) => TX_16X8,
      (16, 32) => TX_16X32,
      (32, 16) => TX_32X16,
      (32, 64) => TX_32X64,
      (64, 32) => TX_64X32,
      (4, 16) => TX_4X16,
      (16, 4) => TX_16X4,
      (8, 32) => TX_8X32,
      (32, 8) => TX_32X8,
      (16, 64) => TX_16X64,
      (64, 16) => TX_64X16,
      _ => unreachable!()
    }
  }
}

pub const TX_TYPES: usize = 16;

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
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
  NEAR0MV,
  NEAR1MV,
  NEAR2MV,
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

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum InterIntraMode {
  II_DC_PRED,
  II_V_PRED,
  II_H_PRED,
  II_SMOOTH_PRED,
  INTERINTRA_MODES
}
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum CompoundType {
  COMPOUND_AVERAGE,
  COMPOUND_WEDGE,
  COMPOUND_DIFFWTD,
  COMPOUND_TYPES,
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum MotionMode {
  SIMPLE_TRANSLATION,
  OBMC_CAUSAL,    // 2-sided OBMC
  WARPED_CAUSAL,  // 2-sided WARPED
  MOTION_MODES
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum PaletteSize {
  TWO_COLORS,
  THREE_COLORS,
  FOUR_COLORS,
  FIVE_COLORS,
  SIX_COLORS,
  SEVEN_COLORS,
  EIGHT_COLORS,
  PALETTE_SIZES
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum PaletteColor {
  PALETTE_COLOR_ONE,
  PALETTE_COLOR_TWO,
  PALETTE_COLOR_THREE,
  PALETTE_COLOR_FOUR,
  PALETTE_COLOR_FIVE,
  PALETTE_COLOR_SIX,
  PALETTE_COLOR_SEVEN,
  PALETTE_COLOR_EIGHT,
  PALETTE_COLORS
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum FilterIntraMode {
  FILTER_DC_PRED,
  FILTER_V_PRED,
  FILTER_H_PRED,
  FILTER_D157_PRED,
  FILTER_PAETH_PRED,
  FILTER_INTRA_MODES
}

#[derive(Copy, Clone)]
pub struct MotionVector {
  pub row: i16,
  pub col: i16
}

pub const NEWMV_MODE_CONTEXTS: usize = 7;
pub const GLOBALMV_MODE_CONTEXTS: usize = 2;
pub const REFMV_MODE_CONTEXTS: usize = 6;
pub const INTER_COMPOUND_MODES: usize = 1 + PredictionMode::NEW_NEWMV as usize
  - PredictionMode::NEAREST_NEARESTMV as usize;

pub const REFMV_OFFSET: usize = 4;
pub const GLOBALMV_OFFSET: usize = 3;
pub const NEWMV_CTX_MASK: usize = ((1 << GLOBALMV_OFFSET) - 1);
pub const GLOBALMV_CTX_MASK: usize =
  ((1 << (REFMV_OFFSET - GLOBALMV_OFFSET)) - 1);
pub const REFMV_CTX_MASK: usize = ((1 << (8 - REFMV_OFFSET)) - 1);

pub static RAV1E_PARTITION_TYPES: &'static [PartitionType] =
  &[PartitionType::PARTITION_NONE, PartitionType::PARTITION_HORZ,
    PartitionType::PARTITION_VERT, PartitionType::PARTITION_SPLIT];

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
  MV_SUBPEL_HIGH_PRECISION
}

/* Symbols for coding which components are zero jointly */
pub const MV_JOINTS: usize = 4;

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum MvJointType {
  MV_JOINT_ZERO = 0,   /* Zero vector */
  MV_JOINT_HNZVZ = 1,  /* Vert zero, hor nonzero */
  MV_JOINT_HZVNZ = 2,  /* Hor zero, vert nonzero */
  MV_JOINT_HNZVNZ = 3  /* Both components nonzero */
}

pub fn get_intra_edges<'a>(
  dst: &'a PlaneSlice<'a>,
  tx_size: TxSize,
  bit_depth: usize,
  p: usize,
  frame_w_in_b: usize,
  frame_h_in_b: usize,
  opt_mode: Option<PredictionMode>
) -> AlignedArray<[u16; 4 * MAX_TX_SIZE + 1]> {

  let mut edge_buf: AlignedArray<[u16; 4 * MAX_TX_SIZE + 1]> =
    UninitializedAlignedArray();
  let base = 128u16 << (bit_depth - 8);

  {
    // left pixels are order from bottom to top and right-aligned
    let (left, not_left) = edge_buf.array.split_at_mut(2*MAX_TX_SIZE);
    let (top_left, above) = not_left.split_at_mut(1);

    let x = dst.x;
    let y = dst.y;

    let mut needs_left = true;
    let mut needs_topleft = true;
    let mut needs_top = true;
    let mut needs_topright = true;
    let mut needs_bottomleft = true;

    if let Some(mut mode) = opt_mode {
      mode = match mode {
        PredictionMode::PAETH_PRED => match (x, y) {
          (0, 0) => PredictionMode::DC_PRED,
          (_, 0) => PredictionMode::H_PRED,
          (0, _) => PredictionMode::V_PRED,
          _ => PredictionMode::PAETH_PRED
        },
        _ => mode
      };

      let dc_or_cfl =
        mode == PredictionMode::DC_PRED || mode == PredictionMode::UV_CFL_PRED;

      needs_left = mode != PredictionMode::V_PRED && (!dc_or_cfl || x != 0);
      needs_topleft = mode == PredictionMode::PAETH_PRED || mode == PredictionMode::D117_PRED
      || mode == PredictionMode::D135_PRED || mode == PredictionMode::D153_PRED;
      needs_top = mode != PredictionMode::H_PRED && (!dc_or_cfl || y != 0);
      needs_topright = mode == PredictionMode::D45_PRED || mode == PredictionMode::D63_PRED;
      needs_bottomleft = mode == PredictionMode::D207_PRED;
    }

    // Needs left
    if needs_left {
      if x != 0 {
        let left_slice = dst.go_left(1);
        for i in 0..tx_size.height() {
          left[2*MAX_TX_SIZE - tx_size.height() + i] = left_slice.p(0, tx_size.height() - 1 - i);
        }
      } else {
        let val = if y != 0 { dst.go_up(1).p(0, 0) } else { base + 1 };
        for v in left[2*MAX_TX_SIZE - tx_size.height()..].iter_mut() {
          *v = val
        }
      }
    }

    // Needs top-left
    if needs_topleft {
      top_left[0] = match (x, y) {
        (0, 0) => base,
        (_, 0) => dst.go_left(1).p(0, 0),
        (0, _) => dst.go_up(1).p(0, 0),
        _ => dst.go_up(1).go_left(1).p(0, 0)
      };
    }

    // Needs top
    if needs_top {
      if y != 0 {
        above[..tx_size.width()].copy_from_slice(&dst.go_up(1).as_slice()[..tx_size.width()]);
      } else {
        let val = if x != 0 { dst.go_left(1).p(0, 0) } else { base - 1 };
        for v in above[..tx_size.width()].iter_mut() {
          *v = val;
        }
      }
    }

    // Needs top right
    if needs_topright {
      let bo = if p == 0 {
        BlockOffset { x: x as usize / 4, y: y as usize / 4 }
      } else {
        BlockOffset { x: x as usize / 2, y: y as usize / 2 }
      };
      let bsize = if p == 0 {
        BlockSize::from_width_and_height(tx_size.width(), tx_size.height())
      } else {
        BlockSize::from_width_and_height(2*tx_size.width(), 2*tx_size.height())
      };
      let num_avail = if y != 0 && has_tr(&bo, bsize) {
        tx_size.height().min(
          (if p == 0 { MI_SIZE } else { MI_SIZE / 2 }) * frame_w_in_b
            - x as usize
            - tx_size.width()
        )
      } else {
        0
      };
      if num_avail > 0 {
        above[tx_size.width()..tx_size.width() + num_avail]
        .copy_from_slice(&dst.go_up(1).as_slice()[tx_size.width()..tx_size.width() + num_avail]);
      }
      if num_avail < tx_size.height() {
        let val = above[tx_size.width() + num_avail - 1];
        for v in above[tx_size.width() + num_avail..tx_size.width() + tx_size.height()].iter_mut() {
          *v = val;
        }
      }
    }

    // Needs bottom left
    if needs_bottomleft {
      let bo = if p == 0 {
        BlockOffset { x: x as usize / 4, y: y as usize / 4 }
      } else {
        BlockOffset { x: x as usize / 2, y: y as usize / 2 }
      };
      let bsize = if p == 0 {
        BlockSize::from_width_and_height(tx_size.width(), tx_size.height())
      } else {
        BlockSize::from_width_and_height(2*tx_size.width(), 2*tx_size.height())
      };
      let num_avail = if x != 0 && has_bl(&bo, bsize) {
        tx_size.width().min(
          (if p == 0 { MI_SIZE } else { MI_SIZE / 2 }) * frame_h_in_b
            - y as usize
            - tx_size.height()
        )
      } else {
        0
      };
      if num_avail > 0 {
        let left_slice = dst.go_left(1);
        for i in 0..num_avail {
          left[2*MAX_TX_SIZE - tx_size.height() - 1 - i] = left_slice.p(0, tx_size.height() + i);
        }
      }
      if num_avail < tx_size.width() {
        let val = left[2 * MAX_TX_SIZE - tx_size.height() - num_avail];
        for v in left[(2 * MAX_TX_SIZE - tx_size.height() - tx_size.width())
          ..(2 * MAX_TX_SIZE - tx_size.height() - num_avail)]
          .iter_mut()
        {
          *v = val;
        }
      }
    }

  }
  edge_buf
}

impl PredictionMode {
  pub fn predict_intra<'a>(
    self, dst: &'a mut PlaneMutSlice<'a>, tx_size: TxSize, bit_depth: usize,
    ac: &[i16], alpha: i16, edge_buf: &AlignedArray<[u16; 4 * MAX_TX_SIZE + 1]>
  ) {
    assert!(self.is_intra());

    match tx_size {
      TxSize::TX_4X4 =>
        self.predict_intra_inner::<Block4x4>(dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_8X8 =>
        self.predict_intra_inner::<Block8x8>(dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_16X16 =>
        self.predict_intra_inner::<Block16x16>(dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_32X32 =>
        self.predict_intra_inner::<Block32x32>(dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_64X64 =>
        self.predict_intra_inner::<Block64x64>(dst, bit_depth, ac, alpha, edge_buf),

      TxSize::TX_4X8 =>
        self.predict_intra_inner::<Block4x8>(dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_8X4 =>
        self.predict_intra_inner::<Block8x4>(dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_8X16 =>
        self.predict_intra_inner::<Block8x16>(dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_16X8 =>
        self.predict_intra_inner::<Block16x8>(dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_16X32 =>
        self.predict_intra_inner::<Block16x32>(dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_32X16 =>
        self.predict_intra_inner::<Block32x16>(dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_32X64 =>
        self.predict_intra_inner::<Block32x64>(dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_64X32 =>
        self.predict_intra_inner::<Block64x32>(dst, bit_depth, ac, alpha, edge_buf),

      TxSize::TX_4X16 =>
        self.predict_intra_inner::<Block4x16>(dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_16X4 =>
        self.predict_intra_inner::<Block16x4>(dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_8X32 =>
        self.predict_intra_inner::<Block8x32>(dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_32X8 =>
        self.predict_intra_inner::<Block32x8>(dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_16X64 =>
        self.predict_intra_inner::<Block16x64>(dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_64X16 =>
        self.predict_intra_inner::<Block64x16>(dst, bit_depth, ac, alpha, edge_buf),
    }
  }

  #[inline(always)]
  fn predict_intra_inner<'a, B: Intra<u16>>(
    self, dst: &'a mut PlaneMutSlice<'a>, bit_depth: usize, ac: &[i16],
    alpha: i16, edge_buf: &AlignedArray<[u16; 4 * MAX_TX_SIZE + 1]>
  ) {
    // left pixels are order from bottom to top and right-aligned
    let (left, not_left) = edge_buf.array.split_at(2*MAX_TX_SIZE);
    let (top_left, above) = not_left.split_at(1);

    let stride = dst.plane.cfg.stride;
    let x = dst.x;
    let y = dst.y;

    let mode: PredictionMode = match self {
      PredictionMode::PAETH_PRED => match (x, y) {
        (0, 0) => PredictionMode::DC_PRED,
        (_, 0) => PredictionMode::H_PRED,
        (0, _) => PredictionMode::V_PRED,
        _ => PredictionMode::PAETH_PRED
      },
      PredictionMode::UV_CFL_PRED =>
        if alpha == 0 {
          PredictionMode::DC_PRED
        } else {
          self
        },
      _ => self
    };

    let slice = dst.as_mut_slice();
    let above_slice = &above[..B::W + B::H];
    let left_slice = &left[2 * MAX_TX_SIZE - B::H..];
    let left_and_left_below_slice = &left[2 * MAX_TX_SIZE - B::H - B::W..];

    match mode {
      PredictionMode::DC_PRED => match (x, y) {
        (0, 0) => B::pred_dc_128(slice, stride, bit_depth),
        (_, 0) => B::pred_dc_left(slice, stride, above_slice, left_slice),
        (0, _) => B::pred_dc_top(slice, stride, above_slice, left_slice),
        _ => B::pred_dc(slice, stride, above_slice, left_slice)
      },
      PredictionMode::UV_CFL_PRED => match (x, y) {
        (0, 0) => B::pred_cfl_128(slice, stride, &ac, alpha, bit_depth),
        (_, 0) => B::pred_cfl_left(
          slice,
          stride,
          &ac,
          alpha,
          bit_depth,
          above_slice,
          left_slice
        ),
        (0, _) => B::pred_cfl_top(
          slice,
          stride,
          &ac,
          alpha,
          bit_depth,
          above_slice,
          left_slice
        ),
        _ => B::pred_cfl(
          slice,
          stride,
          &ac,
          alpha,
          bit_depth,
          above_slice,
          left_slice
        )
      },
      PredictionMode::H_PRED => B::pred_h(slice, stride, left_slice),
      PredictionMode::V_PRED => B::pred_v(slice, stride, above_slice),
      PredictionMode::PAETH_PRED =>
        B::pred_paeth(slice, stride, above_slice, left_slice, top_left[0]),
      PredictionMode::SMOOTH_PRED =>
        B::pred_smooth(slice, stride, above_slice, left_slice),
      PredictionMode::SMOOTH_H_PRED =>
        B::pred_smooth_h(slice, stride, above_slice, left_slice),
      PredictionMode::SMOOTH_V_PRED =>
        B::pred_smooth_v(slice, stride, above_slice, left_slice),
      PredictionMode::D45_PRED =>
        B::pred_directional(slice, stride, above_slice, left_and_left_below_slice, top_left, 45, bit_depth),
      PredictionMode::D135_PRED =>
        B::pred_directional(slice, stride, above_slice, left_and_left_below_slice, top_left, 135, bit_depth),
      PredictionMode::D117_PRED =>
        B::pred_directional(slice, stride, above_slice, left_and_left_below_slice, top_left, 113, bit_depth),
      PredictionMode::D153_PRED =>
        B::pred_directional(slice, stride, above_slice, left_and_left_below_slice, top_left, 157, bit_depth),
      PredictionMode::D207_PRED =>
        B::pred_directional(slice, stride, above_slice, left_and_left_below_slice, top_left, 203, bit_depth),
      PredictionMode::D63_PRED =>
        B::pred_directional(slice, stride, above_slice, left_and_left_below_slice, top_left, 67, bit_depth),
      _ => unimplemented!()
    }
  }

  pub fn is_intra(self) -> bool {
    self < PredictionMode::NEARESTMV
  }

  pub fn is_cfl(self) -> bool {
    self == PredictionMode::UV_CFL_PRED
  }

  pub fn is_directional(self) -> bool {
    self >= PredictionMode::V_PRED && self <= PredictionMode::D63_PRED
  }

  pub fn predict_inter<'a>(
    self, fi: &FrameInvariants, p: usize, po: &PlaneOffset,
    dst: &'a mut PlaneMutSlice<'a>, width: usize, height: usize,
    ref_frames: [usize; 2], mvs: [MotionVector; 2]
  ) {
    assert!(!self.is_intra());

    let mode = FilterMode::REGULAR;
    let is_compound =
      ref_frames[1] > INTRA_FRAME && ref_frames[1] != NONE_FRAME;

    fn get_params<'a>(
      rec_plane: &'a Plane, po: &PlaneOffset, mv: MotionVector
    ) -> (i32, i32, PlaneSlice<'a>) {
      let rec_cfg = &rec_plane.cfg;
      let shift_row = 3 + rec_cfg.ydec;
      let shift_col = 3 + rec_cfg.xdec;
      let row_offset = mv.row as i32 >> shift_row;
      let col_offset = mv.col as i32 >> shift_col;
      let row_frac =
        (mv.row as i32 - (row_offset << shift_row)) << (4 - shift_row);
      let col_frac =
        (mv.col as i32 - (col_offset << shift_col)) << (4 - shift_col);
      let qo = PlaneOffset {
        x: po.x + col_offset as isize - 3,
        y: po.y + row_offset as isize - 3
      };
      (row_frac, col_frac, rec_plane.slice(&qo).clamp().subslice(3, 3))
    };

    if !is_compound {
      match fi.rec_buffer.frames
        [fi.ref_frames[ref_frames[0] - LAST_FRAME] as usize]
      {
        Some(ref rec) => {
          let (row_frac, col_frac, src) =
            get_params(&rec.frame.planes[p], po, mvs[0]);
          put_8tap(
            dst,
            src,
            width,
            height,
            col_frac,
            row_frac,
            mode,
            mode,
            fi.sequence.bit_depth
          );
        }
        None => ()
      }
    } else {
      let mut tmp: [AlignedArray<[i16; 128 * 128]>; 2] =
        [UninitializedAlignedArray(), UninitializedAlignedArray()];
      for i in 0..2 {
        match fi.rec_buffer.frames
          [fi.ref_frames[ref_frames[i] - LAST_FRAME] as usize]
        {
          Some(ref rec) => {
            let (row_frac, col_frac, src) =
              get_params(&rec.frame.planes[p], po, mvs[i]);
            prep_8tap(
              &mut tmp[i].array,
              src,
              width,
              height,
              col_frac,
              row_frac,
              mode,
              mode,
              fi.sequence.bit_depth
            );
          }
          None => ()
        }
      }
      mc_avg(
        dst,
        &tmp[0].array,
        &tmp[1].array,
        width,
        height,
        fi.sequence.bit_depth
      );
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

pub fn has_tr(bo: &BlockOffset, bsize: BlockSize) -> bool {
  let sb_mi_size = BLOCK_64X64.width_mi(); /* Assume 64x64 for now */
  let mask_row = bo.y & LOCAL_BLOCK_MASK;
  let mask_col = bo.x & LOCAL_BLOCK_MASK;
  let target_n4_w = bsize.width_mi();
  let target_n4_h = bsize.height_mi();

  let mut bs = target_n4_w.max(target_n4_h);

  if bs > BLOCK_64X64.width_mi() {
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
  if (target_n4_w < target_n4_h) && (bo.x & target_n4_w) == 0 {
    has_tr = true;
  }

  /* The bottom of two horizontal rectangles never has a top right (as the block
    * to the right won't have been decoded) */
  if (target_n4_w > target_n4_h) && (bo.y & target_n4_h) != 0 {
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

pub fn has_bl(bo: &BlockOffset, bsize: BlockSize) -> bool {
  let sb_mi_size = BLOCK_64X64.width_mi(); /* Assume 64x64 for now */
  let mask_row = bo.y & LOCAL_BLOCK_MASK;
  let mask_col = bo.x & LOCAL_BLOCK_MASK;
  let target_n4_w = bsize.width_mi();
  let target_n4_h = bsize.height_mi();

  let mut bs = target_n4_w.max(target_n4_h);

  if bs > BLOCK_64X64.width_mi() {
    return false;
  }

  let mut has_bl = (mask_row & bs) == 0 && (mask_col & bs) == 0 && bs < sb_mi_size;

  /* TODO: assert its a power of two */

  while 2 * bs < sb_mi_size {
    if (mask_col & bs) == 0 {
      if (mask_col & (2 * bs) == 0) && (mask_row & (2 * bs) == 0) {
        has_bl = true;
        break;
      }
    } else {
      break;
    }
    bs <<= 1;
  }

  /* The right hand of two vertical rectangles never has a bottom left (as the
    * block below won't have been decoded) */
  if (target_n4_w < target_n4_h) && (bo.x & target_n4_w) != 0 {
    has_bl = false;
  }

  /* The top of two horizontal rectangles always has a bottom left (as the block
    * to the left will have been decoded) */
  if (target_n4_w > target_n4_h) && (bo.y & target_n4_h) == 0 {
    has_bl = true;
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

  has_bl
}
