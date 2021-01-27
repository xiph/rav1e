// Copyright (c) 2017-2020, The rav1e contributors. All rights reserved
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
use crate::scan_order::*;
use crate::tiling::*;
use crate::token_cdfs::*;
use crate::transform::TxSize::*;

use crate::transform::*;
use crate::util::*;

use arrayvec::*;
use std::collections::HashMap;
use std::default::Default;
use std::ops::{Add, Index, IndexMut};
use std::*;

const MAX_REF_MV_STACK_SIZE: usize = 8;
pub const REF_CAT_LEVEL: u32 = 640;

pub const FRAME_LF_COUNT: usize = 4;
pub const MAX_LOOP_FILTER: usize = 63;
const DELTA_LF_SMALL: u32 = 3;
pub const DELTA_LF_PROBS: usize = DELTA_LF_SMALL as usize;

const DELTA_Q_SMALL: u32 = 3;
pub const DELTA_Q_PROBS: usize = DELTA_Q_SMALL as usize;

static size_group_lookup: [u8; BlockSize::BLOCK_SIZES_ALL] =
  [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 2, 2];

static num_pels_log2_lookup: [u8; BlockSize::BLOCK_SIZES_ALL] =
  [4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 6, 6, 8, 8, 10, 10];

#[macro_use]
mod cdf_context;
pub use cdf_context::*;

mod partition_unit;
pub use partition_unit::*;

mod superblock_unit;
pub use superblock_unit::*;

mod transform_unit;
pub use transform_unit::TxClass::*;
pub use transform_unit::*;

mod block_unit;
pub use block_unit::*;

mod frame_header;
pub use frame_header::*;

#[inline]
pub fn av1_get_coded_tx_size(tx_size: TxSize) -> TxSize {
  match tx_size {
    TX_64X64 | TX_64X32 | TX_32X64 => TX_32X32,
    TX_16X64 => TX_16X32,
    TX_64X16 => TX_32X16,
    _ => tx_size,
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
pub const MV_UPP: i32 = 1 << MV_IN_USE_BITS;
pub const MV_LOW: i32 = -(1 << MV_IN_USE_BITS);

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
  assert!(MV_LOW <= comp && comp <= MV_UPP);
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
