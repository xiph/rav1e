// Copyright (c) 2017-2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use num_traits::PrimInt;
use std::mem::size_of;

pub trait Fixed {
  fn floor_log2(&self, n: usize) -> usize;
  fn ceil_log2(&self, n: usize) -> usize;
  fn align_power_of_two(&self, n: usize) -> usize;
  fn align_power_of_two_and_shift(&self, n: usize) -> usize;
}

impl Fixed for usize {
  #[inline]
  fn floor_log2(&self, n: usize) -> usize {
    self & !((1 << n) - 1)
  }
  #[inline]
  fn ceil_log2(&self, n: usize) -> usize {
    (self + (1 << n) - 1).floor_log2(n)
  }
  #[inline]
  fn align_power_of_two(&self, n: usize) -> usize {
    self.ceil_log2(n)
  }
  #[inline]
  fn align_power_of_two_and_shift(&self, n: usize) -> usize {
    (self + (1 << n) - 1) >> n
  }
}

pub fn clamp<T: PartialOrd>(input: T, min: T, max: T) -> T {
  if input < min {
    min
  } else if input > max {
    max
  } else {
    input
  }
}

pub trait ILog: PrimInt {
  // Integer binary logarithm of an integer value.
  // Returns floor(log2(self)) + 1, or 0 if self == 0.
  // This is the number of bits that would be required to represent self in two's
  //  complement notation with all of the leading zeros stripped.
  // TODO: Mark const once trait functions can be constant
  fn ilog(self) -> usize {
    size_of::<Self>() * 8 - self.leading_zeros() as usize
  }
}

impl<T> ILog for T where T: PrimInt {}

#[inline(always)]
pub fn msb(x: i32) -> i32 {
  debug_assert!(x > 0);
  31 ^ (x.leading_zeros() as i32)
}

#[inline(always)]
pub const fn round_shift(value: i32, bit: usize) -> i32 {
  (value + (1 << bit >> 1)) >> bit
}
