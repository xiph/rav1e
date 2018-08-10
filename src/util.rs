// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#[repr(align(16))]
struct Align16;

/// A 16 byte aligned array.
/// # Examples
/// ```
/// extern crate rav1e;
/// use rav1e::util::*;
///
/// let mut x: AlignedArray<[i16; 64 * 64]> = AlignedArray([0; 64 * 64]);
/// assert!(x.array.as_ptr() as usize % 16 == 0);
///
/// let mut x: AlignedArray<[i16; 64 * 64]> = UninitializedAlignedArray();
/// assert!(x.array.as_ptr() as usize % 16 == 0);
/// ```
pub struct AlignedArray<ARRAY>
where
  ARRAY: ?Sized
{
  _alignment: [Align16; 0],
  pub array: ARRAY
}

#[allow(non_snake_case)]
pub fn AlignedArray<ARRAY>(array: ARRAY) -> AlignedArray<ARRAY> {
  AlignedArray { _alignment: [], array }
}

#[allow(non_snake_case)]
pub fn UninitializedAlignedArray<ARRAY>() -> AlignedArray<ARRAY> {
  use std::mem;
  AlignedArray(unsafe { mem::uninitialized() })
}

#[test]
fn sanity() {
  let a: AlignedArray<_> = AlignedArray([0u8; 3]);
  assert!(is_aligned(a.array.as_ptr(), 4));
}

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

/// Check alignment.
pub fn is_aligned<T>(ptr: *const T, n: usize) -> bool {
  return ((ptr as usize) & ((1 << n) - 1)) == 0;
}
