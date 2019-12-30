// Copyright (c) 2017-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use std::mem::MaybeUninit;

#[repr(align(32))]
pub struct Align32;

// A 16 byte aligned array.
// # Examples
// ```
// let mut x: AlignedArray<[i16; 64 * 64]> = AlignedArray::new([0; 64 * 64]);
// assert!(x.array.as_ptr() as usize % 16 == 0);
//
// let mut x: AlignedArray<[i16; 64 * 64]> = AlignedArray::uninitialized();
// assert!(x.array.as_ptr() as usize % 16 == 0);
// ```
pub struct AlignedArray<ARRAY> {
  _alignment: [Align32; 0],
  pub array: ARRAY,
}

impl<A> AlignedArray<A> {
  pub const fn new(array: A) -> Self {
    AlignedArray { _alignment: [], array }
  }
  #[allow(clippy::uninit_assumed_init)]
  pub fn uninitialized() -> Self {
    Self::new(unsafe { MaybeUninit::uninit().assume_init() })
  }
}

#[test]
fn sanity() {
  fn is_aligned<T>(ptr: *const T, n: usize) -> bool {
    ((ptr as usize) & ((1 << n) - 1)) == 0
  }

  let a: AlignedArray<_> = AlignedArray::new([0u8; 3]);
  assert!(is_aligned(a.array.as_ptr(), 4));
}
