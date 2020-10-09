// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use std::mem::MaybeUninit;
use std::ops::Bound;
use std::ops::RangeBounds;

#[repr(transparent)]
pub struct Uninit<T>(MaybeUninit<T>);

pub fn init_slice_repeat_mut<T: Copy>(
  slice: &'_ mut [MaybeUninit<T>], value: T,
) -> &'_ mut [T] {
  // Fill all of slice
  for a in slice.iter_mut() {
    *a = MaybeUninit::new(value);
  }

  // Defined behavior, since all elements of slice are initialized
  unsafe { assume_slice_init_mut(slice) }
}

pub trait Init<T> {
  fn init_slice_mut<S: RangeBounds<usize>>(
    &mut self, range: S, value: T,
  ) -> &mut [T]
  where
    T: Copy;
}

impl<T> Init<T> for [Uninit<T>] {
  fn init_slice_mut<S: RangeBounds<usize>>(
    &mut self, range: S, value: T,
  ) -> &mut [T]
  where
    T: Copy,
  {
    let slice = match range.end_bound() {
      Bound::Included(&i) => &mut self[..=i],
      Bound::Excluded(&i) => &mut self[..i],
      Bound::Unbounded => self,
    };
    let slice = match range.start_bound() {
      Bound::Included(&i) => &mut slice[i..],
      Bound::Excluded(&i) => &mut slice[(i + 1)..],
      Bound::Unbounded => slice,
    };
    for x in slice.iter_mut() {
      x.0 = MaybeUninit::new(value);
    }
    unsafe { force_init_slice_mut(slice) }
  }
}

unsafe fn force_init_slice_mut<T: Copy>(
  slice: &'_ mut [Uninit<T>],
) -> &'_ mut [T] {
  &mut *(slice as *mut [Uninit<T>] as *mut [T])
}

/// Assume all the elements are initialized
pub unsafe fn assume_slice_init_mut<T: Copy>(
  slice: &'_ mut [MaybeUninit<T>],
) -> &'_ mut [T] {
  &mut *(slice as *mut [std::mem::MaybeUninit<T>] as *mut [T])
}
