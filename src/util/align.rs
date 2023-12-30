// Copyright (c) 2017-2022, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use std::alloc::{alloc, dealloc, Layout};
use std::mem::MaybeUninit;
use std::ptr;
use std::{fmt, mem};

#[repr(align(64))]
pub struct Align64;

// A 64 byte aligned piece of data.
// # Examples
// ```
// let mut x: Aligned<[i16; 64 * 64]> = Aligned::new([0; 64 * 64]);
// assert!(x.data.as_ptr() as usize % 16 == 0);
//
// let mut x: Aligned<[i16; 64 * 64]> = Aligned::uninitialized();
// assert!(x.data.as_ptr() as usize % 16 == 0);
// ```
pub struct Aligned<T> {
  _alignment: [Align64; 0],
  pub data: T,
}

#[cfg(any(test, feature = "bench"))]
impl<const N: usize, T> Aligned<[T; N]> {
  #[inline(always)]
  pub fn from_fn<F>(cb: F) -> Self
  where
    F: FnMut(usize) -> T,
  {
    Aligned { _alignment: [], data: std::array::from_fn(cb) }
  }
}

impl<const N: usize, T> Aligned<[MaybeUninit<T>; N]> {
  #[inline(always)]
  pub const fn uninit_array() -> Self {
    Aligned {
      _alignment: [],
      // SAFETY: Uninitialized [MaybeUninit<T>; N] is valid.
      data: unsafe { MaybeUninit::uninit().assume_init() },
    }
  }
}

impl<T> Aligned<T> {
  pub const fn new(data: T) -> Self {
    Aligned { _alignment: [], data }
  }
  #[allow(clippy::uninit_assumed_init)]
  /// # Safety
  ///
  /// The resulting `Aligned<T>` *must* be written to before it is read from.
  pub const unsafe fn uninitialized() -> Self {
    Self::new(MaybeUninit::uninit().assume_init())
  }
}

/// An analog to a Box<[T]> where the underlying slice is aligned.
/// Alignment is according to the architecture-specific SIMD constraints.
pub struct AlignedBoxedSlice<T> {
  ptr: std::ptr::NonNull<T>,
  len: usize,
}

impl<T> AlignedBoxedSlice<T> {
  // Data alignment in bytes.
  cfg_if::cfg_if! {
    if #[cfg(target_arch = "wasm32")] {
      // FIXME: wasm32 allocator fails for alignment larger than 3
      const DATA_ALIGNMENT_LOG2: usize = 3;
    } else {
      const DATA_ALIGNMENT_LOG2: usize = 6;
    }
  }

  const fn layout(len: usize) -> Layout {
    // SAFETY: We are ensuring that `align` is non-zero and is a multiple of 2.
    unsafe {
      Layout::from_size_align_unchecked(
        len * mem::size_of::<T>(),
        1 << Self::DATA_ALIGNMENT_LOG2,
      )
    }
  }

  fn alloc(len: usize) -> std::ptr::NonNull<T> {
    // SAFETY: We are not calling this with a null pointer, so it's safe.
    unsafe { ptr::NonNull::new_unchecked(alloc(Self::layout(len)) as *mut T) }
  }

  /// Creates a [`AlignedBoxedSlice`] with a slice of length [`len`] filled with
  /// [`val`].
  pub fn new(len: usize, val: T) -> Self
  where
    T: Clone,
  {
    let mut output = Self { ptr: Self::alloc(len), len };

    for a in output.iter_mut() {
      *a = val.clone();
    }

    output
  }
}

impl<T: fmt::Debug> fmt::Debug for AlignedBoxedSlice<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    fmt::Debug::fmt(&**self, f)
  }
}

impl<T> std::ops::Deref for AlignedBoxedSlice<T> {
  type Target = [T];

  fn deref(&self) -> &[T] {
    // SAFETY: We know that `self.ptr` is not null, and we know its length.
    unsafe {
      let p = self.ptr.as_ptr();

      std::slice::from_raw_parts(p, self.len)
    }
  }
}

impl<T> std::ops::DerefMut for AlignedBoxedSlice<T> {
  fn deref_mut(&mut self) -> &mut [T] {
    // SAFETY: We know that `self.ptr` is not null, and we know its length.
    unsafe {
      let p = self.ptr.as_ptr();

      std::slice::from_raw_parts_mut(p, self.len)
    }
  }
}

impl<T> std::ops::Drop for AlignedBoxedSlice<T> {
  fn drop(&mut self) {
    // SAFETY: We know that the contents of this struct are aligned and valid to drop.
    unsafe {
      for a in self.iter_mut() {
        ptr::drop_in_place(a)
      }

      dealloc(self.ptr.as_ptr() as *mut u8, Self::layout(self.len));
    }
  }
}

unsafe impl<T> Send for AlignedBoxedSlice<T> where T: Send {}
unsafe impl<T> Sync for AlignedBoxedSlice<T> where T: Sync {}

#[cfg(test)]
mod test {
  use super::*;

  fn is_aligned<T>(ptr: *const T, n: usize) -> bool {
    ((ptr as usize) & ((1 << n) - 1)) == 0
  }

  #[test]
  fn sanity_stack() {
    let a: Aligned<_> = Aligned::new([0u8; 3]);
    assert!(is_aligned(a.data.as_ptr(), 4));
  }

  #[test]
  fn sanity_heap() {
    let a: AlignedBoxedSlice<_> = AlignedBoxedSlice::new(3, 0u8);
    assert!(is_aligned(a.as_ptr(), 4));
  }
}
