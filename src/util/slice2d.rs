// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(unused)]

use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::{fmt, slice};

#[derive(Copy, Clone)]
pub struct Slice2D<'a, T> {
  ptr: *const T,
  width: usize,
  height: usize,
  stride: usize,
  phantom: PhantomData<&'a T>,
}

#[derive(Copy, Clone)]
pub struct Slice2DMut<'a, T> {
  ptr: *mut T,
  width: usize,
  height: usize,
  stride: usize,
  phantom: PhantomData<&'a T>,
}

impl<'a, T> Slice2D<'a, T> {
  #[inline(always)]
  pub fn new(
    ptr: *const T, width: usize, height: usize, stride: usize,
  ) -> Self {
    assert!(width <= stride);
    Self { ptr, width, height, stride, phantom: PhantomData }
  }

  #[inline(always)]
  pub const fn as_ptr(&self) -> *const T {
    self.ptr
  }

  #[inline(always)]
  pub const fn width(&self) -> usize {
    self.width
  }

  #[inline(always)]
  pub const fn height(&self) -> usize {
    self.height
  }

  #[inline(always)]
  pub const fn stride(&self) -> usize {
    self.stride
  }

  pub fn rows_iter(&self) -> RowsIter<'_, T> {
    RowsIter {
      data: self.as_ptr(),
      stride: self.stride(),
      width: self.width(),
      remaining: self.height(),
      phantom: PhantomData,
    }
  }
}

impl<T> Index<usize> for Slice2D<'_, T> {
  type Output = [T];
  #[inline(always)]
  fn index(&self, index: usize) -> &Self::Output {
    assert!(index < self.height);
    unsafe {
      slice::from_raw_parts(self.ptr.add(index * self.stride), self.width)
    }
  }
}

impl<T> fmt::Debug for Slice2D<'_, T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "Slice2D {{ ptr: {:?}, size: {}({})x{} }}",
      self.ptr, self.width, self.stride, self.height
    )
  }
}

// Functions shared with Slice2D
impl<'a, T> Slice2DMut<'a, T> {
  #[inline(always)]
  pub const fn as_ptr(&self) -> *const T {
    self.ptr
  }

  #[inline(always)]
  pub const fn width(&self) -> usize {
    self.width
  }

  #[inline(always)]
  pub const fn height(&self) -> usize {
    self.height
  }

  #[inline(always)]
  pub const fn stride(&self) -> usize {
    self.stride
  }

  pub fn rows_iter(&self) -> RowsIter<'_, T> {
    RowsIter {
      data: self.as_ptr(),
      stride: self.stride(),
      width: self.width(),
      remaining: self.height(),
      phantom: PhantomData,
    }
  }
}

// Mutable functions
impl<'a, T> Slice2DMut<'a, T> {
  #[inline(always)]
  pub fn new(ptr: *mut T, width: usize, height: usize, stride: usize) -> Self {
    assert!(width <= stride);
    Self { ptr: ptr as *mut T, width, height, stride, phantom: PhantomData }
  }

  pub const fn as_const(self) -> Slice2D<'a, T> {
    Slice2D {
      ptr: self.ptr,
      width: self.width,
      height: self.height,
      stride: self.stride,
      phantom: PhantomData,
    }
  }

  pub fn as_mut_ptr(&mut self) -> *mut T {
    self.ptr
  }

  pub fn rows_iter_mut(&mut self) -> RowsIterMut<'_, T> {
    RowsIterMut {
      data: self.as_mut_ptr(),
      stride: self.stride(),
      width: self.width(),
      remaining: self.height(),
      phantom: PhantomData,
    }
  }
}

impl<T> Index<usize> for Slice2DMut<'_, T> {
  type Output = [T];
  #[inline(always)]
  fn index(&self, index: usize) -> &Self::Output {
    assert!(index < self.height());
    unsafe {
      let ptr = self.as_ptr().add(index * self.stride());
      slice::from_raw_parts(ptr, self.width())
    }
  }
}

impl<T> IndexMut<usize> for Slice2DMut<'_, T> {
  #[inline(always)]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    assert!(index < self.height());
    unsafe {
      let ptr = self.as_mut_ptr().add(index * self.stride());
      slice::from_raw_parts_mut(ptr, self.width())
    }
  }
}

impl<T> fmt::Debug for Slice2DMut<'_, T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "Slice2D {{ ptr: {:?}, size: {}({})x{} }}",
      self.ptr, self.width, self.stride, self.height
    )
  }
}

/// Iterator over rows
pub struct RowsIter<'a, T> {
  data: *const T,
  stride: usize,
  width: usize,
  remaining: usize,
  phantom: PhantomData<&'a T>,
}

/// Mutable iterator over rows
pub struct RowsIterMut<'a, T> {
  data: *mut T,
  stride: usize,
  width: usize,
  remaining: usize,
  phantom: PhantomData<&'a mut T>,
}

impl<'a, T> Iterator for RowsIter<'a, T> {
  type Item = &'a [T];

  #[inline(always)]
  fn next(&mut self) -> Option<Self::Item> {
    if self.remaining > 0 {
      let row = unsafe {
        let ptr = self.data;
        self.data = self.data.add(self.stride);
        slice::from_raw_parts(ptr, self.width)
      };
      self.remaining -= 1;
      Some(row)
    } else {
      None
    }
  }

  #[inline(always)]
  fn size_hint(&self) -> (usize, Option<usize>) {
    (self.remaining, Some(self.remaining))
  }
}

impl<'a, T> Iterator for RowsIterMut<'a, T> {
  type Item = &'a mut [T];

  #[inline(always)]
  fn next(&mut self) -> Option<Self::Item> {
    if self.remaining > 0 {
      let row = unsafe {
        let ptr = self.data;
        self.data = self.data.add(self.stride);
        slice::from_raw_parts_mut(ptr, self.width)
      };
      self.remaining -= 1;
      Some(row)
    } else {
      None
    }
  }

  #[inline(always)]
  fn size_hint(&self) -> (usize, Option<usize>) {
    (self.remaining, Some(self.remaining))
  }
}

impl<T> ExactSizeIterator for RowsIter<'_, T> {}
impl<T> FusedIterator for RowsIter<'_, T> {}
impl<T> ExactSizeIterator for RowsIterMut<'_, T> {}
impl<T> FusedIterator for RowsIterMut<'_, T> {}
