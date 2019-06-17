// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use std::alloc::{alloc, dealloc, Layout};
use std::iter::FusedIterator;
use std::fmt::{Debug, Display, Formatter};
use std::marker::PhantomData;
use std::mem;
use std::ops::{Index, IndexMut, Range};

use crate::tiling::*;
use crate::util::*;

/// Plane-specific configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaneConfig {
  pub stride: usize,
  pub alloc_height: usize,
  pub width: usize,
  pub height: usize,
  pub xdec: usize,
  pub ydec: usize,
  pub xpad: usize,
  pub ypad: usize,
  pub xorigin: usize,
  pub yorigin: usize
}

/// Absolute offset in pixels inside a plane
#[derive(Clone, Copy, Debug)]
pub struct PlaneOffset {
  pub x: isize,
  pub y: isize
}

#[derive(Debug, PartialEq, Eq)]
pub struct PlaneData<T: Pixel> {
  ptr: std::ptr::NonNull<T>,
  _marker: PhantomData<T>,
  len: usize,
}

unsafe impl<T: Pixel + Send> Send for PlaneData<T> { }
unsafe impl<T: Pixel + Sync> Sync for PlaneData<T> { }

impl<T: Pixel> Clone for PlaneData<T> {
  fn clone(&self) -> Self {
    let mut pd = unsafe { Self::new_uninitialized(self.len) };

    pd.copy_from_slice(self);

    pd
  }
}

impl<T: Pixel> std::ops::Deref for PlaneData<T> {
  type Target = [T];

  fn deref(&self) -> &[T] {
    unsafe {
      let p = self.ptr.as_ptr();

      std::slice::from_raw_parts(p, self.len)
    }
  }
}

impl<T: Pixel> std::ops::DerefMut for PlaneData<T> {
  fn deref_mut(&mut self) -> &mut [T] {
    unsafe {
      let p = self.ptr.as_ptr();

      std::slice::from_raw_parts_mut(p, self.len)
    }
  }
}

impl<T: Pixel> std::ops::Drop for PlaneData<T> {
  fn drop(&mut self) {
    unsafe {
      dealloc(self.ptr.as_ptr() as *mut u8, Self::layout(self.len));
    }
  }
}

impl<T: Pixel> PlaneData<T> {
  /// Data alignment in bytes.
  const DATA_ALIGNMENT_LOG2: usize = 5;

  unsafe fn layout(len: usize) -> Layout {
    Layout::from_size_align_unchecked(
      len * mem::size_of::<T>(),
      1 << Self::DATA_ALIGNMENT_LOG2
    )
  }

  unsafe fn new_uninitialized(len: usize) -> Self {
    let ptr = {
      let ptr = alloc(Self::layout(len)) as *mut T;
      std::ptr::NonNull::new_unchecked(ptr)
    };

    PlaneData {
      ptr,
      len,
      _marker: PhantomData
    }
  }

  pub fn new(len: usize) -> Self {
    let mut pd = unsafe { Self::new_uninitialized(len) };

    for v in pd.iter_mut() {
      *v = T::cast_from(128);
    }

    pd
  }

  pub fn from_slice(data: &[T]) -> Self {
    let mut pd = unsafe { Self::new_uninitialized(data.len()) };

    pd.copy_from_slice(data);

    pd
  }
}

#[derive(Clone, PartialEq, Eq)]
pub struct Plane<T: Pixel> {
  pub data: PlaneData<T>,
  pub cfg: PlaneConfig
}

impl<T: Pixel> Debug for Plane<T>
    where T: Display {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    write!(f, "Plane {{ data: [{}, ...], cfg: {:?} }}", self.data[0], self.cfg)
  }
}

impl<T: Pixel> Plane<T> {
  /// Stride alignment in bytes.
  const STRIDE_ALIGNMENT_LOG2: usize = 5;

  pub fn new(
    width: usize, height: usize, xdec: usize, ydec: usize, xpad: usize,
    ypad: usize
  ) -> Self {
    let xorigin = xpad.align_power_of_two(
      Self::STRIDE_ALIGNMENT_LOG2 + 1 - mem::size_of::<T>()
    );
    let yorigin = ypad;
    let stride = (xorigin + width + xpad).align_power_of_two(
      Self::STRIDE_ALIGNMENT_LOG2 + 1 - mem::size_of::<T>()
    );
    let alloc_height = yorigin + height + ypad;
    let data = PlaneData::new(stride * alloc_height);

    Plane {
      data,
      cfg: PlaneConfig {
        stride,
        alloc_height,
        width,
        height,
        xdec,
        ydec,
        xpad,
        ypad,
        xorigin,
        yorigin
      }
    }
  }

  pub fn wrap(data: Vec<T>, stride: usize) -> Self {
    let len = data.len();

    assert!(len % stride == 0);

    Self {
      data: PlaneData::from_slice(&data),
      cfg: PlaneConfig {
        stride,
        alloc_height: len / stride,
        width: stride,
        height: len / stride,
        xdec: 0,
        ydec: 0,
        xpad: 0,
        ypad: 0,
        xorigin: 0,
        yorigin: 0,
      }
    }
  }

  pub fn pad(&mut self, w: usize, h: usize) {
    let xorigin = self.cfg.xorigin;
    let yorigin = self.cfg.yorigin;
    let stride = self.cfg.stride;
    let alloc_height = self.cfg.alloc_height;
    let width = w >> self.cfg.xdec;
    let height = h >> self.cfg.ydec;

    if xorigin > 0 {
      for y in 0..height {
        let base = (yorigin + y) * stride;
        let fill_val = self.data[base + xorigin];
        for val in &mut self.data[base..base + xorigin] {
          *val = fill_val;
        }
      }
    }

    if xorigin + width < stride {
      for y in 0..height {
        let base = (yorigin + y) * stride + xorigin + width;
        let fill_val = self.data[base - 1];
        for val in &mut self.data[base..base + stride - (xorigin + width)] {
          *val = fill_val;
        }
      }
    }

    if yorigin > 0 {
      let (top, bottom) = self.data.split_at_mut(yorigin * stride);
      let src = &bottom[..stride];
      for y in 0..yorigin {
        let dst = &mut top[y * stride..(y + 1) * stride];
        dst.copy_from_slice(src);
      }
    }

    if yorigin + height < self.cfg.alloc_height {
      let (top, bottom) = self.data.split_at_mut((yorigin + height) * stride);
      let src = &top[(yorigin + height - 1) * stride..];
      for y in 0..alloc_height - (yorigin + height) {
        let dst = &mut bottom[y * stride..(y + 1) * stride];
        dst.copy_from_slice(src);
      }
    }
  }

  pub fn slice(&self, po: PlaneOffset) -> PlaneSlice<'_, T> {
    PlaneSlice { plane: self, x: po.x, y: po.y }
  }

  pub fn mut_slice(&mut self, po: PlaneOffset) -> PlaneMutSlice<'_, T> {
    PlaneMutSlice { plane: self, x: po.x, y: po.y }
  }

  pub fn as_slice(&self) -> PlaneSlice<'_, T> {
    self.slice(PlaneOffset { x: 0, y: 0 })
  }

  pub fn as_mut_slice(&mut self) -> PlaneMutSlice<'_, T> {
    self.mut_slice(PlaneOffset { x: 0, y: 0 })
  }

  #[inline(always)]
  pub fn region(&self, area: Area) -> PlaneRegion<'_, T> {
    let rect = area.to_rect(
      self.cfg.xdec,
      self.cfg.ydec,
      self.cfg.stride - self.cfg.xorigin as usize,
      self.cfg.alloc_height - self.cfg.yorigin as usize,
    );
    PlaneRegion::new(self, rect)
  }

  #[inline(always)]
  pub fn region_mut(&mut self, area: Area) -> PlaneRegionMut<'_, T> {
    let rect = area.to_rect(
      self.cfg.xdec,
      self.cfg.ydec,
      self.cfg.stride - self.cfg.xorigin as usize,
      self.cfg.alloc_height - self.cfg.yorigin as usize,
    );
    PlaneRegionMut::new(self, rect)
  }

  #[inline(always)]
  pub fn as_region(&self) -> PlaneRegion<'_, T> {
    self.region(Area::StartingAt { x: 0, y: 0 })
  }

  #[inline(always)]
  pub fn as_region_mut(&mut self) -> PlaneRegionMut<'_, T> {
    self.region_mut(Area::StartingAt { x: 0, y: 0 })
  }

  #[inline]
  fn index(&self, x: usize, y: usize) -> usize {
    (y + self.cfg.yorigin) * self.cfg.stride + (x + self.cfg.xorigin)
  }

  #[inline]
  pub fn row_range(&self, x: isize, y: isize) -> Range<usize> {
    debug_assert!(self.cfg.yorigin as isize + y >= 0);
    debug_assert!(self.cfg.xorigin as isize + x >= 0);
    let base_y = (self.cfg.yorigin as isize + y) as usize;
    let base_x = (self.cfg.xorigin as isize + x) as usize;
    let base = base_y * self.cfg.stride + base_x;
    let width = self.cfg.stride - base_x;
    base..base + width
  }


  pub fn p(&self, x: usize, y: usize) -> T {
    self.data[self.index(x, y)]
  }

  pub fn data_origin(&self) -> &[T] {
    &self.data[self.index(0, 0)..]
  }

  pub fn data_origin_mut(&mut self) -> &mut [T] {
    let i = self.index(0, 0);
    &mut self.data[i..]
  }

  pub fn copy_from_raw_u8(
    &mut self, source: &[u8], source_stride: usize, source_bytewidth: usize
  ) {
    let stride = self.cfg.stride;
    for (self_row, source_row) in self
      .data_origin_mut()
      .chunks_mut(stride)
      .zip(source.chunks(source_stride))
    {
      match source_bytewidth {
        1 => for (self_pixel, source_pixel) in
          self_row.iter_mut().zip(source_row.iter())
        {
          *self_pixel = T::cast_from(*source_pixel);
        },
        2 => {
          assert!(mem::size_of::<T>() >= 2, "source bytewidth ({}) cannot fit in Plane<u8>", source_bytewidth);
          for (self_pixel, bytes) in
            self_row.iter_mut().zip(source_row.chunks(2))
          {
            *self_pixel = T::cast_from(u16::cast_from(bytes[1]) << 8 | u16::cast_from(bytes[0]));
          }
        },

        _ => {}
      }
    }
  }

  pub fn downsample_from(&mut self, src: &Plane<T>) {
    let width = self.cfg.width;
    let height = self.cfg.height;
    let xorigin = self.cfg.xorigin;
    let yorigin = self.cfg.yorigin;
    let stride = self.cfg.stride;

    assert!(width * 2 == src.cfg.width);
    assert!(height * 2 == src.cfg.height);

    for row in 0..height {
      let base = (yorigin + row) * stride + xorigin;
      let dst = &mut self.data[base..base + width];

      for col in 0..width {
        let mut sum = 0;
        sum += u32::cast_from(src.p(2 * col, 2 * row));
        sum += u32::cast_from(src.p(2 * col + 1, 2 * row));
        sum += u32::cast_from(src.p(2 * col, 2 * row + 1));
        sum += u32::cast_from(src.p(2 * col + 1, 2 * row + 1));
        let avg = (sum + 2) >> 2;
        dst[col] = T::cast_from(avg);
      }
    }
  }

  /// Iterates over the pixels in the `Plane`, skipping stride data.
  pub fn iter(&self) -> PlaneIter<'_, T> {
    PlaneIter::new(self)
  }
}

#[derive(Debug)]
pub struct PlaneIter<'a, T: Pixel> {
  plane: &'a Plane<T>,
  y: usize,
  x: usize,
}

impl<'a, T: Pixel> PlaneIter<'a, T> {
  pub fn new(plane: &'a Plane<T>) -> Self {
    Self {
      plane,
      y: 0,
      x: 0,
    }
  }

  fn width(&self) -> usize {
    self.plane.cfg.width
  }

  fn height(&self) -> usize {
    self.plane.cfg.height
  }
}

impl<'a, T: Pixel> Iterator for PlaneIter<'a, T> {
  type Item = T;

  fn next(&mut self) -> Option<<Self as Iterator>::Item> {
    if self.y == self.height() {
      return None;
    }
    let pixel = self.plane.p(self.x, self.y);
    if self.x == self.width() - 1 {
      self.x = 0;
      self.y += 1;
    } else {
      self.x += 1;
    }
    Some(pixel)
  }
}

#[derive(Clone, Copy, Debug)]
pub struct PlaneSlice<'a, T: Pixel> {
  pub plane: &'a Plane<T>,
  pub x: isize,
  pub y: isize
}

pub struct IterWidth<'a, T: Pixel> {
    ps: PlaneSlice<'a, T>,
    width: usize,
}

impl<'a, T: Pixel> Iterator for IterWidth<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let x = self.ps.plane.cfg.xorigin as isize + self.ps.x;
        let y = self.ps.plane.cfg.yorigin as isize + self.ps.y;
        let stride = self.ps.plane.cfg.stride;
        let base = y as usize * stride + x as usize;

        if self.ps.plane.data.len() < base + self.width {
            None
        } else {
            self.ps.y += 1;
            Some(&self.ps.plane.data[base..base + self.width])
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.ps.plane.cfg.height - self.ps.y as usize;

        (size, Some(size))
    }
}

impl<'a, T: Pixel> ExactSizeIterator for IterWidth<'a, T> { }

impl<'a, T: Pixel> FusedIterator for IterWidth<'a, T> { }

pub struct RowsIter<'a, T: Pixel> {
  plane: &'a Plane<T>,
  x: isize,
  y: isize,
}

impl<'a, T: Pixel> Iterator for RowsIter<'a, T> {
  type Item = &'a [T];

  fn next(&mut self) -> Option<Self::Item> {
    if self.plane.cfg.height as isize > self.y {
      // cannot directly return self.ps.row(row) due to lifetime issue
      let range = self.plane.row_range(self.x, self.y);
      self.y += 1;
      Some(&self.plane.data[range])
    } else {
      None
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    let remaining = self.plane.cfg.height as isize - self.y;
    debug_assert!(remaining >= 0);
    let remaining = remaining as usize;

    (remaining, Some(remaining))
  }
}

impl<'a, T: Pixel> ExactSizeIterator for RowsIter<'a, T> {}
impl<'a, T: Pixel> FusedIterator for RowsIter<'a, T> {}

impl<'a, T: Pixel> PlaneSlice<'a, T> {
  pub fn as_ptr(&self) -> *const T {
    self[0].as_ptr()
  }

  pub fn rows_iter(&self) -> RowsIter<'_, T> {
    RowsIter {
      plane: self.plane,
      x: self.x,
      y: self.y,
    }
  }

  pub fn clamp(&self) -> PlaneSlice<'a, T> {
    PlaneSlice {
      plane: self.plane,
      x: self
        .x
        .min(self.plane.cfg.width as isize)
        .max(-(self.plane.cfg.xorigin as isize)),
      y: self
        .y
        .min(self.plane.cfg.height as isize)
        .max(-(self.plane.cfg.yorigin as isize))
    }
  }

  pub fn iter_width(&self, width: usize) -> IterWidth<'a, T> {
    IterWidth { ps: *self, width }
  }

  pub fn subslice(&self, xo: usize, yo: usize) -> PlaneSlice<'a, T> {
    PlaneSlice {
      plane: self.plane,
      x: self.x + xo as isize,
      y: self.y + yo as isize
    }
  }

  pub fn reslice(&self, xo: isize, yo: isize) -> PlaneSlice<'a, T> {
    PlaneSlice {
      plane: self.plane,
      x: self.x + xo,
      y: self.y + yo
    }
  }

  /// A slice starting i pixels above the current one.
  pub fn go_up(&self, i: usize) -> PlaneSlice<'a, T> {
    PlaneSlice { plane: self.plane, x: self.x, y: self.y - i as isize }
  }

  /// A slice starting i pixels to the left of the current one.
  pub fn go_left(&self, i: usize) -> PlaneSlice<'a, T> {
    PlaneSlice { plane: self.plane, x: self.x - i as isize, y: self.y }
  }

  pub fn p(&self, add_x: usize, add_y: usize) -> T {
    let new_y =
      (self.y + add_y as isize + self.plane.cfg.yorigin as isize) as usize;
    let new_x =
      (self.x + add_x as isize + self.plane.cfg.xorigin as isize) as usize;
    self.plane.data[new_y * self.plane.cfg.stride + new_x]
  }
}

impl<'a, T: Pixel> Index<usize> for PlaneSlice<'a, T> {
  type Output = [T];
  fn index(&self, index: usize) -> &Self::Output {
    let range = self.plane.row_range(self.x, self.y + index as isize);
    &self.plane.data[range]
  }
}

pub struct PlaneMutSlice<'a, T: Pixel> {
  pub plane: &'a mut Plane<T>,
  pub x: isize,
  pub y: isize
}

pub struct RowsIterMut<'a, T: Pixel> {
  plane: *mut Plane<T>,
  x: isize,
  y: isize,
  phantom: PhantomData<&'a mut Plane<T>>,
}

impl<'a, T: Pixel> Iterator for RowsIterMut<'a, T> {
  type Item = &'a mut [T];

  fn next(&mut self) -> Option<Self::Item> {
    // there could not be a concurrent call using a mutable reference to the plane
    let plane = unsafe { &mut *self.plane };
    if plane.cfg.height as isize > self.y {
      // cannot directly return self.ps.row(row) due to lifetime issue
      let range = plane.row_range(self.x, self.y);
      self.y += 1;
      Some(&mut plane.data[range])
    } else {
      None
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    // there could not be a concurrent call using a mutable reference to the plane
    let plane = unsafe { &mut *self.plane };
    let remaining = plane.cfg.height as isize - self.y;
    debug_assert!(remaining >= 0);
    let remaining = remaining as usize;

    (remaining, Some(remaining))
  }
}

impl<'a, T: Pixel> ExactSizeIterator for RowsIterMut<'a, T> {}
impl<'a, T: Pixel> FusedIterator for RowsIterMut<'a, T> {}

impl<'a, T: Pixel> PlaneMutSlice<'a, T> {
  pub fn as_ptr(&self) -> *const T {
    self[0].as_ptr()
  }

  pub fn as_mut_ptr(&mut self) -> *mut T {
    self[0].as_mut_ptr()
  }

  pub fn rows_iter(&self) -> RowsIter<'_, T> {
    RowsIter {
      plane: self.plane,
      x: self.x,
      y: self.y,
    }
  }

  pub fn rows_iter_mut(&mut self) -> RowsIterMut<'_, T> {
    RowsIterMut {
      plane: self.plane as *mut Plane<T>,
      x: self.x,
      y: self.y,
      phantom: PhantomData,
    }
  }

  // FIXME: code duplication with PlaneSlice
  pub fn p(&self, add_x: usize, add_y: usize) -> T {
    let new_y =
      (self.y + add_y as isize + self.plane.cfg.yorigin as isize) as usize;
    let new_x =
      (self.x + add_x as isize + self.plane.cfg.xorigin as isize) as usize;
    self.plane.data[new_y * self.plane.cfg.stride + new_x]
  }
}

impl<'a, T: Pixel> Index<usize> for PlaneMutSlice<'a, T> {
  type Output = [T];
  fn index(&self, index: usize) -> &Self::Output {
    let range = self.plane.row_range(self.x, self.y + index as isize);
    &self.plane.data[range]
  }
}

impl<'a, T: Pixel> IndexMut<usize> for PlaneMutSlice<'a, T> {
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    let range = self.plane.row_range(self.x, self.y + index as isize);
    &mut self.plane.data[range]
  }
}

#[cfg(test)]
pub mod test {
  use super::*;

  #[test]
  fn copy_from_raw_u8() {
    let mut plane = Plane::wrap(
    vec![
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 2, 3, 4, 0, 0,
        0, 0, 8, 7, 6, 5, 0, 0,
        0, 0, 9, 8, 7, 6, 0, 0,
        0, 0, 2, 3, 4, 5, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    ], 8);

    let input = vec![42u8; 64];

    plane.copy_from_raw_u8(&input, 8, 1);

    println!("{:?}", &plane.data[..10]);

    assert_eq!(&input[..64], &plane.data[..64]);
  }

  #[test]
  fn test_plane_pad() {
    let mut plane = Plane::<u8> {
      data: PlaneData::from_slice(&vec![
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 2, 3, 4, 0, 0,
        0, 0, 8, 7, 6, 5, 0, 0,
        0, 0, 9, 8, 7, 6, 0, 0,
        0, 0, 2, 3, 4, 5, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
      ]),
      cfg: PlaneConfig {
        stride: 8,
        alloc_height: 9,
        width: 4,
        height: 4,
        xdec: 0,
        ydec: 0,
        xpad: 0,
        ypad: 0,
        xorigin: 2,
        yorigin: 3,
      },
    };
    plane.pad(4, 4);
    assert_eq!(
      &[
        1u8, 1, 1, 2, 3, 4, 4, 4,
        1, 1, 1, 2, 3, 4, 4, 4,
        1, 1, 1, 2, 3, 4, 4, 4,
        1, 1, 1, 2, 3, 4, 4, 4,
        8, 8, 8, 7, 6, 5, 5, 5,
        9, 9, 9, 8, 7, 6, 6, 6,
        2, 2, 2, 3, 4, 5, 5, 5,
        2, 2, 2, 3, 4, 5, 5, 5,
        2, 2, 2, 3, 4, 5, 5, 5,
      ][..],
      &plane.data[..]
    );
  }
}
