// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use std::iter::FusedIterator;
use std::fmt::{Debug, Display, Formatter};
use std::marker::PhantomData;
use std::mem;
use std::ops::{Index, IndexMut, Range};

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
#[derive(Debug)]
pub struct PlaneOffset {
  pub x: isize,
  pub y: isize
}

#[derive(Clone, PartialEq, Eq)]
pub struct Plane<T: Pixel> {
  pub data: Vec<T>,
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

  /// Data alignment in bytes.
  const DATA_ALIGNMENT_LOG2: usize = 5;

  pub fn new(
    width: usize, height: usize, xdec: usize, ydec: usize, xpad: usize,
    ypad: usize
  ) -> Self {
    let xorigin = xpad.align_power_of_two(Self::DATA_ALIGNMENT_LOG2 + 1 - mem::size_of::<T>());
    let yorigin = ypad;
    let stride = (xorigin + width + xpad)
      .align_power_of_two(Self::STRIDE_ALIGNMENT_LOG2 + 1 - mem::size_of::<T>());
    let alloc_height = yorigin + height + ypad;
    let data = unsafe {
      let mut aligned_data = vec![AlignedArray([T::cast_from(128); 32]);
          (stride * alloc_height + 31) >> Self::DATA_ALIGNMENT_LOG2];
      let new_parts = Vec::from_raw_parts(aligned_data.as_mut_ptr() as *mut T,
          aligned_data.len() << Self::DATA_ALIGNMENT_LOG2,
          aligned_data.capacity() << Self::DATA_ALIGNMENT_LOG2);
      std::mem::forget(aligned_data);
      new_parts
    };
    assert!(is_aligned(data.as_ptr(), Self::DATA_ALIGNMENT_LOG2));
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
      data,
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

  pub fn slice(&self, po: &PlaneOffset) -> PlaneSlice<'_, T> {
    PlaneSlice { plane: self, x: po.x, y: po.y }
  }

  pub fn mut_slice(&mut self, po: &PlaneOffset) -> PlaneMutSlice<'_, T> {
    PlaneMutSlice { plane: self, x: po.x, y: po.y }
  }

  pub fn as_slice(&self) -> PlaneSlice<'_, T> {
    self.slice(&PlaneOffset { x: 0, y: 0 })
  }

  pub fn as_mut_slice(&mut self) -> PlaneMutSlice<'_, T> {
    self.mut_slice(&PlaneOffset { x: 0, y: 0 })
  }

  #[inline]
  fn index(&self, x: usize, y: usize) -> usize {
    (y + self.cfg.yorigin) * self.cfg.stride + (x + self.cfg.xorigin)
  }

  #[inline]
  fn row_range(&self, x: isize, y: isize) -> Range<usize> {
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
  pub fn row(&self, y: usize) -> &[T] {
    let range = self.plane.row_range(self.x, self.y + y as isize);
    &self.plane.data[range]
  }

  pub fn as_ptr(&self) -> *const T {
    self.row(0).as_ptr()
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
    self.row(index)
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
  pub fn row(&self, y: usize) -> &[T] {
    let range = self.plane.row_range(self.x, self.y + y as isize);
    &self.plane.data[range]
  }

  pub fn row_mut(&mut self, y: usize) -> &mut [T] {
    let range = self.plane.row_range(self.x, self.y + y as isize);
    &mut self.plane.data[range]
  }

  pub fn as_ptr(&self) -> *const T {
    self.row(0).as_ptr()
  }

  pub fn as_mut_ptr(&mut self) -> *mut T {
    self.row_mut(0).as_mut_ptr()
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

  /// A slice starting i pixels above the current one.
  pub fn go_up(&self, i: usize) -> PlaneSlice<'_, T> {
    PlaneSlice { plane: self.plane, x: self.x, y: self.y - i as isize }
  }

  /// A slice starting i pixels to the left of the current one.
  pub fn go_left(&self, i: usize) -> PlaneSlice<'_, T> {
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

impl<'a, T: Pixel> Index<usize> for PlaneMutSlice<'a, T> {
  type Output = [T];
  fn index(&self, index: usize) -> &Self::Output {
    self.row(index)
  }
}

impl<'a, T: Pixel> IndexMut<usize> for PlaneMutSlice<'a, T> {
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    self.row_mut(index)
  }
}

#[cfg(test)]
pub mod test {
  use super::*;

  #[test]
  fn test_plane_pad() {
    let mut plane = Plane::<u8> {
      data: vec![
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 2, 3, 4, 0, 0,
        0, 0, 8, 7, 6, 5, 0, 0,
        0, 0, 9, 8, 7, 6, 0, 0,
        0, 0, 2, 3, 4, 5, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
      ],
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
      vec![
        1, 1, 1, 2, 3, 4, 4, 4,
        1, 1, 1, 2, 3, 4, 4, 4,
        1, 1, 1, 2, 3, 4, 4, 4,
        1, 1, 1, 2, 3, 4, 4, 4,
        8, 8, 8, 7, 6, 5, 5, 5,
        9, 9, 9, 8, 7, 6, 6, 6,
        2, 2, 2, 3, 4, 5, 5, 5,
        2, 2, 2, 3, 4, 5, 5, 5,
        2, 2, 2, 3, 4, 5, 5, 5,
      ],
      plane.data
    );
  }
}
