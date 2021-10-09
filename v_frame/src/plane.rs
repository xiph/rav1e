// Copyright (c) 2017-2021, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use rayon::current_num_threads;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;
use rust_hawktracer::*;
use std::cmp;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Index, IndexMut, Range};
use std::{
  alloc::{alloc, dealloc, Layout},
  u32,
};
use std::{
  fmt::{Debug, Display, Formatter},
  usize,
};
use std::{iter::FusedIterator, ops::DerefMut};

use crate::math::*;
use crate::pixel::*;
use crate::serialize::{Deserialize, Serialize};

/// Plane-specific configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlaneConfig {
  /// Data stride.
  pub stride: usize,
  /// Allocated height in pixels.
  pub alloc_height: usize,
  /// Width in pixels.
  pub width: usize,
  /// Height in pixels.
  pub height: usize,
  /// Decimator along the X axis.
  ///
  /// For example, for chroma planes in a 4:2:0 configuration this would be 1.
  pub xdec: usize,
  /// Decimator along the Y axis.
  ///
  /// For example, for chroma planes in a 4:2:0 configuration this would be 1.
  pub ydec: usize,
  /// Number of padding pixels on the right.
  pub xpad: usize,
  /// Number of padding pixels on the bottom.
  pub ypad: usize,
  /// X where the data starts.
  pub xorigin: usize,
  /// Y where the data starts.
  pub yorigin: usize,
}

impl PlaneConfig {
  /// Stride alignment in bytes.
  const STRIDE_ALIGNMENT_LOG2: usize = 6;

  #[inline]
  pub fn new(
    width: usize, height: usize, xdec: usize, ydec: usize, xpad: usize,
    ypad: usize, type_size: usize,
  ) -> Self {
    let xorigin =
      xpad.align_power_of_two(Self::STRIDE_ALIGNMENT_LOG2 + 1 - type_size);
    let yorigin = ypad;
    let stride = (xorigin + width + xpad)
      .align_power_of_two(Self::STRIDE_ALIGNMENT_LOG2 + 1 - type_size);
    let alloc_height = yorigin + height + ypad;

    PlaneConfig {
      stride,
      alloc_height,
      width,
      height,
      xdec,
      ydec,
      xpad,
      ypad,
      xorigin,
      yorigin,
    }
  }
}

/// Absolute offset in pixels inside a plane
#[derive(Clone, Copy, Debug, Default)]
pub struct PlaneOffset {
  pub x: isize,
  pub y: isize,
}

/// Backing buffer for the Plane data
///
/// The buffer is padded and aligned according to the architecture-specific
/// SIMD constraints.
#[derive(Debug, PartialEq, Eq)]
#[cfg_attr(not(feature = "serialize"), derive(Serialize, Deserialize))]
pub struct PlaneData<T: Pixel> {
  ptr: std::ptr::NonNull<T>,
  _marker: PhantomData<T>,
  len: usize,
}

unsafe impl<T: Pixel + Send> Send for PlaneData<T> {}
unsafe impl<T: Pixel + Sync> Sync for PlaneData<T> {}

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

#[cfg(feature = "serialize")]
impl<T: Pixel + Serialize> Serialize for PlaneData<T> {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    use serde::ser::SerializeSeq;
    use std::ops::Deref;

    let mut data = serializer.serialize_seq(Some(self.len))?;
    for byte in self.deref() {
      data.serialize_element(&byte)?;
    }
    data.end()
  }
}

#[cfg(feature = "serialize")]
impl<'de, T: Pixel + Deserialize<'de>> Deserialize<'de> for PlaneData<T> {
  fn deserialize<D>(
    deserializer: D,
  ) -> Result<Self, <D as serde::Deserializer<'de>>::Error>
  where
    D: serde::Deserializer<'de>,
  {
    let deserialized = Vec::deserialize(deserializer)?;
    Ok(Self::from_slice(&deserialized))
  }
}

impl<T: Pixel> PlaneData<T> {
  // Data alignment in bytes.
  cfg_if::cfg_if! {
    if #[cfg(target_arch = "wasm32")] {
      // FIXME: wasm32 allocator fails for alignment larger than 3
      const DATA_ALIGNMENT_LOG2: usize = 3;
    } else {
      const DATA_ALIGNMENT_LOG2: usize = 6;
    }
  }

  unsafe fn layout(len: usize) -> Layout {
    Layout::from_size_align_unchecked(
      len * mem::size_of::<T>(),
      1 << Self::DATA_ALIGNMENT_LOG2,
    )
  }

  unsafe fn new_uninitialized(len: usize) -> Self {
    let ptr = {
      let ptr = alloc(Self::layout(len)) as *mut T;
      std::ptr::NonNull::new_unchecked(ptr)
    };

    PlaneData { ptr, len, _marker: PhantomData }
  }

  pub fn new(len: usize) -> Self {
    let mut pd = unsafe { Self::new_uninitialized(len) };

    for v in pd.iter_mut() {
      *v = T::cast_from(128);
    }

    pd
  }

  fn from_slice(data: &[T]) -> Self {
    let mut pd = unsafe { Self::new_uninitialized(data.len()) };

    pd.copy_from_slice(data);

    pd
  }
}

/// One data plane of a frame.
///
/// For example, a plane can be a Y luma plane or a U or V chroma plane.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plane<T: Pixel> {
  // TODO: it is used by encoder to copy by plane and by tiling, make it
  // private again once tiling is moved and a copy_plane fn is added.
  //
  pub data: PlaneData<T>,
  /// Plane configuration.
  pub cfg: PlaneConfig,
}

impl<T: Pixel> Debug for Plane<T>
where
  T: Display,
{
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    write!(f, "Plane {{ data: [{}, ...], cfg: {:?} }}", self.data[0], self.cfg)
  }
}

impl<T: Pixel> Plane<T> {
  /// Allocates and returns a new plane.
  pub fn new(
    width: usize, height: usize, xdec: usize, ydec: usize, xpad: usize,
    ypad: usize,
  ) -> Self {
    let cfg = PlaneConfig::new(
      width,
      height,
      xdec,
      ydec,
      xpad,
      ypad,
      mem::size_of::<T>(),
    );
    let data = PlaneData::new(cfg.stride * cfg.alloc_height);

    Plane { data, cfg }
  }

  /// Allocates and returns an uninitialized plane.
  unsafe fn new_uninitialized(
    width: usize, height: usize, xdec: usize, ydec: usize, xpad: usize,
    ypad: usize,
  ) -> Self {
    let cfg = PlaneConfig::new(
      width,
      height,
      xdec,
      ydec,
      xpad,
      ypad,
      mem::size_of::<T>(),
    );
    let data = PlaneData::new_uninitialized(cfg.stride * cfg.alloc_height);

    Plane { data, cfg }
  }

  pub fn from_slice(data: &[T], stride: usize) -> Self {
    let len = data.len();

    assert!(len % stride == 0);

    Self {
      data: PlaneData::from_slice(data),
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
      },
    }
  }

  pub fn pad(&mut self, w: usize, h: usize) {
    let xorigin = self.cfg.xorigin;
    let yorigin = self.cfg.yorigin;
    let stride = self.cfg.stride;
    let alloc_height = self.cfg.alloc_height;
    let width = (w + self.cfg.xdec) >> self.cfg.xdec;
    let height = (h + self.cfg.ydec) >> self.cfg.ydec;

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

  /// Minimally test that the plane has been padded.
  pub fn probe_padding(&self, w: usize, h: usize) -> bool {
    let PlaneConfig {
      xorigin, yorigin, stride, alloc_height, xdec, ydec, ..
    } = self.cfg;
    let width = (w + xdec) >> xdec;
    let height = (h + ydec) >> ydec;
    let corner = (yorigin + height - 1) * stride + xorigin + width - 1;
    let corner_value = self.data[corner];

    self.data[(yorigin + height) * stride - 1] == corner_value
      && self.data[(alloc_height - 1) * stride + xorigin + width - 1]
        == corner_value
      && self.data[alloc_height * stride - 1] == corner_value
  }

  pub fn slice(&self, po: PlaneOffset) -> PlaneSlice<'_, T> {
    PlaneSlice { plane: self, x: po.x, y: po.y }
  }

  pub fn mut_slice(&mut self, po: PlaneOffset) -> PlaneMutSlice<'_, T> {
    PlaneMutSlice { plane: self, x: po.x, y: po.y }
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

  /// Returns the pixel at the given coordinates.
  pub fn p(&self, x: usize, y: usize) -> T {
    self.data[self.index(x, y)]
  }

  /// Returns plane data starting from the origin.
  pub fn data_origin(&self) -> &[T] {
    &self.data[self.index(0, 0)..]
  }

  /// Returns mutable plane data starting from the origin.
  pub fn data_origin_mut(&mut self) -> &mut [T] {
    let i = self.index(0, 0);
    &mut self.data[i..]
  }

  /// Copies data into the plane from a pixel array.
  pub fn copy_from_raw_u8(
    &mut self, source: &[u8], source_stride: usize, source_bytewidth: usize,
  ) {
    let stride = self.cfg.stride;
    for (self_row, source_row) in self
      .data_origin_mut()
      .chunks_mut(stride)
      .zip(source.chunks(source_stride))
    {
      match source_bytewidth {
        1 => {
          for (self_pixel, source_pixel) in
            self_row.iter_mut().zip(source_row.iter())
          {
            *self_pixel = T::cast_from(*source_pixel);
          }
        }
        2 => {
          assert!(
            mem::size_of::<T>() >= 2,
            "source bytewidth ({}) cannot fit in Plane<u8>",
            source_bytewidth
          );
          for (self_pixel, bytes) in
            self_row.iter_mut().zip(source_row.chunks_exact(2))
          {
            *self_pixel = T::cast_from(bytes[1]) << 8 | T::cast_from(bytes[0]);
          }
        }

        _ => {}
      }
    }
  }

  /// Copies data from a plane into a pixel array.
  pub fn copy_to_raw_u8(
    &self, dest: &mut [u8], dest_stride: usize, dest_bytewidth: usize,
  ) {
    let stride = self.cfg.stride;
    for (self_row, dest_row) in
      self.data_origin().chunks(stride).zip(dest.chunks_mut(dest_stride))
    {
      match dest_bytewidth {
        1 => {
          for (self_pixel, dest_pixel) in
            self_row[..self.cfg.width].iter().zip(dest_row.iter_mut())
          {
            *dest_pixel = u8::cast_from(*self_pixel);
          }
        }
        2 => {
          assert!(
            mem::size_of::<T>() >= 2,
            "dest bytewidth ({}) cannot fit in Plane<u8>",
            dest_bytewidth
          );
          for (self_pixel, bytes) in
            self_row[..self.cfg.width].iter().zip(dest_row.chunks_mut(2))
          {
            bytes[0] = u16::cast_from(*self_pixel) as u8;
            bytes[1] = (u16::cast_from(*self_pixel) >> 8) as u8;
          }
        }

        _ => {}
      }
    }
  }

  /// Returns plane with half the resolution for width and height.
  /// Downscaled with 2x2 box filter.
  /// Padded to dimensions with frame_width and frame_height.
  pub fn downsampled(
    &self, frame_width: usize, frame_height: usize,
  ) -> Plane<T> {
    let src = self;
    // SAFETY: all pixels initialized in this function
    let mut new = unsafe {
      Plane::new_uninitialized(
        (src.cfg.width + 1) / 2,
        (src.cfg.height + 1) / 2,
        src.cfg.xdec + 1,
        src.cfg.ydec + 1,
        src.cfg.xpad / 2,
        src.cfg.ypad / 2,
      )
    };

    let width = new.cfg.width;
    let height = new.cfg.height;

    assert!(width * 2 <= src.cfg.stride - src.cfg.xorigin);
    assert!(height * 2 <= src.cfg.alloc_height - src.cfg.yorigin);

    let data_origin = src.data_origin();
    for (row_idx, dst_row) in new
      .mut_slice(PlaneOffset::default())
      .rows_iter_mut()
      .enumerate()
      .take(height)
    {
      let src_top_row =
        &data_origin[(src.cfg.stride * row_idx * 2)..][..(2 * width)];
      let src_bottom_row =
        &data_origin[(src.cfg.stride * (row_idx * 2 + 1))..][..(2 * width)];
      for (col, dst) in dst_row.iter_mut().take(width).enumerate() {
        let mut sum = 0;
        unsafe {
          // SAFETY: We perform a bounds check above to ensure these are safe
          sum += u32::cast_from(*src_top_row.get_unchecked(2 * col));
          sum += u32::cast_from(*src_top_row.get_unchecked(2 * col + 1));
          sum += u32::cast_from(*src_bottom_row.get_unchecked(2 * col));
          sum += u32::cast_from(*src_bottom_row.get_unchecked(2 * col + 1));
        }
        let avg = (sum + 2) >> 2;
        *dst = T::cast_from(avg);
      }
    }
    new.pad(frame_width, frame_height);
    new
  }

  /// Returns a plane downscaled from the source plane by a factor of `scale` (not padded)
  pub fn downscale(&self, scale: usize) -> Plane<T> {
    // SAFETY: all pixels initialized when `downscale_in_place` is called
    let mut new_plane = unsafe {
      Plane::new_uninitialized(
        self.cfg.width / scale,
        self.cfg.height / scale,
        0,
        0,
        0,
        0,
      )
    };

    self.downscale_in_place(scale, &mut new_plane);

    new_plane
  }

  /// Downscales the source plane by a factor of `scale`, writing the result to `in_plane` (not padded)
  ///
  /// `in_plane`'s width and height must be sufficient for `scale`.
  #[hawktracer(downscale)]
  pub fn downscale_in_place(&self, scale: usize, in_plane: &mut Plane<T>) {
    let src = self;
    let box_pixels = scale * scale;
    let half_box_pixels = box_pixels as u32 / 2; // Used for rounding int division

    let data_origin = src.data_origin();

    let stride = in_plane.cfg.stride;
    let width = in_plane.cfg.width;
    let height = in_plane.cfg.height;

    // Par iter over dst chunks
    let plane_data_mut_slice = in_plane.data.deref_mut();
    let threads = current_num_threads();
    let chunk_rows = cmp::max((height + threads / 2) / threads, 1);

    let chunk_size = chunk_rows * stride;

    let height_limit = height * stride;
    plane_data_mut_slice[0..height_limit]
      .par_chunks_mut(chunk_size)
      .enumerate()
      .for_each(|(chunk_idx, chunk)| {
        // Iter dst rows
        let dst_rows = chunk.chunks_mut(stride);
        for (row_offset, dst_row) in dst_rows.enumerate() {
          let row_idx = chunk_idx * chunk_rows + row_offset;

          // Iter dst cols
          for (col_idx, dst) in dst_row[0..width].iter_mut().enumerate() {
            let mut sum = half_box_pixels;
            // Sum box of size scale * scale

            // Iter src row
            for y in 0..scale {
              let src_row_idx = row_idx * scale + y;
              let src_row = &data_origin[(src_row_idx * src.cfg.stride)..];

              // Iter src col
              for x in 0..scale {
                let src_col_idx = col_idx * scale + x;
                sum += u32::cast_from(src_row[src_col_idx]);
              }
            }

            // Box average
            let avg = sum as usize / box_pixels;
            *dst = T::cast_from(avg);
          }
        }
      });
  }

  /// Iterates over the pixels in the plane, skipping the padding.
  pub fn iter(&self) -> PlaneIter<'_, T> {
    PlaneIter::new(self)
  }

  /// Iterates over the lines of the plane
  pub fn rows_iter(&self) -> RowsIter<'_, T> {
    RowsIter { plane: self, x: 0, y: 0 }
  }

  /// Return a line
  pub fn row(&self, y: isize) -> &[T] {
    let range = self.row_range(0, y);

    &self.data[range]
  }
}

/// Iterator over plane pixels, skipping padding.
#[derive(Debug)]
pub struct PlaneIter<'a, T: Pixel> {
  plane: &'a Plane<T>,
  y: usize,
  x: usize,
}

impl<'a, T: Pixel> PlaneIter<'a, T> {
  /// Creates a new iterator.
  pub fn new(plane: &'a Plane<T>) -> Self {
    Self { plane, y: 0, x: 0 }
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

impl<T: Pixel> FusedIterator for PlaneIter<'_, T> {}

#[derive(Clone, Copy, Debug)]
pub struct PlaneSlice<'a, T: Pixel> {
  pub plane: &'a Plane<T>,
  pub x: isize,
  pub y: isize,
}

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
  #[allow(unused)]
  pub fn as_ptr(&self) -> *const T {
    self[0].as_ptr()
  }

  pub fn rows_iter(&self) -> RowsIter<'_, T> {
    RowsIter { plane: self.plane, x: self.x, y: self.y }
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
        .max(-(self.plane.cfg.yorigin as isize)),
    }
  }

  pub fn subslice(&self, xo: usize, yo: usize) -> PlaneSlice<'a, T> {
    PlaneSlice {
      plane: self.plane,
      x: self.x + xo as isize,
      y: self.y + yo as isize,
    }
  }

  pub fn reslice(&self, xo: isize, yo: isize) -> PlaneSlice<'a, T> {
    PlaneSlice { plane: self.plane, x: self.x + xo, y: self.y + yo }
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

  pub fn row(&self, y: usize) -> &[T] {
    let y = (self.y + y as isize + self.plane.cfg.yorigin as isize) as usize;
    let x = (self.x + self.plane.cfg.xorigin as isize) as usize;
    let start = y * self.plane.cfg.stride + x;
    let width = self.plane.cfg.stride - x;
    &self.plane.data[start..start + width]
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
  pub y: isize,
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
  #[allow(unused)]
  pub fn rows_iter(&self) -> RowsIter<'_, T> {
    RowsIter { plane: self.plane, x: self.x, y: self.y }
  }

  pub fn rows_iter_mut(&mut self) -> RowsIterMut<'_, T> {
    RowsIterMut {
      plane: self.plane as *mut Plane<T>,
      x: self.x,
      y: self.y,
      phantom: PhantomData,
    }
  }

  #[allow(unused)]
  pub fn subslice(&mut self, xo: usize, yo: usize) -> PlaneMutSlice<'_, T> {
    PlaneMutSlice {
      plane: self.plane,
      x: self.x + xo as isize,
      y: self.y + yo as isize,
    }
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
    #[rustfmt::skip]
    let mut plane = Plane::from_slice(&
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
      ],
      8,
    );

    let input = vec![42u8; 64];

    plane.copy_from_raw_u8(&input, 8, 1);

    println!("{:?}", &plane.data[..10]);

    assert_eq!(&input[..64], &plane.data[..64]);
  }

  #[test]
  fn copy_to_raw_u8() {
    #[rustfmt::skip]
    let plane = Plane::from_slice(&
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
      ],
      8,
    );

    let mut output = vec![42u8; 64];

    plane.copy_to_raw_u8(&mut output, 8, 1);

    println!("{:?}", &plane.data[..10]);

    assert_eq!(&output[..64], &plane.data[..64]);
  }

  #[test]
  fn test_plane_downsample() {
    #[rustfmt::skip]
    let plane = Plane::<u8> {
      data: PlaneData::from_slice(&[
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
    let downsampled = plane.downsampled(4, 4);

    #[rustfmt::skip]
    let expected = &[
      5, 5,
      6, 6,
    ];

    let v: Vec<_> = downsampled.iter().collect();

    assert_eq!(&expected[..], &v[..]);
  }
  #[test]
  fn test_plane_downsample_odd() {
    #[rustfmt::skip]
    let plane = Plane::<u8> {
      data: PlaneData::from_slice(&[
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
        width: 3,
        height: 3,
        xdec: 0,
        ydec: 0,
        xpad: 0,
        ypad: 0,
        xorigin: 2,
        yorigin: 3,
      },
    };
    let downsampled = plane.downsampled(3, 3);

    #[rustfmt::skip]
    let expected = &[
      5, 5,
      6, 6,
    ];

    let v: Vec<_> = downsampled.iter().collect();
    assert_eq!(&expected[..], &v[..]);
  }

  #[test]
  fn test_plane_downscale() {
    #[rustfmt::skip]
    let plane = Plane::<u8> {
      data: PlaneData::from_slice(&[
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 4, 5, 0, 0,
        0, 0, 2, 3, 6, 7, 0, 0,
        0, 0, 8, 9, 7, 5, 0, 0,
        0, 0, 9, 8, 3, 1, 0, 0,
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
    let downscaled = plane.downscale(2);

    #[rustfmt::skip]
    let expected = &[
      2, 6,
      9, 4
    ];

    let v: Vec<_> = downscaled.iter().collect();
    assert_eq!(&expected[..], &v[..]);
  }

  #[test]
  fn test_plane_downscale_odd() {
    #[rustfmt::skip]
  let plane = Plane::<u8> {
      data: PlaneData::from_slice(&[
        1, 2, 3, 4, 1, 2, 3, 4,
        0, 0, 8, 7, 6, 5, 8, 7,
        6, 5, 8, 7, 6, 5, 8, 7,
        6, 5, 8, 7, 0, 0, 2, 3,
        4, 5, 0, 0, 9, 8, 7, 6,
        0, 0, 0, 0, 2, 3, 4, 5,
        0, 0, 0, 0, 2, 3, 4, 5,
      ]),
      cfg: PlaneConfig {
        stride: 8,
        alloc_height: 7,
        width: 8,
        height: 7,
        xdec: 0,
        ydec: 0,
        xpad: 0,
        ypad: 0,
        xorigin: 0,
        yorigin: 0,
      },
    };

    let downscaled = plane.downscale(3);

    #[rustfmt::skip]
    let expected = &[
      4, 5,
      3, 3
    ];

    let v: Vec<_> = downscaled.iter().collect();
    assert_eq!(&expected[..], &v[..]);
  }

  #[test]
  fn test_plane_downscale_odd_2() {
    #[rustfmt::skip]
    let plane = Plane::<u8> {
      data: PlaneData::from_slice(&[
        9, 8, 3, 1, 0, 1, 4, 5, 0, 0,
        0, 1, 4, 5, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 9, 0,
        0, 2, 3, 6, 7, 0, 0, 0, 0, 0,
        0, 0, 8, 9, 7, 5, 0, 0, 0, 0,
        9, 8, 3, 1, 0, 1, 4, 5, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 2, 3, 6, 7, 0,
        0, 0, 0, 0, 0, 0, 8, 9, 7, 5,
        0, 0, 0, 0, 9, 8, 3, 1, 0, 0
      ]),
      cfg: PlaneConfig {
        stride: 10,
        alloc_height: 10,
        width: 10,
        height: 10,
        xdec: 0,
        ydec: 0,
        xpad: 0,
        ypad: 0,
        xorigin: 0,
        yorigin: 0,
      },
    };
    let downscaled = plane.downscale(3);

    #[rustfmt::skip]
    let expected = &[
      3, 1, 2,
      4, 4, 1,
      0, 0, 4,
    ];

    let v: Vec<_> = downscaled.iter().collect();
    assert_eq!(&expected[..], &v[..]);
  }

  #[test]
  fn test_plane_pad() {
    #[rustfmt::skip]
    let mut plane = Plane::<u8> {
      data: PlaneData::from_slice(&[
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

    #[rustfmt::skip]
    assert_eq!(
      &[
        1, 1, 1, 2, 3, 4, 4, 4,
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

  #[test]
  fn test_pixel_iterator() {
    #[rustfmt::skip]
    let plane = Plane::<u8> {
      data: PlaneData::from_slice(&[
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

    let pixels: Vec<u8> = plane.iter().collect();

    assert_eq!(
      &[1, 2, 3, 4, 8, 7, 6, 5, 9, 8, 7, 6, 2, 3, 4, 5,][..],
      &pixels[..]
    );
  }
}
