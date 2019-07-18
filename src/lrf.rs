// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(safe_extern_statics)]

use crate::frame::Frame;
use crate::encoder::FrameInvariants;
use crate::context::PLANES;
use crate::context::MAX_SB_SIZE;
use crate::frame::Plane;
use crate::frame::PlaneSlice;
use crate::frame::PlaneMutSlice;
use crate::frame::PlaneOffset;
use crate::frame::PlaneConfig;
use std::cmp;
use crate::util::clamp;
use crate::util::CastFromPrimitive;
use crate::util::Pixel;

use std::ops::{Index, IndexMut};

pub const RESTORATION_TILESIZE_MAX_LOG2: usize = 8;

pub const RESTORE_NONE: u8 = 0;
pub const RESTORE_SWITCHABLE: u8 = 1;
pub const RESTORE_WIENER: u8 = 2;
pub const RESTORE_SGRPROJ: u8 = 3;

pub const WIENER_TAPS_MIN: [i8; 3] = [ -5, -23, -17 ];
pub const WIENER_TAPS_MID: [i8; 3] = [ 3, -7, 15 ];
pub const WIENER_TAPS_MAX: [i8; 3] = [ 10, 8, 46 ];
#[allow(unused)]
pub const WIENER_TAPS_K:   [i8; 3] = [ 1, 2, 3 ];
pub const WIENER_BITS: usize = 7;

pub const SGRPROJ_XQD_MIN: [i8; 2] = [ -96, -32 ];
pub const SGRPROJ_XQD_MID: [i8; 2] = [ -32, 31 ];
pub const SGRPROJ_XQD_MAX: [i8; 2] = [ 31, 95 ];
pub const SGRPROJ_PRJ_SUBEXP_K: u8 = 4;
pub const SGRPROJ_PRJ_BITS: u8 = 7;
pub const SGRPROJ_PARAMS_BITS: u8 = 4;
pub const SGRPROJ_MTABLE_BITS: u8 = 20;
pub const SGRPROJ_SGR_BITS: u8 = 8;
pub const SGRPROJ_RECIP_BITS: u8 = 12;
pub const SGRPROJ_RST_BITS: u8 = 4;
pub const SGRPROJ_PARAMS_S: [[u32; 2]; 1 << SGRPROJ_PARAMS_BITS] = [
  [140, 3236], [112, 2158], [ 93, 1618], [ 80, 1438],
  [ 70, 1295], [ 58, 1177], [ 47, 1079], [ 37,  996],
  [ 30,  925], [ 25,  863], [  0, 2589], [  0, 1618],
  [  0, 1177], [  0,  925], [ 56,    0], [ 22,    0]
];

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RestorationFilter {
  None,
  Wiener  { coeffs: [[i8; 3]; 2] },
  Sgrproj { set: u8,
            xqd: [i8; 2] },
}

impl RestorationFilter {
  pub fn default() -> RestorationFilter {
    RestorationFilter::None{}
  }
}

#[inline(always)]
fn sgrproj_sum_finish(ssq: u32, sum: u32, n: u32, one_over_n: u32, s: u32, bdm8: usize) -> (u32, u32) {
  let scaled_ssq = (ssq + (1 << (2 * bdm8) >> 1)) >> (2 * bdm8);
  let scaled_sum = (sum + (1 << bdm8 >> 1)) >> bdm8;
  let p = cmp::max(0, (scaled_ssq*n) as i32 - (scaled_sum*scaled_sum) as i32) as u32;
  let z = (p * s + (1 << SGRPROJ_MTABLE_BITS >> 1)) >> SGRPROJ_MTABLE_BITS;
  let a = if z >= 255 {
    256
  } else if z == 0 {
    1
  } else {
    (((z << SGRPROJ_SGR_BITS) + z/2) / (z+1))
  };
  let b = ((1 << SGRPROJ_SGR_BITS) - a) * sum * one_over_n;
  (a, (b + (1 << SGRPROJ_RECIP_BITS >> 1)) >> SGRPROJ_RECIP_BITS)
}

// Using an integral image, compute the sum of a square region
fn get_integral_square(
  iimg: &[u32], stride: usize, x: usize, y: usize, size: usize
) -> u32 {
  // Cancel out overflow in iimg by using wrapping arithmetic
  iimg[y * stride + x]
    .wrapping_add(iimg[(y + size) * stride + x + size])
    .wrapping_sub(iimg[(y + size) * stride + x])
    .wrapping_sub(iimg[y * stride + x + size])
}

// computes an intermediate (ab) column for stripe_h + 2 rows at
// column stripe_x.
// r=1 case computes every row as every row is used (see r2 version below)
fn sgrproj_box_ab_r1(
  af: &mut [u32; 64 + 2], bf: &mut [u32; 64 + 2], iimg: &[u32],
  iimg_sq: &[u32], iimg_stride: usize, stripe_x: isize, stripe_h: usize,
  s: u32, bdm8: usize
) {
  for y in 0..stripe_h + 2 {
    let sum = get_integral_square(iimg, iimg_stride, stripe_x as usize, y, 3);
    let ssq =
      get_integral_square(iimg_sq, iimg_stride, stripe_x as usize, y, 3);
    let (reta, retb) = sgrproj_sum_finish(ssq, sum, 9, 455, s, bdm8);
    af[y] = reta;
    bf[y] = retb;
  }
}

// One oddness about the radius=2 intermediate array computations that
// the spec doesn't make clear: Although the spec defines computation
// of every row (of a, b and f), only half of the rows (every-other
// row) are actually used.  We use the full-size array here but only
// compute the even rows.  This is not so much optimization as trying
// to illustrate what this convoluted filter is actually doing
// (ie not as much as it may appear).
fn sgrproj_box_ab_r2(
  af: &mut [u32; 64 + 2], bf: &mut [u32; 64 + 2], iimg: &[u32],
  iimg_sq: &[u32], iimg_stride: usize, stripe_x: isize, stripe_h: usize,
  s: u32, bdm8: usize
) {
  for y in (0..stripe_h + 2).step_by(2) {
    let sum = get_integral_square(iimg, iimg_stride, stripe_x as usize, y, 5);
    let ssq =
      get_integral_square(iimg_sq, iimg_stride, stripe_x as usize, y, 5);
    let (reta, retb) = sgrproj_sum_finish(ssq, sum, 25, 164, s, bdm8);
    af[y] = reta;
    bf[y] = retb;
  }
}

fn sgrproj_box_f_r0<T: Pixel>(f: &mut[u32; 64], x: usize, y: isize, h: usize, cdeffed: &PlaneSlice<T>) {
  for i in cmp::max(0, -y) as usize..h {
    f[i as usize] = (u32::cast_from(cdeffed.p(x, (y + i as isize) as usize))) << SGRPROJ_RST_BITS;
  }
}

fn sgrproj_box_f_r1<T: Pixel>(af: &[&[u32; 64+2]; 3], bf: &[&[u32; 64+2]; 3], f: &mut[u32; 64],
                              x: usize, y: isize, h: usize, cdeffed: &PlaneSlice<T>) {
  let shift = 5 + SGRPROJ_SGR_BITS - SGRPROJ_RST_BITS;
  for i in cmp::max(0, -y) as usize..h {
    let a =
      3 * (af[0][i] + af[2][i]   + af[0][i+2] + af[2][i+2]) +
      4 * (af[1][i] + af[0][i+1] + af[1][i+1] + af[2][i+1] + af[1][i+2]);
    let b =
      3 * (bf[0][i] + bf[2][i]   + bf[0][i+2] + bf[2][i+2]) +
      4 * (bf[1][i] + bf[0][i+1] + bf[1][i+1] + bf[2][i+1] + bf[1][i+2]);
    let v = a * u32::cast_from(cdeffed.p(x, (y + i as isize) as usize)) + b;
    f[i as usize] = (v + (1 << shift >> 1)) >> shift;
  }
}

fn sgrproj_box_f_r2<T: Pixel>(af: &[&[u32; 64+2]; 3], bf: &[&[u32; 64+2]; 3], f: &mut[u32; 64],
                              x: usize, y: isize, h: usize, cdeffed: &PlaneSlice<T>) {
  let shift = 5 + SGRPROJ_SGR_BITS - SGRPROJ_RST_BITS;
  let shifto = 4 + SGRPROJ_SGR_BITS - SGRPROJ_RST_BITS;
  for i in (cmp::max(0, -y) as usize..h).step_by(2) {
    let a =
      5 * (af[0][i] + af[2][i]) +
      6 * (af[1][i]);
    let b =
      5 * (bf[0][i] + bf[2][i]) +
      6 * (bf[1][i]);
    let ao =
      5 * (af[0][i+2] + af[2][i+2]) +
      6 * (af[1][i+2]);
    let bo =
      5 * (bf[0][i+2] + bf[2][i+2]) +
      6 * (bf[1][i+2]);
    let v = (a + ao) * u32::cast_from(cdeffed.p(x, (y+i as isize) as usize)) + b + bo;
    f[i as usize] = (v + (1 << shift >> 1)) >> shift;
    let vo = ao * u32::cast_from(cdeffed.p(x, (y + i as isize) as usize + 1)) + bo;
    f[i as usize + 1] = (vo + (1 << shifto >> 1)) >> shifto;
  }
}

struct VertPaddedIter<'a, T: Pixel> {
  // The two sources that can be selected when clipping
  deblocked: &'a Plane<T>,
  cdeffed: &'a Plane<T>,
  // x index to choice where on the row to start
  x: isize,
  // y index that will be mutated
  y: isize,
  // The index at which to terminate. Can be larger than the slice length.
  end: isize,
  // Used for source buffer choice/clipping. May (and regularly will)
  // be negative.
  stripe_begin: isize,
  // Also used for source buffer choice/clipping. May specify a stripe boundary
  // less than, equal to, or larger than the buffers we're accessing.
  stripe_end: isize,
  // Active area cropping is done by specifying a value smaller than the height
  // of the plane.
  crop: isize
}

impl<'a, 'b, T: Pixel> VertPaddedIter<'a, T> {
  fn new(
    cdeffed: &PlaneSlice<'a, T>, deblocked: &PlaneSlice<'a, T>,
    stripe_h: usize, crop: usize, r: usize
  ) -> VertPaddedIter<'a, T> {
    // cdeffed and deblocked must start at the same coordinates from their
    // underlying planes. Since cropping is provided via a separate params, the
    // height of the underlying planes do not need to match.
    assert_eq!(cdeffed.x, deblocked.x);
    assert_eq!(cdeffed.y, deblocked.y);

    // The number of rows outside the stripe are needed
    let rows_above = r + 2;
    let rows_below = 2;

    // Offset crop and stripe_h so they are relative to the underlying plane
    // and not the plane slice.
    let crop = crop as isize + deblocked.y;
    let stripe_end = stripe_h as isize + deblocked.y;

    // Move y up the number rows above.
    // If y is negative we repeat the first row
    let y = deblocked.y - rows_above as isize;

    VertPaddedIter {
      deblocked: deblocked.plane,
      cdeffed: cdeffed.plane,
      x: deblocked.x,
      y,
      end: (rows_above + stripe_h + rows_below) as isize + y,
      stripe_begin: deblocked.y,
      stripe_end,
      crop
    }
  }
}

impl<'a, T: Pixel> Iterator for VertPaddedIter<'a, T> {
  type Item = &'a [T];

  #[inline(always)]
  fn next(&mut self) -> Option<Self::Item> {
    if self.end > self.y {
      // clamp before deciding the source
      // clamp vertically to storage at top and passed-in height at bottom
      let cropped_y = clamp(self.y, 0, self.crop - 1);
      // clamp vertically to stripe limits
      let ly = clamp(cropped_y, self.stripe_begin - 2, self.stripe_end + 1);

      // decide if we're vertically inside or outside the strip
      let src_plane =
        if ly >= self.stripe_begin && ly < self.stripe_end as isize {
          self.cdeffed
        } else {
          self.deblocked
        };
      // cannot directly return self.ps.row(row) due to lifetime issue
      let range = src_plane.row_range(self.x, ly);
      self.y += 1;
      Some(&src_plane.data[range])
    } else {
      None
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    let remaining = self.end - self.y;
    debug_assert!(remaining >= 0);
    let remaining = remaining as usize;

    (remaining, Some(remaining))
  }
}

impl<T: Pixel> ExactSizeIterator for VertPaddedIter<'_, T> {}

struct HorzPaddedIter<'a, T: Pixel> {
  // Active area cropping is done using the length of the slice
  slice: &'a [T],
  // x index of the iterator
  // When less than 0, repeat the first element. When greater than end, repeat
  // the last element
  index: isize,
  // The index at which to terminate. Can be larger than the slice length.
  end: usize
}

impl<'a, T: Pixel> HorzPaddedIter<'a, T> {
  fn new(
    slice: &'a [T], start_index: isize, width: usize
  ) -> HorzPaddedIter<'a, T> {
    HorzPaddedIter {
      slice,
      index: start_index,
      end: (width as isize + start_index) as usize
    }
  }
}

impl<'a, T: Pixel> Iterator for HorzPaddedIter<'a, T> {
  type Item = &'a T;

  #[inline(always)]
  fn next(&mut self) -> Option<Self::Item> {
    if self.index < self.end as isize {
      // clamp to the edges of the frame
      let x = clamp(self.index, 0, self.slice.len() as isize - 1) as usize;
      self.index += 1;
      Some(&self.slice[x])
    } else {
      None
    }
  }

  #[inline(always)]
  fn size_hint(&self) -> (usize, Option<usize>) {
    let size: usize = (self.end as isize - self.index) as usize;
    (size, Some(size))
  }
}

impl<T: Pixel> ExactSizeIterator for HorzPaddedIter<'_, T> {}

fn setup_integral_image<'a, T: Pixel>(
  rows_iter: &mut impl Iterator<Item = impl Iterator<Item = &'a T>>,
  integral_image: &mut [u32], sq_integral_image: &mut [u32],
  integral_image_stride: usize
) {
  // Setup the first row
  {
    let mut sum: u32 = 0;
    let mut sq_sum: u32 = 0;
    // Remove the first row and use it outside of the main loop
    let row = rows_iter.next().unwrap();
    for (src, (integral, sq_integral)) in
      row.zip(integral_image.iter_mut().zip(sq_integral_image.iter_mut()))
    {
      let current = u32::cast_from(*src);

      // Wrap adds to prevent undefined behaviour on overflow. Overflow is
      // cancelled out when calculating the sum of a region.
      sum = sum.wrapping_add(current);
      *integral = sum;
      sq_sum = sq_sum.wrapping_add(current * current);
      *sq_integral = sq_sum;
    }
  }
  // Calculate all other rows
  let mut integral_slice = &mut integral_image[..];
  let mut sq_integral_slice = &mut sq_integral_image[..];
  for row in rows_iter {
    let mut sum: u32 = 0;
    let mut sq_sum: u32 = 0;

    // Split the data between the previous row and future rows.
    // This allows us to mutate the current row while accessing the
    // previous row.
    let (integral_row_prev, integral_row) =
      integral_slice.split_at_mut(integral_image_stride);
    let (sq_integral_row_prev, sq_integral_row) =
      sq_integral_slice.split_at_mut(integral_image_stride);
    for (
      src,
      ((integral_above, sq_integral_above), (integral, sq_integral))
    ) in row.zip(
      integral_row_prev
        .iter()
        .zip(sq_integral_row_prev.iter())
        .zip(integral_row.iter_mut().zip(sq_integral_row.iter_mut()))
    ) {
      let current = u32::cast_from(*src);
      // Wrap adds to prevent undefined behaviour on overflow. Overflow is
      // cancelled out when calculating the sum of a region.
      sum = sum.wrapping_add(current);
      *integral = sum.wrapping_add(*integral_above);
      sq_sum = sq_sum.wrapping_add(current * current);
      *sq_integral = sq_sum.wrapping_add(*sq_integral_above);
    }

    // The current row also contains all future rows. Replacing the slice with
    // it moves down a row.
    integral_slice = integral_row;
    sq_integral_slice = sq_integral_row;
  }
}

pub fn sgrproj_stripe_filter<T: Pixel>(set: u8, xqd: [i8; 2], fi: &FrameInvariants<T>,
                                       crop_w: usize, crop_h: usize,
                                       stripe_w: usize, stripe_h: usize,
                                       cdeffed: &PlaneSlice<T>,
                                       deblocked: &PlaneSlice<T>,
                                       out: &mut PlaneMutSlice<T>) {
  const WIDTH_MAX: usize = (1 << RESTORATION_TILESIZE_MAX_LOG2) * 2;
  assert!(stripe_h <= 64);
  assert!(stripe_w <= WIDTH_MAX);

  const INTEGRAL_IMAGE_STRIDE: usize = WIDTH_MAX + 6 + 2;
  const INTEGRAL_IMAGE_HEIGHT: usize = 64 + 6 + 2;
  const INTEGRAL_IMAGE_SIZE: usize =
    INTEGRAL_IMAGE_STRIDE * INTEGRAL_IMAGE_HEIGHT;
  let mut integral_image: [u32; INTEGRAL_IMAGE_SIZE] =
    [0; INTEGRAL_IMAGE_SIZE];
  let mut sq_integral_image: [u32; INTEGRAL_IMAGE_SIZE] =
    [0; INTEGRAL_IMAGE_SIZE];

  let bdm8 = fi.sequence.bit_depth - 8;
  let mut a_r2: [[u32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut b_r2: [[u32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut f_r2: [u32; 64] = [0; 64];
  let mut a_r1: [[u32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut b_r1: [[u32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut f_r1: [u32; 64] = [0; 64];

  let s_r2: u32 = SGRPROJ_PARAMS_S[set as usize][0];
  let s_r1: u32 = SGRPROJ_PARAMS_S[set as usize][1];

  let outstart = cmp::max(0, cmp::max(-cdeffed.y, -out.y)) as usize;

  let max_r: usize = if s_r2 > 0 {
    2
  } else if s_r1 > 0 {
    1
  } else {
    0
  };
  {
    // Number of elements outside the stripe
    let left_w = max_r + 2;
    let right_w = max_r + 1;

    assert_eq!(cdeffed.x, deblocked.x);

    // Find how many unique elements to use to the left and right
    let left_uniques = if cdeffed.x == 0 { 0 } else { left_w };
    let right_uniques = right_w.min(crop_w - stripe_w);

    // Find the total number of unique elements used
    let row_uniques = left_uniques + stripe_w + right_uniques;

    // Negative start indices result in repeating the first element of the row
    let start_index_x = if cdeffed.x == 0 { -(left_w as isize) } else { 0 };

    let mut rows_iter = VertPaddedIter::new(
      // Move left to encompass all the used data
      &cdeffed.go_left(left_uniques),
      &deblocked.go_left(left_uniques),
      stripe_h,
      crop_h,
      max_r
    )
    .map(|row: &[T]| {
      HorzPaddedIter::new(
        // Limit how many unique elements we use
        &row[..row_uniques],
        start_index_x,
        left_w + stripe_w + right_w
      )
    });

    setup_integral_image(
      &mut rows_iter,
      &mut integral_image,
      &mut sq_integral_image,
      INTEGRAL_IMAGE_STRIDE
    );
  }
  /* prime the intermediate arrays */
  if s_r2 > 0 {
    sgrproj_box_ab_r2(&mut a_r2[0], &mut b_r2[0],
                      &integral_image, &sq_integral_image,
                      INTEGRAL_IMAGE_STRIDE,
                      0, stripe_h, s_r2, bdm8);
    sgrproj_box_ab_r2(&mut a_r2[1], &mut b_r2[1],
                      &integral_image, &sq_integral_image,
                      INTEGRAL_IMAGE_STRIDE,
                      1, stripe_h, s_r2, bdm8);
  }
  if s_r1 > 0 {
    let r_diff = max_r - 1;
    let integral_image_offset = r_diff + r_diff * INTEGRAL_IMAGE_STRIDE;
    sgrproj_box_ab_r1(&mut a_r1[0], &mut b_r1[0],
                      &integral_image[integral_image_offset..],
                      &sq_integral_image[integral_image_offset..],
                      INTEGRAL_IMAGE_STRIDE,
                      0, stripe_h, s_r1, bdm8);
    sgrproj_box_ab_r1(&mut a_r1[1], &mut b_r1[1],
                      &integral_image[integral_image_offset..],
                      &sq_integral_image[integral_image_offset..],
                      INTEGRAL_IMAGE_STRIDE,
                      1, stripe_h, s_r1, bdm8);
  }

  /* iterate by column */
  for xi in 0..stripe_w {
    /* build intermediate array columns */
    if s_r2 > 0 {
      sgrproj_box_ab_r2(&mut a_r2[(xi+2)%3], &mut b_r2[(xi+2)%3],
                        &integral_image, &sq_integral_image,
                        INTEGRAL_IMAGE_STRIDE,
                        xi as isize + 2, stripe_h, s_r2, bdm8);
      let ap0: [&[u32; 64+2]; 3] = [&a_r2[xi%3], &a_r2[(xi+1)%3], &a_r2[(xi+2)%3]];
      let bp0: [&[u32; 64+2]; 3] = [&b_r2[xi%3], &b_r2[(xi+1)%3], &b_r2[(xi+2)%3]];
      sgrproj_box_f_r2(&ap0, &bp0, &mut f_r2, xi, 0, stripe_h as usize, &cdeffed);
    } else {
      sgrproj_box_f_r0(&mut f_r2, xi, 0, stripe_h as usize, &cdeffed);
    }
    if s_r1 > 0 {
      let r_diff = max_r - 1;
      let integral_image_offset = r_diff + r_diff * INTEGRAL_IMAGE_STRIDE;
      sgrproj_box_ab_r1(&mut a_r1[(xi+2)%3], &mut b_r1[(xi+2)%3],
                        &integral_image[integral_image_offset..],
                        &sq_integral_image[integral_image_offset..],
                        INTEGRAL_IMAGE_STRIDE,
                        xi as isize + 2, stripe_h, s_r1, bdm8);
      let ap1: [&[u32; 64+2]; 3] = [&a_r1[xi%3], &a_r1[(xi+1)%3], &a_r1[(xi+2)%3]];
      let bp1: [&[u32; 64+2]; 3] = [&b_r1[xi%3], &b_r1[(xi+1)%3], &b_r1[(xi+2)%3]];
      sgrproj_box_f_r1(&ap1, &bp1, &mut f_r1, xi, 0, stripe_h as usize, &cdeffed);
    } else {
      sgrproj_box_f_r0(&mut f_r1, xi, 0, stripe_h as usize, &cdeffed);
    }

    /* apply filter */
    let bit_depth = fi.sequence.bit_depth;
    let w0 = xqd[0] as i32;
    let w1 = xqd[1] as i32;
    let w2 = (1 << SGRPROJ_PRJ_BITS) - w0 - w1;
    for yi in outstart..stripe_h as usize {
      let u = i32::cast_from(cdeffed.p(xi, yi)) << SGRPROJ_RST_BITS;
      let v = w0*f_r2[yi] as i32 + w1*u + w2*f_r1[yi] as i32;
      let s = (v + (1 << (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS) >> 1)) >> (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS);
      out[yi][xi] = T::cast_from(clamp(s, 0, (1 << bit_depth) - 1));
    }
  }
}

// Frame inputs below aren't all equal, and will change as work
// continues.  There's no deblocked reconstruction available at this
// point of RDO, so we use the non-deblocked reconstruction, cdef and
// input.  The input can be a full-sized frame. Cdef input is a partial
// frame constructed specifically for RDO.

// For simplicity, this ignores stripe segmentation (it's possible the
// extra complexity isn't worth it and we'll ignore stripes
// permanently during RDO, but that's not been tested yet). Data
// access inside the cdef frame is monolithic and clipped to the cdef
// borders.

// Input params follow the same rules as sgrproj_stripe_filter.
// Inputs are relative to the colocated slice views.
pub fn sgrproj_solve<T: Pixel>(set: u8, fi: &FrameInvariants<T>,
                               input: &PlaneSlice<T>,
                               cdeffed: &PlaneSlice<T>,
                               cdef_w: usize, cdef_h: usize) -> (i8, i8) {
  assert!(cdef_h <= 64);
  assert!(cdef_w <= 64);
  const INTEGRAL_IMAGE_STRIDE: usize = 64 + 6 + 2;
  let mut integral_image: [u32;
    INTEGRAL_IMAGE_STRIDE * INTEGRAL_IMAGE_STRIDE] =
    [0; INTEGRAL_IMAGE_STRIDE * INTEGRAL_IMAGE_STRIDE];
  let mut sq_integral_image: [u32;
    INTEGRAL_IMAGE_STRIDE * INTEGRAL_IMAGE_STRIDE] =
    [0; INTEGRAL_IMAGE_STRIDE * INTEGRAL_IMAGE_STRIDE];

  let bdm8 = fi.sequence.bit_depth - 8;

  let mut a_r2: [[u32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut b_r2: [[u32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut f_r2: [u32; 64] = [0; 64];
  let mut a_r1: [[u32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut b_r1: [[u32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut f_r1: [u32; 64] = [0; 64];

  let s_r2: u32 = SGRPROJ_PARAMS_S[set as usize][0];
  let s_r1: u32 = SGRPROJ_PARAMS_S[set as usize][1];

  let mut h:[[f64; 2]; 2] = [[0.,0.],[0.,0.]];
  let mut c:[f64; 2] = [0., 0.];

  let max_r = if s_r2 > 0 { 2 } else if s_r1 > 0 { 1 } else { 0 };
  {
    let mut rows_iter = VertPaddedIter::new(
      cdeffed,
      cdeffed,
      cdef_h,
      cdef_h,
      max_r)
    .map(|row: &[T]| {
      let left_w = max_r + 2;
      let right_w = max_r + 1;
      HorzPaddedIter::new(
        &row[..cdef_w],
        -(left_w as isize),
        left_w + cdef_w + right_w
      )
    });
    setup_integral_image(
      &mut rows_iter,
      &mut integral_image,
      &mut sq_integral_image,
      INTEGRAL_IMAGE_STRIDE
    );
  }

  /* prime the intermediate arrays */
  if s_r2 > 0 {
    sgrproj_box_ab_r2(&mut a_r2[0], &mut b_r2[0],
                      &integral_image, &sq_integral_image,
                      INTEGRAL_IMAGE_STRIDE,
                      0, cdef_h, s_r2, bdm8);
    sgrproj_box_ab_r2(&mut a_r2[1], &mut b_r2[1],
                      &integral_image, &sq_integral_image,
                      INTEGRAL_IMAGE_STRIDE,
                      1, cdef_h, s_r2, bdm8);
  }
  if s_r1 > 0 {
    let r_diff = max_r - 1;
    let integral_image_offset = r_diff + r_diff * INTEGRAL_IMAGE_STRIDE;
    sgrproj_box_ab_r1(&mut a_r1[0], &mut b_r1[0],
                      &integral_image[integral_image_offset..],
                      &sq_integral_image[integral_image_offset..],
                      INTEGRAL_IMAGE_STRIDE,
                      0, cdef_h, s_r1, bdm8);
    sgrproj_box_ab_r1(&mut a_r1[1], &mut b_r1[1],
                      &integral_image[integral_image_offset..],
                      &sq_integral_image[integral_image_offset..],
                      INTEGRAL_IMAGE_STRIDE,
                      1, cdef_h, s_r1, bdm8);
  }

  /* iterate by column */
  for xi in 0..cdef_w {
    /* build intermediate array columns */
    if s_r2 > 0 {
      sgrproj_box_ab_r2(&mut a_r2[(xi+2)%3], &mut b_r2[(xi+2)%3],
                        &integral_image, &sq_integral_image,
                        INTEGRAL_IMAGE_STRIDE,
                        xi as isize + 2, cdef_h, s_r2, bdm8);
      let ap0: [&[u32; 64+2]; 3] = [&a_r2[xi%3], &a_r2[(xi+1)%3], &a_r2[(xi+2)%3]];
      let bp0: [&[u32; 64+2]; 3] = [&b_r2[xi%3], &b_r2[(xi+1)%3], &b_r2[(xi+2)%3]];
      sgrproj_box_f_r2(&ap0, &bp0, &mut f_r2, xi, 0, cdef_h as usize, &cdeffed);
    } else {
      sgrproj_box_f_r0(&mut f_r2, xi, 0, cdef_h as usize, &cdeffed);
    }
    if s_r1 > 0 {
      let r_diff = max_r - 1;
      let integral_image_offset = r_diff + r_diff * INTEGRAL_IMAGE_STRIDE;
      sgrproj_box_ab_r1(&mut a_r1[(xi+2)%3], &mut b_r1[(xi+2)%3],
                        &integral_image[integral_image_offset..],
                        &sq_integral_image[integral_image_offset..],
                        INTEGRAL_IMAGE_STRIDE,
                        xi as isize + 2, cdef_h, s_r1, bdm8);
      let ap1: [&[u32; 64+2]; 3] = [&a_r1[xi%3], &a_r1[(xi+1)%3], &a_r1[(xi+2)%3]];
      let bp1: [&[u32; 64+2]; 3] = [&b_r1[xi%3], &b_r1[(xi+1)%3], &b_r1[(xi+2)%3]];
      sgrproj_box_f_r1(&ap1, &bp1, &mut f_r1, xi, 0, cdef_h as usize, &cdeffed);
    } else {
      sgrproj_box_f_r0(&mut f_r1, xi, 0, cdef_h as usize, &cdeffed);
    }

    for yi in 0..cdef_h {
      let u = i32::cast_from(cdeffed.p(xi,yi)) << SGRPROJ_RST_BITS;
      let s = i32::cast_from(input.p(xi,yi)) << SGRPROJ_RST_BITS;
      let f2 = f_r2[yi] as i32 - u;
      let f1 = f_r1[yi] as i32 - u;
      h[0][0] += f2 as f64 * f2 as f64;
      h[1][1] += f1 as f64 * f1 as f64;
      h[0][1] += f1 as f64 * f2 as f64;
      c[0] += f2 as f64 * s as f64;
      c[1] += f1 as f64 * s as f64;
    }
  }

  // this is lifted almost in-tact from libaom
  let n = cdef_w as f64 * cdef_h as f64;
  h[0][0] /= n;
  h[0][1] /= n;
  h[1][1] /= n;
  h[1][0] = h[0][1];
  c[0] /= n;
  c[1] /= n;
  let (xq0, xq1) = if s_r2 == 0 {
    // H matrix is now only the scalar h[1][1]
    // C vector is now only the scalar c[1]
    if h[1][1] == 0. {
      (0, 0)
    } else {
      (0, (c[1] / h[1][1]).round() as i32)
    }
  } else if s_r1 == 0 {
    // H matrix is now only the scalar h[0][0]
    // C vector is now only the scalar c[0]
    if h[0][0] == 0. {
      (0, 0)
    } else {
      ((c[0] / h[0][0]).round() as i32, 0)
    }
  } else {
    let det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
    if det == 0. {
      (0, 0)
    } else {
      // If scaling up dividend would overflow, instead scale down the divisor
      let div1 = (h[1][1] * c[0] - h[0][1] * c[1]) * (1 << SGRPROJ_PRJ_BITS) as f64;
      let div2 = (h[0][0] * c[1] - h[1][0] * c[0]) * (1 << SGRPROJ_PRJ_BITS) as f64;

      ((div1 / det).round() as i32, (div2 / det).round() as i32)
    }
  };
  (clamp(xq0, SGRPROJ_XQD_MIN[0] as i32, SGRPROJ_XQD_MAX[0] as i32) as i8,
   clamp(xq1, SGRPROJ_XQD_MIN[1] as i32, SGRPROJ_XQD_MAX[1] as i32) as i8)
}

fn wiener_stripe_filter<T: Pixel>(coeffs: [[i8; 3]; 2], fi: &FrameInvariants<T>,
                                  crop_w: usize, crop_h: usize,
                                  stripe_w: usize, stripe_h: usize,
                                  stripe_x: usize, stripe_y: isize,
                                  cdeffed: &Plane<T>, deblocked: &Plane<T>, out: &mut Plane<T>) {
  let bit_depth = fi.sequence.bit_depth;
  let round_h = if bit_depth == 12 {5} else {3};
  let round_v = if bit_depth == 12 {9} else {11};
  let offset = 1 << (bit_depth + WIENER_BITS - round_h - 1);
  let limit = (1 << (bit_depth + 1 + WIENER_BITS - round_h)) - 1;

  let mut work: [i32; MAX_SB_SIZE+7] = [0; MAX_SB_SIZE+7];
  let vfilter: [i32; 7] = [ coeffs[0][0] as i32,
                            coeffs[0][1] as i32,
                            coeffs[0][2] as i32,
                            128 - 2 * (coeffs[0][0] as i32 +
                                       coeffs[0][1] as i32 +
                                       coeffs[0][2] as i32 ),
                            coeffs[0][2] as i32,
                            coeffs[0][1] as i32,
                            coeffs[0][0] as i32];
  let hfilter: [i32; 7] = [ coeffs[1][0] as i32,
                            coeffs[1][1] as i32,
                            coeffs[1][2] as i32,
                            128 - 2 * (coeffs[1][0] as i32 +
                                       coeffs[1][1] as i32 +
                                       coeffs[1][2] as i32),
                            coeffs[1][2] as i32,
                            coeffs[1][1] as i32,
                            coeffs[1][0] as i32];

  // unlike x, our y can be negative to start as the first stripe
  // starts off the top of the frame by 8 pixels, and can also run off the end of the frame
  let start_wi = if stripe_y < 0 {-stripe_y} else {0} as usize;
  let start_yi = if stripe_y < 0 {0} else {stripe_y} as usize;
  let end_i = cmp::max(0, if stripe_h as isize + stripe_y > crop_h as isize {
    crop_h as isize - stripe_y - start_wi as isize
  } else {
    stripe_h as isize - start_wi as isize
  }) as usize;

  let mut out_slice = out.mut_slice(PlaneOffset{x: 0, y: start_yi as isize});

  for xi in stripe_x..stripe_x+stripe_w {
    let n = cmp::min(7, crop_w as isize + 3 - xi as isize);
    for yi in stripe_y - 3..stripe_y + stripe_h as isize + 4 {
      let src_plane: &Plane<T>;
      let mut acc = 0;
      let ly;
      if yi < stripe_y {
        ly = cmp::max(clamp(yi, 0, crop_h as isize - 1), stripe_y - 2) as usize;
        src_plane = deblocked;
      } else if yi < stripe_y+stripe_h as isize {
        ly = clamp(yi, 0, crop_h as isize - 1) as usize;
        src_plane = cdeffed;
      } else {
        ly = cmp::min(clamp(yi, 0, crop_h as isize - 1), stripe_y + stripe_h as isize + 1) as usize;
        src_plane = deblocked;
      }

      for i in 0..3 - xi as isize {
        acc += hfilter[i as usize] * i32::cast_from(src_plane.p(0, ly));
      }
      for i in cmp::max(0,3 - (xi as isize))..n {
        acc += hfilter[i as usize] * i32::cast_from(src_plane.p((xi as isize + i - 3) as usize, ly));
      }
      for i in n..7 {
        acc += hfilter[i as usize] * i32::cast_from(src_plane.p(crop_w - 1, ly));
      }

      acc = (acc + (1 << round_h >> 1)) >> round_h;
      work[(yi-stripe_y+3) as usize] = clamp(acc, -offset, limit-offset);
    }

    for (wi, dst) in (start_wi..start_wi+end_i).zip(out_slice.rows_iter_mut().map(|row| &mut row[xi]).take(end_i)) {
      let mut acc = 0;
      for (i,src) in (0..7).zip(work[wi..wi+7].iter_mut()) {
        acc += vfilter[i] * *src;
      }
      *dst = T::cast_from(clamp((acc + (1 << round_v >> 1)) >> round_v, 0, (1 << bit_depth) - 1));
    }
  }
}

#[derive(Copy, Clone, Debug)]
pub struct RestorationUnit {
  pub filter: RestorationFilter,
}

impl RestorationUnit {
  pub fn default() -> RestorationUnit {
    RestorationUnit {
      filter: RestorationFilter::default(),
    }
  }
}

#[derive(Clone, Debug)]
pub struct FrameRestorationUnits {
  units: Box<[RestorationUnit]>,
  pub cols: usize,
  pub rows: usize,
}

impl FrameRestorationUnits {
  pub fn new(cols: usize, rows: usize) -> Self {
    Self {
      units: vec![RestorationUnit::default(); cols * rows].into_boxed_slice(),
      cols,
      rows,
    }
  }
}

impl Index<usize> for FrameRestorationUnits {
  type Output = [RestorationUnit];
  #[inline(always)]
  fn index(&self, index: usize) -> &Self::Output {
    &self.units[index * self.cols..(index + 1) * self.cols]
  }
}

impl IndexMut<usize> for FrameRestorationUnits {
  #[inline(always)]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    &mut self.units[index * self.cols..(index + 1) * self.cols]
  }
}

#[derive(Clone, Debug)]
pub struct RestorationPlaneConfig {
  pub lrf_type: u8,
  pub unit_size: usize,
  // (1 << sb_shift) gives the number of superblocks both horizontally and
  // vertically in a restoration unit, not accounting for RU stretching
  pub sb_shift: usize,
  // stripe height is 64 in all cases except 4:2:0 chroma planes where
  // it is 32.  This is independent of all other setup parameters
  pub stripe_height: usize,
  pub cols: usize,
  pub rows: usize,
}

#[derive(Clone, Debug)]
pub struct RestorationPlane {
  pub cfg: RestorationPlaneConfig,
  pub units: FrameRestorationUnits,
}

#[derive(Clone, Default)]
pub struct RestorationPlaneOffset {
  pub row: usize,
  pub col: usize
}

impl RestorationPlane {
  pub fn new(lrf_type: u8, unit_size: usize, sb_shift: usize, stripe_decimate: usize,
             cols: usize, rows: usize) -> RestorationPlane {
    let stripe_height = if stripe_decimate != 0 {32} else {64};
    RestorationPlane {
      cfg: RestorationPlaneConfig {
        lrf_type,
        unit_size,
        sb_shift,
        stripe_height,
        cols,
        rows,
      },
      units: FrameRestorationUnits::new(cols, rows),
    }
  }

  // Stripes are always 64 pixels high in a non-subsampled
  // frame, and decimated from 64 pixels in chroma.  When
  // filtering, they are not co-located on Y with superblocks.
  fn restoration_unit_index_by_stripe(&self, stripenum: usize, rux: usize) -> (usize, usize) {
    (
      cmp::min(rux, self.cfg.cols - 1),
      cmp::min(stripenum * self.cfg.stripe_height / self.cfg.unit_size, self.cfg.rows - 1),
    )
  }

  pub fn restoration_unit_by_stripe(&self, stripenum: usize, rux: usize) -> &RestorationUnit {
    let (x, y) = self.restoration_unit_index_by_stripe(stripenum, rux);
    &self.units[y][x]
  }
}

#[derive(Clone, Debug)]
pub struct RestorationState {
  pub planes: [RestorationPlane; PLANES]
}

impl RestorationState {
  pub fn new<T: Pixel>(fi: &FrameInvariants<T>, input: &Frame<T>) -> Self {
    let PlaneConfig { xdec, ydec, .. } = input.planes[1].cfg;
    let stripe_uv_decimate = if xdec>0 && ydec>0 {1} else {0};
    // Currrently opt for smallest possible restoration unit size (1
    // superblock) This is *temporary*.  Counting on it will break
    // very shortly; the 1-superblock hardwiring is only until the
    // upper level encoder is capable of dealing with the delayed
    // writes that RU size > SB size will require.
    let lrf_y_shift = if fi.sequence.use_128x128_superblock {1} else {2};
    let lrf_uv_shift = lrf_y_shift + stripe_uv_decimate;

    // derive the rest
    let y_unit_log2 = RESTORATION_TILESIZE_MAX_LOG2 - lrf_y_shift;
    let uv_unit_log2 = RESTORATION_TILESIZE_MAX_LOG2 - lrf_uv_shift;
    let y_unit_size = 1 << y_unit_log2;
    let uv_unit_size = 1 << uv_unit_log2;
    let y_sb_log2 = if fi.sequence.use_128x128_superblock {7} else {6};
    let uv_sb_log2 = y_sb_log2 - stripe_uv_decimate;
    let cols = ((fi.width + (y_unit_size >> 1)) / y_unit_size).max(1);
    let rows = ((fi.height + (y_unit_size >> 1)) / y_unit_size).max(1);

    RestorationState {
      planes: [
        RestorationPlane::new(RESTORE_SWITCHABLE, y_unit_size, y_unit_log2 - y_sb_log2,
                              0, cols, rows),
        RestorationPlane::new(RESTORE_SWITCHABLE, uv_unit_size, uv_unit_log2 - uv_sb_log2,
                              stripe_uv_decimate, cols, rows),
        RestorationPlane::new(RESTORE_SWITCHABLE, uv_unit_size, uv_unit_log2 - uv_sb_log2,
                              stripe_uv_decimate, cols, rows)
      ],
    }
  }

  pub fn lrf_filter_frame<T: Pixel>(&mut self, out: &mut Frame<T>, pre_cdef: &Frame<T>,
                                    fi: &FrameInvariants<T>) {
    let cdeffed = out.clone();

    // unlike the other loop filters that operate over the padded
    // frame dimensions, restoration filtering and source pixel
    // accesses are clipped to the original frame dimensions
    // that's why we use fi.width and fi.height instead of PlaneConfig fields

    // number of stripes (counted according to colocated Y luma position)
    let stripe_n = (fi.height + 7) / 64 + 1;

    for pli in 0..PLANES {
      let rp = &self.planes[pli];
      let xdec = out.planes[pli].cfg.xdec;
      let ydec = out.planes[pli].cfg.ydec;
      let crop_w = (fi.width + (1 << xdec >> 1)) >> xdec;
      let crop_h = (fi.height + (1 << ydec >> 1)) >> ydec;

      for si in 0..stripe_n {
        let (stripe_start_y, stripe_size) = if si == 0 {
          (0, (64 - 8) >> ydec)
        }  else {
          (
            (si as isize * 64 - 8) >> ydec,
            // one past, unlike spec
            64 >> ydec
          )
        };

        // horizontally, go rdu-by-rdu
        for rux in 0..rp.cfg.cols {
          // stripe x pixel locations must be clipped to frame, last may need to stretch
          let x = rux * rp.cfg.unit_size;
          let size = if rux == rp.cfg.cols - 1 {
            crop_w - x
          } else {
            rp.cfg.unit_size
          };
          let ru = rp.restoration_unit_by_stripe(si, rux);
          match ru.filter {
            RestorationFilter::Wiener{coeffs} => {
              wiener_stripe_filter(coeffs, fi,
                                   crop_w, crop_h,
                                   size, stripe_size,
                                   x, stripe_start_y,
                                   &cdeffed.planes[pli], &pre_cdef.planes[pli],
                                   &mut out.planes[pli]);
            },
            RestorationFilter::Sgrproj{set, xqd} => {
              sgrproj_stripe_filter(set, xqd, fi,
                                    crop_w - x,
                                    (crop_h as isize - stripe_start_y) as usize,
                                    size, stripe_size,
                                    &cdeffed.planes[pli].slice(PlaneOffset{x: x as isize,
                                                                           y: stripe_start_y}),
                                    &pre_cdef.planes[pli].slice(PlaneOffset{x: x as isize,
                                                                            y: stripe_start_y}),
                                    &mut out.planes[pli].mut_slice(PlaneOffset{x: x as isize,
                                                                               y: stripe_start_y}));
            },
            RestorationFilter::None => {
              // do nothing
            }
          }
        }
      }
    }
  }
}
