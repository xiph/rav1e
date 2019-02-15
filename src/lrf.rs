// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(safe_extern_statics)]

use crate::encoder::Frame;
use crate::encoder::FrameInvariants;
use crate::context::SuperBlockOffset;
use crate::context::PLANES;
use crate::context::MAX_SB_SIZE;
use crate::plane::Plane;
use crate::plane::PlaneSlice;
use crate::plane::PlaneMutSlice;
use crate::plane::PlaneOffset;
use crate::plane::PlaneConfig;
use std::cmp;
use crate::util::clamp;
use crate::util::CastFromPrimitive;
use crate::util::Pixel;

pub const RESTORATION_TILESIZE_MAX_LOG2: usize = 8;

pub const RESTORE_NONE: u8 = 0;
pub const RESTORE_SWITCHABLE: u8 = 1;
pub const RESTORE_WIENER: u8 = 2;
pub const RESTORE_SGRPROJ: u8 = 3;

pub const WIENER_TAPS_MIN: [i8; 3] = [ -5, -23, -17 ];
pub const WIENER_TAPS_MID: [i8; 3] = [ 3, -7, 15 ];
pub const WIENER_TAPS_MAX: [i8; 3] = [ 10, 8, 46 ];
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
pub const SGRPROJ_PARAMS_S: [[i32; 2]; 1 << SGRPROJ_PARAMS_BITS] = [
  [140, 3236], [112, 2158], [ 93, 1618], [ 80, 1438],
  [ 70, 1295], [ 58, 1177], [ 47, 1079], [ 37,  996],
  [ 30,  925], [ 25,  863], [  0, 2589], [  0, 1618],
  [  0, 1177], [  0,  925], [ 56,    0], [ 22,    0]
];

#[derive(Copy, Clone, Debug)]
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

fn sgrproj_sum_finish(ssq: i32, sum: i32, n: i32, one_over_n: i32, s: i32, bdm8: usize) -> (i32, i32) {
  let scaled_ssq = ssq + (1 << 2*bdm8 >> 1) >> 2*bdm8;
  let scaled_sum = sum + (1 << bdm8 >> 1) >> bdm8;
  let p = cmp::max(0, scaled_ssq*(n as i32) - scaled_sum*scaled_sum);
  let z = p*s + (1 << SGRPROJ_MTABLE_BITS >> 1) >> SGRPROJ_MTABLE_BITS;
  let a = if z >= 255 {
    256
  } else if z == 0 {
    1
  } else {
    ((z << SGRPROJ_SGR_BITS) + z/2) / (z+1)
  };
  let b = ((1 << SGRPROJ_SGR_BITS) - a ) * sum * one_over_n;
  (a, b + (1 << SGRPROJ_RECIP_BITS >> 1) >> SGRPROJ_RECIP_BITS)
}

// The addressing below is a bit confusing, made worse by LRF's odd
// clipping requirements, and our reusing code for partial frames.  So
// I'm documenting the LRF conventions here in detail.

// 'Relative to plane storage' means that a coordinate or bound is
// being applied as if to the full Plane backing the PlaneSlice.  For
// example, a PlaneSlice may represent a subset of a middle of a
// plane, but when we say the top/left bounds are clipped 'relative to
// plane storage', that means relative to 0,0 of the plane, not 0,0 of
// the plane slice.

// 'Relative to the slice view' means that a coordinate or bound is
// counted from the 0,0 of the PlaneSlice, not the Plane from which it
// was sliced.

// Passed in plane slices may be the same size or different sizes;
// filter access will be clipped to 0,0..w,h of the underlying plane
// storage for both planes, depending which is accessed.  Note that
// the passed in w/h that specifies the storage clipping is actually
// relative to the the slice view, not the plane storage (it
// simplifies the math internally).  Eg, if a PlaceSlice has a y
// offset of -2 (meaning its origin is two rows above the top row of
// the backing plane), and we pass in a height of 4, the rows
// 0,1,2,3,4 of the slice address -2, -1, 0, 1, 2 of the backing plane
// with access clipped to 0, 0, 0, 1, 1.

// Active area cropping is done by specifying a w,h smaller
// than the actual underlying plane storage.

// stripe_y is the beginning of the current stripe (used for source
// buffer choice/clipping) relative to the passed in plane view.  It
// may (and regularly will) be negative.

// stripe_h is the hright of the current stripe, again used for source
// buffer choice/clipping).  It may specify a stripe boundary less
// than, eqqal to, or larger than the buffers we're accessing.

// x and y specify the center pixel of the current filter kernel
// application.  They are relative to the passed in slice views.

fn sgrproj_box_sum_slow<T: Pixel>(a: &mut i32, b: &mut i32,
                                  stripe_y: isize, stripe_h: usize,
                                  x: isize, y: isize,
                                  r: usize, n: i32, one_over_n: i32, s: i32, bdm8: usize,
                                  backing: &PlaneSlice<T>, backing_w: usize, backing_h: usize,
                                  cdeffed: &PlaneSlice<T>, cdeffed_w: usize, cdeffed_h: usize) {  
  let mut ssq:i32 = 0;
  let mut sum:i32 = 0;

  for yi in y-r as isize..=y+r as isize {
    let src_plane;
    let src_w;
    let src_h;
    
    // decide if we're vertically inside or outside the stripe
    if yi >= stripe_y && yi < stripe_y + stripe_h as isize {
      src_plane = cdeffed;
      src_w = (cdeffed_w as isize - x + r as isize) as usize;
      src_h = cdeffed_h as isize;
    } else {
      src_plane = backing;
      src_w = (backing_w as isize - x + r as isize) as usize;
      src_h = backing_h as isize;
    }
    // clamp vertically to storage at top and passed-in height at bottom 
    let cropped_y = clamp(yi, -src_plane.y, src_h - 1);
    // clamp vertically to stripe limits
    let ly = clamp(cropped_y, stripe_y - 2, stripe_y + stripe_h as isize + 1);
    // Reslice to avoid a negative X index.
    let p = src_plane.reslice(x - r as isize,ly).as_slice();
    // left-hand addressing limit
    let left = cmp::max(0, r as isize - x - src_plane.x) as usize;
    // right-hand addressing limit
    let right = cmp::min(2*r+1, src_w);

    // run accumulation to left of frame storage (if any)
    for _xi in 0..left {
      let c = i32::cast_from(p[(r as isize - x) as usize]);
      ssq += c*c;
      sum += c;
    }
    // run accumulation in-frame
    for xi in left..right {
      let c = i32::cast_from(p[xi]);
      ssq += c*c;
      sum += c;
    }
    // run accumulation to right of frame (if any)
    for _xi in right..2*r+1 {
      let c = i32::cast_from(p[src_w - 1]);
      ssq += c*c;
      sum += c;
    }
  }
  let (reta, retb) = sgrproj_sum_finish(ssq, sum, n, one_over_n, s, bdm8);
  *a = reta;
  *b = retb;
}

// unrolled computation to be used when all bounds-checking has been satisfied.
fn sgrproj_box_sum_fastxy_r1<T: Pixel>(a: &mut i32, b: &mut i32, x: isize, y: isize,
                                       s: i32, bdm8: usize, p: &PlaneSlice<T>) {
  let mut ssq:i32 = 0;
  let mut sum:i32 = 0;
  for yi in -1..=1 {
    let x = p.reslice(x - 1, y + yi).as_slice();
    ssq += i32::cast_from(x[0]) * i32::cast_from(x[0]) +
      i32::cast_from(x[1]) * i32::cast_from(x[1]) +
      i32::cast_from(x[2]) * i32::cast_from(x[2]);
    sum += i32::cast_from(x[0]) + i32::cast_from(x[1]) + i32::cast_from(x[2]);
  }
  let (reta, retb) = sgrproj_sum_finish(ssq, sum, 9, 455, s, bdm8);
  *a = reta;
  *b = retb;
}

fn sgrproj_box_sum_fastxy_r2<T: Pixel>(a: &mut i32, b: &mut i32, x: isize, y: isize,
                                       s: i32, bdm8: usize, p: &PlaneSlice<T>) {
  let mut ssq:i32 = 0;
  let mut sum:i32 = 0;
  for yi in -2..=2 {
    let x = p.reslice(x - 2, y + yi).as_slice();
    ssq += i32::cast_from(x[0]) * i32::cast_from(x[0]) +
      i32::cast_from(x[1]) * i32::cast_from(x[1]) +
      i32::cast_from(x[2]) * i32::cast_from(x[2]) +
      i32::cast_from(x[3]) * i32::cast_from(x[3]) +
      i32::cast_from(x[4]) * i32::cast_from(x[4]);
    sum += i32::cast_from(x[0]) + i32::cast_from(x[1]) + i32::cast_from(x[2]) +
      i32::cast_from(x[3]) + i32::cast_from(x[4]);
  }
  let (reta, retb) = sgrproj_sum_finish(ssq, sum, 25, 164, s, bdm8);
  *a = reta;
  *b = retb;
}

// unrolled computation to be used when only X bounds-checking has been satisfied.
fn sgrproj_box_sum_fastx_r1<T: Pixel>(a: &mut i32, b: &mut i32,
                                      stripe_y: isize, stripe_h: usize,
                                      x: isize, y: isize,
                                      s: i32, bdm8: usize,
                                      backing: &PlaneSlice<T>, backing_h: usize,
                                      cdeffed: &PlaneSlice<T>, cdeffed_h: usize) {  
  let mut ssq:i32 = 0;
  let mut sum:i32 = 0;
  for yi in y-1..=y+1 {
    let src_plane;
    let src_h;
    
    // decide if we're vertically inside or outside the stripe
    if yi >= stripe_y && yi < stripe_y + stripe_h as isize {
      src_plane = cdeffed;
      src_h = cdeffed_h as isize;
    } else {
      src_plane = backing;
      src_h = backing_h as isize;
    }
    // clamp vertically to storage addressing limit
    let cropped_y = clamp(yi, -src_plane.y, src_h - 1);
    // clamp vertically to stripe limits
    let ly = clamp(cropped_y, stripe_y - 2, stripe_y + stripe_h as isize + 1);
    let x = src_plane.reslice(x - 1, ly).as_slice();
    ssq += i32::cast_from(x[0]) * i32::cast_from(x[0]) +
      i32::cast_from(x[1]) * i32::cast_from(x[1]) +
      i32::cast_from(x[2]) * i32::cast_from(x[2]);
    sum += i32::cast_from(x[0]) + i32::cast_from(x[1]) + i32::cast_from(x[2]);
  }
  let (reta, retb) = sgrproj_sum_finish(ssq, sum, 9, 455, s, bdm8);
  *a = reta;
  *b = retb;
}

fn sgrproj_box_sum_fastx_r2<T: Pixel>(a: &mut i32, b: &mut i32,
                                      stripe_y: isize, stripe_h: usize,
                                      x: isize, y: isize,
                                      s: i32, bdm8: usize,
                                      backing: &PlaneSlice<T>, backing_h: usize,
                                      cdeffed: &PlaneSlice<T>, cdeffed_h: usize) {  
  let mut ssq:i32 = 0;
  let mut sum:i32 = 0;
  for yi in y - 2..=y + 2 {
    let src_plane;
    let src_h;
    
    // decide if we're vertically inside or outside the stripe
    if yi >= stripe_y && yi < stripe_y + stripe_h as isize {
      src_plane = cdeffed;
      src_h = cdeffed_h as isize;
    } else {
      src_plane = backing;
      src_h = backing_h as isize;
    }
    // clamp vertically to storage addressing limit
    let cropped_y = clamp(yi, -src_plane.y, src_h as isize - 1);
    // clamp vertically to stripe limits
    let ly = clamp(cropped_y, stripe_y - 2, stripe_y + stripe_h as isize + 1);
    let x = src_plane.reslice(x - 2, ly).as_slice();
    ssq += i32::cast_from(x[0]) * i32::cast_from(x[0]) +
      i32::cast_from(x[1]) * i32::cast_from(x[1]) +
      i32::cast_from(x[2]) * i32::cast_from(x[2]) +
      i32::cast_from(x[3]) * i32::cast_from(x[3]) +
      i32::cast_from(x[4]) * i32::cast_from(x[4]);
    sum += i32::cast_from(x[0]) + i32::cast_from(x[1]) + i32::cast_from(x[2]) +
      i32::cast_from(x[3]) + i32::cast_from(x[4]);
  }
  let (reta, retb) = sgrproj_sum_finish(ssq, sum, 25, 164, s, bdm8);
  *a = reta;
  *b = retb;
}

// computes an intermediate (ab) column for rows stripe_y through
// stripe_y+stripe_h (no inclusize) at column stripe_x.
// r=1 case computes every row as every row is used (see r2 version below)
fn sgrproj_box_ab_r1<T: Pixel>(af: &mut[i32; 64+2],
                               bf: &mut[i32; 64+2],
                               stripe_x: isize, stripe_y: isize, stripe_h: usize,
                               s: i32, bdm8: usize,
                               backing: &PlaneSlice<T>, backing_w: usize, backing_h: usize,
                               cdeffed: &PlaneSlice<T>, cdeffed_w: usize, cdeffed_h: usize) {
  // we will fill the af and bf arrays from 0..stripe_h+1 (ni),
  // representing stripe_y-1 to stripe_y+stripe_h+1 inclusive
  let boundary0 = 0;
  let boundary3 = stripe_h + 2;
  if backing.x + stripe_x > 0 && stripe_x < backing_w as isize - 1 &&
    cdeffed.x + stripe_x > 0 && stripe_x < cdeffed_w as isize - 1 {
    // Addressing is away from left and right edges of cdeffed storage;
    // no X clipping to worry about, but the top/bottom few rows still
    // need to worry about storage and stripe limits
      
    // boundary1 is the point where we're guaranteed all our y
    // addressing will be both in the stripe and in cdeffed storage  
    let boundary1 = cmp::max(2, 2 - cdeffed.y - stripe_y) as usize;
    // boundary 2 is when we have to bounds check along the bottom of
    // the stripe or bottom of storage
    let boundary2 = cmp::min(cdeffed_h as isize - stripe_y - 1, stripe_h as isize - 1) as usize;

    // top rows (if any), away from left and right columns
    for i in boundary0..boundary1 {
      sgrproj_box_sum_fastx_r1(&mut af[i], &mut bf[i],
                               stripe_y, stripe_h, 
                               stripe_x, stripe_y + i as isize - 1,
                               s, bdm8,
                               backing, backing_h,
                               cdeffed, cdeffed_h);
    }
    // middle rows, away from left and right columns
    for i in boundary1..boundary2 {
      sgrproj_box_sum_fastxy_r1(&mut af[i], &mut bf[i],
                                stripe_x, stripe_y + i as isize - 1, s, bdm8, cdeffed);
    }
    // bottom rows (if any), away from left and right columns
    for i in boundary2..boundary3 {
      sgrproj_box_sum_fastx_r1(&mut af[i], &mut bf[i],
                               stripe_y, stripe_h, 
                               stripe_x, stripe_y + i as isize - 1,
                               s, bdm8,
                               backing, backing_h,
                               cdeffed, cdeffed_h);
    }
  } else {
    // top/bottom rows and left/right columns, where we need to worry about frame and stripe clipping
    for i in boundary0..boundary3 {
      sgrproj_box_sum_slow(&mut af[i], &mut bf[i],                           
                           stripe_y, stripe_h,
                           stripe_x, stripe_y + i as isize - 1,
                           1, 9, 455, s, bdm8,
                           backing, backing_w, backing_h,
                           cdeffed, cdeffed_w, cdeffed_h);
    }
  }
}

// One oddness about the radius=2 intermediate array computations that
// the spec doesn't make clear: Although the spec defines computation
// of every row (of a, b and f), only half of the rows (every-other
// row) are actually used.  We use the full-size array here but only
// compute the even rows.  This is not so much optimization as trying
// to illustrate what this convoluted filter is actually doing
// (ie not as much as it may appear).
fn sgrproj_box_ab_r2<T: Pixel>(af: &mut[i32; 64+2],
                               bf: &mut[i32; 64+2],
                               stripe_x: isize, stripe_y: isize, stripe_h: usize,
                               s: i32, bdm8: usize,
                               backing: &PlaneSlice<T>, backing_w: usize, backing_h: usize,
                               cdeffed: &PlaneSlice<T>, cdeffed_w: usize, cdeffed_h: usize) {  
  // we will fill the af and bf arrays from 0..stripe_h+1 (ni),
  // representing stripe_y-1 to stripe_y+stripe_h+1 inclusive
  let boundary0 = 0; // even
  let boundary3 = stripe_h + 2; // don't care if odd
  if backing.x + stripe_x > 1 && stripe_x < backing_w as isize - 2 &&
    cdeffed.x + stripe_x > 1 && stripe_x < cdeffed_w as isize - 2 {
    // Addressing is away from left and right edges of cdeffed storage;
    // no X clipping to worry about, but the top/bottom few rows still
    // need to worry about storage and stripe limits
      
    // boundary1 is the point where we're guaranteed all our y
    // addressing will be both in the stripe and in cdeffed storage
    // make even and round up  
    let boundary1 = (cmp::max(3, 3 - cdeffed.y - stripe_y) + 1 >> 1 << 1) as usize;
    // boundary 2 is when we have to bounds check along the bottom of
    // the stripe or bottom of storage
    // must be even, rounding of +1 cancels fencepost of -1
    let boundary2 = (cmp::min(cdeffed_h as isize - stripe_y, stripe_h as isize) >> 1 << 1) as usize;

    // top rows, away from left and right columns
    for i in (boundary0..boundary1).step_by(2) {
      sgrproj_box_sum_fastx_r2(&mut af[i], &mut bf[i],
                               stripe_y, stripe_h, 
                               stripe_x, stripe_y + i as isize - 1,
                               s, bdm8,
                               backing, backing_h,
                               cdeffed, cdeffed_h);
    }
    // middle rows, away from left and right columns
    for i in (boundary1..boundary2).step_by(2) {
      sgrproj_box_sum_fastxy_r2(&mut af[i], &mut bf[i],
                                stripe_x, stripe_y + i as isize - 1,
                                s, bdm8, cdeffed);
    }
    // bottom rows, away from left and right columns
    for i in (boundary2..boundary3).step_by(2) {
      sgrproj_box_sum_fastx_r2(&mut af[i], &mut bf[i],
                               stripe_y, stripe_h, 
                               stripe_x, stripe_y + i as isize - 1,
                               s, bdm8,
                               backing, backing_h,
                               cdeffed, cdeffed_h);
    }
  } else {
    // top/bottom rows and left/right columns, where we need to worry about frame and stripe clipping
    for i in (boundary0..boundary3).step_by(2) {
      sgrproj_box_sum_slow(&mut af[i], &mut bf[i],
                           stripe_y, stripe_h,
                           stripe_x, stripe_y + i as isize - 1,
                           2, 25, 164, s, bdm8,
                           backing, backing_w, backing_h,
                           cdeffed, cdeffed_w, cdeffed_h);
    }
  }
}

fn sgrproj_box_f_r0<T: Pixel>(f: &mut[i32; 64], x: usize, y: isize, h: usize, cdeffed: &PlaneSlice<T>) {
  for i in cmp::max(0, -y) as usize..h {
    f[i as usize] = (i32::cast_from(cdeffed.p(x, (y + i as isize) as usize))) << SGRPROJ_RST_BITS;
  }
}

fn sgrproj_box_f_r1<T: Pixel>(af: &[&[i32; 64+2]; 3], bf: &[&[i32; 64+2]; 3], f: &mut[i32; 64],
                              x: usize, y: isize, h: usize, cdeffed: &PlaneSlice<T>) {
  let shift = 5 + SGRPROJ_SGR_BITS - SGRPROJ_RST_BITS;
  for i in cmp::max(0, -y) as usize..h {
    let a =
      3 * (af[0][i+0] + af[2][i+0] + af[0][i+2] + af[2][i+2]) +
      4 * (af[1][i+0] + af[0][i+1] + af[1][i+1] + af[2][i+1] + af[1][i+2]);
    let b =
      3 * (bf[0][i+0] + bf[2][i+0] + bf[0][i+2] + bf[2][i+2]) +
      4 * (bf[1][i+0] + bf[0][i+1] + bf[1][i+1] + bf[2][i+1] + bf[1][i+2]);
    let v = a * i32::cast_from(cdeffed.p(x, (y + i as isize) as usize)) + b;
    f[i as usize] = v + (1 << shift >> 1) >> shift;
  }
}

fn sgrproj_box_f_r2<T: Pixel>(af: &[&[i32; 64+2]; 3], bf: &[&[i32; 64+2]; 3], f: &mut[i32; 64],
                              x: usize, y: isize, h: usize, cdeffed: &PlaneSlice<T>) {
  let shift = 5 + SGRPROJ_SGR_BITS - SGRPROJ_RST_BITS;
  let shifto = 4 + SGRPROJ_SGR_BITS - SGRPROJ_RST_BITS;
  for i in (cmp::max(0, -y) as usize..h).step_by(2) {
    let a =
      5 * (af[0][i+0] + af[2][i+0]) + 
      6 * (af[1][i+0]);
    let b =
      5 * (bf[0][i+0] + bf[2][i+0]) + 
      6 * (bf[1][i+0]);
    let ao =
      5 * (af[0][i+2] + af[2][i+2]) + 
      6 * (af[1][i+2]);
    let bo =
      5 * (bf[0][i+2] + bf[2][i+2]) + 
      6 * (bf[1][i+2]);
    let v = (a + ao) * i32::cast_from(cdeffed.p(x, (y+i as isize) as usize)) + b + bo;
    f[i as usize] = v + (1 << shift >> 1) >> shift;
    let vo = ao * i32::cast_from(cdeffed.p(x, (y + i as isize) as usize + 1)) + bo;
    f[i as usize + 1] = vo + (1 << shifto >> 1) >> shifto;
  }
}

pub fn sgrproj_stripe_filter<T: Pixel>(set: u8, xqd: [i8; 2], fi: &FrameInvariants<T>,
                                       crop_w: usize, crop_h: usize,
                                       stripe_w: usize, stripe_h: usize,
                                       cdeffed: &PlaneSlice<T>,
                                       deblocked: &PlaneSlice<T>,
                                       out: &mut PlaneMutSlice<T>) {
  assert!(stripe_h <= 64);
  let bdm8 = fi.sequence.bit_depth - 8;
  let mut a_r2: [[i32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut b_r2: [[i32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut f_r2: [i32; 64] = [0; 64];
  let mut a_r1: [[i32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut b_r1: [[i32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut f_r1: [i32; 64] = [0; 64];

  let s_r2: i32 = SGRPROJ_PARAMS_S[set as usize][0];
  let s_r1: i32 = SGRPROJ_PARAMS_S[set as usize][1];

  let outstart = cmp::max(0, cmp::max(-cdeffed.y, -out.y)) as usize;
  let outstride = out.plane.cfg.stride; 
  let out_data = out.as_mut_slice();
  
  /* prime the intermediate arrays */
  if s_r2 > 0 {
    sgrproj_box_ab_r2(&mut a_r2[0], &mut b_r2[0],
                      -1, 0, stripe_h,
                      s_r2, bdm8,
                      &deblocked, crop_w, crop_h,
                      &cdeffed, crop_w, crop_h);
    sgrproj_box_ab_r2(&mut a_r2[1], &mut b_r2[1],
                      0, 0, stripe_h,
                      s_r2, bdm8,
                      &deblocked, crop_w, crop_h,
                      &cdeffed, crop_w, crop_h);
  }
  if s_r1 > 0 {
    sgrproj_box_ab_r1(&mut a_r1[0], &mut b_r1[0],
                      -1, 0, stripe_h,
                      s_r1, bdm8,
                      &deblocked, crop_w, crop_h,
                      &cdeffed, crop_w, crop_h);
    sgrproj_box_ab_r1(&mut a_r1[1], &mut b_r1[1],
                      0, 0, stripe_h,
                      s_r1, bdm8,
                      &deblocked, crop_w, crop_h,
                      &cdeffed, crop_w, crop_h);
  }

  /* iterate by column */
  for xi in 0..stripe_w {
    /* build intermediate array columns */
    if s_r2 > 0 {
      sgrproj_box_ab_r2(&mut a_r2[(xi+2)%3], &mut b_r2[(xi+2)%3],
                        xi as isize + 1, 0, stripe_h,
                        s_r2, bdm8,
                        &deblocked, crop_w, crop_h,
                        &cdeffed, crop_w, crop_h);
      let ap0: [&[i32; 64+2]; 3] = [&a_r2[xi%3], &a_r2[(xi+1)%3], &a_r2[(xi+2)%3]];
      let bp0: [&[i32; 64+2]; 3] = [&b_r2[xi%3], &b_r2[(xi+1)%3], &b_r2[(xi+2)%3]];
      sgrproj_box_f_r2(&ap0, &bp0, &mut f_r2, xi, 0, stripe_h as usize, &cdeffed);
    } else {
      sgrproj_box_f_r0(&mut f_r2, xi, 0, stripe_h as usize, &cdeffed);
    }
    if s_r1 > 0 {
      sgrproj_box_ab_r1(&mut a_r1[(xi+2)%3], &mut b_r1[(xi+2)%3],
                        xi as isize + 1, 0, stripe_h,
                        s_r1, bdm8,
                        &deblocked, crop_w, crop_h,
                        &cdeffed, crop_w, crop_h);
      let ap1: [&[i32; 64+2]; 3] = [&a_r1[xi%3], &a_r1[(xi+1)%3], &a_r1[(xi+2)%3]];
      let bp1: [&[i32; 64+2]; 3] = [&b_r1[xi%3], &b_r1[(xi+1)%3], &b_r1[(xi+2)%3]];

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
      let v = w0*f_r2[yi] + w1*u + w2*f_r1[yi];
      let s = v + (1 << SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS >> 1) >> SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS;
      out_data[xi + yi*outstride] = T::cast_from(clamp(s, 0, (1 << bit_depth) - 1));
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
  let bdm8 = fi.sequence.bit_depth - 8;
  let mut a_r2: [[i32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut b_r2: [[i32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut f_r2: [i32; 64] = [0; 64];
  let mut a_r1: [[i32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut b_r1: [[i32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut f_r1: [i32; 64] = [0; 64];

  let s_r2: i32 = SGRPROJ_PARAMS_S[set as usize][0];
  let s_r1: i32 = SGRPROJ_PARAMS_S[set as usize][1];

  let mut h:[[f64; 2]; 2] = [[0.,0.],[0.,0.]];
  let mut c:[f64; 2] = [0., 0.];

  /* prime the intermediate arrays */
  if s_r2 > 0 {
    sgrproj_box_ab_r2(&mut a_r2[0], &mut b_r2[0],
                      -1, 0, cdef_h,
                      s_r2, bdm8,
                      &cdeffed, cdef_w, cdef_h,
                      &cdeffed, cdef_w, cdef_h);
    sgrproj_box_ab_r2(&mut a_r2[1], &mut b_r2[1],
                      0, 0, cdef_h,
                      s_r2, bdm8,
                      &cdeffed, cdef_w, cdef_h,
                      &cdeffed, cdef_w, cdef_h);
  }
  if s_r1 > 0 {
    sgrproj_box_ab_r1(&mut a_r1[0], &mut b_r1[0],
                      -1, 0, cdef_h,
                      s_r1, bdm8,
                      &cdeffed, cdef_w, cdef_h,
                      &cdeffed, cdef_w, cdef_h);
    sgrproj_box_ab_r1(&mut a_r1[1], &mut b_r1[1],
                      0, 0, cdef_h,
                      s_r1, bdm8,
                      &cdeffed, cdef_w, cdef_h,
                      &cdeffed, cdef_w, cdef_h);
  }
  
  /* iterate by column */
  for xi in 0..cdef_w {
    /* build intermediate array columns */
    if s_r2 > 0 {
      sgrproj_box_ab_r2(&mut a_r2[(xi+2)%3], &mut b_r2[(xi+2)%3],
                        xi as isize + 1, 0, cdef_h,
                        s_r2, bdm8,
                        &cdeffed, cdef_w, cdef_h,
                        &cdeffed, cdef_w, cdef_h);
      let ap0: [&[i32; 64+2]; 3] = [&a_r2[xi%3], &a_r2[(xi+1)%3], &a_r2[(xi+2)%3]];
      let bp0: [&[i32; 64+2]; 3] = [&b_r2[xi%3], &b_r2[(xi+1)%3], &b_r2[(xi+2)%3]];
      sgrproj_box_f_r2(&ap0, &bp0, &mut f_r2, xi, 0, cdef_h as usize, &cdeffed);
    } else {
      sgrproj_box_f_r0(&mut f_r2, xi, 0, cdef_h as usize, &cdeffed);
    }
    if s_r1 > 0 {
      sgrproj_box_ab_r1(&mut a_r1[(xi+2)%3], &mut b_r1[(xi+2)%3],
                        xi as isize + 1, 0, cdef_h,
                        s_r1, bdm8,
                        &cdeffed, cdef_w, cdef_h,
                        &cdeffed, cdef_w, cdef_h);
      let ap1: [&[i32; 64+2]; 3] = [&a_r1[xi%3], &a_r1[(xi+1)%3], &a_r1[(xi+2)%3]];
      let bp1: [&[i32; 64+2]; 3] = [&b_r1[xi%3], &b_r1[(xi+1)%3], &b_r1[(xi+2)%3]];

      sgrproj_box_f_r1(&ap1, &bp1, &mut f_r1, xi, 0, cdef_h as usize, &cdeffed);
    } else {
      sgrproj_box_f_r0(&mut f_r1, xi, 0, cdef_h as usize, &cdeffed);
    }

    for yi in 0..cdef_h {
      let u = i32::cast_from(cdeffed.p(yi,xi)) << SGRPROJ_RST_BITS;
      let s = i32::cast_from(input.p(yi,xi)) << SGRPROJ_RST_BITS;
      let f2 = f_r2[yi] - u;
      let f1 = f_r1[yi] - u;
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
  let offset = 1 << bit_depth + WIENER_BITS - round_h - 1;
  let limit = (1 << bit_depth + 1 + WIENER_BITS - round_h) - 1;

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

  let stride = out.cfg.stride;
  let mut out_slice = out.mut_slice(&PlaneOffset{x: 0, y: start_yi as isize});
  let out_data = out_slice.as_mut_slice();

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

      acc = acc + (1 << round_h >> 1) >> round_h;
      work[(yi-stripe_y+3) as usize] = clamp(acc, -offset, limit-offset);
    }

    for (wi, dst) in (start_wi..start_wi+end_i).zip(out_data[xi..].iter_mut().step_by(stride).take(end_i)) {
      let mut acc = 0;
      for (i,src) in (0..7).zip(work[wi..wi+7].iter_mut()) {
        acc += vfilter[i] * *src;
      }
      *dst = T::cast_from(clamp(acc + (1 << round_v >> 1) >> round_v, 0, (1 << bit_depth) - 1));
    }
  }
}

#[derive(Copy, Clone, Debug)]
pub struct RestorationUnit {
  pub filter: RestorationFilter,
  pub coded: bool,
}

impl RestorationUnit {
  pub fn default() -> RestorationUnit {
    RestorationUnit {
      filter: RestorationFilter::default(),
      coded: false,
    }
  }
}

#[derive(Clone, Debug)]
pub struct RestorationPlane {
  pub lrf_type: u8,
  pub unit_size: usize,
  // (1 << sb_shift) gives the number of superblocks having size 1 << SUPERBLOCK_TO_PLANE_SHIFT
  // both horizontally and vertically in a restoration unit, not accounting for RU stretching
  pub sb_shift: usize,
  // stripe height is 64 in all cases except 4:2:0 chroma planes where
  // it is 32.  This is independent of all other setup parameters
  pub stripe_height: usize,
  pub cols: usize,
  pub rows: usize,
  pub wiener_ref: [[i8; 3]; 2],
  pub sgrproj_ref: [i8; 2],
  pub units: Box<[RestorationUnit]>,
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
      lrf_type,
      unit_size,
      sb_shift,
      stripe_height,
      cols,
      rows,
      wiener_ref: [WIENER_TAPS_MID; 2],
      sgrproj_ref: SGRPROJ_XQD_MID,
      units: vec![RestorationUnit::default(); cols * rows].into_boxed_slice(),
    }
  }

  fn restoration_unit_index(&self, sbo: &SuperBlockOffset) -> (usize, usize) {
    (
      (sbo.x >> self.sb_shift).min(self.cols - 1),
      (sbo.y >> self.sb_shift).min(self.rows - 1),
    )
  }

  // Stripes are always 64 pixels high in a non-subsampled
  // frame, and decimated from 64 pixels in chroma.  When
  // filtering, they are not co-located on Y with superblocks.
  fn restoration_unit_index_by_stripe(&self, stripenum: usize, rux: usize) -> (usize, usize) {
    (
      cmp::min(rux, self.cols - 1),
      cmp::min(stripenum * self.stripe_height / self.unit_size, self.rows - 1),
    )
  }

  pub fn restoration_unit(&self, sbo: &SuperBlockOffset) -> &RestorationUnit {
    let (x, y) = self.restoration_unit_index(sbo);
    &self.units[y * self.cols + x]
  }

  pub fn restoration_unit_as_mut(&mut self, sbo: &SuperBlockOffset) -> &mut RestorationUnit {
    let (x, y) = self.restoration_unit_index(sbo);
    &mut self.units[y * self.cols + x]
  }

  pub fn restoration_unit_by_stripe(&self, stripenum: usize, rux: usize) -> &RestorationUnit {
    let (x, y) = self.restoration_unit_index_by_stripe(stripenum, rux);
    &self.units[y * self.cols + x]
  }
}

#[derive(Clone, Debug)]
pub struct RestorationState {
  pub plane: [RestorationPlane; PLANES]
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
      plane: [RestorationPlane::new(RESTORE_SWITCHABLE, y_unit_size, y_unit_log2 - y_sb_log2,
                                    0, cols, rows),
              RestorationPlane::new(RESTORE_SWITCHABLE, uv_unit_size, uv_unit_log2 - uv_sb_log2,
                                    stripe_uv_decimate, cols, rows),
              RestorationPlane::new(RESTORE_SWITCHABLE, uv_unit_size, uv_unit_log2 - uv_sb_log2,
                                    stripe_uv_decimate, cols, rows)],
    }
  }

  pub fn restoration_unit(&self, sbo: &SuperBlockOffset, pli: usize) -> &RestorationUnit {
    self.plane[pli].restoration_unit(sbo)
  }

  pub fn restoration_unit_as_mut(&mut self, sbo: &SuperBlockOffset, pli: usize) -> &mut RestorationUnit {
    self.plane[pli].restoration_unit_as_mut(sbo)
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
      let rp = &self.plane[pli];
      let xdec = out.planes[pli].cfg.xdec;
      let ydec = out.planes[pli].cfg.ydec;
      let crop_w = fi.width + (1 << xdec >> 1) >> xdec;
      let crop_h = fi.height + (1 << ydec >> 1) >> ydec;

      for si in 0..stripe_n {
        // stripe y pixel locations must be able to overspan the frame
        let stripe_start_y = si as isize * 64 - 8 >> ydec;
        let stripe_size = 64 >> ydec; // one past, unlike spec

        // horizontally, go rdu-by-rdu
        for rux in 0..rp.cols {
          // stripe x pixel locations must be clipped to frame, last may need to stretch
          let x = rux * rp.unit_size;
          let size = if rux == rp.cols - 1 {
            crop_w - x
          } else {
            rp.unit_size
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
                                    &cdeffed.planes[pli].slice(&PlaneOffset{x: x as isize,
                                                                           y: stripe_start_y}),
                                    &pre_cdef.planes[pli].slice(&PlaneOffset{x: x as isize,
                                                                            y: stripe_start_y}),
                                    &mut out.planes[pli].mut_slice(&PlaneOffset{x: x as isize,
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
