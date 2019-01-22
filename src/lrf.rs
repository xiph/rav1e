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
    RestorationFilter::Sgrproj {set:4, xqd: SGRPROJ_XQD_MID}
  }
}

fn sgrproj_sum_finish(ssq: i32, sum: i32,
                      n: i32, one_over_n: i32, s: i32,
                      bit_depth: usize) -> (i32, i32) {
  let scaled_ssq = ssq + (1 << 2*(bit_depth-8) >> 1) >> 2*(bit_depth-8);
  let scaled_sum = sum + (1 << bit_depth - 8 >> 1) >> bit_depth - 8;
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
fn sgrproj_box_sum_slow<T: Pixel>(a: &mut i32, b: &mut i32, row: isize, r: usize,
                        crop_w: usize, crop_h: usize, stripe_h: usize,
                        stripe_x: isize, stripe_y: isize,
                        n: i32, one_over_n: i32, s: i32,
                        cdeffed: &Plane<T>, deblocked: &Plane<T>, bit_depth: usize) {  
  let xn = cmp::min(r as isize + 1, crop_w as isize - stripe_x);
  let mut ssq:i32 = 0;
  let mut sum:i32 = 0;

  for yi in row-r as isize..=row+r as isize {
    let src_plane = if yi >= 0 && yi < stripe_h as isize {cdeffed} else {deblocked};
    let ly = clamp(clamp(yi+stripe_y, 0, crop_h as isize - 1),
                   stripe_y - 2, stripe_y + stripe_h as isize + 1) as usize;

    for _xi in -(r as isize)..-stripe_x {
      let c:i32 = src_plane.p(0, ly).as_();
      ssq += c*c;
      sum += c;
    }
    for xi in cmp::max(-(r as isize), -stripe_x)..xn {
      let c:i32 = src_plane.p((xi + stripe_x) as usize, ly).as_();
      ssq += c*c;
      sum += c;
    }
    for _xi in xn..r as isize + 1 {
      let c:i32 = src_plane.p(crop_w - 1, ly).as_();
      ssq += c*c;
      sum += c;
    }
  }
  let (reta, retb) = sgrproj_sum_finish(ssq, sum, n, one_over_n, s, bit_depth);
  *a = reta;
  *b = retb;
}

// unrolled computation to be used when all bounds-checking has been pre-satisfied.
fn sgrproj_box_sum_fastxy_r1<T: Pixel>(a: &mut i32, b: &mut i32, stripe_x: isize, stripe_y: isize,
                                       s: i32, p: &Plane<T>, bit_depth: usize) {
  let mut ssq:i32 = 0;
  let mut sum:i32 = 0;
  for yi in -1..=1 {
    let x = p.slice(&PlaneOffset{x: stripe_x - 1, y: stripe_y + yi}).as_slice();
    ssq += i32::cast_from(x[0]) * i32::cast_from(x[0]) +
      i32::cast_from(x[1]) * i32::cast_from(x[1]) +
      i32::cast_from(x[2]) * i32::cast_from(x[2]);
    sum += i32::cast_from(x[0]) + i32::cast_from(x[1]) + i32::cast_from(x[2]);
  }
  let (reta, retb) = sgrproj_sum_finish(ssq, sum, 9, 455, s, bit_depth);
  *a = reta;
  *b = retb;
}

fn sgrproj_box_sum_fastxy_r2<T: Pixel>(a: &mut i32, b: &mut i32, stripe_x: isize, stripe_y: isize,
                                       s: i32, p: &Plane<T>, bit_depth: usize) {
  let mut ssq:i32 = 0;
  let mut sum:i32 = 0;
  for yi in -2..=2 {
    let x = p.slice(&PlaneOffset{x: stripe_x - 2, y: stripe_y + yi}).as_slice();
    ssq += i32::cast_from(x[0]) * i32::cast_from(x[0]) +
      i32::cast_from(x[1]) * i32::cast_from(x[1]) +
      i32::cast_from(x[2]) * i32::cast_from(x[2]) +
      i32::cast_from(x[3]) * i32::cast_from(x[3]) +
      i32::cast_from(x[4]) * i32::cast_from(x[4]);
    sum += i32::cast_from(x[0]) + i32::cast_from(x[1]) + i32::cast_from(x[2]) +
      i32::cast_from(x[3]) + i32::cast_from(x[4]);
  }
  let (reta, retb) = sgrproj_sum_finish(ssq, sum, 25, 164, s, bit_depth);
  *a = reta;
  *b = retb;
}

// unrolled computation to be used when X bounds-checking has been pre-satisfied.
fn sgrproj_box_sum_fastx_r1<T: Pixel>(a: &mut i32, b: &mut i32, row: isize,
                                      crop_h: usize, stripe_h: usize,
                                      stripe_x: isize, stripe_y: isize,
                                      s: i32, cdeffed: &Plane<T>, deblocked: &Plane<T>,
                                      bit_depth: usize) {  
  let mut ssq:i32 = 0;
  let mut sum:i32 = 0;
  for yi in row-1 as isize..=row+1 as isize {
    let p = if yi >= 0 && yi < stripe_h as isize {cdeffed} else {deblocked};
    let ly = clamp(clamp(yi+stripe_y, 0, crop_h as isize - 1),
                   stripe_y - 2, stripe_y + stripe_h as isize + 1);
    let x = p.slice(&PlaneOffset{x: stripe_x - 1, y: ly}).as_slice();
    ssq += i32::cast_from(x[0]) * i32::cast_from(x[0]) +
      i32::cast_from(x[1]) * i32::cast_from(x[1]) +
      i32::cast_from(x[2]) * i32::cast_from(x[2]);
    sum += i32::cast_from(x[0]) + i32::cast_from(x[1]) + i32::cast_from(x[2]);
  }
  let (reta, retb) = sgrproj_sum_finish(ssq, sum, 9, 455, s, bit_depth);
  *a = reta;
  *b = retb;
}

fn sgrproj_box_sum_fastx_r2<T: Pixel>(a: &mut i32, b: &mut i32, row: isize,
                                      crop_h: usize, stripe_h: usize,
                                      stripe_x: isize, stripe_y: isize,
                                      s: i32, cdeffed: &Plane<T>, deblocked: &Plane<T>,
                                      bit_depth: usize) {  
  let mut ssq:i32 = 0;
  let mut sum:i32 = 0;
  for yi in row-2 as isize..=row+2 as isize {
    let p = if yi >= 0 && yi < stripe_h as isize {cdeffed} else {deblocked};
    let ly = clamp(clamp(yi+stripe_y, 0, crop_h as isize - 1),
                   stripe_y - 2, stripe_y + stripe_h as isize + 1);
    let x = p.slice(&PlaneOffset{x: stripe_x - 2, y: ly}).as_slice();
    ssq += i32::cast_from(x[0]) * i32::cast_from(x[0]) +
      i32::cast_from(x[1]) * i32::cast_from(x[1]) +
      i32::cast_from(x[2]) * i32::cast_from(x[2]) +
      i32::cast_from(x[3]) * i32::cast_from(x[3]) +
      i32::cast_from(x[4]) * i32::cast_from(x[4]);
    sum += i32::cast_from(x[0]) + i32::cast_from(x[1]) + i32::cast_from(x[2]) +
      i32::cast_from(x[3]) + i32::cast_from(x[4]);
  }
  let (reta, retb) = sgrproj_sum_finish(ssq, sum, 25, 164, s, bit_depth);
  *a = reta;
  *b = retb;
}

// r=1 case computes every row as every row is used (see r2 version below)
fn sgrproj_box_ab_r1<T: Pixel>(af: &mut[i32; 64+2],
                               bf: &mut[i32; 64+2],
                               s: i32,
                               crop_w: usize, crop_h: usize, stripe_h: usize,
                               stripe_x: isize, stripe_y: isize,
                               cdeffed: &Plane<T>, deblocked: &Plane<T>, bit_depth: usize) {
  let boundary0 = cmp::max(0, -stripe_y) as usize;
  let boundary3 = stripe_h + 2;
  if stripe_x > 1 && stripe_x < crop_w as isize - 2 {
    // Away from left and right edges; no X clipping to worry about,
    // but top/bottom few rows still need to worry about Y
    let boundary1 = cmp::max(3, 3-stripe_y) as usize;
    let boundary2 = cmp::min(crop_h as isize - stripe_y - 1, stripe_h as isize - 1) as usize;

    // top rows, away from left and right columns
    for i in boundary0..boundary1 {
      sgrproj_box_sum_fastx_r1(&mut af[i], &mut bf[i], i as isize - 1, crop_h, stripe_h,
                           stripe_x, stripe_y, s, cdeffed, deblocked, bit_depth);
    }
    // middle rows, away from left and right columns
    for i in boundary1..boundary2 {
      sgrproj_box_sum_fastxy_r1(&mut af[i], &mut bf[i],
                            stripe_x, stripe_y + i as isize - 1, s, cdeffed, bit_depth);
    }
    // bottom rows, away from left and right columns
    for i in boundary2..boundary3 {
      sgrproj_box_sum_fastx_r1(&mut af[i], &mut bf[i], i as isize - 1, crop_h, stripe_h,
                           stripe_x, stripe_y, s, cdeffed, deblocked, bit_depth);
    }
  } else {
    // top/bottom rows and left/right columns, where we need to worry about frame and stripe clipping
    for i in boundary0..boundary3 {
      sgrproj_box_sum_slow(&mut af[i], &mut bf[i], i as isize - 1, 1, crop_w, crop_h, stripe_h,
                           stripe_x, stripe_y, 9, 455, s, cdeffed, deblocked, bit_depth);
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
                               s: i32,
                               crop_w: usize, crop_h: usize, stripe_h: usize,
                               stripe_x: isize, stripe_y: isize,
                               cdeffed: &Plane<T>, deblocked: &Plane<T>,
                               bit_depth: usize) {
  let boundary0 = cmp::max(0, -stripe_y) as usize; // always even
  let boundary3 = stripe_h + 2; // don't care if odd
  if stripe_x > 1 && stripe_x < crop_w as isize - 2 {
    // make even, round up
    let boundary1 = cmp::max(3, 3-stripe_y) as usize + 1 >> 1 << 1;
    // must be even, rounding of +1 cancels fencepost of -1
    let boundary2 = cmp::min(crop_h as isize - stripe_y, stripe_h as isize) as usize >> 1 << 1;

    // top rows, away from left and right columns
    for i in (boundary0..boundary1).step_by(2) {
      sgrproj_box_sum_fastx_r2(&mut af[i], &mut bf[i], i as isize - 1, crop_h, stripe_h,
                           stripe_x, stripe_y, s, cdeffed, deblocked, bit_depth);
    }
    // middle rows, away from left and right columns
    for i in (boundary1..boundary2).step_by(2) {
      sgrproj_box_sum_fastxy_r2(&mut af[i], &mut bf[i],
                            stripe_x, stripe_y + i as isize - 1, s, cdeffed, bit_depth);
    }
    // bottom rows, away from left and right columns
    for i in (boundary2..boundary3).step_by(2) {
      sgrproj_box_sum_fastx_r2(&mut af[i], &mut bf[i], i as isize - 1, crop_h, stripe_h,
                           stripe_x, stripe_y, s, cdeffed, deblocked, bit_depth);
    }
  } else {
    // top/bottom rows and left/right columns, where we need to worry about frame and stripe clipping
    for i in (boundary0..boundary3).step_by(2) {
      sgrproj_box_sum_slow(&mut af[i], &mut bf[i], i as isize - 1, 2, crop_w, crop_h, stripe_h,
                           stripe_x, stripe_y, 25, 164, s, cdeffed, deblocked, bit_depth);
    }
  }
}

fn sgrproj_box_f_r0<T: Pixel>(f: &mut[i32; 64], x: usize, y: isize, h: usize, cdeffed: &Plane<T>) {
  for i in cmp::max(0, -y) as usize..h {
    f[i as usize] = (i32::cast_from(cdeffed.p(x, (y + i as isize) as usize))) << SGRPROJ_RST_BITS;
  }
}

fn sgrproj_box_f_r1<T: Pixel>(af: &[&[i32; 64+2]; 3], bf: &[&[i32; 64+2]; 3], f: &mut[i32; 64],
                              x: usize, y: isize, h: usize, cdeffed: &Plane<T>) {
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
                              x: usize, y: isize, h: usize, cdeffed: &Plane<T>) {
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

fn sgrproj_stripe_filter<T: Pixel>(set: u8, xqd: [i8; 2], fi: &FrameInvariants<T>,
                                   crop_w: usize, crop_h: usize,
                                   stripe_w: usize, stripe_h: usize,
                                   stripe_x: usize, stripe_y: isize,
                                   cdeffed: &Plane<T>, deblocked: &Plane<T>, out: &mut Plane<T>) {

  assert!(stripe_h <= 64);
  let mut a_r2: [[i32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut b_r2: [[i32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut f_r2: [i32; 64] = [0; 64];
  let mut a_r1: [[i32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut b_r1: [[i32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut f_r1: [i32; 64] = [0; 64];

  let s_r2: i32 = SGRPROJ_PARAMS_S[set as usize][0];
  let s_r1: i32 = SGRPROJ_PARAMS_S[set as usize][1];

  /* prime the intermediate arrays */
  if s_r2 > 0 {
    sgrproj_box_ab_r2(&mut a_r2[0], &mut b_r2[0], s_r2,
                   crop_w, crop_h, stripe_h,
                   stripe_x as isize - 1, stripe_y,
                   cdeffed, deblocked, fi.sequence.bit_depth);
    sgrproj_box_ab_r2(&mut a_r2[1], &mut b_r2[1], s_r2,
                   crop_w, crop_h, stripe_h,
                   stripe_x as isize, stripe_y,
                   cdeffed, deblocked, fi.sequence.bit_depth);
  }
  if s_r1 > 0 {
    sgrproj_box_ab_r1(&mut a_r1[0], &mut b_r1[0], s_r1,
                      crop_w, crop_h, stripe_h,
                      stripe_x as isize - 1, stripe_y,
                      cdeffed, deblocked, fi.sequence.bit_depth);
    sgrproj_box_ab_r1(&mut a_r1[1], &mut b_r1[1], s_r1,
                      crop_w, crop_h, stripe_h,
                      stripe_x as isize, stripe_y,
                      cdeffed, deblocked, fi.sequence.bit_depth);
  }

  /* iterate by column */
  let start = cmp::max(0, -stripe_y) as usize;
  let cdeffed_slice = cdeffed.slice(&PlaneOffset{x: stripe_x as isize, y: stripe_y});
  let outstride = out.cfg.stride;
  let mut out_slice = out.mut_slice(&PlaneOffset{x: stripe_x as isize, y: stripe_y});
  let out_data = out_slice.as_mut_slice();
  for xi in 0..stripe_w {
    /* build intermediate array columns */
    if s_r2 > 0 {
      sgrproj_box_ab_r2(&mut a_r2[(xi+2)%3], &mut b_r2[(xi+2)%3], s_r2,
                     crop_w, crop_h, stripe_h,
                     (stripe_x + xi + 1) as isize, stripe_y,
                     cdeffed, deblocked, fi.sequence.bit_depth);
      let ap0: [&[i32; 64+2]; 3] = [&a_r2[xi%3], &a_r2[(xi+1)%3], &a_r2[(xi+2)%3]];
      let bp0: [&[i32; 64+2]; 3] = [&b_r2[xi%3], &b_r2[(xi+1)%3], &b_r2[(xi+2)%3]];
      sgrproj_box_f_r2(&ap0, &bp0, &mut f_r2, stripe_x + xi, stripe_y, stripe_h as usize, cdeffed);
    } else {
      sgrproj_box_f_r0(&mut f_r2, stripe_x + xi, stripe_y, stripe_h as usize, cdeffed);
    }
    if s_r1 > 0 {
      sgrproj_box_ab_r1(&mut a_r1[(xi+2)%3], &mut b_r1[(xi+2)%3], s_r1,
                        crop_w, crop_h, stripe_h,
                        (stripe_x + xi + 1) as isize, stripe_y,
                        cdeffed, deblocked, fi.sequence.bit_depth);
      let ap1: [&[i32; 64+2]; 3] = [&a_r1[xi%3], &a_r1[(xi+1)%3], &a_r1[(xi+2)%3]];
      let bp1: [&[i32; 64+2]; 3] = [&b_r1[xi%3], &b_r1[(xi+1)%3], &b_r1[(xi+2)%3]];

      sgrproj_box_f_r1(&ap1, &bp1, &mut f_r1, stripe_x + xi, stripe_y, stripe_h as usize, cdeffed);
    } else {
      sgrproj_box_f_r0(&mut f_r1, stripe_x + xi, stripe_y, stripe_h as usize, cdeffed);
    }

    /* apply filter */
    let bit_depth = fi.sequence.bit_depth;
    let w0 = xqd[0] as i32;
    let w1 = xqd[1] as i32;
    let w2 = (1 << SGRPROJ_PRJ_BITS) - w0 - w1;
    for yi in start..stripe_h as usize {
      let u = i32::cast_from(cdeffed_slice.p(xi, yi)) << SGRPROJ_RST_BITS;
      let v = w0*f_r2[yi] + w1*u + w2*f_r1[yi];
      let s = v + (1 << SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS >> 1) >> SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS;
      out_data[xi + yi*outstride] = T::cast_from(clamp(s, 0, (1 << bit_depth) - 1));
    }
  }
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
          let ru_start_x = rux * rp.unit_size;
          let ru_size = if rux == rp.cols - 1 {
            crop_w - ru_start_x
          } else {
            rp.unit_size
          };
          let ru = rp.restoration_unit_by_stripe(si, rux);
          match ru.filter {
            RestorationFilter::Wiener{coeffs} => {          
              wiener_stripe_filter(coeffs, fi,
                                   crop_w, crop_h,
                                   ru_size, stripe_size,
                                   ru_start_x, stripe_start_y,
                                   &cdeffed.planes[pli], &pre_cdef.planes[pli],
                                   &mut out.planes[pli]);
            },
            RestorationFilter::Sgrproj{set, xqd} => {
              sgrproj_stripe_filter(set, xqd, fi,
                                    crop_w, crop_h,
                                    ru_size, stripe_size,
                                    ru_start_x, stripe_start_y,
                                    &cdeffed.planes[pli], &pre_cdef.planes[pli],
                                    &mut out.planes[pli]);
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
