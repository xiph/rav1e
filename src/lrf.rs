// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(safe_extern_statics)]

use encoder::Frame;
use encoder::FrameInvariants;
use context::ContextWriter;
use context::SuperBlockOffset;
use context::PLANES;
use context::MAX_SB_SIZE;
use plane::Plane;
use plane::PlaneOffset;
use plane::PlaneConfig;
use std::cmp;
use util::clamp;

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
pub const SGRPROJ_PARAMS_RADIUS: [[u8; 2]; 1 << SGRPROJ_PARAMS_BITS] = [
  [2, 1], [2, 1], [2, 1], [2, 1],
  [2, 1], [2, 1], [2, 1], [2, 1],
  [2, 1], [2, 1], [0, 1], [0, 1],
  [0, 1], [0, 1], [2, 0], [2, 0],
];
pub const SGRPROJ_PARAMS_EPS: [[u8; 2]; 1 << SGRPROJ_PARAMS_BITS] = [
  [12,  4], [15,  6], [18,  8], [21,  9],
  [24, 10], [29, 11], [36, 12], [45, 13],
  [56, 14], [68, 15], [ 0,  5], [ 0,  8],
  [ 0, 11], [ 0, 14], [30,  0], [75,  0],
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
    RestorationFilter::None
  }
}

fn sgrproj_box_ab(af: &mut[i32; 64+2],
                  bf: &mut[i32; 64+2],
                  r: isize, eps: isize,
                  crop_w: usize, crop_h: usize, stripe_h: isize,
                  stripe_x: isize, stripe_y: isize,
                  cdeffed: &Plane, deblocked: &Plane, bit_depth: usize) {
  let n = ((2*r + 1) * (2*r + 1)) as i32;
  let one_over_n = ((1 << SGRPROJ_RECIP_BITS) + n/2 ) / n;
  let n2e = n*n*eps as i32;
  let s = ((1 << SGRPROJ_MTABLE_BITS) + n2e/2) / n2e;
  let xn = cmp::min(r+1, crop_w as isize - stripe_x);
  for row in -1..1+stripe_h {
    let mut a:i32 = 0;
    let mut b:i32 = 0;

    for yi in stripe_y+row-r..=stripe_y+row+r {
      let mut src_plane: &Plane;
      let ly;
      if yi < stripe_y {
        ly = cmp::max(cmp::max(yi, 0), stripe_y - 2) as usize;
        src_plane = deblocked;
      } else if yi < stripe_y + stripe_h {
        ly = clamp(yi, 0, crop_h as isize - 1) as usize;
        src_plane = cdeffed;
      } else {
        ly = cmp::min(clamp(yi, 0, crop_h as isize - 1), stripe_y+stripe_h+1) as usize;
        src_plane = deblocked;
      }

      for _xi in -r..-stripe_x {
        let c = src_plane.p(0, ly) as i32;
        a += c*c;
        b += c;
      }
      for xi in cmp::max(-r, -stripe_x)..xn {
        let c = src_plane.p((xi + stripe_x) as usize, ly) as i32;
        a += c*c;
        b += c;
      }
      for _xi in xn..r+1 {
        let c = src_plane.p(crop_w - 1, ly) as i32;
        a += c*c;
        b += c;
      }
    }
    a = a + (1 << 2*(bit_depth-8) >> 1) >> 2*(bit_depth-8);
    let d = b + (1 << bit_depth - 8 >> 1) >> bit_depth - 8;
    let p = cmp::max(0, a*(n as i32) - d*d);
    let z = p*s + (1 << SGRPROJ_MTABLE_BITS >> 1) >> SGRPROJ_MTABLE_BITS;
    let a2 = if z >= 255 {
      256
    } else if z == 0 {
      1
    } else {
      ((z << SGRPROJ_SGR_BITS) + z/2) / (z+1)
    };
    let b2 = ((1 << SGRPROJ_SGR_BITS) - a2 ) * b * one_over_n;
    af[(row+1) as usize] = a2;
    bf[(row+1) as usize] = b2 + (1 << SGRPROJ_RECIP_BITS >> 1) >> SGRPROJ_RECIP_BITS;
  }
}

fn sgrproj_box_f(af: &[&[i32; 64+2]; 3], bf: &[&[i32; 64+2]; 3], f: &mut[i32; 64],
                 x: usize, y: isize, h: isize, cdeffed: &Plane, pass: usize) {

  for i in 0..h as usize {
    let shift = if pass == 0 && (i&1) == 1 {4} else {5} + SGRPROJ_SGR_BITS - SGRPROJ_RST_BITS;
    let mut a = 0;
    let mut b = 0;
    for dy in 0..=2 {
      for dx in 0..=2 {
        let weight = if pass == 0 {
          if ((i+dy) & 1) == 0 {
            if dx == 1 {
              6
            } else {
              5
            }
          } else {
            0
          }
        } else {
          if dx == 1 || dy == 1 {
            4
          } else {
            3
          }
        };
        a += weight * af[dx][i+dy];
        b += weight * bf[dx][i+dy];
      }
    }
    let v = a * cdeffed.p(x, (y+i as isize) as usize) as i32 + b;
    f[i as usize] = v + (1 << shift >> 1) >> shift;
  }
}

fn sgrproj_stripe_rdu(set: u8, xqd: [i8; 2], fi: &FrameInvariants,
                      crop_w: usize, crop_h: usize,
                      stripe_w: usize, stripe_h: isize,
                      stripe_x: usize, stripe_y: isize,
                      cdeffed: &Plane, deblocked: &Plane, out: &mut Plane) {

  assert!(stripe_h <= 64);
  let mut a0: [[i32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut a1: [[i32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut b0: [[i32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut b1: [[i32; 64+2]; 3] = [[0; 64+2]; 3];
  let mut f0: [i32; 64] = [0; 64];
  let mut f1: [i32; 64] = [0; 64];

  let r0: u8 = SGRPROJ_PARAMS_RADIUS[set as usize][0];
  let r1: u8 = SGRPROJ_PARAMS_RADIUS[set as usize][1];
  let eps0: u8 = SGRPROJ_PARAMS_EPS[set as usize][0];
  let eps1: u8 = SGRPROJ_PARAMS_EPS[set as usize][1];

  /* prime the intermediate arrays */
  if r0 > 0 {
    sgrproj_box_ab(&mut a0[0], &mut b0[0], r0 as isize, eps0 as isize,
                   crop_w, crop_h, stripe_h,
                   stripe_x as isize - 1, stripe_y,
                   cdeffed, deblocked, fi.sequence.bit_depth);
    sgrproj_box_ab(&mut a0[1], &mut b0[1], r0 as isize, eps0 as isize,
                   crop_w, crop_h, stripe_h,
                   stripe_x as isize, stripe_y,
                   cdeffed, deblocked, fi.sequence.bit_depth);
  }
  if r1 > 0 {
    sgrproj_box_ab(&mut a1[0], &mut b1[0], r1 as isize, eps1 as isize,
                   crop_w, crop_h, stripe_h,
                   stripe_x as isize - 1, stripe_y,
                   cdeffed, deblocked, fi.sequence.bit_depth);
    sgrproj_box_ab(&mut a1[1], &mut b1[1], r1 as isize, eps1 as isize,
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
    if r0 > 0 {
      sgrproj_box_ab(&mut a0[(xi+2)%3], &mut b0[(xi+2)%3], r0 as isize, eps0 as isize,
                     crop_w, crop_h, stripe_h,
                     (stripe_x + xi + 1) as isize, stripe_y,
                     cdeffed, deblocked, fi.sequence.bit_depth);
      let ap0: [&[i32; 64+2]; 3] = [&a0[xi%3], &a0[(xi+1)%3], &a0[(xi+2)%3]];
      let bp0: [&[i32; 64+2]; 3] = [&b0[xi%3], &b0[(xi+1)%3], &b0[(xi+2)%3]];
      sgrproj_box_f(&ap0, &bp0, &mut f0, stripe_x + xi, stripe_y, stripe_h, cdeffed, 0);
    }
    if r1 > 0 {
      sgrproj_box_ab(&mut a1[(xi+2)%3], &mut b1[(xi+2)%3], r1 as isize, eps1 as isize,
                     crop_w, crop_h, stripe_h,
                     (stripe_x + xi + 1) as isize, stripe_y,
                     cdeffed, deblocked, fi.sequence.bit_depth);
      let ap1: [&[i32; 64+2]; 3] = [&a1[xi%3], &a1[(xi+1)%3], &a1[(xi+2)%3]];
      let bp1: [&[i32; 64+2]; 3] = [&b1[xi%3], &b1[(xi+1)%3], &b1[(xi+2)%3]];

      sgrproj_box_f(&ap1, &bp1, &mut f1, stripe_x + xi, stripe_y, stripe_h, cdeffed, 1);
    }
    let bit_depth = fi.sequence.bit_depth;
    if r0 > 0 {
      if r1 > 0 {
        let w0 = xqd[0] as i32;
        let w1 = xqd[1] as i32;
        let w2 = (1 << SGRPROJ_PRJ_BITS) - w0 - w1;
        for yi in start..stripe_h as usize {
          let u = (cdeffed_slice.p(xi, yi) as i32) << SGRPROJ_RST_BITS;
          let v = w0*f0[yi] + w1*u + w2*f1[yi];
          let s = v + (1 << SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS >> 1) >> SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS;
          out_data[xi + yi*outstride] = clamp(s as u16, 0, (1 << bit_depth) - 1);
        }
      } else {
        let w0 = xqd[0] as i32;
        let w = (1 << SGRPROJ_PRJ_BITS) - w0;
        for yi in start..stripe_h as usize {
          let u = (cdeffed_slice.p(xi, yi) as i32) << SGRPROJ_RST_BITS;
          let v = w0*f0[yi] + w*u;
          let s = v + (1 << SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS >> 1) >> SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS;
          out_data[xi + yi*outstride] = clamp(s as u16, 0, (1 << bit_depth) - 1);
        }
      }
    } else {
      /* if r0 is 0, the r1 must be nonzero */
      let w = xqd[0] as i32 + xqd[1] as i32;
      let w2 = (1 << SGRPROJ_PRJ_BITS) - w;
      for yi in start..stripe_h as usize {
        let u = (cdeffed_slice.p(xi, yi) as i32) << SGRPROJ_RST_BITS;
        let v = w*u + w2*f1[yi];
        let s = v + (1 << SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS >> 1) >> SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS;
        out_data[xi + yi*outstride] = clamp(s as u16, 0, (1 << bit_depth) - 1);
      }      
    }
  }
}

fn wiener_stripe_rdu(coeffs: [[i8; 3]; 2], fi: &FrameInvariants,
                      crop_w: usize, crop_h: usize,
                      stripe_w: usize, stripe_h: isize,
                      stripe_x: usize, stripe_y: isize,
                      cdeffed: &Plane, deblocked: &Plane, out: &mut Plane) {
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
  let end_i = cmp::max(0, if stripe_y + stripe_h > crop_h as isize {
    crop_h as isize - stripe_y - start_wi as isize
  } else {
    stripe_h - start_wi as isize
  }) as usize;
  
  let stride = out.cfg.stride;
  let mut out_slice = out.mut_slice(&PlaneOffset{x: 0, y: start_yi as isize});
  let out_data = out_slice.as_mut_slice();

  for xi in stripe_x..stripe_x+stripe_w {
    let n = cmp::min(7, crop_w as isize + 3 - xi as isize);
    for yi in stripe_y-3..stripe_y+stripe_h+4 {
      let mut src_plane: &Plane;
      let mut acc = 0;
      let ly;
      if yi < stripe_y {
        ly = cmp::max(clamp(yi, 0, crop_h as isize - 1), stripe_y - 2) as usize;
        src_plane = deblocked;
      } else if yi < stripe_y+stripe_h {
        ly = clamp(yi, 0, crop_h as isize - 1) as usize;
        src_plane = cdeffed;
      } else {
        ly = cmp::min(clamp(yi, 0, crop_h as isize - 1), stripe_y + stripe_h + 1) as usize;
        src_plane = deblocked;
      }
      
      for i in 0..3 - xi as isize {
        acc += hfilter[i as usize] * src_plane.p(0, ly) as i32;
      }
      for i in cmp::max(0,3 - (xi as isize))..n {
        acc += hfilter[i as usize] * src_plane.p((xi as isize + i - 3) as usize, ly) as i32;
      }
      for i in n..7 {
        acc += hfilter[i as usize] * src_plane.p(crop_w - 1, ly) as i32;
      }
        
      acc = acc + (1 << round_h >> 1) >> round_h;
      work[(yi-stripe_y+3) as usize] = clamp(acc, -offset, limit-offset);
    }

    for (wi, dst) in (start_wi..start_wi+end_i).zip(out_data[xi..].iter_mut().step_by(stride).take(end_i)) {
      let mut acc = 0;
      for (i,src) in (0..7).zip(work[wi..wi+7].iter_mut()) {
        acc += vfilter[i] * *src;
      }
      *dst = clamp(acc + (1 << round_v >> 1) >> round_v, 0, (1 << bit_depth) - 1) as u16;
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
  pub fn new(fi: &FrameInvariants, input: &Frame) -> Self {
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

  pub fn lrf_optimize_superblock(&mut self, _sbo: &SuperBlockOffset, _fi: &FrameInvariants,
                                 _cw: &mut ContextWriter) {
  }

  pub fn lrf_filter_frame(&mut self, out: &mut Frame, pre_cdef: &Frame,
                          fi: &FrameInvariants) {
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
              wiener_stripe_rdu(coeffs, fi,
                                crop_w, crop_h,
                                ru_size, stripe_size,
                                ru_start_x, stripe_start_y,
                                &cdeffed.planes[pli], &pre_cdef.planes[pli],
                                &mut out.planes[pli]);
            },
            RestorationFilter::Sgrproj{set, xqd} => {
              sgrproj_stripe_rdu(set, xqd, fi,
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
