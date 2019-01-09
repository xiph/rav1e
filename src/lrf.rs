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
use encoder::FrameState;
use encoder::FrameInvariants;
use context::ContextWriter;
use context::SuperBlockOffset;
use context::PLANES;
use context::MAX_SB_SIZE;
use plane::Plane;
use plane::PlaneConfig;
use plane::PlaneOffset;
use std::cmp;
use util::clamp;

pub const RESTORATION_TILESIZE_MAX: usize = 256;

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

#[derive(Copy, Clone)]
pub enum RestorationFilter {
  None,
  Wiener  { coeffs: [[i8; 3]; 2] },
  Sgrproj { set: u8,
            xqd: [i8; 2] },
}

impl RestorationFilter {
  pub fn default() -> RestorationFilter {
    RestorationFilter::Sgrproj{set: 5, xqd: SGRPROJ_XQD_MID}
  }
}

fn sgrproj_box_ab(af: &mut[i32; 64+2],
                  bf: &mut[i32; 64+2],
                  r: isize, eps: isize,
                  clipped_cfg: &PlaneConfig,
                  x: isize,
                  stripe_y: isize, stripe_h: isize, y: isize, h: isize,
                  cdeffed: &Plane, deblocked: &Plane,
                  bit_depth: usize) {


  let n = ((2*r + 1) * (2*r + 1)) as i32;
  let one_over_n = ((1 << SGRPROJ_RECIP_BITS) + n/2 ) / n;
  let n2e = n*n*eps as i32;
  let s = ((1 << SGRPROJ_MTABLE_BITS) + n2e/2) / n2e;
  for row in -1..1+h {
    let mut a:i32 = 0;
    let mut b:i32 = 0;
    let xn = cmp::min(r+1, clipped_cfg.width as isize - x);

    for yi in y+row-r..=y+row+r {
      let mut src_plane: &Plane;
      let ly;
      if yi < stripe_y {
        ly = cmp::max(clamp(yi, 0, clipped_cfg.height as isize - 1), y - 2) as usize;
        src_plane = deblocked;
      } else if yi < stripe_y + stripe_h {
        ly = clamp(yi, 0, clipped_cfg.height as isize - 1) as usize;
        src_plane = cdeffed;
      } else {
        ly = cmp::min(clamp(yi, 0, clipped_cfg.height as isize - 1), y+h+1) as usize;
        src_plane = deblocked;
      }

      for _xi in -r..-x {
        let c = src_plane.p(0, ly) as i32;
        a += c*c;
        b += c;
      }
      for xi in cmp::max(-r, -x)..xn {
        let c = src_plane.p((xi + x) as usize, ly) as i32;
        a += c*c;
        b += c;
      }
      for _xi in xn..r+1 {
        let c = src_plane.p(clipped_cfg.width - 1, ly) as i32;
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
                 x: usize, y: usize, h: usize, cdeffed: &Plane, pass: usize) {

  for i in 0..h {
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
    let v = a * cdeffed.p(x, y+i) as i32 + b;
    f[i as usize] = v + (1 << shift >> 1) >> shift;
  }
}

fn sgrproj_stripe_rdu(set: u8, xqd: [i8; 2], clipped_cfg: &PlaneConfig,
                      x: usize, w: usize, stripe_y: isize, stripe_h: isize,
                      cdeffed: &Plane, deblocked: &Plane, out: &mut Plane, bit_depth: usize){

  // unlike x, our y can be negative to start as the first stripe
  // starts off the top of the frame by 8 pixels, and can also run off the end of the frame
  let clipped_y = if stripe_y < 0 {0} else {stripe_y} as usize;
  let clipped_h = cmp::max(0, if stripe_y + stripe_h > clipped_cfg.height as isize {
    clipped_cfg.height - clipped_y
  } else {
    (stripe_y + stripe_h - clipped_y as isize) as usize
  });

  assert!(clipped_h <= 64);

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

  let w0 = xqd[0] as i32;
  let w1 = xqd[1] as i32;
  let w2 = (1 << SGRPROJ_PRJ_BITS) - w0 - w1;

  /* prime the intermediate arrays */
  sgrproj_box_ab(&mut a0[0], &mut b0[0], r0 as isize, eps0 as isize, clipped_cfg,
                 x as isize - 1, stripe_y, stripe_h, clipped_y as isize, clipped_h as isize,
                 cdeffed, deblocked, bit_depth);
  sgrproj_box_ab(&mut a0[1], &mut b0[1], r0 as isize, eps0 as isize, clipped_cfg,
                 x as isize, stripe_y, stripe_h, clipped_y as isize, clipped_h as isize,
                 cdeffed, deblocked, bit_depth);
  sgrproj_box_ab(&mut a1[0], &mut b1[0], r1 as isize, eps1 as isize, clipped_cfg,
                 x as isize - 1, stripe_y, stripe_h, clipped_y as isize, clipped_h as isize,
                 cdeffed, deblocked, bit_depth);
  sgrproj_box_ab(&mut a1[1], &mut b1[1], r1 as isize, eps1 as isize, clipped_cfg,
                 x as isize, stripe_y, stripe_h, clipped_y as isize, clipped_h as isize,
                 cdeffed, deblocked, bit_depth);

  /* iterate by column */
  let cdeffed_slice = cdeffed.slice(&PlaneOffset{x: x as isize, y: clipped_y as isize});
  let outstride = out.cfg.stride;
  let mut out_slice = out.mut_slice(&PlaneOffset{x: x as isize, y: clipped_y as isize});
  let out_data = out_slice.as_mut_slice();
  for xi in 0..w {
    sgrproj_box_ab(&mut a0[(xi+2)%3], &mut b0[(xi+2)%3], r0 as isize, eps0 as isize, clipped_cfg,
                   (x + xi + 1) as isize, stripe_y, stripe_h, clipped_y as isize, clipped_h as isize,
                   cdeffed, deblocked, bit_depth);

    sgrproj_box_ab(&mut a1[(xi+2)%3], &mut b1[(xi+2)%3], r1 as isize, eps1 as isize, clipped_cfg,
                   (x + xi + 1) as isize, stripe_y, stripe_h, clipped_y as isize, clipped_h as isize,
                   cdeffed, deblocked, bit_depth);
    {
      /* build intermediate array column F */
      let ap0: [&[i32; 64+2]; 3] = [&a0[xi%3], &a0[(xi+1)%3], &a0[(xi+2)%3]];
      let ap1: [&[i32; 64+2]; 3] = [&a1[xi%3], &a1[(xi+1)%3], &a1[(xi+2)%3]];
      let bp0: [&[i32; 64+2]; 3] = [&b0[xi%3], &b0[(xi+1)%3], &b0[(xi+2)%3]];
      let bp1: [&[i32; 64+2]; 3] = [&b1[xi%3], &b1[(xi+1)%3], &b1[(xi+2)%3]];

      sgrproj_box_f(&ap0, &bp0, &mut f0, x + xi, clipped_y, clipped_h, cdeffed, 0);
      sgrproj_box_f(&ap1, &bp1, &mut f1, x + xi, clipped_y, clipped_h, cdeffed, 1);
    }

    for yi in 0..clipped_h {
      let u = (cdeffed_slice.p(xi,yi) as i32) << SGRPROJ_RST_BITS;
      let mut v = w1 * u;
      if r0 != 0 {
        v += w0 * f0[yi];
      } else {
        v += w0 * u;
      }
      if r1 != 0 {
        v += w2 * f1[yi];
      } else {
        v += w2 * u;
      }
      let s = v + (1 << SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS >> 1) >> SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS;
      out_data[xi + yi*outstride] = clamp(s as u16, 0, (1 << bit_depth) - 1);
    }
  }
}

fn wiener_stripe_rdu(coeffs: [[i8; 3]; 2], clipped_cfg: &PlaneConfig,
                     x: usize, w: usize, y: isize, h: isize,
                     cdeffed: &Plane, deblocked: &Plane, out: &mut Plane, bit_depth: usize){
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
  let start_wi = if y < 0 {-y} else {0} as usize; 
  let start_yi = if y < 0 {0} else {y} as usize; 
  let end_i = cmp::max(0, if y+h > clipped_cfg.height as isize {
    clipped_cfg.height as isize - y - start_wi as isize
  } else {
    h - start_wi as isize
  }) as usize;
  
  let stride = out.cfg.stride;
  let mut out_slice = out.mut_slice(&PlaneOffset{x: 0, y: start_yi as isize});
  let out_data = out_slice.as_mut_slice();

  for xi in x..x+w {
    let n = cmp::min(7, clipped_cfg.width as isize + 3 - xi as isize);
    for yi in y-3..y+h+4 {
      let mut src_plane: &Plane;
      let mut acc = 0;
      let ly;
      if yi < y {
        ly = cmp::max(clamp(yi, 0, clipped_cfg.height as isize - 1), y - 2) as usize;
        src_plane = deblocked;
      } else if yi < y+h {
        ly = clamp(yi, 0, clipped_cfg.height as isize - 1) as usize;
        src_plane = cdeffed;
      } else {
        ly = cmp::min(clamp(yi, 0, clipped_cfg.height as isize - 1), y+h+1) as usize;
        src_plane = deblocked;
      }
      
      for i in 0..3 - xi as isize {
        acc += hfilter[i as usize] * src_plane.p(0, ly) as i32;
      }
      for i in cmp::max(0,3 - (xi as isize))..n {
        acc += hfilter[i as usize] * src_plane.p((xi as isize + i - 3) as usize, ly) as i32;
      }
      for i in n..7 {
        acc += hfilter[i as usize] * src_plane.p(clipped_cfg.width - 1, ly) as i32;
      }
        
      acc = acc + (1 << round_h >> 1) >> round_h;
      work[(yi-y+3) as usize] = clamp(acc, -offset, limit-offset);
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

#[derive(Copy, Clone)]
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

#[derive(Clone)]
pub struct RestorationPlane {
  // all size units are subsampled if the plane is subsampled
  pub clipped_cfg: PlaneConfig,
  pub lrf_type: u8,
  pub unit_size: usize,
  pub cols: usize,
  pub rows: usize,
  pub wiener_ref: [[i8; 3]; 2],
  pub sgrproj_ref: [i8; 2],
  pub units: Vec<Vec<RestorationUnit>>
}

#[derive(Clone, Default)]
pub struct RestorationPlaneOffset {
  pub row: usize,
  pub col: usize
}

impl RestorationPlane {
  pub fn new(clipped_cfg: &PlaneConfig, lrf_type: u8, unit_size: usize) -> RestorationPlane {
    let PlaneConfig { width, height, .. } = clipped_cfg;
    // bolted to superblock size for now
    let cols = cmp::max((width + (unit_size>>1)) / unit_size, 1);
    let rows = cmp::max((height + (unit_size>>1)) / unit_size, 1);
    RestorationPlane {
      clipped_cfg: clipped_cfg.clone(),
      lrf_type,
      unit_size,
      cols,
      rows,
      wiener_ref: [WIENER_TAPS_MID; 2],
      sgrproj_ref: SGRPROJ_XQD_MID,
      units: vec![vec![RestorationUnit::default(); cols]; rows]
    }
  }

  /// find the restoration unit offset corresponding to the this superblock offset
  /// This encapsulates some minor weirdness due to RU stretching at the frame boundary.
  pub fn restoration_plane_offset(&self, sbo: &SuperBlockOffset) -> RestorationPlaneOffset {
    let po = sbo.plane_offset(&self.clipped_cfg);
    RestorationPlaneOffset {
      row: cmp::min(po.y / self.unit_size as isize, self.rows as isize - 1) as usize,
      col: cmp::min(po.x / self.unit_size as isize, self.cols as isize - 1) as usize
    }
  }

  pub fn restoration_unit(&self, sbo: &SuperBlockOffset) -> &RestorationUnit {
    let rpo = self.restoration_plane_offset(sbo);
    &self.units[rpo.row][rpo.col]
  }

  pub fn restoration_unit_as_mut(&mut self, sbo: &SuperBlockOffset) -> &mut RestorationUnit {
    let rpo = self.restoration_plane_offset(sbo);
    &mut self.units[rpo.row][rpo.col]
  }

  pub fn restoration_unit_by_stripe(&self, stripenum: usize, rux: usize) -> &RestorationUnit {
    &self.units[cmp::min((stripenum * 64 >> self.clipped_cfg.ydec) / self.unit_size, self.rows - 1)]
      [cmp::min(rux, self.cols - 1)]
  }
}

#[derive(Clone)]
pub struct RestorationState {
  pub lrf_type: [u8; PLANES],
  pub unit_size: [usize; PLANES],
  pub plane: [RestorationPlane; PLANES]
}

impl RestorationState {
  pub fn new(fi: &FrameInvariants, input: &Frame) -> Self {
    // unlike the other loop filters that operate over the padded
    // frame dimensions, restoration filtering and source pixel
    // accesses are clipped to the original frame dimensions
    let mut clipped_cfg:[PlaneConfig; 3] = [input.planes[0].cfg.clone(),
                                    input.planes[1].cfg.clone(),
                                    input.planes[2].cfg.clone()];
    clipped_cfg[0].width = fi.width;
    clipped_cfg[0].height = fi.height;
    
    let PlaneConfig { xdec, ydec, .. } = clipped_cfg[1];

    clipped_cfg[1].width = fi.width + (1 << xdec >> 1) >> xdec;
    clipped_cfg[1].height = fi.height + (1 << ydec >> 1) >> ydec;
    clipped_cfg[2].width = fi.width + (1 << xdec >> 1) >> xdec;
    clipped_cfg[2].height = fi.height + (1 << ydec >> 1) >> ydec;

    // Currrently opt for smallest possible restoration unit size
    let lrf_y_shift = if fi.sequence.use_128x128_superblock {1} else {2};
    let lrf_uv_shift = lrf_y_shift + if xdec>0 && ydec>0 {1} else {0};
    let lrf_type: [u8; PLANES] = [RESTORE_SWITCHABLE, RESTORE_SWITCHABLE, RESTORE_SWITCHABLE];
    let unit_size: [usize; PLANES] = [RESTORATION_TILESIZE_MAX >> lrf_y_shift,
                                      RESTORATION_TILESIZE_MAX >> lrf_uv_shift,
                                      RESTORATION_TILESIZE_MAX >> lrf_uv_shift];

    RestorationState {
      lrf_type,
      unit_size,
      plane: [RestorationPlane::new(&clipped_cfg[0], lrf_type[0], unit_size[0]),
              RestorationPlane::new(&clipped_cfg[1], lrf_type[1], unit_size[1]),
              RestorationPlane::new(&clipped_cfg[2], lrf_type[2], unit_size[2])]
    }
  }
  
  pub fn restoration_unit(&self, sbo: &SuperBlockOffset, pli: usize) -> &RestorationUnit {
    let rpo = self.plane[pli].restoration_plane_offset(sbo);
    &self.plane[pli].units[rpo.row][rpo.col]
  }

  pub fn restoration_unit_as_mut(&mut self, sbo: &SuperBlockOffset, pli: usize) -> &mut RestorationUnit {
    let rpo = self.plane[pli].restoration_plane_offset(sbo);
    &mut self.plane[pli].units[rpo.row][rpo.col]
  }  

  pub fn lrf_optimize_superblock(&mut self, _sbo: &SuperBlockOffset, _fi: &FrameInvariants,
                                 _fs: &FrameState, _cw: &mut ContextWriter) {
  }

  pub fn lrf_filter_frame(&mut self, fs: &mut FrameState, pre_cdef: &Frame,
                          bit_depth: usize) {
    let cdeffed = &fs.rec.clone();
    let out = &mut fs.rec;
    
    // number of stripes (counted according to colocated Y luma position)
    let stripe_n = (self.plane[0].clipped_cfg.height + 7) / 64 + 1;
    
    for pli in 0..PLANES {
      let rp = &self.plane[pli];
      let ydec = self.plane[pli].clipped_cfg.ydec;

      for si in 0..stripe_n {
        // stripe y pixel locations must be able to overspan the frame
        let stripe_start_y = si as isize * 64 - 8 >> ydec;
        let stripe_size = 64 >> ydec; // one past, unlike spec
      
        // horizontally, go rdu-by-rdu
        for rux in 0..rp.cols {
          // stripe x pixel locations must be clipped to frame, last may need to stretch
          let ru_start_x = rux * rp.unit_size;
          let ru_size = if rux == rp.cols - 1 {
            self.plane[pli].clipped_cfg.width - ru_start_x
          } else {
            rp.unit_size
          };
          let ru = rp.restoration_unit_by_stripe(si, rux);
          match ru.filter {
            RestorationFilter::Wiener{coeffs} => {          
              wiener_stripe_rdu(coeffs, &self.plane[pli].clipped_cfg,
                                ru_start_x, ru_size, stripe_start_y, stripe_size,
                                &cdeffed.planes[pli], &pre_cdef.planes[pli],
                                &mut out.planes[pli], bit_depth);
            },
            RestorationFilter::Sgrproj{set, xqd} => {
              sgrproj_stripe_rdu(set, xqd, &self.plane[pli].clipped_cfg,
                                 ru_start_x, ru_size, stripe_start_y, stripe_size,
                                 &cdeffed.planes[pli], &pre_cdef.planes[pli],
                                 &mut out.planes[pli], bit_depth);
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
