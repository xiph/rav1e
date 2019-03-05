// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(safe_extern_statics)]

use crate::context::*;
use crate::Frame;
use crate::FrameInvariants;
use crate::plane::*;
use crate::util::{clamp, msb, Pixel, CastFromPrimitive};

use std::cmp;

pub struct CdefDirections {
  dir: [[u8; 8]; 8],
  var: [[i32; 8]; 8]
}

pub const CDEF_VERY_LARGE: u16 = 30000;
const CDEF_SEC_STRENGTHS: u8 = 4;

// Instead of dividing by n between 2 and 8, we multiply by 3*5*7*8/n.
// The output is then 840 times larger, but we don't care for finding
// the max. */
const CDEF_DIV_TABLE: [i32; 9] = [ 0, 840, 420, 280, 210, 168, 140, 120, 105 ];

#[inline]
/// Returns the position and value of the first instance of the max element in
/// a slice as a tuple.
///
/// # Arguments
///
/// * `elems` - A non-empty slice of integers
///
/// # Panics
///
/// Panics if `elems` is empty
fn first_max_element(elems: &[i32]) -> (usize, i32) {
  // In case of a tie, the first element must be selected.
  let (max_idx, max_value) = elems.iter().enumerate().max_by_key(|&(i, v)| (v, -(i as isize))).unwrap();
  (max_idx, *max_value)
}

// Detect direction. 0 means 45-degree up-right, 2 is horizontal, and so on.
// The search minimizes the weighted variance along all the lines in a
// particular direction, i.e. the squared error between the input and a
// "predicted" block where each pixel is replaced by the average along a line
// in a particular direction. Since each direction have the same sum(x^2) term,
// that term is never computed. See Section 2, step 2, of:
// http://jmvalin.ca/notes/intra_paint.pdf
fn cdef_find_dir<T: Pixel>(img: &PlaneSlice<'_, T>, var: &mut i32, coeff_shift: usize) -> i32 {
  let mut cost: [i32; 8] = [0; 8];
  let mut partial: [[i32; 15]; 8] = [[0; 15]; 8];
  for i in 0..8 {
    for j in 0..8 {
      let p: i32 = img[i][j].as_();
      // We subtract 128 here to reduce the maximum range of the squared
      // partial sums.
      debug_assert!(p >> coeff_shift <= 255);
      let x = (p >> coeff_shift) - 128;
      partial[0][i + j] += x;
      partial[1][i + j / 2] += x;
      partial[2][i] += x;
      partial[3][3 + i - j / 2] += x;
      partial[4][7 + i - j] += x;
      partial[5][3 - i / 2 + j] += x;
      partial[6][j] += x;
      partial[7][i / 2 + j] += x;
    }
  }
  for i in 0..8 {
    cost[2] += partial[2][i] * partial[2][i];
    cost[6] += partial[6][i] * partial[6][i];
  }
  cost[2] *= CDEF_DIV_TABLE[8];
  cost[6] *= CDEF_DIV_TABLE[8];
  for i in 0..7 {
    cost[0] += (partial[0][i]*partial[0][i] +
                partial[0][14-i]*partial[0][14-i]) * CDEF_DIV_TABLE[i + 1];
    cost[4] += (partial[4][i]*partial[4][i] +
                partial[4][14-i]*partial[4][14-i]) * CDEF_DIV_TABLE[i + 1];
  }
  cost[0] += partial[0][7] * partial[0][7] * CDEF_DIV_TABLE[8];
  cost[4] += partial[4][7] * partial[4][7] * CDEF_DIV_TABLE[8];
  for i in (1..8).step_by(2) {
    for j in 0..5 {
      cost[i] += partial[i][3 + j] * partial[i][3 + j];
    }
    cost[i] *= CDEF_DIV_TABLE[8];
    for j in 0..3 {
      cost[i] += (partial[i][j]*partial[i][j] +
                  partial[i][10-j]*partial[i][10-j]) * CDEF_DIV_TABLE[2 * j + 2];
    }
  }

  let (best_dir, best_cost) = first_max_element(&cost);
  // Difference between the optimal variance and the variance along the
  // orthogonal direction. Again, the sum(x^2) terms cancel out.
  // We'd normally divide by 840, but dividing by 1024 is close enough
  // for what we're going to do with this. */
  *var = (best_cost - cost[(best_dir + 4) & 7]) >> 10;

  best_dir as i32
}

fn constrain(diff: i32, threshold: i32, damping: i32) -> i32 {
  if threshold != 0 {
    let shift = cmp::max(0, damping - msb(threshold));
    let magnitude = cmp::min(diff.abs(), cmp::max(0, threshold - (diff.abs() >> shift)));
    if diff < 0 {
      -1 * magnitude
    } else {
      magnitude
    }
  } else {
    0
  }
}

// Unlike the AOM code, our block addressing points to the UL corner
// of the 2-pixel padding around the block, not the block itself.
// The destination is unpadded.
unsafe fn cdef_filter_block<T: Pixel>(
  dst: *mut T, dstride: isize, input: *const u16, istride: isize, pri_strength: i32,
  sec_strength: i32, dir: usize, damping: i32, xsize: isize, ysize: isize, coeff_shift: i32
) {
  let cdef_pri_taps = [[4, 2], [3, 3]];
  let cdef_sec_taps = [[2, 1], [2, 1]];
  let pri_taps = cdef_pri_taps[((pri_strength >> coeff_shift) & 1) as usize];
  let sec_taps = cdef_sec_taps[((pri_strength >> coeff_shift) & 1) as usize];
  let cdef_directions = [[-1 * istride + 1, -2 * istride + 2 ],
                         [ 0 * istride + 1, -1 * istride + 2 ],
                         [ 0 * istride + 1,  0 * istride + 2 ],
                         [ 0 * istride + 1,  1 * istride + 2 ],
                         [ 1 * istride + 1,  2 * istride + 2 ],
                         [ 1 * istride + 0,  2 * istride + 1 ],
                         [ 1 * istride + 0,  2 * istride + 0 ],
                         [ 1 * istride + 0,  2 * istride - 1 ]];
  for i in 0..ysize {
    for j in 0..xsize {
      let ptr_in = input.offset((i + 2) * istride + j + 2);
      let ptr_out = dst.offset(i * dstride + j);
      let x = *ptr_in;
      let mut sum = 0 as i32;
      let mut max = x;
      let mut min = x;
      for k in 0..2usize {
        let p0 = *ptr_in.offset(cdef_directions[dir][k]);
        let p1 = *ptr_in.offset(-cdef_directions[dir][k]);
        sum += pri_taps[k] * constrain(i32::cast_from(p0) - i32::cast_from(x), pri_strength, damping);
        sum += pri_taps[k] * constrain(i32::cast_from(p1) - i32::cast_from(x), pri_strength, damping);
        if p0 != CDEF_VERY_LARGE {
          max = cmp::max(p0, max);
        }
        if p1 != CDEF_VERY_LARGE {
          max = cmp::max(p1, max);
        }
        min = cmp::min(p0, min);
        min = cmp::min(p1, min);
        let s0 = *ptr_in.offset(cdef_directions[(dir + 2) & 7][k]);
        let s1 = *ptr_in.offset(-cdef_directions[(dir + 2) & 7][k]);
        let s2 = *ptr_in.offset(cdef_directions[(dir + 6) & 7][k]);
        let s3 = *ptr_in.offset(-cdef_directions[(dir + 6) & 7][k]);
        if s0 != CDEF_VERY_LARGE {
          max = cmp::max(s0, max);
        }
        if s1 != CDEF_VERY_LARGE {
          max = cmp::max(s1, max);
        }
        if s2 != CDEF_VERY_LARGE {
          max = cmp::max(s2, max);
        }
        if s3 != CDEF_VERY_LARGE {
          max = cmp::max(s3, max);
        }
        min = cmp::min(s0, min);
        min = cmp::min(s1, min);
        min = cmp::min(s2, min);
        min = cmp::min(s3, min);
        sum += sec_taps[k] * constrain(i32::cast_from(s0) - i32::cast_from(x), sec_strength, damping);
        sum += sec_taps[k] * constrain(i32::cast_from(s1) - i32::cast_from(x), sec_strength, damping);
        sum += sec_taps[k] * constrain(i32::cast_from(s2) - i32::cast_from(x), sec_strength, damping);
        sum += sec_taps[k] * constrain(i32::cast_from(s3) - i32::cast_from(x), sec_strength, damping);
      }
      let v = T::cast_from(i32::cast_from(x) + ((8 + sum - (sum < 0) as i32) >> 4));
      *ptr_out = clamp(v, T::cast_from(min), T::cast_from(max));
    }
  }
}

// We use the variance of an 8x8 block to adjust the effective filter strength.
fn adjust_strength(strength: i32, var: i32) -> i32 {
  let i = if (var >> 6) != 0 {cmp::min(msb(var >> 6), 12)} else {0};
  if var!=0 {strength * (4 + i) + 8 >> 4} else {0}
}

// For convenience of use alongside cdef_filter_superblock, we assume
// in_frame is padded.  Blocks are not scanned outside the block
// boundaries (padding is untouched here).

pub fn cdef_analyze_superblock<T: Pixel>(
  in_frame: &mut Frame<T>,
  bc_global: &mut BlockContext,
  sbo: &SuperBlockOffset,
  sbo_global: &SuperBlockOffset,
  bit_depth: usize,
) -> CdefDirections {
  let coeff_shift = bit_depth as usize - 8;
  let mut dir: CdefDirections = CdefDirections {dir: [[0; 8]; 8], var: [[0; 8]; 8]};
  // Each direction block is 8x8 in y, and direction computation only looks at y
  for by in 0..8 {
    for bx in 0..8 {
      // The bc and global SBO are only to determine frame
      // boundaries and skips in the event we're passing in a
      // single-SB copy 'frame' that represents some superblock
      // in the main frame.
      let global_block_offset = sbo_global.block_offset(bx<<1, by<<1);
      if global_block_offset.x < bc_global.cols && global_block_offset.y < bc_global.rows {
        let skip = bc_global.at(&global_block_offset).skip
          & bc_global.at(&sbo_global.block_offset(2*bx+1, 2*by)).skip
          & bc_global.at(&sbo_global.block_offset(2*bx, 2*by+1)).skip
          & bc_global.at(&sbo_global.block_offset(2*bx+1, 2*by+1)).skip;

        if !skip {
          let mut var: i32 = 0;
          let in_plane = &mut in_frame.planes[0];
          let in_po = sbo.plane_offset(&in_plane.cfg);
          let in_slice = in_plane.slice(&in_po);
          dir.dir[bx][by] = cdef_find_dir(&in_slice.reslice(8 * bx as isize + 2,
                                                            8 * by as isize + 2),
                                          &mut var, coeff_shift) as u8;
          dir.var[bx][by] = var;
        }
      }
    }
  }
  dir
}


pub fn cdef_sb_frame<T: Pixel>(fi: &FrameInvariants<T>, f: &Frame<T>) -> Frame<T> {
  let sb_size = if fi.sequence.use_128x128_superblock {128} else {64};

  Frame {
    planes: [
      Plane::new(sb_size >> f.planes[0].cfg.xdec, sb_size >> f.planes[0].cfg.ydec,
                 f.planes[0].cfg.xdec, f.planes[0].cfg.ydec, 3, 3),
      Plane::new(sb_size >> f.planes[1].cfg.xdec, sb_size >> f.planes[1].cfg.ydec,
                 f.planes[1].cfg.xdec, f.planes[1].cfg.ydec, 3, 3),
      Plane::new(sb_size >> f.planes[2].cfg.xdec, sb_size >> f.planes[2].cfg.ydec,
                 f.planes[2].cfg.xdec, f.planes[2].cfg.ydec, 3, 3),
    ]
  }
}

pub fn cdef_sb_padded_frame_copy<T: Pixel>(
  fi: &FrameInvariants<T>, sbo: &SuperBlockOffset,
  f: &Frame<T>, pad: usize
) -> Frame<u16> {
  let ipad = pad as isize;
  let sb_size = if fi.sequence.use_128x128_superblock {128} else {64};
  let mut out = Frame {
    planes: [
      Plane::new((sb_size >> f.planes[0].cfg.xdec) + pad*2, (sb_size >> f.planes[0].cfg.ydec) + pad*2,
                 f.planes[0].cfg.xdec, f.planes[0].cfg.ydec, 3, 3),
      Plane::new((sb_size >> f.planes[1].cfg.xdec) + pad*2, (sb_size >> f.planes[1].cfg.ydec) + pad*2,
                 f.planes[1].cfg.xdec, f.planes[1].cfg.ydec, 3, 3),
      Plane::new((sb_size >> f.planes[2].cfg.xdec) + pad*2, (sb_size >> f.planes[2].cfg.ydec) + pad*2,
                 f.planes[2].cfg.xdec, f.planes[2].cfg.ydec, 3, 3),
    ]
  };
  // Copy data into padded frame
  for p in 0..3 {
    let xdec = f.planes[p].cfg.xdec;
    let ydec = f.planes[p].cfg.ydec;
    let h = f.planes[p].cfg.height as isize;
    let w = f.planes[p].cfg.width as isize;
    let offset = sbo.plane_offset(&f.planes[p].cfg);
    for y in 0..((sb_size>>ydec) + pad*2) as isize {
      let mut out_slice = out.planes[p].as_mut_slice();
      let out_row = &mut out_slice[y as usize];
      if offset.y + y < ipad || offset.y+y >= h + ipad {
        // above or below the frame, fill with flag
        for x in 0..(sb_size>>xdec) + pad*2 {
          out_row[x] = CDEF_VERY_LARGE;
        }
      } else {
        let in_slice = f.planes[p].slice(&PlaneOffset {x:0, y:offset.y - ipad});
        let in_row = &in_slice[y as usize];
        // are we guaranteed to be all in frame this row?
        if offset.x < ipad || offset.x + (sb_size as isize >>xdec) + ipad >= w {
          // No; do it the hard way.  off left or right edge, fill with flag.
          for x in 0..(sb_size>>xdec) as isize + ipad*2 {
            if offset.x + x >= ipad && offset.x + x < w + ipad {
              out_row[x as usize] = u16::cast_from(in_row[(offset.x + x - ipad) as usize]);
            } else {
              out_row[x as usize] = CDEF_VERY_LARGE;
            }
          }
        } else {
          // Yes, do it the easy way: just copy
          for x in 0..(sb_size>>xdec) as isize + ipad*2 {
            out_row[x as usize] = u16::cast_from(in_row[(offset.x + x - ipad) as usize]);
          }
        }
      }
    }
  }
  out
}


pub fn cdef_empty_frame<T: Pixel, U: Pixel>(f: &Frame<T>) -> Frame<U> {
  Frame {
    planes: [
      Plane::new(0, 0, f.planes[0].cfg.xdec, f.planes[0].cfg.ydec, 0, 0),
      Plane::new(0, 0, f.planes[0].cfg.xdec, f.planes[0].cfg.ydec, 0, 0),
      Plane::new(0, 0, f.planes[0].cfg.xdec, f.planes[0].cfg.ydec, 0, 0),
    ]
  }
}

// We assume in is padded, and the area we'll write out is at least as
// large as the unpadded area of in
// cdef_index is taken from the block context
pub fn cdef_filter_superblock<T: Pixel>(
  fi: &FrameInvariants<T>,
  in_frame: &mut Frame<u16>,
  out_frame: &mut Frame<T>,
  bc_global: &mut BlockContext,
  sbo: &SuperBlockOffset,
  sbo_global: &SuperBlockOffset,
  cdef_index: u8,
  cdef_dirs: &CdefDirections,
) {
  let coeff_shift = fi.sequence.bit_depth as i32 - 8;
  let cdef_damping = fi.cdef_damping as i32;
  let cdef_y_strength = fi.cdef_y_strengths[cdef_index as usize];
  let cdef_uv_strength = fi.cdef_uv_strengths[cdef_index as usize];
  let cdef_pri_y_strength = (cdef_y_strength / CDEF_SEC_STRENGTHS) as i32;
  let mut cdef_sec_y_strength = (cdef_y_strength % CDEF_SEC_STRENGTHS) as i32;
  let cdef_pri_uv_strength = (cdef_uv_strength / CDEF_SEC_STRENGTHS) as i32;
  let mut cdef_sec_uv_strength = (cdef_uv_strength % CDEF_SEC_STRENGTHS) as i32;
  if cdef_sec_y_strength == 3 {
    cdef_sec_y_strength += 1;
  }
  if cdef_sec_uv_strength == 3 {
    cdef_sec_uv_strength += 1;
  }

  // Each direction block is 8x8 in y, potentially smaller if subsampled in chroma
  for by in 0..8 {
    for bx in 0..8 {
      let global_block_offset = sbo_global.block_offset(bx<<1, by<<1);
      if global_block_offset.x < bc_global.cols && global_block_offset.y < bc_global.rows {
        let skip = bc_global.at(&global_block_offset).skip
          & bc_global.at(&sbo_global.block_offset(2*bx+1, 2*by)).skip
          & bc_global.at(&sbo_global.block_offset(2*bx, 2*by+1)).skip
          & bc_global.at(&sbo_global.block_offset(2*bx+1, 2*by+1)).skip;
        if !skip {
          let dir = cdef_dirs.dir[bx][by];
          let var = cdef_dirs.var[bx][by];
          for p in 0..3 {
            let out_plane = &mut out_frame.planes[p];
            let out_po = sbo.plane_offset(&out_plane.cfg);
            let in_plane = &mut in_frame.planes[p];
            let in_po = sbo.plane_offset(&in_plane.cfg);
            let xdec = in_plane.cfg.xdec;
            let ydec = in_plane.cfg.ydec;

            let in_stride = in_plane.cfg.stride;
            let in_slice = &in_plane.mut_slice(&in_po);
            let out_stride = out_plane.cfg.stride;
            let out_slice = &mut out_plane.mut_slice(&out_po);

            let local_pri_strength;
            let local_sec_strength;
            let mut local_damping: i32 = cdef_damping + coeff_shift;
            let local_dir = if p == 0 {
              local_pri_strength = adjust_strength(cdef_pri_y_strength << coeff_shift, var);
              local_sec_strength = cdef_sec_y_strength << coeff_shift;
              if cdef_pri_y_strength != 0 { dir as usize } else { 0 }
            } else {
              local_pri_strength = cdef_pri_uv_strength << coeff_shift;
              local_sec_strength = cdef_sec_uv_strength << coeff_shift;
              local_damping -= 1;
              if cdef_pri_uv_strength != 0 { dir as usize } else { 0 }
            };

            unsafe {
              let xsize = 8 >> xdec;
              let ysize = 8 >> ydec;
              assert!(out_slice.rows_iter().len() >= (8 * by >> ydec) + ysize);
              assert!(in_slice.rows_iter().len() >= (8 * by >> ydec) + ysize + 4);
              let dst = out_slice[8 * by >> ydec][8 * bx >> xdec..].as_mut_ptr();
              let input = in_slice[8 * by >> ydec][8 * bx >> xdec..].as_ptr();
              cdef_filter_block(dst,
                                out_stride as isize,
                                input,
                                in_stride as isize,
                                local_pri_strength, local_sec_strength, local_dir,
                                local_damping, xsize as isize, ysize as isize,
                                coeff_shift as i32);
            }
          }
        }
      }
    }
  }
}

// Input to this process is the array CurrFrame of reconstructed samples.
// Output from this process is the array CdefFrame containing deringed samples.
// The purpose of CDEF is to perform deringing based on the detected direction of blocks.
// CDEF parameters are stored for each 64 by 64 block of pixels.
// The CDEF filter is applied on each 8 by 8 block of pixels.
// Reference: http://av1-spec.argondesign.com/av1-spec/av1-spec.html#cdef-process
pub fn cdef_filter_frame<T: Pixel>(fi: &FrameInvariants<T>, rec: &mut Frame<T>, bc: &mut BlockContext) {
  // Each filter block is 64x64, except right and/or bottom for non-multiple-of-64 sizes.
  // FIXME: 128x128 SB support will break this, we need FilterBlockOffset etc.
  let fb_width = (rec.planes[0].cfg.width + 63) / 64;
  let fb_height = (rec.planes[0].cfg.height + 63) / 64;

  // Construct a padded copy of the reconstructed frame.
  let mut padded_px: [[usize; 2]; 3] = [[0; 2]; 3];
  for p in 0..3 {
    padded_px[p][0] =  (fb_width*64 >> rec.planes[p].cfg.xdec) + 4;
    padded_px[p][1] =  (fb_height*64 >> rec.planes[p].cfg.ydec) + 4;
  }
  let mut cdef_frame: Frame<u16> = Frame {
    planes: [
      Plane::new(padded_px[0][0], padded_px[0][1], rec.planes[0].cfg.xdec, rec.planes[0].cfg.ydec, 0, 0),
      Plane::new(padded_px[1][0], padded_px[1][1], rec.planes[1].cfg.xdec, rec.planes[1].cfg.ydec, 0, 0),
      Plane::new(padded_px[2][0], padded_px[2][1], rec.planes[2].cfg.xdec, rec.planes[2].cfg.ydec, 0, 0)
    ]
  };
  for p in 0..3 {
    let rec_w = rec.planes[p].cfg.width;
    let rec_h = rec.planes[p].cfg.height;
    let mut cdef_slice = cdef_frame.planes[p].as_mut_slice();
    for row in 0..padded_px[p][1] {
      // pad first two elements of current row
      {
        let cdef_row = &mut cdef_slice[row][..2];
        cdef_row[0] = CDEF_VERY_LARGE;
        cdef_row[1] = CDEF_VERY_LARGE;
      }
      // pad out end of current row
      {
        let cdef_row = &mut cdef_slice[row][rec_w + 2..padded_px[p][0]];
        for x in cdef_row {
          *x = CDEF_VERY_LARGE;
        }
      }
      // copy current row from rec if we're in data, or pad if we're in first two rows/last N rows
      {
        let cdef_row = &mut cdef_slice[row][2..rec_w + 2];
        if row < 2 || row >= rec_h+2 {
          for x in cdef_row {
            *x = CDEF_VERY_LARGE;
          }
        } else {
          let rec_stride = rec.planes[p].cfg.stride;
          for (x, y) in cdef_row.iter_mut().zip(
            rec.planes[p].data_origin()[(row-2)*rec_stride..(row-1)*rec_stride].iter()
          ) {
            *x = u16::cast_from(*y);
          }
        }
      }
    }
  }

  // Perform actual CDEF, using the padded copy as source, and the input rec vector as destination.
  for fby in 0..fb_height {
    for fbx in 0..fb_width {
      let sbo = SuperBlockOffset { x: fbx, y: fby };
      let cdef_index = bc.at(&sbo.block_offset(0, 0)).cdef_index;
      let cdef_dirs = cdef_analyze_superblock(&mut cdef_frame, bc, &sbo, &sbo, fi.sequence.bit_depth);
      cdef_filter_superblock(fi, &mut cdef_frame, rec, bc, &sbo, &sbo, cdef_index, &cdef_dirs);
    }
  }
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn check_max_element() {
    assert_eq!(first_max_element(&[-1, -1, 1, 2, 3, 4, 6, 6]), (6, 6));
    assert_eq!(first_max_element(&[-1, -1, 1, 2, 3, 4, 7, 6]), (6, 7));
    assert_eq!(first_max_element(&[0, 0]), (0, 0));
  }
}
