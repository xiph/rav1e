// Copyright (c) 2017-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::context::*;
use crate::encoder::FrameInvariants;
use crate::frame::Frame;
use crate::frame::*;
use crate::hawktracer::*;
use crate::tiling::*;
use crate::util::{clamp, msb, CastFromPrimitive, Pixel};

use crate::cpu_features::CpuFeatureLevel;
use std::cmp;

cfg_if::cfg_if! {
  if #[cfg(nasm_x86_64)] {
    pub(crate) use crate::asm::x86::cdef::*;
  } else {
    pub(crate) use self::native::*;
  }
}

pub const CDEF_VERY_LARGE: u16 = 0x8000;
pub(crate) const CDEF_SEC_STRENGTHS: u8 = 4;

pub struct CdefDirections {
  dir: [[u8; 8]; 8],
  var: [[i32; 8]; 8],
}

pub(crate) mod native {
  use super::*;

  use simd_helpers::cold_for_target_arch;

  // Instead of dividing by n between 2 and 8, we multiply by 3*5*7*8/n.
  // The output is then 840 times larger, but we don't care for finding
  // the max.
  const CDEF_DIV_TABLE: [i32; 9] = [0, 840, 420, 280, 210, 168, 140, 120, 105];

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
  #[inline]
  fn first_max_element(elems: &[i32]) -> (usize, i32) {
    // In case of a tie, the first element must be selected.
    let (max_idx, max_value) = elems
      .iter()
      .enumerate()
      .max_by_key(|&(i, v)| (v, -(i as isize)))
      .unwrap();
    (max_idx, *max_value)
  }

  // Detect direction. 0 means 45-degree up-right, 2 is horizontal, and so on.
  // The search minimizes the weighted variance along all the lines in a
  // particular direction, i.e. the squared error between the input and a
  // "predicted" block where each pixel is replaced by the average along a line
  // in a particular direction. Since each direction have the same sum(x^2) term,
  // that term is never computed. See Section 2, step 2, of:
  // http://jmvalin.ca/notes/intra_paint.pdf
  pub fn cdef_find_dir<T: Pixel>(
    img: &PlaneSlice<'_, u16>, var: &mut u32, coeff_shift: usize,
    _cpu: CpuFeatureLevel,
  ) -> i32 {
    let mut cost: [i32; 8] = [0; 8];
    let mut partial: [[i32; 15]; 8] = [[0; 15]; 8];
    for i in 0..8 {
      for j in 0..8 {
        let p: i32 = img[i][j] as i32;
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
      cost[0] += (partial[0][i] * partial[0][i]
        + partial[0][14 - i] * partial[0][14 - i])
        * CDEF_DIV_TABLE[i + 1];
      cost[4] += (partial[4][i] * partial[4][i]
        + partial[4][14 - i] * partial[4][14 - i])
        * CDEF_DIV_TABLE[i + 1];
    }
    cost[0] += partial[0][7] * partial[0][7] * CDEF_DIV_TABLE[8];
    cost[4] += partial[4][7] * partial[4][7] * CDEF_DIV_TABLE[8];
    for i in (1..8).step_by(2) {
      for j in 0..5 {
        cost[i] += partial[i][3 + j] * partial[i][3 + j];
      }
      cost[i] *= CDEF_DIV_TABLE[8];
      for j in 0..3 {
        cost[i] += (partial[i][j] * partial[i][j]
          + partial[i][10 - j] * partial[i][10 - j])
          * CDEF_DIV_TABLE[2 * j + 2];
      }
    }

    let (best_dir, best_cost) = first_max_element(&cost);
    // Difference between the optimal variance and the variance along the
    // orthogonal direction. Again, the sum(x^2) terms cancel out.
    // We'd normally divide by 840, but dividing by 1024 is close enough
    // for what we're going to do with this. */
    *var = ((best_cost - cost[(best_dir + 4) & 7]) >> 10) as u32;

    best_dir as i32
  }

  #[inline(always)]
  fn constrain(diff: i32, threshold: i32, damping: i32) -> i32 {
    if threshold != 0 {
      let shift = cmp::max(0, damping - msb(threshold));
      let magnitude =
        cmp::min(diff.abs(), cmp::max(0, threshold - (diff.abs() >> shift)));

      if diff < 0 {
        -magnitude
      } else {
        magnitude
      }
    } else {
      0
    }
  }

  #[cold_for_target_arch("x86_64")]
  #[allow(clippy::erasing_op, clippy::identity_op, clippy::neg_multiply)]
  pub(crate) unsafe fn cdef_filter_block<T: Pixel>(
    dst: &mut PlaneRegionMut<'_, T>, input: *const u16, istride: isize,
    pri_strength: i32, sec_strength: i32, dir: usize, damping: i32,
    bit_depth: usize, xdec: usize, ydec: usize, _cpu: CpuFeatureLevel,
  ) {
    let xsize = (8 >> xdec) as isize;
    let ysize = (8 >> ydec) as isize;
    let coeff_shift = bit_depth as usize - 8;
    let cdef_pri_taps = [[4, 2], [3, 3]];
    let cdef_sec_taps = [[2, 1], [2, 1]];
    let pri_taps = cdef_pri_taps[((pri_strength >> coeff_shift) & 1) as usize];
    let sec_taps = cdef_sec_taps[((pri_strength >> coeff_shift) & 1) as usize];
    let cdef_directions = [
      [-1 * istride + 1, -2 * istride + 2],
      [0 * istride + 1, -1 * istride + 2],
      [0 * istride + 1, 0 * istride + 2],
      [0 * istride + 1, 1 * istride + 2],
      [1 * istride + 1, 2 * istride + 2],
      [1 * istride + 0, 2 * istride + 1],
      [1 * istride + 0, 2 * istride + 0],
      [1 * istride + 0, 2 * istride - 1],
    ];
    for i in 0..ysize {
      for j in 0..xsize {
        let ptr_in = input.offset(i * istride + j);
        let x = *ptr_in;
        let mut sum = 0 as i32;
        let mut max = x;
        let mut min = x;
        for k in 0..2usize {
          let cdef_dirs = [
            cdef_directions[dir][k],
            cdef_directions[(dir + 2) & 7][k],
            cdef_directions[(dir + 6) & 7][k],
          ];
          let pri_tap = pri_taps[k];
          let p =
            [*ptr_in.offset(cdef_dirs[0]), *ptr_in.offset(-cdef_dirs[0])];
          for p_elem in p.iter() {
            sum += pri_tap
              * constrain(
                i32::cast_from(*p_elem) - i32::cast_from(x),
                pri_strength,
                damping,
              );
            if *p_elem != CDEF_VERY_LARGE {
              max = cmp::max(*p_elem, max);
            }
            min = cmp::min(*p_elem, min);
          }

          let s = [
            *ptr_in.offset(cdef_dirs[1]),
            *ptr_in.offset(-cdef_dirs[1]),
            *ptr_in.offset(cdef_dirs[2]),
            *ptr_in.offset(-cdef_dirs[2]),
          ];
          let sec_tap = sec_taps[k];
          for s_elem in s.iter() {
            if *s_elem != CDEF_VERY_LARGE {
              max = cmp::max(*s_elem, max);
            }
            min = cmp::min(*s_elem, min);
            sum += sec_tap
              * constrain(
                i32::cast_from(*s_elem) - i32::cast_from(x),
                sec_strength,
                damping,
              );
          }
        }
        let v = i32::cast_from(x) + ((8 + sum - (sum < 0) as i32) >> 4);
        dst[i as usize][j as usize] =
          T::cast_from(clamp(v, min as i32, max as i32));
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
}

// We use the variance of an 8x8 block to adjust the effective filter strength.
fn adjust_strength(strength: i32, var: i32) -> i32 {
  let i = if (var >> 6) != 0 { cmp::min(msb(var >> 6), 12) } else { 0 };
  if var != 0 {
    (strength * (4 + i) + 8) >> 4
  } else {
    0
  }
}

// For convenience of use alongside cdef_filter_superblock, we assume
// in_frame is padded.  Blocks are not scanned outside the block
// boundaries (padding is untouched here).

pub fn cdef_analyze_superblock_range<T: Pixel>(
  fi: &FrameInvariants<T>, in_frame: &Frame<u16>, blocks: &TileBlocks<'_>,
  sbo: TileSuperBlockOffset, sbo_global: TileSuperBlockOffset, sb_w: usize,
  sb_h: usize
) -> Vec<CdefDirections> {
  let mut ret = Vec::<CdefDirections>::with_capacity(sb_h * sb_w);
  for sby in 0..sb_h {
    for sbx in 0..sb_w {
      let local_sbo = TileSuperBlockOffset(SuperBlockOffset {
        x: sbo.0.x + sbx,
        y: sbo.0.y + sby,
      });
      let toplevel_sbo = TileSuperBlockOffset(SuperBlockOffset {
        x: sbo_global.0.x + sbx,
        y: sbo_global.0.y + sby,
      });
      ret.push(cdef_analyze_superblock(
        fi,
        in_frame,
        blocks,
        local_sbo,
        toplevel_sbo,
      ));
    }
  }
  ret
}

// For convenience of use alongside cdef_filter_superblock, we assume
// in_frame is padded.  Blocks are not scanned outside the block
// boundaries (padding is untouched here).

pub fn cdef_analyze_superblock<T: Pixel>(
  fi: &FrameInvariants<T>, in_frame: &Frame<u16>, blocks: &TileBlocks<'_>,
  sbo: TileSuperBlockOffset, sbo_global: TileSuperBlockOffset
) -> CdefDirections {
  let coeff_shift = fi.sequence.bit_depth as usize - 8;
  let mut dir: CdefDirections =
    CdefDirections { dir: [[0; 8]; 8], var: [[0; 8]; 8] };
  // Each direction block is 8x8 in y, and direction computation only looks at y
  for by in 0..8 {
    for bx in 0..8 {
      // The blocks and global SBO are only to determine frame
      // boundaries and skips in the event we're passing in a
      // single-SB copy 'frame' that represents some superblock
      // in the main frame.
      let global_block_offset = sbo_global.block_offset(bx << 1, by << 1);
      if global_block_offset.0.x < blocks.cols()
        && global_block_offset.0.y < blocks.rows()
      {
        let skip = blocks[global_block_offset].skip
          & blocks[sbo_global.block_offset(2 * bx + 1, 2 * by)].skip
          & blocks[sbo_global.block_offset(2 * bx, 2 * by + 1)].skip
          & blocks[sbo_global.block_offset(2 * bx + 1, 2 * by + 1)].skip;

        if !skip {
          let mut var: u32 = 0;
          let in_plane = &in_frame.planes[0];
          let in_po = sbo.plane_offset(&in_plane.cfg);
          let in_slice = in_plane.slice(in_po);
          dir.dir[bx][by] = cdef_find_dir::<T>(
            &in_slice.reslice(8 * bx as isize, 8 * by as isize),
            &mut var,
            coeff_shift,
            fi.cpu_feature_level,
          ) as u8;
          dir.var[bx][by] = var as i32;
        }
      }
    }
  }
  dir
}

// Allocates and returns a new Frame with its own memory that is
// patterned on the decimation of the Frame backing the passed-in
// Tile.  The width and hight are in units of 8-pixel (undecimated)
// blocks, the minimum working unit of the CDEF filters.
pub fn cdef_block8_frame<T: Pixel>(
  w_8: usize, h_8: usize, pattern_tile: &Tile<'_, T>,
) -> Frame<u16> {
  Frame {
    planes: [
      {
        let &PlaneConfig { xdec, ydec, .. } = pattern_tile.planes[0].plane_cfg;
        Plane::new(w_8 << 3 >> xdec, h_8 << 3 >> ydec, xdec, ydec, 0, 0)
      },
      {
        let &PlaneConfig { xdec, ydec, .. } = pattern_tile.planes[1].plane_cfg;
        Plane::new(w_8 << 3 >> xdec, h_8 << 3 >> ydec, xdec, ydec, 0, 0)
      },
      {
        let &PlaneConfig { xdec, ydec, .. } = pattern_tile.planes[2].plane_cfg;
        Plane::new(w_8 << 3 >> xdec, h_8 << 3 >> ydec, xdec, ydec, 0, 0)
      },
    ],
  }
}

// Allocates and returns a new Frame with its own memory that is
// patterned on the decimation of the Frame backing the passed-in
// Tile.  The width and hight are in units of 8-pixel (undecimated)
// blocks, the minimum working unit of the CDEF filters, and the
// padding is in units of individual pixels.  The full padding is
// applied even to decimated planes.  The contents of the tile,
// beginning at the passed in superblock offset, are copied into the
// new Frame.  The padding is also filled from the passed in Tile,
// where pixels are available.  Those portions of the new Frame that
// do not overlap visible pixels int he passed in tile are filled with
// the CDEF_VERY_LARGE flag.
pub fn cdef_padded_tile_copy<T: Pixel>(
  tile: &Tile<'_, T>,
  sbo: TileSuperBlockOffset,
  w_8: usize,
  h_8: usize,
  pad: usize,
) -> Frame<u16> {
  let ipad = pad as isize;
  let mut out = {
    Frame {
      planes: {
        let new_plane = |pli: usize| {
          let &PlaneConfig { xdec, ydec, .. } = tile.planes[pli].plane_cfg;
          Plane::new(
            w_8 << 3 >> xdec,
            h_8 << 3 >> ydec,
            xdec,
            ydec,
            pad,
            pad,)
        };
        [new_plane(0), new_plane(1), new_plane(2)]
      },
    }
  };
  // Copy data into padded frame
  for pli in 0..3 {
    let PlaneOffset{x, y} = sbo.plane_offset(&tile.planes[pli].plane_cfg);
    let in_width = tile.planes[pli].rect().width as isize;
    let in_height = tile.planes[pli].rect().height as isize;
    let out_width = out.planes[pli].cfg.width;
    let out_height = out.planes[pli].cfg.height;
    // we copy pixels from the input tile for padding, but don't
    // exceed the bounds of the tile (do not contend with other
    // threads!)
    let mut out_region =
      out.planes[pli].region_mut(Area::StartingAt { x: -ipad, y: -ipad });
    for yi in 0..(out_height + pad * 2) as isize {
      let out_row = &mut out_region[yi as usize];
      if y + yi - ipad < 0 || y + yi - ipad >= in_height as isize {
        // above or below the visible frame, fill with flag.
        // This flag needs to go away (since it forces us to use a 16-bit range)
        // but that requires some deep changes to the filtering code
        // and buffer offsetting in loop filter RDO
        for xi in 0..out_width + pad * 2 {
          out_row[xi] = CDEF_VERY_LARGE;
        }
      } else {
        let in_row = &tile.planes[pli][(y + yi - ipad) as usize];
        for xi in 0..out_width as isize + ipad * 2 {
          if x + xi - ipad >= 0 && x + xi - ipad < in_width as isize {
            out_row[xi as usize] =
              u16::cast_from(in_row[(x + xi - ipad) as usize]);
          } else {
            out_row[xi as usize] = CDEF_VERY_LARGE;
          }
        }
      }
    }
  }
  out
}

// We assume in is padded, and the area we'll write out is at least as
// large as the unpadded area of in
// cdef_index is taken from the block context
pub fn cdef_filter_superblock<T: Pixel, U: Pixel>(
  fi: &FrameInvariants<T>, in_frame: &Frame<u16>, out_frame: &mut Frame<U>,
  blocks: &TileBlocks<'_>, sbo: TileSuperBlockOffset,
  sbo_global: TileSuperBlockOffset, cdef_index: u8,
  cdef_dirs: &CdefDirections,
) {
  let bit_depth = fi.sequence.bit_depth;
  let coeff_shift = fi.sequence.bit_depth as i32 - 8;
  let cdef_damping = fi.cdef_damping as i32;
  let cdef_y_strength = fi.cdef_y_strengths[cdef_index as usize];
  let cdef_uv_strength = fi.cdef_uv_strengths[cdef_index as usize];
  let cdef_pri_y_strength = (cdef_y_strength / CDEF_SEC_STRENGTHS) as i32;
  let mut cdef_sec_y_strength = (cdef_y_strength % CDEF_SEC_STRENGTHS) as i32;
  let cdef_pri_uv_strength = (cdef_uv_strength / CDEF_SEC_STRENGTHS) as i32;
  let mut cdef_sec_uv_strength =
    (cdef_uv_strength % CDEF_SEC_STRENGTHS) as i32;
  if cdef_sec_y_strength == 3 {
    cdef_sec_y_strength += 1;
  }
  if cdef_sec_uv_strength == 3 {
    cdef_sec_uv_strength += 1;
  }

  // Each direction block is 8x8 in y, potentially smaller if subsampled in chroma
  for by in 0..8 {
    for bx in 0..8 {
      let global_block_offset = sbo_global.block_offset(bx << 1, by << 1);
      if global_block_offset.0.x < blocks.cols()
        && global_block_offset.0.y < blocks.rows()
      {
        let skip = blocks[global_block_offset].skip
          & blocks[sbo_global.block_offset(2 * bx + 1, 2 * by)].skip
          & blocks[sbo_global.block_offset(2 * bx, 2 * by + 1)].skip
          & blocks[sbo_global.block_offset(2 * bx + 1, 2 * by + 1)].skip;
        let dir = cdef_dirs.dir[bx][by];
        let var = cdef_dirs.var[bx][by];
        for p in 0..3 {
          let out_plane = &mut out_frame.planes[p];
          let in_plane = &in_frame.planes[p];
          let in_po = sbo.plane_offset(&in_plane.cfg);
          let xdec = in_plane.cfg.xdec;
          let ydec = in_plane.cfg.ydec;
          let in_stride = in_plane.cfg.stride;
          let in_slice = &in_plane.slice(in_po);
          let out_region = &mut out_plane.region_mut(Area::BlockStartingAt {
            bo: sbo.block_offset(0, 0).0,
          });
          let xsize = 8 >> xdec;
          let ysize = 8 >> ydec;

          if !skip {
            let local_pri_strength;
            let local_sec_strength;
            let mut local_damping: i32 = cdef_damping + coeff_shift;
            let local_dir = if p == 0 {
              local_pri_strength =
                adjust_strength(cdef_pri_y_strength << coeff_shift, var);
              local_sec_strength = cdef_sec_y_strength << coeff_shift;
              if cdef_pri_y_strength != 0 {
                dir as usize
              } else {
                0
              }
            } else {
              local_pri_strength = cdef_pri_uv_strength << coeff_shift;
              local_sec_strength = cdef_sec_uv_strength << coeff_shift;
              local_damping -= 1;
              if cdef_pri_uv_strength != 0 {
                dir as usize
              } else {
                0
              }
            };

            unsafe {
              let PlaneConfig { ypad, xpad, .. } = in_slice.plane.cfg;
              assert!(
                in_slice.rows_iter().len() + ypad
                  >= ((8 * by) >> ydec) + ysize + 2
              );
              assert!(in_slice.x - 2 >= -(xpad as isize));
              assert!(in_slice.y - 2 >= -(ypad as isize));

              let mut dst = out_region.subregion_mut(Area::BlockRect {
                bo: BlockOffset { x: 2 * bx, y: 2 * by },
                width: xsize,
                height: ysize,
              });
              let input =
                in_slice[(8 * by) >> ydec][(8 * bx) >> xdec..].as_ptr();
              cdef_filter_block(
                &mut dst,
                input,
                in_stride as isize,
                local_pri_strength,
                local_sec_strength,
                local_dir,
                local_damping,
                bit_depth,
                xdec,
                ydec,
                fi.cpu_feature_level,
              );
            }
          } else {
            // we need to copy input to output
            let in_block =
              in_slice.subslice((8 * bx) >> xdec, (8 * by) >> ydec);
            let mut out_block = out_region.subregion_mut(Area::BlockRect {
              bo: BlockOffset { x: 2 * bx, y: 2 * by },
              width: xsize,
              height: ysize,
            });
            for i in 0..ysize {
              for j in 0..xsize {
                out_block[i][j] = U::cast_from(in_block[i][j]);
              }
            }
          }
        }
      }
    }
  }
}

// Input to this process is the Frame rec of reconstructed samples.
// Output from this process is the Frame rec containing deringed samples.
// The purpose of CDEF is to perform deringing based on the detected direction of blocks.
// CDEF parameters are stored for each 64 by 64 block of pixels.
// The CDEF filter is applied on each 8 by 8 block of pixels.
// Reference: http://av1-spec.argondesign.com/av1-spec/av1-spec.html#cdef-process
#[hawktracer(cdef_filter_frame)]
pub fn cdef_filter_frame<T: Pixel>(
  fi: &FrameInvariants<T>, rec: &mut Frame<T>, blocks: &FrameBlocks,
) {
  // Each filter block is 64x64, except right and/or bottom for non-multiple-of-64 sizes.
  // FIXME: 128x128 SB support will break this, we need FilterBlockOffset etc.
  let fb_width = (rec.planes[0].cfg.width + 63) / 64;
  let fb_height = (rec.planes[0].cfg.height + 63) / 64;

  // Construct a padded copy of the reconstructed frame.
  let mut cdef_frame: Frame<u16> = Frame {
    planes: {
      let new_plane = |pli: usize| {
        Plane::new(
          (fb_width * 64) >> rec.planes[pli].cfg.xdec,
          (fb_height * 64) >> rec.planes[pli].cfg.ydec,
          rec.planes[pli].cfg.xdec,
          rec.planes[pli].cfg.ydec,
          2,
          2,
        )
      };
      [new_plane(0), new_plane(1), new_plane(2)]
    },
  };

  for p in 0..3 {
    let rec_w = rec.planes[p].cfg.width;
    let rec_h = rec.planes[p].cfg.height;
    let mut cdef_region = cdef_frame.planes[p].region_mut(Area::Rect {
      x: -2,
      y: -2,
      width: rec_w + 4,
      height: rec_h + 4,
    });
    for row in 0..cdef_region.rect().height {
      // pad first two elements of current row
      {
        let cdef_row = &mut cdef_region[row][..2];
        cdef_row[0] = CDEF_VERY_LARGE;
        cdef_row[1] = CDEF_VERY_LARGE;
      }
      // pad out end of current row
      {
        let cdef_row = &mut cdef_region[row][rec_w + 2..];
        for x in cdef_row {
          *x = CDEF_VERY_LARGE;
        }
      }
      // copy current row from rec if we're in data, or pad if we're in first two rows/last N rows
      {
        let cdef_row = &mut cdef_region[row][2..rec_w + 2];
        if row < 2 || row >= rec_h + 2 {
          for x in cdef_row {
            *x = CDEF_VERY_LARGE;
          }
        } else {
          let rec_stride = rec.planes[p].cfg.stride;
          for (x, y) in cdef_row.iter_mut().zip(
            rec.planes[p].data_origin()
              [(row - 2) * rec_stride..(row - 1) * rec_stride]
              .iter(),
          ) {
            *x = u16::cast_from(*y);
          }
        }
      }
    }
  }

  let tb = blocks.as_tile_blocks();

  // Perform actual CDEF, using the padded copy as source, and the input rec vector as destination.
  for fby in 0..fb_height {
    for fbx in 0..fb_width {
      let sbo = PlaneSuperBlockOffset(SuperBlockOffset { x: fbx, y: fby });
      let cdef_index = blocks[sbo.block_offset(0, 0)].cdef_index;

      // In this particular instance CDEF application operates on the whole
      // frame as if it were one tile.
      let sbo = TileSuperBlockOffset(sbo.0);
      let cdef_dirs = cdef_analyze_superblock(
        fi,
        &cdef_frame,
        &tb,
        sbo,
        sbo,
      );
      
      cdef_filter_superblock(
        fi,
        &cdef_frame,
        rec,
        &tb,
        sbo,
        sbo,
        cdef_index,
        &cdef_dirs,
      );
    }
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::api::*;
  use crate::encoder::*;

  fn create_frame() -> (Frame<u16>, FrameInvariants<u16>) {
    let mut frame = Frame::<u16>::new(512, 512, ChromaSampling::Cs420);

    // in this test, each pixel contains the sum of its row and column indices:
    //
    //  0 1 2 3 4 . .
    //  1 2 3 4 5 . .
    //  2 3 4 5 6 . .
    //  3 4 5 6 7 . .
    //  4 5 6 7 8 . .
    //  . . . . . . .
    //  . . . . . . .
    for plane in &mut frame.planes {
      let PlaneConfig { width, height, .. } = plane.cfg;
      let mut slice = plane.mut_slice(PlaneOffset::default());
      for col in 0..width {
        for row in 0..height {
          slice[row][col] = (row + col) as u16;
        }
      }
    }

    let config = EncoderConfig {
      width: 512,
      height: 512,
      quantizer: 100,
      speed_settings: SpeedSettings::from_preset(10),
      ..Default::default()
    };
    let sequence = Sequence::new(&config);
    let fi = FrameInvariants::new(config, sequence);
    (frame, fi)
  }

  #[test]
  fn test_padded_frame_copy() {
    let (frame, _fi) = create_frame();
    let tile = frame.as_tile();
    // a super-block in the middle (not near frame borders)
    let pad = 2;
    let padded_frame = cdef_padded_tile_copy(
      &tile,
      TileSuperBlockOffset(SuperBlockOffset{x: 1, y: 2}),
      64 >> 3,
      64 >> 3,
      pad,
    );

    // index (0, 0) of padded_frame should match index (64, 128) of the source
    // frame, have 2 cols and rows padding from the source frame on all sides,
    // and have a size of (64, 64)
    assert_eq!(padded_frame.planes[0].cfg.width, 64);
    assert_eq!(padded_frame.planes[0].cfg.height, 64);

    let po = PlaneOffset { x: 62, y: 126 };
    let in_luma_slice = frame.planes[0].slice(po);
    let out_luma_region =
      padded_frame.planes[0].region(Area::StartingAt { x: -2, y: -2 });

    // this region does not overlap the frame padding, so it contains only
    // values from the input frame
    for row in 0..68 {
      for col in 0..68 {
        let in_pixel = in_luma_slice[row][col];
        let out_pixel = out_luma_region[row][col];
        assert_eq!(in_pixel, out_pixel);
      }
    }
  }

  #[test]
  fn test_padded_frame_copy_outside_input() {
    let (frame, _fi) = create_frame();
    let tile = frame.as_tile();
    // the top-right super-block (near top and right frame borders)
    let pad = 2;
    let padded_frame = cdef_padded_tile_copy(
      &tile,
      TileSuperBlockOffset(SuperBlockOffset{x: 7, y: 0}),
      64 >> 3,
      64 >> 3,
      pad,
    );

    // index (0, 0) of padded_frame should match index (448, 0) of the source
    // frame, have 2 cols/rows from the source frame left and below, 2
    // cols/rows of padding value right and above, and have a size of (64, 64)
    assert_eq!(padded_frame.planes[0].cfg.width, 64);
    assert_eq!(padded_frame.planes[0].cfg.height, 64);

    let po = PlaneOffset { x: 446, y: 0 };
    let in_luma_slice = frame.planes[0].slice(po);
    let out_luma_slice =
      padded_frame.planes[0].region(Area::StartingAt { x: -2, y: 0 });

    // this region does not overlap the frame padding, so it contains only
    // values from the input frame
    for row in 0..66 {
      for col in 0..66 {
        let in_pixel = in_luma_slice[row][col];
        let out_pixel = out_luma_slice[row][col];
        assert_eq!(out_pixel, in_pixel);
      }
      // right frame padding
      for col in 66..68 {
        let out_pixel = out_luma_slice[row][col];
        assert_eq!(out_pixel, CDEF_VERY_LARGE);
      }
    }

    // top frame padding
    let out_luma_slice =
      padded_frame.planes[0].region(Area::StartingAt { x: -2, y: -2 });
    for row in 0..2 {
      for col in 0..68 {
        let out_pixel = out_luma_slice[row][col];
        assert_eq!(out_pixel, CDEF_VERY_LARGE);
      }
    }
  }
}
