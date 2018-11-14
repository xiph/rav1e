// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use context::BlockOffset;
use context::BLOCK_TO_PLANE_SHIFT;
use context::MI_SIZE;
use partition::*;
use plane::*;
use util::*;
use FrameInvariants;
use FrameState;

use libc;

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(windows), feature = "nasm"))]
extern {
  fn rav1e_sad_4x4_hbd_ssse3(
    src: *const u16, src_stride: libc::ptrdiff_t, dst: *const u16,
    dst_stride: libc::ptrdiff_t
  ) -> u32;
}

#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
extern {
  fn rav1e_sad_8x8_hbd10_ssse3(
    src: *const u16, src_stride: libc::ptrdiff_t, dst: *const u16,
    dst_stride: libc::ptrdiff_t
  ) -> u32;

  fn rav1e_sad_16x16_hbd_ssse3(
    src: *const u16, src_stride: libc::ptrdiff_t, dst: *const u16,
    dst_stride: libc::ptrdiff_t
  ) -> u32;

  fn rav1e_sad_32x32_hbd10_ssse3(
    src: *const u16, src_stride: libc::ptrdiff_t, dst: *const u16,
    dst_stride: libc::ptrdiff_t
  ) -> u32;

  fn rav1e_sad_64x64_hbd10_ssse3(
    src: *const u16, src_stride: libc::ptrdiff_t, dst: *const u16,
    dst_stride: libc::ptrdiff_t
  ) -> u32;

  fn rav1e_sad_128x128_hbd10_ssse3(
    src: *const u16, src_stride: libc::ptrdiff_t, dst: *const u16,
    dst_stride: libc::ptrdiff_t
  ) -> u32;
}

#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
#[target_feature(enable = "ssse3")]
unsafe fn sad_ssse3(
  plane_org: &PlaneSlice, plane_ref: &PlaneSlice, blk_h: usize, blk_w: usize,
  bit_depth: usize
) -> u32 {
  let mut sum = 0 as u32;
  // TODO: stride *2??? What is the correct way to do this?
  let org_stride = plane_org.plane.cfg.stride as libc::ptrdiff_t * 2;
  let ref_stride = plane_ref.plane.cfg.stride as libc::ptrdiff_t * 2;
  assert!(blk_h >= 4 && blk_w >= 4);
  let step_size =
    blk_h.min(blk_w).min(if bit_depth <= 10 { 128 } else { 4 });
  let func = match step_size.ilog() {
    3 => rav1e_sad_4x4_hbd_ssse3,
    4 => rav1e_sad_8x8_hbd10_ssse3,
    5 => rav1e_sad_16x16_hbd_ssse3,
    6 => rav1e_sad_32x32_hbd10_ssse3,
    7 => rav1e_sad_64x64_hbd10_ssse3,
    8 => rav1e_sad_128x128_hbd10_ssse3,
    _ => rav1e_sad_128x128_hbd10_ssse3
  };
  for r in (0..blk_h).step_by(step_size) {
    for c in (0..blk_w).step_by(step_size) {
      let org_slice = plane_org.subslice(c, r);
      let ref_slice = plane_ref.subslice(c, r);
      let org_ptr = org_slice.as_slice().as_ptr();
      let ref_ptr = ref_slice.as_slice().as_ptr();
      sum += func(org_ptr, org_stride, ref_ptr, ref_stride);
    }
  }
  return sum;
}

#[inline(always)]
pub fn get_sad(
  plane_org: &PlaneSlice, plane_ref: &PlaneSlice, blk_h: usize, blk_w: usize,
  bit_depth: usize
) -> u32 {
  #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
  {
    if is_x86_feature_detected!("ssse3") && blk_h >= 4 && blk_w >= 4 {
      return unsafe {
        sad_ssse3(plane_org, plane_ref, blk_h, blk_w, bit_depth)
      };
    }
  }
  let mut sum = 0 as u32;

  let org_iter = plane_org.iter_width(blk_w);
  let ref_iter = plane_ref.iter_width(blk_w);

  for (slice_org, slice_ref) in org_iter.take(blk_h).zip(ref_iter) {
      sum += slice_org
        .iter()
        .zip(slice_ref)
        .map(|(&a, &b)| (a as i32 - b as i32).abs() as u32)
        .sum::<u32>();
  }

  sum
}

fn get_mv_range(fi: &FrameInvariants, bo: &BlockOffset, blk_w: usize, blk_h: usize) -> (isize, isize, isize, isize) {
  let border_w = 128 + blk_w as isize * 8;
  let border_h = 128 + blk_h as isize * 8;
  let mvx_min = -(bo.x as isize) * (8 * MI_SIZE) as isize - border_w;
  let mvx_max = (fi.w_in_b - bo.x - blk_w / MI_SIZE) as isize * (8 * MI_SIZE) as isize + border_w;
  let mvy_min = -(bo.y as isize) * (8 * MI_SIZE) as isize - border_h;
  let mvy_max = (fi.h_in_b - bo.y - blk_h / MI_SIZE) as isize * (8 * MI_SIZE) as isize + border_h;

  (mvx_min, mvx_max, mvy_min, mvy_max)
}

pub fn motion_estimation(
  fi: &FrameInvariants, fs: &FrameState, bsize: BlockSize, bo: &BlockOffset,
  ref_frame: usize, pmv: MotionVector, bit_depth: usize
) -> MotionVector {
  match fi.rec_buffer.frames[fi.ref_frames[ref_frame - LAST_FRAME] as usize] {
    Some(ref rec) => {
      let po = PlaneOffset {
        x: (bo.x as isize) << BLOCK_TO_PLANE_SHIFT,
        y: (bo.y as isize) << BLOCK_TO_PLANE_SHIFT
      };
      let range = 16;
      let blk_w = bsize.width();
      let blk_h = bsize.height();
      let (mvx_min, mvx_max, mvy_min, mvy_max) = get_mv_range(fi, bo, blk_w, blk_h);
      let x_lo = po.x + ((-range + (pmv.col / 8) as isize).max(mvx_min / 8));
      let x_hi = po.x + ((range + (pmv.col / 8) as isize).min(mvx_max / 8));
      let y_lo = po.y + ((-range + (pmv.row / 8) as isize).max(mvy_min / 8));
      let y_hi = po.y + ((range + (pmv.row / 8) as isize).min(mvy_max / 8));

      let mut lowest_sad = 128 * 128 * 4096 as u32;
      let mut best_mv = MotionVector { row: 0, col: 0 };

      full_search(
        x_lo,
        x_hi,
        y_lo,
        y_hi,
        blk_h,
        blk_w,
        &fs.input.planes[0],
        &rec.frame.planes[0],
        &mut best_mv,
        &mut lowest_sad,
        &po,
        2,
        bit_depth
      );

      let mode = PredictionMode::NEWMV;
      let mut tmp_plane = Plane::new(blk_w, blk_h, 0, 0, 0, 0);

      let mut steps = vec![8, 4, 2];
      if fi.allow_high_precision_mv {
        steps.push(1);
      }

      for step in steps {
        let center_mv_h = best_mv;
        for i in 0..3 {
          for j in 0..3 {
            // Skip the center point that was already tested
            if i == 1 && j == 1 {
              continue;
            }

            let cand_mv = MotionVector {
              row: center_mv_h.row + step * (i as i16 - 1),
              col: center_mv_h.col + step * (j as i16 - 1)
            };

            if (cand_mv.col as isize) < mvx_min || (cand_mv.col as isize) > mvx_max {
              continue;
            }
            if (cand_mv.row as isize) < mvy_min || (cand_mv.row as isize) > mvy_max {
              continue;
            }

            {
              let tmp_slice =
                &mut tmp_plane.mut_slice(&PlaneOffset { x: 0, y: 0 });

              mode.predict_inter(
                fi, 0, &po, tmp_slice, blk_w, blk_h, [ref_frame, NONE_FRAME],
                [cand_mv, MotionVector{ row: 0, col: 0 }], 8,
              );
            }

            let plane_org = fs.input.planes[0].slice(&po);
            let plane_ref = tmp_plane.slice(&PlaneOffset { x: 0, y: 0 });

            let sad = get_sad(&plane_org, &plane_ref, blk_h, blk_w, bit_depth);

            if sad < lowest_sad {
              lowest_sad = sad;
              best_mv = cand_mv;
            }
          }
        }
      }

      best_mv
    }

    None => MotionVector { row: 0, col: 0 }
  }
}

fn full_search(
  x_lo: isize, x_hi: isize, y_lo: isize, y_hi: isize, blk_h: usize,
  blk_w: usize, p_org: &Plane, p_ref: &Plane, best_mv: &mut MotionVector,
  lowest_sad: &mut u32, po: &PlaneOffset, step: usize, bit_depth: usize
) {
  for y in (y_lo..y_hi).step_by(step) {
    for x in (x_lo..x_hi).step_by(step) {
      let plane_org = p_org.slice(po);
      let plane_ref = p_ref.slice(&PlaneOffset { x, y });

      let sad = get_sad(&plane_org, &plane_ref, blk_h, blk_w, bit_depth);

      if sad < *lowest_sad {
        *lowest_sad = sad;
        *best_mv = MotionVector {
          row: 8 * (y as i16 - po.y as i16),
          col: 8 * (x as i16 - po.x as i16)
        }
      }
    }
  }
}

// Adjust block offset such that entire block lies within frame boundaries
fn adjust_bo(bo: &BlockOffset, fi: &FrameInvariants, blk_w: usize, blk_h: usize) -> BlockOffset {
  BlockOffset {
    x: (bo.x as isize).min(fi.w_in_b as isize - blk_w as isize / 4).max(0) as usize,
    y: (bo.y as isize).min(fi.h_in_b as isize - blk_h as isize / 4).max(0) as usize
  }
}

pub fn estimate_motion_ss4(
  fi: &FrameInvariants, fs: &FrameState, bsize: BlockSize, ref_idx: usize,
  bo: &BlockOffset, bit_depth: usize
) -> Option<MotionVector> {
  if let Some(ref rec) = fi.rec_buffer.frames[ref_idx] {
    let blk_w = bsize.width();
    let blk_h = bsize.height();
    let bo_adj = adjust_bo(bo, fi, blk_w, blk_h);
    let po = PlaneOffset {
      x: (bo_adj.x as isize) << BLOCK_TO_PLANE_SHIFT >> 2,
      y: (bo_adj.y as isize) << BLOCK_TO_PLANE_SHIFT >> 2
    };
    let range = 64 * fi.me_range_scale as isize;
    let (mvx_min, mvx_max, mvy_min, mvy_max) = get_mv_range(fi, &bo_adj, blk_w, blk_h);
    let x_lo = po.x + (((-range).max(mvx_min / 8)) >> 2);
    let x_hi = po.x + (((range).min(mvx_max / 8)) >> 2);
    let y_lo = po.y + (((-range).max(mvy_min / 8)) >> 2);
    let y_hi = po.y + (((range).min(mvy_max / 8)) >> 2);

    let mut lowest_sad = ((blk_w >> 2) * (blk_h >> 2) * 4096) as u32;
    let mut best_mv = MotionVector { row: 0, col: 0 };

    full_search(
      x_lo,
      x_hi,
      y_lo,
      y_hi,
      blk_h >> 2,
      blk_w >> 2,
      &fs.input_qres,
      &rec.input_qres,
      &mut best_mv,
      &mut lowest_sad,
      &po,
      1,
      bit_depth
    );

    Some(MotionVector { row: best_mv.row * 4, col: best_mv.col * 4 })
  } else {
    None
  }
}

pub fn estimate_motion_ss2(
  fi: &FrameInvariants, fs: &FrameState, bsize: BlockSize, ref_idx: usize,
  bo: &BlockOffset, pmvs: &[Option<MotionVector>; 3], bit_depth: usize
) -> Option<MotionVector> {
  if let Some(ref rec) = fi.rec_buffer.frames[ref_idx] {
    let blk_w = bsize.width();
    let blk_h = bsize.height();
    let bo_adj = adjust_bo(bo, fi, blk_w, blk_h);
    let po = PlaneOffset {
      x: (bo_adj.x as isize) << BLOCK_TO_PLANE_SHIFT >> 1,
      y: (bo_adj.y as isize) << BLOCK_TO_PLANE_SHIFT >> 1
    };
    let range = 16;
    let (mvx_min, mvx_max, mvy_min, mvy_max) = get_mv_range(fi, &bo_adj, blk_w, blk_h);

    let mut lowest_sad = ((blk_w >> 1) * (blk_h >> 1) * 4096) as u32;
    let mut best_mv = MotionVector { row: 0, col: 0 };

    for omv in pmvs.iter() {
      if let Some(pmv) = omv {
        let x_lo = po.x + (((pmv.col as isize / 8 - range).max(mvx_min / 8)) >> 1);
        let x_hi = po.x + (((pmv.col as isize / 8 + range).min(mvx_max / 8)) >> 1);
        let y_lo = po.y + (((pmv.row as isize / 8 - range).max(mvy_min / 8)) >> 1);
        let y_hi = po.y + (((pmv.row as isize / 8 + range).min(mvy_max / 8)) >> 1);

        full_search(
          x_lo,
          x_hi,
          y_lo,
          y_hi,
          blk_h >> 1,
          blk_w >> 1,
          &fs.input_hres,
          &rec.input_hres,
          &mut best_mv,
          &mut lowest_sad,
          &po,
          1,
          bit_depth
        );
      }
    }

    Some(MotionVector { row: best_mv.row * 2, col: best_mv.col * 2 })
  } else {
    None
  }
}

#[cfg(test)]
pub mod test {
  use super::*;

  // Generate plane data for get_sad_same()
  fn setup_sad() -> (Plane, Plane) {
    let mut input_plane = Plane::new(640, 480, 0, 0, 128 + 8, 128 + 8);
    let mut rec_plane = input_plane.clone();

    for (i, row) in input_plane.data.chunks_mut(input_plane.cfg.stride).enumerate() {
      for (j, mut pixel) in row.into_iter().enumerate() {
        let val = ((j + i) as i32 & 255i32) as u16;
        assert!(val >= u8::min_value().into() &&
            val <= u8::max_value().into());
        *pixel = val;
      }
    }

    for (i, row) in rec_plane.data.chunks_mut(rec_plane.cfg.stride).enumerate() {
      for (j, mut pixel) in row.into_iter().enumerate() {
        let val = (j as i32 - i as i32 & 255i32) as u16;
        assert!(val >= u8::min_value().into() &&
            val <= u8::max_value().into());
        *pixel = val;
      }
    }

    (input_plane, rec_plane)
  }

  // Regression and validation test for SAD computation
  #[test]
  fn get_sad_same() {
    use partition::BlockSize;
    use partition::BlockSize::*;

    let blocks: Vec<(BlockSize, u32)> = vec![
      (BLOCK_4X4, 1912),
      (BLOCK_4X8, 3496),
      (BLOCK_8X4, 4296),
      (BLOCK_8X8, 7824),
      (BLOCK_8X16, 14416),
      (BLOCK_16X8, 16592),
      (BLOCK_16X16, 31136),
      (BLOCK_16X32, 59552),
      (BLOCK_32X16, 60064),
      (BLOCK_32X32, 120128),
      (BLOCK_32X64, 250176),
      (BLOCK_64X32, 186688),
      (BLOCK_64X64, 438912),
      (BLOCK_64X128, 1016768),
      (BLOCK_128X64, 654272),
      (BLOCK_128X128, 1689792),
      (BLOCK_4X16, 6664),
      (BLOCK_16X4, 8680),
      (BLOCK_8X32, 27600),
      (BLOCK_32X8, 31056),
      (BLOCK_16X64, 116384),
      (BLOCK_64X16, 93344),
    ];

    let bit_depth: usize = 8;
    let (input_plane, rec_plane) = setup_sad();

    for block in blocks {
      let bsw = block.0.width();
      let bsh = block.0.height();
      let po = PlaneOffset { x: 40, y: 40 };

      let mut input_slice = input_plane.slice(&po);
      let mut rec_slice = rec_plane.slice(&po);

      assert_eq!(
        block.1,
        get_sad(&mut input_slice, &mut rec_slice, bsw, bsh, bit_depth)
      );
    }
  }
}
