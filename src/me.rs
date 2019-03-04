// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
pub use self::nasm::get_sad;
#[cfg(any(not(target_arch = "x86_64"), windows, not(feature = "nasm")))]
pub use self::native::get_sad;
use crate::context::{BlockOffset, BLOCK_TO_PLANE_SHIFT, MI_SIZE};
use crate::encoder::ReferenceFrame;
use crate::FrameInvariants;
use crate::FrameState;
use crate::partition::*;
use crate::plane::*;
use crate::util::Pixel;

use std::sync::Arc;

#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
mod nasm {
  use crate::plane::*;
  use crate::util::*;
  use std::mem;

  use libc;

  extern {
    fn rav1e_sad_4x4_hbd_ssse3(
      src: *const u16, src_stride: libc::ptrdiff_t, dst: *const u16,
      dst_stride: libc::ptrdiff_t
    ) -> u32;

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

  #[target_feature(enable = "ssse3")]
  unsafe fn sad_ssse3<T: Pixel>(
    plane_org: &PlaneSlice<'_, T>, plane_ref: &PlaneSlice<'_, T>, blk_h: usize,
    blk_w: usize, bit_depth: usize
  ) -> u32 {
    assert!(mem::size_of::<T>() == 2, "only implemented for u16 for now");
    let mut sum = 0 as u32;
    let org_stride = (plane_org.plane.cfg.stride * mem::size_of::<T>()) as libc::ptrdiff_t;
    let ref_stride = (plane_ref.plane.cfg.stride * mem::size_of::<T>()) as libc::ptrdiff_t;
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
        let org_ptr = org_slice.as_ptr();
        let ref_ptr = ref_slice.as_ptr();
        // FIXME for now, T == u16
        let org_ptr = org_ptr as *const u16;
        let ref_ptr = ref_ptr as *const u16;
        sum += func(org_ptr, org_stride, ref_ptr, ref_stride);
      }
    }
    sum
  }

  #[inline(always)]
  pub fn get_sad<T: Pixel>(
    plane_org: &PlaneSlice<'_, T>, plane_ref: &PlaneSlice<'_, T>, blk_h: usize,
    blk_w: usize, bit_depth: usize
  ) -> u32 {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if is_x86_feature_detected!("ssse3") && blk_h >= 4 && blk_w >= 4 && mem::size_of::<T>() == 2 {
        return unsafe {
          sad_ssse3(plane_org, plane_ref, blk_h, blk_w, bit_depth)
        };
      }
    }
    super::native::get_sad(plane_org, plane_ref, blk_h, blk_w, bit_depth)
  }
}

mod native {
  use crate::plane::*;
  use crate::util::*;

  #[inline(always)]
  pub fn get_sad<T: Pixel>(
    plane_org: &PlaneSlice<'_, T>, plane_ref: &PlaneSlice<'_, T>, blk_h: usize,
    blk_w: usize, _bit_depth: usize
  ) -> u32 {
    let mut sum = 0 as u32;

    let org_iter = plane_org.iter_width(blk_w);
    let ref_iter = plane_ref.iter_width(blk_w);

    for (slice_org, slice_ref) in org_iter.take(blk_h).zip(ref_iter) {
      sum += slice_org
        .iter()
        .zip(slice_ref)
        .map(|(&a, &b)| (i32::cast_from(a) - i32::cast_from(b)).abs() as u32)
        .sum::<u32>();
    }

    sum
  }
}

fn get_mv_range(
  w_in_b: usize, h_in_b: usize, bo: &BlockOffset, blk_w: usize, blk_h: usize
) -> (isize, isize, isize, isize) {
  let border_w = 128 + blk_w as isize * 8;
  let border_h = 128 + blk_h as isize * 8;
  let mvx_min = -(bo.x as isize) * (8 * MI_SIZE) as isize - border_w;
  let mvx_max = (w_in_b - bo.x - blk_w / MI_SIZE) as isize * (8 * MI_SIZE) as isize + border_w;
  let mvy_min = -(bo.y as isize) * (8 * MI_SIZE) as isize - border_h;
  let mvy_max = (h_in_b - bo.y - blk_h / MI_SIZE) as isize * (8 * MI_SIZE) as isize + border_h;

  (mvx_min, mvx_max, mvy_min, mvy_max)
}

pub fn get_subset_predictors<T: Pixel>(
  fi: &FrameInvariants<T>, bo: &BlockOffset, cmv: MotionVector,
  frame_mvs: &[MotionVector], frame_ref_opt: &Option<Arc<ReferenceFrame<T>>>,
  ref_slot: usize
) -> (Vec<MotionVector>) {
  let mut predictors = Vec::new();

  // EPZS subset A and B predictors.

  if bo.x > 0 {
    let left = frame_mvs[bo.y * fi.w_in_b + bo.x - 1];
    predictors.push(left);
  }
  if bo.y > 0 {
    let top = frame_mvs[(bo.y - 1) * fi.w_in_b + bo.x];
    predictors.push(top);

    if bo.x < fi.w_in_b - 1 {
      let top_right = frame_mvs[(bo.y - 1) * fi.w_in_b + bo.x + 1];
      predictors.push(top_right);
    }
  }

  if predictors.len() > 0 {
    let mut median_mv = MotionVector::default();
    for mv in predictors.iter() {
      median_mv = median_mv + *mv;
    }
    median_mv = median_mv / (predictors.len() as i16);

    predictors.push(median_mv.quantize_to_fullpel());
  }

  predictors.push(MotionVector::default());

  // Coarse motion estimation.

  predictors.push(cmv.quantize_to_fullpel());

  // EPZS subset C predictors.

  if let Some(ref frame_ref) = frame_ref_opt {
    let prev_frame_mvs = &frame_ref.frame_mvs[ref_slot];

    if bo.x > 0 {
      let left = prev_frame_mvs[bo.y * fi.w_in_b + bo.x - 1];
      predictors.push(left);
    }
    if bo.y > 0 {
      let top = prev_frame_mvs[(bo.y - 1) * fi.w_in_b + bo.x];
      predictors.push(top);
    }
    if bo.x < fi.w_in_b - 1 {
      let right = prev_frame_mvs[bo.y * fi.w_in_b + bo.x + 1];
      predictors.push(right);
    }
    if bo.y < fi.h_in_b - 1 {
      let bottom = prev_frame_mvs[(bo.y + 1) * fi.w_in_b + bo.x];
      predictors.push(bottom);
    }

    predictors.push(prev_frame_mvs[bo.y * fi.w_in_b + bo.x]);
  }

  predictors
}

pub fn motion_estimation<T: Pixel>(
  fi: &FrameInvariants<T>, fs: &FrameState<T>, bsize: BlockSize, bo: &BlockOffset,
  ref_frame: usize, cmv: MotionVector, pmv: [MotionVector; 2],
  ref_slot: usize
) -> MotionVector {
  match fi.rec_buffer.frames[fi.ref_frames[ref_frame - LAST_FRAME] as usize] {
    Some(ref rec) => {

      let po = PlaneOffset {
        x: (bo.x as isize) << BLOCK_TO_PLANE_SHIFT,
        y: (bo.y as isize) << BLOCK_TO_PLANE_SHIFT
      };
      let blk_w = bsize.width();
      let blk_h = bsize.height();
      let (mvx_min, mvx_max, mvy_min, mvy_max) = get_mv_range(fi.w_in_b, fi.h_in_b, bo, blk_w, blk_h);

      // 0.5 is a fudge factor
      let lambda = (fi.me_lambda * 256.0 * 0.5) as u32;

      // Full-pixel motion estimation

      let mut lowest_cost = std::u64::MAX;
      let mut best_mv = MotionVector::default();

      let frame_mvs = &fs.frame_mvs[ref_slot];
      let frame_ref = &fi.rec_buffer.frames[fi.ref_frames[0] as usize];

      if fi.config.speed_settings.diamond_me {
        let predictors = get_subset_predictors(fi, bo, cmv,
          frame_mvs, frame_ref, ref_slot);

        diamond_me_search(
          fi, &po,
          &fs.input.planes[0], &rec.frame.planes[0],
          &predictors, fi.sequence.bit_depth,
          pmv, lambda,
          mvx_min, mvx_max, mvy_min, mvy_max,
          blk_w, blk_h,
          &mut best_mv, &mut lowest_cost);
      } else {
        let range = 16;
        let x_lo = po.x + ((-range + (cmv.col / 8) as isize).max(mvx_min / 8).min(mvx_max / 8));
        let x_hi = po.x + ((range + (cmv.col / 8) as isize).max(mvx_min / 8).min(mvx_max / 8));
        let y_lo = po.y + ((-range + (cmv.row / 8) as isize).max(mvy_min / 8).min(mvy_max / 8));
        let y_hi = po.y + ((range + (cmv.row / 8) as isize).max(mvy_min / 8).min(mvy_max / 8));

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
          &mut lowest_cost,
          &po,
          2,
          fi.sequence.bit_depth,
          lambda,
          pmv,
          fi.allow_high_precision_mv
        );
      }

      // Sub-pixel motion estimation

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
                fi,
                0,
                &po,
                tmp_slice,
                blk_w,
                blk_h,
                [ref_frame, NONE_FRAME],
                [cand_mv, MotionVector::default()]
              );
            }

            let plane_org = fs.input.planes[0].slice(&po);
            let plane_ref = tmp_plane.slice(&PlaneOffset { x: 0, y: 0 });

            let sad = get_sad(&plane_org, &plane_ref, blk_h, blk_w, fi.sequence.bit_depth);

            let rate1 = get_mv_rate(cand_mv, pmv[0], fi.allow_high_precision_mv);
            let rate2 = get_mv_rate(cand_mv, pmv[1], fi.allow_high_precision_mv);
            let rate = rate1.min(rate2 + 1);
            let cost = 256 * sad as u64 + rate as u64 * lambda as u64;

            if cost < lowest_cost {
              lowest_cost = cost;
              best_mv = cand_mv;
            }
          }
        }
      }

      best_mv
    }

    None => MotionVector::default()
  }
}

fn get_best_predictor<T: Pixel>(
  fi: &FrameInvariants<T>,
  po: &PlaneOffset, p_org: &Plane<T>, p_ref: &Plane<T>,
  predictors: &[MotionVector],
  bit_depth: usize, pmv: [MotionVector; 2], lambda: u32,
  mvx_min: isize, mvx_max: isize, mvy_min: isize, mvy_max: isize,
  blk_w: usize, blk_h: usize,
  center_mv: &mut MotionVector, center_mv_cost: &mut u64) {
  *center_mv = MotionVector::default();
  *center_mv_cost = std::u64::MAX;

  for &init_mv in predictors.iter() {
    let cost = get_mv_rd_cost(
      fi, po, p_org, p_ref, bit_depth,
      pmv, lambda, mvx_min, mvx_max, mvy_min, mvy_max,
      blk_w, blk_h, init_mv);

    if cost < *center_mv_cost {
      *center_mv = init_mv;
      *center_mv_cost = cost;
    }
  }
}

fn diamond_me_search<T: Pixel>(
  fi: &FrameInvariants<T>,
  po: &PlaneOffset, p_org: &Plane<T>, p_ref: &Plane<T>,
  predictors: &[MotionVector],
  bit_depth: usize, pmv: [MotionVector; 2], lambda: u32,
  mvx_min: isize, mvx_max: isize, mvy_min: isize, mvy_max: isize,
  blk_w: usize, blk_h: usize,
  center_mv: &mut MotionVector, center_mv_cost: &mut u64)
{
  let diamond_pattern = [(1i16, 0i16), (0, 1), (-1, 0), (0, -1)];
  let mut diamond_radius: i16 = 16;

  get_best_predictor(
    fi, po, p_org, p_ref, &predictors,
    bit_depth, pmv, lambda, mvx_min, mvx_max, mvy_min, mvy_max,
    blk_w, blk_h, center_mv, center_mv_cost);

  loop {
    let mut best_diamond_rd_cost = std::u64::MAX;
    let mut best_diamond_mv = MotionVector::default();

    for p in diamond_pattern.iter() {

        let cand_mv = MotionVector {
          row: center_mv.row + diamond_radius * p.0,
          col: center_mv.col + diamond_radius * p.1
        };

        let rd_cost = get_mv_rd_cost(
          fi, &po, p_org, p_ref, bit_depth,
          pmv, lambda, mvx_min, mvx_max, mvy_min, mvy_max,
          blk_w, blk_h, cand_mv);

        if rd_cost < best_diamond_rd_cost {
          best_diamond_rd_cost = rd_cost;
          best_diamond_mv = cand_mv;
        }
    }

    if *center_mv_cost <= best_diamond_rd_cost {
      if diamond_radius == 8 {
        break;
      } else {
        diamond_radius /= 2;
      }
    }
    else {
      *center_mv = best_diamond_mv;
      *center_mv_cost = best_diamond_rd_cost;
    }
  }

  assert!(*center_mv_cost < std::u64::MAX);
}

fn get_mv_rd_cost<T: Pixel>(
  fi: &FrameInvariants<T>,
  po: &PlaneOffset, p_org: &Plane<T>, p_ref: &Plane<T>, bit_depth: usize,
  pmv: [MotionVector; 2], lambda: u32,
  mvx_min: isize, mvx_max: isize, mvy_min: isize, mvy_max: isize,
  blk_w: usize, blk_h: usize,
  cand_mv: MotionVector) -> u64
{
  if (cand_mv.col as isize) < mvx_min || (cand_mv.col as isize) > mvx_max {
    return std::u64::MAX;
  }
  if (cand_mv.row as isize) < mvy_min || (cand_mv.row as isize) > mvy_max {
    return std::u64::MAX;
  }

  let plane_org = p_org.slice(po);
  let plane_ref = p_ref.slice(&PlaneOffset {
    x: po.x + (cand_mv.col / 8) as isize,
    y: po.y + (cand_mv.row / 8) as isize
  });

  let sad = get_sad(&plane_org, &plane_ref, blk_h, blk_w, bit_depth);

  let rate1 = get_mv_rate(cand_mv, pmv[0], fi.allow_high_precision_mv);
  let rate2 = get_mv_rate(cand_mv, pmv[1], fi.allow_high_precision_mv);
  let rate = rate1.min(rate2 + 1);

  256 * sad as u64 + rate as u64 * lambda as u64
}

fn full_search<T: Pixel>(
  x_lo: isize, x_hi: isize, y_lo: isize, y_hi: isize, blk_h: usize,
  blk_w: usize, p_org: &Plane<T>, p_ref: &Plane<T>, best_mv: &mut MotionVector,
  lowest_cost: &mut u64, po: &PlaneOffset, step: usize, bit_depth: usize,
  lambda: u32, pmv: [MotionVector; 2], allow_high_precision_mv: bool
) {
    let search_range_y = (y_lo..=y_hi).step_by(step);
    let search_range_x = (x_lo..=x_hi).step_by(step);
    let search_area = search_range_y.flat_map(|y| { search_range_x.clone().map(move |x| (y, x)) });

    let (cost, mv) = search_area.map(|(y, x)| {
      let plane_org = p_org.slice(po);
      let plane_ref = p_ref.slice(&PlaneOffset { x, y });

      let sad = get_sad(&plane_org, &plane_ref, blk_h, blk_w, bit_depth);

      let mv = MotionVector {
        row: 8 * (y as i16 - po.y as i16),
        col: 8 * (x as i16 - po.x as i16)
      };

      let rate1 = get_mv_rate(mv, pmv[0], allow_high_precision_mv);
      let rate2 = get_mv_rate(mv, pmv[1], allow_high_precision_mv);
      let rate = rate1.min(rate2 + 1);
      let cost = 256 * sad as u64 + rate as u64 * lambda as u64;

      (cost, mv)
  }).min_by_key(|(c, _)| *c).unwrap();

    *lowest_cost = cost;
    *best_mv = mv;
}

// Adjust block offset such that entire block lies within frame boundaries
fn adjust_bo<T: Pixel>(bo: &BlockOffset, fi: &FrameInvariants<T>, blk_w: usize, blk_h: usize) -> BlockOffset {
  BlockOffset {
    x: (bo.x as isize).min(fi.w_in_b as isize - blk_w as isize / 4).max(0) as usize,
    y: (bo.y as isize).min(fi.h_in_b as isize - blk_h as isize / 4).max(0) as usize
  }
}

fn get_mv_rate(a: MotionVector, b: MotionVector, allow_high_precision_mv: bool) -> u32 {
  fn diff_to_rate(diff: i16, allow_high_precision_mv: bool) -> u32 {
    let d = if allow_high_precision_mv { diff } else { diff >> 1 };
    if d == 0 {
      0
    } else {
      2 * (16 - d.abs().leading_zeros())
    }
  }

  diff_to_rate(a.row - b.row, allow_high_precision_mv) + diff_to_rate(a.col - b.col, allow_high_precision_mv)
}

pub fn estimate_motion_ss4<T: Pixel>(
  fi: &FrameInvariants<T>, fs: &FrameState<T>, bsize: BlockSize, ref_idx: usize,
  bo: &BlockOffset
) -> Option<MotionVector> {
  if let Some(ref rec) = fi.rec_buffer.frames[ref_idx] {
    let blk_w = bsize.width();
    let blk_h = bsize.height();
    let bo_adj = adjust_bo(bo, fi, blk_w, blk_h);
    let po = PlaneOffset {
      x: (bo_adj.x as isize) << BLOCK_TO_PLANE_SHIFT >> 2,
      y: (bo_adj.y as isize) << BLOCK_TO_PLANE_SHIFT >> 2
    };

    let range_x = 192 * fi.me_range_scale as isize;
    let range_y = 64 * fi.me_range_scale as isize;
    let (mvx_min, mvx_max, mvy_min, mvy_max) = get_mv_range(fi.w_in_b, fi.h_in_b, &bo_adj, blk_w, blk_h);
    let x_lo = po.x + (((-range_x).max(mvx_min / 8)) >> 2);
    let x_hi = po.x + (((range_x).min(mvx_max / 8)) >> 2);
    let y_lo = po.y + (((-range_y).max(mvy_min / 8)) >> 2);
    let y_hi = po.y + (((range_y).min(mvy_max / 8)) >> 2);

    let mut lowest_cost = std::u64::MAX;
    let mut best_mv = MotionVector::default();

    // Divide by 16 to account for subsampling, 0.125 is a fudge factor
    let lambda = (fi.me_lambda * 256.0 / 16.0 * 0.125) as u32;

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
      &mut lowest_cost,
      &po,
      1,
      fi.sequence.bit_depth,
      lambda,
      [MotionVector::default(); 2],
      fi.allow_high_precision_mv
    );

    Some(MotionVector { row: best_mv.row * 4, col: best_mv.col * 4 })
  } else {
    None
  }
}

pub fn estimate_motion_ss2<T: Pixel>(
  fi: &FrameInvariants<T>, fs: &FrameState<T>, bsize: BlockSize, ref_idx: usize,
  bo: &BlockOffset, pmvs: &[Option<MotionVector>; 3]
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
    let (mvx_min, mvx_max, mvy_min, mvy_max) = get_mv_range(fi.w_in_b, fi.h_in_b, &bo_adj, blk_w, blk_h);

    let mut lowest_cost = std::u64::MAX;
    let mut best_mv = MotionVector::default();

    // Divide by 4 to account for subsampling, 0.125 is a fudge factor
    let lambda = (fi.me_lambda * 256.0 / 4.0 * 0.125) as u32;

    for omv in pmvs.iter() {
      if let Some(pmv) = omv {
        let x_lo = po.x + (((pmv.col as isize / 8 - range).max(mvx_min / 8).min(mvx_max / 8)) >> 1);
        let x_hi = po.x + (((pmv.col as isize / 8 + range).max(mvx_min / 8).min(mvx_max / 8)) >> 1);
        let y_lo = po.y + (((pmv.row as isize / 8 - range).max(mvy_min / 8).min(mvy_max / 8)) >> 1);
        let y_hi = po.y + (((pmv.row as isize / 8 + range).max(mvy_min / 8).min(mvy_max / 8)) >> 1);

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
          &mut lowest_cost,
          &po,
          1,
          fi.sequence.bit_depth,
          lambda,
          [MotionVector::default(); 2],
          fi.allow_high_precision_mv
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
  use crate::partition::BlockSize;
  use crate::partition::BlockSize::*;

  // Generate plane data for get_sad_same()
  fn setup_sad() -> (Plane<u16>, Plane<u16>) {
    let mut input_plane = Plane::new(640, 480, 0, 0, 128 + 8, 128 + 8);
    let mut rec_plane = input_plane.clone();

    let pad_off = (input_plane.cfg.xorigin - input_plane.cfg.xpad) as i32;

    for (i, row) in input_plane.data.chunks_mut(input_plane.cfg.stride).enumerate() {
      for (j, pixel) in row.into_iter().enumerate() {
        let val = ((j + i) as i32 - pad_off & 255i32) as u16;
        assert!(val >= u8::min_value().into() &&
            val <= u8::max_value().into());
        *pixel = val;
      }
    }

    for (i, row) in rec_plane.data.chunks_mut(rec_plane.cfg.stride).enumerate() {
      for (j, pixel) in row.into_iter().enumerate() {
        let val = (j as i32 - i as i32 - pad_off & 255i32) as u16;
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
