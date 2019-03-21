// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
pub use self::nasm::get_sad;
#[cfg(any(not(target_arch = "x86_64"), not(feature = "nasm")))]
pub use self::native::get_sad;
use crate::context::{BlockOffset, BLOCK_TO_PLANE_SHIFT, MI_SIZE};
use crate::encoder::ReferenceFrame;
use crate::FrameInvariants;
use crate::FrameState;
use crate::partition::*;
use crate::plane::*;
use crate::util::Pixel;

use std::ops::{Index, IndexMut};
use std::sync::Arc;

#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
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

    fn rav1e_sad4x4_sse2(
      src: *const u8, src_stride: libc::ptrdiff_t, dst: *const u8,
      dst_stride: libc::ptrdiff_t
    ) -> u32;

    fn rav1e_sad8x8_sse2(
      src: *const u8, src_stride: libc::ptrdiff_t, dst: *const u8,
      dst_stride: libc::ptrdiff_t
    ) -> u32;

    fn rav1e_sad16x16_sse2(
      src: *const u8, src_stride: libc::ptrdiff_t, dst: *const u8,
      dst_stride: libc::ptrdiff_t
    ) -> u32;

    fn rav1e_sad32x32_sse2(
      src: *const u8, src_stride: libc::ptrdiff_t, dst: *const u8,
      dst_stride: libc::ptrdiff_t
    ) -> u32;

    fn rav1e_sad64x64_sse2(
      src: *const u8, src_stride: libc::ptrdiff_t, dst: *const u8,
      dst_stride: libc::ptrdiff_t
    ) -> u32;

    fn rav1e_sad128x128_sse2(
      src: *const u8, src_stride: libc::ptrdiff_t, dst: *const u8,
      dst_stride: libc::ptrdiff_t
    ) -> u32;
  }

  #[target_feature(enable = "ssse3")]
  unsafe fn sad_hbd_ssse3(
    plane_org: &PlaneSlice<'_, u16>, plane_ref: &PlaneSlice<'_, u16>, blk_h: usize,
    blk_w: usize, bit_depth: usize
  ) -> u32 {
    let mut sum = 0 as u32;
    let org_stride = (plane_org.plane.cfg.stride * 2) as libc::ptrdiff_t;
    let ref_stride = (plane_ref.plane.cfg.stride * 2) as libc::ptrdiff_t;
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

  #[target_feature(enable = "sse2")]
  unsafe fn sad_sse2(
    plane_org: &PlaneSlice<'_, u8>, plane_ref: &PlaneSlice<'_, u8>, blk_h: usize,
    blk_w: usize
  ) -> u32 {
    // FIXME unaligned blocks coming from hres/qres ME search
    let ptr_align_log2 = (plane_org.as_ptr() as usize).trailing_zeros() as usize;
    // The largest unaligned-safe function is for 8x8
    let ptr_align = 1 << ptr_align_log2.max(3);
    let mut sum = 0 as u32;
    let org_stride = plane_org.plane.cfg.stride as libc::ptrdiff_t;
    let ref_stride = plane_ref.plane.cfg.stride as libc::ptrdiff_t;
    assert!(blk_h >= 4 && blk_w >= 4);
    let step_size = blk_h.min(blk_w).min(ptr_align);
    let func = match step_size.ilog() {
      3 => rav1e_sad4x4_sse2,
      4 => rav1e_sad8x8_sse2,
      5 => rav1e_sad16x16_sse2,
      6 => rav1e_sad32x32_sse2,
      7 => rav1e_sad64x64_sse2,
      8 => rav1e_sad128x128_sse2,
      _ => rav1e_sad128x128_sse2
    };
    for r in (0..blk_h).step_by(step_size) {
      for c in (0..blk_w).step_by(step_size) {
        let org_slice = plane_org.subslice(c, r);
        let ref_slice = plane_ref.subslice(c, r);
        let org_ptr = org_slice.as_ptr();
        let ref_ptr = ref_slice.as_ptr();
        // FIXME for now, T == u8
        let org_ptr = org_ptr as *const u8;
        let ref_ptr = ref_ptr as *const u8;
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
    #[cfg(all(target_arch = "x86_64", feature = "nasm"))]
    {
      if mem::size_of::<T>() == 2 && is_x86_feature_detected!("ssse3") && blk_h >= 4 && blk_w >= 4 {
        return unsafe {
          let plane_org = &*(plane_org as *const _ as *const PlaneSlice<'_, u16>);
          let plane_ref = &*(plane_ref as *const _ as *const PlaneSlice<'_, u16>);
          sad_hbd_ssse3(plane_org, plane_ref, blk_h, blk_w, bit_depth)
        };
      }
      if mem::size_of::<T>() == 1 && is_x86_feature_detected!("sse2") && blk_h >= 4 && blk_w >= 4 {
        return unsafe {
          let plane_org = &*(plane_org as *const _ as *const PlaneSlice<'_, u8>);
          let plane_ref = &*(plane_ref as *const _ as *const PlaneSlice<'_, u8>);
          sad_sse2(plane_org, plane_ref, blk_h, blk_w)
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

#[derive(Debug, Clone)]
pub struct FrameMotionVectors {
  mvs: Box<[MotionVector]>,
  pub cols: usize,
  pub rows: usize,
}

impl FrameMotionVectors {
  pub fn new(cols: usize, rows: usize) -> Self {
    Self {
      mvs: vec![MotionVector::default(); cols * rows].into_boxed_slice(),
      cols,
      rows,
    }
  }
}

impl Index<usize> for FrameMotionVectors {
  type Output = [MotionVector];
  #[inline]
  fn index(&self, index: usize) -> &Self::Output {
    &self.mvs[index * self.cols..(index + 1) * self.cols]
  }
}

impl IndexMut<usize> for FrameMotionVectors {
  #[inline]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    &mut self.mvs[index * self.cols..(index + 1) * self.cols]
  }
}

fn get_mv_range(
  w_in_b: usize, h_in_b: usize, bo: BlockOffset, blk_w: usize, blk_h: usize
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
  bo: BlockOffset, cmv: MotionVector,
  w_in_b: usize, h_in_b: usize,
  frame_mvs: &FrameMotionVectors, frame_ref_opt: &Option<Arc<ReferenceFrame<T>>>,
  ref_frame_id: usize
) -> (Vec<MotionVector>) {
  let mut predictors = Vec::new();

  // Zero motion vector
  predictors.push(MotionVector::default());

  // Coarse motion estimation.
  predictors.push(cmv.quantize_to_fullpel());

  // EPZS subset A and B predictors.

  let mut median_preds = Vec::new();
  if bo.x > 0 {
    let left = frame_mvs[bo.y][bo.x - 1];
    median_preds.push(left);
    if !left.is_zero() { predictors.push(left); }
  }
  if bo.y > 0 {
    let top = frame_mvs[bo.y - 1][bo.x];
    median_preds.push(top);
    if !top.is_zero() { predictors.push(top); }

    if bo.x < w_in_b - 1 {
      let top_right = frame_mvs[bo.y - 1][bo.x + 1];
      median_preds.push(top_right);
      if !top_right.is_zero() { predictors.push(top_right); }
    }
  }

  if !median_preds.is_empty() {
    let mut median_mv = MotionVector::default();
    for mv in median_preds.iter() {
      median_mv = median_mv + *mv;
    }
    median_mv = median_mv / (median_preds.len() as i16);
    let median_mv_quant = median_mv.quantize_to_fullpel();
    if !median_mv_quant.is_zero() { predictors.push(median_mv_quant); }
  }

  // EPZS subset C predictors.

  if let Some(ref frame_ref) = frame_ref_opt {
    let prev_frame_mvs = &frame_ref.frame_mvs[ref_frame_id];

    if bo.x > 0 {
      let left = prev_frame_mvs[bo.y][bo.x - 1];
      if !left.is_zero() { predictors.push(left); }
    }
    if bo.y > 0 {
      let top = prev_frame_mvs[bo.y - 1][bo.x];
      if !top.is_zero() { predictors.push(top); }
    }
    if bo.x < w_in_b - 1 {
      let right = prev_frame_mvs[bo.y][bo.x + 1];
      if !right.is_zero() { predictors.push(right); }
    }
    if bo.y < h_in_b - 1 {
      let bottom = prev_frame_mvs[bo.y + 1][bo.x];
      if !bottom.is_zero() { predictors.push(bottom); }
    }

    let previous = prev_frame_mvs[bo.y][bo.x];
    if !previous.is_zero() { predictors.push(previous); }
  }

  predictors
}

pub trait MotionEstimation {
  fn full_pixel_me<T: Pixel>(
    fi: &FrameInvariants<T>, fs: &FrameState<T>, rec: &Arc<ReferenceFrame<T>>,
    bo: BlockOffset, lambda: u32,
    cmv: MotionVector, pmv: [MotionVector; 2],
    mvx_min: isize, mvx_max: isize, mvy_min: isize, mvy_max: isize,
    blk_w: usize, blk_h: usize, best_mv: &mut MotionVector,
    lowest_cost: &mut u64, ref_frame: usize
  );

  fn sub_pixel_me<T: Pixel>(
    fi: &FrameInvariants<T>, fs: &FrameState<T>, rec: &Arc<ReferenceFrame<T>>,
    bo: BlockOffset, lambda: u32, pmv: [MotionVector; 2],
    mvx_min: isize, mvx_max: isize, mvy_min: isize, mvy_max: isize,
    blk_w: usize, blk_h: usize, best_mv: &mut MotionVector,
    lowest_cost: &mut u64, ref_frame: usize
  );

  fn motion_estimation<T: Pixel> (
    fi: &FrameInvariants<T>, fs: &FrameState<T>, bsize: BlockSize,
    bo: BlockOffset, ref_frame: usize, cmv: MotionVector,
    pmv: [MotionVector; 2]
  ) -> MotionVector {
    match fi.rec_buffer.frames[fi.ref_frames[ref_frame - LAST_FRAME] as usize]
    {
      Some(ref rec) => {
        let blk_w = bsize.width();
        let blk_h = bsize.height();
        let (mvx_min, mvx_max, mvy_min, mvy_max) =
          get_mv_range(fi.w_in_b, fi.h_in_b, bo, blk_w, blk_h);

        // 0.5 is a fudge factor
        let lambda = (fi.me_lambda * 256.0 * 0.5) as u32;

        // Full-pixel motion estimation

        let mut lowest_cost = std::u64::MAX;
        let mut best_mv = MotionVector::default();

        Self::full_pixel_me(fi, fs, rec, bo, lambda, cmv, pmv,
                           mvx_min, mvx_max, mvy_min, mvy_max, blk_w, blk_h,
                           &mut best_mv, &mut lowest_cost, ref_frame);

        Self::sub_pixel_me(fi, fs, rec, bo, lambda, pmv,
                           mvx_min, mvx_max, mvy_min, mvy_max, blk_w, blk_h,
                           &mut best_mv, &mut lowest_cost, ref_frame);

        best_mv
      }

      None => MotionVector::default()
    }
  }

  fn estimate_motion_ss2<T: Pixel>(
    fi: &FrameInvariants<T>, fs: &FrameState<T>, bsize: BlockSize, ref_idx: usize,
    bo: BlockOffset, pmvs: &[Option<MotionVector>; 3], ref_frame: usize
  ) -> Option<MotionVector> {
    if let Some(ref rec) = fi.rec_buffer.frames[ref_idx] {
      let blk_w = bsize.width();
      let blk_h = bsize.height();
      let bo_adj = adjust_bo(bo, fi, blk_w, blk_h);
      let bo_adj_h = BlockOffset{x: bo_adj.x >> 1, y: bo_adj.y >> 1};
      let po = PlaneOffset {
        x: (bo_adj.x as isize) << BLOCK_TO_PLANE_SHIFT >> 1,
        y: (bo_adj.y as isize) << BLOCK_TO_PLANE_SHIFT >> 1
      };
      let (mvx_min, mvx_max, mvy_min, mvy_max) = get_mv_range(fi.w_in_b, fi.h_in_b, bo_adj, blk_w, blk_h);

      let global_mv = [MotionVector{row: 0, col: 0}; 2];
      let frame_mvs = &fs.frame_mvs[ref_frame];
      let frame_ref_opt = &fi.rec_buffer.frames[fi.ref_frames[0] as usize];

      let mut lowest_cost = std::u64::MAX;
      let mut best_mv = MotionVector::default();

      // Divide by 4 to account for subsampling, 0.125 is a fudge factor
      let lambda = (fi.me_lambda * 256.0 / 4.0 * 0.125) as u32;

      Self::me_ss2(
        fi, fs, pmvs, bo_adj_h,
        frame_mvs, frame_ref_opt, po, rec, global_mv, lambda,
        mvx_min, mvx_max, mvy_min, mvy_max, blk_w, blk_h,
        &mut best_mv, &mut lowest_cost
      );

      Some(MotionVector { row: best_mv.row * 2, col: best_mv.col * 2 })
    } else {
      None
    }
  }

  fn me_ss2<T: Pixel>(
    fi: &FrameInvariants<T>, fs: &FrameState<T>,
    pmvs: &[Option<MotionVector>; 3], bo_adj_h: BlockOffset,
    frame_mvs: &FrameMotionVectors, frame_ref_opt: &Option<Arc<ReferenceFrame<T>>>,
    po: PlaneOffset, rec: &Arc<ReferenceFrame<T>>,
    global_mv: [MotionVector; 2], lambda: u32,
    mvx_min: isize, mvx_max: isize, mvy_min: isize, mvy_max: isize,
    blk_w: usize, blk_h: usize,
    best_mv: &mut MotionVector, lowest_cost: &mut u64
  );
}

pub struct DiamondSearch {}
pub struct FullSearch {}

impl MotionEstimation for DiamondSearch {
  fn full_pixel_me<T: Pixel>(
    fi: &FrameInvariants<T>, fs: &FrameState<T>, rec: &Arc<ReferenceFrame<T>>,
    bo: BlockOffset, lambda: u32,
    cmv: MotionVector, pmv: [MotionVector; 2], mvx_min: isize, mvx_max: isize,
    mvy_min: isize, mvy_max: isize, blk_w: usize, blk_h: usize,
    best_mv: &mut MotionVector, lowest_cost: &mut u64, ref_frame: usize
  ) {
    let frame_mvs = &fs.frame_mvs[ref_frame - LAST_FRAME];
    let frame_ref = &fi.rec_buffer.frames[fi.ref_frames[0] as usize];
    let predictors =
      get_subset_predictors(bo, cmv, fi.w_in_b, fi.h_in_b, frame_mvs, frame_ref, ref_frame - LAST_FRAME);

    diamond_me_search(
      fi,
      bo.to_luma_plane_offset(),
      &fs.input.planes[0],
      &rec.frame.planes[0],
      &predictors,
      fi.sequence.bit_depth,
      pmv,
      lambda,
      mvx_min,
      mvx_max,
      mvy_min,
      mvy_max,
      blk_w,
      blk_h,
      best_mv,
      lowest_cost,
      false,
      ref_frame
    );
  }

  fn sub_pixel_me<T: Pixel>(
    fi: &FrameInvariants<T>, fs: &FrameState<T>, rec: &Arc<ReferenceFrame<T>>,
    bo: BlockOffset, lambda: u32,
    pmv: [MotionVector; 2], mvx_min: isize, mvx_max: isize,
    mvy_min: isize, mvy_max: isize, blk_w: usize, blk_h: usize,
    best_mv: &mut MotionVector, lowest_cost: &mut u64, ref_frame: usize,
  )
  {
    let predictors = vec![*best_mv];
    diamond_me_search(
      fi,
      bo.to_luma_plane_offset(),
      &fs.input.planes[0],
      &rec.frame.planes[0],
      &predictors,
      fi.sequence.bit_depth,
      pmv,
      lambda,
      mvx_min,
      mvx_max,
      mvy_min,
      mvy_max,
      blk_w,
      blk_h,
      best_mv,
      lowest_cost,
      true,
      ref_frame
    );
  }

  fn me_ss2<T: Pixel>(
    fi: &FrameInvariants<T>, fs: &FrameState<T>,
    pmvs: &[Option<MotionVector>; 3], bo_adj_h: BlockOffset,
    frame_mvs: &FrameMotionVectors, frame_ref_opt: &Option<Arc<ReferenceFrame<T>>>,
    po: PlaneOffset, rec: &Arc<ReferenceFrame<T>>,
    global_mv: [MotionVector; 2], lambda: u32,
    mvx_min: isize, mvx_max: isize, mvy_min: isize, mvy_max: isize,
    blk_w: usize, blk_h: usize,
    best_mv: &mut MotionVector, lowest_cost: &mut u64
  ) {
    for omv in pmvs.iter() {
      if let Some(pmv) = omv {
        let mut predictors = get_subset_predictors::<T>(
          bo_adj_h,
          MotionVector{row: pmv.row, col: pmv.col},
          fi.w_in_b, fi.h_in_b,
          &frame_mvs, frame_ref_opt, 0
        );

        for predictor in &mut predictors {
          predictor.row >>= 1;
          predictor.col >>= 1;
        }

        diamond_me_search(
          fi, po,
          &fs.input_hres, &rec.input_hres,
          &predictors, fi.sequence.bit_depth,
          global_mv, lambda,
          mvx_min >> 1, mvx_max >> 1, mvy_min >> 1, mvy_max >> 1,
          blk_w >> 1, blk_h >> 1,
          best_mv, lowest_cost,
          false, 0
        );
      }
    }
  }
}

impl MotionEstimation for FullSearch {
  fn full_pixel_me<T: Pixel>(
    fi: &FrameInvariants<T>, fs: &FrameState<T>, rec: &Arc<ReferenceFrame<T>>,
    bo: BlockOffset, lambda: u32,
    cmv: MotionVector, pmv: [MotionVector; 2], mvx_min: isize, mvx_max: isize,
    mvy_min: isize, mvy_max: isize, blk_w: usize, blk_h: usize,
    best_mv: &mut MotionVector, lowest_cost: &mut u64, _ref_frame: usize
  ) {
    let po = bo.to_luma_plane_offset();
    let range = 16;
    let x_lo = po.x
      + ((-range + (cmv.col / 8) as isize).max(mvx_min / 8).min(mvx_max / 8));
    let x_hi = po.x
      + ((range + (cmv.col / 8) as isize).max(mvx_min / 8).min(mvx_max / 8));
    let y_lo = po.y
      + ((-range + (cmv.row / 8) as isize).max(mvy_min / 8).min(mvy_max / 8));
    let y_hi = po.y
      + ((range + (cmv.row / 8) as isize).max(mvy_min / 8).min(mvy_max / 8));

    full_search(
      x_lo,
      x_hi,
      y_lo,
      y_hi,
      blk_h,
      blk_w,
      &fs.input.planes[0],
      &rec.frame.planes[0],
      best_mv,
      lowest_cost,
      po,
      2,
      fi.sequence.bit_depth,
      lambda,
      pmv,
      fi.allow_high_precision_mv
    );
  }

  fn sub_pixel_me<T: Pixel>(
    fi: &FrameInvariants<T>, fs: &FrameState<T>, _rec: &Arc<ReferenceFrame<T>>,
    bo: BlockOffset, lambda: u32,
    pmv: [MotionVector; 2], mvx_min: isize, mvx_max: isize,
    mvy_min: isize, mvy_max: isize, blk_w: usize, blk_h: usize,
    best_mv: &mut MotionVector, lowest_cost: &mut u64, ref_frame: usize,
  )
  {
    telescopic_subpel_search(
      fi,
      fs,
      bo.to_luma_plane_offset(),
      lambda,
      ref_frame,
      pmv,
      mvx_min,
      mvx_max,
      mvy_min,
      mvy_max,
      blk_w,
      blk_h,
      best_mv,
      lowest_cost
    );
  }

  fn me_ss2<T: Pixel>(
    fi: &FrameInvariants<T>, fs: &FrameState<T>,
    pmvs: &[Option<MotionVector>; 3], _bo_adj_h: BlockOffset,
    _frame_mvs: &FrameMotionVectors, _frame_ref_opt: &Option<Arc<ReferenceFrame<T>>>,
    po: PlaneOffset, rec: &Arc<ReferenceFrame<T>>,
    _global_mv: [MotionVector; 2], lambda: u32,
    mvx_min: isize, mvx_max: isize, mvy_min: isize, mvy_max: isize,
    blk_w: usize, blk_h: usize,
    best_mv: &mut MotionVector, lowest_cost: &mut u64
  ) {
    let range = 16;
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
          best_mv,
          lowest_cost,
          po,
          1,
          fi.sequence.bit_depth,
          lambda,
          [MotionVector::default(); 2],
          fi.allow_high_precision_mv
        );
      }
    }
  }
}

fn get_best_predictor<T: Pixel>(
  fi: &FrameInvariants<T>,
  po: PlaneOffset, p_org: &Plane<T>, p_ref: &Plane<T>,
  predictors: &[MotionVector],
  bit_depth: usize, pmv: [MotionVector; 2], lambda: u32,
  mvx_min: isize, mvx_max: isize, mvy_min: isize, mvy_max: isize,
  blk_w: usize, blk_h: usize,
  center_mv: &mut MotionVector, center_mv_cost: &mut u64,
  tmp_plane_opt: &mut Option<Plane<T>>, ref_frame: usize) {
  *center_mv = MotionVector::default();
  *center_mv_cost = std::u64::MAX;

  for &init_mv in predictors.iter() {
    let cost = get_mv_rd_cost(
      fi, po, p_org, p_ref, bit_depth,
      pmv, lambda, mvx_min, mvx_max, mvy_min, mvy_max,
      blk_w, blk_h, init_mv, tmp_plane_opt, ref_frame);

    if cost < *center_mv_cost {
      *center_mv = init_mv;
      *center_mv_cost = cost;
    }
  }
}

fn diamond_me_search<T: Pixel>(
  fi: &FrameInvariants<T>,
  po: PlaneOffset, p_org: &Plane<T>, p_ref: &Plane<T>,
  predictors: &[MotionVector],
  bit_depth: usize, pmv: [MotionVector; 2], lambda: u32,
  mvx_min: isize, mvx_max: isize, mvy_min: isize, mvy_max: isize,
  blk_w: usize, blk_h: usize,
  center_mv: &mut MotionVector, center_mv_cost: &mut u64,
  subpixel: bool, ref_frame: usize)
{
  let diamond_pattern = [(1i16, 0i16), (0, 1), (-1, 0), (0, -1)];
  let (mut diamond_radius, diamond_radius_end, mut tmp_plane_opt) = {
    if subpixel {
      // Sub-pixel motion estimation
      (
        4i16,
        if fi.allow_high_precision_mv {1i16} else {2i16},
        Some(Plane::new(blk_w, blk_h, 0, 0, 0, 0)),
      )
    } else {
      // Full pixel motion estimation
      (16i16, 8i16, None)
    }
  };

  get_best_predictor(
    fi, po, p_org, p_ref, &predictors,
    bit_depth, pmv, lambda, mvx_min, mvx_max, mvy_min, mvy_max,
    blk_w, blk_h, center_mv, center_mv_cost,
    &mut tmp_plane_opt, ref_frame);

  loop {
    let mut best_diamond_rd_cost = std::u64::MAX;
    let mut best_diamond_mv = MotionVector::default();

    for p in diamond_pattern.iter() {

        let cand_mv = MotionVector {
          row: center_mv.row + diamond_radius * p.0,
          col: center_mv.col + diamond_radius * p.1
        };

        let rd_cost = get_mv_rd_cost(
          fi, po, p_org, p_ref, bit_depth,
          pmv, lambda, mvx_min, mvx_max, mvy_min, mvy_max,
          blk_w, blk_h, cand_mv, &mut tmp_plane_opt, ref_frame);

        if rd_cost < best_diamond_rd_cost {
          best_diamond_rd_cost = rd_cost;
          best_diamond_mv = cand_mv;
        }
    }

    if *center_mv_cost <= best_diamond_rd_cost {
      if diamond_radius == diamond_radius_end {
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
  po: PlaneOffset, p_org: &Plane<T>, p_ref: &Plane<T>, bit_depth: usize,
  pmv: [MotionVector; 2], lambda: u32,
  mvx_min: isize, mvx_max: isize, mvy_min: isize, mvy_max: isize,
  blk_w: usize, blk_h: usize,
  cand_mv: MotionVector, tmp_plane_opt: &mut Option<Plane<T>>,
  ref_frame: usize) -> u64
{
  if (cand_mv.col as isize) < mvx_min || (cand_mv.col as isize) > mvx_max {
    return std::u64::MAX;
  }
  if (cand_mv.row as isize) < mvy_min || (cand_mv.row as isize) > mvy_max {
    return std::u64::MAX;
  }

  let plane_org = p_org.slice(po);

  if let Some(ref mut tmp_plane) = tmp_plane_opt {
    let mut tmp_slice = &mut tmp_plane.mut_slice(PlaneOffset { x: 0, y: 0 });
    PredictionMode::NEWMV.predict_inter(
      fi,
      0,
      po,
      &mut tmp_slice,
      blk_w,
      blk_h,
      [ref_frame, NONE_FRAME],
      [cand_mv, MotionVector { row: 0, col: 0 }]
    );
    let plane_ref = tmp_plane.slice(PlaneOffset { x: 0, y: 0 });
    compute_mv_rd_cost(
      fi, pmv, lambda, bit_depth, blk_w, blk_h, cand_mv,
      &plane_org, &plane_ref
    )
  } else {
    // Full pixel motion vector
    let plane_ref = p_ref.slice(PlaneOffset {
      x: po.x + (cand_mv.col / 8) as isize,
      y: po.y + (cand_mv.row / 8) as isize
    });
    compute_mv_rd_cost(
      fi, pmv, lambda, bit_depth, blk_w, blk_h, cand_mv,
      &plane_org, &plane_ref
    )
  }
}

fn compute_mv_rd_cost<T: Pixel>(
  fi: &FrameInvariants<T>,
  pmv: [MotionVector; 2], lambda: u32,
  bit_depth: usize, blk_w: usize, blk_h: usize, cand_mv: MotionVector,
  plane_org: &PlaneSlice<T>, plane_ref: &PlaneSlice<T>
) -> u64
{
  let sad = get_sad(&plane_org, &plane_ref, blk_h, blk_w, bit_depth);

  let rate1 = get_mv_rate(cand_mv, pmv[0], fi.allow_high_precision_mv);
  let rate2 = get_mv_rate(cand_mv, pmv[1], fi.allow_high_precision_mv);
  let rate = rate1.min(rate2 + 1);

  256 * sad as u64 + rate as u64 * lambda as u64
}

fn telescopic_subpel_search<T: Pixel>(
  fi: &FrameInvariants<T>, fs: &FrameState<T>, po: PlaneOffset,
  lambda: u32, ref_frame: usize, pmv: [MotionVector; 2],
  mvx_min: isize, mvx_max: isize, mvy_min: isize, mvy_max: isize,
  blk_w: usize, blk_h: usize,
  best_mv: &mut MotionVector, lowest_cost: &mut u64
) {
  let mode = PredictionMode::NEWMV;

  let mut steps = vec![8, 4, 2];
  if fi.allow_high_precision_mv {
    steps.push(1);
  }

  let mut tmp_plane = Plane::new(blk_w, blk_h, 0, 0, 0, 0);

  for step in steps {
    let center_mv_h = *best_mv;
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
            &mut tmp_plane.mut_slice(PlaneOffset { x: 0, y: 0 });

          mode.predict_inter(
            fi,
            0,
            po,
            tmp_slice,
            blk_w,
            blk_h,
            [ref_frame, NONE_FRAME],
            [cand_mv, MotionVector { row: 0, col: 0 }]
          );
        }

        let plane_org = fs.input.planes[0].slice(po);
        let plane_ref = tmp_plane.slice(PlaneOffset { x: 0, y: 0 });

        let sad = get_sad(&plane_org, &plane_ref, blk_h, blk_w, fi.sequence.bit_depth);

        let rate1 = get_mv_rate(cand_mv, pmv[0], fi.allow_high_precision_mv);
        let rate2 = get_mv_rate(cand_mv, pmv[1], fi.allow_high_precision_mv);
        let rate = rate1.min(rate2 + 1);
        let cost = 256 * sad as u64 + rate as u64 * lambda as u64;

        if cost < *lowest_cost {
          *lowest_cost = cost;
          *best_mv = cand_mv;
        }
      }
    }
  }
}

fn full_search<T: Pixel>(
  x_lo: isize, x_hi: isize, y_lo: isize, y_hi: isize, blk_h: usize,
  blk_w: usize, p_org: &Plane<T>, p_ref: &Plane<T>, best_mv: &mut MotionVector,
  lowest_cost: &mut u64, po: PlaneOffset, step: usize, bit_depth: usize,
  lambda: u32, pmv: [MotionVector; 2], allow_high_precision_mv: bool
) {
    let search_range_y = (y_lo..=y_hi).step_by(step);
    let search_range_x = (x_lo..=x_hi).step_by(step);
    let search_area = search_range_y.flat_map(|y| { search_range_x.clone().map(move |x| (y, x)) });

    let (cost, mv) = search_area.map(|(y, x)| {
      let plane_org = p_org.slice(po);
      let plane_ref = p_ref.slice(PlaneOffset { x, y });

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
fn adjust_bo<T: Pixel>(bo: BlockOffset, fi: &FrameInvariants<T>, blk_w: usize, blk_h: usize) -> BlockOffset {
  BlockOffset {
    x: (bo.x as isize).min(fi.w_in_b as isize - blk_w as isize / 4).max(0) as usize,
    y: (bo.y as isize).min(fi.h_in_b as isize - blk_h as isize / 4).max(0) as usize
  }
}

#[inline(always)]
fn get_mv_rate(a: MotionVector, b: MotionVector, allow_high_precision_mv: bool) -> u32 {
  #[inline(always)]
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
  bo: BlockOffset
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
    let (mvx_min, mvx_max, mvy_min, mvy_max) = get_mv_range(fi.w_in_b, fi.h_in_b, bo_adj, blk_w, blk_h);
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
      po,
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

#[cfg(test)]
pub mod test {
  use super::*;
  use crate::partition::BlockSize;
  use crate::partition::BlockSize::*;

  // Generate plane data for get_sad_same()
  fn setup_sad<T: Pixel>() -> (Plane<T>, Plane<T>) {
    let mut input_plane = Plane::new(640, 480, 0, 0, 128 + 8, 128 + 8);
    let mut rec_plane = input_plane.clone();
    // Make the test pattern robust to data alignment
    let xpad_off = (input_plane.cfg.xorigin - input_plane.cfg.xpad) as i32 - 8i32;

    for (i, row) in input_plane.data.chunks_mut(input_plane.cfg.stride).enumerate() {
      for (j, pixel) in row.into_iter().enumerate() {
        let val = (j + i) as i32 - xpad_off & 255i32;
        assert!(val >= u8::min_value().into() &&
            val <= u8::max_value().into());
        *pixel = T::cast_from(val);
      }
    }

    for (i, row) in rec_plane.data.chunks_mut(rec_plane.cfg.stride).enumerate() {
      for (j, pixel) in row.into_iter().enumerate() {
        let val = j as i32 - i as i32 - xpad_off & 255i32;
        assert!(val >= u8::min_value().into() &&
            val <= u8::max_value().into());
        *pixel = T::cast_from(val);
      }
    }

    (input_plane, rec_plane)
  }

  // Regression and validation test for SAD computation
  fn get_sad_same_inner<T: Pixel>() {
    let blocks: Vec<(BlockSize, u32)> = vec![
      (BLOCK_4X4, 1912),
      (BLOCK_4X8, 4296),
      (BLOCK_8X4, 3496),
      (BLOCK_8X8, 7824),
      (BLOCK_8X16, 16592),
      (BLOCK_16X8, 14416),
      (BLOCK_16X16, 31136),
      (BLOCK_16X32, 60064),
      (BLOCK_32X16, 59552),
      (BLOCK_32X32, 120128),
      (BLOCK_32X64, 186688),
      (BLOCK_64X32, 250176),
      (BLOCK_64X64, 438912),
      (BLOCK_64X128, 654272),
      (BLOCK_128X64, 1016768),
      (BLOCK_128X128, 1689792),
      (BLOCK_4X16, 8680),
      (BLOCK_16X4, 6664),
      (BLOCK_8X32, 31056),
      (BLOCK_32X8, 27600),
      (BLOCK_16X64, 93344),
      (BLOCK_64X16, 116384),
    ];

    let bit_depth: usize = 8;
    let (input_plane, rec_plane) = setup_sad::<T>();

    for block in blocks {
      let bsw = block.0.width();
      let bsh = block.0.height();
      let po = PlaneOffset { x: 32, y: 40 };

      let mut input_slice = input_plane.slice(po);
      let mut rec_slice = rec_plane.slice(po);

      assert_eq!(
        block.1,
        get_sad(&mut input_slice, &mut rec_slice, bsh, bsw, bit_depth)
      );
    }
  }

  #[test]
  fn get_sad_same_u8() {
    get_sad_same_inner::<u8>();
  }

  #[test]
  fn get_sad_same_u16() {
    get_sad_same_inner::<u16>();
  }
}
