// Copyright (c) 2017-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::context::{
  BlockOffset, PlaneBlockOffset, TileBlockOffset, BLOCK_TO_PLANE_SHIFT,
  MI_SIZE,
};
use crate::dist::*;
use crate::encoder::ReferenceFrame;
use crate::frame::*;
use crate::mc::MotionVector;
use crate::partition::*;
use crate::predict::PredictionMode;
use crate::tiling::*;
use crate::util::Pixel;
use crate::FrameInvariants;

use arrayvec::*;

use std::convert::identity;
use std::iter;
use std::ops::{Index, IndexMut};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct FrameMotionVectors {
  mvs: Box<[MotionVector]>,
  pub cols: usize,
  pub rows: usize,
}

impl FrameMotionVectors {
  pub fn new(cols: usize, rows: usize) -> Self {
    Self {
      // dynamic allocation: once per frame
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

const fn get_mv_range(
  w_in_b: usize, h_in_b: usize, bo: PlaneBlockOffset, blk_w: usize,
  blk_h: usize,
) -> (isize, isize, isize, isize) {
  let border_w = 128 + blk_w as isize * 8;
  let border_h = 128 + blk_h as isize * 8;
  let mvx_min = -(bo.0.x as isize) * (8 * MI_SIZE) as isize - border_w;
  let mvx_max = (w_in_b - bo.0.x - blk_w / MI_SIZE) as isize
    * (8 * MI_SIZE) as isize
    + border_w;
  let mvy_min = -(bo.0.y as isize) * (8 * MI_SIZE) as isize - border_h;
  let mvy_max = (h_in_b - bo.0.y - blk_h / MI_SIZE) as isize
    * (8 * MI_SIZE) as isize
    + border_h;

  (mvx_min, mvx_max, mvy_min, mvy_max)
}

pub fn get_subset_predictors<T: Pixel>(
  tile_bo: TileBlockOffset, cmvs: ArrayVec<[MotionVector; 7]>,
  tile_mvs: &TileMotionVectors<'_>, frame_ref_opt: Option<&ReferenceFrame<T>>,
  ref_frame_id: usize,
) -> ArrayVec<[MotionVector; 17]> {
  let mut predictors = ArrayVec::<[_; 17]>::new();

  // Zero motion vector
  predictors.push(MotionVector::default());

  // Coarse motion estimation.
  predictors.extend(cmvs.into_iter().map(MotionVector::quantize_to_fullpel));

  // EPZS subset A and B predictors.

  let mut median_preds = ArrayVec::<[_; 3]>::new();
  if tile_bo.0.x > 0 {
    let left = tile_mvs[tile_bo.0.y][tile_bo.0.x - 1];
    median_preds.push(left);
    if !left.is_zero() {
      predictors.push(left);
    }
  }
  if tile_bo.0.y > 0 {
    let top = tile_mvs[tile_bo.0.y - 1][tile_bo.0.x];
    median_preds.push(top);
    if !top.is_zero() {
      predictors.push(top);
    }

    if tile_bo.0.x < tile_mvs.cols() - 1 {
      let top_right = tile_mvs[tile_bo.0.y - 1][tile_bo.0.x + 1];
      median_preds.push(top_right);
      if !top_right.is_zero() {
        predictors.push(top_right);
      }
    }
  }

  if !median_preds.is_empty() {
    let mut median_mv = MotionVector::default();
    for mv in median_preds.iter() {
      median_mv = median_mv + *mv;
    }
    median_mv = median_mv / (median_preds.len() as i16);
    let median_mv_quant = median_mv.quantize_to_fullpel();
    if !median_mv_quant.is_zero() {
      predictors.push(median_mv_quant);
    }
  }

  // EPZS subset C predictors.

  if let Some(frame_ref) = frame_ref_opt {
    let prev_frame_mvs = &frame_ref.frame_mvs[ref_frame_id];

    let frame_bo = PlaneBlockOffset(BlockOffset {
      x: tile_mvs.x() + tile_bo.0.x,
      y: tile_mvs.y() + tile_bo.0.y,
    });
    if frame_bo.0.x > 0 {
      let left = prev_frame_mvs[frame_bo.0.y][frame_bo.0.x - 1];
      if !left.is_zero() {
        predictors.push(left);
      }
    }
    if frame_bo.0.y > 0 {
      let top = prev_frame_mvs[frame_bo.0.y - 1][frame_bo.0.x];
      if !top.is_zero() {
        predictors.push(top);
      }
    }
    if frame_bo.0.x < prev_frame_mvs.cols - 1 {
      let right = prev_frame_mvs[frame_bo.0.y][frame_bo.0.x + 1];
      if !right.is_zero() {
        predictors.push(right);
      }
    }
    if frame_bo.0.y < prev_frame_mvs.rows - 1 {
      let bottom = prev_frame_mvs[frame_bo.0.y + 1][frame_bo.0.x];
      if !bottom.is_zero() {
        predictors.push(bottom);
      }
    }

    let previous = prev_frame_mvs[frame_bo.0.y][frame_bo.0.x];
    if !previous.is_zero() {
      predictors.push(previous);
    }
  }

  predictors
}

pub trait MotionEstimation {
  fn full_pixel_me<T: Pixel>(
    fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>,
    rec: &ReferenceFrame<T>, tile_bo: TileBlockOffset, lambda: u32,
    cmvs: ArrayVec<[MotionVector; 7]>, pmv: [MotionVector; 2], mvx_min: isize,
    mvx_max: isize, mvy_min: isize, mvy_max: isize, bsize: BlockSize,
    best_mv: &mut MotionVector, lowest_cost: &mut u64, ref_frame: RefType,
  );

  fn sub_pixel_me<T: Pixel>(
    fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>,
    rec: &ReferenceFrame<T>, tile_bo: TileBlockOffset, lambda: u32,
    pmv: [MotionVector; 2], mvx_min: isize, mvx_max: isize, mvy_min: isize,
    mvy_max: isize, bsize: BlockSize, use_satd: bool,
    best_mv: &mut MotionVector, lowest_cost: &mut u64, ref_frame: RefType,
  );

  fn motion_estimation<T: Pixel>(
    fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>, bsize: BlockSize,
    tile_bo: TileBlockOffset, ref_frame: RefType, cmv: MotionVector,
    pmv: [MotionVector; 2],
  ) -> MotionVector {
    match fi.rec_buffer.frames[fi.ref_frames[ref_frame.to_index()] as usize] {
      Some(ref rec) => {
        let blk_w = bsize.width();
        let blk_h = bsize.height();
        let frame_bo = ts.to_frame_block_offset(tile_bo);
        let (mvx_min, mvx_max, mvy_min, mvy_max) =
          get_mv_range(fi.w_in_b, fi.h_in_b, frame_bo, blk_w, blk_h);

        // 0.5 is a fudge factor
        let lambda = (fi.me_lambda * 256.0 * 0.5) as u32;

        // Full-pixel motion estimation

        let mut lowest_cost = std::u64::MAX;
        let mut best_mv = MotionVector::default();

        Self::full_pixel_me(
          fi,
          ts,
          rec,
          tile_bo,
          lambda,
          iter::once(cmv).collect(),
          pmv,
          mvx_min,
          mvx_max,
          mvy_min,
          mvy_max,
          bsize,
          &mut best_mv,
          &mut lowest_cost,
          ref_frame,
        );

        let use_satd: bool = fi.config.speed_settings.use_satd_subpel;
        if use_satd {
          lowest_cost = get_mv_rd_cost(
            fi,
            frame_bo.to_luma_plane_offset(),
            &ts.input.planes[0],
            &rec.frame.planes[0],
            fi.sequence.bit_depth,
            pmv,
            lambda,
            use_satd,
            mvx_min,
            mvx_max,
            mvy_min,
            mvy_max,
            bsize,
            best_mv,
            None,
            ref_frame,
          );
        }

        Self::sub_pixel_me(
          fi,
          ts,
          rec,
          tile_bo,
          lambda,
          pmv,
          mvx_min,
          mvx_max,
          mvy_min,
          mvy_max,
          bsize,
          use_satd,
          &mut best_mv,
          &mut lowest_cost,
          ref_frame,
        );

        best_mv
      }

      None => MotionVector::default(),
    }
  }

  fn estimate_motion_ss2<T: Pixel>(
    fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>, bsize: BlockSize,
    tile_bo: TileBlockOffset, pmvs: &[Option<MotionVector>; 3],
    ref_frame: RefType,
  ) -> Option<MotionVector> {
    if let Some(ref rec) =
      fi.rec_buffer.frames[fi.ref_frames[ref_frame.to_index()] as usize]
    {
      let blk_w = bsize.width();
      let blk_h = bsize.height();
      let tile_bo_adj =
        adjust_bo(tile_bo, ts.mi_width, ts.mi_height, blk_w, blk_h);
      let frame_bo_adj = ts.to_frame_block_offset(tile_bo_adj);
      let (mvx_min, mvx_max, mvy_min, mvy_max) =
        get_mv_range(fi.w_in_b, fi.h_in_b, frame_bo_adj, blk_w, blk_h);

      let global_mv = [MotionVector { row: 0, col: 0 }; 2];

      let mut lowest_cost = std::u64::MAX;
      let mut best_mv = MotionVector::default();

      // Divide by 4 to account for subsampling, 0.125 is a fudge factor
      let lambda = (fi.me_lambda * 256.0 / 4.0 * 0.125) as u32;

      Self::me_ss2(
        fi,
        ts,
        pmvs,
        tile_bo_adj,
        rec,
        global_mv,
        lambda,
        mvx_min,
        mvx_max,
        mvy_min,
        mvy_max,
        bsize,
        &mut best_mv,
        &mut lowest_cost,
        ref_frame,
      );

      Some(MotionVector { row: best_mv.row * 2, col: best_mv.col * 2 })
    } else {
      None
    }
  }

  fn me_ss2<T: Pixel>(
    fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>,
    pmvs: &[Option<MotionVector>; 3], tile_bo_adj: TileBlockOffset,
    rec: &ReferenceFrame<T>, global_mv: [MotionVector; 2], lambda: u32,
    mvx_min: isize, mvx_max: isize, mvy_min: isize, mvy_max: isize,
    bsize: BlockSize, best_mv: &mut MotionVector, lowest_cost: &mut u64,
    ref_frame: RefType,
  );

  fn estimate_motion<T: Pixel>(
    fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>, bsize: BlockSize,
    tile_bo: TileBlockOffset, pmvs: &[Option<MotionVector>],
    ref_frame: RefType,
  ) -> Option<MotionVector> {
    debug_assert!(pmvs.len() <= 7);

    if let Some(ref rec) =
      fi.rec_buffer.frames[fi.ref_frames[ref_frame.to_index()] as usize]
    {
      let blk_w = bsize.width();
      let blk_h = bsize.height();
      let tile_bo_adj =
        adjust_bo(tile_bo, ts.mi_width, ts.mi_height, blk_w, blk_h);
      let frame_bo_adj = ts.to_frame_block_offset(tile_bo_adj);
      let (mvx_min, mvx_max, mvy_min, mvy_max) =
        get_mv_range(fi.w_in_b, fi.h_in_b, frame_bo_adj, blk_w, blk_h);

      let global_mv = [MotionVector { row: 0, col: 0 }; 2];

      let mut lowest_cost = std::u64::MAX;
      let mut best_mv = MotionVector::default();

      // 0.5 is a fudge factor
      let lambda = (fi.me_lambda * 256.0 * 0.5) as u32;

      Self::full_pixel_me(
        fi,
        ts,
        rec,
        tile_bo_adj,
        lambda,
        pmvs.iter().cloned().filter_map(identity).collect(),
        global_mv,
        mvx_min,
        mvx_max,
        mvy_min,
        mvy_max,
        bsize,
        &mut best_mv,
        &mut lowest_cost,
        ref_frame,
      );

      Some(MotionVector { row: best_mv.row, col: best_mv.col })
    } else {
      None
    }
  }
}

pub struct DiamondSearch {}
pub struct FullSearch {}

impl MotionEstimation for DiamondSearch {
  fn full_pixel_me<T: Pixel>(
    fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>,
    rec: &ReferenceFrame<T>, tile_bo: TileBlockOffset, lambda: u32,
    cmvs: ArrayVec<[MotionVector; 7]>, pmv: [MotionVector; 2], mvx_min: isize,
    mvx_max: isize, mvy_min: isize, mvy_max: isize, bsize: BlockSize,
    best_mv: &mut MotionVector, lowest_cost: &mut u64, ref_frame: RefType,
  ) {
    let tile_mvs = &ts.mvs[ref_frame.to_index()].as_const();
    let frame_ref = fi.rec_buffer.frames[fi.ref_frames[0] as usize]
      .as_ref()
      .map(Arc::as_ref);
    let predictors = get_subset_predictors(
      tile_bo,
      cmvs,
      tile_mvs,
      frame_ref,
      ref_frame.to_index(),
    );

    let frame_bo = ts.to_frame_block_offset(tile_bo);
    diamond_me_search(
      fi,
      frame_bo.to_luma_plane_offset(),
      &ts.input.planes[0],
      &rec.frame.planes[0],
      &predictors,
      fi.sequence.bit_depth,
      pmv,
      lambda,
      mvx_min,
      mvx_max,
      mvy_min,
      mvy_max,
      bsize,
      false,
      best_mv,
      lowest_cost,
      false,
      ref_frame,
    );
  }

  fn sub_pixel_me<T: Pixel>(
    fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>,
    rec: &ReferenceFrame<T>, tile_bo: TileBlockOffset, lambda: u32,
    pmv: [MotionVector; 2], mvx_min: isize, mvx_max: isize, mvy_min: isize,
    mvy_max: isize, bsize: BlockSize, use_satd: bool,
    best_mv: &mut MotionVector, lowest_cost: &mut u64, ref_frame: RefType,
  ) {
    let predictors = [*best_mv; 1];
    let frame_bo = ts.to_frame_block_offset(tile_bo);
    diamond_me_search(
      fi,
      frame_bo.to_luma_plane_offset(),
      &ts.input.planes[0],
      &rec.frame.planes[0],
      &predictors,
      fi.sequence.bit_depth,
      pmv,
      lambda,
      mvx_min,
      mvx_max,
      mvy_min,
      mvy_max,
      bsize,
      use_satd,
      best_mv,
      lowest_cost,
      true,
      ref_frame,
    );
  }

  fn me_ss2<T: Pixel>(
    fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>,
    pmvs: &[Option<MotionVector>; 3], tile_bo_adj: TileBlockOffset,
    rec: &ReferenceFrame<T>, global_mv: [MotionVector; 2], lambda: u32,
    mvx_min: isize, mvx_max: isize, mvy_min: isize, mvy_max: isize,
    bsize: BlockSize, best_mv: &mut MotionVector, lowest_cost: &mut u64,
    ref_frame: RefType,
  ) {
    let frame_bo_adj = ts.to_frame_block_offset(tile_bo_adj);
    let frame_po = PlaneOffset {
      x: (frame_bo_adj.0.x as isize) << BLOCK_TO_PLANE_SHIFT >> 1,
      y: (frame_bo_adj.0.y as isize) << BLOCK_TO_PLANE_SHIFT >> 1,
    };

    let tile_mvs = &ts.mvs[ref_frame.to_index()].as_const();
    let frame_ref = fi.rec_buffer.frames[fi.ref_frames[0] as usize]
      .as_ref()
      .map(Arc::as_ref);

    let mut predictors = get_subset_predictors::<T>(
      tile_bo_adj,
      pmvs.iter().cloned().filter_map(identity).collect(),
      tile_mvs,
      frame_ref,
      ref_frame.to_index(),
    );

    for predictor in &mut predictors {
      predictor.row >>= 1;
      predictor.col >>= 1;
    }

    diamond_me_search(
      fi,
      frame_po,
      ts.input_hres,
      &rec.input_hres,
      &predictors,
      fi.sequence.bit_depth,
      global_mv,
      lambda,
      mvx_min >> 1,
      mvx_max >> 1,
      mvy_min >> 1,
      mvy_max >> 1,
      BlockSize::from_width_and_height(
        bsize.width() >> 1,
        bsize.height() >> 1,
      ),
      false,
      best_mv,
      lowest_cost,
      false,
      ref_frame,
    );
  }
}

impl MotionEstimation for FullSearch {
  fn full_pixel_me<T: Pixel>(
    fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>,
    rec: &ReferenceFrame<T>, tile_bo: TileBlockOffset, lambda: u32,
    cmvs: ArrayVec<[MotionVector; 7]>, pmv: [MotionVector; 2], mvx_min: isize,
    mvx_max: isize, mvy_min: isize, mvy_max: isize, bsize: BlockSize,
    best_mv: &mut MotionVector, lowest_cost: &mut u64, _ref_frame: RefType,
  ) {
    let frame_bo = ts.to_frame_block_offset(tile_bo);
    let frame_po = frame_bo.to_luma_plane_offset();
    let range = 16;
    for cmv in cmvs.into_iter() {
      let x_lo = frame_po.x
        + ((-range + (cmv.col / 8) as isize)
          .max(mvx_min / 8)
          .min(mvx_max / 8));
      let x_hi = frame_po.x
        + ((range + (cmv.col / 8) as isize).max(mvx_min / 8).min(mvx_max / 8));
      let y_lo = frame_po.y
        + ((-range + (cmv.row / 8) as isize)
          .max(mvy_min / 8)
          .min(mvy_max / 8));
      let y_hi = frame_po.y
        + ((range + (cmv.row / 8) as isize).max(mvy_min / 8).min(mvy_max / 8));

      full_search(
        fi,
        x_lo,
        x_hi,
        y_lo,
        y_hi,
        bsize,
        &ts.input.planes[0],
        &rec.frame.planes[0],
        best_mv,
        lowest_cost,
        frame_po,
        2,
        lambda,
        pmv,
      );
    }
  }

  fn sub_pixel_me<T: Pixel>(
    fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>,
    _rec: &ReferenceFrame<T>, tile_bo: TileBlockOffset, lambda: u32,
    pmv: [MotionVector; 2], mvx_min: isize, mvx_max: isize, mvy_min: isize,
    mvy_max: isize, bsize: BlockSize, use_satd: bool,
    best_mv: &mut MotionVector, lowest_cost: &mut u64, ref_frame: RefType,
  ) {
    let frame_bo = ts.to_frame_block_offset(tile_bo);
    telescopic_subpel_search(
      fi,
      ts,
      frame_bo.to_luma_plane_offset(),
      lambda,
      ref_frame,
      pmv,
      mvx_min,
      mvx_max,
      mvy_min,
      mvy_max,
      bsize,
      use_satd,
      best_mv,
      lowest_cost,
    );
  }

  fn me_ss2<T: Pixel>(
    fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>,
    pmvs: &[Option<MotionVector>; 3], tile_bo_adj: TileBlockOffset,
    rec: &ReferenceFrame<T>, _global_mv: [MotionVector; 2], lambda: u32,
    mvx_min: isize, mvx_max: isize, mvy_min: isize, mvy_max: isize,
    bsize: BlockSize, best_mv: &mut MotionVector, lowest_cost: &mut u64,
    _ref_frame: RefType,
  ) {
    let frame_bo_adj = ts.to_frame_block_offset(tile_bo_adj);
    let frame_po = PlaneOffset {
      x: (frame_bo_adj.0.x as isize) << BLOCK_TO_PLANE_SHIFT >> 1,
      y: (frame_bo_adj.0.y as isize) << BLOCK_TO_PLANE_SHIFT >> 1,
    };
    let range = 16;
    for omv in pmvs.iter() {
      if let Some(pmv) = omv {
        let x_lo = frame_po.x
          + (((pmv.col as isize / 8 - range)
            .max(mvx_min / 8)
            .min(mvx_max / 8))
            >> 1);
        let x_hi = frame_po.x
          + (((pmv.col as isize / 8 + range)
            .max(mvx_min / 8)
            .min(mvx_max / 8))
            >> 1);
        let y_lo = frame_po.y
          + (((pmv.row as isize / 8 - range)
            .max(mvy_min / 8)
            .min(mvy_max / 8))
            >> 1);
        let y_hi = frame_po.y
          + (((pmv.row as isize / 8 + range)
            .max(mvy_min / 8)
            .min(mvy_max / 8))
            >> 1);
        full_search(
          fi,
          x_lo,
          x_hi,
          y_lo,
          y_hi,
          BlockSize::from_width_and_height(
            bsize.width() >> 1,
            bsize.height() >> 1,
          ),
          ts.input_hres,
          &rec.input_hres,
          best_mv,
          lowest_cost,
          frame_po,
          1,
          lambda,
          [MotionVector::default(); 2],
        );
      }
    }
  }
}

fn get_best_predictor<T: Pixel>(
  fi: &FrameInvariants<T>, po: PlaneOffset, p_org: &Plane<T>,
  p_ref: &Plane<T>, predictors: &[MotionVector], bit_depth: usize,
  pmv: [MotionVector; 2], lambda: u32, use_satd: bool, mvx_min: isize,
  mvx_max: isize, mvy_min: isize, mvy_max: isize, bsize: BlockSize,
  center_mv: &mut MotionVector, center_mv_cost: &mut u64,
  tmp_region_opt: &mut Option<PlaneRegionMut<T>>, ref_frame: RefType,
) {
  *center_mv = MotionVector::default();
  *center_mv_cost = std::u64::MAX;

  for &init_mv in predictors.iter() {
    let cost = get_mv_rd_cost(
      fi,
      po,
      p_org,
      p_ref,
      bit_depth,
      pmv,
      lambda,
      use_satd,
      mvx_min,
      mvx_max,
      mvy_min,
      mvy_max,
      bsize,
      init_mv,
      tmp_region_opt.as_mut(),
      ref_frame,
    );

    if cost < *center_mv_cost {
      *center_mv = init_mv;
      *center_mv_cost = cost;
    }
  }
}

fn diamond_me_search<T: Pixel>(
  fi: &FrameInvariants<T>, po: PlaneOffset, p_org: &Plane<T>,
  p_ref: &Plane<T>, predictors: &[MotionVector], bit_depth: usize,
  pmv: [MotionVector; 2], lambda: u32, mvx_min: isize, mvx_max: isize,
  mvy_min: isize, mvy_max: isize, bsize: BlockSize, use_satd: bool,
  center_mv: &mut MotionVector, center_mv_cost: &mut u64, subpixel: bool,
  ref_frame: RefType,
) {
  use crate::util::Aligned;

  let cfg = PlaneConfig::new(
    bsize.width(),
    bsize.height(),
    0,
    0,
    0,
    0,
    std::mem::size_of::<T>(),
  );

  let mut buf: Aligned<[T; 128 * 128]> = Aligned::uninitialized();

  let diamond_pattern = [(1i16, 0i16), (0, 1), (-1, 0), (0, -1)];
  let (mut diamond_radius, diamond_radius_end, mut tmp_region_opt) = {
    if subpixel {
      let rect = Rect { x: 0, y: 0, width: cfg.width, height: cfg.height };

      // Sub-pixel motion estimation
      (
        4i16,
        if fi.allow_high_precision_mv { 1i16 } else { 2i16 },
        Some(PlaneRegionMut::from_slice(&mut buf.data, &cfg, rect)),
      )
    } else {
      // Full pixel motion estimation
      (16i16, 8i16, None)
    }
  };

  get_best_predictor(
    fi,
    po,
    p_org,
    p_ref,
    predictors,
    bit_depth,
    pmv,
    lambda,
    use_satd,
    mvx_min,
    mvx_max,
    mvy_min,
    mvy_max,
    bsize,
    center_mv,
    center_mv_cost,
    &mut tmp_region_opt,
    ref_frame,
  );

  loop {
    let mut best_diamond_rd_cost = std::u64::MAX;
    let mut best_diamond_mv = MotionVector::default();

    for p in diamond_pattern.iter() {
      let cand_mv = MotionVector {
        row: center_mv.row + diamond_radius * p.0,
        col: center_mv.col + diamond_radius * p.1,
      };

      let rd_cost = get_mv_rd_cost(
        fi,
        po,
        p_org,
        p_ref,
        bit_depth,
        pmv,
        lambda,
        use_satd,
        mvx_min,
        mvx_max,
        mvy_min,
        mvy_max,
        bsize,
        cand_mv,
        tmp_region_opt.as_mut(),
        ref_frame,
      );

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
    } else {
      *center_mv = best_diamond_mv;
      *center_mv_cost = best_diamond_rd_cost;
    }
  }

  assert!(*center_mv_cost < std::u64::MAX);
}

fn get_mv_rd_cost<T: Pixel>(
  fi: &FrameInvariants<T>, po: PlaneOffset, p_org: &Plane<T>,
  p_ref: &Plane<T>, bit_depth: usize, pmv: [MotionVector; 2], lambda: u32,
  use_satd: bool, mvx_min: isize, mvx_max: isize, mvy_min: isize,
  mvy_max: isize, bsize: BlockSize, cand_mv: MotionVector,
  tmp_region_opt: Option<&mut PlaneRegionMut<T>>, ref_frame: RefType,
) -> u64 {
  if (cand_mv.col as isize) < mvx_min || (cand_mv.col as isize) > mvx_max {
    return std::u64::MAX;
  }
  if (cand_mv.row as isize) < mvy_min || (cand_mv.row as isize) > mvy_max {
    return std::u64::MAX;
  }

  let plane_org = p_org.region(Area::StartingAt { x: po.x, y: po.y });

  if let Some(region) = tmp_region_opt {
    let tile_rect = TileRect {
      x: 0,
      y: 0,
      width: region.plane_cfg.width,
      height: region.plane_cfg.height,
    };
    PredictionMode::NEWMV.predict_inter_single(
      fi,
      tile_rect,
      0,
      po,
      region,
      bsize.width(),
      bsize.height(),
      ref_frame,
      cand_mv,
    );
    let plane_ref = region.as_const();
    compute_mv_rd_cost(
      fi, pmv, lambda, use_satd, bit_depth, bsize, cand_mv, &plane_org,
      &plane_ref,
    )
  } else {
    // Full pixel motion vector
    let plane_ref = p_ref.region(Area::StartingAt {
      x: po.x + (cand_mv.col / 8) as isize,
      y: po.y + (cand_mv.row / 8) as isize,
    });
    compute_mv_rd_cost(
      fi, pmv, lambda, use_satd, bit_depth, bsize, cand_mv, &plane_org,
      &plane_ref,
    )
  }
}

#[inline(always)]
fn compute_mv_rd_cost<T: Pixel>(
  fi: &FrameInvariants<T>, pmv: [MotionVector; 2], lambda: u32,
  use_satd: bool, bit_depth: usize, bsize: BlockSize, cand_mv: MotionVector,
  plane_org: &PlaneRegion<'_, T>, plane_ref: &PlaneRegion<'_, T>,
) -> u64 {
  let sad = if use_satd {
    get_satd(plane_org, plane_ref, bsize, bit_depth, fi.cpu_feature_level)
  } else {
    get_sad(plane_org, plane_ref, bsize, bit_depth, fi.cpu_feature_level)
  };

  let rate1 = get_mv_rate(cand_mv, pmv[0], fi.allow_high_precision_mv);
  let rate2 = get_mv_rate(cand_mv, pmv[1], fi.allow_high_precision_mv);
  let rate = rate1.min(rate2 + 1);

  256 * sad as u64 + rate as u64 * lambda as u64
}

fn telescopic_subpel_search<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>, po: PlaneOffset,
  lambda: u32, ref_frame: RefType, pmv: [MotionVector; 2], mvx_min: isize,
  mvx_max: isize, mvy_min: isize, mvy_max: isize, bsize: BlockSize,
  use_satd: bool, best_mv: &mut MotionVector, lowest_cost: &mut u64,
) {
  let mode = PredictionMode::NEWMV;
  let blk_w = bsize.width();
  let blk_h = bsize.height();

  let steps: &[_] =
    if fi.allow_high_precision_mv { &[8, 4, 2, 1] } else { &[8, 4, 2] };

  let mut tmp_plane = Plane::new(blk_w, blk_h, 0, 0, 0, 0);
  let tile_rect = TileRect {
    x: 0,
    y: 0,
    width: tmp_plane.cfg.width,
    height: tmp_plane.cfg.height,
  };

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
          col: center_mv_h.col + step * (j as i16 - 1),
        };

        if (cand_mv.col as isize) < mvx_min || (cand_mv.col as isize) > mvx_max
        {
          continue;
        }
        if (cand_mv.row as isize) < mvy_min || (cand_mv.row as isize) > mvy_max
        {
          continue;
        }

        {
          mode.predict_inter_single(
            fi,
            tile_rect,
            0,
            po,
            &mut tmp_plane.as_region_mut(),
            blk_w,
            blk_h,
            ref_frame,
            cand_mv,
          );
        }

        let plane_org =
          ts.input.planes[0].region(Area::StartingAt { x: po.x, y: po.y });
        let plane_ref = tmp_plane.as_region();

        let cost = compute_mv_rd_cost(
          fi,
          pmv,
          lambda,
          use_satd,
          fi.sequence.bit_depth,
          bsize,
          cand_mv,
          &plane_org,
          &plane_ref,
        );

        if cost < *lowest_cost {
          *lowest_cost = cost;
          *best_mv = cand_mv;
        }
      }
    }
  }
}

fn full_search<T: Pixel>(
  fi: &FrameInvariants<T>, x_lo: isize, x_hi: isize, y_lo: isize, y_hi: isize,
  bsize: BlockSize, p_org: &Plane<T>, p_ref: &Plane<T>,
  best_mv: &mut MotionVector, lowest_cost: &mut u64, po: PlaneOffset,
  step: usize, lambda: u32, pmv: [MotionVector; 2],
) {
  let blk_w = bsize.width();
  let blk_h = bsize.height();
  let plane_org = p_org.region(Area::StartingAt { x: po.x, y: po.y });
  let search_region = p_ref.region(Area::Rect {
    x: x_lo,
    y: y_lo,
    width: (x_hi - x_lo) as usize + blk_w,
    height: (y_hi - y_lo) as usize + blk_h,
  });

  // Select rectangular regions within search region with vert+horz windows
  for vert_window in search_region.vert_windows(blk_h).step_by(step) {
    for ref_window in vert_window.horz_windows(blk_w).step_by(step) {
      let &Rect { x, y, .. } = ref_window.rect();

      let mv = MotionVector {
        row: 8 * (y as i16 - po.y as i16),
        col: 8 * (x as i16 - po.x as i16),
      };

      let cost = compute_mv_rd_cost(
        fi,
        pmv,
        lambda,
        false,
        fi.sequence.bit_depth,
        bsize,
        mv,
        &plane_org,
        &ref_window,
      );

      if cost < *lowest_cost {
        *lowest_cost = cost;
        *best_mv = mv;
      }
    }
  }
}

// Adjust block offset such that entire block lies within boundaries
// Align to block width, to admit aligned SAD instructions
fn adjust_bo(
  bo: TileBlockOffset, mi_width: usize, mi_height: usize, blk_w: usize,
  blk_h: usize,
) -> TileBlockOffset {
  TileBlockOffset(BlockOffset {
    x: (bo.0.x as isize).min(mi_width as isize - blk_w as isize / 4).max(0)
      as usize
      & !(blk_w / 4 - 1),
    y: (bo.0.y as isize).min(mi_height as isize - blk_h as isize / 4).max(0)
      as usize,
  })
}

#[inline(always)]
fn get_mv_rate(
  a: MotionVector, b: MotionVector, allow_high_precision_mv: bool,
) -> u32 {
  #[inline(always)]
  fn diff_to_rate(diff: i16, allow_high_precision_mv: bool) -> u32 {
    let d = if allow_high_precision_mv { diff } else { diff >> 1 };
    if d == 0 {
      0
    } else {
      2 * (16 - d.abs().leading_zeros())
    }
  }

  diff_to_rate(a.row - b.row, allow_high_precision_mv)
    + diff_to_rate(a.col - b.col, allow_high_precision_mv)
}

pub fn estimate_motion_ss4<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>, bsize: BlockSize,
  ref_idx: usize, tile_bo: TileBlockOffset,
) -> Option<MotionVector> {
  if let Some(ref rec) = fi.rec_buffer.frames[ref_idx] {
    let blk_w = bsize.width();
    let blk_h = bsize.height();
    let tile_bo_adj =
      adjust_bo(tile_bo, ts.mi_width, ts.mi_height, blk_w, blk_h);
    let frame_bo_adj = ts.to_frame_block_offset(tile_bo_adj);
    let po = PlaneOffset {
      x: (frame_bo_adj.0.x as isize) << BLOCK_TO_PLANE_SHIFT >> 2,
      y: (frame_bo_adj.0.y as isize) << BLOCK_TO_PLANE_SHIFT >> 2,
    };

    let range_x = 192 * fi.me_range_scale as isize;
    let range_y = 64 * fi.me_range_scale as isize;
    let (mvx_min, mvx_max, mvy_min, mvy_max) =
      get_mv_range(fi.w_in_b, fi.h_in_b, frame_bo_adj, blk_w, blk_h);
    let x_lo = po.x + (((-range_x).max(mvx_min / 8)) >> 2);
    let x_hi = po.x + (((range_x).min(mvx_max / 8)) >> 2);
    let y_lo = po.y + (((-range_y).max(mvy_min / 8)) >> 2);
    let y_hi = po.y + (((range_y).min(mvy_max / 8)) >> 2);

    let mut lowest_cost = std::u64::MAX;
    let mut best_mv = MotionVector::default();

    // Divide by 16 to account for subsampling, 0.125 is a fudge factor
    let lambda = (fi.me_lambda * 256.0 / 16.0 * 0.125) as u32;

    full_search(
      fi,
      x_lo,
      x_hi,
      y_lo,
      y_hi,
      BlockSize::from_width_and_height(blk_w >> 2, blk_h >> 2),
      ts.input_qres,
      &rec.input_qres,
      &mut best_mv,
      &mut lowest_cost,
      po,
      1,
      lambda,
      [MotionVector::default(); 2],
    );

    Some(MotionVector { row: best_mv.row * 4, col: best_mv.col * 4 })
  } else {
    None
  }
}
