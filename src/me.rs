// Copyright (c) 2017-2020, The rav1e contributors. All rights reserved
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

use crate::util::ILog;
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

#[derive(Debug, Copy, Clone)]
pub struct MVSearchResult {
  mv: MotionVector,
  cost: u64,
}

const fn get_mv_range(
  w_in_b: usize, h_in_b: usize, bo: PlaneBlockOffset, blk_w: usize,
  blk_h: usize,
) -> (isize, isize, isize, isize) {
  let border_w = 128 + blk_w as isize * 8;
  let border_h = 128 + blk_h as isize * 8;
  let mvx_min = -(bo.0.x as isize) * (8 * MI_SIZE) as isize - border_w;
  let mvx_max = ((w_in_b - bo.0.x) as isize - (blk_w / MI_SIZE) as isize)
    * (8 * MI_SIZE) as isize
    + border_w;
  let mvy_min = -(bo.0.y as isize) * (8 * MI_SIZE) as isize - border_h;
  let mvy_max = ((h_in_b - bo.0.y) as isize - (blk_h / MI_SIZE) as isize)
    * (8 * MI_SIZE) as isize
    + border_h;

  (mvx_min, mvx_max, mvy_min, mvy_max)
}

pub fn get_subset_predictors<T: Pixel>(
  tile_bo: TileBlockOffset, cmvs: ArrayVec<[MotionVector; 7]>,
  tile_mvs: &TileMotionVectors<'_>, frame_ref_opt: Option<&ReferenceFrame<T>>,
  ref_frame_id: usize, bsize: BlockSize,
) -> ArrayVec<[MotionVector; 17]> {
  let mut predictors = ArrayVec::<[_; 17]>::new();
  let w = bsize.width_mi();
  let h = bsize.height_mi();

  // Add a candidate predictor, aligning to fullpel and filtering out zero mvs.
  let add_cand = |predictors: &mut ArrayVec<[MotionVector; 17]>,
                  cand_mv: MotionVector| {
    let cand_mv = cand_mv.quantize_to_fullpel();
    if !cand_mv.is_zero() {
      predictors.push(cand_mv)
    }
  };

  // Zero motion vector, don't use add_cand since it skips zero vectors.
  predictors.push(MotionVector::default());

  // Coarse motion estimation.
  for mv in cmvs {
    add_cand(&mut predictors, mv);
  }

  // Get predictors from the same frame.

  let clipped_half_w = (w >> 1).min(tile_mvs.cols() - 1 - tile_bo.0.x);
  let clipped_half_h = (h >> 1).min(tile_mvs.rows() - 1 - tile_bo.0.y);

  // Sample the center of the current block.
  add_cand(
    &mut predictors,
    tile_mvs[tile_bo.0.y + clipped_half_h][tile_bo.0.x + clipped_half_w],
  );

  // Sample the middle of all blocks bordering this one.
  // Note: If motion vectors haven't been precomputed to a given blocksize, then
  // the right and bottom edges will be duplicates of the center predictor when
  // processing in raster order.

  // left
  if tile_bo.0.x > 0 {
    add_cand(
      &mut predictors,
      tile_mvs[tile_bo.0.y + clipped_half_h][tile_bo.0.x - 1],
    );
  }
  // top
  if tile_bo.0.y > 0 {
    add_cand(
      &mut predictors,
      tile_mvs[tile_bo.0.y - 1][tile_bo.0.x + clipped_half_w],
    );
  }
  // right
  if tile_mvs.cols() > w && tile_bo.0.x < tile_mvs.cols() - w {
    add_cand(
      &mut predictors,
      tile_mvs[tile_bo.0.y + clipped_half_h][tile_bo.0.x + w],
    );
  }
  // bottom
  if tile_mvs.rows() > h && tile_bo.0.y < tile_mvs.rows() - h {
    add_cand(
      &mut predictors,
      tile_mvs[tile_bo.0.y + h][tile_bo.0.x + clipped_half_w],
    );
  }

  // EPZS subset C predictors.
  // Sample the middle of bordering side of the left, right, top and bottom
  // blocks of the previous frame.
  // Sample the middle of this block in the previous frame.

  if let Some(frame_ref) = frame_ref_opt {
    let prev_frame_mvs = &frame_ref.frame_mvs[ref_frame_id];

    let frame_bo = PlaneBlockOffset(BlockOffset {
      x: tile_mvs.x() + tile_bo.0.x,
      y: tile_mvs.y() + tile_bo.0.y,
    });
    let clipped_half_w = (w >> 1).min(prev_frame_mvs.cols - 1 - frame_bo.0.x);
    let clipped_half_h = (h >> 1).min(prev_frame_mvs.rows - 1 - frame_bo.0.y);

    if frame_bo.0.x > 0 {
      let left =
        prev_frame_mvs[frame_bo.0.y + clipped_half_h][frame_bo.0.x - 1];
      add_cand(&mut predictors, left);
    }
    if frame_bo.0.y > 0 {
      let top =
        prev_frame_mvs[frame_bo.0.y - 1][frame_bo.0.x + clipped_half_w];
      add_cand(&mut predictors, top);
    }
    if prev_frame_mvs.cols > w && frame_bo.0.x < prev_frame_mvs.cols - w {
      let right =
        prev_frame_mvs[frame_bo.0.y + clipped_half_h][frame_bo.0.x + w];
      add_cand(&mut predictors, right);
    }
    if prev_frame_mvs.rows > h && frame_bo.0.y < prev_frame_mvs.rows - h {
      let bottom =
        prev_frame_mvs[frame_bo.0.y + h][frame_bo.0.x + clipped_half_w];
      add_cand(&mut predictors, bottom);
    }

    let previous = prev_frame_mvs[frame_bo.0.y + clipped_half_h]
      [frame_bo.0.x + clipped_half_w];
    add_cand(&mut predictors, previous);
  }

  predictors
}

pub fn motion_estimation<T: Pixel>(
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

      let po = frame_bo.to_luma_plane_offset();
      let area = Area::BlockStartingAt { bo: tile_bo.0 };
      let org_region: &PlaneRegion<T> =
        &ts.input_tile.planes[0].subregion(area);
      let p_ref: &Plane<T> = &rec.frame.planes[0];

      let mut best = full_pixel_me(
        fi,
        ts,
        org_region,
        p_ref,
        tile_bo,
        lambda,
        iter::once(cmv).collect(),
        pmv,
        mvx_min,
        mvx_max,
        mvy_min,
        mvy_max,
        bsize,
        ref_frame,
      );

      let use_satd: bool = fi.config.speed_settings.use_satd_subpel;
      if use_satd {
        best.cost = get_fullpel_mv_rd_cost(
          fi,
          po,
          org_region,
          p_ref,
          fi.sequence.bit_depth,
          pmv,
          lambda,
          use_satd,
          mvx_min,
          mvx_max,
          mvy_min,
          mvy_max,
          bsize,
          best.mv,
        );
      }

      sub_pixel_me(
        fi, po, org_region, p_ref, lambda, pmv, mvx_min, mvx_max, mvy_min,
        mvy_max, bsize, use_satd, &mut best, ref_frame,
      );

      best.mv
    }

    None => MotionVector::default(),
  }
}

pub fn estimate_motion_ss2<T: Pixel>(
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

    // Divide by 4 to account for subsampling, 0.125 is a fudge factor
    let lambda = (fi.me_lambda * 256.0 / 4.0 * 0.125) as u32;

    let best = me_ss2(
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
      ref_frame,
    );

    Some(MotionVector { row: best.mv.row * 2, col: best.mv.col * 2 })
  } else {
    None
  }
}

pub fn estimate_motion<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>, bsize: BlockSize,
  tile_bo: TileBlockOffset, pmvs: &[Option<MotionVector>], ref_frame: RefType,
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

    // 0.5 is a fudge factor
    let lambda = (fi.me_lambda * 256.0 * 0.5) as u32;

    let area = Area::BlockStartingAt { bo: tile_bo_adj.0 };
    let org_region: &PlaneRegion<T> = &ts.input_tile.planes[0].subregion(area);

    let MVSearchResult { mv: best_mv, .. } = full_pixel_me(
      fi,
      ts,
      org_region,
      &rec.frame.planes[0],
      tile_bo_adj,
      lambda,
      pmvs.iter().cloned().filter_map(identity).collect(),
      global_mv,
      mvx_min,
      mvx_max,
      mvy_min,
      mvy_max,
      bsize,
      ref_frame,
    );

    Some(MotionVector { row: best_mv.row, col: best_mv.col })
  } else {
    None
  }
}

fn full_pixel_me<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>,
  org_region: &PlaneRegion<T>, p_ref: &Plane<T>, tile_bo: TileBlockOffset,
  lambda: u32, cmvs: ArrayVec<[MotionVector; 7]>, pmv: [MotionVector; 2],
  mvx_min: isize, mvx_max: isize, mvy_min: isize, mvy_max: isize,
  bsize: BlockSize, ref_frame: RefType,
) -> MVSearchResult {
  let tile_mvs = &ts.mvs[ref_frame.to_index()].as_const();
  let frame_ref =
    fi.rec_buffer.frames[fi.ref_frames[0] as usize].as_ref().map(Arc::as_ref);
  let predictors = get_subset_predictors(
    tile_bo,
    cmvs,
    tile_mvs,
    frame_ref,
    ref_frame.to_index(),
    bsize,
  );

  let frame_bo = ts.to_frame_block_offset(tile_bo);
  let po = frame_bo.to_luma_plane_offset();
  fullpel_diamond_me_search(
    fi,
    po,
    org_region,
    p_ref,
    &predictors,
    fi.sequence.bit_depth,
    pmv,
    lambda,
    mvx_min,
    mvx_max,
    mvy_min,
    mvy_max,
    bsize,
  )
}

fn sub_pixel_me<T: Pixel>(
  fi: &FrameInvariants<T>, po: PlaneOffset, org_region: &PlaneRegion<T>,
  p_ref: &Plane<T>, lambda: u32, pmv: [MotionVector; 2], mvx_min: isize,
  mvx_max: isize, mvy_min: isize, mvy_max: isize, bsize: BlockSize,
  use_satd: bool, best: &mut MVSearchResult, ref_frame: RefType,
) {
  subpel_diamond_me_search(
    fi,
    po,
    org_region,
    p_ref,
    fi.sequence.bit_depth,
    pmv,
    lambda,
    mvx_min,
    mvx_max,
    mvy_min,
    mvy_max,
    bsize,
    use_satd,
    best,
    ref_frame,
  );
}

fn me_ss2<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>,
  pmvs: &[Option<MotionVector>; 3], tile_bo_adj: TileBlockOffset,
  rec: &ReferenceFrame<T>, global_mv: [MotionVector; 2], lambda: u32,
  mvx_min: isize, mvx_max: isize, mvy_min: isize, mvy_max: isize,
  bsize: BlockSize, ref_frame: RefType,
) -> MVSearchResult {
  let frame_bo_adj = ts.to_frame_block_offset(tile_bo_adj);
  let po = PlaneOffset {
    x: (frame_bo_adj.0.x as isize) << BLOCK_TO_PLANE_SHIFT >> 1,
    y: (frame_bo_adj.0.y as isize) << BLOCK_TO_PLANE_SHIFT >> 1,
  };
  let org_region =
    &ts.input_hres.region(Area::StartingAt { x: po.x, y: po.y });

  let tile_mvs = &ts.mvs[ref_frame.to_index()].as_const();
  let frame_ref =
    fi.rec_buffer.frames[fi.ref_frames[0] as usize].as_ref().map(Arc::as_ref);

  let mut predictors = get_subset_predictors::<T>(
    tile_bo_adj,
    pmvs.iter().cloned().filter_map(identity).collect(),
    tile_mvs,
    frame_ref,
    ref_frame.to_index(),
    bsize,
  );

  for predictor in &mut predictors {
    predictor.row >>= 1;
    predictor.col >>= 1;
  }

  fullpel_diamond_me_search(
    fi,
    po,
    org_region,
    &rec.input_hres,
    &predictors,
    fi.sequence.bit_depth,
    global_mv,
    lambda,
    mvx_min >> 1,
    mvx_max >> 1,
    mvy_min >> 1,
    mvy_max >> 1,
    BlockSize::from_width_and_height(bsize.width() >> 1, bsize.height() >> 1),
  )
}

fn get_best_predictor<T: Pixel>(
  fi: &FrameInvariants<T>, po: PlaneOffset, org_region: &PlaneRegion<T>,
  p_ref: &Plane<T>, predictors: &[MotionVector], bit_depth: usize,
  pmv: [MotionVector; 2], lambda: u32, mvx_min: isize, mvx_max: isize,
  mvy_min: isize, mvy_max: isize, bsize: BlockSize,
) -> MVSearchResult {
  let mut best: MVSearchResult =
    MVSearchResult { mv: MotionVector::default(), cost: u64::MAX };

  for &init_mv in predictors.iter() {
    let cost = get_fullpel_mv_rd_cost(
      fi, po, org_region, p_ref, bit_depth, pmv, lambda, false, mvx_min,
      mvx_max, mvy_min, mvy_max, bsize, init_mv,
    );

    if cost < best.cost {
      best.mv = init_mv;
      best.cost = cost;
    }
  }

  best
}

fn fullpel_diamond_me_search<T: Pixel>(
  fi: &FrameInvariants<T>, po: PlaneOffset, org_region: &PlaneRegion<T>,
  p_ref: &Plane<T>, predictors: &[MotionVector], bit_depth: usize,
  pmv: [MotionVector; 2], lambda: u32, mvx_min: isize, mvx_max: isize,
  mvy_min: isize, mvy_max: isize, bsize: BlockSize,
) -> MVSearchResult {
  let diamond_pattern = [(1i16, 0i16), (0, 1), (-1, 0), (0, -1)];
  let (mut diamond_radius, diamond_radius_end) = (4u8, 3u8);

  let mut center = get_best_predictor(
    fi, po, org_region, p_ref, predictors, bit_depth, pmv, lambda, mvx_min,
    mvx_max, mvy_min, mvy_max, bsize,
  );

  loop {
    let mut best_diamond: MVSearchResult =
      MVSearchResult { mv: MotionVector::default(), cost: u64::MAX };

    for p in diamond_pattern.iter() {
      let cand_mv = MotionVector {
        row: center.mv.row + (p.0 << diamond_radius),
        col: center.mv.col + (p.1 << diamond_radius),
      };

      let rd_cost = get_fullpel_mv_rd_cost(
        fi, po, org_region, p_ref, bit_depth, pmv, lambda, false, mvx_min,
        mvx_max, mvy_min, mvy_max, bsize, cand_mv,
      );

      if rd_cost < best_diamond.cost {
        best_diamond.mv = cand_mv;
        best_diamond.cost = rd_cost;
      }
    }

    if center.cost <= best_diamond.cost {
      if diamond_radius == diamond_radius_end {
        break;
      } else {
        diamond_radius -= 1;
      }
    } else {
      center = best_diamond;
    }
  }

  assert!(center.cost < std::u64::MAX);

  center
}

fn subpel_diamond_me_search<T: Pixel>(
  fi: &FrameInvariants<T>, po: PlaneOffset, org_region: &PlaneRegion<T>,
  _p_ref: &Plane<T>, bit_depth: usize, pmv: [MotionVector; 2], lambda: u32,
  mvx_min: isize, mvx_max: isize, mvy_min: isize, mvy_max: isize,
  bsize: BlockSize, use_satd: bool, center: &mut MVSearchResult,
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
  let (mut diamond_radius, diamond_radius_end, mut tmp_region) = {
    let rect = Rect { x: 0, y: 0, width: cfg.width, height: cfg.height };

    // start at 1/2 pel and end at 1/4 or 1/8 pel
    (
      2u8,
      if fi.allow_high_precision_mv { 0u8 } else { 1u8 },
      PlaneRegionMut::from_slice(&mut buf.data, &cfg, rect),
    )
  };

  loop {
    let mut best_diamond: MVSearchResult =
      MVSearchResult { mv: MotionVector::default(), cost: u64::MAX };

    for p in diamond_pattern.iter() {
      let cand_mv = MotionVector {
        row: center.mv.row + (p.0 << diamond_radius),
        col: center.mv.col + (p.1 << diamond_radius),
      };

      let rd_cost = get_subpel_mv_rd_cost(
        fi,
        po,
        org_region,
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
        &mut tmp_region,
        ref_frame,
      );

      if rd_cost < best_diamond.cost {
        best_diamond.mv = cand_mv;
        best_diamond.cost = rd_cost;
      }
    }

    if center.cost <= best_diamond.cost {
      if diamond_radius == diamond_radius_end {
        break;
      } else {
        diamond_radius -= 1;
      }
    } else {
      *center = best_diamond;
    }
  }

  assert!(center.cost < std::u64::MAX);
}

#[inline]
fn get_fullpel_mv_rd_cost<T: Pixel>(
  fi: &FrameInvariants<T>, po: PlaneOffset, org_region: &PlaneRegion<T>,
  p_ref: &Plane<T>, bit_depth: usize, pmv: [MotionVector; 2], lambda: u32,
  use_satd: bool, mvx_min: isize, mvx_max: isize, mvy_min: isize,
  mvy_max: isize, bsize: BlockSize, cand_mv: MotionVector,
) -> u64 {
  if (cand_mv.col as isize) < mvx_min
    || (cand_mv.col as isize) > mvx_max
    || (cand_mv.row as isize) < mvy_min
    || (cand_mv.row as isize) > mvy_max
  {
    return std::u64::MAX;
  }

  // Full pixel motion vector
  let plane_ref = p_ref.region(Area::StartingAt {
    x: po.x + (cand_mv.col / 8) as isize,
    y: po.y + (cand_mv.row / 8) as isize,
  });
  compute_mv_rd_cost(
    fi, pmv, lambda, use_satd, bit_depth, bsize, cand_mv, org_region,
    &plane_ref,
  )
}

fn get_subpel_mv_rd_cost<T: Pixel>(
  fi: &FrameInvariants<T>, po: PlaneOffset, org_region: &PlaneRegion<T>,
  bit_depth: usize, pmv: [MotionVector; 2], lambda: u32, use_satd: bool,
  mvx_min: isize, mvx_max: isize, mvy_min: isize, mvy_max: isize,
  bsize: BlockSize, cand_mv: MotionVector, tmp_region: &mut PlaneRegionMut<T>,
  ref_frame: RefType,
) -> u64 {
  if (cand_mv.col as isize) < mvx_min
    || (cand_mv.col as isize) > mvx_max
    || (cand_mv.row as isize) < mvy_min
    || (cand_mv.row as isize) > mvy_max
  {
    return std::u64::MAX;
  }

  let tile_rect = TileRect {
    x: 0,
    y: 0,
    width: tmp_region.plane_cfg.width,
    height: tmp_region.plane_cfg.height,
  };
  PredictionMode::NEWMV.predict_inter_single(
    fi,
    tile_rect,
    0,
    po,
    tmp_region,
    bsize.width(),
    bsize.height(),
    ref_frame,
    cand_mv,
  );
  let plane_ref = tmp_region.as_const();
  compute_mv_rd_cost(
    fi, pmv, lambda, use_satd, bit_depth, bsize, cand_mv, org_region,
    &plane_ref,
  )
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
    2 * d.abs().ilog() as u32
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
