// Copyright (c) 2018-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::context::*;
use crate::partition::BlockSize;
use crate::quantize::{ac_q, select_ac_qi};
use crate::rdo::DistortionScale;
use crate::tiling::TileStateMut;
use crate::util::Pixel;
use crate::FrameInvariants;
use crate::FrameState;
use arrayvec::ArrayVec;

pub fn segmentation_optimize<T: Pixel>(
  fi: &FrameInvariants<T>, fs: &mut FrameState<T>,
) {
  assert!(fi.enable_segmentation);
  fs.segmentation.enabled = true;

  if fs.segmentation.enabled {
    fs.segmentation.update_map = true;

    // Update the values on every frame, as they vary by frame type.
    fs.segmentation.update_data = true;

    if !fs.segmentation.update_data {
      return;
    }

    // Select target quantizers for each segment by fitting to log(scale).
    let mut log_scales_q24 = fi
      .distortion_scales
      .iter()
      .zip(fi.activity_scales.iter())
      .map(|(&d, &a)| d.mul_blog_q24(a))
      .collect::<Vec<_>>();
    // Minimize the total distance from a small set of values to all scales.
    let centroids_q24 = kmeans(64, &mut log_scales_q24);
    // For the selected centroids, derive a target quantizer:
    //   scale Q'^2 = Q^2
    // See `distortion_scale_for` for more information.
    let qidx_diffs: ArrayVec<_, 3> = {
      use crate::rate::{bexp64, blog64, q24_to_q57};
      let log_ac_q_q57 =
        blog64(ac_q(fi.base_q_idx, 0, fi.sequence.bit_depth) as i64);
      centroids_q24
        .iter()
        // Rewrite in log form and exponentiate:
        //   scale Q'^2 = Q^2
        //           Q' = Q / sqrt(scale)
        //      log(Q') = log(Q) - 0.5 log(scale)
        .map(|&c| bexp64(log_ac_q_q57 - (q24_to_q57(c) >> 1)))
        // Find the index of the nearest quantizer to the target,
        // and take the delta from the base quantizer index.
        .map(|q| {
          // Avoid going into lossless mode by never bringing qidx below 1.
          select_ac_qi(q, fi.sequence.bit_depth).max(1) as i16
            - fi.base_q_idx as i16
        })
        .collect()
    };
    // Precompute the midpoints between selected centroids, in log(scale) form.
    let thresholds: ArrayVec<_, 2> = centroids_q24
      .iter()
      .zip(centroids_q24.iter().skip(1))
      .map(|(&c1, &c2)| (c1 + c2 + 1) >> 1)
      // Convert to scale for simple comparisons in `select_segment`.
      .map(DistortionScale::bexp_q24)
      .collect();

    for (i, &alt_q) in qidx_diffs.iter().enumerate() {
      fs.segmentation.features[i][SegLvl::SEG_LVL_ALT_Q as usize] = true;
      fs.segmentation.data[i][SegLvl::SEG_LVL_ALT_Q as usize] = alt_q;
    }
    fs.segmentation.thresholds.copy_from_slice(&thresholds);

    /* Figure out parameters */
    fs.segmentation.preskip = false;
    fs.segmentation.last_active_segid = 0;
    if fs.segmentation.enabled {
      for i in 0..8 {
        for j in 0..SegLvl::SEG_LVL_MAX as usize {
          if fs.segmentation.features[i][j] {
            fs.segmentation.last_active_segid = i as u8;
            if j >= SegLvl::SEG_LVL_REF_FRAME as usize {
              fs.segmentation.preskip = true;
            }
          }
        }
      }
    }
  }
}

pub fn select_segment<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>, tile_bo: TileBlockOffset,
  bsize: BlockSize, skip: bool,
) -> std::ops::RangeInclusive<u8> {
  use crate::api::SegmentationLevel;
  use crate::rdo::spatiotemporal_scale;

  // If skip is true or segmentation is turned off, sidx is not coded.
  if skip || !fi.enable_segmentation {
    return 0..=0;
  }

  if fi.config.speed_settings.segmentation == SegmentationLevel::Full {
    return 0..=2;
  }

  let frame_bo = ts.to_frame_block_offset(tile_bo);
  let scale = spatiotemporal_scale(fi, frame_bo, bsize);

  let sidx = if scale.0 < ts.segmentation.thresholds[0].0 {
    0
  } else if scale.0 > ts.segmentation.thresholds[1].0 {
    2
  } else {
    1
  };

  sidx..=sidx
}

fn kmeans(limit: usize, data: &mut [i32]) -> [i32; 3] {
  data.sort_unstable();
  let mut centroids = [data[0], data[data.len() / 2], data[data.len() - 1]];

  let mut low = [0, data.len() / 2, data.len()];
  let mut high = low;
  let mut sum = [0i64; 3];

  for _ in 0..limit {
    for (i, threshold) in centroids
      .iter()
      .zip(centroids.iter().skip(1))
      .map(|(&c1, &c2)| ((c1 as i64 + c2 as i64 + 1) >> 1) as i32)
      .enumerate()
    {
      let mut n = high[i];
      let mut s = sum[i];
      for &d in data[..n].iter().rev().take_while(|&d| *d > threshold) {
        s -= d as i64;
        n -= 1;
      }
      for &d in data[n..].iter().take_while(|&d| *d <= threshold) {
        s += d as i64;
        n += 1;
      }
      high[i] = n;
      sum[i] = s;

      let mut n = low[i + 1];
      let mut s = sum[i + 1];
      for &d in data[n..].iter().take_while(|&d| *d < threshold) {
        s -= d as i64;
        n += 1;
      }
      for &d in data[..n].iter().rev().take_while(|&d| *d >= threshold) {
        s += d as i64;
        n -= 1;
      }
      low[i + 1] = n;
      sum[i + 1] = s;
    }
    let mut changed = false;
    for (i, c) in centroids.iter_mut().enumerate() {
      let count = (high[i] - low[i]) as i64;
      assert!(count != 0);
      let new_centroid = ((sum[i] + (count >> 1)) / count) as i32;
      changed |= *c != new_centroid;
      *c = new_centroid;
    }
    if !changed {
      break;
    }
  }

  centroids
}
