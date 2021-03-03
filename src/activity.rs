// Copyright (c) 2017-2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::frame::*;
use crate::tiling::*;
use crate::util::*;
use itertools::izip;
use rust_hawktracer::*;

#[derive(Debug, Default, Clone)]
pub struct ActivityMask {
  variances: Vec<u32>,
  // Width and height of the original frame that is masked
  width: usize,
  height: usize,
  // Side of unit (square) activity block in log2
  granularity: usize,
}

impl ActivityMask {
  #[hawktracer(activity_mask_from_plane)]
  pub fn from_plane<T: Pixel>(luma_plane: &Plane<T>) -> ActivityMask {
    let PlaneConfig { width, height, .. } = luma_plane.cfg;

    let granularity = 3;

    let aligned_luma = Rect {
      x: 0_isize,
      y: 0_isize,
      width: (width >> granularity) << granularity,
      height: (height >> granularity) << granularity,
    };
    let luma = PlaneRegion::new(luma_plane, aligned_luma);

    let mut variances =
      Vec::with_capacity((height >> granularity) * (width >> granularity));

    for y in 0..height >> granularity {
      for x in 0..width >> granularity {
        let block_rect = Area::Rect {
          x: (x << granularity) as isize,
          y: (y << granularity) as isize,
          width: 8,
          height: 8,
        };

        let block = luma.subregion(block_rect);
        let variance = variance_8x8(&block);
        variances.push(variance);
      }
    }
    ActivityMask { variances, width, height, granularity }
  }
}

// Adapted from the source variance calculation in cdef_dist_wxh_8x8.
#[inline(never)]
fn variance_8x8<T: Pixel>(src: &PlaneRegion<'_, T>) -> u32 {
  debug_assert!(src.plane_cfg.xdec == 0);
  debug_assert!(src.plane_cfg.ydec == 0);

  // Sum into columns to improve auto-vectorization
  let mut sum_s_cols: [u16; 8] = [0; 8];
  let mut sum_s2_cols: [u32; 8] = [0; 8];

  // Check upfront that 8 rows are available.
  let _row = &src[7];

  for j in 0..8 {
    let row = &src[j][0..8];
    for (sum_s, sum_s2, s) in izip!(&mut sum_s_cols, &mut sum_s2_cols, row) {
      // Don't convert directly to u32 to allow better vectorization
      let s: u16 = u16::cast_from(*s);
      *sum_s += s;

      // Convert to u32 to avoid overflows when multiplying
      let s: u32 = s as u32;
      *sum_s2 += s * s;
    }
  }

  // Sum together the sum of columns
  let sum_s = sum_s_cols.iter().map(|&a| u32::cast_from(a)).sum::<u32>();
  let sum_s2 = sum_s2_cols.iter().sum::<u32>();

  // Use sums to calculate variance
  sum_s2 - ((sum_s * sum_s + 32) >> 6)
}
