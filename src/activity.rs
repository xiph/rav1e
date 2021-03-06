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
use rust_hawktracer::*;

#[derive(Debug, Default, Clone)]
pub struct ActivityMask {
  variances: Vec<f64>,
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

        let mean: f64 = block
          .rows_iter()
          .flatten()
          .map(|&pix| {
            let pix: i16 = CastFromPrimitive::cast_from(pix);
            pix as f64
          })
          .sum::<f64>()
          / 64.0_f64;
        let variance: f64 = block
          .rows_iter()
          .flatten()
          .map(|&pix| {
            let pix: i16 = CastFromPrimitive::cast_from(pix);
            (pix as f64 - mean).powi(2)
          })
          .sum::<f64>();
        variances.push(variance);
      }
    }
    ActivityMask { variances, width, height, granularity }
  }
}
