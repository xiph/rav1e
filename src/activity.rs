// Copyright (c) 2017-2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::frame::*;
use crate::hawktracer::*;
use crate::tiling::*;
use crate::util::*;

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

  pub fn variance_at(&self, x: usize, y: usize) -> Option<f64> {
    let (dec_width, dec_height) =
      (self.width >> self.granularity, self.height >> self.granularity);
    if x > dec_width || y > dec_height {
      None
    } else {
      Some(*self.variances.get(x + dec_width * y).unwrap())
    }
  }

  pub fn mean_activity_of(&self, rect: Rect) -> Option<f64> {
    let Rect { x, y, width, height } = rect;
    let (x, y) = (x as usize, y as usize);
    let granularity = self.granularity;
    let (dec_x, dec_y) = (x >> granularity, y >> granularity);
    let (dec_width, dec_height) =
      (width >> granularity, height >> granularity);

    if x > self.width
      || y > self.height
      || (x + width) > self.width
      || (y + height) > self.height
      || dec_width == 0
      || dec_height == 0
    {
      // Region lies out of the frame or is smaller than 8x8 on some axis
      None
    } else {
      let activity = self
        .variances
        .chunks_exact(self.width >> granularity)
        .skip(dec_y)
        .take(dec_height)
        .map(|row| row.iter().skip(dec_x).take(dec_width).sum::<f64>())
        .sum::<f64>()
        / (dec_width as f64 * dec_height as f64);

      Some(activity.cbrt().sqrt())
    }
  }
}
