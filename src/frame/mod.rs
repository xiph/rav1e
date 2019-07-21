// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::util::*;
use crate::api::ChromaSampling;
use crate::context::MAX_SB_SIZE;
use crate::mc::SUBPEL_FILTER_SIZE;

#[cfg(test)]
use crate::tiling::*;

mod plane;
pub use plane::*;

const FRAME_MARGIN: usize = 16 + SUBPEL_FILTER_SIZE;

/// Optional per-frame encoder parameters
pub struct FrameParameters {
  /// Force the frame to be emitted as keyframe
  pub keyframe: bool,
}

#[derive(Debug, Clone)]
pub struct Frame<T: Pixel> {
  pub planes: [Plane<T>; 3]
}

impl<T: Pixel> Frame<T> {
  pub fn new(width: usize, height: usize, chroma_sampling: ChromaSampling) -> Self {
    let luma_width = width.align_power_of_two(3);
    let luma_height = height.align_power_of_two(3);
    let luma_padding = MAX_SB_SIZE + FRAME_MARGIN;

    let (chroma_sampling_period_x, chroma_sampling_period_y) =
      chroma_sampling.sampling_period();
    let chroma_width = luma_width / chroma_sampling_period_x;
    let chroma_height = luma_height / chroma_sampling_period_y;
    let chroma_padding_x = luma_padding / chroma_sampling_period_x;
    let chroma_padding_y = luma_padding / chroma_sampling_period_y;
    let chroma_decimation_x = chroma_sampling_period_x - 1;
    let chroma_decimation_y = chroma_sampling_period_y - 1;

    Frame {
      planes: [
        Plane::new(
          luma_width, luma_height,
          0, 0,
          luma_padding, luma_padding
        ),
        Plane::new(
          chroma_width, chroma_height,
          chroma_decimation_x, chroma_decimation_y,
          chroma_padding_x, chroma_padding_y
        ),
        Plane::new(
          chroma_width, chroma_height,
          chroma_decimation_x, chroma_decimation_y,
          chroma_padding_x, chroma_padding_y
        )
      ]
    }
  }

  pub(crate) fn pad(&mut self, w: usize, h: usize) {
    for p in self.planes.iter_mut() {
      p.pad(w, h);
    }
  }

  #[inline(always)]
  #[cfg(test)]
  pub(crate) fn as_tile(&self) -> Tile<'_, T> {
    let PlaneConfig { width, height, .. } = self.planes[0].cfg;
    Tile::new(self, TileRect { x: 0, y: 0, width, height })
  }

  #[inline(always)]
  #[cfg(test)]
  pub fn as_tile_mut(&mut self) -> TileMut<'_, T> {
    let PlaneConfig { width, height, .. } = self.planes[0].cfg;
    TileMut::new(self, TileRect { x: 0, y: 0, width, height })
  }

  /// Returns a `PixelIter` containing the data of this frame's planes in YUV format.
  /// Each point in the `PixelIter` is a triple consisting of a Y, U, and V component.
  /// The `PixelIter` is laid out as contiguous rows, e.g. to get a given 0-indexed row
  /// you could use `data.skip(width * row_idx).take(width)`.
  ///
  /// This data retains any padding, e.g. it uses the width and height specifed in
  /// the Y-plane's `cfg` struct, and not the display width and height specied in
  /// `FrameInvariants`.
  pub(crate) fn iter(&self) -> PixelIter<'_, T> {
    PixelIter::new(&self.planes)
  }
}

#[derive(Debug)]
pub(crate) struct PixelIter<'a, T: Pixel> {
  planes: &'a [Plane<T>; 3],
  y: usize,
  x: usize,
}

impl<'a, T: Pixel> PixelIter<'a, T> {
  pub fn new(planes: &'a [Plane<T>; 3]) -> Self {
    PixelIter {
      planes,
      y: 0,
      x: 0,
    }
  }

  fn width(&self) -> usize {
    self.planes[0].cfg.width
  }

  fn height(&self) -> usize {
    self.planes[0].cfg.height
  }
}

impl<'a, T: Pixel> Iterator for PixelIter<'a, T> {
  type Item = (T, T, T);

  fn next(&mut self) -> Option<<Self as Iterator>::Item> {
    if self.y == self.height() - 1 && self.x == self.width() - 1 {
      return None;
    }
    let pixel = (
      self.planes[0].p(self.x, self.y),
      self.planes[1].p(self.x >> self.planes[1].cfg.xdec, self.y >> self.planes[1].cfg.ydec),
      self.planes[2].p(self.x >> self.planes[2].cfg.xdec, self.y >> self.planes[2].cfg.ydec),
    );
    if self.x == self.width() - 1 {
      self.x = 0;
      self.y += 1;
    } else {
      self.x += 1;
    }
    Some(pixel)
  }
}





