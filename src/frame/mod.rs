// Copyright (c) 2018-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use num_derive::FromPrimitive;

use crate::api::ChromaSampling;
use crate::context::SB_SIZE;
use crate::mc::SUBPEL_FILTER_SIZE;
use crate::util::*;

use crate::tiling::*;

mod plane;
pub use plane::*;

const FRAME_MARGIN: usize = 16 + SUBPEL_FILTER_SIZE;

/// Override the frame type decision
///
/// Only certain frame types can be selected.
#[derive(Debug, PartialEq, Clone, Copy, FromPrimitive)]
#[repr(C)]
pub enum FrameTypeOverride {
  /// Do not force any decision.
  No,
  /// Force the frame to be a Keyframe.
  Key,
}

/// Optional per-frame encoder parameters
#[derive(Debug, Clone, Copy)]
pub struct FrameParameters {
  /// Force emitted frame to be of the type selected
  pub frame_type_override: FrameTypeOverride,
}

/// One video frame.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Frame<T: Pixel> {
  /// Planes constituting the frame.
  pub planes: [Plane<T>; 3],
}

impl<T: Pixel> Frame<T> {
  /// Creates a new frame with the given parameters.
  ///
  /// Allocates data for the planes.
  pub fn new(
    width: usize, height: usize, chroma_sampling: ChromaSampling,
  ) -> Self {
    let luma_width = width.align_power_of_two(3);
    let luma_height = height.align_power_of_two(3);
    let luma_padding = SB_SIZE + FRAME_MARGIN;

    let (chroma_decimation_x, chroma_decimation_y) =
      chroma_sampling.get_decimation().unwrap_or((0, 0));
    let (chroma_width, chroma_height) =
      chroma_sampling.get_chroma_dimensions(luma_width, luma_height);
    let chroma_padding_x = luma_padding >> chroma_decimation_x;
    let chroma_padding_y = luma_padding >> chroma_decimation_y;

    Frame {
      planes: [
        Plane::new(luma_width, luma_height, 0, 0, luma_padding, luma_padding),
        Plane::new(
          chroma_width,
          chroma_height,
          chroma_decimation_x,
          chroma_decimation_y,
          chroma_padding_x,
          chroma_padding_y,
        ),
        Plane::new(
          chroma_width,
          chroma_height,
          chroma_decimation_x,
          chroma_decimation_y,
          chroma_padding_x,
          chroma_padding_y,
        ),
      ],
    }
  }

  pub(crate) fn pad(&mut self, w: usize, h: usize) {
    for p in self.planes.iter_mut() {
      p.pad(w, h);
    }
  }

  #[inline(always)]
  pub(crate) fn as_tile(&self) -> Tile<'_, T> {
    let PlaneConfig { width, height, .. } = self.planes[0].cfg;
    Tile::new(self, TileRect { x: 0, y: 0, width, height })
  }

  #[inline(always)]
  pub fn as_tile_mut(&mut self) -> TileMut<'_, T> {
    let PlaneConfig { width, height, .. } = self.planes[0].cfg;
    TileMut::new(self, TileRect { x: 0, y: 0, width, height })
  }
}
