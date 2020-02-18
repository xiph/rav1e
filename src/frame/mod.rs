// Copyright (c) 2018-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use num_derive::FromPrimitive;

use crate::context::SB_SIZE;
use crate::mc::SUBPEL_FILTER_SIZE;
use crate::util::*;

#[cfg(test)]
use crate::tiling::*;

mod plane;
pub use plane::*;

const FRAME_MARGIN: usize = 16 + SUBPEL_FILTER_SIZE;
const LUMA_PADDING: usize = SB_SIZE + FRAME_MARGIN;

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

pub use v_frame::frame::Frame;

/// Public Trait Interface for Frame Allocation
pub trait FrameAlloc {
  /// Initialise new frame default type
  fn new(width: usize, height: usize, chroma_sampling: ChromaSampling)
    -> Self;
}

impl<T: Pixel> FrameAlloc for Frame<T> {
  /// Creates a new frame with the given parameters.
  /// new function calls new_with_padding function which takes luma_padding
  /// as parameter
  fn new(
    width: usize, height: usize, chroma_sampling: ChromaSampling,
  ) -> Self {
    v_frame::frame::Frame::new_with_padding(
      width,
      height,
      chroma_sampling,
      LUMA_PADDING,
    )
  }
}

/// Public Trait for calulating Padding
pub(crate) trait FramePad {
  fn pad(&mut self, w: usize, h: usize);
}

impl<T: Pixel> FramePad for Frame<T> {
  fn pad(&mut self, w: usize, h: usize) {
    for p in self.planes.iter_mut() {
      p.pad(w, h);
    }
  }
}

/// Public Trait for new Tile of a frame
pub(crate) trait AsTile<T: Pixel> {
  #[cfg(test)]
  fn as_tile(&self) -> Tile<'_, T>;
  #[cfg(test)]
  fn as_tile_mut(&mut self) -> TileMut<'_, T>;
}

#[cfg(test)]
impl<T: Pixel> AsTile<T> for Frame<T> {
  #[inline(always)]
  fn as_tile(&self) -> Tile<'_, T> {
    let PlaneConfig { width, height, .. } = self.planes[0].cfg;
    Tile::new(self, TileRect { x: 0, y: 0, width, height })
  }
  #[inline(always)]
  fn as_tile_mut(&mut self) -> TileMut<'_, T> {
    let PlaneConfig { width, height, .. } = self.planes[0].cfg;
    TileMut::new(self, TileRect { x: 0, y: 0, width, height })
  }
}
