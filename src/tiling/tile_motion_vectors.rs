// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::mc::MotionVector;
use crate::me::*;
use crate::util::{Slice2D, Slice2DMut};

use std::ops::{Index, IndexMut};

/// Tiled view of FrameMotionVectors
#[derive(Debug)]
pub struct TileMotionVectors<'a> {
  data: Slice2D<'a, MotionVector>,
  // expressed in mi blocks
  // private to guarantee borrowing rules
  x: usize,
  y: usize,
}

/// Mutable tiled view of FrameMotionVectors
#[derive(Debug)]
pub struct TileMotionVectorsMut<'a> {
  data: Slice2DMut<'a, MotionVector>,
  // expressed in mi blocks
  // private to guarantee borrowing rules
  x: usize,
  y: usize,
}

// common impl for TileMotionVectors and TileMotionVectorsMut
macro_rules! tile_motion_vectors_common {
  // $name: TileMotionVectors or TileMotionVectorsMut
  // $opt_mut: nothing or mut
  ($name:ident, $slice:ident $(,$opt_mut:tt)?) => {
    impl<'a> $name<'a> {

      #[inline(always)]
      pub fn new(
        frame_mvs: &'a $($opt_mut)? FrameMotionVectors,
        x: usize,
        y: usize,
        cols: usize,
        rows: usize,
      ) -> Self {
        assert!(x + cols <= frame_mvs.cols);
        assert!(y + rows <= frame_mvs.rows);
        Self {
          data: $slice ::new(& $($opt_mut)? frame_mvs[y][x], cols, rows, frame_mvs.cols),
          x,
          y,
        }
      }

      #[inline(always)]
      pub fn x(&self) -> usize {
        self.x
      }

      #[inline(always)]
      pub fn y(&self) -> usize {
        self.y
      }

      #[inline(always)]
      pub fn cols(&self) -> usize {
        self.data.width()
      }

      #[inline(always)]
      pub fn rows(&self) -> usize {
        self.data.height()
      }
    }

    unsafe impl Send for $name<'_> {}
    unsafe impl Sync for $name<'_> {}

    impl Index<usize> for $name<'_> {
      type Output = [MotionVector];

      #[inline(always)]
      fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
      }
    }
  }
}

tile_motion_vectors_common!(TileMotionVectors, Slice2D);
tile_motion_vectors_common!(TileMotionVectorsMut, Slice2DMut, mut);

impl TileMotionVectorsMut<'_> {
  #[inline(always)]
  pub fn as_const(&self) -> TileMotionVectors<'_> {
    TileMotionVectors { data: self.data.as_const(), x: self.x, y: self.y }
  }
}

impl IndexMut<usize> for TileMotionVectorsMut<'_> {
  #[inline(always)]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    &mut self.data[index]
  }
}
