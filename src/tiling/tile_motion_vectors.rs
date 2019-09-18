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

use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::slice;

/// Tiled view of FrameMotionVectors
#[derive(Debug)]
pub struct TileMotionVectors<'a> {
  data: *const MotionVector,
  // expressed in mi blocks
  // private to guarantee borrowing rules
  x: usize,
  y: usize,
  cols: usize,
  rows: usize,
  stride: usize, // number of cols in the underlying FrameMotionVectors
  phantom: PhantomData<&'a MotionVector>,
}

/// Mutable tiled view of FrameMotionVectors
#[derive(Debug)]
pub struct TileMotionVectorsMut<'a> {
  data: *mut MotionVector,
  // expressed in mi blocks
  // private to guarantee borrowing rules
  x: usize,
  y: usize,
  cols: usize,
  rows: usize,
  stride: usize, // number of cols in the underlying FrameMotionVectors
  phantom: PhantomData<&'a mut MotionVector>,
}

// common impl for TileMotionVectors and TileMotionVectorsMut
macro_rules! tile_motion_vectors_common {
  // $name: TileMotionVectors or TileMotionVectorsMut
  // $opt_mut: nothing or mut
  ($name:ident $(,$opt_mut:tt)?) => {
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
          data: & $($opt_mut)? frame_mvs[y][x],
          x,
          y,
          cols,
          rows,
          stride: frame_mvs.cols,
          phantom: PhantomData,
        }
      }

      #[inline(always)]
      pub const fn x(&self) -> usize {
        self.x
      }

      #[inline(always)]
      pub const fn y(&self) -> usize {
        self.y
      }

      #[inline(always)]
      pub const fn cols(&self) -> usize {
        self.cols
      }

      #[inline(always)]
      pub const fn rows(&self) -> usize {
        self.rows
      }
    }

    unsafe impl Send for $name<'_> {}
    unsafe impl Sync for $name<'_> {}

    impl Index<usize> for $name<'_> {
      type Output = [MotionVector];

      #[inline(always)]
      fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.rows);
        unsafe {
          let ptr = self.data.add(index * self.stride);
          slice::from_raw_parts(ptr, self.cols)
        }
      }
    }
  }
}

tile_motion_vectors_common!(TileMotionVectors);
tile_motion_vectors_common!(TileMotionVectorsMut, mut);

impl TileMotionVectorsMut<'_> {
  #[inline(always)]
  pub const fn as_const(&self) -> TileMotionVectors<'_> {
    TileMotionVectors {
      data: self.data,
      x: self.x,
      y: self.y,
      cols: self.cols,
      rows: self.rows,
      stride: self.stride,
      phantom: PhantomData,
    }
  }
}

impl IndexMut<usize> for TileMotionVectorsMut<'_> {
  #[inline(always)]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    assert!(index < self.rows);
    unsafe {
      let ptr = self.data.add(index * self.stride);
      slice::from_raw_parts_mut(ptr, self.cols)
    }
  }
}
