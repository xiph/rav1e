// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::context::*;
use crate::frame::*;
use crate::util::*;

use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::slice;

/// Rectangle of a plane region, in pixels
#[derive(Debug, Clone, Copy)]
pub struct Rect {
  // coordinates relative to the plane origin (xorigin, yorigin)
  pub x: isize,
  pub y: isize,
  pub width: usize,
  pub height: usize,
}

impl Rect {
  #[inline(always)]
  pub fn decimated(&self, xdec: usize, ydec: usize) -> Self {
    Self {
      x: self.x >> xdec,
      y: self.y >> ydec,
      width: self.width >> xdec,
      height: self.height >> ydec,
    }
  }
}

// Structure to describe a rectangle area in several ways
//
// To retrieve a subregion from a region, we need to provide the subregion
// bounds, relative to its parent region. The subregion must always be included
// in its parent region.
//
// For that purpose, we could just use a rectangle (x, y, width, height), but
// this would be too cumbersome to use in practice. For example, we often need
// to pass a subregion from an offset, using the same bottom-right corner as
// its parent, or to pass a subregion expressed in block offset instead of
// pixel offset.
//
// Area provides a flexible way to describe a subregion.
#[derive(Debug, Clone, Copy)]
pub enum Area {
  /// A well-defined rectangle
  Rect { x: isize, y: isize, width: usize, height: usize },
  /// A rectangle starting at offset (x, y) and ending at the bottom-right
  /// corner of the parent
  StartingAt { x: isize, y: isize },
  /// A well-defined rectangle with offset expressed in blocks
  BlockRect { bo: BlockOffset, width: usize, height: usize },
  /// a rectangle starting at given block offset until the bottom-right corner
  /// of the parent
  BlockStartingAt { bo: BlockOffset },
}

impl Area {
  #[inline(always)]
  pub fn to_rect(
    &self, xdec: usize, ydec: usize, parent_width: usize, parent_height: usize,
  ) -> Rect {
    match *self {
      Area::Rect { x, y, width, height } => Rect { x, y, width, height },
      Area::StartingAt { x, y } => Rect {
        x,
        y,
        width: (parent_width as isize - x) as usize,
        height: (parent_height as isize - y) as usize,
      },
      Area::BlockRect { bo, width, height } => Rect {
        x: (bo.x >> xdec << BLOCK_TO_PLANE_SHIFT) as isize,
        y: (bo.y >> ydec << BLOCK_TO_PLANE_SHIFT) as isize,
        width,
        height,
      },
      Area::BlockStartingAt { bo } => {
        let x = (bo.x >> xdec << BLOCK_TO_PLANE_SHIFT) as isize;
        let y = (bo.y >> ydec << BLOCK_TO_PLANE_SHIFT) as isize;
        Rect {
          x,
          y,
          width: (parent_width as isize - x) as usize,
          height: (parent_height as isize - y) as usize,
        }
      }
    }
  }
}

/// Bounded region of a plane
///
/// This allows to give access to a rectangular area of a plane without
/// giving access to the whole plane.
#[derive(Debug)]
pub struct PlaneRegion<'a, T: Pixel> {
  data: *const T, // points to (plane_cfg.x, plane_cfg.y)
  pub plane_cfg: &'a PlaneConfig,
  // private to guarantee borrowing rules
  rect: Rect,
  phantom: PhantomData<&'a T>,
}

/// Mutable bounded region of a plane
///
/// This allows to give mutable access to a rectangular area of the plane
/// without giving access to the whole plane.
#[derive(Debug)]
pub struct PlaneRegionMut<'a, T: Pixel> {
  data: *mut T, // points to (plane_cfg.x, plane_cfg.y)
  pub plane_cfg: &'a PlaneConfig,
  rect: Rect,
  phantom: PhantomData<&'a mut T>,
}

// common impl for PlaneRegion and PlaneRegionMut
macro_rules! plane_region_common {
  // $name: PlaneRegion or PlaneRegionMut
  // $as_ptr: as_ptr or as_mut_ptr
  // $opt_mut: nothing or mut
  ($name:ident, $as_ptr:ident $(,$opt_mut:tt)?) => {
    impl<'a, T: Pixel> $name<'a, T> {

      #[inline(always)]
      pub fn new(plane: &'a $($opt_mut)? Plane<T>, rect: Rect) -> Self {
        assert!(rect.x >= -(plane.cfg.xorigin as isize));
        assert!(rect.y >= -(plane.cfg.yorigin as isize));
        assert!(plane.cfg.xorigin as isize + rect.x + rect.width as isize <= plane.cfg.stride as isize);
        assert!(plane.cfg.yorigin as isize + rect.y + rect.height as isize <= plane.cfg.alloc_height as isize);
        let origin = (plane.cfg.yorigin as isize + rect.y) * plane.cfg.stride as isize
                    + plane.cfg.xorigin as isize + rect.x;
        Self {
          data: unsafe { plane.data.$as_ptr().offset(origin) },
          plane_cfg: &plane.cfg,
          rect,
          phantom: PhantomData,
        }
      }

      #[inline(always)]
      pub fn data_ptr(&self) -> *const T {
        self.data
      }

      #[inline(always)]
      pub fn rect(&self) -> &Rect {
        &self.rect
      }

      #[inline(always)]
      pub fn rows_iter(&self) -> RowsIter<'_, T> {
        RowsIter {
          data: self.data,
          stride: self.plane_cfg.stride,
          width: self.rect.width,
          remaining: self.rect.height,
          phantom: PhantomData,
        }
      }

      // Return a view to a subregion of the plane
      //
      // The subregion must be included in (i.e. must not exceed) this region.
      //
      // It is described by an `Area`, relative to this region.
      //
      // # Example
      //
      // ``` ignore
      // # use rav1e::tiling::*;
      // # fn f(region: &PlaneRegion<'_, u16>) {
      // // a subregion from (10, 8) to the end of the region
      // let subregion = region.subregion(Area::StartingAt { x: 10, y: 8 });
      // # }
      // ```
      //
      // ``` ignore
      // # use rav1e::context::*;
      // # use rav1e::tiling::*;
      // # fn f(region: &PlaneRegion<'_, u16>) {
      // // a subregion from the top-left of block (2, 3) having size (64, 64)
      // let bo = BlockOffset { x: 2, y: 3 };
      // let subregion = region.subregion(Area::BlockRect { bo, width: 64, height: 64 });
      // # }
      // ```
      #[inline(always)]
      pub fn subregion(&self, area: Area) -> PlaneRegion<'_, T> {
        let rect = area.to_rect(
          self.plane_cfg.xdec,
          self.plane_cfg.ydec,
          self.rect.width,
          self.rect.height,
        );
        assert!(rect.x >= 0 && rect.x as usize <= self.rect.width);
        assert!(rect.y >= 0 && rect.y as usize <= self.rect.height);
        let data = unsafe {
          self.data.add(rect.y as usize * self.plane_cfg.stride + rect.x as usize)
        };
        let absolute_rect = Rect {
          x: self.rect.x + rect.x,
          y: self.rect.y + rect.y,
          width: rect.width,
          height: rect.height,
        };
        PlaneRegion {
          data,
          plane_cfg: &self.plane_cfg,
          rect: absolute_rect,
          phantom: PhantomData,
        }
      }

      #[inline(always)]
      pub fn to_frame_plane_offset(&self, tile_po: PlaneOffset) -> PlaneOffset {
        PlaneOffset {
          x: self.rect.x + tile_po.x,
          y: self.rect.y + tile_po.y,
        }
      }

      #[inline(always)]
      pub fn to_frame_block_offset(&self, tile_bo: TileBlockOffset) -> PlaneBlockOffset {
        debug_assert!(self.rect.x >= 0);
        debug_assert!(self.rect.y >= 0);
        let PlaneConfig { xdec, ydec, .. } = self.plane_cfg;
        debug_assert!(self.rect.x as usize % (MI_SIZE >> xdec) == 0);
        debug_assert!(self.rect.y as usize % (MI_SIZE >> ydec) == 0);
        let bx = self.rect.x as usize >> MI_SIZE_LOG2 - xdec;
        let by = self.rect.y as usize >> MI_SIZE_LOG2 - ydec;
        PlaneBlockOffset(BlockOffset {
          x: bx + tile_bo.0.x,
          y: by + tile_bo.0.y,
        })
      }

      #[inline(always)]
      pub fn to_frame_super_block_offset(
        &self,
        tile_sbo: TileSuperBlockOffset,
        sb_size_log2: usize
      ) -> PlaneSuperBlockOffset {
        debug_assert!(sb_size_log2 == 6 || sb_size_log2 == 7);
        debug_assert!(self.rect.x >= 0);
        debug_assert!(self.rect.y >= 0);
        let PlaneConfig { xdec, ydec, .. } = self.plane_cfg;
        debug_assert!(self.rect.x as usize % (1 << sb_size_log2 - xdec) == 0);
        debug_assert!(self.rect.y as usize % (1 << sb_size_log2 - ydec) == 0);
        let sbx = self.rect.x as usize >> sb_size_log2 - xdec;
        let sby = self.rect.y as usize >> sb_size_log2 - ydec;
        PlaneSuperBlockOffset(SuperBlockOffset {
          x: sbx + tile_sbo.0.x,
          y: sby + tile_sbo.0.y,
        })
      }
    }

    unsafe impl<T: Pixel> Send for $name<'_, T> {}
    unsafe impl<T: Pixel> Sync for $name<'_, T> {}

    impl<T: Pixel> Index<usize> for $name<'_, T> {
      type Output = [T];

      #[inline(always)]
      fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.rect.height);
        unsafe {
          let ptr = self.data.add(index * self.plane_cfg.stride);
          slice::from_raw_parts(ptr, self.rect.width)
        }
      }
    }
  }
}

plane_region_common!(PlaneRegion, as_ptr);
plane_region_common!(PlaneRegionMut, as_mut_ptr, mut);

impl<'a, T: Pixel> PlaneRegionMut<'a, T> {
  #[inline(always)]
  pub fn data_ptr_mut(&mut self) -> *mut T {
    self.data
  }

  #[inline(always)]
  pub fn rows_iter_mut(&mut self) -> RowsIterMut<'_, T> {
    RowsIterMut {
      data: self.data,
      stride: self.plane_cfg.stride,
      width: self.rect.width,
      remaining: self.rect.height,
      phantom: PhantomData,
    }
  }

  // Return a mutable view to a subregion of the plane
  //
  // The subregion must be included in (i.e. must not exceed) this region.
  //
  // It is described by an `Area`, relative to this region.
  //
  // # Example
  //
  // ``` ignore
  // # use rav1e::tiling::*;
  // # fn f(region: &mut PlaneRegionMut<'_, u16>) {
  // // a mutable subregion from (10, 8) having size (32, 32)
  // let subregion = region.subregion_mut(Area::Rect { x: 10, y: 8, width: 32, height: 32 });
  // # }
  // ```
  //
  // ``` ignore
  // # use rav1e::context::*;
  // # use rav1e::tiling::*;
  // # fn f(region: &mut PlaneRegionMut<'_, u16>) {
  // // a mutable subregion from the top-left of block (2, 3) to the end of the region
  // let bo = BlockOffset { x: 2, y: 3 };
  // let subregion = region.subregion_mut(Area::BlockStartingAt { bo });
  // # }
  // ```
  #[inline(always)]
  pub fn subregion_mut(&mut self, area: Area) -> PlaneRegionMut<'_, T> {
    let rect = area.to_rect(
      self.plane_cfg.xdec,
      self.plane_cfg.ydec,
      self.rect.width,
      self.rect.height,
    );
    assert!(rect.x >= 0 && rect.x as usize <= self.rect.width);
    assert!(rect.y >= 0 && rect.y as usize <= self.rect.height);
    let data = unsafe {
      self.data.add(rect.y as usize * self.plane_cfg.stride + rect.x as usize)
    };
    let absolute_rect = Rect {
      x: self.rect.x + rect.x,
      y: self.rect.y + rect.y,
      width: rect.width,
      height: rect.height,
    };
    PlaneRegionMut {
      data,
      plane_cfg: &self.plane_cfg,
      rect: absolute_rect,
      phantom: PhantomData,
    }
  }

  #[inline(always)]
  pub fn as_const(&self) -> PlaneRegion<'_, T> {
    PlaneRegion {
      data: self.data,
      plane_cfg: self.plane_cfg,
      rect: self.rect,
      phantom: PhantomData,
    }
  }
}

impl<T: Pixel> IndexMut<usize> for PlaneRegionMut<'_, T> {
  #[inline(always)]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    assert!(index < self.rect.height);
    unsafe {
      let ptr = self.data.add(index * self.plane_cfg.stride);
      slice::from_raw_parts_mut(ptr, self.rect.width)
    }
  }
}

/// Iterator over plane region rows
pub struct RowsIter<'a, T: Pixel> {
  data: *const T,
  stride: usize,
  width: usize,
  remaining: usize,
  phantom: PhantomData<&'a T>,
}

/// Mutable iterator over plane region rows
pub struct RowsIterMut<'a, T: Pixel> {
  data: *mut T,
  stride: usize,
  width: usize,
  remaining: usize,
  phantom: PhantomData<&'a mut T>,
}

impl<'a, T: Pixel> Iterator for RowsIter<'a, T> {
  type Item = &'a [T];

  #[inline(always)]
  fn next(&mut self) -> Option<Self::Item> {
    if self.remaining > 0 {
      let row = unsafe {
        let ptr = self.data;
        self.data = self.data.add(self.stride);
        slice::from_raw_parts(ptr, self.width)
      };
      self.remaining -= 1;
      Some(row)
    } else {
      None
    }
  }

  #[inline(always)]
  fn size_hint(&self) -> (usize, Option<usize>) {
    (self.remaining, Some(self.remaining))
  }
}

impl<'a, T: Pixel> Iterator for RowsIterMut<'a, T> {
  type Item = &'a mut [T];

  #[inline(always)]
  fn next(&mut self) -> Option<Self::Item> {
    if self.remaining > 0 {
      let row = unsafe {
        let ptr = self.data;
        self.data = self.data.add(self.stride);
        slice::from_raw_parts_mut(ptr, self.width)
      };
      self.remaining -= 1;
      Some(row)
    } else {
      None
    }
  }

  #[inline(always)]
  fn size_hint(&self) -> (usize, Option<usize>) {
    (self.remaining, Some(self.remaining))
  }
}

impl<T: Pixel> ExactSizeIterator for RowsIter<'_, T> {}
impl<T: Pixel> ExactSizeIterator for RowsIterMut<'_, T> {}
