// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::context::*;
use crate::lrf::*;
use crate::encoder::FrameInvariants;
use crate::util::Pixel;

use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::ptr;
use std::slice;

/// Tiled view of RestorationUnits
#[derive(Debug)]
pub struct TileRestorationUnits<'a> {
  data: *const RestorationUnit,
  // private to guarantee borrowing rules
  x: usize,
  y: usize,
  cols: usize,
  rows: usize,
  stride: usize, // number of cols in the underlying FrameRestorationUnits
  phantom: PhantomData<&'a RestorationUnit>,
}

/// Mutable tiled view of RestorationUnits
#[derive(Debug)]
pub struct TileRestorationUnitsMut<'a> {
  data: *mut RestorationUnit,
  // private to guarantee borrowing rules
  x: usize,
  y: usize,
  cols: usize,
  rows: usize,
  stride: usize, // number of cols in the underlying FrameRestorationUnits
  phantom: PhantomData<&'a mut RestorationUnit>,
}

// common impl for TileRestorationUnits and TileRestorationUnitsMut
macro_rules! tile_restoration_units_common {
  // $name: TileRestorationUnits or TileRestorationUnitsMut
  // $null: null or null_mut
  // $opt_mut: nothing or mut
  ($name:ident, $null:ident $(,$opt_mut:tt)?) => {
    impl<'a> $name<'a> {

      #[inline(always)]
      pub fn new(
        frame_units: &'a $($opt_mut)? FrameRestorationUnits,
        x: usize,
        y: usize,
        cols: usize,
        rows: usize,
      ) -> Self {
        Self {
          data: if x < frame_units.cols && y < frame_units.rows {
            & $($opt_mut)? frame_units[y][x]
          } else {
            // on edges, a tile may contain no restoration units
            ptr::$null()
          },
          x,
          y,
          cols,
          rows,
          stride: frame_units.cols,
          phantom: PhantomData,
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
        self.cols
      }

      #[inline(always)]
      pub fn rows(&self) -> usize {
        self.rows
      }
    }

    unsafe impl Send for $name<'_> {}
    unsafe impl Sync for $name<'_> {}

    impl Index<usize> for $name<'_> {
      type Output = [RestorationUnit];

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

tile_restoration_units_common!(TileRestorationUnits, null);
tile_restoration_units_common!(TileRestorationUnitsMut, null_mut, mut);

impl TileRestorationUnitsMut<'_> {
  #[inline(always)]
  pub fn as_const(&self) -> TileRestorationUnits<'_> {
    TileRestorationUnits {
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

impl IndexMut<usize> for TileRestorationUnitsMut<'_> {
  #[inline(always)]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    assert!(index < self.rows);
    unsafe {
      let ptr = self.data.add(index * self.stride);
      slice::from_raw_parts_mut(ptr, self.cols)
    }
  }
}

/// Tiled view of RestorationPlane
#[derive(Debug)]
pub struct TileRestorationPlane<'a> {
  pub rp_cfg: &'a RestorationPlaneConfig,
  pub wiener_ref: [[i8; 3]; 2],
  pub sgrproj_ref: [i8; 2],
  pub units: TileRestorationUnits<'a>,
}

/// Mutable tiled view of RestorationPlane
#[derive(Debug)]
pub struct TileRestorationPlaneMut<'a> {
  pub rp_cfg: &'a RestorationPlaneConfig,
  pub wiener_ref: [[i8; 3]; 2],
  pub sgrproj_ref: [i8; 2],
  pub units: TileRestorationUnitsMut<'a>,
}

// common impl for TileRestorationPlane and TileRestorationPlaneMut
macro_rules! tile_restoration_plane_common {
  // $name: TileRestorationPlane or TileRestorationPlaneMut
  // $tru_type: TileRestorationUnits or TileRestorationUnitsMut
  // $opt_mut: nothing or mut
  ($name:ident, $tru_type:ident $(,$opt_mut:tt)?) => {
    impl<'a> $name<'a> {

      #[inline(always)]
      pub fn new(
        rp: &'a $($opt_mut)? RestorationPlane,
        units_x: usize,
        units_y: usize,
        units_cols: usize,
        units_rows: usize,
      ) -> Self {
        Self {
          rp_cfg: &rp.cfg,
          wiener_ref: [WIENER_TAPS_MID; 2],
          sgrproj_ref: SGRPROJ_XQD_MID,
          units: $tru_type::new(& $($opt_mut)? rp.units, units_x, units_y, units_cols, units_rows),
        }
      }

      pub fn restoration_unit_index(&self, sbo: TileSuperBlockOffset) -> Option<(usize, usize)> {
        // is this a stretch block?
        let x_stretch = sbo.0.x < self.rp_cfg.sb_cols &&
          sbo.0.x >> self.rp_cfg.sb_shift >= self.units.cols;
        let y_stretch = sbo.0.y < self.rp_cfg.sb_rows &&
          sbo.0.y >> self.rp_cfg.sb_shift >= self.units.rows;

        let x = (sbo.0.x >> self.rp_cfg.sb_shift) - if x_stretch { 1 } else { 0 };
        let y = (sbo.0.y >> self.rp_cfg.sb_shift) - if y_stretch { 1 } else { 0 };
        if x < self.units.cols && y < self.units.rows {
          Some((x, y))
        } else {
          None
        }
      }

      pub fn restoration_unit_countable(&self, sbo: TileSuperBlockOffset) -> usize {
        if let Some((x,y)) = self.restoration_unit_index(sbo) {
          y*self.units.cols+x
        } else {
          unreachable!()
        }
      }

      // Is this the last sb (in scan order) in the restoration unit?
      // Stretch makes this a bit annoying to compute.
      pub fn restoration_unit_last_sb<T: Pixel>(&self, fi: &FrameInvariants<T>,
                                            sbo: TileSuperBlockOffset) -> bool {
        // there is 1 restoration unit for (1 << sb_shift) super-blocks
        let mask = (1 << self.rp_cfg.sb_shift) - 1;
        // is this a stretch block?
        let x_stretch = sbo.0.x >> self.rp_cfg.sb_shift >= self.units.cols;
        let y_stretch = sbo.0.y >> self.rp_cfg.sb_shift >= self.units.rows;

        let last_x = (sbo.0.x & mask == mask && !x_stretch) || sbo.0.x == fi.sb_width-1;
        let last_y = (sbo.0.y & mask == mask && !y_stretch) || sbo.0.y == fi.sb_height-1;

        last_x && last_y
      }

      #[inline(always)]
      pub fn restoration_unit(&self, sbo: TileSuperBlockOffset) -> Option<&RestorationUnit> {
        self.restoration_unit_index(sbo).map(|(x, y)| &self.units[y][x])
      }
    }
  }
}

tile_restoration_plane_common!(TileRestorationPlane, TileRestorationUnits);
tile_restoration_plane_common!(
  TileRestorationPlaneMut,
  TileRestorationUnitsMut,
  mut
);

impl<'a> TileRestorationPlaneMut<'a> {
  #[inline(always)]
  pub fn restoration_unit_mut(
    &mut self, sbo: TileSuperBlockOffset,
  ) -> Option<&mut RestorationUnit> {
    // cannot use map() due to lifetime constraints
    if let Some((x, y)) = self.restoration_unit_index(sbo) {
      Some(&mut self.units[y][x])
    } else {
      None
    }
  }

  #[inline(always)]
  pub fn as_const(&self) -> TileRestorationPlane<'_> {
    TileRestorationPlane {
      rp_cfg: self.rp_cfg,
      wiener_ref: self.wiener_ref,
      sgrproj_ref: self.sgrproj_ref,
      units: self.units.as_const(),
    }
  }
}

/// Tiled view of RestorationState
#[derive(Debug)]
pub struct TileRestorationState<'a> {
  pub planes: [TileRestorationPlane<'a>; PLANES],
}

/// Mutable tiled view of RestorationState
#[derive(Debug)]
pub struct TileRestorationStateMut<'a> {
  pub planes: [TileRestorationPlaneMut<'a>; PLANES],
}

// common impl for TileRestorationState and TileRestorationStateMut
macro_rules! tile_restoration_state_common {
  // $name: TileRestorationState or TileRestorationStateMut
  // $trp_type: TileRestorationPlane or TileRestorationPlaneMut
  // $iter: iter or iter_mut
  // $opt_mut: nothing or mut
  ($name:ident, $trp_type:ident, $iter:ident $(,$opt_mut:tt)?) => {
    impl<'a> $name<'a> {

      #[inline(always)]
      pub fn new(
        rs: &'a $($opt_mut)? RestorationState,
        sbo: PlaneSuperBlockOffset,
        sb_width: usize,
        sb_height: usize,
      ) -> Self {
        let (units_x0, units_y0, units_cols0, units_rows0) =
          Self::get_units_region(rs, sbo, sb_width, sb_height, 0);
        let (units_x1, units_y1, units_cols1, units_rows1) =
          Self::get_units_region(rs, sbo, sb_width, sb_height, 1);
        let (units_x2, units_y2, units_cols2, units_rows2) =
          Self::get_units_region(rs, sbo, sb_width, sb_height, 2);
        // we cannot retrieve &mut of slice items directly and safely
        let mut planes_iter = rs.planes.$iter();
        Self {
          planes: [
            {
              let plane = planes_iter.next().unwrap();
              $trp_type::new(plane, units_x0, units_y0, units_cols0, units_rows0)
            },
            {
              let plane = planes_iter.next().unwrap();
              $trp_type::new(plane, units_x1, units_y1, units_cols1, units_rows1)
            },
            {
              let plane = planes_iter.next().unwrap();
              $trp_type::new(plane, units_x2, units_y2, units_cols2, units_rows2)
            },
          ],
        }
      }

      #[inline(always)]
      fn get_units_region(
        rs: &RestorationState,
        sbo: PlaneSuperBlockOffset,
        sb_width: usize,
        sb_height: usize,
        pli: usize,
      ) -> (usize, usize, usize, usize) {
        let sb_shift = rs.planes[pli].cfg.sb_shift;
        // there may be several super-blocks per restoration unit
        // the given super-block offset must match the start of a restoration unit
        debug_assert!(sbo.0.x % (1 << sb_shift) == 0);
        debug_assert!(sbo.0.y % (1 << sb_shift) == 0);

        let units_x = sbo.0.x >> sb_shift;
        let units_y = sbo.0.y >> sb_shift;
        let units_cols = sb_width + (1 << sb_shift) - 1 >> sb_shift;
        let units_rows = sb_height + (1<< sb_shift) - 1 >> sb_shift;

        let FrameRestorationUnits { cols: rs_cols, rows: rs_rows, .. } = rs.planes[pli].units;
        // +1 because the last super-block may use the "stretched" restoration unit
        // from its neighbours
        // <https://github.com/xiph/rav1e/issues/631#issuecomment-454419152>
        debug_assert!(units_x < rs_cols + 1);
        debug_assert!(units_y < rs_rows + 1);
        debug_assert!(units_x + units_cols <= rs_cols + 1);
        debug_assert!(units_y + units_rows <= rs_rows + 1);

        let units_x = units_x.min(rs_cols);
        let units_y = units_y.min(rs_rows);
        let units_cols = units_cols.min(rs_cols - units_x);
        let units_rows = units_rows.min(rs_rows - units_y);
        (units_x, units_y, units_cols, units_rows)
      }

      #[inline(always)]
      pub fn has_restoration_unit(&self, sbo: TileSuperBlockOffset, pli: usize) -> bool {
        let is_some = self.planes[pli].restoration_unit(sbo).is_some();
        is_some
      }
    }
  }
}

tile_restoration_state_common!(
  TileRestorationState,
  TileRestorationPlane,
  iter
);
tile_restoration_state_common!(
  TileRestorationStateMut,
  TileRestorationPlaneMut,
  iter_mut,
  mut
);

impl<'a> TileRestorationStateMut<'a> {
  #[inline(always)]
  pub fn as_const(&self) -> TileRestorationState {
    TileRestorationState {
      planes: [
        self.planes[0].as_const(),
        self.planes[1].as_const(),
        self.planes[2].as_const(),
      ],
    }
  }
}
