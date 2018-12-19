// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use context::*;
use encoder::*;
use plane::*;

use std::marker::PhantomData;
use std::slice;

#[derive(Debug, Clone)]
pub struct PlaneRegionConfig {
  // coordinates of the region relative to the raw plane (including padding)
  pub xorigin: usize,
  pub yorigin: usize,
  pub width: usize,
  pub height: usize,
}

#[derive(Debug, Clone)]
pub struct PlaneRegionMut<'a> {
  data: *mut u16,
  pub plane_cfg: &'a PlaneConfig,
  pub cfg: PlaneRegionConfig,
  phantom: PhantomData<&'a mut u16>,
}

impl<'a> PlaneRegionMut<'a> {
  // exposed as unsafe because nothing prevents the caller to retrieve overlapping regions
  unsafe fn new(plane: *mut Plane, cfg: PlaneRegionConfig) -> Self {
    PlaneRegionMut {
      data: (*plane).data.as_mut_ptr(),
      plane_cfg: &(*plane).cfg,
      cfg,
      phantom: PhantomData,
    }
  }

  pub fn row(&self, y: usize) -> &[u16] {
    assert!(y < self.cfg.height);
    let offset =
      (self.cfg.yorigin + y) * self.plane_cfg.stride + self.cfg.xorigin;
    unsafe { slice::from_raw_parts(self.data.add(offset), self.cfg.width) }
  }

  pub fn row_mut(&mut self, y: usize) -> &mut [u16] {
    assert!(y < self.cfg.height);
    let offset =
      (self.cfg.yorigin + y) * self.plane_cfg.stride + self.cfg.xorigin;
    unsafe { slice::from_raw_parts_mut(self.data.add(offset), self.cfg.width) }
  }
}

#[derive(Debug, Clone)]
pub struct TileMut<'a> {
  pub planes: [PlaneRegionMut<'a>; PLANES],
}

#[derive(Debug, Clone, Copy)]
struct TileInfo {
  width: usize,
  height: usize,
}

pub struct TileIterMut<'a> {
  planes: *mut [Plane; PLANES],
  tile_info: TileInfo,
  x: usize,
  y: usize,
  phantom: PhantomData<&'a mut Plane>,
}

impl<'a> TileIterMut<'a> {
  pub fn from_frame(
    frame: &'a mut Frame,
    tile_width: usize,
    tile_height: usize,
  ) -> Self {
    let tile_info = TileInfo { width: tile_width, height: tile_height };
    Self {
      planes: &mut frame.planes,
      tile_info,
      x: 0,
      y: 0,
      phantom: PhantomData,
    }
  }
}

impl<'a> Iterator for TileIterMut<'a> {
  type Item = TileMut<'a>;

  fn next(&mut self) -> Option<TileMut<'a>> {
    let planes = unsafe { &mut *self.planes };
    let width = planes[0].cfg.width;
    let height = planes[0].cfg.height;
    if self.x >= width || self.y >= height {
      None
    } else {
      // use current (x, y) for the current TileMut
      let (x, y) = (self.x, self.y);

      // update (self.x, self.y) for the next TileMut
      self.x += self.tile_info.width;
      if self.x >= width {
        self.x = 0;
        self.y += self.tile_info.height;
      }

      Some(TileMut {
        planes: unsafe {
          [
            Self::next_region(&mut planes[0], x, y, self.tile_info),
            Self::next_region(&mut planes[1], x, y, self.tile_info),
            Self::next_region(&mut planes[2], x, y, self.tile_info),
          ]
        },
      })
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    let planes = unsafe { &mut *self.planes };
    let width = planes[0].cfg.width;
    let height = planes[0].cfg.height;

    let tile_width = self.tile_info.width;
    let tile_height = self.tile_info.height;

    let cols = (width + tile_width - 1) / tile_width;
    let rows = (height + tile_height - 1) / tile_height;

    let consumed = (self.y / tile_height) * rows + (self.x / tile_width);
    let remaining = cols * rows - consumed;

    (remaining, Some(remaining))
  }
}

impl<'a> TileIterMut<'a> {
  unsafe fn next_region(
    plane: *mut Plane,
    luma_x: usize,
    luma_y: usize,
    tile_info: TileInfo,
  ) -> PlaneRegionMut<'a> {
    let plane = &mut *plane;

    let x = luma_x >> plane.cfg.xdec;
    let y = luma_y >> plane.cfg.ydec;
    let w = tile_info.width >> plane.cfg.xdec;
    let h = tile_info.height >> plane.cfg.ydec;

    assert!(plane.cfg.xorigin >= 0);
    assert!(plane.cfg.yorigin >= 0);

    let xorigin = plane.cfg.xorigin as usize + x;
    let yorigin = plane.cfg.yorigin as usize + y;
    let width = w.min(plane.cfg.width - x);
    let height = h.min(plane.cfg.height - y);

    let cfg = PlaneRegionConfig { xorigin, yorigin, width, height };

    PlaneRegionMut::new(plane, cfg)
  }
}

#[cfg(test)]
pub mod test {
  use super::*;

  #[test]
  fn test_tile_count() {
    let mut frame = Frame::new(80, 60, ChromaSampling::Cs420);

    {
      let mut iter = frame.tile_iter_mut(40, 30);
      assert_eq!(4, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(3, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(2, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(1, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(0, iter.size_hint().0);
      assert!(iter.next().is_none());
    }

    {
      let mut iter = frame.tile_iter_mut(32, 24);
      assert_eq!(9, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(8, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(7, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(6, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(5, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(4, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(3, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(2, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(1, iter.size_hint().0);
      assert!(iter.next().is_some());
      assert_eq!(0, iter.size_hint().0);
      assert!(iter.next().is_none());
    }
  }

  #[test]
  fn test_tile_area() {
    let mut frame = Frame::new(72, 68, ChromaSampling::Cs420);

    let planes_origins = [
      (
        frame.planes[0].cfg.xorigin as usize,
        frame.planes[0].cfg.yorigin as usize,
      ),
      (
        frame.planes[1].cfg.xorigin as usize,
        frame.planes[1].cfg.yorigin as usize,
      ),
      (
        frame.planes[2].cfg.xorigin as usize,
        frame.planes[2].cfg.yorigin as usize,
      ),
    ];

    let tiles = frame.tile_iter_mut(32, 32).collect::<Vec<_>>();
    // the frame must be split into 9 tiles:
    //
    //       luma (Y)             chroma (U)            chroma (V)
    //   32x32 32x32  8x32     16x16 16x16  4x16     16x16 16x16  4x16
    //   32x32 32x32  8x32     16x16 16x16  4x16     16x16 16x16  4x16
    //   32x 4 32x 4  8x 4     16x 2 16x 2  4x 2     16x 2 16x 2  4x 2

    assert_eq!(9, tiles.len());

    let tile = &tiles[0]; // the top-left tile
    assert_eq!(32, tile.planes[0].cfg.width);
    assert_eq!(32, tile.planes[0].cfg.height);
    assert_eq!(16, tile.planes[1].cfg.width);
    assert_eq!(16, tile.planes[1].cfg.height);
    assert_eq!(16, tile.planes[2].cfg.width);
    assert_eq!(16, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1, tile.planes[2].cfg.yorigin);

    let tile = &tiles[1]; // the top-middle tile
    assert_eq!(32, tile.planes[0].cfg.width);
    assert_eq!(32, tile.planes[0].cfg.height);
    assert_eq!(16, tile.planes[1].cfg.width);
    assert_eq!(16, tile.planes[1].cfg.height);
    assert_eq!(16, tile.planes[2].cfg.width);
    assert_eq!(16, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0 + 32, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0 + 16, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0 + 16, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1, tile.planes[2].cfg.yorigin);

    let tile = &tiles[2]; // the top-right tile
    assert_eq!(8, tile.planes[0].cfg.width);
    assert_eq!(32, tile.planes[0].cfg.height);
    assert_eq!(4, tile.planes[1].cfg.width);
    assert_eq!(16, tile.planes[1].cfg.height);
    assert_eq!(4, tile.planes[2].cfg.width);
    assert_eq!(16, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0 + 64, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0 + 32, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0 + 32, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1, tile.planes[2].cfg.yorigin);

    let tile = &tiles[3]; // the middle-left tile
    assert_eq!(32, tile.planes[0].cfg.width);
    assert_eq!(32, tile.planes[0].cfg.height);
    assert_eq!(16, tile.planes[1].cfg.width);
    assert_eq!(16, tile.planes[1].cfg.height);
    assert_eq!(16, tile.planes[2].cfg.width);
    assert_eq!(16, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1 + 32, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1 + 16, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1 + 16, tile.planes[2].cfg.yorigin);

    let tile = &tiles[4]; // the center tile
    assert_eq!(32, tile.planes[0].cfg.width);
    assert_eq!(32, tile.planes[0].cfg.height);
    assert_eq!(16, tile.planes[1].cfg.width);
    assert_eq!(16, tile.planes[1].cfg.height);
    assert_eq!(16, tile.planes[2].cfg.width);
    assert_eq!(16, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0 + 32, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1 + 32, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0 + 16, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1 + 16, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0 + 16, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1 + 16, tile.planes[2].cfg.yorigin);

    let tile = &tiles[5]; // the middle-right tile
    assert_eq!(8, tile.planes[0].cfg.width);
    assert_eq!(32, tile.planes[0].cfg.height);
    assert_eq!(4, tile.planes[1].cfg.width);
    assert_eq!(16, tile.planes[1].cfg.height);
    assert_eq!(4, tile.planes[2].cfg.width);
    assert_eq!(16, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0 + 64, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1 + 32, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0 + 32, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1 + 16, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0 + 32, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1 + 16, tile.planes[2].cfg.yorigin);

    let tile = &tiles[6]; // the bottom-left tile
    assert_eq!(32, tile.planes[0].cfg.width);
    assert_eq!(4, tile.planes[0].cfg.height);
    assert_eq!(16, tile.planes[1].cfg.width);
    assert_eq!(2, tile.planes[1].cfg.height);
    assert_eq!(16, tile.planes[2].cfg.width);
    assert_eq!(2, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1 + 64, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1 + 32, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1 + 32, tile.planes[2].cfg.yorigin);

    let tile = &tiles[7]; // the bottom-middle tile
    assert_eq!(32, tile.planes[0].cfg.width);
    assert_eq!(4, tile.planes[0].cfg.height);
    assert_eq!(16, tile.planes[1].cfg.width);
    assert_eq!(2, tile.planes[1].cfg.height);
    assert_eq!(16, tile.planes[2].cfg.width);
    assert_eq!(2, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0 + 32, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1 + 64, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0 + 16, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1 + 32, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0 + 16, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1 + 32, tile.planes[2].cfg.yorigin);

    let tile = &tiles[8]; // the bottom-right tile
    assert_eq!(8, tile.planes[0].cfg.width);
    assert_eq!(4, tile.planes[0].cfg.height);
    assert_eq!(4, tile.planes[1].cfg.width);
    assert_eq!(2, tile.planes[1].cfg.height);
    assert_eq!(4, tile.planes[2].cfg.width);
    assert_eq!(2, tile.planes[2].cfg.height);
    assert_eq!(planes_origins[0].0 + 64, tile.planes[0].cfg.xorigin);
    assert_eq!(planes_origins[0].1 + 64, tile.planes[0].cfg.yorigin);
    assert_eq!(planes_origins[1].0 + 32, tile.planes[1].cfg.xorigin);
    assert_eq!(planes_origins[1].1 + 32, tile.planes[1].cfg.yorigin);
    assert_eq!(planes_origins[2].0 + 32, tile.planes[2].cfg.xorigin);
    assert_eq!(planes_origins[2].1 + 32, tile.planes[2].cfg.yorigin);
  }

  #[test]
  fn test_tile_write() {
    let mut frame = Frame::new(72, 68, ChromaSampling::Cs420);

    {
      let mut tiles = frame.tile_iter_mut(32, 32).collect::<Vec<_>>();

      {
        // row 12 of Y-plane of the top-left tile
        let row = tiles[0].planes[0].row_mut(12);
        assert_eq!(32, row.len());
        &mut row[5..11].copy_from_slice(&[4, 42, 12, 18, 15, 31]);
      }

      {
        // row 8 of U-plane of the middle-right tile
        let row = tiles[5].planes[1].row_mut(8);
        assert_eq!(4, row.len());
        &mut row[..].copy_from_slice(&[14, 121, 1, 3]);
      }

      {
        // row 1 of V-plane of the bottom-middle tile
        let row = tiles[7].planes[2].row_mut(1);
        assert_eq!(16, row.len());
        &mut row[11..16].copy_from_slice(&[6, 5, 2, 11, 8]);
      }
    }

    // check that writes on tiles correctly affected the underlying frame

    let plane = &frame.planes[0];
    let y = plane.cfg.yorigin as usize + 12;
    let x = plane.cfg.xorigin as usize + 5;
    let idx = y * plane.cfg.stride + x;
    assert_eq!(&[4, 42, 12, 18, 15, 31], &plane.data[idx..idx + 6]);

    let plane = &frame.planes[1];
    let offset = (32, 16); // middle-right tile, chroma plane
    let y = plane.cfg.yorigin as usize + offset.1 + 8;
    let x = plane.cfg.xorigin as usize + offset.0;
    let idx = y * plane.cfg.stride + x;
    assert_eq!(&[14, 121, 1, 3], &plane.data[idx..idx + 4]);

    let plane = &frame.planes[2];
    let offset = (16, 32); // bottom-middle tile, chroma plane
    let y = plane.cfg.yorigin as usize + offset.1 + 1;
    let x = plane.cfg.xorigin as usize + offset.0 + 11;
    let idx = y * plane.cfg.stride + x;
    assert_eq!(&[6, 5, 2, 11, 8], &plane.data[idx..idx + 5]);
  }
}
