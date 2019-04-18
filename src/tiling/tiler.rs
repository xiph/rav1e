// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::util::*;

pub const MAX_TILE_WIDTH: usize = 4096;
pub const MAX_TILE_AREA: usize = 4096 * 2304;
pub const MAX_TILE_COLS: usize = 64;
pub const MAX_TILE_ROWS: usize = 64;

/// Tiling information
///
/// This stores everything necessary to split a frame into tiles, and write
/// headers fields into the bitstream.
///
/// The method tile_iter_mut() actually provides tiled views of FrameState
/// and FrameBlocks.
#[derive(Debug, Clone, Copy)]
pub struct TilingInfo {
  pub frame_width: usize,
  pub frame_height: usize,
  pub tile_width_sb: usize,
  pub tile_height_sb: usize,
  pub cols: usize, // number of columns of tiles within the whole frame
  pub rows: usize, // number of rows of tiles within the whole frame
  pub sb_size_log2: usize,
}

impl TilingInfo {
  pub fn new(
    sb_size_log2: usize,
    frame_width: usize,
    frame_height: usize,
    tile_cols_log2: usize,
    tile_rows_log2: usize,
  ) -> Self {
    // <https://aomediacodec.github.io/av1-spec/#tile-info-syntax>

    // Frame::new() aligns to the next multiple of 8
    let frame_width = frame_width.align_power_of_two(3);
    let frame_height = frame_height.align_power_of_two(3);
    let frame_width_sb =
      frame_width.align_power_of_two_and_shift(sb_size_log2);
    let frame_height_sb =
      frame_height.align_power_of_two_and_shift(sb_size_log2);
    let sb_cols = frame_width.align_power_of_two_and_shift(sb_size_log2);
    let sb_rows = frame_height.align_power_of_two_and_shift(sb_size_log2);

    let max_tile_width_sb = MAX_TILE_WIDTH >> sb_size_log2;
    let max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size_log2);
    let min_tile_cols_log2 = Self::tile_log2(max_tile_width_sb, sb_cols);
    let max_tile_cols_log2 = Self::tile_log2(1, sb_cols.min(MAX_TILE_COLS));
    let max_tile_rows_log2 = Self::tile_log2(1, sb_rows.min(MAX_TILE_ROWS));

    let min_tiles_log2 = min_tile_cols_log2
      .max(Self::tile_log2(max_tile_area_sb, sb_cols * sb_rows));

    let tile_cols_log2 =
      tile_cols_log2.max(min_tile_cols_log2).min(max_tile_cols_log2);
    let tile_width_sb = sb_cols.align_power_of_two_and_shift(tile_cols_log2);

    let min_tile_rows_log2 = if min_tiles_log2 > tile_cols_log2 {
      min_tiles_log2 - tile_cols_log2
    } else {
      0
    };
    let tile_rows_log2 =
      tile_rows_log2.max(min_tile_rows_log2).min(max_tile_rows_log2);
    let tile_height_sb = sb_rows.align_power_of_two_and_shift(tile_rows_log2);

    let cols = (frame_width_sb + tile_width_sb - 1) / tile_width_sb;
    let rows = (frame_height_sb + tile_height_sb - 1) / tile_height_sb;

    Self {
      frame_width,
      frame_height,
      tile_width_sb,
      tile_height_sb,
      cols,
      rows,
      sb_size_log2,
    }
  }

  /// Return the smallest value for `k` such that `blkSize << k` is greater than
  /// or equal to `target`.
  ///
  /// <https://aomediacodec.github.io/av1-spec/#tile-size-calculation-function>
  fn tile_log2(blk_size: usize, target: usize) -> usize {
    let mut k = 0;
    while (blk_size << k) < target {
      k += 1;
    }
    k
  }

  #[inline(always)]
  pub fn tile_count(&self) -> usize {
    self.cols * self.rows
  }
}
