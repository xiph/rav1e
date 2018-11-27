// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(safe_extern_statics)]

use encoder::Frame;
use encoder::FrameState;
use encoder::FrameInvariants;
use context::BlockContext;
use context::ContextWriter;
use context::SuperBlockOffset;
use context::PLANES;
use encoder::Sequence;
use plane::PlaneConfig;
use std::cmp;

pub const RESTORATION_TILESIZE_MAX: usize = 256;

pub const RESTORE_NONE: u8 = 0;
pub const RESTORE_SWITCHABLE: u8 = 1;
pub const RESTORE_WIENER: u8 = 2;
pub const RESTORE_SGRPROJ: u8 = 3;

pub const WIENER_TAPS_MIN: [i8; 3] = [ -5, -23, -17 ];
pub const WIENER_TAPS_MID: [i8; 3] = [ 3, -7, 15 ];
pub const WIENER_TAPS_MAX: [i8; 3] = [ 10, 8, 46 ];
pub const WIENER_TAPS_K:   [i8; 3] = [ 1, 2, 3 ];

pub const SGRPROJ_XQD_MIN: [i8; 2] = [ -96, -32 ];
pub const SGRPROJ_XQD_MID: [i8; 2] = [ -32, 31 ];
pub const SGRPROJ_XQD_MAX: [i8; 2] = [ 31, 95 ];
pub const SGRPROJ_PRJ_SUBEXP_K: u8 = 4;
pub const SGRPROJ_PRJ_BITS: u8 = 7;
pub const SGRPROJ_PARAMS_BITS: u8 = 4;
pub const SGRPROJ_PARAMS_RADIUS: [[u8; 2]; 1 << SGRPROJ_PARAMS_BITS] = [
  [2, 1], [2, 1], [2, 1], [2, 1],
  [2, 1], [2, 1], [2, 1], [2, 1],
  [2, 1], [2, 1], [0, 1], [0, 1],
  [0, 1], [0, 1], [2, 0], [2, 0],
];
pub const SGRPROJ_PARAMS_EPS: [[u8; 2]; 1 << SGRPROJ_PARAMS_BITS] = [
  [12,  4], [15,  6], [18,  8], [21,  9],
  [24, 10], [29, 11], [36, 12], [45, 13],
  [56, 14], [68, 15], [ 0,  5], [ 0,  8],
  [ 0, 11], [ 0, 14], [30,  0], [75,  0],
];

#[derive(Copy, Clone)]
pub enum RestorationFilter {
  None,
  Wiener  { coeffs: [[i8; 3]; 2] },
  Sgrproj { set: i8,
            xqd: [i8; 2] },
}

impl RestorationFilter {
  pub fn default() -> RestorationFilter {
    RestorationFilter::None
  }
}

#[derive(Copy, Clone)]
pub struct RestorationUnit {
  pub filter: RestorationFilter,
  pub coded: bool,
}

impl RestorationUnit {
  pub fn default() -> RestorationUnit {
    RestorationUnit {
      filter: RestorationFilter::default(),
      coded: false,
    }
  }
}

#[derive(Clone)]
pub struct RestorationPlane {
  pub parent_cfg: PlaneConfig,
  pub lrf_type: u8,
  pub unit_size: usize,
  pub cols: usize,
  pub rows: usize,
  pub wiener_ref: [[i8; 3]; 2],
  pub sgrproj_ref: [i8; 2],
  pub units: Vec<Vec<RestorationUnit>>
}

#[derive(Clone, Default)]
pub struct RestorationPlaneOffset {
  pub row: usize,
  pub col: usize
}

impl RestorationPlane {
  pub fn new(parent_cfg: &PlaneConfig, lrf_type: u8, unit_size: usize) -> RestorationPlane {
    let PlaneConfig { width, height, .. } = parent_cfg;
    // bolted to superblock size for now
    let cols = cmp::max((width + (unit_size>>1)) / unit_size, 1);
    let rows = cmp::max((height + (unit_size>>1)) / unit_size, 1);
    RestorationPlane {
      parent_cfg: parent_cfg.clone(),
      lrf_type,
      unit_size,
      cols,
      rows,
      wiener_ref: [WIENER_TAPS_MID; 2],
      sgrproj_ref: SGRPROJ_XQD_MID,
      units: vec![vec![RestorationUnit::default(); cols]; rows]
    }
  }

  /// find the restoration unit offset corresponding to the this superblock offset
  /// This encapsulates some minor weirdness due to RU stretching at the frame boundary.
  pub fn restoration_plane_offset(&self, sbo: &SuperBlockOffset) -> RestorationPlaneOffset {
    let po = sbo.plane_offset(&self.parent_cfg);
    RestorationPlaneOffset {
      row: cmp::min((po.y + self.unit_size as isize - 1) / self.unit_size as isize,
                    self.rows as isize - 1) as usize,
      col: cmp::min((po.x + self.unit_size as isize - 1) / self.unit_size as isize,
                    self.cols as isize - 1) as usize
    }
  }

  pub fn restoration_unit(&self, sbo: &SuperBlockOffset) -> &RestorationUnit {
    let rpo = self.restoration_plane_offset(sbo);
    &self.units[rpo.row][rpo.col]
  }

  pub fn restoration_unit_as_mut(&mut self, sbo: &SuperBlockOffset) -> &mut RestorationUnit {
    let rpo = self.restoration_plane_offset(sbo);
    &mut self.units[rpo.row][rpo.col]
  }  
}

#[derive(Clone)]
pub struct RestorationState {
  pub lrf_type: [u8; PLANES],
  pub unit_size: [usize; PLANES],
  pub plane: [RestorationPlane; PLANES]
}

impl RestorationState {
  pub fn new(seq: &Sequence, input: &Frame) -> Self {
    let PlaneConfig { xdec, ydec, .. } = input.planes[1].cfg;

    // Currrently opt for smallest possible restoration unit size
    let lrf_y_shift = if seq.use_128x128_superblock {1} else {2};
    let lrf_uv_shift = lrf_y_shift + if xdec>0 && ydec>0 {1} else {0};
    let lrf_type: [u8; PLANES] = [RESTORE_SWITCHABLE, RESTORE_SWITCHABLE, RESTORE_SWITCHABLE];
    let unit_size: [usize; PLANES] = [RESTORATION_TILESIZE_MAX >> lrf_y_shift,
                                      RESTORATION_TILESIZE_MAX >> lrf_uv_shift,
                                      RESTORATION_TILESIZE_MAX >> lrf_uv_shift];
    RestorationState {
      lrf_type,
      unit_size,
      plane: [RestorationPlane::new(&input.planes[0].cfg, lrf_type[0], unit_size[0]),
              RestorationPlane::new(&input.planes[1].cfg, lrf_type[1], unit_size[1]),
              RestorationPlane::new(&input.planes[2].cfg, lrf_type[2], unit_size[2])]
    }
  }
  
  pub fn restoration_unit(&self, sbo: &SuperBlockOffset, pli: usize) -> &RestorationUnit {
    let rpo = self.plane[pli].restoration_plane_offset(sbo);
    &self.plane[pli].units[rpo.row][rpo.col]
  }

  pub fn restoration_unit_as_mut(&mut self, sbo: &SuperBlockOffset, pli: usize) -> &mut RestorationUnit {
    let rpo = self.plane[pli].restoration_plane_offset(sbo);
    &mut self.plane[pli].units[rpo.row][rpo.col]
  }  

  pub fn lrf_optimize_superblock(&mut self, _sbo: &SuperBlockOffset, _fi: &FrameInvariants,
                             _fs: &FrameState, _cw: &mut ContextWriter,
                             _bit_depth: usize) -> RestorationUnit {
    RestorationUnit::default()
  }

  pub fn lrf_filter_frame(&mut self, _fs: &FrameState, _pre_cdef: &Frame,
                          _bc: &BlockContext, _bit_depth: usize) {
    // stub
  }
}



