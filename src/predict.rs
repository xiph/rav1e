// Copyright (c) 2017-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]

cfg_if::cfg_if! {
  if #[cfg(nasm_x86_64)] {
    pub use crate::asm::x86::predict::*;
  } else if #[cfg(asm_neon)] {
    pub use crate::asm::aarch64::predict::*;
  } else {
    pub use self::native::*;
  }
}

use crate::context::{INTRA_MODES, MAX_TX_SIZE};
use crate::cpu_features::CpuFeatureLevel;
use crate::encoder::FrameInvariants;
use crate::frame::*;
use crate::mc::*;
use crate::partition::*;
use crate::tiling::*;
use crate::transform::*;
use crate::util::*;

pub static RAV1E_INTRA_MODES: &[PredictionMode] = &[
  PredictionMode::DC_PRED,
  PredictionMode::H_PRED,
  PredictionMode::V_PRED,
  PredictionMode::SMOOTH_PRED,
  PredictionMode::SMOOTH_H_PRED,
  PredictionMode::SMOOTH_V_PRED,
  PredictionMode::PAETH_PRED,
  PredictionMode::D45_PRED,
  PredictionMode::D135_PRED,
  PredictionMode::D117_PRED,
  PredictionMode::D153_PRED,
  PredictionMode::D207_PRED,
  PredictionMode::D63_PRED,
];

pub static RAV1E_INTER_MODES_MINIMAL: &[PredictionMode] =
  &[PredictionMode::NEARESTMV];

pub static RAV1E_INTER_COMPOUND_MODES: &[PredictionMode] = &[
  PredictionMode::GLOBAL_GLOBALMV,
  PredictionMode::NEAREST_NEARESTMV,
  PredictionMode::NEW_NEWMV,
  PredictionMode::NEAREST_NEWMV,
  PredictionMode::NEW_NEARESTMV,
  PredictionMode::NEAR_NEARMV,
];

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Eq, Ord)]
pub enum PredictionMode {
  DC_PRED,     // Average of above and left pixels
  V_PRED,      // Vertical
  H_PRED,      // Horizontal
  D45_PRED,    // Directional 45  deg = round(arctan(1/1) * 180/pi)
  D135_PRED,   // Directional 135 deg = 180 - 45
  D117_PRED,   // Directional 117 deg = 180 - 63
  D153_PRED,   // Directional 153 deg = 180 - 27
  D207_PRED,   // Directional 207 deg = 180 + 27
  D63_PRED,    // Directional 63  deg = round(arctan(2/1) * 180/pi)
  SMOOTH_PRED, // Combination of horizontal and vertical interpolation
  SMOOTH_V_PRED,
  SMOOTH_H_PRED,
  PAETH_PRED,
  UV_CFL_PRED,
  NEARESTMV,
  NEAR0MV,
  NEAR1MV,
  NEAR2MV,
  GLOBALMV,
  NEWMV,
  // Compound ref compound modes
  NEAREST_NEARESTMV,
  NEAR_NEARMV,
  NEAREST_NEWMV,
  NEW_NEARESTMV,
  NEAR_NEWMV,
  NEW_NEARMV,
  GLOBAL_GLOBALMV,
  NEW_NEWMV,
}

#[derive(Copy, Clone, Debug)]
pub enum PredictionVariant {
  NONE,
  LEFT,
  TOP,
  BOTH,
}

impl PredictionVariant {
  fn new(x: usize, y: usize) -> Self {
    match (x, y) {
      (0, 0) => PredictionVariant::NONE,
      (_, 0) => PredictionVariant::LEFT,
      (0, _) => PredictionVariant::TOP,
      _ => PredictionVariant::BOTH,
    }
  }
}

impl Default for PredictionMode {
  fn default() -> Self {
    PredictionMode::DC_PRED
  }
}

impl PredictionMode {
  pub fn is_compound(self) -> bool {
    self >= PredictionMode::NEAREST_NEARESTMV
  }
  pub fn has_near(self) -> bool {
    (self >= PredictionMode::NEAR0MV && self <= PredictionMode::NEAR2MV)
      || self == PredictionMode::NEAR_NEARMV
      || self == PredictionMode::NEAR_NEWMV
      || self == PredictionMode::NEW_NEARMV
  }
  pub fn predict_intra<T: Pixel>(
    self, tile_rect: TileRect, dst: &mut PlaneRegionMut<'_, T>,
    tx_size: TxSize, bit_depth: usize, ac: &[i16], alpha: i16,
    edge_buf: &AlignedArray<[T; 4 * MAX_TX_SIZE + 1]>, cpu: CpuFeatureLevel,
  ) {
    assert!(self.is_intra());
    let &Rect { x: frame_x, y: frame_y, .. } = dst.rect();
    debug_assert!(frame_x >= 0 && frame_y >= 0);
    // x and y are expressed relative to the tile
    let x = frame_x as usize - tile_rect.x;
    let y = frame_y as usize - tile_rect.y;

    let variant = PredictionVariant::new(x, y);

    let mode = match self {
      PredictionMode::PAETH_PRED => match variant {
        PredictionVariant::NONE => PredictionMode::DC_PRED,
        PredictionVariant::LEFT => PredictionMode::H_PRED,
        PredictionVariant::TOP => PredictionMode::V_PRED,
        PredictionVariant::BOTH => PredictionMode::PAETH_PRED,
      },
      PredictionMode::UV_CFL_PRED if alpha == 0 => PredictionMode::DC_PRED,
      _ => self,
    };

    let angle = match mode {
      PredictionMode::UV_CFL_PRED => alpha as isize,
      PredictionMode::D45_PRED => 45,
      PredictionMode::D135_PRED => 135,
      PredictionMode::D117_PRED => 113,
      PredictionMode::D153_PRED => 157,
      PredictionMode::D207_PRED => 203,
      PredictionMode::D63_PRED => 67,
      _ => 0,
    };

    dispatch_predict_intra::<T>(
      mode, variant, dst, tx_size, bit_depth, ac, angle, edge_buf, cpu,
    );
  }

  pub fn is_intra(self) -> bool {
    self < PredictionMode::NEARESTMV
  }

  pub fn is_cfl(self) -> bool {
    self == PredictionMode::UV_CFL_PRED
  }

  pub fn is_directional(self) -> bool {
    self >= PredictionMode::V_PRED && self <= PredictionMode::D63_PRED
  }

  pub fn predict_inter<T: Pixel>(
    self, fi: &FrameInvariants<T>, tile_rect: TileRect, p: usize,
    po: PlaneOffset, dst: &mut PlaneRegionMut<'_, T>, width: usize,
    height: usize, ref_frames: [RefType; 2], mvs: [MotionVector; 2],
  ) {
    assert!(!self.is_intra());
    let frame_po = tile_rect.to_frame_plane_offset(po);

    let mode = fi.default_filter;
    let is_compound = ref_frames[1] != RefType::INTRA_FRAME
      && ref_frames[1] != RefType::NONE_FRAME;

    fn get_params<'a, T: Pixel>(
      rec_plane: &'a Plane<T>, po: PlaneOffset, mv: MotionVector,
    ) -> (i32, i32, PlaneSlice<'a, T>) {
      let rec_cfg = &rec_plane.cfg;
      let shift_row = 3 + rec_cfg.ydec;
      let shift_col = 3 + rec_cfg.xdec;
      let row_offset = mv.row as i32 >> shift_row;
      let col_offset = mv.col as i32 >> shift_col;
      let row_frac =
        (mv.row as i32 - (row_offset << shift_row)) << (4 - shift_row);
      let col_frac =
        (mv.col as i32 - (col_offset << shift_col)) << (4 - shift_col);
      let qo = PlaneOffset {
        x: po.x + col_offset as isize - 3,
        y: po.y + row_offset as isize - 3,
      };
      (row_frac, col_frac, rec_plane.slice(qo).clamp().subslice(3, 3))
    };

    if !is_compound {
      if let Some(ref rec) =
        fi.rec_buffer.frames[fi.ref_frames[ref_frames[0].to_index()] as usize]
      {
        let (row_frac, col_frac, src) =
          get_params(&rec.frame.planes[p], frame_po, mvs[0]);
        put_8tap(
          dst,
          src,
          width,
          height,
          col_frac,
          row_frac,
          mode,
          mode,
          fi.sequence.bit_depth,
          fi.cpu_feature_level,
        );
      }
    } else {
      let mut tmp: [AlignedArray<[i16; 128 * 128]>; 2] =
        [AlignedArray::uninitialized(), AlignedArray::uninitialized()];
      for i in 0..2 {
        if let Some(ref rec) = fi.rec_buffer.frames
          [fi.ref_frames[ref_frames[i].to_index()] as usize]
        {
          let (row_frac, col_frac, src) =
            get_params(&rec.frame.planes[p], frame_po, mvs[i]);
          prep_8tap(
            &mut tmp[i].array,
            src,
            width,
            height,
            col_frac,
            row_frac,
            mode,
            mode,
            fi.sequence.bit_depth,
            fi.cpu_feature_level,
          );
        }
      }
      mc_avg(
        dst,
        &tmp[0].array,
        &tmp[1].array,
        width,
        height,
        fi.sequence.bit_depth,
        fi.cpu_feature_level,
      );
    }
  }
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum InterIntraMode {
  II_DC_PRED,
  II_V_PRED,
  II_H_PRED,
  II_SMOOTH_PRED,
  INTERINTRA_MODES,
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum CompoundType {
  COMPOUND_AVERAGE,
  COMPOUND_WEDGE,
  COMPOUND_DIFFWTD,
  COMPOUND_TYPES,
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum MotionMode {
  SIMPLE_TRANSLATION,
  OBMC_CAUSAL,   // 2-sided OBMC
  WARPED_CAUSAL, // 2-sided WARPED
  MOTION_MODES,
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum PaletteSize {
  TWO_COLORS,
  THREE_COLORS,
  FOUR_COLORS,
  FIVE_COLORS,
  SIX_COLORS,
  SEVEN_COLORS,
  EIGHT_COLORS,
  PALETTE_SIZES,
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum PaletteColor {
  PALETTE_COLOR_ONE,
  PALETTE_COLOR_TWO,
  PALETTE_COLOR_THREE,
  PALETTE_COLOR_FOUR,
  PALETTE_COLOR_FIVE,
  PALETTE_COLOR_SIX,
  PALETTE_COLOR_SEVEN,
  PALETTE_COLOR_EIGHT,
  PALETTE_COLORS,
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum FilterIntraMode {
  FILTER_DC_PRED,
  FILTER_V_PRED,
  FILTER_H_PRED,
  FILTER_D157_PRED,
  FILTER_PAETH_PRED,
  FILTER_INTRA_MODES,
}

// Weights are quadratic from '1' to '1 / block_size', scaled by 2^sm_weight_log2_scale.
const sm_weight_log2_scale: u8 = 8;

// Smooth predictor weights
#[rustfmt::skip]
static sm_weight_arrays: [u8; 2 * MAX_TX_SIZE] = [
    // Unused, because we always offset by bs, which is at least 2.
    0, 0,
    // bs = 2
    255, 128,
    // bs = 4
    255, 149, 85, 64,
    // bs = 8
    255, 197, 146, 105, 73, 50, 37, 32,
    // bs = 16
    255, 225, 196, 170, 145, 123, 102, 84, 68, 54, 43, 33, 26, 20, 17, 16,
    // bs = 32
    255, 240, 225, 210, 196, 182, 169, 157, 145, 133, 122, 111, 101, 92, 83, 74,
    66, 59, 52, 45, 39, 34, 29, 25, 21, 17, 14, 12, 10, 9, 8, 8,
    // bs = 64
    255, 248, 240, 233, 225, 218, 210, 203, 196, 189, 182, 176, 169, 163, 156,
    150, 144, 138, 133, 127, 121, 116, 111, 106, 101, 96, 91, 86, 82, 77, 73, 69,
    65, 61, 57, 54, 50, 47, 44, 41, 38, 35, 32, 29, 27, 25, 22, 20, 18, 16, 15,
    13, 12, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 4,
];

const NEED_LEFT: u8 = 1 << 1;
const NEED_ABOVE: u8 = 1 << 2;
const NEED_ABOVERIGHT: u8 = 1 << 3;
const NEED_ABOVELEFT: u8 = 1 << 4;
#[allow(unused)]
const NEED_BOTTOMLEFT: u8 = 1 << 5;

/*const INTRA_EDGE_FILT: usize = 3;
const INTRA_EDGE_TAPS: usize = 5;
const MAX_UPSAMPLE_SZ: usize = 16;*/

#[allow(unused)]
pub static extend_modes: [u8; INTRA_MODES] = [
  NEED_ABOVE | NEED_LEFT,                  // DC
  NEED_ABOVE,                              // V
  NEED_LEFT,                               // H
  NEED_ABOVE | NEED_ABOVERIGHT,            // D45
  NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT, // D135
  NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT, // D113
  NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT, // D157
  NEED_LEFT | NEED_BOTTOMLEFT,             // D203
  NEED_ABOVE | NEED_ABOVERIGHT,            // D67
  NEED_LEFT | NEED_ABOVE,                  // SMOOTH
  NEED_LEFT | NEED_ABOVE,                  // SMOOTH_V
  NEED_LEFT | NEED_ABOVE,                  // SMOOTH_H
  NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT, // PAETH
];

#[inline(always)]
fn get_scaled_luma_q0(alpha_q3: i16, ac_pred_q3: i16) -> i32 {
  let scaled_luma_q6 = (alpha_q3 as i32) * (ac_pred_q3 as i32);
  let abs_scaled_luma_q0 = (scaled_luma_q6.abs() + 32) >> 6;
  if scaled_luma_q6 < 0 {
    -abs_scaled_luma_q0
  } else {
    abs_scaled_luma_q0
  }
}

pub(crate) mod native {
  use super::*;
  use crate::context::MAX_TX_SIZE;
  use crate::cpu_features::CpuFeatureLevel;
  use crate::tiling::PlaneRegionMut;
  use crate::transform::TxSize;
  use crate::util::round_shift;
  use crate::util::AlignedArray;
  use crate::Pixel;
  use std::mem::size_of;

  #[inline(always)]
  pub fn dispatch_predict_intra<T: Pixel>(
    mode: PredictionMode, variant: PredictionVariant,
    dst: &mut PlaneRegionMut<'_, T>, tx_size: TxSize, bit_depth: usize,
    ac: &[i16], angle: isize,
    edge_buf: &AlignedArray<[T; 4 * MAX_TX_SIZE + 1]>, _cpu: CpuFeatureLevel,
  ) {
    let width = tx_size.width();
    let height = tx_size.height();

    // left pixels are order from bottom to top and right-aligned
    let (left, not_left) = edge_buf.array.split_at(2 * MAX_TX_SIZE);
    let (top_left, above) = not_left.split_at(1);

    let above_slice = &above[..width + height];
    let left_slice = &left[2 * MAX_TX_SIZE - height..];
    let left_and_left_below_slice = &left[2 * MAX_TX_SIZE - height - width..];

    match mode {
      PredictionMode::DC_PRED => {
        (match variant {
          PredictionVariant::NONE => pred_dc_128,
          PredictionVariant::LEFT => pred_dc_left,
          PredictionVariant::TOP => pred_dc_top,
          PredictionVariant::BOTH => pred_dc,
        })(dst, above_slice, left_slice, width, height, bit_depth)
      }
      PredictionMode::UV_CFL_PRED => (match variant {
        PredictionVariant::NONE => pred_cfl_128,
        PredictionVariant::LEFT => pred_cfl_left,
        PredictionVariant::TOP => pred_cfl_top,
        PredictionVariant::BOTH => pred_cfl,
      })(
        dst,
        &ac,
        angle as i16,
        above_slice,
        left_slice,
        width,
        height,
        bit_depth,
      ),
      PredictionMode::H_PRED => pred_h(dst, left_slice, width, height),
      PredictionMode::V_PRED => pred_v(dst, above_slice, width, height),
      PredictionMode::PAETH_PRED => {
        pred_paeth(dst, above_slice, left_slice, top_left[0], width, height)
      }
      PredictionMode::SMOOTH_PRED => {
        pred_smooth(dst, above_slice, left_slice, width, height)
      }
      PredictionMode::SMOOTH_H_PRED => {
        pred_smooth_h(dst, above_slice, left_slice, width, height)
      }
      PredictionMode::SMOOTH_V_PRED => {
        pred_smooth_v(dst, above_slice, left_slice, width, height)
      }
      PredictionMode::D45_PRED
      | PredictionMode::D135_PRED
      | PredictionMode::D117_PRED
      | PredictionMode::D153_PRED
      | PredictionMode::D207_PRED
      | PredictionMode::D63_PRED => pred_directional(
        dst,
        above_slice,
        left_and_left_below_slice,
        top_left,
        angle as usize,
        width,
        height,
        bit_depth,
      ),
      _ => unimplemented!(),
    }
  }

  pub(crate) fn pred_dc<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T], width: usize,
    height: usize, _bit_depth: usize,
  ) {
    let edges = left[..height].iter().chain(above[..width].iter());
    let len = (width + height) as u32;
    let avg = (edges.fold(0u32, |acc, &v| {
      let v: u32 = v.into();
      v + acc
    }) + (len >> 1))
      / len;
    let avg = T::cast_from(avg);

    for line in output.rows_iter_mut().take(height) {
      for v in &mut line[..width] {
        *v = avg;
      }
    }
  }

  pub(crate) fn pred_dc_128<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>, _above: &[T], _left: &[T],
    width: usize, height: usize, bit_depth: usize,
  ) {
    let v = T::cast_from(128u32 << (bit_depth - 8));
    for y in 0..height {
      for x in 0..width {
        output[y][x] = v;
      }
    }
  }

  pub(crate) fn pred_dc_left<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>, _above: &[T], left: &[T],
    width: usize, height: usize, _bit_depth: usize,
  ) {
    let sum = left[..].iter().fold(0u32, |acc, &v| {
      let v: u32 = v.into();
      v + acc
    });
    let avg = T::cast_from((sum + (height >> 1) as u32) / height as u32);
    for line in output.rows_iter_mut().take(height) {
      line[..width].iter_mut().for_each(|v| *v = avg);
    }
  }

  pub(crate) fn pred_dc_top<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>, above: &[T], _left: &[T],
    width: usize, height: usize, _bit_depth: usize,
  ) {
    let sum = above[..width].iter().fold(0u32, |acc, &v| {
      let v: u32 = v.into();
      v + acc
    });
    let avg = T::cast_from((sum + (width >> 1) as u32) / width as u32);
    for line in output.rows_iter_mut().take(height) {
      line[..width].iter_mut().for_each(|v| *v = avg);
    }
  }

  pub(crate) fn pred_h<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>, left: &[T], width: usize,
    height: usize,
  ) {
    for (line, l) in output.rows_iter_mut().zip(left[..height].iter().rev()) {
      for v in &mut line[..width] {
        *v = *l;
      }
    }
  }

  pub(crate) fn pred_v<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>, above: &[T], width: usize,
    height: usize,
  ) {
    for line in output.rows_iter_mut().take(height) {
      line[..width].clone_from_slice(&above[..width])
    }
  }

  pub(crate) fn pred_paeth<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T],
    above_left: T, width: usize, height: usize,
  ) {
    for r in 0..height {
      let row = &mut output[r];
      for c in 0..width {
        // Top-left pixel is fixed in libaom
        let raw_top_left: i32 = above_left.into();
        let raw_left: i32 = left[height - 1 - r].into();
        let raw_top: i32 = above[c].into();

        let p_base = raw_top + raw_left - raw_top_left;
        let p_left = (p_base - raw_left).abs();
        let p_top = (p_base - raw_top).abs();
        let p_top_left = (p_base - raw_top_left).abs();

        // Return nearest to base of left, top and top_left
        if p_left <= p_top && p_left <= p_top_left {
          row[c] = T::cast_from(raw_left);
        } else if p_top <= p_top_left {
          row[c] = T::cast_from(raw_top);
        } else {
          row[c] = T::cast_from(raw_top_left);
        }
      }
    }
  }

  pub(crate) fn pred_smooth<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T], width: usize,
    height: usize,
  ) {
    let below_pred = left[0]; // estimated by bottom-left pixel
    let right_pred = above[width - 1]; // estimated by top-right pixel
    let sm_weights_w = &sm_weight_arrays[width..];
    let sm_weights_h = &sm_weight_arrays[height..];

    let log2_scale = 1 + sm_weight_log2_scale;
    let scale = 1_u16 << sm_weight_log2_scale;

    // Weights sanity checks
    assert!((sm_weights_w[0] as u16) < scale);
    assert!((sm_weights_h[0] as u16) < scale);
    assert!((scale - sm_weights_w[width - 1] as u16) < scale);
    assert!((scale - sm_weights_h[height - 1] as u16) < scale);
    // ensures no overflow when calculating predictor
    assert!(log2_scale as usize + size_of::<T>() < 31);

    for r in 0..height {
      let row = &mut output[r];
      for c in 0..width {
        let pixels = [above[c], below_pred, left[height - 1 - r], right_pred];

        let weights = [
          sm_weights_h[r] as u16,
          scale - sm_weights_h[r] as u16,
          sm_weights_w[c] as u16,
          scale - sm_weights_w[c] as u16,
        ];

        assert!(
          scale >= (sm_weights_h[r] as u16)
            && scale >= (sm_weights_w[c] as u16)
        );

        // Sum up weighted pixels
        let mut this_pred: u32 = weights
          .iter()
          .zip(pixels.iter())
          .map(|(w, p)| {
            let p: u32 = (*p).into();
            (*w as u32) * p
          })
          .sum();
        this_pred = (this_pred + (1 << (log2_scale - 1))) >> log2_scale;

        row[c] = T::cast_from(this_pred);
      }
    }
  }

  pub(crate) fn pred_smooth_h<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T], width: usize,
    height: usize,
  ) {
    let right_pred = above[width - 1]; // estimated by top-right pixel
    let sm_weights = &sm_weight_arrays[width..];

    let log2_scale = sm_weight_log2_scale;
    let scale = 1_u16 << sm_weight_log2_scale;

    // Weights sanity checks
    assert!((sm_weights[0] as u16) < scale);
    assert!((scale - sm_weights[width - 1] as u16) < scale);
    // ensures no overflow when calculating predictor
    assert!(log2_scale as usize + size_of::<T>() < 31);

    for r in 0..height {
      let row = &mut output[r];
      for c in 0..width {
        let pixels = [left[height - 1 - r], right_pred];
        let weights = [sm_weights[c] as u16, scale - sm_weights[c] as u16];

        assert!(scale >= sm_weights[c] as u16);

        let mut this_pred: u32 = weights
          .iter()
          .zip(pixels.iter())
          .map(|(w, p)| {
            let p: u32 = (*p).into();
            (*w as u32) * p
          })
          .sum();
        this_pred = (this_pred + (1 << (log2_scale - 1))) >> log2_scale;

        row[c] = T::cast_from(this_pred);
      }
    }
  }

  pub(crate) fn pred_smooth_v<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T], width: usize,
    height: usize,
  ) {
    let below_pred = left[0]; // estimated by bottom-left pixel
    let sm_weights = &sm_weight_arrays[height..];

    let log2_scale = sm_weight_log2_scale;
    let scale = 1_u16 << sm_weight_log2_scale;

    // Weights sanity checks
    assert!((sm_weights[0] as u16) < scale);
    assert!((scale - sm_weights[height - 1] as u16) < scale);
    // ensures no overflow when calculating predictor
    assert!(log2_scale as usize + size_of::<T>() < 31);

    for r in 0..height {
      let row = &mut output[r];
      for c in 0..width {
        let pixels = [above[c], below_pred];
        let weights = [sm_weights[r] as u16, scale - sm_weights[r] as u16];

        assert!(scale >= sm_weights[r] as u16);

        let mut this_pred: u32 = weights
          .iter()
          .zip(pixels.iter())
          .map(|(w, p)| {
            let p: u32 = (*p).into();
            (*w as u32) * p
          })
          .sum();
        this_pred = (this_pred + (1 << (log2_scale - 1))) >> log2_scale;

        row[c] = T::cast_from(this_pred);
      }
    }
  }

  pub(crate) fn pred_cfl_inner<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>, ac: &[i16], alpha: i16, width: usize,
    height: usize, bit_depth: usize,
  ) {
    if alpha == 0 {
      return;
    }
    assert!(32 >= width);
    assert!(ac.len() >= 32 * (height - 1) + width);
    assert!(output.plane_cfg.stride >= width);
    assert!(output.rows_iter().len() >= height);

    let sample_max = (1 << bit_depth) - 1;
    let avg: i32 = output[0][0].into();

    for (line, luma) in
      output.rows_iter_mut().zip(ac.chunks(width)).take(height)
    {
      for (v, &l) in line[..width].iter_mut().zip(luma[..width].iter()) {
        *v = T::cast_from(
          (avg + get_scaled_luma_q0(alpha, l)).max(0).min(sample_max),
        );
      }
    }
  }

  pub(crate) fn pred_cfl<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>, ac: &[i16], alpha: i16, above: &[T],
    left: &[T], width: usize, height: usize, bit_depth: usize,
  ) {
    pred_dc(output, above, left, width, height, bit_depth);
    pred_cfl_inner(output, &ac, alpha, width, height, bit_depth);
  }

  pub(crate) fn pred_cfl_128<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>, ac: &[i16], alpha: i16, above: &[T],
    left: &[T], width: usize, height: usize, bit_depth: usize,
  ) {
    pred_dc_128(output, above, left, width, height, bit_depth);
    pred_cfl_inner(output, &ac, alpha, width, height, bit_depth);
  }

  pub(crate) fn pred_cfl_left<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>, ac: &[i16], alpha: i16, above: &[T],
    left: &[T], width: usize, height: usize, bit_depth: usize,
  ) {
    pred_dc_left(output, above, left, width, height, bit_depth);
    pred_cfl_inner(output, &ac, alpha, width, height, bit_depth);
  }

  pub(crate) fn pred_cfl_top<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>, ac: &[i16], alpha: i16, above: &[T],
    left: &[T], width: usize, height: usize, bit_depth: usize,
  ) {
    pred_dc_top(output, above, left, width, height, bit_depth);
    pred_cfl_inner(output, &ac, alpha, width, height, bit_depth);
  }

  pub(crate) fn pred_directional<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T],
    top_left: &[T], angle: usize, width: usize, height: usize,
    bit_depth: usize,
  ) {
    let sample_max = ((1 << bit_depth) - 1) as i32;
    let _angle_delta = 0;

    let p_angle = angle; // TODO use Mode_to_Angle

    let upsample_above = 0;
    let upsample_left = 0;

    let enable_intra_edge_filter = false; // FIXME

    if enable_intra_edge_filter {
      // TODO
    }

    fn dr_intra_derivative(p_angle: usize) -> usize {
      match p_angle {
        4 => 1023,
        7 => 547,
        10 => 372,
        14 => 273,
        17 => 215,
        20 => 178,
        23 => 151,
        26 => 132,
        29 => 116,
        32 => 102,
        36 => 90,
        39 => 80,
        42 => 71,
        45 => 64,
        48 => 57,
        51 => 51,
        54 => 45,
        58 => 40,
        61 => 35,
        64 => 31,
        67 => 27,
        70 => 23,
        73 => 19,
        76 => 15,
        81 => 11,
        84 => 7,
        87 => 3,
        _ => 0,
      }
    }

    let dx = if p_angle < 90 {
      dr_intra_derivative(p_angle)
    } else if p_angle > 90 && p_angle < 180 {
      dr_intra_derivative(180 - p_angle)
    } else {
      0 // undefined
    };

    let dy = if p_angle > 90 && p_angle < 180 {
      dr_intra_derivative(p_angle - 90)
    } else if p_angle > 180 {
      dr_intra_derivative(270 - p_angle)
    } else {
      0 // undefined
    };

    if p_angle < 90 {
      for i in 0..height {
        let row = &mut output[i];
        for j in 0..width {
          let idx = (i + 1) * dx;
          let base = (idx >> (6 - upsample_above)) + (j << upsample_above);
          let shift = (((idx << upsample_above) >> 1) & 31) as i32;
          let max_base_x = (height + width - 1) << upsample_above;
          let v = if base < max_base_x {
            let a: i32 = above[base].into();
            let b: i32 = above[base + 1].into();
            round_shift(a * (32 - shift) + b * shift, 5)
          } else {
            let c: i32 = above[max_base_x].into();
            c
          }
          .max(0)
          .min(sample_max);
          row[j] = T::cast_from(v);
        }
      }
    } else if p_angle > 90 && p_angle < 180 {
      for i in 0..height {
        let row = &mut output[i];
        for j in 0..width {
          let idx = (j << 6) as isize - ((i + 1) * dx) as isize;
          let base = idx >> (6 - upsample_above);
          if base >= -(1 << upsample_above) {
            let shift = (((idx << upsample_above) >> 1) & 31) as i32;
            let a: i32 =
              if base < 0 { top_left[0] } else { above[base as usize] }.into();
            let b: i32 = above[(base + 1) as usize].into();
            let v = round_shift(a * (32 - shift) + b * shift, 5)
              .max(0)
              .min(sample_max);
            row[j] = T::cast_from(v);
          } else {
            let idx = (i << 6) as isize - ((j + 1) * dy) as isize;
            let base = idx >> (6 - upsample_left);
            let shift = (((idx << upsample_left) >> 1) & 31) as i32;
            let a: i32 = if base < 0 {
              top_left[0]
            } else {
              left[width + height - 1 - base as usize]
            }
            .into();
            let b: i32 = left[width + height - (2 + base) as usize].into();
            let v = round_shift(a * (32 - shift) + b * shift, 5)
              .max(0)
              .min(sample_max);
            row[j] = T::cast_from(v);
          }
        }
      }
    } else if p_angle > 180 {
      for i in 0..height {
        let row = &mut output[i];
        for j in 0..width {
          let idx = (j + 1) * dy;
          let base = (idx >> (6 - upsample_left)) + (i << upsample_left);
          let shift = (((idx << upsample_left) >> 1) & 31) as i32;
          let a: i32 = left[width + height - 1 - base].into();
          let b: i32 = left[width + height - 2 - base].into();
          let v = round_shift(a * (32 - shift) + b * shift, 5)
            .max(0)
            .min(sample_max);
          row[j] = T::cast_from(v);
        }
      }
    }
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::frame::AsRegion;
  use crate::predict::native::*;
  use num_traits::*;

  #[test]
  fn pred_matches_u8() {
    let mut edge_buf: AlignedArray<[u8; 2 * MAX_TX_SIZE + 1]> =
      AlignedArray::uninitialized();
    for i in 0..edge_buf.array.len() {
      edge_buf.array[i] = (i + 32).saturating_sub(MAX_TX_SIZE).as_();
    }
    let left = &edge_buf.array[MAX_TX_SIZE - 4..MAX_TX_SIZE];
    let above = &edge_buf.array[MAX_TX_SIZE + 1..MAX_TX_SIZE + 5];
    let top_left = edge_buf.array[MAX_TX_SIZE];

    let mut output = Plane::wrap(vec![0u8; 4 * 4], 4);

    pred_dc(&mut output.as_region_mut(), above, left, 4, 4, 8);
    assert_eq!(&output.data[..], [32u8; 16]);

    pred_dc_top(&mut output.as_region_mut(), above, left, 4, 4, 8);
    assert_eq!(&output.data[..], [35u8; 16]);

    pred_dc_left(&mut output.as_region_mut(), above, left, 4, 4, 8);
    assert_eq!(&output.data[..], [30u8; 16]);

    pred_dc_128(&mut output.as_region_mut(), above, left, 4, 4, 8);
    assert_eq!(&output.data[..], [128u8; 16]);

    pred_v(&mut output.as_region_mut(), above, 4, 4);
    assert_eq!(
      &output.data[..],
      [33, 34, 35, 36, 33, 34, 35, 36, 33, 34, 35, 36, 33, 34, 35, 36]
    );

    pred_h(&mut output.as_region_mut(), left, 4, 4);
    assert_eq!(
      &output.data[..],
      [31, 31, 31, 31, 30, 30, 30, 30, 29, 29, 29, 29, 28, 28, 28, 28]
    );

    pred_paeth(&mut output.as_region_mut(), above, left, top_left, 4, 4);
    assert_eq!(
      &output.data[..],
      [32, 34, 35, 36, 30, 32, 32, 36, 29, 32, 32, 32, 28, 28, 32, 32]
    );

    pred_smooth(&mut output.as_region_mut(), above, left, 4, 4);
    assert_eq!(
      &output.data[..],
      [32, 34, 35, 35, 30, 32, 33, 34, 29, 31, 32, 32, 29, 30, 32, 32]
    );

    pred_smooth_h(&mut output.as_region_mut(), above, left, 4, 4);
    assert_eq!(
      &output.data[..],
      [31, 33, 34, 35, 30, 33, 34, 35, 29, 32, 34, 34, 28, 31, 33, 34]
    );

    pred_smooth_v(&mut output.as_region_mut(), above, left, 4, 4);
    assert_eq!(
      &output.data[..],
      [33, 34, 35, 36, 31, 31, 32, 33, 30, 30, 30, 31, 29, 30, 30, 30]
    );
  }

  #[test]
  fn pred_max() {
    let max12bit = 4096 - 1;
    let above = [max12bit; 32];
    let left = [max12bit; 32];

    let mut o = Plane::wrap(vec![0u16; 32 * 32], 32);

    pred_dc(&mut o.as_region_mut(), &above[..4], &left[..4], 4, 4, 16);

    for l in o.data.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    pred_h(&mut o.as_region_mut(), &left[..4], 4, 4);

    for l in o.data.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    pred_v(&mut o.as_region_mut(), &above[..4], 4, 4);

    for l in o.data.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    let above_left = unsafe { *above.as_ptr().offset(-1) };

    pred_paeth(
      &mut o.as_region_mut(),
      &above[..4],
      &left[..4],
      above_left,
      4,
      4,
    );

    for l in o.data.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    pred_smooth(&mut o.as_region_mut(), &above[..4], &left[..4], 4, 4);

    for l in o.data.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    pred_smooth_h(&mut o.as_region_mut(), &above[..4], &left[..4], 4, 4);

    for l in o.data.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    pred_smooth_v(&mut o.as_region_mut(), &above[..4], &left[..4], 4, 4);

    for l in o.data.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }
  }
}
