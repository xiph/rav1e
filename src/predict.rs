// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
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

use crate::context::{INTRA_MODES, MAX_TX_SIZE};
use crate::encoder::FrameInvariants;
use crate::mc::*;
use crate::partition::*;
use crate::frame::*;
use crate::tiling::*;
use crate::transform::*;
use crate::util::*;

#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
use libc;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::mem::*;
#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
use std::ptr;

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

pub static RAV1E_INTER_MODES_MINIMAL: &[PredictionMode] = &[
  PredictionMode::NEARESTMV
];

pub static RAV1E_INTER_COMPOUND_MODES: &[PredictionMode] = &[
  PredictionMode::GLOBAL_GLOBALMV,
  PredictionMode::NEAREST_NEARESTMV,
  PredictionMode::NEW_NEWMV,
  PredictionMode::NEAREST_NEWMV,
  PredictionMode::NEW_NEARESTMV,
  PredictionMode::NEAR_NEARMV,
];

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
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
  NEW_NEWMV
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
    self, tile_rect: TileRect, dst: &mut PlaneRegionMut<'_, T>, tx_size: TxSize, bit_depth: usize,
    ac: &[i16], alpha: i16, edge_buf: &AlignedArray<[T; 4 * MAX_TX_SIZE + 1]>
  ) {
    assert!(self.is_intra());

    match tx_size {
      TxSize::TX_4X4 =>
        self.predict_intra_inner::<Block4x4, _>(tile_rect, dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_8X8 =>
        self.predict_intra_inner::<Block8x8, _>(tile_rect, dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_16X16 =>
        self.predict_intra_inner::<Block16x16, _>(tile_rect, dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_32X32 =>
        self.predict_intra_inner::<Block32x32, _>(tile_rect, dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_64X64 =>
        self.predict_intra_inner::<Block64x64, _>(tile_rect, dst, bit_depth, ac, alpha, edge_buf),

      TxSize::TX_4X8 =>
        self.predict_intra_inner::<Block4x8, _>(tile_rect, dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_8X4 =>
        self.predict_intra_inner::<Block8x4, _>(tile_rect, dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_8X16 =>
        self.predict_intra_inner::<Block8x16, _>(tile_rect, dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_16X8 =>
        self.predict_intra_inner::<Block16x8, _>(tile_rect, dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_16X32 =>
        self.predict_intra_inner::<Block16x32, _>(tile_rect, dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_32X16 =>
        self.predict_intra_inner::<Block32x16, _>(tile_rect, dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_32X64 =>
        self.predict_intra_inner::<Block32x64, _>(tile_rect, dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_64X32 =>
        self.predict_intra_inner::<Block64x32, _>(tile_rect, dst, bit_depth, ac, alpha, edge_buf),

      TxSize::TX_4X16 =>
        self.predict_intra_inner::<Block4x16, _>(tile_rect, dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_16X4 =>
        self.predict_intra_inner::<Block16x4, _>(tile_rect, dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_8X32 =>
        self.predict_intra_inner::<Block8x32, _>(tile_rect, dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_32X8 =>
        self.predict_intra_inner::<Block32x8, _>(tile_rect, dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_16X64 =>
        self.predict_intra_inner::<Block16x64, _>(tile_rect, dst, bit_depth, ac, alpha, edge_buf),
      TxSize::TX_64X16 =>
        self.predict_intra_inner::<Block64x16, _>(tile_rect, dst, bit_depth, ac, alpha, edge_buf),
    }
  }

  #[inline(always)]
  fn predict_intra_inner<B: Intra<T>, T: Pixel>(
    self, tile_rect: TileRect, dst: &mut PlaneRegionMut<'_, T>, bit_depth: usize, ac: &[i16],
    alpha: i16, edge_buf: &AlignedArray<[T; 4 * MAX_TX_SIZE + 1]>
  ) {
    // left pixels are order from bottom to top and right-aligned
    let (left, not_left) = edge_buf.array.split_at(2*MAX_TX_SIZE);
    let (top_left, above) = not_left.split_at(1);

    let &Rect { x: frame_x, y: frame_y, .. } = dst.rect();
    debug_assert!(frame_x >= 0 && frame_y >= 0);
    // x and y are expressed relative to the tile
    let x = frame_x as usize - tile_rect.x;
    let y = frame_y as usize - tile_rect.y;

    let mode: PredictionMode = match self {
      PredictionMode::PAETH_PRED => match (x, y) {
        (0, 0) => PredictionMode::DC_PRED,
        (_, 0) => PredictionMode::H_PRED,
        (0, _) => PredictionMode::V_PRED,
        _ => PredictionMode::PAETH_PRED
      },
      PredictionMode::UV_CFL_PRED =>
        if alpha == 0 {
          PredictionMode::DC_PRED
        } else {
          self
        },
      _ => self
    };

    let above_slice = &above[..B::W + B::H];
    let left_slice = &left[2 * MAX_TX_SIZE - B::H..];
    let left_and_left_below_slice = &left[2 * MAX_TX_SIZE - B::H - B::W..];

    match mode {
      PredictionMode::DC_PRED => match (x, y) {
        (0, 0) => B::pred_dc_128(dst, bit_depth),
        (_, 0) => B::pred_dc_left(dst, above_slice, left_slice),
        (0, _) => B::pred_dc_top(dst, above_slice, left_slice),
        _ => B::pred_dc(dst, above_slice, left_slice)
      },
      PredictionMode::UV_CFL_PRED => match (x, y) {
        (0, 0) => B::pred_cfl_128(dst, &ac, alpha, bit_depth),
        (_, 0) => B::pred_cfl_left(
          dst,
          &ac,
          alpha,
          bit_depth,
          above_slice,
          left_slice
        ),
        (0, _) => B::pred_cfl_top(
          dst,
          &ac,
          alpha,
          bit_depth,
          above_slice,
          left_slice
        ),
        _ => B::pred_cfl(
          dst,
          &ac,
          alpha,
          bit_depth,
          above_slice,
          left_slice
        )
      },
      PredictionMode::H_PRED => B::pred_h(dst, left_slice),
      PredictionMode::V_PRED => B::pred_v(dst, above_slice),
      PredictionMode::PAETH_PRED =>
        B::pred_paeth(dst, above_slice, left_slice, top_left[0]),
      PredictionMode::SMOOTH_PRED =>
        B::pred_smooth(dst, above_slice, left_slice),
      PredictionMode::SMOOTH_H_PRED =>
        B::pred_smooth_h(dst, above_slice, left_slice),
      PredictionMode::SMOOTH_V_PRED =>
        B::pred_smooth_v(dst, above_slice, left_slice),
      PredictionMode::D45_PRED =>
        B::pred_directional(dst, above_slice, left_and_left_below_slice, top_left, 45, bit_depth),
      PredictionMode::D135_PRED =>
        B::pred_directional(dst, above_slice, left_and_left_below_slice, top_left, 135, bit_depth),
      PredictionMode::D117_PRED =>
        B::pred_directional(dst, above_slice, left_and_left_below_slice, top_left, 113, bit_depth),
      PredictionMode::D153_PRED =>
        B::pred_directional(dst, above_slice, left_and_left_below_slice, top_left, 157, bit_depth),
      PredictionMode::D207_PRED =>
        B::pred_directional(dst, above_slice, left_and_left_below_slice, top_left, 203, bit_depth),
      PredictionMode::D63_PRED =>
        B::pred_directional(dst, above_slice, left_and_left_below_slice, top_left, 67, bit_depth),
      _ => unimplemented!()
    }
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
    self, fi: &FrameInvariants<T>, tile_rect: TileRect, p: usize, po: PlaneOffset,
    dst: &mut PlaneRegionMut<'_, T>, width: usize, height: usize,
    ref_frames: [RefType; 2], mvs: [MotionVector; 2]
  ) {
    assert!(!self.is_intra());
    let frame_po = tile_rect.to_frame_plane_offset(po);

    let mode = fi.default_filter;
    let is_compound =
      ref_frames[1] != RefType::INTRA_FRAME && ref_frames[1] != RefType::NONE_FRAME;

    fn get_params<'a, T: Pixel>(
      rec_plane: &'a Plane<T>, po: PlaneOffset, mv: MotionVector
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
        y: po.y + row_offset as isize - 3
      };
      (row_frac, col_frac, rec_plane.slice(qo).clamp().subslice(3, 3))
    };

    if !is_compound {
      if let Some(ref rec) = fi.rec_buffer.frames[fi.ref_frames[ref_frames[0].to_index()] as usize] {
        let (row_frac, col_frac, src) = get_params(&rec.frame.planes[p], frame_po, mvs[0]);
        put_8tap(
          dst,
          src,
          width,
          height,
          col_frac,
          row_frac,
          mode,
          mode,
          fi.sequence.bit_depth
        );
      }
    } else {
      let mut tmp: [AlignedArray<[i16; 128 * 128]>; 2] =
        [UninitializedAlignedArray(), UninitializedAlignedArray()];
      for i in 0..2 {
        if let Some(ref rec) = fi.rec_buffer.frames[fi.ref_frames[ref_frames[i].to_index()] as usize] {
          let (row_frac, col_frac, src) = get_params(&rec.frame.planes[p], frame_po, mvs[i]);
          prep_8tap(
            &mut tmp[i].array,
            src,
            width,
            height,
            col_frac,
            row_frac,
            mode,
            mode,
            fi.sequence.bit_depth
          );
        }
      }
      mc_avg(
        dst,
        &tmp[0].array,
        &tmp[1].array,
        width,
        height,
        fi.sequence.bit_depth
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
  INTERINTRA_MODES
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
  OBMC_CAUSAL,    // 2-sided OBMC
  WARPED_CAUSAL,  // 2-sided WARPED
  MOTION_MODES
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
  PALETTE_SIZES
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
  PALETTE_COLORS
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum FilterIntraMode {
  FILTER_DC_PRED,
  FILTER_V_PRED,
  FILTER_H_PRED,
  FILTER_D157_PRED,
  FILTER_PAETH_PRED,
  FILTER_INTRA_MODES
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
  NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT  // PAETH
];

pub trait Dim {
  const W: usize;
  const H: usize;
}

macro_rules! block_dimension {
  ($W:expr, $H:expr) => {
    paste::item! {
      pub struct [<Block $W x $H>];

      impl Dim for [<Block $W x $H>] {
        const W: usize = $W;
        const H: usize = $H;
      }

      impl<T: Pixel> Intra<T> for [<Block $W x $H>] {}
    }
  };
}

macro_rules! blocks_dimension {
  ($(($W:expr, $H:expr)),+) => {
    $(
      block_dimension! { $W, $H }
    )*
  }
}

blocks_dimension! { (4, 4), (8, 8), (16, 16), (32, 32), (64, 64) }
blocks_dimension! { (4, 8), (8, 16), (16, 32), (32, 64) }
blocks_dimension! { (8, 4), (16, 8), (32, 16), (64, 32) }
blocks_dimension! { (4, 16), (8, 32), (16, 64) }
blocks_dimension! { (16, 4), (32, 8), (64, 16) }

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

#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
macro_rules! decl_angular_ipred_fn {
  ($f:ident) => {
    extern {
      fn $f(
        dst: *mut u8, stride: libc::ptrdiff_t, topleft: *const u8,
        width: libc::c_int, height: libc::c_int, angle: libc::c_int
      );
    }
  };
}

#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
decl_angular_ipred_fn!(rav1e_ipred_dc_avx2);
#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
decl_angular_ipred_fn!(rav1e_ipred_dc_128_avx2);
#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
decl_angular_ipred_fn!(rav1e_ipred_dc_left_avx2);
#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
decl_angular_ipred_fn!(rav1e_ipred_dc_top_avx2);
#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
decl_angular_ipred_fn!(rav1e_ipred_h_avx2);
#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
decl_angular_ipred_fn!(rav1e_ipred_v_avx2);
#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
decl_angular_ipred_fn!(rav1e_ipred_paeth_avx2);
#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
decl_angular_ipred_fn!(rav1e_ipred_smooth_avx2);
#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
decl_angular_ipred_fn!(rav1e_ipred_smooth_h_avx2);
#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
decl_angular_ipred_fn!(rav1e_ipred_smooth_v_avx2);

#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
macro_rules! decl_cfl_pred_fn {
  ($f:ident) => {
    extern {
      fn $f(
        dst: *mut u8, stride: libc::ptrdiff_t, topleft: *const u8,
        width: libc::c_int, height: libc::c_int, ac: *const u8,
        alpha: libc::c_int
      );
    }
  };
}

#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
decl_cfl_pred_fn!(rav1e_ipred_cfl_avx2);
#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
decl_cfl_pred_fn!(rav1e_ipred_cfl_128_avx2);
#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
decl_cfl_pred_fn!(rav1e_ipred_cfl_left_avx2);
#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
decl_cfl_pred_fn!(rav1e_ipred_cfl_top_avx2);

pub trait Intra<T>: Dim
where
  T: Pixel,
{
  fn pred_dc(output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T]) {
    #[cfg(all(target_arch = "x86_64", feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_dc_avx2(
            output.data_ptr_mut() as *mut _,
            output.plane_cfg.stride as libc::ptrdiff_t,
            above.as_ptr().offset(-1) as *const _,
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            0
          )
        };
      }
    }
    let edges = left[..Self::H].iter().chain(above[..Self::W].iter());
    let len = (Self::W + Self::H) as u32;
    let avg = (edges.fold(0u32, |acc, &v| { let v: u32 = v.into(); v + acc }) + (len >> 1)) / len;
    let avg = T::cast_from(avg);

    for line in output.rows_iter_mut().take(Self::H) {
      for v in &mut line[..Self::W] {
        *v = avg;
      }
    }
  }

  fn pred_dc_128(output: &mut PlaneRegionMut<'_, T>, bit_depth: usize) {
    #[cfg(all(target_arch = "x86_64", feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_dc_128_avx2(
            output.data_ptr_mut() as *mut _,
            output.plane_cfg.stride as libc::ptrdiff_t,
            ptr::null(),
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            0
          )
        };
      }
    }
    let v = T::cast_from(128u32 << (bit_depth - 8));
    for y in 0..Self::H {
      for x in 0..Self::W {
        output[y][x] = v;
      }
    }
  }

  fn pred_dc_left(output: &mut PlaneRegionMut<'_, T>, _above: &[T], left: &[T]) {
    #[cfg(all(target_arch = "x86_64", feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_dc_left_avx2(
            output.data_ptr_mut() as *mut _,
            output.plane_cfg.stride as libc::ptrdiff_t,
            left.as_ptr().add(Self::H) as *const _,
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            0
          )
        };
      }
    }
    let sum = left[..Self::H].iter().fold(0u32, |acc, &v| { let v: u32 = v.into(); v + acc });
    let avg = T::cast_from((sum + (Self::H >> 1) as u32) / Self::H as u32);
    for line in output.rows_iter_mut().take(Self::H) {
      line[..Self::W].iter_mut().for_each(|v| *v = avg);
    }
  }

  fn pred_dc_top(output: &mut PlaneRegionMut<'_, T>, above: &[T], _left: &[T]) {
    #[cfg(all(target_arch = "x86_64", feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_dc_top_avx2(
            output.data_ptr_mut() as *mut _,
            output.plane_cfg.stride as libc::ptrdiff_t,
            above.as_ptr().offset(-1) as *const _,
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            0
          )
        };
      }
    }
    let sum = above[..Self::W].iter().fold(0u32, |acc, &v| { let v: u32 = v.into(); v + acc });
    let avg = T::cast_from((sum + (Self::W >> 1) as u32) / Self::W as u32);
    for line in output.rows_iter_mut().take(Self::H) {
      line[..Self::W].iter_mut().for_each(|v| *v = avg);
    }
  }

  fn pred_h(output: &mut PlaneRegionMut<'_, T>, left: &[T]) {
    #[cfg(all(target_arch = "x86_64", feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_h_avx2(
            output.data_ptr_mut() as *mut _,
            output.plane_cfg.stride as libc::ptrdiff_t,
            left.as_ptr().add(Self::H) as *const _,
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            0
          )
        };
      }
    }
    for (line, l) in
      output.rows_iter_mut().zip(left[..Self::H].iter().rev())
    {
      for v in &mut line[..Self::W] {
        *v = *l;
      }
    }
  }

  fn pred_v(output: &mut PlaneRegionMut<'_, T>, above: &[T]) {
    #[cfg(all(target_arch = "x86_64", feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_v_avx2(
            output.data_ptr_mut() as *mut _,
            output.plane_cfg.stride as libc::ptrdiff_t,
            above.as_ptr().offset(-1) as *const _,
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            0
          )
        };
      }
    }
    for line in output.rows_iter_mut().take(Self::H) {
      line[..Self::W].clone_from_slice(&above[..Self::W])
    }
  }

  fn pred_paeth(
    output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T],
    above_left: T
  ) {
    #[cfg(all(target_arch = "x86_64", feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_paeth_avx2(
            output.data_ptr_mut() as *mut _,
            output.plane_cfg.stride as libc::ptrdiff_t,
            above.as_ptr().offset(-1) as *const _,
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            0
          )
        };
      }
    }
    for r in 0..Self::H {
      let row = &mut output[r];
      for c in 0..Self::W {
        // Top-left pixel is fixed in libaom
        let raw_top_left: i32 = above_left.into();
        let raw_left: i32 = left[Self::H - 1 - r].into();
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

  fn pred_smooth(
    output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T]
  ) {
    #[cfg(all(target_arch = "x86_64", feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_smooth_avx2(
            output.data_ptr_mut() as *mut _,
            output.plane_cfg.stride as libc::ptrdiff_t,
            above.as_ptr().offset(-1) as *const _,
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            0
          )
        };
      }
    }
    let below_pred = left[0]; // estimated by bottom-left pixel
    let right_pred = above[Self::W - 1]; // estimated by top-right pixel
    let sm_weights_w = &sm_weight_arrays[Self::W..];
    let sm_weights_h = &sm_weight_arrays[Self::H..];

    let log2_scale = 1 + sm_weight_log2_scale;
    let scale = 1_u16 << sm_weight_log2_scale;

    // Weights sanity checks
    assert!((sm_weights_w[0] as u16) < scale);
    assert!((sm_weights_h[0] as u16) < scale);
    assert!((scale - sm_weights_w[Self::W - 1] as u16) < scale);
    assert!((scale - sm_weights_h[Self::H - 1] as u16) < scale);
    // ensures no overflow when calculating predictor
    assert!(log2_scale as usize + size_of::<T>() < 31);

    for r in 0..Self::H {
      let row = &mut output[r];
      for c in 0..Self::W {
        let pixels = [above[c], below_pred, left[Self::H - 1 - r], right_pred];

        let weights = [
          sm_weights_h[r] as u16,
          scale - sm_weights_h[r] as u16,
          sm_weights_w[c] as u16,
          scale - sm_weights_w[c] as u16
        ];

        assert!(
          scale >= (sm_weights_h[r] as u16)
            && scale >= (sm_weights_w[c] as u16)
        );

        // Sum up weighted pixels
        let mut this_pred: u32 = weights
          .iter()
          .zip(pixels.iter())
          .map(|(w, p)| { let p: u32 = (*p).into(); (*w as u32) * p })
          .sum();
        this_pred = (this_pred + (1 << (log2_scale - 1))) >> log2_scale;

        row[c] = T::cast_from(this_pred);
      }
    }
  }

  fn pred_smooth_h(
    output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T]
  ) {
    #[cfg(all(target_arch = "x86_64", feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_smooth_h_avx2(
            output.data_ptr_mut() as *mut _,
            output.plane_cfg.stride as libc::ptrdiff_t,
            above.as_ptr().offset(-1) as *const _,
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            0
          )
        };
      }
    }
    let right_pred = above[Self::W - 1]; // estimated by top-right pixel
    let sm_weights = &sm_weight_arrays[Self::W..];

    let log2_scale = sm_weight_log2_scale;
    let scale = 1_u16 << sm_weight_log2_scale;

    // Weights sanity checks
    assert!((sm_weights[0] as u16) < scale);
    assert!((scale - sm_weights[Self::W - 1] as u16) < scale);
    // ensures no overflow when calculating predictor
    assert!(log2_scale as usize + size_of::<T>() < 31);

    for r in 0..Self::H {
      let row = &mut output[r];
      for c in 0..Self::W {
        let pixels = [left[Self::H - 1 - r], right_pred];
        let weights = [sm_weights[c] as u16, scale - sm_weights[c] as u16];

        assert!(scale >= sm_weights[c] as u16);

        let mut this_pred: u32 = weights
          .iter()
          .zip(pixels.iter())
          .map(|(w, p)| { let p: u32 = (*p).into(); (*w as u32) * p })
          .sum();
        this_pred = (this_pred + (1 << (log2_scale - 1))) >> log2_scale;

        row[c] = T::cast_from(this_pred);
      }
    }
  }

  fn pred_smooth_v(
    output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T]
  ) {
    #[cfg(all(target_arch = "x86_64", feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_smooth_v_avx2(
            output.data_ptr_mut() as *mut _,
            output.plane_cfg.stride as libc::ptrdiff_t,
            above.as_ptr().offset(-1) as *const _,
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            0
          )
        };
      }
    }
    let below_pred = left[0]; // estimated by bottom-left pixel
    let sm_weights = &sm_weight_arrays[Self::H..];

    let log2_scale = sm_weight_log2_scale;
    let scale = 1_u16 << sm_weight_log2_scale;

    // Weights sanity checks
    assert!((sm_weights[0] as u16) < scale);
    assert!((scale - sm_weights[Self::H - 1] as u16) < scale);
    // ensures no overflow when calculating predictor
    assert!(log2_scale as usize + size_of::<T>() < 31);

    for r in 0..Self::H {
      let row = &mut output[r];
      for c in 0..Self::W {
        let pixels = [above[c], below_pred];
        let weights = [sm_weights[r] as u16, scale - sm_weights[r] as u16];

        assert!(scale >= sm_weights[r] as u16);

        let mut this_pred: u32 = weights
          .iter()
          .zip(pixels.iter())
          .map(|(w, p)| { let p: u32 = (*p).into(); (*w as u32) * p })
          .sum();
        this_pred = (this_pred + (1 << (log2_scale - 1))) >> log2_scale;

        row[c] = T::cast_from(this_pred);
      }
    }
  }

  #[target_feature(enable = "ssse3")]
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  unsafe fn pred_cfl_ssse3(
    output: *mut T, stride: usize, ac: *const i16, alpha: i16,
    bit_depth: usize
  ) {
    let alpha_sign = _mm_set1_epi16(alpha);
    let alpha_q12 = _mm_slli_epi16(_mm_abs_epi16(alpha_sign), 9);
    let dc_scalar: u32 = (*output).into();
    let dc_q0 = _mm_set1_epi16(dc_scalar as i16);
    let max = _mm_set1_epi16((1 << bit_depth) - 1);

    for j in 0..Self::H {
      let luma = ac.add(Self::W * j);
      let line = output.add(stride * j);

      let mut i = 0isize;
      let mut last = _mm_setzero_si128();
      while (i as usize) < Self::W {
        let ac_q3 = _mm_loadu_si128(luma.offset(i) as *const _);
        let ac_sign = _mm_sign_epi16(alpha_sign, ac_q3);
        let abs_scaled_luma_q0 =
          _mm_mulhrs_epi16(_mm_abs_epi16(ac_q3), alpha_q12);
        let scaled_luma_q0 = _mm_sign_epi16(abs_scaled_luma_q0, ac_sign);
        let pred = _mm_add_epi16(scaled_luma_q0, dc_q0);
        if size_of::<T>() == 1 {
          if Self::W < 16 {
            let res = _mm_packus_epi16(pred, pred);
            if Self::W == 4 {
               *(line.offset(i) as *mut i32) = _mm_cvtsi128_si32(res);
            } else {
              _mm_storel_epi64(line.offset(i) as *mut _, res);
            }
          } else if (i & 15) == 0 {
            last = pred;
          } else {
            let res = _mm_packus_epi16(last, pred);
            _mm_storeu_si128(line.offset(i - 8) as *mut _, res);
          }
        } else {
          let res = _mm_min_epi16(max, _mm_max_epi16(pred, _mm_setzero_si128()));
          if Self::W == 4 {
            _mm_storel_epi64(line.offset(i) as *mut _, res);
          } else {
            _mm_storeu_si128(line.offset(i) as *mut _, res);
          }
        }
        i += 8;
      }
    }
  }

  fn pred_cfl_inner(
    output: &mut PlaneRegionMut<'_, T>, ac: &[i16], alpha: i16, bit_depth: usize
  ) {
    if alpha == 0 {
      return;
    }
    assert!(32 >= Self::W);
    assert!(ac.len() >= 32 * (Self::H - 1) + Self::W);
    assert!(output.plane_cfg.stride >= Self::W);
    assert!(output.rows_iter().len() >= Self::H);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
      if is_x86_feature_detected!("ssse3") {
        return unsafe {
          Self::pred_cfl_ssse3(output.data_ptr_mut(), output.plane_cfg.stride, ac.as_ptr(), alpha, bit_depth)
        };
      }
    }

    let sample_max = (1 << bit_depth) - 1;
    let avg: i32 = output[0][0].into();


    for (line, luma) in
      output.rows_iter_mut().zip(ac.chunks(Self::W)).take(Self::H)
    {
      for (v, &l) in line[..Self::W].iter_mut().zip(luma[..Self::W].iter()) {
        *v = T::cast_from(
          (avg + get_scaled_luma_q0(alpha, l)).max(0).min(sample_max));
      }
    }
  }

  fn pred_cfl(
    output: &mut PlaneRegionMut<'_, T>, ac: &[i16], alpha: i16, bit_depth: usize,
    above: &[T], left: &[T]
  ) {
    #[cfg(all(target_arch = "x86_64", feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_cfl_avx2(
            output.data_ptr_mut() as *mut _,
            output.plane_cfg.stride as libc::ptrdiff_t,
            above.as_ptr().offset(-1) as *const _,
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            ac.as_ptr() as *const _,
            alpha as libc::c_int
          )
        }
      }
    }
    Self::pred_dc(output, above, left);
    Self::pred_cfl_inner(output, &ac, alpha, bit_depth);
  }

  fn pred_cfl_128(
    output: &mut PlaneRegionMut<'_, T>, ac: &[i16], alpha: i16, bit_depth: usize
  ) {
    #[cfg(all(target_arch = "x86_64", feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_cfl_128_avx2(
            output.data_ptr_mut() as *mut _,
            output.plane_cfg.stride as libc::ptrdiff_t,
            ptr::null(),
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            ac.as_ptr() as *const _,
            alpha as libc::c_int
          )
        }
      }
    }
    Self::pred_dc_128(output, bit_depth);
    Self::pred_cfl_inner(output, &ac, alpha, bit_depth);
  }

  fn pred_cfl_left(
    output: &mut PlaneRegionMut<'_, T>, ac: &[i16], alpha: i16, bit_depth: usize,
    above: &[T], left: &[T]
  ) {
    #[cfg(all(target_arch = "x86_64", feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_cfl_left_avx2(
            output.data_ptr_mut() as *mut _,
            output.plane_cfg.stride as libc::ptrdiff_t,
            above.as_ptr().offset(-1) as *const _,
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            ac.as_ptr() as *const _,
            alpha as libc::c_int
          )
        }
      }
    }
    Self::pred_dc_left(output, above, left);
    Self::pred_cfl_inner(output, &ac, alpha, bit_depth);
  }

  fn pred_cfl_top(
    output: &mut PlaneRegionMut<'_, T>, ac: &[i16], alpha: i16, bit_depth: usize,
    above: &[T], left: &[T]
  ) {
    #[cfg(all(target_arch = "x86_64", feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_cfl_top_avx2(
            output.data_ptr_mut() as *mut _,
            output.plane_cfg.stride as libc::ptrdiff_t,
            above.as_ptr().offset(-1) as *const _,
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            ac.as_ptr() as *const _,
            alpha as libc::c_int
          )
        }
      }
    }
    Self::pred_dc_top(output, above, left);
    Self::pred_cfl_inner(output, &ac, alpha, bit_depth);
  }

  fn pred_directional(
    output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T], top_left: &[T], angle: usize, bit_depth: usize
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
        _ => 0
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
      for i in 0..Self::H {
        let row = &mut output[i];
        for j in 0..Self::W {
          let idx = (i + 1) * dx;
          let base = (idx >> (6 - upsample_above)) + (j << upsample_above);
          let shift = (((idx << upsample_above) >> 1) & 31) as i32;
          let max_base_x = (Self::H + Self::W - 1) << upsample_above;
          let v = if base < max_base_x {
            let a: i32 = above[base].into();
            let b: i32 = above[base + 1].into();
            round_shift(a * (32 - shift) + b * shift, 5)
          } else {
            let c: i32 = above[max_base_x].into();
            c
          }.max(0).min(sample_max);
          row[j] = T::cast_from(v);
        }
      }
    } else if p_angle > 90 && p_angle < 180 {
      for i in 0..Self::H {
        let row = &mut output[i];
        for j in 0..Self::W {
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
              left[Self::W + Self::H - 1 - base as usize]
            }
            .into();
            let b: i32 = left[Self::W + Self::H - (2 + base) as usize].into();
            let v = round_shift(a * (32 - shift) + b * shift, 5)
                .max(0)
                .min(sample_max);
            row[j] = T::cast_from(v);
          }
        }
      }
    } else if p_angle > 180 {
      for i in 0..Self::H {
        let row = &mut output[i];
        for j in 0..Self::W {
          let idx = (j + 1) * dy;
          let base = (idx >> (6 - upsample_left)) + (i << upsample_left);
          let shift = (((idx << upsample_left) >> 1) & 31) as i32;
          let a: i32 = left[Self::W + Self::H - 1 - base].into();
          let b: i32 = left[Self::W + Self::H - 2 - base].into();
          let v = round_shift(a * (32 - shift) + b * shift, 5)
              .max(0)
              .min(sample_max);
          row[j] = T::cast_from(v);
        }
      }
    }
  }
}


pub trait Inter: Dim {}

#[cfg(test)]
mod test {
  use super::*;
  use num_traits::*;

  #[test]
  fn pred_matches_u8() {
    let mut edge_buf: AlignedArray<[u8; 2 * MAX_TX_SIZE + 1]> =
      UninitializedAlignedArray();
    for i in 0..edge_buf.array.len() {
      edge_buf.array[i] = (i + 32).saturating_sub(MAX_TX_SIZE).as_();
    }
    let left = &edge_buf.array[MAX_TX_SIZE - 4..MAX_TX_SIZE];
    let above = &edge_buf.array[MAX_TX_SIZE + 1..MAX_TX_SIZE + 5];
    let top_left = edge_buf.array[MAX_TX_SIZE];

    let mut output = Plane::wrap(vec![0u8; 4 * 4], 4);

    Block4x4::pred_dc(&mut output.as_region_mut(), above, left);
    assert_eq!(&output.data[..], [32u8; 16]);

    Block4x4::pred_dc_top(&mut output.as_region_mut(), above, left);
    assert_eq!(&output.data[..], [35u8; 16]);

    Block4x4::pred_dc_left(&mut output.as_region_mut(), above, left);
    assert_eq!(&output.data[..], [30u8; 16]);

    Block4x4::pred_dc_128(&mut output.as_region_mut(), 8);
    assert_eq!(&output.data[..], [128u8; 16]);

    Block4x4::pred_v(&mut output.as_region_mut(), above);
    assert_eq!(
      &output.data[..],
      [33, 34, 35, 36, 33, 34, 35, 36, 33, 34, 35, 36, 33, 34, 35, 36]
    );

    Block4x4::pred_h(&mut output.as_region_mut(), left);
    assert_eq!(
      &output.data[..],
      [31, 31, 31, 31, 30, 30, 30, 30, 29, 29, 29, 29, 28, 28, 28, 28]
    );

    Block4x4::pred_paeth(&mut output.as_region_mut(), above, left, top_left);
    assert_eq!(
      &output.data[..],
      [32, 34, 35, 36, 30, 32, 32, 36, 29, 32, 32, 32, 28, 28, 32, 32]
    );

    Block4x4::pred_smooth(&mut output.as_region_mut(), above, left);
    assert_eq!(
      &output.data[..],
      [32, 34, 35, 35, 30, 32, 33, 34, 29, 31, 32, 32, 29, 30, 32, 32]
    );

    Block4x4::pred_smooth_h(&mut output.as_region_mut(), above, left);
    assert_eq!(
      &output.data[..],
      [31, 33, 34, 35, 30, 33, 34, 35, 29, 32, 34, 34, 28, 31, 33, 34]
    );

    Block4x4::pred_smooth_v(&mut output.as_region_mut(), above, left);
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

    Block4x4::pred_dc(&mut o.as_region_mut(), &above[..4], &left[..4]);

    for l in o.data.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    Block4x4::pred_h(&mut o.as_region_mut(), &left[..4]);

    for l in o.data.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    Block4x4::pred_v(&mut o.as_region_mut(), &above[..4]);

    for l in o.data.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    let above_left = unsafe { *above.as_ptr().offset(-1) };

    Block4x4::pred_paeth(&mut o.as_region_mut(), &above[..4], &left[..4], above_left);

    for l in o.data.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    Block4x4::pred_smooth(&mut o.as_region_mut(), &above[..4], &left[..4]);

    for l in o.data.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    Block4x4::pred_smooth_h(&mut o.as_region_mut(), &above[..4], &left[..4]);

    for l in o.data.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    Block4x4::pred_smooth_v(&mut o.as_region_mut(), &above[..4], &left[..4]);

    for l in o.data.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }
  }
}
