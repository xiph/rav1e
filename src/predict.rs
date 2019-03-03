// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(non_upper_case_globals)]

use crate::context::{INTRA_MODES, MAX_TX_SIZE};
use crate::partition::*;
use crate::plane::*;
use crate::util::*;

#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
use libc;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::mem::*;
#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
use std::ptr;

pub static RAV1E_INTRA_MODES: &'static [PredictionMode] = &[
  PredictionMode::DC_PRED,
  PredictionMode::H_PRED,
  PredictionMode::V_PRED,
  PredictionMode::SMOOTH_PRED,
  PredictionMode::SMOOTH_H_PRED,
  PredictionMode::SMOOTH_V_PRED,
  PredictionMode::PAETH_PRED,
  // reconstruction of directional prediction is currently broken
  // PredictionMode::D45_PRED,
  // PredictionMode::D135_PRED,
  // PredictionMode::D117_PRED,
  // PredictionMode::D153_PRED,
  // PredictionMode::D207_PRED,
  // PredictionMode::D63_PRED,
];

// Intra prediction modes tested at high speed levels
#[rustfmt::skip]
pub static RAV1E_INTRA_MODES_MINIMAL: &'static [PredictionMode] = &[
    PredictionMode::DC_PRED,
    PredictionMode::H_PRED,
    PredictionMode::V_PRED
];

pub static RAV1E_INTER_MODES_MINIMAL: &'static [PredictionMode] = &[
  PredictionMode::NEARESTMV
];

pub static RAV1E_INTER_COMPOUND_MODES: &'static [PredictionMode] = &[
  PredictionMode::GLOBAL_GLOBALMV,
  PredictionMode::NEAREST_NEARESTMV,
  PredictionMode::NEW_NEWMV,
  PredictionMode::NEAREST_NEWMV,
  PredictionMode::NEW_NEARESTMV
];

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
const NEED_BOTTOMLEFT: u8 = 1 << 5;

/*const INTRA_EDGE_FILT: usize = 3;
const INTRA_EDGE_TAPS: usize = 5;
const MAX_UPSAMPLE_SZ: usize = 16;*/

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

#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
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

#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
decl_angular_ipred_fn!(rav1e_ipred_dc_avx2);
#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
decl_angular_ipred_fn!(rav1e_ipred_dc_128_avx2);
#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
decl_angular_ipred_fn!(rav1e_ipred_dc_left_avx2);
#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
decl_angular_ipred_fn!(rav1e_ipred_dc_top_avx2);
#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
decl_angular_ipred_fn!(rav1e_ipred_h_avx2);
#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
decl_angular_ipred_fn!(rav1e_ipred_v_avx2);
#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
decl_angular_ipred_fn!(rav1e_ipred_paeth_avx2);
#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
decl_angular_ipred_fn!(rav1e_ipred_smooth_avx2);
#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
decl_angular_ipred_fn!(rav1e_ipred_smooth_h_avx2);
#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
decl_angular_ipred_fn!(rav1e_ipred_smooth_v_avx2);

#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
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

#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
decl_cfl_pred_fn!(rav1e_ipred_cfl_avx2);
#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
decl_cfl_pred_fn!(rav1e_ipred_cfl_128_avx2);
#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
decl_cfl_pred_fn!(rav1e_ipred_cfl_left_avx2);
#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
decl_cfl_pred_fn!(rav1e_ipred_cfl_top_avx2);

pub trait Intra<T>: Dim
where
  T: Pixel,
{
  fn pred_dc(output: &mut PlaneMutSlice<'_, T>, above: &[T], left: &[T]) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_dc_avx2(
            output.as_mut_ptr() as *mut _,
            output.plane.cfg.stride as libc::ptrdiff_t,
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

  fn pred_dc_128(output: &mut PlaneMutSlice<'_, T>, bit_depth: usize) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_dc_128_avx2(
            output.as_mut_ptr() as *mut _,
            output.plane.cfg.stride as libc::ptrdiff_t,
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

  fn pred_dc_left(output: &mut PlaneMutSlice<'_, T>, _above: &[T], left: &[T]) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_dc_left_avx2(
            output.as_mut_ptr() as *mut _,
            output.plane.cfg.stride as libc::ptrdiff_t,
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

  fn pred_dc_top(output: &mut PlaneMutSlice<'_, T>, above: &[T], _left: &[T]) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_dc_top_avx2(
            output.as_mut_ptr() as *mut _,
            output.plane.cfg.stride as libc::ptrdiff_t,
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

  fn pred_h(output: &mut PlaneMutSlice<'_, T>, left: &[T]) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_h_avx2(
            output.as_mut_ptr() as *mut _,
            output.plane.cfg.stride as libc::ptrdiff_t,
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

  fn pred_v(output: &mut PlaneMutSlice<'_, T>, above: &[T]) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_v_avx2(
            output.as_mut_ptr() as *mut _,
            output.plane.cfg.stride as libc::ptrdiff_t,
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
    output: &mut PlaneMutSlice<'_, T>, above: &[T], left: &[T],
    above_left: T
  ) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_paeth_avx2(
            output.as_mut_ptr() as *mut _,
            output.plane.cfg.stride as libc::ptrdiff_t,
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
    output: &mut PlaneMutSlice<'_, T>, above: &[T], left: &[T]
  ) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_smooth_avx2(
            output.as_mut_ptr() as *mut _,
            output.plane.cfg.stride as libc::ptrdiff_t,
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
    output: &mut PlaneMutSlice<'_, T>, above: &[T], left: &[T]
  ) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_smooth_h_avx2(
            output.as_mut_ptr() as *mut _,
            output.plane.cfg.stride as libc::ptrdiff_t,
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
    output: &mut PlaneMutSlice<'_, T>, above: &[T], left: &[T]
  ) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_smooth_v_avx2(
            output.as_mut_ptr() as *mut _,
            output.plane.cfg.stride as libc::ptrdiff_t,
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
    output: &mut PlaneMutSlice<'_, T>, ac: &[i16], alpha: i16, bit_depth: usize
  ) {
    if alpha == 0 {
      return;
    }
    assert!(32 >= Self::W);
    assert!(ac.len() >= 32 * (Self::H - 1) + Self::W);
    assert!(output.plane.cfg.stride >= Self::W);
    assert!(output.rows_iter().len() >= Self::H);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
      if is_x86_feature_detected!("ssse3") {
        return unsafe {
          Self::pred_cfl_ssse3(output.as_mut_ptr(), output.plane.cfg.stride, ac.as_ptr(), alpha, bit_depth)
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
    output: &mut PlaneMutSlice<'_, T>, ac: &[i16], alpha: i16, bit_depth: usize,
    above: &[T], left: &[T]
  ) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_cfl_avx2(
            output.as_mut_ptr() as *mut _,
            output.plane.cfg.stride as libc::ptrdiff_t,
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
    output: &mut PlaneMutSlice<'_, T>, ac: &[i16], alpha: i16, bit_depth: usize
  ) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_cfl_128_avx2(
            output.as_mut_ptr() as *mut _,
            output.plane.cfg.stride as libc::ptrdiff_t,
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
    output: &mut PlaneMutSlice<'_, T>, ac: &[i16], alpha: i16, bit_depth: usize,
    above: &[T], left: &[T]
  ) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_cfl_left_avx2(
            output.as_mut_ptr() as *mut _,
            output.plane.cfg.stride as libc::ptrdiff_t,
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
    output: &mut PlaneMutSlice<'_, T>, ac: &[i16], alpha: i16, bit_depth: usize,
    above: &[T], left: &[T]
  ) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_cfl_top_avx2(
            output.as_mut_ptr() as *mut _,
            output.plane.cfg.stride as libc::ptrdiff_t,
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
    output: &mut PlaneMutSlice<'_, T>, above: &[T], left: &[T], top_left: &[T], angle: usize, bit_depth: usize
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

    Block4x4::pred_dc(&mut output.as_mut_slice(), above, left);
    assert_eq!(output.data, [32u8; 16]);

    Block4x4::pred_dc_top(&mut output.as_mut_slice(), above, left);
    assert_eq!(output.data, [35u8; 16]);

    Block4x4::pred_dc_left(&mut output.as_mut_slice(), above, left);
    assert_eq!(output.data, [30u8; 16]);

    Block4x4::pred_dc_128(&mut output.as_mut_slice(), 8);
    assert_eq!(output.data, [128u8; 16]);

    Block4x4::pred_v(&mut output.as_mut_slice(), above);
    assert_eq!(
      output.data,
      [33, 34, 35, 36, 33, 34, 35, 36, 33, 34, 35, 36, 33, 34, 35, 36]
    );

    Block4x4::pred_h(&mut output.as_mut_slice(), left);
    assert_eq!(
      output.data,
      [31, 31, 31, 31, 30, 30, 30, 30, 29, 29, 29, 29, 28, 28, 28, 28]
    );

    Block4x4::pred_paeth(&mut output.as_mut_slice(), above, left, top_left);
    assert_eq!(
      output.data,
      [32, 34, 35, 36, 30, 32, 32, 36, 29, 32, 32, 32, 28, 28, 32, 32]
    );

    Block4x4::pred_smooth(&mut output.as_mut_slice(), above, left);
    assert_eq!(
      output.data,
      [32, 34, 35, 35, 30, 32, 33, 34, 29, 31, 32, 32, 29, 30, 32, 32]
    );

    Block4x4::pred_smooth_h(&mut output.as_mut_slice(), above, left);
    assert_eq!(
      output.data,
      [31, 33, 34, 35, 30, 33, 34, 35, 29, 32, 34, 34, 28, 31, 33, 34]
    );

    Block4x4::pred_smooth_v(&mut output.as_mut_slice(), above, left);
    assert_eq!(
      output.data,
      [33, 34, 35, 36, 31, 31, 32, 33, 30, 30, 30, 31, 29, 30, 30, 30]
    );
  }

  #[test]
  fn pred_max() {
    let max12bit = 4096 - 1;
    let above = [max12bit; 32];
    let left = [max12bit; 32];

    let mut o = Plane::wrap(vec![0u16; 32 * 32], 32);

    Block4x4::pred_dc(&mut o.as_mut_slice(), &above[..4], &left[..4]);

    for l in o.data.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    Block4x4::pred_h(&mut o.as_mut_slice(), &left[..4]);

    for l in o.data.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    Block4x4::pred_v(&mut o.as_mut_slice(), &above[..4]);

    for l in o.data.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    let above_left = unsafe { *above.as_ptr().offset(-1) };

    Block4x4::pred_paeth(&mut o.as_mut_slice(), &above[..4], &left[..4], above_left);

    for l in o.data.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    Block4x4::pred_smooth(&mut o.as_mut_slice(), &above[..4], &left[..4]);

    for l in o.data.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    Block4x4::pred_smooth_h(&mut o.as_mut_slice(), &above[..4], &left[..4]);

    for l in o.data.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    Block4x4::pred_smooth_v(&mut o.as_mut_slice(), &above[..4], &left[..4]);

    for l in o.data.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }
  }
}
