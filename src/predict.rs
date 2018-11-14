// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(non_upper_case_globals)]
#![cfg_attr(feature = "cargo-clippy", allow(cast_lossless))]
#![cfg_attr(feature = "cargo-clippy", allow(needless_range_loop))]

#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
use libc;
use num_traits::*;

use context::INTRA_MODES;
use context::MAX_TX_SIZE;
use partition::*;
use util::*;
use std::mem::*;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub static RAV1E_INTRA_MODES: &'static [PredictionMode] = &[
  PredictionMode::DC_PRED,
  PredictionMode::H_PRED,
  PredictionMode::V_PRED,
  PredictionMode::SMOOTH_PRED,
  PredictionMode::SMOOTH_H_PRED,
  PredictionMode::SMOOTH_V_PRED,
  PredictionMode::PAETH_PRED
];

// Intra prediction modes tested at high speed levels
#[cfg_attr(rustfmt, rustfmt_skip)]
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
#[cfg_attr(rustfmt, rustfmt_skip)]
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

      impl Intra<u8> for [<Block $W x $H>] {}
      impl Intra<u16> for [<Block $W x $H>] {}
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

pub trait Intra<T>: Dim
where
  T: Pixel,
  i32: AsPrimitive<T>,
  u32: AsPrimitive<T>,
  usize: AsPrimitive<T>
{
  #[cfg_attr(feature = "comparative_bench", inline(never))]
  fn pred_dc(output: &mut [T], stride: usize, above: &[T], left: &[T]) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_dc_avx2(
            output.as_mut_ptr() as *mut _,
            stride as libc::ptrdiff_t,
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
    let avg =
      ((edges.fold(0u32, |acc, &v| { let v: u32 = v.into(); v + acc }) + (len >> 1)) / len).as_();

    for line in output.chunks_mut(stride).take(Self::H) {
      for v in &mut line[..Self::W] {
        *v = avg;
      }
    }
  }

  #[cfg_attr(feature = "comparative_bench", inline(never))]
  fn pred_dc_128(output: &mut [T], stride: usize, bit_depth: usize) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      use std::ptr;
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_dc_128_avx2(
            output.as_mut_ptr() as *mut _,
            stride as libc::ptrdiff_t,
            ptr::null(),
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            0
          )
        };
      }
    }
    for y in 0..Self::H {
      for x in 0..Self::W {
        output[y * stride + x] = (128u32 << (bit_depth - 8)).as_();
      }
    }
  }

  #[cfg_attr(feature = "comparative_bench", inline(never))]
  fn pred_dc_left(output: &mut [T], stride: usize, _above: &[T], left: &[T]) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_dc_left_avx2(
            output.as_mut_ptr() as *mut _,
            stride as libc::ptrdiff_t,
            left.as_ptr().add(Self::H) as *const _,
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            0
          )
        };
      }
    }
    let sum = left[..Self::W].iter().fold(0u32, |acc, &v| { let v: u32 = v.into(); v + acc });
    let avg = ((sum + (Self::W >> 1) as u32) / Self::W as u32).as_();
    for line in output.chunks_mut(stride).take(Self::H) {
      line[..Self::W].iter_mut().for_each(|v| *v = avg);
    }
  }

  #[cfg_attr(feature = "comparative_bench", inline(never))]
  fn pred_dc_top(output: &mut [T], stride: usize, above: &[T], _left: &[T]) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_dc_top_avx2(
            output.as_mut_ptr() as *mut _,
            stride as libc::ptrdiff_t,
            above.as_ptr().offset(-1) as *const _,
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            0
          )
        };
      }
    }
    let sum = above[..Self::W].iter().fold(0u32, |acc, &v| { let v: u32 = v.into(); v + acc });
    let avg = ((sum + (Self::W >> 1) as u32) / Self::W as u32).as_();
    for line in output.chunks_mut(stride).take(Self::H) {
      line[..Self::W].iter_mut().for_each(|v| *v = avg);
    }
  }

  #[cfg_attr(feature = "comparative_bench", inline(never))]
  fn pred_h(output: &mut [T], stride: usize, left: &[T]) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_h_avx2(
            output.as_mut_ptr() as *mut _,
            stride as libc::ptrdiff_t,
            left.as_ptr().add(Self::H) as *const _,
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            0
          )
        };
      }
    }
    for (line, l) in
      output.chunks_mut(stride).zip(left[..Self::H].iter().rev())
    {
      for v in &mut line[..Self::W] {
        *v = *l;
      }
    }
  }

  #[cfg_attr(feature = "comparative_bench", inline(never))]
  fn pred_v(output: &mut [T], stride: usize, above: &[T]) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_v_avx2(
            output.as_mut_ptr() as *mut _,
            stride as libc::ptrdiff_t,
            above.as_ptr().offset(-1) as *const _,
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            0
          )
        };
      }
    }
    for line in output.chunks_mut(stride).take(Self::H) {
      line[..Self::W].clone_from_slice(&above[..Self::W])
    }
  }

  #[cfg_attr(feature = "comparative_bench", inline(never))]
  fn pred_paeth(
    output: &mut [T], stride: usize, above: &[T], left: &[T],
    above_left: T
  ) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_paeth_avx2(
            output.as_mut_ptr() as *mut _,
            stride as libc::ptrdiff_t,
            above.as_ptr().offset(-1) as *const _,
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            0
          )
        };
      }
    }
    for r in 0..Self::H {
      for c in 0..Self::W {
        // Top-left pixel is fixed in libaom
        let raw_top_left: i32 = above_left.into();
        let raw_left: i32 = left[Self::H - 1 - r].into();
        let raw_top: i32 = above[c].into();

        let p_base = raw_top + raw_left - raw_top_left;
        let p_left = (p_base - raw_left).abs();
        let p_top = (p_base - raw_top).abs();
        let p_top_left = (p_base - raw_top_left).abs();

        let output_index = r * stride + c;

        // Return nearest to base of left, top and top_left
        if p_left <= p_top && p_left <= p_top_left {
          output[output_index] = raw_left.as_();
        } else if p_top <= p_top_left {
          output[output_index] = raw_top.as_();
        } else {
          output[output_index] = raw_top_left.as_();
        }
      }
    }
  }

  #[cfg_attr(feature = "comparative_bench", inline(never))]
  fn pred_smooth(
    output: &mut [T], stride: usize, above: &[T], left: &[T]
  ) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_smooth_avx2(
            output.as_mut_ptr() as *mut _,
            stride as libc::ptrdiff_t,
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
    assert!(log2_scale as usize + size_of_val(&output[0]) < 31); // ensures no overflow when calculating predictor

    for r in 0..Self::H {
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

        let output_index = r * stride + c;

        output[output_index] = this_pred.as_();
      }
    }
  }

  #[cfg_attr(feature = "comparative_bench", inline(never))]
  fn pred_smooth_h(
    output: &mut [T], stride: usize, above: &[T], left: &[T]
  ) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_smooth_h_avx2(
            output.as_mut_ptr() as *mut _,
            stride as libc::ptrdiff_t,
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
    assert!(log2_scale as usize + size_of_val(&output[0]) < 31); // ensures no overflow when calculating predictor

    for r in 0..Self::H {
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

        let output_index = r * stride + c;

        output[output_index] = this_pred.as_();
      }
    }
  }

  #[cfg_attr(feature = "comparative_bench", inline(never))]
  fn pred_smooth_v(
    output: &mut [T], stride: usize, above: &[T], left: &[T]
  ) {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
        return unsafe {
          rav1e_ipred_smooth_v_avx2(
            output.as_mut_ptr() as *mut _,
            stride as libc::ptrdiff_t,
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
    assert!(log2_scale as usize + size_of_val(&output[0]) < 31); // ensures no overflow when calculating predictor

    for r in 0..Self::H {
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

        let output_index = r * stride + c;

        output[output_index] = this_pred.as_();
      }
    }
  }

  #[target_feature(enable = "ssse3")]
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  unsafe fn pred_cfl_ssse3(
    output: &mut [T], stride: usize, ac: &[i16], alpha: i16,
    bit_depth: usize
  ) {
    let alpha_sign = _mm_set1_epi16(alpha);
    let alpha_q12 = _mm_slli_epi16(_mm_abs_epi16(alpha_sign), 9);
    let dc_scalar: u32 = (*output.as_ptr()).into();
    let dc_q0 = _mm_set1_epi16(dc_scalar as i16);
    let max = _mm_set1_epi16((1 << bit_depth) - 1);

    for j in 0..Self::H {
      let luma = ac.as_ptr().add(32 * j);
      let line = output.as_mut_ptr().add(stride * j);

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

  #[cfg_attr(feature = "comparative_bench", inline(never))]
  fn pred_cfl(
    output: &mut [T], stride: usize, ac: &[i16], alpha: i16,
    bit_depth: usize
  ) {
    if alpha == 0 {
      return;
    }
    assert!(32 >= Self::W);
    assert!(ac.len() >= 32 * (Self::H - 1) + Self::W);
    assert!(stride >= Self::W);
    assert!(output.len() >= stride * (Self::H - 1) + Self::W);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
      if is_x86_feature_detected!("ssse3") {
        return unsafe {
          Self::pred_cfl_ssse3(output, stride, ac, alpha, bit_depth)
        };
      }
    }

    let sample_max = (1 << bit_depth) - 1;
    let avg: i32 = output[0].into();

    for (line, luma) in
      output.chunks_mut(stride).zip(ac.chunks(32)).take(Self::H)
    {
      for (v, &l) in line[..Self::W].iter_mut().zip(luma[..Self::W].iter()) {
        *v =
          (avg + get_scaled_luma_q0(alpha, l)).max(0).min(sample_max).as_();
      }
    }
  }
}

pub trait Inter: Dim {}

#[cfg(all(test, feature = "aom"))]
pub mod test {
  use super::*;
  use rand::{ChaChaRng, Rng, SeedableRng};

  const MAX_ITER: usize = 50000;

  fn setup_pred(
    ra: &mut ChaChaRng
  ) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
    let output = vec![0u16; 32 * 32];
    let above: Vec<u16> = (0..32).map(|_| ra.gen()).collect();
    let left: Vec<u16> = (0..32).map(|_| ra.gen()).collect();

    let o1 = output.clone();
    let o2 = output.clone();

    (above, left, o1, o2)
  }

  macro_rules! wrap_aom_pred_fn {
    ($fn_4x4:ident, $aom_fn:ident) => {
      extern {
        fn $aom_fn(
          dst: *mut u16, stride: libc::ptrdiff_t, bw: libc::c_int,
          bh: libc::c_int, above: *const u16, left: *const u16,
          bd: libc::c_int
        );
      }

      fn $fn_4x4(
        output: &mut [u16], stride: usize, above: &[u16], left: &[u16]
      ) {
        let mut left = left.to_vec();
        left.reverse();
        unsafe {
          $aom_fn(
            output.as_mut_ptr(),
            stride as libc::ptrdiff_t,
            4,
            4,
            above.as_ptr(),
            left.as_ptr(),
            8
          );
        }
      }
    };
  }

  wrap_aom_pred_fn!(pred_dc_4x4, highbd_dc_predictor);
  wrap_aom_pred_fn!(pred_dc_left_4x4, highbd_dc_left_predictor);
  wrap_aom_pred_fn!(pred_dc_top_4x4, highbd_dc_top_predictor);
  wrap_aom_pred_fn!(pred_h_4x4, highbd_h_predictor);
  wrap_aom_pred_fn!(pred_v_4x4, highbd_v_predictor);
  wrap_aom_pred_fn!(pred_paeth_4x4, highbd_paeth_predictor);
  wrap_aom_pred_fn!(pred_smooth_4x4, highbd_smooth_predictor);
  wrap_aom_pred_fn!(pred_smooth_h_4x4, highbd_smooth_h_predictor);
  wrap_aom_pred_fn!(pred_smooth_v_4x4, highbd_smooth_v_predictor);

  extern {
    fn cfl_predict_hbd_c(
      ac_buf_q3: *const i16, dst: *mut u16, stride: libc::ptrdiff_t,
      alpha_q3: libc::c_int, bd: libc::c_int, bw: libc::c_int,
      bh: libc::c_int
    );
  }

  pub fn pred_cfl_4x4(
    output: &mut [u16], stride: usize, ac: &[i16], alpha: i16, bd: i32
  ) {
    unsafe {
      cfl_predict_hbd_c(
        ac.as_ptr(),
        output.as_mut_ptr(),
        stride as libc::ptrdiff_t,
        alpha as libc::c_int,
        bd,
        4,
        4
      );
    }
  }

  fn do_dc_pred(ra: &mut ChaChaRng) -> (Vec<u16>, Vec<u16>) {
    let (above, left, mut o1, mut o2) = setup_pred(ra);

    pred_dc_4x4(&mut o1, 32, &above[..4], &left[..4]);
    Block4x4::pred_dc(&mut o2, 32, &above[..4], &left[..4]);

    (o1, o2)
  }

  fn do_dc_left_pred(ra: &mut ChaChaRng) -> (Vec<u16>, Vec<u16>) {
    let (above, left, mut o1, mut o2) = setup_pred(ra);

    pred_dc_left_4x4(&mut o1, 32, &above[..4], &left[..4]);
    Block4x4::pred_dc_left(&mut o2, 32, &above[..4], &left[..4]);

    (o1, o2)
  }

  fn do_dc_top_pred(ra: &mut ChaChaRng) -> (Vec<u16>, Vec<u16>) {
    let (above, left, mut o1, mut o2) = setup_pred(ra);

    pred_dc_top_4x4(&mut o1, 32, &above[..4], &left[..4]);
    Block4x4::pred_dc_top(&mut o2, 32, &above[..4], &left[..4]);

    (o1, o2)
  }

  fn do_h_pred(ra: &mut ChaChaRng) -> (Vec<u16>, Vec<u16>) {
    let (above, left, mut o1, mut o2) = setup_pred(ra);

    pred_h_4x4(&mut o1, 32, &above[..4], &left[..4]);
    Block4x4::pred_h(&mut o2, 32, &left[..4]);

    (o1, o2)
  }

  fn do_v_pred(ra: &mut ChaChaRng) -> (Vec<u16>, Vec<u16>) {
    let (above, left, mut o1, mut o2) = setup_pred(ra);

    pred_v_4x4(&mut o1, 32, &above[..4], &left[..4]);
    Block4x4::pred_v(&mut o2, 32, &above[..4]);

    (o1, o2)
  }

  fn do_paeth_pred(ra: &mut ChaChaRng) -> (Vec<u16>, Vec<u16>) {
    let (above, left, mut o1, mut o2) = setup_pred(ra);
    let above_left = unsafe { *above.as_ptr().offset(-1) };

    pred_paeth_4x4(&mut o1, 32, &above[..4], &left[..4]);
    Block4x4::pred_paeth(&mut o2, 32, &above[..4], &left[..4], above_left);

    (o1, o2)
  }

  fn do_smooth_pred(ra: &mut ChaChaRng) -> (Vec<u16>, Vec<u16>) {
    let (above, left, mut o1, mut o2) = setup_pred(ra);

    pred_smooth_4x4(&mut o1, 32, &above[..4], &left[..4]);
    Block4x4::pred_smooth(&mut o2, 32, &above[..4], &left[..4]);

    (o1, o2)
  }

  fn do_smooth_h_pred(ra: &mut ChaChaRng) -> (Vec<u16>, Vec<u16>) {
    let (above, left, mut o1, mut o2) = setup_pred(ra);

    pred_smooth_h_4x4(&mut o1, 32, &above[..4], &left[..4]);
    Block4x4::pred_smooth_h(&mut o2, 32, &above[..4], &left[..4]);

    (o1, o2)
  }

  fn do_smooth_v_pred(ra: &mut ChaChaRng) -> (Vec<u16>, Vec<u16>) {
    let (above, left, mut o1, mut o2) = setup_pred(ra);

    pred_smooth_v_4x4(&mut o1, 32, &above[..4], &left[..4]);
    Block4x4::pred_smooth_v(&mut o2, 32, &above[..4], &left[..4]);

    (o1, o2)
  }

  fn setup_cfl_pred(
    ra: &mut ChaChaRng, bit_depth: usize
  ) -> (Vec<u16>, Vec<u16>, Vec<i16>, i16, Vec<u16>, Vec<u16>) {
    let o1 = vec![0u16; 32 * 32];
    let o2 = vec![0u16; 32 * 32];
    let max: u16 = (1 << bit_depth) - 1;
    let above: Vec<u16> =
      (0..32).map(|_| ra.gen()).map(|v: u16| v & max).collect();
    let left: Vec<u16> =
      (0..32).map(|_| ra.gen()).map(|v: u16| v & max).collect();
    let luma_max: i16 = (1 << (bit_depth + 3)) - 1;
    let ac: Vec<i16> = (0..(32 * 32))
      .map(|_| ra.gen())
      .map(|v: i16| (v & luma_max) - (luma_max >> 1))
      .collect();
    let alpha = -1 as i16;

    (above, left, ac, alpha, o1, o2)
  }

  fn do_cfl_pred(ra: &mut ChaChaRng) -> (Vec<u16>, Vec<u16>) {
    let (above, left, ac, alpha, mut o1, mut o2) = setup_cfl_pred(ra, 8);

    pred_dc_4x4(&mut o1, 32, &above[..4], &left[..4]);
    Block4x4::pred_dc(&mut o2, 32, &above[..4], &left[..4]);

    pred_cfl_4x4(&mut o1, 32, &ac, alpha, 8);
    Block4x4::pred_cfl(&mut o2, 32, &ac, alpha, 8);

    (o1, o2)
  }

  fn assert_same(o2: Vec<u16>) {
    for l in o2.chunks(32).take(4) {
      for v in l[..4].windows(2) {
        assert_eq!(v[0], v[1]);
      }
    }
  }

  #[test]
  fn pred_matches() {
    let mut ra = ChaChaRng::from_seed([0; 32]);
    for _ in 0..MAX_ITER {
      let (o1, o2) = do_dc_pred(&mut ra);
      assert_eq!(o1, o2);

      let (o1, o2) = do_dc_left_pred(&mut ra);
      assert_eq!(o1, o2);

      let (o1, o2) = do_dc_top_pred(&mut ra);
      assert_eq!(o1, o2);

      let (o1, o2) = do_h_pred(&mut ra);
      assert_eq!(o1, o2);

      let (o1, o2) = do_v_pred(&mut ra);
      assert_eq!(o1, o2);

      let (o1, o2) = do_paeth_pred(&mut ra);
      assert_eq!(o1, o2);

      let (o1, o2) = do_smooth_pred(&mut ra);
      assert_eq!(o1, o2);

      let (o1, o2) = do_smooth_h_pred(&mut ra);
      assert_eq!(o1, o2);

      let (o1, o2) = do_smooth_v_pred(&mut ra);
      assert_eq!(o1, o2);

      let (o1, o2) = do_cfl_pred(&mut ra);
      assert_eq!(o1, o2);
    }
  }

  #[test]
  fn pred_matches_u8() {
    use util::*;
    let mut edge_buf: AlignedArray<[u8; 2 * MAX_TX_SIZE + 1]> =
      UninitializedAlignedArray();
    for i in 0..edge_buf.array.len() {
      edge_buf.array[i] = (i + 32).saturating_sub(MAX_TX_SIZE).as_();
    }
    let left = &edge_buf.array[MAX_TX_SIZE - 4..MAX_TX_SIZE];
    let above = &edge_buf.array[MAX_TX_SIZE + 1..MAX_TX_SIZE + 5];
    let top_left = edge_buf.array[MAX_TX_SIZE];

    let stride = 4;
    let mut output = vec![0u8; 4 * 4];

    Block4x4::pred_dc(&mut output, stride, above, left);
    assert_eq!(output, [32u8; 16]);

    Block4x4::pred_dc_top(&mut output, stride, above, left);
    assert_eq!(output, [35u8; 16]);

    Block4x4::pred_dc_left(&mut output, stride, above, left);
    assert_eq!(output, [30u8; 16]);

    Block4x4::pred_dc_128(&mut output, stride, 8);
    assert_eq!(output, [128u8; 16]);

    Block4x4::pred_v(&mut output, stride, above);
    assert_eq!(
      output,
      [33, 34, 35, 36, 33, 34, 35, 36, 33, 34, 35, 36, 33, 34, 35, 36]
    );

    Block4x4::pred_h(&mut output, stride, left);
    assert_eq!(
      output,
      [31, 31, 31, 31, 30, 30, 30, 30, 29, 29, 29, 29, 28, 28, 28, 28]
    );

    Block4x4::pred_paeth(&mut output, stride, above, left, top_left);
    assert_eq!(
      output,
      [32, 34, 35, 36, 30, 32, 32, 36, 29, 32, 32, 32, 28, 28, 32, 32]
    );

    Block4x4::pred_smooth(&mut output, stride, above, left);
    assert_eq!(
      output,
      [32, 34, 35, 35, 30, 32, 33, 34, 29, 31, 32, 32, 29, 30, 32, 32]
    );

    Block4x4::pred_smooth_h(&mut output, stride, above, left);
    assert_eq!(
      output,
      [31, 33, 34, 35, 30, 33, 34, 35, 29, 32, 34, 34, 28, 31, 33, 34]
    );

    Block4x4::pred_smooth_v(&mut output, stride, above, left);
    assert_eq!(
      output,
      [33, 34, 35, 36, 31, 31, 32, 33, 30, 30, 30, 31, 29, 30, 30, 30]
    );
  }

  #[test]
  fn pred_same() {
    let mut ra = ChaChaRng::from_seed([0; 32]);
    for _ in 0..MAX_ITER {
      let (_, o2) = do_dc_pred(&mut ra);

      assert_same(o2)
    }
  }

  #[test]
  fn pred_max() {
    let max12bit = 4096 - 1;
    let above = [max12bit; 32];
    let left = [max12bit; 32];

    let mut o = vec![0u16; 32 * 32];

    Block4x4::pred_dc(&mut o, 32, &above[..4], &left[..4]);

    for l in o.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    Block4x4::pred_h(&mut o, 32, &left[..4]);

    for l in o.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    Block4x4::pred_v(&mut o, 32, &above[..4]);

    for l in o.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    let above_left = unsafe { *above.as_ptr().offset(-1) };

    Block4x4::pred_paeth(&mut o, 32, &above[..4], &left[..4], above_left);

    for l in o.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    Block4x4::pred_smooth(&mut o, 32, &above[..4], &left[..4]);

    for l in o.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    Block4x4::pred_smooth_h(&mut o, 32, &above[..4], &left[..4]);

    for l in o.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }

    Block4x4::pred_smooth_v(&mut o, 32, &above[..4], &left[..4]);

    for l in o.chunks(32).take(4) {
      for v in l[..4].iter() {
        assert_eq!(*v, max12bit);
      }
    }
  }
}
