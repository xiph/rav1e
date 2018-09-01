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

use libc;

use context::MAX_TX_SIZE;
use partition::*;
use std::mem::*;
use context::INTRA_MODES;

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

pub static RAV1E_INTER_MODES: &'static [PredictionMode] = &[
  PredictionMode::GLOBALMV,
  PredictionMode::NEARESTMV,
  PredictionMode::NEWMV,
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
    // TODO: enable extra weights for TX64X64
    // bs = 64
    /*255, 248, 240, 233, 225, 218, 210, 203, 196, 189, 182, 176, 169, 163, 156,
    150, 144, 138, 133, 127, 121, 116, 111, 106, 101, 96, 91, 86, 82, 77, 73, 69,
    65, 61, 57, 54, 50, 47, 44, 41, 38, 35, 32, 29, 27, 25, 22, 20, 18, 16, 15,
    13, 12, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 4,*/
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
  NEED_ABOVE | NEED_LEFT,                   // DC
  NEED_ABOVE,                               // V
  NEED_LEFT,                                // H
  NEED_ABOVE | NEED_ABOVERIGHT,             // D45
  NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT,  // D135
  NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT,  // D113
  NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT,  // D157
  NEED_LEFT | NEED_BOTTOMLEFT,              // D203
  NEED_ABOVE | NEED_ABOVERIGHT,             // D67
  NEED_LEFT | NEED_ABOVE,                   // SMOOTH
  NEED_LEFT | NEED_ABOVE,                   // SMOOTH_V
  NEED_LEFT | NEED_ABOVE,                   // SMOOTH_H
  NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT,  // PAETH
];

extern {
  #[cfg(test)]
  fn highbd_dc_predictor(
    dst: *mut u16, stride: libc::ptrdiff_t, bw: libc::c_int, bh: libc::c_int,
    above: *const u16, left: *const u16, bd: libc::c_int
  );

  fn highbd_dc_left_predictor(
    dst: *mut u16, stride: libc::ptrdiff_t, bw: libc::c_int, bh: libc::c_int,
    above: *const u16, left: *const u16, bd: libc::c_int
  );

  fn highbd_dc_top_predictor(
    dst: *mut u16, stride: libc::ptrdiff_t, bw: libc::c_int, bh: libc::c_int,
    above: *const u16, left: *const u16, bd: libc::c_int
  );

  #[cfg(test)]
  fn highbd_h_predictor(
    dst: *mut u16, stride: libc::ptrdiff_t, bw: libc::c_int, bh: libc::c_int,
    above: *const u16, left: *const u16, bd: libc::c_int
  );

  #[cfg(test)]
  fn highbd_v_predictor(
    dst: *mut u16, stride: libc::ptrdiff_t, bw: libc::c_int, bh: libc::c_int,
    above: *const u16, left: *const u16, bd: libc::c_int
  );

  #[cfg(test)]
  fn highbd_paeth_predictor(
    dst: *mut u16, stride: libc::ptrdiff_t, bw: libc::c_int, bh: libc::c_int,
    above: *const u16, left: *const u16, bd: libc::c_int
  );

  #[cfg(test)]
  fn highbd_smooth_predictor(
    dst: *mut u16, stride: libc::ptrdiff_t, bw: libc::c_int, bh: libc::c_int,
    above: *const u16, left: *const u16, bd: libc::c_int
  );

  #[cfg(test)]
  fn highbd_smooth_h_predictor(
    dst: *mut u16, stride: libc::ptrdiff_t, bw: libc::c_int, bh: libc::c_int,
    above: *const u16, left: *const u16, bd: libc::c_int
  );

  #[cfg(test)]
  fn highbd_smooth_v_predictor(
    dst: *mut u16, stride: libc::ptrdiff_t, bw: libc::c_int, bh: libc::c_int,
    above: *const u16, left: *const u16, bd: libc::c_int
  );

  #[cfg(test)]
  fn cfl_predict_hbd_c(
    ac_buf_q3: *const i16, dst: *mut u16, stride: libc::ptrdiff_t,
    alpha_q3: libc::c_int, bd: libc::c_int, bw: libc::c_int, bh: libc::c_int
  );
}

pub trait Dim {
  const W: usize;
  const H: usize;
}

pub struct Block4x4;

impl Dim for Block4x4 {
  const W: usize = 4;
  const H: usize = 4;
}

pub struct Block8x8;

impl Dim for Block8x8 {
  const W: usize = 8;
  const H: usize = 8;
}

pub struct Block16x16;

impl Dim for Block16x16 {
  const W: usize = 16;
  const H: usize = 16;
}

pub struct Block32x32;

impl Dim for Block32x32 {
  const W: usize = 32;
  const H: usize = 32;
}

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

pub trait Intra: Dim {
  #[cfg_attr(feature = "comparative_bench", inline(never))]
  fn pred_dc(output: &mut [u16], stride: usize, above: &[u16], left: &[u16]) {
    let edges = left[..Self::H].iter().chain(above[..Self::W].iter());
    let len = (Self::W + Self::H) as u32;
    let avg =
      ((edges.fold(0, |acc, &v| acc + v as u32) + (len >> 1)) / len) as u16;

    for line in output.chunks_mut(stride).take(Self::H) {
      for v in &mut line[..Self::W] {
        *v = avg;
      }
    }
  }

  #[cfg_attr(feature = "comparative_bench", inline(never))]
  fn pred_dc_128(output: &mut [u16], stride: usize, bit_depth: usize) {
    for y in 0..Self::H {
      for x in 0..Self::W {
        output[y * stride + x] = 128 << (bit_depth - 8);
      }
    }
  }

  #[cfg_attr(feature = "comparative_bench", inline(never))]
  fn pred_dc_left(
    output: &mut [u16], stride: usize, above: &[u16], left: &[u16], bit_depth: usize
  ) {
    unsafe {
      highbd_dc_left_predictor(
        output.as_mut_ptr(),
        stride as libc::ptrdiff_t,
        Self::W as libc::c_int,
        Self::H as libc::c_int,
        above.as_ptr(),
        left.as_ptr(),
        bit_depth as libc::c_int
      );
    }
  }

  #[cfg_attr(feature = "comparative_bench", inline(never))]
  fn pred_dc_top(
    output: &mut [u16], stride: usize, above: &[u16], left: &[u16], bit_depth: usize
  ) {
    unsafe {
      highbd_dc_top_predictor(
        output.as_mut_ptr(),
        stride as libc::ptrdiff_t,
        Self::W as libc::c_int,
        Self::H as libc::c_int,
        above.as_ptr(),
        left.as_ptr(),
        bit_depth as libc::c_int
      );
    }
  }

  #[cfg_attr(feature = "comparative_bench", inline(never))]
  fn pred_h(output: &mut [u16], stride: usize, left: &[u16]) {
    for (line, l) in output.chunks_mut(stride).zip(left[..Self::H].iter()) {
      for v in &mut line[..Self::W] {
        *v = *l;
      }
    }
  }

  #[cfg_attr(feature = "comparative_bench", inline(never))]
  fn pred_v(output: &mut [u16], stride: usize, above: &[u16]) {
    for line in output.chunks_mut(stride).take(Self::H) {
      line[..Self::W].clone_from_slice(&above[..Self::W])
    }
  }

  #[cfg_attr(feature = "comparative_bench", inline(never))]
  fn pred_paeth(
    output: &mut [u16], stride: usize, above: &[u16], left: &[u16],
    above_left: u16
  ) {
    for r in 0..Self::H {
      for c in 0..Self::W {
        // Top-left pixel is fixed in libaom
        let raw_top_left = above_left as i32;
        let raw_left = left[r] as i32;
        let raw_top = above[c] as i32;

        let p_base = raw_top + raw_left - raw_top_left;
        let p_left = (p_base - raw_left).abs();
        let p_top = (p_base - raw_top).abs();
        let p_top_left = (p_base - raw_top_left).abs();

        let output_index = r * stride + c;

        // Return nearest to base of left, top and top_left
        if p_left <= p_top && p_left <= p_top_left {
          output[output_index] = raw_left as u16;
        } else if p_top <= p_top_left {
          output[output_index] = raw_top as u16;
        } else {
          output[output_index] = raw_top_left as u16;
        }
      }
    }
  }

  #[cfg_attr(feature = "comparative_bench", inline(never))]
  fn pred_smooth(
    output: &mut [u16], stride: usize, above: &[u16], left: &[u16]
  ) {
    let below_pred = left[Self::H - 1]; // estimated by bottom-left pixel
    let right_pred = above[Self::W - 1]; // estimated by top-right pixel
    let sm_weights_w = &sm_weight_arrays[Self::W..];
    let sm_weights_h = &sm_weight_arrays[Self::H..];

    let log2_scale = 1 + sm_weight_log2_scale;
    let scale = 1_u16 << sm_weight_log2_scale as u16;

    // Weights sanity checks
    assert!((sm_weights_w[0] as u16) < scale);
    assert!((sm_weights_h[0] as u16) < scale);
    assert!((scale - sm_weights_w[Self::W - 1] as u16) < scale);
    assert!((scale - sm_weights_h[Self::H - 1] as u16) < scale);
    assert!(log2_scale as usize + size_of_val(&output[0]) < 31); // ensures no overflow when calculating predictor

    for r in 0..Self::H {
      for c in 0..Self::W {
        let pixels = [above[c], below_pred, left[r], right_pred];

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
          .map(|(w, p)| (*w as u32) * (*p as u32))
          .sum();
        this_pred = (this_pred + (1 << (log2_scale - 1))) >> log2_scale;

        let output_index = r * stride + c;

        output[output_index] = this_pred as u16;
      }
    }
  }

  #[cfg_attr(feature = "comparative_bench", inline(never))]
  fn pred_smooth_h(
    output: &mut [u16], stride: usize, above: &[u16], left: &[u16]
  ) {
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
        let pixels = [left[r], right_pred];
        let weights = [sm_weights[c] as u16, scale - sm_weights[c] as u16];

        assert!(scale >= sm_weights[c] as u16);

        let mut this_pred: u32 = weights
          .iter()
          .zip(pixels.iter())
          .map(|(w, p)| (*w as u32) * (*p as u32))
          .sum();
        this_pred = (this_pred + (1 << (log2_scale - 1))) >> log2_scale;

        let output_index = r * stride + c;

        output[output_index] = this_pred as u16;
      }
    }
  }

  #[cfg_attr(feature = "comparative_bench", inline(never))]
  fn pred_smooth_v(
    output: &mut [u16], stride: usize, above: &[u16], left: &[u16]
  ) {
    let below_pred = left[Self::H - 1]; // estimated by bottom-left pixel
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
          .map(|(w, p)| (*w as u32) * (*p as u32))
          .sum();
        this_pred = (this_pred + (1 << (log2_scale - 1))) >> log2_scale;

        let output_index = r * stride + c;

        output[output_index] = this_pred as u16;
      }
    }
  }

  #[target_feature(enable = "ssse3")]
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  unsafe fn pred_cfl_ssse3(
    output: &mut [u16], stride: usize, ac: &[i16], alpha: i16,
    bit_depth: usize
  ) {
    let alpha_sign = _mm_set1_epi16(alpha);
    let alpha_q12 = _mm_slli_epi16(_mm_abs_epi16(alpha_sign), 9);
    let dc_q0 = _mm_set1_epi16(*output.as_ptr() as i16);
    let max = _mm_set1_epi16((1 << bit_depth) - 1);

    for j in 0..Self::H {
      let luma = ac.as_ptr().offset((32 * j) as isize);
      let line = output.as_mut_ptr().offset((stride * j) as isize);

      let mut i = 0isize;
      while (i as usize) < Self::W {
        let ac_q3 = _mm_loadu_si128(luma.offset(i) as *const _);
        let ac_sign = _mm_sign_epi16(alpha_sign, ac_q3);
        let abs_scaled_luma_q0 =
          _mm_mulhrs_epi16(_mm_abs_epi16(ac_q3), alpha_q12);
        let scaled_luma_q0 = _mm_sign_epi16(abs_scaled_luma_q0, ac_sign);
        let pred = _mm_add_epi16(scaled_luma_q0, dc_q0);
        let res = _mm_min_epi16(max, _mm_max_epi16(pred, _mm_setzero_si128()));
        if Self::W == 4 {
          _mm_storel_epi64(line.offset(i) as *mut _, res);
        } else {
          _mm_storeu_si128(line.offset(i) as *mut _, res);
        }
        i += 8;
      }
    }
  }

  #[cfg_attr(feature = "comparative_bench", inline(never))]
  fn pred_cfl(
    output: &mut [u16], stride: usize, ac: &[i16], alpha: i16,
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
    let avg = output[0] as i32;

    for (line, luma) in
      output.chunks_mut(stride).zip(ac.chunks(32)).take(Self::H)
    {
      for (v, &l) in line[..Self::W].iter_mut().zip(luma[..Self::W].iter()) {
        *v =
          (avg + get_scaled_luma_q0(alpha, l)).max(0).min(sample_max) as u16;
      }
    }
  }
}

pub trait Inter: Dim {

}

impl Intra for Block4x4 {}
impl Intra for Block8x8 {}
impl Intra for Block16x16 {}
impl Intra for Block32x32 {}

#[cfg(test)]
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

  fn pred_dc_4x4(
    output: &mut [u16], stride: usize, above: &[u16], left: &[u16]
  ) {
    unsafe {
      highbd_dc_predictor(
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

  pub fn pred_h_4x4(
    output: &mut [u16], stride: usize, above: &[u16], left: &[u16]
  ) {
    unsafe {
      highbd_h_predictor(
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

  pub fn pred_v_4x4(
    output: &mut [u16], stride: usize, above: &[u16], left: &[u16]
  ) {
    unsafe {
      highbd_v_predictor(
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

  pub fn pred_paeth_4x4(
    output: &mut [u16], stride: usize, above: &[u16], left: &[u16]
  ) {
    unsafe {
      highbd_paeth_predictor(
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

  pub fn pred_smooth_4x4(
    output: &mut [u16], stride: usize, above: &[u16], left: &[u16]
  ) {
    unsafe {
      highbd_smooth_predictor(
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

  pub fn pred_smooth_h_4x4(
    output: &mut [u16], stride: usize, above: &[u16], left: &[u16]
  ) {
    unsafe {
      highbd_smooth_h_predictor(
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

  pub fn pred_smooth_v_4x4(
    output: &mut [u16], stride: usize, above: &[u16], left: &[u16]
  ) {
    unsafe {
      highbd_smooth_v_predictor(
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
