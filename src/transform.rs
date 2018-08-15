// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

extern crate libc;

use partition::TxSize;
use partition::TxType;

use context::clamp;

static COSPI_INV : [i32; 64] =
  [ 4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973,
    3948, 3920, 3889, 3857, 3822, 3784, 3745, 3703, 3659, 3612, 3564,
    3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967, 2896,
    2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019,
    1931, 1842, 1751, 1660, 1567, 1474, 1380, 1285, 1189, 1092, 995,
    897,  799,  700,  601,  501,  401,  301,  201,  101 ];

#[inline]
fn half_btf(w0: i32, in0: i32, w1: i32, in1: i32, bit: i32) -> i32 {
  let result = (w0 * in0) + (w1 * in1);
  round_shift(result, bit)
}

#[inline]
fn round_shift(value: i32, bit: i32) -> i32 {
  (value + (1 << (bit - 1))) >> bit
}

#[inline]
fn clamp_value(value: i32, bit: i32) -> i32 {
  // Do nothing for invalid clamp bit.
  if bit <= 0 {
    value
  }
  else {
    let max_value: i32 = ((1i64 << (bit - 1)) - 1) as i32;
    let min_value: i32 = (-(1i64 << (bit - 1))) as i32;
    clamp(value, min_value, max_value)
  }
}

fn av1_idct4(input: [i32; 4], output: &mut[i32], range: i32) {
  let cos_bit = 12;
  let mut bf0: [i32; 4] = input;
  let mut bf1: [i32; 4] = [0; 4];

  // stage 0

  // stage 1
  bf1[0] = bf0[0];
  bf1[1] = bf0[2];
  bf1[2] = bf0[1];
  bf1[3] = bf0[3];

  // stage 2
  bf0[0] = half_btf(COSPI_INV[32], bf1[0], COSPI_INV[32], bf1[1], cos_bit);
  bf0[1] = half_btf(COSPI_INV[32], bf1[0], -COSPI_INV[32], bf1[1], cos_bit);
  bf0[2] = half_btf(COSPI_INV[48], bf1[2], -COSPI_INV[16], bf1[3], cos_bit);
  bf0[3] = half_btf(COSPI_INV[16], bf1[2], COSPI_INV[48], bf1[3], cos_bit);

  // stage 3
  output[0] = clamp_value(bf0[0] + bf0[3], range);
  output[1] = clamp_value(bf0[1] + bf0[2], range);
  output[2] = clamp_value(bf0[1] - bf0[2], range);
  output[3] = clamp_value(bf0[0] - bf0[3], range);
}

static INV_RANGES : [[i32; 2]; 3] = [[16, 16], [18, 16], [20, 18]];

fn get_ranges(bd: i32) -> [i32; 2] {
  INV_RANGES[((bd - 8) >> 1) as usize]
}

fn inv_txfm2d_add_4x4_rs(input: &[i32], output: &mut [u16],
                         stride: usize,
                         bd : i32) {
  let ranges = get_ranges(bd);
  let mut buffer = [0i32; 4*4];
  for r in 0..4 {
    let input_slice = &input[r*4..(r+1)*4];
    let buffer_slice = &mut buffer[r*4..(r+1)*4];
    let mut temp_in: [i32; 4] = [0; 4];
    for c in 0..4 {
      temp_in[c] = input_slice[c];
    }
    av1_idct4(temp_in, buffer_slice, ranges[0]);
  }

  for c in 0..4 {
    let mut temp_in: [i32; 4] = [0; 4];
    let mut temp_out: [i32; 4] = [0; 4];
    for r in 0..4 {
      temp_in[r] = buffer[r * 4 + c];
    }
    av1_idct4(temp_in, &mut temp_out, ranges[1]);
    for r in 0..4 {
      output[r * stride + c] = clamp(output[r * stride + c] as i32 + round_shift(temp_out[r], 4), 0, (1 << bd) - 1) as u16;
    }
  }
}

// In libaom, functions that have more than one specialization use function
// pointers, so we need to declare them as static fields and call them
// indirectly. Otherwise, we call SSE or C variants directly. To fully
// understand what's going on here you should first understand the Perl code
// in libaom that generates these function bindings.

extern {
  fn av1_fwd_txfm2d_4x4_c(
    input: *const i16, output: *mut i32, stride: libc::c_int,
    tx_type: libc::c_int, bd: libc::c_int
  );
  fn av1_fwd_txfm2d_8x8_c(
    input: *const i16, output: *mut i32, stride: libc::c_int,
    tx_type: libc::c_int, bd: libc::c_int
  );
  fn av1_fwd_txfm2d_16x16_c(
    input: *const i16, output: *mut i32, stride: libc::c_int,
    tx_type: libc::c_int, bd: libc::c_int
  );
  fn av1_fwd_txfm2d_32x32_c(
    input: *const i16, output: *mut i32, stride: libc::c_int,
    tx_type: libc::c_int, bd: libc::c_int
  );
}

extern "C" {
  static av1_inv_txfm2d_add_4x4: extern fn(
    input: *const i32,
    output: *mut u16,
    stride: libc::c_int,
    tx_type: libc::c_int,
    bd: libc::c_int
  );
  fn av1_inv_txfm2d_add_4x4_c(
    input: *const i32,
    output: *mut u16,
    stride: libc::c_int,
    tx_type: libc::c_int,
    bd: libc::c_int
  );
  static av1_inv_txfm2d_add_8x8: extern fn(
    input: *const i32,
    output: *mut u16,
    stride: libc::c_int,
    tx_type: libc::c_int,
    bd: libc::c_int
  ) -> ();
  fn av1_inv_txfm2d_add_8x8_c(
    input: *const i32,
    output: *mut u16,
    stride: libc::c_int,
    tx_type: libc::c_int,
    bd: libc::c_int
  ) -> ();
  static av1_inv_txfm2d_add_16x16: extern fn(
    input: *const i32,
    output: *mut u16,
    stride: libc::c_int,
    tx_type: libc::c_int,
    bd: libc::c_int
  );
  fn av1_inv_txfm2d_add_16x16_c(
    input: *const i32, output: *mut u16, stride: libc::c_int,
    tx_type: libc::c_int, bd: libc::c_int
  );
  static av1_inv_txfm2d_add_32x32: extern fn(
    input: *const i32,
    output: *mut u16,
    stride: libc::c_int,
    tx_type: libc::c_int,
    bd: libc::c_int
  );
  fn av1_inv_txfm2d_add_32x32_c(
    input: *const i32,
    output: *mut u16,
    stride: libc::c_int,
    tx_type: libc::c_int,
    bd: libc::c_int
  );
}

pub fn forward_transform(
  input: &[i16], output: &mut [i32], stride: usize, tx_size: TxSize,
  tx_type: TxType, bit_depth: usize
) {
  match tx_size {
    TxSize::TX_4X4 => fht4x4(input, output, stride, tx_type, bit_depth),
    TxSize::TX_8X8 => fht8x8(input, output, stride, tx_type, bit_depth),
    TxSize::TX_16X16 => fht16x16(input, output, stride, tx_type, bit_depth),
    TxSize::TX_32X32 => fht32x32(input, output, stride, tx_type, bit_depth),
    _ => panic!("unimplemented tx size")
  }
}

pub fn inverse_transform_add(
  input: &[i32], output: &mut [u16], stride: usize, tx_size: TxSize,
  tx_type: TxType, bit_depth: usize
) {
  match tx_size {
    TxSize::TX_4X4 => iht4x4_add(input, output, stride, tx_type, bit_depth),
    TxSize::TX_8X8 => iht8x8_add(input, output, stride, tx_type, bit_depth),
    TxSize::TX_16X16 => iht16x16_add(input, output, stride, tx_type, bit_depth),
    TxSize::TX_32X32 => iht32x32_add(input, output, stride, tx_type, bit_depth),
    _ => panic!("unimplemented tx size")
  }
}

fn fht4x4(input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType, bit_depth: usize) {
  unsafe {
    av1_fwd_txfm2d_4x4_c(
      input.as_ptr(),
      output.as_mut_ptr(),
      stride as libc::c_int,
      tx_type as libc::c_int,
      bit_depth as libc::c_int
    );
  }
}

fn iht4x4_add(
  input: &[i32], output: &mut [u16], stride: usize, tx_type: TxType, bit_depth: usize
) {
  // SIMD code may assert for transform types beyond TxType::IDTX.
  if tx_type < TxType::IDTX {
    if tx_type == TxType::DCT_DCT {
      inv_txfm2d_add_4x4_rs(input, output, stride, 8);
    }
    else {
    unsafe {
      av1_inv_txfm2d_add_4x4(
        input.as_ptr(),
        output.as_mut_ptr(),
        stride as libc::c_int,
        tx_type as libc::c_int,
        bit_depth as libc::c_int
      );
    }
    }
  } else {
    unsafe {
      av1_inv_txfm2d_add_4x4_c(
        input.as_ptr(),
        output.as_mut_ptr(),
        stride as libc::c_int,
        tx_type as libc::c_int,
        bit_depth as libc::c_int
      );
    }
  }
}

fn fht8x8(input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType, bit_depth: usize) {
  unsafe {
    av1_fwd_txfm2d_8x8_c(
      input.as_ptr(),
      output.as_mut_ptr(),
      stride as libc::c_int,
      tx_type as libc::c_int,
      bit_depth as libc::c_int
    );
  }
}

fn iht8x8_add(
  input: &[i32], output: &mut [u16], stride: usize, tx_type: TxType, bit_depth: usize
) {
  // SIMD code may assert for transform types beyond TxType::IDTX.
  if tx_type < TxType::IDTX {
    unsafe {
      av1_inv_txfm2d_add_8x8(
        input.as_ptr(),
        output.as_mut_ptr(),
        stride as libc::c_int,
        tx_type as libc::c_int,
        bit_depth as libc::c_int
      );
    }
  } else {
    unsafe {
      av1_inv_txfm2d_add_8x8_c(
        input.as_ptr(),
        output.as_mut_ptr(),
        stride as libc::c_int,
        tx_type as libc::c_int,
        bit_depth as libc::c_int
      );
    }
  }
}

fn fht16x16(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType, bit_depth: usize
) {
  unsafe {
    av1_fwd_txfm2d_16x16_c(
      input.as_ptr(),
      output.as_mut_ptr(),
      stride as libc::c_int,
      tx_type as libc::c_int,
      bit_depth as libc::c_int
    );
  }
}

fn iht16x16_add(
  input: &[i32], output: &mut [u16], stride: usize, tx_type: TxType, bit_depth: usize
) {
  unsafe {
    if tx_type < TxType::IDTX {
      // SSE C code asserts for transform types beyond TxType::IDTX.
      av1_inv_txfm2d_add_16x16(
        input.as_ptr(),
        output.as_mut_ptr(),
        stride as libc::c_int,
        tx_type as libc::c_int,
        bit_depth as libc::c_int
      );
    } else {
      av1_inv_txfm2d_add_16x16_c(
        input.as_ptr(),
        output.as_mut_ptr(),
        stride as libc::c_int,
        tx_type as libc::c_int,
        bit_depth as libc::c_int
      );
    }
  }
}

fn fht32x32(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType, bit_depth: usize
) {
  unsafe {
    av1_fwd_txfm2d_32x32_c(
      input.as_ptr(),
      output.as_mut_ptr(),
      stride as libc::c_int,
      tx_type as libc::c_int,
      bit_depth as libc::c_int
    );
  }
}

fn iht32x32_add(
  input: &[i32], output: &mut [u16], stride: usize, tx_type: TxType, bit_depth: usize
) {
  unsafe {
    if tx_type < TxType::IDTX {
      // SIMDI code may assert for transform types beyond TxType::IDTX.
      av1_inv_txfm2d_add_32x32(
        input.as_ptr(),
        output.as_mut_ptr(),
        stride as libc::c_int,
        tx_type as libc::c_int,
        bit_depth as libc::c_int
      );
    } else {
      av1_inv_txfm2d_add_32x32_c(
        input.as_ptr(),
        output.as_mut_ptr(),
        stride as libc::c_int,
        tx_type as libc::c_int,
        bit_depth as libc::c_int
      );
    }
  }
}
