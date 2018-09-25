// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

extern crate libc;

use std::cmp;

use partition::TxSize;
use partition::TxType;

use util::*;

// Blocks
use predict::*;

static SQRT2: i32 = 5793;

static COSPI_INV: [i32; 64] = [
  4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973, 3948,
  3920, 3889, 3857, 3822, 3784, 3745, 3703, 3659, 3612, 3564, 3513, 3461,
  3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967, 2896, 2824, 2751, 2675,
  2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019, 1931, 1842, 1751, 1660,
  1567, 1474, 1380, 1285, 1189, 1092, 995, 897, 799, 700, 601, 501, 401, 301,
  201, 101,
];

static SINPI_INV: [i32; 5] = [0, 1321, 2482, 3344, 3803];

// performs half a butterfly
#[inline]
fn half_btf(w0: i32, in0: i32, w1: i32, in1: i32, bit: usize) -> i32 {
  let result = (w0 * in0) + (w1 * in1);
  round_shift(result, bit)
}

#[inline]
fn round_shift(value: i32, bit: usize) -> i32 {
  if bit <= 0 {
    value
  } else {
    (value + (1 << (bit - 1))) >> bit
  }
}

// clamps value to a signed integer type of bit bits
#[inline]
fn clamp_value(value: i32, bit: usize) -> i32 {
  let max_value: i32 = ((1i64 << (bit - 1)) - 1) as i32;
  let min_value: i32 = (-(1i64 << (bit - 1))) as i32;
  clamp(value, min_value, max_value)
}

pub fn av1_idct4(input: &[i32], output: &mut [i32], range: usize) {
  let cos_bit = 12;
  // stage 0

  // stage 1
  let stg1 = [input[0], input[2], input[1], input[3]];

  // stage 2
  let stg2 = [
    half_btf(COSPI_INV[32], stg1[0], COSPI_INV[32], stg1[1], cos_bit),
    half_btf(COSPI_INV[32], stg1[0], -COSPI_INV[32], stg1[1], cos_bit),
    half_btf(COSPI_INV[48], stg1[2], -COSPI_INV[16], stg1[3], cos_bit),
    half_btf(COSPI_INV[16], stg1[2], COSPI_INV[48], stg1[3], cos_bit)
  ];

  // stage 3
  output[0] = clamp_value(stg2[0] + stg2[3], range);
  output[1] = clamp_value(stg2[1] + stg2[2], range);
  output[2] = clamp_value(stg2[1] - stg2[2], range);
  output[3] = clamp_value(stg2[0] - stg2[3], range);
}

fn av1_iadst4(input: &[i32], output: &mut [i32], _range: usize) {
  let bit = 12;

  let x0 = input[0];
  let x1 = input[1];
  let x2 = input[2];
  let x3 = input[3];

  // stage 1
  let s0 = SINPI_INV[1] * x0;
  let s1 = SINPI_INV[2] * x0;
  let s2 = SINPI_INV[3] * x1;
  let s3 = SINPI_INV[4] * x2;
  let s4 = SINPI_INV[1] * x2;
  let s5 = SINPI_INV[2] * x3;
  let s6 = SINPI_INV[4] * x3;

  // stage 2
  let s7 = (x0 - x2) + x3;

  // stage 3
  let s0 = s0 + s3;
  let s1 = s1 - s4;
  let s3 = s2;
  let s2 = SINPI_INV[3] * s7;

  // stage 4
  let s0 = s0 + s5;
  let s1 = s1 - s6;

  // stage 5
  let x0 = s0 + s3;
  let x1 = s1 + s3;
  let x2 = s2;
  let x3 = s0 + s1;

  // stage 6
  let x3 = x3 - s3;

  output[0] = round_shift(x0, bit);
  output[1] = round_shift(x1, bit);
  output[2] = round_shift(x2, bit);
  output[3] = round_shift(x3, bit);
}

fn av1_iidentity4(input: &[i32], output: &mut [i32], _range: usize) {
  for i in 0..4 {
    output[i] = round_shift(SQRT2 * input[i], 12);
  }
}

pub fn av1_idct8(input: &[i32], output: &mut [i32], range: usize) {
  // call idct4
  let temp_in = [ input[0], input[2], input[4], input[6] ];
  let mut temp_out: [i32; 4] = [0; 4];
  av1_idct4(&temp_in, &mut temp_out, range);

  let cos_bit = 12;

  // stage 0

  // stage 1
  let stg1 = [ input[1], input[5], input[3], input[7] ];

  // stage 2
  let stg2 = [
    half_btf(COSPI_INV[56], stg1[0], -COSPI_INV[8], stg1[3], cos_bit),
    half_btf(COSPI_INV[24], stg1[1], -COSPI_INV[40], stg1[2], cos_bit),
    half_btf(COSPI_INV[40], stg1[1], COSPI_INV[24], stg1[2], cos_bit),
    half_btf(COSPI_INV[8], stg1[0], COSPI_INV[56], stg1[3], cos_bit)
  ];

  // stage 3
  let stg3 = [
    clamp_value(stg2[0] + stg2[1], range),
    clamp_value(stg2[0] - stg2[1], range),
    clamp_value(-stg2[2] + stg2[3], range),
    clamp_value(stg2[2] + stg2[3], range)
  ];

  // stage 4
  let stg4 = [
    stg3[0],
    half_btf(-COSPI_INV[32], stg3[1], COSPI_INV[32], stg3[2], cos_bit),
    half_btf(COSPI_INV[32], stg3[1], COSPI_INV[32], stg3[2], cos_bit),
    stg3[3]
  ];

  // stage 5
  output[0] = clamp_value(temp_out[0] + stg4[3], range);
  output[1] = clamp_value(temp_out[1] + stg4[2], range);
  output[2] = clamp_value(temp_out[2] + stg4[1], range);
  output[3] = clamp_value(temp_out[3] + stg4[0], range);
  output[4] = clamp_value(temp_out[3] - stg4[0], range);
  output[5] = clamp_value(temp_out[2] - stg4[1], range);
  output[6] = clamp_value(temp_out[1] - stg4[2], range);
  output[7] = clamp_value(temp_out[0] - stg4[3], range);
}

fn av1_iadst8(input: &[i32], output: &mut [i32], range: usize) {
  let cos_bit = 12;
  // stage 0

  // stage 1
  let stg1 = [
    input[7], input[0], input[5], input[2], input[3], input[4], input[1],
    input[6],
  ];

  // stage 2
  let stg2 = [
    half_btf(COSPI_INV[4], stg1[0], COSPI_INV[60], stg1[1], cos_bit),
    half_btf(COSPI_INV[60], stg1[0], -COSPI_INV[4], stg1[1], cos_bit),
    half_btf(COSPI_INV[20], stg1[2], COSPI_INV[44], stg1[3], cos_bit),
    half_btf(COSPI_INV[44], stg1[2], -COSPI_INV[20], stg1[3], cos_bit),
    half_btf(COSPI_INV[36], stg1[4], COSPI_INV[28], stg1[5], cos_bit),
    half_btf(COSPI_INV[28], stg1[4], -COSPI_INV[36], stg1[5], cos_bit),
    half_btf(COSPI_INV[52], stg1[6], COSPI_INV[12], stg1[7], cos_bit),
    half_btf(COSPI_INV[12], stg1[6], -COSPI_INV[52], stg1[7], cos_bit)
  ];

  // stage 3
  let stg3 = [
    clamp_value(stg2[0] + stg2[4], range),
    clamp_value(stg2[1] + stg2[5], range),
    clamp_value(stg2[2] + stg2[6], range),
    clamp_value(stg2[3] + stg2[7], range),
    clamp_value(stg2[0] - stg2[4], range),
    clamp_value(stg2[1] - stg2[5], range),
    clamp_value(stg2[2] - stg2[6], range),
    clamp_value(stg2[3] - stg2[7], range)
  ];

  // stage 4
  let stg4 = [
    stg3[0],
    stg3[1],
    stg3[2],
    stg3[3],
    half_btf(COSPI_INV[16], stg3[4], COSPI_INV[48], stg3[5], cos_bit),
    half_btf(COSPI_INV[48], stg3[4], -COSPI_INV[16], stg3[5], cos_bit),
    half_btf(-COSPI_INV[48], stg3[6], COSPI_INV[16], stg3[7], cos_bit),
    half_btf(COSPI_INV[16], stg3[6], COSPI_INV[48], stg3[7], cos_bit)
  ];

  // stage 5
  let stg5 = [
    clamp_value(stg4[0] + stg4[2], range),
    clamp_value(stg4[1] + stg4[3], range),
    clamp_value(stg4[0] - stg4[2], range),
    clamp_value(stg4[1] - stg4[3], range),
    clamp_value(stg4[4] + stg4[6], range),
    clamp_value(stg4[5] + stg4[7], range),
    clamp_value(stg4[4] - stg4[6], range),
    clamp_value(stg4[5] - stg4[7], range)
  ];

  // stage 6
  let stg6 = [
    stg5[0],
    stg5[1],
    half_btf(COSPI_INV[32], stg5[2], COSPI_INV[32], stg5[3], cos_bit),
    half_btf(COSPI_INV[32], stg5[2], -COSPI_INV[32], stg5[3], cos_bit),
    stg5[4],
    stg5[5],
    half_btf(COSPI_INV[32], stg5[6], COSPI_INV[32], stg5[7], cos_bit),
    half_btf(COSPI_INV[32], stg5[6], -COSPI_INV[32], stg5[7], cos_bit)
  ];

  // stage 7
  output[0] = stg6[0];
  output[1] = -stg6[4];
  output[2] = stg6[6];
  output[3] = -stg6[2];
  output[4] = stg6[3];
  output[5] = -stg6[7];
  output[6] = stg6[5];
  output[7] = -stg6[1];
}

fn av1_iidentity8(input: &[i32], output: &mut [i32], _range: usize) {
  for i in 0..8 {
    output[i] = 2 * input[i];
  }
}

fn av1_idct16(input: &[i32], output: &mut [i32], range: usize) {
  let cos_bit = 12;
  // stage 0

  // stage 1
  let stg1 = [
    input[0], input[8], input[4], input[12], input[2], input[10], input[6],
    input[14], input[1], input[9], input[5], input[13], input[3], input[11],
    input[7], input[15],
  ];

  // stage 2
  let stg2 = [
    stg1[0],
    stg1[1],
    stg1[2],
    stg1[3],
    stg1[4],
    stg1[5],
    stg1[6],
    stg1[7],
    half_btf(COSPI_INV[60], stg1[8], -COSPI_INV[4], stg1[15], cos_bit),
    half_btf(COSPI_INV[28], stg1[9], -COSPI_INV[36], stg1[14], cos_bit),
    half_btf(COSPI_INV[44], stg1[10], -COSPI_INV[20], stg1[13], cos_bit),
    half_btf(COSPI_INV[12], stg1[11], -COSPI_INV[52], stg1[12], cos_bit),
    half_btf(COSPI_INV[52], stg1[11], COSPI_INV[12], stg1[12], cos_bit),
    half_btf(COSPI_INV[20], stg1[10], COSPI_INV[44], stg1[13], cos_bit),
    half_btf(COSPI_INV[36], stg1[9], COSPI_INV[28], stg1[14], cos_bit),
    half_btf(COSPI_INV[4], stg1[8], COSPI_INV[60], stg1[15], cos_bit)
  ];

  // stage 3
  let stg3 = [
    stg2[0],
    stg2[1],
    stg2[2],
    stg2[3],
    half_btf(COSPI_INV[56], stg2[4], -COSPI_INV[8], stg2[7], cos_bit),
    half_btf(COSPI_INV[24], stg2[5], -COSPI_INV[40], stg2[6], cos_bit),
    half_btf(COSPI_INV[40], stg2[5], COSPI_INV[24], stg2[6], cos_bit),
    half_btf(COSPI_INV[8], stg2[4], COSPI_INV[56], stg2[7], cos_bit),
    clamp_value(stg2[8] + stg2[9], range),
    clamp_value(stg2[8] - stg2[9], range),
    clamp_value(-stg2[10] + stg2[11], range),
    clamp_value(stg2[10] + stg2[11], range),
    clamp_value(stg2[12] + stg2[13], range),
    clamp_value(stg2[12] - stg2[13], range),
    clamp_value(-stg2[14] + stg2[15], range),
    clamp_value(stg2[14] + stg2[15], range)
  ];

  // stage 4
  let stg4 = [
    half_btf(COSPI_INV[32], stg3[0], COSPI_INV[32], stg3[1], cos_bit),
    half_btf(COSPI_INV[32], stg3[0], -COSPI_INV[32], stg3[1], cos_bit),
    half_btf(COSPI_INV[48], stg3[2], -COSPI_INV[16], stg3[3], cos_bit),
    half_btf(COSPI_INV[16], stg3[2], COSPI_INV[48], stg3[3], cos_bit),
    clamp_value(stg3[4] + stg3[5], range),
    clamp_value(stg3[4] - stg3[5], range),
    clamp_value(-stg3[6] + stg3[7], range),
    clamp_value(stg3[6] + stg3[7], range),
    stg3[8],
    half_btf(-COSPI_INV[16], stg3[9], COSPI_INV[48], stg3[14], cos_bit),
    half_btf(-COSPI_INV[48], stg3[10], -COSPI_INV[16], stg3[13], cos_bit),
    stg3[11],
    stg3[12],
    half_btf(-COSPI_INV[16], stg3[10], COSPI_INV[48], stg3[13], cos_bit),
    half_btf(COSPI_INV[48], stg3[9], COSPI_INV[16], stg3[14], cos_bit),
    stg3[15]
  ];

  // stage 5
  let stg5 = [
    clamp_value(stg4[0] + stg4[3], range),
    clamp_value(stg4[1] + stg4[2], range),
    clamp_value(stg4[1] - stg4[2], range),
    clamp_value(stg4[0] - stg4[3], range),
    stg4[4],
    half_btf(-COSPI_INV[32], stg4[5], COSPI_INV[32], stg4[6], cos_bit),
    half_btf(COSPI_INV[32], stg4[5], COSPI_INV[32], stg4[6], cos_bit),
    stg4[7],
    clamp_value(stg4[8] + stg4[11], range),
    clamp_value(stg4[9] + stg4[10], range),
    clamp_value(stg4[9] - stg4[10], range),
    clamp_value(stg4[8] - stg4[11], range),
    clamp_value(-stg4[12] + stg4[15], range),
    clamp_value(-stg4[13] + stg4[14], range),
    clamp_value(stg4[13] + stg4[14], range),
    clamp_value(stg4[12] + stg4[15], range)
  ];

  // stage 6
  let stg6 = [
    clamp_value(stg5[0] + stg5[7], range),
    clamp_value(stg5[1] + stg5[6], range),
    clamp_value(stg5[2] + stg5[5], range),
    clamp_value(stg5[3] + stg5[4], range),
    clamp_value(stg5[3] - stg5[4], range),
    clamp_value(stg5[2] - stg5[5], range),
    clamp_value(stg5[1] - stg5[6], range),
    clamp_value(stg5[0] - stg5[7], range),
    stg5[8],
    stg5[9],
    half_btf(-COSPI_INV[32], stg5[10], COSPI_INV[32], stg5[13], cos_bit),
    half_btf(-COSPI_INV[32], stg5[11], COSPI_INV[32], stg5[12], cos_bit),
    half_btf(COSPI_INV[32], stg5[11], COSPI_INV[32], stg5[12], cos_bit),
    half_btf(COSPI_INV[32], stg5[10], COSPI_INV[32], stg5[13], cos_bit),
    stg5[14],
    stg5[15]
  ];

  // stage 7
  output[0] = clamp_value(stg6[0] + stg6[15], range);
  output[1] = clamp_value(stg6[1] + stg6[14], range);
  output[2] = clamp_value(stg6[2] + stg6[13], range);
  output[3] = clamp_value(stg6[3] + stg6[12], range);
  output[4] = clamp_value(stg6[4] + stg6[11], range);
  output[5] = clamp_value(stg6[5] + stg6[10], range);
  output[6] = clamp_value(stg6[6] + stg6[9], range);
  output[7] = clamp_value(stg6[7] + stg6[8], range);
  output[8] = clamp_value(stg6[7] - stg6[8], range);
  output[9] = clamp_value(stg6[6] - stg6[9], range);
  output[10] = clamp_value(stg6[5] - stg6[10], range);
  output[11] = clamp_value(stg6[4] - stg6[11], range);
  output[12] = clamp_value(stg6[3] - stg6[12], range);
  output[13] = clamp_value(stg6[2] - stg6[13], range);
  output[14] = clamp_value(stg6[1] - stg6[14], range);
  output[15] = clamp_value(stg6[0] - stg6[15], range);
}

fn av1_iadst16(input: &[i32], output: &mut [i32], range: usize) {
  let cos_bit = 12;
  // stage 0

  // stage 1
  let stg1 = [
    input[15], input[0], input[13], input[2], input[11], input[4], input[9],
    input[6], input[7], input[8], input[5], input[10], input[3], input[12],
    input[1], input[14],
  ];

  // stage 2
  let stg2 = [
    half_btf(COSPI_INV[2], stg1[0], COSPI_INV[62], stg1[1], cos_bit),
    half_btf(COSPI_INV[62], stg1[0], -COSPI_INV[2], stg1[1], cos_bit),
    half_btf(COSPI_INV[10], stg1[2], COSPI_INV[54], stg1[3], cos_bit),
    half_btf(COSPI_INV[54], stg1[2], -COSPI_INV[10], stg1[3], cos_bit),
    half_btf(COSPI_INV[18], stg1[4], COSPI_INV[46], stg1[5], cos_bit),
    half_btf(COSPI_INV[46], stg1[4], -COSPI_INV[18], stg1[5], cos_bit),
    half_btf(COSPI_INV[26], stg1[6], COSPI_INV[38], stg1[7], cos_bit),
    half_btf(COSPI_INV[38], stg1[6], -COSPI_INV[26], stg1[7], cos_bit),
    half_btf(COSPI_INV[34], stg1[8], COSPI_INV[30], stg1[9], cos_bit),
    half_btf(COSPI_INV[30], stg1[8], -COSPI_INV[34], stg1[9], cos_bit),
    half_btf(COSPI_INV[42], stg1[10], COSPI_INV[22], stg1[11], cos_bit),
    half_btf(COSPI_INV[22], stg1[10], -COSPI_INV[42], stg1[11], cos_bit),
    half_btf(COSPI_INV[50], stg1[12], COSPI_INV[14], stg1[13], cos_bit),
    half_btf(COSPI_INV[14], stg1[12], -COSPI_INV[50], stg1[13], cos_bit),
    half_btf(COSPI_INV[58], stg1[14], COSPI_INV[6], stg1[15], cos_bit),
    half_btf(COSPI_INV[6], stg1[14], -COSPI_INV[58], stg1[15], cos_bit)
  ];

  // stage 3
  let stg3 = [
    clamp_value(stg2[0] + stg2[8], range),
    clamp_value(stg2[1] + stg2[9], range),
    clamp_value(stg2[2] + stg2[10], range),
    clamp_value(stg2[3] + stg2[11], range),
    clamp_value(stg2[4] + stg2[12], range),
    clamp_value(stg2[5] + stg2[13], range),
    clamp_value(stg2[6] + stg2[14], range),
    clamp_value(stg2[7] + stg2[15], range),
    clamp_value(stg2[0] - stg2[8], range),
    clamp_value(stg2[1] - stg2[9], range),
    clamp_value(stg2[2] - stg2[10], range),
    clamp_value(stg2[3] - stg2[11], range),
    clamp_value(stg2[4] - stg2[12], range),
    clamp_value(stg2[5] - stg2[13], range),
    clamp_value(stg2[6] - stg2[14], range),
    clamp_value(stg2[7] - stg2[15], range)
  ];

  // stage 4
  let stg4 = [
    stg3[0],
    stg3[1],
    stg3[2],
    stg3[3],
    stg3[4],
    stg3[5],
    stg3[6],
    stg3[7],
    half_btf(COSPI_INV[8], stg3[8], COSPI_INV[56], stg3[9], cos_bit),
    half_btf(COSPI_INV[56], stg3[8], -COSPI_INV[8], stg3[9], cos_bit),
    half_btf(COSPI_INV[40], stg3[10], COSPI_INV[24], stg3[11], cos_bit),
    half_btf(COSPI_INV[24], stg3[10], -COSPI_INV[40], stg3[11], cos_bit),
    half_btf(-COSPI_INV[56], stg3[12], COSPI_INV[8], stg3[13], cos_bit),
    half_btf(COSPI_INV[8], stg3[12], COSPI_INV[56], stg3[13], cos_bit),
    half_btf(-COSPI_INV[24], stg3[14], COSPI_INV[40], stg3[15], cos_bit),
    half_btf(COSPI_INV[40], stg3[14], COSPI_INV[24], stg3[15], cos_bit)
  ];

  // stage 5
  let stg5 = [
    clamp_value(stg4[0] + stg4[4], range),
    clamp_value(stg4[1] + stg4[5], range),
    clamp_value(stg4[2] + stg4[6], range),
    clamp_value(stg4[3] + stg4[7], range),
    clamp_value(stg4[0] - stg4[4], range),
    clamp_value(stg4[1] - stg4[5], range),
    clamp_value(stg4[2] - stg4[6], range),
    clamp_value(stg4[3] - stg4[7], range),
    clamp_value(stg4[8] + stg4[12], range),
    clamp_value(stg4[9] + stg4[13], range),
    clamp_value(stg4[10] + stg4[14], range),
    clamp_value(stg4[11] + stg4[15], range),
    clamp_value(stg4[8] - stg4[12], range),
    clamp_value(stg4[9] - stg4[13], range),
    clamp_value(stg4[10] - stg4[14], range),
    clamp_value(stg4[11] - stg4[15], range)
  ];

  // stage 6
  let stg6 = [
    stg5[0],
    stg5[1],
    stg5[2],
    stg5[3],
    half_btf(COSPI_INV[16], stg5[4], COSPI_INV[48], stg5[5], cos_bit),
    half_btf(COSPI_INV[48], stg5[4], -COSPI_INV[16], stg5[5], cos_bit),
    half_btf(-COSPI_INV[48], stg5[6], COSPI_INV[16], stg5[7], cos_bit),
    half_btf(COSPI_INV[16], stg5[6], COSPI_INV[48], stg5[7], cos_bit),
    stg5[8],
    stg5[9],
    stg5[10],
    stg5[11],
    half_btf(COSPI_INV[16], stg5[12], COSPI_INV[48], stg5[13], cos_bit),
    half_btf(COSPI_INV[48], stg5[12], -COSPI_INV[16], stg5[13], cos_bit),
    half_btf(-COSPI_INV[48], stg5[14], COSPI_INV[16], stg5[15], cos_bit),
    half_btf(COSPI_INV[16], stg5[14], COSPI_INV[48], stg5[15], cos_bit)
  ];

  // stage 7
  let stg7 = [
    clamp_value(stg6[0] + stg6[2], range),
    clamp_value(stg6[1] + stg6[3], range),
    clamp_value(stg6[0] - stg6[2], range),
    clamp_value(stg6[1] - stg6[3], range),
    clamp_value(stg6[4] + stg6[6], range),
    clamp_value(stg6[5] + stg6[7], range),
    clamp_value(stg6[4] - stg6[6], range),
    clamp_value(stg6[5] - stg6[7], range),
    clamp_value(stg6[8] + stg6[10], range),
    clamp_value(stg6[9] + stg6[11], range),
    clamp_value(stg6[8] - stg6[10], range),
    clamp_value(stg6[9] - stg6[11], range),
    clamp_value(stg6[12] + stg6[14], range),
    clamp_value(stg6[13] + stg6[15], range),
    clamp_value(stg6[12] - stg6[14], range),
    clamp_value(stg6[13] - stg6[15], range)
  ];

  // stage 8
  let stg8 = [
    stg7[0],
    stg7[1],
    half_btf(COSPI_INV[32], stg7[2], COSPI_INV[32], stg7[3], cos_bit),
    half_btf(COSPI_INV[32], stg7[2], -COSPI_INV[32], stg7[3], cos_bit),
    stg7[4],
    stg7[5],
    half_btf(COSPI_INV[32], stg7[6], COSPI_INV[32], stg7[7], cos_bit),
    half_btf(COSPI_INV[32], stg7[6], -COSPI_INV[32], stg7[7], cos_bit),
    stg7[8],
    stg7[9],
    half_btf(COSPI_INV[32], stg7[10], COSPI_INV[32], stg7[11], cos_bit),
    half_btf(COSPI_INV[32], stg7[10], -COSPI_INV[32], stg7[11], cos_bit),
    stg7[12],
    stg7[13],
    half_btf(COSPI_INV[32], stg7[14], COSPI_INV[32], stg7[15], cos_bit),
    half_btf(COSPI_INV[32], stg7[14], -COSPI_INV[32], stg7[15], cos_bit)
  ];

  // stage 9
  output[0] = stg8[0];
  output[1] = -stg8[8];
  output[2] = stg8[12];
  output[3] = -stg8[4];
  output[4] = stg8[6];
  output[5] = -stg8[14];
  output[6] = stg8[10];
  output[7] = -stg8[2];
  output[8] = stg8[3];
  output[9] = -stg8[11];
  output[10] = stg8[15];
  output[11] = -stg8[7];
  output[12] = stg8[5];
  output[13] = -stg8[13];
  output[14] = stg8[9];
  output[15] = -stg8[1];
}

fn av1_iidentity16(input: &[i32], output: &mut [i32], _range: usize) {
  for i in 0..16 {
    output[i] = round_shift(SQRT2 * 2 * input[i], 12);
  }
}

static INV_TXFM_FNS: [[fn(&[i32], &mut [i32], usize); 3]; 3] = [
  [av1_idct4, av1_idct8, av1_idct16],
  [av1_iadst4, av1_iadst8, av1_iadst16],
  [av1_iidentity4, av1_iidentity8, av1_iidentity16]
];

enum TxType1D {
  DCT,
  ADST,
  IDTX
}

// Option can be removed when the table is completely filled
fn get_1d_tx_types(tx_type: TxType) -> Option<(TxType1D, TxType1D)> {
  match tx_type {
    TxType::DCT_DCT => Some((TxType1D::DCT, TxType1D::DCT)),
    TxType::ADST_DCT => Some((TxType1D::ADST, TxType1D::DCT)),
    TxType::DCT_ADST => Some((TxType1D::DCT, TxType1D::ADST)),
    TxType::ADST_ADST => Some((TxType1D::ADST, TxType1D::ADST)),
    TxType::IDTX => Some((TxType1D::IDTX, TxType1D::IDTX)),
    TxType::V_DCT => Some((TxType1D::DCT, TxType1D::IDTX)),
    TxType::H_DCT => Some((TxType1D::IDTX, TxType1D::DCT)),
    TxType::V_ADST => Some((TxType1D::ADST, TxType1D::IDTX)),
    TxType::H_ADST => Some((TxType1D::IDTX, TxType1D::ADST)),
    _ => None
  }
}

trait InvTxfm2D: Dim {
  const INTERMEDIATE_SHIFT: usize;

  fn inv_txfm2d_add_rs(
    input: &[i32], output: &mut [u16], stride: usize, tx_type: TxType,
    bd: usize
  ) {
    let buffer = &mut [0i32; 64 * 64][..Self::W * Self::H];
    let tx_types_1d = get_1d_tx_types(tx_type)
      .expect("TxType not supported by rust txfm code.");
    // perform inv txfm on every row
    let range = bd + 8;
    let txfm_fn =
      INV_TXFM_FNS[tx_types_1d.1 as usize][Self::W.ilog() - 3];
    for (input_slice, buffer_slice) in
      input.chunks(Self::W).zip(buffer.chunks_mut(Self::W))
    {
      let mut temp_in: [i32; 64] = [0; 64];
      for (raw, clamped) in input_slice.iter().zip(temp_in.iter_mut()) {
        *clamped = clamp_value(*raw, range);
      }
      txfm_fn(&temp_in, buffer_slice, range);
    }

    // perform inv txfm on every col
    let range = cmp::max(bd + 6, 16);
    let txfm_fn =
      INV_TXFM_FNS[tx_types_1d.0 as usize][Self::H.ilog() - 3];
    for c in 0..Self::H {
      let mut temp_in: [i32; 64] = [0; 64];
      let mut temp_out: [i32; 64] = [0; 64];
      for (raw, clamped) in
        buffer[c..].iter().step_by(Self::W).zip(temp_in.iter_mut())
      {
        *clamped =
          clamp_value(round_shift(*raw, Self::INTERMEDIATE_SHIFT), range);
      }
      txfm_fn(&temp_in, &mut temp_out, range);
      for (temp, out) in temp_out
        .iter()
        .zip(output[c..].iter_mut().step_by(stride).take(Self::H))
      {
        *out =
          clamp(*out as i32 + round_shift(*temp, 4), 0, (1 << bd) - 1) as u16;
      }
    }
  }
}

impl InvTxfm2D for Block4x4 {
  const INTERMEDIATE_SHIFT: usize = 0;
}

impl InvTxfm2D for Block8x8 {
  const INTERMEDIATE_SHIFT: usize = 1;
}

impl InvTxfm2D for Block16x16 {
  const INTERMEDIATE_SHIFT: usize = 2;
}

// In libaom, functions that have more than one specialization use function
// pointers, so we need to declare them as static fields and call them
// indirectly. Otherwise, we call SSE or C variants directly. To fully
// understand what's going on here you should first understand the Perl code
// in libaom that generates these function bindings.

extern {
  static av1_fwd_txfm2d_4x4: extern fn(
    input: *const i16,
    output: *mut i32,
    stride: libc::c_int,
    tx_type: libc::c_int,
    bd: libc::c_int
  );
  fn av1_fwd_txfm2d_4x4_c(
    input: *const i16, output: *mut i32, stride: libc::c_int,
    tx_type: libc::c_int, bd: libc::c_int
  );
  static av1_fwd_txfm2d_8x8: extern fn(
    input: *const i16,
    output: *mut i32,
    stride: libc::c_int,
    tx_type: libc::c_int,
    bd: libc::c_int
  );
  fn av1_fwd_txfm2d_8x8_c(
    input: *const i16, output: *mut i32, stride: libc::c_int,
    tx_type: libc::c_int, bd: libc::c_int
  );
  static av1_fwd_txfm2d_16x16: extern fn(
    input: *const i16,
    output: *mut i32,
    stride: libc::c_int,
    tx_type: libc::c_int,
    bd: libc::c_int
  );
  fn av1_fwd_txfm2d_16x16_c(
    input: *const i16, output: *mut i32, stride: libc::c_int,
    tx_type: libc::c_int, bd: libc::c_int
  );
  static av1_fwd_txfm2d_32x32: extern fn(
    input: *const i16,
    output: *mut i32,
    stride: libc::c_int,
    tx_type: libc::c_int,
    bd: libc::c_int
  );
  fn av1_fwd_txfm2d_32x32_c(
    input: *const i16, output: *mut i32, stride: libc::c_int,
    tx_type: libc::c_int, bd: libc::c_int
  );
}

extern {
  static av1_inv_txfm2d_add_4x4: extern fn(
    input: *const i32,
    output: *mut u16,
    stride: libc::c_int,
    tx_type: libc::c_int,
    bd: libc::c_int
  );
  fn av1_inv_txfm2d_add_4x4_c(
    input: *const i32, output: *mut u16, stride: libc::c_int,
    tx_type: libc::c_int, bd: libc::c_int
  );
  static av1_inv_txfm2d_add_8x8: extern fn(
    input: *const i32,
    output: *mut u16,
    stride: libc::c_int,
    tx_type: libc::c_int,
    bd: libc::c_int
  ) -> ();
  fn av1_inv_txfm2d_add_8x8_c(
    input: *const i32, output: *mut u16, stride: libc::c_int,
    tx_type: libc::c_int, bd: libc::c_int
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
    input: *const i32, output: *mut u16, stride: libc::c_int,
    tx_type: libc::c_int, bd: libc::c_int
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
    TxSize::TX_16X16 =>
      iht16x16_add(input, output, stride, tx_type, bit_depth),
    TxSize::TX_32X32 =>
      iht32x32_add(input, output, stride, tx_type, bit_depth),
    _ => panic!("unimplemented tx size")
  }
}

fn fht4x4(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  // SIMD code may assert for transform types beyond TxType::IDTX.
  if tx_type < TxType::IDTX {
    unsafe {
      av1_fwd_txfm2d_4x4(
        input.as_ptr(),
        output.as_mut_ptr(),
        stride as libc::c_int,
        tx_type as libc::c_int,
        bit_depth as libc::c_int
      );
    }
  } else {
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
}

fn iht4x4_add(
  input: &[i32], output: &mut [u16], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  // SIMD code may assert for transform types beyond TxType::IDTX.
  if tx_type <= TxType::ADST_ADST
    || (tx_type >= TxType::IDTX && tx_type <= TxType::H_ADST)
  {
    Block4x4::inv_txfm2d_add_rs(input, output, stride, tx_type, bit_depth);
  } else if tx_type < TxType::IDTX {
    unsafe {
      av1_inv_txfm2d_add_4x4(
        input.as_ptr(),
        output.as_mut_ptr(),
        stride as libc::c_int,
        tx_type as libc::c_int,
        bit_depth as libc::c_int
      );
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

fn fht8x8(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  // SIMD code may assert for transform types beyond TxType::IDTX.
  if tx_type < TxType::IDTX {
    unsafe {
      av1_fwd_txfm2d_8x8(
        input.as_ptr(),
        output.as_mut_ptr(),
        stride as libc::c_int,
        tx_type as libc::c_int,
        bit_depth as libc::c_int
      );
    }
  } else {
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
}

fn iht8x8_add(
  input: &[i32], output: &mut [u16], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  // SIMD code may assert for transform types beyond TxType::IDTX.
  if tx_type <= TxType::ADST_ADST
    || (tx_type >= TxType::IDTX && tx_type <= TxType::H_ADST)
  {
    Block8x8::inv_txfm2d_add_rs(input, output, stride, tx_type, bit_depth);
  } else if tx_type < TxType::IDTX {
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
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  // SIMD code may assert for transform types beyond TxType::IDTX.
  if tx_type < TxType::IDTX {
    unsafe {
      av1_fwd_txfm2d_16x16(
        input.as_ptr(),
        output.as_mut_ptr(),
        stride as libc::c_int,
        tx_type as libc::c_int,
        bit_depth as libc::c_int
      );
    }
  } else {
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
}

fn iht16x16_add(
  input: &[i32], output: &mut [u16], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  unsafe {
    if tx_type <= TxType::ADST_ADST
      || (tx_type >= TxType::IDTX && tx_type <= TxType::H_ADST)
    {
      Block16x16::inv_txfm2d_add_rs(input, output, stride, tx_type, bit_depth);
    } else if tx_type < TxType::IDTX {
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
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  // SIMD code may assert for transform types that aren't TxType::DCT_DCT.
  if tx_type == TxType::DCT_DCT {
    unsafe {
      av1_fwd_txfm2d_32x32(
        input.as_ptr(),
        output.as_mut_ptr(),
        stride as libc::c_int,
        tx_type as libc::c_int,
        bit_depth as libc::c_int
      );
    }
  } else {
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
}

fn iht32x32_add(
  input: &[i32], output: &mut [u16], stride: usize, tx_type: TxType,
  bit_depth: usize
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
