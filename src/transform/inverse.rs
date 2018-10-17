use super::*;

use std::cmp;

use partition::TxType;

static COSPI_INV: [i32; 64] = [
  4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973, 3948,
  3920, 3889, 3857, 3822, 3784, 3745, 3703, 3659, 3612, 3564, 3513, 3461,
  3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967, 2896, 2824, 2751, 2675,
  2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019, 1931, 1842, 1751, 1660,
  1567, 1474, 1380, 1285, 1189, 1092, 995, 897, 799, 700, 601, 501, 401, 301,
  201, 101,
];

static SINPI_INV: [i32; 5] = [0, 1321, 2482, 3344, 3803];

const INV_COS_BIT: usize = 12;

fn av1_idct4(input: &[i32], output: &mut [i32], range: usize) {
  // stage 1
  let stg1 = [input[0], input[2], input[1], input[3]];

  // stage 2
  let stg2 = [
    half_btf(COSPI_INV[32], stg1[0], COSPI_INV[32], stg1[1], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg1[0], -COSPI_INV[32], stg1[1], INV_COS_BIT),
    half_btf(COSPI_INV[48], stg1[2], -COSPI_INV[16], stg1[3], INV_COS_BIT),
    half_btf(COSPI_INV[16], stg1[2], COSPI_INV[48], stg1[3], INV_COS_BIT)
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

fn av1_idct8(input: &[i32], output: &mut [i32], range: usize) {
  // call idct4
  let temp_in = [input[0], input[2], input[4], input[6]];
  let mut temp_out: [i32; 4] = [0; 4];
  av1_idct4(&temp_in, &mut temp_out, range);

  // stage 0

  // stage 1
  let stg1 = [input[1], input[5], input[3], input[7]];

  // stage 2
  let stg2 = [
    half_btf(COSPI_INV[56], stg1[0], -COSPI_INV[8], stg1[3], INV_COS_BIT),
    half_btf(COSPI_INV[24], stg1[1], -COSPI_INV[40], stg1[2], INV_COS_BIT),
    half_btf(COSPI_INV[40], stg1[1], COSPI_INV[24], stg1[2], INV_COS_BIT),
    half_btf(COSPI_INV[8], stg1[0], COSPI_INV[56], stg1[3], INV_COS_BIT)
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
    half_btf(-COSPI_INV[32], stg3[1], COSPI_INV[32], stg3[2], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg3[1], COSPI_INV[32], stg3[2], INV_COS_BIT),
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
  // stage 1
  let stg1 = [
    input[7], input[0], input[5], input[2], input[3], input[4], input[1],
    input[6],
  ];

  // stage 2
  let stg2 = [
    half_btf(COSPI_INV[4], stg1[0], COSPI_INV[60], stg1[1], INV_COS_BIT),
    half_btf(COSPI_INV[60], stg1[0], -COSPI_INV[4], stg1[1], INV_COS_BIT),
    half_btf(COSPI_INV[20], stg1[2], COSPI_INV[44], stg1[3], INV_COS_BIT),
    half_btf(COSPI_INV[44], stg1[2], -COSPI_INV[20], stg1[3], INV_COS_BIT),
    half_btf(COSPI_INV[36], stg1[4], COSPI_INV[28], stg1[5], INV_COS_BIT),
    half_btf(COSPI_INV[28], stg1[4], -COSPI_INV[36], stg1[5], INV_COS_BIT),
    half_btf(COSPI_INV[52], stg1[6], COSPI_INV[12], stg1[7], INV_COS_BIT),
    half_btf(COSPI_INV[12], stg1[6], -COSPI_INV[52], stg1[7], INV_COS_BIT)
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
    half_btf(COSPI_INV[16], stg3[4], COSPI_INV[48], stg3[5], INV_COS_BIT),
    half_btf(COSPI_INV[48], stg3[4], -COSPI_INV[16], stg3[5], INV_COS_BIT),
    half_btf(-COSPI_INV[48], stg3[6], COSPI_INV[16], stg3[7], INV_COS_BIT),
    half_btf(COSPI_INV[16], stg3[6], COSPI_INV[48], stg3[7], INV_COS_BIT)
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
    half_btf(COSPI_INV[32], stg5[2], COSPI_INV[32], stg5[3], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg5[2], -COSPI_INV[32], stg5[3], INV_COS_BIT),
    stg5[4],
    stg5[5],
    half_btf(COSPI_INV[32], stg5[6], COSPI_INV[32], stg5[7], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg5[6], -COSPI_INV[32], stg5[7], INV_COS_BIT)
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
  // call idct8
  let temp_in = [
    input[0], input[2], input[4], input[6], input[8], input[10], input[12],
    input[14],
  ];
  let mut temp_out: [i32; 8] = [0; 8];
  av1_idct8(&temp_in, &mut temp_out, range);

  // stage 1
  let stg1 = [
    input[1], input[9], input[5], input[13], input[3], input[11], input[7],
    input[15],
  ];

  // stage 2
  let stg2 = [
    half_btf(COSPI_INV[60], stg1[0], -COSPI_INV[4], stg1[7], INV_COS_BIT),
    half_btf(COSPI_INV[28], stg1[1], -COSPI_INV[36], stg1[6], INV_COS_BIT),
    half_btf(COSPI_INV[44], stg1[2], -COSPI_INV[20], stg1[5], INV_COS_BIT),
    half_btf(COSPI_INV[12], stg1[3], -COSPI_INV[52], stg1[4], INV_COS_BIT),
    half_btf(COSPI_INV[52], stg1[3], COSPI_INV[12], stg1[4], INV_COS_BIT),
    half_btf(COSPI_INV[20], stg1[2], COSPI_INV[44], stg1[5], INV_COS_BIT),
    half_btf(COSPI_INV[36], stg1[1], COSPI_INV[28], stg1[6], INV_COS_BIT),
    half_btf(COSPI_INV[4], stg1[0], COSPI_INV[60], stg1[7], INV_COS_BIT)
  ];

  // stage 3
  let stg3 = [
    clamp_value(stg2[0] + stg2[1], range),
    clamp_value(stg2[0] - stg2[1], range),
    clamp_value(-stg2[2] + stg2[3], range),
    clamp_value(stg2[2] + stg2[3], range),
    clamp_value(stg2[4] + stg2[5], range),
    clamp_value(stg2[4] - stg2[5], range),
    clamp_value(-stg2[6] + stg2[7], range),
    clamp_value(stg2[6] + stg2[7], range)
  ];

  // stage 4
  let stg4 = [
    stg3[0],
    half_btf(-COSPI_INV[16], stg3[1], COSPI_INV[48], stg3[6], INV_COS_BIT),
    half_btf(-COSPI_INV[48], stg3[2], -COSPI_INV[16], stg3[5], INV_COS_BIT),
    stg3[3],
    stg3[4],
    half_btf(-COSPI_INV[16], stg3[2], COSPI_INV[48], stg3[5], INV_COS_BIT),
    half_btf(COSPI_INV[48], stg3[1], COSPI_INV[16], stg3[6], INV_COS_BIT),
    stg3[7]
  ];

  // stage 5
  let stg5 = [
    clamp_value(stg4[0] + stg4[3], range),
    clamp_value(stg4[1] + stg4[2], range),
    clamp_value(stg4[1] - stg4[2], range),
    clamp_value(stg4[0] - stg4[3], range),
    clamp_value(-stg4[4] + stg4[7], range),
    clamp_value(-stg4[5] + stg4[6], range),
    clamp_value(stg4[5] + stg4[6], range),
    clamp_value(stg4[4] + stg4[7], range)
  ];

  // stage 6
  let stg6 = [
    stg5[0],
    stg5[1],
    half_btf(-COSPI_INV[32], stg5[2], COSPI_INV[32], stg5[5], INV_COS_BIT),
    half_btf(-COSPI_INV[32], stg5[3], COSPI_INV[32], stg5[4], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg5[3], COSPI_INV[32], stg5[4], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg5[2], COSPI_INV[32], stg5[5], INV_COS_BIT),
    stg5[6],
    stg5[7]
  ];

  // stage 7
  output[0] = clamp_value(temp_out[0] + stg6[7], range);
  output[1] = clamp_value(temp_out[1] + stg6[6], range);
  output[2] = clamp_value(temp_out[2] + stg6[5], range);
  output[3] = clamp_value(temp_out[3] + stg6[4], range);
  output[4] = clamp_value(temp_out[4] + stg6[3], range);
  output[5] = clamp_value(temp_out[5] + stg6[2], range);
  output[6] = clamp_value(temp_out[6] + stg6[1], range);
  output[7] = clamp_value(temp_out[7] + stg6[0], range);
  output[8] = clamp_value(temp_out[7] - stg6[0], range);
  output[9] = clamp_value(temp_out[6] - stg6[1], range);
  output[10] = clamp_value(temp_out[5] - stg6[2], range);
  output[11] = clamp_value(temp_out[4] - stg6[3], range);
  output[12] = clamp_value(temp_out[3] - stg6[4], range);
  output[13] = clamp_value(temp_out[2] - stg6[5], range);
  output[14] = clamp_value(temp_out[1] - stg6[6], range);
  output[15] = clamp_value(temp_out[0] - stg6[7], range);
}

fn av1_iadst16(input: &[i32], output: &mut [i32], range: usize) {
  // stage 1
  let stg1 = [
    input[15], input[0], input[13], input[2], input[11], input[4], input[9],
    input[6], input[7], input[8], input[5], input[10], input[3], input[12],
    input[1], input[14],
  ];

  // stage 2
  let stg2 = [
    half_btf(COSPI_INV[2], stg1[0], COSPI_INV[62], stg1[1], INV_COS_BIT),
    half_btf(COSPI_INV[62], stg1[0], -COSPI_INV[2], stg1[1], INV_COS_BIT),
    half_btf(COSPI_INV[10], stg1[2], COSPI_INV[54], stg1[3], INV_COS_BIT),
    half_btf(COSPI_INV[54], stg1[2], -COSPI_INV[10], stg1[3], INV_COS_BIT),
    half_btf(COSPI_INV[18], stg1[4], COSPI_INV[46], stg1[5], INV_COS_BIT),
    half_btf(COSPI_INV[46], stg1[4], -COSPI_INV[18], stg1[5], INV_COS_BIT),
    half_btf(COSPI_INV[26], stg1[6], COSPI_INV[38], stg1[7], INV_COS_BIT),
    half_btf(COSPI_INV[38], stg1[6], -COSPI_INV[26], stg1[7], INV_COS_BIT),
    half_btf(COSPI_INV[34], stg1[8], COSPI_INV[30], stg1[9], INV_COS_BIT),
    half_btf(COSPI_INV[30], stg1[8], -COSPI_INV[34], stg1[9], INV_COS_BIT),
    half_btf(COSPI_INV[42], stg1[10], COSPI_INV[22], stg1[11], INV_COS_BIT),
    half_btf(COSPI_INV[22], stg1[10], -COSPI_INV[42], stg1[11], INV_COS_BIT),
    half_btf(COSPI_INV[50], stg1[12], COSPI_INV[14], stg1[13], INV_COS_BIT),
    half_btf(COSPI_INV[14], stg1[12], -COSPI_INV[50], stg1[13], INV_COS_BIT),
    half_btf(COSPI_INV[58], stg1[14], COSPI_INV[6], stg1[15], INV_COS_BIT),
    half_btf(COSPI_INV[6], stg1[14], -COSPI_INV[58], stg1[15], INV_COS_BIT)
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
    half_btf(COSPI_INV[8], stg3[8], COSPI_INV[56], stg3[9], INV_COS_BIT),
    half_btf(COSPI_INV[56], stg3[8], -COSPI_INV[8], stg3[9], INV_COS_BIT),
    half_btf(COSPI_INV[40], stg3[10], COSPI_INV[24], stg3[11], INV_COS_BIT),
    half_btf(COSPI_INV[24], stg3[10], -COSPI_INV[40], stg3[11], INV_COS_BIT),
    half_btf(-COSPI_INV[56], stg3[12], COSPI_INV[8], stg3[13], INV_COS_BIT),
    half_btf(COSPI_INV[8], stg3[12], COSPI_INV[56], stg3[13], INV_COS_BIT),
    half_btf(-COSPI_INV[24], stg3[14], COSPI_INV[40], stg3[15], INV_COS_BIT),
    half_btf(COSPI_INV[40], stg3[14], COSPI_INV[24], stg3[15], INV_COS_BIT)
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
    half_btf(COSPI_INV[16], stg5[4], COSPI_INV[48], stg5[5], INV_COS_BIT),
    half_btf(COSPI_INV[48], stg5[4], -COSPI_INV[16], stg5[5], INV_COS_BIT),
    half_btf(-COSPI_INV[48], stg5[6], COSPI_INV[16], stg5[7], INV_COS_BIT),
    half_btf(COSPI_INV[16], stg5[6], COSPI_INV[48], stg5[7], INV_COS_BIT),
    stg5[8],
    stg5[9],
    stg5[10],
    stg5[11],
    half_btf(COSPI_INV[16], stg5[12], COSPI_INV[48], stg5[13], INV_COS_BIT),
    half_btf(COSPI_INV[48], stg5[12], -COSPI_INV[16], stg5[13], INV_COS_BIT),
    half_btf(-COSPI_INV[48], stg5[14], COSPI_INV[16], stg5[15], INV_COS_BIT),
    half_btf(COSPI_INV[16], stg5[14], COSPI_INV[48], stg5[15], INV_COS_BIT)
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
    half_btf(COSPI_INV[32], stg7[2], COSPI_INV[32], stg7[3], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg7[2], -COSPI_INV[32], stg7[3], INV_COS_BIT),
    stg7[4],
    stg7[5],
    half_btf(COSPI_INV[32], stg7[6], COSPI_INV[32], stg7[7], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg7[6], -COSPI_INV[32], stg7[7], INV_COS_BIT),
    stg7[8],
    stg7[9],
    half_btf(COSPI_INV[32], stg7[10], COSPI_INV[32], stg7[11], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg7[10], -COSPI_INV[32], stg7[11], INV_COS_BIT),
    stg7[12],
    stg7[13],
    half_btf(COSPI_INV[32], stg7[14], COSPI_INV[32], stg7[15], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg7[14], -COSPI_INV[32], stg7[15], INV_COS_BIT)
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

fn av1_idct32(input: &[i32], output: &mut [i32], range: usize) {
  // stage 1;
  let stg1 = [
    input[0], input[16], input[8], input[24], input[4], input[20], input[12],
    input[28], input[2], input[18], input[10], input[26], input[6], input[22],
    input[14], input[30], input[1], input[17], input[9], input[25], input[5],
    input[21], input[13], input[29], input[3], input[19], input[11],
    input[27], input[7], input[23], input[15], input[31],
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
    stg1[8],
    stg1[9],
    stg1[10],
    stg1[11],
    stg1[12],
    stg1[13],
    stg1[14],
    stg1[15],
    half_btf(COSPI_INV[62], stg1[16], -COSPI_INV[2], stg1[31], INV_COS_BIT),
    half_btf(COSPI_INV[30], stg1[17], -COSPI_INV[34], stg1[30], INV_COS_BIT),
    half_btf(COSPI_INV[46], stg1[18], -COSPI_INV[18], stg1[29], INV_COS_BIT),
    half_btf(COSPI_INV[14], stg1[19], -COSPI_INV[50], stg1[28], INV_COS_BIT),
    half_btf(COSPI_INV[54], stg1[20], -COSPI_INV[10], stg1[27], INV_COS_BIT),
    half_btf(COSPI_INV[22], stg1[21], -COSPI_INV[42], stg1[26], INV_COS_BIT),
    half_btf(COSPI_INV[38], stg1[22], -COSPI_INV[26], stg1[25], INV_COS_BIT),
    half_btf(COSPI_INV[6], stg1[23], -COSPI_INV[58], stg1[24], INV_COS_BIT),
    half_btf(COSPI_INV[58], stg1[23], COSPI_INV[6], stg1[24], INV_COS_BIT),
    half_btf(COSPI_INV[26], stg1[22], COSPI_INV[38], stg1[25], INV_COS_BIT),
    half_btf(COSPI_INV[42], stg1[21], COSPI_INV[22], stg1[26], INV_COS_BIT),
    half_btf(COSPI_INV[10], stg1[20], COSPI_INV[54], stg1[27], INV_COS_BIT),
    half_btf(COSPI_INV[50], stg1[19], COSPI_INV[14], stg1[28], INV_COS_BIT),
    half_btf(COSPI_INV[18], stg1[18], COSPI_INV[46], stg1[29], INV_COS_BIT),
    half_btf(COSPI_INV[34], stg1[17], COSPI_INV[30], stg1[30], INV_COS_BIT),
    half_btf(COSPI_INV[2], stg1[16], COSPI_INV[62], stg1[31], INV_COS_BIT)
  ];

  // stage 3
  let stg3 = [
    stg2[0],
    stg2[1],
    stg2[2],
    stg2[3],
    stg2[4],
    stg2[5],
    stg2[6],
    stg2[7],
    half_btf(COSPI_INV[60], stg2[8], -COSPI_INV[4], stg2[15], INV_COS_BIT),
    half_btf(COSPI_INV[28], stg2[9], -COSPI_INV[36], stg2[14], INV_COS_BIT),
    half_btf(COSPI_INV[44], stg2[10], -COSPI_INV[20], stg2[13], INV_COS_BIT),
    half_btf(COSPI_INV[12], stg2[11], -COSPI_INV[52], stg2[12], INV_COS_BIT),
    half_btf(COSPI_INV[52], stg2[11], COSPI_INV[12], stg2[12], INV_COS_BIT),
    half_btf(COSPI_INV[20], stg2[10], COSPI_INV[44], stg2[13], INV_COS_BIT),
    half_btf(COSPI_INV[36], stg2[9], COSPI_INV[28], stg2[14], INV_COS_BIT),
    half_btf(COSPI_INV[4], stg2[8], COSPI_INV[60], stg2[15], INV_COS_BIT),
    clamp_value(stg2[16] + stg2[17], range),
    clamp_value(stg2[16] - stg2[17], range),
    clamp_value(-stg2[18] + stg2[19], range),
    clamp_value(stg2[18] + stg2[19], range),
    clamp_value(stg2[20] + stg2[21], range),
    clamp_value(stg2[20] - stg2[21], range),
    clamp_value(-stg2[22] + stg2[23], range),
    clamp_value(stg2[22] + stg2[23], range),
    clamp_value(stg2[24] + stg2[25], range),
    clamp_value(stg2[24] - stg2[25], range),
    clamp_value(-stg2[26] + stg2[27], range),
    clamp_value(stg2[26] + stg2[27], range),
    clamp_value(stg2[28] + stg2[29], range),
    clamp_value(stg2[28] - stg2[29], range),
    clamp_value(-stg2[30] + stg2[31], range),
    clamp_value(stg2[30] + stg2[31], range)
  ];

  // stage 4
  let stg4 = [
    stg3[0],
    stg3[1],
    stg3[2],
    stg3[3],
    half_btf(COSPI_INV[56], stg3[4], -COSPI_INV[8], stg3[7], INV_COS_BIT),
    half_btf(COSPI_INV[24], stg3[5], -COSPI_INV[40], stg3[6], INV_COS_BIT),
    half_btf(COSPI_INV[40], stg3[5], COSPI_INV[24], stg3[6], INV_COS_BIT),
    half_btf(COSPI_INV[8], stg3[4], COSPI_INV[56], stg3[7], INV_COS_BIT),
    clamp_value(stg3[8] + stg3[9], range),
    clamp_value(stg3[8] - stg3[9], range),
    clamp_value(-stg3[10] + stg3[11], range),
    clamp_value(stg3[10] + stg3[11], range),
    clamp_value(stg3[12] + stg3[13], range),
    clamp_value(stg3[12] - stg3[13], range),
    clamp_value(-stg3[14] + stg3[15], range),
    clamp_value(stg3[14] + stg3[15], range),
    stg3[16],
    half_btf(-COSPI_INV[8], stg3[17], COSPI_INV[56], stg3[30], INV_COS_BIT),
    half_btf(-COSPI_INV[56], stg3[18], -COSPI_INV[8], stg3[29], INV_COS_BIT),
    stg3[19],
    stg3[20],
    half_btf(-COSPI_INV[40], stg3[21], COSPI_INV[24], stg3[26], INV_COS_BIT),
    half_btf(-COSPI_INV[24], stg3[22], -COSPI_INV[40], stg3[25], INV_COS_BIT),
    stg3[23],
    stg3[24],
    half_btf(-COSPI_INV[40], stg3[22], COSPI_INV[24], stg3[25], INV_COS_BIT),
    half_btf(COSPI_INV[24], stg3[21], COSPI_INV[40], stg3[26], INV_COS_BIT),
    stg3[27],
    stg3[28],
    half_btf(-COSPI_INV[8], stg3[18], COSPI_INV[56], stg3[29], INV_COS_BIT),
    half_btf(COSPI_INV[56], stg3[17], COSPI_INV[8], stg3[30], INV_COS_BIT),
    stg3[31]
  ];

  // stage 5
  let stg5 = [
    half_btf(COSPI_INV[32], stg4[0], COSPI_INV[32], stg4[1], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg4[0], -COSPI_INV[32], stg4[1], INV_COS_BIT),
    half_btf(COSPI_INV[48], stg4[2], -COSPI_INV[16], stg4[3], INV_COS_BIT),
    half_btf(COSPI_INV[16], stg4[2], COSPI_INV[48], stg4[3], INV_COS_BIT),
    clamp_value(stg4[4] + stg4[5], range),
    clamp_value(stg4[4] - stg4[5], range),
    clamp_value(-stg4[6] + stg4[7], range),
    clamp_value(stg4[6] + stg4[7], range),
    stg4[8],
    half_btf(-COSPI_INV[16], stg4[9], COSPI_INV[48], stg4[14], INV_COS_BIT),
    half_btf(-COSPI_INV[48], stg4[10], -COSPI_INV[16], stg4[13], INV_COS_BIT),
    stg4[11],
    stg4[12],
    half_btf(-COSPI_INV[16], stg4[10], COSPI_INV[48], stg4[13], INV_COS_BIT),
    half_btf(COSPI_INV[48], stg4[9], COSPI_INV[16], stg4[14], INV_COS_BIT),
    stg4[15],
    clamp_value(stg4[16] + stg4[19], range),
    clamp_value(stg4[17] + stg4[18], range),
    clamp_value(stg4[17] - stg4[18], range),
    clamp_value(stg4[16] - stg4[19], range),
    clamp_value(-stg4[20] + stg4[23], range),
    clamp_value(-stg4[21] + stg4[22], range),
    clamp_value(stg4[21] + stg4[22], range),
    clamp_value(stg4[20] + stg4[23], range),
    clamp_value(stg4[24] + stg4[27], range),
    clamp_value(stg4[25] + stg4[26], range),
    clamp_value(stg4[25] - stg4[26], range),
    clamp_value(stg4[24] - stg4[27], range),
    clamp_value(-stg4[28] + stg4[31], range),
    clamp_value(-stg4[29] + stg4[30], range),
    clamp_value(stg4[29] + stg4[30], range),
    clamp_value(stg4[28] + stg4[31], range)
  ];

  // stage 6
  let stg6 = [
    clamp_value(stg5[0] + stg5[3], range),
    clamp_value(stg5[1] + stg5[2], range),
    clamp_value(stg5[1] - stg5[2], range),
    clamp_value(stg5[0] - stg5[3], range),
    stg5[4],
    half_btf(-COSPI_INV[32], stg5[5], COSPI_INV[32], stg5[6], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg5[5], COSPI_INV[32], stg5[6], INV_COS_BIT),
    stg5[7],
    clamp_value(stg5[8] + stg5[11], range),
    clamp_value(stg5[9] + stg5[10], range),
    clamp_value(stg5[9] - stg5[10], range),
    clamp_value(stg5[8] - stg5[11], range),
    clamp_value(-stg5[12] + stg5[15], range),
    clamp_value(-stg5[13] + stg5[14], range),
    clamp_value(stg5[13] + stg5[14], range),
    clamp_value(stg5[12] + stg5[15], range),
    stg5[16],
    stg5[17],
    half_btf(-COSPI_INV[16], stg5[18], COSPI_INV[48], stg5[29], INV_COS_BIT),
    half_btf(-COSPI_INV[16], stg5[19], COSPI_INV[48], stg5[28], INV_COS_BIT),
    half_btf(-COSPI_INV[48], stg5[20], -COSPI_INV[16], stg5[27], INV_COS_BIT),
    half_btf(-COSPI_INV[48], stg5[21], -COSPI_INV[16], stg5[26], INV_COS_BIT),
    stg5[22],
    stg5[23],
    stg5[24],
    stg5[25],
    half_btf(-COSPI_INV[16], stg5[21], COSPI_INV[48], stg5[26], INV_COS_BIT),
    half_btf(-COSPI_INV[16], stg5[20], COSPI_INV[48], stg5[27], INV_COS_BIT),
    half_btf(COSPI_INV[48], stg5[19], COSPI_INV[16], stg5[28], INV_COS_BIT),
    half_btf(COSPI_INV[48], stg5[18], COSPI_INV[16], stg5[29], INV_COS_BIT),
    stg5[30],
    stg5[31]
  ];

  // stage 7
  let stg7 = [
    clamp_value(stg6[0] + stg6[7], range),
    clamp_value(stg6[1] + stg6[6], range),
    clamp_value(stg6[2] + stg6[5], range),
    clamp_value(stg6[3] + stg6[4], range),
    clamp_value(stg6[3] - stg6[4], range),
    clamp_value(stg6[2] - stg6[5], range),
    clamp_value(stg6[1] - stg6[6], range),
    clamp_value(stg6[0] - stg6[7], range),
    stg6[8],
    stg6[9],
    half_btf(-COSPI_INV[32], stg6[10], COSPI_INV[32], stg6[13], INV_COS_BIT),
    half_btf(-COSPI_INV[32], stg6[11], COSPI_INV[32], stg6[12], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg6[11], COSPI_INV[32], stg6[12], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg6[10], COSPI_INV[32], stg6[13], INV_COS_BIT),
    stg6[14],
    stg6[15],
    clamp_value(stg6[16] + stg6[23], range),
    clamp_value(stg6[17] + stg6[22], range),
    clamp_value(stg6[18] + stg6[21], range),
    clamp_value(stg6[19] + stg6[20], range),
    clamp_value(stg6[19] - stg6[20], range),
    clamp_value(stg6[18] - stg6[21], range),
    clamp_value(stg6[17] - stg6[22], range),
    clamp_value(stg6[16] - stg6[23], range),
    clamp_value(-stg6[24] + stg6[31], range),
    clamp_value(-stg6[25] + stg6[30], range),
    clamp_value(-stg6[26] + stg6[29], range),
    clamp_value(-stg6[27] + stg6[28], range),
    clamp_value(stg6[27] + stg6[28], range),
    clamp_value(stg6[26] + stg6[29], range),
    clamp_value(stg6[25] + stg6[30], range),
    clamp_value(stg6[24] + stg6[31], range)
  ];

  // stage 8
  let stg8 = [
    clamp_value(stg7[0] + stg7[15], range),
    clamp_value(stg7[1] + stg7[14], range),
    clamp_value(stg7[2] + stg7[13], range),
    clamp_value(stg7[3] + stg7[12], range),
    clamp_value(stg7[4] + stg7[11], range),
    clamp_value(stg7[5] + stg7[10], range),
    clamp_value(stg7[6] + stg7[9], range),
    clamp_value(stg7[7] + stg7[8], range),
    clamp_value(stg7[7] - stg7[8], range),
    clamp_value(stg7[6] - stg7[9], range),
    clamp_value(stg7[5] - stg7[10], range),
    clamp_value(stg7[4] - stg7[11], range),
    clamp_value(stg7[3] - stg7[12], range),
    clamp_value(stg7[2] - stg7[13], range),
    clamp_value(stg7[1] - stg7[14], range),
    clamp_value(stg7[0] - stg7[15], range),
    stg7[16],
    stg7[17],
    stg7[18],
    stg7[19],
    half_btf(-COSPI_INV[32], stg7[20], COSPI_INV[32], stg7[27], INV_COS_BIT),
    half_btf(-COSPI_INV[32], stg7[21], COSPI_INV[32], stg7[26], INV_COS_BIT),
    half_btf(-COSPI_INV[32], stg7[22], COSPI_INV[32], stg7[25], INV_COS_BIT),
    half_btf(-COSPI_INV[32], stg7[23], COSPI_INV[32], stg7[24], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg7[23], COSPI_INV[32], stg7[24], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg7[22], COSPI_INV[32], stg7[25], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg7[21], COSPI_INV[32], stg7[26], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg7[20], COSPI_INV[32], stg7[27], INV_COS_BIT),
    stg7[28],
    stg7[29],
    stg7[30],
    stg7[31]
  ];

  // stage 9
  output[0] = clamp_value(stg8[0] + stg8[31], range);
  output[1] = clamp_value(stg8[1] + stg8[30], range);
  output[2] = clamp_value(stg8[2] + stg8[29], range);
  output[3] = clamp_value(stg8[3] + stg8[28], range);
  output[4] = clamp_value(stg8[4] + stg8[27], range);
  output[5] = clamp_value(stg8[5] + stg8[26], range);
  output[6] = clamp_value(stg8[6] + stg8[25], range);
  output[7] = clamp_value(stg8[7] + stg8[24], range);
  output[8] = clamp_value(stg8[8] + stg8[23], range);
  output[9] = clamp_value(stg8[9] + stg8[22], range);
  output[10] = clamp_value(stg8[10] + stg8[21], range);
  output[11] = clamp_value(stg8[11] + stg8[20], range);
  output[12] = clamp_value(stg8[12] + stg8[19], range);
  output[13] = clamp_value(stg8[13] + stg8[18], range);
  output[14] = clamp_value(stg8[14] + stg8[17], range);
  output[15] = clamp_value(stg8[15] + stg8[16], range);
  output[16] = clamp_value(stg8[15] - stg8[16], range);
  output[17] = clamp_value(stg8[14] - stg8[17], range);
  output[18] = clamp_value(stg8[13] - stg8[18], range);
  output[19] = clamp_value(stg8[12] - stg8[19], range);
  output[20] = clamp_value(stg8[11] - stg8[20], range);
  output[21] = clamp_value(stg8[10] - stg8[21], range);
  output[22] = clamp_value(stg8[9] - stg8[22], range);
  output[23] = clamp_value(stg8[8] - stg8[23], range);
  output[24] = clamp_value(stg8[7] - stg8[24], range);
  output[25] = clamp_value(stg8[6] - stg8[25], range);
  output[26] = clamp_value(stg8[5] - stg8[26], range);
  output[27] = clamp_value(stg8[4] - stg8[27], range);
  output[28] = clamp_value(stg8[3] - stg8[28], range);
  output[29] = clamp_value(stg8[2] - stg8[29], range);
  output[30] = clamp_value(stg8[1] - stg8[30], range);
  output[31] = clamp_value(stg8[0] - stg8[31], range);
}

fn av1_iidentity32(input: &[i32], output: &mut [i32], _range: usize) {
  for i in 0..32 {
    output[i] = input[i] * 4;
  }
}

static INV_TXFM_FNS: [[fn(&[i32], &mut [i32], usize); 4]; 4] = [
  [av1_idct4, av1_idct8, av1_idct16, av1_idct32],
  [av1_iadst4, av1_iadst8, av1_iadst16, |_, _, _| unimplemented!()],
  [
    |_, _, _| unimplemented!(),
    |_, _, _| unimplemented!(),
    |_, _, _| unimplemented!(),
    |_, _, _| unimplemented!()
  ],
  [av1_iidentity4, av1_iidentity8, av1_iidentity16, av1_iidentity32]
];

trait InvTxfm2D: Dim {
  const INTERMEDIATE_SHIFT: usize;

  fn inv_txfm2d_add(
    input: &[i32], output: &mut [u16], stride: usize, tx_type: TxType,
    bd: usize
  ) {
    // TODO: Implement SSE version
    Self::inv_txfm2d_add_rs(input, output, stride, tx_type, bd);
  }

  fn inv_txfm2d_add_rs(
    input: &[i32], output: &mut [u16], stride: usize, tx_type: TxType,
    bd: usize
  ) {
    let buffer = &mut [0i32; 64 * 64][..Self::W * Self::H];
    let tx_types_1d = get_1d_tx_types(tx_type)
      .expect("TxType not supported by rust txfm code.");
    // perform inv txfm on every row
    let range = bd + 8;
    let txfm_fn = INV_TXFM_FNS[tx_types_1d.1 as usize][Self::W.ilog() - 3];
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
    let txfm_fn = INV_TXFM_FNS[tx_types_1d.0 as usize][Self::H.ilog() - 3];
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

impl InvTxfm2D for Block32x32 {
  const INTERMEDIATE_SHIFT: usize = 2;
}

pub fn iht4x4_add(
  input: &[i32], output: &mut [u16], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  // SIMD code may assert for transform types beyond TxType::IDTX.
  if tx_type < TxType::IDTX {
    Block4x4::inv_txfm2d_add(input, output, stride, tx_type, bit_depth);
  } else {
    Block4x4::inv_txfm2d_add_rs(input, output, stride, tx_type, bit_depth);
  }
}

pub fn iht8x8_add(
  input: &[i32], output: &mut [u16], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  if tx_type < TxType::IDTX {
    Block8x8::inv_txfm2d_add(input, output, stride, tx_type, bit_depth);
  } else {
    Block8x8::inv_txfm2d_add_rs(input, output, stride, tx_type, bit_depth);
  }
}

pub fn iht16x16_add(
  input: &[i32], output: &mut [u16], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  if tx_type < TxType::IDTX {
    // SSE C code asserts for transform types beyond TxType::IDTX.
    Block16x16::inv_txfm2d_add(input, output, stride, tx_type, bit_depth);
  } else {
    Block16x16::inv_txfm2d_add_rs(input, output, stride, tx_type, bit_depth);
  }
}

pub fn iht32x32_add(
  input: &[i32], output: &mut [u16], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  if tx_type < TxType::IDTX {
    // SIMD code may assert for transform types beyond TxType::IDTX.
    Block32x32::inv_txfm2d_add(input, output, stride, tx_type, bit_depth);
  } else {
    Block32x32::inv_txfm2d_add_rs(input, output, stride, tx_type, bit_depth);
  }
}
