// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use super::*;
use partition::{TxSize, TxType};

type TxfmShift = [i8; 3];
type TxfmShifts = [TxfmShift; 3];

// Shift so that the first shift is 4 - (bd - 8) to align with the initial
// design of daala_tx
// 8 bit 4x4 is an exception and only shifts by 3 in the first stage
const FWD_SHIFT_4X4: TxfmShifts = [[3, 0, 0], [2, 0, 1], [0, 0, 3]];
const FWD_SHIFT_8X8: TxfmShifts = [[4, -1, 0], [2, 0, 1], [0, 0, 3]];
const FWD_SHIFT_16X16: TxfmShifts = [[4, -1, 0], [2, 0, 1], [0, 0, 3]];
const FWD_SHIFT_32X32: TxfmShifts = [[4, -2, 0], [2, 0, 0], [0, 0, 2]];
const FWD_SHIFT_64X64: TxfmShifts = [[4, -1, -2], [2, 0, -1], [0, 0, 1]];
const FWD_SHIFT_4X8: TxfmShifts = [[4, -1, 0], [2, 0, 1], [0, 0, 3]];
const FWD_SHIFT_8X4: TxfmShifts = [[4, -1, 0], [2, 0, 1], [0, 0, 3]];
const FWD_SHIFT_8X16: TxfmShifts = [[4, -1, 0], [2, 0, 1], [0, 0, 3]];
const FWD_SHIFT_16X8: TxfmShifts = [[4, -1, 0], [2, 0, 1], [0, 0, 3]];
const FWD_SHIFT_16X32: TxfmShifts = [[4, -2, 0], [2, 0, 0], [0, 0, 2]];
const FWD_SHIFT_32X16: TxfmShifts = [[4, -2, 0], [2, 0, 0], [0, 0, 2]];
const FWD_SHIFT_32X64: TxfmShifts = [[4, -1, -2], [2, 0, -1], [0, 0, 1]];
const FWD_SHIFT_64X32: TxfmShifts = [[4, -1, -2], [2, 0, -1], [0, 0, 1]];
const FWD_SHIFT_4X16: TxfmShifts = [[4, -1, 0], [2, 0, 1], [0, 0, 3]];
const FWD_SHIFT_16X4: TxfmShifts = [[4, -1, 0], [2, 0, 1], [0, 0, 3]];
const FWD_SHIFT_8X32: TxfmShifts = [[4, -1, 0], [2, 0, 1], [0, 0, 3]];
const FWD_SHIFT_32X8: TxfmShifts = [[4, -1, 0], [2, 0, 1], [0, 0, 3]];
const FWD_SHIFT_16X64: TxfmShifts = [[4, -2, 0], [2, 0, 0], [0, 0, 2]];
const FWD_SHIFT_64X16: TxfmShifts = [[4, -2, 0], [2, 0, 0], [0, 0, 2]];

const FWD_TXFM_SHIFT_LS: [TxfmShifts; TxSize::TX_SIZES_ALL] = [
  FWD_SHIFT_4X4,
  FWD_SHIFT_8X8,
  FWD_SHIFT_16X16,
  FWD_SHIFT_32X32,
  FWD_SHIFT_64X64,
  FWD_SHIFT_4X8,
  FWD_SHIFT_8X4,
  FWD_SHIFT_8X16,
  FWD_SHIFT_16X8,
  FWD_SHIFT_16X32,
  FWD_SHIFT_32X16,
  FWD_SHIFT_32X64,
  FWD_SHIFT_64X32,
  FWD_SHIFT_4X16,
  FWD_SHIFT_16X4,
  FWD_SHIFT_8X32,
  FWD_SHIFT_32X8,
  FWD_SHIFT_16X64,
  FWD_SHIFT_64X16
];

type TxfmFunc = Fn(&[i32], &mut [i32]);

use std::ops::*;

fn tx_mul(a: i32, mul: (i32, i32)) -> i32 {
  ((a * mul.0) + (1 << mul.1 >> 1)) >> mul.1
}

fn copy_fn(a: i32) -> i32 {
  a
}

fn rshift1(a: i32) -> i32 {
  (a + if a < 0 { 1 } else { 0 }) >> 1
}

fn add_avg(a: i32, b: i32) -> i32 {
  (a + b) >> 1
}

fn sub_avg(a: i32, b: i32) -> i32 {
  (a - b) >> 1
}

trait RotateKernelPi4 {
  const ADD: fn(i32, i32) -> i32;
  const SUB: fn(i32, i32) -> i32;

  fn kernel(p0: i32, p1: i32, m: ((i32, i32), (i32, i32))) -> (i32, i32) {
    let t = Self::ADD(p1, p0);
    let (a, out0) = (tx_mul(p0, m.0), tx_mul(t, m.1));
    let out1 = Self::SUB(a, out0);
    (out0, out1)
  }
}

struct RotatePi4Add;
struct RotatePi4AddAvg;
struct RotatePi4Sub;
struct RotatePi4SubAvg;

impl RotateKernelPi4 for RotatePi4Add {
  const ADD: fn(i32, i32) -> i32 = Add::add;
  const SUB: fn(i32, i32) -> i32 = Sub::sub;
}

impl RotateKernelPi4 for RotatePi4AddAvg {
  const ADD: fn(i32, i32) -> i32 = add_avg;
  const SUB: fn(i32, i32) -> i32 = Sub::sub;
}

impl RotateKernelPi4 for RotatePi4Sub {
  const ADD: fn(i32, i32) -> i32 = Sub::sub;
  const SUB: fn(i32, i32) -> i32 = Add::add;
}

impl RotateKernelPi4 for RotatePi4SubAvg {
  const ADD: fn(i32, i32) -> i32 = sub_avg;
  const SUB: fn(i32, i32) -> i32 = Add::add;
}

trait RotateKernel {
  const ADD: fn(i32, i32) -> i32;
  const SUB: fn(i32, i32) -> i32;
  const SHIFT: fn(i32) -> i32;

  fn half_kernel(
    p0: (i32, i32), p1: i32, m: ((i32, i32), (i32, i32), (i32, i32))
  ) -> (i32, i32) {
    let t = Self::ADD(p1, p0.0);
    let (a, b, c) = (tx_mul(p0.1, m.0), tx_mul(p1, m.1), tx_mul(t, m.2));
    let out0 = b + c;
    let shifted = Self::SHIFT(c);
    let out1 = Self::SUB(a, shifted);
    (out0, out1)
  }

  fn kernel(
    p0: i32, p1: i32, m: ((i32, i32), (i32, i32), (i32, i32))
  ) -> (i32, i32) {
    Self::half_kernel((p0, p0), p1, m)
  }
}

trait RotateKernelNeg {
  const ADD: fn(i32, i32) -> i32;
  fn kernel(
    p0: i32, p1: i32, m: ((i32, i32), (i32, i32), (i32, i32))
  ) -> (i32, i32) {
    let t = Self::ADD(p0, p1);
    let (a, b, c) = (tx_mul(p0, m.0), tx_mul(p1, m.1), tx_mul(t, m.2));
    let out0 = b - c;
    let out1 = c - a;
    (out0, out1)
  }
}

struct RotateAdd;
struct RotateAddAvg;
struct RotateAddShift;
struct RotateSub;
struct RotateSubAvg;
struct RotateSubShift;
struct RotateNeg;
struct RotateNegAvg;

impl RotateKernel for RotateAdd {
  const ADD: fn(i32, i32) -> i32 = Add::add;
  const SUB: fn(i32, i32) -> i32 = Sub::sub;
  const SHIFT: fn(i32) -> i32 = copy_fn;
}

impl RotateKernel for RotateAddAvg {
  const ADD: fn(i32, i32) -> i32 = add_avg;
  const SUB: fn(i32, i32) -> i32 = Sub::sub;
  const SHIFT: fn(i32) -> i32 = copy_fn;
}

impl RotateKernel for RotateAddShift {
  const ADD: fn(i32, i32) -> i32 = Add::add;
  const SUB: fn(i32, i32) -> i32 = Sub::sub;
  const SHIFT: fn(i32) -> i32 = rshift1;
}

impl RotateKernel for RotateSub {
  const ADD: fn(i32, i32) -> i32 = Sub::sub;
  const SUB: fn(i32, i32) -> i32 = Add::add;
  const SHIFT: fn(i32) -> i32 = copy_fn;
}

impl RotateKernel for RotateSubAvg {
  const ADD: fn(i32, i32) -> i32 = sub_avg;
  const SUB: fn(i32, i32) -> i32 = Add::add;
  const SHIFT: fn(i32) -> i32 = copy_fn;
}

impl RotateKernel for RotateSubShift {
  const ADD: fn(i32, i32) -> i32 = Sub::sub;
  const SUB: fn(i32, i32) -> i32 = Add::add;
  const SHIFT: fn(i32) -> i32 = rshift1;
}

impl RotateKernelNeg for RotateNeg {
  const ADD: fn(i32, i32) -> i32 = Sub::sub;
}

impl RotateKernelNeg for RotateNegAvg {
  const ADD: fn(i32, i32) -> i32 = sub_avg;
}

fn butterfly_add(p0: i32, p1: i32) -> ((i32, i32), i32) {
  let p0 = p0 + p1;
  let p0h = rshift1(p0);
  let p1h = p1 - p0h;
  ((p0h, p0), p1h)
}

fn butterfly_sub(p0: i32, p1: i32) -> ((i32, i32), i32) {
  let p0 = p0 - p1;
  let p0h = rshift1(p0);
  let p1h = p1 + p0h;
  ((p0h, p0), p1h)
}

fn butterfly_neg(p0: i32, p1: i32) -> (i32, (i32, i32)) {
  let p1 = p0 - p1;
  let p1h = rshift1(p1);
  let p0h = p0 - p1h;
  (p0h, (p1h, p1))
}

fn butterfly_add_asym(p0: (i32, i32), p1h: i32) -> (i32, i32) {
  let p1 = p1h + p0.0;
  let p0 = p0.1 - p1;
  (p0, p1)
}

fn butterfly_sub_asym(p0: (i32, i32), p1h: i32) -> (i32, i32) {
  let p1 = p1h - p0.0;
  let p0 = p0.1 + p1;
  (p0, p1)
}

fn butterfly_neg_asym(p0h: i32, p1: (i32, i32)) -> (i32, i32) {
  let p0 = p0h + p1.0;
  let p1 = p0 - p1.1;
  (p0, p1)
}

macro_rules! store_coeffs {
  ( $arr:expr, $( $x:expr ),* ) => {
      {
      let mut i: i32 = -1;
      $(
        i += 1;
        $arr[i as usize] = $x;
      )*
    }
  };
}

fn daala_fdct_ii_2_asym(p0h: i32, p1: (i32, i32)) -> (i32, i32) {
  butterfly_neg_asym(p0h, p1)
}

fn daala_fdst_iv_2_asym(p0: (i32, i32), p1h: (i32)) -> (i32, i32) {
  //   473/512 = (Sin[3*Pi/8] + Cos[3*Pi/8])/Sqrt[2] = 0.9238795325112867
  // 3135/4096 = (Sin[3*Pi/8] - Cos[3*Pi/8])*Sqrt[2] = 0.7653668647301795
  // 4433/8192 = Cos[3*Pi/8]*Sqrt[2]                 = 0.5411961001461971
  RotateAdd::half_kernel(p0, p1h, ((473, 9), (3135, 12), (4433, 13)))
}

fn daala_fdct_ii_4(q0: i32, q1: i32, q2: i32, q3: i32, output: &mut [i32]) {
  // +/- Butterflies with asymmetric output.
  let (q0h, q3) = butterfly_neg(q0, q3);
  let (q1, q2h) = butterfly_add(q1, q2);

  // Embedded 2-point transforms with asymmetric input.
  let (q0, q1) = daala_fdct_ii_2_asym(q0h, q1);
  let (q3, q2) = daala_fdst_iv_2_asym(q3, q2h);

  store_coeffs!(output, q0, q1, q2, q3);
}

fn daala_fdct4(input: &[i32], output: &mut [i32]) {
  let mut temp_out: [i32; 4] = [0; 4];
  daala_fdct_ii_4(input[0], input[1], input[2], input[3], &mut temp_out);

  output[0] = temp_out[0];
  output[1] = temp_out[2];
  output[2] = temp_out[1];
  output[3] = temp_out[3];
}

fn daala_fdst_vii_4(input: &[i32], output: &mut [i32]) {
  let q0 = input[0];
  let q1 = input[1];
  let q2 = input[2];
  let q3 = input[3];
  let t0 = q1 + q3;
  // t1 = (q0 + q1 - q3)/2
  let t1 = q1 + sub_avg(q0, t0);
  let t2 = q0 - q1;
  let t3 = q2;
  let t4 = q0 + q3;
  // 7021/16384 ~= 2*Sin[2*Pi/9]/3 ~= 0.428525073124360
  let t0 = tx_mul(t0, (7021, 14));
  // 37837/32768 ~= 4*Sin[3*Pi/9]/3 ~= 1.154700538379252
  let t1 = tx_mul(t1, (37837, 15));
  // 21513/32768 ~= 2*Sin[4*Pi/9]/3 ~= 0.656538502008139
  let t2 = tx_mul(t2, (21513, 15));
  // 37837/32768 ~= 4*Sin[3*Pi/9]/3 ~= 1.154700538379252
  let t3 = tx_mul(t3, (37837, 15));
  // 467/2048 ~= 2*Sin[1*Pi/9]/3 ~= 0.228013428883779
  let t4 = tx_mul(t4, (467, 11));
  let t3h = rshift1(t3);
  let u4 = t4 + t3h;
  output[0] = t0 + u4;
  output[1] = t1;
  output[2] = t0 + (t2 - t3h);
  output[3] = t2 + (t3 - u4);
}

fn daala_fdct_ii_2(p0: i32, p1: i32) -> (i32, i32) {
  // 11585/8192 = Sin[Pi/4] + Cos[Pi/4]  = 1.4142135623730951
  // 11585/8192 = 2*Cos[Pi/4]            = 1.4142135623730951
  let (p1, p0) = RotatePi4SubAvg::kernel(p1, p0, ((11585, 13), (11585, 13)));
  (p0, p1)
}

fn daala_fdst_iv_2(p0: i32, p1: i32) -> (i32, i32) {
  // 10703/8192 = Sin[3*Pi/8] + Cos[3*Pi/8]  = 1.3065629648763766
  // 8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8]  = 0.5411961001461971
  //  3135/4096 = 2*Cos[3*Pi/8]              = 0.7653668647301796
  RotateAddAvg::kernel(p0, p1, ((10703, 13), (8867, 14), (3135, 12)))
}

fn daala_fdct_ii_4_asym(
  q0h: i32, q1: (i32, i32), q2h: i32, q3: (i32, i32), output: &mut [i32]
) {
  // +/- Butterflies with asymmetric input.
  let (q0, q3) = butterfly_neg_asym(q0h, q3);
  let (q1, q2) = butterfly_sub_asym(q1, q2h);

  // Embedded 2-point orthonormal transforms.
  let (q0, q1) = daala_fdct_ii_2(q0, q1);
  let (q3, q2) = daala_fdst_iv_2(q3, q2);

  store_coeffs!(output, q0, q1, q2, q3);
}

fn daala_fdst_iv_4_asym(
  q0: (i32, i32), q1h: i32, q2: (i32, i32), q3h: i32, output: &mut [i32]
) {
  // Stage 0
  //  9633/16384 = (Sin[7*Pi/16] + Cos[7*Pi/16])/2 = 0.5879378012096793
  //  12873/8192 = (Sin[7*Pi/16] - Cos[7*Pi/16])*2 = 1.5713899167742045
  // 12785/32768 = Cos[7*Pi/16]*2                  = 0.3901806440322565
  let (q0, q3) = RotateAddShift::half_kernel(
    q0,
    q3h,
    ((9633, 14), (12873, 13), (12785, 15))
  );
  // 11363/16384 = (Sin[5*Pi/16] + Cos[5*Pi/16])/2 = 0.6935199226610738
  // 18081/32768 = (Sin[5*Pi/16] - Cos[5*Pi/16])*2 = 0.5517987585658861
  //  4551/4096 = Cos[5*Pi/16]*2                  = 1.1111404660392044
  let (q2, q1) = RotateSubShift::half_kernel(
    q2,
    q1h,
    ((11363, 14), (18081, 15), (4551, 12))
  );

  // Stage 1
  let (q2, q3) = butterfly_sub_asym((rshift1(q2), q2), q3);
  let (q0, q1) = butterfly_sub_asym((rshift1(q0), q0), q1);

  // Stage 2
  // 11585/8192 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951
  // 11585/8192 = 2*Cos[Pi/4]           = 1.4142135623730951
  let (q2, q1) = RotatePi4AddAvg::kernel(q2, q1, ((11585, 13), (11585, 13)));

  store_coeffs!(output, q0, q1, q2, q3);
}

fn daala_fdct_ii_8(
  r0: i32, r1: i32, r2: i32, r3: i32, r4: i32, r5: i32, r6: i32, r7: i32,
  output: &mut [i32]
) {
  // +/- Butterflies with asymmetric output.
  let (r0h, r7) = butterfly_neg(r0, r7);
  let (r1, r6h) = butterfly_add(r1, r6);
  let (r2h, r5) = butterfly_neg(r2, r5);
  let (r3, r4h) = butterfly_add(r3, r4);

  // Embedded 4-point transforms with asymmetric input.
  daala_fdct_ii_4_asym(r0h, r1, r2h, r3, &mut output[0..4]);
  daala_fdst_iv_4_asym(r7, r6h, r5, r4h, &mut output[4..8]);
  output[4..8].reverse();
}

fn daala_fdct8(input: &[i32], output: &mut [i32]) {
  let mut temp_out: [i32; 8] = [0; 8];
  daala_fdct_ii_8(
    input[0],
    input[1],
    input[2],
    input[3],
    input[4],
    input[5],
    input[6],
    input[7],
    &mut temp_out
  );

  output[0] = temp_out[0];
  output[1] = temp_out[4];
  output[2] = temp_out[2];
  output[3] = temp_out[6];
  output[4] = temp_out[1];
  output[5] = temp_out[5];
  output[6] = temp_out[3];
  output[7] = temp_out[7];
}

fn daala_fdst_iv_8(
  r0: i32, r1: i32, r2: i32, r3: i32, r4: i32, r5: i32, r6: i32, r7: i32,
  output: &mut [i32]
) {
  // Stage 0
  // 17911/16384 = Sin[15*Pi/32] + Cos[15*Pi/32] = 1.0932018670017576
  // 14699/16384 = Sin[15*Pi/32] - Cos[15*Pi/32] = 0.8971675863426363
  //    803/8192 = Cos[15*Pi/32]                 = 0.0980171403295606
  let (r0, r7) =
    RotateAdd::kernel(r0, r7, ((17911, 14), (14699, 14), (803, 13)));
  // 20435/16384 = Sin[13*Pi/32] + Cos[13*Pi/32] = 1.24722501298667123
  // 21845/32768 = Sin[13*Pi/32] - Cos[13*Pi/32] = 0.66665565847774650
  //   1189/4096 = Cos[13*Pi/32]                 = 0.29028467725446233
  let (r6, r1) =
    RotateSub::kernel(r6, r1, ((20435, 14), (21845, 15), (1189, 12)));
  // 22173/16384 = Sin[11*Pi/32] + Cos[11*Pi/32] = 1.3533180011743526
  //   3363/8192 = Sin[11*Pi/32] - Cos[11*Pi/32] = 0.4105245275223574
  // 15447/32768 = Cos[11*Pi/32]                 = 0.47139673682599764
  let (r2, r5) =
    RotateAdd::kernel(r2, r5, ((22173, 14), (3363, 13), (15447, 15)));
  // 23059/16384 = Sin[9*Pi/32] + Cos[9*Pi/32] = 1.4074037375263826
  //  2271/16384 = Sin[9*Pi/32] - Cos[9*Pi/32] = 0.1386171691990915
  //   5197/8192 = Cos[9*Pi/32]                = 0.6343932841636455
  let (r4, r3) =
    RotateSub::kernel(r4, r3, ((23059, 14), (2271, 14), (5197, 13)));

  // Stage 1
  let (r0, r3h) = butterfly_add(r0, r3);
  let (r2, r1h) = butterfly_sub(r2, r1);
  let (r5, r6h) = butterfly_add(r5, r6);
  let (r7, r4h) = butterfly_sub(r7, r4);

  // Stage 2
  let (r7, r6) = butterfly_add_asym(r7, r6h);
  let (r5, r3) = butterfly_add_asym(r5, r3h);
  let (r2, r4) = butterfly_add_asym(r2, r4h);
  let (r0, r1) = butterfly_sub_asym(r0, r1h);

  // Stage 3
  // 10703/8192 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766
  // 8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969
  //  3135/4096 = 2*Cos[3*Pi/8]             = 0.7653668647301796
  let (r3, r4) =
    RotateSubAvg::kernel(r3, r4, ((10703, 13), (8867, 14), (3135, 12)));
  // 10703/8192 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766
  // 8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969
  //  3135/4096 = 2*Cos[3*Pi/8]             = 0.7653668647301796
  let (r2, r5) =
    RotateNegAvg::kernel(r2, r5, ((10703, 13), (8867, 14), (3135, 12)));
  // 11585/8192 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951
  // 11585/8192 = 2*Cos[Pi/4]           = 1.4142135623730951
  let (r1, r6) = RotatePi4SubAvg::kernel(r1, r6, ((11585, 13), (11585, 13)));

  store_coeffs!(output, r0, r1, r2, r3, r4, r5, r6, r7);
}

fn daala_fdst8(input: &[i32], output: &mut [i32]) {
  let mut temp_out: [i32; 8] = [0; 8];
  daala_fdst_iv_8(
    input[0],
    input[1],
    input[2],
    input[3],
    input[4],
    input[5],
    input[6],
    input[7],
    &mut temp_out
  );

  output[0] = temp_out[0];
  output[1] = temp_out[4];
  output[2] = temp_out[2];
  output[3] = temp_out[6];
  output[4] = temp_out[1];
  output[5] = temp_out[5];
  output[6] = temp_out[3];
  output[7] = temp_out[7];
}

fn daala_fdst_iv_4(q0: i32, q1: i32, q2: i32, q3: i32, output: &mut [i32]) {
  // Stage 0
  // 13623/16384 = (Sin[7*Pi/16] + Cos[7*Pi/16])/Sqrt[2] = 0.831469612302545
  //   4551/4096 = (Sin[7*Pi/16] - Cos[7*Pi/16])*Sqrt[2] = 1.111140466039204
  //  9041/32768 = Cos[7*Pi/16]*Sqrt[2]                  = 0.275899379282943
  let (q0, q3) =
    RotateAddShift::kernel(q0, q3, ((13623, 14), (4551, 12), (565, 11)));
  // 16069/16384 = (Sin[5*Pi/16] + Cos[5*Pi/16])/Sqrt[2] = 0.9807852804032304
  // 12785/32768 = (Sin[5*Pi/16] - Cos[5*Pi/16])*Sqrt[2] = 0.3901806440322566
  //   1609/2048 = Cos[5*Pi/16]*Sqrt[2]                  = 0.7856949583871021
  let (q2, q1) =
    RotateSubShift::kernel(q2, q1, ((16069, 14), (12785, 15), (1609, 11)));

  // Stage 1
  let (q2, q3) = butterfly_sub_asym((rshift1(q2), q2), q3);
  let (q0, q1) = butterfly_sub_asym((rshift1(q0), q0), q1);

  // Stage 2
  // 11585/8192 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951
  // 11585/8192 = 2*Cos[Pi/4]           = 1.4142135623730951
  let (q2, q1) = RotatePi4AddAvg::kernel(q2, q1, ((11585, 13), (11585, 13)));

  store_coeffs!(output, q0, q1, q2, q3);
}

fn daala_fdct_ii_8_asym(
  r0h: i32, r1: (i32, i32), r2h: i32, r3: (i32, i32), r4h: i32,
  r5: (i32, i32), r6h: i32, r7: (i32, i32), output: &mut [i32]
) {
  // +/- Butterflies with asymmetric input.
  let (r0, r7) = butterfly_neg_asym(r0h, r7);
  let (r1, r6) = butterfly_sub_asym(r1, r6h);
  let (r2, r5) = butterfly_neg_asym(r2h, r5);
  let (r3, r4) = butterfly_sub_asym(r3, r4h);

  // Embedded 4-point orthonormal transforms.
  daala_fdct_ii_4(r0, r1, r2, r3, &mut output[0..4]);
  daala_fdst_iv_4(r7, r6, r5, r4, &mut output[4..8]);
  output[4..8].reverse();
}

fn daala_fdst_iv_8_asym(
  r0: (i32, i32), r1h: i32, r2: (i32, i32), r3h: i32, r4: (i32, i32),
  r5h: i32, r6: (i32, i32), r7h: i32, output: &mut [i32]
) {
  // Stage 0
  // 12665/16384 = (Sin[15*Pi/32] + Cos[15*Pi/32])/Sqrt[2] = 0.77301045336274
  //   5197/4096 = (Sin[15*Pi/32] - Cos[15*Pi/32])*Sqrt[2] = 1.26878656832729
  //  2271/16384 = Cos[15*Pi/32]*Sqrt[2]                   = 0.13861716919909
  let (r0, r7) =
    RotateAdd::half_kernel(r0, r7h, ((12665, 14), (5197, 12), (2271, 14)));
  // 14449/16384 = Sin[13*Pi/32] + Cos[13*Pi/32])/Sqrt[2] = 0.881921264348355
  // 30893/32768 = Sin[13*Pi/32] - Cos[13*Pi/32])*Sqrt[2] = 0.942793473651995
  //   3363/8192 = Cos[13*Pi/32]*Sqrt[2]                  = 0.410524527522357
  let (r6, r1) =
    RotateSub::half_kernel(r6, r1h, ((14449, 14), (30893, 15), (3363, 13)));
  // 15679/16384 = Sin[11*Pi/32] + Cos[11*Pi/32])/Sqrt[2] = 0.956940335732209
  //   1189/2048 = Sin[11*Pi/32] - Cos[11*Pi/32])*Sqrt[2] = 0.580569354508925
  //   5461/8192 = Cos[11*Pi/32]*Sqrt[2]                  = 0.666655658477747
  let (r2, r5) =
    RotateAdd::half_kernel(r2, r5h, ((15679, 14), (1189, 11), (5461, 13)));
  // 16305/16384 = (Sin[9*Pi/32] + Cos[9*Pi/32])/Sqrt[2] = 0.9951847266721969
  //    803/4096 = (Sin[9*Pi/32] - Cos[9*Pi/32])*Sqrt[2] = 0.1960342806591213
  // 14699/16384 = Cos[9*Pi/32]*Sqrt[2]                  = 0.8971675863426364
  let (r4, r3) =
    RotateSub::half_kernel(r4, r3h, ((16305, 14), (803, 12), (14699, 14)));

  // Stage 1
  let (r0, r3h) = butterfly_add(r0, r3);
  let (r2, r1h) = butterfly_sub(r2, r1);
  let (r5, r6h) = butterfly_add(r5, r6);
  let (r7, r4h) = butterfly_sub(r7, r4);

  // Stage 2
  let (r7, r6) = butterfly_add_asym(r7, r6h);
  let (r5, r3) = butterfly_add_asym(r5, r3h);
  let (r2, r4) = butterfly_add_asym(r2, r4h);
  let (r0, r1) = butterfly_sub_asym(r0, r1h);

  // Stage 3
  // 10703/8192 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766
  // 8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969
  //  3135/4096 = 2*Cos[3*Pi/8]             = 0.7653668647301796
  let (r3, r4) =
    RotateSubAvg::kernel(r3, r4, ((669, 9), (8867, 14), (3135, 12)));
  // 10703/8192 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766
  // 8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969
  //  3135/4096 = 2*Cos[3*Pi/8]             = 0.7653668647301796
  let (r2, r5) =
    RotateNegAvg::kernel(r2, r5, ((669, 9), (8867, 14), (3135, 12)));
  // 11585/8192 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951
  // 11585/8192 = 2*Cos[Pi/4]           = 1.4142135623730951
  let (r1, r6) = RotatePi4SubAvg::kernel(r1, r6, ((5793, 12), (11585, 13)));

  store_coeffs!(output, r0, r1, r2, r3, r4, r5, r6, r7);
}

fn daala_fdct_ii_16(
  s0: i32, s1: i32, s2: i32, s3: i32, s4: i32, s5: i32, s6: i32, s7: i32,
  s8: i32, s9: i32, sa: i32, sb: i32, sc: i32, sd: i32, se: i32, sf: i32,
  output: &mut [i32]
) {
  // +/- Butterflies with asymmetric output.
  let (s0h, sf) = butterfly_neg(s0, sf);
  let (s1, seh) = butterfly_add(s1, se);
  let (s2h, sd) = butterfly_neg(s2, sd);
  let (s3, sch) = butterfly_add(s3, sc);
  let (s4h, sb) = butterfly_neg(s4, sb);
  let (s5, sah) = butterfly_add(s5, sa);
  let (s6h, s9) = butterfly_neg(s6, s9);
  let (s7, s8h) = butterfly_add(s7, s8);

  // Embedded 8-point transforms with asymmetric input.
  daala_fdct_ii_8_asym(s0h, s1, s2h, s3, s4h, s5, s6h, s7, &mut output[0..8]);
  daala_fdst_iv_8_asym(sf, seh, sd, sch, sb, sah, s9, s8h, &mut output[8..16]);
  output[8..16].reverse();
}

fn daala_fdct16(input: &[i32], output: &mut [i32]) {
  let mut temp_out: [i32; 16] = [0; 16];
  daala_fdct_ii_16(
    input[0],
    input[1],
    input[2],
    input[3],
    input[4],
    input[5],
    input[6],
    input[7],
    input[8],
    input[9],
    input[10],
    input[11],
    input[12],
    input[13],
    input[14],
    input[15],
    &mut temp_out
  );

  output[0] = temp_out[0];
  output[1] = temp_out[8];
  output[2] = temp_out[4];
  output[3] = temp_out[12];
  output[4] = temp_out[2];
  output[5] = temp_out[10];
  output[6] = temp_out[6];
  output[7] = temp_out[14];
  output[8] = temp_out[1];
  output[9] = temp_out[9];
  output[10] = temp_out[5];
  output[11] = temp_out[13];
  output[12] = temp_out[3];
  output[13] = temp_out[11];
  output[14] = temp_out[7];
  output[15] = temp_out[15];
}

fn daala_fdst_iv_16(
  s0: i32, s1: i32, s2: i32, s3: i32, s4: i32, s5: i32, s6: i32, s7: i32,
  s8: i32, s9: i32, sa: i32, sb: i32, sc: i32, sd: i32, se: i32, sf: i32,
  output: &mut [i32]
) {
  // Stage 0
  // 24279/32768 = (Sin[31*Pi/64] + Cos[31*Pi/64])/Sqrt[2] = 0.74095112535496
  //  11003/8192 = (Sin[31*Pi/64] - Cos[31*Pi/64])*Sqrt[2] = 1.34311790969404
  //  1137/16384 = Cos[31*Pi/64]*Sqrt[2]                   = 0.06939217050794
  let (s0, sf) =
    RotateAddShift::kernel(s0, sf, ((24279, 15), (11003, 13), (1137, 14)));
  // 1645/2048 = (Sin[29*Pi/64] + Cos[29*Pi/64])/Sqrt[2] = 0.8032075314806449
  //   305/256 = (Sin[29*Pi/64] - Cos[29*Pi/64])*Sqrt[2] = 1.1913986089848667
  //  425/2048 = Cos[29*Pi/64]*Sqrt[2]                   = 0.2075082269882116
  let (se, s1) =
    RotateSubShift::kernel(se, s1, ((1645, 11), (305, 8), (425, 11)));
  // 14053/32768 = (Sin[27*Pi/64] + Cos[27*Pi/64])/Sqrt[2] = 0.85772861000027
  //   8423/8192 = (Sin[27*Pi/64] - Cos[27*Pi/64])*Sqrt[2] = 1.02820548838644
  //   2815/8192 = Cos[27*Pi/64]*Sqrt[2]                   = 0.34362586580705
  let (s2, sd) =
    RotateAddShift::kernel(s2, sd, ((14053, 14), (8423, 13), (2815, 13)));
  // 14811/16384 = (Sin[25*Pi/64] + Cos[25*Pi/64])/Sqrt[2] = 0.90398929312344
  //   7005/8192 = (Sin[25*Pi/64] - Cos[25*Pi/64])*Sqrt[2] = 0.85511018686056
  //   3903/8192 = Cos[25*Pi/64]*Sqrt[2]                   = 0.47643419969316
  let (sc, s3) =
    RotateSubShift::kernel(sc, s3, ((14811, 14), (7005, 13), (3903, 13)));
  // 30853/32768 = (Sin[23*Pi/64] + Cos[23*Pi/64])/Sqrt[2] = 0.94154406518302
  // 11039/16384 = (Sin[23*Pi/64] - Cos[23*Pi/64])*Sqrt[2] = 0.67377970678444
  //  9907/16384 = Cos[23*Pi/64]*Sqrt[2]                   = 0.60465421179080
  let (s4, sb) =
    RotateAddShift::kernel(s4, sb, ((30853, 15), (11039, 14), (9907, 14)));
  // 15893/16384 = (Sin[21*Pi/64] + Cos[21*Pi/64])/Sqrt[2] = 0.97003125319454
  //   3981/8192 = (Sin[21*Pi/64] - Cos[21*Pi/64])*Sqrt[2] = 0.89716758634264
  //   1489/2048 = Cos[21*Pi/64]*Sqrt[2]                   = 0.72705107329128
  let (sa, s5) =
    RotateSubShift::kernel(sa, s5, ((15893, 14), (3981, 13), (1489, 11)));
  // 32413/32768 = (Sin[19*Pi/64] + Cos[19*Pi/64])/Sqrt[2] = 0.98917650996478
  //    601/2048 = (Sin[19*Pi/64] - Cos[19*Pi/64])*Sqrt[2] = 0.29346094891072
  // 13803/16384 = Cos[19*Pi/64]*Sqrt[2]                   = 0.84244603550942
  let (s6, s9) =
    RotateAddShift::kernel(s6, s9, ((32413, 15), (601, 11), (13803, 14)));
  // 32729/32768 = (Sin[17*Pi/64] + Cos[17*Pi/64])/Sqrt[2] = 0.99879545620517
  //    201/2048 = (Sin[17*Pi/64] - Cos[17*Pi/64])*Sqrt[2] = 0.09813534865484
  //   1945/2048 = Cos[17*Pi/64]*Sqrt[2]                   = 0.94972778187775
  let (s8, s7) =
    RotateSubShift::kernel(s8, s7, ((32729, 15), (201, 11), (1945, 11)));

  // Stage 1
  let (s0, s7) = butterfly_sub_asym((rshift1(s0), s0), s7);
  let (s8, sf) = butterfly_sub_asym((rshift1(s8), s8), sf);
  let (s4, s3) = butterfly_add_asym((rshift1(s4), s4), s3);
  let (sc, sb) = butterfly_add_asym((rshift1(sc), sc), sb);
  let (s2, s5) = butterfly_sub_asym((rshift1(s2), s2), s5);
  let (sa, sd) = butterfly_sub_asym((rshift1(sa), sa), sd);
  let (s6, s1) = butterfly_add_asym((rshift1(s6), s6), s1);
  let (se, s9) = butterfly_add_asym((rshift1(se), se), s9);

  // Stage 2
  let ((_s8h, s8), s4h) = butterfly_add(s8, s4);
  let ((_s7h, s7), sbh) = butterfly_add(s7, sb);
  let ((_sah, sa), s6h) = butterfly_sub(sa, s6);
  let ((_s5h, s5), s9h) = butterfly_sub(s5, s9);
  let (s0, s3h) = butterfly_add(s0, s3);
  let (sd, seh) = butterfly_add(sd, se);
  let (s2, s1h) = butterfly_sub(s2, s1);
  let (sf, sch) = butterfly_sub(sf, sc);

  // Stage 3
  //     301/256 = Sin[7*Pi/16] + Cos[7*Pi/16] = 1.1758756024193586
  //   1609/2048 = Sin[7*Pi/16] - Cos[7*Pi/16] = 0.7856949583871022
  // 12785/32768 = 2*Cos[7*Pi/16]              = 0.3901806440322565
  let (s8, s7) =
    RotateAddAvg::kernel(s8, s7, ((301, 8), (1609, 11), (12785, 15)));
  // 11363/8192 = Sin[5*Pi/16] + Cos[5*Pi/16] = 1.3870398453221475
  // 9041/32768 = Sin[5*Pi/16] - Cos[5*Pi/16] = 0.2758993792829431
  //  4551/8192 = Cos[5*Pi/16]                = 0.5555702330196022
  let (s9, s6) =
    RotateAdd::kernel(s9h, s6h, ((11363, 13), (9041, 15), (4551, 13)));
  //  5681/4096 = Sin[5*Pi/16] + Cos[5*Pi/16] = 1.3870398453221475
  // 9041/32768 = Sin[5*Pi/16] - Cos[5*Pi/16] = 0.2758993792829431
  //  4551/4096 = 2*Cos[5*Pi/16]              = 1.1111404660392044
  let (s5, sa) =
    RotateNegAvg::kernel(s5, sa, ((5681, 12), (9041, 15), (4551, 12)));
  //   9633/8192 = Sin[7*Pi/16] + Cos[7*Pi/16] = 1.1758756024193586
  // 12873/16384 = Sin[7*Pi/16] - Cos[7*Pi/16] = 0.7856949583871022
  //  6393/32768 = Cos[7*Pi/16]                = 0.1950903220161283
  let (s4, sb) =
    RotateNeg::kernel(s4h, sbh, ((9633, 13), (12873, 14), (6393, 15)));

  // Stage 4
  let (s2, sc) = butterfly_add_asym(s2, sch);
  let (s0, s1) = butterfly_sub_asym(s0, s1h);
  let (sf, se) = butterfly_add_asym(sf, seh);
  let (sd, s3) = butterfly_add_asym(sd, s3h);
  let (s7, s6) = butterfly_add_asym((rshift1(s7), s7), s6);
  let (s8, s9) = butterfly_sub_asym((rshift1(s8), s8), s9);
  let (sa, sb) = butterfly_sub_asym((rshift1(sa), sa), sb);
  let (s5, s4) = butterfly_add_asym((rshift1(s5), s5), s4);

  // Stage 5
  //    669/512 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766
  // 8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969
  //  3135/4096 = 2*Cos[7*Pi/8]             = 0.7653668647301796
  let (sc, s3) =
    RotateAddAvg::kernel(sc, s3, ((669, 9), (8867, 14), (3135, 12)));
  //    669/512 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3870398453221475
  // 8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969
  //  3135/4096 = 2*Cos[3*Pi/8]             = 0.7653668647301796
  let (s2, sd) =
    RotateNegAvg::kernel(s2, sd, ((669, 9), (8867, 14), (3135, 12)));
  //  5793/4096 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951
  // 11585/8192 = 2*Cos[Pi/4]           = 1.4142135623730951
  let (sa, s5) = RotatePi4AddAvg::kernel(sa, s5, ((5793, 12), (11585, 13)));
  //  5793/4096 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951
  // 11585/8192 = 2*Cos[Pi/4]           = 1.4142135623730951
  let (s6, s9) = RotatePi4AddAvg::kernel(s6, s9, ((5793, 12), (11585, 13)));
  //  5793/4096 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951
  // 11585/8192 = 2*Cos[Pi/4]           = 1.4142135623730951
  let (se, s1) = RotatePi4AddAvg::kernel(se, s1, ((5793, 12), (11585, 13)));

  store_coeffs!(
    output, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sa, sb, sc, sd, se, sf
  );
}

fn daala_fdst16(input: &[i32], output: &mut [i32]) {
  let mut temp_out: [i32; 16] = [0; 16];
  daala_fdst_iv_16(
    input[0],
    input[1],
    input[2],
    input[3],
    input[4],
    input[5],
    input[6],
    input[7],
    input[8],
    input[9],
    input[10],
    input[11],
    input[12],
    input[13],
    input[14],
    input[15],
    &mut temp_out
  );

  output[0] = temp_out[0];
  output[1] = temp_out[8];
  output[2] = temp_out[4];
  output[3] = temp_out[12];
  output[4] = temp_out[2];
  output[5] = temp_out[10];
  output[6] = temp_out[6];
  output[7] = temp_out[14];
  output[8] = temp_out[1];
  output[9] = temp_out[9];
  output[10] = temp_out[5];
  output[11] = temp_out[13];
  output[12] = temp_out[3];
  output[13] = temp_out[11];
  output[14] = temp_out[7];
  output[15] = temp_out[15];
}

fn daala_fdct_ii_16_asym(
  s0h: i32, s1: (i32, i32), s2h: i32, s3: (i32, i32), s4h: i32,
  s5: (i32, i32), s6h: i32, s7: (i32, i32), s8h: i32, s9: (i32, i32),
  sah: i32, sb: (i32, i32), sch: i32, sd: (i32, i32), seh: i32,
  sf: (i32, i32), output: &mut [i32]
) {
  // +/- Butterflies with asymmetric input.
  let (s0, sf) = butterfly_neg_asym(s0h, sf);
  let (s1, se) = butterfly_sub_asym(s1, seh);
  let (s2, sd) = butterfly_neg_asym(s2h, sd);
  let (s3, sc) = butterfly_sub_asym(s3, sch);
  let (s4, sb) = butterfly_neg_asym(s4h, sb);
  let (s5, sa) = butterfly_sub_asym(s5, sah);
  let (s6, s9) = butterfly_neg_asym(s6h, s9);
  let (s7, s8) = butterfly_sub_asym(s7, s8h);

  // Embedded 8-point orthonormal transforms.
  daala_fdct_ii_8(s0, s1, s2, s3, s4, s5, s6, s7, &mut output[0..8]);
  daala_fdst_iv_8(sf, se, sd, sc, sb, sa, s9, s8, &mut output[8..16]);
  output[8..16].reverse();
}

fn daala_fdst_iv_16_asym(
  s0: (i32, i32), s1h: i32, s2: (i32, i32), s3h: i32, s4: (i32, i32),
  s5h: i32, s6: (i32, i32), s7h: i32, s8: (i32, i32), s9h: i32,
  sa: (i32, i32), sbh: i32, sc: (i32, i32), sdh: i32, se: (i32, i32),
  sfh: i32, output: &mut [i32]
) {
  // Stage 0
  //   1073/2048 = (Sin[31*Pi/64] + Cos[31*Pi/64])/2 = 0.5239315652662953
  // 62241/32768 = (Sin[31*Pi/64] - Cos[31*Pi/64])*2 = 1.8994555637555088
  //   201/16384 = Cos[31*Pi/64]*2                   = 0.0981353486548360
  let (s0, sf) =
    RotateAddShift::half_kernel(s0, sfh, ((1073, 11), (62241, 15), (201, 11)));
  // 18611/32768 = (Sin[29*Pi/64] + Cos[29*Pi/64])/2 = 0.5679534922100714
  // 55211/32768 = (Sin[29*Pi/64] - Cos[29*Pi/64])*2 = 1.6848920710188384
  //    601/2048 = Cos[29*Pi/64]*2                   = 0.2934609489107235
  let (se, s1) = RotateSubShift::half_kernel(
    se,
    s1h,
    ((18611, 15), (55211, 15), (601, 11))
  );
  //  9937/16384 = (Sin[27*Pi/64] + Cos[27*Pi/64])/2 = 0.6065057165489039
  //   1489/1024 = (Sin[27*Pi/64] - Cos[27*Pi/64])*2 = 1.4541021465825602
  //   3981/8192 = Cos[27*Pi/64]*2                   = 0.4859603598065277
  let (s2, sd) =
    RotateAddShift::half_kernel(s2, sdh, ((9937, 14), (1489, 10), (3981, 13)));
  // 10473/16384 = (Sin[25*Pi/64] + Cos[25*Pi/64])/2 = 0.6392169592876205
  // 39627/32768 = (Sin[25*Pi/64] - Cos[25*Pi/64])*2 = 1.2093084235816014
  // 11039/16384 = Cos[25*Pi/64]*2                   = 0.6737797067844401
  let (sc, s3) = RotateSubShift::half_kernel(
    sc,
    s3h,
    ((10473, 14), (39627, 15), (11039, 14))
  );
  // 2727/4096 = (Sin[23*Pi/64] + Cos[23*Pi/64])/2 = 0.6657721932768628
  // 3903/4096 = (Sin[23*Pi/64] - Cos[23*Pi/64])*2 = 0.9528683993863225
  // 7005/8192 = Cos[23*Pi/64]*2                   = 0.8551101868605642
  let (s4, sb) =
    RotateAddShift::half_kernel(s4, sbh, ((2727, 12), (3903, 12), (7005, 13)));
  // 5619/8192 = (Sin[21*Pi/64] + Cos[21*Pi/64])/2 = 0.6859156770967569
  // 2815/4096 = (Sin[21*Pi/64] - Cos[21*Pi/64])*2 = 0.6872517316141069
  // 8423/8192 = Cos[21*Pi/64]*2                   = 1.0282054883864433
  let (sa, s5) =
    RotateSubShift::half_kernel(sa, s5h, ((5619, 13), (2815, 12), (8423, 13)));
  //   2865/4096 = (Sin[19*Pi/64] + Cos[19*Pi/64])/2 = 0.6994534179865391
  // 13588/32768 = (Sin[19*Pi/64] - Cos[19*Pi/64])*2 = 0.4150164539764232
  //     305/256 = Cos[19*Pi/64]*2                   = 1.1913986089848667
  let (s6, s9) =
    RotateAddShift::half_kernel(s6, s9h, ((2865, 12), (13599, 15), (305, 8)));
  // 23143/32768 = (Sin[17*Pi/64] + Cos[17*Pi/64])/2 = 0.7062550401009887
  //   1137/8192 = (Sin[17*Pi/64] - Cos[17*Pi/64])*2 = 0.1387843410158816
  //  11003/8192 = Cos[17*Pi/64]*2                   = 1.3431179096940367
  let (s8, s7) = RotateSubShift::half_kernel(
    s8,
    s7h,
    ((23143, 15), (1137, 13), (11003, 13))
  );

  // Stage 1
  let (s0, s7) = butterfly_sub_asym((rshift1(s0), s0), s7);
  let (s8, sf) = butterfly_sub_asym((rshift1(s8), s8), sf);
  let (s4, s3) = butterfly_add_asym((rshift1(s4), s4), s3);
  let (sc, sb) = butterfly_add_asym((rshift1(sc), sc), sb);
  let (s2, s5) = butterfly_sub_asym((rshift1(s2), s2), s5);
  let (sa, sd) = butterfly_sub_asym((rshift1(sa), sa), sd);
  let (s6, s1) = butterfly_add_asym((rshift1(s6), s6), s1);
  let (se, s9) = butterfly_add_asym((rshift1(se), se), s9);

  // Stage 2
  let ((_s8h, s8), s4h) = butterfly_add(s8, s4);
  let ((_s7h, s7), sbh) = butterfly_add(s7, sb);
  let ((_sah, sa), s6h) = butterfly_sub(sa, s6);
  let ((_s5h, s5), s9h) = butterfly_sub(s5, s9);
  let (s0, s3h) = butterfly_add(s0, s3);
  let (sd, seh) = butterfly_add(sd, se);
  let (s2, s1h) = butterfly_sub(s2, s1);
  let (sf, sch) = butterfly_sub(sf, sc);

  // Stage 3
  //   9633/8192 = Sin[7*Pi/16] + Cos[7*Pi/16] = 1.1758756024193586
  // 12873/16384 = Sin[7*Pi/16] - Cos[7*Pi/16] = 0.7856949583871022
  //  6393/32768 = Cos[7*Pi/16]                = 0.1950903220161283
  let (s8, s7) =
    RotateAdd::kernel(s8, s7, ((9633, 13), (12873, 14), (6393, 15)));
  // 22725/16384 = Sin[5*Pi/16] + Cos[5*Pi/16] = 1.3870398453221475
  //  9041/32768 = Sin[5*Pi/16] - Cos[5*Pi/16] = 0.2758993792829431
  //   4551/8192 = Cos[5*Pi/16]                = 0.5555702330196022
  let (s9, s6) =
    RotateAdd::kernel(s9h, s6h, ((22725, 14), (9041, 15), (4551, 13)));
  //  11363/8192 = Sin[5*Pi/16] + Cos[5*Pi/16] = 1.3870398453221475
  //  9041/32768 = Sin[5*Pi/16] - Cos[5*Pi/16] = 0.2758993792829431
  //   4551/8192 = Cos[5*Pi/16]                = 0.5555702330196022
  let (s5, sa) =
    RotateNeg::kernel(s5, sa, ((11363, 13), (9041, 15), (4551, 13)));
  //  9633/32768 = Sin[7*Pi/16] + Cos[7*Pi/16] = 1.1758756024193586
  // 12873/16384 = Sin[7*Pi/16] - Cos[7*Pi/16] = 0.7856949583871022
  //  6393/32768 = Cos[7*Pi/16]                = 0.1950903220161283
  let (s4, sb) =
    RotateNeg::kernel(s4h, sbh, ((9633, 13), (12873, 14), (6393, 15)));

  // Stage 4
  let (s2, sc) = butterfly_add_asym(s2, sch);
  let (s0, s1) = butterfly_sub_asym(s0, s1h);
  let (sf, se) = butterfly_add_asym(sf, seh);
  let (sd, s3) = butterfly_add_asym(sd, s3h);
  let (s7, s6) = butterfly_add_asym((rshift1(s7), s7), s6);
  let (s8, s9) = butterfly_sub_asym((rshift1(s8), s8), s9);
  let (sa, sb) = butterfly_sub_asym((rshift1(sa), sa), sb);
  let (s5, s4) = butterfly_add_asym((rshift1(s5), s5), s4);

  // Stage 5
  // 10703/8192 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766
  // 8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969
  //  3135/8192 = Cos[3*Pi/8]               = 0.3826834323650898
  let (sc, s3) =
    RotateAdd::kernel(sc, s3, ((10703, 13), (8867, 14), (3135, 13)));
  // 10703/8192 = Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3870398453221475
  // 8867/16384 = Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969
  //  3135/8192 = Cos[3*Pi/8]               = 0.3826834323650898
  let (s2, sd) =
    RotateNeg::kernel(s2, sd, ((10703, 13), (8867, 14), (3135, 13)));
  // 11585/8192 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951
  //  5793/8192 = Cos[Pi/4]             = 0.7071067811865475
  let (sa, s5) = RotatePi4Add::kernel(sa, s5, ((11585, 13), (5793, 13)));
  // 11585/8192 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951
  //  5793/8192 = Cos[Pi/4]             = 0.7071067811865475
  let (s6, s9) = RotatePi4Add::kernel(s6, s9, ((11585, 13), (5793, 13)));
  // 11585/8192 = Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951
  //  5793/8192 = Cos[Pi/4]             = 0.7071067811865475
  let (se, s1) = RotatePi4Add::kernel(se, s1, ((11585, 13), (5793, 13)));

  store_coeffs!(
    output, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sa, sb, sc, sd, se, sf
  );
}

fn daala_fdct_ii_32(
  t0: i32, t1: i32, t2: i32, t3: i32, t4: i32, t5: i32, t6: i32, t7: i32,
  t8: i32, t9: i32, ta: i32, tb: i32, tc: i32, td: i32, te: i32, tf: i32,
  tg: i32, th: i32, ti: i32, tj: i32, tk: i32, tl: i32, tm: i32, tn: i32,
  to: i32, tp: i32, tq: i32, tr: i32, ts: i32, tt: i32, tu: i32, tv: i32,
  output: &mut [i32]
) {
  // +/- Butterflies with asymmetric output.
  let (t0h, tv) = butterfly_neg(t0, tv);
  let (t1, tuh) = butterfly_add(t1, tu);
  let (t2h, tt) = butterfly_neg(t2, tt);
  let (t3, tsh) = butterfly_add(t3, ts);
  let (t4h, tr) = butterfly_neg(t4, tr);
  let (t5, tqh) = butterfly_add(t5, tq);
  let (t6h, tp) = butterfly_neg(t6, tp);
  let (t7, toh) = butterfly_add(t7, to);
  let (t8h, tn) = butterfly_neg(t8, tn);
  let (t9, tmh) = butterfly_add(t9, tm);
  let (tah, tl) = butterfly_neg(ta, tl);
  let (tb, tkh) = butterfly_add(tb, tk);
  let (tch, tj) = butterfly_neg(tc, tj);
  let (td, tih) = butterfly_add(td, ti);
  let (teh, th) = butterfly_neg(te, th);
  let (tf, tgh) = butterfly_add(tf, tg);

  // Embedded 16-point transforms with asymmetric input.
  daala_fdct_ii_16_asym(
    t0h,
    t1,
    t2h,
    t3,
    t4h,
    t5,
    t6h,
    t7,
    t8h,
    t9,
    tah,
    tb,
    tch,
    td,
    teh,
    tf,
    &mut output[0..16]
  );
  daala_fdst_iv_16_asym(
    tv,
    tuh,
    tt,
    tsh,
    tr,
    tqh,
    tp,
    toh,
    tn,
    tmh,
    tl,
    tkh,
    tj,
    tih,
    th,
    tgh,
    &mut output[16..32]
  );
  output[16..32].reverse();
}

fn daala_fdct32(input: &[i32], output: &mut [i32]) {
  let mut temp_out: [i32; 32] = [0; 32];
  daala_fdct_ii_32(
    input[0],
    input[1],
    input[2],
    input[3],
    input[4],
    input[5],
    input[6],
    input[7],
    input[8],
    input[9],
    input[10],
    input[11],
    input[12],
    input[13],
    input[14],
    input[15],
    input[16],
    input[17],
    input[18],
    input[19],
    input[20],
    input[21],
    input[22],
    input[23],
    input[24],
    input[25],
    input[26],
    input[27],
    input[28],
    input[29],
    input[30],
    input[31],
    &mut temp_out
  );

  output[0] = temp_out[0];
  output[1] = temp_out[16];
  output[2] = temp_out[8];
  output[3] = temp_out[24];
  output[4] = temp_out[4];
  output[5] = temp_out[20];
  output[6] = temp_out[12];
  output[7] = temp_out[28];
  output[8] = temp_out[2];
  output[9] = temp_out[18];
  output[10] = temp_out[10];
  output[11] = temp_out[26];
  output[12] = temp_out[6];
  output[13] = temp_out[22];
  output[14] = temp_out[14];
  output[15] = temp_out[30];
  output[16] = temp_out[1];
  output[17] = temp_out[17];
  output[18] = temp_out[9];
  output[19] = temp_out[25];
  output[20] = temp_out[5];
  output[21] = temp_out[21];
  output[22] = temp_out[13];
  output[23] = temp_out[29];
  output[24] = temp_out[3];
  output[25] = temp_out[19];
  output[26] = temp_out[11];
  output[27] = temp_out[27];
  output[28] = temp_out[7];
  output[29] = temp_out[23];
  output[30] = temp_out[15];
  output[31] = temp_out[31];
}

fn daala_fdct_ii_32_asym(
  t0h: i32, t1: (i32, i32), t2h: i32, t3: (i32, i32), t4h: i32,
  t5: (i32, i32), t6h: i32, t7: (i32, i32), t8h: i32, t9: (i32, i32),
  tah: i32, tb: (i32, i32), tch: i32, td: (i32, i32), teh: i32,
  tf: (i32, i32), tgh: i32, th: (i32, i32), tih: i32, tj: (i32, i32),
  tkh: i32, tl: (i32, i32), tmh: i32, tn: (i32, i32), toh: i32,
  tp: (i32, i32), tqh: i32, tr: (i32, i32), tsh: i32, tt: (i32, i32),
  tuh: i32, tv: (i32, i32), output: &mut [i32]
) {
  // +/- Butterflies with asymmetric input.
  let (t0, tv) = butterfly_neg_asym(t0h, tv);
  let (t1, tu) = butterfly_sub_asym(t1, tuh);
  let (t2, tt) = butterfly_neg_asym(t2h, tt);
  let (t3, ts) = butterfly_sub_asym(t3, tsh);
  let (t4, tr) = butterfly_neg_asym(t4h, tr);
  let (t5, tq) = butterfly_sub_asym(t5, tqh);
  let (t6, tp) = butterfly_neg_asym(t6h, tp);
  let (t7, to) = butterfly_sub_asym(t7, toh);
  let (t8, tn) = butterfly_neg_asym(t8h, tn);
  let (t9, tm) = butterfly_sub_asym(t9, tmh);
  let (ta, tl) = butterfly_neg_asym(tah, tl);
  let (tb, tk) = butterfly_sub_asym(tb, tkh);
  let (tc, tj) = butterfly_neg_asym(tch, tj);
  let (td, ti) = butterfly_sub_asym(td, tih);
  let (te, th) = butterfly_neg_asym(teh, th);
  let (tf, tg) = butterfly_sub_asym(tf, tgh);

  // Embedded 16-point orthonormal transforms.
  daala_fdct_ii_16(
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    t8,
    t9,
    ta,
    tb,
    tc,
    td,
    te,
    tf,
    &mut output[0..16]
  );
  daala_fdst_iv_16(
    tv,
    tu,
    tt,
    ts,
    tr,
    tq,
    tp,
    to,
    tn,
    tm,
    tl,
    tk,
    tj,
    ti,
    th,
    tg,
    &mut output[16..32]
  );
  output[16..32].reverse();
}

fn daala_fdst_iv_32_asym(
  t0: (i32, i32), t1h: i32, t2: (i32, i32), t3h: i32, t4: (i32, i32),
  t5h: i32, t6: (i32, i32), t7h: i32, t8: (i32, i32), t9h: i32,
  ta: (i32, i32), tbh: i32, tc: (i32, i32), tdh: i32, te: (i32, i32),
  tfh: i32, tg: (i32, i32), thh: i32, ti: (i32, i32), tjh: i32,
  tk: (i32, i32), tlh: i32, tm: (i32, i32), tnh: i32, to: (i32, i32),
  tph: i32, tq: (i32, i32), trh: i32, ts: (i32, i32), tth: i32,
  tu: (i32, i32), tvh: i32, output: &mut [i32]
) {
  // Stage 0
  // Sin[63*Pi/128] + Cos[63*Pi/128] = 1.0242400472191164
  // Sin[63*Pi/128] - Cos[63*Pi/128] = 0.9751575901732919
  // Cos[63*Pi/128]                  = 0.0245412285229123
  let (t0, tv) =
    RotateAdd::half_kernel(t0, tvh, ((5933, 13), (22595, 14), (1137, 15)));
  // Sin[61*Pi/128] + Cos[61*Pi/128] = 1.0708550202783576
  // Sin[61*Pi/128] - Cos[61*Pi/128] = 0.9237258930790228
  // Cos[61*Pi/128]                  = 0.0735645635996674
  let (tu, t1) =
    RotateSub::half_kernel(tu, t1h, ((6203, 13), (21403, 14), (3409, 15)));
  // Sin[59*Pi/128] + Cos[59*Pi/128] = 1.1148902097979262
  // Sin[59*Pi/128] - Cos[59*Pi/128] = 0.8700688593994937
  // Cos[59*Pi/128]                  = 0.1224106751992162
  let (t2, tt) =
    RotateAdd::half_kernel(t2, tth, ((25833, 15), (315, 8), (5673, 15)));
  // Sin[57*Pi/128] + Cos[57*Pi/128] = 1.1562395311492424
  // Sin[57*Pi/128] - Cos[57*Pi/128] = 0.8143157536286401
  // Cos[57*Pi/128]                  = 0.1709618887603012
  let (ts, t3) =
    RotateSub::half_kernel(ts, t3h, ((26791, 15), (4717, 12), (7923, 15)));
  // Sin[55*Pi/128] + Cos[55*Pi/128] = 1.1948033701953984
  // Sin[55*Pi/128] - Cos[55*Pi/128] = 0.7566008898816587
  // Cos[55*Pi/128]                  = 0.2191012401568698
  let (t4, tr) =
    RotateAdd::half_kernel(t4, trh, ((6921, 13), (17531, 14), (10153, 15)));
  // Sin[53*Pi/128] + Cos[53*Pi/128] = 1.2304888232703382
  // Sin[53*Pi/128] - Cos[53*Pi/128] = 0.6970633083205415
  // Cos[53*Pi/128]                  = 0.2667127574748984
  let (tq, t5) =
    RotateSub::half_kernel(tq, t5h, ((28511, 15), (32303, 15), (1545, 12)));
  // Sin[51*Pi/128] + Cos[51*Pi/128] = 1.2632099209919283
  // Sin[51*Pi/128] - Cos[51*Pi/128] = 0.6358464401941452
  // Cos[51*Pi/128]                  = 0.3136817403988915
  let (t6, tp) =
    RotateAdd::half_kernel(t6, tph, ((29269, 15), (14733, 14), (1817, 12)));
  // Sin[49*Pi/128] + Cos[49*Pi/128] = 1.2928878353697270
  // Sin[49*Pi/128] - Cos[49*Pi/128] = 0.5730977622997508
  // Cos[49*Pi/128]                  = 0.3598950365349881
  let (to, t7) =
    RotateSub::half_kernel(to, t7h, ((29957, 15), (13279, 14), (8339, 14)));
  // Sin[47*Pi/128] + Cos[47*Pi/128] = 1.3194510697085207
  // Sin[47*Pi/128] - Cos[47*Pi/128] = 0.5089684416985408
  // Cos[47*Pi/128]                  = 0.4052413140049899
  let (t8, tn) =
    RotateAdd::half_kernel(t8, tnh, ((7643, 13), (11793, 14), (18779, 15)));
  // Sin[45*Pi/128] + Cos[45*Pi/128] = 1.3428356308501219
  // Sin[45*Pi/128] - Cos[45*Pi/128] = 0.4436129715409088
  // Cos[45*Pi/128]                  = 0.4496113296546065
  let (tm, t9) =
    RotateSub::half_kernel(tm, t9h, ((15557, 14), (20557, 15), (20835, 15)));
  // Sin[43*Pi/128] + Cos[43*Pi/128] = 1.3629851833384956
  // Sin[43*Pi/128] - Cos[43*Pi/128] = 0.3771887988789274
  // Cos[43*Pi/128]                  = 0.4928981922297840
  let (ta, tl) =
    RotateAdd::half_kernel(ta, tlh, ((31581, 15), (17479, 15), (22841, 15)));
  // Sin[41*Pi/128] + Cos[41*Pi/128] = 1.3798511851368043
  // Sin[41*Pi/128] - Cos[41*Pi/128] = 0.3098559453626100
  // Cos[41*Pi/128]                  = 0.5349976198870972
  let (tk, tb) =
    RotateSub::half_kernel(tk, tbh, ((7993, 13), (14359, 15), (3099, 12)));
  // Sin[39*Pi/128] + Cos[39*Pi/128] = 1.3933930045694290
  // Sin[39*Pi/128] - Cos[39*Pi/128] = 0.2417766217337384
  // Cos[39*Pi/128]                  = 0.5758081914178453
  let (tc, tj) =
    RotateAdd::half_kernel(tc, tjh, ((16143, 14), (2801, 13), (26683, 15)));
  // Sin[37*Pi/128] + Cos[37*Pi/128] = 1.4035780182072331
  // Sin[37*Pi/128] - Cos[37*Pi/128] = 0.1731148370459795
  // Cos[37*Pi/128]                  = 0.6152315905806268
  let (ti, td) =
    RotateSub::half_kernel(ti, tdh, ((16261, 14), (4011, 14), (14255, 14)));
  // Sin[35*Pi/128] + Cos[35*Pi/128] = 1.4103816894602614
  // Sin[35*Pi/128] - Cos[35*Pi/128] = 0.1040360035527078
  // Cos[35*Pi/128]                  = 0.6531728429537768
  let (te, th) =
    RotateAdd::half_kernel(te, thh, ((32679, 15), (4821, 15), (30269, 15)));
  // Sin[33*Pi/128] + Cos[33*Pi/128] = 1.4137876276885337
  // Sin[33*Pi/128] - Cos[33*Pi/128] = 0.0347065382144002
  // Cos[33*Pi/128]                  = 0.6895405447370668
  let (tg, tf) =
    RotateSub::half_kernel(tg, tfh, ((16379, 14), (201, 12), (15977, 14)));

  // Stage 1
  let (t0, tfh) = butterfly_add(t0, tf);
  let (tv, tgh) = butterfly_sub(tv, tg);
  let (th, tuh) = butterfly_add(th, tu);
  let (te, t1h) = butterfly_sub(te, t1);
  let (t2, tdh) = butterfly_add(t2, td);
  let (tt, tih) = butterfly_sub(tt, ti);
  let (tj, tsh) = butterfly_add(tj, ts);
  let (tc, t3h) = butterfly_sub(tc, t3);
  let (t4, tbh) = butterfly_add(t4, tb);
  let (tr, tkh) = butterfly_sub(tr, tk);
  let (tl, tqh) = butterfly_add(tl, tq);
  let (ta, t5h) = butterfly_sub(ta, t5);
  let (t6, t9h) = butterfly_add(t6, t9);
  let (tp, tmh) = butterfly_sub(tp, tm);
  let (tn, toh) = butterfly_add(tn, to);
  let (t8, t7h) = butterfly_sub(t8, t7);

  // Stage 2
  let (t0, t7) = butterfly_sub_asym(t0, t7h);
  let (tv, to) = butterfly_add_asym(tv, toh);
  let (tp, tu) = butterfly_sub_asym(tp, tuh);
  let (t6, t1) = butterfly_add_asym(t6, t1h);
  let (t2, t5) = butterfly_sub_asym(t2, t5h);
  let (tt, tq) = butterfly_add_asym(tt, tqh);
  let (tr, ts) = butterfly_sub_asym(tr, tsh);
  let (t4, t3) = butterfly_add_asym(t4, t3h);
  let (t8, tg) = butterfly_add_asym(t8, tgh);
  let (te, tm) = butterfly_sub_asym(te, tmh);
  let (tn, tf) = butterfly_add_asym(tn, tfh);
  let (th, t9) = butterfly_sub_asym(th, t9h);
  let (ta, ti) = butterfly_add_asym(ta, tih);
  let (tc, tk) = butterfly_sub_asym(tc, tkh);
  let (tl, td) = butterfly_add_asym(tl, tdh);
  let (tj, tb) = butterfly_sub_asym(tj, tbh);

  // Stage 3
  // Sin[15*Pi/32] + Cos[15*Pi/32] = 1.0932018670017576
  // Sin[15*Pi/32] - Cos[15*Pi/32] = 0.8971675863426363
  // Cos[15*Pi/32]                 = 0.0980171403295606
  let (tf, tg) =
    RotateSub::kernel(tf, tg, ((17911, 14), (14699, 14), (803, 13)));
  // Sin[13*Pi/32] + Cos[13*Pi/32] = 1.2472250129866712
  // Sin[13*Pi/32] - Cos[13*Pi/32] = 0.6666556584777465
  // Cos[13*Pi/32]                 = 0.2902846772544623
  let (th, te) =
    RotateAdd::kernel(th, te, ((10217, 13), (5461, 13), (1189, 12)));
  // Sin[11*Pi/32] + Cos[11*Pi/32] = 1.3533180011743526
  // Sin[11*Pi/32] - Cos[11*Pi/32] = 0.4105245275223574
  // Cos[11*Pi/32]                 = 0.4713967368259976
  let (ti, td) =
    RotateAdd::kernel(ti, td, ((5543, 12), (3363, 13), (7723, 14)));
  // Sin[9*Pi/32] + Cos[9*Pi/32] = 1.4074037375263826
  // Sin[9*Pi/32] - Cos[9*Pi/32] = 0.1386171691990915
  // Cos[9*Pi/32]                = 0.6343932841636455
  let (tc, tj) =
    RotateSub::kernel(tc, tj, ((11529, 13), (2271, 14), (5197, 13)));
  // Sin[9*Pi/32] + Cos[9*Pi/32] = 1.4074037375263826
  // Sin[9*Pi/32] - Cos[9*Pi/32] = 0.1386171691990915
  // Cos[9*Pi/32]                = 0.6343932841636455
  let (tb, tk) =
    RotateNeg::kernel(tb, tk, ((11529, 13), (2271, 14), (5197, 13)));
  // Sin[11*Pi/32] + Cos[11*Pi/32] = 1.3533180011743526
  // Sin[11*Pi/32] - Cos[11*Pi/32] = 0.4105245275223574
  // Cos[11*Pi/32]                 = 0.4713967368259976
  let (ta, tl) =
    RotateNeg::kernel(ta, tl, ((5543, 12), (3363, 13), (7723, 14)));
  // Sin[13*Pi/32] + Cos[13*Pi/32] = 1.2472250129866712
  // Sin[13*Pi/32] - Cos[13*Pi/32] = 0.6666556584777465
  // Cos[13*Pi/32]                 = 0.2902846772544623
  let (t9, tm) =
    RotateNeg::kernel(t9, tm, ((10217, 13), (5461, 13), (1189, 12)));
  // Sin[15*Pi/32] + Cos[15*Pi/32] = 1.0932018670017576
  // Sin[15*Pi/32] - Cos[15*Pi/32] = 0.8971675863426363
  // Cos[15*Pi/32]                 = 0.0980171403295606
  let (t8, tn) =
    RotateNeg::kernel(t8, tn, ((17911, 14), (14699, 14), (803, 13)));

  // Stage 4
  let (t3, t0h) = butterfly_sub(t3, t0);
  let (ts, tvh) = butterfly_add(ts, tv);
  let (tu, tth) = butterfly_sub(tu, tt);
  let (t1, t2h) = butterfly_add(t1, t2);
  let ((_toh, to), t4h) = butterfly_add(to, t4);
  let ((_tqh, tq), t6h) = butterfly_sub(tq, t6);
  let ((_t7h, t7), trh) = butterfly_add(t7, tr);
  let ((_t5h, t5), tph) = butterfly_sub(t5, tp);
  let (tb, t8h) = butterfly_sub(tb, t8);
  let (tk, tnh) = butterfly_add(tk, tn);
  let (tm, tlh) = butterfly_sub(tm, tl);
  let (t9, tah) = butterfly_add(t9, ta);
  let (tf, tch) = butterfly_sub(tf, tc);
  let (tg, tjh) = butterfly_add(tg, tj);
  let (ti, thh) = butterfly_sub(ti, th);
  let (td, teh) = butterfly_add(td, te);

  // Stage 5
  // Sin[7*Pi/16] + Cos[7*Pi/16] = 1.1758756024193586
  // Sin[7*Pi/16] - Cos[7*Pi/16] = 0.7856949583871022
  // Cos[7*Pi/16]                = 0.1950903220161283
  let (to, t7) = RotateAdd::kernel(to, t7, ((301, 8), (1609, 11), (6393, 15)));
  // Sin[5*Pi/16] + Cos[5*Pi/16] = 1.3870398453221475
  // Sin[5*Pi/16] - Cos[5*Pi/16] = 0.2758993792829431
  // Cos[5*Pi/16]                = 0.5555702330196022
  let (tph, t6h) =
    RotateAdd::kernel(tph, t6h, ((11363, 13), (9041, 15), (4551, 13)));
  // Sin[5*Pi/16] + Cos[5*Pi/16] = 1.3870398453221475
  // Sin[5*Pi/16] - Cos[5*Pi/16] = 0.2758993792829431
  // Cos[5*Pi/16]                = 0.5555702330196022
  let (t5, tq) =
    RotateNeg::kernel(t5, tq, ((5681, 12), (9041, 15), (4551, 13)));
  // Sin[7*Pi/16] + Cos[7*Pi/16] = 1.1758756024193586
  // Sin[7*Pi/16] - Cos[7*Pi/16] = 0.7856949583871022
  // Cos[7*Pi/16]                = 0.1950903220161283
  let (t4h, trh) =
    RotateNeg::kernel(t4h, trh, ((9633, 13), (12873, 14), (6393, 15)));

  // Stage 6
  let (t1, t0) = butterfly_add_asym(t1, t0h);
  let (tu, tv) = butterfly_sub_asym(tu, tvh);
  let (ts, t2) = butterfly_sub_asym(ts, t2h);
  let (t3, tt) = butterfly_sub_asym(t3, tth);
  let (t5, t4) = butterfly_add_asym((rshift1(t5), t5), t4h);
  let (tq, tr) = butterfly_sub_asym((rshift1(tq), tq), trh);
  let (t7, t6) = butterfly_add_asym((rshift1(t7), t7), t6h);
  let (to, tp) = butterfly_sub_asym((rshift1(to), to), tph);
  let (t9, t8) = butterfly_add_asym(t9, t8h);
  let (tm, tn) = butterfly_sub_asym(tm, tnh);
  let (tk, ta) = butterfly_sub_asym(tk, tah);
  let (tb, tl) = butterfly_sub_asym(tb, tlh);
  let (ti, tc) = butterfly_add_asym(ti, tch);
  let (td, tj) = butterfly_add_asym(td, tjh);
  let (tf, te) = butterfly_add_asym(tf, teh);
  let (tg, th) = butterfly_sub_asym(tg, thh);

  // Stage 7
  // Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766
  // Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969
  // Cos[3*Pi/8]               = 0.3826834323650898
  let (t2, tt) = RotateNeg::kernel(t2, tt, ((669, 9), (8867, 14), (3135, 13)));
  // Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766
  // Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969
  // Cos[3*Pi/8]               = 0.3826834323650898
  let (ts, t3) = RotateAdd::kernel(ts, t3, ((669, 9), (8867, 14), (3135, 13)));
  // Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766
  // Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969
  // Cos[3*Pi/8]               = 0.3826834323650898
  let (ta, tl) = RotateNeg::kernel(ta, tl, ((669, 9), (8867, 14), (3135, 13)));
  // Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766
  // Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969
  // Cos[3*Pi/8]               = 0.3826834323650898
  let (tk, tb) = RotateAdd::kernel(tk, tb, ((669, 9), (8867, 14), (3135, 13)));
  // Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766
  // Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969
  // Cos[3*Pi/8]               = 0.3826834323650898
  let (tc, tj) = RotateAdd::kernel(tc, tj, ((669, 9), (8867, 14), (3135, 13)));
  // Sin[3*Pi/8] + Cos[3*Pi/8] = 1.3065629648763766
  // Sin[3*Pi/8] - Cos[3*Pi/8] = 0.5411961001461969
  // Cos[3*Pi/8]               = 0.3826834323650898
  let (ti, td) = RotateNeg::kernel(ti, td, ((669, 9), (8867, 14), (3135, 13)));
  // Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951
  // Cos[Pi/4]             = 0.7071067811865475
  let (tu, t1) = RotatePi4Add::kernel(tu, t1, ((5793, 12), (5793, 13)));
  // Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951
  // Cos[Pi/4]             = 0.7071067811865475
  let (tq, t5) = RotatePi4Add::kernel(tq, t5, ((5793, 12), (5793, 13)));
  // Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951
  // Cos[Pi/4]             = 0.7071067811865475
  let (tp, t6) = RotatePi4Sub::kernel(tp, t6, ((5793, 12), (5793, 13)));
  // Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951
  // Cos[Pi/4]             = 0.7071067811865475
  let (tm, t9) = RotatePi4Add::kernel(tm, t9, ((5793, 12), (5793, 13)));
  // Sin[Pi/4] + Cos[Pi/4] = 1.4142135623730951
  // Cos[Pi/4]             = 0.7071067811865475
  let (te, th) = RotatePi4Add::kernel(te, th, ((5793, 12), (5793, 13)));

  store_coeffs!(
    output, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf,
    tg, th, ti, tj, tk, tl, tm, tn, to, tp, tq, tr, ts, tt, tu, tv
  );
}

fn daala_fdct64(input: &[i32], output: &mut [i32]) {
  // Use arrays to avoid ridiculous variable names
  let mut asym: [(i32, i32); 32] = [(0, 0); 32];
  let mut half: [i32; 32] = [0; 32];
  // +/- Butterflies with asymmetric output.
  {
    let mut butterfly_pair = |i: usize| {
      let j = i * 2;
      let (ah, c) = butterfly_neg(input[j], input[63 - j]);
      let (b, dh) = butterfly_add(input[j + 1], input[63 - j - 1]);
      half[i] = ah;
      half[31 - i] = dh;
      asym[i] = b;
      asym[31 - i] = c;
    };
    butterfly_pair(0);
    butterfly_pair(1);
    butterfly_pair(2);
    butterfly_pair(3);
    butterfly_pair(4);
    butterfly_pair(5);
    butterfly_pair(6);
    butterfly_pair(7);
    butterfly_pair(8);
    butterfly_pair(9);
    butterfly_pair(10);
    butterfly_pair(11);
    butterfly_pair(12);
    butterfly_pair(13);
    butterfly_pair(14);
    butterfly_pair(15);
  }

  let mut temp_out: [i32; 64] = [0; 64];
  // Embedded 2-point transforms with asymmetric input.
  daala_fdct_ii_32_asym(
    half[0],
    asym[0],
    half[1],
    asym[1],
    half[2],
    asym[2],
    half[3],
    asym[3],
    half[4],
    asym[4],
    half[5],
    asym[5],
    half[6],
    asym[6],
    half[7],
    asym[7],
    half[8],
    asym[8],
    half[9],
    asym[9],
    half[10],
    asym[10],
    half[11],
    asym[11],
    half[12],
    asym[12],
    half[13],
    asym[13],
    half[14],
    asym[14],
    half[15],
    asym[15],
    &mut temp_out[0..32]
  );
  daala_fdst_iv_32_asym(
    asym[31],
    half[31],
    asym[30],
    half[30],
    asym[29],
    half[29],
    asym[28],
    half[28],
    asym[27],
    half[27],
    asym[26],
    half[26],
    asym[25],
    half[25],
    asym[24],
    half[24],
    asym[23],
    half[23],
    asym[22],
    half[22],
    asym[21],
    half[21],
    asym[20],
    half[20],
    asym[19],
    half[19],
    asym[18],
    half[18],
    asym[17],
    half[17],
    asym[16],
    half[16],
    &mut temp_out[32..64]
  );
  temp_out[32..64].reverse();

  // Store a reordered version of output in temp_out
  let mut reorder_4 = |i: usize, j: usize| {
    output[0 + i * 4] = temp_out[0 + j];
    output[1 + i * 4] = temp_out[32 + j];
    output[2 + i * 4] = temp_out[16 + j];
    output[3 + i * 4] = temp_out[48 + j];
  };
  reorder_4(0, 0);
  reorder_4(1, 8);
  reorder_4(2, 4);
  reorder_4(3, 12);
  reorder_4(4, 2);
  reorder_4(5, 10);
  reorder_4(6, 6);
  reorder_4(7, 14);

  reorder_4(8, 1);
  reorder_4(9, 9);
  reorder_4(10, 5);
  reorder_4(11, 13);
  reorder_4(12, 3);
  reorder_4(13, 11);
  reorder_4(14, 7);
  reorder_4(15, 15);
}

fn fidentity4(input: &[i32], output: &mut [i32]) {
  for i in 0..4 {
    output[i] = input[i];
  }
}

fn fidentity8(input: &[i32], output: &mut [i32]) {
  for i in 0..8 {
    output[i] = input[i];
  }
}

fn fidentity16(input: &[i32], output: &mut [i32]) {
  for i in 0..16 {
    output[i] = input[i];
  }
}

fn fidentity32(input: &[i32], output: &mut [i32]) {
  for i in 0..32 {
    output[i] = input[i];
  }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TxfmType {
  DCT4,
  DCT8,
  DCT16,
  DCT32,
  DCT64,
  ADST4,
  ADST8,
  ADST16,
  Identity4,
  Identity8,
  Identity16,
  Identity32,
  Invalid
}

impl TxfmType {
  const TX_TYPES_1D: usize = 4;
  const AV1_TXFM_TYPE_LS: [[TxfmType; Self::TX_TYPES_1D]; 5] = [
    [TxfmType::DCT4, TxfmType::ADST4, TxfmType::ADST4, TxfmType::Identity4],
    [TxfmType::DCT8, TxfmType::ADST8, TxfmType::ADST8, TxfmType::Identity8],
    [
      TxfmType::DCT16,
      TxfmType::ADST16,
      TxfmType::ADST16,
      TxfmType::Identity16
    ],
    [
      TxfmType::DCT32,
      TxfmType::Invalid,
      TxfmType::Invalid,
      TxfmType::Identity32
    ],
    [TxfmType::DCT64, TxfmType::Invalid, TxfmType::Invalid, TxfmType::Invalid]
  ];

  fn get_func(self) -> &'static TxfmFunc {
    match self {
      TxfmType::DCT4 => &daala_fdct4,
      TxfmType::DCT8 => &daala_fdct8,
      TxfmType::DCT16 => &daala_fdct16,
      TxfmType::DCT32 => &daala_fdct32,
      TxfmType::DCT64 => &daala_fdct64,
      TxfmType::ADST4 => &daala_fdst_vii_4,
      TxfmType::ADST8 => &daala_fdst8,
      TxfmType::ADST16 => &daala_fdst16,
      TxfmType::Identity4 => &fidentity4,
      TxfmType::Identity8 => &fidentity8,
      TxfmType::Identity16 => &fidentity16,
      TxfmType::Identity32 => &fidentity32,
      _ => unreachable!()
    }
  }
}

#[derive(Debug, Clone, Copy)]
struct Txfm2DFlipCfg {
  tx_size: TxSize,
  /// Flip upside down
  ud_flip: bool,
  /// Flip left to right
  lr_flip: bool,
  shift: TxfmShift,
  txfm_type_col: TxfmType,
  txfm_type_row: TxfmType
}

impl Txfm2DFlipCfg {
  fn fwd(tx_type: TxType, tx_size: TxSize, bd: usize) -> Self {
    let tx_type_1d_col = VTX_TAB[tx_type as usize];
    let tx_type_1d_row = HTX_TAB[tx_type as usize];
    let txw_idx = tx_size.width_index();
    let txh_idx = tx_size.height_index();
    let txfm_type_col =
      TxfmType::AV1_TXFM_TYPE_LS[txh_idx][tx_type_1d_col as usize];
    let txfm_type_row =
      TxfmType::AV1_TXFM_TYPE_LS[txw_idx][tx_type_1d_row as usize];
    assert_ne!(txfm_type_col, TxfmType::Invalid);
    assert_ne!(txfm_type_row, TxfmType::Invalid);
    let (ud_flip, lr_flip) = Self::get_flip_cfg(tx_type);
    let cfg = Txfm2DFlipCfg {
      tx_size,
      ud_flip,
      lr_flip,
      shift: FWD_TXFM_SHIFT_LS[tx_size as usize][(bd - 8) / 2],
      txfm_type_col,
      txfm_type_row
    };
    cfg
  }

  /// Determine the flip config, returning (ud_flip, lr_flip)
  fn get_flip_cfg(tx_type: TxType) -> (bool, bool) {
    match tx_type {
      TxType::DCT_DCT
      | TxType::ADST_DCT
      | TxType::DCT_ADST
      | TxType::ADST_ADST
      | TxType::IDTX
      | TxType::V_DCT
      | TxType::H_DCT
      | TxType::V_ADST
      | TxType::H_ADST => (false, false),
      TxType::FLIPADST_DCT | TxType::FLIPADST_ADST | TxType::V_FLIPADST =>
        (true, false),
      TxType::DCT_FLIPADST | TxType::ADST_FLIPADST | TxType::H_FLIPADST =>
        (false, true),
      TxType::FLIPADST_FLIPADST => (true, true)
    }
  }
}

trait FwdTxfm2D: Dim {
  fn fwd_txfm2d_daala(
    input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
    bd: usize
  ) {
    let buf = &mut [0i32; 64 * 64][..Self::W * Self::H];
    let cfg =
      Txfm2DFlipCfg::fwd(tx_type, TxSize::by_dims(Self::W, Self::H), bd);

    // Note when assigning txfm_size_col, we use the txfm_size from the
    // row configuration and vice versa. This is intentionally done to
    // accurately perform rectangular transforms. When the transform is
    // rectangular, the number of columns will be the same as the
    // txfm_size stored in the row cfg struct. It will make no difference
    // for square transforms.
    let txfm_size_col = TxSize::width(cfg.tx_size);
    let txfm_size_row = TxSize::height(cfg.tx_size);

    let txfm_func_col = cfg.txfm_type_col.get_func();
    let txfm_func_row = cfg.txfm_type_row.get_func();

    // Columns
    for c in 0..txfm_size_col {
      if cfg.ud_flip {
        // flip upside down
        for r in 0..txfm_size_row {
          output[r] = (input[(txfm_size_row - r - 1) * stride + c]).into();
        }
      } else {
        for r in 0..txfm_size_row {
          output[r] = (input[r * stride + c]).into();
        }
      }
      av1_round_shift_array(output, txfm_size_row, -cfg.shift[0]);
      txfm_func_col(
        &output[..txfm_size_row].to_vec(),
        &mut output[txfm_size_row..]
      );
      av1_round_shift_array(
        &mut output[txfm_size_row..],
        txfm_size_row,
        -cfg.shift[1]
      );
      if cfg.lr_flip {
        for r in 0..txfm_size_row {
          // flip from left to right
          buf[r * txfm_size_col + (txfm_size_col - c - 1)] =
            output[txfm_size_row + r];
        }
      } else {
        for r in 0..txfm_size_row {
          buf[r * txfm_size_col + c] = output[txfm_size_row + r];
        }
      }
    }

    // Rows
    for r in 0..txfm_size_row {
      txfm_func_row(
        &buf[r * txfm_size_col..],
        &mut output[r * txfm_size_col..]
      );
      av1_round_shift_array(
        &mut output[r * txfm_size_col..],
        txfm_size_col,
        -cfg.shift[2]
      );
    }
  }
}

macro_rules! impl_fwd_txs {
  ($(($W:expr, $H:expr)),+) => {
    $(
      paste::item! {
        impl FwdTxfm2D for [<Block $W x $H>] {}
      }
    )*
  }
}

impl_fwd_txs! { (4, 4), (8, 8), (16, 16), (32, 32), (64, 64) }
impl_fwd_txs! { (4, 8), (8, 16), (16, 32), (32, 64) }
impl_fwd_txs! { (8, 4), (16, 8), (32, 16), (64, 32) }
impl_fwd_txs! { (4, 16), (8, 32), (16, 64) }
impl_fwd_txs! { (16, 4), (32, 8), (64, 16) }

pub fn fht4x4(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  Block4x4::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth);
}

pub fn fht8x8(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  Block8x8::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth);
}

pub fn fht16x16(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  Block16x16::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth);
}

pub fn fht32x32(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  Block32x32::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth);
}

pub fn fht64x64(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  assert!(tx_type == TxType::DCT_DCT);
  let mut tmp = [0 as i32; 4096];

  //Block64x64::fwd_txfm2d(input, &mut tmp, stride, tx_type, bit_depth);
  Block64x64::fwd_txfm2d_daala(input, &mut tmp, stride, tx_type, bit_depth);

  for i in 0..2 {
    for (row_out, row_in) in output[2048*i..].chunks_mut(32).zip(tmp[32*i..].chunks(64)).take(64) {
      row_out.copy_from_slice(&row_in[..32]);
    }
  }
}

pub fn fht4x8(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  Block4x8::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth);
}

pub fn fht8x4(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  Block8x4::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth);
}

pub fn fht8x16(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  Block8x16::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth);
}

pub fn fht16x8(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  Block16x8::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth);
}

pub fn fht16x32(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  assert!(tx_type == TxType::DCT_DCT || tx_type == TxType::IDTX);
  Block16x32::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth);
}

pub fn fht32x16(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  assert!(tx_type == TxType::DCT_DCT || tx_type == TxType::IDTX);
  Block32x16::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth);
}

pub fn fht32x64(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  assert!(tx_type == TxType::DCT_DCT);
  let mut tmp = [0 as i32; 2048];

  Block32x64::fwd_txfm2d_daala(input, &mut tmp, stride, tx_type, bit_depth);

  for (row_out, row_in) in output.chunks_mut(32).
    zip(tmp.chunks(32)).take(64) {
    row_out.copy_from_slice(&row_in[..32]);
  }
}

pub fn fht64x32(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  assert!(tx_type == TxType::DCT_DCT);
  let mut tmp = [0 as i32; 2048];

  Block64x32::fwd_txfm2d_daala(input, &mut tmp, stride, tx_type, bit_depth);

  for i in 0..2 {
    for (row_out, row_in) in output[1024*i..].chunks_mut(32).
      zip(tmp[32*i..].chunks(64)).take(32) {
      row_out.copy_from_slice(&row_in[..32]);
    }
  }
}

pub fn fht4x16(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  Block4x16::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth);
}

pub fn fht16x4(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  Block16x4::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth);
}

pub fn fht8x32(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  assert!(tx_type == TxType::DCT_DCT || tx_type == TxType::IDTX);
  Block8x32::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth);
}

pub fn fht32x8(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  assert!(tx_type == TxType::DCT_DCT || tx_type == TxType::IDTX);
  Block32x8::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth);
}

pub fn fht16x64(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  assert!(tx_type == TxType::DCT_DCT);
  let mut tmp = [0 as i32; 1024];

  Block16x64::fwd_txfm2d_daala(input, &mut tmp, stride, tx_type, bit_depth);

  for (row_out, row_in) in output.chunks_mut(16).
    zip(tmp.chunks(16)).take(64) {
    row_out.copy_from_slice(&row_in[..16]);
  }
}

pub fn fht64x16(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  assert!(tx_type == TxType::DCT_DCT);
  let mut tmp = [0 as i32; 1024];

  Block64x16::fwd_txfm2d_daala(input, &mut tmp, stride, tx_type, bit_depth);

  for i in 0..2 {
    for (row_out, row_in) in output[512*i..].chunks_mut(32).
      zip(tmp[32*i..].chunks(64)).take(16) {
      row_out.copy_from_slice(&row_in[..32]);
    }
  }
}
