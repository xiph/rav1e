// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use partition::TxSize;

extern {
  static dc_qlookup_Q3: [i16; 256];
  static ac_qlookup_Q3: [i16; 256];
}

fn get_tx_scale(tx_size: TxSize) -> u8 {
  let tx_scale = match tx_size {
    TxSize::TX_64X64 => 4,
    TxSize::TX_32X32 => 2,
    _ => 1
  };

  tx_scale
}

pub fn dc_q(qindex: usize) -> i16 {
  unsafe { dc_qlookup_Q3[qindex] }
}

pub fn ac_q(qindex: usize) -> i16 {
  unsafe { ac_qlookup_Q3[qindex] }
}

pub fn quantize_in_place(qindex: usize, coeffs: &mut [i32], tx_size: TxSize) {
  let tx_scale = get_tx_scale(tx_size) as i32;

  let dc_quant = dc_q(qindex) as i32;
  let ac_quant = ac_q(qindex) as i32;

  // using 21/64=0.328125 as rounding offset. To be tuned
  let dc_offset = dc_quant * 21 / 64 as i32;
  let ac_offset = ac_quant * 21 / 64 as i32;

  coeffs[0] *= tx_scale;
  coeffs[0] += coeffs[0].signum() * dc_offset;
  coeffs[0] /= dc_quant;

  for c in coeffs[1..].iter_mut() {
    *c *= tx_scale;
    *c += c.signum() * ac_offset;
    *c /= ac_quant;
  }
}

pub fn dequantize(
  qindex: usize, coeffs: &[i32], rcoeffs: &mut [i32], tx_size: TxSize
) {
  let tx_scale = get_tx_scale(tx_size) as i32;

  rcoeffs[0] = (coeffs[0] * dc_q(qindex) as i32) / tx_scale;
  let ac_quant = ac_q(qindex) as i32;

  for (r, &c) in rcoeffs.iter_mut().zip(coeffs.iter()).skip(1) {
    *r = c * ac_quant / tx_scale;
  }
}
