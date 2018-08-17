// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![cfg_attr(feature = "cargo-clippy", allow(cast_lossless))]

use partition::TxSize;
use std::mem;

extern {
  static dc_qlookup_Q3: [i16; 256];
  static dc_qlookup_10_Q3: [i16; 256];
  static dc_qlookup_12_Q3: [i16; 256];

  static ac_qlookup_Q3: [i16; 256];  
  static ac_qlookup_10_Q3: [i16; 256];
  static ac_qlookup_12_Q3: [i16; 256];
}

fn get_log_tx_scale(tx_size: TxSize) -> i32 {
  match tx_size {
    TxSize::TX_64X64 => 2,
    TxSize::TX_32X32 => 1,
    _ => 0
  }
}

pub fn dc_q(qindex: usize, bit_depth: usize) -> i16 {
  let &table = match bit_depth {
    8 => &dc_qlookup_Q3,
    10 => &dc_qlookup_10_Q3,
    12 => &dc_qlookup_12_Q3,
    _ => unimplemented!()
  };

  table[qindex]
}

pub fn ac_q(qindex: usize, bit_depth: usize) -> i16 {
  let &table = match bit_depth {
    8 => &ac_qlookup_Q3,
    10 => &ac_qlookup_10_Q3,
    12 => &ac_qlookup_12_Q3,
    _ => unimplemented!()
  };

  table[qindex]
}

#[derive(Debug, Default, Clone, Copy)]
pub struct QuantizationContext {
  log_tx_scale: i32,
  dc_quant: u32,
  dc_offset: i32,
  dc_mul_add: (u32, u32, u32),

  ac_quant: u32,
  ac_offset: i32,
  ac_mul_add: (u32, u32, u32)
}

fn divu_gen(d: u32) -> (u32, u32, u32) {
  let nbits = (mem::size_of_val(&d) as u64) * 8;
  let m = nbits - d.leading_zeros() as u64 - 1;
  if (d & (d - 1)) == 0 {
    (0xFFFFFFFF, 0xFFFFFFFF, m as u32)
  } else {
    let d = d as u64;
    let t = (1u64 << (m + nbits)) / d;
    let r = (t * d + d) & ((1 << nbits) - 1);
    if r <= 1u64 << m {
      (t as u32 + 1, 0u32, m as u32)
    } else {
      (t as u32, t as u32, m as u32)
    }
  }
}

#[inline]
fn divu_pair(x: i32, d: (u32, u32, u32)) -> i32 {
  let y = if x < 0 { -x } else { x } as u64;
  let (a, b, shift) = d;
  let shift = shift as u64;
  let a = a as u64;
  let b = b as u64;

  let y = (((a * y + b) >> 32) >> shift) as i32;
  if x < 0 {
    -y
  } else {
    y
  }
}

#[cfg(test)]
mod test {
  use super::*;
  #[test]
  fn test_divu_pair() {
    for d in 1..1024 {
      for x in -1000..1000 {
        let ab = divu_gen(d as u32);
        assert_eq!(x / d, divu_pair(x, ab));
      }
    }
  }
  #[test]
  fn gen_divu_table() {
    let b: Vec<(u32, u32, u32)> =
      dc_qlookup_Q3.iter().map(|&v| divu_gen(v as u32)).collect();

    println!("{:?}", b);
  }
}

impl QuantizationContext {
  pub fn update(&mut self, qindex: usize, tx_size: TxSize, is_intra: bool, bit_depth: usize) {
    self.log_tx_scale = get_log_tx_scale(tx_size);

    self.dc_quant = dc_q(qindex, bit_depth) as u32;
    self.dc_mul_add = divu_gen(self.dc_quant);

    self.ac_quant = ac_q(qindex, bit_depth) as u32;
    self.ac_mul_add = divu_gen(self.ac_quant);

    self.dc_offset = self.dc_quant as i32 * (if is_intra {21} else {15}) / 64;
    self.ac_offset = self.ac_quant as i32 * (if is_intra {21} else {15}) / 64;
  }

  #[inline]
  pub fn quantize(&self, coeffs: &mut [i32]) {
    coeffs[0] <<= self.log_tx_scale;
    coeffs[0] += coeffs[0].signum() * self.dc_offset;
    coeffs[0] = divu_pair(coeffs[0], self.dc_mul_add);

    for c in coeffs[1..].iter_mut() {
      *c <<= self.log_tx_scale;
      *c += c.signum() * self.ac_offset;
      *c = divu_pair(*c, self.ac_mul_add);
    }
  }
}

pub fn quantize_in_place(qindex: usize, coeffs: &mut [i32], tx_size: TxSize, bit_depth: usize) {
  let log_tx_scale = get_log_tx_scale(tx_size);

  let dc_quant = dc_q(qindex, bit_depth) as i32;
  let ac_quant = ac_q(qindex, bit_depth) as i32;

  // using 21/64=0.328125 as rounding offset. To be tuned
  let dc_offset = dc_quant * 21 / 64 as i32;
  let ac_offset = ac_quant * 21 / 64 as i32;

  coeffs[0] <<= log_tx_scale;
  coeffs[0] += coeffs[0].signum() * dc_offset;
  coeffs[0] /= dc_quant;

  for c in coeffs[1..].iter_mut() {
    *c <<= log_tx_scale;
    *c += c.signum() * ac_offset;
    *c /= ac_quant;
  }
}

pub fn dequantize(
  qindex: usize, coeffs: &[i32], rcoeffs: &mut [i32], tx_size: TxSize, bit_depth: usize
) {
  let log_tx_scale = get_log_tx_scale(tx_size) as i32;

  let val = coeffs[0] * dc_q(qindex, bit_depth) as i32;
  rcoeffs[0] = (val + if val < 0 { (1 << log_tx_scale) - 1 } else { 0 }) >> log_tx_scale;

  let ac_quant = ac_q(qindex, bit_depth) as i32;

  for (r, &c) in rcoeffs.iter_mut().zip(coeffs.iter()).skip(1) {
    *r = (c * ac_quant + if c < 0 { (1 << log_tx_scale) - 1 } else { 0 }) >> log_tx_scale;
  }
}
