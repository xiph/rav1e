// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

pub use self::forward::*;
pub use self::inverse::*;

use partition::{TxSize, TxType, TX_TYPES};
use predict::*;
use util::*;

mod forward;
mod inverse;

static SQRT2_BITS: usize = 12;
static SQRT2: i32 = 5793;       // 2^12 * sqrt(2)
static INV_SQRT2: i32 = 2896;   // 2^12 / sqrt(2)

/// Utility function that returns the log of the ratio of the col and row sizes.
#[inline]
pub fn get_rect_tx_log_ratio(col: usize, row: usize) -> i8 {
  debug_assert!(col > 0 && row > 0);
  col.ilog() as i8 - row.ilog() as i8
}

// performs half a butterfly
#[inline]
fn half_btf(w0: i32, in0: i32, w1: i32, in1: i32, bit: usize) -> i32 {
  // Ensure defined behaviour for when w0*in0 + w1*in1 is negative and
  //   overflows, but w0*in0 + w1*in1 + rounding isn't.
  let result = (w0 * in0).wrapping_add(w1 * in1);
  // Implement a version of round_shift with wrapping
  if bit <= 0 {
    result
  } else {
    result.wrapping_add(1 << (bit - 1)) >> bit
  }
}

// clamps value to a signed integer type of bit bits
#[inline]
fn clamp_value(value: i32, bit: usize) -> i32 {
  let max_value: i32 = ((1i64 << (bit - 1)) - 1) as i32;
  let min_value: i32 = (-(1i64 << (bit - 1))) as i32;
  clamp(value, min_value, max_value)
}

pub fn av1_round_shift_array(arr: &mut [i32], size: usize, bit: i8) {
  // FIXME
  //  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  //      {
  //        if is_x86_feature_detected!("sse4.1") {
  //          return unsafe {
  //            x86_asm::av1_round_shift_array_sse4_1(arr, size, bit)
  //          };
  //        }
  //      }
  av1_round_shift_array_rs(arr, size, bit)
}

fn av1_round_shift_array_rs(arr: &mut [i32], size: usize, bit: i8) {
  if bit == 0 {
    return;
  }
  if bit > 0 {
    let bit = bit as usize;
    for i in 0..size {
      arr[i] = round_shift(arr[i], bit);
    }
  } else {
    for i in 0..size {
      arr[i] =
        clamp((1 << (-bit)) * arr[i], i32::min_value(), i32::max_value());
    }
  }
}

#[derive(Debug, Clone, Copy)]
enum TxType1D {
  DCT,
  ADST,
  FLIPADST,
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

const VTX_TAB: [TxType1D; TX_TYPES] = [
  TxType1D::DCT,
  TxType1D::ADST,
  TxType1D::DCT,
  TxType1D::ADST,
  TxType1D::FLIPADST,
  TxType1D::DCT,
  TxType1D::FLIPADST,
  TxType1D::ADST,
  TxType1D::FLIPADST,
  TxType1D::IDTX,
  TxType1D::DCT,
  TxType1D::IDTX,
  TxType1D::ADST,
  TxType1D::IDTX,
  TxType1D::FLIPADST,
  TxType1D::IDTX
];

const HTX_TAB: [TxType1D; TX_TYPES] = [
  TxType1D::DCT,
  TxType1D::DCT,
  TxType1D::ADST,
  TxType1D::ADST,
  TxType1D::DCT,
  TxType1D::FLIPADST,
  TxType1D::FLIPADST,
  TxType1D::FLIPADST,
  TxType1D::ADST,
  TxType1D::IDTX,
  TxType1D::IDTX,
  TxType1D::DCT,
  TxType1D::IDTX,
  TxType1D::ADST,
  TxType1D::IDTX,
  TxType1D::FLIPADST
];

pub fn forward_transform(
  input: &[i16], output: &mut [i32], stride: usize, tx_size: TxSize,
  tx_type: TxType, bit_depth: usize
) {
  use self::TxSize::*;
  match tx_size {
    TX_4X4 => fht4x4(input, output, stride, tx_type, bit_depth),
    TX_8X8 => fht8x8(input, output, stride, tx_type, bit_depth),
    TX_16X16 => fht16x16(input, output, stride, tx_type, bit_depth),
    TX_32X32 => fht32x32(input, output, stride, tx_type, bit_depth),
    TX_64X64 => fht64x64(input, output, stride, tx_type, bit_depth),

    TX_4X8 => fht4x8(input, output, stride, tx_type, bit_depth),
    TX_8X4 => fht8x4(input, output, stride, tx_type, bit_depth),
    TX_8X16 => fht8x16(input, output, stride, tx_type, bit_depth),
    TX_16X8 => fht16x8(input, output, stride, tx_type, bit_depth),
    TX_16X32 => fht16x32(input, output, stride, tx_type, bit_depth),
    TX_32X16 => fht32x16(input, output, stride, tx_type, bit_depth),
    TX_32X64 => fht32x64(input, output, stride, tx_type, bit_depth),
    TX_64X32 => fht64x32(input, output, stride, tx_type, bit_depth),

    TX_4X16 => fht4x16(input, output, stride, tx_type, bit_depth),
    TX_16X4 => fht16x4(input, output, stride, tx_type, bit_depth),
    TX_8X32 => fht8x32(input, output, stride, tx_type, bit_depth),
    TX_32X8 => fht32x8(input, output, stride, tx_type, bit_depth),
    TX_16X64 => fht16x64(input, output, stride, tx_type, bit_depth),
    TX_64X16 => fht64x16(input, output, stride, tx_type, bit_depth),
  }
}

pub fn inverse_transform_add(
  input: &[i32], output: &mut [u16], stride: usize, tx_size: TxSize,
  tx_type: TxType, bit_depth: usize
) {
  use self::TxSize::*;
  match tx_size {
    TX_4X4 => iht4x4_add(input, output, stride, tx_type, bit_depth),
    TX_8X8 => iht8x8_add(input, output, stride, tx_type, bit_depth),
    TX_16X16 => iht16x16_add(input, output, stride, tx_type, bit_depth),
    TX_32X32 => iht32x32_add(input, output, stride, tx_type, bit_depth),
    TX_64X64 => iht64x64_add(input, output, stride, tx_type, bit_depth),

    TX_4X8 => iht4x8_add(input, output, stride, tx_type, bit_depth),
    TX_8X4 => iht8x4_add(input, output, stride, tx_type, bit_depth),
    TX_8X16 => iht8x16_add(input, output, stride, tx_type, bit_depth),
    TX_16X8 => iht16x8_add(input, output, stride, tx_type, bit_depth),
    TX_16X32 => iht16x32_add(input, output, stride, tx_type, bit_depth),
    TX_32X16 => iht32x16_add(input, output, stride, tx_type, bit_depth),
    TX_32X64 => iht32x64_add(input, output, stride, tx_type, bit_depth),
    TX_64X32 => iht64x32_add(input, output, stride, tx_type, bit_depth),

    TX_4X16 => iht4x16_add(input, output, stride, tx_type, bit_depth),
    TX_16X4 => iht16x4_add(input, output, stride, tx_type, bit_depth),
    TX_8X32 => iht8x32_add(input, output, stride, tx_type, bit_depth),
    TX_32X8 => iht32x8_add(input, output, stride, tx_type, bit_depth),
    TX_16X64 => iht16x64_add(input, output, stride, tx_type, bit_depth),
    TX_64X16 => iht64x16_add(input, output, stride, tx_type, bit_depth),
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use rand::random;

  fn test_roundtrip(tx_size: TxSize, tx_type: TxType, tolerance: i16) {
    let mut src_storage = [0u16; 64 * 64];
    let src = &mut src_storage[..tx_size.area()];
    let mut dst_storage = [0u16; 64 * 64];
    let dst = &mut dst_storage[..tx_size.area()];
    let mut res_storage = [0i16; 64 * 64];
    let res = &mut res_storage[..tx_size.area()];
    let mut freq_storage = [0i32; 64 * 64];
    let freq = &mut freq_storage[..tx_size.area()];
    for ((r, s), d) in res.iter_mut().zip(src.iter_mut()).zip(dst.iter_mut()) {
      *s = random::<u8>() as u16;
      *d = random::<u8>() as u16;
      *r = (*s as i16) - (*d as i16);
    }
    forward_transform(res, freq, tx_size.width(), tx_size, tx_type, 8);
    inverse_transform_add(freq, dst, tx_size.width(), tx_size, tx_type, 8);

    for (s, d) in src.iter().zip(dst) {
      assert!(i16::abs((*s as i16) - (*d as i16)) <= tolerance);
    }
  }

  #[test]
  fn log_tx_ratios() {
    let combinations = [
      (TxSize::TX_4X4, 0),
      (TxSize::TX_8X8, 0),
      (TxSize::TX_16X16, 0),
      (TxSize::TX_32X32, 0),
      (TxSize::TX_64X64, 0),

      (TxSize::TX_4X8, -1),
      (TxSize::TX_8X4, 1),
      (TxSize::TX_8X16, -1),
      (TxSize::TX_16X8, 1),
      (TxSize::TX_16X32, -1),
      (TxSize::TX_32X16, 1),
      (TxSize::TX_32X64, -1),
      (TxSize::TX_64X32, 1),

      (TxSize::TX_4X16, -2),
      (TxSize::TX_16X4, 2),
      (TxSize::TX_8X32, -2),
      (TxSize::TX_32X8, 2),
      (TxSize::TX_16X64, -2),
      (TxSize::TX_64X16, 2),
    ];

    for &(tx_size, expected) in combinations.iter() {
      println!("Testing combination {:?}, {:?}", tx_size.width(), tx_size.height());
      assert!(get_rect_tx_log_ratio(tx_size.width(), tx_size.height()) == expected);
    }
  }

  #[test]
  fn roundtrips() {
    use partition::TxSize::*;
    use partition::TxType::*;
    let combinations = [
      (TX_4X4, DCT_DCT, 0),
      (TX_4X4, ADST_DCT, 0),
      (TX_4X4, DCT_ADST, 0),
      (TX_4X4, ADST_ADST, 0),
      (TX_4X4, IDTX, 0),
      (TX_4X4, V_DCT, 0),
      (TX_4X4, H_DCT, 0),
      (TX_4X4, V_ADST, 0),
      (TX_4X4, H_ADST, 0),
      (TX_8X8, DCT_DCT, 1),
      (TX_8X8, ADST_DCT, 1),
      (TX_8X8, DCT_ADST, 1),
      (TX_8X8, ADST_ADST, 1),
      (TX_8X8, IDTX, 0),
      (TX_8X8, V_DCT, 0),
      (TX_8X8, H_DCT, 0),
      (TX_8X8, V_ADST, 0),
      (TX_8X8, H_ADST, 1),
      (TX_16X16, DCT_DCT, 1),
      (TX_16X16, ADST_DCT, 1),
      (TX_16X16, DCT_ADST, 1),
      (TX_16X16, ADST_ADST, 1),
      (TX_16X16, IDTX, 0),
      (TX_16X16, V_DCT, 1),
      (TX_16X16, H_DCT, 1),
      // 32x tranforms only use DCT_DCT and IDTX
      (TX_32X32, DCT_DCT, 2),
      (TX_32X32, IDTX, 0),
      // 64x tranforms only use DCT_DCT and IDTX
      //(TX_64X64, DCT_DCT, 0),
      (TX_4X8, DCT_DCT, 1),
      (TX_8X4, DCT_DCT, 1),
      (TX_4X16, DCT_DCT, 1),
      (TX_16X4, DCT_DCT, 1),
      (TX_8X16, DCT_DCT, 1),
      (TX_16X8, DCT_DCT, 1),
      (TX_8X32, DCT_DCT, 2),
      (TX_32X8, DCT_DCT, 2),
      (TX_16X32, DCT_DCT, 2),
      (TX_32X16, DCT_DCT, 2),
    ];
    for &(tx_size, tx_type, tolerance) in combinations.iter() {
      println!("Testing combination {:?}, {:?}", tx_size, tx_type);
      test_roundtrip(tx_size, tx_type, tolerance);
    }
  }
}
