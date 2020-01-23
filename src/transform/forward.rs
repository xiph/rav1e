// Copyright (c) 2018-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::cpu_features::CpuFeatureLevel;
use crate::util::*;

use super::TxType;

cfg_if::cfg_if! {
  if #[cfg(nasm_x86_64)] {
    pub use crate::asm::x86::transform::forward::*;
  } else {
    pub use self::native::*;
  }
}

pub mod native {
  use super::*;

  use crate::transform::av1_round_shift_array;
  use crate::transform::forward_shared::*;
  use crate::transform::Dim;
  use crate::transform::TxSize;

  pub trait TxOperations: Copy {
    fn zero() -> Self;

    fn tx_mul(self, _: (i32, i32)) -> Self;
    fn rshift1(self) -> Self;
    fn add(self, b: Self) -> Self;
    fn sub(self, b: Self) -> Self;
    fn add_avg(self, b: Self) -> Self;
    fn sub_avg(self, b: Self) -> Self;

    fn copy_fn(self) -> Self {
      self
    }
  }

  impl TxOperations for i32 {
    fn zero() -> Self {
      0
    }

    fn tx_mul(self, mul: (i32, i32)) -> Self {
      ((self * mul.0) + (1 << mul.1 >> 1)) >> mul.1
    }

    fn rshift1(self) -> Self {
      (self + if self < 0 { 1 } else { 0 }) >> 1
    }

    fn add(self, b: Self) -> Self {
      self + b
    }

    fn sub(self, b: Self) -> Self {
      self - b
    }

    fn add_avg(self, b: Self) -> Self {
      (self + b) >> 1
    }

    fn sub_avg(self, b: Self) -> Self {
      (self - b) >> 1
    }
  }

  impl_1d_tx!();

  type TxfmFunc = dyn Fn(&mut [i32]);

  fn get_func(t: TxfmType) -> &'static TxfmFunc {
    use self::TxfmType::*;
    match t {
      DCT4 => &daala_fdct4,
      DCT8 => &daala_fdct8,
      DCT16 => &daala_fdct16,
      DCT32 => &daala_fdct32,
      DCT64 => &daala_fdct64,
      ADST4 => &daala_fdst_vii_4,
      ADST8 => &daala_fdst8,
      ADST16 => &daala_fdst16,
      Identity4 => &fidentity,
      Identity8 => &fidentity,
      Identity16 => &fidentity,
      Identity32 => &fidentity,
      _ => unreachable!(),
    }
  }

  fn fwd_txfm2d<T: Coefficient>(
    input: &[i16], output: &mut [T], stride: usize, tx_size: TxSize,
    tx_type: TxType, bd: usize,
  ) {
    // Note when assigning txfm_size_col, we use the txfm_size from the
    // row configuration and vice versa. This is intentionally done to
    // accurately perform rectangular transforms. When the transform is
    // rectangular, the number of columns will be the same as the
    // txfm_size stored in the row cfg struct. It will make no difference
    // for square transforms.
    let txfm_size_col = tx_size.width();
    let txfm_size_row = tx_size.height();

    let mut tmp: Aligned<[i32; 64 * 64]> = Aligned::uninitialized();
    let buf = &mut tmp.data[..txfm_size_col * txfm_size_row];

    let cfg = Txfm2DFlipCfg::fwd(tx_type, tx_size, bd);

    let txfm_func_col = get_func(cfg.txfm_type_col);
    let txfm_func_row = get_func(cfg.txfm_type_row);

    // Columns
    for c in 0..txfm_size_col {
      let mut col_coeffs_backing: Aligned<[i32; 64]> =
        Aligned::uninitialized();
      let col_coeffs = &mut col_coeffs_backing.data[..txfm_size_row];
      if cfg.ud_flip {
        // flip upside down
        for r in 0..txfm_size_row {
          col_coeffs[r] = (input[(txfm_size_row - r - 1) * stride + c]).into();
        }
      } else {
        for r in 0..txfm_size_row {
          col_coeffs[r] = (input[r * stride + c]).into();
        }
      }

      av1_round_shift_array(col_coeffs, txfm_size_row, -cfg.shift[0]);
      txfm_func_col(col_coeffs);
      av1_round_shift_array(col_coeffs, txfm_size_row, -cfg.shift[1]);
      if cfg.lr_flip {
        for r in 0..txfm_size_row {
          // flip from left to right
          buf[r * txfm_size_col + (txfm_size_col - c - 1)] = col_coeffs[r];
        }
      } else {
        for r in 0..txfm_size_row {
          buf[r * txfm_size_col + c] = col_coeffs[r];
        }
      }
    }

    // Rows
    for r in 0..txfm_size_row {
      let row_coeffs = &mut buf[r * txfm_size_col..];
      txfm_func_row(row_coeffs);
      av1_round_shift_array(row_coeffs, txfm_size_col, -cfg.shift[2]);

      // Store output in at most 32x32 chunks so that the first 32x32
      // coefficients are stored first. When we don't have 64 rows, there is no
      // change in order. With 64 rows, the chunks are in this order
      //  - First 32 rows and first 32 cols
      //  - Last 32 rows and first 32 cols
      //  - First 32 rows and last 32 cols
      //  - Last 32 rows and last 32 cols

      // Output is grouped into 32x32 chunks so a stride of at most 32 is
      // used for each chunk.
      let output_stride = txfm_size_row.min(32);

      // Split the first 32 rows from the last 32 rows
      let output = &mut output
        [(r >= 32) as usize * output_stride * txfm_size_col.min(32)..];

      for cg in (0..txfm_size_col).step_by(32) {
        // Split the first 32 cols from the last 32 cols
        let output = &mut output[txfm_size_row * cg..];

        for c in 0..txfm_size_col.min(32) {
          output[c * output_stride + (r & 31)] =
            T::cast_from(row_coeffs[c + cg]);
        }
      }
    }
  }

  pub trait FwdTxfm2D: Dim {
    fn fwd_txfm2d_daala<T: Coefficient>(
      input: &[i16], output: &mut [T], stride: usize, tx_type: TxType,
      bd: usize, _cpu: CpuFeatureLevel,
    ) {
      fwd_txfm2d(
        input,
        output,
        stride,
        TxSize::by_dims(Self::W, Self::H),
        tx_type,
        bd,
      );
    }
  }

  impl_fwd_txs!();
}

pub fn fht4x4<T: Coefficient>(
  input: &[i16], output: &mut [T], stride: usize, tx_type: TxType,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  Block4x4::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth, cpu);
}

pub fn fht8x8<T: Coefficient>(
  input: &[i16], output: &mut [T], stride: usize, tx_type: TxType,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  Block8x8::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth, cpu);
}

pub fn fht16x16<T: Coefficient>(
  input: &[i16], output: &mut [T], stride: usize, tx_type: TxType,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  Block16x16::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth, cpu);
}

pub fn fht32x32<T: Coefficient>(
  input: &[i16], output: &mut [T], stride: usize, tx_type: TxType,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  Block32x32::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth, cpu);
}

pub fn fht64x64<T: Coefficient>(
  input: &[i16], output: &mut [T], stride: usize, tx_type: TxType,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  assert!(tx_type == TxType::DCT_DCT);
  Block64x64::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth, cpu);
}

pub fn fht4x8<T: Coefficient>(
  input: &[i16], output: &mut [T], stride: usize, tx_type: TxType,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  Block4x8::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth, cpu);
}

pub fn fht8x4<T: Coefficient>(
  input: &[i16], output: &mut [T], stride: usize, tx_type: TxType,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  Block8x4::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth, cpu);
}

pub fn fht8x16<T: Coefficient>(
  input: &[i16], output: &mut [T], stride: usize, tx_type: TxType,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  Block8x16::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth, cpu);
}

pub fn fht16x8<T: Coefficient>(
  input: &[i16], output: &mut [T], stride: usize, tx_type: TxType,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  Block16x8::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth, cpu);
}

pub fn fht16x32<T: Coefficient>(
  input: &[i16], output: &mut [T], stride: usize, tx_type: TxType,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  assert!(tx_type == TxType::DCT_DCT || tx_type == TxType::IDTX);
  Block16x32::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth, cpu);
}

pub fn fht32x16<T: Coefficient>(
  input: &[i16], output: &mut [T], stride: usize, tx_type: TxType,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  assert!(tx_type == TxType::DCT_DCT || tx_type == TxType::IDTX);
  Block32x16::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth, cpu);
}

pub fn fht32x64<T: Coefficient>(
  input: &[i16], output: &mut [T], stride: usize, tx_type: TxType,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  assert!(tx_type == TxType::DCT_DCT);
  Block32x64::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth, cpu);
}

pub fn fht64x32<T: Coefficient>(
  input: &[i16], output: &mut [T], stride: usize, tx_type: TxType,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  assert!(tx_type == TxType::DCT_DCT);
  Block64x32::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth, cpu);
}

pub fn fht4x16<T: Coefficient>(
  input: &[i16], output: &mut [T], stride: usize, tx_type: TxType,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  Block4x16::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth, cpu);
}

pub fn fht16x4<T: Coefficient>(
  input: &[i16], output: &mut [T], stride: usize, tx_type: TxType,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  Block16x4::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth, cpu);
}

pub fn fht8x32<T: Coefficient>(
  input: &[i16], output: &mut [T], stride: usize, tx_type: TxType,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  assert!(tx_type == TxType::DCT_DCT || tx_type == TxType::IDTX);
  Block8x32::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth, cpu);
}

pub fn fht32x8<T: Coefficient>(
  input: &[i16], output: &mut [T], stride: usize, tx_type: TxType,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  assert!(tx_type == TxType::DCT_DCT || tx_type == TxType::IDTX);
  Block32x8::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth, cpu);
}

pub fn fht16x64<T: Coefficient>(
  input: &[i16], output: &mut [T], stride: usize, tx_type: TxType,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  assert!(tx_type == TxType::DCT_DCT);
  Block16x64::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth, cpu);
}

pub fn fht64x16<T: Coefficient>(
  input: &[i16], output: &mut [T], stride: usize, tx_type: TxType,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  assert!(tx_type == TxType::DCT_DCT);
  Block64x16::fwd_txfm2d_daala(input, output, stride, tx_type, bit_depth, cpu);
}
