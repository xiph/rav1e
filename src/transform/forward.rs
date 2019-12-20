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

  type TxfmFunc = dyn Fn(&[i32], &mut [i32]);

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
      Identity4 => &fidentity4,
      Identity8 => &fidentity8,
      Identity16 => &fidentity16,
      Identity32 => &fidentity32,
      _ => unreachable!(),
    }
  }

  pub trait FwdTxfm2D: Dim {
    fn fwd_txfm2d_daala<T: Coefficient>(
      input: &[i16], output: &mut [T], stride: usize, tx_type: TxType,
      bd: usize, _cpu: CpuFeatureLevel,
    ) {
      let mut tmp1: AlignedArray<[i32; 64 * 64]> =
        AlignedArray::uninitialized();
      let mut tmp2: AlignedArray<[i32; 64 * 64]> =
        AlignedArray::uninitialized();
      let buf1 = &mut tmp1.array[..Self::W * Self::H];
      let buf2 = &mut tmp2.array[..Self::W * Self::H];

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

      let txfm_func_col = get_func(cfg.txfm_type_col);
      let txfm_func_row = get_func(cfg.txfm_type_row);

      // Columns
      for c in 0..txfm_size_col {
        let mut col_flip_backing: AlignedArray<[i32; 64]> =
          AlignedArray::uninitialized();
        let col_flip = &mut col_flip_backing.array[..txfm_size_row];
        if cfg.ud_flip {
          // flip upside down
          for r in 0..txfm_size_row {
            col_flip[r] = (input[(txfm_size_row - r - 1) * stride + c]).into();
          }
        } else {
          for r in 0..txfm_size_row {
            col_flip[r] = (input[r * stride + c]).into();
          }
        }

        let mut tx_output_backing: AlignedArray<[i32; 64]> =
          AlignedArray::uninitialized();
        let tx_output = &mut tx_output_backing.array[..txfm_size_row];
        av1_round_shift_array(col_flip, txfm_size_row, -cfg.shift[0]);
        txfm_func_col(&col_flip, tx_output);
        av1_round_shift_array(tx_output, txfm_size_row, -cfg.shift[1]);
        if cfg.lr_flip {
          for r in 0..txfm_size_row {
            // flip from left to right
            buf1[r * txfm_size_col + (txfm_size_col - c - 1)] = tx_output[r];
          }
        } else {
          for r in 0..txfm_size_row {
            buf1[r * txfm_size_col + c] = tx_output[r];
          }
        }
      }

      // Rows
      for r in 0..txfm_size_row {
        txfm_func_row(
          &buf1[r * txfm_size_col..],
          &mut buf2[r * txfm_size_col..],
        );
        av1_round_shift_array(
          &mut buf2[r * txfm_size_col..],
          txfm_size_col,
          -cfg.shift[2],
        );
        for c in 0..txfm_size_col {
          output[c * txfm_size_row + r] =
            T::cast_from(buf2[r * txfm_size_col + c]);
        }
      }
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
  let mut aligned: AlignedArray<[T; 4096]> = AlignedArray::uninitialized();
  let tmp = &mut aligned.array;

  //Block64x64::fwd_txfm2d(input, &mut tmp, stride, tx_type, bit_depth, cpu);
  Block64x64::fwd_txfm2d_daala(input, tmp, stride, tx_type, bit_depth, cpu);

  for i in 0..2 {
    for (row_out, row_in) in
      output[2048 * i..].chunks_mut(32).zip(tmp[32 * i..].chunks(64)).take(64)
    {
      row_out.copy_from_slice(&row_in[..32]);
    }
  }
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
  let mut aligned: AlignedArray<[T; 2048]> = AlignedArray::uninitialized();
  let tmp = &mut aligned.array;

  Block32x64::fwd_txfm2d_daala(input, tmp, stride, tx_type, bit_depth, cpu);

  for i in 0..2 {
    for (row_out, row_in) in
      output[1024 * i..].chunks_mut(32).zip(tmp[32 * i..].chunks(64)).take(32)
    {
      row_out.copy_from_slice(&row_in[..32]);
    }
  }
}

pub fn fht64x32<T: Coefficient>(
  input: &[i16], output: &mut [T], stride: usize, tx_type: TxType,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  assert!(tx_type == TxType::DCT_DCT);
  let mut aligned: AlignedArray<[T; 2048]> = AlignedArray::uninitialized();
  let tmp = &mut aligned.array;

  Block64x32::fwd_txfm2d_daala(input, tmp, stride, tx_type, bit_depth, cpu);

  for (row_out, row_in) in output.chunks_mut(32).zip(tmp.chunks(32)).take(64) {
    row_out.copy_from_slice(&row_in[..32]);
  }
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
  let mut aligned: AlignedArray<[T; 1024]> = AlignedArray::uninitialized();
  let tmp = &mut aligned.array;

  Block16x64::fwd_txfm2d_daala(input, tmp, stride, tx_type, bit_depth, cpu);

  for i in 0..2 {
    for (row_out, row_in) in
      output[512 * i..].chunks_mut(32).zip(tmp[32 * i..].chunks(64)).take(16)
    {
      row_out.copy_from_slice(&row_in[..32]);
    }
  }
}

pub fn fht64x16<T: Coefficient>(
  input: &[i16], output: &mut [T], stride: usize, tx_type: TxType,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  assert!(tx_type == TxType::DCT_DCT);
  let mut aligned: AlignedArray<[T; 1024]> = AlignedArray::uninitialized();
  let tmp = &mut aligned.array;

  Block64x16::fwd_txfm2d_daala(input, tmp, stride, tx_type, bit_depth, cpu);

  for (row_out, row_in) in output.chunks_mut(16).zip(tmp.chunks(16)).take(64) {
    row_out.copy_from_slice(&row_in[..16]);
  }
}
