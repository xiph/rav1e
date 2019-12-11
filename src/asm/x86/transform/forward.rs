// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::cpu_features::CpuFeatureLevel;
use crate::transform::forward::native;
use crate::transform::forward_shared::*;
use crate::transform::*;
use crate::util::*;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

type TxfmFuncI32X8 = unsafe fn(&[I32X8], &mut [I32X8]);

fn get_func_i32x8(t: TxfmType) -> TxfmFuncI32X8 {
  use self::TxfmType::*;
  match t {
    DCT4 => daala_fdct4,
    DCT8 => daala_fdct8,
    DCT16 => daala_fdct16,
    DCT32 => daala_fdct32,
    DCT64 => daala_fdct64,
    ADST4 => daala_fdst_vii_4,
    ADST8 => daala_fdst8,
    ADST16 => daala_fdst16,
    Identity4 => fidentity4,
    Identity8 => fidentity8,
    Identity16 => fidentity16,
    Identity32 => fidentity32,
    _ => unreachable!(),
  }
}

pub trait TxOperations: Copy {
  unsafe fn zero() -> Self;

  unsafe fn tx_mul(self, _: (i32, i32)) -> Self;
  unsafe fn rshift1(self) -> Self;
  unsafe fn add(self, b: Self) -> Self;
  unsafe fn sub(self, b: Self) -> Self;
  unsafe fn add_avg(self, b: Self) -> Self;
  unsafe fn sub_avg(self, b: Self) -> Self;

  unsafe fn copy_fn(self) -> Self {
    self
  }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[derive(Copy, Clone)]
struct I32X8 {
  data: __m256i,
}

impl I32X8 {
  #[target_feature(enable = "avx2")]
  unsafe fn vec(self) -> __m256i {
    self.data
  }

  #[target_feature(enable = "avx2")]
  unsafe fn new(a: __m256i) -> I32X8 {
    I32X8 { data: a }
  }
}

impl TxOperations for I32X8 {
  #[target_feature(enable = "avx2")]
  unsafe fn zero() -> Self {
    I32X8::new(_mm256_setzero_si256())
  }

  #[target_feature(enable = "avx2")]
  unsafe fn tx_mul(self, mul: (i32, i32)) -> Self {
    I32X8::new(_mm256_srav_epi32(
      _mm256_add_epi32(
        _mm256_mullo_epi32(self.vec(), _mm256_set1_epi32(mul.0)),
        _mm256_set1_epi32(1 << mul.1 >> 1),
      ),
      _mm256_set1_epi32(mul.1),
    ))
  }

  #[target_feature(enable = "avx2")]
  unsafe fn rshift1(self) -> Self {
    I32X8::new(_mm256_srai_epi32(
      _mm256_sub_epi32(
        self.vec(),
        _mm256_cmpgt_epi32(_mm256_setzero_si256(), self.vec()),
      ),
      1,
    ))
  }

  #[target_feature(enable = "avx2")]
  unsafe fn add(self, b: Self) -> Self {
    I32X8::new(_mm256_add_epi32(self.vec(), b.vec()))
  }

  #[target_feature(enable = "avx2")]
  unsafe fn sub(self, b: Self) -> Self {
    I32X8::new(_mm256_sub_epi32(self.vec(), b.vec()))
  }

  #[target_feature(enable = "avx2")]
  unsafe fn add_avg(self, b: Self) -> Self {
    I32X8::new(_mm256_srai_epi32(_mm256_add_epi32(self.vec(), b.vec()), 1))
  }

  #[target_feature(enable = "avx2")]
  unsafe fn sub_avg(self, b: Self) -> Self {
    I32X8::new(_mm256_srai_epi32(_mm256_sub_epi32(self.vec(), b.vec()), 1))
  }
}

impl_1d_tx!(target_feature(enable = "avx2"), unsafe);

#[target_feature(enable = "avx2")]
unsafe fn transpose_8x8_avx2(
  input: (I32X8, I32X8, I32X8, I32X8, I32X8, I32X8, I32X8, I32X8),
) -> (I32X8, I32X8, I32X8, I32X8, I32X8, I32X8, I32X8, I32X8) {
  let stage1 = (
    _mm256_unpacklo_epi32(input.0.vec(), input.1.vec()),
    _mm256_unpackhi_epi32(input.0.vec(), input.1.vec()),
    _mm256_unpacklo_epi32(input.2.vec(), input.3.vec()),
    _mm256_unpackhi_epi32(input.2.vec(), input.3.vec()),
    _mm256_unpacklo_epi32(input.4.vec(), input.5.vec()),
    _mm256_unpackhi_epi32(input.4.vec(), input.5.vec()),
    _mm256_unpacklo_epi32(input.6.vec(), input.7.vec()),
    _mm256_unpackhi_epi32(input.6.vec(), input.7.vec()),
  );

  let stage2 = (
    _mm256_unpacklo_epi64(stage1.0, stage1.2),
    _mm256_unpackhi_epi64(stage1.0, stage1.2),
    _mm256_unpacklo_epi64(stage1.1, stage1.3),
    _mm256_unpackhi_epi64(stage1.1, stage1.3),
    _mm256_unpacklo_epi64(stage1.4, stage1.6),
    _mm256_unpackhi_epi64(stage1.4, stage1.6),
    _mm256_unpacklo_epi64(stage1.5, stage1.7),
    _mm256_unpackhi_epi64(stage1.5, stage1.7),
  );

  #[allow(clippy::identity_op)]
  const LO: i32 = (2 << 4) | 0;
  const HI: i32 = (3 << 4) | 1;
  (
    I32X8::new(_mm256_permute2x128_si256(stage2.0, stage2.4, LO)),
    I32X8::new(_mm256_permute2x128_si256(stage2.1, stage2.5, LO)),
    I32X8::new(_mm256_permute2x128_si256(stage2.2, stage2.6, LO)),
    I32X8::new(_mm256_permute2x128_si256(stage2.3, stage2.7, LO)),
    I32X8::new(_mm256_permute2x128_si256(stage2.0, stage2.4, HI)),
    I32X8::new(_mm256_permute2x128_si256(stage2.1, stage2.5, HI)),
    I32X8::new(_mm256_permute2x128_si256(stage2.2, stage2.6, HI)),
    I32X8::new(_mm256_permute2x128_si256(stage2.3, stage2.7, HI)),
  )
}

#[target_feature(enable = "avx2")]
unsafe fn transpose_8x4_avx2(
  input: (I32X8, I32X8, I32X8, I32X8, I32X8, I32X8, I32X8, I32X8),
) -> (I32X8, I32X8, I32X8, I32X8) {
  // Last 8 are empty
  let stage1 = (
    //0101
    _mm256_unpacklo_epi32(input.0.vec(), input.1.vec()),
    _mm256_unpackhi_epi32(input.0.vec(), input.1.vec()),
    _mm256_unpacklo_epi32(input.2.vec(), input.3.vec()),
    _mm256_unpackhi_epi32(input.2.vec(), input.3.vec()),
    _mm256_unpacklo_epi32(input.4.vec(), input.5.vec()),
    _mm256_unpackhi_epi32(input.4.vec(), input.5.vec()),
    _mm256_unpacklo_epi32(input.6.vec(), input.7.vec()),
    _mm256_unpackhi_epi32(input.6.vec(), input.7.vec()),
  );

  let stage2 = (
    _mm256_unpacklo_epi64(stage1.0, stage1.2),
    _mm256_unpackhi_epi64(stage1.0, stage1.2),
    _mm256_unpacklo_epi64(stage1.1, stage1.3),
    _mm256_unpackhi_epi64(stage1.1, stage1.3),
    _mm256_unpacklo_epi64(stage1.4, stage1.6),
    _mm256_unpackhi_epi64(stage1.4, stage1.6),
    _mm256_unpacklo_epi64(stage1.5, stage1.7),
    _mm256_unpackhi_epi64(stage1.5, stage1.7),
  );

  #[allow(clippy::identity_op)]
  const LO: i32 = (2 << 4) | 0;
  (
    I32X8::new(_mm256_permute2x128_si256(stage2.0, stage2.4, LO)),
    I32X8::new(_mm256_permute2x128_si256(stage2.1, stage2.5, LO)),
    I32X8::new(_mm256_permute2x128_si256(stage2.2, stage2.6, LO)),
    I32X8::new(_mm256_permute2x128_si256(stage2.3, stage2.7, LO)),
  )
}

#[target_feature(enable = "avx2")]
unsafe fn transpose_4x8_avx2(
  input: (I32X8, I32X8, I32X8, I32X8),
) -> (I32X8, I32X8, I32X8, I32X8, I32X8, I32X8, I32X8, I32X8) {
  let stage1 = (
    // 0101
    _mm256_unpacklo_epi32(input.0.vec(), input.1.vec()),
    _mm256_unpackhi_epi32(input.0.vec(), input.1.vec()),
    // 2323
    _mm256_unpacklo_epi32(input.2.vec(), input.3.vec()),
    _mm256_unpackhi_epi32(input.2.vec(), input.3.vec()),
  );

  let stage2 = (
    // 01234567
    _mm256_unpacklo_epi64(stage1.0, stage1.2),
    _mm256_unpackhi_epi64(stage1.0, stage1.2),
    _mm256_unpacklo_epi64(stage1.1, stage1.3),
    _mm256_unpackhi_epi64(stage1.1, stage1.3),
  );

  (
    I32X8::new(stage2.0),
    I32X8::new(stage2.1),
    I32X8::new(stage2.2),
    I32X8::new(stage2.3),
    I32X8::new(_mm256_castsi128_si256(_mm256_extractf128_si256(stage2.0, 1))),
    I32X8::new(_mm256_castsi128_si256(_mm256_extractf128_si256(stage2.1, 1))),
    I32X8::new(_mm256_castsi128_si256(_mm256_extractf128_si256(stage2.2, 1))),
    I32X8::new(_mm256_castsi128_si256(_mm256_extractf128_si256(stage2.3, 1))),
  )
}

#[target_feature(enable = "avx2")]
unsafe fn transpose_4x4_avx2(
  input: (I32X8, I32X8, I32X8, I32X8),
) -> (I32X8, I32X8, I32X8, I32X8) {
  let stage1 = (
    _mm256_unpacklo_epi32(input.0.vec(), input.1.vec()),
    _mm256_unpackhi_epi32(input.0.vec(), input.1.vec()),
    _mm256_unpacklo_epi32(input.2.vec(), input.3.vec()),
    _mm256_unpackhi_epi32(input.2.vec(), input.3.vec()),
  );

  (
    I32X8::new(_mm256_unpacklo_epi64(stage1.0, stage1.2)),
    I32X8::new(_mm256_unpackhi_epi64(stage1.0, stage1.2)),
    I32X8::new(_mm256_unpacklo_epi64(stage1.1, stage1.3)),
    I32X8::new(_mm256_unpackhi_epi64(stage1.1, stage1.3)),
  )
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn shift_left(a: I32X8, shift: u8) -> I32X8 {
  I32X8::new(_mm256_sllv_epi32(a.vec(), _mm256_set1_epi32(shift as i32)))
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn shift_right(a: I32X8, shift: u8) -> I32X8 {
  I32X8::new(_mm256_srav_epi32(
    _mm256_add_epi32(a.vec(), _mm256_set1_epi32(1 << (shift as i32) >> 1)),
    _mm256_set1_epi32(shift as i32),
  ))
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn round_shift_array_avx2(arr: &mut [I32X8], size: usize, bit: i8) {
  if bit == 0 {
    return;
  }
  if bit > 0 {
    let shift = bit as u8;
    for i in (0..size).step_by(4) {
      let s = &mut arr[i..i + 4];
      s[0] = shift_right(s[0], shift);
      s[1] = shift_right(s[1], shift);
      s[2] = shift_right(s[2], shift);
      s[3] = shift_right(s[3], shift);
    }
  } else {
    let shift = (-bit) as u8;
    for i in (0..size).step_by(4) {
      let s = &mut arr[i..i + 4];
      s[0] = shift_left(s[0], shift);
      s[1] = shift_left(s[1], shift);
      s[2] = shift_left(s[2], shift);
      s[3] = shift_left(s[3], shift);
    }
  }
}

pub trait FwdTxfm2D: native::FwdTxfm2D {
  fn fwd_txfm2d_daala(
    input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
    bd: usize, cpu: CpuFeatureLevel,
  ) {
    if cpu >= CpuFeatureLevel::AVX2 {
      return unsafe {
        Self::fwd_txfm2d_daala_avx2(input, output, stride, tx_type, bd);
      };
    }

    <Self as native::FwdTxfm2D>::fwd_txfm2d_daala(
      input, output, stride, tx_type, bd, cpu,
    );
  }

  #[allow(clippy::identity_op, clippy::erasing_op)]
  #[target_feature(enable = "avx2")]
  unsafe fn fwd_txfm2d_daala_avx2(
    input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
    bd: usize,
  ) {
    //let mut tmp: AlignedArray<[i32; 64 * 64]> = AlignedArray::uninitialized();
    //let buf = &mut tmp.array[..Self::W * Self::H];
    let mut tmp: AlignedArray<[I32X8; 64 * 64 / 8]> =
      AlignedArray::uninitialized();
    let buf = &mut tmp.array[..Self::W * (Self::H / 8).max(1)];
    let temp_out = &mut [I32X8::zero(); 128];
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

    let txfm_func_col = get_func_i32x8(cfg.txfm_type_col);
    let txfm_func_row = get_func_i32x8(cfg.txfm_type_row);

    // Columns
    for cg in (0..txfm_size_col).step_by(8) {
      let shift = cfg.shift[0] as u8;
      #[target_feature(enable = "avx2")]
      #[inline]
      unsafe fn load_columns(input_ptr: *const i16, shift: u8) -> I32X8 {
        shift_left(
          I32X8::new(_mm256_cvtepi16_epi32(_mm_loadu_si128(
            input_ptr as *const _,
          ))),
          shift,
        )
      }
      if cfg.ud_flip {
        // flip upside down
        for r in 0..txfm_size_row {
          let input_ptr =
            input[(txfm_size_row - r - 1) * stride + cg..].as_ptr();
          temp_out[r] = load_columns(input_ptr, shift);
        }
      } else {
        // TODO: load less of x4
        for r in (0..txfm_size_row).step_by(4) {
          let output = &mut temp_out[r..r + 4];
          let input_ptr = input[r * stride + cg..].as_ptr();
          output[0] = load_columns(input_ptr, shift);
          output[1] = load_columns(input_ptr.add(stride), shift);
          output[2] = load_columns(input_ptr.add(2 * stride), shift);
          output[3] = load_columns(input_ptr.add(3 * stride), shift);
        }
      }

      let (temp_in, mut temp_out) = temp_out.split_at_mut(txfm_size_row);

      txfm_func_col(&temp_in, &mut temp_out);

      round_shift_array_avx2(&mut temp_out, txfm_size_row, -cfg.shift[1]);

      for rg in (0..txfm_size_row).step_by(8) {
        if txfm_size_row >= 8 && txfm_size_col >= 8 {
          let buf = &mut buf[(rg / 8 * txfm_size_col) + cg..];
          let buf = &mut buf[..8];
          let input = &temp_out[rg..];
          let input = &input[..8];
          let transposed = transpose_8x8_avx2((
            input[0], input[1], input[2], input[3], input[4], input[5],
            input[6], input[7],
          ));

          buf[0] = transposed.0;
          buf[1] = transposed.1;
          buf[2] = transposed.2;
          buf[3] = transposed.3;
          buf[4] = transposed.4;
          buf[5] = transposed.5;
          buf[6] = transposed.6;
          buf[7] = transposed.7;
        } else if txfm_size_row >= 8 && txfm_size_col == 4 {
          let buf = &mut buf[(rg / 8 * txfm_size_col) + cg..];
          let buf = &mut buf[..4];
          let input = &temp_out[rg..];
          let input = &input[..8];
          let transposed = transpose_8x4_avx2((
            input[0], input[1], input[2], input[3], input[4], input[5],
            input[6], input[7],
          ));

          buf[0] = transposed.0;
          buf[1] = transposed.1;
          buf[2] = transposed.2;
          buf[3] = transposed.3;
        } else if txfm_size_row == 4 && txfm_size_col >= 8 {
          let buf = &mut buf[(rg / 8 * txfm_size_col) + cg..];
          let buf = &mut buf[..8];
          let input = &temp_out[rg..];
          let input = &input[..8];
          let transposed =
            transpose_4x8_avx2((input[0], input[1], input[2], input[3]));

          buf[0] = transposed.0;
          buf[1] = transposed.1;
          buf[2] = transposed.2;
          buf[3] = transposed.3;
          buf[4] = transposed.4;
          buf[5] = transposed.5;
          buf[6] = transposed.6;
          buf[7] = transposed.7;
        } else if txfm_size_row == 4 && txfm_size_col == 4 {
          let buf = &mut buf[(rg / 8 * txfm_size_col) + cg..];
          let buf = &mut buf[..4];
          let input = &temp_out[rg..];
          let input = &input[..4];
          let transposed =
            transpose_4x4_avx2((input[0], input[1], input[2], input[3]));

          buf[0] = transposed.0;
          buf[1] = transposed.1;
          buf[2] = transposed.2;
          buf[3] = transposed.3;
        }
      }
    }

    // Rows
    for rg in (0..txfm_size_row).step_by(8) {
      if cfg.lr_flip {
        buf[rg / 8 * txfm_size_col..][..txfm_size_col].reverse();
      }
      txfm_func_row(&buf[rg / 8 * txfm_size_col..], &mut temp_out[..]);
      round_shift_array_avx2(temp_out, txfm_size_col, -cfg.shift[2]);
      for cg in (0..txfm_size_col).step_by(8) {
        if txfm_size_row >= 8 && txfm_size_col >= 8 {
          let output_ptr = output[rg * txfm_size_col + cg..].as_mut_ptr();
          let input = &temp_out[cg..];

          let transposed = transpose_8x8_avx2((
            input[0], input[1], input[2], input[3], input[4], input[5],
            input[6], input[7],
          ));

          _mm256_storeu_si256(
            output_ptr.add(0 * txfm_size_col) as *mut _,
            transposed.0.vec(),
          );
          _mm256_storeu_si256(
            output_ptr.add(1 * txfm_size_col) as *mut _,
            transposed.1.vec(),
          );
          _mm256_storeu_si256(
            output_ptr.add(2 * txfm_size_col) as *mut _,
            transposed.2.vec(),
          );
          _mm256_storeu_si256(
            output_ptr.add(3 * txfm_size_col) as *mut _,
            transposed.3.vec(),
          );
          _mm256_storeu_si256(
            output_ptr.add(4 * txfm_size_col) as *mut _,
            transposed.4.vec(),
          );
          _mm256_storeu_si256(
            output_ptr.add(5 * txfm_size_col) as *mut _,
            transposed.5.vec(),
          );
          _mm256_storeu_si256(
            output_ptr.add(6 * txfm_size_col) as *mut _,
            transposed.6.vec(),
          );
          _mm256_storeu_si256(
            output_ptr.add(7 * txfm_size_col) as *mut _,
            transposed.7.vec(),
          );
        } else if txfm_size_row == 4 && txfm_size_col >= 8 {
          let output_ptr = output[rg * txfm_size_col + cg..].as_mut_ptr();
          let input = &temp_out[cg..];

          let transposed = transpose_8x4_avx2((
            input[0], input[1], input[2], input[3], input[4], input[5],
            input[6], input[7],
          ));

          _mm256_storeu_si256(
            output_ptr.add(0 * txfm_size_col) as *mut _,
            transposed.0.vec(),
          );
          _mm256_storeu_si256(
            output_ptr.add(1 * txfm_size_col) as *mut _,
            transposed.1.vec(),
          );
          _mm256_storeu_si256(
            output_ptr.add(2 * txfm_size_col) as *mut _,
            transposed.2.vec(),
          );
          _mm256_storeu_si256(
            output_ptr.add(3 * txfm_size_col) as *mut _,
            transposed.3.vec(),
          );
        } else if txfm_size_row >= 8 && txfm_size_col == 4 {
          let output_ptr = output[rg * txfm_size_col + cg..].as_mut_ptr();
          let input = &temp_out[cg..];

          let transposed =
            transpose_4x8_avx2((input[0], input[1], input[2], input[3]));

          _mm_storeu_si128(
            output_ptr.add(0 * txfm_size_col) as *mut _,
            _mm256_castsi256_si128(transposed.0.vec()),
          );
          _mm_storeu_si128(
            output_ptr.add(1 * txfm_size_col) as *mut _,
            _mm256_castsi256_si128(transposed.1.vec()),
          );
          _mm_storeu_si128(
            output_ptr.add(2 * txfm_size_col) as *mut _,
            _mm256_castsi256_si128(transposed.2.vec()),
          );
          _mm_storeu_si128(
            output_ptr.add(3 * txfm_size_col) as *mut _,
            _mm256_castsi256_si128(transposed.3.vec()),
          );
          _mm_storeu_si128(
            output_ptr.add(4 * txfm_size_col) as *mut _,
            _mm256_castsi256_si128(transposed.4.vec()),
          );
          _mm_storeu_si128(
            output_ptr.add(5 * txfm_size_col) as *mut _,
            _mm256_castsi256_si128(transposed.5.vec()),
          );
          _mm_storeu_si128(
            output_ptr.add(6 * txfm_size_col) as *mut _,
            _mm256_castsi256_si128(transposed.6.vec()),
          );
          _mm_storeu_si128(
            output_ptr.add(7 * txfm_size_col) as *mut _,
            _mm256_castsi256_si128(transposed.7.vec()),
          );
        } else if txfm_size_row == 4 && txfm_size_col == 4 {
          let output_ptr = output[rg * txfm_size_col + cg..].as_mut_ptr();
          let input = &temp_out[cg..];

          let transposed =
            transpose_4x4_avx2((input[0], input[1], input[2], input[3]));

          _mm_storeu_si128(
            output_ptr.add(0 * txfm_size_col) as *mut _,
            _mm256_castsi256_si128(transposed.0.vec()),
          );
          _mm_storeu_si128(
            output_ptr.add(1 * txfm_size_col) as *mut _,
            _mm256_castsi256_si128(transposed.1.vec()),
          );
          _mm_storeu_si128(
            output_ptr.add(2 * txfm_size_col) as *mut _,
            _mm256_castsi256_si128(transposed.2.vec()),
          );
          _mm_storeu_si128(
            output_ptr.add(3 * txfm_size_col) as *mut _,
            _mm256_castsi256_si128(transposed.3.vec()),
          );
        }
      }
    }
  }
}

impl_fwd_txs!();
