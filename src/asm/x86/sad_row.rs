// Copyright (c) 2021, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::cpu_features::CpuFeatureLevel;
use crate::sad_row::*;
use crate::util::{Pixel, PixelType};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::arch::asm;
use std::hint::unreachable_unchecked;
use std::mem;

/// SAFETY: src and dst must be the same length and less than 16 elements
#[inline(always)]
unsafe fn sad_scalar(src: &[u8], dst: &[u8]) -> i64 {
  if src.len() != dst.len() {
    unreachable_unchecked()
  }

  let sum = 0;

  for i in 0..src.len() {
    // We use inline assembly here to force the compiler to not auto-vectorize the loop,
    // since it is already vectorized manually.
    asm!(
      "add {sum}, {x}",
      sum = in(reg) sum,
      x = in(reg) (*src.get_unchecked(i) as i64 - *dst.get_unchecked(i) as i64).abs(),
      options(nostack)
    );
  }

  sum
}

/// SAFETY: src and dst must be the same length and less than 32 elements
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn sad_below32_8bpc_sse2(src: &[u8], dst: &[u8]) -> i64 {
  if src.len() != dst.len() {
    unreachable_unchecked()
  }

  if src.len() >= 16 {
    let src_u8x16 = _mm_loadu_si128(src.as_ptr() as *const _);
    let dst_u8x16 = _mm_loadu_si128(dst.as_ptr() as *const _);
    let result = _mm_sad_epu8(src_u8x16, dst_u8x16);
    let mut sum = mem::transmute::<_, [i64; 2]>(result).iter().sum::<i64>();

    // cannot overflow because src.len() >= 16
    let remaining = src.len() - 16;

    if remaining != 0 {
      let src_extra = src.get_unchecked(16..);
      let dst_extra = dst.get_unchecked(16..);
      sum += sad_scalar(src_extra, dst_extra);
    }

    sum
  } else {
    sad_scalar(src, dst)
  }
}

/// SAFETY: src and dst must be the same length
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn sad_8bpc_avx2(src: &[u8], dst: &[u8]) -> i64 {
  if src.len() != dst.len() {
    unreachable_unchecked()
  }

  let src_chunks = src.chunks_exact(32);
  let dst_chunks = dst.chunks_exact(32);

  let (src_rem, dst_rem) = (src_chunks.remainder(), dst_chunks.remainder());

  if src_chunks.len() == 0 {
    sad_below32_8bpc_sse2(src_rem, dst_rem)
  } else {
    let main_sum = src_chunks
      .zip(dst_chunks)
      .map(|(src_chunk, dst_chunk)| {
        let src = _mm256_loadu_si256(src_chunk.as_ptr() as *const _);
        let dst = _mm256_loadu_si256(dst_chunk.as_ptr() as *const _);

        _mm256_sad_epu8(src, dst)
      })
      .reduce(|a, b| _mm256_add_epi32(a, b))
      .unwrap_or_else(|| unreachable_unchecked());

    let mut main_sum =
      mem::transmute::<_, [i64; 4]>(main_sum).iter().sum::<i64>();

    if !src_rem.is_empty() {
      main_sum += sad_below32_8bpc_sse2(src_rem, dst_rem);
    }

    main_sum
  }
}

/// SAFETY: src and dst must be the same length
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn sad_8bpc_sse2(src: &[u8], dst: &[u8]) -> i64 {
  if src.len() != dst.len() {
    unreachable_unchecked()
  }

  let src_chunks = src.chunks_exact(16);
  let dst_chunks = dst.chunks_exact(16);

  let (src_rem, dst_rem) = (src_chunks.remainder(), dst_chunks.remainder());

  if src_chunks.len() == 0 {
    sad_scalar(src_rem, dst_rem)
  } else {
    let main_sum = src_chunks
      .zip(dst_chunks)
      .map(|(src_chunk, dst_chunk)| {
        let src = _mm_loadu_si128(src_chunk.as_ptr() as *const _);
        let dst = _mm_loadu_si128(dst_chunk.as_ptr() as *const _);

        _mm_sad_epu8(src, dst)
      })
      .reduce(|a, b| _mm_add_epi32(a, b))
      .unwrap_or_else(|| unreachable_unchecked());

    let mut main_sum =
      mem::transmute::<_, [i64; 2]>(main_sum).iter().sum::<i64>();

    if !src_rem.is_empty() {
      main_sum += sad_scalar(src_rem, dst_rem);
    }

    main_sum
  }
}

pub(crate) fn sad_row_internal<T: Pixel>(
  src: &[T], dst: &[T], cpu: CpuFeatureLevel,
) -> u64 {
  assert!(src.len() == dst.len());

  match T::type_enum() {
    PixelType::U8 => {
      // helper macro to reduce boilerplate
      macro_rules! call_asm {
        ($func:ident, $src:expr, $dst:expr, $cpu:expr) => {
          unsafe {
            let result =
              $func(mem::transmute($src), mem::transmute($dst)) as u64;

            #[cfg(feature = "check_asm")]
            assert_eq!(result, rust::sad_row_internal($src, $dst, $cpu));

            result
          }
        };
      }

      if cpu >= CpuFeatureLevel::AVX2 {
        call_asm!(sad_8bpc_avx2, src, dst, cpu)
      } else if cpu >= CpuFeatureLevel::SSE2 {
        call_asm!(sad_8bpc_sse2, src, dst, cpu)
      } else {
        rust::sad_row_internal(src, dst, cpu)
      }
    }
    PixelType::U16 => rust::sad_row_internal(src, dst, cpu),
  }
}
