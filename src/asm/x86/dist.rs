// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::dist::*;
use crate::partition::BlockSize;
use crate::tiling::*;
use crate::util::*;

type SadFn = unsafe extern fn(
  src: *const u8,
  src_stride: isize,
  dst: *const u8,
  dst_stride: isize,
) -> u32;

type SatdFn = SadFn;

type SadHBDFn = unsafe extern fn(
  src: *const u16,
  src_stride: isize,
  dst: *const u16,
  dst_stride: isize,
) -> u32;

macro_rules! declare_asm_dist_fn {
  ($(($name: ident, $T: ident)),+) => (
    $(
      extern { fn $name (
        src: *const $T, src_stride: isize, dst: *const $T, dst_stride: isize
      ) -> u32; }
    )+
  )
}

declare_asm_dist_fn![
  // SSSE3
  (rav1e_sad_4x4_hbd_ssse3, u16),
  (rav1e_sad_16x16_hbd_ssse3, u16),
  // SSE2
  (rav1e_sad4x4_sse2, u8),
  (rav1e_sad4x8_sse2, u8),
  (rav1e_sad4x16_sse2, u8),
  (rav1e_sad8x4_sse2, u8),
  (rav1e_sad8x8_sse2, u8),
  (rav1e_sad8x16_sse2, u8),
  (rav1e_sad8x32_sse2, u8),
  (rav1e_sad16x16_sse2, u8),
  (rav1e_sad32x32_sse2, u8),
  (rav1e_sad64x64_sse2, u8),
  (rav1e_sad128x128_sse2, u8),
  // AVX
  (rav1e_sad16x4_avx2, u8),
  (rav1e_sad16x8_avx2, u8),
  (rav1e_sad16x16_avx2, u8),
  (rav1e_sad16x32_avx2, u8),
  (rav1e_sad16x64_avx2, u8),
  (rav1e_sad32x8_avx2, u8),
  (rav1e_sad32x16_avx2, u8),
  (rav1e_sad32x32_avx2, u8),
  (rav1e_sad32x64_avx2, u8),
  (rav1e_sad64x16_avx2, u8),
  (rav1e_sad64x32_avx2, u8),
  (rav1e_sad64x64_avx2, u8),
  (rav1e_sad64x128_avx2, u8),
  (rav1e_sad128x64_avx2, u8),
  (rav1e_sad128x128_avx2, u8),
  (rav1e_satd_4x4_avx2, u8),
  (rav1e_satd_8x8_avx2, u8),
  (rav1e_satd_16x16_avx2, u8),
  (rav1e_satd_32x32_avx2, u8),
  (rav1e_satd_64x64_avx2, u8),
  (rav1e_satd_128x128_avx2, u8),
  (rav1e_satd_4x8_avx2, u8),
  (rav1e_satd_8x4_avx2, u8),
  (rav1e_satd_8x16_avx2, u8),
  (rav1e_satd_16x8_avx2, u8),
  (rav1e_satd_16x32_avx2, u8),
  (rav1e_satd_32x16_avx2, u8),
  (rav1e_satd_32x64_avx2, u8),
  (rav1e_satd_64x32_avx2, u8),
  (rav1e_satd_64x128_avx2, u8),
  (rav1e_satd_128x64_avx2, u8),
  (rav1e_satd_4x16_avx2, u8),
  (rav1e_satd_16x4_avx2, u8),
  (rav1e_satd_8x32_avx2, u8),
  (rav1e_satd_32x8_avx2, u8),
  (rav1e_satd_16x64_avx2, u8),
  (rav1e_satd_64x16_avx2, u8)
];

#[inline(always)]
#[allow(clippy::let_and_return)]
pub fn get_sad<T: Pixel>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, bsize: BlockSize,
  bit_depth: usize,
) -> u32 {
  #[cfg(feature = "check_asm")]
  let ref_dist = SadNative::get_sad(dst, src, bsize, bit_depth);

  let dist = if is_x86_feature_detected!("avx2") {
    SadAvx2::get_sad(src, dst, bsize, bit_depth)
  } else if is_x86_feature_detected!("ssse3") {
    SadSsse3::get_sad(src, dst, bsize, bit_depth)
  } else if is_x86_feature_detected!("sse2") {
    SadSse2::get_sad(src, dst, bsize, bit_depth)
  } else {
    SadNative::get_sad(src, dst, bsize, bit_depth)
  };

  #[cfg(feature = "check_asm")]
  assert_eq!(dist, ref_dist);

  dist
}

#[inline(always)]
#[allow(clippy::let_and_return)]
pub fn get_satd<T: Pixel>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, bsize: BlockSize,
  bit_depth: usize,
) -> u32 {
  #[cfg(feature = "check_asm")]
  let ref_dist = SatdNative::get_satd(dst, src, bsize, bit_depth);

  let dist = if is_x86_feature_detected!("avx2") {
    SatdAvx2::get_satd(src, dst, bsize, bit_depth)
  } else {
    SatdNative::get_satd(src, dst, bsize, bit_depth)
  };

  #[cfg(feature = "check_asm")]
  assert_eq!(dist, ref_dist);

  dist
}

trait Sad<T: Pixel> {
  fn get_sad(
    src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, bsize: BlockSize,
    bit_depth: usize,
  ) -> u32;
}

struct SadAvx2;

impl<T: Pixel> Sad<T> for SadAvx2 {
  fn get_sad(
    src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, bsize: BlockSize,
    bit_depth: usize,
  ) -> u32 {
    if T::type_enum() == PixelType::U8 {
      let func: Option<SadFn> = match bsize {
        BlockSize::BLOCK_4X4 => Some(rav1e_sad4x4_sse2),
        BlockSize::BLOCK_4X8 => Some(rav1e_sad4x8_sse2),
        BlockSize::BLOCK_4X16 => Some(rav1e_sad4x16_sse2),

        BlockSize::BLOCK_8X4 => Some(rav1e_sad8x4_sse2),
        BlockSize::BLOCK_8X8 => Some(rav1e_sad8x8_sse2),
        BlockSize::BLOCK_8X16 => Some(rav1e_sad8x16_sse2),
        BlockSize::BLOCK_8X32 => Some(rav1e_sad8x32_sse2),

        BlockSize::BLOCK_16X4 => Some(rav1e_sad16x4_avx2),
        BlockSize::BLOCK_16X8 => Some(rav1e_sad16x8_avx2),
        BlockSize::BLOCK_16X16 => Some(rav1e_sad16x16_avx2),
        BlockSize::BLOCK_16X32 => Some(rav1e_sad16x32_avx2),
        BlockSize::BLOCK_16X64 => Some(rav1e_sad16x64_avx2),

        BlockSize::BLOCK_32X8 => Some(rav1e_sad32x8_avx2),
        BlockSize::BLOCK_32X16 => Some(rav1e_sad32x16_avx2),
        BlockSize::BLOCK_32X32 => Some(rav1e_sad32x32_avx2),
        BlockSize::BLOCK_32X64 => Some(rav1e_sad32x64_avx2),

        BlockSize::BLOCK_64X16 => Some(rav1e_sad64x16_avx2),
        BlockSize::BLOCK_64X32 => Some(rav1e_sad64x32_avx2),
        BlockSize::BLOCK_64X64 => Some(rav1e_sad64x64_avx2),
        BlockSize::BLOCK_64X128 => Some(rav1e_sad64x128_avx2),

        BlockSize::BLOCK_128X64 => Some(rav1e_sad128x64_avx2),
        BlockSize::BLOCK_128X128 => Some(rav1e_sad128x128_avx2),
        _ => None,
      };
      if let Some(func) = func {
        unsafe {
          return func(
            src.data_ptr() as *const _,
            T::to_asm_stride(src.plane_cfg.stride),
            dst.data_ptr() as *const _,
            T::to_asm_stride(dst.plane_cfg.stride),
          );
        }
      }
    }

    SadSsse3::get_sad(src, dst, bsize, bit_depth)
  }
}

struct SadSsse3;

impl<T: Pixel> Sad<T> for SadSsse3 {
  fn get_sad(
    src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, bsize: BlockSize,
    bit_depth: usize,
  ) -> u32 {
    if T::type_enum() == PixelType::U16 {
      let func: Option<SadHBDFn> = match bsize {
        BlockSize::BLOCK_4X4 => Some(rav1e_sad_4x4_hbd_ssse3),
        BlockSize::BLOCK_16X16 => Some(rav1e_sad_16x16_hbd_ssse3),
        _ => None,
      };
      if let Some(func) = func {
        unsafe {
          return func(
            src.data_ptr() as *const _,
            T::to_asm_stride(src.plane_cfg.stride),
            dst.data_ptr() as *const _,
            T::to_asm_stride(dst.plane_cfg.stride),
          );
        }
      }
    }

    SadSse2::get_sad(src, dst, bsize, bit_depth)
  }
}

struct SadSse2;

impl<T: Pixel> Sad<T> for SadSse2 {
  fn get_sad(
    src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, bsize: BlockSize,
    bit_depth: usize,
  ) -> u32 {
    if T::type_enum() == PixelType::U8 {
      let func: Option<SadFn> = match bsize {
        BlockSize::BLOCK_4X4 => Some(rav1e_sad4x4_sse2),
        BlockSize::BLOCK_4X8 => Some(rav1e_sad4x8_sse2),
        BlockSize::BLOCK_4X16 => Some(rav1e_sad4x16_sse2),

        BlockSize::BLOCK_8X4 => Some(rav1e_sad8x4_sse2),
        BlockSize::BLOCK_8X8 => Some(rav1e_sad8x8_sse2),
        BlockSize::BLOCK_8X16 => Some(rav1e_sad8x16_sse2),
        BlockSize::BLOCK_8X32 => Some(rav1e_sad8x32_sse2),

        BlockSize::BLOCK_16X16 => Some(rav1e_sad16x16_sse2),
        BlockSize::BLOCK_32X32 => Some(rav1e_sad32x32_sse2),
        BlockSize::BLOCK_64X64 => Some(rav1e_sad64x64_sse2),
        BlockSize::BLOCK_128X128 => Some(rav1e_sad128x128_sse2),
        _ => None,
      };
      if let Some(func) = func {
        unsafe {
          return func(
            src.data_ptr() as *const _,
            T::to_asm_stride(src.plane_cfg.stride),
            dst.data_ptr() as *const _,
            T::to_asm_stride(dst.plane_cfg.stride),
          );
        }
      }
    }

    SadNative::get_sad(src, dst, bsize, bit_depth)
  }
}

struct SadNative;

impl<T: Pixel> Sad<T> for SadNative {
  #[inline(always)]
  fn get_sad(
    src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, bsize: BlockSize,
    bit_depth: usize,
  ) -> u32 {
    native::get_sad(src, dst, bsize, bit_depth)
  }
}

trait Satd<T: Pixel> {
  fn get_satd(
    src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, bsize: BlockSize,
    bit_depth: usize,
  ) -> u32;
}

struct SatdAvx2;

impl<T: Pixel> Satd<T> for SatdAvx2 {
  fn get_satd(
    src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, bsize: BlockSize,
    bit_depth: usize,
  ) -> u32 {
    if T::type_enum() == PixelType::U8 {
      let func: Option<SatdFn> = match bsize {
        BlockSize::BLOCK_4X4 => Some(rav1e_satd_4x4_avx2),
        BlockSize::BLOCK_8X8 => Some(rav1e_satd_8x8_avx2),
        BlockSize::BLOCK_16X16 => Some(rav1e_satd_16x16_avx2),
        BlockSize::BLOCK_32X32 => Some(rav1e_satd_32x32_avx2),
        BlockSize::BLOCK_64X64 => Some(rav1e_satd_64x64_avx2),
        BlockSize::BLOCK_128X128 => Some(rav1e_satd_128x128_avx2),

        BlockSize::BLOCK_4X8 => Some(rav1e_satd_4x8_avx2),
        BlockSize::BLOCK_8X4 => Some(rav1e_satd_8x4_avx2),
        BlockSize::BLOCK_8X16 => Some(rav1e_satd_8x16_avx2),
        BlockSize::BLOCK_16X8 => Some(rav1e_satd_16x8_avx2),
        BlockSize::BLOCK_16X32 => Some(rav1e_satd_16x32_avx2),
        BlockSize::BLOCK_32X16 => Some(rav1e_satd_32x16_avx2),
        BlockSize::BLOCK_32X64 => Some(rav1e_satd_32x64_avx2),
        BlockSize::BLOCK_64X32 => Some(rav1e_satd_64x32_avx2),
        BlockSize::BLOCK_64X128 => Some(rav1e_satd_64x128_avx2),
        BlockSize::BLOCK_128X64 => Some(rav1e_satd_128x64_avx2),

        BlockSize::BLOCK_4X16 => Some(rav1e_satd_4x16_avx2),
        BlockSize::BLOCK_16X4 => Some(rav1e_satd_16x4_avx2),
        BlockSize::BLOCK_8X32 => Some(rav1e_satd_8x32_avx2),
        BlockSize::BLOCK_32X8 => Some(rav1e_satd_32x8_avx2),
        BlockSize::BLOCK_16X64 => Some(rav1e_satd_16x64_avx2),
        BlockSize::BLOCK_64X16 => Some(rav1e_satd_64x16_avx2),
        _ => None,
      };
      if let Some(func) = func {
        unsafe {
          return func(
            src.data_ptr() as *const _,
            T::to_asm_stride(src.plane_cfg.stride),
            dst.data_ptr() as *const _,
            T::to_asm_stride(dst.plane_cfg.stride),
          );
        }
      }
    }

    SatdNative::get_satd(src, dst, bsize, bit_depth)
  }
}

struct SatdNative;

impl<T: Pixel> Satd<T> for SatdNative {
  #[inline(always)]
  fn get_satd(
    src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, bsize: BlockSize,
    bit_depth: usize,
  ) -> u32 {
    native::get_satd(src, dst, bsize, bit_depth)
  }
}
