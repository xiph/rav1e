// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::cpu_features::CpuFeatureLevel;
use crate::dist::*;
use crate::partition::BlockSize;
use crate::tiling::*;
use crate::util::*;

#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
use x86_64::*;

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

type SatdHBDFn = SadHBDFn;

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

// BlockSize::BLOCK_SIZES.next_power_of_two();
const DIST_FNS_LENGTH: usize = 32;

fn to_index(bsize: BlockSize) -> usize {
  bsize as usize & (DIST_FNS_LENGTH - 1)
}

#[inline(always)]
#[allow(clippy::let_and_return)]
pub fn get_sad<T: Pixel>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, bsize: BlockSize,
  bit_depth: usize, cpu: CpuFeatureLevel,
) -> u32 {
  let call_native =
    || -> u32 { native::get_sad(dst, src, bsize, bit_depth, cpu) };

  #[cfg(feature = "check_asm")]
  let ref_dist = call_native();

  let dist = match T::type_enum() {
    PixelType::U8 => match SAD_FNS[cpu.as_index()][to_index(bsize)] {
      Some(func) => unsafe {
        (func)(
          src.data_ptr() as *const _,
          T::to_asm_stride(src.plane_cfg.stride),
          dst.data_ptr() as *const _,
          T::to_asm_stride(dst.plane_cfg.stride),
        )
      },
      None => call_native(),
    },
    PixelType::U16 => match SAD_HBD_FNS[cpu.as_index()][to_index(bsize)] {
      Some(func) => unsafe {
        (func)(
          src.data_ptr() as *const _,
          T::to_asm_stride(src.plane_cfg.stride),
          dst.data_ptr() as *const _,
          T::to_asm_stride(dst.plane_cfg.stride),
        )
      },
      None => call_native(),
    },
  };

  #[cfg(feature = "check_asm")]
  assert_eq!(dist, ref_dist);

  dist
}

#[inline(always)]
#[allow(clippy::let_and_return)]
pub fn get_satd<T: Pixel>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, bsize: BlockSize,
  bit_depth: usize, cpu: CpuFeatureLevel,
) -> u32 {
  let call_native =
    || -> u32 { native::get_satd(dst, src, bsize, bit_depth, cpu) };

  #[cfg(feature = "check_asm")]
  let ref_dist = call_native();

  let dist = match T::type_enum() {
    PixelType::U8 => match SATD_FNS[cpu.as_index()][to_index(bsize)] {
      Some(func) => unsafe {
        (func)(
          src.data_ptr() as *const _,
          T::to_asm_stride(src.plane_cfg.stride),
          dst.data_ptr() as *const _,
          T::to_asm_stride(dst.plane_cfg.stride),
        )
      },
      None => call_native(),
    },
    PixelType::U16 => match SATD_HBD_FNS[cpu.as_index()][to_index(bsize)] {
      Some(func) => unsafe {
        (func)(
          src.data_ptr() as *const _,
          T::to_asm_stride(src.plane_cfg.stride),
          dst.data_ptr() as *const _,
          T::to_asm_stride(dst.plane_cfg.stride),
        )
      },
      None => call_native(),
    },
  };

  #[cfg(feature = "check_asm")]
  assert_eq!(dist, ref_dist);

  dist
}

mod x86_64 {
  use super::*;

  static SAD_FNS_SSE2: [Option<SadFn>; DIST_FNS_LENGTH] = {
    let mut out: [Option<SadFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

    use BlockSize::*;

    out[BLOCK_4X4 as usize] = Some(rav1e_sad4x4_sse2);
    out[BLOCK_4X8 as usize] = Some(rav1e_sad4x8_sse2);
    out[BLOCK_4X16 as usize] = Some(rav1e_sad4x16_sse2);

    out[BLOCK_8X4 as usize] = Some(rav1e_sad8x4_sse2);
    out[BLOCK_8X8 as usize] = Some(rav1e_sad8x8_sse2);
    out[BLOCK_8X16 as usize] = Some(rav1e_sad8x16_sse2);
    out[BLOCK_8X32 as usize] = Some(rav1e_sad8x32_sse2);

    out[BLOCK_16X16 as usize] = Some(rav1e_sad16x16_sse2);
    out[BLOCK_32X32 as usize] = Some(rav1e_sad32x32_sse2);
    out[BLOCK_64X64 as usize] = Some(rav1e_sad64x64_sse2);
    out[BLOCK_128X128 as usize] = Some(rav1e_sad128x128_sse2);

    out
  };

  static SAD_FNS_AVX2: [Option<SadFn>; DIST_FNS_LENGTH] = {
    let mut out: [Option<SadFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

    use BlockSize::*;

    out[BLOCK_4X4 as usize] = Some(rav1e_sad4x4_sse2);
    out[BLOCK_4X8 as usize] = Some(rav1e_sad4x8_sse2);
    out[BLOCK_4X16 as usize] = Some(rav1e_sad4x16_sse2);

    out[BLOCK_8X4 as usize] = Some(rav1e_sad8x4_sse2);
    out[BLOCK_8X8 as usize] = Some(rav1e_sad8x8_sse2);
    out[BLOCK_8X16 as usize] = Some(rav1e_sad8x16_sse2);
    out[BLOCK_8X32 as usize] = Some(rav1e_sad8x32_sse2);

    out[BLOCK_16X4 as usize] = Some(rav1e_sad16x4_avx2);
    out[BLOCK_16X8 as usize] = Some(rav1e_sad16x8_avx2);
    out[BLOCK_16X16 as usize] = Some(rav1e_sad16x16_avx2);
    out[BLOCK_16X32 as usize] = Some(rav1e_sad16x32_avx2);
    out[BLOCK_16X64 as usize] = Some(rav1e_sad16x64_avx2);

    out[BLOCK_32X8 as usize] = Some(rav1e_sad32x8_avx2);
    out[BLOCK_32X16 as usize] = Some(rav1e_sad32x16_avx2);
    out[BLOCK_32X32 as usize] = Some(rav1e_sad32x32_avx2);
    out[BLOCK_32X64 as usize] = Some(rav1e_sad32x64_avx2);

    out[BLOCK_64X16 as usize] = Some(rav1e_sad64x16_avx2);
    out[BLOCK_64X32 as usize] = Some(rav1e_sad64x32_avx2);
    out[BLOCK_64X64 as usize] = Some(rav1e_sad64x64_avx2);
    out[BLOCK_64X128 as usize] = Some(rav1e_sad64x128_avx2);

    out[BLOCK_128X64 as usize] = Some(rav1e_sad128x64_avx2);
    out[BLOCK_128X128 as usize] = Some(rav1e_sad128x128_avx2);

    out
  };

  pub static SAD_FNS: [[Option<SadFn>; DIST_FNS_LENGTH];
    CpuFeatureLevel::len()] = {
    let mut out = [[None; DIST_FNS_LENGTH]; CpuFeatureLevel::len()];

    out[CpuFeatureLevel::SSE2 as usize] = SAD_FNS_SSE2;
    out[CpuFeatureLevel::SSSE3 as usize] = SAD_FNS_SSE2;
    out[CpuFeatureLevel::AVX2 as usize] = SAD_FNS_AVX2;

    out
  };

  static SAD_HBD_FNS_SSSE3: [Option<SadHBDFn>; DIST_FNS_LENGTH] = {
    let mut out: [Option<SadHBDFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

    use BlockSize::*;

    out[BLOCK_4X4 as usize] = Some(rav1e_sad_4x4_hbd_ssse3);
    out[BLOCK_16X16 as usize] = Some(rav1e_sad_16x16_hbd_ssse3);

    out
  };

  pub(crate) static SAD_HBD_FNS: [[Option<SadHBDFn>; DIST_FNS_LENGTH];
    CpuFeatureLevel::len()] = {
    let mut out = [[None; DIST_FNS_LENGTH]; CpuFeatureLevel::len()];

    out[CpuFeatureLevel::SSE2 as usize] = SAD_HBD_FNS_SSSE3;
    out[CpuFeatureLevel::SSSE3 as usize] = SAD_HBD_FNS_SSSE3;
    out[CpuFeatureLevel::AVX2 as usize] = SAD_HBD_FNS_SSSE3;

    out
  };

  static SATD_FNS_AVX2: [Option<SatdFn>; DIST_FNS_LENGTH] = {
    let mut out: [Option<SatdFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

    use BlockSize::*;

    out[BLOCK_4X4 as usize] = Some(rav1e_satd_4x4_avx2);
    out[BLOCK_8X8 as usize] = Some(rav1e_satd_8x8_avx2);
    out[BLOCK_16X16 as usize] = Some(rav1e_satd_16x16_avx2);
    out[BLOCK_32X32 as usize] = Some(rav1e_satd_32x32_avx2);
    out[BLOCK_64X64 as usize] = Some(rav1e_satd_64x64_avx2);
    out[BLOCK_128X128 as usize] = Some(rav1e_satd_128x128_avx2);

    out[BLOCK_4X8 as usize] = Some(rav1e_satd_4x8_avx2);
    out[BLOCK_8X4 as usize] = Some(rav1e_satd_8x4_avx2);
    out[BLOCK_8X16 as usize] = Some(rav1e_satd_8x16_avx2);
    out[BLOCK_16X8 as usize] = Some(rav1e_satd_16x8_avx2);
    out[BLOCK_16X32 as usize] = Some(rav1e_satd_16x32_avx2);
    out[BLOCK_32X16 as usize] = Some(rav1e_satd_32x16_avx2);
    out[BLOCK_32X64 as usize] = Some(rav1e_satd_32x64_avx2);
    out[BLOCK_64X32 as usize] = Some(rav1e_satd_64x32_avx2);
    out[BLOCK_64X128 as usize] = Some(rav1e_satd_64x128_avx2);
    out[BLOCK_128X64 as usize] = Some(rav1e_satd_128x64_avx2);

    out[BLOCK_4X16 as usize] = Some(rav1e_satd_4x16_avx2);
    out[BLOCK_16X4 as usize] = Some(rav1e_satd_16x4_avx2);
    out[BLOCK_8X32 as usize] = Some(rav1e_satd_8x32_avx2);
    out[BLOCK_32X8 as usize] = Some(rav1e_satd_32x8_avx2);
    out[BLOCK_16X64 as usize] = Some(rav1e_satd_16x64_avx2);
    out[BLOCK_64X16 as usize] = Some(rav1e_satd_64x16_avx2);

    out
  };

  pub(crate) static SATD_FNS: [[Option<SatdFn>; DIST_FNS_LENGTH];
    CpuFeatureLevel::len()] = {
    let mut out = [[None; DIST_FNS_LENGTH]; CpuFeatureLevel::len()];

    out[CpuFeatureLevel::AVX2 as usize] = SATD_FNS_AVX2;

    out
  };

  pub(crate) static SATD_HBD_FNS: [[Option<SatdHBDFn>; DIST_FNS_LENGTH];
    CpuFeatureLevel::len()] =
    [[None; DIST_FNS_LENGTH]; CpuFeatureLevel::len()];
}
