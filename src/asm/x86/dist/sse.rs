// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::cpu_features::CpuFeatureLevel;
use crate::dist::*;
use crate::encoder::IMPORTANCE_BLOCK_SIZE;
use crate::partition::BlockSize;
use crate::tiling::PlaneRegion;
use crate::util::*;

type WeightedSseFn = unsafe extern fn(
  src: *const u8,
  src_stride: isize,
  dst: *const u8,
  dst_stride: isize,
  scale: *const u32,
  scale_stride: isize,
) -> u64;

type WeightedSseHBDFn = unsafe extern fn(
  src: *const u16,
  src_stride: isize,
  dst: *const u16,
  dst_stride: isize,
  scale: *const u32,
  scale_stride: isize,
) -> u64;

macro_rules! declare_asm_sse_fn {
  ($($name: ident),+) => (
    $(
      extern { fn $name (
        src: *const u8, src_stride: isize, dst: *const u8, dst_stride: isize, scale: *const u32, scale_stride: isize
      ) -> u64; }
    )+
  )
}

macro_rules! declare_asm_hbd_sse_fn {
  ($($name: ident),+) => (
    $(
      extern { fn $name (
        src: *const u16, src_stride: isize, dst: *const u16, dst_stride: isize, scale: *const u32, scale_stride: isize
      ) -> u64; }
    )+
  )
}

declare_asm_sse_fn![
  // SSSE3
  rav1e_weighted_sse_4x4_ssse3,
  rav1e_weighted_sse_4x8_ssse3,
  rav1e_weighted_sse_4x16_ssse3,
  rav1e_weighted_sse_8x4_ssse3,
  rav1e_weighted_sse_8x8_ssse3,
  rav1e_weighted_sse_8x16_ssse3,
  rav1e_weighted_sse_8x32_ssse3,
  // AVX2
  rav1e_weighted_sse_16x4_avx2,
  rav1e_weighted_sse_16x8_avx2,
  rav1e_weighted_sse_16x16_avx2,
  rav1e_weighted_sse_16x32_avx2,
  rav1e_weighted_sse_16x64_avx2,
  rav1e_weighted_sse_32x8_avx2,
  rav1e_weighted_sse_32x16_avx2,
  rav1e_weighted_sse_32x32_avx2,
  rav1e_weighted_sse_32x64_avx2,
  rav1e_weighted_sse_64x16_avx2,
  rav1e_weighted_sse_64x32_avx2,
  rav1e_weighted_sse_64x64_avx2,
  rav1e_weighted_sse_64x128_avx2,
  rav1e_weighted_sse_128x64_avx2,
  rav1e_weighted_sse_128x128_avx2
];

declare_asm_hbd_sse_fn![
  // SSE2
  rav1e_weighted_sse_4x4_hbd_sse2
];

/// # Panics
///
/// - If in `check_asm` mode, panics on mismatch between native and ASM results.
#[inline(always)]
#[allow(clippy::let_and_return)]
pub fn get_weighted_sse<T: Pixel>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, scale: &[u32],
  scale_stride: usize, w: usize, h: usize, bit_depth: usize,
  cpu: CpuFeatureLevel,
) -> u64 {
  // Assembly breaks if imp block size changes.
  assert_eq!(IMPORTANCE_BLOCK_SIZE >> 1, 4);

  let bsize_opt = BlockSize::from_width_and_height_opt(w, h);

  let call_rust = || -> u64 {
    rust::get_weighted_sse(dst, src, scale, scale_stride, w, h, bit_depth, cpu)
  };

  #[cfg(feature = "check_asm")]
  let ref_dist = call_rust();

  #[inline]
  const fn size_of_element<T: Sized>(_: &[T]) -> usize {
    std::mem::size_of::<T>()
  }

  let dist = match (bsize_opt, T::type_enum()) {
    (Err(_), _) => call_rust(),
    (Ok(bsize), PixelType::U8) => {
      match SSE_FNS[cpu.as_index()][to_index(bsize)] {
        // SAFETY: Calls Assembly code.
        Some(func) => unsafe {
          (func)(
            src.data_ptr() as *const _,
            T::to_asm_stride(src.plane_cfg.stride),
            dst.data_ptr() as *const _,
            T::to_asm_stride(dst.plane_cfg.stride),
            scale.as_ptr(),
            (scale_stride * size_of_element(scale)) as isize,
          ) as u64
        },
        None => call_rust(),
      }
    }
    (Ok(bsize), PixelType::U16) => {
      match SSE_HBD_FNS[cpu.as_index()][to_index(bsize)] {
        // SAFETY: Calls Assembly code.
        Some(func) => unsafe {
          (func)(
            src.data_ptr() as *const _,
            T::to_asm_stride(src.plane_cfg.stride) as isize,
            dst.data_ptr() as *const _,
            T::to_asm_stride(dst.plane_cfg.stride) as isize,
            scale.as_ptr(),
            (scale_stride * size_of_element(scale)) as isize,
          )
        },
        None => call_rust(),
      }
    }
  };

  #[cfg(feature = "check_asm")]
  assert_eq!(
    dist, ref_dist,
    "Weighted SSE {:?}: Assembly doesn't match reference code.",
    bsize_opt
  );

  dist
}

static SSE_FNS_SSSE3: [Option<WeightedSseFn>; DIST_FNS_LENGTH] = {
  let mut out: [Option<WeightedSseFn>; DIST_FNS_LENGTH] =
    [None; DIST_FNS_LENGTH];

  use BlockSize::*;
  out[BLOCK_4X4 as usize] = Some(rav1e_weighted_sse_4x4_ssse3);
  out[BLOCK_4X8 as usize] = Some(rav1e_weighted_sse_4x8_ssse3);
  out[BLOCK_4X16 as usize] = Some(rav1e_weighted_sse_4x16_ssse3);
  out[BLOCK_8X4 as usize] = Some(rav1e_weighted_sse_8x4_ssse3);
  out[BLOCK_8X8 as usize] = Some(rav1e_weighted_sse_8x8_ssse3);
  out[BLOCK_8X16 as usize] = Some(rav1e_weighted_sse_8x16_ssse3);
  out[BLOCK_8X32 as usize] = Some(rav1e_weighted_sse_8x32_ssse3);

  out
};

static SSE_FNS_AVX2: [Option<WeightedSseFn>; DIST_FNS_LENGTH] = {
  let mut out: [Option<WeightedSseFn>; DIST_FNS_LENGTH] = SSE_FNS_SSSE3;

  use BlockSize::*;
  out[BLOCK_16X4 as usize] = Some(rav1e_weighted_sse_16x4_avx2);
  out[BLOCK_16X8 as usize] = Some(rav1e_weighted_sse_16x8_avx2);
  out[BLOCK_16X16 as usize] = Some(rav1e_weighted_sse_16x16_avx2);
  out[BLOCK_16X32 as usize] = Some(rav1e_weighted_sse_16x32_avx2);
  out[BLOCK_16X64 as usize] = Some(rav1e_weighted_sse_16x64_avx2);
  out[BLOCK_32X8 as usize] = Some(rav1e_weighted_sse_32x8_avx2);
  out[BLOCK_32X16 as usize] = Some(rav1e_weighted_sse_32x16_avx2);
  out[BLOCK_32X32 as usize] = Some(rav1e_weighted_sse_32x32_avx2);
  out[BLOCK_32X64 as usize] = Some(rav1e_weighted_sse_32x64_avx2);
  out[BLOCK_64X16 as usize] = Some(rav1e_weighted_sse_64x16_avx2);
  out[BLOCK_64X32 as usize] = Some(rav1e_weighted_sse_64x32_avx2);
  out[BLOCK_64X64 as usize] = Some(rav1e_weighted_sse_64x64_avx2);
  out[BLOCK_64X128 as usize] = Some(rav1e_weighted_sse_64x128_avx2);
  out[BLOCK_128X64 as usize] = Some(rav1e_weighted_sse_128x64_avx2);
  out[BLOCK_128X128 as usize] = Some(rav1e_weighted_sse_128x128_avx2);

  out
};

static SSE_HBD_FNS_SSE2: [Option<WeightedSseHBDFn>; DIST_FNS_LENGTH] = {
  let mut out: [Option<WeightedSseHBDFn>; DIST_FNS_LENGTH] =
    [None; DIST_FNS_LENGTH];

  use BlockSize::*;
  out[BLOCK_4X4 as usize] = Some(rav1e_weighted_sse_4x4_hbd_sse2);

  out
};

cpu_function_lookup_table!(
  SSE_FNS: [[Option<WeightedSseFn>; DIST_FNS_LENGTH]],
  default: [None; DIST_FNS_LENGTH],
  [SSSE3, AVX2]
);

cpu_function_lookup_table!(
  SSE_HBD_FNS: [[Option<WeightedSseHBDFn>; DIST_FNS_LENGTH]],
  default: [None; DIST_FNS_LENGTH],
  [SSE2]
);

#[cfg(test)]
pub mod test {
  use super::*;
  use crate::frame::*;
  use crate::rdo::DistortionScale;
  use crate::tiling::Area;
  use rand::{thread_rng, Rng};

  fn random_planes<T: Pixel>(bd: usize) -> (Plane<T>, Plane<T>) {
    let mut rng = thread_rng();

    // Two planes with different strides
    let mut input_plane = Plane::new(640, 480, 0, 0, 128 + 8, 128 + 8);
    let mut rec_plane = Plane::new(640, 480, 0, 0, 2 * 128 + 8, 2 * 128 + 8);

    for rows in input_plane.as_region_mut().rows_iter_mut() {
      for c in rows {
        *c = T::cast_from(rng.gen_range(0u16..(1 << bd)));
      }
    }

    for rows in rec_plane.as_region_mut().rows_iter_mut() {
      for c in rows {
        *c = T::cast_from(rng.gen_range(0u16..(1 << bd)));
      }
    }

    (input_plane, rec_plane)
  }

  // Create planes with the max difference between the two values.
  fn max_diff_planes<T: Pixel>(bd: usize) -> (Plane<T>, Plane<T>) {
    // Two planes with different strides
    let mut input_plane = Plane::new(640, 480, 0, 0, 128 + 8, 128 + 8);
    let mut rec_plane = Plane::new(640, 480, 0, 0, 2 * 128 + 8, 2 * 128 + 8);

    for rows in input_plane.as_region_mut().rows_iter_mut() {
      for c in rows {
        *c = T::cast_from(0);
      }
    }

    for rows in rec_plane.as_region_mut().rows_iter_mut() {
      for c in rows {
        *c = T::cast_from((1 << bd) - 1);
      }
    }

    (input_plane, rec_plane)
  }

  /// Fill data for scaling of one (i.e. no scaling between blocks)
  fn scaling_one(scales: &mut [u32]) {
    for a in scales.iter_mut() {
      *a = DistortionScale::default().0;
    }
  }

  /// Fill data for scaling of one
  fn scaling_random(scales: &mut [u32]) {
    let mut rng = thread_rng();
    for a in scales.iter_mut() {
      *a = rng
        .gen_range(DistortionScale::from(0.5).0..DistortionScale::from(1.5).0);
    }
  }

  /// Fill the max value for scaling
  /// TODO: Pair with max difference test
  fn scaling_large(scales: &mut [u32]) {
    for a in scales.iter_mut() {
      *a = DistortionScale::from(f64::MAX).0;
    }
  }

  #[test]
  fn weighted_sse_simd_no_scaling() {
    weighted_sse_simd_tester(8, scaling_one, random_planes::<u8>);
  }

  #[test]
  fn weighted_sse_simd_random() {
    weighted_sse_simd_tester(8, scaling_random, random_planes::<u8>);
  }

  #[test]
  fn weighted_sse_simd_large() {
    weighted_sse_simd_tester(8, scaling_large, max_diff_planes::<u8>);
  }

  #[test]
  fn weighted_sse_hbd_simd_no_scaling() {
    weighted_sse_simd_tester(12, scaling_one, random_planes::<u16>);
  }

  #[test]
  fn weighted_sse_hbd_simd_random() {
    weighted_sse_simd_tester(12, scaling_random, random_planes::<u16>);
  }

  #[test]
  fn weighted_sse_hbd_simd_large() {
    weighted_sse_simd_tester(12, scaling_large, max_diff_planes::<u16>);
  }

  fn weighted_sse_simd_tester<T: Pixel>(
    bd: usize, fill_scales: fn(scales: &mut [u32]),
    gen_planes: fn(bd: usize) -> (Plane<T>, Plane<T>),
  ) {
    use BlockSize::*;
    let blocks = vec![
      BLOCK_4X4,
      BLOCK_4X8,
      BLOCK_8X4,
      BLOCK_8X8,
      BLOCK_8X16,
      BLOCK_16X8,
      BLOCK_16X16,
      BLOCK_16X32,
      BLOCK_32X16,
      BLOCK_32X32,
      BLOCK_32X64,
      BLOCK_64X32,
      BLOCK_64X64,
      BLOCK_64X128,
      BLOCK_128X64,
      BLOCK_128X128,
      BLOCK_4X16,
      BLOCK_16X4,
      BLOCK_8X32,
      BLOCK_32X8,
      BLOCK_16X64,
      BLOCK_64X16,
    ];

    const SCALE_STRIDE: usize = 256;
    let mut scaling_storage = Aligned::new([0u32; 256 * SCALE_STRIDE]);
    let scaling = &mut scaling_storage.data;
    fill_scales(scaling);

    let (input_plane, rec_plane) = gen_planes(bd);

    let mut fail = false;

    for block in blocks {
      // Start at block width to test alignment.
      let area = Area::StartingAt { x: block.width() as isize, y: 40 };

      let input_region = input_plane.region(area);
      let rec_region = rec_plane.region(area);

      let rust = rust::get_weighted_sse(
        &input_region,
        &rec_region,
        scaling,
        SCALE_STRIDE,
        block.width(),
        block.height(),
        bd,
        CpuFeatureLevel::default(),
      );

      let simd = get_weighted_sse(
        &input_region,
        &rec_region,
        scaling,
        SCALE_STRIDE,
        block.width(),
        block.height(),
        bd,
        CpuFeatureLevel::default(),
      );

      if simd != rust {
        eprintln!(
          "Weighted SSE {}: Assembly doesn't match reference code \
          \t {} (asm) != {} (ref)",
          block, simd, rust
        );
        fail = true;
      }
    }

    if fail {
      panic!();
    }
  }
}
