// Copyright (c) 2022, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::activity::apply_ssim_boost;
use crate::cpu_features::CpuFeatureLevel;
use crate::dist::*;
use crate::tiling::PlaneRegion;
use crate::util::Pixel;
use crate::util::PixelType;

type CdefDistKernelFn = unsafe extern fn(
  src: *const u8,
  src_stride: isize,
  dst: *const u8,
  dst_stride: isize,
  ret_ptr: *mut u32,
);

extern {
  fn rav1e_cdef_dist_kernel_4x4_sse2(
    src: *const u8, src_stride: isize, dst: *const u8, dst_stride: isize,
    ret_ptr: *mut u32,
  );
  fn rav1e_cdef_dist_kernel_4x8_sse2(
    src: *const u8, src_stride: isize, dst: *const u8, dst_stride: isize,
    ret_ptr: *mut u32,
  );
  fn rav1e_cdef_dist_kernel_8x4_sse2(
    src: *const u8, src_stride: isize, dst: *const u8, dst_stride: isize,
    ret_ptr: *mut u32,
  );
  fn rav1e_cdef_dist_kernel_8x8_sse2(
    src: *const u8, src_stride: isize, dst: *const u8, dst_stride: isize,
    ret_ptr: *mut u32,
  );
}

#[allow(clippy::let_and_return)]
pub fn cdef_dist_kernel<T: Pixel>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, w: usize, h: usize,
  bit_depth: usize, cpu: CpuFeatureLevel,
) -> u32 {
  debug_assert!(src.plane_cfg.xdec == 0);
  debug_assert!(src.plane_cfg.ydec == 0);
  debug_assert!(dst.plane_cfg.xdec == 0);
  debug_assert!(dst.plane_cfg.ydec == 0);

  // Limit kernel to 8x8
  debug_assert!(w <= 8);
  debug_assert!(h <= 8);

  let call_rust =
    || -> u32 { rust::cdef_dist_kernel(dst, src, w, h, bit_depth, cpu) };
  #[cfg(feature = "check_asm")]
  let ref_dist = call_rust();

  let mut ret_buf = [0u32; 3];
  match T::type_enum() {
    PixelType::U8 => {
      if let Some(func) =
        CDEF_DIST_KERNEL_FNS[cpu.as_index()][kernel_fn_index(w, h)]
      {
        unsafe {
          func(
            src.data_ptr() as *const _,
            T::to_asm_stride(src.plane_cfg.stride),
            dst.data_ptr() as *const _,
            T::to_asm_stride(dst.plane_cfg.stride),
            ret_buf.as_mut_ptr(),
          )
        }
      } else {
        return call_rust();
      }
    }
    PixelType::U16 => return call_rust(),
  }

  let svar = ret_buf[0];
  let dvar = ret_buf[1];
  let sse = ret_buf[2];

  let dist = apply_ssim_boost(sse, svar, dvar, bit_depth);
  #[cfg(feature = "check_asm")]
  assert_eq!(
    dist, ref_dist,
    "CDEF Distortion {}x{}: Assembly doesn't match reference code.",
    w, h
  );

  dist
}

/// Store functions in a 8x8 grid. Most will be empty.
const CDEF_DIST_KERNEL_FNS_LENGTH: usize = 8 * 8;

const fn kernel_fn_index(w: usize, h: usize) -> usize {
  ((w - 1) << 3) | (h - 1)
}

static CDEF_DIST_KERNEL_FNS_SSE2: [Option<CdefDistKernelFn>;
  CDEF_DIST_KERNEL_FNS_LENGTH] = {
  let mut out: [Option<CdefDistKernelFn>; CDEF_DIST_KERNEL_FNS_LENGTH] =
    [None; CDEF_DIST_KERNEL_FNS_LENGTH];

  out[kernel_fn_index(4, 4)] = Some(rav1e_cdef_dist_kernel_4x4_sse2);
  out[kernel_fn_index(4, 8)] = Some(rav1e_cdef_dist_kernel_4x8_sse2);
  out[kernel_fn_index(8, 4)] = Some(rav1e_cdef_dist_kernel_8x4_sse2);
  out[kernel_fn_index(8, 8)] = Some(rav1e_cdef_dist_kernel_8x8_sse2);

  out
};

cpu_function_lookup_table!(
  CDEF_DIST_KERNEL_FNS:
    [[Option<CdefDistKernelFn>; CDEF_DIST_KERNEL_FNS_LENGTH]],
  default: [None; CDEF_DIST_KERNEL_FNS_LENGTH],
  [SSE2]
);

#[cfg(test)]
pub mod test {
  use super::*;
  use crate::frame::*;
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

  /// Create planes with the max values for pixels.
  fn max_planes<T: Pixel>(bd: usize) -> (Plane<T>, Plane<T>) {
    // Two planes with different strides
    let mut input_plane = Plane::new(640, 480, 0, 0, 128 + 8, 128 + 8);
    let mut rec_plane = Plane::new(640, 480, 0, 0, 2 * 128 + 8, 2 * 128 + 8);

    for rows in input_plane.as_region_mut().rows_iter_mut() {
      for c in rows {
        *c = T::cast_from((1 << bd) - 1);
      }
    }

    for rows in rec_plane.as_region_mut().rows_iter_mut() {
      for c in rows {
        *c = T::cast_from((1 << bd) - 1);
      }
    }

    (input_plane, rec_plane)
  }

  /// Create planes with the max difference between the two values.
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

  #[test]
  fn cdef_dist_simd_random() {
    cdef_diff_tester(8, random_planes::<u8>);
  }

  #[test]
  fn cdef_dist_simd_large() {
    cdef_diff_tester(8, max_planes::<u8>);
  }

  #[test]
  fn cdef_dist_simd_large_diff() {
    cdef_diff_tester(8, max_diff_planes::<u8>);
  }

  fn cdef_diff_tester<T: Pixel>(
    bd: usize, gen_planes: fn(bd: usize) -> (Plane<T>, Plane<T>),
  ) {
    let (src_plane, dst_plane) = gen_planes(bd);

    let mut fail = false;

    for w in 1..=8 {
      for h in 1..=8 {
        // Test alignment by choosing starting location based on width.
        let area = Area::StartingAt { x: if w <= 4 { 4 } else { 8 }, y: 40 };

        let src_region = src_plane.region(area);
        let dst_region = dst_plane.region(area);

        let rust = rust::cdef_dist_kernel(
          &src_region,
          &dst_region,
          w,
          h,
          bd,
          CpuFeatureLevel::default(),
        );

        let simd = cdef_dist_kernel(
          &src_region,
          &dst_region,
          w,
          h,
          bd,
          CpuFeatureLevel::default(),
        );

        if simd != rust {
          eprintln!(
            "CDEF Distortion {}x{}: Assembly doesn't match reference code \
          \t {} (asm) != {} (ref)",
            w, h, simd, rust
          );
          fail = true;
        }
      }

      if fail {
        panic!();
      }
    }
  }
}
