// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::cdef::*;
use crate::cpu_features::CpuFeatureLevel;
#[cfg(feature = "check_asm")]
use crate::frame::*;
use crate::tiling::PlaneRegionMut;
use crate::util::*;

type CdefFilterFn = unsafe extern fn(
  dst: *mut u8,
  dst_stride: isize,
  tmp: *const u16,
  tmp_stride: isize,
  pri_strength: i32,
  sec_strength: i32,
  dir: i32,
  damping: i32,
);

type CdefFilterHBDFn = unsafe extern fn(
  dst: *mut u8,
  dst_stride: isize,
  tmp: *const u16,
  tmp_stride: isize,
  pri_strength: i32,
  sec_strength: i32,
  dir: i32,
  damping: i32,
  bit_depth: i32,
);

#[inline(always)]
const fn decimate_index(xdec: usize, ydec: usize) -> usize {
  ((ydec << 1) | xdec) & 3
}

pub unsafe fn cdef_filter_block<T: Pixel>(
  dst: &mut PlaneRegionMut<'_, T>, src: *const u16, src_stride: isize,
  pri_strength: i32, sec_strength: i32, dir: usize, damping: i32,
  bit_depth: usize, xdec: usize, ydec: usize, cpu: CpuFeatureLevel,
) {
  let call_native = |dst: &mut PlaneRegionMut<T>| {
    native::cdef_filter_block(
      dst,
      src,
      src_stride,
      pri_strength,
      sec_strength,
      dir,
      damping,
      bit_depth,
      xdec,
      ydec,
      cpu,
    );
  };
  #[cfg(feature = "check_asm")]
  let ref_dst = {
    let mut copy = dst.scratch_copy();
    call_native(&mut copy.as_region_mut());
    copy
  };
  match T::type_enum() {
    PixelType::U8 => {
      match CDEF_FILTER_FNS[cpu.as_index()][decimate_index(xdec, ydec)] {
        Some(func) => {
          (func)(
            dst.data_ptr_mut() as *mut _,
            T::to_asm_stride(dst.plane_cfg.stride),
            src,
            src_stride,
            pri_strength,
            sec_strength,
            dir as i32,
            damping,
          );
        }
        None => call_native(dst),
      }
    }
    PixelType::U16 => {
      match CDEF_FILTER_HBD_FNS[cpu.as_index()][decimate_index(xdec, ydec)] {
        Some(func) => {
          (func)(
            dst.data_ptr_mut() as *mut _,
            T::to_asm_stride(dst.plane_cfg.stride),
            src,
            src_stride,
            pri_strength,
            sec_strength,
            dir as i32,
            damping,
            bit_depth as i32,
          );
        }
        None => call_native(dst),
      }
    }
  }
  #[cfg(feature = "check_asm")]
  {
    for (dst_row, ref_row) in
      dst.rows_iter().zip(ref_dst.as_region().rows_iter())
    {
      for (dst, reference) in dst_row.iter().zip(ref_row) {
        assert_eq!(*dst, *reference);
      }
    }
  }
}

extern {
  fn rav1e_cdef_filter_4x4_avx2(
    dst: *mut u8, dst_stride: isize, tmp: *const u16, tmp_stride: isize,
    pri_strength: i32, sec_strength: i32, dir: i32, damping: i32,
  );

  fn rav1e_cdef_filter_4x8_avx2(
    dst: *mut u8, dst_stride: isize, tmp: *const u16, tmp_stride: isize,
    pri_strength: i32, sec_strength: i32, dir: i32, damping: i32,
  );

  fn rav1e_cdef_filter_8x8_avx2(
    dst: *mut u8, dst_stride: isize, tmp: *const u16, tmp_stride: isize,
    pri_strength: i32, sec_strength: i32, dir: i32, damping: i32,
  );
}

static CDEF_FILTER_FNS_AVX2: [Option<CdefFilterFn>; 4] = {
  let mut out: [Option<CdefFilterFn>; 4] = [None; 4];
  out[decimate_index(1, 1)] = Some(rav1e_cdef_filter_4x4_avx2);
  out[decimate_index(1, 0)] = Some(rav1e_cdef_filter_4x8_avx2);
  out[decimate_index(0, 0)] = Some(rav1e_cdef_filter_8x8_avx2);
  out
};

pub(crate) static CDEF_FILTER_FNS: [[Option<CdefFilterFn>; 4];
  CpuFeatureLevel::len()] = {
  let mut out: [[Option<CdefFilterFn>; 4]; CpuFeatureLevel::len()] =
    [[None; 4]; CpuFeatureLevel::len()];
  out[CpuFeatureLevel::AVX2 as usize] = CDEF_FILTER_FNS_AVX2;
  out
};

pub(crate) static CDEF_FILTER_HBD_FNS: [[Option<CdefFilterHBDFn>; 4];
  CpuFeatureLevel::len()] = [[None; 4]; CpuFeatureLevel::len()];
