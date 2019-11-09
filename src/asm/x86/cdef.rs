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

pub(crate) unsafe fn cdef_filter_block<T: Pixel>(
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

#[cfg(test)]
mod test {
  use super::*;
  use crate::frame::{AsRegion, Plane};
  use interpolate_name::interpolate_test;
  use rand::random;
  use std::str::FromStr;

  macro_rules! test_cdef_fns {
    ($(($XDEC:expr, $YDEC:expr)),*, $OPT:ident, $OPTLIT:literal) => {
      $(
        paste::item! {
          #[interpolate_test(dir_0, 0)]
          #[interpolate_test(dir_1, 1)]
          #[interpolate_test(dir_2, 2)]
          #[interpolate_test(dir_3, 3)]
          #[interpolate_test(dir_4, 4)]
          #[interpolate_test(dir_5, 5)]
          #[interpolate_test(dir_6, 6)]
          #[interpolate_test(dir_7, 7)]
          fn [<cdef_filter_block_dec_ $XDEC _ $YDEC _ $OPT>](dir: usize) {
            if !is_x86_feature_detected!($OPTLIT) {
              eprintln!("Ignoring {} test, not supported on this machine!", $OPTLIT);
              return;
            }

            let width = 8 >> $XDEC;
            let height = 8 >> $YDEC;
            let area = width * height;
            let mut src = vec![0u16; area];
            let mut dst = Plane::wrap(vec![0u8; area], width);
            for (s, d) in src.iter_mut().zip(dst.data.iter_mut()) {
              *s = random::<u8>() as u16;
              *d = random::<u8>();
            }
            let mut native_dst = dst.clone();

            let src_stride = width as isize;
            let pri_strength = 1;
            let sec_strength = 0;
            let damping = 2;
            let bit_depth = 8;

            unsafe {
              cdef_filter_block(&mut dst.as_region_mut(), src.as_ptr(), src_stride, pri_strength, sec_strength, dir, damping, bit_depth, $XDEC, $YDEC, CpuFeatureLevel::from_str($OPTLIT).unwrap());
              cdef_filter_block(&mut native_dst.as_region_mut(), src.as_ptr(), src_stride, pri_strength, sec_strength, dir, damping, bit_depth, $XDEC, $YDEC, CpuFeatureLevel::NATIVE);
              assert_eq!(native_dst.data_origin(), dst.data_origin());
            }
          }
        }
      )*
    }
  }

  test_cdef_fns!((1, 1), (1, 0), (0, 0), avx2, "avx2");
}
