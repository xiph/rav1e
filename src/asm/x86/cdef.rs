// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::cdef::*;
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

pub unsafe fn cdef_filter_block<T: Pixel>(
  dst: &mut PlaneRegionMut<'_, T>, src: *const u16, src_stride: isize,
  pri_strength: i32, sec_strength: i32, dir: usize, damping: i32,
  bit_depth: usize, xdec: usize, ydec: usize,
) {
  #[cfg(feature = "check_asm")]
  let ref_dst = {
    let mut copy = dst.scratch_copy();
    CdefFilterNative::cdef_filter_block(
      &mut copy.as_region_mut(),
      src,
      src_stride,
      pri_strength,
      sec_strength,
      dir,
      damping,
      bit_depth,
      xdec,
      ydec,
    );
    copy
  };

  if is_x86_feature_detected!("avx2") {
    CdefFilterAvx2::cdef_filter_block(
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
    );
  } else {
    CdefFilterNative::cdef_filter_block(
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
    );
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

trait CdefFilter<T: Pixel> {
  unsafe fn cdef_filter_block(
    dst: &mut PlaneRegionMut<'_, T>, src: *const u16, src_stride: isize,
    pri_strength: i32, sec_strength: i32, dir: usize, damping: i32,
    bit_depth: usize, xdec: usize, ydec: usize,
  );
}

struct CdefFilterAvx2;

impl<T: Pixel> CdefFilter<T> for CdefFilterAvx2 {
  unsafe fn cdef_filter_block(
    dst: &mut PlaneRegionMut<'_, T>, src: *const u16, src_stride: isize,
    pri_strength: i32, sec_strength: i32, dir: usize, damping: i32,
    bit_depth: usize, xdec: usize, ydec: usize,
  ) {
    if T::type_enum() == PixelType::U8 {
      let func: Option<CdefFilterFn> = match (xdec, ydec) {
        (1, 1) => Some(rav1e_cdef_filter_4x4_avx2),
        (1, 0) => Some(rav1e_cdef_filter_4x8_avx2),
        (0, 0) => Some(rav1e_cdef_filter_8x8_avx2),
        _ => None,
      };
      if let Some(func) = func {
        return func(
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
    }

    CdefFilterNative::cdef_filter_block(
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
    )
  }
}

struct CdefFilterNative;

impl<T: Pixel> CdefFilter<T> for CdefFilterNative {
  #[inline(always)]
  unsafe fn cdef_filter_block(
    dst: &mut PlaneRegionMut<'_, T>, src: *const u16, src_stride: isize,
    pri_strength: i32, sec_strength: i32, dir: usize, damping: i32,
    bit_depth: usize, xdec: usize, ydec: usize,
  ) {
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
    );
  }
}
