// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::frame::*;
use crate::mc::FilterMode::*;
use crate::mc::*;
use crate::tiling::*;
use crate::util::*;

type PutFn = unsafe extern fn(
  dst: *mut u8,
  dst_stride: isize,
  src: *const u8,
  src_stride: isize,
  width: i32,
  height: i32,
  col_frac: i32,
  row_frac: i32,
);

type PrepFn = unsafe extern fn(
  tmp: *mut i16,
  src: *const u8,
  src_stride: isize,
  width: i32,
  height: i32,
  col_frac: i32,
  row_frac: i32,
);

type AvgFn = unsafe extern fn(
  dst: *mut u8,
  dst_stride: isize,
  tmp1: *const i16,
  tmp2: *const i16,
  width: i32,
  height: i32,
);

pub fn put_8tap<T: Pixel>(
  dst: &mut PlaneRegionMut<'_, T>, src: PlaneSlice<'_, T>, width: usize,
  height: usize, col_frac: i32, row_frac: i32, mode_x: FilterMode,
  mode_y: FilterMode, bit_depth: usize,
) {
  #[cfg(feature = "check_asm")]
  let ref_dst = {
    let mut copy = dst.scratch_copy();
    Put8TapNative::put_8tap(
      &mut copy.as_region_mut(),
      src,
      width,
      height,
      col_frac,
      row_frac,
      mode_x,
      mode_y,
      bit_depth,
    );
    copy
  };

  if is_x86_feature_detected!("avx2") {
    Put8TapAvx2::put_8tap(
      dst, src, width, height, col_frac, row_frac, mode_x, mode_y, bit_depth,
    )
  } else {
    Put8TapNative::put_8tap(
      dst, src, width, height, col_frac, row_frac, mode_x, mode_y, bit_depth,
    )
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

pub fn prep_8tap<T: Pixel>(
  tmp: &mut [i16], src: PlaneSlice<'_, T>, width: usize, height: usize,
  col_frac: i32, row_frac: i32, mode_x: FilterMode, mode_y: FilterMode,
  bit_depth: usize,
) {
  #[cfg(feature = "check_asm")]
  let ref_tmp = {
    let mut copy = vec![0; width * height];
    copy[..].copy_from_slice(&tmp[..width * height]);
    Prep8TapNative::prep_8tap(
      &mut copy, src, width, height, col_frac, row_frac, mode_x, mode_y,
      bit_depth,
    );
    copy
  };

  if is_x86_feature_detected!("avx2") {
    Prep8TapAvx2::prep_8tap(
      tmp, src, width, height, col_frac, row_frac, mode_x, mode_y, bit_depth,
    )
  } else {
    Prep8TapNative::prep_8tap(
      tmp, src, width, height, col_frac, row_frac, mode_x, mode_y, bit_depth,
    )
  }

  #[cfg(feature = "check_asm")]
  {
    assert_eq!(&tmp[..width * height], &ref_tmp[..]);
  }
}

pub fn mc_avg<T: Pixel>(
  dst: &mut PlaneRegionMut<'_, T>, tmp1: &[i16], tmp2: &[i16], width: usize,
  height: usize, bit_depth: usize,
) {
  #[cfg(feature = "check_asm")]
  let ref_dst = {
    let mut copy = dst.scratch_copy();
    McAvgNative::mc_avg(
      &mut copy.as_region_mut(),
      tmp1,
      tmp2,
      width,
      height,
      bit_depth,
    );
    copy
  };

  if is_x86_feature_detected!("avx2") {
    McAvgAvx2::mc_avg(dst, tmp1, tmp2, width, height, bit_depth)
  } else {
    McAvgNative::mc_avg(dst, tmp1, tmp2, width, height, bit_depth)
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

trait Put8Tap<T: Pixel> {
  fn put_8tap(
    dst: &mut PlaneRegionMut<'_, T>, src: PlaneSlice<'_, T>, width: usize,
    height: usize, col_frac: i32, row_frac: i32, mode_x: FilterMode,
    mode_y: FilterMode, bit_depth: usize,
  );
}

macro_rules! decl_mc_fns {
  ($(($mode_x:expr, $mode_y:expr, $func_name:ident)),+) => {
    extern {
      $(
        fn $func_name(
          dst: *mut u8, dst_stride: isize, src: *const u8, src_stride: isize,
          w: i32, h: i32, mx: i32, my: i32
        );
      )*
    }
  }
}

decl_mc_fns!(
  (REGULAR, REGULAR, rav1e_put_8tap_regular_avx2),
  (REGULAR, SMOOTH, rav1e_put_8tap_regular_smooth_avx2),
  (REGULAR, SHARP, rav1e_put_8tap_regular_sharp_avx2),
  (SMOOTH, REGULAR, rav1e_put_8tap_smooth_regular_avx2),
  (SMOOTH, SMOOTH, rav1e_put_8tap_smooth_avx2),
  (SMOOTH, SHARP, rav1e_put_8tap_smooth_sharp_avx2),
  (SHARP, REGULAR, rav1e_put_8tap_sharp_regular_avx2),
  (SHARP, SMOOTH, rav1e_put_8tap_sharp_smooth_avx2),
  (SHARP, SHARP, rav1e_put_8tap_sharp_avx2),
  (BILINEAR, BILINEAR, rav1e_put_bilin_avx2)
);

struct Put8TapAvx2;

impl<T: Pixel> Put8Tap<T> for Put8TapAvx2 {
  fn put_8tap(
    dst: &mut PlaneRegionMut<'_, T>, src: PlaneSlice<'_, T>, width: usize,
    height: usize, col_frac: i32, row_frac: i32, mode_x: FilterMode,
    mode_y: FilterMode, bit_depth: usize,
  ) {
    if T::type_enum() == PixelType::U8 {
      let func: Option<PutFn> = match (mode_x, mode_y) {
        (REGULAR, REGULAR) => Some(rav1e_put_8tap_regular_avx2),
        (REGULAR, SMOOTH) => Some(rav1e_put_8tap_regular_smooth_avx2),
        (REGULAR, SHARP) => Some(rav1e_put_8tap_regular_sharp_avx2),
        (SMOOTH, REGULAR) => Some(rav1e_put_8tap_smooth_regular_avx2),
        (SMOOTH, SMOOTH) => Some(rav1e_put_8tap_smooth_avx2),
        (SMOOTH, SHARP) => Some(rav1e_put_8tap_smooth_sharp_avx2),
        (SHARP, REGULAR) => Some(rav1e_put_8tap_sharp_regular_avx2),
        (SHARP, SMOOTH) => Some(rav1e_put_8tap_sharp_smooth_avx2),
        (SHARP, SHARP) => Some(rav1e_put_8tap_sharp_avx2),
        (BILINEAR, BILINEAR) => Some(rav1e_put_bilin_avx2),
        _ => None,
      };
      if let Some(func) = func {
        unsafe {
          return func(
            dst.data_ptr_mut() as *mut _,
            u8::to_asm_stride(dst.plane_cfg.stride),
            src.as_ptr() as *const _,
            u8::to_asm_stride(src.plane.cfg.stride),
            width as i32,
            height as i32,
            col_frac,
            row_frac,
          );
        }
      }
    }

    Put8TapNative::put_8tap(
      dst, src, width, height, col_frac, row_frac, mode_x, mode_y, bit_depth,
    );
  }
}

struct Put8TapNative;

impl<T: Pixel> Put8Tap<T> for Put8TapNative {
  #[inline(always)]
  fn put_8tap(
    dst: &mut PlaneRegionMut<'_, T>, src: PlaneSlice<'_, T>, width: usize,
    height: usize, col_frac: i32, row_frac: i32, mode_x: FilterMode,
    mode_y: FilterMode, bit_depth: usize,
  ) {
    native::put_8tap(
      dst, src, width, height, col_frac, row_frac, mode_x, mode_y, bit_depth,
    );
  }
}

macro_rules! decl_mct_fns {
  ($(($mode_x:expr, $mode_y:expr, $func_name:ident)),+) => {
    extern {
      $(
        fn $func_name(
          tmp: *mut i16, src: *const u8, src_stride: libc::ptrdiff_t, w: i32,
          h: i32, mx: i32, my: i32
        );
      )*
    }
  }
}

decl_mct_fns!(
  (REGULAR, REGULAR, rav1e_prep_8tap_regular_avx2),
  (REGULAR, SMOOTH, rav1e_prep_8tap_regular_smooth_avx2),
  (REGULAR, SHARP, rav1e_prep_8tap_regular_sharp_avx2),
  (SMOOTH, REGULAR, rav1e_prep_8tap_smooth_regular_avx2),
  (SMOOTH, SMOOTH, rav1e_prep_8tap_smooth_avx2),
  (SMOOTH, SHARP, rav1e_prep_8tap_smooth_sharp_avx2),
  (SHARP, REGULAR, rav1e_prep_8tap_sharp_regular_avx2),
  (SHARP, SMOOTH, rav1e_prep_8tap_sharp_smooth_avx2),
  (SHARP, SHARP, rav1e_prep_8tap_sharp_avx2),
  (BILINEAR, BILINEAR, rav1e_prep_bilin_avx2)
);

trait Prep8Tap<T: Pixel> {
  fn prep_8tap(
    tmp: &mut [i16], src: PlaneSlice<'_, T>, width: usize, height: usize,
    col_frac: i32, row_frac: i32, mode_x: FilterMode, mode_y: FilterMode,
    bit_depth: usize,
  );
}

struct Prep8TapAvx2;

impl<T: Pixel> Prep8Tap<T> for Prep8TapAvx2 {
  fn prep_8tap(
    tmp: &mut [i16], src: PlaneSlice<'_, T>, width: usize, height: usize,
    col_frac: i32, row_frac: i32, mode_x: FilterMode, mode_y: FilterMode,
    bit_depth: usize,
  ) {
    if T::type_enum() == PixelType::U8 {
      let func: Option<PrepFn> = match (mode_x, mode_y) {
        (REGULAR, REGULAR) => Some(rav1e_prep_8tap_regular_avx2),
        (REGULAR, SMOOTH) => Some(rav1e_prep_8tap_regular_smooth_avx2),
        (REGULAR, SHARP) => Some(rav1e_prep_8tap_regular_sharp_avx2),
        (SMOOTH, REGULAR) => Some(rav1e_prep_8tap_smooth_regular_avx2),
        (SMOOTH, SMOOTH) => Some(rav1e_prep_8tap_smooth_avx2),
        (SMOOTH, SHARP) => Some(rav1e_prep_8tap_smooth_sharp_avx2),
        (SHARP, REGULAR) => Some(rav1e_prep_8tap_sharp_regular_avx2),
        (SHARP, SMOOTH) => Some(rav1e_prep_8tap_sharp_smooth_avx2),
        (SHARP, SHARP) => Some(rav1e_prep_8tap_sharp_avx2),
        (BILINEAR, BILINEAR) => Some(rav1e_prep_bilin_avx2),
        _ => None,
      };
      if let Some(func) = func {
        unsafe {
          return func(
            tmp.as_mut_ptr(),
            src.as_ptr() as *const _,
            T::to_asm_stride(src.plane.cfg.stride),
            width as i32,
            height as i32,
            col_frac,
            row_frac,
          );
        }
      }
    }

    Prep8TapNative::prep_8tap(
      tmp, src, width, height, col_frac, row_frac, mode_x, mode_y, bit_depth,
    );
  }
}

struct Prep8TapNative;

impl<T: Pixel> Prep8Tap<T> for Prep8TapNative {
  #[inline(always)]
  fn prep_8tap(
    tmp: &mut [i16], src: PlaneSlice<'_, T>, width: usize, height: usize,
    col_frac: i32, row_frac: i32, mode_x: FilterMode, mode_y: FilterMode,
    bit_depth: usize,
  ) {
    native::prep_8tap(
      tmp, src, width, height, col_frac, row_frac, mode_x, mode_y, bit_depth,
    );
  }
}

extern {
  fn rav1e_avg_avx2(
    dst: *mut u8, dst_stride: libc::ptrdiff_t, tmp1: *const i16,
    tmp2: *const i16, w: i32, h: i32,
  );
}

trait McAvg<T: Pixel> {
  fn mc_avg(
    dst: &mut PlaneRegionMut<'_, T>, tmp1: &[i16], tmp2: &[i16], width: usize,
    height: usize, bit_depth: usize,
  );
}

struct McAvgAvx2;

impl<T: Pixel> McAvg<T> for McAvgAvx2 {
  fn mc_avg(
    dst: &mut PlaneRegionMut<'_, T>, tmp1: &[i16], tmp2: &[i16], width: usize,
    height: usize, bit_depth: usize,
  ) {
    if T::type_enum() == PixelType::U8 {
      unsafe {
        return rav1e_avg_avx2(
          dst.data_ptr_mut() as *mut _,
          T::to_asm_stride(dst.plane_cfg.stride),
          tmp1.as_ptr(),
          tmp2.as_ptr(),
          width as i32,
          height as i32,
        );
      }
    }

    McAvgNative::mc_avg(dst, tmp1, tmp2, width, height, bit_depth);
  }
}

struct McAvgNative;

impl<T: Pixel> McAvg<T> for McAvgNative {
  #[inline(always)]
  fn mc_avg(
    dst: &mut PlaneRegionMut<'_, T>, tmp1: &[i16], tmp2: &[i16], width: usize,
    height: usize, bit_depth: usize,
  ) {
    native::mc_avg(dst, tmp1, tmp2, width, height, bit_depth);
  }
}
