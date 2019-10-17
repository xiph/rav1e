// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::cpu_features::CpuFeatureLevel;
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

type PutHBDFn = unsafe extern fn(
  dst: *mut u8,
  dst_stride: isize,
  src: *const u8,
  src_stride: isize,
  width: i32,
  height: i32,
  col_frac: i32,
  row_frac: i32,
  bit_depth: i32,
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

type PrepHBDFn = unsafe extern fn(
  tmp: *mut i16,
  src: *const u16,
  src_stride: isize,
  width: i32,
  height: i32,
  col_frac: i32,
  row_frac: i32,
  bit_depth: i32,
);

type AvgFn = unsafe extern fn(
  dst: *mut u8,
  dst_stride: isize,
  tmp1: *const i16,
  tmp2: *const i16,
  width: i32,
  height: i32,
);

type AvgHBDFn = unsafe extern fn(
  dst: *mut u16,
  dst_stride: isize,
  tmp1: *const i16,
  tmp2: *const i16,
  width: i32,
  height: i32,
  bit_depth: i32,
);

// gets an index that can be mapped to a function for a pair of filter modes
const fn get_2d_mode_idx(mode_x: FilterMode, mode_y: FilterMode) -> usize {
  (mode_x as usize + 4 * (mode_y as usize)) & 15
}

pub fn put_8tap<T: Pixel>(
  dst: &mut PlaneRegionMut<'_, T>, src: PlaneSlice<'_, T>, width: usize,
  height: usize, col_frac: i32, row_frac: i32, mode_x: FilterMode,
  mode_y: FilterMode, bit_depth: usize, cpu: CpuFeatureLevel,
) {
  let call_native = |dst: &mut PlaneRegionMut<'_, T>| {
    native::put_8tap(
      dst, src, width, height, col_frac, row_frac, mode_x, mode_y, bit_depth,
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
      match PUT_FNS[cpu.as_index()][get_2d_mode_idx(mode_x, mode_y)] {
        Some(func) => unsafe {
          (func)(
            dst.data_ptr_mut() as *mut _,
            T::to_asm_stride(dst.plane_cfg.stride),
            src.as_ptr() as *const _,
            T::to_asm_stride(src.plane.cfg.stride),
            width as i32,
            height as i32,
            col_frac,
            row_frac,
          );
        },
        None => call_native(dst),
      }
    }
    PixelType::U16 => {
      match PUT_HBD_FNS[cpu.as_index()][get_2d_mode_idx(mode_x, mode_y)] {
        Some(func) => unsafe {
          (func)(
            dst.data_ptr_mut() as *mut _,
            T::to_asm_stride(dst.plane_cfg.stride),
            src.as_ptr() as *const _,
            T::to_asm_stride(src.plane.cfg.stride),
            width as i32,
            height as i32,
            col_frac,
            row_frac,
            bit_depth as i32,
          );
        },
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

pub fn prep_8tap<T: Pixel>(
  tmp: &mut [i16], src: PlaneSlice<'_, T>, width: usize, height: usize,
  col_frac: i32, row_frac: i32, mode_x: FilterMode, mode_y: FilterMode,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  let call_native = |tmp: &mut [i16]| {
    native::prep_8tap(
      tmp, src, width, height, col_frac, row_frac, mode_x, mode_y, bit_depth,
      cpu,
    );
  };
  #[cfg(feature = "check_asm")]
  let ref_tmp = {
    let mut copy = vec![0; width * height];
    copy[..].copy_from_slice(&tmp[..width * height]);
    call_native(&mut copy);
    copy
  };
  match T::type_enum() {
    PixelType::U8 => {
      match PREP_FNS[cpu.as_index()][get_2d_mode_idx(mode_x, mode_y)] {
        Some(func) => unsafe {
          (func)(
            tmp.as_mut_ptr(),
            src.as_ptr() as *const _,
            T::to_asm_stride(src.plane.cfg.stride),
            width as i32,
            height as i32,
            col_frac,
            row_frac,
          );
        },
        None => call_native(tmp),
      }
    }
    PixelType::U16 => {
      match PREP_HBD_FNS[cpu.as_index()][get_2d_mode_idx(mode_x, mode_y)] {
        Some(func) => unsafe {
          (func)(
            tmp.as_mut_ptr() as *mut _,
            src.as_ptr() as *const _,
            T::to_asm_stride(src.plane.cfg.stride),
            width as i32,
            height as i32,
            col_frac,
            row_frac,
            bit_depth as i32,
          );
        },
        None => call_native(tmp),
      }
    }
  }
  #[cfg(feature = "check_asm")]
  {
    assert_eq!(&tmp[..width * height], &ref_tmp[..]);
  }
}

pub fn mc_avg<T: Pixel>(
  dst: &mut PlaneRegionMut<'_, T>, tmp1: &[i16], tmp2: &[i16], width: usize,
  height: usize, bit_depth: usize, cpu: CpuFeatureLevel,
) {
  let call_native = |dst: &mut PlaneRegionMut<'_, T>| {
    native::mc_avg(dst, tmp1, tmp2, width, height, bit_depth, cpu);
  };
  #[cfg(feature = "check_asm")]
  let ref_dst = {
    let mut copy = dst.scratch_copy();
    call_native(&mut copy.as_region_mut());
    copy
  };
  match T::type_enum() {
    PixelType::U8 => match AVG_FNS[cpu.as_index()] {
      Some(func) => unsafe {
        (func)(
          dst.data_ptr_mut() as *mut _,
          T::to_asm_stride(dst.plane_cfg.stride),
          tmp1.as_ptr(),
          tmp2.as_ptr(),
          width as i32,
          height as i32,
        );
      },
      None => call_native(dst),
    },
    PixelType::U16 => match AVG_HBD_FNS[cpu.as_index()] {
      Some(func) => unsafe {
        (func)(
          dst.data_ptr_mut() as *mut _,
          T::to_asm_stride(dst.plane_cfg.stride),
          tmp1.as_ptr(),
          tmp2.as_ptr(),
          width as i32,
          height as i32,
          bit_depth as i32,
        );
      },
      None => call_native(dst),
    },
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

    static PUT_FNS_AVX2: [Option<PutFn>; 16] = {
      let mut out: [Option<PutFn>; 16] = [None; 16];
      $(
        out[get_2d_mode_idx($mode_x, $mode_y)] = Some($func_name);
      )*
      out
    };
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

pub(crate) static PUT_FNS: [[Option<PutFn>; 16]; CpuFeatureLevel::len()] = {
  let mut out = [[None; 16]; CpuFeatureLevel::len()];
  out[CpuFeatureLevel::AVX2 as usize] = PUT_FNS_AVX2;
  out
};

pub(crate) static PUT_HBD_FNS: [[Option<PutHBDFn>; 16];
  CpuFeatureLevel::len()] = [[None; 16]; CpuFeatureLevel::len()];

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

    static PREP_FNS_AVX2: [Option<PrepFn>; 16] = {
      let mut out: [Option<PrepFn>; 16] = [None; 16];
      $(
        out[get_2d_mode_idx($mode_x, $mode_y)] = Some($func_name);
      )*
      out
    };
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

pub(crate) static PREP_FNS: [[Option<PrepFn>; 16]; CpuFeatureLevel::len()] = {
  let mut out = [[None; 16]; CpuFeatureLevel::len()];
  out[CpuFeatureLevel::AVX2 as usize] = PREP_FNS_AVX2;
  out
};

pub(crate) static PREP_HBD_FNS: [[Option<PrepHBDFn>; 16];
  CpuFeatureLevel::len()] = [[None; 16]; CpuFeatureLevel::len()];

extern {
  fn rav1e_avg_ssse3(
    dst: *mut u8, dst_stride: libc::ptrdiff_t, tmp1: *const i16,
    tmp2: *const i16, w: i32, h: i32,
  );

  fn rav1e_avg_avx2(
    dst: *mut u8, dst_stride: libc::ptrdiff_t, tmp1: *const i16,
    tmp2: *const i16, w: i32, h: i32,
  );
}

pub(crate) static AVG_FNS: [Option<AvgFn>; CpuFeatureLevel::len()] = {
  let mut out: [Option<AvgFn>; CpuFeatureLevel::len()] =
    [None; CpuFeatureLevel::len()];
  out[CpuFeatureLevel::SSSE3 as usize] = Some(rav1e_avg_ssse3);
  out[CpuFeatureLevel::AVX2 as usize] = Some(rav1e_avg_avx2);
  out
};

pub(crate) static AVG_HBD_FNS: [Option<AvgHBDFn>; CpuFeatureLevel::len()] =
  [None; CpuFeatureLevel::len()];
