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
  fn rav1e_avg_avx2(
    dst: *mut u8, dst_stride: libc::ptrdiff_t, tmp1: *const i16,
    tmp2: *const i16, w: i32, h: i32,
  );
}

pub(crate) static AVG_FNS: [Option<AvgFn>; CpuFeatureLevel::len()] = {
  let mut out: [Option<AvgFn>; CpuFeatureLevel::len()] =
    [None; CpuFeatureLevel::len()];
  out[CpuFeatureLevel::AVX2 as usize] = Some(rav1e_avg_avx2);
  out
};

pub(crate) static AVG_HBD_FNS: [Option<AvgHBDFn>; CpuFeatureLevel::len()] =
  [None; CpuFeatureLevel::len()];

#[rustfmt::skip]
static MC_SUBPEL_FILTERS: AlignedArray<[i8; 600]> = AlignedArray::new([
0,   1,  -3,  63,   4,  -1,   0,   0, // REGULAR
0,   1,  -5,  61,   9,  -2,   0,   0,
0,   1,  -6,  58,  14,  -4,   1,   0,
0,   1,  -7,  55,  19,  -5,   1,   0,
0,   1,  -7,  51,  24,  -6,   1,   0,
0,   1,  -8,  47,  29,  -6,   1,   0,
0,   1,  -7,  42,  33,  -6,   1,   0,
0,   1,  -7,  38,  38,  -7,   1,   0,
0,   1,  -6,  33,  42,  -7,   1,   0,
0,   1,  -6,  29,  47,  -8,   1,   0,
0,   1,  -6,  24,  51,  -7,   1,   0,
0,   1,  -5,  19,  55,  -7,   1,   0,
0,   1,  -4,  14,  58,  -6,   1,   0,
0,   0,  -2,   9,  61,  -5,   1,   0,
0,   0,  -1,   4,  63,  -3,   1,   0,

0,   1,  14,  31,  17,   1,   0,   0, // SMOOTH
0,   0,  13,  31,  18,   2,   0,   0,
0,   0,  11,  31,  20,   2,   0,   0,
0,   0,  10,  30,  21,   3,   0,   0,
0,   0,   9,  29,  22,   4,   0,   0,
0,   0,   8,  28,  23,   5,   0,   0,
0,  -1,   8,  27,  24,   6,   0,   0,
0,  -1,   7,  26,  26,   7,  -1,   0,
0,   0,   6,  24,  27,   8,  -1,   0,
0,   0,   5,  23,  28,   8,   0,   0,
0,   0,   4,  22,  29,   9,   0,   0,
0,   0,   3,  21,  30,  10,   0,   0,
0,   0,   2,  20,  31,  11,   0,   0,
0,   0,   2,  18,  31,  13,   0,   0,
0,   0,   1,  17,  31,  14,   1,   0,

1,   1,  -3,  63,   4,  -1,   1,   0, // SHARP
1,   3,  -6,  62,   8,  -3,   2,  -1,
1,   4,  -9,  60,  13,  -5,   3,  -1,
2,   5, -11,  58,  19,  -7,   3,  -1,
2,   5, -11,  54,  24,  -9,   4,  -1,
2,   5, -12,  50,  30, -10,   4,  -1,
2,   5, -12,  45,  35, -11,   5,  -1,
2,   6, -12,  40,  40, -12,   6,  -2,
1,   5, -11,  35,  45, -12,   5,  -2,
1,   4, -10,  30,  50, -12,   5,  -2,
1,   4,  -9,  24,  54, -11,   5,  -2,
1,   3,  -7,  19,  58, -11,   5,  -2,
1,   3,  -5,  13,  60,  -9,   4,  -1,
1,   2,  -3,   8,  62,  -6,   3,  -1,
0,   1,  -1,   4,  63,  -3,   1,  -1,

0,   0,  -2,  63,   4,  -1,   0,   0, // REGULAR 4
0,   0,  -4,  61,   9,  -2,   0,   0,
0,   0,  -5,  58,  14,  -3,   0,   0,
0,   0,  -6,  55,  19,  -4,   0,   0,
0,   0,  -6,  51,  24,  -5,   0,   0,
0,   0,  -7,  47,  29,  -5,   0,   0,
0,   0,  -6,  42,  33,  -5,   0,   0,
0,   0,  -6,  38,  38,  -6,   0,   0,
0,   0,  -5,  33,  42,  -6,   0,   0,
0,   0,  -5,  29,  47,  -7,   0,   0,
0,   0,  -5,  24,  51,  -6,   0,   0,
0,   0,  -4,  19,  55,  -6,   0,   0,
0,   0,  -3,  14,  58,  -5,   0,   0,
0,   0,  -2,   9,  61,  -4,   0,   0,
0,   0,  -1,   4,  63,  -2,   0,   0,

0,   0,  15,  31,  17,   1,   0,   0, // SMOOTH 4
0,   0,  13,  31,  18,   2,   0,   0,
0,   0,  11,  31,  20,   2,   0,   0,
0,   0,  10,  30,  21,   3,   0,   0,
0,   0,   9,  29,  22,   4,   0,   0,
0,   0,   8,  28,  23,   5,   0,   0,
0,   0,   7,  27,  24,   6,   0,   0,
0,   0,   6,  26,  26,   6,   0,   0,
0,   0,   6,  24,  27,   7,   0,   0,
0,   0,   5,  23,  28,   8,   0,   0,
0,   0,   4,  22,  29,   9,   0,   0,
0,   0,   3,  21,  30,  10,   0,   0,
0,   0,   2,  20,  31,  11,   0,   0,
0,   0,   2,  18,  31,  13,   0,   0,
0,   0,   1,  17,  31,  15,   0,   0 ]);

#[no_mangle]
pub static rav1e_mc_subpel_filters: &[i8] = &MC_SUBPEL_FILTERS.array;
