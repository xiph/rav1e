// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::cdef::*;
use crate::cpu_features::CpuFeatureLevel;
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
  dst: *mut u16,
  dst_stride: isize,
  tmp: *const u16,
  tmp_stride: isize,
  pri_strength: i32,
  sec_strength: i32,
  dir: i32,
  damping: i32,
  bitdepth_max: i32,
);

#[inline(always)]
const fn decimate_index(xdec: usize, ydec: usize) -> usize {
  ((ydec << 1) | xdec) & 3
}

pub(crate) unsafe fn cdef_filter_block<T: Pixel>(
  dst: &mut PlaneRegionMut<'_, T>, src: *const T, src_stride: isize,
  pri_strength: i32, sec_strength: i32, dir: usize, damping: i32,
  bit_depth: usize, xdec: usize, ydec: usize, edges: u8, cpu: CpuFeatureLevel,
) {
  let call_rust = |dst: &mut PlaneRegionMut<T>| {
    rust::cdef_filter_block(
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
      edges,
      cpu,
    );
  };

  // TODO: handle padding in the fast path
  if edges != CDEF_HAVE_ALL {
    call_rust(dst);
  } else {
    #[cfg(feature = "check_asm")]
    let ref_dst = {
      let mut copy = dst.scratch_copy();
      call_rust(&mut copy.as_region_mut());
      copy
    };
    match T::type_enum() {
      PixelType::U8 => {
        match CDEF_FILTER_FNS[cpu.as_index()][decimate_index(xdec, ydec)] {
          Some(func) => {
            // current cdef_filter_block asm does 16->8 for historical
            // reasons.  Copy into tmp space for now (also handling
            // padding) until asm is updated
            const TMPSTRIDE: isize =
              std::mem::align_of::<Aligned<u16>>() as isize;
            /* 256 or 512-bit alignment, greater than 2 * (8>>xdec) + 2 */
            const TMPSIZE: usize =
              ((2 + 8 + 2) * TMPSTRIDE + TMPSTRIDE) as usize;
            let mut tmp: Aligned<[u16; TMPSIZE]> =
              Aligned::new([CDEF_VERY_LARGE; TMPSIZE]);
            rust::pad_into_tmp16(
              tmp.data.as_mut_ptr().offset(TMPSTRIDE - 2), // points to
              // *padding* upper left; the -2 is to make sure the
              // block area is SIMD-aligned, not the padding
              TMPSTRIDE,
              src, // points to *block* upper left
              src_stride,
              8 >> xdec,
              8 >> ydec,
              edges,
            );
            (func)(
              dst.data_ptr_mut() as *mut _,
              T::to_asm_stride(dst.plane_cfg.stride),
              tmp.data.as_ptr().offset(3 * TMPSTRIDE) as *const u16,
              TMPSTRIDE,
              pri_strength,
              sec_strength,
              dir as i32,
              damping,
            );
          }
          None => call_rust(dst),
        }
      }
      PixelType::U16 => {
        match CDEF_FILTER_HBD_FNS[cpu.as_index()][decimate_index(xdec, ydec)] {
          Some(func) => {
            (func)(
              dst.data_ptr_mut() as *mut _,
              T::to_asm_stride(dst.plane_cfg.stride),
              src as *const _,
              src_stride,
              pri_strength,
              sec_strength,
              dir as i32,
              damping,
              (1 << bit_depth) - 1,
            );
          }
          None => call_rust(dst),
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
}

extern {
  fn rav1e_cdef_filter_4x4_neon(
    dst: *mut u8, dst_stride: isize, tmp: *const u16, tmp_stride: isize,
    pri_strength: i32, sec_strength: i32, dir: i32, damping: i32,
  );

  fn rav1e_cdef_filter_4x8_neon(
    dst: *mut u8, dst_stride: isize, tmp: *const u16, tmp_stride: isize,
    pri_strength: i32, sec_strength: i32, dir: i32, damping: i32,
  );

  fn rav1e_cdef_filter_8x8_neon(
    dst: *mut u8, dst_stride: isize, tmp: *const u16, tmp_stride: isize,
    pri_strength: i32, sec_strength: i32, dir: i32, damping: i32,
  );
}

static CDEF_FILTER_FNS_NEON: [Option<CdefFilterFn>; 4] = {
  let mut out: [Option<CdefFilterFn>; 4] = [None; 4];
  out[decimate_index(1, 1)] = Some(rav1e_cdef_filter_4x4_neon);
  out[decimate_index(1, 0)] = Some(rav1e_cdef_filter_4x8_neon);
  out[decimate_index(0, 0)] = Some(rav1e_cdef_filter_8x8_neon);
  out
};

cpu_function_lookup_table!(
  CDEF_FILTER_FNS: [[Option<CdefFilterFn>; 4]],
  default: [None; 4],
  // [NEON]
  []
);

cpu_function_lookup_table!(
  CDEF_FILTER_HBD_FNS: [[Option<CdefFilterHBDFn>; 4]],
  default: [None; 4],
  []
);

type CdefDirLBDFn =
  unsafe extern fn(tmp: *const u8, tmp_stride: isize, var: *mut u32) -> i32;
type CdefDirHBDFn =
  unsafe extern fn(tmp: *const u16, tmp_stride: isize, var: *mut u32) -> i32;

#[inline(always)]
#[allow(clippy::let_and_return)]
pub(crate) fn cdef_find_dir<T: Pixel>(
  img: &PlaneSlice<'_, T>, var: &mut u32, coeff_shift: usize,
  cpu: CpuFeatureLevel,
) -> i32 {
  let call_rust =
    |var: &mut u32| rust::cdef_find_dir::<T>(img, var, coeff_shift, cpu);

  #[cfg(feature = "check_asm")]
  let (ref_dir, ref_var) = {
    let mut var: u32 = 0;
    let dir = call_rust(&mut var);
    (dir, var)
  };

  let dir = match T::type_enum() {
    PixelType::U8 => {
      if let Some(func) = CDEF_DIR_LBD_FNS[cpu.as_index()] {
        unsafe {
          (func)(
            img.as_ptr() as *const _,
            T::to_asm_stride(img.plane.cfg.stride),
            var as *mut u32,
          )
        }
      } else {
        call_rust(var)
      }
    }
    PixelType::U16 => {
      if let Some(func) = CDEF_DIR_HBD_FNS[cpu.as_index()] {
        unsafe {
          (func)(
            img.as_ptr() as *const _,
            T::to_asm_stride(img.plane.cfg.stride),
            var as *mut u32,
          )
        }
      } else {
        call_rust(var)
      }
    }
  };

  #[cfg(feature = "check_asm")]
  {
    assert_eq!(dir, ref_dir);
    assert_eq!(*var, ref_var);
  }

  dir
}

extern {
  fn rav1e_cdef_find_dir_8bpc_neon(
    tmp: *const u8, tmp_stride: isize, var: *mut u32,
  ) -> i32;
}

extern {
  fn rav1e_cdef_find_dir_16bpc_neon(
    tmp: *const u16, tmp_stride: isize, var: *mut u32,
  ) -> i32;
}

cpu_function_lookup_table!(
  CDEF_DIR_LBD_FNS: [Option<CdefDirLBDFn>],
  default: None,
  [(NEON, Some(rav1e_cdef_find_dir_8bpc_neon))]
);

cpu_function_lookup_table!(
  CDEF_DIR_HBD_FNS: [Option<CdefDirHBDFn>],
  default: None,
  [(NEON, Some(rav1e_cdef_find_dir_16bpc_neon))]
);

