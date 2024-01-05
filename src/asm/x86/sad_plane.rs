// Copyright (c) 2022, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::cpu_features::CpuFeatureLevel;
use crate::sad_plane::*;
use crate::util::{Pixel, PixelType};

use v_frame::plane::Plane;

use std::mem;

macro_rules! decl_sad_plane_fn {
  ($($f:ident),+) => {
    extern {
      $(
        fn $f(
          src: *const u8, dst: *const u8, stride: libc::size_t,
          width: libc::size_t, rows: libc::size_t
        ) -> u64;
      )*
    }
  };
}

decl_sad_plane_fn!(rav1e_sad_plane_8bpc_sse2, rav1e_sad_plane_8bpc_avx2);

pub(crate) fn sad_plane_internal<T: Pixel>(
  src: &Plane<T>, dst: &Plane<T>, cpu: CpuFeatureLevel,
) -> u64 {
  debug_assert!(src.cfg.width == dst.cfg.width);
  debug_assert!(src.cfg.stride == dst.cfg.stride);
  debug_assert!(src.cfg.height == dst.cfg.height);
  debug_assert!(src.cfg.width <= src.cfg.stride);

  match T::type_enum() {
    PixelType::U8 => {
      // helper macro to reduce boilerplate
      macro_rules! call_asm {
        ($func:ident, $src:expr, $dst:expr, $cpu:expr) => {
          // SAFETY: Calls Assembly code.
          unsafe {
            let result = $func(
              mem::transmute(src.data_origin().as_ptr()),
              mem::transmute(dst.data_origin().as_ptr()),
              src.cfg.stride,
              src.cfg.width,
              src.cfg.height,
            );

            #[cfg(feature = "check_asm")]
            assert_eq!(result, rust::sad_plane_internal($src, $dst, $cpu));

            result
          }
        };
      }

      if cpu >= CpuFeatureLevel::AVX2 {
        call_asm!(rav1e_sad_plane_8bpc_avx2, src, dst, cpu)
      } else if cpu >= CpuFeatureLevel::SSE2 {
        call_asm!(rav1e_sad_plane_8bpc_sse2, src, dst, cpu)
      } else {
        rust::sad_plane_internal(src, dst, cpu)
      }
    }
    PixelType::U16 => rust::sad_plane_internal(src, dst, cpu),
  }
}
