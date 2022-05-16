// Copyright (c) 2021, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::cpu_features::CpuFeatureLevel;
use crate::sad_row::*;
use crate::util::{Pixel, PixelType};

use std::mem;

macro_rules! decl_sad_row_fn {
  ($($f:ident),+) => {
    extern {
      $(
        fn $f(
          src: *const u8, dst: *const u8, len: libc::size_t,
        ) -> u64;
      )*
    }
  };
}

decl_sad_row_fn!(rav1e_sad_row_8bpc_sse2, rav1e_sad_row_8bpc_avx2);

pub(crate) fn sad_row_internal<T: Pixel>(
  src: &[T], dst: &[T], cpu: CpuFeatureLevel,
) -> u64 {
  assert!(src.len() == dst.len());

  match T::type_enum() {
    PixelType::U8 => {
      // helper macro to reduce boilerplate
      macro_rules! call_asm {
        ($func:ident, $src:expr, $dst:expr, $cpu:expr) => {
          // SAFETY: Calls Assembly code.
          //
          // FIXME: Remove `allow` once https://github.com/rust-lang/rust-clippy/issues/8264 fixed
          #[allow(clippy::undocumented_unsafe_blocks)]
          unsafe {
            let result =
              $func(mem::transmute(($src).as_ptr()), mem::transmute(($dst).as_ptr()), src.len()) as u64;

            #[cfg(feature = "check_asm")]
            assert_eq!(result, rust::sad_row_internal($src, $dst, $cpu));

            result
          }
        };
      }

      if cpu >= CpuFeatureLevel::AVX2 {
        call_asm!(rav1e_sad_row_8bpc_avx2, src, dst, cpu)
      } else if cpu >= CpuFeatureLevel::SSE2 {
        call_asm!(rav1e_sad_row_8bpc_sse2, src, dst, cpu)
      } else {
        rust::sad_row_internal(src, dst, cpu)
      }
    }
    PixelType::U16 => rust::sad_row_internal(src, dst, cpu),
  }
}
