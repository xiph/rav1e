// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::cpu_features::CpuFeatureLevel;
use crate::predict::{self, native};
use crate::tiling::PlaneRegionMut;
use crate::Pixel;
use libc;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::mem::size_of;
use std::ptr;

macro_rules! decl_angular_ipred_fn {
  ($($f:ident),+) => {
    extern {
      $(
        fn $f(
          dst: *mut u8, stride: libc::ptrdiff_t, topleft: *const u8,
          width: libc::c_int, height: libc::c_int, angle: libc::c_int,
        );
      )*
    }
  };
}

decl_angular_ipred_fn! {
  rav1e_ipred_dc_avx2,
  rav1e_ipred_dc_ssse3,
  rav1e_ipred_dc_128_avx2,
  rav1e_ipred_dc_128_ssse3,
  rav1e_ipred_dc_left_avx2,
  rav1e_ipred_dc_left_ssse3,
  rav1e_ipred_dc_top_avx2,
  rav1e_ipred_dc_top_ssse3,
  rav1e_ipred_h_avx2,
  rav1e_ipred_h_ssse3,
  rav1e_ipred_v_avx2,
  rav1e_ipred_v_ssse3,
  rav1e_ipred_paeth_avx2,
  rav1e_ipred_smooth_avx2,
  rav1e_ipred_smooth_ssse3,
  rav1e_ipred_smooth_h_avx2,
  rav1e_ipred_smooth_h_ssse3,
  rav1e_ipred_smooth_v_avx2,
  rav1e_ipred_smooth_v_ssse3
}

macro_rules! decl_cfl_pred_fn {
  ($($f:ident),+) => {
    extern {
      $(
        fn $f(
          dst: *mut u8, stride: libc::ptrdiff_t, topleft: *const u8,
          width: libc::c_int, height: libc::c_int, ac: *const u8,
          alpha: libc::c_int,
        );
      )*
    }
  };
}

decl_cfl_pred_fn! {
  rav1e_ipred_cfl_avx2,
  rav1e_ipred_cfl_ssse3,
  rav1e_ipred_cfl_128_avx2,
  rav1e_ipred_cfl_128_ssse3,
  rav1e_ipred_cfl_left_avx2,
  rav1e_ipred_cfl_left_ssse3,
  rav1e_ipred_cfl_top_avx2,
  rav1e_ipred_cfl_top_ssse3
}

pub trait Intra<T>: native::Intra<T>
where
  T: Pixel,
{
  fn pred_dc(
    output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T],
    cpu: CpuFeatureLevel,
  ) {
    if size_of::<T>() == 1 && cpu >= CpuFeatureLevel::SSSE3 {
      return unsafe {
        (if cpu >= CpuFeatureLevel::AVX2 {
          rav1e_ipred_dc_avx2
        } else {
          rav1e_ipred_dc_ssse3
        })(
          output.data_ptr_mut() as *mut _,
          output.plane_cfg.stride as libc::ptrdiff_t,
          above.as_ptr().offset(-1) as *const _,
          Self::W as libc::c_int,
          Self::H as libc::c_int,
          0,
        )
      };
    }

    <Self as native::Intra<T>>::pred_dc(output, above, left, cpu)
  }

  fn pred_dc_128(
    output: &mut PlaneRegionMut<'_, T>, bit_depth: usize, cpu: CpuFeatureLevel,
  ) {
    if size_of::<T>() == 1 && cpu >= CpuFeatureLevel::SSSE3 {
      return unsafe {
        (if cpu >= CpuFeatureLevel::AVX2 {
          rav1e_ipred_dc_128_avx2
        } else {
          rav1e_ipred_dc_128_ssse3
        })(
          output.data_ptr_mut() as *mut _,
          output.plane_cfg.stride as libc::ptrdiff_t,
          ptr::null(),
          Self::W as libc::c_int,
          Self::H as libc::c_int,
          0,
        )
      };
    }

    <Self as native::Intra<T>>::pred_dc_128(output, bit_depth, cpu)
  }

  fn pred_dc_left(
    output: &mut PlaneRegionMut<'_, T>, _above: &[T], left: &[T],
    cpu: CpuFeatureLevel,
  ) {
    if size_of::<T>() == 1 && cpu >= CpuFeatureLevel::SSSE3 {
      return unsafe {
        (if cpu >= CpuFeatureLevel::AVX2 {
          rav1e_ipred_dc_left_avx2
        } else {
          rav1e_ipred_dc_left_ssse3
        })(
          output.data_ptr_mut() as *mut _,
          output.plane_cfg.stride as libc::ptrdiff_t,
          left.as_ptr().add(Self::H) as *const _,
          Self::W as libc::c_int,
          Self::H as libc::c_int,
          0,
        )
      };
    }

    <Self as native::Intra<T>>::pred_dc_left(output, _above, left, cpu)
  }

  fn pred_dc_top(
    output: &mut PlaneRegionMut<'_, T>, above: &[T], _left: &[T],
    cpu: CpuFeatureLevel,
  ) {
    if size_of::<T>() == 1 && cpu >= CpuFeatureLevel::SSSE3 {
      return unsafe {
        (if cpu >= CpuFeatureLevel::AVX2 {
          rav1e_ipred_dc_top_avx2
        } else {
          rav1e_ipred_dc_top_ssse3
        })(
          output.data_ptr_mut() as *mut _,
          output.plane_cfg.stride as libc::ptrdiff_t,
          above.as_ptr().offset(-1) as *const _,
          Self::W as libc::c_int,
          Self::H as libc::c_int,
          0,
        )
      };
    }

    <Self as native::Intra<T>>::pred_dc_top(output, above, _left, cpu)
  }

  fn pred_h(
    output: &mut PlaneRegionMut<'_, T>, left: &[T], cpu: CpuFeatureLevel,
  ) {
    if size_of::<T>() == 1 && cpu >= CpuFeatureLevel::SSSE3 {
      return unsafe {
        (if cpu >= CpuFeatureLevel::AVX2 {
          rav1e_ipred_h_avx2
        } else {
          rav1e_ipred_h_ssse3
        })(
          output.data_ptr_mut() as *mut _,
          output.plane_cfg.stride as libc::ptrdiff_t,
          left.as_ptr().add(Self::H) as *const _,
          Self::W as libc::c_int,
          Self::H as libc::c_int,
          0,
        )
      };
    }

    <Self as native::Intra<T>>::pred_h(output, left, cpu)
  }

  fn pred_v(
    output: &mut PlaneRegionMut<'_, T>, above: &[T], cpu: CpuFeatureLevel,
  ) {
    if size_of::<T>() == 1 && cpu >= CpuFeatureLevel::SSSE3 {
      return unsafe {
        (if cpu >= CpuFeatureLevel::AVX2 {
          rav1e_ipred_v_avx2
        } else {
          rav1e_ipred_v_ssse3
        })(
          output.data_ptr_mut() as *mut _,
          output.plane_cfg.stride as libc::ptrdiff_t,
          above.as_ptr().offset(-1) as *const _,
          Self::W as libc::c_int,
          Self::H as libc::c_int,
          0,
        )
      };
    }

    <Self as native::Intra<T>>::pred_v(output, above, cpu)
  }

  fn pred_paeth(
    output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T],
    above_left: T, cpu: CpuFeatureLevel,
  ) {
    if size_of::<T>() == 1 && cpu >= CpuFeatureLevel::AVX2 {
      return unsafe {
        rav1e_ipred_paeth_avx2(
          output.data_ptr_mut() as *mut _,
          output.plane_cfg.stride as libc::ptrdiff_t,
          above.as_ptr().offset(-1) as *const _,
          Self::W as libc::c_int,
          Self::H as libc::c_int,
          0,
        )
      };
    }

    <Self as native::Intra<T>>::pred_paeth(
      output, above, left, above_left, cpu,
    )
  }

  fn pred_smooth(
    output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T],
    cpu: CpuFeatureLevel,
  ) {
    if size_of::<T>() == 1 && cpu >= CpuFeatureLevel::SSSE3 {
      return unsafe {
        (if cpu >= CpuFeatureLevel::AVX2 {
          rav1e_ipred_smooth_avx2
        } else {
          rav1e_ipred_smooth_ssse3
        })(
          output.data_ptr_mut() as *mut _,
          output.plane_cfg.stride as libc::ptrdiff_t,
          above.as_ptr().offset(-1) as *const _,
          Self::W as libc::c_int,
          Self::H as libc::c_int,
          0,
        )
      };
    }

    <Self as native::Intra<T>>::pred_smooth(output, above, left, cpu)
  }

  fn pred_smooth_h(
    output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T],
    cpu: CpuFeatureLevel,
  ) {
    if size_of::<T>() == 1 && cpu >= CpuFeatureLevel::SSSE3 {
      return unsafe {
        (if cpu >= CpuFeatureLevel::AVX2 {
          rav1e_ipred_smooth_h_avx2
        } else {
          rav1e_ipred_smooth_h_ssse3
        })(
          output.data_ptr_mut() as *mut _,
          output.plane_cfg.stride as libc::ptrdiff_t,
          above.as_ptr().offset(-1) as *const _,
          Self::W as libc::c_int,
          Self::H as libc::c_int,
          0,
        )
      };
    }

    <Self as native::Intra<T>>::pred_smooth_h(output, above, left, cpu)
  }

  fn pred_smooth_v(
    output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T],
    cpu: CpuFeatureLevel,
  ) {
    if size_of::<T>() == 1 && cpu >= CpuFeatureLevel::SSSE3 {
      return unsafe {
        (if cpu >= CpuFeatureLevel::AVX2 {
          rav1e_ipred_smooth_v_avx2
        } else {
          rav1e_ipred_smooth_v_ssse3
        })(
          output.data_ptr_mut() as *mut _,
          output.plane_cfg.stride as libc::ptrdiff_t,
          above.as_ptr().offset(-1) as *const _,
          Self::W as libc::c_int,
          Self::H as libc::c_int,
          0,
        )
      };
    }

    <Self as native::Intra<T>>::pred_smooth_v(output, above, left, cpu)
  }

  #[target_feature(enable = "ssse3")]
  unsafe fn pred_cfl_ssse3(
    output: *mut T, stride: usize, ac: *const i16, alpha: i16,
    bit_depth: usize,
  ) {
    let alpha_sign = _mm_set1_epi16(alpha);
    let alpha_q12 = _mm_slli_epi16(_mm_abs_epi16(alpha_sign), 9);
    let dc_scalar: u32 = (*output).into();
    let dc_q0 = _mm_set1_epi16(dc_scalar as i16);
    let max = _mm_set1_epi16((1 << bit_depth) - 1);

    for j in 0..Self::H {
      let luma = ac.add(Self::W * j);
      let line = output.add(stride * j);

      let mut i = 0isize;
      let mut last = _mm_setzero_si128();
      while (i as usize) < Self::W {
        let ac_q3 = _mm_loadu_si128(luma.offset(i) as *const _);
        let ac_sign = _mm_sign_epi16(alpha_sign, ac_q3);
        let abs_scaled_luma_q0 =
          _mm_mulhrs_epi16(_mm_abs_epi16(ac_q3), alpha_q12);
        let scaled_luma_q0 = _mm_sign_epi16(abs_scaled_luma_q0, ac_sign);
        let pred = _mm_add_epi16(scaled_luma_q0, dc_q0);
        if size_of::<T>() == 1 {
          if Self::W < 16 {
            let res = _mm_packus_epi16(pred, pred);
            if Self::W == 4 {
              *(line.offset(i) as *mut i32) = _mm_cvtsi128_si32(res);
            } else {
              _mm_storel_epi64(line.offset(i) as *mut _, res);
            }
          } else if (i & 15) == 0 {
            last = pred;
          } else {
            let res = _mm_packus_epi16(last, pred);
            _mm_storeu_si128(line.offset(i - 8) as *mut _, res);
          }
        } else {
          let res =
            _mm_min_epi16(max, _mm_max_epi16(pred, _mm_setzero_si128()));
          if Self::W == 4 {
            _mm_storel_epi64(line.offset(i) as *mut _, res);
          } else {
            _mm_storeu_si128(line.offset(i) as *mut _, res);
          }
        }
        i += 8;
      }
    }
  }

  fn pred_cfl_inner(
    output: &mut PlaneRegionMut<'_, T>, ac: &[i16], alpha: i16,
    bit_depth: usize, cpu: CpuFeatureLevel,
  ) {
    if alpha == 0 {
      return;
    }
    assert!(32 >= Self::W);
    assert!(ac.len() >= 32 * (Self::H - 1) + Self::W);
    assert!(output.plane_cfg.stride >= Self::W);
    assert!(output.rows_iter().len() >= Self::H);

    if cpu >= CpuFeatureLevel::SSSE3 {
      return unsafe {
        Self::pred_cfl_ssse3(
          output.data_ptr_mut(),
          output.plane_cfg.stride,
          ac.as_ptr(),
          alpha,
          bit_depth,
        )
      };
    }

    <Self as Intra<T>>::pred_cfl_inner(output, &ac, alpha, bit_depth, cpu);
  }

  fn pred_cfl(
    output: &mut PlaneRegionMut<'_, T>, ac: &[i16], alpha: i16,
    bit_depth: usize, above: &[T], left: &[T], cpu: CpuFeatureLevel,
  ) {
    {
      if size_of::<T>() == 1 && cpu >= CpuFeatureLevel::SSSE3 {
        return unsafe {
          (if cpu >= CpuFeatureLevel::AVX2 {
            rav1e_ipred_cfl_avx2
          } else {
            rav1e_ipred_cfl_ssse3
          })(
            output.data_ptr_mut() as *mut _,
            output.plane_cfg.stride as libc::ptrdiff_t,
            above.as_ptr().offset(-1) as *const _,
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            ac.as_ptr() as *const _,
            alpha as libc::c_int,
          )
        };
      }
    }
    <Self as Intra<T>>::pred_dc(output, above, left, cpu);
    <Self as Intra<T>>::pred_cfl_inner(output, &ac, alpha, bit_depth, cpu);
  }

  fn pred_cfl_128(
    output: &mut PlaneRegionMut<'_, T>, ac: &[i16], alpha: i16,
    bit_depth: usize, cpu: CpuFeatureLevel,
  ) {
    {
      if size_of::<T>() == 1 && cpu >= CpuFeatureLevel::SSSE3 {
        return unsafe {
          (if cpu >= CpuFeatureLevel::AVX2 {
            rav1e_ipred_cfl_128_avx2
          } else {
            rav1e_ipred_cfl_128_ssse3
          })(
            output.data_ptr_mut() as *mut _,
            output.plane_cfg.stride as libc::ptrdiff_t,
            ptr::null(),
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            ac.as_ptr() as *const _,
            alpha as libc::c_int,
          )
        };
      }
    }
    <Self as Intra<T>>::pred_dc_128(output, bit_depth, cpu);
    <Self as Intra<T>>::pred_cfl_inner(output, &ac, alpha, bit_depth, cpu);
  }

  fn pred_cfl_left(
    output: &mut PlaneRegionMut<'_, T>, ac: &[i16], alpha: i16,
    bit_depth: usize, above: &[T], left: &[T], cpu: CpuFeatureLevel,
  ) {
    {
      if size_of::<T>() == 1 && cpu >= CpuFeatureLevel::SSSE3 {
        return unsafe {
          (if cpu >= CpuFeatureLevel::AVX2 {
            rav1e_ipred_cfl_left_avx2
          } else {
            rav1e_ipred_cfl_left_ssse3
          })(
            output.data_ptr_mut() as *mut _,
            output.plane_cfg.stride as libc::ptrdiff_t,
            above.as_ptr().offset(-1) as *const _,
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            ac.as_ptr() as *const _,
            alpha as libc::c_int,
          )
        };
      }
    }
    <Self as Intra<T>>::pred_dc_left(output, above, left, cpu);
    <Self as Intra<T>>::pred_cfl_inner(output, &ac, alpha, bit_depth, cpu);
  }

  fn pred_cfl_top(
    output: &mut PlaneRegionMut<'_, T>, ac: &[i16], alpha: i16,
    bit_depth: usize, above: &[T], left: &[T], cpu: CpuFeatureLevel,
  ) {
    {
      if size_of::<T>() == 1 && cpu >= CpuFeatureLevel::SSSE3 {
        return unsafe {
          (if cpu >= CpuFeatureLevel::AVX2 {
            rav1e_ipred_cfl_top_avx2
          } else {
            rav1e_ipred_cfl_top_ssse3
          })(
            output.data_ptr_mut() as *mut _,
            output.plane_cfg.stride as libc::ptrdiff_t,
            above.as_ptr().offset(-1) as *const _,
            Self::W as libc::c_int,
            Self::H as libc::c_int,
            ac.as_ptr() as *const _,
            alpha as libc::c_int,
          )
        };
      }
    }
    <Self as Intra<T>>::pred_dc_top(output, above, left, cpu);
    <Self as Intra<T>>::pred_cfl_inner(output, &ac, alpha, bit_depth, cpu);
  }
}

macro_rules! block_intra_asm_impl {
  ($W:expr, $H:expr) => {
    paste::item! {
      impl<T: Pixel> Intra<T> for predict::[<Block $W x $H>] {}
    }
  };
}

macro_rules! blocks_intra_asm_impl {
  ($(($W:expr, $H:expr)),+) => {
    $(
      block_intra_asm_impl! { $W, $H }
    )*
  }
}

blocks_intra_asm_impl! { (4, 4), (8, 8), (16, 16), (32, 32), (64, 64) }
blocks_intra_asm_impl! { (4, 8), (8, 16), (16, 32), (32, 64) }
blocks_intra_asm_impl! { (8, 4), (16, 8), (32, 16), (64, 32) }
blocks_intra_asm_impl! { (4, 16), (8, 32), (16, 64) }
blocks_intra_asm_impl! { (16, 4), (32, 8), (64, 16) }
