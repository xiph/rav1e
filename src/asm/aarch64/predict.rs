// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::context::MAX_TX_SIZE;
use crate::cpu_features::CpuFeatureLevel;
use crate::predict::{native, PredictionMode, PredictionVariant};
use crate::tiling::PlaneRegionMut;
use crate::transform::TxSize;
use crate::util::AlignedArray;
use crate::Pixel;
use libc;
use std::mem::size_of;

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
  rav1e_ipred_dc_neon,
  rav1e_ipred_dc_128_neon,
  rav1e_ipred_dc_left_neon,
  rav1e_ipred_dc_top_neon,
  rav1e_ipred_v_neon,
  rav1e_ipred_h_neon,
  rav1e_ipred_smooth_neon,
  rav1e_ipred_smooth_v_neon,
  rav1e_ipred_smooth_h_neon,
  rav1e_ipred_paeth_neon
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
  rav1e_ipred_cfl_neon,
  rav1e_ipred_cfl_128_neon,
  rav1e_ipred_cfl_left_neon,
  rav1e_ipred_cfl_top_neon
}

#[inline(always)]
pub fn dispatch_predict_intra<T: Pixel>(
  mode: PredictionMode, variant: PredictionVariant,
  dst: &mut PlaneRegionMut<'_, T>, tx_size: TxSize, bit_depth: usize,
  ac: &[i16], angle: isize, edge_buf: &AlignedArray<[T; 4 * MAX_TX_SIZE + 1]>,
  cpu: CpuFeatureLevel,
) {
  let call_native = |dst: &mut PlaneRegionMut<'_, T>| {
    native::dispatch_predict_intra(
      mode, variant, dst, tx_size, bit_depth, ac, angle, edge_buf, cpu,
    );
  };

  if size_of::<T>() != 1 {
    return call_native(dst);
  }

  unsafe {
    let dst_ptr = dst.data_ptr_mut() as *mut _;
    let stride = dst.plane_cfg.stride as libc::ptrdiff_t;
    let edge_ptr =
      edge_buf.array.as_ptr().offset(2 * MAX_TX_SIZE as isize) as *const _;
    let w = tx_size.width() as libc::c_int;
    let h = tx_size.height() as libc::c_int;
    let angle = angle as libc::c_int;

    if cpu >= CpuFeatureLevel::NEON {
      match mode {
        PredictionMode::DC_PRED => {
          (match variant {
            PredictionVariant::NONE => rav1e_ipred_dc_128_neon,
            PredictionVariant::LEFT => rav1e_ipred_dc_left_neon,
            PredictionVariant::TOP => rav1e_ipred_dc_top_neon,
            PredictionVariant::BOTH => rav1e_ipred_dc_neon,
          })(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::V_PRED if angle == 90 => {
          rav1e_ipred_v_neon(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::H_PRED if angle == 180 => {
          rav1e_ipred_h_neon(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::SMOOTH_PRED => {
          rav1e_ipred_smooth_neon(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::SMOOTH_V_PRED => {
          rav1e_ipred_smooth_v_neon(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::SMOOTH_H_PRED => {
          rav1e_ipred_smooth_h_neon(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::PAETH_PRED => {
          rav1e_ipred_paeth_neon(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::UV_CFL_PRED => {
          let ac_ptr = ac.as_ptr() as *const _;
          (match variant {
            PredictionVariant::NONE => rav1e_ipred_cfl_128_neon,
            PredictionVariant::LEFT => rav1e_ipred_cfl_left_neon,
            PredictionVariant::TOP => rav1e_ipred_cfl_top_neon,
            PredictionVariant::BOTH => rav1e_ipred_cfl_neon,
          })(dst_ptr, stride, edge_ptr, w, h, ac_ptr, angle);
        }
        _ => call_native(dst),
      }
    } else {
      call_native(dst);
    }
  }
}
