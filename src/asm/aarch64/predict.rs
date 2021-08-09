// Copyright (c) 2019-2021, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::context::MAX_TX_SIZE;
use crate::cpu_features::CpuFeatureLevel;
use crate::predict::{
  rust, IntraEdgeFilterParameters, PredictionMode, PredictionVariant,
};
use crate::tiling::PlaneRegionMut;
use crate::transform::TxSize;
use crate::util::Aligned;
use crate::{Pixel, PixelType};
use libc;

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
  rav1e_ipred_dc_8bpc_neon,
  rav1e_ipred_dc_128_8bpc_neon,
  rav1e_ipred_dc_left_8bpc_neon,
  rav1e_ipred_dc_top_8bpc_neon,
  rav1e_ipred_v_8bpc_neon,
  rav1e_ipred_h_8bpc_neon,
  rav1e_ipred_smooth_8bpc_neon,
  rav1e_ipred_smooth_v_8bpc_neon,
  rav1e_ipred_smooth_h_8bpc_neon,
  rav1e_ipred_paeth_8bpc_neon
}

macro_rules! decl_angular_ipred_hbd_fn {
  ($($f:ident),+) => {
    extern {
      $(
        fn $f(
          dst: *mut u16, stride: libc::ptrdiff_t, topleft: *const u16,
          width: libc::c_int, height: libc::c_int, angle: libc::c_int,
          max_width: libc::c_int, max_height: libc::c_int,
          bit_depth_max: libc::c_int,
        );
      )*
    }
  };
}

decl_angular_ipred_hbd_fn! {
  rav1e_ipred_dc_16bpc_neon,
  rav1e_ipred_dc_128_16bpc_neon,
  rav1e_ipred_dc_left_16bpc_neon,
  rav1e_ipred_dc_top_16bpc_neon,
  rav1e_ipred_v_16bpc_neon,
  rav1e_ipred_h_16bpc_neon,
  rav1e_ipred_smooth_16bpc_neon,
  rav1e_ipred_smooth_v_16bpc_neon,
  rav1e_ipred_smooth_h_16bpc_neon,
  rav1e_ipred_paeth_16bpc_neon
}

macro_rules! decl_cfl_pred_fn {
  ($($f:ident),+) => {
    extern {
      $(
        fn $f(
          dst: *mut u8, stride: libc::ptrdiff_t, topleft: *const u8,
          width: libc::c_int, height: libc::c_int, ac: *const i16,
          alpha: libc::c_int,
        );
      )*
    }
  };
}

decl_cfl_pred_fn! {
  rav1e_ipred_cfl_8bpc_neon,
  rav1e_ipred_cfl_128_8bpc_neon,
  rav1e_ipred_cfl_left_8bpc_neon,
  rav1e_ipred_cfl_top_8bpc_neon
}

macro_rules! decl_cfl_pred_hbd_fn {
  ($($f:ident),+) => {
    extern {
      $(
        fn $f(
          dst: *mut u16, stride: libc::ptrdiff_t, topleft: *const u16,
          width: libc::c_int, height: libc::c_int, ac: *const i16,
          alpha: libc::c_int, bit_depth_max: libc::c_int,
        );
      )*
    }
  };
}

decl_cfl_pred_hbd_fn! {
  rav1e_ipred_cfl_16bpc_neon,
  rav1e_ipred_cfl_128_16bpc_neon,
  rav1e_ipred_cfl_left_16bpc_neon,
  rav1e_ipred_cfl_top_16bpc_neon
}

#[inline(always)]
pub fn dispatch_predict_intra<T: Pixel>(
  mode: PredictionMode, variant: PredictionVariant,
  dst: &mut PlaneRegionMut<'_, T>, tx_size: TxSize, bit_depth: usize,
  ac: &[i16], angle: isize, ief_params: Option<IntraEdgeFilterParameters>,
  edge_buf: &Aligned<[T; 4 * MAX_TX_SIZE + 1]>, cpu: CpuFeatureLevel,
) {
  let call_rust = |dst: &mut PlaneRegionMut<'_, T>| {
    rust::dispatch_predict_intra(
      mode, variant, dst, tx_size, bit_depth, ac, angle, ief_params, edge_buf,
      cpu,
    );
  };

  if cpu < CpuFeatureLevel::NEON {
    return call_rust(dst);
  }

  unsafe {
    let dst_ptr = dst.data_ptr_mut() as *mut _;
    let dst_u16 = dst.data_ptr_mut() as *mut u16;
    let stride = T::to_asm_stride(dst.plane_cfg.stride) as libc::ptrdiff_t;
    let edge_ptr =
      edge_buf.data.as_ptr().offset(2 * MAX_TX_SIZE as isize) as *const _;
    let edge_u16 =
      edge_buf.data.as_ptr().offset(2 * MAX_TX_SIZE as isize) as *const u16;
    let w = tx_size.width() as libc::c_int;
    let h = tx_size.height() as libc::c_int;
    let angle = angle as libc::c_int;
    let bd_max = (1 << bit_depth) - 1;
    match T::type_enum() {
      PixelType::U8 => match mode {
        PredictionMode::DC_PRED => {
          (match variant {
            PredictionVariant::NONE => rav1e_ipred_dc_128_8bpc_neon,
            PredictionVariant::LEFT => rav1e_ipred_dc_left_8bpc_neon,
            PredictionVariant::TOP => rav1e_ipred_dc_top_8bpc_neon,
            PredictionVariant::BOTH => rav1e_ipred_dc_8bpc_neon,
          })(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::V_PRED if angle == 90 => {
          rav1e_ipred_v_8bpc_neon(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::H_PRED if angle == 180 => {
          rav1e_ipred_h_8bpc_neon(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::SMOOTH_PRED => {
          rav1e_ipred_smooth_8bpc_neon(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::SMOOTH_V_PRED => {
          rav1e_ipred_smooth_v_8bpc_neon(
            dst_ptr, stride, edge_ptr, w, h, angle,
          );
        }
        PredictionMode::SMOOTH_H_PRED => {
          rav1e_ipred_smooth_h_8bpc_neon(
            dst_ptr, stride, edge_ptr, w, h, angle,
          );
        }
        PredictionMode::PAETH_PRED => {
          rav1e_ipred_paeth_8bpc_neon(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::UV_CFL_PRED => {
          let ac_ptr = ac.as_ptr() as *const _;
          (match variant {
            PredictionVariant::NONE => rav1e_ipred_cfl_128_8bpc_neon,
            PredictionVariant::LEFT => rav1e_ipred_cfl_left_8bpc_neon,
            PredictionVariant::TOP => rav1e_ipred_cfl_top_8bpc_neon,
            PredictionVariant::BOTH => rav1e_ipred_cfl_8bpc_neon,
          })(dst_ptr, stride, edge_ptr, w, h, ac_ptr, angle);
        }
        _ => call_rust(dst),
      },
      PixelType::U16 if bit_depth > 8 => match mode {
        PredictionMode::DC_PRED => {
          (match variant {
            PredictionVariant::NONE => rav1e_ipred_dc_128_16bpc_neon,
            PredictionVariant::LEFT => rav1e_ipred_dc_left_16bpc_neon,
            PredictionVariant::TOP => rav1e_ipred_dc_top_16bpc_neon,
            PredictionVariant::BOTH => rav1e_ipred_dc_16bpc_neon,
          })(dst_u16, stride, edge_u16, w, h, angle, 0, 0, bd_max);
        }
        PredictionMode::V_PRED if angle == 90 => {
          rav1e_ipred_v_16bpc_neon(
            dst_u16, stride, edge_u16, w, h, angle, 0, 0, bd_max,
          );
        }
        PredictionMode::H_PRED if angle == 180 => {
          rav1e_ipred_h_16bpc_neon(
            dst_u16, stride, edge_u16, w, h, angle, 0, 0, bd_max,
          );
        }
        PredictionMode::SMOOTH_PRED => {
          rav1e_ipred_smooth_16bpc_neon(
            dst_u16, stride, edge_u16, w, h, angle, 0, 0, bd_max,
          );
        }
        PredictionMode::SMOOTH_V_PRED => {
          rav1e_ipred_smooth_v_16bpc_neon(
            dst_u16, stride, edge_u16, w, h, angle, 0, 0, bd_max,
          );
        }
        PredictionMode::SMOOTH_H_PRED => {
          rav1e_ipred_smooth_h_16bpc_neon(
            dst_u16, stride, edge_u16, w, h, angle, 0, 0, bd_max,
          );
        }
        PredictionMode::PAETH_PRED => {
          rav1e_ipred_paeth_16bpc_neon(
            dst_u16, stride, edge_u16, w, h, angle, 0, 0, bd_max,
          );
        }
        PredictionMode::UV_CFL_PRED => {
          let ac_ptr = ac.as_ptr() as *const _;
          (match variant {
            PredictionVariant::NONE => rav1e_ipred_cfl_128_16bpc_neon,
            PredictionVariant::LEFT => rav1e_ipred_cfl_left_16bpc_neon,
            PredictionVariant::TOP => rav1e_ipred_cfl_top_16bpc_neon,
            PredictionVariant::BOTH => rav1e_ipred_cfl_16bpc_neon,
          })(dst_u16, stride, edge_u16, w, h, ac_ptr, angle, bd_max);
        }
        _ => call_rust(dst),
      },
      _ => call_rust(dst),
    }
  }
}
