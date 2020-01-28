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
use crate::predict::{
  native, IntraEdgeFilterParameters, PredictionMode, PredictionVariant,
};
use crate::tiling::PlaneRegionMut;
use crate::transform::TxSize;
use crate::util::Aligned;
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
  rav1e_ipred_dc_avx2,
  rav1e_ipred_dc_ssse3,
  rav1e_ipred_dc_128_avx2,
  rav1e_ipred_dc_128_ssse3,
  rav1e_ipred_dc_left_avx2,
  rav1e_ipred_dc_left_ssse3,
  rav1e_ipred_dc_top_avx2,
  rav1e_ipred_dc_top_ssse3,
  rav1e_ipred_v_avx2,
  rav1e_ipred_v_ssse3,
  rav1e_ipred_h_avx2,
  rav1e_ipred_h_ssse3,
  rav1e_ipred_z1_avx2,
  rav1e_ipred_z3_avx2,
  rav1e_ipred_smooth_avx2,
  rav1e_ipred_smooth_ssse3,
  rav1e_ipred_smooth_v_avx2,
  rav1e_ipred_smooth_v_ssse3,
  rav1e_ipred_smooth_h_avx2,
  rav1e_ipred_smooth_h_ssse3,
  rav1e_ipred_paeth_avx2,
  rav1e_ipred_paeth_ssse3
}

// For z2 prediction, we need to provide extra parameters, dx and dy, which indicate
// the distance between the predicted block's top-left pixel and the frame's edge.
// It is required for the intra edge filtering process.
extern {
  fn rav1e_ipred_z2_avx2(
    dst: *mut u8, stride: libc::ptrdiff_t, topleft: *const u8,
    width: libc::c_int, height: libc::c_int, angle: libc::c_int,
    dx: libc::c_int, dy: libc::c_int,
  );
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

#[inline(always)]
pub fn dispatch_predict_intra<T: Pixel>(
  mode: PredictionMode, variant: PredictionVariant,
  dst: &mut PlaneRegionMut<'_, T>, tx_size: TxSize, bit_depth: usize,
  ac: &[i16], angle: isize, ief_params: Option<IntraEdgeFilterParameters>,
  edge_buf: &Aligned<[T; 4 * MAX_TX_SIZE + 1]>, cpu: CpuFeatureLevel,
) {
  let call_native = |dst: &mut PlaneRegionMut<'_, T>| {
    native::dispatch_predict_intra(
      mode, variant, dst, tx_size, bit_depth, ac, angle, ief_params, edge_buf,
      cpu,
    );
  };

  if size_of::<T>() != 1 {
    return call_native(dst);
  }

  unsafe {
    let dst_ptr = dst.data_ptr_mut() as *mut _;
    let stride = dst.plane_cfg.stride as libc::ptrdiff_t;
    let edge_ptr =
      edge_buf.data.as_ptr().offset(2 * MAX_TX_SIZE as isize) as *const _;
    let w = tx_size.width() as libc::c_int;
    let h = tx_size.height() as libc::c_int;
    let angle = angle as libc::c_int;

    if cpu >= CpuFeatureLevel::AVX2 {
      match mode {
        PredictionMode::DC_PRED => {
          (match variant {
            PredictionVariant::NONE => rav1e_ipred_dc_128_avx2,
            PredictionVariant::LEFT => rav1e_ipred_dc_left_avx2,
            PredictionVariant::TOP => rav1e_ipred_dc_top_avx2,
            PredictionVariant::BOTH => rav1e_ipred_dc_avx2,
          })(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::V_PRED if angle == 90 => {
          rav1e_ipred_v_avx2(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::H_PRED if angle == 180 => {
          rav1e_ipred_h_avx2(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::V_PRED
        | PredictionMode::H_PRED
        | PredictionMode::D45_PRED
        | PredictionMode::D135_PRED
        | PredictionMode::D113_PRED
        | PredictionMode::D157_PRED
        | PredictionMode::D203_PRED
        | PredictionMode::D67_PRED => {
          let (enable_ief, ief_smooth_filter) =
            if let Some(params) = ief_params {
              (true as libc::c_int, params.use_smooth_filter() as libc::c_int)
            } else {
              (false as libc::c_int, false as libc::c_int)
            };

          // dav1d assembly uses the unused integer bits to hold IEF parameters
          let angle_arg =
            angle | (enable_ief << 10) | (ief_smooth_filter << 9);

          // From dav1d, bw and bh are the frame width and height rounded to 8px units
          let (bw, bh) = (
            ((dst.plane_cfg.width + 7) >> 3) << 3,
            ((dst.plane_cfg.height + 7) >> 3) << 3,
          );
          // From dav1d, dx and dy are the distance from the predicted block to the frame edge
          let (dx, dy) = (
            (bw as isize - dst.rect().x as isize) as libc::c_int,
            (bh as isize - dst.rect().y as isize) as libc::c_int,
          );

          if angle <= 90 {
            rav1e_ipred_z1_avx2(dst_ptr, stride, edge_ptr, w, h, angle_arg);
          } else if angle < 180 {
            rav1e_ipred_z2_avx2(
              dst_ptr, stride, edge_ptr, w, h, angle_arg, dx, dy,
            );
          } else {
            rav1e_ipred_z3_avx2(dst_ptr, stride, edge_ptr, w, h, angle_arg);
          }
        }
        PredictionMode::SMOOTH_PRED => {
          rav1e_ipred_smooth_avx2(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::SMOOTH_V_PRED => {
          rav1e_ipred_smooth_v_avx2(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::SMOOTH_H_PRED => {
          rav1e_ipred_smooth_h_avx2(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::PAETH_PRED => {
          rav1e_ipred_paeth_avx2(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::UV_CFL_PRED => {
          let ac_ptr = ac.as_ptr() as *const _;
          (match variant {
            PredictionVariant::NONE => rav1e_ipred_cfl_128_avx2,
            PredictionVariant::LEFT => rav1e_ipred_cfl_left_avx2,
            PredictionVariant::TOP => rav1e_ipred_cfl_top_avx2,
            PredictionVariant::BOTH => rav1e_ipred_cfl_avx2,
          })(dst_ptr, stride, edge_ptr, w, h, ac_ptr, angle);
        }
        _ => call_native(dst),
      }
    } else if cpu >= CpuFeatureLevel::SSSE3 {
      match mode {
        PredictionMode::DC_PRED => {
          (match variant {
            PredictionVariant::NONE => rav1e_ipred_dc_128_ssse3,
            PredictionVariant::LEFT => rav1e_ipred_dc_left_ssse3,
            PredictionVariant::TOP => rav1e_ipred_dc_top_ssse3,
            PredictionVariant::BOTH => rav1e_ipred_dc_ssse3,
          })(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::V_PRED if angle == 90 => {
          rav1e_ipred_v_ssse3(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::H_PRED if angle == 180 => {
          rav1e_ipred_h_ssse3(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::SMOOTH_PRED => {
          rav1e_ipred_smooth_ssse3(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::SMOOTH_V_PRED => {
          rav1e_ipred_smooth_v_ssse3(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::SMOOTH_H_PRED => {
          rav1e_ipred_smooth_h_ssse3(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::PAETH_PRED => {
          rav1e_ipred_paeth_ssse3(dst_ptr, stride, edge_ptr, w, h, angle);
        }
        PredictionMode::UV_CFL_PRED => {
          let ac_ptr = ac.as_ptr() as *const _;
          (match variant {
            PredictionVariant::NONE => rav1e_ipred_cfl_128_ssse3,
            PredictionVariant::LEFT => rav1e_ipred_cfl_left_ssse3,
            PredictionVariant::TOP => rav1e_ipred_cfl_top_ssse3,
            PredictionVariant::BOTH => rav1e_ipred_cfl_ssse3,
          })(dst_ptr, stride, edge_ptr, w, h, ac_ptr, angle);
        }
        _ => call_native(dst),
      }
    } else {
      call_native(dst);
    }
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::frame::{AsRegion, Plane};
  use crate::predict::native;
  use num_traits::*;

  #[test]
  fn pred_matches_u8() {
    let tx_size = TxSize::TX_4X4;
    let bit_depth = 8;
    let cpu = CpuFeatureLevel::default();
    let ac = [0i16; 32 * 32];

    let mut edge_buf: Aligned<[u8; 4 * MAX_TX_SIZE + 1]> =
      Aligned::uninitialized();
    for i in 0..edge_buf.data.len() {
      edge_buf.data[i] = (i + 32).saturating_sub(2 * MAX_TX_SIZE).as_();
    }

    for (mode, variant) in [
      (PredictionMode::DC_PRED, PredictionVariant::BOTH),
      (PredictionMode::DC_PRED, PredictionVariant::TOP),
      (PredictionMode::DC_PRED, PredictionVariant::LEFT),
      (PredictionMode::DC_PRED, PredictionVariant::NONE),
      (PredictionMode::V_PRED, PredictionVariant::BOTH),
      (PredictionMode::H_PRED, PredictionVariant::BOTH),
      (PredictionMode::D45_PRED, PredictionVariant::BOTH),
      (PredictionMode::D135_PRED, PredictionVariant::BOTH),
      (PredictionMode::D113_PRED, PredictionVariant::BOTH),
      (PredictionMode::D157_PRED, PredictionVariant::BOTH),
      (PredictionMode::D203_PRED, PredictionVariant::BOTH),
      (PredictionMode::D67_PRED, PredictionVariant::BOTH),
      (PredictionMode::SMOOTH_PRED, PredictionVariant::BOTH),
      (PredictionMode::SMOOTH_V_PRED, PredictionVariant::BOTH),
      (PredictionMode::SMOOTH_H_PRED, PredictionVariant::BOTH),
      (PredictionMode::PAETH_PRED, PredictionVariant::BOTH),
    ]
    .iter()
    {
      let angles = match mode {
        PredictionMode::V_PRED => [81, 84, 87, 90, 93, 96, 99].iter(),
        PredictionMode::H_PRED => [171, 174, 177, 180, 183, 186, 189].iter(),
        PredictionMode::D45_PRED => [36, 39, 42, 45, 48, 51, 54].iter(),
        PredictionMode::D135_PRED => {
          [126, 129, 132, 135, 138, 141, 144].iter()
        }
        PredictionMode::D113_PRED => {
          [104, 107, 110, 113, 116, 119, 122].iter()
        }
        PredictionMode::D157_PRED => {
          [148, 151, 154, 157, 160, 163, 166].iter()
        }
        PredictionMode::D203_PRED => {
          [194, 197, 200, 203, 206, 209, 212].iter()
        }
        PredictionMode::D67_PRED => [58, 61, 64, 67, 70, 73, 76].iter(),
        _ => [0].iter(),
      };
      for angle in angles {
        let expected = {
          let mut plane = Plane::wrap(vec![0u8; 4 * 4], 4);
          native::dispatch_predict_intra(
            *mode,
            *variant,
            &mut plane.as_region_mut(),
            tx_size,
            bit_depth,
            &ac,
            *angle,
            None,
            &edge_buf,
            cpu,
          );
          let mut data = [0u8; 4 * 4];
          for (v, d) in data.iter_mut().zip(plane.data[..].iter()) {
            *v = *d;
          }
          data
        };

        let mut output = Plane::wrap(vec![0u8; 4 * 4], 4);
        dispatch_predict_intra(
          *mode,
          *variant,
          &mut output.as_region_mut(),
          tx_size,
          bit_depth,
          &ac,
          *angle,
          None,
          &edge_buf,
          cpu,
        );
        assert_eq!(
          expected,
          &output.data[..],
          "mode={:?} variant={:?} angle={}",
          *mode,
          *variant,
          *angle
        );
      }
    }
  }
}
