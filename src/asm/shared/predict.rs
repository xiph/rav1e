// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#[cfg(test)]
mod test {
  use crate::context::MAX_TX_SIZE;
  use crate::cpu_features::CpuFeatureLevel;
  use crate::frame::{AsRegion, Plane};
  use crate::predict::dispatch_predict_intra;
  use crate::predict::rust;
  use crate::predict::{
    IntraEdgeFilterParameters, PredictionMode, PredictionVariant,
  };
  use crate::transform::TxSize;
  use crate::util::Aligned;
  use crate::Pixel;

  #[test]
  fn pred_matches() {
    for cpu in
      &CpuFeatureLevel::all()[..=CpuFeatureLevel::default().as_index()]
    {
      pred_matches_inner::<u8>(*cpu, 8);
      pred_matches_inner::<u16>(*cpu, 10);
      pred_matches_inner::<u16>(*cpu, 12);
    }
  }

  fn pred_matches_inner<T: Pixel>(cpu: CpuFeatureLevel, bit_depth: usize) {
    let tx_size = TxSize::TX_4X4;
    // SAFETY: We write to the array below before reading from it.
    let mut ac: Aligned<[i16; 32 * 32]> = unsafe { Aligned::uninitialized() };
    for i in 0..ac.data.len() {
      ac.data[i] = i as i16 - 16 * 32;
    }
    // SAFETY: We write to the array below before reading from it.
    let mut edge_buf: Aligned<[T; 4 * MAX_TX_SIZE + 1]> =
      unsafe { Aligned::uninitialized() };
    for i in 0..edge_buf.data.len() {
      edge_buf.data[i] =
        T::cast_from(((i ^ 1) + 32).saturating_sub(2 * MAX_TX_SIZE));
    }

    let ief_params_all = [
      None,
      Some(IntraEdgeFilterParameters::default()),
      Some(IntraEdgeFilterParameters {
        above_mode: Some(PredictionMode::SMOOTH_PRED),
        left_mode: Some(PredictionMode::SMOOTH_PRED),
        ..Default::default()
      }),
    ];

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
      (PredictionMode::UV_CFL_PRED, PredictionVariant::BOTH),
      (PredictionMode::UV_CFL_PRED, PredictionVariant::TOP),
      (PredictionMode::UV_CFL_PRED, PredictionVariant::LEFT),
      (PredictionMode::UV_CFL_PRED, PredictionVariant::NONE),
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
        PredictionMode::UV_CFL_PRED => [
          -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 1,
          2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        ]
        .iter(),
        _ => [0].iter(),
      };
      for angle in angles {
        for ief_params in match mode {
          PredictionMode::V_PRED if *angle == 90 => [None].iter(),
          PredictionMode::H_PRED if *angle == 180 => [None].iter(),
          PredictionMode::V_PRED
          | PredictionMode::H_PRED
          | PredictionMode::D45_PRED
          | PredictionMode::D135_PRED
          | PredictionMode::D113_PRED
          | PredictionMode::D157_PRED
          | PredictionMode::D203_PRED
          | PredictionMode::D67_PRED => ief_params_all.iter(),
          _ => [None].iter(),
        } {
          let expected = {
            let mut plane = Plane::from_slice(&[T::zero(); 4 * 4], 4);
            rust::dispatch_predict_intra(
              *mode,
              *variant,
              &mut plane.as_region_mut(),
              tx_size,
              bit_depth,
              &ac.data,
              *angle,
              *ief_params,
              &edge_buf,
              cpu,
            );
            let mut data = [T::zero(); 4 * 4];
            for (v, d) in data.iter_mut().zip(plane.data[..].iter()) {
              *v = *d;
            }
            data
          };

          let mut output = Plane::from_slice(&[T::zero(); 4 * 4], 4);
          dispatch_predict_intra(
            *mode,
            *variant,
            &mut output.as_region_mut(),
            tx_size,
            bit_depth,
            &ac.data,
            *angle,
            *ief_params,
            &edge_buf,
            cpu,
          );
          assert_eq!(
            expected,
            &output.data[..],
            "mode={:?} variant={:?} angle={} ief_params.use_smooth_filter={:?} bit_depth={} cpu={:?}",
            *mode,
            *variant,
            *angle,
            ief_params.map(|p| p.use_smooth_filter()),
            bit_depth,
            cpu
          );
        }
      }
    }
  }
}
