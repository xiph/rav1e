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
  use crate::predict::{PredictionMode, PredictionVariant};
  use crate::transform::TxSize;
  use crate::util::Aligned;
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
          let mut plane = Plane::from_slice(&vec![0u8; 4 * 4], 4);
          rust::dispatch_predict_intra(
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

        let mut output = Plane::from_slice(&vec![0u8; 4 * 4], 4);
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
