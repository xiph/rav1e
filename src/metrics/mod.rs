// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::api::ChromaSampling;
use crate::frame::Frame;
use crate::util::Pixel;

mod ciede;
mod psnr;
mod psnr_hvs;
mod ssim;

#[derive(Debug, Clone, Copy)]
pub struct FrameMetrics {
  pub y: f64,
  pub u: f64,
  pub v: f64,
  pub weighted_avg: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct QualityMetrics {
  /// Peak Signal-to-Noise Ratio for Y, U, and V planes
  pub psnr: Option<FrameMetrics>,
  /// Peak Signal-to-Noise Ratio as perceived by the Human Visual System--
  /// taking into account Contrast Sensitivity Function (CSF)
  pub psnr_hvs: Option<FrameMetrics>,
  /// Structural Similarity
  pub ssim: Option<FrameMetrics>,
  /// Multi-Scale Structural Similarity
  pub ms_ssim: Option<FrameMetrics>,
  /// CIEDE 2000 color difference algorithm: https://en.wikipedia.org/wiki/Color_difference#CIEDE2000
  pub ciede: Option<f64>,
  /// Aligned Peak Signal-to-Noise Ratio for Y, U, and V planes
  pub apsnr: Option<FrameMetrics>,
  /// Netflix's Video Multimethod Assessment Fusion
  pub vmaf: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetricsEnabled {
  /// Don't calculate any metrics.
  None,
  /// Calculate the PSNR of each plane, but no other metrics.
  Psnr,
  /// Calculate all implemented metrics. Currently implemented metrics match what is available via AWCY.
  All,
}

pub fn calculate_frame_metrics<T: Pixel>(
  frame1: &Frame<T>, frame2: &Frame<T>, bit_depth: usize, cs: ChromaSampling,
  metrics: MetricsEnabled,
) -> QualityMetrics {
  match metrics {
    MetricsEnabled::None => QualityMetrics::default(),
    MetricsEnabled::Psnr => {
      let mut metrics = QualityMetrics::default();
      metrics.psnr =
        Some(psnr::calculate_frame_psnr(frame1, frame2, bit_depth, cs));
      metrics
    }
    MetricsEnabled::All => {
      let mut metrics = QualityMetrics::default();
      metrics.psnr =
        Some(psnr::calculate_frame_psnr(frame1, frame2, bit_depth, cs));
      metrics.psnr_hvs = Some(psnr_hvs::calculate_frame_psnr_hvs(
        frame1, frame2, bit_depth, cs,
      ));
      let ssim = ssim::calculate_frame_ssim(frame1, frame2, bit_depth, cs);
      metrics.ssim = Some(ssim);
      let ms_ssim =
        ssim::calculate_frame_msssim(frame1, frame2, bit_depth, cs);
      metrics.ms_ssim = Some(ms_ssim);
      let ciede = ciede::calculate_frame_ciede(frame1, frame2, bit_depth);
      metrics.ciede = Some(ciede);
      // TODO: APSNR
      // TODO: VMAF
      metrics
    }
  }
}
