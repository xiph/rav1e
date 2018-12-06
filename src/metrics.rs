// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use encoder::Frame;
use plane::Plane;

/// Calculates the PSNR for a `Frame` by comparing the original (uncompressed) to the compressed
/// version of the frame. Higher PSNR is better--PSNR is capped at 100 in order to avoid skewed
/// statistics from e.g. all black frames, which would otherwise show a PSNR of infinity.
///
/// See https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio for more details.
pub fn calculate_frame_psnr(original: &Frame, compressed: &Frame, bit_depth: usize) -> (f64, f64, f64) {
  (calculate_plane_psnr(&original.planes[0], &compressed.planes[0], bit_depth),
    calculate_plane_psnr(&original.planes[1], &compressed.planes[1], bit_depth),
    calculate_plane_psnr(&original.planes[2], &compressed.planes[2], bit_depth))
}

/// Calculate the PSNR for a `Plane` by comparing the original (uncompressed) to the compressed
/// version.
fn calculate_plane_psnr(original: &Plane, compressed: &Plane, bit_depth: usize) -> f64 {
  let mse = calculate_plane_mse(original, compressed);
  if mse <= 0.0000000001 {
    return 100.0;
  }
  let max = ((1 << bit_depth) - 1) as f64;
  20.0 * max.log10() - 10.0 * mse.log10()
}

/// Calculate the mean squared error for a `Plane` by comparing the original (uncompressed)
/// to the compressed version.
fn calculate_plane_mse(original: &Plane, compressed: &Plane) -> f64 {
  original.iter().zip(compressed.iter())
    .map(|(a, b)| (a as i32 - b as i32).abs() as u64)
    .map(|err| err * err)
    .sum::<u64>() as f64 / (original.cfg.width * original.cfg.height) as f64
}