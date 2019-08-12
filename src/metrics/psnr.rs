use crate::api::ChromaSampling;
use crate::frame::Frame;
use crate::frame::Plane;
use crate::metrics::FrameMetrics;
use crate::util::{CastFromPrimitive, Pixel};

/// Calculates the PSNR for a `Frame` by comparing the original (uncompressed) to the compressed
/// version of the frame. Higher PSNR is better--PSNR is capped at 100 in order to avoid skewed
/// statistics from e.g. all black frames, which would otherwise show a PSNR of infinity.
///
/// See https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio for more details.
pub(crate) fn calculate_frame_psnr<T: Pixel>(
  frame1: &Frame<T>, frame2: &Frame<T>, bit_depth: usize, cs: ChromaSampling,
) -> FrameMetrics {
  let y =
    calculate_plane_psnr(&frame1.planes[0], &frame2.planes[0], bit_depth);
  let u =
    calculate_plane_psnr(&frame1.planes[1], &frame2.planes[1], bit_depth);
  let v =
    calculate_plane_psnr(&frame1.planes[2], &frame2.planes[2], bit_depth);
  FrameMetrics { y, u, v, weighted_avg: weighted_average((y, u, v), cs) }
}

/// Calculate the PSNR for a `Plane` by comparing the original (uncompressed) to the compressed
/// version.
fn calculate_plane_psnr<T: Pixel>(
  frame1: &Plane<T>, frame2: &Plane<T>, bit_depth: usize,
) -> f64 {
  let mse = calculate_plane_mse(frame1, frame2);
  if mse <= std::f64::EPSILON {
    return 100.0;
  }
  let max = ((1 << bit_depth) - 1) as f64;
  20.0 * max.log10() - 10.0 * mse.log10()
}

/// Calculate the mean squared error for a `Plane` by comparing the original (uncompressed)
/// to the compressed version.
fn calculate_plane_mse<T: Pixel>(frame1: &Plane<T>, frame2: &Plane<T>) -> f64 {
  frame1
    .iter()
    .zip(frame2.iter())
    .map(|(a, b)| (i32::cast_from(a) - i32::cast_from(b)).abs() as u64)
    .map(|err| err * err)
    .sum::<u64>() as f64
    / (frame1.cfg.width * frame1.cfg.height) as f64
}

fn weighted_average(results: (f64, f64, f64), cs: ChromaSampling) -> f64 {
  let cweight = cs.get_chroma_weight();
  (results.0 + cweight * (results.1 + results.2)) / (1.0 + 2.0 * cweight)
}
