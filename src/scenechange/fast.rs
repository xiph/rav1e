use std::{cmp, sync::Arc};

use crate::{
  api::SceneDetectionSpeed,
  encoder::Sequence,
  frame::{Frame, Plane},
  sad_plane,
  scenechange::fast_idiv,
};
use debug_unreachable::debug_unreachable;
use v_frame::pixel::Pixel;

use super::{ScaleFunction, SceneChangeDetector, ScenecutResult};

/// Experiments have determined this to be an optimal threshold
pub(super) const FAST_THRESHOLD: f64 = 18.0;

impl<T: Pixel> SceneChangeDetector<T> {
  /// The fast algorithm detects fast cuts using a raw difference
  /// in pixel values between the scaled frames.
  #[profiling::function]
  pub(super) fn fast_scenecut(
    &mut self, frame1: Arc<Frame<T>>, frame2: Arc<Frame<T>>,
  ) -> ScenecutResult {
    if let Some(scale_func) = &self.scale_func {
      // downscale both frames for faster comparison
      if let Some(frame_buffer) = &mut self.downscaled_frame_buffer {
        frame_buffer.swap(0, 1);
        (scale_func.downscale_in_place)(
          &frame2.planes[0],
          &mut frame_buffer[1],
        );
      } else {
        self.downscaled_frame_buffer = Some([
          (scale_func.downscale)(&frame1.planes[0]),
          (scale_func.downscale)(&frame2.planes[0]),
        ]);
      }

      if let Some(frame_buffer) = &self.downscaled_frame_buffer {
        let &[first, second] = &frame_buffer;
        let delta = self.delta_in_planes(first, second);

        ScenecutResult {
          threshold: self.threshold,
          inter_cost: delta,
          imp_block_cost: delta,
          forward_adjusted_cost: delta,
          backward_adjusted_cost: delta,
        }
      } else {
        // SAFETY: `downscaled_frame_buffer` is always initialized to `Some(..)` with a valid state
        // before this if/else block is reached.
        unsafe { debug_unreachable!() }
      }
    } else {
      let delta = self.delta_in_planes(&frame1.planes[0], &frame2.planes[0]);

      ScenecutResult {
        threshold: self.threshold,
        inter_cost: delta,
        imp_block_cost: delta,
        backward_adjusted_cost: delta,
        forward_adjusted_cost: delta,
      }
    }
  }

  /// Calculates the average sum of absolute difference (SAD) per pixel between 2 planes
  #[profiling::function]
  fn delta_in_planes(&self, plane1: &Plane<T>, plane2: &Plane<T>) -> f64 {
    let delta = sad_plane::sad_plane(plane1, plane2, self.cpu_feature_level);

    delta as f64 / self.pixels as f64
  }
}

/// Scaling factor for frame in scene detection
pub(super) fn detect_scale_factor<T: Pixel>(
  sequence: &Arc<Sequence>, speed_mode: SceneDetectionSpeed,
) -> Option<ScaleFunction<T>> {
  let small_edge =
    cmp::min(sequence.max_frame_height, sequence.max_frame_width) as usize;
  let scale_func = if speed_mode == SceneDetectionSpeed::Fast {
    match small_edge {
      0..=240 => None,
      241..=480 => Some(ScaleFunction::from_scale::<2>()),
      481..=720 => Some(ScaleFunction::from_scale::<4>()),
      721..=1080 => Some(ScaleFunction::from_scale::<8>()),
      1081..=1600 => Some(ScaleFunction::from_scale::<16>()),
      1601..=usize::MAX => Some(ScaleFunction::from_scale::<32>()),
      _ => None,
    }
  } else {
    None
  };

  if let Some(scale_factor) = scale_func.as_ref().map(|x| x.factor) {
    debug!(
      "Scene detection scale factor {}, [{},{}] -> [{},{}]",
      scale_factor,
      sequence.max_frame_width,
      sequence.max_frame_height,
      fast_idiv(sequence.max_frame_width as usize, scale_factor),
      fast_idiv(sequence.max_frame_height as usize, scale_factor)
    );
  }

  scale_func
}
