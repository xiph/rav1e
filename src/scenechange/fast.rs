use std::{cmp, sync::Arc};

use crate::{
  api::SceneDetectionSpeed,
  encoder::Sequence,
  frame::{Frame, Plane},
  sad_row,
  scenechange::fast_idiv,
};
use debug_unreachable::debug_unreachable;
use rust_hawktracer::*;
use v_frame::pixel::Pixel;

use super::{ScaleFunction, SceneChangeDetector, ScenecutResult};

/// Experiments have determined this to be an optimal threshold
pub(super) const FAST_THRESHOLD: f64 = 18.0;

impl<T: Pixel> SceneChangeDetector<T> {
  /// The fast algorithm detects fast cuts using a raw difference
  /// in pixel values between the scaled frames.
  #[hawktracer(fast_scenecut)]
  pub(super) fn fast_scenecut(
    &mut self, frame1: Arc<Frame<T>>, frame2: Arc<Frame<T>>,
  ) -> ScenecutResult {
    if let Some(scale_func) = &self.scale_func {
      // downscale both frames for faster comparison
      if let Some((frame_buffer, is_initialized)) =
        &mut self.downscaled_frame_buffer
      {
        let frame_buffer = &mut *frame_buffer;
        if *is_initialized {
          frame_buffer.swap(0, 1);
          (scale_func.downscale_in_place)(
            &frame2.planes[0],
            &mut frame_buffer[1],
          );
        } else {
          // both frames are in an irrelevant and invalid state, so we have to reinitialize
          // them, but we can reuse their allocations
          (scale_func.downscale_in_place)(
            &frame1.planes[0],
            &mut frame_buffer[0],
          );
          (scale_func.downscale_in_place)(
            &frame2.planes[0],
            &mut frame_buffer[1],
          );
          *is_initialized = true;
        }
      } else {
        self.downscaled_frame_buffer = Some((
          [
            (scale_func.downscale)(&frame1.planes[0]),
            (scale_func.downscale)(&frame2.planes[0]),
          ],
          true, // the frame buffer is initialized and in a valid state
        ));
      }

      if let Some((frame_buffer, _)) = &self.downscaled_frame_buffer {
        let frame_buffer = &*frame_buffer;
        let delta = self.delta_in_planes(&frame_buffer[0], &frame_buffer[1]);

        ScenecutResult {
          threshold: self.threshold as f64,
          inter_cost: delta as f64,
          imp_block_cost: delta as f64,
          forward_adjusted_cost: delta as f64,
          backward_adjusted_cost: delta as f64,
        }
      } else {
        // SAFETY: `downscaled_frame_buffer` is always initialized to `Some(..)` with a valid state
        // before this if/else block is reached.
        unsafe { debug_unreachable!() }
      }
    } else {
      if let Some(frame_buffer) = &mut self.frame_ref_buffer {
        frame_buffer.swap(0, 1);
        frame_buffer[1] = frame2;
      } else {
        self.frame_ref_buffer = Some([frame1, frame2]);
      }

      if let Some(frame_buffer) = &self.frame_ref_buffer {
        let delta = self.delta_in_planes(
          &frame_buffer[0].planes[0],
          &frame_buffer[1].planes[0],
        );

        ScenecutResult {
          threshold: self.threshold as f64,
          inter_cost: delta as f64,
          imp_block_cost: delta as f64,
          backward_adjusted_cost: delta as f64,
          forward_adjusted_cost: delta as f64,
        }
      } else {
        // SAFETY: `frame_ref_buffer` is always initialized to `Some(..)` at the start
        // of this code block if it was `None`.
        unsafe { debug_unreachable!() }
      }
    }
  }

  /// Calculates the average sum of absolute difference (SAD) per pixel between 2 planes
  #[hawktracer(delta_in_planes)]
  fn delta_in_planes(&self, plane1: &Plane<T>, plane2: &Plane<T>) -> f64 {
    let mut delta = 0;

    let lines = plane1.rows_iter().zip(plane2.rows_iter());

    for (l1, l2) in lines {
      let l1 = l1.get(..plane1.cfg.width).unwrap_or(l1);
      let l2 = l2.get(..plane1.cfg.width).unwrap_or(l2);
      delta += sad_row::sad_row(l1, l2, self.cpu_feature_level);
    }
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
      // SAFETY: We ensure that scale_factor is set based on nonzero powers of 2.
      unsafe { fast_idiv(sequence.max_frame_width as usize, scale_factor) },
      unsafe { fast_idiv(sequence.max_frame_height as usize, scale_factor) }
    );
  }

  scale_func
}
