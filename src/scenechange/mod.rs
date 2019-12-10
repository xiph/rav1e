// Copyright (c) 2018-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::api::lookahead::*;
use crate::api::{EncoderConfig, InterConfig};
use crate::cpu_features::CpuFeatureLevel;
use crate::encoder::Sequence;
use crate::frame::*;
use crate::hawktracer::*;
use crate::util::CastFromPrimitive;
use crate::util::Pixel;
use std::collections::BTreeSet;
use std::sync::Arc;

/// Runs keyframe detection on frames from the lookahead queue.
pub(crate) struct SceneChangeDetector {
  /// Minimum average difference between YUV deltas that will trigger a scene change.
  threshold: u64,
  /// Fast scene cut detection mode, uses simple SAD instead of encoder cost estimates.
  fast_mode: bool,
  /// Frames that cannot be marked as keyframes due to the algorithm excluding them.
  /// Storing the frame numbers allows us to avoid looking back more than one frame.
  excluded_frames: BTreeSet<u64>,
  /// The bit depth of the video.
  bit_depth: usize,
  /// The CPU feature level to be used.
  cpu_feature_level: CpuFeatureLevel,
  encoder_config: EncoderConfig,
  sequence: Sequence,
}

impl SceneChangeDetector {
  pub fn new(
    bit_depth: usize, fast_mode: bool, cpu_feature_level: CpuFeatureLevel,
    encoder_config: EncoderConfig, sequence: Sequence,
  ) -> Self {
    // This implementation is based on a Python implementation at
    // https://pyscenedetect.readthedocs.io/en/latest/reference/detection-methods/.
    // The Python implementation uses HSV values and a threshold of 30. Comparing the
    // YUV values was sufficient in most cases, and avoided a more costly YUV->RGB->HSV
    // conversion, but the deltas needed to be scaled down. The deltas for keyframes
    // in YUV were about 1/3 to 1/2 of what they were in HSV, but non-keyframes were
    // very unlikely to have a delta greater than 3 in YUV, whereas they may reach into
    // the double digits in HSV. Therefore, 12 was chosen as a reasonable default threshold.
    // This may be adjusted later.
    //
    // This threshold is only used for the fast scenecut implementation.
    const BASE_THRESHOLD: u64 = 12;
    Self {
      threshold: BASE_THRESHOLD * bit_depth as u64 / 8,
      fast_mode,
      excluded_frames: BTreeSet::new(),
      bit_depth,
      cpu_feature_level,
      encoder_config,
      sequence,
    }
  }

  /// Runs keyframe detection on the next frame in the lookahead queue.
  ///
  /// This function requires that a subset of input frames
  /// is passed to it in order, and that `keyframes` is only
  /// updated from this method. `input_frameno` should correspond
  /// to the second frame in `frame_set`.
  ///
  /// This will gracefully handle the first frame in the video as well.
  #[hawktracer(analyze_next_frame)]
  pub fn analyze_next_frame<T: Pixel>(
    &mut self, frame_set: &[Arc<Frame<T>>], input_frameno: u64,
    previous_keyframe: u64, config: &EncoderConfig, inter_cfg: &InterConfig,
  ) -> bool {
    // Find the distance to the previous keyframe.
    let distance = input_frameno - previous_keyframe;

    // Handle minimum and maximum key frame intervals.
    if distance < config.min_key_frame_interval {
      return false;
    }
    if distance >= config.max_key_frame_interval {
      return true;
    }

    if config.speed_settings.no_scene_detection {
      return false;
    }

    self.exclude_scene_flashes(
      &frame_set,
      input_frameno,
      inter_cfg,
      previous_keyframe,
    );

    self.is_key_frame(
      frame_set[0].clone(),
      frame_set[1].clone(),
      input_frameno,
      previous_keyframe,
    )
  }

  /// Determines if `current_frame` should be a keyframe.
  fn is_key_frame<T: Pixel>(
    &self, previous_frame: Arc<Frame<T>>, current_frame: Arc<Frame<T>>,
    current_frameno: u64, previous_keyframe: u64,
  ) -> bool {
    if self.excluded_frames.contains(&current_frameno) {
      return false;
    }

    let result = self.has_scenecut(
      previous_frame,
      current_frame,
      current_frameno,
      previous_keyframe,
    );
    debug!(
      "[SC-Detect] Frame {} to {}: I={:.3} T={:.3} P={:.3} {}",
      current_frameno - 1,
      current_frameno,
      result.intra_cost,
      result.threshold,
      result.inter_cost,
      if result.has_scenecut { "Scenecut" } else { "No cut" }
    );
    result.has_scenecut
  }

  /// Uses lookahead to avoid coding short flashes as scenecuts.
  /// Saves excluded frame numbers in `self.excluded_frames`.
  fn exclude_scene_flashes<T: Pixel>(
    &mut self, frame_subset: &[Arc<Frame<T>>], frameno: u64,
    inter_cfg: &InterConfig, previous_keyframe: u64,
  ) {
    let lookahead_distance = inter_cfg.keyframe_lookahead_distance() as usize;

    if frame_subset.len() - 1 < lookahead_distance {
      // Don't add a keyframe in the last frame pyramid.
      // It's effectively the same as a scene flash,
      // and really wasteful for compression.
      for frame in
        frameno..=(frameno + inter_cfg.keyframe_lookahead_distance())
      {
        self.excluded_frames.insert(frame);
      }
      return;
    }

    // Where A and B are scenes: AAAAAABBBAAAAAA
    // If BBB is shorter than lookahead_distance, it is detected as a flash
    // and not considered a scenecut.
    //
    // Search starting with the furthest frame,
    // to enable early loop exit if we find a scene flash.
    for j in (1..=lookahead_distance).rev() {
      let result = self.has_scenecut(
        frame_subset[0].clone(),
        frame_subset[j].clone(),
        frameno - 1 + j as u64,
        previous_keyframe,
      );
      debug!(
        "[SF-Detect-1] Frame {} to {}: I={:.3} T={:.3} P={:.3} {}",
        frameno - 1,
        frameno - 1 + j as u64,
        result.intra_cost,
        result.threshold,
        result.inter_cost,
        if result.has_scenecut { "No flash" } else { "Scene flash" }
      );
      if !result.has_scenecut {
        // Any frame in between `0` and `j` cannot be a real scenecut.
        for i in 0..=j {
          let frameno = frameno + i as u64 - 1;
          self.excluded_frames.insert(frameno);
        }
        // Because all frames in this gap are already excluded,
        // exit the loop early as an optimization.
        break;
      }
    }

    // Where A-F are scenes: AAAAABBCCDDEEFFFFFF
    // If each of BB ... EE are shorter than `lookahead_distance`, they are
    // detected as flashes and not considered scenecuts.
    // Instead, the first F frame becomes a scenecut.
    // If the video ends before F, no frame becomes a scenecut.
    for i in 1..lookahead_distance {
      let result = self.has_scenecut(
        frame_subset[i].clone(),
        frame_subset[lookahead_distance].clone(),
        frameno - 1 + lookahead_distance as u64,
        previous_keyframe,
      );
      debug!(
        "[SF-Detect-2] Frame {} to {}: I={:.3} T={:.3} P={:.3} {}",
        frameno - 1 + i as u64,
        frameno - 1 + lookahead_distance as u64,
        result.intra_cost,
        result.threshold,
        result.inter_cost,
        if result.has_scenecut { "Scene flash" } else { "No flash" }
      );
      if result.has_scenecut {
        // If the current frame is the frame before a scenecut, it cannot also be the frame of a scenecut.
        let frameno = frameno + i as u64 - 1;
        self.excluded_frames.insert(frameno);
      }
    }
  }

  /// Run a comparison between two frames to determine if they qualify for a scenecut.
  ///
  /// The standard algorithm uses block intra and inter costs
  /// to determine which method would be more efficient
  /// for coding this frame.
  ///
  /// The fast algorithm detects fast cuts using a raw difference
  /// in pixel values between the frames.
  /// It does not handle pans well, but the scene flash detection compensates for this
  /// in many cases.
  fn has_scenecut<T: Pixel>(
    &self, frame1: Arc<Frame<T>>, frame2: Arc<Frame<T>>, frameno: u64,
    previous_keyframe: u64,
  ) -> ScenecutResult {
    if self.fast_mode {
      let len = frame2.planes[0].cfg.width * frame2.planes[0].cfg.height;
      let delta = self.delta_in_planes(&frame1.planes[0], &frame2.planes[0]);
      let threshold = self.threshold * len as u64;
      ScenecutResult {
        intra_cost: threshold as f64,
        threshold: threshold as f64,
        inter_cost: delta as f64,
        has_scenecut: delta >= threshold,
      }
    } else {
      let intra_costs =
        estimate_intra_costs(&*frame2, self.bit_depth, self.cpu_feature_level);
      let intra_cost = intra_costs.iter().map(|&cost| cost as u64).sum::<u64>()
        as f64
        / intra_costs.len() as f64;

      let inter_costs = estimate_inter_costs(
        frame2,
        frame1,
        self.bit_depth,
        self.encoder_config,
        self.sequence,
      );
      let inter_cost = inter_costs.iter().map(|&cost| cost as u64).sum::<u64>()
        as f64
        / inter_costs.len() as f64;

      // Sliding scale, more likely to choose a keyframe
      // as we get farther from the last keyframe.
      // Based on x264 scenecut code.
      //
      // `THRESH_MAX` determines how likely we are
      // to choose a keyframe, between 0.0-1.0.
      // Higher values mean we are more likely to choose a keyframe.
      // `0.4` was chosen based on trials of the `scenecut-720p` set in AWCY,
      // as it appeared to provide the best average compression.
      // This also matches the default scenecut threshold in x264.
      const THRESH_MAX: f64 = 0.4;
      const THRESH_MIN: f64 = THRESH_MAX * 0.25;
      let distance_from_keyframe = frameno - previous_keyframe;
      let min_keyint = self.encoder_config.min_key_frame_interval;
      let max_keyint = self.encoder_config.max_key_frame_interval;
      let bias = if distance_from_keyframe <= min_keyint / 4 {
        THRESH_MIN / 4.0
      } else if distance_from_keyframe <= min_keyint {
        THRESH_MIN * distance_from_keyframe as f64 / min_keyint as f64
      } else {
        THRESH_MIN
          + (THRESH_MAX - THRESH_MIN)
            * (distance_from_keyframe - min_keyint) as f64
            / (max_keyint - min_keyint) as f64
      };
      let threshold = intra_cost * (1.0 - bias);

      ScenecutResult {
        intra_cost,
        threshold,
        inter_cost,
        has_scenecut: inter_cost > threshold,
      }
    }
  }

  fn delta_in_planes<T: Pixel>(
    &self, plane1: &Plane<T>, plane2: &Plane<T>,
  ) -> u64 {
    let mut delta = 0;
    let lines = plane1.rows_iter().zip(plane2.rows_iter());

    for (l1, l2) in lines {
      let delta_line = l1
        .iter()
        .zip(l2.iter())
        .map(|(&p1, &p2)| {
          (i16::cast_from(p1) - i16::cast_from(p2)).abs() as u64
        })
        .sum::<u64>();
      delta += delta_line;
    }
    delta
  }
}

/// This struct primarily exists for returning metrics to the caller
/// for logging debug information.
#[derive(Debug, Clone, Copy)]
struct ScenecutResult {
  intra_cost: f64,
  inter_cost: f64,
  threshold: f64,
  has_scenecut: bool,
}
