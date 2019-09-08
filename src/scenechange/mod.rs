// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::api::InterConfig;
use crate::api::{EncoderConfig, FrameType};
use crate::util::Pixel;

use crate::encoder::FrameInvariants;
use std::cmp;
use std::collections::BTreeSet;

const SCENECUT_THRESHOLD: f32 = 40.;

/// Runs keyframe detection on frames from the lookahead queue.
#[derive(Default)]
pub(crate) struct SceneChangeDetector {
  /// Frames that cannot be marked as keyframes due to the algorithm excluding them.
  /// Storing the frame numbers allows us to avoid looking back more than one frame.
  excluded_frames: BTreeSet<u64>,
}

impl SceneChangeDetector {
  /// Runs keyframe detection on the next frame in the lookahead queue.
  ///
  /// This function requires that a subset of input frames
  /// is passed to it in order, and that `keyframes` is only
  /// updated from this method. `input_frameno` should correspond
  /// to the first frame in `frame_set`.
  ///
  /// This will gracefully handle the first frame in the video as well.
  pub fn analyze_next_frame<T: Pixel>(
    &mut self, frame_set: &[&FrameInvariants<T>], config: &EncoderConfig,
    inter_cfg: &InterConfig, keyframes: &mut BTreeSet<u64>,
    keyframes_forced: &BTreeSet<u64>,
  ) {
    let input_frameno = frame_set[0].input_frameno;
    if keyframes.iter().any(|f| *f >= input_frameno) {
      // Already analyzed past this frame--this is only a concern due to
      // frame reordering.
      return;
    }

    self.exclude_scene_flashes(
      &frame_set,
      input_frameno,
      keyframes,
      inter_cfg,
      config,
    );

    if self.is_key_frame(
      &frame_set[0],
      input_frameno,
      config,
      keyframes,
      keyframes_forced,
    ) {
      keyframes.insert(input_frameno);
    }
  }

  /// Determines if `current_frame` should be a keyframe.
  fn is_key_frame<T: Pixel>(
    &self, current_frame: &FrameInvariants<T>, current_frameno: u64,
    config: &EncoderConfig, keyframes: &mut BTreeSet<u64>,
    keyframes_forced: &BTreeSet<u64>,
  ) -> bool {
    if keyframes_forced.contains(&current_frameno) {
      return true;
    }

    // Find the distance to the previous keyframe.
    let previous_keyframe = keyframes.iter().last().unwrap();
    let distance = current_frameno - previous_keyframe;

    // Handle minimum and maximum key frame intervals.
    if distance < config.min_key_frame_interval {
      return false;
    }
    if distance == config.max_key_frame_interval {
      return true;
    }
    if distance > config.max_key_frame_interval {
      // This resolves an issue where more frames than necessary
      // would be selected as keyframes, due to scene analysis not
      // always being performed in input frame order.
      //
      // The frame will be properly analyzed on the next pass
      // after the FIs are reset.
      return false;
    }

    // Skip smart scene detection if it's disabled
    if config.speed_settings.no_scene_detection {
      return false;
    }

    if self.excluded_frames.contains(&current_frameno) {
      return false;
    }

    self.has_scenecut(current_frame, 1, keyframes, config)
  }

  /// Uses lookahead to avoid coding short flashes as scenecuts.
  /// Saves excluded frame numbers in `self.excluded_frames`.
  fn exclude_scene_flashes<T: Pixel>(
    &mut self, frame_subset: &[&FrameInvariants<T>], frameno: u64,
    keyframes: &mut BTreeSet<u64>, inter_cfg: &InterConfig,
    config: &EncoderConfig,
  ) {
    let lookahead_distance = cmp::min(
      inter_cfg.keyframe_lookahead_distance() as usize,
      frame_subset.len(),
    );

    // Where A and B are scenes: AAAAAABBBAAAAAA
    // If BBB is shorter than lookahead_distance, it is detected as a flash
    // and not considered a scenecut.
    if lookahead_distance > 1 && frameno > 0 {
      for j in 1..lookahead_distance {
        if !self.has_scenecut(&frame_subset[j], j + 1, keyframes, config) {
          // Any frame in between the previous frame and `j` cannot be a real scenecut.
          for i in 0..=(j + 1) {
            let frameno = frameno + i as u64 - 1;
            self.excluded_frames.insert(frameno);
          }
        }
      }
    }

    // Where A-F are scenes: AAAAABBCCDDEEFFFFFF
    // If each of BB ... EE are shorter than `lookahead_distance`, they are
    // detected as flashes and not considered scenecuts.
    // Instead, the first F frame becomes a scenecut.
    // If the video ends before F, no frame becomes a scenecut.
    for i in 0..lookahead_distance {
      if self.has_scenecut(
        &frame_subset[lookahead_distance - 1],
        lookahead_distance - i,
        keyframes,
        config,
      ) {
        // If the current frame is the frame before a scenecut, it cannot also be the frame of a scenecut.
        let frameno = frameno + i as u64 - 1;
        self.excluded_frames.insert(frameno);
      }
    }
  }

  /// Check the inter costs between two frames to determine if they qualify for a scenecut.
  /// Based on the x264 algorithm.
  fn has_scenecut<T: Pixel>(
    &self, fi: &FrameInvariants<T>, ref_frame_distance: usize,
    keyframes: &BTreeSet<u64>, config: &EncoderConfig,
  ) -> bool {
    debug_assert!(fi.input_frameno >= ref_frame_distance as u64);
    if fi.frame_type == FrameType::KEY {
      return true;
    }

    let intra_cost =
      &fi.lookahead_intra_costs.iter().map(|&v| v as f32).sum::<f32>()
        / fi.lookahead_intra_costs.len() as f32;
    let inter_cost = &fi.lookahead_inter_costs[ref_frame_distance - 1]
      .iter()
      .map(|&v| v as f32)
      .sum::<f32>()
      / fi.lookahead_inter_costs[ref_frame_distance - 1].len() as f32;
    // Avoid a division by zero error
    if intra_cost < std::f32::EPSILON {
      if inter_cost < std::f32::EPSILON {
        // This is going to be all-SKIP anyway, so mark it a non-scenechange.
        debug!(
          "frame {} to {}: no scenecut; icost 0; pcost 0",
          fi.input_frameno,
          fi.input_frameno - ref_frame_distance as u64
        );
        return false;
      } else {
        debug!(
          "frame {} to {}: scenecut; icost 0; pcost {:.3}",
          fi.input_frameno,
          fi.input_frameno - ref_frame_distance as u64,
          inter_cost
        );
        return true;
      }
    }

    let distance_from_last_keyframe =
      fi.input_frameno - *keyframes.iter().last().unwrap();
    let thresh_max = SCENECUT_THRESHOLD / 100.;
    let thresh_min = thresh_max * 0.25;

    let bias = if distance_from_last_keyframe
      <= config.min_key_frame_interval / 4
    {
      thresh_min / 4.
    } else if distance_from_last_keyframe <= config.min_key_frame_interval {
      thresh_min * distance_from_last_keyframe as f32
        / config.min_key_frame_interval as f32
    } else if config.min_key_frame_interval == config.max_key_frame_interval {
      thresh_max
    } else {
      thresh_min
        + (thresh_max - thresh_min)
          * (distance_from_last_keyframe - config.min_key_frame_interval)
            as f32
          / (config.max_key_frame_interval - config.min_key_frame_interval)
            as f32
    };

    let is_scenecut = inter_cost >= (1. - bias) * intra_cost;
    debug!(
      "frame {} to {}: {}; icost {}; pcost {}; ratio {:.3}; thresh {:.3}",
      fi.input_frameno,
      fi.input_frameno - ref_frame_distance as u64,
      if is_scenecut { "scenecut" } else { "no scenecut" },
      intra_cost,
      inter_cost,
      1. - inter_cost / intra_cost,
      bias
    );
    is_scenecut
  }
}
