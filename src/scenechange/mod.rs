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
const FRAME_PCT_THRESHOLD: f32 = 0.75;

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

    let previous_keyframe = keyframes.iter().last().unwrap();
    let distance = input_frameno - previous_keyframe;
    if distance > config.max_key_frame_interval {
      // This resolves an issue where more frames than necessary
      // would be selected as keyframes, due to scene analysis not
      // always being performed in input frame order.
      //
      // The frame will be properly analyzed on the next pass
      // after the FIs are reset.
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
      inter_cfg,
      keyframes,
      keyframes_forced,
    ) {
      keyframes.insert(input_frameno);
    }
  }

  /// Determines if `current_frame` should be a keyframe.
  fn is_key_frame<T: Pixel>(
    &self, current_frame: &FrameInvariants<T>, current_frameno: u64,
    config: &EncoderConfig, inter_cfg: &InterConfig,
    keyframes: &mut BTreeSet<u64>, keyframes_forced: &BTreeSet<u64>,
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

    // Skip smart scene detection if it's disabled
    if config.speed_settings.no_scene_detection {
      return false;
    }

    if self.excluded_frames.contains(&current_frameno) {
      return false;
    }

    self.has_scenecut(current_frame, keyframes, config, inter_cfg, 0)
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
    // if lookahead_distance > 1 && frameno > 0 {
    //   for j in 1..lookahead_distance {
    //     if !self.has_scenecut(
    //       &frame_subset[j],
    //       keyframes,
    //       config,
    //       inter_cfg,
    //       j,
    //     ) {
    //       // Any frame in between the previous frame and `j` cannot be a real scenecut.
    //       for i in 0..=(j + 1) {
    //         let frameno = frameno + i as u64 - 1;
    //         self.excluded_frames.insert(frameno);
    //       }
    //     }
    //   }
    // }

    // Where A-F are scenes: AAAAABBCCDDEEFFFFFF
    // If each of BB ... EE are shorter than `lookahead_distance`, they are
    // detected as flashes and not considered scenecuts.
    // Instead, the first F frame becomes a scenecut.
    // If the video ends before F, no frame becomes a scenecut.
    // for i in 0..lookahead_distance {
    //   if self.has_scenecut(
    //     &frame_subset[lookahead_distance - 1],
    //     lookahead_distance - i,
    //     keyframes,
    //     config,
    //   ) {
    //     // If the current frame is the frame before a scenecut, it cannot also be the frame of a scenecut.
    //     let frameno = frameno + i as u64 - 1;
    //     self.excluded_frames.insert(frameno);
    //   }
    // }
  }

  /// Check a frame to see if it qualifies as a scenecut.
  /// At its core, this uses x264's cost-based algorithm, but it
  /// uses a windowed detection method which is intended to reduce
  /// false detections of scene flashes which would exclude a normally
  /// good keyframe.
  /// This method will look back `lookahead_distance` from the current
  /// frame, calculating the cost ratio and averaging it over the frames.
  /// If the average meets the threshold, this frame is detected as a
  /// scenecut.
  fn has_scenecut<T: Pixel>(
    &self, fi: &FrameInvariants<T>, keyframes: &BTreeSet<u64>,
    config: &EncoderConfig, inter_cfg: &InterConfig, skip_distance: usize,
  ) -> bool {
    if fi.frame_type == FrameType::KEY {
      // This frame is already marked as a keyframe, so we don't need to
      // test it again.
      return true;
    }

    let intra_cost =
      &fi.lookahead_intra_costs.iter().map(|&v| v as f32).sum::<f32>()
        / fi.lookahead_intra_costs.len() as f32;
    let inter_distance =
      cmp::min(fi.input_frameno, inter_cfg.keyframe_lookahead_distance())
        as usize;
    let inter_costs = &fi
      .lookahead_inter_costs
      .iter()
      .take(inter_distance)
      .skip(skip_distance)
      .map(|costs| {
        costs.iter().map(|&v| v as f32).sum::<f32>() / costs.len() as f32
      })
      .collect::<Vec<_>>();
    let inter_avg = inter_costs.iter().sum::<f32>() / inter_costs.len() as f32;

    if inter_costs[0] < std::f32::EPSILON {
      // This is going to be all-SKIP, so always mark it as a non-scenechange.
      if skip_distance == 0 {
        debug!(
          "frame {} to {}: no scenecut; icost {}; pcost 0",
          fi.input_frameno,
          fi.input_frameno - inter_distance as u64,
          intra_cost
        );
      }
      return false;
    }

    if intra_cost < std::f32::EPSILON {
      // This avoids a division by zero error.
      if skip_distance == 0 {
        debug!(
          "frame {} to {}: scenecut; icost 0; pcost {:.3}",
          fi.input_frameno,
          fi.input_frameno - inter_distance as u64,
          inter_avg
        );
      }
      return true;
    }

    let distance_from_last_keyframe =
      fi.input_frameno - *keyframes.iter().last().unwrap();
    let thresh_max = SCENECUT_THRESHOLD / 100.;
    let thresh_min = thresh_max * 0.25;

    // The threshold gradually increases to the maximum as you get further from
    // the previous keyframe. This is to account for the fact that quality gradually
    // diminishes as you get further from the previous keyframe, therefore we want
    // a keyframe to be more likely to be chosen if we are far from the previous keyframe.
    let threshold = if distance_from_last_keyframe
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
          * (cmp::min(
            distance_from_last_keyframe - config.min_key_frame_interval,
            config.max_key_frame_interval - config.min_key_frame_interval,
          )) as f32
          / (config.max_key_frame_interval - config.min_key_frame_interval)
            as f32
    };

    // This algorithm wants all frames in the set to be below the max scenecut threshold,
    // wants at least 75% of frames in the set to be below the adjusted scenecut threshold,
    // and wants the average of frames in the set to be below the adjusted scenecut threshold.
    let frames_below_threshold = inter_costs
      .iter()
      .filter(|&&cost| 1. - cost / intra_cost <= threshold)
      .count();
    let all_frames_below_max_threshold = inter_costs
      .iter()
      .filter(|&&cost| 1. - cost / intra_cost <= SCENECUT_THRESHOLD)
      .count()
      == inter_costs.len();
    let avg_below_threshold = inter_avg >= (1. - threshold) * intra_cost;
    let enough_frames_below_threshold = frames_below_threshold as f32
      / inter_costs.len() as f32
      >= FRAME_PCT_THRESHOLD;

    let is_scenecut = avg_below_threshold
      && enough_frames_below_threshold
      && all_frames_below_max_threshold;
    if skip_distance == 0 {
      debug!(
        "frame {} to {}: {}; icost {}; pcost {}; ratio {:.3}; thresh {:.3}; below thresh: {}/{}; all below max: {}",
        fi.input_frameno,
        fi.input_frameno - inter_distance as u64,
        if is_scenecut { "scenecut" } else { "no scenecut" },
        intra_cost,
        inter_avg,
        1. - inter_avg / intra_cost,
        threshold,
        frames_below_threshold,
        inter_costs.len(),
        all_frames_below_max_threshold
      );
    }
    is_scenecut
  }
}
