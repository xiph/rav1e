// Copyright (c) 2018-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::api::EncoderConfig;
use crate::api::InterConfig;
use crate::frame::*;
use crate::util::CastFromPrimitive;
use crate::util::Pixel;

use crate::hawktracer::*;

use std::cmp;
use std::collections::BTreeSet;
use std::sync::Arc;

/// Runs keyframe detection on frames from the lookahead queue.
pub(crate) struct SceneChangeDetector {
  /// Minimum average difference between YUV deltas that will trigger a scene change.
  threshold: u8,
  /// Fast scene cut detection mode, ignoing chroma planes.
  fast_mode: bool,
  /// Frames that cannot be marked as keyframes due to the algorithm excluding them.
  /// Storing the frame numbers allows us to avoid looking back more than one frame.
  excluded_frames: BTreeSet<u64>,
}

impl SceneChangeDetector {
  pub fn new(bit_depth: u8, fast_mode: bool) -> Self {
    // This implementation is based on a Python implementation at
    // https://pyscenedetect.readthedocs.io/en/latest/reference/detection-methods/.
    // The Python implementation uses HSV values and a threshold of 30. Comparing the
    // YUV values was sufficient in most cases, and avoided a more costly YUV->RGB->HSV
    // conversion, but the deltas needed to be scaled down. The deltas for keyframes
    // in YUV were about 1/3 to 1/2 of what they were in HSV, but non-keyframes were
    // very unlikely to have a delta greater than 3 in YUV, whereas they may reach into
    // the double digits in HSV. Therefore, 12 was chosen as a reasonable default threshold.
    // This may be adjusted later.
    const BASE_THRESHOLD: u8 = 12;
    Self {
      threshold: BASE_THRESHOLD * bit_depth / 8,
      fast_mode,
      excluded_frames: BTreeSet::new(),
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

    self.exclude_scene_flashes(&frame_set, input_frameno, inter_cfg);

    self.is_key_frame(&frame_set[0], &frame_set[1], input_frameno)
  }

  /// Determines if `current_frame` should be a keyframe.
  fn is_key_frame<T: Pixel>(
    &self, previous_frame: &Frame<T>, current_frame: &Frame<T>,
    current_frameno: u64,
  ) -> bool {
    if self.excluded_frames.contains(&current_frameno) {
      return false;
    }

    self.has_scenecut(previous_frame, current_frame)
  }

  /// Uses lookahead to avoid coding short flashes as scenecuts.
  /// Saves excluded frame numbers in `self.excluded_frames`.
  fn exclude_scene_flashes<T: Pixel>(
    &mut self, frame_subset: &[Arc<Frame<T>>], frameno: u64,
    inter_cfg: &InterConfig,
  ) {
    let lookahead_distance = cmp::min(
      inter_cfg.keyframe_lookahead_distance() as usize,
      frame_subset.len() - 1,
    );

    // Where A and B are scenes: AAAAAABBBAAAAAA
    // If BBB is shorter than lookahead_distance, it is detected as a flash
    // and not considered a scenecut.
    for j in 1..=lookahead_distance {
      if !self.has_scenecut(&frame_subset[0], &frame_subset[j]) {
        // Any frame in between `0` and `j` cannot be a real scenecut.
        for i in 0..=j {
          let frameno = frameno + i as u64 - 1;
          self.excluded_frames.insert(frameno);
        }
      }
    }

    // Where A-F are scenes: AAAAABBCCDDEEFFFFFF
    // If each of BB ... EE are shorter than `lookahead_distance`, they are
    // detected as flashes and not considered scenecuts.
    // Instead, the first F frame becomes a scenecut.
    // If the video ends before F, no frame becomes a scenecut.
    for i in 1..lookahead_distance {
      if self.has_scenecut(&frame_subset[i], &frame_subset[lookahead_distance])
      {
        // If the current frame is the frame before a scenecut, it cannot also be the frame of a scenecut.
        let frameno = frameno + i as u64 - 1;
        self.excluded_frames.insert(frameno);
      }
    }
  }

  /// Run a comparison between two frames to determine if they qualify for a scenecut.
  ///
  /// The current algorithm detects fast cuts using changes in colour and intensity between frames.
  /// Since the difference between frames is used, only fast cuts are detected
  /// with this method. This is intended to change via https://github.com/xiph/rav1e/issues/794.
  fn has_scenecut<T: Pixel>(
    &self, frame1: &Frame<T>, frame2: &Frame<T>,
  ) -> bool {
    let mut len = frame2.planes[0].cfg.width * frame2.planes[0].cfg.height;
    let mut delta = 0;

    delta += self.delta_in_planes(&frame1.planes[0], &frame2.planes[0]);

    if !self.fast_mode {
      let u_dec = frame1.planes[1].cfg.xdec + frame1.planes[1].cfg.ydec;
      len +=
        (frame2.planes[1].cfg.width * frame2.planes[1].cfg.height) << u_dec;

      delta +=
        self.delta_in_planes(&frame1.planes[1], &frame2.planes[1]) << u_dec;

      let v_dec = frame1.planes[2].cfg.xdec + frame1.planes[2].cfg.ydec;
      len +=
        (frame2.planes[2].cfg.width * frame2.planes[2].cfg.height) << v_dec;
      delta +=
        self.delta_in_planes(&frame1.planes[2], &frame2.planes[2]) << v_dec;
    }

    delta >= self.threshold as u64 * len as u64
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
