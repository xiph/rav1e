// Copyright (c) 2018-2022, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

mod fast;
mod standard;

use crate::api::{EncoderConfig, SceneDetectionSpeed};
use crate::cpu_features::CpuFeatureLevel;
use crate::encoder::Sequence;
use crate::frame::*;
use crate::me::RefMEStats;
use crate::util::Pixel;
use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::{cmp, u64};

use self::fast::{detect_scale_factor, FAST_THRESHOLD};

/// Experiments have determined this to be an optimal threshold
const IMP_BLOCK_DIFF_THRESHOLD: f64 = 7.0;

/// Fast integer division where divisor is a nonzero power of 2
#[inline(always)]
pub(crate) fn fast_idiv(n: usize, d: NonZeroUsize) -> usize {
  debug_assert!(d.is_power_of_two());

  n >> d.trailing_zeros()
}

struct ScaleFunction<T: Pixel> {
  downscale_in_place:
    fn(/* &self: */ &Plane<T>, /* in_plane: */ &mut Plane<T>),
  downscale: fn(/* &self: */ &Plane<T>) -> Plane<T>,
  factor: NonZeroUsize,
}

impl<T: Pixel> ScaleFunction<T> {
  fn from_scale<const SCALE: usize>() -> Self {
    assert!(
      SCALE.is_power_of_two(),
      "Scaling factor needs to be a nonzero power of two"
    );

    Self {
      downscale: Plane::downscale::<SCALE>,
      downscale_in_place: Plane::downscale_in_place::<SCALE>,
      factor: NonZeroUsize::new(SCALE).unwrap(),
    }
  }
}
/// Runs keyframe detection on frames from the lookahead queue.
pub struct SceneChangeDetector<T: Pixel> {
  /// Minimum average difference between YUV deltas that will trigger a scene change.
  threshold: f64,
  /// Fast scene cut detection mode, uses simple SAD instead of encoder cost estimates.
  speed_mode: SceneDetectionSpeed,
  /// Downscaling function for fast scene detection
  scale_func: Option<ScaleFunction<T>>,
  /// Frame buffer for scaled frames
  downscaled_frame_buffer: Option<[Plane<T>; 2]>,
  /// Buffer for FrameMEStats for cost scenecut
  frame_me_stats_buffer: Option<RefMEStats>,
  /// Deque offset for current
  lookahead_offset: usize,
  /// Start deque offset based on lookahead
  deque_offset: usize,
  /// Scenechange results for adaptive threshold
  score_deque: Vec<ScenecutResult>,
  /// Number of pixels in scaled frame for fast mode
  pixels: usize,
  /// The bit depth of the video.
  bit_depth: usize,
  /// The CPU feature level to be used.
  cpu_feature_level: CpuFeatureLevel,
  encoder_config: EncoderConfig,
  sequence: Arc<Sequence>,
  /// Calculated intra costs for each input frame.
  /// These are cached for reuse later in rav1e.
  pub(crate) intra_costs: BTreeMap<u64, Box<[u32]>>,
  /// Temporary buffer used by estimate_intra_costs.
  pub(crate) temp_plane: Option<Plane<T>>,
}

impl<T: Pixel> SceneChangeDetector<T> {
  pub fn new(
    encoder_config: EncoderConfig, cpu_feature_level: CpuFeatureLevel,
    lookahead_distance: usize, sequence: Arc<Sequence>,
  ) -> Self {
    let bit_depth = encoder_config.bit_depth;
    let speed_mode = if encoder_config.low_latency {
      SceneDetectionSpeed::Fast
    } else {
      encoder_config.speed_settings.scene_detection_mode
    };

    // Downscaling function for fast scene detection
    let scale_func = detect_scale_factor(&sequence, speed_mode);

    // Set lookahead offset to 5 if normal lookahead available
    let lookahead_offset = if lookahead_distance >= 5 { 5 } else { 0 };
    let deque_offset = lookahead_offset;

    let score_deque = Vec::with_capacity(5 + lookahead_distance);

    // Downscaling factor for fast scenedetect (is currently always a power of 2)
    let factor =
      scale_func.as_ref().map_or(NonZeroUsize::new(1).unwrap(), |x| x.factor);

    let pixels = if speed_mode == SceneDetectionSpeed::Fast {
      fast_idiv(sequence.max_frame_height as usize, factor)
        * fast_idiv(sequence.max_frame_width as usize, factor)
    } else {
      1
    };

    let threshold = FAST_THRESHOLD * (bit_depth as f64) / 8.0;

    Self {
      threshold,
      speed_mode,
      scale_func,
      downscaled_frame_buffer: None,
      frame_me_stats_buffer: None,
      lookahead_offset,
      deque_offset,
      score_deque,
      pixels,
      bit_depth,
      cpu_feature_level,
      encoder_config,
      sequence,
      intra_costs: BTreeMap::new(),
      temp_plane: None,
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
  #[profiling::function]
  pub fn analyze_next_frame(
    &mut self, frame_set: &[&Arc<Frame<T>>], input_frameno: u64,
    previous_keyframe: u64,
  ) -> bool {
    // Use score deque for adaptive threshold for scene cut
    // Declare score_deque offset based on lookahead  for scene change scores

    // Find the distance to the previous keyframe.
    let distance = input_frameno - previous_keyframe;

    if frame_set.len() <= self.lookahead_offset {
      // Don't insert keyframes in the last few frames of the video
      // This is basically a scene flash and a waste of bits
      return false;
    }

    if self.encoder_config.speed_settings.scene_detection_mode
      == SceneDetectionSpeed::None
    {
      if let Some(true) = self.handle_min_max_intervals(distance) {
        return true;
      };
      return false;
    }

    // Initialization of score deque
    // based on frame set length
    if self.deque_offset > 0
      && frame_set.len() > self.deque_offset + 1
      && self.score_deque.is_empty()
    {
      self.initialize_score_deque(frame_set, input_frameno, self.deque_offset);
    } else if self.score_deque.is_empty() {
      self.initialize_score_deque(
        frame_set,
        input_frameno,
        frame_set.len() - 1,
      );

      self.deque_offset = frame_set.len() - 2;
    }
    // Running single frame comparison and adding it to deque
    // Decrease deque offset if there is no new frames
    if frame_set.len() > self.deque_offset + 1 {
      self.run_comparison(
        frame_set[self.deque_offset].clone(),
        frame_set[self.deque_offset + 1].clone(),
        input_frameno + self.deque_offset as u64,
      );
    } else {
      self.deque_offset -= 1;
    }

    // Adaptive scenecut check
    let (scenecut, score) = self.adaptive_scenecut();
    let scenecut = self.handle_min_max_intervals(distance).unwrap_or(scenecut);
    debug!(
      "[SC-Detect] Frame {}: Raw={:5.1}  ImpBl={:5.1}  Bwd={:5.1}  Fwd={:5.1}  Th={:.1}  {}",
      input_frameno,
      score.inter_cost,
      score.imp_block_cost,
      score.backward_adjusted_cost,
      score.forward_adjusted_cost,
      score.threshold,
      if scenecut { "Scenecut" } else { "No cut" }
    );

    // Keep score deque of 5 backward frames
    // and forward frames of length of lookahead offset
    if self.score_deque.len() > 5 + self.lookahead_offset {
      self.score_deque.pop();
    }

    scenecut
  }

  fn handle_min_max_intervals(&mut self, distance: u64) -> Option<bool> {
    // Handle minimum and maximum keyframe intervals.
    if distance < self.encoder_config.min_key_frame_interval {
      return Some(false);
    }
    if distance >= self.encoder_config.max_key_frame_interval {
      return Some(true);
    }
    None
  }

  // Initially fill score deque with frame scores
  fn initialize_score_deque(
    &mut self, frame_set: &[&Arc<Frame<T>>], input_frameno: u64,
    init_len: usize,
  ) {
    for x in 0..init_len {
      self.run_comparison(
        frame_set[x].clone(),
        frame_set[x + 1].clone(),
        input_frameno + x as u64,
      );
    }
  }

  /// Runs scene change comparison beetween 2 given frames
  /// Insert result to start of score deque
  fn run_comparison(
    &mut self, frame1: Arc<Frame<T>>, frame2: Arc<Frame<T>>,
    input_frameno: u64,
  ) {
    let mut result = if self.speed_mode == SceneDetectionSpeed::Fast {
      self.fast_scenecut(frame1, frame2)
    } else {
      self.cost_scenecut(frame1, frame2, input_frameno)
    };

    // Subtract the highest metric value of surrounding frames from the current one
    // It makes the peaks in the metric more distinct
    if self.speed_mode != SceneDetectionSpeed::Fast && self.deque_offset > 0 {
      if input_frameno == 1 {
        // Accounts for the second frame not having a score to adjust against.
        // It should always be 0 because the first frame of the video is always a keyframe.
        result.backward_adjusted_cost = 0.0;
      } else {
        let mut adjusted_cost = f64::MAX;
        for other_cost in
          self.score_deque.iter().take(self.deque_offset).map(|i| i.inter_cost)
        {
          let this_cost = result.inter_cost - other_cost;
          if this_cost < adjusted_cost {
            adjusted_cost = this_cost;
          }
          if adjusted_cost < 0.0 {
            adjusted_cost = 0.0;
            break;
          }
        }
        result.backward_adjusted_cost = adjusted_cost;
      }
      if !self.score_deque.is_empty() {
        for i in 0..(cmp::min(self.deque_offset, self.score_deque.len())) {
          let adjusted_cost =
            self.score_deque[i].inter_cost - result.inter_cost;
          if i == 0
            || adjusted_cost < self.score_deque[i].forward_adjusted_cost
          {
            self.score_deque[i].forward_adjusted_cost = adjusted_cost;
          }
          if self.score_deque[i].forward_adjusted_cost < 0.0 {
            self.score_deque[i].forward_adjusted_cost = 0.0;
          }
        }
      }
    }
    self.score_deque.insert(0, result);
  }

  /// Compares current scene score to adapted threshold based on previous scores
  /// Value of current frame is offset by lookahead, if lookahead >=5
  /// Returns true if current scene score is higher than adapted threshold
  fn adaptive_scenecut(&mut self) -> (bool, ScenecutResult) {
    let score = self.score_deque[self.deque_offset];

    // We use the importance block algorithm's cost metrics as a secondary algorithm
    // because, although it struggles in certain scenarios such as
    // finding the end of a pan, it is very good at detecting hard scenecuts
    // or detecting if a pan exists.
    // Because of this, we only consider a frame for a scenechange if
    // the importance block algorithm is over the threshold either on this frame (hard scenecut)
    // or within the past few frames (pan). This helps filter out a few false positives
    // produced by the cost-based algorithm.
    let imp_block_threshold =
      IMP_BLOCK_DIFF_THRESHOLD * (self.bit_depth as f64) / 8.0;
    if !&self.score_deque[self.deque_offset..]
      .iter()
      .any(|result| result.imp_block_cost >= imp_block_threshold)
    {
      return (false, score);
    }

    let cost = score.forward_adjusted_cost;
    if cost >= score.threshold {
      let back_deque = &self.score_deque[self.deque_offset + 1..];
      let forward_deque = &self.score_deque[..self.deque_offset];
      let back_over_tr_count = back_deque
        .iter()
        .filter(|result| result.backward_adjusted_cost >= result.threshold)
        .count();
      let forward_over_tr_count = forward_deque
        .iter()
        .filter(|result| result.forward_adjusted_cost >= result.threshold)
        .count();

      // Check for scenecut after the flashes
      // No frames over threshold forward
      // and some frames over threshold backward
      let back_count_req = if self.speed_mode == SceneDetectionSpeed::Fast {
        // Fast scenecut is more sensitive to false flash detection,
        // so we want more "evidence" of there being a flash before creating a keyframe.
        2
      } else {
        1
      };
      if forward_over_tr_count == 0 && back_over_tr_count >= back_count_req {
        return (true, score);
      }

      // Check for scenecut before flash
      // If distance longer than max flash length
      if back_over_tr_count == 0
        && forward_over_tr_count == 1
        && forward_deque[0].forward_adjusted_cost >= forward_deque[0].threshold
      {
        return (true, score);
      }

      if back_over_tr_count != 0 || forward_over_tr_count != 0 {
        return (false, score);
      }
    }

    (cost >= score.threshold, score)
  }
}

#[derive(Debug, Clone, Copy)]
struct ScenecutResult {
  inter_cost: f64,
  imp_block_cost: f64,
  backward_adjusted_cost: f64,
  forward_adjusted_cost: f64,
  threshold: f64,
}
