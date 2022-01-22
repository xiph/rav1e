// Copyright (c) 2018-2021, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::api::lookahead::*;
use crate::api::{EncoderConfig, SceneDetectionSpeed};
use crate::cpu_features::CpuFeatureLevel;
use crate::encoder::Sequence;
use crate::frame::*;
use crate::sad_row;
use crate::util::Pixel;
use rust_hawktracer::*;
use std::sync::Arc;
use std::{cmp, u64};

// The fast implementation is based on a Python implementation at
// https://pyscenedetect.readthedocs.io/en/latest/reference/detection-methods/.
// The Python implementation uses HSV values and a threshold of 30. Comparing the
// YUV values was sufficient in most cases, and avoided a more costly YUV->RGB->HSV
// conversion, but the deltas needed to be scaled down. The deltas for keyframes
// in YUV were about 1/3 to 1/2 of what they were in HSV, but non-keyframes were
// very unlikely to have a delta greater than 3 in YUV, whereas they may reach into
// the double digits in HSV.
//
// Experiments have shown that these thresholds are optimal.
const FAST_THRESHOLD: f64 = 18.0;
const IMP_BLOCK_DIFF_THRESHOLD: f64 = 7.0;

/// Runs keyframe detection on frames from the lookahead queue.
pub struct SceneChangeDetector<T: Pixel> {
  /// Minimum average difference between YUV deltas that will trigger a scene change.
  threshold: f64,
  /// Fast scene cut detection mode, uses simple SAD instead of encoder cost estimates.
  speed_mode: SceneDetectionSpeed,
  /// scaling factor for fast scene detection
  scale_factor: usize,
  /// Frame buffer for scaled frames
  downscaled_frame_buffer: Option<(
    Box<[Plane<T>; 2]>,
    // `true` if the data is valid and initialized, or `false`
    // if it should be assumed that the data is uninitialized.
    bool,
  )>,
  /// Frame buffer for holding references to source frames.
  ///
  /// Useful for not copying data into the downscaled frame buffer
  /// when using a downscale factor of 1.
  frame_ref_buffer: Option<Box<[Arc<Frame<T>>; 2]>>,
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

    // Scale factor for fast and medium scene detection
    let scale_factor = detect_scale_factor(&sequence, speed_mode);

    // Set lookahead offset to 5 if normal lookahead available
    let lookahead_offset = if lookahead_distance >= 5 { 5 } else { 0 };
    let deque_offset = lookahead_offset;

    let score_deque = Vec::with_capacity(5 + lookahead_distance);

    // Pixel count for fast scenedetect
    let pixels = if speed_mode == SceneDetectionSpeed::Fast {
      (sequence.max_frame_height as usize / scale_factor)
        * (sequence.max_frame_width as usize / scale_factor)
    } else {
      1
    };

    let threshold = FAST_THRESHOLD * (bit_depth as f64) / 8.0;

    Self {
      threshold,
      speed_mode,
      scale_factor,
      downscaled_frame_buffer: None,
      frame_ref_buffer: None,
      lookahead_offset,
      deque_offset,
      score_deque,
      pixels,
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
  pub fn analyze_next_frame(
    &mut self, frame_set: &[Arc<Frame<T>>], input_frameno: u64,
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

    // Initiallization of score deque
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
        input_frameno,
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
    &mut self, frame_set: &[Arc<Frame<T>>], input_frameno: u64,
    init_len: usize,
  ) {
    for x in 0..init_len {
      self.run_comparison(
        frame_set[x].clone(),
        frame_set[x + 1].clone(),
        input_frameno,
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
      self.cost_scenecut(frame1, frame2)
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

  /// The fast algorithm detects fast cuts using a raw difference
  /// in pixel values between the scaled frames.
  #[hawktracer(fast_scenecut)]
  fn fast_scenecut(
    &mut self, frame1: Arc<Frame<T>>, frame2: Arc<Frame<T>>,
  ) -> ScenecutResult {
    if self.scale_factor == 1 {
      if let Some(frame_buffer) = self.frame_ref_buffer.as_deref_mut() {
        frame_buffer.swap(0, 1);
        frame_buffer[1] = frame2;
      } else {
        self.frame_ref_buffer = Some(Box::new([frame1, frame2]));
      }

      if let Some(frame_buffer) = self.frame_ref_buffer.as_deref() {
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
        unreachable!()
      }
    } else {
      // downscale both frames for faster comparison
      if let Some((frame_buffer, is_initialized)) =
        &mut self.downscaled_frame_buffer
      {
        let frame_buffer = &mut **frame_buffer;
        if *is_initialized {
          frame_buffer.swap(0, 1);
          frame2.planes[0]
            .downscale_in_place(self.scale_factor, &mut frame_buffer[1]);
        } else {
          // both frames are in an irrelevant and invalid state, so we have to reinitialize
          // them, but we can reuse their allocations
          frame1.planes[0]
            .downscale_in_place(self.scale_factor, &mut frame_buffer[0]);
          frame2.planes[0]
            .downscale_in_place(self.scale_factor, &mut frame_buffer[1]);
          *is_initialized = true;
        }
      } else {
        self.downscaled_frame_buffer = Some((
          Box::new([
            frame1.planes[0].downscale(self.scale_factor),
            frame2.planes[0].downscale(self.scale_factor),
          ]),
          true, // the frame buffer is initialized and in a valid state
        ));
      }

      if let Some((frame_buffer, _)) = &self.downscaled_frame_buffer {
        let frame_buffer = &**frame_buffer;
        let delta = self.delta_in_planes(&frame_buffer[0], &frame_buffer[1]);

        ScenecutResult {
          threshold: self.threshold as f64,
          inter_cost: delta as f64,
          imp_block_cost: delta as f64,
          forward_adjusted_cost: delta as f64,
          backward_adjusted_cost: delta as f64,
        }
      } else {
        unreachable!()
      }
    }
  }

  /// Run a comparison between two frames to determine if they qualify for a scenecut.
  ///
  /// We gather both intra and inter costs for the frames,
  /// as well as an importance-block-based difference,
  /// and use all three metrics.
  #[hawktracer(cost_scenecut)]
  fn cost_scenecut(
    &self, frame1: Arc<Frame<T>>, frame2: Arc<Frame<T>>,
  ) -> ScenecutResult {
    let frame2_inter_ref = Arc::clone(&frame2);
    let frame1_imp_ref = Arc::clone(&frame1);
    let frame2_imp_ref = Arc::clone(&frame2);

    let mut intra_cost = 0.0;
    let mut mv_inter_cost = 0.0;
    let mut imp_block_cost = 0.0;
    crate::rayon::scope(|s| {
      s.spawn(|_| {
        let intra_costs = estimate_intra_costs(
          &*frame2,
          self.bit_depth,
          self.cpu_feature_level,
        );
        intra_cost = intra_costs.iter().map(|&cost| cost as u64).sum::<u64>()
          as f64
          / intra_costs.len() as f64
      });
      s.spawn(|_| {
        let inter_costs = estimate_inter_costs(
          frame2_inter_ref,
          frame1,
          self.bit_depth,
          self.encoder_config,
          self.sequence.clone(),
        );

        mv_inter_cost =
          inter_costs.iter().map(|&cost| cost as u64).sum::<u64>() as f64
            / inter_costs.len() as f64
      });
      s.spawn(|_| {
        imp_block_cost =
          estimate_importance_block_difference(frame2_imp_ref, frame1_imp_ref);
      });
    });

    // `BIAS` determines how likely we are
    // to choose a keyframe, between 0.0-1.0.
    // Higher values mean we are more likely to choose a keyframe.
    // This value was chosen based on trials using the new
    // adaptive scenecut code.
    const BIAS: f64 = 0.7;
    let threshold = intra_cost * (1.0 - BIAS);

    ScenecutResult {
      inter_cost: mv_inter_cost,
      imp_block_cost,
      threshold,
      backward_adjusted_cost: 0.0,
      forward_adjusted_cost: 0.0,
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
fn detect_scale_factor(
  sequence: &Arc<Sequence>, speed_mode: SceneDetectionSpeed,
) -> usize {
  let small_edge =
    cmp::min(sequence.max_frame_height, sequence.max_frame_width) as usize;
  let scale_factor;
  if speed_mode == SceneDetectionSpeed::Fast {
    scale_factor = match small_edge {
      0..=240 => 1,
      241..=480 => 2,
      481..=720 => 4,
      721..=1080 => 8,
      1081..=1600 => 16,
      1601..=usize::MAX => 32,
      _ => 1,
    } as usize
  } else {
    scale_factor = match small_edge {
      0..=1600 => 1,
      1601..=2160 => 2,
      2161..=usize::MAX => 4,
      _ => 1,
    } as usize
  };

  debug!(
    "Scene detection scale factor {}, [{},{}] -> [{},{}]",
    scale_factor,
    sequence.max_frame_width,
    sequence.max_frame_height,
    sequence.max_frame_width as usize / scale_factor,
    sequence.max_frame_height as usize / scale_factor
  );
  scale_factor
}

#[derive(Debug, Clone, Copy)]
struct ScenecutResult {
  inter_cost: f64,
  imp_block_cost: f64,
  backward_adjusted_cost: f64,
  forward_adjusted_cost: f64,
  threshold: f64,
}
