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
use crate::util::{CastFromPrimitive, Pixel};
use rust_hawktracer::*;
use std::sync::Arc;
use std::{cmp, u64};
// use crate::api::*;

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
  score_deque: Vec<(f64, f64)>,
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
    // This implementation is based on a Python implementation at
    // https://pyscenedetect.readthedocs.io/en/latest/reference/detection-methods/.
    // The Python implementation uses HSV values and a threshold of 30. Comparing the
    // YUV values was sufficient in most cases, and avoided a more costly YUV->RGB->HSV
    // conversion, but the deltas needed to be scaled down. The deltas for keyframes
    // in YUV were about 1/3 to 1/2 of what they were in HSV, but non-keyframes were
    // very unlikely to have a delta greater than 3 in YUV, whereas they may reach into
    // the double digits in HSV.
    //
    // Experiments have shown that these thresholds is optimal.
    const FAST_THRESHOLD: f64 = 18.0;
    const SLOW_THRESHOLD: f64 = 7.0;

    let bit_depth = encoder_config.bit_depth;
    let speed_mode = if encoder_config.low_latency {
      SceneDetectionSpeed::Fast
    } else {
      encoder_config.speed_settings.fast_scene_detection
    };

    // Scale factor for fast and medium scene detection
    let scale_factor = if speed_mode != SceneDetectionSpeed::Slow {
      detect_scale_factor(&sequence, speed_mode)
    } else {
      1_usize
    };

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

    let threshold = if speed_mode == SceneDetectionSpeed::Fast {
      FAST_THRESHOLD * (bit_depth as f64) / 8.0
    } else {
      SLOW_THRESHOLD * (bit_depth as f64) / 8.0
    };

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

    if frame_set.len() < 2 {
      return false;
    }

    // Handle minimum and maximum keyframe intervals.
    if distance < self.encoder_config.min_key_frame_interval {
      return false;
    }
    if distance >= self.encoder_config.max_key_frame_interval {
      // Clear buffers and `score_deque`
      if let Some((_, is_initialized)) = &mut self.downscaled_frame_buffer {
        *is_initialized = false;
      }
      debug!("[SC-score-deque]{:.0?}", self.score_deque);
      self.score_deque.clear();

      return true;
    }

    if self.encoder_config.speed_settings.no_scene_detection {
      return false;
    }

    // Initiallization of score deque
    // based on frame set length
    if self.deque_offset > 0
      && frame_set.len() > self.deque_offset + 1
      && self.score_deque.is_empty()
    {
      self.initialize_score_deque(
        frame_set,
        input_frameno,
        previous_keyframe,
        self.deque_offset,
      );
    } else if self.score_deque.is_empty() {
      self.initialize_score_deque(
        frame_set,
        input_frameno,
        previous_keyframe,
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
        previous_keyframe,
      );
    } else {
      self.deque_offset -= 1;
    }

    // Adaptive scenecut check
    let scenecut = self.adaptive_scenecut();
    debug!(
      "[SC-Detect] Frame {}: I={:4.0}  T= {:.0} {}",
      input_frameno,
      self.score_deque[self.deque_offset].0,
      self.score_deque[self.deque_offset].1,
      if scenecut { "Scenecut" } else { "No cut" }
    );

    if scenecut {
      // Clear buffers and `score_deque`
      if let Some((_, is_initialized)) = &mut self.downscaled_frame_buffer {
        *is_initialized = false;
      }
      debug!("[SC-score-deque]{:.0?}", self.score_deque);
      self.score_deque.clear();
    } else {
      // Keep score deque of 5 backward frames
      // and forward frames of lenght of lookahead offset
      if self.score_deque.len() > 5 + self.lookahead_offset {
        self.score_deque.pop();
      }
    }

    scenecut
  }

  // Initially fill score deque with frame scores
  fn initialize_score_deque(
    &mut self, frame_set: &[Arc<Frame<T>>], input_frameno: u64,
    previous_keyframe: u64, init_len: usize,
  ) {
    for x in 0..init_len {
      self.run_comparison(
        frame_set[x].clone(),
        frame_set[x + 1].clone(),
        input_frameno,
        previous_keyframe,
      );
    }
  }

  /// Runs scene change comparison beetween 2 given frames
  /// Insert result to start of score deque
  fn run_comparison(
    &mut self, frame1: Arc<Frame<T>>, frame2: Arc<Frame<T>>,
    input_frameno: u64, previous_keyframe: u64,
  ) {
    let result = if self.speed_mode == SceneDetectionSpeed::Fast {
      self.fast_scenecut(frame1, frame2)
    } else {
      self.cost_scenecut(frame1, frame2, input_frameno, previous_keyframe)
    };
    self
      .score_deque
      .insert(0, (result.inter_cost as f64, result.threshold as f64));
  }

  /// Compares current scene score to adapted threshold based on previous scores
  /// Value of current frame is offset by lookahead, if lookahead >=5
  /// Returns true if current scene score is higher than adapted threshold
  fn adaptive_scenecut(&mut self) -> bool {
    // Subtract the previous metric value from the current one
    // It makes the peaks in the metric more distinctive
    if (self.speed_mode != SceneDetectionSpeed::Fast) && self.deque_offset > 0
    {
      let previous_scene_score = self.score_deque[self.deque_offset - 1].0;
      self.score_deque[self.deque_offset].0 -= previous_scene_score;
    }

    let scene_score = self.score_deque[self.deque_offset].0;
    let scene_threshold = self.score_deque[self.deque_offset].1;

    if scene_score >= scene_threshold as f64 {
      let back_deque = &self.score_deque[self.deque_offset + 1..];
      let forward_deque = &self.score_deque[..self.deque_offset];

      let back_over_tr_count =
        back_deque.iter().filter(|(x, y)| x > y).count();
      let forward_over_tr_count =
        forward_deque.iter().filter(|(x, y)| x > y).count();

      // Check for scenecut after the flashes
      // No frames over threshold forward
      // and some frames over threshold backward
      if forward_over_tr_count == 0 && back_over_tr_count > 1 {
        return true;
      }

      // Check for scenecut before flash
      // If distance longer than max flash length
      if back_over_tr_count == 0
        && forward_over_tr_count == 1
        && forward_deque[0].0 > forward_deque[0].1
      {
        return true;
      }

      if back_over_tr_count != 0 || forward_over_tr_count != 0 {
        return false;
      }
    }

    scene_score >= scene_threshold
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
        }
      } else {
        unreachable!()
      }
    }
  }

  /// Run a comparison between two frames to determine if they qualify for a scenecut.
  ///
  /// Using block intra and inter costs
  /// to determine which method would be more efficient
  /// for coding this frame.
  #[hawktracer(cost_scenecut)]
  fn cost_scenecut(
    &self, frame1: Arc<Frame<T>>, frame2: Arc<Frame<T>>, frameno: u64,
    previous_keyframe: u64,
  ) -> ScenecutResult {
    let frame2_ref2 = Arc::clone(&frame2);

    let (intra_cost, inter_cost) = crate::rayon::join(
      move || {
        let intra_costs = estimate_intra_costs(
          &*frame2,
          self.bit_depth,
          self.cpu_feature_level,
        );
        intra_costs.iter().map(|&cost| cost as u64).sum::<u64>() as f64
          / intra_costs.len() as f64
      },
      move || {
        let inter_costs = if self.speed_mode == SceneDetectionSpeed::Medium {
          estimate_inter_costs(
            frame2_ref2,
            frame1,
            self.bit_depth,
            self.encoder_config,
            self.sequence.clone(),
          )
        } else {
          estimate_inter_costs_histogram_blocks(frame2_ref2, frame1)
        };

        inter_costs.iter().map(|&cost| cost as u64).sum::<u64>() as f64
          / inter_costs.len() as f64
      },
    );

    // Sliding scale, more likely to choose a keyframe
    // as we get farther from the last keyframe.
    // Based on x264 scenecut code.
    //
    // `THRESH_MAX` determines how likely we are
    // to choose a keyframe, between 0.0-1.0.
    // Higher values mean we are more likely to choose a keyframe.
    // `0.833` was chosen based on trials using the new
    // adaptive scenecut code.
    const THRESH_MAX: f64 = 0.833;
    const THRESH_MIN: f64 = 0.75;
    let distance_from_keyframe = frameno - previous_keyframe;
    let min_keyint = self.encoder_config.min_key_frame_interval;
    let max_keyint = self.encoder_config.max_key_frame_interval;
    debug_assert!(distance_from_keyframe >= min_keyint);
    let bias = THRESH_MIN
      + (THRESH_MAX - THRESH_MIN)
        * (distance_from_keyframe - min_keyint) as f64
        / (max_keyint - min_keyint) as f64;

    // Adaptive threshold for medium version, static thresholf for the slow one
    let threshold = if self.speed_mode == SceneDetectionSpeed::Medium {
      intra_cost * (1.0 - bias)
    } else {
      self.threshold as f64
    };

    ScenecutResult { inter_cost, threshold }
  }

  /// Calculates delta beetween 2 planes
  /// returns average for pixel
  #[hawktracer(delta_in_planes)]
  fn delta_in_planes(&self, plane1: &Plane<T>, plane2: &Plane<T>) -> f64 {
    let mut delta = 0;

    let lines = plane1.rows_iter().zip(plane2.rows_iter());

    for (l1, l2) in lines {
      let l1 = l1.get(..plane1.cfg.width).unwrap_or(l1);
      let l2 = l2.get(..plane1.cfg.width).unwrap_or(l2);
      let delta_line = l1
        .iter()
        .zip(l2.iter())
        .map(|(&p1, &p2)| {
          (i16::cast_from(p1) - i16::cast_from(p2)).abs() as u32
        })
        .sum::<u32>();
      delta += delta_line as u64;
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

/// This struct primarily exists for returning metrics to the caller
#[derive(Debug, Clone, Copy)]
struct ScenecutResult {
  inter_cost: f64,
  threshold: f64,
}
