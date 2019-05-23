// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::frame::Frame;
use crate::util::{CastFromPrimitive, Pixel};

use std::sync::Arc;

/// Detects fast cuts using changes in colour and intensity between frames.
/// Since the difference between frames is used, only fast cuts are detected
/// with this method. This is probably fine for the purpose of choosing keyframes.
pub struct SceneChangeDetector<T: Pixel> {
  /// Minimum average difference between YUV deltas that will trigger a scene change.
  threshold: u8,
  /// Frame number and frame reference of the last frame analyzed
  last_frame: Option<(usize, Arc<Frame<T>>)>,
}

impl<T: Pixel> Default for SceneChangeDetector<T> {
  fn default() -> Self {
    Self {
      // This implementation is based on a Python implementation at
      // https://pyscenedetect.readthedocs.io/en/latest/reference/detection-methods/.
      // The Python implementation uses HSV values and a threshold of 30. Comparing the
      // YUV values was sufficient in most cases, and avoided a more costly YUV->RGB->HSV
      // conversion, but the deltas needed to be scaled down. The deltas for keyframes
      // in YUV were about 1/3 to 1/2 of what they were in HSV, but non-keyframes were
      // very unlikely to have a delta greater than 3 in YUV, whereas they may reach into
      // the double digits in HSV. Therefore, 12 was chosen as a reasonable default threshold.
      // This may be adjusted later.
      threshold: 12,
      last_frame: None,
    }
  }
}

impl<T: Pixel> SceneChangeDetector<T> {
  pub fn new(bit_depth: usize) -> Self {
    let mut detector = Self::default();
    detector.threshold = detector.threshold * bit_depth as u8 / 8;
    detector
  }

  pub fn set_last_frame(
    &mut self,
    ref_frame: Arc<Frame<T>>,
    frame_num: usize,
  ) {
    self.last_frame = Some((frame_num, ref_frame));
  }

  pub fn detect_scene_change(
    &mut self,
    curr_frame: Arc<Frame<T>>,
    frame_num: usize,
  ) -> bool {
    let mut is_change = false;

    match self.last_frame {
      Some((last_num, ref last_frame)) if last_num == frame_num - 1 => {
        let len =
          curr_frame.planes[0].cfg.width * curr_frame.planes[0].cfg.height;
        let delta_yuv = last_frame
          .iter()
          .zip(curr_frame.iter())
          .map(|(last, cur)| {
            (
              (i16::cast_from(cur.0) - i16::cast_from(last.0)).abs() as u64,
              (i16::cast_from(cur.1) - i16::cast_from(last.1)).abs() as u64,
              (i16::cast_from(cur.2) - i16::cast_from(last.2)).abs() as u64,
            )
          })
          .fold((0, 0, 0), |(ht, st, vt), (h, s, v)| (ht + h, st + s, vt + v));
        let delta_yuv = (
          (delta_yuv.0 / len as u64) as u16,
          (delta_yuv.1 / len as u64) as u16,
          (delta_yuv.2 / len as u64) as u16,
        );
        let delta_avg = ((delta_yuv.0 + delta_yuv.1 + delta_yuv.2) / 3) as u8;
        is_change = delta_avg >= self.threshold;
      }
      _ => (),
    }
    self.last_frame = Some((frame_num, curr_frame));
    is_change
  }
}
