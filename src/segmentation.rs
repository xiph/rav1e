// Copyright (c) 2018-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use std::cmp::Ordering;
use crate::context::*;
use crate::header::PRIMARY_REF_NONE;
use crate::util::Pixel;
use crate::FrameInvariants;
use crate::FrameState;

pub fn segmentation_optimize<T: Pixel>(
  fi: &FrameInvariants<T>, fs: &mut FrameState<T>,
) {
  assert!(fi.enable_segmentation);
  fs.segmentation.enabled = fi.config.aq_mult != 0;

  if fs.segmentation.enabled {
    fs.segmentation.update_map = true;

    // We don't change the values between frames.
    fs.segmentation.update_data = fi.primary_ref_frame == PRIMARY_REF_NONE;

    if !fs.segmentation.update_data {
      return;
    }

    // Avoid going into lossless mode by never bringing qidx below 1.
    // Because base_q_idx changes more frequently than the segmentation
    // data, it is still possible for a segment to enter lossless, so
    // enforcement elsewhere is needed.
    let offset_lower_limit = 1 - fi.base_q_idx as i16;

    let mult = fi.config.aq_mult as f64;

    let avg_var = fi.activity_mask.avg_var;
    let mut seg_bins = fi.activity_mask.seg_bins;
    let threshold = fi.activity_mask.tot_bins / 6;

    let mut num_neg = 0usize;
    let mut num_pos = 0usize;

    let mut tmp_delta = [0f64; 8];
    for i in 0..8 {
        tmp_delta[i] = (avg_var.ceil() - (i as f64))*mult;
        num_pos += (tmp_delta[i] > 0f64) as usize;
        num_neg += (tmp_delta[i] < 0f64) as usize;
    }

    let mut remap_segment_tab: [usize; 8] = [ 0, 1, 2, 3, 4, 5, 6, 7 ];
    let mut num_segments = 8;

    loop {
        let mut changed = false;

        assert!(threshold > 0);

        if num_segments < 4 { break; }

        for i in 0..8 {
            if seg_bins[remap_segment_tab[i]] >= threshold { continue };
            if seg_bins[remap_segment_tab[i]] == 0 { continue }; /* Already eliminated */

            let prev_id = remap_segment_tab[i];

            #[derive(Debug, Default, Clone, Copy)]
            struct ScoreTab { idx: usize, score: f64 };
            let mut s_array = [ScoreTab { idx: usize::max_value(), score: std::f64::MAX }; 8];

            for j in 0..8 {
                s_array[j].idx = remap_segment_tab[j];
                if (remap_segment_tab[j] == prev_id) ||
                   (seg_bins[remap_segment_tab[j]] == 0) ||
                   (((num_neg < 2) || (num_pos < 2)) && (tmp_delta[remap_segment_tab[j]].signum() != tmp_delta[prev_id].signum())) {
                    s_array[j].score = std::f64::MAX;
                } else {
                    s_array[j].score = (tmp_delta[remap_segment_tab[j]] - tmp_delta[prev_id]).abs();
                }
            }

            s_array.sort_by(|a, b| (a.score).partial_cmp(&b.score).unwrap_or(Ordering::Less));

            if s_array[0].score == std::f64::MAX {
                continue;
            }

            /* Remap any old mappings to the current segment as well */
            for j in 0..8 {
                if remap_segment_tab[j] == prev_id {
                   remap_segment_tab[j] = s_array[0].idx;
                }
            }

            println!("Remapped = {} to {}, ss {} ({}) and {} ({})", prev_id, remap_segment_tab[i], seg_bins[prev_id], tmp_delta[prev_id], seg_bins[remap_segment_tab[i]], tmp_delta[remap_segment_tab[i]]);

            let num_2bins = seg_bins[remap_segment_tab[i]] + seg_bins[prev_id];
            let mut ratio_new = (seg_bins[remap_segment_tab[i]] as f64) / (num_2bins as f64);
            let mut ratio_old = (seg_bins[prev_id] as f64) / (num_2bins as f64);

            ratio_new *= tmp_delta[remap_segment_tab[i]];
            ratio_old *= tmp_delta[prev_id];

            num_pos -= (tmp_delta[prev_id] > 0f64) as usize;
            num_neg -= (tmp_delta[prev_id] < 0f64) as usize;

            tmp_delta[remap_segment_tab[i]] = ratio_new + ratio_old;
            tmp_delta[prev_id] = std::f64::MAX;

            seg_bins[remap_segment_tab[i]] += seg_bins[prev_id];
            seg_bins[prev_id] = 0;

            num_segments -= 1;

            changed = true;
            break;
        }

        if changed == false { break; }
    }

    /* Get all unique values in the intentionally unsorted array (its a LUT) */
    let mut uniq_array = [0usize; 8];
    let mut num_segments = 0;
    for i in 0..8 {
        let mut seen_match = false;
        for j in 0..num_segments {
            if remap_segment_tab[i] == uniq_array[j] {
                seen_match = true;
            }
        }
        if seen_match == false {
            uniq_array[num_segments] = remap_segment_tab[i];
            num_segments += 1;
        }
    }

    let mut seg_delta = [0f64; 8];
    for i in 0..num_segments {
        /* Collect all used segment deltas into the actual segment map */
        seg_delta[i] = tmp_delta[uniq_array[i]];

        /* Remap the LUT to make it match the layout of the seg deltaq map */
        for j in 0..8 {
            if remap_segment_tab[j] == uniq_array[i] {
                remap_segment_tab[j] = i;
            }
        }
    }

    println!("Num seg = {}, Center = {}", num_segments, avg_var);
    for i in 0..8 {
        println!("    {} -> {} : {}", i, remap_segment_tab[i], seg_delta[remap_segment_tab[i]]);
    }

    fs.segmentation.act_lut = remap_segment_tab;

    for i in 0..num_segments {
        fs.segmentation.features[i][SegLvl::SEG_LVL_ALT_Q as usize] = true;
        fs.segmentation.data[i][SegLvl::SEG_LVL_ALT_Q as usize] = (seg_delta[i].round() as i16).max(offset_lower_limit);

        println!("Seg {} = {}", i, fs.segmentation.data[i][SegLvl::SEG_LVL_ALT_Q as usize]);
    }

    /* Figure out parameters */
    fs.segmentation.preskip = false;
    fs.segmentation.last_active_segid = 0;
    if fs.segmentation.enabled {
      for i in 0..8 {
        for j in 0..SegLvl::SEG_LVL_MAX as usize {
          if fs.segmentation.features[i][j] {
            fs.segmentation.last_active_segid = i as u8;
            if j >= SegLvl::SEG_LVL_REF_FRAME as usize {
              fs.segmentation.preskip = true;
            }
          }
        }
      }
    }
  }
}
