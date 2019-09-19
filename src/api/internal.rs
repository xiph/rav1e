// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.
#![deny(missing_docs)]

use crate::api::{EncoderConfig, EncoderStatus, FrameType, Packet};

use arrayvec::ArrayVec;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::context::*;
use crate::context::{FrameBlocks, SuperBlockOffset, TileSuperBlockOffset};
use crate::dist::get_satd;
use crate::encoder::*;
use crate::frame::*;
use crate::metrics::calculate_frame_psnr;
use crate::partition::*;
use crate::predict::PredictionMode;
use crate::rate::RCState;
use crate::rate::FRAME_NSUBTYPES;
use crate::rate::FRAME_SUBTYPE_I;
use crate::rate::FRAME_SUBTYPE_P;
use crate::rate::FRAME_SUBTYPE_SEF;
use crate::scenechange::SceneChangeDetector;
use crate::stats::EncoderStats;
use crate::tiling::{Area, TileRect};
use crate::transform::TxSize;
use crate::util::Pixel;

use std::cmp;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;

/// The set of options that controls frame re-ordering and reference picture
///  selection.
/// The options stored here are invariant over the whole encode.
#[derive(Debug, Clone, Copy)]
pub(crate) struct InterConfig {
  /// Whether frame re-ordering is enabled.
  reorder: bool,
  /// Whether P-frames can use multiple references.
  pub(crate) multiref: bool,
  /// The depth of the re-ordering pyramid.
  /// The current code cannot support values larger than 2.
  pub(crate) pyramid_depth: u64,
  /// Number of input frames in group.
  pub(crate) group_input_len: u64,
  /// Number of output frames in group.
  /// This includes both hidden frames and "show existing frame" frames.
  group_output_len: u64,
}

impl InterConfig {
  fn new(enc_config: &EncoderConfig) -> InterConfig {
    let reorder = !enc_config.low_latency;
    // A group always starts with (group_output_len - group_input_len) hidden
    //  frames, followed by group_input_len shown frames.
    // The shown frames iterate over the input frames in order, with frames
    //  already encoded as hidden frames now displayed with Show Existing
    //  Frame.
    // For example, for a pyramid depth of 2, the group is as follows:
    //                      |TU         |TU |TU |TU
    // idx_in_group_output:   0   1   2   3   4   5
    // input_frameno:         4   2   1  SEF  3  SEF
    // output_frameno:        1   2   3   4   5   6
    // level:                 0   1   2   1   2   0
    //                        ^^^^^   ^^^^^^^^^^^^^
    //                        hidden      shown
    // TODO: This only works for pyramid_depth <= 2 --- after that we need
    //  more hidden frames in the middle of the group.
    let pyramid_depth = if reorder { 2 } else { 0 };
    let group_input_len = 1 << pyramid_depth;
    let group_output_len = group_input_len + pyramid_depth;
    InterConfig {
      reorder,
      multiref: reorder || enc_config.speed_settings.multiref,
      pyramid_depth,
      group_input_len,
      group_output_len,
    }
  }

  /// Get the index of an output frame in its re-ordering group given the output
  ///  frame number of the frame in the current keyframe gop.
  /// When re-ordering is disabled, this always returns 0.
  pub(crate) fn get_idx_in_group_output(
    &self, output_frameno_in_gop: u64,
  ) -> u64 {
    // The first frame in the GOP should be a keyframe and is not re-ordered,
    //  so we should not be calling this function on it.
    debug_assert!(output_frameno_in_gop > 0);
    (output_frameno_in_gop - 1) % self.group_output_len
  }

  /// Get the order-hint of an output frame given the output frame number of the
  ///  frame in the current keyframe gop and the index of that output frame
  ///  in its re-ordering gorup.
  pub(crate) fn get_order_hint(
    &self, output_frameno_in_gop: u64, idx_in_group_output: u64,
  ) -> u32 {
    // The first frame in the GOP should be a keyframe, but currently this
    //  function only handles inter frames.
    // We could return 0 for keyframes if keyframe support is needed.
    debug_assert!(output_frameno_in_gop > 0);
    // Which P-frame group in the current gop is this output frame in?
    // Subtract 1 because the first frame in the gop is always a keyframe.
    let group_idx = (output_frameno_in_gop - 1) / self.group_output_len;
    // Get the offset to the corresponding input frame.
    // TODO: This only works with pyramid_depth <= 2.
    let offset = if idx_in_group_output < self.pyramid_depth {
      self.group_input_len >> idx_in_group_output
    } else {
      idx_in_group_output - self.pyramid_depth + 1
    };
    // Construct the final order hint relative to the start of the group.
    (self.group_input_len * group_idx + offset) as u32
  }

  /// Get the level of the current frame in the pyramid.
  pub(crate) fn get_level(&self, idx_in_group_output: u64) -> u64 {
    if !self.reorder {
      0
    } else if idx_in_group_output < self.pyramid_depth {
      // Hidden frames are output first (to be shown in the future).
      idx_in_group_output
    } else {
      // Shown frames
      // TODO: This only works with pyramid_depth <= 2.
      pos_to_lvl(
        idx_in_group_output - self.pyramid_depth + 1,
        self.pyramid_depth,
      )
    }
  }

  pub(crate) fn get_slot_idx(&self, level: u64, order_hint: u32) -> u32 {
    // Frames with level == 0 are stored in slots 0..4, and frames with higher
    //  values of level in slots 4..8
    if level == 0 {
      (order_hint >> self.pyramid_depth) & 3
    } else {
      // This only works with pyramid_depth <= 4.
      3 + level as u32
    }
  }

  pub(crate) const fn get_show_frame(&self, idx_in_group_output: u64) -> bool {
    idx_in_group_output >= self.pyramid_depth
  }

  pub(crate) fn get_show_existing_frame(
    &self, idx_in_group_output: u64,
  ) -> bool {
    // The self.reorder test here is redundant, but short-circuits the rest,
    //  avoiding a bunch of work when it's false.
    self.reorder
      && self.get_show_frame(idx_in_group_output)
      && (idx_in_group_output - self.pyramid_depth + 1).count_ones() == 1
      && idx_in_group_output != self.pyramid_depth
  }

  pub(crate) fn get_input_frameno(
    &self, output_frameno_in_gop: u64, gop_input_frameno_start: u64,
  ) -> u64 {
    if output_frameno_in_gop == 0 {
      gop_input_frameno_start
    } else {
      let idx_in_group_output =
        self.get_idx_in_group_output(output_frameno_in_gop);
      let order_hint =
        self.get_order_hint(output_frameno_in_gop, idx_in_group_output);
      gop_input_frameno_start + order_hint as u64
    }
  }

  const fn max_reordering_latency(&self) -> u64 {
    self.group_input_len
  }

  pub(crate) fn keyframe_lookahead_distance(&self) -> u64 {
    cmp::max(1, self.max_reordering_latency()) + 1
  }
}
// the fields pub(super) are accessed only by the tests
pub(crate) struct ContextInner<T: Pixel> {
  pub(crate) frame_count: u64,
  pub(crate) limit: Option<u64>,
  pub(crate) output_frameno: u64,
  pub(super) inter_cfg: InterConfig,
  pub(super) frames_processed: u64,
  /// Maps *input_frameno* to frames
  pub(super) frame_q: BTreeMap<u64, Option<Arc<Frame<T>>>>, //    packet_q: VecDeque<Packet>
  /// Maps *output_frameno* to frame data
  pub(super) frame_invariants: BTreeMap<u64, FrameInvariants<T>>,
  /// A list of the input_frameno for keyframes in this encode.
  /// Needed so that we don't need to keep all of the frame_invariants in
  ///  memory for the whole life of the encode.
  // TODO: Is this needed at all?
  keyframes: BTreeSet<u64>,
  // TODO: Is this needed at all?
  keyframes_forced: BTreeSet<u64>,
  /// A storage space for reordered frames.
  packet_data: Vec<u8>,
  /// Maps `output_frameno` to `gop_output_frameno_start`.
  gop_output_frameno_start: BTreeMap<u64, u64>,
  /// Maps `output_frameno` to `gop_input_frameno_start`.
  pub(crate) gop_input_frameno_start: BTreeMap<u64, u64>,
  keyframe_detector: SceneChangeDetector,
  pub(crate) config: EncoderConfig,
  seq: Sequence,
  pub(crate) rc_state: RCState,
  maybe_prev_log_base_q: Option<i64>,
  /// The next `input_frameno` to be processed by lookahead.
  next_lookahead_frame: u64,
  /// The next `output_frameno` to be computed by lookahead.
  next_lookahead_output_frameno: u64,
}

impl<T: Pixel> ContextInner<T> {
  pub fn new(enc: &EncoderConfig) -> Self {
    // initialize with temporal delimiter
    let packet_data = TEMPORAL_DELIMITER.to_vec();

    let maybe_ac_qi_max =
      if enc.quantizer < 255 { Some(enc.quantizer as u8) } else { None };

    ContextInner {
      frame_count: 0,
      limit: None,
      inter_cfg: InterConfig::new(enc),
      output_frameno: 0,
      frames_processed: 0,
      frame_q: BTreeMap::new(),
      frame_invariants: BTreeMap::new(),
      keyframes: BTreeSet::new(),
      keyframes_forced: BTreeSet::new(),
      packet_data,
      gop_output_frameno_start: BTreeMap::new(),
      gop_input_frameno_start: BTreeMap::new(),
      keyframe_detector: SceneChangeDetector::new(enc.bit_depth as u8),
      config: enc.clone(),
      seq: Sequence::new(enc),
      rc_state: RCState::new(
        enc.width as i32,
        enc.height as i32,
        enc.time_base.den as i64,
        enc.time_base.num as i64,
        enc.bitrate,
        maybe_ac_qi_max,
        enc.min_quantizer,
        enc.max_key_frame_interval as i32,
        enc.reservoir_frame_delay,
      ),
      maybe_prev_log_base_q: None,
      next_lookahead_frame: 0,
      next_lookahead_output_frameno: 0,
    }
  }

  pub fn send_frame(
    &mut self, frame: Option<Arc<Frame<T>>>, params: Option<FrameParameters>,
  ) -> Result<(), EncoderStatus> {
    let input_frameno = self.frame_count;
    if frame.is_some() {
      self.frame_count += 1;
    }
    self.frame_q.insert(input_frameno, frame);

    if let Some(params) = params {
      if params.frame_type_override == FrameTypeOverride::Key {
        self.keyframes_forced.insert(input_frameno);
      }
    }

    self.compute_lookahead_data();
    Ok(())
  }

  fn get_frame(&self, input_frameno: u64) -> Arc<Frame<T>> {
    // Clones only the arc, so low cost overhead
    self
      .frame_q
      .get(&input_frameno)
      .as_ref()
      .unwrap()
      .as_ref()
      .unwrap()
      .clone()
  }

  /// Indicates whether more frames need to be read into the frame queue
  /// in order for frame queue lookahead to be full.
  fn needs_more_frame_q_lookahead(&self, input_frameno: u64) -> bool {
    let lookahead_end = self.frame_q.keys().last().cloned().unwrap_or(0);
    let frames_needed =
      input_frameno + self.inter_cfg.keyframe_lookahead_distance() + 1;
    lookahead_end < frames_needed && self.needs_more_frames(lookahead_end)
  }

  /// Indicates whether more frames need to be processed into FrameInvariants
  /// in order for FI lookahead to be full.
  fn needs_more_fi_lookahead(&self) -> bool {
    let ready_frames = self.get_rdo_lookahead_frames().count();
    ready_frames < self.config.rdo_lookahead_frames + 1
      && self.needs_more_frames(self.next_lookahead_frame)
  }

  pub fn needs_more_frames(&self, frame_count: u64) -> bool {
    self.limit.map(|limit| frame_count < limit).unwrap_or(true)
  }

  fn get_rdo_lookahead_frames(
    &self,
  ) -> impl Iterator<Item = (&u64, &FrameInvariants<T>)> {
    self
      .frame_invariants
      .iter()
      .skip_while(move |(&output_frameno, _)| {
        output_frameno < self.output_frameno
      })
      .filter(|(_, fi)| !fi.invalid && !fi.show_existing_frame)
      .take(self.config.rdo_lookahead_frames + 1)
  }

  fn next_keyframe_input_frameno(
    &self, gop_input_frameno_start: u64, ignore_limit: bool,
  ) -> u64 {
    let next_detected = self
      .keyframes
      .iter()
      .find(|&&input_frameno| input_frameno > gop_input_frameno_start)
      .cloned();
    let mut next_limit =
      gop_input_frameno_start + self.config.max_key_frame_interval;
    if !ignore_limit && self.limit.is_some() {
      next_limit = next_limit.min(self.limit.unwrap());
    }
    if next_detected.is_none() {
      return next_limit;
    }
    cmp::min(next_detected.unwrap(), next_limit)
  }

  fn set_frame_properties(
    &mut self, output_frameno: u64,
  ) -> Result<(), EncoderStatus> {
    let fi = self.build_frame_properties(output_frameno)?;
    self.frame_invariants.insert(output_frameno, fi);

    Ok(())
  }

  fn build_frame_properties(
    &mut self, output_frameno: u64,
  ) -> Result<FrameInvariants<T>, EncoderStatus> {
    let (prev_gop_output_frameno_start, prev_gop_input_frameno_start) =
      if output_frameno == 0 {
        (0, 0)
      } else {
        (
          self.gop_output_frameno_start[&(output_frameno - 1)],
          self.gop_input_frameno_start[&(output_frameno - 1)],
        )
      };

    self
      .gop_output_frameno_start
      .insert(output_frameno, prev_gop_output_frameno_start);
    self
      .gop_input_frameno_start
      .insert(output_frameno, prev_gop_input_frameno_start);

    let output_frameno_in_gop =
      output_frameno - self.gop_output_frameno_start[&output_frameno];
    let mut input_frameno = self.inter_cfg.get_input_frameno(
      output_frameno_in_gop,
      self.gop_input_frameno_start[&output_frameno],
    );

    if self.needs_more_frame_q_lookahead(input_frameno) {
      return Err(EncoderStatus::NeedMoreData);
    }

    if output_frameno_in_gop > 0 {
      let next_keyframe_input_frameno = self.next_keyframe_input_frameno(
        self.gop_input_frameno_start[&output_frameno],
        false,
      );
      let prev_input_frameno =
        self.frame_invariants[&(output_frameno - 1)].input_frameno;
      if input_frameno >= next_keyframe_input_frameno {
        if !self.inter_cfg.reorder
          || ((output_frameno_in_gop - 1) % self.inter_cfg.group_output_len
            == 0
            && prev_input_frameno == (next_keyframe_input_frameno - 1))
        {
          input_frameno = next_keyframe_input_frameno;

          // If we'll return early, do it before modifying the state.
          match self.frame_q.get(&input_frameno) {
            Some(Some(_)) => {}
            _ => {
              return Err(EncoderStatus::NeedMoreData);
            }
          }

          *self.gop_output_frameno_start.get_mut(&output_frameno).unwrap() =
            output_frameno;
          *self.gop_input_frameno_start.get_mut(&output_frameno).unwrap() =
            next_keyframe_input_frameno;
        } else {
          let fi = FrameInvariants::new_inter_frame(
            &self.frame_invariants[&(output_frameno - 1)],
            &self.inter_cfg,
            self.gop_input_frameno_start[&output_frameno],
            output_frameno_in_gop,
            next_keyframe_input_frameno,
          );
          assert!(fi.invalid);
          return Ok(fi);
        }
      }
    }

    match self.frame_q.get(&input_frameno) {
      Some(Some(_)) => {}
      _ => {
        return Err(EncoderStatus::NeedMoreData);
      }
    }

    // Now that we know the input_frameno, look up the correct frame type
    let frame_type = if self.keyframes.contains(&input_frameno) {
      FrameType::KEY
    } else {
      FrameType::INTER
    };
    if frame_type == FrameType::KEY {
      *self.gop_output_frameno_start.get_mut(&output_frameno).unwrap() =
        output_frameno;
      *self.gop_input_frameno_start.get_mut(&output_frameno).unwrap() =
        input_frameno;
    }

    let output_frameno_in_gop =
      output_frameno - self.gop_output_frameno_start[&output_frameno];
    if output_frameno_in_gop == 0 {
      let fi = FrameInvariants::new_key_frame(
        self.config.clone(),
        self.seq,
        self.gop_input_frameno_start[&output_frameno],
      );
      assert!(!fi.invalid);
      Ok(fi)
    } else {
      let next_keyframe_input_frameno = self.next_keyframe_input_frameno(
        self.gop_input_frameno_start[&output_frameno],
        false,
      );
      let fi = FrameInvariants::new_inter_frame(
        &self.frame_invariants[&(output_frameno - 1)],
        &self.inter_cfg,
        self.gop_input_frameno_start[&output_frameno],
        output_frameno_in_gop,
        next_keyframe_input_frameno,
      );
      assert!(!fi.invalid);
      Ok(fi)
    }
  }

  pub(crate) fn done_processing(&self) -> bool {
    self.limit.map(|limit| self.frames_processed == limit).unwrap_or(false)
  }

  /// Computes lookahead motion vectors and fills in `lookahead_mvs`,
  /// `rec_buffer` and `lookahead_rec_buffer` on the `FrameInvariants`. This
  /// function must be called after every new `FrameInvariants` is initially
  /// computed.
  fn compute_lookahead_motion_vectors(&mut self, output_frameno: u64) {
    let fi = self.frame_invariants.get_mut(&output_frameno).unwrap();

    // We're only interested in valid frames which are not show-existing-frame.
    // Those two don't modify the rec_buffer so there's no need to do anything
    // special about it either, it'll propagate on its own.
    if fi.invalid || fi.show_existing_frame {
      return;
    }

    let frame = self.frame_q[&fi.input_frameno].as_ref().unwrap();

    // TODO: some of this work, like downsampling, could be reused in the
    // actual encoding.
    let mut fs = FrameState::new_with_frame(fi, frame.clone());
    fs.input_hres.downsample_from(&frame.planes[0]);
    fs.input_hres.pad(fi.width, fi.height);
    fs.input_qres.downsample_from(&fs.input_hres);
    fs.input_qres.pad(fi.width, fi.height);

    #[cfg(feature = "dump_lookahead_data")]
    {
      let plane = &fs.input_qres;
      image::GrayImage::from_fn(
        plane.cfg.width as u32,
        plane.cfg.height as u32,
        |x, y| image::Luma([plane.p(x as usize, y as usize).as_()]),
      )
      .save(format!("{}-qres.png", fi.input_frameno))
      .unwrap();
      let plane = &fs.input_hres;
      image::GrayImage::from_fn(
        plane.cfg.width as u32,
        plane.cfg.height as u32,
        |x, y| image::Luma([plane.p(x as usize, y as usize).as_()]),
      )
      .save(format!("{}-hres.png", fi.input_frameno))
      .unwrap();
    }

    // Do not modify the next output frame's FrameInvariants.
    if self.output_frameno == output_frameno {
      // We do want to propagate the lookahead_rec_buffer though.
      let rfs = Arc::new(ReferenceFrame {
        order_hint: fi.order_hint,
        // Use the original frame contents.
        frame: frame.clone(),
        input_hres: fs.input_hres,
        input_qres: fs.input_qres,
        cdfs: fs.cdfs,
        // TODO: can we set MVs here? We can probably even compute these MVs
        // right now instead of in encode_tile?
        frame_mvs: fs.frame_mvs,
        output_frameno,
      });
      for i in 0..(REF_FRAMES as usize) {
        if (fi.refresh_frame_flags & (1 << i)) != 0 {
          fi.lookahead_rec_buffer.frames[i] = Some(Arc::clone(&rfs));
          fi.lookahead_rec_buffer.deblock[i] = fs.deblock;
        }
      }

      return;
    }

    // Our lookahead_rec_buffer should be filled with correct original frame
    // data from the previous frames. Copy it into rec_buffer because that's
    // what the MV search uses. During the actual encoding rec_buffer is
    // overwritten with its correct values anyway.
    fi.rec_buffer = fi.lookahead_rec_buffer.clone();

    // TODO: as in the encoding code, key frames will have no references.
    // However, for block importance purposes we want key frames to act as
    // P-frames in this instance.
    //
    // Compute the motion vectors.
    let mut blocks = FrameBlocks::new(fi.w_in_b, fi.h_in_b);

    fi.tiling
      .tile_iter_mut(&mut fs, &mut blocks)
      .collect::<Vec<_>>()
      .into_par_iter()
      .for_each(|mut ctx| {
        let ts = &mut ctx.ts;

        // Compute the quarter-resolution motion vectors.
        let tile_pmvs = build_coarse_pmvs(fi, ts);

        // Compute the half-resolution motion vectors.
        let mut half_res_pmvs = Vec::with_capacity(ts.sb_height * ts.sb_width);

        for sby in 0..ts.sb_height {
          for sbx in 0..ts.sb_width {
            let tile_sbo =
              TileSuperBlockOffset(SuperBlockOffset { x: sbx, y: sby });
            half_res_pmvs
              .push(build_half_res_pmvs(fi, ts, tile_sbo, &tile_pmvs));
          }
        }

        // Compute the full-resolution motion vectors.
        for sby in 0..ts.sb_height {
          for sbx in 0..ts.sb_width {
            let tile_sbo =
              TileSuperBlockOffset(SuperBlockOffset { x: sbx, y: sby });
            build_full_res_pmvs(fi, ts, tile_sbo, &half_res_pmvs);
          }
        }
      });

    // Save the motion vectors to FrameInvariants.
    fi.lookahead_mvs = fs.frame_mvs.clone().into_boxed_slice();

    #[cfg(feature = "dump_lookahead_data")]
    {
      use crate::partition::RefType::*;

      let second_ref_frame = if !self.inter_cfg.multiref {
        LAST_FRAME // make second_ref_frame match first
      } else if fi.idx_in_group_output == 0 {
        LAST2_FRAME
      } else {
        ALTREF_FRAME
      };

      // Use the default index, it corresponds to the last P-frame or to the
      // backwards lower reference (so the closest previous frame).
      let index = if second_ref_frame.to_index() != 0 { 0 } else { 1 };

      let mvs = &fs.frame_mvs[index];
      use byteorder::{NativeEndian, WriteBytesExt};
      let mut buf = vec![];
      buf.write_u64::<NativeEndian>(mvs.rows as u64).unwrap();
      buf.write_u64::<NativeEndian>(mvs.cols as u64).unwrap();
      for y in 0..mvs.rows {
        for x in 0..mvs.cols {
          let mv = mvs[y][x];
          buf.write_i16::<NativeEndian>(mv.row).unwrap();
          buf.write_i16::<NativeEndian>(mv.col).unwrap();
        }
      }
      ::std::fs::write(format!("{}-mvs.bin", fi.input_frameno), buf).unwrap();
    }

    // Set lookahead_rec_buffer on this FrameInvariants for future
    // FrameInvariants to pick it up.
    let rfs = Arc::new(ReferenceFrame {
      order_hint: fi.order_hint,
      // Use the original frame contents.
      frame: frame.clone(),
      input_hres: fs.input_hres,
      input_qres: fs.input_qres,
      cdfs: fs.cdfs,
      frame_mvs: fs.frame_mvs,
      output_frameno,
    });
    for i in 0..(REF_FRAMES as usize) {
      if (fi.refresh_frame_flags & (1 << i)) != 0 {
        fi.lookahead_rec_buffer.frames[i] = Some(Arc::clone(&rfs));
        fi.lookahead_rec_buffer.deblock[i] = fs.deblock;
      }
    }
  }

  /// Computes lookahead intra cost approximations and fills in
  /// `lookahead_intra_costs` on the `FrameInvariants`.
  fn compute_lookahead_intra_costs(&mut self, output_frameno: u64) {
    let fi = self.frame_invariants.get_mut(&output_frameno).unwrap();

    // We're only interested in valid frames which are not show-existing-frame.
    if fi.invalid || fi.show_existing_frame {
      return;
    }

    let frame = self.frame_q[&fi.input_frameno].as_ref().unwrap();

    let mut plane_after_prediction = frame.planes[0].clone();

    let bsize = BlockSize::from_width_and_height(
      IMPORTANCE_BLOCK_SIZE,
      IMPORTANCE_BLOCK_SIZE,
    );

    for y in 0..fi.h_in_imp_b {
      for x in 0..fi.w_in_imp_b {
        let plane_org = frame.planes[0].region(Area::Rect {
          x: (x * IMPORTANCE_BLOCK_SIZE) as isize,
          y: (y * IMPORTANCE_BLOCK_SIZE) as isize,
          width: IMPORTANCE_BLOCK_SIZE,
          height: IMPORTANCE_BLOCK_SIZE,
        });

        // TODO: other intra prediction modes.
        let edge_buf = get_intra_edges(
          &frame.planes[0].as_region(),
          TileBlockOffset(BlockOffset { x, y }),
          0,
          0,
          BlockSize::BLOCK_8X8,
          PlaneOffset {
            x: (x * IMPORTANCE_BLOCK_SIZE) as isize,
            y: (y * IMPORTANCE_BLOCK_SIZE) as isize,
          },
          TxSize::TX_8X8,
          fi.sequence.bit_depth,
          Some(PredictionMode::DC_PRED),
        );

        let mut plane_after_prediction_region = plane_after_prediction
          .region_mut(Area::Rect {
            x: (x * IMPORTANCE_BLOCK_SIZE) as isize,
            y: (y * IMPORTANCE_BLOCK_SIZE) as isize,
            width: IMPORTANCE_BLOCK_SIZE,
            height: IMPORTANCE_BLOCK_SIZE,
          });

        PredictionMode::DC_PRED.predict_intra(
          TileRect {
            x: x * IMPORTANCE_BLOCK_SIZE,
            y: y * IMPORTANCE_BLOCK_SIZE,
            width: IMPORTANCE_BLOCK_SIZE,
            height: IMPORTANCE_BLOCK_SIZE,
          },
          &mut plane_after_prediction_region,
          TxSize::TX_8X8,
          fi.sequence.bit_depth,
          &[], // Not used by DC_PRED.
          0,   // Not used by DC_PRED.
          &edge_buf,
        );

        let plane_after_prediction_region =
          plane_after_prediction.region(Area::Rect {
            x: (x * IMPORTANCE_BLOCK_SIZE) as isize,
            y: (y * IMPORTANCE_BLOCK_SIZE) as isize,
            width: IMPORTANCE_BLOCK_SIZE,
            height: IMPORTANCE_BLOCK_SIZE,
          });

        let intra_cost = get_satd(
          &plane_org,
          &plane_after_prediction_region,
          bsize,
          self.config.bit_depth,
          fi.cpu_feature_level,
        );

        fi.lookahead_intra_costs[y * fi.w_in_imp_b + x] = intra_cost;
      }
    }
  }

  pub fn compute_lookahead_data(&mut self) {
    let lookahead_frames = self
      .frame_q
      .iter()
      .filter_map(|(&input_frameno, frame)| {
        if input_frameno >= self.next_lookahead_frame {
          frame.clone()
        } else {
          None
        }
      })
      .collect::<Vec<_>>();
    let mut lookahead_idx = 0;

    while !self.needs_more_frame_q_lookahead(self.next_lookahead_frame) {
      // Process the next unprocessed frame
      // Start by getting that frame and all frames after it in the queue
      let current_lookahead_frames = &lookahead_frames[lookahead_idx..];

      if current_lookahead_frames.is_empty() {
        // All frames have been processed
        break;
      }

      self.keyframe_detector.analyze_next_frame(
        if self.next_lookahead_frame == 0 || self.config.still_picture {
          None
        } else {
          self
            .frame_q
            .get(&(self.next_lookahead_frame - 1))
            .map(|f| f.as_ref().unwrap().clone())
        },
        &current_lookahead_frames,
        self.next_lookahead_frame,
        &self.config,
        &self.inter_cfg,
        &mut self.keyframes,
        &self.keyframes_forced,
      );

      self.next_lookahead_frame += 1;
      lookahead_idx += 1;
    }

    // Compute the frame invariants.
    while self.set_frame_properties(self.next_lookahead_output_frameno).is_ok()
    {
      self
        .compute_lookahead_motion_vectors(self.next_lookahead_output_frameno);
      self.compute_lookahead_intra_costs(self.next_lookahead_output_frameno);
      self.next_lookahead_output_frameno += 1;
    }
  }

  /// Computes the block importances for the current output frame.
  fn compute_block_importances(&mut self) {
    // SEF don't need block importances.
    if self.frame_invariants[&self.output_frameno].show_existing_frame {
      return;
    }

    // Get a list of output_framenos that we want to propagate through.
    let output_framenos = self
      .get_rdo_lookahead_frames()
      .map(|(&output_frameno, _)| output_frameno)
      .collect::<Vec<_>>();

    // The first one should be the current output frame.
    assert_eq!(output_framenos[0], self.output_frameno);

    // First, initialize them all with zeros.
    for output_frameno in output_framenos.iter() {
      let fi = self.frame_invariants.get_mut(output_frameno).unwrap();
      for x in fi.block_importances.iter_mut() {
        *x = 0.;
      }
    }

    // Now compute and propagate the block importances from the end. The
    // current output frame will get its block importances from the future
    // frames.
    const MV_UNITS_PER_PIXEL: i64 = 8;
    const BLOCK_SIZE_IN_MV_UNITS: i64 =
      IMPORTANCE_BLOCK_SIZE as i64 * MV_UNITS_PER_PIXEL;
    const BLOCK_AREA_IN_MV_UNITS: i64 =
      BLOCK_SIZE_IN_MV_UNITS * BLOCK_SIZE_IN_MV_UNITS;

    let bsize = BlockSize::from_width_and_height(
      IMPORTANCE_BLOCK_SIZE,
      IMPORTANCE_BLOCK_SIZE,
    );

    for &output_frameno in output_framenos.iter().skip(1).rev() {
      // Remove fi from the map temporarily and put it back in in the end of
      // the iteration. This is required because we need to mutably borrow
      // referenced fis from the map, and that wouldn't be possible if this was
      // an active borrow.
      let fi = self.frame_invariants.remove(&output_frameno).unwrap();

      // TODO: see comment above about key frames not having references.
      if fi.frame_type == FrameType::KEY {
        self.frame_invariants.insert(output_frameno, fi);
        continue;
      }

      let frame = self.frame_q[&fi.input_frameno].as_ref().unwrap();

      // There can be at most 3 of these.
      let mut unique_indices = ArrayVec::<[_; 3]>::new();

      for (mv_index, &rec_index) in fi.ref_frames.iter().enumerate() {
        if unique_indices.iter().find(|&&(_, r)| r == rec_index).is_none() {
          unique_indices.push((mv_index, rec_index));
        }
      }

      // Compute and propagate the importance, split evenly between the
      // referenced frames.
      for &(mv_index, rec_index) in unique_indices.iter() {
        // Use rec_buffer here rather than lookahead_rec_buffer because
        // rec_buffer still contains the reference frames for the current frame
        // (it's only overwritten when the frame is encoded), while
        // lookahead_rec_buffer already contains reference frames for the next
        // frame (for the reference propagation to work correctly).
        let reference =
          fi.rec_buffer.frames[rec_index as usize].as_ref().unwrap();
        let reference_frame = &reference.frame;
        let reference_output_frameno = reference.output_frameno;

        // We should never use frame as its own reference.
        assert_ne!(reference_output_frameno, output_frameno);

        for y in 0..fi.h_in_imp_b {
          for x in 0..fi.w_in_imp_b {
            let mv = fi.lookahead_mvs[mv_index][y * 2][x * 2];

            // Coordinates of the top-left corner of the reference block, in MV
            // units.
            let reference_x =
              x as i64 * BLOCK_SIZE_IN_MV_UNITS + mv.col as i64;
            let reference_y =
              y as i64 * BLOCK_SIZE_IN_MV_UNITS + mv.row as i64;

            let plane_org = frame.planes[0].region(Area::Rect {
              x: (x * IMPORTANCE_BLOCK_SIZE) as isize,
              y: (y * IMPORTANCE_BLOCK_SIZE) as isize,
              width: IMPORTANCE_BLOCK_SIZE,
              height: IMPORTANCE_BLOCK_SIZE,
            });

            let plane_ref = reference_frame.planes[0].region(Area::Rect {
              x: reference_x as isize / MV_UNITS_PER_PIXEL as isize,
              y: reference_y as isize / MV_UNITS_PER_PIXEL as isize,
              width: IMPORTANCE_BLOCK_SIZE,
              height: IMPORTANCE_BLOCK_SIZE,
            });

            let inter_cost = get_satd(
              &plane_org,
              &plane_ref,
              bsize,
              self.config.bit_depth,
              fi.cpu_feature_level,
            ) as f32;

            let intra_cost =
              fi.lookahead_intra_costs[y * fi.w_in_imp_b + x] as f32;

            let future_importance =
              fi.block_importances[y * fi.w_in_imp_b + x];

            let propagate_fraction = (1. - inter_cost / intra_cost).max(0.);
            let propagate_amount = (intra_cost + future_importance)
              * propagate_fraction
              / unique_indices.len() as f32;

            if let Some(reference_frame_block_importances) = self
              .frame_invariants
              .get_mut(&reference_output_frameno)
              .map(|fi| &mut fi.block_importances)
            {
              let mut propagate =
                |block_x_in_mv_units, block_y_in_mv_units, fraction| {
                  let x = block_x_in_mv_units / BLOCK_SIZE_IN_MV_UNITS;
                  let y = block_y_in_mv_units / BLOCK_SIZE_IN_MV_UNITS;

                  // TODO: propagate partially if the block is partially off-frame
                  // (possible on right and bottom edges)?
                  if x >= 0
                    && y >= 0
                    && (x as usize) < fi.w_in_imp_b
                    && (y as usize) < fi.h_in_imp_b
                  {
                    reference_frame_block_importances
                      [y as usize * fi.w_in_imp_b + x as usize] +=
                      propagate_amount * fraction;
                  }
                };

              // Coordinates of the top-left corner of the block intersecting the
              // reference block from the top-left.
              let top_left_block_x = (reference_x
                - if reference_x < 0 {
                  BLOCK_SIZE_IN_MV_UNITS - 1
                } else {
                  0
                })
                / BLOCK_SIZE_IN_MV_UNITS
                * BLOCK_SIZE_IN_MV_UNITS;
              let top_left_block_y = (reference_y
                - if reference_y < 0 {
                  BLOCK_SIZE_IN_MV_UNITS - 1
                } else {
                  0
                })
                / BLOCK_SIZE_IN_MV_UNITS
                * BLOCK_SIZE_IN_MV_UNITS;

              debug_assert!(reference_x >= top_left_block_x);
              debug_assert!(reference_y >= top_left_block_y);

              let top_right_block_x =
                top_left_block_x + BLOCK_SIZE_IN_MV_UNITS;
              let top_right_block_y = top_left_block_y;
              let bottom_left_block_x = top_left_block_x;
              let bottom_left_block_y =
                top_left_block_y + BLOCK_SIZE_IN_MV_UNITS;
              let bottom_right_block_x = top_right_block_x;
              let bottom_right_block_y = bottom_left_block_y;

              let top_left_block_fraction = ((top_right_block_x - reference_x)
                * (bottom_left_block_y - reference_y))
                as f32
                / BLOCK_AREA_IN_MV_UNITS as f32;

              propagate(
                top_left_block_x,
                top_left_block_y,
                top_left_block_fraction,
              );

              let top_right_block_fraction =
                ((reference_x + BLOCK_SIZE_IN_MV_UNITS - top_right_block_x)
                  * (bottom_left_block_y - reference_y))
                  as f32
                  / BLOCK_AREA_IN_MV_UNITS as f32;

              propagate(
                top_right_block_x,
                top_right_block_y,
                top_right_block_fraction,
              );

              let bottom_left_block_fraction = ((top_right_block_x
                - reference_x)
                * (reference_y + BLOCK_SIZE_IN_MV_UNITS - bottom_left_block_y))
                as f32
                / BLOCK_AREA_IN_MV_UNITS as f32;

              propagate(
                bottom_left_block_x,
                bottom_left_block_y,
                bottom_left_block_fraction,
              );

              let bottom_right_block_fraction =
                ((reference_x + BLOCK_SIZE_IN_MV_UNITS - top_right_block_x)
                  * (reference_y + BLOCK_SIZE_IN_MV_UNITS
                    - bottom_left_block_y)) as f32
                  / BLOCK_AREA_IN_MV_UNITS as f32;

              propagate(
                bottom_right_block_x,
                bottom_right_block_y,
                bottom_right_block_fraction,
              );
            }
          }
        }
      }

      self.frame_invariants.insert(output_frameno, fi);
    }

    // Get the final block importance values for the current output frame.
    if !output_framenos.is_empty() {
      let fi = self.frame_invariants.get_mut(&output_framenos[0]).unwrap();

      for y in 0..fi.h_in_imp_b {
        for x in 0..fi.w_in_imp_b {
          let intra_cost =
            fi.lookahead_intra_costs[y * fi.w_in_imp_b + x] as f32;

          let importance = &mut fi.block_importances[y * fi.w_in_imp_b + x];
          if intra_cost > 0. {
            *importance = (1. + *importance / intra_cost).log2();
          } else {
            *importance = 0.;
          }

          assert!(*importance >= 0.);
        }
      }

      #[cfg(feature = "dump_lookahead_data")]
      {
        let data = &fi.block_importances;
        use byteorder::{NativeEndian, WriteBytesExt};
        let mut buf = vec![];
        buf.write_u64::<NativeEndian>(fi.h_in_imp_b as u64).unwrap();
        buf.write_u64::<NativeEndian>(fi.w_in_imp_b as u64).unwrap();
        for y in 0..fi.h_in_imp_b {
          for x in 0..fi.w_in_imp_b {
            let importance = data[y * fi.w_in_imp_b + x];
            buf.write_f32::<NativeEndian>(importance).unwrap();
          }
        }
        ::std::fs::write(format!("{}-imps.bin", fi.input_frameno), buf)
          .unwrap();
      }
    }
  }

  pub fn receive_packet(&mut self) -> Result<Packet<T>, EncoderStatus> {
    if self.done_processing() {
      return Err(EncoderStatus::LimitReached);
    }

    if self.needs_more_fi_lookahead() {
      return Err(EncoderStatus::NeedMoreData);
    }

    // Find the next output_frameno corresponding to a non-skipped frame.
    self.output_frameno = self
      .frame_invariants
      .iter()
      .skip_while(|(&output_frameno, _)| output_frameno < self.output_frameno)
      .find(|(_, fi)| !fi.invalid)
      .map(|(&output_frameno, _)| output_frameno)
      .ok_or(EncoderStatus::NeedMoreData)?; // TODO: doesn't play well with the below check?

    let input_frameno =
      self.frame_invariants[&self.output_frameno].input_frameno;
    if !self.needs_more_frames(input_frameno) {
      return Err(EncoderStatus::LimitReached);
    }

    // Compute the block importances for the current output frame.
    self.compute_block_importances();

    let cur_output_frameno = self.output_frameno;

    let ret = {
      let fi = self.frame_invariants.get(&cur_output_frameno).unwrap();
      if fi.show_existing_frame {
        if !self.rc_state.ready() {
          return Err(EncoderStatus::NotReady);
        }
        let mut fs = FrameState::new(fi);

        let sef_data = encode_show_existing_frame(fi, &mut fs);
        let bits = (sef_data.len() * 8) as i64;
        self.packet_data.extend(sef_data);
        self.rc_state.update_state(
          bits,
          FRAME_SUBTYPE_SEF,
          fi.show_frame,
          0,
          false,
          false,
        );
        let rec = if fi.show_frame { Some(fs.rec) } else { None };
        self.output_frameno += 1;

        let input_frameno = fi.input_frameno;
        let frame_type = fi.frame_type;
        let bit_depth = fi.sequence.bit_depth;
        let qp = fi.base_q_idx;
        self.finalize_packet(
          rec,
          input_frameno,
          frame_type,
          bit_depth,
          qp,
          fs.enc_stats,
        )
      } else if let Some(f) = self.frame_q.get(&fi.input_frameno) {
        if !self.rc_state.ready() {
          return Err(EncoderStatus::NotReady);
        }
        if let Some(frame) = f.clone() {
          let fti = fi.get_frame_subtype();
          let qps = self.rc_state.select_qi(
            self,
            self.output_frameno,
            fti,
            self.maybe_prev_log_base_q,
          );
          let fi = self.frame_invariants.get_mut(&cur_output_frameno).unwrap();
          fi.set_quantizers(&qps);

          if self.rc_state.needs_trial_encode(fti) {
            let mut fs = FrameState::new_with_frame(fi, frame.clone());
            let data = encode_frame(fi, &mut fs);
            self.rc_state.update_state(
              (data.len() * 8) as i64,
              fti,
              fi.show_frame,
              qps.log_target_q,
              true,
              false,
            );
            let qps = self.rc_state.select_qi(
              self,
              self.output_frameno,
              fti,
              self.maybe_prev_log_base_q,
            );
            let fi =
              self.frame_invariants.get_mut(&cur_output_frameno).unwrap();
            fi.set_quantizers(&qps);
          }

          let fi = self.frame_invariants.get_mut(&cur_output_frameno).unwrap();
          let mut fs = FrameState::new_with_frame(fi, frame.clone());
          let data = encode_frame(fi, &mut fs);
          let enc_stats = fs.enc_stats.clone();
          self.maybe_prev_log_base_q = Some(qps.log_base_q);
          // TODO: Add support for dropping frames.
          self.rc_state.update_state(
            (data.len() * 8) as i64,
            fti,
            fi.show_frame,
            qps.log_target_q,
            false,
            false,
          );
          self.packet_data.extend(data);

          fs.rec.pad(fi.width, fi.height);

          // TODO avoid the clone by having rec Arc.
          let rec = if fi.show_frame { Some(fs.rec.clone()) } else { None };

          update_rec_buffer(self.output_frameno, fi, fs);

          // Copy persistent fields into subsequent FrameInvariants.
          let rec_buffer = fi.rec_buffer.clone();
          for subsequent_fi in self
            .frame_invariants
            .iter_mut()
            .skip_while(|(&output_frameno, _)| {
              output_frameno <= cur_output_frameno
            })
            .map(|(_, fi)| fi)
            // Here we want the next valid non-show-existing-frame inter frame.
            //
            // Copying to show-existing-frame frames isn't actually required
            // for correct encoding, but it's needed for the reconstruction to
            // work correctly.
            .filter(|fi| !fi.invalid)
            .take_while(|fi| fi.frame_type != FrameType::KEY)
          {
            subsequent_fi.rec_buffer = rec_buffer.clone();
            subsequent_fi.set_ref_frame_sign_bias();

            // Stop after the first non-show-existing-frame.
            if !subsequent_fi.show_existing_frame {
              break;
            }
          }

          let fi = self.frame_invariants.get(&self.output_frameno).unwrap();

          self.output_frameno += 1;

          if fi.show_frame {
            let input_frameno = fi.input_frameno;
            let frame_type = fi.frame_type;
            let bit_depth = fi.sequence.bit_depth;
            let qp = fi.base_q_idx;
            self.finalize_packet(
              rec,
              input_frameno,
              frame_type,
              bit_depth,
              qp,
              enc_stats,
            )
          } else {
            Err(EncoderStatus::Encoded)
          }
        } else {
          Err(EncoderStatus::NeedMoreData)
        }
      } else {
        Err(EncoderStatus::NeedMoreData)
      }
    };

    if let Ok(ref pkt) = ret {
      self.garbage_collect(pkt.input_frameno);
    }

    ret
  }

  fn finalize_packet(
    &mut self, rec: Option<Frame<T>>, input_frameno: u64,
    frame_type: FrameType, bit_depth: usize, qp: u8, enc_stats: EncoderStats,
  ) -> Result<Packet<T>, EncoderStatus> {
    let data = self.packet_data.clone();
    self.packet_data.clear();
    if write_temporal_delimiter(&mut self.packet_data).is_err() {
      return Err(EncoderStatus::Failure);
    }

    let mut psnr = None;
    if self.config.show_psnr {
      if let Some(ref rec) = rec {
        let original_frame = self.get_frame(input_frameno);
        psnr = Some(calculate_frame_psnr(&*original_frame, rec, bit_depth));
      }
    }

    self.frames_processed += 1;
    Ok(Packet { data, rec, input_frameno, frame_type, psnr, qp, enc_stats })
  }

  fn garbage_collect(&mut self, cur_input_frameno: u64) {
    if cur_input_frameno == 0 {
      return;
    }
    let frame_q_start = self.frame_q.keys().next().cloned().unwrap_or(0);
    for i in frame_q_start..cur_input_frameno {
      self.frame_q.remove(&i);
    }

    if self.output_frameno < 2 {
      return;
    }
    let fi_start = self.frame_invariants.keys().next().cloned().unwrap_or(0);
    for i in fi_start..(self.output_frameno - 1) {
      self.frame_invariants.remove(&i);
      self.gop_output_frameno_start.remove(&i);
      self.gop_input_frameno_start.remove(&i);
    }
  }

  /// Counts the number of output frames of each subtype in the next
  ///  reservoir_frame_delay temporal units (needed for rate control).
  /// Returns the number of output frames (excluding SEF frames) and output TUs
  ///  until the last keyframe in the next reservoir_frame_delay temporal units,
  ///  or the end of the interval, whichever comes first.
  /// The former is needed because it indicates the number of rate estimates we
  ///  will make.
  /// The latter is needed because it indicates the number of times new bitrate
  ///  is added to the buffer.
  pub(crate) fn guess_frame_subtypes(
    &self, nframes: &mut [i32; FRAME_NSUBTYPES + 1],
    reservoir_frame_delay: i32,
  ) -> (i32, i32) {
    for fti in 0..=FRAME_NSUBTYPES {
      nframes[fti] = 0;
    }

    // Two-pass calls this function before receive_packet(), and in particular
    // before the very first send_frame(), when the following maps are empty.
    // In this case, return 0 as the default value.
    let mut prev_keyframe_input_frameno = *self
      .gop_input_frameno_start
      .get(&self.output_frameno)
      .unwrap_or_else(|| {
        assert!(self.output_frameno == 0);
        &0
      });
    let mut prev_keyframe_output_frameno = *self
      .gop_output_frameno_start
      .get(&self.output_frameno)
      .unwrap_or_else(|| {
        assert!(self.output_frameno == 0);
        &0
      });

    let mut prev_keyframe_ntus = 0;
    // Does not include SEF frames.
    let mut prev_keyframe_nframes = 0;
    let mut acc: [i32; FRAME_NSUBTYPES + 1] = [0; FRAME_NSUBTYPES + 1];
    // Updates the frame counts with the accumulated values when we hit a
    //  keyframe.
    fn collect_counts(
      nframes: &mut [i32; FRAME_NSUBTYPES + 1],
      acc: &mut [i32; FRAME_NSUBTYPES + 1],
    ) {
      for fti in 0..=FRAME_NSUBTYPES {
        nframes[fti] += acc[fti];
        acc[fti] = 0;
      }
      acc[FRAME_SUBTYPE_I] += 1;
    }
    let mut output_frameno = self.output_frameno;
    let mut ntus = 0;
    // Does not include SEF frames.
    let mut nframes_total = 0;
    while ntus < reservoir_frame_delay {
      let output_frameno_in_gop =
        output_frameno - prev_keyframe_output_frameno;
      let is_kf = if let Some(fi) = self.frame_invariants.get(&output_frameno)
      {
        if fi.frame_type == FrameType::KEY {
          prev_keyframe_input_frameno = fi.input_frameno;
          // We do not currently use forward keyframes, so they should always
          //  end the current TU (thus we always increment ntus below).
          debug_assert!(fi.show_frame);
          true
        } else {
          false
        }
      } else {
        // It is possible to be invoked for the first time from twopass_out()
        //  before receive_packet() is called, in which case frame_invariants
        //  will not be populated.
        // Force the first frame in each GOP to be a keyframe in that case.
        output_frameno_in_gop == 0
      };
      if is_kf {
        collect_counts(nframes, &mut acc);
        prev_keyframe_output_frameno = output_frameno;
        prev_keyframe_ntus = ntus;
        prev_keyframe_nframes = nframes_total;
        output_frameno += 1;
        ntus += 1;
        nframes_total += 1;
        continue;
      }
      let idx_in_group_output =
        self.inter_cfg.get_idx_in_group_output(output_frameno_in_gop);
      let input_frameno = prev_keyframe_input_frameno
        + self
          .inter_cfg
          .get_order_hint(output_frameno_in_gop, idx_in_group_output)
          as u64;
      // For rate control purposes, ignore any limit on frame count that has
      //  been set.
      // We pretend that we will keep encoding frames forever to prevent the
      //  control loop from driving us into the rails as we come up against a
      //  hard stop (with no more chance to correct outstanding errors).
      let next_keyframe_input_frameno =
        self.next_keyframe_input_frameno(prev_keyframe_input_frameno, true);
      // If we are re-ordering, we may skip some output frames in the final
      //  re-order group of the GOP.
      if input_frameno >= next_keyframe_input_frameno {
        // If we have encoded enough whole groups to reach the next keyframe,
        //  then start the next keyframe gop.
        if 1
          + (output_frameno - prev_keyframe_output_frameno)
            / self.inter_cfg.group_output_len
            * self.inter_cfg.group_input_len
          >= next_keyframe_input_frameno - prev_keyframe_input_frameno
        {
          collect_counts(nframes, &mut acc);
          prev_keyframe_input_frameno = input_frameno;
          prev_keyframe_output_frameno = output_frameno;
          prev_keyframe_ntus = ntus;
          prev_keyframe_nframes = nframes_total;
          // We do not currently use forward keyframes, so they should always
          //  end the current TU.
          output_frameno += 1;
          ntus += 1;
        }
        output_frameno += 1;
        continue;
      }
      if self.inter_cfg.get_show_existing_frame(idx_in_group_output) {
        acc[FRAME_SUBTYPE_SEF] += 1;
      } else {
        // TODO: Implement golden P-frames.
        let fti = FRAME_SUBTYPE_P
          + (self.inter_cfg.get_level(idx_in_group_output) as usize);
        acc[fti] += 1;
        nframes_total += 1;
      }
      if self.inter_cfg.get_show_frame(idx_in_group_output) {
        ntus += 1;
      }
      output_frameno += 1;
    }
    if prev_keyframe_output_frameno <= self.output_frameno {
      // If there were no keyframes at all, or only the first frame was a
      //  keyframe, the accumulators never flushed and still contain counts for
      //  the entire buffer.
      // In both cases, we return these counts.
      collect_counts(nframes, &mut acc);
      (nframes_total, ntus)
    } else {
      // Otherwise, we discard what remains in the accumulators as they contain
      //  the counts from and past the last keyframe.
      (prev_keyframe_nframes, prev_keyframe_ntus)
    }
  }
}
