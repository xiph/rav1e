// Copyright (c) 2018-2022, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.
#![deny(missing_docs)]

use crate::activity::ActivityMask;
use crate::api::lookahead::*;
use crate::api::{
  EncoderConfig, EncoderStatus, FrameType, Opaque, Packet, T35,
};
use crate::color::ChromaSampling::Cs400;
use crate::cpu_features::CpuFeatureLevel;
use crate::dist::get_satd;
use crate::encoder::*;
use crate::frame::*;
use crate::partition::*;
use crate::rate::{
  RCState, FRAME_NSUBTYPES, FRAME_SUBTYPE_I, FRAME_SUBTYPE_P,
  FRAME_SUBTYPE_SEF,
};
use crate::scenechange::SceneChangeDetector;
use crate::stats::EncoderStats;
use crate::tiling::Area;
use crate::util::Pixel;
use arrayvec::ArrayVec;
use debug_unreachable::debug_unreachable;
use itertools::Itertools;
use rust_hawktracer::*;
use std::cmp;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub(crate) struct MiniGopConfig {
  pub group_input_len: usize,
  pub pyramid_depth: u8,
}

// Thin wrapper for frame-related data
// that gets cached and reused throughout the life of a frame.
#[derive(Clone)]
pub(crate) struct FrameData<T: Pixel> {
  pub(crate) fi: FrameInvariants<T>,
  pub(crate) fs: FrameState<T>,
}

impl<T: Pixel> FrameData<T> {
  pub(crate) fn new(fi: FrameInvariants<T>, frame: Arc<Frame<T>>) -> Self {
    let fs = FrameState::new_with_frame(&fi, frame);
    FrameData { fi, fs }
  }
}

type FrameQueue<T> = BTreeMap<u64, Option<Arc<Frame<T>>>>;
type FrameDataQueue<T> = BTreeMap<u64, Option<FrameData<T>>>;

// the fields pub(super) are accessed only by the tests
pub(crate) struct ContextInner<T: Pixel> {
  pub(crate) frame_count: u64,
  pub(crate) limit: Option<u64>,
  pub(crate) output_frameno: u64,
  pub(super) minigop_config: MiniGopConfig,
  pub(super) frames_processed: u64,
  /// Maps *input_frameno* to frames
  pub(super) frame_q: FrameQueue<T>,
  /// Maps *output_frameno* to frame data
  pub(super) frame_data: FrameDataQueue<T>,
  /// A list of the precomputed frame types and pyramid depth for frames within the lookahead.
  /// This allows us to have dynamic pyramid depths and widths by computing them before
  /// creating the frame invariants.
  frame_depths: BTreeMap<u64, FrameDepth>,
  // TODO: Is this needed at all?
  keyframes_forced: BTreeSet<u64>,
  /// A storage space for reordered frames.
  packet_data: Vec<u8>,
  /// Maps `output_frameno` to `gop_output_frameno_start`.
  gop_output_frameno_start: BTreeMap<u64, u64>,
  /// Maps `output_frameno` to `gop_input_frameno_start`.
  pub(crate) gop_input_frameno_start: BTreeMap<u64, u64>,
  /// Maps `output_frameno` to `minigop_output_frameno_start`.
  minigop_output_frameno_start: BTreeMap<u64, u64>,
  /// Maps `output_frameno` to `minigop_input_frameno_start`.
  pub(crate) minigop_input_frameno_start: BTreeMap<u64, u64>,
  frame_type_lookahead_distance: usize,
  keyframe_detector: SceneChangeDetector<T>,
  pub(crate) config: Arc<EncoderConfig>,
  seq: Arc<Sequence>,
  pub(crate) rc_state: RCState,
  maybe_prev_log_base_q: Option<i64>,
  /// The next `input_frameno` to be processed by lookahead.
  next_lookahead_frame: u64,
  /// The next `output_frameno` to be computed by lookahead.
  next_lookahead_output_frameno: u64,
  /// Optional opaque to be sent back to the user
  opaque_q: BTreeMap<u64, Opaque>,
  /// Optional T35 metadata per frame
  t35_q: BTreeMap<u64, Box<[T35]>>,
}

impl<T: Pixel> ContextInner<T> {
  pub fn new(enc: &EncoderConfig) -> Self {
    // initialize with temporal delimiter
    let packet_data = TEMPORAL_DELIMITER.to_vec();
    let mut frame_depths = BTreeMap::new();
    frame_depths.insert(0, FrameDepth::Intra);

    let maybe_ac_qi_max =
      if enc.quantizer < 255 { Some(enc.quantizer as u8) } else { None };

    let seq = Arc::new(Sequence::new(enc));
    let lookahead_distance = enc.speed_settings.rdo_lookahead_frames.min(32);

    ContextInner {
      frame_count: 0,
      limit: None,
      minigop_config: MiniGopConfig { group_input_len: 1, pyramid_depth: 0 },
      output_frameno: 0,
      frames_processed: 0,
      frame_q: BTreeMap::new(),
      frame_data: BTreeMap::new(),
      frame_depths,
      keyframes_forced: BTreeSet::new(),
      packet_data,
      gop_output_frameno_start: BTreeMap::new(),
      gop_input_frameno_start: BTreeMap::new(),
      minigop_output_frameno_start: BTreeMap::new(),
      minigop_input_frameno_start: BTreeMap::new(),
      frame_type_lookahead_distance: lookahead_distance,
      keyframe_detector: SceneChangeDetector::new(
        enc.clone(),
        CpuFeatureLevel::default(),
        lookahead_distance,
        seq.clone(),
      ),
      config: Arc::new(enc.clone()),
      seq,
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
      next_lookahead_frame: 1,
      next_lookahead_output_frameno: 0,
      opaque_q: BTreeMap::new(),
      t35_q: BTreeMap::new(),
    }
  }

  pub(crate) fn allowed_ref_frames(&self) -> &[RefType] {
    use crate::partition::RefType::*;

    let reorder = self.config.reorder();
    let multiref = self.config.multiref();
    if reorder {
      &ALL_INTER_REFS
    } else if multiref {
      &[LAST_FRAME, LAST2_FRAME, LAST3_FRAME, GOLDEN_FRAME]
    } else {
      &[LAST_FRAME]
    }
  }

  #[hawktracer(send_frame)]
  pub fn send_frame(
    &mut self, mut frame: Option<Arc<Frame<T>>>,
    params: Option<FrameParameters>,
  ) -> Result<(), EncoderStatus> {
    if let Some(ref mut frame) = frame {
      use crate::api::color::ChromaSampling;
      let EncoderConfig { width, height, chroma_sampling, .. } = *self.config;
      let planes =
        if chroma_sampling == ChromaSampling::Cs400 { 1 } else { 3 };
      // Try to add padding
      if let Some(ref mut frame) = Arc::get_mut(frame) {
        for plane in frame.planes[..planes].iter_mut() {
          plane.pad(width, height);
        }
      }
      // Enforce that padding is added
      for (p, plane) in frame.planes[..planes].iter().enumerate() {
        assert!(
          plane.probe_padding(width, height),
          "Plane {p} was not padded before passing Frame to send_frame()."
        );
      }
    }

    let input_frameno = self.frame_count;
    let is_flushing = frame.is_none();
    if !is_flushing {
      self.frame_count += 1;
    }
    self.frame_q.insert(input_frameno, frame);

    if let Some(params) = params {
      if params.frame_type_override == FrameTypeOverride::Key {
        self.keyframes_forced.insert(input_frameno);
      }
      if let Some(op) = params.opaque {
        self.opaque_q.insert(input_frameno, op);
      }
      self.t35_q.insert(input_frameno, params.t35_metadata);
    }

    if !self.needs_more_frame_q_lookahead(self.next_lookahead_frame) {
      let lookahead_frames = self
        .frame_q
        .range(self.next_lookahead_frame - 1..)
        .filter_map(|(&_input_frameno, frame)| frame.as_ref().map(Arc::clone))
        .collect::<Vec<Arc<Frame<T>>>>();

      if is_flushing {
        // This is the last time send_frame is called, process all the
        // remaining frames.
        for cur_lookahead_frames in
          std::iter::successors(Some(&lookahead_frames[..]), |s| s.get(1..))
        {
          if cur_lookahead_frames.len() < 2 {
            // All frames have been processed
            break;
          }

          self.compute_frame_placement(cur_lookahead_frames);
        }
      } else {
        self.compute_frame_placement(&lookahead_frames);
      }
    }

    self.compute_frame_invariants();

    Ok(())
  }

  /// Indicates whether more frames need to be read into the frame queue
  /// in order for frame queue lookahead to be full.
  fn needs_more_frame_q_lookahead(&self, input_frameno: u64) -> bool {
    let lookahead_end = self.frame_q.keys().last().cloned().unwrap_or(0);
    let frames_needed =
      input_frameno + self.frame_type_lookahead_distance as u64 + 1;
    lookahead_end < frames_needed && self.needs_more_frames(lookahead_end)
  }

  /// Indicates whether more frames need to be processed into `FrameInvariants`
  /// in order for FI lookahead to be full.
  pub fn needs_more_fi_lookahead(&self) -> bool {
    let ready_frames = self.get_rdo_lookahead_frames().count();
    ready_frames < self.config.speed_settings.rdo_lookahead_frames + 1
      && self.needs_more_frames(self.next_lookahead_frame)
  }

  pub fn needs_more_frames(&self, frame_count: u64) -> bool {
    self.limit.map(|limit| frame_count < limit).unwrap_or(true)
  }

  fn get_rdo_lookahead_frames(
    &self,
  ) -> impl Iterator<Item = (&u64, &FrameData<T>)> {
    self
      .frame_data
      .iter()
      .skip_while(move |(&output_frameno, _)| {
        output_frameno < self.output_frameno
      })
      .filter_map(|(fno, data)| data.as_ref().map(|data| (fno, data)))
      .filter(|(_, data)| !data.fi.is_show_existing_frame())
      .take(self.config.speed_settings.rdo_lookahead_frames + 1)
  }

  fn next_minigop_input_frameno(
    &self, minigop_input_frameno_start: u64, ignore_limit: bool,
  ) -> u64 {
    let next_detected = self
      .frame_depths
      .iter()
      .find(|&(&input_frameno, frame_depth)| {
        (frame_depth == &FrameDepth::Intra
          || frame_depth
            == &FrameDepth::Inter { depth: 0, is_minigop_start: true })
          && input_frameno > minigop_input_frameno_start
      })
      .map(|(input_frameno, _)| *input_frameno);
    let mut next_limit =
      minigop_input_frameno_start + self.config.max_key_frame_interval;
    if !ignore_limit && self.limit.is_some() {
      next_limit = next_limit.min(self.limit.unwrap());
    }
    if next_detected.is_none() {
      return next_limit;
    }
    cmp::min(next_detected.unwrap(), next_limit)
  }

  fn next_keyframe_input_frameno(
    &self, gop_input_frameno_start: u64, ignore_limit: bool,
  ) -> u64 {
    let next_detected = self
      .frame_depths
      .iter()
      .find(|&(&input_frameno, frame_depth)| {
        frame_depth == &FrameDepth::Intra
          && input_frameno > gop_input_frameno_start
      })
      .map(|(input_frameno, _)| *input_frameno);
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

    self.frame_data.insert(
      output_frameno,
      fi.map(|fi| {
        let frame = self
          .frame_q
          .get(&fi.input_frameno)
          .as_ref()
          .unwrap()
          .as_ref()
          .unwrap();
        FrameData::new(fi, frame.clone())
      }),
    );

    Ok(())
  }

  #[allow(unused)]
  pub fn build_dump_properties() -> PathBuf {
    let mut data_location = PathBuf::new();
    if env::var_os("RAV1E_DATA_PATH").is_some() {
      data_location.push(&env::var_os("RAV1E_DATA_PATH").unwrap());
    } else {
      data_location.push(&env::current_dir().unwrap());
      data_location.push(".lookahead_data");
    }
    fs::create_dir_all(&data_location).unwrap();
    data_location
  }

  fn get_input_frameno(
    &mut self, output_frameno: u64, minigop_input_frameno_start: u64,
  ) -> u64 {
    let next_minigop_start = self
      .frame_depths
      .range((minigop_input_frameno_start + 1)..)
      .find(|&(_, depth)| depth.is_minigop_start())
      .map(|(frameno, _)| *frameno);
    let minigop_end = next_minigop_start
      .unwrap_or_else(|| *self.frame_depths.keys().last().unwrap());
    let minigop_depth = self
      .frame_depths
      .range(minigop_input_frameno_start..=minigop_end)
      .map(|(_, depth)| depth.depth())
      .max()
      .unwrap();
    let last_fi = &self.frame_data.last_key_value().unwrap().1.unwrap().fi;
    let next_input_frameno = self
      .frame_depths
      .range(minigop_input_frameno_start..=minigop_end)
      .find(|(frameno, depth)| {
        depth.depth() == last_fi.pyramid_level
          && **frameno > last_fi.input_frameno
      })
      .or_else(|| {
        self
          .frame_depths
          .range(minigop_input_frameno_start..=minigop_end)
          .find(|(_, depth)| depth.depth() == last_fi.pyramid_level + 1)
      })
      .map(|(frameno, _)| *frameno);
    if let Some(frameno) = next_input_frameno {
      frameno
    } else {
      // This frame starts a new minigop
      let input_frameno = last_fi.input_frameno + 1;
      self.minigop_output_frameno_start.insert(output_frameno, output_frameno);
      self.minigop_input_frameno_start.insert(output_frameno, input_frameno);
      self.minigop_config = MiniGopConfig {
        group_input_len: (minigop_end - minigop_input_frameno_start + 1)
          as usize,
        pyramid_depth: minigop_depth,
      };
      input_frameno
    }
  }

  fn build_frame_properties(
    &mut self, output_frameno: u64,
  ) -> Result<Option<FrameInvariants<T>>, EncoderStatus> {
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

    let (prev_minigop_output_frameno_start, prev_minigop_input_frameno_start) =
      if output_frameno == 0 {
        (0, 0)
      } else {
        (
          self.minigop_output_frameno_start[&(output_frameno - 1)],
          self.minigop_input_frameno_start[&(output_frameno - 1)],
        )
      };

    self
      .minigop_output_frameno_start
      .insert(output_frameno, prev_minigop_output_frameno_start);
    self
      .minigop_input_frameno_start
      .insert(output_frameno, prev_minigop_input_frameno_start);

    let mut input_frameno = self.get_input_frameno(
      output_frameno,
      self.minigop_input_frameno_start[&output_frameno],
    );

    if self.needs_more_frame_q_lookahead(input_frameno) {
      return Err(EncoderStatus::NeedMoreData);
    }

    let t35_metadata = if let Some(t35) = self.t35_q.remove(&input_frameno) {
      t35
    } else {
      Box::new([])
    };

    match self.frame_q.get(&input_frameno) {
      Some(Some(_)) => {}
      _ => {
        return Err(EncoderStatus::NeedMoreData);
      }
    }

    // Now that we know the input_frameno, look up the correct frame type
    let frame_type = if self.frame_depths[&input_frameno] == FrameDepth::Intra
    {
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
        self.seq.clone(),
        self.gop_input_frameno_start[&output_frameno],
        t35_metadata,
      );
      Ok(Some(fi))
    } else {
      let minigop_input_frameno_start =
        self.minigop_input_frameno_start[&output_frameno];
      let next_minigop_input_frameno = self.next_minigop_input_frameno(
        self.gop_input_frameno_start[&output_frameno],
        false,
      );
      // Show frame if all previous input frames have already been shown
      let show_frame = self
        .frame_data
        .range(minigop_input_frameno_start..)
        .filter(|(_, data)| data.unwrap().fi.show_frame)
        .map(|(_, data)| data.unwrap().fi.input_frameno)
        .sorted()
        .unique()
        .filter(|frameno| *frameno < input_frameno)
        .count() as u64
        == input_frameno - minigop_input_frameno_start;
      let show_existing_frame = self
        .frame_data
        .range(minigop_input_frameno_start..)
        .any(|(_, data)| data.unwrap().fi.input_frameno == input_frameno);
      if show_existing_frame {
        assert!(show_frame);
      }
      let fi = FrameInvariants::new_inter_frame(
        self.get_previous_coded_fi(output_frameno),
        input_frameno,
        self.gop_input_frameno_start[&output_frameno],
        output_frameno_in_gop,
        minigop_input_frameno_start,
        output_frameno - self.minigop_output_frameno_start[&output_frameno],
        next_minigop_input_frameno,
        show_frame,
        show_existing_frame,
        &self.frame_depths,
        &self.config,
        t35_metadata,
        &self.minigop_config,
      );
      assert!(fi.is_some());
      Ok(fi)
    }
  }

  fn get_previous_fi(&self, output_frameno: u64) -> &FrameInvariants<T> {
    let res = self
      .frame_data
      .iter()
      .filter(|(fno, _)| **fno < output_frameno)
      .rfind(|(_, fd)| fd.is_some())
      .unwrap();
    &res.1.as_ref().unwrap().fi
  }

  fn get_previous_coded_fi(&self, output_frameno: u64) -> &FrameInvariants<T> {
    let res = self
      .frame_data
      .iter()
      .filter(|(fno, _)| **fno < output_frameno)
      .rfind(|(_, fd)| {
        fd.as_ref().map(|fd| !fd.fi.is_show_existing_frame()).unwrap_or(false)
      })
      .unwrap();
    &res.1.as_ref().unwrap().fi
  }

  pub(crate) fn done_processing(&self) -> bool {
    self.limit.map(|limit| self.frames_processed == limit).unwrap_or(false)
  }

  /// Computes lookahead motion vectors and fills in `lookahead_mvs`,
  /// `rec_buffer` and `lookahead_rec_buffer` on the `FrameInvariants`. This
  /// function must be called after every new `FrameInvariants` is initially
  /// computed.
  #[hawktracer(compute_lookahead_motion_vectors)]
  fn compute_lookahead_motion_vectors(&mut self, output_frameno: u64) {
    let frame_data = self.frame_data.get(&output_frameno).unwrap();

    // We're only interested in valid frames which are not show-existing-frame.
    // Those two don't modify the rec_buffer so there's no need to do anything
    // special about it either, it'll propagate on its own.
    if frame_data
      .as_ref()
      .map(|fd| fd.fi.is_show_existing_frame())
      .unwrap_or(true)
    {
      return;
    }

    let qps = {
      let fti = frame_data.as_ref().unwrap().fi.get_frame_subtype();
      self.rc_state.select_qi(
        self,
        output_frameno,
        fti,
        self.maybe_prev_log_base_q,
        0,
      )
    };

    let frame_data =
      self.frame_data.get_mut(&output_frameno).unwrap().as_mut().unwrap();
    let fs = &mut frame_data.fs;
    let fi = &mut frame_data.fi;
    let coded_data = fi.coded_frame_data.as_mut().unwrap();

    #[cfg(feature = "dump_lookahead_data")]
    {
      let data_location = Self::build_dump_properties();
      let plane = &fs.input_qres;
      let mut file_name = format!("{:010}-qres", fi.input_frameno);
      let buf: Vec<_> = plane.iter().map(|p| p.as_()).collect();
      image::GrayImage::from_vec(
        plane.cfg.width as u32,
        plane.cfg.height as u32,
        buf,
      )
      .unwrap()
      .save(data_location.join(file_name).with_extension("png"))
      .unwrap();
      let plane = &fs.input_hres;
      file_name = format!("{:010}-hres", fi.input_frameno);
      let buf: Vec<_> = plane.iter().map(|p| p.as_()).collect();
      image::GrayImage::from_vec(
        plane.cfg.width as u32,
        plane.cfg.height as u32,
        buf,
      )
      .unwrap()
      .save(data_location.join(file_name).with_extension("png"))
      .unwrap();
    }

    // Do not modify the next output frame's FrameInvariants.
    if self.output_frameno == output_frameno {
      // We do want to propagate the lookahead_rec_buffer though.
      let rfs = Arc::new(ReferenceFrame {
        order_hint: fi.order_hint,
        width: fi.width as u32,
        height: fi.height as u32,
        render_width: fi.render_width,
        render_height: fi.render_height,
        // Use the original frame contents.
        frame: fs.input.clone(),
        input_hres: fs.input_hres.clone(),
        input_qres: fs.input_qres.clone(),
        cdfs: fs.cdfs,
        frame_me_stats: fs.frame_me_stats.clone(),
        output_frameno,
        segmentation: fs.segmentation,
      });
      for i in 0..REF_FRAMES {
        if (fi.refresh_frame_flags & (1 << i)) != 0 {
          coded_data.lookahead_rec_buffer.frames[i] = Some(Arc::clone(&rfs));
          coded_data.lookahead_rec_buffer.deblock[i] = fs.deblock;
        }
      }

      return;
    }

    // Our lookahead_rec_buffer should be filled with correct original frame
    // data from the previous frames. Copy it into rec_buffer because that's
    // what the MV search uses. During the actual encoding rec_buffer is
    // overwritten with its correct values anyway.
    fi.rec_buffer = coded_data.lookahead_rec_buffer.clone();

    // Estimate lambda with rate-control dry-run
    fi.set_quantizers(&qps);

    // TODO: as in the encoding code, key frames will have no references.
    // However, for block importance purposes we want key frames to act as
    // P-frames in this instance.
    //
    // Compute the motion vectors.
    compute_motion_vectors(fi, fs, self.allowed_ref_frames());

    let coded_data = fi.coded_frame_data.as_mut().unwrap();

    #[cfg(feature = "dump_lookahead_data")]
    {
      use crate::partition::RefType::*;
      let data_location = Self::build_dump_properties();
      let file_name = format!("{:010}-mvs", fi.input_frameno);
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

      let me_stats = &fs.frame_me_stats.read().expect("poisoned lock")[index];
      use byteorder::{NativeEndian, WriteBytesExt};
      // dynamic allocation: debugging only
      let mut buf = vec![];
      buf.write_u64::<NativeEndian>(me_stats.rows as u64).unwrap();
      buf.write_u64::<NativeEndian>(me_stats.cols as u64).unwrap();
      for y in 0..me_stats.rows {
        for x in 0..me_stats.cols {
          let mv = me_stats[y][x].mv;
          buf.write_i16::<NativeEndian>(mv.row).unwrap();
          buf.write_i16::<NativeEndian>(mv.col).unwrap();
        }
      }
      ::std::fs::write(
        data_location.join(file_name).with_extension("bin"),
        buf,
      )
      .unwrap();
    }

    // Set lookahead_rec_buffer on this FrameInvariants for future
    // FrameInvariants to pick it up.
    let rfs = Arc::new(ReferenceFrame {
      order_hint: fi.order_hint,
      width: fi.width as u32,
      height: fi.height as u32,
      render_width: fi.render_width,
      render_height: fi.render_height,
      // Use the original frame contents.
      frame: fs.input.clone(),
      input_hres: fs.input_hres.clone(),
      input_qres: fs.input_qres.clone(),
      cdfs: fs.cdfs,
      frame_me_stats: fs.frame_me_stats.clone(),
      output_frameno,
      segmentation: fs.segmentation,
    });
    for i in 0..REF_FRAMES {
      if (fi.refresh_frame_flags & (1 << i)) != 0 {
        coded_data.lookahead_rec_buffer.frames[i] = Some(Arc::clone(&rfs));
        coded_data.lookahead_rec_buffer.deblock[i] = fs.deblock;
      }
    }
  }

  /// Computes lookahead intra cost approximations and fills in
  /// `lookahead_intra_costs` on the `FrameInvariants`.
  fn compute_lookahead_intra_costs(&mut self, output_frameno: u64) {
    let frame_data = self.frame_data.get(&output_frameno).unwrap();
    let fd = &frame_data.as_ref();

    // We're only interested in valid frames which are not show-existing-frame.
    if fd.map(|fd| fd.fi.is_show_existing_frame()).unwrap_or(true) {
      return;
    }

    let fi = &fd.unwrap().fi;

    self
      .frame_data
      .get_mut(&output_frameno)
      .unwrap()
      .as_mut()
      .unwrap()
      .fi
      .coded_frame_data
      .as_mut()
      .unwrap()
      .lookahead_intra_costs = self
      .keyframe_detector
      .intra_costs
      .remove(&fi.input_frameno)
      .unwrap_or_else(|| {
        let frame = self.frame_q[&fi.input_frameno].as_ref().unwrap();

        let temp_plane = self
          .keyframe_detector
          .temp_plane
          .get_or_insert_with(|| frame.planes[0].clone());

        // We use the cached values from scenechange if available,
        // otherwise we need to calculate them here.
        estimate_intra_costs(
          temp_plane,
          &**frame,
          fi.sequence.bit_depth,
          fi.cpu_feature_level,
        )
      });
  }

  #[hawktracer(compute_keyframe_placement)]
  pub fn compute_frame_placement(
    &mut self, lookahead_frames: &[Arc<Frame<T>>],
  ) {
    if self.keyframes_forced.contains(&self.next_lookahead_frame) {
      self.frame_depths.insert(self.next_lookahead_frame, FrameDepth::Intra);
    } else {
      let is_keyframe = self.keyframe_detector.analyze_next_frame(
        lookahead_frames,
        self.next_lookahead_frame,
        *self.frame_depths.iter().last().unwrap().0,
      );
      if is_keyframe {
        self.keyframe_detector.inter_costs.remove(&self.next_lookahead_frame);
        self.frame_depths.insert(self.next_lookahead_frame, FrameDepth::Intra);
      } else if self.frame_depths[&(self.next_lookahead_frame - 1)]
        == FrameDepth::Intra
        || self.config.low_latency
      {
        // The last frame is a keyframe, so this one must start a new mini-GOP.
        // Or, in the case of low latency, every frame is a separate mini-GOP.
        self.keyframe_detector.inter_costs.remove(&self.next_lookahead_frame);
        self.frame_depths.insert(
          self.next_lookahead_frame,
          FrameDepth::Inter { depth: 0, is_minigop_start: true },
        );
      } else {
        self.compute_current_minigop_cost();
      };
    }

    self.next_lookahead_frame += 1;
  }

  fn compute_current_minigop_cost(&mut self) {
    let minigop_start_frame = *self
      .frame_depths
      .iter()
      .rev()
      .find(|(_, d)| {
        **d == FrameDepth::Inter { depth: 0, is_minigop_start: true }
      })
      .unwrap()
      .0;

    let current_width =
      (self.next_lookahead_frame - minigop_start_frame) as u8;
    let max_pyramid_width = self.frame_type_lookahead_distance as u8;

    let mut need_new_minigop = false;
    if current_width == max_pyramid_width {
      // Since we hit the max width, we must start a new mini-GOP.
      need_new_minigop = true;
    } else {
      let current_minigop_cost = self
        .keyframe_detector
        .inter_costs
        .range(minigop_start_frame..=self.next_lookahead_frame)
        .map(|cost| {
          // Adjust the inter cost down to 8-bit scaling
          *cost.1 / (1 << (self.config.bit_depth - 8)) as f64
        })
        .sum::<f64>();
      let allowance = match current_width + 1 {
        // Depth 0
        1..=2 => 18000.0,
        // Depth 1
        3 => 20000.0,
        // Depth 2
        4 => 20000.0,
        // Depth 3
        5..=8 => 18000.0,
        // Depth 4
        9..=16 => 12000.0,
        // Depth 5
        17.. => 10000.0,
      };
      if current_minigop_cost > allowance {
        need_new_minigop = true;
      }
    }

    if need_new_minigop {
      self.compute_minigop_frame_order(
        minigop_start_frame,
        self.next_lookahead_frame - 1,
      );
      self.frame_depths.insert(
        self.next_lookahead_frame,
        FrameDepth::Inter { depth: 0, is_minigop_start: true },
      );
      for frameno in minigop_start_frame..=self.next_lookahead_frame {
        self.keyframe_detector.inter_costs.remove(&frameno);
      }
    }
  }

  // Start and end frame are inclusive
  fn compute_minigop_frame_order(&mut self, start_frame: u64, end_frame: u64) {
    // By this point, `start_frame` should already be inserted at depth 0
    if start_frame == end_frame {
      return;
    }

    let mut frames = ((start_frame + 1)..=end_frame).collect::<BTreeSet<_>>();
    let mut current_depth = 0;
    while !frames.is_empty() {
      if current_depth == 0 {
        // Special case for depth 0, we generally want the last frame at this depth
        let frameno = frames.pop_last().unwrap();
        self.frame_depths.insert(
          frameno,
          FrameDepth::Inter { depth: 0, is_minigop_start: frames.is_empty() },
        );
        current_depth += 1;
      } else {
        let max_frames_in_level = 1 << (current_depth - 1);
        if frames.len() <= max_frames_in_level {
          for frameno in frames.into_iter() {
            self.frame_depths.insert(
              frameno,
              FrameDepth::Inter {
                depth: current_depth,
                is_minigop_start: false,
              },
            );
          }
          break;
        } else {
          let mut breakpoints = vec![*frames.first().unwrap()];
          let mut prev_val = *frames.first().unwrap();
          for frameno in &frames {
            if *frameno > prev_val + 1 {
              breakpoints.push(*frameno);
            }
            prev_val = *frameno;
          }
          breakpoints.push(*frames.last().unwrap());
          for (start, end) in breakpoints.into_iter().tuple_windows() {
            let midpoint = (end - start + 1) / 2;
            frames.remove(&midpoint);
            self.frame_depths.insert(
              midpoint,
              FrameDepth::Inter {
                depth: current_depth,
                is_minigop_start: false,
              },
            );
          }
          current_depth += 1;
        }
      }
    }
  }

  #[hawktracer(compute_frame_invariants)]
  pub fn compute_frame_invariants(&mut self) {
    while self.set_frame_properties(self.next_lookahead_output_frameno).is_ok()
    {
      self
        .compute_lookahead_motion_vectors(self.next_lookahead_output_frameno);
      if self.config.temporal_rdo() {
        self.compute_lookahead_intra_costs(self.next_lookahead_output_frameno);
      }
      self.next_lookahead_output_frameno += 1;
    }
  }

  #[hawktracer(update_block_importances)]
  fn update_block_importances(
    fi: &FrameInvariants<T>, me_stats: &crate::me::FrameMEStats,
    frame: &Frame<T>, reference_frame: &Frame<T>, bit_depth: usize,
    bsize: BlockSize, len: usize,
    reference_frame_block_importances: &mut [f32],
  ) {
    let coded_data = fi.coded_frame_data.as_ref().unwrap();
    let plane_org = &frame.planes[0];
    let plane_ref = &reference_frame.planes[0];
    let lookahead_intra_costs_lines =
      coded_data.lookahead_intra_costs.chunks_exact(coded_data.w_in_imp_b);
    let block_importances_lines =
      coded_data.block_importances.chunks_exact(coded_data.w_in_imp_b);

    lookahead_intra_costs_lines
      .zip(block_importances_lines)
      .zip(me_stats.rows_iter().step_by(2))
      .enumerate()
      .flat_map(
        |(y, ((lookahead_intra_costs, block_importances), me_stats_line))| {
          lookahead_intra_costs
            .iter()
            .zip(block_importances.iter())
            .zip(me_stats_line.iter().step_by(2))
            .enumerate()
            .map(move |(x, ((&intra_cost, &future_importance), &me_stat))| {
              let mv = me_stat.mv;

              // Coordinates of the top-left corner of the reference block, in MV
              // units.
              let reference_x =
                x as i64 * IMP_BLOCK_SIZE_IN_MV_UNITS + mv.col as i64;
              let reference_y =
                y as i64 * IMP_BLOCK_SIZE_IN_MV_UNITS + mv.row as i64;

              let region_org = plane_org.region(Area::Rect {
                x: (x * IMPORTANCE_BLOCK_SIZE) as isize,
                y: (y * IMPORTANCE_BLOCK_SIZE) as isize,
                width: IMPORTANCE_BLOCK_SIZE,
                height: IMPORTANCE_BLOCK_SIZE,
              });

              let region_ref = plane_ref.region(Area::Rect {
                x: reference_x as isize
                  / IMP_BLOCK_MV_UNITS_PER_PIXEL as isize,
                y: reference_y as isize
                  / IMP_BLOCK_MV_UNITS_PER_PIXEL as isize,
                width: IMPORTANCE_BLOCK_SIZE,
                height: IMPORTANCE_BLOCK_SIZE,
              });

              let inter_cost = get_satd(
                &region_org,
                &region_ref,
                bsize.width(),
                bsize.height(),
                bit_depth,
                fi.cpu_feature_level,
              ) as f32;

              let intra_cost = intra_cost as f32;
              //          let intra_cost = lookahead_intra_costs[x] as f32;
              //          let future_importance = block_importances[x];

              let propagate_fraction = if intra_cost <= inter_cost {
                0.
              } else {
                1. - inter_cost / intra_cost
              };

              let propagate_amount = (intra_cost + future_importance)
                * propagate_fraction
                / len as f32;
              (propagate_amount, reference_x, reference_y)
            })
        },
      )
      .for_each(|(propagate_amount, reference_x, reference_y)| {
        let mut propagate =
          |block_x_in_mv_units, block_y_in_mv_units, fraction| {
            let x = block_x_in_mv_units / IMP_BLOCK_SIZE_IN_MV_UNITS;
            let y = block_y_in_mv_units / IMP_BLOCK_SIZE_IN_MV_UNITS;

            // TODO: propagate partially if the block is partially off-frame
            // (possible on right and bottom edges)?
            if x >= 0
              && y >= 0
              && (x as usize) < coded_data.w_in_imp_b
              && (y as usize) < coded_data.h_in_imp_b
            {
              reference_frame_block_importances
                [y as usize * coded_data.w_in_imp_b + x as usize] +=
                propagate_amount * fraction;
            }
          };

        // Coordinates of the top-left corner of the block intersecting the
        // reference block from the top-left.
        let top_left_block_x = (reference_x
          - if reference_x < 0 { IMP_BLOCK_SIZE_IN_MV_UNITS - 1 } else { 0 })
          / IMP_BLOCK_SIZE_IN_MV_UNITS
          * IMP_BLOCK_SIZE_IN_MV_UNITS;
        let top_left_block_y = (reference_y
          - if reference_y < 0 { IMP_BLOCK_SIZE_IN_MV_UNITS - 1 } else { 0 })
          / IMP_BLOCK_SIZE_IN_MV_UNITS
          * IMP_BLOCK_SIZE_IN_MV_UNITS;

        debug_assert!(reference_x >= top_left_block_x);
        debug_assert!(reference_y >= top_left_block_y);

        let top_right_block_x = top_left_block_x + IMP_BLOCK_SIZE_IN_MV_UNITS;
        let top_right_block_y = top_left_block_y;
        let bottom_left_block_x = top_left_block_x;
        let bottom_left_block_y =
          top_left_block_y + IMP_BLOCK_SIZE_IN_MV_UNITS;
        let bottom_right_block_x = top_right_block_x;
        let bottom_right_block_y = bottom_left_block_y;

        let top_left_block_fraction = ((top_right_block_x - reference_x)
          * (bottom_left_block_y - reference_y))
          as f32
          / IMP_BLOCK_AREA_IN_MV_UNITS as f32;

        propagate(top_left_block_x, top_left_block_y, top_left_block_fraction);

        let top_right_block_fraction =
          ((reference_x + IMP_BLOCK_SIZE_IN_MV_UNITS - top_right_block_x)
            * (bottom_left_block_y - reference_y)) as f32
            / IMP_BLOCK_AREA_IN_MV_UNITS as f32;

        propagate(
          top_right_block_x,
          top_right_block_y,
          top_right_block_fraction,
        );

        let bottom_left_block_fraction = ((top_right_block_x - reference_x)
          * (reference_y + IMP_BLOCK_SIZE_IN_MV_UNITS - bottom_left_block_y))
          as f32
          / IMP_BLOCK_AREA_IN_MV_UNITS as f32;

        propagate(
          bottom_left_block_x,
          bottom_left_block_y,
          bottom_left_block_fraction,
        );

        let bottom_right_block_fraction =
          ((reference_x + IMP_BLOCK_SIZE_IN_MV_UNITS - top_right_block_x)
            * (reference_y + IMP_BLOCK_SIZE_IN_MV_UNITS - bottom_left_block_y))
            as f32
            / IMP_BLOCK_AREA_IN_MV_UNITS as f32;

        propagate(
          bottom_right_block_x,
          bottom_right_block_y,
          bottom_right_block_fraction,
        );
      });
  }

  /// Computes the block importances for the current output frame.
  #[hawktracer(compute_block_importances)]
  fn compute_block_importances(&mut self) {
    // SEF don't need block importances.
    if self.frame_data[&self.output_frameno]
      .as_ref()
      .unwrap()
      .fi
      .is_show_existing_frame()
    {
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
      let fi = &mut self
        .frame_data
        .get_mut(output_frameno)
        .unwrap()
        .as_mut()
        .unwrap()
        .fi;
      for x in
        fi.coded_frame_data.as_mut().unwrap().block_importances.iter_mut()
      {
        *x = 0.;
      }
    }

    // Now compute and propagate the block importances from the end. The
    // current output frame will get its block importances from the future
    // frames.
    let bsize = BlockSize::from_width_and_height(
      IMPORTANCE_BLOCK_SIZE,
      IMPORTANCE_BLOCK_SIZE,
    );

    for &output_frameno in output_framenos.iter().skip(1).rev() {
      // TODO: see comment above about key frames not having references.
      if self
        .frame_data
        .get(&output_frameno)
        .unwrap()
        .as_ref()
        .unwrap()
        .fi
        .frame_type
        == FrameType::KEY
      {
        continue;
      }

      // Remove fi from the map temporarily and put it back in in the end of
      // the iteration. This is required because we need to mutably borrow
      // referenced fis from the map, and that wouldn't be possible if this was
      // an active borrow.
      //
      // Performance note: Contrary to intuition,
      // removing the data and re-inserting it at the end
      // is more performant because it avoids a very expensive clone.
      let output_frame_data =
        self.frame_data.remove(&output_frameno).unwrap().unwrap();
      {
        let fi = &output_frame_data.fi;
        let fs = &output_frame_data.fs;

        let frame = self.frame_q[&fi.input_frameno].as_ref().unwrap();

        // There can be at most 3 of these.
        let mut unique_indices = ArrayVec::<_, 3>::new();

        for (mv_index, &rec_index) in fi.ref_frames.iter().enumerate() {
          if !unique_indices.iter().any(|&(_, r)| r == rec_index) {
            unique_indices.push((mv_index, rec_index));
          }
        }

        let bit_depth = self.config.bit_depth;
        let frame_data = &mut self.frame_data;
        let len = unique_indices.len();

        let lookahead_me_stats =
          fs.frame_me_stats.read().expect("poisoned lock");

        // Compute and propagate the importance, split evenly between the
        // referenced frames.
        unique_indices.iter().for_each(|&(mv_index, rec_index)| {
          // Use rec_buffer here rather than lookahead_rec_buffer because
          // rec_buffer still contains the reference frames for the current frame
          // (it's only overwritten when the frame is encoded), while
          // lookahead_rec_buffer already contains reference frames for the next
          // frame (for the reference propagation to work correctly).
          let reference =
            fi.rec_buffer.frames[rec_index as usize].as_ref().unwrap();
          let reference_frame = &reference.frame;
          let reference_output_frameno = reference.output_frameno;
          let me_stats = &lookahead_me_stats[mv_index];

          // We should never use frame as its own reference.
          assert_ne!(reference_output_frameno, output_frameno);

          if let Some(reference_frame_block_importances) =
            frame_data.get_mut(&reference_output_frameno).map(|data| {
              &mut data
                .as_mut()
                .unwrap()
                .fi
                .coded_frame_data
                .as_mut()
                .unwrap()
                .block_importances
            })
          {
            Self::update_block_importances(
              fi,
              me_stats,
              frame,
              reference_frame,
              bit_depth,
              bsize,
              len,
              reference_frame_block_importances,
            );
          }
        });
      }
      self.frame_data.insert(output_frameno, Some(output_frame_data));
    }

    if !output_framenos.is_empty() {
      let fi = &mut self
        .frame_data
        .get_mut(&output_framenos[0])
        .unwrap()
        .as_mut()
        .unwrap()
        .fi;
      let coded_data = fi.coded_frame_data.as_mut().unwrap();
      let block_importances = coded_data.block_importances.iter();
      let lookahead_intra_costs = coded_data.lookahead_intra_costs.iter();
      let distortion_scales = coded_data.distortion_scales.iter_mut();
      for ((&propagate_cost, &intra_cost), distortion_scale) in
        block_importances.zip(lookahead_intra_costs).zip(distortion_scales)
      {
        *distortion_scale = crate::rdo::distortion_scale_for(
          propagate_cost as f64,
          intra_cost as f64,
        );
      }
      #[cfg(feature = "dump_lookahead_data")]
      {
        use byteorder::{NativeEndian, WriteBytesExt};

        let coded_data = fi.coded_frame_data.as_ref().unwrap();

        let mut buf = vec![];
        let data_location = Self::build_dump_properties();
        let file_name = format!("{:010}-imps", fi.input_frameno);
        buf.write_u64::<NativeEndian>(coded_data.h_in_imp_b as u64).unwrap();
        buf.write_u64::<NativeEndian>(coded_data.w_in_imp_b as u64).unwrap();
        buf.write_u64::<NativeEndian>(fi.get_frame_subtype() as u64).unwrap();
        for y in 0..coded_data.h_in_imp_b {
          for x in 0..coded_data.w_in_imp_b {
            buf
              .write_f32::<NativeEndian>(f64::from(
                coded_data.distortion_scales[y * coded_data.w_in_imp_b + x],
              ) as f32)
              .unwrap();
          }
        }
        ::std::fs::write(
          data_location.join(file_name).with_extension("bin"),
          buf,
        )
        .unwrap();
      }
    }
  }

  pub(crate) fn encode_packet(
    &mut self, cur_output_frameno: u64,
  ) -> Result<Packet<T>, EncoderStatus> {
    if self
      .frame_data
      .get(&cur_output_frameno)
      .unwrap()
      .as_ref()
      .unwrap()
      .fi
      .is_show_existing_frame()
    {
      if !self.rc_state.ready() {
        return Err(EncoderStatus::NotReady);
      }

      self.encode_show_existing_packet(cur_output_frameno)
    } else if let Some(Some(_)) = self.frame_q.get(
      &self
        .frame_data
        .get(&cur_output_frameno)
        .unwrap()
        .as_ref()
        .unwrap()
        .fi
        .input_frameno,
    ) {
      if !self.rc_state.ready() {
        return Err(EncoderStatus::NotReady);
      }

      self.encode_normal_packet(cur_output_frameno)
    } else {
      Err(EncoderStatus::NeedMoreData)
    }
  }

  #[hawktracer(encode_show_existing_packet)]
  pub fn encode_show_existing_packet(
    &mut self, cur_output_frameno: u64,
  ) -> Result<Packet<T>, EncoderStatus> {
    let frame_data =
      self.frame_data.get_mut(&cur_output_frameno).unwrap().as_mut().unwrap();
    let sef_data =
      encode_show_existing_frame(&frame_data.fi, &mut frame_data.fs);
    let bits = (sef_data.len() * 8) as i64;
    self.packet_data.extend(sef_data);
    self.rc_state.update_state(
      bits,
      FRAME_SUBTYPE_SEF,
      frame_data.fi.show_frame,
      0,
      false,
      false,
    );
    let (rec, source) = if frame_data.fi.show_frame {
      (Some(frame_data.fs.rec.clone()), Some(frame_data.fs.input.clone()))
    } else {
      (None, None)
    };

    self.output_frameno += 1;

    let input_frameno = frame_data.fi.input_frameno;
    let frame_type = frame_data.fi.frame_type;
    let qp = frame_data.fi.base_q_idx;
    let enc_stats = frame_data.fs.enc_stats.clone();
    self.finalize_packet(rec, source, input_frameno, frame_type, qp, enc_stats)
  }

  #[hawktracer(encode_normal_packet)]
  pub fn encode_normal_packet(
    &mut self, cur_output_frameno: u64,
  ) -> Result<Packet<T>, EncoderStatus> {
    let mut frame_data =
      self.frame_data.remove(&cur_output_frameno).unwrap().unwrap();

    let mut log_isqrt_mean_scale = 0i64;

    if let Some(coded_data) = frame_data.fi.coded_frame_data.as_mut() {
      if self.config.tune == Tune::Psychovisual {
        let frame =
          self.frame_q[&frame_data.fi.input_frameno].as_ref().unwrap();
        coded_data.activity_mask = ActivityMask::from_plane(&frame.planes[0]);
        coded_data.activity_mask.fill_scales(
          frame_data.fi.sequence.bit_depth,
          &mut coded_data.activity_scales,
        );
        log_isqrt_mean_scale = coded_data.compute_spatiotemporal_scores();
      } else {
        coded_data.activity_mask = ActivityMask::default();
        log_isqrt_mean_scale = coded_data.compute_temporal_scores();
      }
      #[cfg(feature = "dump_lookahead_data")]
      {
        use crate::encoder::Scales::*;
        let input_frameno = frame_data.fi.input_frameno;
        if self.config.tune == Tune::Psychovisual {
          coded_data.dump_scales(
            Self::build_dump_properties(),
            ActivityScales,
            input_frameno,
          );
          coded_data.dump_scales(
            Self::build_dump_properties(),
            SpatiotemporalScales,
            input_frameno,
          );
        }
        coded_data.dump_scales(
          Self::build_dump_properties(),
          DistortionScales,
          input_frameno,
        );
      }
    }

    let fti = frame_data.fi.get_frame_subtype();
    let qps = self.rc_state.select_qi(
      self,
      cur_output_frameno,
      fti,
      self.maybe_prev_log_base_q,
      log_isqrt_mean_scale,
    );
    frame_data.fi.set_quantizers(&qps);

    if self.rc_state.needs_trial_encode(fti) {
      let mut trial_fs = frame_data.fs.clone();
      let data = encode_frame(&frame_data.fi, &mut trial_fs);
      self.rc_state.update_state(
        (data.len() * 8) as i64,
        fti,
        frame_data.fi.show_frame,
        qps.log_target_q,
        true,
        false,
      );
      let qps = self.rc_state.select_qi(
        self,
        cur_output_frameno,
        fti,
        self.maybe_prev_log_base_q,
        log_isqrt_mean_scale,
      );
      frame_data.fi.set_quantizers(&qps);
    }

    let data = encode_frame(&frame_data.fi, &mut frame_data.fs);
    #[cfg(feature = "dump_lookahead_data")]
    {
      let input_frameno = frame_data.fi.input_frameno;
      let data_location = Self::build_dump_properties();
      frame_data.fs.segmentation.dump_threshold(data_location, input_frameno);
    }
    let enc_stats = frame_data.fs.enc_stats.clone();
    self.maybe_prev_log_base_q = Some(qps.log_base_q);
    // TODO: Add support for dropping frames.
    self.rc_state.update_state(
      (data.len() * 8) as i64,
      fti,
      frame_data.fi.show_frame,
      qps.log_target_q,
      false,
      false,
    );
    self.packet_data.extend(data);

    let planes =
      if frame_data.fi.sequence.chroma_sampling == Cs400 { 1 } else { 3 };

    Arc::get_mut(&mut frame_data.fs.rec).unwrap().pad(
      frame_data.fi.width,
      frame_data.fi.height,
      planes,
    );

    let (rec, source) = if frame_data.fi.show_frame {
      (Some(frame_data.fs.rec.clone()), Some(frame_data.fs.input.clone()))
    } else {
      (None, None)
    };

    update_rec_buffer(cur_output_frameno, &mut frame_data.fi, &frame_data.fs);

    // Copy persistent fields into subsequent FrameInvariants.
    let rec_buffer = frame_data.fi.rec_buffer.clone();
    for subsequent_fi in self
      .frame_data
      .iter_mut()
      .skip_while(|(&output_frameno, _)| output_frameno <= cur_output_frameno)
      // Here we want the next valid non-show-existing-frame inter frame.
      //
      // Copying to show-existing-frame frames isn't actually required
      // for correct encoding, but it's needed for the reconstruction to
      // work correctly.
      .filter_map(|(_, frame_data)| frame_data.as_mut().map(|fd| &mut fd.fi))
      .take_while(|fi| fi.frame_type != FrameType::KEY)
    {
      subsequent_fi.rec_buffer = rec_buffer.clone();
      subsequent_fi.set_ref_frame_sign_bias();

      // Stop after the first non-show-existing-frame.
      if !subsequent_fi.is_show_existing_frame() {
        break;
      }
    }

    self.frame_data.insert(cur_output_frameno, Some(frame_data));
    let frame_data =
      self.frame_data.get(&cur_output_frameno).unwrap().as_ref().unwrap();
    let fi = &frame_data.fi;

    self.output_frameno += 1;

    if fi.show_frame {
      let input_frameno = fi.input_frameno;
      let frame_type = fi.frame_type;
      let qp = fi.base_q_idx;
      self.finalize_packet(
        rec,
        source,
        input_frameno,
        frame_type,
        qp,
        enc_stats,
      )
    } else {
      Err(EncoderStatus::Encoded)
    }
  }

  #[hawktracer(receive_packet)]
  pub fn receive_packet(&mut self) -> Result<Packet<T>, EncoderStatus> {
    if self.done_processing() {
      return Err(EncoderStatus::LimitReached);
    }

    if self.needs_more_fi_lookahead() {
      return Err(EncoderStatus::NeedMoreData);
    }

    // Find the next output_frameno corresponding to a non-skipped frame.
    self.output_frameno = self
      .frame_data
      .iter()
      .skip_while(|(&output_frameno, _)| output_frameno < self.output_frameno)
      .find(|(_, data)| data.is_some())
      .map(|(&output_frameno, _)| output_frameno)
      .ok_or(EncoderStatus::NeedMoreData)?; // TODO: doesn't play well with the below check?

    let input_frameno =
      self.frame_data[&self.output_frameno].as_ref().unwrap().fi.input_frameno;
    if !self.needs_more_frames(input_frameno) {
      return Err(EncoderStatus::LimitReached);
    }

    if self.config.temporal_rdo() {
      // Compute the block importances for the current output frame.
      self.compute_block_importances();
    }

    let cur_output_frameno = self.output_frameno;

    let mut ret = self.encode_packet(cur_output_frameno);

    if let Ok(ref mut pkt) = ret {
      self.garbage_collect(pkt.input_frameno);
      pkt.opaque = self.opaque_q.remove(&pkt.input_frameno);
    }

    ret
  }

  fn finalize_packet(
    &mut self, rec: Option<Arc<Frame<T>>>, source: Option<Arc<Frame<T>>>,
    input_frameno: u64, frame_type: FrameType, qp: u8,
    enc_stats: EncoderStats,
  ) -> Result<Packet<T>, EncoderStatus> {
    let data = self.packet_data.clone();
    self.packet_data.clear();
    if write_temporal_delimiter(&mut self.packet_data).is_err() {
      return Err(EncoderStatus::Failure);
    }

    self.frames_processed += 1;
    Ok(Packet {
      data,
      rec,
      source,
      input_frameno,
      frame_type,
      qp,
      enc_stats,
      opaque: None,
    })
  }

  #[hawktracer(garbage_collect)]
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
    let fi_start = self.frame_data.keys().next().cloned().unwrap_or(0);
    for i in fi_start..(self.output_frameno - 1) {
      self.frame_data.remove(&i);
      self.gop_output_frameno_start.remove(&i);
      self.gop_input_frameno_start.remove(&i);
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrameDepth {
  Intra,
  Inter { depth: u8, is_minigop_start: bool },
}

impl FrameDepth {
  pub fn depth(self) -> u8 {
    match self {
      FrameDepth::Intra => 0,
      FrameDepth::Inter { depth, .. } => depth,
    }
  }

  pub fn is_minigop_start(self) -> bool {
    match self {
      FrameDepth::Intra => true,
      FrameDepth::Inter { is_minigop_start, .. } => is_minigop_start,
    }
  }
}
