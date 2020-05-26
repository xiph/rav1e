// Copyright (c) 2018-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::activity::*;
use crate::api::*;
use crate::cdef::*;
use crate::context::*;
use crate::deblock::*;
use crate::ec::*;
use crate::frame::*;
use crate::header::*;
use crate::lrf::*;
use crate::mc::FilterMode;
use crate::mc::MotionVector;
use crate::me::*;
use crate::partition::PartitionType::*;
use crate::partition::RefType::*;
use crate::partition::*;
use crate::predict::{
  AngleDelta, IntraEdgeFilterParameters, IntraParam, PredictionMode,
};
use crate::quantize::*;
use crate::rate::bexp64;
use crate::rate::q57;
use crate::rate::QuantizerParameters;
use crate::rate::FRAME_SUBTYPE_I;
use crate::rate::FRAME_SUBTYPE_P;
use crate::rate::QSCALE;
use crate::rdo::*;
use crate::segmentation::*;
use crate::serialize::{Deserialize, Serialize};
use crate::stats::EncoderStats;
use crate::tiling::*;
use crate::transform::*;
use crate::util::*;

use arg_enum_proc_macro::ArgEnum;
use arrayvec::*;
use bitstream_io::{BigEndian, BitWriter};

use std::collections::VecDeque;
use std::io::Write;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::{fmt, io, mem};

use crate::hawktracer::*;
use crate::rayon::iter::*;

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub enum CDEFSearchMethod {
  PickFromQ,
  FastSearch,
  FullSearch,
}

#[inline(always)]
fn poly2(q: f32, a: f32, b: f32, c: f32, max: i32) -> i32 {
  clamp((q * q * a + q * b + c).round() as i32, 0, max)
}

pub static TEMPORAL_DELIMITER: [u8; 2] = [0x12, 0x00];

const MAX_NUM_TEMPORAL_LAYERS: usize = 8;
const MAX_NUM_SPATIAL_LAYERS: usize = 4;
const MAX_NUM_OPERATING_POINTS: usize =
  MAX_NUM_TEMPORAL_LAYERS * MAX_NUM_SPATIAL_LAYERS;

/// Size of blocks for the importance computation, in pixels.
pub const IMPORTANCE_BLOCK_SIZE: usize =
  1 << (IMPORTANCE_BLOCK_TO_BLOCK_SHIFT + BLOCK_TO_PLANE_SHIFT);

#[derive(Debug, Clone)]
pub struct ReferenceFrame<T: Pixel> {
  pub order_hint: u32,
  pub frame: Arc<Frame<T>>,
  pub input_hres: Arc<Plane<T>>,
  pub input_qres: Arc<Plane<T>>,
  pub cdfs: CDFContext,
  pub frame_mvs: Arc<Vec<FrameMotionVectors>>,
  pub output_frameno: u64,
  pub segmentation: SegmentationState,
}

#[derive(Debug, Clone, Default)]
pub struct ReferenceFramesSet<T: Pixel> {
  pub frames: [Option<Arc<ReferenceFrame<T>>>; (REF_FRAMES as usize)],
  pub deblock: [DeblockState; (REF_FRAMES as usize)],
}

impl<T: Pixel> ReferenceFramesSet<T> {
  pub fn new() -> Self {
    Self { frames: Default::default(), deblock: Default::default() }
  }
}

#[derive(ArgEnum, Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum Tune {
  Psnr,
  Psychovisual,
}

impl Default for Tune {
  fn default() -> Self {
    Tune::Psychovisual
  }
}

const FRAME_ID_LENGTH: u32 = 15;
const DELTA_FRAME_ID_LENGTH: u32 = 14;

#[derive(Copy, Clone, Debug)]
pub struct Sequence {
  // OBU Sequence header of AV1
  pub profile: u8,
  pub num_bits_width: u32,
  pub num_bits_height: u32,
  pub bit_depth: usize,
  pub chroma_sampling: ChromaSampling,
  pub chroma_sample_position: ChromaSamplePosition,
  pub pixel_range: PixelRange,
  pub color_description: Option<ColorDescription>,
  pub mastering_display: Option<MasteringDisplay>,
  pub content_light: Option<ContentLight>,
  pub max_frame_width: u32,
  pub max_frame_height: u32,
  pub frame_id_numbers_present_flag: bool,
  pub frame_id_length: u32,
  pub delta_frame_id_length: u32,
  pub use_128x128_superblock: bool,
  pub order_hint_bits_minus_1: u32,
  pub force_screen_content_tools: u32, // 0 - force off
  // 1 - force on
  // 2 - adaptive
  pub force_integer_mv: u32, // 0 - Not to force. MV can be in 1/4 or 1/8
  // 1 - force to integer
  // 2 - adaptive
  pub still_picture: bool, // Video is a single frame still picture
  pub reduced_still_picture_hdr: bool, // Use reduced header for still picture
  pub enable_filter_intra: bool, // enables/disables filter_intra
  pub enable_intra_edge_filter: bool, // enables/disables corner/edge filtering and upsampling
  pub enable_interintra_compound: bool, // enables/disables interintra_compound
  pub enable_masked_compound: bool,   // enables/disables masked compound
  pub enable_dual_filter: bool,       // 0 - disable dual interpolation filter
  // 1 - enable vert/horiz filter selection
  pub enable_order_hint: bool, // 0 - disable order hint, and related tools
  // jnt_comp, ref_frame_mvs, frame_sign_bias
  // if 0, enable_jnt_comp and
  // enable_ref_frame_mvs must be set zs 0.
  pub enable_jnt_comp: bool, // 0 - disable joint compound modes
  // 1 - enable it
  pub enable_ref_frame_mvs: bool, // 0 - disable ref frame mvs
  // 1 - enable it
  pub enable_warped_motion: bool, // 0 - disable warped motion for sequence
  // 1 - enable it for the sequence
  pub enable_superres: bool, // 0 - Disable superres for the sequence, and disable
  //     transmitting per-frame superres enabled flag.
  // 1 - Enable superres for the sequence, and also
  //     enable per-frame flag to denote if superres is
  //     enabled for that frame.
  pub enable_cdef: bool,        // To turn on/off CDEF
  pub enable_restoration: bool, // To turn on/off loop restoration
  pub enable_large_lru: bool, // To turn on/off larger-than-superblock loop restoration units
  pub enable_delayed_loopfilter_rdo: bool, // allow encoder to delay loop filter RDO/coding until after frame reconstruciton is complete
  pub operating_points_cnt_minus_1: usize,
  pub operating_point_idc: [u16; MAX_NUM_OPERATING_POINTS],
  pub display_model_info_present_flag: bool,
  pub decoder_model_info_present_flag: bool,
  pub level: [[usize; 2]; MAX_NUM_OPERATING_POINTS], // minor, major
  pub tier: [usize; MAX_NUM_OPERATING_POINTS], // seq_tier in the spec. One bit: 0
  // or 1.
  pub film_grain_params_present: bool,
  pub separate_uv_delta_q: bool,
  pub timing_info_present: bool,
}

impl Sequence {
  pub fn new(config: &EncoderConfig) -> Sequence {
    let width_bits = 32 - (config.width as u32).leading_zeros();
    let height_bits = 32 - (config.height as u32).leading_zeros();
    assert!(width_bits <= 16);
    assert!(height_bits <= 16);

    let profile = if config.bit_depth == 12
      || config.chroma_sampling == ChromaSampling::Cs422
    {
      2
    } else if config.chroma_sampling == ChromaSampling::Cs444 {
      1
    } else {
      0
    };

    let mut operating_point_idc = [0 as u16; MAX_NUM_OPERATING_POINTS];
    let mut level = [[1, 2 as usize]; MAX_NUM_OPERATING_POINTS];
    let mut tier = [0 as usize; MAX_NUM_OPERATING_POINTS];

    for i in 0..MAX_NUM_OPERATING_POINTS {
      operating_point_idc[i] = 0;
      level[i][0] = 1; // minor
      level[i][1] = 2; // major
      tier[i] = 0;
    }

    // Restoration filters are not useful for very small frame sizes,
    // so disable them in that case.
    let enable_restoration_filters = config.width >= 32 && config.height >= 32;

    Sequence {
      profile,
      num_bits_width: width_bits,
      num_bits_height: height_bits,
      bit_depth: config.bit_depth,
      chroma_sampling: config.chroma_sampling,
      chroma_sample_position: config.chroma_sample_position,
      pixel_range: config.pixel_range,
      color_description: config.color_description,
      mastering_display: config.mastering_display,
      content_light: config.content_light,
      max_frame_width: config.width as u32,
      max_frame_height: config.height as u32,
      frame_id_numbers_present_flag: false,
      frame_id_length: FRAME_ID_LENGTH,
      delta_frame_id_length: DELTA_FRAME_ID_LENGTH,
      use_128x128_superblock: false,
      order_hint_bits_minus_1: 5,
      force_screen_content_tools: if config.still_picture { 2 } else { 0 },
      force_integer_mv: 2,
      still_picture: config.still_picture,
      reduced_still_picture_hdr: config.still_picture,
      enable_filter_intra: false,
      enable_intra_edge_filter: true,
      enable_interintra_compound: false,
      enable_masked_compound: false,
      enable_dual_filter: false,
      enable_order_hint: !config.still_picture,
      enable_jnt_comp: false,
      enable_ref_frame_mvs: false,
      enable_warped_motion: false,
      enable_superres: false,
      enable_cdef: config.speed_settings.cdef
        && config.chroma_sampling != ChromaSampling::Cs422
        && enable_restoration_filters,
      enable_restoration: config.speed_settings.lrf
        && config.chroma_sampling != ChromaSampling::Cs422
        && enable_restoration_filters,
      enable_large_lru: true,
      enable_delayed_loopfilter_rdo: true,
      operating_points_cnt_minus_1: 0,
      operating_point_idc,
      display_model_info_present_flag: false,
      decoder_model_info_present_flag: false,
      level,
      tier,
      film_grain_params_present: false,
      separate_uv_delta_q: true,
      timing_info_present: config.enable_timing_info,
    }
  }

  pub const fn get_relative_dist(&self, a: u32, b: u32) -> i32 {
    let diff = a as i32 - b as i32;
    let m = 1 << self.order_hint_bits_minus_1;
    (diff & (m - 1)) - (diff & m)
  }

  pub fn get_skip_mode_allowed<T: Pixel>(
    &self, fi: &FrameInvariants<T>, inter_cfg: &InterConfig,
    reference_select: bool,
  ) -> bool {
    if fi.intra_only || !reference_select || !self.enable_order_hint {
      return false;
    }

    let mut forward_idx: isize = -1;
    let mut backward_idx: isize = -1;
    let mut forward_hint = 0;
    let mut backward_hint = 0;

    for i in inter_cfg.allowed_ref_frames().iter().map(|rf| rf.to_index()) {
      if let Some(ref rec) = fi.rec_buffer.frames[fi.ref_frames[i] as usize] {
        let ref_hint = rec.order_hint;

        if self.get_relative_dist(ref_hint, fi.order_hint) < 0 {
          if forward_idx < 0
            || self.get_relative_dist(ref_hint, forward_hint) > 0
          {
            forward_idx = i as isize;
            forward_hint = ref_hint;
          }
        } else if self.get_relative_dist(ref_hint, fi.order_hint) > 0
          && (backward_idx < 0
            || self.get_relative_dist(ref_hint, backward_hint) > 0)
        {
          backward_idx = i as isize;
          backward_hint = ref_hint;
        }
      }
    }

    if forward_idx < 0 {
      false
    } else if backward_idx >= 0 {
      // set skip_mode_frame
      true
    } else {
      let mut second_forward_idx: isize = -1;
      let mut second_forward_hint = 0;

      for i in inter_cfg.allowed_ref_frames().iter().map(|rf| rf.to_index()) {
        if let Some(ref rec) = fi.rec_buffer.frames[fi.ref_frames[i] as usize]
        {
          let ref_hint = rec.order_hint;

          if self.get_relative_dist(ref_hint, forward_hint) < 0
            && (second_forward_idx < 0
              || self.get_relative_dist(ref_hint, second_forward_hint) > 0)
          {
            second_forward_idx = i as isize;
            second_forward_hint = ref_hint;
          }
        }
      }

      // TODO: Set skip_mode_frame, when second_forward_idx is not less than 0.
      second_forward_idx >= 0
    }
  }

  #[inline(always)]
  pub const fn sb_size_log2(&self) -> usize {
    6 + (self.use_128x128_superblock as usize)
  }
}

impl Default for Sequence {
  fn default() -> Self {
    Sequence::new(&EncoderConfig::default())
  }
}

pub type BlockPmv = [[Option<MotionVector>; REF_FRAMES]; 5];

#[derive(Debug, Clone)]
pub struct FrameState<T: Pixel> {
  pub sb_size_log2: usize,
  pub input: Arc<Frame<T>>,
  pub input_hres: Arc<Plane<T>>, // half-resolution version of input luma
  pub input_qres: Arc<Plane<T>>, // quarter-resolution version of input luma
  pub rec: Arc<Frame<T>>,
  pub cdfs: CDFContext,
  pub context_update_tile_id: usize, // tile id used for the CDFontext
  pub max_tile_size_bytes: u32,
  pub deblock: DeblockState,
  pub segmentation: SegmentationState,
  pub restoration: RestorationState,
  // Because we only reference these within a tile context,
  // these are stored per-tile for easier access.
  pub half_res_pmvs: Vec<(PlaneSuperBlockOffset, Vec<BlockPmv>)>,
  pub frame_mvs: Arc<Vec<FrameMotionVectors>>,
  pub enc_stats: EncoderStats,
}

impl<T: Pixel> FrameState<T> {
  pub fn new(fi: &FrameInvariants<T>) -> Self {
    // TODO(negge): Use fi.cfg.chroma_sampling when we store VideoDetails in FrameInvariants
    FrameState::new_with_frame(
      fi,
      Arc::new(Frame::new(fi.width, fi.height, fi.sequence.chroma_sampling)),
    )
  }

  pub fn new_with_frame(
    fi: &FrameInvariants<T>, frame: Arc<Frame<T>>,
  ) -> Self {
    let rs = RestorationState::new(fi, &frame);
    let luma_width = frame.planes[0].cfg.width;
    let luma_height = frame.planes[0].cfg.height;

    let hres = frame.planes[0].downsampled(fi.width, fi.height);
    let qres = hres.downsampled(fi.width, fi.height);

    Self {
      sb_size_log2: fi.sb_size_log2(),
      input: frame,
      input_hres: Arc::new(hres),
      input_qres: Arc::new(qres),
      rec: Arc::new(Frame::new(
        luma_width,
        luma_height,
        fi.sequence.chroma_sampling,
      )),
      cdfs: CDFContext::new(0),
      context_update_tile_id: 0,
      max_tile_size_bytes: 0,
      deblock: Default::default(),
      segmentation: Default::default(),
      restoration: rs,
      half_res_pmvs: Vec::with_capacity(fi.tiling.cols * fi.tiling.rows),
      frame_mvs: {
        let mut vec = Vec::with_capacity(REF_FRAMES);
        for _ in 0..REF_FRAMES {
          vec.push(FrameMotionVectors::new(fi.w_in_b, fi.h_in_b));
        }
        Arc::new(vec)
      },
      enc_stats: Default::default(),
    }
  }

  #[inline(always)]
  pub fn as_tile_state_mut(&mut self) -> TileStateMut<'_, T> {
    let PlaneConfig { width, height, .. } = self.rec.planes[0].cfg;
    let sbo_0 = PlaneSuperBlockOffset(SuperBlockOffset { x: 0, y: 0 });
    TileStateMut::new(self, sbo_0, self.sb_size_log2, width, height)
  }
}

#[derive(Copy, Clone, Debug)]
pub struct DeblockState {
  pub levels: [u8; PLANES + 1], // Y vertical edges, Y horizontal, U, V
  pub sharpness: u8,
  pub deltas_enabled: bool,
  pub delta_updates_enabled: bool,
  pub ref_deltas: [i8; REF_FRAMES],
  pub mode_deltas: [i8; 2],
  pub block_deltas_enabled: bool,
  pub block_delta_shift: u8,
  pub block_delta_multi: bool,
}

impl Default for DeblockState {
  fn default() -> Self {
    DeblockState {
      levels: [8, 8, 4, 4],
      sharpness: 0,
      deltas_enabled: false, // requires delta_q_enabled
      delta_updates_enabled: false,
      ref_deltas: [1, 0, 0, 0, 0, -1, -1, -1],
      mode_deltas: [0, 0],
      block_deltas_enabled: false,
      block_delta_shift: 0,
      block_delta_multi: false,
    }
  }
}

#[derive(Copy, Clone, Debug)]
pub struct SegmentationState {
  pub enabled: bool,
  pub update_data: bool,
  pub update_map: bool,
  pub preskip: bool,
  pub last_active_segid: u8,
  pub features: [[bool; SegLvl::SEG_LVL_MAX as usize]; 8],
  pub data: [[i16; SegLvl::SEG_LVL_MAX as usize]; 8],
}

impl Default for SegmentationState {
  fn default() -> Self {
    SegmentationState {
      enabled: false,
      update_data: false,
      update_map: false,
      preskip: false,
      last_active_segid: 0,
      features: [[false; SegLvl::SEG_LVL_MAX as usize]; 8],
      data: [[0; SegLvl::SEG_LVL_MAX as usize]; 8],
    }
  }
}

// Frame Invariants are invariant inside a frame
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct FrameInvariants<T: Pixel> {
  pub sequence: Sequence,
  pub width: usize,
  pub height: usize,
  pub sb_width: usize,
  pub sb_height: usize,
  pub w_in_b: usize,
  pub h_in_b: usize,
  pub tiling: TilingInfo,
  pub input_frameno: u64,
  pub order_hint: u32,
  pub show_frame: bool,
  pub showable_frame: bool,
  pub error_resilient: bool,
  pub intra_only: bool,
  pub allow_high_precision_mv: bool,
  pub frame_type: FrameType,
  pub show_existing_frame: bool,
  pub frame_to_show_map_idx: u32,
  pub use_reduced_tx_set: bool,
  pub reference_mode: ReferenceMode,
  pub use_prev_frame_mvs: bool,
  pub partition_range: PartitionRange,
  pub globalmv_transformation_type: [GlobalMVMode; INTER_REFS_PER_FRAME],
  pub num_tg: usize,
  pub large_scale_tile: bool,
  pub disable_cdf_update: bool,
  pub allow_screen_content_tools: u32,
  pub force_integer_mv: u32,
  pub primary_ref_frame: u32,
  pub refresh_frame_flags: u32, // a bitmask that specifies which
  // reference frame slots will be updated with the current frame
  // after it is decoded.
  pub allow_intrabc: bool,
  pub use_ref_frame_mvs: bool,
  pub is_filter_switchable: bool,
  pub is_motion_mode_switchable: bool,
  pub disable_frame_end_update_cdf: bool,
  pub allow_warped_motion: bool,
  pub cdef_search_method: CDEFSearchMethod,
  pub cdef_damping: u8,
  pub cdef_bits: u8,
  pub cdef_y_strengths: [u8; 8],
  pub cdef_uv_strengths: [u8; 8],
  pub delta_q_present: bool,
  pub config: EncoderConfig,
  pub ref_frames: [u8; INTER_REFS_PER_FRAME],
  pub ref_frame_sign_bias: [bool; INTER_REFS_PER_FRAME],
  pub rec_buffer: ReferenceFramesSet<T>,
  pub base_q_idx: u8,
  pub dc_delta_q: [i8; 3],
  pub ac_delta_q: [i8; 3],
  pub lambda: f64,
  pub me_lambda: f64,
  pub dist_scale: [f64; 3],
  pub me_range_scale: u8,
  pub use_tx_domain_distortion: bool,
  pub use_tx_domain_rate: bool,
  pub idx_in_group_output: u64,
  pub pyramid_level: u64,
  pub enable_early_exit: bool,
  pub tx_mode_select: bool,
  pub enable_inter_txfm_split: bool,
  pub default_filter: FilterMode,
  /// If true, this `FrameInvariants` corresponds to an invalid frame and
  /// should be ignored. Invalid frames occur when a subgop is prematurely
  /// ended, for example, by a key frame or the end of the video.
  pub invalid: bool,
  /// Motion vectors to the _original_ reference frames (not reconstructed).
  /// Used for lookahead purposes.
  pub lookahead_mvs: Arc<Vec<FrameMotionVectors>>,
  /// The lookahead version of `rec_buffer`, used for storing and propagating
  /// the original reference frames (rather than reconstructed ones). The
  /// lookahead uses both `rec_buffer` and `lookahead_rec_buffer`, where
  /// `rec_buffer` contains the current frame's reference frames and
  /// `lookahead_rec_buffer` contains the next frame's reference frames.
  pub lookahead_rec_buffer: ReferenceFramesSet<T>,
  /// Frame width in importance blocks.
  pub w_in_imp_b: usize,
  /// Frame height in importance blocks.
  pub h_in_imp_b: usize,
  /// Intra prediction cost estimations for each importance block.
  pub lookahead_intra_costs: Box<[u32]>,
  /// Future importance values for each importance block. That is, a value
  /// indicating how much future frames depend on the block (for example, via
  /// inter-prediction).
  pub block_importances: Box<[f32]>,
  /// Pre-computed distortion_scale.
  pub distortion_scales: Box<[DistortionScale]>,

  /// Target CPU feature level.
  pub cpu_feature_level: crate::cpu_features::CpuFeatureLevel,
  pub activity_mask: ActivityMask,
  pub enable_segmentation: bool,
}

pub(crate) const fn pos_to_lvl(pos: u64, pyramid_depth: u64) -> u64 {
  // Derive level within pyramid for a frame with a given coding order position
  // For example, with a pyramid of depth 2, the 2 least significant bits of the
  // position determine the level:
  // 00 -> 0
  // 01 -> 2
  // 10 -> 1
  // 11 -> 2
  pyramid_depth - (pos | (1 << pyramid_depth)).trailing_zeros() as u64
}

impl<T: Pixel> FrameInvariants<T> {
  #[allow(clippy::erasing_op, clippy::identity_op)]
  pub fn new(config: EncoderConfig, sequence: Sequence) -> Self {
    assert!(
      sequence.bit_depth <= mem::size_of::<T>() * 8,
      "bit depth cannot fit into u8"
    );
    let use_reduced_tx_set = config.speed_settings.reduced_tx_set;
    let use_tx_domain_distortion =
      config.tune == Tune::Psnr && config.speed_settings.tx_domain_distortion;
    let use_tx_domain_rate = config.speed_settings.tx_domain_rate;

    let w_in_b = 2 * config.width.align_power_of_two_and_shift(3); // MiCols, ((width+7)/8)<<3 >> MI_SIZE_LOG2
    let h_in_b = 2 * config.height.align_power_of_two_and_shift(3); // MiRows, ((height+7)/8)<<3 >> MI_SIZE_LOG2
    let frame_rate = config.frame_rate();

    let mut tiling = TilingInfo::from_target_tiles(
      sequence.sb_size_log2(),
      config.width,
      config.height,
      frame_rate,
      TilingInfo::tile_log2(1, config.tile_cols).unwrap(),
      TilingInfo::tile_log2(1, config.tile_rows).unwrap(),
    );

    if config.tiles > 0 {
      let mut tile_rows_log2 = 0;
      let mut tile_cols_log2 = 0;
      while (tile_rows_log2 < tiling.max_tile_rows_log2)
        || (tile_cols_log2 < tiling.max_tile_cols_log2)
      {
        tiling = TilingInfo::from_target_tiles(
          sequence.sb_size_log2(),
          config.width,
          config.height,
          frame_rate,
          tile_cols_log2,
          tile_rows_log2,
        );

        if tiling.rows * tiling.cols >= config.tiles {
          break;
        };

        if ((tiling.tile_height_sb >= tiling.tile_width_sb)
          && (tiling.tile_rows_log2 < tiling.max_tile_rows_log2))
          || (tile_cols_log2 >= tiling.max_tile_cols_log2)
        {
          tile_rows_log2 += 1;
        } else {
          tile_cols_log2 += 1;
        }
      }
    }

    // Width and height are padded to 8×8 block size.
    let w_in_imp_b = w_in_b / 2;
    let h_in_imp_b = h_in_b / 2;

    Self {
      sequence,
      width: config.width,
      height: config.height,
      sb_width: config.width.align_power_of_two_and_shift(6),
      sb_height: config.height.align_power_of_two_and_shift(6),
      w_in_b,
      h_in_b,
      tiling,
      input_frameno: 0,
      order_hint: 0,
      show_frame: true,
      showable_frame: !sequence.reduced_still_picture_hdr,
      error_resilient: false,
      intra_only: true,
      allow_high_precision_mv: false,
      frame_type: FrameType::KEY,
      show_existing_frame: false,
      frame_to_show_map_idx: 0,
      use_reduced_tx_set,
      reference_mode: ReferenceMode::SINGLE,
      use_prev_frame_mvs: false,
      partition_range: config.speed_settings.partition_range,
      globalmv_transformation_type: [GlobalMVMode::IDENTITY;
        INTER_REFS_PER_FRAME],
      num_tg: 1,
      large_scale_tile: false,
      disable_cdf_update: false,
      allow_screen_content_tools: sequence.force_screen_content_tools,
      force_integer_mv: 1,
      primary_ref_frame: PRIMARY_REF_NONE,
      refresh_frame_flags: ALL_REF_FRAMES_MASK,
      allow_intrabc: false,
      use_ref_frame_mvs: false,
      is_filter_switchable: false,
      is_motion_mode_switchable: false, // 0: only the SIMPLE motion mode will be used.
      disable_frame_end_update_cdf: sequence.reduced_still_picture_hdr,
      allow_warped_motion: false,
      cdef_search_method: CDEFSearchMethod::PickFromQ,
      cdef_damping: 3,
      cdef_bits: 0,
      cdef_y_strengths: [
        0 * 4 + 0,
        1 * 4 + 0,
        2 * 4 + 1,
        3 * 4 + 1,
        5 * 4 + 2,
        7 * 4 + 3,
        10 * 4 + 3,
        13 * 4 + 3,
      ],
      cdef_uv_strengths: [
        0 * 4 + 0,
        1 * 4 + 0,
        2 * 4 + 1,
        3 * 4 + 1,
        5 * 4 + 2,
        7 * 4 + 3,
        10 * 4 + 3,
        13 * 4 + 3,
      ],
      delta_q_present: false,
      ref_frames: [0; INTER_REFS_PER_FRAME],
      ref_frame_sign_bias: [false; INTER_REFS_PER_FRAME],
      rec_buffer: ReferenceFramesSet::new(),
      base_q_idx: config.quantizer as u8,
      dc_delta_q: [0; 3],
      ac_delta_q: [0; 3],
      lambda: 0.0,
      dist_scale: [1.0; 3],
      me_lambda: 0.0,
      me_range_scale: 1,
      use_tx_domain_distortion,
      use_tx_domain_rate,
      idx_in_group_output: 0,
      pyramid_level: 0,
      enable_early_exit: true,
      config,
      tx_mode_select: false,
      default_filter: FilterMode::REGULAR,
      invalid: false,
      lookahead_mvs: {
        let mut vec = Vec::with_capacity(REF_FRAMES);
        for _ in 0..REF_FRAMES {
          vec.push(FrameMotionVectors::new(w_in_b, h_in_b));
        }
        Arc::new(vec)
      },
      lookahead_rec_buffer: ReferenceFramesSet::new(),
      w_in_imp_b,
      h_in_imp_b,
      // dynamic allocation: once per frame
      lookahead_intra_costs: vec![0; w_in_imp_b * h_in_imp_b]
        .into_boxed_slice(),
      // dynamic allocation: once per frame
      block_importances: vec![0.; w_in_imp_b * h_in_imp_b].into_boxed_slice(),
      distortion_scales: vec![
        DistortionScale::default();
        w_in_imp_b * h_in_imp_b
      ]
      .into_boxed_slice(),
      cpu_feature_level: Default::default(),
      activity_mask: Default::default(),
      enable_segmentation: config.speed_settings.enable_segmentation,
      enable_inter_txfm_split: config.speed_settings.enable_inter_tx_split,
    }
  }

  pub fn new_key_frame(
    config: EncoderConfig, sequence: Sequence, gop_input_frameno_start: u64,
  ) -> Self {
    let mut fi = Self::new(config, sequence);
    fi.input_frameno = gop_input_frameno_start;
    fi.tx_mode_select = fi.config.speed_settings.rdo_tx_decision;
    fi
  }

  /// Returns the created FrameInvariants along with a bool indicating success.
  /// This interface provides simpler usage, because we always need the produced
  /// FrameInvariants regardless of success or failure.
  pub(crate) fn new_inter_frame(
    previous_fi: &Self, inter_cfg: &InterConfig, gop_input_frameno_start: u64,
    output_frameno_in_gop: u64, next_keyframe_input_frameno: u64,
    error_resilient: bool,
  ) -> Self {
    let mut fi = previous_fi.clone();
    fi.intra_only = false;
    fi.force_integer_mv = 0; // note: should be 1 if fi.intra_only is true
    fi.idx_in_group_output =
      inter_cfg.get_idx_in_group_output(output_frameno_in_gop);
    fi.tx_mode_select = fi.enable_inter_txfm_split;

    fi.order_hint =
      inter_cfg.get_order_hint(output_frameno_in_gop, fi.idx_in_group_output);
    let input_frameno = inter_cfg
      .get_input_frameno(output_frameno_in_gop, gop_input_frameno_start);
    if input_frameno >= next_keyframe_input_frameno {
      fi.frame_type = FrameType::INTER;
      fi.show_existing_frame = false;
      fi.show_frame = false;
      fi.invalid = true;
      return fi;
    } else {
      fi.invalid = false;
    }

    fi.pyramid_level = inter_cfg.get_level(fi.idx_in_group_output);

    fi.frame_type = if (inter_cfg.switch_frame_interval > 0)
      && (output_frameno_in_gop % inter_cfg.switch_frame_interval == 0)
      && (fi.pyramid_level == 0)
    {
      FrameType::SWITCH
    } else {
      FrameType::INTER
    };
    fi.error_resilient =
      if fi.frame_type == FrameType::SWITCH { true } else { error_resilient };

    // this is the slot that the current frame is going to be saved into
    let slot_idx = inter_cfg.get_slot_idx(fi.pyramid_level, fi.order_hint);
    fi.show_frame = inter_cfg.get_show_frame(fi.idx_in_group_output);
    fi.show_existing_frame =
      inter_cfg.get_show_existing_frame(fi.idx_in_group_output);
    fi.frame_to_show_map_idx = slot_idx;
    fi.refresh_frame_flags = if fi.frame_type == FrameType::SWITCH {
      ALL_REF_FRAMES_MASK
    } else if fi.show_existing_frame {
      0
    } else {
      1 << slot_idx
    };

    let second_ref_frame =
      if fi.idx_in_group_output == 0 { LAST2_FRAME } else { ALTREF_FRAME };
    let ref_in_previous_group = LAST3_FRAME;

    // reuse probability estimates from previous frames only in top level frames
    fi.primary_ref_frame = if fi.error_resilient || (fi.pyramid_level > 2) {
      PRIMARY_REF_NONE
    } else {
      (ref_in_previous_group.to_index()) as u32
    };

    if fi.pyramid_level == 0 {
      // level 0 has no forward references
      // default to last P frame
      fi.ref_frames = [
        // calculations done relative to the slot_idx for this frame.
        // the last four frames can be found by subtracting from the current slot_idx
        // add 4 to prevent underflow
        // TODO: maybe use order_hint here like in get_slot_idx?
        // this is the previous P frame
        (slot_idx + 4 - 1) as u8 % 4
          ; INTER_REFS_PER_FRAME];
      if inter_cfg.multiref {
        // use the second-previous p frame as a second reference frame
        fi.ref_frames[second_ref_frame.to_index()] =
          (slot_idx + 4 - 2) as u8 % 4;
      }
    } else {
      debug_assert!(inter_cfg.multiref);

      // fill in defaults
      // default to backwards reference in lower level
      fi.ref_frames = [{
        let oh = fi.order_hint
          - (inter_cfg.group_input_len as u32 >> fi.pyramid_level);
        let lvl1 = pos_to_lvl(oh as u64, inter_cfg.pyramid_depth);
        if lvl1 == 0 {
          ((oh >> inter_cfg.pyramid_depth) % 4) as u8
        } else {
          3 + lvl1 as u8
        }
      }; INTER_REFS_PER_FRAME];
      // use forward reference in lower level as a second reference frame
      fi.ref_frames[second_ref_frame.to_index()] = {
        let oh = fi.order_hint
          + (inter_cfg.group_input_len as u32 >> fi.pyramid_level);
        let lvl2 = pos_to_lvl(oh as u64, inter_cfg.pyramid_depth);
        if lvl2 == 0 {
          ((oh >> inter_cfg.pyramid_depth) % 4) as u8
        } else {
          3 + lvl2 as u8
        }
      };
      // use a reference to the previous frame in the same level
      // (horizontally) as a third reference
      fi.ref_frames[ref_in_previous_group.to_index()] = { slot_idx as u8 }
    }

    fi.set_ref_frame_sign_bias();

    fi.reference_mode = if inter_cfg.multiref && fi.idx_in_group_output != 0 {
      ReferenceMode::SELECT
    } else {
      ReferenceMode::SINGLE
    };
    fi.input_frameno = input_frameno;
    fi.me_range_scale = (inter_cfg.group_input_len >> fi.pyramid_level) as u8;
    fi
  }

  pub fn set_ref_frame_sign_bias(&mut self) {
    for i in 0..INTER_REFS_PER_FRAME {
      self.ref_frame_sign_bias[i] = if !self.sequence.enable_order_hint {
        false
      } else if let Some(ref rec) =
        self.rec_buffer.frames[self.ref_frames[i] as usize]
      {
        let hint = rec.order_hint;
        self.sequence.get_relative_dist(hint, self.order_hint) > 0
      } else {
        false
      };
    }
  }

  pub fn get_frame_subtype(&self) -> usize {
    if self.frame_type == FrameType::KEY {
      FRAME_SUBTYPE_I
    } else {
      FRAME_SUBTYPE_P + (self.pyramid_level as usize)
    }
  }

  fn pick_strength_from_q(&mut self, qps: &QuantizerParameters) {
    self.cdef_damping = 3 + (self.base_q_idx >> 6);
    let q = bexp64(qps.log_target_q as i64 + q57(QSCALE)) as f32;
    /* These coefficients were trained on libaom. */
    let (y_f1, y_f2, uv_f1, uv_f2) = if !self.intra_only {
      (
        poly2(q, -0.0000023593946_f32, 0.0068615186_f32, 0.02709886_f32, 15),
        poly2(q, -0.00000057629734_f32, 0.0013993345_f32, 0.03831067_f32, 3),
        poly2(q, -0.0000007095069_f32, 0.0034628846_f32, 0.00887099_f32, 15),
        poly2(q, 0.00000023874085_f32, 0.00028223585_f32, 0.05576307_f32, 3),
      )
    } else {
      (
        poly2(q, 0.0000033731974_f32, 0.008070594_f32, 0.0187634_f32, 15),
        poly2(q, 0.0000029167343_f32, 0.0027798624_f32, 0.0079405_f32, 3),
        poly2(q, -0.0000130790995_f32, 0.012892405_f32, -0.00748388_f32, 15),
        poly2(q, 0.0000032651783_f32, 0.00035520183_f32, 0.00228092_f32, 3),
      )
    };
    self.cdef_y_strengths[0] = (y_f1 * CDEF_SEC_STRENGTHS as i32 + y_f2) as u8;
    self.cdef_uv_strengths[0] =
      (uv_f1 * CDEF_SEC_STRENGTHS as i32 + uv_f2) as u8;
  }

  pub fn set_quantizers(&mut self, qps: &QuantizerParameters) {
    self.base_q_idx = qps.ac_qi[0];
    let base_q_idx = self.base_q_idx as i32;
    for pi in 0..3 {
      debug_assert!(qps.dc_qi[pi] as i32 - base_q_idx >= -128);
      debug_assert!((qps.dc_qi[pi] as i32 - base_q_idx) < 128);
      debug_assert!(qps.ac_qi[pi] as i32 - base_q_idx >= -128);
      debug_assert!((qps.ac_qi[pi] as i32 - base_q_idx) < 128);
      self.dc_delta_q[pi] = (qps.dc_qi[pi] as i32 - base_q_idx) as i8;
      self.ac_delta_q[pi] = (qps.ac_qi[pi] as i32 - base_q_idx) as i8;
    }
    self.lambda =
      qps.lambda * ((1 << (2 * (self.sequence.bit_depth - 8))) as f64);
    self.me_lambda = self.lambda.sqrt();
    self.dist_scale = qps.dist_scale;

    match self.cdef_search_method {
      CDEFSearchMethod::PickFromQ => {
        self.pick_strength_from_q(qps);
      }
      // TODO: implement FastSearch and FullSearch
      _ => unreachable!(),
    }
  }

  #[inline(always)]
  pub fn sb_size_log2(&self) -> usize {
    self.sequence.sb_size_log2()
  }
}

impl<T: Pixel> Default for FrameInvariants<T> {
  fn default() -> Self {
    FrameInvariants::new(EncoderConfig::default(), Sequence::default())
  }
}

impl<T: Pixel> fmt::Display for FrameInvariants<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "Input Frame {} - {}", self.input_frameno, self.frame_type)
  }
}

pub fn write_temporal_delimiter(packet: &mut dyn io::Write) -> io::Result<()> {
  packet.write_all(&TEMPORAL_DELIMITER)?;
  Ok(())
}

fn write_obus<T: Pixel>(
  packet: &mut dyn io::Write, fi: &FrameInvariants<T>, fs: &FrameState<T>,
  inter_cfg: &InterConfig,
) -> io::Result<()> {
  let obu_extension = 0 as u32;

  let mut buf1 = Vec::new();

  // write sequence header obu if KEY_FRAME, preceded by 4-byte size
  if fi.frame_type == FrameType::KEY {
    let mut buf2 = Vec::new();
    {
      let mut bw2 = BitWriter::endian(&mut buf2, BigEndian);
      bw2.write_sequence_header_obu(fi)?;
      bw2.write_bit(true)?; // trailing bit
      bw2.byte_align()?;
    }

    {
      let mut bw1 = BitWriter::endian(&mut buf1, BigEndian);
      bw1.write_obu_header(ObuType::OBU_SEQUENCE_HEADER, obu_extension)?;
    }
    packet.write_all(&buf1).unwrap();
    buf1.clear();

    {
      let mut bw1 = BitWriter::endian(&mut buf1, BigEndian);
      bw1.write_uleb128(buf2.len() as u64)?;
    }
    packet.write_all(&buf1).unwrap();
    buf1.clear();

    packet.write_all(&buf2).unwrap();
    buf2.clear();

    if fi.sequence.content_light.is_some() {
      let mut bw1 = BitWriter::endian(&mut buf1, BigEndian);
      bw1.write_metadata_obu(ObuMetaType::OBU_META_HDR_CLL, fi.sequence)?;
      packet.write_all(&buf1).unwrap();
      buf1.clear();
    }

    if fi.sequence.mastering_display.is_some() {
      let mut bw1 = BitWriter::endian(&mut buf1, BigEndian);
      bw1.write_metadata_obu(ObuMetaType::OBU_META_HDR_MDCV, fi.sequence)?;
      packet.write_all(&buf1).unwrap();
      buf1.clear();
    }
  }

  let mut buf2 = Vec::new();
  {
    let mut bw2 = BitWriter::endian(&mut buf2, BigEndian);
    bw2.write_frame_header_obu(fi, fs, inter_cfg)?;
  }

  {
    let mut bw1 = BitWriter::endian(&mut buf1, BigEndian);
    bw1.write_obu_header(ObuType::OBU_FRAME_HEADER, obu_extension)?;
  }
  packet.write_all(&buf1).unwrap();
  buf1.clear();

  {
    let mut bw1 = BitWriter::endian(&mut buf1, BigEndian);
    bw1.write_uleb128(buf2.len() as u64)?;
  }

  packet.write_all(&buf1).unwrap();
  buf1.clear();

  packet.write_all(&buf2).unwrap();
  buf2.clear();

  Ok(())
}

/// Write into `dst` the difference between the blocks at `src1` and `src2`
fn diff<T: Pixel>(
  dst: &mut [i16], src1: &PlaneRegion<'_, T>, src2: &PlaneRegion<'_, T>,
  width: usize, height: usize,
) {
  for ((l, s1), s2) in dst
    .chunks_mut(width)
    .take(height)
    .zip(src1.rows_iter())
    .zip(src2.rows_iter())
  {
    for ((r, v1), v2) in l.iter_mut().zip(s1).zip(s2) {
      *r = i16::cast_from(*v1) - i16::cast_from(*v2);
    }
  }
}

fn get_qidx<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>, cw: &ContextWriter,
  tile_bo: TileBlockOffset,
) -> u8 {
  let mut qidx = fi.base_q_idx;
  let sidx = cw.bc.blocks[tile_bo].segmentation_idx as usize;
  if ts.segmentation.features[sidx][SegLvl::SEG_LVL_ALT_Q as usize] {
    let delta = ts.segmentation.data[sidx][SegLvl::SEG_LVL_ALT_Q as usize];
    qidx = clamp((qidx as i16) + delta, 0, 255) as u8;
  }
  qidx
}

// For a transform block,
// predict, transform, quantize, write coefficients to a bitstream,
// dequantize, inverse-transform.
pub fn encode_tx_block<T: Pixel>(
  fi: &FrameInvariants<T>,
  ts: &mut TileStateMut<'_, T>,
  cw: &mut ContextWriter,
  w: &mut dyn Writer,
  p: usize,
  // Offset in the luma plane of the partition enclosing this block.
  tile_partition_bo: TileBlockOffset,
  // tx block position within a partition, unit: tx block number
  bx: usize,
  by: usize,
  // Offset in the luma plane where this tx block is colocated. Note that for
  // a chroma block, this offset might be outside of the current partition.
  // For example in 4:2:0, four 4x4 luma partitions share one 4x4 chroma block,
  // this block is part of the last 4x4 partition, but its `tx_bo` offset
  // matches the offset of the first 4x4 partition.
  tx_bo: TileBlockOffset,
  mode: PredictionMode,
  tx_size: TxSize,
  tx_type: TxType,
  bsize: BlockSize,
  po: PlaneOffset,
  skip: bool,
  qidx: u8,
  ac: &[i16],
  pred_intra_param: IntraParam,
  rdo_type: RDOType,
  need_recon_pixel: bool,
) -> (bool, ScaledDistortion) {
  let PlaneConfig { xdec, ydec, .. } = ts.input.planes[p].cfg;
  let tile_rect = ts.tile_rect().decimated(xdec, ydec);
  let area = Area::BlockStartingAt { bo: tx_bo.0 };

  debug_assert!(
    tx_size.sqr() <= TxSize::TX_32X32 || tx_type == TxType::DCT_DCT
  );

  let plane_bsize = bsize.subsampled_size(xdec, ydec);

  debug_assert!(p != 0 || !mode.is_intra() || tx_size.block_size() == plane_bsize || need_recon_pixel,
    "mode.is_intra()={:#?}, plane={:#?}, tx_size.block_size()={:#?}, plane_bsize={:#?}, need_recon_pixel={:#?}",
    mode.is_intra(), p, tx_size.block_size(), plane_bsize, need_recon_pixel);

  let ief_params = if mode.is_directional()
    && fi.sequence.enable_intra_edge_filter
  {
    let above_block_info = ts.above_block_info(tile_partition_bo, p);
    let left_block_info = ts.left_block_info(tile_partition_bo, p);
    Some(IntraEdgeFilterParameters::new(p, above_block_info, left_block_info))
  } else {
    None
  };

  let rec = &mut ts.rec.planes[p];

  if mode.is_intra() {
    let bit_depth = fi.sequence.bit_depth;
    let edge_buf = get_intra_edges(
      &rec.as_const(),
      tile_partition_bo,
      bx,
      by,
      bsize,
      po,
      tx_size,
      bit_depth,
      Some(mode),
      fi.sequence.enable_intra_edge_filter,
      pred_intra_param,
    );
    mode.predict_intra(
      tile_rect,
      &mut rec.subregion_mut(area),
      tx_size,
      bit_depth,
      ac,
      pred_intra_param,
      ief_params,
      &edge_buf,
      fi.cpu_feature_level,
    );
  }

  if skip {
    return (false, ScaledDistortion::zero());
  }

  let coded_tx_area = av1_get_coded_tx_size(tx_size).area();
  let mut residual_storage: Aligned<[i16; 64 * 64]> = Aligned::uninitialized();
  let mut coeffs_storage: Aligned<[T::Coeff; 64 * 64]> =
    Aligned::uninitialized();
  let mut qcoeffs_storage: Aligned<[MaybeUninit<T::Coeff>; 32 * 32]> =
    Aligned::uninitialized();
  let mut rcoeffs_storage: Aligned<[T::Coeff; 32 * 32]> =
    Aligned::uninitialized();
  let residual = &mut residual_storage.data[..tx_size.area()];
  let coeffs = &mut coeffs_storage.data[..tx_size.area()];
  let qcoeffs = init_slice_repeat_mut(
    &mut qcoeffs_storage.data[..coded_tx_area],
    T::Coeff::cast_from(0),
  );
  let rcoeffs = &mut rcoeffs_storage.data[..coded_tx_area];

  diff(
    residual,
    &ts.input_tile.planes[p].subregion(area),
    &rec.subregion(area),
    tx_size.width(),
    tx_size.height(),
  );

  forward_transform(
    residual,
    coeffs,
    tx_size.width(),
    tx_size,
    tx_type,
    fi.sequence.bit_depth,
    fi.cpu_feature_level,
  );

  let eob = ts.qc.quantize(coeffs, qcoeffs, tx_size, tx_type);

  let has_coeff = if need_recon_pixel || rdo_type.needs_coeff_rate() {
    cw.write_coeffs_lv_map(
      w,
      p,
      tx_bo,
      qcoeffs,
      eob,
      mode,
      tx_size,
      tx_type,
      plane_bsize,
      xdec,
      ydec,
      fi.use_reduced_tx_set,
    )
  } else {
    true
  };

  // Reconstruct
  dequantize(
    qidx,
    qcoeffs,
    eob,
    rcoeffs,
    tx_size,
    fi.sequence.bit_depth,
    fi.dc_delta_q[p],
    fi.ac_delta_q[p],
    fi.cpu_feature_level,
  );

  if !fi.use_tx_domain_distortion || need_recon_pixel {
    inverse_transform_add(
      rcoeffs,
      &mut rec.subregion_mut(area),
      eob,
      tx_size,
      tx_type,
      fi.sequence.bit_depth,
      fi.cpu_feature_level,
    );
  }

  let tx_dist = if rdo_type.needs_tx_dist() {
    // Store tx-domain distortion of this block
    // rcoeffs above 32 rows/cols aren't held in the array, because they are
    // always 0. The first 32x32 is stored first in coeffs so we can iterate
    // over coeffs and rcoeffs for the first 32 rows/cols. For the
    // coefficients above 32 rows/cols, we iterate over the rest of coeffs
    // with the assumption that rcoeff coefficients are zero.
    let mut raw_tx_dist = coeffs
      .iter()
      .zip(rcoeffs.iter())
      .map(|(&a, &b)| {
        let c = i32::cast_from(a) - i32::cast_from(b);
        (c * c) as u64
      })
      .sum::<u64>()
      + coeffs[rcoeffs.len()..]
        .iter()
        .map(|&a| {
          let c = i32::cast_from(a);
          (c * c) as u64
        })
        .sum::<u64>();

    let tx_dist_scale_bits = 2 * (3 - get_log_tx_scale(tx_size));
    let tx_dist_scale_rounding_offset = 1 << (tx_dist_scale_bits - 1);

    raw_tx_dist =
      (raw_tx_dist + tx_dist_scale_rounding_offset) >> tx_dist_scale_bits;

    if rdo_type == RDOType::TxDistEstRate {
      // look up rate and distortion in table
      let estimated_rate = estimate_rate(fi.base_q_idx, tx_size, raw_tx_dist);
      w.add_bits_frac(estimated_rate as u32);
    }

    let bias = distortion_scale(fi, ts.to_frame_block_offset(tx_bo), bsize);
    RawDistortion::new(raw_tx_dist) * bias * fi.dist_scale[p]
  } else {
    ScaledDistortion::zero()
  };

  (has_coeff, tx_dist)
}

pub fn motion_compensate<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &mut TileStateMut<'_, T>,
  cw: &mut ContextWriter, luma_mode: PredictionMode, ref_frames: [RefType; 2],
  mvs: [MotionVector; 2], bsize: BlockSize, tile_bo: TileBlockOffset,
  luma_only: bool,
) {
  debug_assert!(!luma_mode.is_intra());

  let PlaneConfig { xdec: u_xdec, ydec: u_ydec, .. } = ts.input.planes[1].cfg;

  // Inter mode prediction can take place once for a whole partition,
  // instead of each tx-block.
  let num_planes = 1
    + if !luma_only && has_chroma(tile_bo, bsize, u_xdec, u_ydec) {
      2
    } else {
      0
    };

  let luma_tile_rect = ts.tile_rect();
  for p in 0..num_planes {
    let plane_bsize =
      if p == 0 { bsize } else { bsize.subsampled_size(u_xdec, u_ydec) };

    let rec = &mut ts.rec.planes[p];
    let po = tile_bo.plane_offset(rec.plane_cfg);
    let &PlaneConfig { xdec, ydec, .. } = rec.plane_cfg;
    let tile_rect = luma_tile_rect.decimated(xdec, ydec);

    let area = Area::BlockStartingAt { bo: tile_bo.0 };
    if p > 0 && bsize < BlockSize::BLOCK_8X8 {
      let mut some_use_intra = false;
      if bsize == BlockSize::BLOCK_4X4 || bsize == BlockSize::BLOCK_4X8 {
        some_use_intra |=
          cw.bc.blocks[tile_bo.with_offset(-1, 0)].mode.is_intra();
      };
      if !some_use_intra && bsize == BlockSize::BLOCK_4X4
        || bsize == BlockSize::BLOCK_8X4
      {
        some_use_intra |=
          cw.bc.blocks[tile_bo.with_offset(0, -1)].mode.is_intra();
      };
      if !some_use_intra && bsize == BlockSize::BLOCK_4X4 {
        some_use_intra |=
          cw.bc.blocks[tile_bo.with_offset(-1, -1)].mode.is_intra();
      };

      if some_use_intra {
        luma_mode.predict_inter(
          fi,
          tile_rect,
          p,
          po,
          &mut rec.subregion_mut(area),
          plane_bsize.width(),
          plane_bsize.height(),
          ref_frames,
          mvs,
        );
      } else {
        assert!(u_xdec == 1 && u_ydec == 1);
        // TODO: these are absolutely only valid for 4:2:0
        if bsize == BlockSize::BLOCK_4X4 {
          let mv0 = cw.bc.blocks[tile_bo.with_offset(-1, -1)].mv;
          let rf0 = cw.bc.blocks[tile_bo.with_offset(-1, -1)].ref_frames;
          let mv1 = cw.bc.blocks[tile_bo.with_offset(0, -1)].mv;
          let rf1 = cw.bc.blocks[tile_bo.with_offset(0, -1)].ref_frames;
          let po1 = PlaneOffset { x: po.x + 2, y: po.y };
          let area1 = Area::StartingAt { x: po1.x, y: po1.y };
          let mv2 = cw.bc.blocks[tile_bo.with_offset(-1, 0)].mv;
          let rf2 = cw.bc.blocks[tile_bo.with_offset(-1, 0)].ref_frames;
          let po2 = PlaneOffset { x: po.x, y: po.y + 2 };
          let area2 = Area::StartingAt { x: po2.x, y: po2.y };
          let po3 = PlaneOffset { x: po.x + 2, y: po.y + 2 };
          let area3 = Area::StartingAt { x: po3.x, y: po3.y };
          luma_mode.predict_inter(
            fi,
            tile_rect,
            p,
            po,
            &mut rec.subregion_mut(area),
            2,
            2,
            rf0,
            mv0,
          );
          luma_mode.predict_inter(
            fi,
            tile_rect,
            p,
            po1,
            &mut rec.subregion_mut(area1),
            2,
            2,
            rf1,
            mv1,
          );
          luma_mode.predict_inter(
            fi,
            tile_rect,
            p,
            po2,
            &mut rec.subregion_mut(area2),
            2,
            2,
            rf2,
            mv2,
          );
          luma_mode.predict_inter(
            fi,
            tile_rect,
            p,
            po3,
            &mut rec.subregion_mut(area3),
            2,
            2,
            ref_frames,
            mvs,
          );
        }
        if bsize == BlockSize::BLOCK_8X4 {
          let mv1 = cw.bc.blocks[tile_bo.with_offset(0, -1)].mv;
          let rf1 = cw.bc.blocks[tile_bo.with_offset(0, -1)].ref_frames;
          luma_mode.predict_inter(
            fi,
            tile_rect,
            p,
            po,
            &mut rec.subregion_mut(area),
            4,
            2,
            rf1,
            mv1,
          );
          let po3 = PlaneOffset { x: po.x, y: po.y + 2 };
          let area3 = Area::StartingAt { x: po3.x, y: po3.y };
          luma_mode.predict_inter(
            fi,
            tile_rect,
            p,
            po3,
            &mut rec.subregion_mut(area3),
            4,
            2,
            ref_frames,
            mvs,
          );
        }
        if bsize == BlockSize::BLOCK_4X8 {
          let mv2 = cw.bc.blocks[tile_bo.with_offset(-1, 0)].mv;
          let rf2 = cw.bc.blocks[tile_bo.with_offset(-1, 0)].ref_frames;
          luma_mode.predict_inter(
            fi,
            tile_rect,
            p,
            po,
            &mut rec.subregion_mut(area),
            2,
            4,
            rf2,
            mv2,
          );
          let po3 = PlaneOffset { x: po.x + 2, y: po.y };
          let area3 = Area::StartingAt { x: po3.x, y: po3.y };
          luma_mode.predict_inter(
            fi,
            tile_rect,
            p,
            po3,
            &mut rec.subregion_mut(area3),
            2,
            4,
            ref_frames,
            mvs,
          );
        }
      }
    } else {
      luma_mode.predict_inter(
        fi,
        tile_rect,
        p,
        po,
        &mut rec.subregion_mut(area),
        plane_bsize.width(),
        plane_bsize.height(),
        ref_frames,
        mvs,
      );
    }
  }
}

pub fn save_block_motion<T: Pixel>(
  ts: &mut TileStateMut<'_, T>, bsize: BlockSize, tile_bo: TileBlockOffset,
  ref_frame: usize, mv: MotionVector,
) {
  let tile_mvs = &mut ts.mvs[ref_frame];
  let tile_bo_x_end = (tile_bo.0.x + bsize.width_mi()).min(ts.mi_width);
  let tile_bo_y_end = (tile_bo.0.y + bsize.height_mi()).min(ts.mi_height);
  for mi_y in tile_bo.0.y..tile_bo_y_end {
    for mi_x in tile_bo.0.x..tile_bo_x_end {
      tile_mvs[mi_y][mi_x] = mv;
    }
  }
}

pub fn encode_block_pre_cdef<T: Pixel>(
  seq: &Sequence, ts: &TileStateMut<'_, T>, cw: &mut ContextWriter,
  w: &mut dyn Writer, bsize: BlockSize, tile_bo: TileBlockOffset, skip: bool,
) -> bool {
  cw.bc.blocks.set_skip(tile_bo, bsize, skip);
  if ts.segmentation.enabled
    && ts.segmentation.update_map
    && ts.segmentation.preskip
  {
    cw.write_segmentation(
      w,
      tile_bo,
      bsize,
      false,
      ts.segmentation.last_active_segid,
    );
  }
  cw.write_skip(w, tile_bo, skip);
  if ts.segmentation.enabled
    && ts.segmentation.update_map
    && !ts.segmentation.preskip
  {
    cw.write_segmentation(
      w,
      tile_bo,
      bsize,
      skip,
      ts.segmentation.last_active_segid,
    );
  }
  if !skip && seq.enable_cdef {
    cw.bc.cdef_coded = true;
  }
  cw.bc.cdef_coded
}

pub fn encode_block_post_cdef<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &mut TileStateMut<'_, T>,
  cw: &mut ContextWriter, w: &mut dyn Writer, luma_mode: PredictionMode,
  chroma_mode: PredictionMode, angle_delta: AngleDelta,
  ref_frames: [RefType; 2], mvs: [MotionVector; 2], bsize: BlockSize,
  tile_bo: TileBlockOffset, skip: bool, cfl: CFLParams, tx_size: TxSize,
  tx_type: TxType, mode_context: usize, mv_stack: &[CandidateMV],
  rdo_type: RDOType, need_recon_pixel: bool, record_stats: bool,
) -> (bool, ScaledDistortion) {
  let is_inter = !luma_mode.is_intra();
  if is_inter {
    assert!(luma_mode == chroma_mode);
  };
  let sb_size = if fi.sequence.use_128x128_superblock {
    BlockSize::BLOCK_128X128
  } else {
    BlockSize::BLOCK_64X64
  };
  let PlaneConfig { xdec, ydec, .. } = ts.input.planes[1].cfg;
  if skip {
    cw.bc.reset_skip_context(tile_bo, bsize, xdec, ydec);
  }
  cw.bc.blocks.set_block_size(tile_bo, bsize);
  cw.bc.blocks.set_mode(tile_bo, bsize, luma_mode);
  cw.bc.blocks.set_tx_size(tile_bo, bsize, tx_size);
  cw.bc.blocks.set_ref_frames(tile_bo, bsize, ref_frames);
  cw.bc.blocks.set_motion_vectors(tile_bo, bsize, mvs);

  //write_q_deltas();
  if cw.bc.code_deltas
    && ts.deblock.block_deltas_enabled
    && (bsize < sb_size || !skip)
  {
    cw.write_block_deblock_deltas(w, tile_bo, ts.deblock.block_delta_multi);
  }
  cw.bc.code_deltas = false;

  if fi.frame_type.has_inter() {
    cw.write_is_inter(w, tile_bo, is_inter);
    if is_inter {
      cw.fill_neighbours_ref_counts(tile_bo);
      cw.write_ref_frames(w, fi, tile_bo);

      if luma_mode.is_compound() {
        cw.write_compound_mode(w, luma_mode, mode_context);
      } else {
        cw.write_inter_mode(w, luma_mode, mode_context);
      }

      let ref_mv_idx = 0;
      let num_mv_found = mv_stack.len();

      if luma_mode == PredictionMode::NEWMV
        || luma_mode == PredictionMode::NEW_NEWMV
      {
        if luma_mode == PredictionMode::NEW_NEWMV {
          assert!(num_mv_found >= 2);
        }
        for idx in 0..2 {
          if num_mv_found > idx + 1 {
            let drl_mode = ref_mv_idx > idx;
            let ctx: usize = (mv_stack[idx].weight < REF_CAT_LEVEL) as usize
              + (mv_stack[idx + 1].weight < REF_CAT_LEVEL) as usize;
            cw.write_drl_mode(w, drl_mode, ctx);
            if !drl_mode {
              break;
            }
          }
        }
      }

      let ref_mvs = if num_mv_found > 0 {
        [mv_stack[ref_mv_idx].this_mv, mv_stack[ref_mv_idx].comp_mv]
      } else {
        [MotionVector::default(); 2]
      };

      let mv_precision = if fi.force_integer_mv != 0 {
        MvSubpelPrecision::MV_SUBPEL_NONE
      } else if fi.allow_high_precision_mv {
        MvSubpelPrecision::MV_SUBPEL_HIGH_PRECISION
      } else {
        MvSubpelPrecision::MV_SUBPEL_LOW_PRECISION
      };

      if luma_mode == PredictionMode::NEWMV
        || luma_mode == PredictionMode::NEW_NEWMV
        || luma_mode == PredictionMode::NEW_NEARESTMV
      {
        cw.write_mv(w, mvs[0], ref_mvs[0], mv_precision);
      }
      if luma_mode == PredictionMode::NEW_NEWMV
        || luma_mode == PredictionMode::NEAREST_NEWMV
      {
        cw.write_mv(w, mvs[1], ref_mvs[1], mv_precision);
      }

      if luma_mode.has_near() {
        let ref_mv_idx = if luma_mode >= PredictionMode::NEAR0MV
          && luma_mode <= PredictionMode::NEAR2MV
        {
          luma_mode as usize - PredictionMode::NEAR0MV as usize + 1
        } else {
          1
        };
        if luma_mode != PredictionMode::NEAR0MV {
          assert!(num_mv_found > ref_mv_idx);
        }

        for idx in 1..3 {
          if num_mv_found > idx + 1 {
            let drl_mode = ref_mv_idx > idx;
            let ctx: usize = (mv_stack[idx].weight < REF_CAT_LEVEL) as usize
              + (mv_stack[idx + 1].weight < REF_CAT_LEVEL) as usize;

            cw.write_drl_mode(w, drl_mode, ctx);
            if !drl_mode {
              break;
            }
          }
        }
        if mv_stack.len() > 1 {
          assert!(mv_stack[ref_mv_idx].this_mv.row == mvs[0].row);
          assert!(mv_stack[ref_mv_idx].this_mv.col == mvs[0].col);
        } else {
          assert!(0 == mvs[0].row);
          assert!(0 == mvs[0].col);
        }
      } else if luma_mode == PredictionMode::NEARESTMV {
        if mv_stack.is_empty() {
          assert_eq!(mvs[0].row, 0);
          assert_eq!(mvs[0].col, 0);
        } else {
          assert_eq!(mvs[0].row, mv_stack[0].this_mv.row);
          assert_eq!(mvs[0].col, mv_stack[0].this_mv.col);
        }
      }
    } else {
      cw.write_intra_mode(w, bsize, luma_mode);
    }
  } else {
    cw.write_intra_mode_kf(w, tile_bo, luma_mode);
  }

  if !is_inter {
    if luma_mode.is_directional() && bsize >= BlockSize::BLOCK_8X8 {
      cw.write_angle_delta(w, angle_delta.y, luma_mode);
    }
    if has_chroma(tile_bo, bsize, xdec, ydec) {
      cw.write_intra_uv_mode(w, chroma_mode, luma_mode, bsize);
      if chroma_mode.is_cfl() {
        assert!(bsize.cfl_allowed());
        cw.write_cfl_alphas(w, cfl);
      }
      if chroma_mode.is_directional() && bsize >= BlockSize::BLOCK_8X8 {
        cw.write_angle_delta(w, angle_delta.uv, chroma_mode);
      }
    }

    if fi.allow_screen_content_tools > 0
      && bsize >= BlockSize::BLOCK_8X8
      && bsize.width() <= 64
      && bsize.height() <= 64
    {
      cw.write_use_palette_mode(
        w,
        false,
        bsize,
        tile_bo,
        luma_mode,
        chroma_mode,
        xdec,
        ydec,
      );
    }

    if fi.sequence.enable_filter_intra
      && luma_mode == PredictionMode::DC_PRED
      && bsize.width() <= 32
      && bsize.height() <= 32
    {
      cw.write_use_filter_intra(w, false, bsize); // turn off FILTER_INTRA
    }
  }

  // write tx_size here
  if fi.tx_mode_select {
    if bsize > BlockSize::BLOCK_4X4 && (!is_inter || !skip) {
      if !is_inter {
        cw.write_tx_size_intra(w, tile_bo, bsize, tx_size);
        cw.bc.update_tx_size_context(tile_bo, bsize, tx_size, false);
      } else {
        // write var_tx_size
        // if here, bsize > BLOCK_4X4 && is_inter && !skip && !Lossless
        debug_assert!(fi.tx_mode_select);
        debug_assert!(bsize > BlockSize::BLOCK_4X4);
        debug_assert!(is_inter);
        debug_assert!(!skip);
        let max_tx_size = max_txsize_rect_lookup[bsize as usize];
        debug_assert!(max_tx_size.block_size() <= BlockSize::BLOCK_64X64);

        //TODO: "&& tx_size.block_size() < bsize" will be replaced with tx-split info for a partition
        //  once it is available.
        let txfm_split =
          fi.enable_inter_txfm_split && tx_size.block_size() < bsize;

        // TODO: Revise write_tx_size_inter() for txfm_split = true
        cw.write_tx_size_inter(
          w,
          tile_bo,
          bsize,
          max_tx_size,
          txfm_split,
          0,
          0,
          0,
        );
      }
    } else {
      cw.bc.update_tx_size_context(tile_bo, bsize, tx_size, is_inter && skip);
    }
  }

  if record_stats {
    let pixels = tx_size.area();
    ts.enc_stats.block_size_counts[bsize as usize] += pixels;
    ts.enc_stats.tx_type_counts[tx_type as usize] += pixels;
    ts.enc_stats.luma_pred_mode_counts[luma_mode as usize] += pixels;
    ts.enc_stats.chroma_pred_mode_counts[chroma_mode as usize] += pixels;
    if skip {
      ts.enc_stats.skip_block_count += pixels;
    }
  }

  if fi.sequence.enable_intra_edge_filter {
    for y in 0..bsize.height_mi() {
      for x in 0..bsize.width_mi() {
        let bi = &mut ts.coded_block_info[tile_bo.0.y + y][tile_bo.0.x + x];
        bi.luma_mode = luma_mode;
        bi.chroma_mode = chroma_mode;
        bi.reference_types = ref_frames;
      }
    }
  }

  if is_inter {
    motion_compensate(
      fi, ts, cw, luma_mode, ref_frames, mvs, bsize, tile_bo, false,
    );
    write_tx_tree(
      fi,
      ts,
      cw,
      w,
      luma_mode,
      angle_delta.y,
      tile_bo,
      bsize,
      tx_size,
      tx_type,
      skip,
      false,
      rdo_type,
      need_recon_pixel,
    )
  } else {
    write_tx_blocks(
      fi,
      ts,
      cw,
      w,
      luma_mode,
      chroma_mode,
      angle_delta,
      tile_bo,
      bsize,
      tx_size,
      tx_type,
      skip,
      cfl,
      false,
      rdo_type,
      need_recon_pixel,
    )
  }
}

pub fn luma_ac<T: Pixel>(
  ac: &mut [i16], ts: &mut TileStateMut<'_, T>, tile_bo: TileBlockOffset,
  bsize: BlockSize,
) {
  let PlaneConfig { xdec, ydec, .. } = ts.input.planes[1].cfg;
  let plane_bsize = bsize.subsampled_size(xdec, ydec);
  let bo = if bsize.is_sub8x8(xdec, ydec) {
    let offset = bsize.sub8x8_offset(xdec, ydec);
    tile_bo.with_offset(offset.0, offset.1)
  } else {
    tile_bo
  };
  let rec = &ts.rec.planes[0];
  let luma = &rec.subregion(Area::BlockStartingAt { bo: bo.0 });

  let mut sum: i32 = 0;
  for sub_y in 0..plane_bsize.height() {
    for sub_x in 0..plane_bsize.width() {
      let y = sub_y << ydec;
      let x = sub_x << xdec;
      let mut sample: i16 = i16::cast_from(luma[y][x]);
      if xdec != 0 {
        sample += i16::cast_from(luma[y][x + 1]);
      }
      if ydec != 0 {
        debug_assert!(xdec != 0);
        sample +=
          i16::cast_from(luma[y + 1][x]) + i16::cast_from(luma[y + 1][x + 1]);
      }
      sample <<= 3 - xdec - ydec;
      ac[sub_y * plane_bsize.width() + sub_x] = sample;
      sum += sample as i32;
    }
  }
  let shift = plane_bsize.width_log2() + plane_bsize.height_log2();
  let average = ((sum + (1 << (shift - 1))) >> shift) as i16;
  for sub_y in 0..plane_bsize.height() {
    for sub_x in 0..plane_bsize.width() {
      ac[sub_y * plane_bsize.width() + sub_x] -= average;
    }
  }
}

pub fn write_tx_blocks<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &mut TileStateMut<'_, T>,
  cw: &mut ContextWriter, w: &mut dyn Writer, luma_mode: PredictionMode,
  chroma_mode: PredictionMode, angle_delta: AngleDelta,
  tile_bo: TileBlockOffset, bsize: BlockSize, tx_size: TxSize,
  tx_type: TxType, skip: bool, cfl: CFLParams, luma_only: bool,
  rdo_type: RDOType, need_recon_pixel: bool,
) -> (bool, ScaledDistortion) {
  let bw = bsize.width_mi() / tx_size.width_mi();
  let bh = bsize.height_mi() / tx_size.height_mi();
  let qidx = get_qidx(fi, ts, cw, tile_bo);
  assert_ne!(qidx, 0); // lossless is not yet supported

  let PlaneConfig { xdec, ydec, .. } = ts.input.planes[1].cfg;
  let mut ac: Aligned<[i16; 32 * 32]> = Aligned::uninitialized();
  let mut partition_has_coeff: bool = false;
  let mut tx_dist = ScaledDistortion::zero();
  let do_chroma = has_chroma(tile_bo, bsize, xdec, ydec);

  ts.qc.update(
    qidx,
    tx_size,
    luma_mode.is_intra(),
    fi.sequence.bit_depth,
    fi.dc_delta_q[0],
    0,
  );

  for by in 0..bh {
    for bx in 0..bw {
      let tx_bo = TileBlockOffset(BlockOffset {
        x: tile_bo.0.x + bx * tx_size.width_mi(),
        y: tile_bo.0.y + by * tx_size.height_mi(),
      });

      let po = tx_bo.plane_offset(&ts.input.planes[0].cfg);
      let (has_coeff, dist) = encode_tx_block(
        fi,
        ts,
        cw,
        w,
        0,
        tile_bo,
        bx,
        by,
        tx_bo,
        luma_mode,
        tx_size,
        tx_type,
        bsize,
        po,
        skip,
        qidx,
        &ac.data,
        IntraParam::AngleDelta(angle_delta.y),
        rdo_type,
        need_recon_pixel,
      );
      partition_has_coeff |= has_coeff;
      tx_dist += dist;
    }
  }

  if luma_only {
    return (partition_has_coeff, tx_dist);
  };

  let uv_tx_size = bsize.largest_chroma_tx_size(xdec, ydec);

  let mut bw_uv = (bw * tx_size.width_mi()) >> xdec;
  let mut bh_uv = (bh * tx_size.height_mi()) >> ydec;

  if (bw_uv == 0 || bh_uv == 0) && do_chroma {
    bw_uv = 1;
    bh_uv = 1;
  }

  bw_uv /= uv_tx_size.width_mi();
  bh_uv /= uv_tx_size.height_mi();

  if chroma_mode.is_cfl() {
    luma_ac(&mut ac.data, ts, tile_bo, bsize);
  }

  if bw_uv > 0 && bh_uv > 0 {
    let uv_tx_type = if uv_tx_size.width() >= 32 || uv_tx_size.height() >= 32 {
      TxType::DCT_DCT
    } else {
      uv_intra_mode_to_tx_type_context(chroma_mode)
    };

    for p in 1..3 {
      ts.qc.update(
        qidx,
        uv_tx_size,
        true,
        fi.sequence.bit_depth,
        fi.dc_delta_q[p],
        fi.ac_delta_q[p],
      );
      let alpha = cfl.alpha(p - 1);
      for by in 0..bh_uv {
        for bx in 0..bw_uv {
          let tx_bo = TileBlockOffset(BlockOffset {
            x: tile_bo.0.x + ((bx * uv_tx_size.width_mi()) << xdec)
              - ((bw * tx_size.width_mi() == 1) as usize) * xdec,
            y: tile_bo.0.y + ((by * uv_tx_size.height_mi()) << ydec)
              - ((bh * tx_size.height_mi() == 1) as usize) * ydec,
          });

          let mut po = tile_bo.plane_offset(&ts.input.planes[p].cfg);
          po.x += (bx * uv_tx_size.width()) as isize;
          po.y += (by * uv_tx_size.height()) as isize;
          let (has_coeff, dist) = encode_tx_block(
            fi,
            ts,
            cw,
            w,
            p,
            tile_bo,
            bx,
            by,
            tx_bo,
            chroma_mode,
            uv_tx_size,
            uv_tx_type,
            bsize,
            po,
            skip,
            qidx,
            &ac.data,
            if chroma_mode.is_cfl() {
              IntraParam::Alpha(alpha)
            } else {
              IntraParam::AngleDelta(angle_delta.uv)
            },
            rdo_type,
            need_recon_pixel,
          );
          partition_has_coeff |= has_coeff;
          tx_dist += dist;
        }
      }
    }
  }

  (partition_has_coeff, tx_dist)
}

pub fn write_tx_tree<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &mut TileStateMut<'_, T>,
  cw: &mut ContextWriter, w: &mut dyn Writer, luma_mode: PredictionMode,
  angle_delta_y: i8, tile_bo: TileBlockOffset, bsize: BlockSize,
  tx_size: TxSize, tx_type: TxType, skip: bool, luma_only: bool,
  rdo_type: RDOType, need_recon_pixel: bool,
) -> (bool, ScaledDistortion) {
  if skip {
    return (false, ScaledDistortion::zero());
  }
  let bw = bsize.width_mi() / tx_size.width_mi();
  let bh = bsize.height_mi() / tx_size.height_mi();
  let qidx = get_qidx(fi, ts, cw, tile_bo);

  let PlaneConfig { xdec, ydec, .. } = ts.input.planes[1].cfg;
  let ac = &[0i16; 0];
  let mut partition_has_coeff: bool = false;
  let mut tx_dist = ScaledDistortion::zero();

  ts.qc.update(
    qidx,
    tx_size,
    luma_mode.is_intra(),
    fi.sequence.bit_depth,
    fi.dc_delta_q[0],
    0,
  );

  // TODO: If tx-parition more than only 1-level, this code does not work.
  // It should recursively traverse the tx block that are split recursivelty by calling write_tx_tree(),
  // as defined in https://aomediacodec.github.io/av1-spec/#transform-tree-syntax
  for by in 0..bh {
    for bx in 0..bw {
      let tx_bo = TileBlockOffset(BlockOffset {
        x: tile_bo.0.x + bx * tx_size.width_mi(),
        y: tile_bo.0.y + by * tx_size.height_mi(),
      });

      let po = tx_bo.plane_offset(&ts.input.planes[0].cfg);
      let (has_coeff, dist) = encode_tx_block(
        fi,
        ts,
        cw,
        w,
        0,
        tile_bo,
        0,
        0,
        tx_bo,
        luma_mode,
        tx_size,
        tx_type,
        bsize,
        po,
        skip,
        qidx,
        ac,
        IntraParam::AngleDelta(angle_delta_y),
        rdo_type,
        need_recon_pixel,
      );
      partition_has_coeff |= has_coeff;
      tx_dist += dist;
    }
  }

  if luma_only {
    return (partition_has_coeff, tx_dist);
  };

  let max_tx_size = max_txsize_rect_lookup[bsize as usize];
  debug_assert!(max_tx_size.block_size() <= BlockSize::BLOCK_64X64);
  let uv_tx_size = bsize.largest_chroma_tx_size(xdec, ydec);

  let mut bw_uv = max_tx_size.width_mi() >> xdec;
  let mut bh_uv = max_tx_size.height_mi() >> ydec;

  if (bw_uv == 0 || bh_uv == 0) && has_chroma(tile_bo, bsize, xdec, ydec) {
    bw_uv = 1;
    bh_uv = 1;
  }

  bw_uv /= uv_tx_size.width_mi();
  bh_uv /= uv_tx_size.height_mi();

  if bw_uv > 0 && bh_uv > 0 {
    let uv_tx_type =
      if partition_has_coeff { tx_type } else { TxType::DCT_DCT }; // if inter mode, uv_tx_type == tx_type

    for p in 1..3 {
      ts.qc.update(
        qidx,
        uv_tx_size,
        false,
        fi.sequence.bit_depth,
        fi.dc_delta_q[p],
        fi.ac_delta_q[p],
      );

      for by in 0..bh_uv {
        for bx in 0..bw_uv {
          let tx_bo = TileBlockOffset(BlockOffset {
            x: tile_bo.0.x + ((bx * uv_tx_size.width_mi()) << xdec)
              - (max_tx_size.width_mi() == 1) as usize * xdec,
            y: tile_bo.0.y + ((by * uv_tx_size.height_mi()) << ydec)
              - (max_tx_size.height_mi() == 1) as usize * ydec,
          });

          let mut po = tile_bo.plane_offset(&ts.input.planes[p].cfg);
          po.x += (bx * uv_tx_size.width()) as isize;
          po.y += (by * uv_tx_size.height()) as isize;
          let (has_coeff, dist) = encode_tx_block(
            fi,
            ts,
            cw,
            w,
            p,
            tile_bo,
            bx,
            by,
            tx_bo,
            luma_mode,
            uv_tx_size,
            uv_tx_type,
            bsize,
            po,
            skip,
            qidx,
            ac,
            IntraParam::AngleDelta(angle_delta_y),
            rdo_type,
            need_recon_pixel,
          );
          partition_has_coeff |= has_coeff;
          tx_dist += dist;
        }
      }
    }
  }

  (partition_has_coeff, tx_dist)
}

pub fn encode_block_with_modes<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &mut TileStateMut<'_, T>,
  cw: &mut ContextWriter, w_pre_cdef: &mut dyn Writer,
  w_post_cdef: &mut dyn Writer, bsize: BlockSize, tile_bo: TileBlockOffset,
  mode_decision: &PartitionParameters, rdo_type: RDOType, record_stats: bool,
) {
  let (mode_luma, mode_chroma) =
    (mode_decision.pred_mode_luma, mode_decision.pred_mode_chroma);
  let cfl = mode_decision.pred_cfl_params;
  let ref_frames = mode_decision.ref_frames;
  let mvs = mode_decision.mvs;
  let mut skip = mode_decision.skip;
  let mut cdef_coded = cw.bc.cdef_coded;
  let (tx_size, tx_type) = (mode_decision.tx_size, mode_decision.tx_type);

  // Set correct segmentation ID before encoding and before
  // rdo_tx_size_type().
  cw.bc.blocks.set_segmentation_idx(tile_bo, bsize, mode_decision.sidx);

  let mut mv_stack = ArrayVec::<[CandidateMV; 9]>::new();
  let is_compound = ref_frames[1] != NONE_FRAME;
  let mode_context =
    cw.find_mvrefs(tile_bo, ref_frames, &mut mv_stack, bsize, fi, is_compound);

  if !mode_decision.skip && !mode_decision.has_coeff {
    skip = true;
  }
  cdef_coded = encode_block_pre_cdef(
    &fi.sequence,
    ts,
    cw,
    if cdef_coded { w_post_cdef } else { w_pre_cdef },
    bsize,
    tile_bo,
    skip,
  );
  encode_block_post_cdef(
    fi,
    ts,
    cw,
    if cdef_coded { w_post_cdef } else { w_pre_cdef },
    mode_luma,
    mode_chroma,
    mode_decision.angle_delta,
    ref_frames,
    mvs,
    bsize,
    tile_bo,
    skip,
    cfl,
    tx_size,
    tx_type,
    mode_context,
    &mv_stack,
    rdo_type,
    true,
    record_stats,
  );
}

fn encode_partition_bottomup<T: Pixel, W: Writer>(
  fi: &FrameInvariants<T>, ts: &mut TileStateMut<'_, T>,
  cw: &mut ContextWriter, w_pre_cdef: &mut W, w_post_cdef: &mut W,
  bsize: BlockSize, tile_bo: TileBlockOffset, pmv_idx: usize,
  ref_rd_cost: f64, inter_cfg: &InterConfig,
) -> PartitionGroupParameters {
  let rdo_type = RDOType::PixelDistRealRate;
  let mut rd_cost = std::f64::MAX;
  let mut best_rd = std::f64::MAX;
  let mut rdo_output = PartitionGroupParameters {
    rd_cost,
    part_type: PartitionType::PARTITION_INVALID,
    part_modes: ArrayVec::new(),
  };

  if tile_bo.0.x >= cw.bc.blocks.cols() || tile_bo.0.y >= cw.bc.blocks.rows() {
    return rdo_output;
  }

  let bsw = bsize.width_mi();
  let bsh = bsize.height_mi();
  let is_square = bsize.is_sqr();

  // TODO: Update for 128x128 superblocks
  assert!(fi.partition_range.max <= BlockSize::BLOCK_64X64);
  // Always split if the current partition is too large, i.e. right or bottom tile border
  let must_split = (tile_bo.0.x + bsw as usize > ts.mi_width
    || tile_bo.0.y + bsh as usize > ts.mi_height
    || bsize > fi.partition_range.max)
    && is_square;

  // must_split overrides the minimum partition size when applicable
  let can_split = // FIXME: sub-8x8 inter blocks not supported for non-4:2:0 sampling
    if fi.frame_type.has_inter() &&
    fi.config.chroma_sampling != ChromaSampling::Cs420 &&
    bsize <= BlockSize::BLOCK_8X8 {
    false
  } else {
    (bsize > fi.partition_range.min && is_square) || must_split
  };
  let mut best_partition = PartitionType::PARTITION_INVALID;

  let cw_checkpoint = cw.checkpoint();
  let w_pre_checkpoint = w_pre_cdef.checkpoint();
  let w_post_checkpoint = w_post_cdef.checkpoint();

  // Code the whole block
  if !must_split {
    let cost = if bsize >= BlockSize::BLOCK_8X8 && is_square {
      let w: &mut W = if cw.bc.cdef_coded { w_post_cdef } else { w_pre_cdef };
      let tell = w.tell_frac();
      cw.write_partition(w, tile_bo, PartitionType::PARTITION_NONE, bsize);
      compute_rd_cost(fi, w.tell_frac() - tell, ScaledDistortion::zero())
    } else {
      0.0
    };

    let pmv_inner_idx = if bsize > BlockSize::BLOCK_32X32 {
      0
    } else {
      ((tile_bo.0.x & 32) >> 5) + ((tile_bo.0.y & 32) >> 4) + 1
    };

    let mode_decision = rdo_mode_decision(
      fi,
      ts,
      cw,
      bsize,
      tile_bo,
      (pmv_idx, pmv_inner_idx),
      inter_cfg,
    );

    if !mode_decision.pred_mode_luma.is_intra() {
      // Fill the saved motion structure
      save_block_motion(
        ts,
        mode_decision.bsize,
        mode_decision.bo,
        mode_decision.ref_frames[0].to_index(),
        mode_decision.mvs[0],
      );
    }

    rd_cost = mode_decision.rd_cost + cost;

    best_partition = PartitionType::PARTITION_NONE;
    best_rd = rd_cost;
    rdo_output.part_modes.push(mode_decision.clone());

    if !can_split {
      encode_block_with_modes(
        fi,
        ts,
        cw,
        w_pre_cdef,
        w_post_cdef,
        bsize,
        tile_bo,
        &mode_decision,
        rdo_type,
        true,
      );
    }
  }

  // Test all partition types other than PARTITION_NONE by comparing their RD costs
  if can_split {
    debug_assert!(is_square);

    for &partition in RAV1E_PARTITION_TYPES {
      if partition == PartitionType::PARTITION_NONE {
        continue;
      }
      if fi.sequence.chroma_sampling == ChromaSampling::Cs422
        && partition == PartitionType::PARTITION_VERT
      {
        continue;
      }

      if must_split {
        let cbw = (ts.mi_width - tile_bo.0.x).min(bsw); // clipped block width, i.e. having effective pixels
        let cbh = (ts.mi_height - tile_bo.0.y).min(bsh);
        let mut split_vert = false;
        let mut split_horz = false;
        if cbw == bsw / 2 && cbh == bsh {
          split_vert = true;
        }
        if cbh == bsh / 2 && cbw == bsw {
          split_horz = true;
        }
        if !split_horz && partition == PartitionType::PARTITION_HORZ {
          continue;
        };
        if !split_vert && partition == PartitionType::PARTITION_VERT {
          continue;
        };
      } else if !fi.config.speed_settings.non_square_partition
        && (partition == PartitionType::PARTITION_HORZ
          || partition == PartitionType::PARTITION_VERT)
      {
        continue;
      }
      cw.rollback(&cw_checkpoint);
      w_pre_cdef.rollback(&w_pre_checkpoint);
      w_post_cdef.rollback(&w_post_checkpoint);

      let subsize = bsize.subsize(partition);
      let hbsw = subsize.width_mi(); // Half the block size width in blocks
      let hbsh = subsize.height_mi(); // Half the block size height in blocks
      let mut child_modes = ArrayVec::<[PartitionParameters; 4]>::new();
      rd_cost = 0.0;

      if bsize >= BlockSize::BLOCK_8X8 {
        let w: &mut W =
          if cw.bc.cdef_coded { w_post_cdef } else { w_pre_cdef };
        let tell = w.tell_frac();
        cw.write_partition(w, tile_bo, partition, bsize);
        rd_cost =
          compute_rd_cost(fi, w.tell_frac() - tell, ScaledDistortion::zero());
      }

      let four_partitions = [
        tile_bo,
        TileBlockOffset(BlockOffset {
          x: tile_bo.0.x + hbsw as usize,
          y: tile_bo.0.y,
        }),
        TileBlockOffset(BlockOffset {
          x: tile_bo.0.x,
          y: tile_bo.0.y + hbsh as usize,
        }),
        TileBlockOffset(BlockOffset {
          x: tile_bo.0.x + hbsw as usize,
          y: tile_bo.0.y + hbsh as usize,
        }),
      ];
      let partitions = get_sub_partitions(&four_partitions, partition);
      let mut early_exit = false;

      // If either of horz or vert partition types is being tested,
      // two partitioned rectangles, defined in 'partitions', of the current block
      // is passed to encode_partition_bottomup()
      for offset in partitions {
        let child_rdo_output = encode_partition_bottomup(
          fi,
          ts,
          cw,
          w_pre_cdef,
          w_post_cdef,
          subsize,
          offset,
          pmv_idx,
          best_rd,
          inter_cfg,
        );
        let cost = child_rdo_output.rd_cost;
        assert!(cost >= 0.0);

        if cost != std::f64::MAX {
          rd_cost += cost;
          if fi.enable_early_exit
            && (rd_cost >= best_rd || rd_cost >= ref_rd_cost)
          {
            assert!(cost != std::f64::MAX);
            early_exit = true;
            break;
          } else if partition != PartitionType::PARTITION_SPLIT {
            child_modes.push(child_rdo_output.part_modes[0].clone());
          }
        }
      }

      if !early_exit && rd_cost < best_rd {
        best_rd = rd_cost;
        best_partition = partition;
        if partition != PartitionType::PARTITION_SPLIT {
          assert!(!child_modes.is_empty());
          rdo_output.part_modes = child_modes;
        }
      }
    }

    debug_assert!(best_partition != PartitionType::PARTITION_INVALID);

    // If the best partition is not PARTITION_SPLIT, recode it
    if best_partition != PartitionType::PARTITION_SPLIT {
      assert!(!rdo_output.part_modes.is_empty());

      cw.rollback(&cw_checkpoint);
      w_pre_cdef.rollback(&w_pre_checkpoint);
      w_post_cdef.rollback(&w_post_checkpoint);

      assert!(best_partition != PartitionType::PARTITION_NONE || !must_split);
      let subsize = bsize.subsize(best_partition);

      if bsize >= BlockSize::BLOCK_8X8 {
        let w: &mut W =
          if cw.bc.cdef_coded { w_post_cdef } else { w_pre_cdef };
        cw.write_partition(w, tile_bo, best_partition, bsize);
      }
      for mode in rdo_output.part_modes.clone() {
        assert!(subsize == mode.bsize);

        if !mode.pred_mode_luma.is_intra() {
          save_block_motion(
            ts,
            mode.bsize,
            mode.bo,
            mode.ref_frames[0].to_index(),
            mode.mvs[0],
          );
        }

        // FIXME: redundant block re-encode
        encode_block_with_modes(
          fi,
          ts,
          cw,
          w_pre_cdef,
          w_post_cdef,
          mode.bsize,
          mode.bo,
          &mode,
          rdo_type,
          true,
        );
      }
    }
  }

  assert!(best_partition != PartitionType::PARTITION_INVALID);

  if is_square
    && bsize >= BlockSize::BLOCK_8X8
    && (bsize == BlockSize::BLOCK_8X8
      || best_partition != PartitionType::PARTITION_SPLIT)
  {
    cw.bc.update_partition_context(
      tile_bo,
      bsize.subsize(best_partition),
      bsize,
    );
  }

  rdo_output.rd_cost = best_rd;
  rdo_output.part_type = best_partition;

  if best_partition != PartitionType::PARTITION_NONE {
    rdo_output.part_modes.clear();
  }
  rdo_output
}

fn encode_partition_topdown<T: Pixel, W: Writer>(
  fi: &FrameInvariants<T>, ts: &mut TileStateMut<'_, T>,
  cw: &mut ContextWriter, w_pre_cdef: &mut W, w_post_cdef: &mut W,
  bsize: BlockSize, tile_bo: TileBlockOffset,
  block_output: &Option<PartitionGroupParameters>, pmv_idx: usize,
  inter_cfg: &InterConfig,
) {
  if tile_bo.0.x >= cw.bc.blocks.cols() || tile_bo.0.y >= cw.bc.blocks.rows() {
    return;
  }
  let bsw = bsize.width_mi();
  let bsh = bsize.height_mi();
  let is_square = bsize.is_sqr();
  let rdo_type = RDOType::PixelDistRealRate;

  // TODO: Update for 128x128 superblocks
  assert!(fi.partition_range.max <= BlockSize::BLOCK_64X64);
  // Always split if the current partition is too large, i.e. right or bottom tile border
  let must_split = (tile_bo.0.x + bsw as usize > ts.mi_width
    || tile_bo.0.y + bsh as usize > ts.mi_height
    || bsize > fi.partition_range.max)
    && is_square;

  let mut rdo_output =
    block_output.clone().unwrap_or(PartitionGroupParameters {
      part_type: PartitionType::PARTITION_INVALID,
      rd_cost: std::f64::MAX,
      part_modes: ArrayVec::new(),
    });
  let partition: PartitionType;
  let mut split_vert = false;
  let mut split_horz = false;
  if must_split {
    let cbw = (ts.mi_width - tile_bo.0.x).min(bsw); // clipped block width, i.e. having effective pixels
    let cbh = (ts.mi_height - tile_bo.0.y).min(bsh);

    if cbw == bsw / 2
      && cbh == bsh
      && fi.sequence.chroma_sampling != ChromaSampling::Cs422
    {
      split_vert = true;
    }
    if cbh == bsh / 2 && cbw == bsw {
      split_horz = true;
    }
  }

  if must_split && (!split_vert && !split_horz) {
    // Oversized blocks are split automatically
    partition = PartitionType::PARTITION_SPLIT;
  } else if (must_split || (bsize > fi.partition_range.min && is_square))
    && (
      // FIXME: sub-8x8 inter blocks not supported for non-4:2:0 sampling
      !fi.frame_type.has_inter()
        || fi.config.chroma_sampling == ChromaSampling::Cs420
        || bsize > BlockSize::BLOCK_8X8
    )
  {
    debug_assert!(bsize.is_sqr());
    // Blocks of sizes within the supported range are subjected to a partitioning decision
    let mut partition_types = ArrayVec::<[PartitionType; 3]>::new();
    if must_split {
      partition_types.push(PartitionType::PARTITION_SPLIT);
      if split_horz {
        partition_types.push(PartitionType::PARTITION_HORZ);
      };
      if split_vert {
        partition_types.push(PartitionType::PARTITION_VERT);
      };
    } else {
      partition_types.push(PartitionType::PARTITION_NONE);
      partition_types.push(PartitionType::PARTITION_SPLIT);
    }
    rdo_output = rdo_partition_decision(
      fi,
      ts,
      cw,
      w_pre_cdef,
      w_post_cdef,
      bsize,
      tile_bo,
      &rdo_output,
      pmv_idx,
      &partition_types,
      rdo_type,
      inter_cfg,
    );
    partition = rdo_output.part_type;
  } else {
    // Blocks of sizes below the supported range are encoded directly
    partition = PartitionType::PARTITION_NONE;
  }

  assert!(
    PartitionType::PARTITION_NONE <= partition
      && partition < PartitionType::PARTITION_INVALID
  );

  let subsize = bsize.subsize(partition);

  if bsize >= BlockSize::BLOCK_8X8 && is_square {
    let w: &mut W = if cw.bc.cdef_coded { w_post_cdef } else { w_pre_cdef };
    cw.write_partition(w, tile_bo, partition, bsize);
  }

  match partition {
    PartitionType::PARTITION_NONE => {
      let part_decision = if !rdo_output.part_modes.is_empty() {
        // The optimal prediction mode is known from a previous iteration
        rdo_output.part_modes[0].clone()
      } else {
        let pmv_inner_idx = if bsize > BlockSize::BLOCK_32X32 {
          0
        } else {
          ((tile_bo.0.x & 32) >> 5) + ((tile_bo.0.y & 32) >> 4) + 1
        };

        // Make a prediction mode decision for blocks encoded with no rdo_partition_decision call (e.g. edges)
        rdo_mode_decision(
          fi,
          ts,
          cw,
          bsize,
          tile_bo,
          (pmv_idx, pmv_inner_idx),
          inter_cfg,
        )
      };

      let mut mode_luma = part_decision.pred_mode_luma;
      let mut mode_chroma = part_decision.pred_mode_chroma;

      let cfl = part_decision.pred_cfl_params;
      let skip = part_decision.skip;
      let ref_frames = part_decision.ref_frames;
      let mvs = part_decision.mvs;
      let mut cdef_coded = cw.bc.cdef_coded;

      // Set correct segmentation ID before encoding and before
      // rdo_tx_size_type().
      cw.bc.blocks.set_segmentation_idx(tile_bo, bsize, part_decision.sidx);

      // NOTE: Cannot avoid calling rdo_tx_size_type() here again,
      // because, with top-down partition RDO, the neighnoring contexts
      // of current partition can change, i.e. neighboring partitions can split down more.
      let (tx_size, tx_type) = rdo_tx_size_type(
        fi, ts, cw, bsize, tile_bo, mode_luma, ref_frames, mvs, skip,
      );

      let mut mv_stack = ArrayVec::<[CandidateMV; 9]>::new();
      let is_compound = ref_frames[1] != NONE_FRAME;
      let mode_context = cw.find_mvrefs(
        tile_bo,
        ref_frames,
        &mut mv_stack,
        bsize,
        fi,
        is_compound,
      );

      // TODO: proper remap when is_compound is true
      if !mode_luma.is_intra() {
        if is_compound && mode_luma != PredictionMode::GLOBAL_GLOBALMV {
          let match0 = mv_stack[0].this_mv.row == mvs[0].row
            && mv_stack[0].this_mv.col == mvs[0].col;
          let match1 = mv_stack[0].comp_mv.row == mvs[1].row
            && mv_stack[0].comp_mv.col == mvs[1].col;

          let match2 = mv_stack[1].this_mv.row == mvs[0].row
            && mv_stack[1].this_mv.col == mvs[0].col;
          let match3 = mv_stack[1].comp_mv.row == mvs[1].row
            && mv_stack[1].comp_mv.col == mvs[1].col;

          mode_luma = if match0 && match1 {
            PredictionMode::NEAREST_NEARESTMV
          } else if match2 && match3 {
            PredictionMode::NEAR_NEARMV
          } else if match0 {
            PredictionMode::NEAREST_NEWMV
          } else if match1 {
            PredictionMode::NEW_NEARESTMV
          } else {
            PredictionMode::NEW_NEWMV
          };

          if mode_luma != PredictionMode::NEAREST_NEARESTMV
            && mvs[0].row == 0
            && mvs[0].col == 0
            && mvs[1].row == 0
            && mvs[1].col == 0
          {
            mode_luma = PredictionMode::GLOBAL_GLOBALMV;
          }
          mode_chroma = mode_luma;
        } else if !is_compound && mode_luma != PredictionMode::GLOBALMV {
          mode_luma = PredictionMode::NEWMV;
          for (c, m) in mv_stack.iter().take(4).zip(
            [
              PredictionMode::NEARESTMV,
              PredictionMode::NEAR0MV,
              PredictionMode::NEAR1MV,
              PredictionMode::NEAR2MV,
            ]
            .iter(),
          ) {
            if c.this_mv.row == mvs[0].row && c.this_mv.col == mvs[0].col {
              mode_luma = *m;
            }
          }
          if mode_luma == PredictionMode::NEWMV
            && mvs[0].row == 0
            && mvs[0].col == 0
          {
            mode_luma = if mv_stack.is_empty() {
              PredictionMode::NEARESTMV
            } else if mv_stack.len() == 1 {
              PredictionMode::NEAR0MV
            } else {
              PredictionMode::GLOBALMV
            };
          }
          mode_chroma = mode_luma;
        }

        save_block_motion(
          ts,
          part_decision.bsize,
          part_decision.bo,
          part_decision.ref_frames[0].to_index(),
          part_decision.mvs[0],
        );
      }

      // FIXME: every final block that has gone through the RDO decision process is encoded twice
      cdef_coded = encode_block_pre_cdef(
        &fi.sequence,
        ts,
        cw,
        if cdef_coded { w_post_cdef } else { w_pre_cdef },
        bsize,
        tile_bo,
        skip,
      );
      encode_block_post_cdef(
        fi,
        ts,
        cw,
        if cdef_coded { w_post_cdef } else { w_pre_cdef },
        mode_luma,
        mode_chroma,
        part_decision.angle_delta,
        ref_frames,
        mvs,
        bsize,
        tile_bo,
        skip,
        cfl,
        tx_size,
        tx_type,
        mode_context,
        &mv_stack,
        RDOType::PixelDistRealRate,
        true,
        true,
      );
    }
    PARTITION_SPLIT | PARTITION_HORZ | PARTITION_VERT => {
      if !rdo_output.part_modes.is_empty() {
        // The optimal prediction modes for each split block is known from an rdo_partition_decision() call
        assert!(subsize != BlockSize::BLOCK_INVALID);

        for mode in rdo_output.part_modes {
          use std::iter::{once, FromIterator};
          // Each block is subjected to a new splitting decision
          encode_partition_topdown(
            fi,
            ts,
            cw,
            w_pre_cdef,
            w_post_cdef,
            subsize,
            mode.bo,
            &Some(PartitionGroupParameters {
              rd_cost: mode.rd_cost,
              part_type: PartitionType::PARTITION_NONE,
              part_modes: ArrayVec::from_iter(once(mode)),
            }),
            pmv_idx,
            inter_cfg,
          );
        }
      } else {
        let hbsw = subsize.width_mi(); // Half the block size width in blocks
        let hbsh = subsize.height_mi(); // Half the block size height in blocks
        let four_partitions = [
          tile_bo,
          TileBlockOffset(BlockOffset {
            x: tile_bo.0.x + hbsw as usize,
            y: tile_bo.0.y,
          }),
          TileBlockOffset(BlockOffset {
            x: tile_bo.0.x,
            y: tile_bo.0.y + hbsh as usize,
          }),
          TileBlockOffset(BlockOffset {
            x: tile_bo.0.x + hbsw as usize,
            y: tile_bo.0.y + hbsh as usize,
          }),
        ];
        let partitions = get_sub_partitions(&four_partitions, partition);

        partitions.iter().for_each(|&offset| {
          encode_partition_topdown(
            fi,
            ts,
            cw,
            w_pre_cdef,
            w_post_cdef,
            subsize,
            offset,
            &None,
            pmv_idx,
            inter_cfg,
          );
        });
      }
    }
    _ => unreachable!(),
  }

  if is_square
    && bsize >= BlockSize::BLOCK_8X8
    && (bsize == BlockSize::BLOCK_8X8
      || partition != PartitionType::PARTITION_SPLIT)
  {
    cw.bc.update_partition_context(tile_bo, subsize, bsize);
  }
}

#[inline(always)]
pub(crate) fn build_coarse_pmvs<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>, inter_cfg: &InterConfig,
) -> Vec<[Option<MotionVector>; REF_FRAMES]> {
  assert!(!fi.sequence.use_128x128_superblock);
  if ts.mi_width >= 16 && ts.mi_height >= 16 {
    let mut frame_pmvs = Vec::with_capacity(ts.sb_width * ts.sb_height);
    for sby in 0..ts.sb_height {
      for sbx in 0..ts.sb_width {
        let sbo = TileSuperBlockOffset(SuperBlockOffset { x: sbx, y: sby });
        let bo = sbo.block_offset(0, 0);
        let mut pmvs: [Option<MotionVector>; REF_FRAMES] = [None; REF_FRAMES];
        for i in inter_cfg.allowed_ref_frames().iter().map(|rf| rf.to_index())
        {
          let r = fi.ref_frames[i] as usize;
          if pmvs[r].is_none() {
            pmvs[r] =
              estimate_motion_ss4(fi, ts, BlockSize::BLOCK_64X64, r, bo);
          }
        }
        frame_pmvs.push(pmvs);
      }
    }
    frame_pmvs
  } else {
    // the block use for motion estimation would be smaller than the whole image
    // dynamic allocation: once per frmae
    vec![[None; REF_FRAMES]; ts.sb_width * ts.sb_height]
  }
}

fn get_initial_cdfcontext<T: Pixel>(fi: &FrameInvariants<T>) -> CDFContext {
  let cdf = if fi.primary_ref_frame == PRIMARY_REF_NONE {
    None
  } else {
    let ref_frame_idx = fi.ref_frames[fi.primary_ref_frame as usize] as usize;
    let ref_frame = fi.rec_buffer.frames[ref_frame_idx].as_ref();
    ref_frame.map(|rec| rec.cdfs)
  };

  // return the retrieved instance if any, a new one otherwise
  cdf.unwrap_or_else(|| CDFContext::new(fi.base_q_idx))
}

#[hawktracer(encode_tile_group)]
fn encode_tile_group<T: Pixel>(
  fi: &FrameInvariants<T>, fs: &mut FrameState<T>, inter_cfg: &InterConfig,
) -> Vec<u8> {
  let mut blocks = FrameBlocks::new(fi.w_in_b, fi.h_in_b);
  let ti = &fi.tiling;

  let initial_cdf = get_initial_cdfcontext(fi);
  // dynamic allocation: once per frame
  let mut cdfs = vec![initial_cdf; ti.tile_count()];

  let (raw_tiles, tile_states): (Vec<_>, Vec<_>) = ti
    .tile_iter_mut(fs, &mut blocks)
    .zip(cdfs.iter_mut())
    .collect::<Vec<_>>()
    .into_par_iter()
    .map(|(mut ctx, cdf)| {
      let raw = encode_tile(fi, &mut ctx.ts, cdf, &mut ctx.tb, inter_cfg);
      (raw, ctx.ts)
    })
    .unzip();

  let stats =
    tile_states.into_iter().map(|ts| ts.enc_stats).collect::<Vec<_>>();
  for tile_stats in stats {
    fs.enc_stats += &tile_stats;
  }

  /* Frame deblocking operates over a single large tile wrapping the
   * frame rather than the frame itself so that deblocking is
   * available inside RDO when needed */
  /* TODO: Don't apply if lossless */
  let levels;
  {
    let ts = &mut fs.as_tile_state_mut();
    let rec = &mut ts.rec;
    levels = deblock_filter_optimize(
      fi,
      &rec.as_const(),
      &ts.input.as_tile(),
      &blocks.as_tile_blocks(),
      fi.width,
      fi.height,
    );
  }
  fs.deblock.levels = levels;
  if fs.deblock.levels[0] != 0 || fs.deblock.levels[1] != 0 {
    let ts = &mut fs.as_tile_state_mut();
    let rec = &mut ts.rec;
    deblock_filter_frame(
      ts.deblock,
      rec,
      &blocks.as_tile_blocks(),
      fi.width,
      fi.height,
      fi.sequence.bit_depth,
    );
  }

  if fi.sequence.enable_restoration {
    // Until the loop filters are pipelined, we'll need to keep
    // around a copy of both the pre- and post-cdef frame.
    let pre_cdef_frame = fs.rec.clone();

    /* TODO: Don't apply if lossless */
    if fi.sequence.enable_cdef {
      cdef_filter_tile_group(fi, fs, &mut blocks);
    }
    /* TODO: Don't apply if lossless */
    fs.restoration.lrf_filter_frame(
      Arc::make_mut(&mut fs.rec),
      &pre_cdef_frame,
      fi,
    );
  } else {
    /* TODO: Don't apply if lossless */
    if fi.sequence.enable_cdef {
      cdef_filter_tile_group(fi, fs, &mut blocks);
    }
  }

  let (idx_max, max_len) = raw_tiles
    .iter()
    .map(Vec::len)
    .enumerate()
    .max_by_key(|&(_, len)| len)
    .unwrap();

  if !fi.disable_frame_end_update_cdf {
    // use the biggest tile (in bytes) for CDF update
    fs.context_update_tile_id = idx_max;
    fs.cdfs = cdfs[idx_max];
    fs.cdfs.reset_counts();
  }

  let max_tile_size_bytes = ((max_len.ilog() + 7) / 8) as u32;
  debug_assert!(max_tile_size_bytes > 0 && max_tile_size_bytes <= 4);
  fs.max_tile_size_bytes = max_tile_size_bytes;

  build_raw_tile_group(ti, &raw_tiles, max_tile_size_bytes)
}

fn build_raw_tile_group(
  ti: &TilingInfo, raw_tiles: &[Vec<u8>], max_tile_size_bytes: u32,
) -> Vec<u8> {
  // <https://aomediacodec.github.io/av1-spec/#general-tile-group-obu-syntax>
  let mut raw = Vec::new();
  let mut bw = BitWriter::endian(&mut raw, BigEndian);
  if ti.cols * ti.rows > 1 {
    // tile_start_and_end_present_flag
    bw.write_bit(false).unwrap();
  }
  bw.byte_align().unwrap();
  for (i, raw_tile) in raw_tiles.iter().enumerate() {
    let last = raw_tiles.len() - 1;
    if i != last {
      let tile_size_minus_1 = raw_tile.len() - 1;
      bw.write_le(max_tile_size_bytes, tile_size_minus_1 as u64).unwrap();
    }
    bw.write_bytes(raw_tile).unwrap();
  }
  raw
}

pub(crate) fn build_half_res_pmvs<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &mut TileStateMut<'_, T>,
  tile_sbo: TileSuperBlockOffset,
  tile_pmvs: &[[Option<MotionVector>; REF_FRAMES]],
) -> BlockPmv {
  let estimate_motion_ss2 = if fi.config.speed_settings.diamond_me {
    crate::me::DiamondSearch::estimate_motion_ss2
  } else {
    crate::me::FullSearch::estimate_motion_ss2
  };

  let TileSuperBlockOffset(SuperBlockOffset { x: sbx, y: sby }) = tile_sbo;
  let mut pmvs: BlockPmv = [[None; REF_FRAMES]; 5];

  // The pmvs array stores 5 motion vectors in the following order:
  //
  //       64×64
  // ┌───────┬───────┐
  // │       │       │
  // │   1   │   2   │
  // │       ╵       │
  // ├────── 0 ──────┤
  // │       ╷       │
  // │   3   │   4   │
  // │       │       │
  // └───────┴───────┘
  //
  // That is, 0 is the motion vector for the whole 64×64 block, obtained from
  // the quarter-resolution search, and 1 through 4 are the motion vectors for
  // the 32×32 blocks, obtained below from the half-resolution search.
  //
  // Each of the four half-resolution searches uses three quarter-resolution
  // candidates: one from the current 64×64 block and two from the two
  // immediately adjacent 64×64 blocks.
  //
  //          ┌───────┐
  //          │       │
  //          │   n   │
  //          │       │
  //          └───────┘
  // ┌───────┐┌───┬───┐┌───────┐
  // │       ││ 1 ╵ 2 ││       │
  // │   w   │├── 0 ──┤│   e   │
  // │       ││ 3 ╷ 4 ││       │
  // └───────┘└───┴───┘└───────┘
  //          ┌───────┐
  //          │       │
  //          │   s   │
  //          │       │
  //          └───────┘

  if ts.mi_width >= 8 && ts.mi_height >= 8 {
    for &i in ALL_INTER_REFS.iter() {
      let r = fi.ref_frames[i.to_index()] as usize;
      if pmvs[0][r].is_none() {
        pmvs[0][r] = tile_pmvs[sby * ts.sb_width + sbx][r];
        if let Some(pmv) = pmvs[0][r] {
          let pmv_w = if sbx > 0 {
            tile_pmvs[sby * ts.sb_width + sbx - 1][r]
          } else {
            None
          };
          let pmv_e = if sbx < ts.sb_width - 1 {
            tile_pmvs[sby * ts.sb_width + sbx + 1][r]
          } else {
            None
          };
          let pmv_n = if sby > 0 {
            tile_pmvs[sby * ts.sb_width + sbx - ts.sb_width][r]
          } else {
            None
          };
          let pmv_s = if sby < ts.sb_height - 1 {
            tile_pmvs[sby * ts.sb_width + sbx + ts.sb_width][r]
          } else {
            None
          };

          assert!(!fi.sequence.use_128x128_superblock);
          pmvs[1][r] = estimate_motion_ss2(
            fi,
            ts,
            BlockSize::BLOCK_32X32,
            tile_sbo.block_offset(0, 0),
            &[Some(pmv), pmv_w, pmv_n],
            i,
          );
          pmvs[2][r] = estimate_motion_ss2(
            fi,
            ts,
            BlockSize::BLOCK_32X32,
            tile_sbo.block_offset(8, 0),
            &[Some(pmv), pmv_e, pmv_n],
            i,
          );
          pmvs[3][r] = estimate_motion_ss2(
            fi,
            ts,
            BlockSize::BLOCK_32X32,
            tile_sbo.block_offset(0, 8),
            &[Some(pmv), pmv_w, pmv_s],
            i,
          );
          pmvs[4][r] = estimate_motion_ss2(
            fi,
            ts,
            BlockSize::BLOCK_32X32,
            tile_sbo.block_offset(8, 8),
            &[Some(pmv), pmv_e, pmv_s],
            i,
          );

          if let Some(mv) = pmvs[1][r] {
            save_block_motion(
              ts,
              BlockSize::BLOCK_32X32,
              tile_sbo.block_offset(0, 0),
              i.to_index(),
              mv,
            );
          }
          if let Some(mv) = pmvs[2][r] {
            save_block_motion(
              ts,
              BlockSize::BLOCK_32X32,
              tile_sbo.block_offset(8, 0),
              i.to_index(),
              mv,
            );
          }
          if let Some(mv) = pmvs[3][r] {
            save_block_motion(
              ts,
              BlockSize::BLOCK_32X32,
              tile_sbo.block_offset(0, 8),
              i.to_index(),
              mv,
            );
          }
          if let Some(mv) = pmvs[4][r] {
            save_block_motion(
              ts,
              BlockSize::BLOCK_32X32,
              tile_sbo.block_offset(8, 8),
              i.to_index(),
              mv,
            );
          }
        }
      }
    }
  }

  pmvs
}

pub(crate) fn build_full_res_pmvs<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &mut TileStateMut<'_, T>,
  tile_sbo: TileSuperBlockOffset,
  half_res_pmvs: &[[[Option<MotionVector>; REF_FRAMES]; 5]],
) {
  let estimate_motion = if fi.config.speed_settings.diamond_me {
    crate::me::DiamondSearch::estimate_motion
  } else {
    crate::me::FullSearch::estimate_motion
  };

  let TileSuperBlockOffset(SuperBlockOffset { x: sbx, y: sby }) = tile_sbo;
  let mut pmvs: [Option<MotionVector>; REF_FRAMES] = [None; REF_FRAMES];
  let half_res_pmvs_this_block = half_res_pmvs[sby * ts.sb_width + sbx];

  if ts.mi_width >= 8 && ts.mi_height >= 8 {
    for &i in ALL_INTER_REFS.iter() {
      let r = fi.ref_frames[i.to_index()] as usize;
      if pmvs[r].is_none() {
        pmvs[r] = half_res_pmvs_this_block[0][r];
        if let Some(pmv) = pmvs[r] {
          assert!(!fi.sequence.use_128x128_superblock);

          let pmvs_w = if sbx > 0 {
            half_res_pmvs[sby * ts.sb_width + sbx - 1]
          } else {
            [[None; REF_FRAMES]; 5]
          };
          let pmvs_e = if sbx < ts.sb_width - 1 {
            half_res_pmvs[sby * ts.sb_width + sbx + 1]
          } else {
            [[None; REF_FRAMES]; 5]
          };
          let pmvs_n = if sby > 0 {
            half_res_pmvs[sby * ts.sb_width + sbx - ts.sb_width]
          } else {
            [[None; REF_FRAMES]; 5]
          };
          let pmvs_s = if sby < ts.sb_height - 1 {
            half_res_pmvs[sby * ts.sb_width + sbx + ts.sb_width]
          } else {
            [[None; REF_FRAMES]; 5]
          };

          for y in 0..4 {
            for x in 0..4 {
              let bo = tile_sbo.block_offset(x * 4, y * 4);

              // We start from half_res_pmvs which include five motion vectors
              // for a 64×64 block, as described in build_half_res_pmvs. In
              // this loop we go one level down and search motion vectors for
              // 16×16 blocks using the full-resolution frames:
              //
              //               64×64
              // ┏━━━━━━━┯━━━━━━━┳━━━━━━━┯━━━━━━━┓
              // ┃       │       ┃       │       ┃
              // ┃       │       ┃       │       ┃
              // ┃       ╵       ┃       ╵       ┃
              // ┠────── 1 ──────╂────── 2 ──────┨
              // ┃       ╷       ┃       ╷       ┃
              // ┃       │       ┃       │       ┃
              // ┃       │       ╹       │       ┃
              // ┣━━━━━━━┿━━━━━━ 0 ━━━━━━┿━━━━━━━┫
              // ┃       │       ╻       │       ┃
              // ┃       │       ┃       │       ┃
              // ┃       ╵       ┃       ╵       ┃
              // ┠────── 3 ──────╂────── 4 ──────┨
              // ┃       ╷       ┃       ╷       ┃
              // ┃       │       ┃       │       ┃
              // ┃       │       ┃       │       ┃
              // ┗━━━━━━━┷━━━━━━━┻━━━━━━━┷━━━━━━━┛
              //
              // Each search receives all covering and adjacent motion vectors
              // as candidates. Additionally, the middle two rows of blocks
              // also receive the 32×32 motion vectors from neighboring 64×64
              // blocks, even though not directly adjacent; same with middle
              // two columns.
              let covering_half_res = match (x, y) {
                (0..=1, 0..=1) => (half_res_pmvs_this_block[1][r]),
                (2..=3, 0..=1) => (half_res_pmvs_this_block[2][r]),
                (0..=1, 2..=3) => (half_res_pmvs_this_block[3][r]),
                (2..=3, 2..=3) => (half_res_pmvs_this_block[4][r]),
                _ => unreachable!(),
              };

              let (vertical_candidate_1, vertical_candidate_2) = match (x, y) {
                (0..=1, 0) => (pmvs_n[0][r], pmvs_n[3][r]),
                (2..=3, 0) => (pmvs_n[0][r], pmvs_n[4][r]),
                (0..=1, 1) => (pmvs_n[3][r], half_res_pmvs_this_block[3][r]),
                (2..=3, 1) => (pmvs_n[4][r], half_res_pmvs_this_block[4][r]),
                (0..=1, 2) => (pmvs_s[1][r], half_res_pmvs_this_block[1][r]),
                (2..=3, 2) => (pmvs_s[2][r], half_res_pmvs_this_block[2][r]),
                (0..=1, 3) => (pmvs_s[0][r], pmvs_s[1][r]),
                (2..=3, 3) => (pmvs_s[0][r], pmvs_s[2][r]),
                _ => unreachable!(),
              };

              let (horizontal_candidate_1, horizontal_candidate_2) =
                match (x, y) {
                  (0, 0..=1) => (pmvs_w[0][r], pmvs_w[2][r]),
                  (0, 2..=3) => (pmvs_w[0][r], pmvs_w[4][r]),
                  (1, 0..=1) => (pmvs_w[2][r], half_res_pmvs_this_block[2][r]),
                  (1, 2..=3) => (pmvs_w[4][r], half_res_pmvs_this_block[4][r]),
                  (2, 0..=1) => (pmvs_e[1][r], half_res_pmvs_this_block[1][r]),
                  (2, 2..=3) => (pmvs_e[3][r], half_res_pmvs_this_block[3][r]),
                  (3, 0..=1) => (pmvs_e[0][r], pmvs_e[2][r]),
                  (3, 2..=3) => (pmvs_e[0][r], pmvs_e[4][r]),
                  _ => unreachable!(),
                };

              if let Some(mv) = estimate_motion(
                fi,
                ts,
                BlockSize::BLOCK_16X16,
                bo,
                &[
                  Some(pmv),
                  covering_half_res,
                  vertical_candidate_1,
                  vertical_candidate_2,
                  horizontal_candidate_1,
                  horizontal_candidate_2,
                ],
                i,
              ) {
                save_block_motion(
                  ts,
                  BlockSize::BLOCK_16X16,
                  bo,
                  i.to_index(),
                  mv,
                );
              }
            }
          }
        }
      }
    }
  }
}

pub struct SBSQueueEntry {
  pub sbo: TileSuperBlockOffset,
  pub lru_index: [i32; PLANES],
  pub cdef_coded: bool,
  pub w_pre_cdef: WriterBase<WriterRecorder>,
  pub w_post_cdef: WriterBase<WriterRecorder>,
}

fn check_lf_queue<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &mut TileStateMut<'_, T>,
  cw: &mut ContextWriter, w: &mut WriterBase<WriterEncoder>,
  sbs_q: &mut VecDeque<SBSQueueEntry>, last_lru_ready: &mut [i32; 3],
  last_lru_rdoed: &mut [i32; 3], last_lru_coded: &mut [i32; 3],
  deblock_p: bool,
) {
  let mut check_queue = true;

  // Walk queue from the head, see if anything is ready for RDO and flush
  while check_queue {
    if let Some(qe) = sbs_q.front_mut() {
      for pli in 0..PLANES {
        if qe.lru_index[pli] > last_lru_ready[pli] {
          check_queue = false;
          break;
        }
      }
      if check_queue {
        // yes, this entry is ready
        if qe.cdef_coded || fi.sequence.enable_restoration {
          // only RDO once for a given LRU.

          // One quirk worth noting: LRUs in different planes
          // may be different sizes; eg, one chroma LRU may
          // cover four luma LRUs. However, we won't get here
          // until all are ready for RDO because the smaller
          // ones all fit inside the biggest, and the biggest
          // doesn't trigger until everything is done.

          // RDO happens on all LRUs within the confines of the
          // biggest, all together.  If any of this SB's planes'
          // LRUs are RDOed, in actuality they all are.

          // SBs tagged with a lru index of -1 are ignored in
          // LRU coding/rdoing decisions (but still need to rdo
          // for cdef).
          let mut already_rdoed = false;
          for pli in 0..PLANES {
            if qe.lru_index[pli] != -1
              && qe.lru_index[pli] <= last_lru_rdoed[pli]
            {
              already_rdoed = true;
              break;
            }
          }
          if !already_rdoed {
            rdo_loop_decision(qe.sbo, fi, ts, cw, w, deblock_p);
            for pli in 0..PLANES {
              if qe.lru_index[pli] != -1
                && last_lru_rdoed[pli] < qe.lru_index[pli]
              {
                last_lru_rdoed[pli] = qe.lru_index[pli];
              }
            }
          }
        }
        // write LRF information
        if fi.sequence.enable_restoration {
          for pli in 0..PLANES {
            if qe.lru_index[pli] != -1
              && last_lru_coded[pli] < qe.lru_index[pli]
            {
              last_lru_coded[pli] = qe.lru_index[pli];
              cw.write_lrf(w, fi, &mut ts.restoration, qe.sbo, pli);
            }
          }
        }
        // Now that loop restoration is coded, we can replay the initial block bits
        qe.w_pre_cdef.replay(w);
        // Now code CDEF into the middle of the block
        if qe.cdef_coded {
          let cdef_index = cw.bc.blocks.get_cdef(qe.sbo);
          cw.write_cdef(w, cdef_index, fi.cdef_bits);
          // Code queued symbols that come after the CDEF index
          qe.w_post_cdef.replay(w);
        }
        sbs_q.pop_front();
      }
    } else {
      check_queue = false;
    }
  }
}

#[hawktracer(encode_tile)]
fn encode_tile<'a, T: Pixel>(
  fi: &FrameInvariants<T>, ts: &mut TileStateMut<'_, T>,
  fc: &'a mut CDFContext, blocks: &'a mut TileBlocksMut<'a>,
  inter_cfg: &InterConfig,
) -> Vec<u8> {
  let mut w = WriterEncoder::new();

  let bc = BlockContext::new(blocks);
  let mut cw = ContextWriter::new(fc, bc);
  let mut sbs_q: VecDeque<SBSQueueEntry> = VecDeque::new();
  let mut last_lru_ready = [-1; 3];
  let mut last_lru_rdoed = [-1; 3];
  let mut last_lru_coded = [-1; 3];

  // main loop
  for sby in 0..ts.sb_height {
    cw.bc.reset_left_contexts();

    for sbx in 0..ts.sb_width {
      let tile_sbo = TileSuperBlockOffset(SuperBlockOffset { x: sbx, y: sby });
      let mut sbs_qe = SBSQueueEntry {
        sbo: tile_sbo,
        lru_index: [-1; PLANES],
        cdef_coded: false,
        w_pre_cdef: WriterRecorder::new(),
        w_post_cdef: WriterRecorder::new(),
      };

      let tile_bo = tile_sbo.block_offset(0, 0);
      cw.bc.cdef_coded = false;
      cw.bc.code_deltas = fi.delta_q_present;

      let pmv_idx = sbx + sby * ts.sb_width;

      // Encode SuperBlock
      if fi.config.speed_settings.encode_bottomup {
        encode_partition_bottomup(
          fi,
          ts,
          &mut cw,
          &mut sbs_qe.w_pre_cdef,
          &mut sbs_qe.w_post_cdef,
          BlockSize::BLOCK_64X64,
          tile_bo,
          pmv_idx,
          std::f64::MAX,
          inter_cfg,
        );
      } else {
        encode_partition_topdown(
          fi,
          ts,
          &mut cw,
          &mut sbs_qe.w_pre_cdef,
          &mut sbs_qe.w_post_cdef,
          BlockSize::BLOCK_64X64,
          tile_bo,
          &None,
          pmv_idx,
          inter_cfg,
        );
      }

      {
        let mut check_queue = false;
        // queue our superblock for when the LRU is complete
        sbs_qe.cdef_coded = cw.bc.cdef_coded;
        for pli in 0..PLANES {
          if let Some((lru_x, lru_y)) =
            ts.restoration.planes[pli].restoration_unit_index(tile_sbo, false)
          {
            let lru_index = ts.restoration.planes[pli]
              .restoration_unit_countable(lru_x, lru_y)
              as i32;
            sbs_qe.lru_index[pli] = lru_index;
            if ts.restoration.planes[pli]
              .restoration_unit_last_sb_for_rdo(fi, ts.sbo, tile_sbo)
            {
              last_lru_ready[pli] = lru_index;
              check_queue = true;
            }
          } else {
            // we're likely in an area stretched into a new tile
            // tag this SB to be ignored in LRU decisions
            sbs_qe.lru_index[pli] = -1;
            check_queue = true;
          }
        }
        sbs_q.push_back(sbs_qe);

        if check_queue && !fi.sequence.enable_delayed_loopfilter_rdo {
          check_lf_queue(
            fi,
            ts,
            &mut cw,
            &mut w,
            &mut sbs_q,
            &mut last_lru_ready,
            &mut last_lru_rdoed,
            &mut last_lru_coded,
            true,
          );
        }
      }
    }
  }

  if fi.sequence.enable_delayed_loopfilter_rdo {
    // Solve deblocking for just this tile
    /* TODO: Don't apply if lossless */
    let deblock_levels = deblock_filter_optimize(
      fi,
      &ts.rec.as_const(),
      &ts.input_tile,
      &cw.bc.blocks.as_const(),
      fi.width,
      fi.height,
    );

    if deblock_levels[0] != 0 || deblock_levels[1] != 0 {
      // copy reconstruction to a temp frame to restore it later
      let rec_copy = Frame {
        planes: [
          ts.rec.planes[0].scratch_copy(),
          ts.rec.planes[1].scratch_copy(),
          ts.rec.planes[2].scratch_copy(),
        ],
      };

      // copy ts.deblock because we need to set some of our own values here
      let mut deblock_copy = *ts.deblock;
      deblock_copy.levels = deblock_levels;

      // temporarily deblock the reference
      deblock_filter_frame(
        &deblock_copy,
        &mut ts.rec,
        &cw.bc.blocks.as_const(),
        fi.width,
        fi.height,
        fi.sequence.bit_depth,
      );

      // rdo lf and write
      check_lf_queue(
        fi,
        ts,
        &mut cw,
        &mut w,
        &mut sbs_q,
        &mut last_lru_ready,
        &mut last_lru_rdoed,
        &mut last_lru_coded,
        false,
      );

      // copy original reference back in
      for pli in 0..PLANES {
        let dst = &mut ts.rec.planes[pli];
        let src = &rec_copy.planes[pli];
        for (dst_row, src_row) in dst.rows_iter_mut().zip(src.rows_iter()) {
          for (out, input) in dst_row.iter_mut().zip(src_row) {
            *out = *input;
          }
        }
      }
    } else {
      // rdo lf and write
      check_lf_queue(
        fi,
        ts,
        &mut cw,
        &mut w,
        &mut sbs_q,
        &mut last_lru_ready,
        &mut last_lru_rdoed,
        &mut last_lru_coded,
        false,
      );
    }
  }

  assert!(
    sbs_q.is_empty(),
    "Superblock queue not empty in tile at offset {}:{}",
    ts.sbo.0.x,
    ts.sbo.0.y
  );
  w.done()
}

#[allow(unused)]
fn write_tile_group_header(tile_start_and_end_present_flag: bool) -> Vec<u8> {
  let mut buf = Vec::new();
  {
    let mut bw = BitWriter::endian(&mut buf, BigEndian);
    bw.write_bit(tile_start_and_end_present_flag).unwrap();
    bw.byte_align().unwrap();
  }
  buf
}

// Write a packet containing only the placeholder that tells the decoder
// to present the already decoded frame present at `frame_to_show_map_idx`
//
// See `av1-spec` Section 6.8.2 and 7.18.
pub fn encode_show_existing_frame<T: Pixel>(
  fi: &FrameInvariants<T>, fs: &mut FrameState<T>, inter_cfg: &InterConfig,
) -> Vec<u8> {
  debug_assert!(fi.show_existing_frame);
  let mut packet = Vec::new();

  write_obus(&mut packet, fi, fs, inter_cfg).unwrap();
  let map_idx = fi.frame_to_show_map_idx as usize;
  if let Some(ref rec) = fi.rec_buffer.frames[map_idx] {
    let fs_rec = Arc::make_mut(&mut fs.rec);
    for p in 0..3 {
      fs_rec.planes[p].data.copy_from_slice(&rec.frame.planes[p].data);
    }
  }
  packet
}

fn get_initial_segmentation<T: Pixel>(
  fi: &FrameInvariants<T>,
) -> SegmentationState {
  let segmentation = if fi.primary_ref_frame == PRIMARY_REF_NONE {
    None
  } else {
    let ref_frame_idx = fi.ref_frames[fi.primary_ref_frame as usize] as usize;
    let ref_frame = fi.rec_buffer.frames[ref_frame_idx].as_ref();
    ref_frame.map(|rec| rec.segmentation)
  };

  // return the retrieved instance if any, a new one otherwise
  segmentation.unwrap_or_default()
}

pub fn encode_frame<T: Pixel>(
  fi: &FrameInvariants<T>, fs: &mut FrameState<T>, inter_cfg: &InterConfig,
) -> Vec<u8> {
  debug_assert!(!fi.show_existing_frame);
  debug_assert!(!fi.invalid);
  let mut packet = Vec::new();

  if fi.enable_segmentation {
    fs.segmentation = get_initial_segmentation(fi);
    segmentation_optimize(fi, fs);
  }
  let tile_group = encode_tile_group(fi, fs, inter_cfg);

  write_obus(&mut packet, fi, fs, inter_cfg).unwrap();
  let mut buf1 = Vec::new();
  {
    let mut bw1 = BitWriter::endian(&mut buf1, BigEndian);
    bw1.write_obu_header(ObuType::OBU_TILE_GROUP, 0).unwrap();
  }
  packet.write_all(&buf1).unwrap();
  buf1.clear();

  {
    let mut bw1 = BitWriter::endian(&mut buf1, BigEndian);
    bw1.write_uleb128(tile_group.len() as u64).unwrap();
  }
  packet.write_all(&buf1).unwrap();
  buf1.clear();

  packet.write_all(&tile_group).unwrap();
  packet
}

pub fn update_rec_buffer<T: Pixel>(
  output_frameno: u64, fi: &mut FrameInvariants<T>, fs: &FrameState<T>,
) {
  let rfs = Arc::new(ReferenceFrame {
    order_hint: fi.order_hint,
    frame: fs.rec.clone(),
    input_hres: fs.input_hres.clone(),
    input_qres: fs.input_qres.clone(),
    cdfs: fs.cdfs,
    frame_mvs: fs.frame_mvs.clone(),
    output_frameno,
    segmentation: fs.segmentation,
  });
  for i in 0..(REF_FRAMES as usize) {
    if (fi.refresh_frame_flags & (1 << i)) != 0 {
      fi.rec_buffer.frames[i] = Some(Arc::clone(&rfs));
      fi.rec_buffer.deblock[i] = fs.deblock;
    }
  }
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn check_partition_types_order() {
    assert_eq!(
      RAV1E_PARTITION_TYPES[RAV1E_PARTITION_TYPES.len() - 1],
      PartitionType::PARTITION_SPLIT
    );
  }
}
