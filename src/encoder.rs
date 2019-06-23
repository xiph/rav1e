// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::api::*;
use crate::cdef::*;
use crate::context::*;
use crate::deblock::*;
use crate::ec::*;
use crate::lrf::*;
use crate::mc::MotionVector;
use crate::me::*;
use crate::partition::*;
use crate::predict::PredictionMode;
use crate::frame::*;
use crate::quantize::*;
use crate::rate::QuantizerParameters;
use crate::rate::FRAME_SUBTYPE_I;
use crate::rate::FRAME_SUBTYPE_P;
use crate::rdo::*;
use crate::segmentation::*;
use crate::tiling::*;
use crate::transform::*;
use crate::util::*;
use crate::partition::PartitionType::*;
use crate::partition::RefType::*;
use crate::header::*;

use arg_enum_proc_macro::ArgEnum;
use bitstream_io::{BitWriter, BigEndian};
use bincode::{serialize, deserialize};
use rayon::iter::*;
use std;
use std::{fmt, io, mem};
use std::io::Write;
use std::io::Read;
use std::sync::Arc;
use std::fs::File;
use arrayvec::*;

pub static TEMPORAL_DELIMITER: [u8; 2] = [0x12, 0x00];

const MAX_NUM_TEMPORAL_LAYERS: usize = 8;
const MAX_NUM_SPATIAL_LAYERS: usize = 4;
const MAX_NUM_OPERATING_POINTS: usize = MAX_NUM_TEMPORAL_LAYERS * MAX_NUM_SPATIAL_LAYERS;

#[derive(Debug, Clone)]
pub struct ReferenceFrame<T: Pixel> {
  pub order_hint: u32,
  pub frame: Frame<T>,
  pub input_hres: Plane<T>,
  pub input_qres: Plane<T>,
  pub cdfs: CDFContext,
  pub frame_mvs: Vec<FrameMotionVectors>,
}

#[derive(Debug, Clone, Default)]
pub struct ReferenceFramesSet<T: Pixel> {
  pub frames: [Option<Arc<ReferenceFrame<T>>>; (REF_FRAMES as usize)],
  pub deblock: [DeblockState; (REF_FRAMES as usize)]
}

impl<T: Pixel> ReferenceFramesSet<T> {
  pub fn new() -> Self {
    Self {
      frames: Default::default(),
      deblock: Default::default()
    }
  }
}



#[derive(ArgEnum, Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub enum Tune {
  Psnr,
  Psychovisual
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
  pub force_screen_content_tools: u32,  // 0 - force off
  // 1 - force on
  // 2 - adaptive
  pub force_integer_mv: u32,      // 0 - Not to force. MV can be in 1/4 or 1/8
  // 1 - force to integer
  // 2 - adaptive
  pub still_picture: bool,               // Video is a single frame still picture
  pub reduced_still_picture_hdr: bool,   // Use reduced header for still picture
  pub enable_filter_intra: bool,         // enables/disables filter_intra
  pub enable_intra_edge_filter: bool,    // enables/disables corner/edge/upsampling
  pub enable_interintra_compound: bool,  // enables/disables interintra_compound
  pub enable_masked_compound: bool,      // enables/disables masked compound
  pub enable_dual_filter: bool,         // 0 - disable dual interpolation filter
  // 1 - enable vert/horiz filter selection
  pub enable_order_hint: bool,     // 0 - disable order hint, and related tools
  // jnt_comp, ref_frame_mvs, frame_sign_bias
  // if 0, enable_jnt_comp and
  // enable_ref_frame_mvs must be set zs 0.
  pub enable_jnt_comp: bool,        // 0 - disable joint compound modes
  // 1 - enable it
  pub enable_ref_frame_mvs: bool,  // 0 - disable ref frame mvs
  // 1 - enable it
  pub enable_warped_motion: bool,   // 0 - disable warped motion for sequence
  // 1 - enable it for the sequence
  pub enable_superres: bool,// 0 - Disable superres for the sequence, and disable
  //     transmitting per-frame superres enabled flag.
  // 1 - Enable superres for the sequence, and also
  //     enable per-frame flag to denote if superres is
  //     enabled for that frame.
  pub enable_cdef: bool,         // To turn on/off CDEF
  pub enable_restoration: bool,  // To turn on/off loop restoration
  pub operating_points_cnt_minus_1: usize,
  pub operating_point_idc: [u16; MAX_NUM_OPERATING_POINTS],
  pub display_model_info_present_flag: bool,
  pub decoder_model_info_present_flag: bool,
  pub level: [[usize; 2]; MAX_NUM_OPERATING_POINTS],	// minor, major
  pub tier: [usize; MAX_NUM_OPERATING_POINTS],  // seq_tier in the spec. One bit: 0
  // or 1.
  pub film_grain_params_present: bool,
  pub separate_uv_delta_q: bool,
}

impl Sequence {
  pub fn new(config: &EncoderConfig) -> Sequence {
    let width_bits = 32 - (config.width as u32).leading_zeros();
    let height_bits = 32 - (config.height as u32).leading_zeros();
    assert!(width_bits <= 16);
    assert!(height_bits <= 16);

    let profile = if config.bit_depth == 12 ||
      config.chroma_sampling == ChromaSampling::Cs422 {
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
      level[i][0] = 1;    // minor
      level[i][1] = 2;    // major
      tier[i] = 0;
    }

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
      force_screen_content_tools: 0,
      force_integer_mv: 2,
      still_picture: false,
      reduced_still_picture_hdr: false,
      enable_filter_intra: false,
      enable_intra_edge_filter: false,
      enable_interintra_compound: false,
      enable_masked_compound: false,
      enable_dual_filter: false,
      enable_order_hint: true,
      enable_jnt_comp: false,
      enable_ref_frame_mvs: false,
      enable_warped_motion: false,
      enable_superres: false,
      enable_cdef: config.speed_settings.cdef && config.chroma_sampling != ChromaSampling::Cs422,
      enable_restoration: config.chroma_sampling != ChromaSampling::Cs422 &&
        config.chroma_sampling != ChromaSampling::Cs444, // FIXME: not working yet
      operating_points_cnt_minus_1: 0,
      operating_point_idc,
      display_model_info_present_flag: false,
      decoder_model_info_present_flag: false,
      level,
      tier,
      film_grain_params_present: false,
      separate_uv_delta_q: true,
    }
  }

  pub fn get_relative_dist(&self, a: u32, b: u32) -> i32 {
    let diff = a as i32 - b as i32;
    let m = 1 << self.order_hint_bits_minus_1;
    (diff & (m - 1)) - (diff & m)
  }

  pub fn get_skip_mode_allowed<T: Pixel>(&self, fi: &FrameInvariants<T>, reference_select: bool) -> bool {
    if fi.intra_only || !reference_select || !self.enable_order_hint {
      return false;
    }

    let mut forward_idx: isize = -1;
    let mut backward_idx: isize = -1;
    let mut forward_hint = 0;
    let mut backward_hint = 0;

    for i in 0..INTER_REFS_PER_FRAME {
      if let Some(ref rec) = fi.rec_buffer.frames[fi.ref_frames[i] as usize] {
        let ref_hint = rec.order_hint;

        if self.get_relative_dist(ref_hint, fi.order_hint) < 0 {
          if forward_idx < 0 || self.get_relative_dist(ref_hint, forward_hint) > 0 {
            forward_idx = i as isize;
            forward_hint = ref_hint;
          }
        } else if self.get_relative_dist(ref_hint, fi.order_hint) > 0 &&
          (backward_idx < 0 || self.get_relative_dist(ref_hint, backward_hint) > 0) {
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

      for i in 0..INTER_REFS_PER_FRAME {
        if let Some(ref rec) = fi.rec_buffer.frames[fi.ref_frames[i] as usize] {
          let ref_hint = rec.order_hint;

          if self.get_relative_dist(ref_hint, forward_hint) < 0 &&
            (second_forward_idx < 0 || self.get_relative_dist(ref_hint, second_forward_hint) > 0) {
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
  pub fn sb_size_log2(&self) -> usize {
    if self.use_128x128_superblock { 7 } else { 6 }
  }

  #[inline(always)]
  pub fn sb_size(&self) -> usize {
    1 << self.sb_size_log2()
  }
}

#[derive(Debug)]
pub struct FrameState<T: Pixel> {
  pub sb_size_log2: usize,
  pub input: Arc<Frame<T>>,
  pub input_hres: Plane<T>, // half-resolution version of input luma
  pub input_qres: Plane<T>, // quarter-resolution version of input luma
  pub rec: Frame<T>,
  pub cdfs: CDFContext,
  pub context_update_tile_id: usize, // tile id used for the CDFontext
  pub max_tile_size_bytes: u32,
  pub deblock: DeblockState,
  pub segmentation: SegmentationState,
  pub restoration: RestorationState,
  pub frame_mvs: Vec<FrameMotionVectors>,
  pub t: RDOTracker,
}

impl<T: Pixel> FrameState<T> {
  pub fn new(fi: &FrameInvariants<T>) -> Self {
    // TODO(negge): Use fi.cfg.chroma_sampling when we store VideoDetails in FrameInvariants
    FrameState::new_with_frame(fi, Arc::new(Frame::new(
      fi.width, fi.height, fi.sequence.chroma_sampling)))
  }

  pub fn new_with_frame(fi: &FrameInvariants<T>, frame: Arc<Frame<T>>) -> Self {
    let rs = RestorationState::new(fi, &frame);
    let luma_width = frame.planes[0].cfg.width;
    let luma_height = frame.planes[0].cfg.height;
    let luma_padding_x = frame.planes[0].cfg.xpad;
    let luma_padding_y = frame.planes[0].cfg.ypad;

    Self {
      sb_size_log2: fi.sb_size_log2(),
      input: frame,
      input_hres: Plane::new(luma_width / 2, luma_height / 2, 1, 1, luma_padding_x / 2, luma_padding_y / 2),
      input_qres: Plane::new(luma_width / 4, luma_height / 4, 2, 2, luma_padding_x / 4, luma_padding_y / 4),
      rec: Frame::new(luma_width, luma_height, fi.sequence.chroma_sampling),
      cdfs: CDFContext::new(0),
      context_update_tile_id: 0,
      max_tile_size_bytes: 0,
      deblock: Default::default(),
      segmentation: Default::default(),
      restoration: rs,
      frame_mvs: {
        let mut vec = Vec::with_capacity(REF_FRAMES);
        for _ in 0..REF_FRAMES {
          vec.push(FrameMotionVectors::new(fi.w_in_b, fi.h_in_b));
        }
        vec
      },
      t: RDOTracker::new()
    }
  }

  #[inline(always)]
  pub fn as_tile_state_mut(&mut self) -> TileStateMut<'_, T> {
    let PlaneConfig { width, height, .. } = self.rec.planes[0].cfg;
    let sbo_0 = SuperBlockOffset { x: 0, y: 0 };
    TileStateMut::new(self, sbo_0, self.sb_size_log2, width, height)
  }
}

#[derive(Copy, Clone, Debug)]
pub struct DeblockState {
  pub levels: [u8; PLANES+1],  // Y vertical edges, Y horizontal, U, V
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
      levels: [8,8,4,4],
      sharpness: 0,
      deltas_enabled: false, // requires delta_q_enabled
      delta_updates_enabled: false,
      ref_deltas: [1, 0, 0, 0, 0, -1, -1, -1],
      mode_deltas: [0, 0],
      block_deltas_enabled: false,
      block_delta_shift: 0,
      block_delta_multi: false
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
      preskip: true,
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
  pub min_partition_size: BlockSize,
  pub globalmv_transformation_type: [GlobalMVMode; INTER_REFS_PER_FRAME],
  pub num_tg: usize,
  pub large_scale_tile: bool,
  pub disable_cdf_update: bool,
  pub allow_screen_content_tools: u32,
  pub force_integer_mv: u32,
  pub primary_ref_frame: u32,
  pub refresh_frame_flags: u32,  // a bitmask that specifies which
  // reference frame slots will be updated with the current frame
  // after it is decoded.
  pub allow_intrabc: bool,
  pub use_ref_frame_mvs: bool,
  pub is_filter_switchable: bool,
  pub is_motion_mode_switchable: bool,
  pub disable_frame_end_update_cdf: bool,
  pub allow_warped_motion: bool,
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
  pub me_range_scale: u8,
  pub use_tx_domain_distortion: bool,
  pub use_tx_domain_rate: bool,
  pub idx_in_group_output: u64,
  pub pyramid_level: u64,
  pub enable_early_exit: bool,
  pub tx_mode_select: bool,
}

pub(crate) fn pos_to_lvl(pos: u64, pyramid_depth: u64) -> u64 {
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
    assert!(sequence.bit_depth <= mem::size_of::<T>() * 8, "bit depth cannot fit into u8");
    // Speed level decides the minimum partition size, i.e. higher speed --> larger min partition size,
    // with exception that SBs on right or bottom frame borders split down to BLOCK_4X4.
    // At speed = 0, RDO search is exhaustive.
    let min_partition_size = config.speed_settings.min_block_size;
    assert!(min_partition_size.is_sqr());
    let use_reduced_tx_set = config.speed_settings.reduced_tx_set;
    let use_tx_domain_distortion = config.tune == Tune::Psnr && config.speed_settings.tx_domain_distortion;
    let use_tx_domain_rate = config.speed_settings.tx_domain_rate;

    let w_in_b = 2 * config.width.align_power_of_two_and_shift(3); // MiCols, ((width+7)/8)<<3 >> MI_SIZE_LOG2
    let h_in_b = 2 * config.height.align_power_of_two_and_shift(3); // MiRows, ((height+7)/8)<<3 >> MI_SIZE_LOG2

    let frame_rate = (config.time_base.den / config.time_base.num) as usize;
    let min_tile_cols = (config.width - 1) / 4096;
    // The minimum number of tiles is determined based on the following requirements:
    // - A tile cannot be wider than 4096 pixels
    // - A tile cannot be larger than 4096x2304 pixels
    // - The tile size * the frame rate * a temporal ratio cannot exceed a given fixed rate
    //   corresponding to 4K60 resolution with a 1.1 ratio.
    let min_tiles = 1 + min_tile_cols.max(
      (config.width * config.height) / (4096 * 2304)).max(
      (config.width * config.height * frame_rate) / (4096_f64 * 2176_f64 * 60_f64 * 1.1) as usize
    );

    let mut tiling = TilingInfo::new(
      sequence.sb_size_log2(),
      config.width,
      config.height,
      config.tile_cols_log2.min(min_tile_cols),
      config.tile_rows_log2
    );

    if config.tiles > 0 || tiling.rows * tiling.cols < min_tiles {
      // If a number of automatically-assigned tiles is specified,
      // start with the bare minimum number of tile rows and columns.
      // Otherwise, the tile assignment is triggered because we need
      // to add more tiles than the number set in the configuration.
      let mut tile_rows_log2 = if config.tiles == 0 { config.tile_rows_log2 } else { 0 };
      let mut tile_cols_log2 = if config.tiles == 0 { config.tile_cols_log2 } else { min_tile_cols };
      while (tile_rows_log2 < tiling.max_tile_rows_log2) || (tile_cols_log2 < tiling.max_tile_cols_log2) {

        tiling = TilingInfo::new(
          sequence.sb_size_log2(),
          config.width,
          config.height,
          tile_cols_log2,
          tile_rows_log2
        );

        if tiling.rows * tiling.cols >= config.tiles &&
          tiling.rows * tiling.cols >= min_tiles {
            break;
        }

        if ((tiling.tile_height_sb >= tiling.tile_width_sb) &&
            (tiling.tile_rows_log2 < tiling.max_tile_rows_log2))
          || (tile_cols_log2 >= tiling.max_tile_cols_log2) {
            tile_rows_log2 += 1;
          } else {
            tile_cols_log2 += 1;
          }
      }
    }

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
      showable_frame: true,
      error_resilient: false,
      intra_only: false,
      allow_high_precision_mv: false,
      frame_type: FrameType::KEY,
      show_existing_frame: false,
      frame_to_show_map_idx: 0,
      use_reduced_tx_set,
      reference_mode: ReferenceMode::SINGLE,
      use_prev_frame_mvs: false,
      min_partition_size,
      globalmv_transformation_type: [GlobalMVMode::IDENTITY; INTER_REFS_PER_FRAME],
      num_tg: 1,
      large_scale_tile: false,
      disable_cdf_update: false,
      allow_screen_content_tools: 0,
      force_integer_mv: 0,
      primary_ref_frame: PRIMARY_REF_NONE,
      refresh_frame_flags: 0,
      allow_intrabc: false,
      use_ref_frame_mvs: false,
      is_filter_switchable: false,
      is_motion_mode_switchable: false, // 0: only the SIMPLE motion mode will be used.
      disable_frame_end_update_cdf: false,
      allow_warped_motion: false,
      cdef_damping: 3,
      cdef_bits: 3,
      cdef_y_strengths: [0*4+0, 1*4+0, 2*4+1, 3*4+1, 5*4+2, 7*4+3, 10*4+3, 13*4+3],
      cdef_uv_strengths: [0*4+0, 1*4+0, 2*4+1, 3*4+1, 5*4+2, 7*4+3, 10*4+3, 13*4+3],
      delta_q_present: false,
      ref_frames: [0; INTER_REFS_PER_FRAME],
      ref_frame_sign_bias: [false; INTER_REFS_PER_FRAME],
      rec_buffer: ReferenceFramesSet::new(),
      base_q_idx: config.quantizer as u8,
      dc_delta_q: [0; 3],
      ac_delta_q: [0; 3],
      lambda: 0.0,
      me_lambda: 0.0,
      me_range_scale: 1,
      use_tx_domain_distortion,
      use_tx_domain_rate,
      idx_in_group_output: 0,
      pyramid_level: 0,
      enable_early_exit: true,
      config,
      tx_mode_select : false,
    }
  }

  pub fn new_key_frame(previous_fi: &Self,
   segment_input_frameno_start: u64) -> Self {
    let mut fi = previous_fi.clone();
    fi.frame_type = FrameType::KEY;
    fi.intra_only = true;
    fi.idx_in_group_output = 0;
    fi.pyramid_level = 0;
    fi.order_hint = 0;
    fi.refresh_frame_flags = ALL_REF_FRAMES_MASK;
    fi.show_frame = true;
    fi.show_existing_frame = false;
    fi.frame_to_show_map_idx = 0;
    fi.primary_ref_frame = PRIMARY_REF_NONE;
    fi.input_frameno = segment_input_frameno_start;
    for i in 0..INTER_REFS_PER_FRAME {
      fi.ref_frames[i] = 0;
    }

    // Until has_tr() and has_bl() is fixed to use partition info, disable intra tx partition
    fi.tx_mode_select = false;

    fi
  }

  /// Returns the created FrameInvariants along with a bool indicating success.
  /// This interface provides simpler usage, because we always need the produced
  /// FrameInvariants regardless of success or failure.
  pub(crate) fn new_inter_frame(
    previous_fi: &Self, inter_cfg: &InterConfig,
    segment_input_frameno_start: u64, output_frameno_in_segment: u64,
    next_keyframe_input_frameno: u64
  ) -> (Self, bool) {
    let mut fi = previous_fi.clone();
    fi.frame_type = FrameType::INTER;
    fi.intra_only = false;
    fi.idx_in_group_output =
     inter_cfg.get_idx_in_group_output(output_frameno_in_segment);
    fi.tx_mode_select = false;

    fi.order_hint = inter_cfg.get_order_hint(output_frameno_in_segment,
     fi.idx_in_group_output);
    let input_frameno = segment_input_frameno_start + fi.order_hint as u64;
    if input_frameno >= next_keyframe_input_frameno {
      fi.show_existing_frame = false;
      fi.show_frame = false;
      return (fi, false);
    }

    fi.pyramid_level = inter_cfg.get_level(fi.idx_in_group_output);

    let slot_idx = inter_cfg.get_slot_idx(fi.pyramid_level, fi.order_hint);
    fi.show_frame = inter_cfg.get_show_frame(fi.idx_in_group_output);
    fi.show_existing_frame =
     inter_cfg.get_show_existing_frame(fi.idx_in_group_output);
    fi.frame_to_show_map_idx = slot_idx;
    fi.refresh_frame_flags = if fi.show_existing_frame {
      0
    } else {
      1 << slot_idx
    };

    let second_ref_frame = if !inter_cfg.multiref {
      LAST_FRAME // make second_ref_frame match first
    } else if fi.idx_in_group_output == 0 {
      LAST2_FRAME
    } else {
      ALTREF_FRAME
    };
    let ref_in_previous_group = LAST3_FRAME;

    // reuse probability estimates from previous frames only in top level frames
    fi.primary_ref_frame = if fi.pyramid_level > 0 {
      PRIMARY_REF_NONE
    } else {
      (ref_in_previous_group.to_index()) as u32
    };

    for i in 0..INTER_REFS_PER_FRAME {
      fi.ref_frames[i] = if fi.pyramid_level == 0 {
        if i == second_ref_frame.to_index() {
          (slot_idx + 4 - 2) as u8 % 4
        } else {
          (slot_idx + 4 - 1) as u8 % 4
        }
      } else if i == second_ref_frame.to_index() {
        let oh = fi.order_hint
         + (inter_cfg.group_input_len as u32 >> fi.pyramid_level);
        let lvl2 = pos_to_lvl(oh as u64, inter_cfg.pyramid_depth);
        if lvl2 == 0 {
          ((oh >> inter_cfg.pyramid_depth) % 4) as u8
        } else {
          3 + lvl2 as u8
        }
      } else if i == ref_in_previous_group.to_index() {
        if fi.pyramid_level == 0 {
          (slot_idx + 4 - 1) as u8 % 4
        } else {
          slot_idx as u8
        }
      } else {
        let oh = fi.order_hint
         - (inter_cfg.group_input_len as u32 >> fi.pyramid_level);
        let lvl1 = pos_to_lvl(oh as u64, inter_cfg.pyramid_depth);
        if lvl1 == 0 {
          ((oh >> inter_cfg.pyramid_depth) % 4) as u8
        } else {
          3 + lvl1 as u8
        }
      };
      fi.ref_frame_sign_bias[i] = if !fi.sequence.enable_order_hint {
        false
      } else if let Some(ref rec) =
        fi.rec_buffer.frames[fi.ref_frames[i] as usize]
      {
        let hint = rec.order_hint;
        fi.sequence.get_relative_dist(hint, fi.order_hint) > 0
      } else {
        false
      };
    }

    fi.reference_mode = if inter_cfg.multiref && fi.idx_in_group_output != 0 {
      ReferenceMode::SELECT
    } else {
      ReferenceMode::SINGLE
    };
    fi.input_frameno = input_frameno;
    fi.me_range_scale = (inter_cfg.group_input_len >> fi.pyramid_level) as u8;
    (fi, true)
  }

  pub fn get_frame_subtype(&self) -> usize {
    if self.frame_type == FrameType::KEY {
      FRAME_SUBTYPE_I
    } else {
      FRAME_SUBTYPE_P + (self.pyramid_level as usize)
    }
  }

  pub fn set_quantizers(&mut self, qps: &QuantizerParameters) {
    self.base_q_idx = qps.ac_qi[0];
    if self.frame_type != FrameType::KEY {
      self.cdef_bits = 3 - ((self.base_q_idx.max(128) - 128) >> 5);
    } else {
      self.cdef_bits = 3;
    }
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
  }

  #[inline(always)]
  pub fn sb_size_log2(&self) -> usize {
    self.sequence.sb_size_log2()
  }

  #[inline(always)]
  pub fn sb_size(&self) -> usize {
    self.sequence.sb_size()
  }
}

impl<T: Pixel> fmt::Display for FrameInvariants<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "Input Frame {} - {}", self.input_frameno, self.frame_type)
  }
}

pub fn write_temporal_delimiter(
  packet: &mut dyn io::Write
) -> io::Result<()> {
  packet.write_all(&TEMPORAL_DELIMITER)?;
  Ok(())
}

fn write_obus<T: Pixel>(
  packet: &mut dyn io::Write, fi: &FrameInvariants<T>, fs: &FrameState<T>
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
    bw2.write_frame_header_obu(fi, fs)?;
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
  dst: &mut [i16],
  src1: &PlaneRegion<'_, T>,
  src2: &PlaneRegion<'_, T>,
  width: usize,
  height: usize,
) {
  for ((l, s1), s2) in dst.chunks_mut(width).take(height)
    .zip(src1.rows_iter())
    .zip(src2.rows_iter()) {
      for ((r, v1), v2) in l.iter_mut().zip(s1).zip(s2) {
        *r = i16::cast_from(*v1) - i16::cast_from(*v2);
      }
    }
}

fn get_qidx<T: Pixel>(fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>, cw: &ContextWriter, tile_bo: BlockOffset) -> u8 {
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
  fi: &FrameInvariants<T>, ts: &mut TileStateMut<'_, T>, cw: &mut ContextWriter,
  w: &mut dyn Writer, p: usize, tile_bo: BlockOffset, mode: PredictionMode,
  tx_size: TxSize, tx_type: TxType, plane_bsize: BlockSize, po: PlaneOffset,
  skip: bool, ac: &[i16], alpha: i16, rdo_type: RDOType, need_recon_pixel: bool
) -> (bool, i64) {
  let qidx = get_qidx(fi, ts, cw, tile_bo);
  let PlaneConfig { xdec, ydec, .. } = ts.input.planes[p].cfg;
  let tile_rect = ts.tile_rect().decimated(xdec, ydec);
  let rec = &mut ts.rec.planes[p];
  let area = Area::BlockStartingAt { bo: tile_bo };

  assert!(tx_size.sqr() <= TxSize::TX_32X32 || tx_type == TxType::DCT_DCT);
  debug_assert!(p != 0 || !mode.is_intra() || tx_size.block_size() == plane_bsize || need_recon_pixel,
    "mode.is_intra()={:#?}, plane={:#?}, tx_size.block_size()={:#?}, plane_bsize={:#?}, need_recon_pixel={:#?}",
    mode.is_intra(), p, tx_size.block_size(), plane_bsize, need_recon_pixel);

  if mode.is_intra() {
    let bit_depth = fi.sequence.bit_depth;
    let edge_buf = get_intra_edges(&rec.as_const(), po, tx_size, bit_depth, Some(mode));
    mode.predict_intra(tile_rect, &mut rec.subregion_mut(area), tx_size, bit_depth, &ac, alpha, &edge_buf);
  }

  if skip { return (false, -1); }

  let mut residual_storage: AlignedArray<[i16; 64 * 64]> = UninitializedAlignedArray();
  let mut coeffs_storage: AlignedArray<[i32; 64 * 64]> = UninitializedAlignedArray();
  let mut qcoeffs_storage: AlignedArray<[i32; 64 * 64]> = UninitializedAlignedArray();
  let mut rcoeffs_storage: AlignedArray<[i32; 64 * 64]> = UninitializedAlignedArray();
  let residual = &mut residual_storage.array[..tx_size.area()];
  let coeffs = &mut coeffs_storage.array[..tx_size.area()];
  let qcoeffs = &mut qcoeffs_storage.array[..tx_size.area()];
  let rcoeffs = &mut rcoeffs_storage.array[..tx_size.area()];

  diff(
    residual,
    &ts.input_tile.planes[p].subregion(area),
    &rec.subregion(area),
    tx_size.width(),
    tx_size.height());

  forward_transform(residual, coeffs, tx_size.width(), tx_size, tx_type, fi.sequence.bit_depth);

  let coded_tx_size = av1_get_coded_tx_size(tx_size).area();
  ts.qc.quantize(coeffs, qcoeffs, coded_tx_size);

  let tell_coeffs = w.tell_frac();
  let has_coeff = if need_recon_pixel || rdo_type.needs_coeff_rate() {
    cw.write_coeffs_lv_map(w, p, tile_bo, &qcoeffs, mode, tx_size, tx_type, plane_bsize, xdec, ydec,
                           fi.use_reduced_tx_set)
  } else {
    true
  };
  let cost_coeffs = w.tell_frac() - tell_coeffs;
  // Reconstruct
  dequantize(qidx, qcoeffs, rcoeffs, tx_size, fi.sequence.bit_depth, fi.dc_delta_q[p], fi.ac_delta_q[p]);

  let mut tx_dist: i64 = -1;

  if !fi.use_tx_domain_distortion || need_recon_pixel {
    inverse_transform_add(rcoeffs, &mut rec.subregion_mut(area), tx_size, tx_type, fi.sequence.bit_depth);
  }
  if rdo_type.needs_tx_dist() {
    // Store tx-domain distortion of this block
    tx_dist = coeffs
      .iter()
      .zip(rcoeffs)
      .map(|(a, b)| {
        let c = *a as i32 - *b as i32;
        (c * c) as u64
      }).sum::<u64>() as i64;

    let tx_dist_scale_bits = 2*(3 - get_log_tx_scale(tx_size));
    let tx_dist_scale_rounding_offset = 1 << (tx_dist_scale_bits - 1);
    tx_dist = (tx_dist + tx_dist_scale_rounding_offset) >> tx_dist_scale_bits;
  }
  if fi.config.train_rdo {
    ts.rdo.add_rate(fi.base_q_idx, tx_size, tx_dist as u64, cost_coeffs as u64);
  }

  if rdo_type == RDOType::TxDistEstRate {
    // look up rate and distortion in table
    let estimated_rate = estimate_rate(fi.base_q_idx, tx_size, tx_dist as u64);
    w.add_bits_frac(estimated_rate as u32);
  }
  (has_coeff, tx_dist)
}

pub fn motion_compensate<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &mut TileStateMut<'_, T>, cw: &mut ContextWriter,
  luma_mode: PredictionMode, ref_frames: [RefType; 2], mvs: [MotionVector; 2],
  bsize: BlockSize, tile_bo: BlockOffset, luma_only: bool
) {
  debug_assert!(!luma_mode.is_intra());

  let PlaneConfig { xdec: u_xdec, ydec: u_ydec, .. } = ts.input.planes[1].cfg;

  // Inter mode prediction can take place once for a whole partition,
  // instead of each tx-block.
  let num_planes = 1 + if !luma_only && has_chroma(tile_bo, bsize, u_xdec, u_ydec) { 2 } else { 0 };

  let luma_tile_rect = ts.tile_rect();
  for p in 0..num_planes {
    let plane_bsize = if p == 0 { bsize }
    else { bsize.subsampled_size(u_xdec, u_ydec) };

    let rec = &mut ts.rec.planes[p];
    let po = tile_bo.plane_offset(&rec.plane_cfg);
    let &PlaneConfig { xdec, ydec, .. } = rec.plane_cfg;
    let tile_rect = luma_tile_rect.decimated(xdec, ydec);

    let area = Area::BlockStartingAt { bo: tile_bo };
    if p > 0 && bsize < BlockSize::BLOCK_8X8 {
      let mut some_use_intra = false;
      if bsize == BlockSize::BLOCK_4X4 || bsize == BlockSize::BLOCK_4X8 {
        some_use_intra |= cw.bc.blocks[tile_bo.with_offset(-1,0)].mode.is_intra(); };
      if !some_use_intra && bsize == BlockSize::BLOCK_4X4 || bsize == BlockSize::BLOCK_8X4 {
        some_use_intra |= cw.bc.blocks[tile_bo.with_offset(0,-1)].mode.is_intra(); };
      if !some_use_intra && bsize == BlockSize::BLOCK_4X4 {
        some_use_intra |= cw.bc.blocks[tile_bo.with_offset(-1,-1)].mode.is_intra(); };

      if some_use_intra {
        luma_mode.predict_inter(fi, tile_rect, p, po, &mut rec.subregion_mut(area), plane_bsize.width(),
                                plane_bsize.height(), ref_frames, mvs);
      } else {
        assert!(u_xdec == 1 && u_ydec == 1);
        // TODO: these are absolutely only valid for 4:2:0
        if bsize == BlockSize::BLOCK_4X4 {
          let mv0 = cw.bc.blocks[tile_bo.with_offset(-1,-1)].mv;
          let rf0 = cw.bc.blocks[tile_bo.with_offset(-1,-1)].ref_frames;
          let mv1 = cw.bc.blocks[tile_bo.with_offset(0,-1)].mv;
          let rf1 = cw.bc.blocks[tile_bo.with_offset(0,-1)].ref_frames;
          let po1 = PlaneOffset { x: po.x+2, y: po.y };
          let area1 = Area::StartingAt { x: po1.x, y: po1.y };
          let mv2 = cw.bc.blocks[tile_bo.with_offset(-1,0)].mv;
          let rf2 = cw.bc.blocks[tile_bo.with_offset(-1,0)].ref_frames;
          let po2 = PlaneOffset { x: po.x, y: po.y+2 };
          let area2 = Area::StartingAt { x: po2.x, y: po2.y };
          let po3 = PlaneOffset { x: po.x+2, y: po.y+2 };
          let area3 = Area::StartingAt { x: po3.x, y: po3.y };
          luma_mode.predict_inter(fi, tile_rect, p, po, &mut rec.subregion_mut(area), 2, 2, rf0, mv0);
          luma_mode.predict_inter(fi, tile_rect, p, po1, &mut rec.subregion_mut(area1), 2, 2, rf1, mv1);
          luma_mode.predict_inter(fi, tile_rect, p, po2, &mut rec.subregion_mut(area2), 2, 2, rf2, mv2);
          luma_mode.predict_inter(fi, tile_rect, p, po3, &mut rec.subregion_mut(area3), 2, 2, ref_frames, mvs);
        }
        if bsize == BlockSize::BLOCK_8X4 {
          let mv1 = cw.bc.blocks[tile_bo.with_offset(0,-1)].mv;
          let rf1 = cw.bc.blocks[tile_bo.with_offset(0,-1)].ref_frames;
          luma_mode.predict_inter(fi, tile_rect, p, po, &mut rec.subregion_mut(area), 4, 2, rf1, mv1);
          let po3 = PlaneOffset { x: po.x, y: po.y+2 };
          let area3 = Area::StartingAt { x: po3.x, y: po3.y };
          luma_mode.predict_inter(fi, tile_rect, p, po3, &mut rec.subregion_mut(area3), 4, 2, ref_frames, mvs);
        }
        if bsize == BlockSize::BLOCK_4X8 {
          let mv2 = cw.bc.blocks[tile_bo.with_offset(-1,0)].mv;
          let rf2 = cw.bc.blocks[tile_bo.with_offset(-1,0)].ref_frames;
          luma_mode.predict_inter(fi, tile_rect, p, po, &mut rec.subregion_mut(area), 2, 4, rf2, mv2);
          let po3 = PlaneOffset { x: po.x+2, y: po.y };
          let area3 = Area::StartingAt { x: po3.x, y: po3.y };
          luma_mode.predict_inter(fi, tile_rect, p, po3, &mut rec.subregion_mut(area3), 2, 4, ref_frames, mvs);
        }
      }
    } else {
      luma_mode.predict_inter(fi, tile_rect, p, po, &mut rec.subregion_mut(area), plane_bsize.width(),
                              plane_bsize.height(), ref_frames, mvs);
    }
  }
}

pub fn save_block_motion<T: Pixel>(
   ts: &mut TileStateMut<'_, T>,
   bsize: BlockSize, tile_bo: BlockOffset,
   ref_frame: usize, mv: MotionVector,
) {
  let tile_mvs = &mut ts.mvs[ref_frame];
  let tile_bo_x_end = (tile_bo.x + bsize.width_mi()).min(ts.mi_width);
  let tile_bo_y_end = (tile_bo.y + bsize.height_mi()).min(ts.mi_height);
  for mi_y in tile_bo.y..tile_bo_y_end {
    for mi_x in tile_bo.x..tile_bo_x_end {
      tile_mvs[mi_y][mi_x] = mv;
    }
  }
}

pub fn encode_block_pre_cdef<T: Pixel>(
  seq: &Sequence, ts: &TileStateMut<'_, T>,
  cw: &mut ContextWriter, w: &mut dyn Writer,
  bsize: BlockSize, tile_bo: BlockOffset, skip: bool
) -> bool {
  cw.bc.blocks.set_skip(tile_bo, bsize, skip);
  if ts.segmentation.enabled && ts.segmentation.update_map && ts.segmentation.preskip {
    cw.write_segmentation(w, tile_bo, bsize, false, ts.segmentation.last_active_segid);
  }
  cw.write_skip(w, tile_bo, skip);
  if ts.segmentation.enabled && ts.segmentation.update_map && !ts.segmentation.preskip {
    cw.write_segmentation(w, tile_bo, bsize, skip, ts.segmentation.last_active_segid);
  }
  if !skip && seq.enable_cdef {
    cw.bc.cdef_coded = true;
  }
  cw.bc.cdef_coded
}

pub fn encode_block_post_cdef<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &mut TileStateMut<'_, T>,
  cw: &mut ContextWriter, w: &mut dyn Writer,
  luma_mode: PredictionMode, chroma_mode: PredictionMode,
  ref_frames: [RefType; 2], mvs: [MotionVector; 2],
  bsize: BlockSize, tile_bo: BlockOffset, skip: bool,
  cfl: CFLParams, tx_size: TxSize, tx_type: TxType,
  mode_context: usize, mv_stack: &[CandidateMV],
  rdo_type: RDOType, need_recon_pixel: bool
) -> i64 {
  let is_inter = !luma_mode.is_intra();
  if is_inter { assert!(luma_mode == chroma_mode); };
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
  if cw.bc.code_deltas && ts.deblock.block_deltas_enabled && (bsize < sb_size || !skip) {
    cw.write_block_deblock_deltas(w, tile_bo, ts.deblock.block_delta_multi);
  }
  cw.bc.code_deltas = false;

  if fi.frame_type == FrameType::INTER {
    cw.write_is_inter(w, tile_bo, is_inter);
    if is_inter {
      cw.fill_neighbours_ref_counts(tile_bo);
      cw.write_ref_frames(w, fi, tile_bo);

      if luma_mode >= PredictionMode::NEAREST_NEARESTMV {
        cw.write_compound_mode(w, luma_mode, mode_context);
      } else {
        cw.write_inter_mode(w, luma_mode, mode_context);
      }

      let ref_mv_idx = 0;
      let num_mv_found = mv_stack.len();

      if luma_mode == PredictionMode::NEWMV || luma_mode == PredictionMode::NEW_NEWMV {
        if luma_mode == PredictionMode::NEW_NEWMV { assert!(num_mv_found >= 2); }
        for idx in 0..2 {
          if num_mv_found > idx + 1 {
            let drl_mode = ref_mv_idx > idx;
            let ctx: usize = (mv_stack[idx].weight < REF_CAT_LEVEL) as usize
              + (mv_stack[idx + 1].weight < REF_CAT_LEVEL) as usize;
            cw.write_drl_mode(w, drl_mode, ctx);
            if !drl_mode { break; }
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

      if luma_mode == PredictionMode::NEWMV ||
        luma_mode == PredictionMode::NEW_NEWMV ||
        luma_mode == PredictionMode::NEW_NEARESTMV {
          cw.write_mv(w, mvs[0], ref_mvs[0], mv_precision);
        }
      if luma_mode == PredictionMode::NEW_NEWMV ||
        luma_mode == PredictionMode::NEAREST_NEWMV {
          cw.write_mv(w, mvs[1], ref_mvs[1], mv_precision);
        }

      if luma_mode >= PredictionMode::NEAR0MV && luma_mode <= PredictionMode::NEAR2MV {
        let ref_mv_idx = luma_mode as usize - PredictionMode::NEAR0MV as usize + 1;
        if luma_mode != PredictionMode::NEAR0MV { assert!(num_mv_found > ref_mv_idx); }

        for idx in 1..3 {
          if num_mv_found > idx + 1 {
            let drl_mode = ref_mv_idx > idx;
            let ctx: usize = (mv_stack[idx].weight < REF_CAT_LEVEL) as usize
              + (mv_stack[idx + 1].weight < REF_CAT_LEVEL) as usize;

            cw.write_drl_mode(w, drl_mode, ctx);
            if !drl_mode { break; }
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
      cw.write_angle_delta(w, 0, luma_mode);
    }
    if has_chroma(tile_bo, bsize, xdec, ydec) {
      cw.write_intra_uv_mode(w, chroma_mode, luma_mode, bsize);
      if chroma_mode.is_cfl() {
        assert!(bsize.cfl_allowed());
        cw.write_cfl_alphas(w, cfl);
      }
      if chroma_mode.is_directional() && bsize >= BlockSize::BLOCK_8X8 {
        cw.write_angle_delta(w, 0, chroma_mode);
      }
    }
    // TODO: Extra condition related to palette mode, see `read_filter_intra_mode_info` in decodemv.c
    if fi.sequence.enable_filter_intra &&
      luma_mode == PredictionMode::DC_PRED && bsize.width() <= 32 && bsize.height() <= 32 {
      cw.write_use_filter_intra(w,false, bsize); // turn off FILTER_INTRA
    }
  }

  // write tx_size here
  if fi.tx_mode_select {
    if bsize.greater_than(BlockSize::BLOCK_4X4) && !(is_inter && skip) {
      if !is_inter {
        cw.write_tx_size_intra(w, tile_bo, bsize, tx_size);
        cw.bc.update_tx_size_context(tile_bo, bsize, tx_size, false);
      } /*else {  // TODO (yushin): write_tx_size_inter(), i.e. var-tx

      }*/
    } else {
      cw.bc.update_tx_size_context(tile_bo, bsize, tx_size, is_inter && skip);
    }
  }

  if is_inter {
    motion_compensate(fi, ts, cw, luma_mode, ref_frames, mvs, bsize, tile_bo, false);
    write_tx_tree(fi, ts, cw, w, luma_mode, tile_bo, bsize, tx_size, tx_type, skip, false, rdo_type, need_recon_pixel)
  } else {
    write_tx_blocks(fi, ts, cw, w, luma_mode, chroma_mode, tile_bo, bsize, tx_size, tx_type, skip, cfl, false, rdo_type, need_recon_pixel)
  }
}

pub fn luma_ac<T: Pixel>(
  ac: &mut [i16], ts: &mut TileStateMut<'_, T>, tile_bo: BlockOffset, bsize: BlockSize
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
  let luma = &rec.subregion(Area::BlockStartingAt { bo });

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
        sample += i16::cast_from(luma[y + 1][x])
          + i16::cast_from(luma[y + 1][x + 1]);
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
  cw: &mut ContextWriter, w: &mut dyn Writer,
  luma_mode: PredictionMode, chroma_mode: PredictionMode, tile_bo: BlockOffset,
  bsize: BlockSize, tx_size: TxSize, tx_type: TxType, skip: bool,
  cfl: CFLParams, luma_only: bool, rdo_type: RDOType, need_recon_pixel: bool
) -> i64 {
  let bw = bsize.width_mi() / tx_size.width_mi();
  let bh = bsize.height_mi() / tx_size.height_mi();
  let qidx = get_qidx(fi, ts, cw, tile_bo);

  let PlaneConfig { xdec, ydec, .. } = ts.input.planes[1].cfg;
  let mut ac: AlignedArray<[i16; 32 * 32]> = UninitializedAlignedArray();
  let mut tx_dist: i64 = 0;
  let do_chroma = has_chroma(tile_bo, bsize, xdec, ydec);

  ts.qc.update(qidx, tx_size, luma_mode.is_intra(), fi.sequence.bit_depth, fi.dc_delta_q[0], 0);

  for by in 0..bh {
    for bx in 0..bw {
      let tx_bo = BlockOffset {
        x: tile_bo.x + bx * tx_size.width_mi(),
        y: tile_bo.y + by * tx_size.height_mi()
      };

      let po = tx_bo.plane_offset(&ts.input.planes[0].cfg);
      let (_, dist) =
        encode_tx_block(
          fi, ts, cw, w, 0, tx_bo, luma_mode, tx_size, tx_type, bsize, po,
          skip, &ac.array, 0, rdo_type, need_recon_pixel
        );
      assert!(!fi.use_tx_domain_distortion || need_recon_pixel || skip || dist >= 0);
      tx_dist += dist;
    }
  }

  if luma_only { return tx_dist };

  let uv_tx_size = bsize.largest_chroma_tx_size(xdec, ydec);

  let mut bw_uv = (bw * tx_size.width_mi()) >> xdec;
  let mut bh_uv = (bh * tx_size.height_mi()) >> ydec;

  if (bw_uv == 0 || bh_uv == 0) && do_chroma {
    bw_uv = 1;
    bh_uv = 1;
  }

  bw_uv /= uv_tx_size.width_mi();
  bh_uv /= uv_tx_size.height_mi();

  let plane_bsize = bsize.subsampled_size(xdec, ydec);

  if chroma_mode.is_cfl() {
    luma_ac(&mut ac.array, ts, tile_bo, bsize);
  }

  if bw_uv > 0 && bh_uv > 0 {
    // TODO: Disable these asserts temporarilly, since chroma_sampling_422_aom and chroma_sampling_444_aom
    // tests seems trigerring them as well, which should not
    // TODO: Not valid if partition > 64x64 && chroma != 420
    /*if xdec == 1 && ydec == 1 {
      assert!(bw_uv == 1, "bw_uv = {}, bh_uv = {}", bw_uv, bh_uv);
      assert!(bh_uv == 1, "bw_uv = {}, bh_uv = {}", bw_uv, bh_uv);
    }*/
    let uv_tx_type = if uv_tx_size.width() >= 32 || uv_tx_size.height() >= 32 {
      TxType::DCT_DCT
    } else {
      uv_intra_mode_to_tx_type_context(chroma_mode)
    };

    for p in 1..3 {
      ts.qc.update(fi.base_q_idx, uv_tx_size, true, fi.sequence.bit_depth, fi.dc_delta_q[p], fi.ac_delta_q[p]);
      let alpha = cfl.alpha(p - 1);
      for by in 0..bh_uv {
        for bx in 0..bw_uv {
          let tx_bo =
            BlockOffset {
              x: tile_bo.x + ((bx * uv_tx_size.width_mi()) << xdec) -
                ((bw * tx_size.width_mi() == 1) as usize) * xdec,
              y: tile_bo.y + ((by * uv_tx_size.height_mi()) << ydec) -
                ((bh * tx_size.height_mi() == 1) as usize) * ydec
            };

          let mut po = tile_bo.plane_offset(&ts.input.planes[p].cfg);
          po.x += (bx * uv_tx_size.width()) as isize;
          po.y += (by * uv_tx_size.height()) as isize;
          let (_, dist) =
            encode_tx_block(fi, ts, cw, w, p, tx_bo, chroma_mode, uv_tx_size, uv_tx_type,
                            plane_bsize, po, skip, &ac.array, alpha, rdo_type, need_recon_pixel);
          assert!(!fi.use_tx_domain_distortion || need_recon_pixel || skip || dist >= 0);
          tx_dist += dist;
        }
      }
    }
  }

  tx_dist
}

// FIXME: For now, assume tx_mode is LARGEST_TX, so var-tx is not implemented yet,
// which means only one tx block exist for a inter mode partition.
pub fn write_tx_tree<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &mut TileStateMut<'_, T>, cw: &mut ContextWriter, w: &mut dyn Writer,
  luma_mode: PredictionMode, tile_bo: BlockOffset,
  bsize: BlockSize, tx_size: TxSize, tx_type: TxType, skip: bool,
  luma_only: bool, rdo_type: RDOType, need_recon_pixel: bool
) -> i64 {
  let bw = bsize.width_mi() / tx_size.width_mi();
  let bh = bsize.height_mi() / tx_size.height_mi();
  let qidx = get_qidx(fi, ts, cw, tile_bo);

  let PlaneConfig { xdec, ydec, .. } = ts.input.planes[1].cfg;
  let ac = &[0i16; 0];
  let mut tx_dist: i64 = 0;

  ts.qc.update(qidx, tx_size, luma_mode.is_intra(), fi.sequence.bit_depth, fi.dc_delta_q[0], 0);

  let po = tile_bo.plane_offset(&ts.input.planes[0].cfg);
  let (has_coeff, dist) = encode_tx_block(
    fi, ts, cw, w, 0, tile_bo, luma_mode, tx_size, tx_type, bsize, po, skip, ac, 0, rdo_type, need_recon_pixel
  );
  assert!(!fi.use_tx_domain_distortion || need_recon_pixel || skip || dist >= 0);
  tx_dist += dist;

  if luma_only { return tx_dist };

  let uv_tx_size = bsize.largest_chroma_tx_size(xdec, ydec);

  let mut bw_uv = (bw * tx_size.width_mi()) >> xdec;
  let mut bh_uv = (bh * tx_size.height_mi()) >> ydec;

  if (bw_uv == 0 || bh_uv == 0) && has_chroma(tile_bo, bsize, xdec, ydec) {
    bw_uv = 1;
    bh_uv = 1;
  }

  bw_uv /= uv_tx_size.width_mi();
  bh_uv /= uv_tx_size.height_mi();

  let plane_bsize = bsize.subsampled_size(xdec, ydec);

  if bw_uv > 0 && bh_uv > 0 {
    // TODO: Disable these asserts temporarilly, since chroma_sampling_422_aom and chroma_sampling_444_aom
    // tests seems trigerring them as well, which should not
    // TODO: Not valid if partition > 64x64 && chroma != 420
    /*if xdec == 1 && ydec == 1 {
      debug_assert!(bw_uv == 1, "bw_uv = {}, bh_uv = {}", bw_uv, bh_uv);
      debug_assert!(bh_uv == 1, "bw_uv = {}, bh_uv = {}", bw_uv, bh_uv);
    }*/
    let uv_tx_type = if has_coeff {tx_type} else {TxType::DCT_DCT}; // if inter mode, uv_tx_type == tx_type

    for p in 1..3 {
      ts.qc.update(qidx, uv_tx_size, false, fi.sequence.bit_depth, fi.dc_delta_q[p], fi.ac_delta_q[p]);
      let tx_bo = BlockOffset {
        x: tile_bo.x  - ((bw * tx_size.width_mi() == 1) as usize),
        y: tile_bo.y  - ((bh * tx_size.height_mi() == 1) as usize)
      };

      let po = tile_bo.plane_offset(&ts.input.planes[p].cfg);
      let (_, dist) =
        encode_tx_block(fi, ts, cw, w, p, tx_bo, luma_mode, uv_tx_size, uv_tx_type,
                        plane_bsize, po, skip, ac, 0, rdo_type, need_recon_pixel);
      assert!(!fi.use_tx_domain_distortion || need_recon_pixel || skip || dist >= 0);
      tx_dist += dist;
    }
  }

  tx_dist
}

pub fn encode_block_with_modes<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &mut TileStateMut<'_, T>,
  cw: &mut ContextWriter, w_pre_cdef: &mut dyn Writer, w_post_cdef: &mut dyn Writer,
  bsize: BlockSize, tile_bo: BlockOffset, mode_decision: &RDOPartitionOutput,
  rdo_type: RDOType
) {
  let (mode_luma, mode_chroma) =
    (mode_decision.pred_mode_luma, mode_decision.pred_mode_chroma);
  let cfl = mode_decision.pred_cfl_params;
  let ref_frames = mode_decision.ref_frames;
  let mvs = mode_decision.mvs;
  let skip = mode_decision.skip;
  let mut cdef_coded = cw.bc.cdef_coded;
  let (tx_size, tx_type) = (mode_decision.tx_size, mode_decision.tx_type);

  debug_assert!((tx_size, tx_type) ==
                rdo_tx_size_type(fi, ts, cw, bsize, tile_bo, mode_luma, ref_frames, mvs, skip));

  let mut mv_stack = ArrayVec::<[CandidateMV; 9]>::new();
  let is_compound = ref_frames[1] != NONE_FRAME;
  let mode_context = cw.find_mvrefs(tile_bo, ref_frames, &mut mv_stack, bsize, fi, is_compound);

  cdef_coded = encode_block_pre_cdef(&fi.sequence, ts, cw, if cdef_coded  {w_post_cdef} else {w_pre_cdef},
                              bsize, tile_bo, skip);
  encode_block_post_cdef(fi, ts, cw, if cdef_coded  {w_post_cdef} else {w_pre_cdef},
                 mode_luma, mode_chroma, ref_frames, mvs, bsize, tile_bo, skip, cfl,
                 tx_size, tx_type, mode_context, &mv_stack, rdo_type, true);
}

fn encode_partition_bottomup<T: Pixel, W: Writer>(
  fi: &FrameInvariants<T>, ts: &mut TileStateMut<'_, T>, cw: &mut ContextWriter,
  w_pre_cdef: &mut W, w_post_cdef: &mut W, bsize: BlockSize,
  tile_bo: BlockOffset, pmvs: &mut [[Option<MotionVector>; REF_FRAMES]; 5],
  ref_rd_cost: f64
) -> (RDOOutput) {
  let rdo_type = RDOType::PixelDistRealRate;
  let mut rd_cost = std::f64::MAX;
  let mut best_rd = std::f64::MAX;
  let mut rdo_output = RDOOutput {
    rd_cost,
    part_type: PartitionType::PARTITION_INVALID,
    part_modes: Vec::new()
  };

  if tile_bo.x >= cw.bc.blocks.cols() || tile_bo.y >= cw.bc.blocks.rows() {
    return rdo_output
  }

  let bsw = bsize.width_mi();
  let bsh = bsize.height_mi();
  let is_square = bsize.is_sqr();

  // Always split if the current partition is too large
  let must_split = (tile_bo.x + bsw as usize > ts.mi_width ||
                    tile_bo.y + bsh as usize > ts.mi_height ||
                    bsize.greater_than(BlockSize::BLOCK_64X64)) && is_square;

  // must_split overrides the minimum partition size when applicable
  let can_split = (bsize > fi.min_partition_size && is_square) || must_split;

  let mut best_partition = PartitionType::PARTITION_INVALID;

  let cw_checkpoint = cw.checkpoint();
  let w_pre_checkpoint = w_pre_cdef.checkpoint();
  let w_post_checkpoint = w_post_cdef.checkpoint();

  // Code the whole block
  if !must_split {
    let cost = if bsize.gte(BlockSize::BLOCK_8X8) && is_square {
      let w: &mut W = if cw.bc.cdef_coded {w_post_cdef} else {w_pre_cdef};
      let tell = w.tell_frac();
      cw.write_partition(w, tile_bo, PartitionType::PARTITION_NONE, bsize);
      (w.tell_frac() - tell) as f64 * fi.lambda / ((1 << OD_BITRES) as f64)
    } else {
      0.0
    };

    let pmv_idx = if bsize.greater_than(BlockSize::BLOCK_32X32) {
      0
    } else {
      ((tile_bo.x & 32) >> 5) + ((tile_bo.y & 32) >> 4) + 1
    };
    let spmvs = &mut pmvs[pmv_idx];

    let mode_decision = rdo_mode_decision(fi, ts, cw, bsize, tile_bo, spmvs);

    if !mode_decision.pred_mode_luma.is_intra() {
      // Fill the saved motion structure
      save_block_motion(
        ts, mode_decision.bsize, mode_decision.bo,
        mode_decision.ref_frames[0].to_index(), mode_decision.mvs[0]
      );
    }

    rd_cost = mode_decision.rd_cost + cost;

    best_partition = PartitionType::PARTITION_NONE;
    best_rd = rd_cost;
    rdo_output.part_modes.push(mode_decision.clone());

    if !can_split {
      encode_block_with_modes(fi, ts, cw, w_pre_cdef, w_post_cdef, bsize, tile_bo,
                              &mode_decision, rdo_type);
    }
  }

  // Test all partition types other than PARTITION_NONE by comparing their RD costs
  if can_split {
    debug_assert!(is_square);

    for &partition in RAV1E_PARTITION_TYPES {
      if partition == PartitionType::PARTITION_NONE { continue; }
      if fi.sequence.chroma_sampling == ChromaSampling::Cs422 &&
        partition == PartitionType::PARTITION_VERT { continue; }

      if must_split {
        let cbw = (ts.mi_width - tile_bo.x).min(bsw); // clipped block width, i.e. having effective pixels
        let cbh = (ts.mi_height - tile_bo.y).min(bsh);
        let mut split_vert = false;
        let mut split_horz = false;
        if cbw == bsw/2 && cbh == bsh { split_vert = true; }
        if cbh == bsh/2 && cbw == bsw { split_horz = true; }
        if !split_horz && partition == PartitionType::PARTITION_HORZ { continue; };
        if !split_vert && partition == PartitionType::PARTITION_VERT { continue; };
      }
      cw.rollback(&cw_checkpoint);
      w_pre_cdef.rollback(&w_pre_checkpoint);
      w_post_cdef.rollback(&w_post_checkpoint);

      let subsize = bsize.subsize(partition);
      let hbsw = subsize.width_mi(); // Half the block size width in blocks
      let hbsh = subsize.height_mi(); // Half the block size height in blocks
      let mut child_modes: Vec<RDOPartitionOutput> = Vec::new();
      rd_cost = 0.0;

      if bsize.gte(BlockSize::BLOCK_8X8) {
        let w: &mut W = if cw.bc.cdef_coded { w_post_cdef } else { w_pre_cdef };
        let tell = w.tell_frac();
        cw.write_partition(w, tile_bo, partition, bsize);
        rd_cost = (w.tell_frac() - tell) as f64 * fi.lambda
          / ((1 << OD_BITRES) as f64);
      }

      let four_partitions = [
        tile_bo,
        BlockOffset{ x: tile_bo.x + hbsw as usize, y: tile_bo.y },
        BlockOffset{ x: tile_bo.x, y: tile_bo.y + hbsh as usize },
        BlockOffset{ x: tile_bo.x + hbsw as usize, y: tile_bo.y + hbsh as usize }
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
          pmvs,//&best_decision.mvs[0]
          best_rd
        );
        let cost = child_rdo_output.rd_cost;
        assert!(cost >= 0.0);

        if cost != std::f64::MAX {
          rd_cost += cost;
          if fi.enable_early_exit && (rd_cost >= best_rd || rd_cost >= ref_rd_cost) {
            assert!(cost != std::f64::MAX);
            early_exit = true;
            break;
          } else if partition != PartitionType::PARTITION_SPLIT {
            child_modes.push(child_rdo_output.part_modes[0].clone());
          }
        }
      };

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

        if bsize.gte(BlockSize::BLOCK_8X8) {
          let w: &mut W = if cw.bc.cdef_coded { w_post_cdef } else { w_pre_cdef };
          cw.write_partition(w, tile_bo, best_partition, bsize);
        }
        for mode in rdo_output.part_modes.clone() {
          assert!(subsize == mode.bsize);

          if !mode.pred_mode_luma.is_intra() {
            save_block_motion(
              ts, mode.bsize, mode.bo,
              mode.ref_frames[0].to_index(), mode.mvs[0]
            );
          }

          // FIXME: redundant block re-encode
          encode_block_with_modes(fi, ts, cw, w_pre_cdef, w_post_cdef,
                                  mode.bsize, mode.bo, &mode, rdo_type);
        }
      }
  }

  assert!(best_partition != PartitionType::PARTITION_INVALID);

  if is_square && bsize.gte(BlockSize::BLOCK_8X8) &&
    (bsize == BlockSize::BLOCK_8X8 || best_partition != PartitionType::PARTITION_SPLIT) {
      cw.bc.update_partition_context(tile_bo, bsize.subsize(best_partition), bsize);
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
  bsize: BlockSize, tile_bo: BlockOffset, block_output: &Option<RDOOutput>,
  pmvs: &mut [[Option<MotionVector>; REF_FRAMES]; 5]
) {

  if tile_bo.x >= cw.bc.blocks.cols() || tile_bo.y >= cw.bc.blocks.rows() {
    return;
  }
  let bsw = bsize.width_mi();
  let bsh = bsize.height_mi();
  let is_square = bsize.is_sqr();
  let rdo_type = RDOType::PixelDistRealRate;

  // Always split if the current partition is too large
  let must_split = (tile_bo.x + bsw as usize > ts.mi_width ||
                    tile_bo.y + bsh as usize > ts.mi_height ||
                    bsize.greater_than(BlockSize::BLOCK_64X64)) && is_square;

  let mut rdo_output = block_output.clone().unwrap_or(RDOOutput {
    part_type: PartitionType::PARTITION_INVALID,
    rd_cost: std::f64::MAX,
    part_modes: Vec::new()
  });
  let partition: PartitionType;
  let mut split_vert = false;
  let mut split_horz = false;
  if must_split {
    let cbw = (ts.mi_width - tile_bo.x).min(bsw); // clipped block width, i.e. having effective pixels
    let cbh = (ts.mi_height - tile_bo.y).min(bsh);

    if cbw == bsw/2 && cbh == bsh &&
      fi.sequence.chroma_sampling != ChromaSampling::Cs422 { split_vert = true; }
    if cbh == bsh/2 && cbw == bsw { split_horz = true; }
  }

  if must_split && (!split_vert && !split_horz) {
    // Oversized blocks are split automatically
    partition = PartitionType::PARTITION_SPLIT;
  } else if must_split || (bsize > fi.min_partition_size && is_square) {
    debug_assert!(bsize.is_sqr());
    // Blocks of sizes within the supported range are subjected to a partitioning decision
    let mut partition_types: Vec<PartitionType> = Vec::new();
    if must_split {
      partition_types.push(PartitionType::PARTITION_SPLIT);
      if split_horz { partition_types.push(PartitionType::PARTITION_HORZ); };
      if split_vert { partition_types.push(PartitionType::PARTITION_VERT); };
    } else if bsize.width_log2() == fi.min_partition_size.width_log2() + 1 {
      partition_types.push(PartitionType::PARTITION_NONE);
      partition_types.push(PartitionType::PARTITION_SPLIT);
      partition_types.push(PartitionType::PARTITION_HORZ);

      if fi.sequence.chroma_sampling != ChromaSampling::Cs422 {
        partition_types.push(PartitionType::PARTITION_VERT);
      }
    } else {
      partition_types.push(PartitionType::PARTITION_NONE);
      partition_types.push(PartitionType::PARTITION_SPLIT);
    }
    rdo_output = rdo_partition_decision(fi, ts, cw,
                                        w_pre_cdef, w_post_cdef, bsize, tile_bo, &rdo_output, pmvs, &partition_types, rdo_type);
    partition = rdo_output.part_type;
  } else {
    // Blocks of sizes below the supported range are encoded directly
    partition = PartitionType::PARTITION_NONE;
  }

  assert!(PartitionType::PARTITION_NONE <= partition &&
          partition < PartitionType::PARTITION_INVALID);

  let subsize = bsize.subsize(partition);

  if bsize.gte(BlockSize::BLOCK_8X8) && is_square {
    let w: &mut W = if cw.bc.cdef_coded { w_post_cdef } else { w_pre_cdef };
    cw.write_partition(w, tile_bo, partition, bsize);
  }

  match partition {
    PartitionType::PARTITION_NONE => {
      let part_decision = if !rdo_output.part_modes.is_empty() {
        // The optimal prediction mode is known from a previous iteration
        rdo_output.part_modes[0].clone()
      } else {
        let pmv_idx = if bsize.greater_than(BlockSize::BLOCK_32X32) {
          0
        } else {
          ((tile_bo.x & 32) >> 5) + ((tile_bo.y & 32) >> 4) + 1
        };
        let spmvs = &mut pmvs[pmv_idx];

        // Make a prediction mode decision for blocks encoded with no rdo_partition_decision call (e.g. edges)
        rdo_mode_decision(fi, ts, cw, bsize, tile_bo, spmvs)
      };

      let mut mode_luma = part_decision.pred_mode_luma;
      let mut mode_chroma = part_decision.pred_mode_chroma;

      let cfl = part_decision.pred_cfl_params;
      let skip = part_decision.skip;
      let ref_frames = part_decision.ref_frames;
      let mvs = part_decision.mvs;
      let mut cdef_coded = cw.bc.cdef_coded;

      // NOTE: Cannot avoid calling rdo_tx_size_type() here again,
      // because, with top-down partition RDO, the neighnoring contexts
      // of current partition can change, i.e. neighboring partitions can split down more.
      let (tx_size, tx_type) =
        rdo_tx_size_type(fi, ts, cw, bsize, tile_bo, mode_luma, ref_frames, mvs, skip);

      let mut mv_stack = ArrayVec::<[CandidateMV; 9]>::new();
      let is_compound = ref_frames[1] != NONE_FRAME;
      let mode_context = cw.find_mvrefs(tile_bo, ref_frames, &mut mv_stack, bsize, fi, is_compound);

      // TODO: proper remap when is_compound is true
      if !mode_luma.is_intra() {
        if is_compound && mode_luma != PredictionMode::GLOBAL_GLOBALMV {
          let match0 = mv_stack[0].this_mv.row == mvs[0].row && mv_stack[0].this_mv.col == mvs[0].col;
          let match1 = mv_stack[0].comp_mv.row == mvs[1].row && mv_stack[0].comp_mv.col == mvs[1].col;

          mode_luma = if match0 && match1 {
            PredictionMode::NEAREST_NEARESTMV
          } else if match0 {
            PredictionMode::NEAREST_NEWMV
          } else if match1 {
            PredictionMode::NEW_NEARESTMV
          } else {
            PredictionMode::NEW_NEWMV
          };
          if mode_luma != PredictionMode::NEAREST_NEARESTMV && mvs[0].row == 0 && mvs[0].col == 0 &&
            mvs[1].row == 0 && mvs[1].col == 0 {
              mode_luma = PredictionMode::GLOBAL_GLOBALMV;
            }
          mode_chroma = mode_luma;
        } else if !is_compound && mode_luma != PredictionMode::GLOBALMV {
          mode_luma = PredictionMode::NEWMV;
          for (c, m) in mv_stack.iter().take(4)
            .zip([PredictionMode::NEARESTMV, PredictionMode::NEAR0MV,
                  PredictionMode::NEAR1MV, PredictionMode::NEAR2MV].iter()) {
              if c.this_mv.row == mvs[0].row && c.this_mv.col == mvs[0].col {
                mode_luma = *m;
              }
            }
          if mode_luma == PredictionMode::NEWMV && mvs[0].row == 0 && mvs[0].col == 0 {
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
          ts, part_decision.bsize, part_decision.bo,
          part_decision.ref_frames[0].to_index(), part_decision.mvs[0]
        );
      }

      // FIXME: every final block that has gone through the RDO decision process is encoded twice
      cdef_coded = encode_block_pre_cdef(&fi.sequence, ts, cw, if cdef_coded  {w_post_cdef} else {w_pre_cdef},
                                  bsize, tile_bo, skip);
      encode_block_post_cdef(fi, ts, cw, if cdef_coded  {w_post_cdef} else {w_pre_cdef},
                     mode_luma, mode_chroma, ref_frames, mvs, bsize, tile_bo, skip, cfl,
                     tx_size, tx_type, mode_context, &mv_stack, RDOType::PixelDistRealRate, true);
    },
    PARTITION_SPLIT |
    PARTITION_HORZ |
    PARTITION_VERT => {
      if !rdo_output.part_modes.is_empty() {
        // The optimal prediction modes for each split block is known from an rdo_partition_decision() call
        assert!(subsize != BlockSize::BLOCK_INVALID);

        for mode in rdo_output.part_modes {
          // Each block is subjected to a new splitting decision
          encode_partition_topdown(fi, ts, cw, w_pre_cdef, w_post_cdef, subsize, mode.bo,
                                   &Some(RDOOutput {
                                     rd_cost: mode.rd_cost,
                                     part_type: PartitionType::PARTITION_NONE,
                                     part_modes: vec![mode] }), pmvs);
        }
      }
      else {
        let hbsw = subsize.width_mi(); // Half the block size width in blocks
        let hbsh = subsize.height_mi(); // Half the block size height in blocks
        let four_partitions = [
          tile_bo,
          BlockOffset{ x: tile_bo.x + hbsw as usize, y: tile_bo.y },
          BlockOffset{ x: tile_bo.x, y: tile_bo.y + hbsh as usize },
          BlockOffset{ x: tile_bo.x + hbsw as usize, y: tile_bo.y + hbsh as usize }
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
            pmvs
          );
        });
      }
    },
    _ => unreachable!(),
  }

  if is_square && bsize.gte(BlockSize::BLOCK_8X8) &&
    (bsize == BlockSize::BLOCK_8X8 || partition != PartitionType::PARTITION_SPLIT) {
      cw.bc.update_partition_context(tile_bo, subsize, bsize);
    }
}

#[inline(always)]
fn build_coarse_pmvs<T: Pixel>(fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>) -> Vec<[Option<MotionVector>; REF_FRAMES]> {
  assert!(!fi.sequence.use_128x128_superblock);
  if ts.mi_width >= 16 && ts.mi_height >= 16 {
    let mut frame_pmvs = Vec::with_capacity(ts.sb_width * ts.sb_height);
    for sby in 0..ts.sb_height {
      for sbx in 0..ts.sb_width {
        let sbo = SuperBlockOffset { x: sbx, y: sby };
        let bo = sbo.block_offset(0, 0);
        let mut pmvs: [Option<MotionVector>; REF_FRAMES] = [None; REF_FRAMES];
        for i in 0..INTER_REFS_PER_FRAME {
          let r = fi.ref_frames[i] as usize;
          if pmvs[r].is_none() {
            pmvs[r] = estimate_motion_ss4(fi, ts, BlockSize::BLOCK_64X64, r, bo);
          }
        }
        frame_pmvs.push(pmvs);
      }
    }
    frame_pmvs
  } else {
    // the block use for motion estimation would be smaller than the whole image
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

fn encode_tile_group<T: Pixel>(fi: &FrameInvariants<T>, fs: &mut FrameState<T>) -> Vec<u8> {
  let mut blocks = FrameBlocks::new(fi.w_in_b, fi.h_in_b);
  let ti = &fi.tiling;

  let initial_cdf = get_initial_cdfcontext(fi);
  let mut cdfs = vec![initial_cdf; ti.tile_count()];

  let (raw_tiles, rdo_trackers): (Vec<_>, Vec<_>) = ti
    .tile_iter_mut(fs, &mut blocks)
    .zip(cdfs.iter_mut())
    .collect::<Vec<_>>()
    .into_par_iter()
    .map(|(mut ctx, cdf)| {
      let raw = encode_tile(fi, &mut ctx.ts, cdf, &mut ctx.tb);
      (raw, ctx.ts.rdo)
    })
    .unzip();

  /* TODO: Don't apply if lossless */
  deblock_filter_optimize(fi, fs, &blocks);
  if fs.deblock.levels[0] != 0 || fs.deblock.levels[1] != 0 {
    deblock_filter_frame(fi, fs, &blocks);
  }

  // Until the loop filters are pipelined, we'll need to keep
  // around a copy of both the pre- and post-cdef frame.
  let pre_cdef_frame = fs.rec.clone();

  /* TODO: Don't apply if lossless */
  if fi.sequence.enable_cdef {
    cdef_filter_frame(fi, &mut fs.rec, &blocks);
  }
  /* TODO: Don't apply if lossless */
  if fi.sequence.enable_restoration {
    fs.restoration.lrf_filter_frame(&mut fs.rec, &pre_cdef_frame, &fi);
  }

  if fi.config.train_rdo {
    eprintln!("train rdo");
    for rdo_tracker in &rdo_trackers {
      fs.t.merge_in(&rdo_tracker);
    }
    if let Ok(mut file) = File::open("rdo.dat") {
      let mut data = vec![];
      file.read_to_end(&mut data).unwrap();
      fs.t.merge_in(&deserialize(data.as_slice()).unwrap());
    }
    let mut rdo_file = File::create("rdo.dat").unwrap();
    rdo_file.write_all(&serialize(&fs.t).unwrap()).unwrap();
    fs.t.print_code();
  }

  let (idx_max, max_len) = raw_tiles
    .iter()
    .map(Vec::len)
    .enumerate()
    .max_by_key(|&(_, len)| len)
    .unwrap();

  // use the biggest tile (in bytes) for CDF update
  fs.context_update_tile_id = idx_max;
  fs.cdfs = cdfs[idx_max];
  fs.cdfs.reset_counts();

  let max_tile_size_bytes = ((max_len as u32).ilog() + 7) / 8;
  debug_assert!(max_tile_size_bytes > 0 && max_tile_size_bytes <= 4);
  fs.max_tile_size_bytes = max_tile_size_bytes;

  build_raw_tile_group(ti, &raw_tiles, max_tile_size_bytes)
}

fn build_raw_tile_group(ti: &TilingInfo, raw_tiles: &[Vec<u8>], max_tile_size_bytes: u32) -> Vec<u8> {
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
    bw.write_bytes(&raw_tile).unwrap();
  }
  raw
}

fn encode_tile<'a, T: Pixel>(
  fi: &FrameInvariants<T>,
  ts: &mut TileStateMut<'_, T>,
  fc: &'a mut CDFContext,
  blocks: &'a mut TileBlocksMut<'a>,
) -> Vec<u8> {
  let mut w = WriterEncoder::new();

  let estimate_motion_ss2 = if fi.config.speed_settings.diamond_me {
    crate::me::DiamondSearch::estimate_motion_ss2
  } else {
    crate::me::FullSearch::estimate_motion_ss2
  };

  let bc = BlockContext::new(blocks);
  // For now, restoration unit size is locked to superblock size.
  let mut cw = ContextWriter::new(fc, bc);

  let tile_pmvs = build_coarse_pmvs(fi, ts);

  // main loop
  for sby in 0..ts.sb_height {
    cw.bc.reset_left_contexts();

    for sbx in 0..ts.sb_width {
      let mut w_pre_cdef = WriterRecorder::new();
      let mut w_post_cdef = WriterRecorder::new();
      let tile_sbo = SuperBlockOffset { x: sbx, y: sby };
      let tile_bo = tile_sbo.block_offset(0, 0);
      cw.bc.cdef_coded = false;
      cw.bc.code_deltas = fi.delta_q_present;

      // Do subsampled ME
      let mut pmvs: [[Option<MotionVector>; REF_FRAMES]; 5] = [[None; REF_FRAMES]; 5];
      if ts.mi_width >= 8 && ts.mi_height >= 8 {
        for i in 0..INTER_REFS_PER_FRAME {
          let r = fi.ref_frames[i] as usize;
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
                fi, ts, BlockSize::BLOCK_32X32, r, tile_sbo.block_offset(0, 0), &[Some(pmv), pmv_w, pmv_n], i
              );
              pmvs[2][r] = estimate_motion_ss2(
                fi, ts, BlockSize::BLOCK_32X32, r, tile_sbo.block_offset(8, 0), &[Some(pmv), pmv_e, pmv_n], i
              );
              pmvs[3][r] = estimate_motion_ss2(
                fi, ts, BlockSize::BLOCK_32X32, r, tile_sbo.block_offset(0, 8), &[Some(pmv), pmv_w, pmv_s], i
              );
              pmvs[4][r] = estimate_motion_ss2(
                fi, ts, BlockSize::BLOCK_32X32, r, tile_sbo.block_offset(8, 8), &[Some(pmv), pmv_e, pmv_s], i
              );

              if let Some(mv) = pmvs[1][r] {
                save_block_motion(ts, BlockSize::BLOCK_32X32, tile_sbo.block_offset(0, 0), i, mv);
              }
              if let Some(mv) = pmvs[2][r] {
                save_block_motion(ts, BlockSize::BLOCK_32X32, tile_sbo.block_offset(8, 0), i, mv);
              }
              if let Some(mv) = pmvs[3][r] {
                save_block_motion(ts, BlockSize::BLOCK_32X32, tile_sbo.block_offset(0, 8), i, mv);
              }
              if let Some(mv) = pmvs[4][r] {
                save_block_motion(ts, BlockSize::BLOCK_32X32, tile_sbo.block_offset(8, 8), i, mv);
              }
            }
          }
        }
      }

      // Encode SuperBlock
      if fi.config.speed_settings.encode_bottomup {
        encode_partition_bottomup(fi, ts, &mut cw,
                                  &mut w_pre_cdef, &mut w_post_cdef,
                                  BlockSize::BLOCK_64X64, tile_bo, &mut pmvs, std::f64::MAX);
      }
      else {
        encode_partition_topdown(fi, ts, &mut cw,
                                 &mut w_pre_cdef, &mut w_post_cdef,
                                 BlockSize::BLOCK_64X64, tile_bo, &None, &mut pmvs);
      }

      // CDEF has to be decided before loop restoration, but coded after.
      // loop restoration must be decided last but coded before anything else.
      if cw.bc.cdef_coded || fi.sequence.enable_restoration {
        rdo_loop_decision(tile_sbo, fi, ts, &mut cw, &mut w);
      }

      if fi.sequence.enable_restoration {
        cw.write_lrf(&mut w, fi, &mut ts.restoration, tile_sbo);
      }

      // Once loop restoration is coded, we can replay the initial block bits
      w_pre_cdef.replay(&mut w);

      if cw.bc.cdef_coded {
        // CDEF index must be written in the middle, we can code it now
        let cdef_index = cw.bc.blocks.get_cdef(tile_sbo);
        cw.write_cdef(&mut w, cdef_index, fi.cdef_bits);
        // ...and then finally code what comes after the CDEF index
        w_post_cdef.replay(&mut w);
      }
    }
  }

  w.done()
}

#[allow(unused)]
fn write_tile_group_header(tile_start_and_end_present_flag: bool) ->
  Vec<u8> {
    let mut buf = Vec::new();
    {
      let mut bw = BitWriter::endian(&mut buf, BigEndian);
      bw.write_bit(tile_start_and_end_present_flag).unwrap();
      bw.byte_align().unwrap();
    }
    buf.clone()
  }

// Write a packet containing only the placeholder that tells the decoder
// to present the already decoded frame present at `frame_to_show_map_idx`
//
// See `av1-spec` Section 6.8.2 and 7.18.
pub fn encode_show_existing_frame<T: Pixel>(
  fi: &mut FrameInvariants<T>, fs: &mut FrameState<T>
) -> Vec<u8> {
  debug_assert!(fi.show_existing_frame);
  let mut packet = Vec::new();

  write_obus(&mut packet, fi, fs).unwrap();
  let map_idx = fi.frame_to_show_map_idx as usize;
  if let Some(ref rec) = fi.rec_buffer.frames[map_idx] {
    for p in 0..3 {
      fs.rec.planes[p].data.copy_from_slice(&rec.frame.planes[p].data);
    }
  }
  packet
}

pub fn encode_frame<T: Pixel>(
  fi: &FrameInvariants<T>, fs: &mut FrameState<T>
) -> Vec<u8> {
  debug_assert!(!fi.show_existing_frame);
  let mut packet = Vec::new();

  fs.input_hres.downsample_from(&fs.input.planes[0]);
  fs.input_hres.pad(fi.width, fi.height);
  fs.input_qres.downsample_from(&fs.input_hres);
  fs.input_qres.pad(fi.width, fi.height);

  segmentation_optimize(fi, fs);

  let tile_group = encode_tile_group(fi, fs);

  write_obus(&mut packet, fi, fs).unwrap();
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

pub fn update_rec_buffer<T: Pixel>(fi: &mut FrameInvariants<T>, fs: FrameState<T>) {
  let rfs = Arc::new(
    ReferenceFrame {
      order_hint: fi.order_hint,
      frame: fs.rec,
      input_hres: fs.input_hres,
      input_qres: fs.input_qres,
      cdfs: fs.cdfs,
      frame_mvs: fs.frame_mvs,
    }
  );
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
    assert_eq!(RAV1E_PARTITION_TYPES[RAV1E_PARTITION_TYPES.len() - 1],
               PartitionType::PARTITION_SPLIT);
  }
}
