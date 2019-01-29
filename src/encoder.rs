// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use api::*;
use cdef::*;
use context::*;
use deblock::*;
use ec::*;
use lrf::*;
use mc::*;
use me::*;
use partition::*;
use plane::*;
use quantize::*;
use rdo::*;
use segmentation::*;
use transform::*;
use util::*;
use partition::PartitionType::*;

use bitstream_io::{BitWriter, BigEndian, LittleEndian};
use std;
use std::{fmt, io};
use std::io::Write;
use std::rc::Rc;
use std::sync::Arc;

extern {
    pub fn av1_rtcd();
    pub fn aom_dsp_rtcd();
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub planes: [Plane; 3]
}

const FRAME_MARGIN: usize = 16 + SUBPEL_FILTER_SIZE;

impl Frame {
    pub fn new(width: usize, height: usize, chroma_sampling: ChromaSampling) -> Frame {
        let chroma_sampling_period = chroma_sampling.sampling_period();
        let (chroma_width, chroma_height, chroma_padding, chroma_xdec, chroma_ydec) = (
            width / chroma_sampling_period.0,
            height / chroma_sampling_period.1,
            MAX_SB_SIZE / chroma_sampling_period.0 + FRAME_MARGIN,
            chroma_sampling_period.0 - 1,
            chroma_sampling_period.1 - 1
        );

        Frame {
            planes: [
                Plane::new(
                    width, height,
                    0, 0,
                    MAX_SB_SIZE + FRAME_MARGIN, MAX_SB_SIZE + FRAME_MARGIN
                ),
                Plane::new(
                    chroma_width, chroma_height,
                    chroma_xdec, chroma_ydec,
                    chroma_padding, chroma_padding
                ),
                Plane::new(
                    chroma_width, chroma_height,
                    chroma_xdec, chroma_ydec,
                    chroma_padding, chroma_padding
                )
            ]
        }
    }

    pub fn pad(&mut self, w: usize, h: usize) {
        for p in self.planes.iter_mut() {
            p.pad(w, h);
        }
    }

    /// Returns a `PixelIter` containing the data of this frame's planes in YUV format.
    /// Each point in the `PixelIter` is a triple consisting of a Y, U, and V component.
    /// The `PixelIter` is laid out as contiguous rows, e.g. to get a given 0-indexed row
    /// you could use `data.skip(width * row_idx).take(width)`.
    ///
    /// This data retains any padding, e.g. it uses the width and height specifed in
    /// the Y-plane's `cfg` struct, and not the display width and height specied in
    /// `FrameInvariants`.
    pub fn iter(&self) -> PixelIter {
      PixelIter::new(&self.planes)
    }
}

#[derive(Debug)]
pub struct PixelIter<'a> {
  planes: &'a [Plane; 3],
  y: usize,
  x: usize,
}

impl<'a> PixelIter<'a> {
  pub fn new(planes: &'a [Plane; 3]) -> Self {
    PixelIter {
      planes,
      y: 0,
      x: 0,
    }
  }

  fn width(&self) -> usize {
    self.planes[0].cfg.width
  }

  fn height(&self) -> usize {
    self.planes[0].cfg.height
  }
}

impl<'a> Iterator for PixelIter<'a> {
  type Item = (u16, u16, u16);

  fn next(&mut self) -> Option<<Self as Iterator>::Item> {
    if self.y == self.height() - 1 && self.x == self.width() - 1 {
      return None;
    }
    let pixel = (
      self.planes[0].p(self.x, self.y),
      self.planes[1].p(self.x / 2, self.y / 2),
      self.planes[2].p(self.x / 2, self.y / 2),
    );
    if self.x == self.width() - 1 {
      self.x = 0;
      self.y += 1;
    } else {
      self.x += 1;
    }
    Some(pixel)
  }
}

#[derive(Debug, Clone)]
pub struct ReferenceFrame {
  pub order_hint: u32,
  pub frame: Frame,
  pub input_hres: Plane,
  pub input_qres: Plane,
  pub cdfs: CDFContext
}

#[derive(Debug, Clone)]
pub struct ReferenceFramesSet {
    pub frames: [Option<Rc<ReferenceFrame>>; (REF_FRAMES as usize)],
    pub deblock: [DeblockState; (REF_FRAMES as usize)]
}

impl ReferenceFramesSet {
    pub fn new() -> ReferenceFramesSet {
        ReferenceFramesSet {
            frames: Default::default(),
            deblock: Default::default()
        }
    }
}

const MAX_NUM_TEMPORAL_LAYERS: usize = 8;
const MAX_NUM_SPATIAL_LAYERS: usize = 4;
const MAX_NUM_OPERATING_POINTS: usize = MAX_NUM_TEMPORAL_LAYERS * MAX_NUM_SPATIAL_LAYERS;

pub const PRIMARY_REF_NONE: u32 = 7;
const PRIMARY_REF_BITS: u32 = 3;

arg_enum!{
    #[derive(Copy, Clone, Debug, PartialEq)]
    #[repr(C)]
    pub enum Tune {
        Psnr,
        Psychovisual
    }
}

impl Default for Tune {
    fn default() -> Self {
        Tune::Psnr
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub enum ChromaSampling {
    Cs420,
    Cs422,
    Cs444
}

impl Default for ChromaSampling {
    fn default() -> Self {
        ChromaSampling::Cs420
    }
}

impl ChromaSampling {
    // Provides the sampling period in the horizontal and vertical axes.
    pub fn sampling_period(self) -> (usize, usize) {
        match self {
            ChromaSampling::Cs420 => (2, 2),
            ChromaSampling::Cs422 => (2, 1),
            ChromaSampling::Cs444 => (1, 1)
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub enum ChromaSamplePosition {
    Unknown,
    Vertical,
    Colocated
}

impl Default for ChromaSamplePosition {
    fn default() -> Self {
        ChromaSamplePosition::Unknown
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Sequence {
    // OBU Sequence header of AV1
    pub profile: u8,
    pub num_bits_width: u32,
    pub num_bits_height: u32,
    pub bit_depth: usize,
    pub chroma_sampling: ChromaSampling,
    pub chroma_sample_position: ChromaSamplePosition,
    pub color_description: Option<ColorDescription>,
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
    pub monochrome: bool,                  // Monochrome video
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
    pub fn new(info: &FrameInfo) -> Sequence {
        let width_bits = 32 - (info.width as u32).leading_zeros();
        let height_bits = 32 - (info.height as u32).leading_zeros();
        assert!(width_bits <= 16);
        assert!(height_bits <= 16);

        let profile = if info.bit_depth == 12 {
            2
        } else if info.chroma_sampling == ChromaSampling::Cs444 {
            1
        } else {
            0
        };

        let mut operating_point_idc = [0 as u16; MAX_NUM_OPERATING_POINTS];
        let mut level = [[1, 2 as usize]; MAX_NUM_OPERATING_POINTS];
        let mut tier = [0 as usize; MAX_NUM_OPERATING_POINTS];

        for i in 0..MAX_NUM_OPERATING_POINTS {
            operating_point_idc[i] = 0;
            level[i][0] = 1;	// minor
            level[i][1] = 2;	// major
            tier[i] = 0;
        }

        Sequence {
            profile: profile,
            num_bits_width: width_bits,
            num_bits_height: height_bits,
            bit_depth: info.bit_depth,
            chroma_sampling: info.chroma_sampling,
            chroma_sample_position: info.chroma_sample_position,
            color_description: None,
            max_frame_width: info.width as u32,
            max_frame_height: info.height as u32,
            frame_id_numbers_present_flag: false,
            frame_id_length: 0,
            delta_frame_id_length: 0,
            use_128x128_superblock: false,
            order_hint_bits_minus_1: 5,
            force_screen_content_tools: 0,
            force_integer_mv: 2,
            still_picture: false,
            reduced_still_picture_hdr: false,
            monochrome: false,
            enable_intra_edge_filter: false,
            enable_interintra_compound: false,
            enable_masked_compound: false,
            enable_dual_filter: false,
            enable_order_hint: true,
            enable_jnt_comp: false,
            enable_ref_frame_mvs: false,
            enable_warped_motion: false,
            enable_superres: false,
            enable_cdef: true,
            enable_restoration: true,
            operating_points_cnt_minus_1: 0,
            operating_point_idc: operating_point_idc,
            display_model_info_present_flag: false,
            decoder_model_info_present_flag: false,
            level: level,
            tier: tier,
            film_grain_params_present: false,
            separate_uv_delta_q: false,
        }
    }

    pub fn get_relative_dist(&self, a: u32, b: u32) -> i32 {
        let diff = a as i32 - b as i32;
        let m = 1 << self.order_hint_bits_minus_1;
        (diff & (m - 1)) - (diff & m)
    }

    pub fn get_skip_mode_allowed(&self, fi: &FrameInvariants, reference_select: bool) -> bool {
      if fi.intra_only || !reference_select || !self.enable_order_hint {
        false
      } else {
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
            } else if self.get_relative_dist(ref_hint, fi.order_hint) > 0 {
              if backward_idx < 0 || self.get_relative_dist(ref_hint, backward_hint) > 0 {
                backward_idx = i as isize;
                backward_hint = ref_hint;
              }
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
              if self.get_relative_dist(ref_hint, forward_hint) < 0 {
                if second_forward_idx < 0 || self.get_relative_dist(ref_hint, second_forward_hint) > 0 {
                  second_forward_idx = i as isize;
                  second_forward_hint = ref_hint;
                }
              }
            }
          }
          if second_forward_idx < 0 {
            false
          } else {
            // set skip_mode_frame
            true
          }
        }
      }
    }
}

#[derive(Debug)]
pub struct FrameState {
    pub input: Arc<Frame>,
    pub input_hres: Plane, // half-resolution version of input luma
    pub input_qres: Plane, // quarter-resolution version of input luma
    pub rec: Frame,
    pub qc: QuantizationContext,
    pub cdfs: CDFContext,
    pub deblock: DeblockState,
    pub segmentation: SegmentationState,
    pub restoration: RestorationState,
}

impl FrameState {
    pub fn new(fi: &FrameInvariants) -> FrameState {
        FrameState::new_with_frame(fi, Arc::new(Frame::new(
            fi.padded_w, fi.padded_h, fi.sequence.chroma_sampling)))
    }

    pub fn new_with_frame(fi: &FrameInvariants, frame: Arc<Frame>) -> FrameState {
        let rs = RestorationState::new(fi, &frame);
        FrameState {
            input: frame,
            input_hres: Plane::new(
                fi.padded_w/2, fi.padded_h/2,
                1, 1,
                (MAX_SB_SIZE + FRAME_MARGIN) / 2, (MAX_SB_SIZE + FRAME_MARGIN) / 2
            ),
            input_qres: Plane::new(
                fi.padded_w/4, fi.padded_h/4,
                2, 2,
                (MAX_SB_SIZE + FRAME_MARGIN) / 4, (MAX_SB_SIZE + FRAME_MARGIN) / 4
            ),
            rec: Frame::new(fi.padded_w, fi.padded_h, fi.sequence.chroma_sampling),
            qc: Default::default(),
            cdfs: CDFContext::new(0),
            deblock: Default::default(),
            segmentation: Default::default(),
            restoration: rs,
        }
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
pub struct FrameInvariants {
    pub sequence: Sequence,
    pub width: usize,
    pub height: usize,
    pub padded_w: usize,
    pub padded_h: usize,
    pub sb_width: usize,
    pub sb_height: usize,
    pub w_in_b: usize,
    pub h_in_b: usize,
    pub number: u64,
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
    pub globalmv_transformation_type: [GlobalMVMode; ALTREF_FRAME + 1],
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
    pub rec_buffer: ReferenceFramesSet,
    pub base_q_idx: u8,
    pub dc_delta_q: [i8; 3],
    pub ac_delta_q: [i8; 3],
    pub me_range_scale: u8,
    pub use_tx_domain_distortion: bool,
    pub inter_cfg: Option<InterPropsConfig>,
}

impl FrameInvariants {
    pub fn new(width: usize, height: usize,
        config: EncoderConfig, sequence: Sequence) -> FrameInvariants {
        // Speed level decides the minimum partition size, i.e. higher speed --> larger min partition size,
        // with exception that SBs on right or bottom frame borders split down to BLOCK_4X4.
        // At speed = 0, RDO search is exhaustive.
        let mut min_partition_size = config.speed_settings.min_block_size;

        if config.tune == Tune::Psychovisual {
            if min_partition_size < BlockSize::BLOCK_8X8 {
                // TODO: Display message that min partition size is enforced to 8x8
                min_partition_size = BlockSize::BLOCK_8X8;
                println!("If tune=Psychovisual is used, min partition size is enforced to 8x8");
            }
        }
        let use_reduced_tx_set = config.speed_settings.reduced_tx_set;
        let use_tx_domain_distortion = config.tune == Tune::Psnr && config.speed_settings.tx_domain_distortion;

        FrameInvariants {
            sequence,
            width,
            height,
            padded_w: width.align_power_of_two(3),
            padded_h: height.align_power_of_two(3),
            sb_width: width.align_power_of_two_and_shift(6),
            sb_height: height.align_power_of_two_and_shift(6),
            w_in_b: 2 * width.align_power_of_two_and_shift(3), // MiCols, ((width+7)/8)<<3 >> MI_SIZE_LOG2
            h_in_b: 2 * height.align_power_of_two_and_shift(3), // MiRows, ((height+7)/8)<<3 >> MI_SIZE_LOG2
            number: 0,
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
            globalmv_transformation_type: [GlobalMVMode::IDENTITY; ALTREF_FRAME + 1],
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
            config,
            ref_frames: [0; INTER_REFS_PER_FRAME],
            ref_frame_sign_bias: [false; INTER_REFS_PER_FRAME],
            rec_buffer: ReferenceFramesSet::new(),
            base_q_idx: config.quantizer as u8,
            dc_delta_q: [0; 3],
            ac_delta_q: [0; 3],
            me_range_scale: 1,
            use_tx_domain_distortion: use_tx_domain_distortion,
            inter_cfg: None,
        }
    }

  pub fn new_key_frame(previous_fi: &Self, segment_start_frame: u64) -> Self {
    let mut fi = previous_fi.clone();
    fi.frame_type = FrameType::KEY;
    fi.intra_only = true;
    fi.inter_cfg = None;
    fi.order_hint = 0;
    fi.refresh_frame_flags = ALL_REF_FRAMES_MASK;
    fi.show_frame = true;
    fi.show_existing_frame = false;
    fi.frame_to_show_map_idx = 0;
    let q_boost = 15;
    fi.base_q_idx = (fi.config.quantizer.max(1 + q_boost).min(255 + q_boost) - q_boost) as u8;
    fi.cdef_bits = 3;
    fi.primary_ref_frame = PRIMARY_REF_NONE;
    fi.number = segment_start_frame;
    for i in 0..INTER_REFS_PER_FRAME {
      fi.ref_frames[i] = 0;
    }
    fi
  }

  fn apply_inter_props_cfg(&mut self, idx_in_segment: u64) {
    let reorder = !self.config.low_latency;
    let multiref = reorder || self.config.speed_settings.multiref;

    let pyramid_depth = if reorder { 2 } else { 0 };
    let group_src_len = 1 << pyramid_depth;
    let group_len = group_src_len + if reorder { pyramid_depth } else { 0 };

    let idx_in_group = (idx_in_segment - 1) % group_len;
    let group_idx = (idx_in_segment - 1) / group_len;

    self.inter_cfg = Some(InterPropsConfig {
      reorder,
      multiref,
      pyramid_depth,
      group_src_len,
      group_len,
      idx_in_group,
      group_idx,
    })
  }

  /// Returns the created FrameInvariants along with a bool indicating success.
  /// This interface provides simpler usage, because we always need the produced
  /// FrameInvariants regardless of success or failure.
  pub fn new_inter_frame(previous_fi: &Self, segment_start_frame: u64, idx_in_segment: u64, next_keyframe: u64) -> (Self, bool) {
    let mut fi = previous_fi.clone();
    fi.frame_type = FrameType::INTER;
    fi.intra_only = false;
    fi.apply_inter_props_cfg(idx_in_segment);
    let inter_cfg = fi.inter_cfg.unwrap();

    fi.order_hint = (inter_cfg.group_src_len * inter_cfg.group_idx +
      if inter_cfg.reorder && inter_cfg.idx_in_group < inter_cfg.pyramid_depth {
        inter_cfg.group_src_len >> inter_cfg.idx_in_group
      } else {
        inter_cfg.idx_in_group - inter_cfg.pyramid_depth + 1
      }) as u32;
    let number = segment_start_frame + fi.order_hint as u64;
    if number >= next_keyframe {
      fi.show_existing_frame = false;
      fi.show_frame = false;
      return (fi, false);
    }

    fn pos_to_lvl(pos: u64, pyramid_depth: u64) -> u64 {
      // Derive level within pyramid for a frame with a given coding order position
      // For example, with a pyramid of depth 2, the 2 least significant bits of the
      // position determine the level:
      // 00 -> 0
      // 01 -> 2
      // 10 -> 1
      // 11 -> 2
      pyramid_depth - (pos | (1 << pyramid_depth)).trailing_zeros() as u64
    }

    let lvl = if !inter_cfg.reorder {
      0
    } else if inter_cfg.idx_in_group < inter_cfg.pyramid_depth {
      inter_cfg.idx_in_group
    } else {
      pos_to_lvl(inter_cfg.idx_in_group - inter_cfg.pyramid_depth + 1, inter_cfg.pyramid_depth)
    };

    // Frames with lvl == 0 are stored in slots 0..4 and frames with higher values
    // of lvl in slots 4..8
    let slot_idx = if lvl == 0 {
      (fi.order_hint >> inter_cfg.pyramid_depth) % 4 as u32
    } else {
      3 + lvl as u32
    };
    fi.show_frame = !inter_cfg.reorder || inter_cfg.idx_in_group >= inter_cfg.pyramid_depth;
    fi.show_existing_frame = fi.show_frame && inter_cfg.reorder &&
      (inter_cfg.idx_in_group - inter_cfg.pyramid_depth + 1).count_ones() == 1 &&
      inter_cfg.idx_in_group != inter_cfg.pyramid_depth;
    fi.frame_to_show_map_idx = slot_idx;
    fi.refresh_frame_flags = if fi.show_existing_frame {
      0
    } else {
      1 << slot_idx
    };

    let q_drop = 15 * lvl as usize;
    fi.base_q_idx = (fi.config.quantizer.min(255 - q_drop) + q_drop) as u8;
    fi.cdef_bits = 3 - ((fi.base_q_idx.max(128) - 128) >> 5);
    let second_ref_frame = if !inter_cfg.multiref {
      NONE_FRAME
    } else if !inter_cfg.reorder || inter_cfg.idx_in_group == 0 {
      LAST2_FRAME
    } else {
      ALTREF_FRAME
    };
    let ref_in_previous_group = LAST3_FRAME;

    // reuse probability estimates from previous frames only in top level frames
    fi.primary_ref_frame = if lvl > 0 { PRIMARY_REF_NONE } else { (ref_in_previous_group - LAST_FRAME) as u32 };

    for i in 0..INTER_REFS_PER_FRAME {
      fi.ref_frames[i] = if lvl == 0 {
        if i == second_ref_frame - LAST_FRAME {
          (slot_idx + 4 - 2) as u8 % 4
        } else {
          (slot_idx + 4 - 1) as u8 % 4
        }
      } else {
        if i == second_ref_frame - LAST_FRAME {
          let oh = fi.order_hint + (inter_cfg.group_src_len as u32 >> lvl);
          let lvl2 = pos_to_lvl(oh as u64, inter_cfg.pyramid_depth);
          if lvl2 == 0 {
            ((oh >> inter_cfg.pyramid_depth) % 4) as u8
          } else {
            3 + lvl2 as u8
          }
        } else if i == ref_in_previous_group - LAST_FRAME {
          if lvl == 0 {
            (slot_idx + 4 - 1) as u8 % 4
          } else {
            slot_idx as u8
          }
        } else {
          let oh = fi.order_hint - (inter_cfg.group_src_len as u32 >> lvl);
          let lvl1 = pos_to_lvl(oh as u64, inter_cfg.pyramid_depth);
          if lvl1 == 0 {
            ((oh >> inter_cfg.pyramid_depth) % 4) as u8
          } else {
            3 + lvl1 as u8
          }
        }
      }
    }

    fi.reference_mode = if inter_cfg.multiref && inter_cfg.reorder && inter_cfg.idx_in_group != 0 {
      ReferenceMode::SELECT
    } else {
      ReferenceMode::SINGLE
    };
    fi.number = number;
    fi.me_range_scale = (inter_cfg.group_src_len >> lvl) as u8;
    (fi, true)
  }
}

impl fmt::Display for FrameInvariants {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Frame {} - {}", self.number, self.frame_type)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InterPropsConfig {
  pub reorder: bool,
  pub multiref: bool,
  pub pyramid_depth: u64,
  pub group_src_len: u64,
  pub group_len: u64,
  pub idx_in_group: u64,
  pub group_idx: u64,
}

#[allow(dead_code,non_camel_case_types)]
#[derive(Debug,PartialEq,Clone,Copy)]
#[repr(C)]
pub enum FrameType {
    KEY,
    INTER,
    INTRA_ONLY,
    SWITCH,
}

//const REFERENCE_MODES: usize = 3;

#[allow(dead_code,non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReferenceMode {
  SINGLE = 0,
  COMPOUND = 1,
  SELECT = 2,
}

pub const ALL_REF_FRAMES_MASK: u32 = (1 << REF_FRAMES) - 1;

impl fmt::Display for FrameType{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FrameType::KEY => write!(f, "Key frame"),
            FrameType::INTER => write!(f, "Inter frame"),
            FrameType::INTRA_ONLY => write!(f, "Intra only frame"),
            FrameType::SWITCH => write!(f, "Switching frame"),
        }
    }
}

pub fn write_ivf_header(output_file: &mut dyn io::Write, width: usize, height: usize, num: usize, den: usize) {
    let mut bw = BitWriter::endian(output_file, LittleEndian);
    bw.write_bytes(b"DKIF").unwrap();
    bw.write(16, 0).unwrap(); // version
    bw.write(16, 32).unwrap(); // version
    bw.write_bytes(b"AV01").unwrap();
    bw.write(16, width as u16).unwrap();
    bw.write(16, height as u16).unwrap();
    bw.write(32, num as u32).unwrap();
    bw.write(32, den as u32).unwrap();
    bw.write(32, 0).unwrap();
    bw.write(32, 0).unwrap();
}

pub fn write_ivf_frame(output_file: &mut dyn io::Write, pts: u64, data: &[u8]) {
    let mut bw = BitWriter::endian(output_file, LittleEndian);
    bw.write(32, data.len() as u32).unwrap();
    bw.write(64, pts).unwrap();
    bw.write_bytes(data).unwrap();
}

trait UncompressedHeader {
    // Start of OBU Headers
    fn write_obu_header(&mut self, obu_type: OBU_Type, obu_extension: u32)
            -> io::Result<()>;
    fn write_sequence_header_obu(&mut self, fi: &mut FrameInvariants)
            -> io::Result<()>;
    fn write_frame_header_obu(&mut self, fi: &FrameInvariants, fs: &FrameState)
          -> io::Result<()>;
    fn write_sequence_header(&mut self, fi: &mut FrameInvariants)
                                    -> io::Result<()>;
    fn write_color_config(&mut self, seq: &mut Sequence) -> io::Result<()>;
    // End of OBU Headers

    fn write_frame_size(&mut self, fi: &FrameInvariants) -> io::Result<()>;
    fn write_deblock_filter_a(&mut self, fi: &FrameInvariants, deblock: &DeblockState) -> io::Result<()>;
    fn write_deblock_filter_b(&mut self, fi: &FrameInvariants, deblock: &DeblockState) -> io::Result<()>;
    fn write_frame_cdef(&mut self, fi: &FrameInvariants) -> io::Result<()>;
    fn write_frame_lrf(&mut self, fi: &FrameInvariants, rs: &RestorationState) -> io::Result<()>;
    fn write_segment_data(&mut self, fi: &FrameInvariants, segmentation: &SegmentationState) -> io::Result<()>;
    fn write_delta_q(&mut self, delta_q: i8) -> io::Result<()>;
}
#[allow(unused)]
const OP_POINTS_IDC_BITS:usize = 12;
#[allow(unused)]
const LEVEL_MAJOR_MIN:usize = 2;
#[allow(unused)]
const LEVEL_MAJOR_BITS:usize = 3;
#[allow(unused)]
const LEVEL_MINOR_BITS:usize = 2;
#[allow(unused)]
const LEVEL_BITS:usize = LEVEL_MAJOR_BITS + LEVEL_MINOR_BITS;
const FRAME_ID_LENGTH: usize = 15;
const DELTA_FRAME_ID_LENGTH: usize = 14;

impl<W: io::Write> UncompressedHeader for BitWriter<W, BigEndian> {
    // Start of OBU Headers
    // Write OBU Header syntax
    fn write_obu_header(&mut self, obu_type: OBU_Type, obu_extension: u32)
            -> io::Result<()>{
        self.write_bit(false)?; // forbidden bit.
        self.write(4, obu_type as u32)?;
        self.write_bit(obu_extension != 0)?;
        self.write_bit(true)?; // obu_has_payload_length_field
        self.write_bit(false)?; // reserved

        if obu_extension != 0 {
            unimplemented!();
            //self.write(8, obu_extension & 0xFF)?; size += 8;
        }

        Ok(())
    }

    fn write_sequence_header_obu(&mut self, fi: &mut FrameInvariants)
        -> io::Result<()> {
        self.write(3, fi.sequence.profile)?; // profile, 3 bits
        self.write(1, 0)?; // still_picture
        self.write(1, 0)?; // reduced_still_picture
        self.write_bit(false)?; // display model present
        self.write_bit(false)?; // no timing info present
        self.write(5, 0)?; // one operating point
        self.write(12,0)?; // idc
        self.write(5, 31)?; // level
        self.write(1, 0)?; // tier
        if fi.sequence.reduced_still_picture_hdr {
            unimplemented!();
        }

        self.write_sequence_header(fi)?;

        self.write_color_config(&mut fi.sequence)?;

        self.write_bit(fi.sequence.film_grain_params_present)?;

        self.write_bit(true)?; // trailing bit

        Ok(())
    }

    fn write_sequence_header(&mut self, fi: &mut FrameInvariants)
        -> io::Result<()> {
        self.write_frame_size(fi)?;

        let seq = &mut fi.sequence;

        if !seq.reduced_still_picture_hdr {
            seq.frame_id_numbers_present_flag = false;
            seq.frame_id_length = FRAME_ID_LENGTH as u32;
            seq.delta_frame_id_length = DELTA_FRAME_ID_LENGTH as u32;

            self.write_bit(seq.frame_id_numbers_present_flag)?;

            if seq.frame_id_numbers_present_flag {
              // We must always have delta_frame_id_length < frame_id_length,
              // in order for a frame to be referenced with a unique delta.
              // Avoid wasting bits by using a coding that enforces this restriction.
              self.write(4, seq.delta_frame_id_length - 2)?;
              self.write(3, seq.frame_id_length - seq.delta_frame_id_length - 1)?;
            }
        }

        self.write_bit(seq.use_128x128_superblock)?;
        self.write_bit(true)?; // enable filter intra
        self.write_bit(seq.enable_intra_edge_filter)?;

        if !seq.reduced_still_picture_hdr {
            self.write_bit(seq.enable_interintra_compound)?;
            self.write_bit(seq.enable_masked_compound)?;
            self.write_bit(seq.enable_warped_motion)?;
            self.write_bit(seq.enable_dual_filter)?;
            self.write_bit(seq.enable_order_hint)?;

            if seq.enable_order_hint {
              self.write_bit(seq.enable_jnt_comp)?;
              self.write_bit(seq.enable_ref_frame_mvs)?;
            }
            if seq.force_screen_content_tools == 2 {
              self.write_bit(true)?;
            } else {
              self.write_bit(false)?;
              self.write_bit(seq.force_screen_content_tools != 0)?;
            }
            if seq.force_screen_content_tools > 0 {
              if seq.force_integer_mv == 2 {
                self.write_bit(true)?;
              } else {
                self.write_bit(false)?;
                self.write_bit(seq.force_integer_mv != 0)?;
              }
            } else {
              assert!(seq.force_integer_mv == 2);
            }
            if seq.enable_order_hint {
              self.write(3, seq.order_hint_bits_minus_1)?;
            }
        }

        self.write_bit(seq.enable_superres)?;
        self.write_bit(seq.enable_cdef)?;
        self.write_bit(seq.enable_restoration)?;

        Ok(())
    }

    fn write_color_config(&mut self, seq: &mut Sequence) -> io::Result<()> {
        let high_bd = seq.bit_depth > 8;

        self.write_bit(high_bd)?;

        if seq.bit_depth == 12 {
            self.write_bit(true)?;
        }

        if seq.profile != 1 {
            self.write_bit(seq.monochrome)?;
        }

        if seq.monochrome {
            unimplemented!();
        }

        if let Some(color_description) = seq.color_description {
            self.write_bit(true)?; // color description present
            self.write(8, color_description.color_primaries as u8)?;
            self.write(8, color_description.transfer_characteristics as u8)?;
            self.write(8, color_description.matrix_coefficients as u8)?;
        } else {
            self.write_bit(false)?; // no color description present
        }

        self.write_bit(false)?; // full color range

        let subsampling_x = seq.chroma_sampling != ChromaSampling::Cs444;
        let subsampling_y = seq.chroma_sampling == ChromaSampling::Cs420;

        if seq.bit_depth == 12 {
            self.write_bit(subsampling_x)?;

            if subsampling_x {
                self.write_bit(subsampling_y)?;
            }
        }

        if !subsampling_y {
            unimplemented!(); // 4:2:2 or 4:4:4 sampling
        }

        self.write(2, seq.chroma_sample_position as u32)?;

        self.write_bit(seq.separate_uv_delta_q)?;

        Ok(())
    }

#[allow(unused)]
    fn write_frame_header_obu(&mut self, fi: &FrameInvariants, fs: &FrameState)
        -> io::Result<()> {
      if fi.sequence.reduced_still_picture_hdr {
        assert!(fi.show_existing_frame);
        assert!(fi.frame_type == FrameType::KEY);
        assert!(fi.show_frame);
      } else {
        self.write_bit(fi.show_existing_frame)?;

        if fi.show_existing_frame {
          self.write(3, fi.frame_to_show_map_idx)?;

          //TODO:
          /* temporal_point_info();
            if fi.sequence.decoder_model_info_present_flag &&
              timing_info.equal_picture_interval == 0 {
            // write frame_presentation_delay;
          }
          if fi.sequence.frame_id_numbers_present_flag {
            // write display_frame_id;
          }*/

          self.write_bit(true)?; // trailing bit
          self.byte_align()?;
          return Ok(());
        }

        self.write(2, fi.frame_type as u32)?;
        self.write_bit(fi.show_frame)?; // show frame

        if fi.show_frame {
          //TODO:
          /* temporal_point_info();
              if fi.sequence.decoder_model_info_present_flag &&
              timing_info.equal_picture_interval == 0 {
            // write frame_presentation_delay;*/
        } else {
          self.write_bit(fi.showable_frame)?;
        }

        if fi.frame_type == FrameType::SWITCH {
          assert!(fi.error_resilient);
        } else {
          if !(fi.frame_type == FrameType::KEY && fi.show_frame) {
            self.write_bit(fi.error_resilient)?; // error resilient
          }
        }
      }

      self.write_bit(fi.disable_cdf_update)?;

      if fi.sequence.force_screen_content_tools == 2 {
        self.write_bit(fi.allow_screen_content_tools != 0)?;
      } else {
        assert!(fi.allow_screen_content_tools ==
                fi.sequence.force_screen_content_tools);
      }

      if fi.allow_screen_content_tools == 2 {
        if fi.sequence.force_integer_mv == 2 {
          self.write_bit(fi.force_integer_mv != 0)?;
        } else {
          assert!(fi.force_integer_mv == fi.sequence.force_integer_mv);
        }
      } else {
        assert!(fi.allow_screen_content_tools ==
                fi.sequence.force_screen_content_tools);
      }

      if fi.sequence.frame_id_numbers_present_flag {
        unimplemented!();

        //TODO:
        //let frame_id_len = fi.sequence.frame_id_length;
        //self.write(frame_id_len, fi.current_frame_id);
      }

      let mut frame_size_override_flag = false;
      if fi.frame_type == FrameType::SWITCH {
        frame_size_override_flag = true;
      } else if fi.sequence.reduced_still_picture_hdr {
        frame_size_override_flag = false;
      } else {
        self.write_bit(frame_size_override_flag)?; // frame size overhead flag
      }

      if fi.sequence.enable_order_hint {
        let n = fi.sequence.order_hint_bits_minus_1 + 1;
        let mask = (1 << n) - 1;
        self.write(n, fi.order_hint & mask)?;
      }

      if fi.error_resilient || fi.intra_only {
      } else {
        self.write(PRIMARY_REF_BITS, fi.primary_ref_frame)?;
      }

      if fi.sequence.decoder_model_info_present_flag {
        unimplemented!();
      }

      if fi.frame_type == FrameType::KEY {
        if !fi.show_frame {  // unshown keyframe (forward keyframe)
          unimplemented!();
          self.write(REF_FRAMES as u32, fi.refresh_frame_flags)?;
        } else {
          assert!(fi.refresh_frame_flags == ALL_REF_FRAMES_MASK);
        }
      } else { // Inter frame info goes here
        if fi.intra_only {
          assert!(fi.refresh_frame_flags != ALL_REF_FRAMES_MASK);
          self.write(REF_FRAMES as u32, fi.refresh_frame_flags)?;
        } else {
          // TODO: This should be set once inter mode is used
          self.write(REF_FRAMES as u32, fi.refresh_frame_flags)?;
        }

      };

      if (!fi.intra_only || fi.refresh_frame_flags != ALL_REF_FRAMES_MASK) {
        // Write all ref frame order hints if error_resilient_mode == 1
        if (fi.error_resilient && fi.sequence.enable_order_hint) {
          unimplemented!();
          //for _ in 0..REF_FRAMES {
          //  self.write(order_hint_bits_minus_1,ref_order_hint[i])?; // order_hint
          //}
        }
      }

      // if KEY or INTRA_ONLY frame
      // FIXME: Not sure whether putting frame/render size here is good idea
      if fi.intra_only {
        if frame_size_override_flag {
          unimplemented!();
        }
        if fi.sequence.enable_superres {
          unimplemented!();
        }
        self.write_bit(false)?; // render_and_frame_size_different
        //if render_and_frame_size_different { }
        if fi.allow_screen_content_tools != 0 && true /* UpscaledWidth == FrameWidth */ {
          self.write_bit(fi.allow_intrabc)?;
        }
      }

      let frame_refs_short_signaling = false;
      if fi.frame_type == FrameType::KEY {
        // Done by above
      } else {
        if fi.intra_only {
          // Done by above
        } else {
          if fi.sequence.enable_order_hint {
            self.write_bit(frame_refs_short_signaling)?;
            if frame_refs_short_signaling {
              unimplemented!();
            }
          }

          for i in 0..INTER_REFS_PER_FRAME {
            if !frame_refs_short_signaling {
              self.write(REF_FRAMES_LOG2 as u32, fi.ref_frames[i] as u8)?;
            }
            if fi.sequence.frame_id_numbers_present_flag {
              unimplemented!();
            }
          }
          if fi.error_resilient && frame_size_override_flag {
            unimplemented!();
          } else {
            if frame_size_override_flag {
               unimplemented!();
            }
            if fi.sequence.enable_superres {
              unimplemented!();
            }
            self.write_bit(false)?; // render_and_frame_size_different
          }
          if fi.force_integer_mv != 0 {
          } else {
            self.write_bit(fi.allow_high_precision_mv);
          }
          self.write_bit(fi.is_filter_switchable)?;
          self.write_bit(fi.is_motion_mode_switchable)?;
          self.write(2,0)?; // EIGHTTAP_REGULAR
          if fi.error_resilient || !fi.sequence.enable_ref_frame_mvs {
          } else {
            self.write_bit(fi.use_ref_frame_mvs)?;
          }
        }
      }

      if !fi.sequence.reduced_still_picture_hdr && !fi.disable_cdf_update {
        self.write_bit(fi.disable_frame_end_update_cdf)?;
      }

      // tile
      self.write_bit(true)?; // uniform_tile_spacing_flag
      if fi.width > 64 {
        // TODO: if tile_cols > 1, write more increment_tile_cols_log2 bits
        self.write_bit(false)?; // tile cols
      }
      if fi.height > 64 {
        // TODO: if tile_rows > 1, write increment_tile_rows_log2 bits
        self.write_bit(false)?; // tile rows
      }
      // TODO: if tile_cols * tile_rows > 1 {
      // write context_update_tile_id and tile_size_bytes_minus_1 }

      // quantization
      assert!(fi.base_q_idx > 0);
      self.write(8, fi.base_q_idx)?; // base_q_idx
      self.write_delta_q(fi.dc_delta_q[0])?;
      assert!(fi.ac_delta_q[0] == 0);
      let diff_uv_delta = fi.sequence.separate_uv_delta_q
        && (fi.dc_delta_q[1] != fi.dc_delta_q[2]
          || fi.ac_delta_q[1] != fi.ac_delta_q[2]);
      if fi.sequence.separate_uv_delta_q {
        self.write_bit(diff_uv_delta)?;
      } else {
        assert!(fi.dc_delta_q[1] == fi.dc_delta_q[2]);
        assert!(fi.ac_delta_q[1] == fi.ac_delta_q[2]);
      }
      self.write_delta_q(fi.dc_delta_q[1])?;
      self.write_delta_q(fi.ac_delta_q[1])?;
      if diff_uv_delta {
        self.write_delta_q(fi.dc_delta_q[2])?;
        self.write_delta_q(fi.ac_delta_q[2])?;
      }
      self.write_bit(false)?; // no qm

      // segmentation
      self.write_segment_data(fi, &fs.segmentation)?;

      // delta_q
      self.write_bit(false)?; // delta_q_present_flag: no delta q

      // delta_lf_params in the spec
      self.write_deblock_filter_a(fi, &fs.deblock)?;

      // code for features not yet implemented....

      // loop_filter_params in the spec
      self.write_deblock_filter_b(fi, &fs.deblock)?;

      // cdef
      self.write_frame_cdef(fi)?;

      // loop restoration
      self.write_frame_lrf(fi, &fs.restoration)?;

      self.write_bit(false)?; // tx mode == TX_MODE_SELECT ?

      let mut reference_select = false;
      if !fi.intra_only {
        reference_select = fi.reference_mode != ReferenceMode::SINGLE;
        self.write_bit(reference_select)?;
      }

      let skip_mode_allowed = fi.sequence.get_skip_mode_allowed(fi, reference_select);
      if skip_mode_allowed {
        self.write_bit(false)?; // skip_mode_present
      }

      if fi.intra_only || fi.error_resilient || !fi.sequence.enable_warped_motion {
      } else {
        self.write_bit(fi.allow_warped_motion)?; // allow_warped_motion
      }

      self.write_bit(fi.use_reduced_tx_set)?; // reduced tx

      // global motion
      if !fi.intra_only {
          for i in LAST_FRAME..ALTREF_FRAME+1 {
              let mode = fi.globalmv_transformation_type[i];
              self.write_bit(mode != GlobalMVMode::IDENTITY)?;
              if mode != GlobalMVMode::IDENTITY {
                  self.write_bit(mode == GlobalMVMode::ROTZOOM)?;
                  if mode != GlobalMVMode::ROTZOOM {
                      self.write_bit(mode == GlobalMVMode::TRANSLATION)?;
                  }
              }
              match mode {
                  GlobalMVMode::IDENTITY => { /* Nothing to do */ }
                  GlobalMVMode::TRANSLATION => {
                      let mv_x = 0;
                      let mv_x_ref = 0;
                      let mv_y = 0;
                      let mv_y_ref = 0;
                      let bits = 12 - 6 + 3 - !fi.allow_high_precision_mv as u8;
                      let bits_diff = 12 - 3 + fi.allow_high_precision_mv as u8;
                      BCodeWriter::write_s_refsubexpfin(self, (1 << bits) + 1,
                                                        3, mv_x_ref >> bits_diff,
                                                        mv_x >> bits_diff)?;
                      BCodeWriter::write_s_refsubexpfin(self, (1 << bits) + 1,
                                                        3, mv_y_ref >> bits_diff,
                                                        mv_y >> bits_diff)?;
                  }
                  GlobalMVMode::ROTZOOM => unimplemented!(),
                  GlobalMVMode::AFFINE => unimplemented!(),
              };
          }
      }

      if fi.sequence.film_grain_params_present && fi.show_frame {
          unimplemented!();
      }

      if fi.large_scale_tile {
          unimplemented!();
      }
      self.write_bit(true)?; // trailing bit
      self.byte_align()?;

      Ok(())
    }
    // End of OBU Headers

    fn write_frame_size(&mut self, fi: &FrameInvariants) -> io::Result<()> {
        // width_bits and height_bits will have to be moved to the sequence header OBU
        // when we add support for it.
        let width_bits = 32 - (fi.width as u32).leading_zeros();
        let height_bits = 32 - (fi.height as u32).leading_zeros();
        assert!(width_bits <= 16);
        assert!(height_bits <= 16);
        self.write(4, width_bits - 1)?;
        self.write(4, height_bits - 1)?;
        self.write(width_bits, (fi.width - 1) as u16)?;
        self.write(height_bits, (fi.height - 1) as u16)?;
        Ok(())
    }

    fn write_deblock_filter_a(&mut self, fi: &FrameInvariants, deblock: &DeblockState) -> io::Result<()> {
        if fi.delta_q_present {
            if !fi.allow_intrabc {
                self.write_bit(deblock.block_deltas_enabled)?;
            }
            if deblock.block_deltas_enabled {
                self.write(2, deblock.block_delta_shift)?;
                self.write_bit(deblock.block_delta_multi)?;
            }
        }
        Ok(())
    }

    fn write_deblock_filter_b(&mut self, fi: &FrameInvariants, deblock: &DeblockState) -> io::Result<()> {
        assert!(deblock.levels[0] < 64);
        self.write(6, deblock.levels[0])?; // loop deblocking filter level 0
        assert!(deblock.levels[1] < 64);
        self.write(6, deblock.levels[1])?; // loop deblocking filter level 1
        if PLANES > 1 && (deblock.levels[0] > 0 || deblock.levels[1] > 0) {
            assert!(deblock.levels[2] < 64);
            self.write(6, deblock.levels[2])?; // loop deblocking filter level 2
            assert!(deblock.levels[3] < 64);
            self.write(6, deblock.levels[3])?; // loop deblocking filter level 3
        }
        self.write(3, deblock.sharpness)?; // deblocking filter sharpness
        self.write_bit(deblock.deltas_enabled)?; // loop deblocking filter deltas enabled
        if deblock.deltas_enabled {
            self.write_bit(deblock.delta_updates_enabled)?; // deltas updates enabled
            if deblock.delta_updates_enabled {
                // conditionally write ref delta updates
                let prev_ref_deltas = if fi.primary_ref_frame == PRIMARY_REF_NONE {
                    [1, 0, 0, 0, 0, -1, -1, -1]
                } else {
                    fi.rec_buffer.deblock[fi.ref_frames[fi.primary_ref_frame as usize] as usize].ref_deltas
                };
                for i in 0..REF_FRAMES {
                    let update = deblock.ref_deltas[i] != prev_ref_deltas[i];
                    self.write_bit(update)?;
                    if update {
                        self.write_signed(7, deblock.ref_deltas[i])?;
                    }
                }
                // conditionally write mode delta updates
                let prev_mode_deltas = if fi.primary_ref_frame == PRIMARY_REF_NONE {
                    [0, 0]
                } else {
                    fi.rec_buffer.deblock[fi.ref_frames[fi.primary_ref_frame as usize] as usize].mode_deltas
                };
                for i in 0..2 {
                    let update = deblock.mode_deltas[i] != prev_mode_deltas[i];
                    self.write_bit(update)?;
                    if update {
                        self.write_signed(7, deblock.mode_deltas[i])?;
                    }
                }
            }
        }
        Ok(())
    }

    fn write_frame_cdef(&mut self, fi: &FrameInvariants) -> io::Result<()> {
        if fi.sequence.enable_cdef {
            assert!(fi.cdef_damping >= 3);
            assert!(fi.cdef_damping <= 6);
            self.write(2, fi.cdef_damping - 3)?;
            assert!(fi.cdef_bits < 4);
            self.write(2,fi.cdef_bits)?; // cdef bits
            for i in 0..(1<<fi.cdef_bits) {
                let j = i << (3 - fi.cdef_bits);
                assert!(fi.cdef_y_strengths[j]<64);
                assert!(fi.cdef_uv_strengths[j]<64);
                self.write(6,fi.cdef_y_strengths[j])?; // cdef y strength
                self.write(6,fi.cdef_uv_strengths[j])?; // cdef uv strength
            }
        }
        Ok(())
    }

    fn write_frame_lrf(&mut self, fi: &FrameInvariants,
                       rs: &RestorationState) -> io::Result<()> {
      if fi.sequence.enable_restoration && !fi.allow_intrabc { // && !self.lossless
        let mut use_lrf = false;
        let mut use_chroma_lrf = false;
        for i in 0..PLANES {
          self.write(2, rs.plane[i].lrf_type)?; // filter type by plane
          if rs.plane[i].lrf_type != RESTORE_NONE {
            use_lrf = true;
            if i > 0 { use_chroma_lrf = true; }
          }
        }
        if use_lrf {
          // The Y shift value written here indicates shift up from superblock size
          if !fi.sequence.use_128x128_superblock {
            self.write(1, if rs.plane[0].unit_size > 64 {1} else {0})?;
          }
          if rs.plane[0].unit_size > 64 {
            self.write(1, if rs.plane[0].unit_size > 128 {1} else {0})?;
          }

          if use_chroma_lrf {
            if fi.sequence.chroma_sampling == ChromaSampling::Cs420 {
              self.write(1, if rs.plane[0].unit_size > rs.plane[1].unit_size {1} else {0})?;
            }
          }
        }
      }
      Ok(())
    }

    fn write_segment_data(&mut self, fi: &FrameInvariants, segmentation: &SegmentationState) -> io::Result<()> {
        self.write_bit(segmentation.enabled)?;
        if segmentation.enabled {
            if fi.primary_ref_frame == PRIMARY_REF_NONE {
                assert_eq!(segmentation.update_map, true);
                assert_eq!(segmentation.update_data, true);
            } else {
                self.write_bit(segmentation.update_map)?;
                if segmentation.update_map {
                    self.write_bit(false)?; /* Without using temporal prediction */
                }
                self.write_bit(segmentation.update_data)?;
            }
            if segmentation.update_data {
                for i in 0..8 {
                    for j in 0..SegLvl::SEG_LVL_MAX as usize {
                        self.write_bit(segmentation.features[i][j])?;
                        if segmentation.features[i][j] {
                            let bits = seg_feature_bits[j];
                            let data = segmentation.data[i][j];
                            if seg_feature_is_signed[j] {
                                self.write_signed(bits + 1, data)?;
                            } else {
                                self.write(bits, data)?;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn write_delta_q(&mut self, delta_q: i8) -> io::Result<()> {
        self.write_bit(delta_q != 0)?;
        if delta_q != 0 {
            self.write_signed(6 + 1, delta_q)?;
        }
        Ok(())
    }
}

#[allow(non_camel_case_types)]
pub enum OBU_Type {
  OBU_SEQUENCE_HEADER = 1,
  OBU_TEMPORAL_DELIMITER = 2,
  OBU_FRAME_HEADER = 3,
  OBU_TILE_GROUP = 4,
  OBU_METADATA = 5,
  OBU_FRAME = 6,
  OBU_REDUNDANT_FRAME_HEADER = 7,
  OBU_TILE_LIST = 8,
  OBU_PADDING = 15,
}

// NOTE from libaom:
// Disallow values larger than 32-bits to ensure consistent behavior on 32 and
// 64 bit targets: value is typically used to determine buffer allocation size
// when decoded.
fn aom_uleb_size_in_bytes(mut value: u64) -> usize {
  let mut size = 0;
  loop {
    size += 1;
    value = value >> 7;
    if value == 0 { break; }
  }
  size
}

fn aom_uleb_encode(mut value: u64, coded_value: &mut [u8]) -> usize {
  let leb_size = aom_uleb_size_in_bytes(value);

  for i in 0..leb_size {
    let mut byte = (value & 0x7f) as u8;
    value >>= 7;
    if value != 0 { byte |= 0x80 };  // Signal that more bytes follow.
    coded_value[i] = byte;
  }

  leb_size
}

fn write_obus(packet: &mut dyn io::Write,
              fi: &mut FrameInvariants, fs: &FrameState)
         -> io::Result<()> {
    let obu_extension = 0 as u32;

    let mut buf1 = Vec::new();
    {
      let mut bw1 = BitWriter::endian(&mut buf1, BigEndian);
      bw1.write_obu_header(OBU_Type::OBU_TEMPORAL_DELIMITER, obu_extension)?;
      bw1.write(8,0)?;	// size of payload == 0, one byte
    }
    packet.write_all(&buf1).unwrap();
    buf1.clear();

    // write sequence header obu if KEY_FRAME, preceded by 4-byte size
    if fi.frame_type == FrameType::KEY {
        let mut buf2 = Vec::new();
        {
            let mut bw2 = BitWriter::endian(&mut buf2, BigEndian);
            bw2.write_sequence_header_obu(fi)?;
            bw2.byte_align()?;
        }

        {
            let mut bw1 = BitWriter::endian(&mut buf1, BigEndian);
            bw1.write_obu_header(OBU_Type::OBU_SEQUENCE_HEADER, obu_extension)?;
        }
        packet.write_all(&buf1).unwrap();
        buf1.clear();

        let obu_payload_size = buf2.len() as u64;
        {
            let mut bw1 = BitWriter::endian(&mut buf1, BigEndian);
            // uleb128()
            let mut coded_payload_length = [0 as u8; 8];
            let leb_size = aom_uleb_encode(obu_payload_size, &mut coded_payload_length);
            for i in 0..leb_size {
                bw1.write(8, coded_payload_length[i])?;
            }
        }
        packet.write_all(&buf1).unwrap();
        buf1.clear();

        packet.write_all(&buf2).unwrap();
        buf2.clear();
    }

    let mut buf2 = Vec::new();
    {
        let mut bw2 = BitWriter::endian(&mut buf2, BigEndian);
        bw2.write_frame_header_obu(fi, fs)?;
    }

    {
        let mut bw1 = BitWriter::endian(&mut buf1, BigEndian);
        bw1.write_obu_header(OBU_Type::OBU_FRAME_HEADER, obu_extension)?;
    }
    packet.write_all(&buf1).unwrap();
    buf1.clear();

    let obu_payload_size = buf2.len() as u64;
    {
        let mut bw1 = BitWriter::endian(&mut buf1, BigEndian);
        // uleb128()
        let mut coded_payload_length = [0 as u8; 8];
        let leb_size = aom_uleb_encode(obu_payload_size, &mut coded_payload_length);
        for i in 0..leb_size {
            bw1.write(8, coded_payload_length[i])?;
        }
    }
    packet.write_all(&buf1).unwrap();
    buf1.clear();

    packet.write_all(&buf2).unwrap();
    buf2.clear();

    Ok(())
}

/// Write into `dst` the difference between the blocks at `src1` and `src2`
fn diff(dst: &mut [i16], src1: &PlaneSlice<'_>, src2: &PlaneSlice<'_>, width: usize, height: usize) {
  let src1_stride = src1.plane.cfg.stride;
  let src2_stride = src2.plane.cfg.stride;

  for ((l, s1), s2) in dst.chunks_mut(width).take(height)
                        .zip(src1.as_slice().chunks(src1_stride))
                        .zip(src2.as_slice().chunks(src2_stride)) {
    for ((r, v1), v2) in l.iter_mut().zip(s1).zip(s2) {
      *r = *v1 as i16 - *v2 as i16;
    }
  }
}

fn get_qidx(fi: &FrameInvariants, fs: &FrameState, cw: &ContextWriter, bo: &BlockOffset) -> u8 {
    let mut qidx = fi.base_q_idx;
    let sidx = cw.bc.at(bo).segmentation_idx as usize;
    if fs.segmentation.features[sidx][SegLvl::SEG_LVL_ALT_Q as usize] {
        let delta = fs.segmentation.data[sidx][SegLvl::SEG_LVL_ALT_Q as usize];
        qidx = clamp((qidx as i16) + delta, 0, 255) as u8;
    }
    qidx
}

// For a transform block,
// predict, transform, quantize, write coefficients to a bitstream,
// dequantize, inverse-transform.
pub fn encode_tx_block(
  fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter,
  w: &mut dyn Writer, p: usize, bo: &BlockOffset, mode: PredictionMode,
  tx_size: TxSize, tx_type: TxType, plane_bsize: BlockSize, po: &PlaneOffset,
  skip: bool, ac: &[i16], alpha: i16, for_rdo_use: bool
) -> (bool, i64) {
    let qidx = get_qidx(fi, fs, cw, bo);
    let rec = &mut fs.rec.planes[p];
    let PlaneConfig { stride, xdec, ydec, .. } = fs.input.planes[p].cfg;

    assert!(tx_size.sqr() <= TxSize::TX_32X32 || tx_type == TxType::DCT_DCT);

    if mode.is_intra() {
      let bit_depth = fi.sequence.bit_depth;
      let edge_buf = get_intra_edges(&rec.slice(po), tx_size, bit_depth, p, fi.w_in_b, fi.h_in_b, Some(mode));
      mode.predict_intra(&mut rec.mut_slice(po), tx_size, bit_depth, &ac, alpha, &edge_buf);
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

    diff(residual,
         &fs.input.planes[p].slice(po),
         &rec.slice(po),
         tx_size.width(),
         tx_size.height());

    forward_transform(residual, coeffs, tx_size.width(), tx_size, tx_type, fi.sequence.bit_depth);

    let coded_tx_size = av1_get_coded_tx_size(tx_size).area();
    fs.qc.quantize(coeffs, qcoeffs, coded_tx_size);

    let has_coeff = cw.write_coeffs_lv_map(w, p, bo, &qcoeffs, mode, tx_size, tx_type, plane_bsize, xdec, ydec,
                            fi.use_reduced_tx_set);

    // Reconstruct
    dequantize(qidx, qcoeffs, rcoeffs, tx_size, fi.sequence.bit_depth, fi.dc_delta_q[p], fi.ac_delta_q[p]);

    let mut tx_dist: i64 = -1;

    if !fi.use_tx_domain_distortion || !for_rdo_use {
        inverse_transform_add(rcoeffs, &mut rec.mut_slice(po).as_mut_slice(), stride, tx_size, tx_type, fi.sequence.bit_depth);
    } else {
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
    (has_coeff, tx_dist)
}

pub fn motion_compensate(fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter,
                         luma_mode: PredictionMode, ref_frames: [usize; 2], mvs: [MotionVector; 2],
                         bsize: BlockSize, bo: &BlockOffset, luma_only: bool) {
  debug_assert!(!luma_mode.is_intra());

  let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;

  // Inter mode prediction can take place once for a whole partition,
  // instead of each tx-block.
  let num_planes = 1 + if !luma_only && has_chroma(bo, bsize, xdec, ydec) { 2 } else { 0 };

  for p in 0..num_planes {
    let plane_bsize = if p == 0 { bsize }
    else { get_plane_block_size(bsize, xdec, ydec) };

    let po = bo.plane_offset(&fs.input.planes[p].cfg);
    let rec = &mut fs.rec.planes[p];

    if p > 0 && bsize < BlockSize::BLOCK_8X8 {
      let mut some_use_intra = false;
      if bsize == BlockSize::BLOCK_4X4 || bsize == BlockSize::BLOCK_4X8 {
          some_use_intra |= cw.bc.at(&bo.with_offset(-1,0)).mode.is_intra(); };
      if !some_use_intra && bsize == BlockSize::BLOCK_4X4 || bsize == BlockSize::BLOCK_8X4 {
          some_use_intra |= cw.bc.at(&bo.with_offset(0,-1)).mode.is_intra(); };
      if !some_use_intra && bsize == BlockSize::BLOCK_4X4 {
          some_use_intra |= cw.bc.at(&bo.with_offset(-1,-1)).mode.is_intra(); };

      if some_use_intra {
        luma_mode.predict_inter(fi, p, &po, &mut rec.mut_slice(&po), plane_bsize.width(),
          plane_bsize.height(), ref_frames, mvs);
      } else {
        assert!(xdec == 1 && ydec == 1);
        // TODO: these are absolutely only valid for 4:2:0
        if bsize == BlockSize::BLOCK_4X4 {
            let mv0 = cw.bc.at(&bo.with_offset(-1,-1)).mv;
            let rf0 = cw.bc.at(&bo.with_offset(-1,-1)).ref_frames;
            let mv1 = cw.bc.at(&bo.with_offset(0,-1)).mv;
            let rf1 = cw.bc.at(&bo.with_offset(0,-1)).ref_frames;
            let po1 = PlaneOffset { x: po.x+2, y: po.y };
            let mv2 = cw.bc.at(&bo.with_offset(-1,0)).mv;
            let rf2 = cw.bc.at(&bo.with_offset(-1,0)).ref_frames;
            let po2 = PlaneOffset { x: po.x, y: po.y+2 };
            let po3 = PlaneOffset { x: po.x+2, y: po.y+2 };
            luma_mode.predict_inter(fi, p, &po, &mut rec.mut_slice(&po), 2, 2, rf0, mv0);
            luma_mode.predict_inter(fi, p, &po1, &mut rec.mut_slice(&po1), 2, 2, rf1, mv1);
            luma_mode.predict_inter(fi, p, &po2, &mut rec.mut_slice(&po2), 2, 2, rf2, mv2);
            luma_mode.predict_inter(fi, p, &po3, &mut rec.mut_slice(&po3), 2, 2, ref_frames, mvs);
        }
        if bsize == BlockSize::BLOCK_8X4 {
            let mv1 = cw.bc.at(&bo.with_offset(0,-1)).mv;
            let rf1 = cw.bc.at(&bo.with_offset(0,-1)).ref_frames;
            luma_mode.predict_inter(fi, p, &po, &mut rec.mut_slice(&po), 4, 2, rf1, mv1);
            let po3 = PlaneOffset { x: po.x, y: po.y+2 };
            luma_mode.predict_inter(fi, p, &po3, &mut rec.mut_slice(&po3), 4, 2, ref_frames, mvs);
        }
        if bsize == BlockSize::BLOCK_4X8 {
            let mv2 = cw.bc.at(&bo.with_offset(-1,0)).mv;
            let rf2 = cw.bc.at(&bo.with_offset(-1,0)).ref_frames;
            luma_mode.predict_inter(fi, p, &po, &mut rec.mut_slice(&po), 2, 4, rf2, mv2);
            let po3 = PlaneOffset { x: po.x+2, y: po.y };
            luma_mode.predict_inter(fi, p, &po3, &mut rec.mut_slice(&po3), 2, 4, ref_frames, mvs);
        }
      }
    } else {
      luma_mode.predict_inter(fi, p, &po, &mut rec.mut_slice(&po), plane_bsize.width(),
        plane_bsize.height(), ref_frames, mvs);
    }
  }
}

pub fn encode_block_a(seq: &Sequence, fs: &FrameState,
                 cw: &mut ContextWriter, w: &mut dyn Writer,
                 bsize: BlockSize, bo: &BlockOffset, skip: bool) -> bool {
    cw.bc.set_skip(bo, bsize, skip);
    if fs.segmentation.enabled && fs.segmentation.update_map && fs.segmentation.preskip {
        cw.write_segmentation(w, bo, bsize, false, fs.segmentation.last_active_segid);
    }
    cw.write_skip(w, bo, skip);
    if fs.segmentation.enabled && fs.segmentation.update_map && !fs.segmentation.preskip {
        cw.write_segmentation(w, bo, bsize, skip, fs.segmentation.last_active_segid);
    }
    if !skip && seq.enable_cdef {
        cw.bc.cdef_coded = true;
    }
    cw.bc.cdef_coded
}

pub fn encode_block_b(fi: &FrameInvariants, fs: &mut FrameState,
                 cw: &mut ContextWriter, w: &mut dyn Writer,
                 luma_mode: PredictionMode, chroma_mode: PredictionMode,
                 ref_frames: [usize; 2], mvs: [MotionVector; 2],
                 bsize: BlockSize, bo: &BlockOffset, skip: bool,
                 cfl: CFLParams, tx_size: TxSize, tx_type: TxType,
                 mode_context: usize, mv_stack: &[CandidateMV], for_rdo_use: bool)
                 -> i64 {
    let is_inter = !luma_mode.is_intra();
    if is_inter { assert!(luma_mode == chroma_mode); };
    let sb_size = if fi.sequence.use_128x128_superblock {
        BlockSize::BLOCK_128X128
    } else {
        BlockSize::BLOCK_64X64
    };
    let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;
    if skip {
        cw.bc.reset_skip_context(bo, bsize, xdec, ydec);
    }
    cw.bc.set_block_size(bo, bsize);
    cw.bc.set_mode(bo, bsize, luma_mode);
    cw.bc.set_ref_frames(bo, bsize, ref_frames);
    cw.bc.set_motion_vectors(bo, bsize, mvs);

    //write_q_deltas();
    if cw.bc.code_deltas && fs.deblock.block_deltas_enabled && (bsize < sb_size || !skip) {
        cw.write_block_deblock_deltas(w, bo, fs.deblock.block_delta_multi);
    }
    cw.bc.code_deltas = false;

    if fi.frame_type == FrameType::INTER {
        cw.write_is_inter(w, bo, is_inter);
        if is_inter {
            cw.fill_neighbours_ref_counts(bo);
            cw.write_ref_frames(w, fi, bo);

            // NOTE: Until rav1e supports other inter modes than GLOBALMV
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
              [MotionVector{ row: 0, col: 0 }; 2]
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
              if mv_stack.len() > 0 {
                assert!(mv_stack[0].this_mv.row == mvs[0].row);
                assert!(mv_stack[0].this_mv.col == mvs[0].col);
              } else {
                assert!(0 == mvs[0].row);
                assert!(0 == mvs[0].col);
              }
            }
        } else {
            cw.write_intra_mode(w, bsize, luma_mode);
        }
    } else {
        cw.write_intra_mode_kf(w, bo, luma_mode);
    }

    if !is_inter {
        if luma_mode.is_directional() && bsize >= BlockSize::BLOCK_8X8 {
            cw.write_angle_delta(w, 0, luma_mode);
        }
        if has_chroma(bo, bsize, xdec, ydec) {
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
        if luma_mode == PredictionMode::DC_PRED && bsize.width() <= 32 && bsize.height() <= 32 {
            cw.write_use_filter_intra(w,false, bsize); // Always turn off FILTER_INTRA
        }
    }

    if is_inter {
      motion_compensate(fi, fs, cw, luma_mode, ref_frames, mvs, bsize, bo, false);
      write_tx_tree(fi, fs, cw, w, luma_mode, bo, bsize, tx_size, tx_type, skip, false, for_rdo_use)
    } else {
      write_tx_blocks(fi, fs, cw, w, luma_mode, chroma_mode, bo, bsize, tx_size, tx_type, skip, cfl, false, for_rdo_use)
    }
}

pub fn luma_ac(
  ac: &mut [i16], fs: &mut FrameState, bo: &BlockOffset, bsize: BlockSize
) {
  let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;
  let plane_bsize = get_plane_block_size(bsize, xdec, ydec);
  let po = if bsize.is_sub8x8() {
    let offset = bsize.sub8x8_offset();
    bo.with_offset(offset.0, offset.1).plane_offset(&fs.input.planes[0].cfg)
  } else {
    bo.plane_offset(&fs.input.planes[0].cfg)
  };
  let rec = &fs.rec.planes[0];
  let luma = &rec.slice(&po);

  let mut sum: i32 = 0;
  for sub_y in 0..plane_bsize.height() {
    for sub_x in 0..plane_bsize.width() {
      let y = sub_y << ydec;
      let x = sub_x << xdec;
      let sample = ((luma.p(x, y)
        + luma.p(x + 1, y)
        + luma.p(x, y + 1)
        + luma.p(x + 1, y + 1))
        << 1) as i16;
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

pub fn write_tx_blocks(fi: &FrameInvariants, fs: &mut FrameState,
                       cw: &mut ContextWriter, w: &mut dyn Writer,
                       luma_mode: PredictionMode, chroma_mode: PredictionMode, bo: &BlockOffset,
                       bsize: BlockSize, tx_size: TxSize, tx_type: TxType, skip: bool,
                       cfl: CFLParams, luma_only: bool, for_rdo_use: bool) -> i64 {
    let bw = bsize.width_mi() / tx_size.width_mi();
    let bh = bsize.height_mi() / tx_size.height_mi();
    let qidx = get_qidx(fi, fs, cw, bo);

    let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;
    let ac = &mut [0i16; 32 * 32];
    let mut tx_dist: i64 = 0;
    let do_chroma = has_chroma(bo, bsize, xdec, ydec);

    fs.qc.update(qidx, tx_size, luma_mode.is_intra(), fi.sequence.bit_depth, fi.dc_delta_q[0], 0);

    for by in 0..bh {
        for bx in 0..bw {
            let tx_bo = BlockOffset {
                x: bo.x + bx * tx_size.width_mi(),
                y: bo.y + by * tx_size.height_mi()
            };

            let po = tx_bo.plane_offset(&fs.input.planes[0].cfg);
            let (_, dist) =
            encode_tx_block(
              fi, fs, cw, w, 0, &tx_bo, luma_mode, tx_size, tx_type, bsize, &po,
              skip, ac, 0, for_rdo_use
            );
            assert!(!fi.use_tx_domain_distortion || !for_rdo_use || skip || dist >= 0);
            tx_dist += dist;
        }
    }

    if luma_only { return tx_dist };

    let uv_tx_size = bsize.largest_uv_tx_size(fi.sequence.chroma_sampling);

    let mut bw_uv = (bw * tx_size.width_mi()) >> xdec;
    let mut bh_uv = (bh * tx_size.height_mi()) >> ydec;

    if (bw_uv == 0 || bh_uv == 0) && do_chroma {
        bw_uv = 1;
        bh_uv = 1;
    }

    bw_uv /= uv_tx_size.width_mi();
    bh_uv /= uv_tx_size.height_mi();

    let plane_bsize = get_plane_block_size(bsize, xdec, ydec);

    if chroma_mode.is_cfl() {
      luma_ac(ac, fs, bo, bsize);
    }

    if bw_uv > 0 && bh_uv > 0 {
        let uv_tx_type = if uv_tx_size.width() >= 32 || uv_tx_size.height() >= 32 {
            TxType::DCT_DCT
        } else {
            uv_intra_mode_to_tx_type_context(chroma_mode)
        };

        for p in 1..3 {
            fs.qc.update(fi.base_q_idx, uv_tx_size, true, fi.sequence.bit_depth, fi.dc_delta_q[p], fi.ac_delta_q[p]);
            let alpha = cfl.alpha(p - 1);
            for by in 0..bh_uv {
                for bx in 0..bw_uv {
                    let tx_bo =
                        BlockOffset {
                            x: bo.x + ((bx * uv_tx_size.width_mi()) << xdec) -
                                ((bw * tx_size.width_mi() == 1) as usize),
                            y: bo.y + ((by * uv_tx_size.height_mi()) << ydec) -
                                ((bh * tx_size.height_mi() == 1) as usize)
                        };

                    let mut po = bo.plane_offset(&fs.input.planes[p].cfg);
                    po.x += (bx * uv_tx_size.width()) as isize;
                    po.y += (by * uv_tx_size.height()) as isize;
                    let (_, dist) =
                    encode_tx_block(fi, fs, cw, w, p, &tx_bo, chroma_mode, uv_tx_size, uv_tx_type,
                                    plane_bsize, &po, skip, ac, alpha, for_rdo_use);
                    assert!(!fi.use_tx_domain_distortion || !for_rdo_use || skip || dist >= 0);
                    tx_dist += dist;
                }
            }
        }
    }

    tx_dist
}

// FIXME: For now, assume tx_mode is LARGEST_TX, so var-tx is not implemented yet
// but only one tx block exist for a inter mode partition.
pub fn write_tx_tree(fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter, w: &mut dyn Writer,
                       luma_mode: PredictionMode, bo: &BlockOffset,
                       bsize: BlockSize, tx_size: TxSize, tx_type: TxType, skip: bool,
                       luma_only: bool, for_rdo_use: bool) -> i64 {
    let bw = bsize.width_mi() / tx_size.width_mi();
    let bh = bsize.height_mi() / tx_size.height_mi();
    let qidx = get_qidx(fi, fs, cw, bo);

    let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;
    let ac = &[0i16; 32 * 32];
    let mut tx_dist: i64 = 0;

    fs.qc.update(qidx, tx_size, luma_mode.is_intra(), fi.sequence.bit_depth, fi.dc_delta_q[0], 0);

    let po = bo.plane_offset(&fs.input.planes[0].cfg);
    let (has_coeff, dist) = encode_tx_block(
      fi, fs, cw, w, 0, &bo, luma_mode, tx_size, tx_type, bsize, &po, skip, ac, 0, for_rdo_use
    );
    assert!(!fi.use_tx_domain_distortion || !for_rdo_use || skip || dist >= 0);
    tx_dist += dist;

    if luma_only { return tx_dist };

    let uv_tx_size = bsize.largest_uv_tx_size(fi.sequence.chroma_sampling);

    let mut bw_uv = (bw * tx_size.width_mi()) >> xdec;
    let mut bh_uv = (bh * tx_size.height_mi()) >> ydec;

    if (bw_uv == 0 || bh_uv == 0) && has_chroma(bo, bsize, xdec, ydec) {
        bw_uv = 1;
        bh_uv = 1;
    }

    bw_uv /= uv_tx_size.width_mi();
    bh_uv /= uv_tx_size.height_mi();

    let plane_bsize = get_plane_block_size(bsize, xdec, ydec);

    if bw_uv > 0 && bh_uv > 0 {
        let uv_tx_type = if has_coeff {tx_type} else {TxType::DCT_DCT}; // if inter mode, uv_tx_type == tx_type

        for p in 1..3 {
            fs.qc.update(qidx, uv_tx_size, false, fi.sequence.bit_depth, fi.dc_delta_q[p], fi.ac_delta_q[p]);
            let tx_bo = BlockOffset {
                x: bo.x  - ((bw * tx_size.width_mi() == 1) as usize),
                y: bo.y  - ((bh * tx_size.height_mi() == 1) as usize)
            };

            let po = bo.plane_offset(&fs.input.planes[p].cfg);
            let (_, dist) =
            encode_tx_block(fi, fs, cw, w, p, &tx_bo, luma_mode, uv_tx_size, uv_tx_type,
                            plane_bsize, &po, skip, ac, 0, for_rdo_use);
            assert!(!fi.use_tx_domain_distortion || !for_rdo_use || skip || dist >= 0);
            tx_dist += dist;
        }
    }

    tx_dist
}

pub fn encode_block_with_modes(fi: &FrameInvariants, fs: &mut FrameState,
    cw: &mut ContextWriter, w_pre_cdef: &mut dyn Writer, w_post_cdef: &mut dyn Writer,
    bsize: BlockSize, bo: &BlockOffset, mode_decision: &RDOPartitionOutput) {
    let (mode_luma, mode_chroma) =
        (mode_decision.pred_mode_luma, mode_decision.pred_mode_chroma);
    let cfl = mode_decision.pred_cfl_params;
    let ref_frames = mode_decision.ref_frames;
    let mvs = mode_decision.mvs;
    let skip = mode_decision.skip;
    let mut cdef_coded = cw.bc.cdef_coded;
    let (tx_size, tx_type) = (mode_decision.tx_size, mode_decision.tx_type);

    debug_assert!((tx_size, tx_type) ==
        rdo_tx_size_type(fi, fs, cw, bsize, bo, mode_luma, ref_frames, mvs, skip));
    cw.bc.set_tx_size(bo, tx_size);

    let mut mv_stack = Vec::new();
    let is_compound = ref_frames[1] != NONE_FRAME;
    let mode_context = cw.find_mvrefs(bo, ref_frames, &mut mv_stack, bsize, fi, is_compound);

    cdef_coded = encode_block_a(&fi.sequence, fs, cw, if cdef_coded  {w_post_cdef} else {w_pre_cdef},
                                bsize, bo, skip);
    encode_block_b(fi, fs, cw, if cdef_coded  {w_post_cdef} else {w_pre_cdef},
                    mode_luma, mode_chroma, ref_frames, mvs, bsize, bo, skip, cfl,
                    tx_size, tx_type, mode_context, &mv_stack, false);
}

fn encode_partition_bottomup(fi: &FrameInvariants, fs: &mut FrameState,
                             cw: &mut ContextWriter, w_pre_cdef: &mut dyn Writer, w_post_cdef: &mut dyn Writer,
                             bsize: BlockSize, bo: &BlockOffset, pmvs: &[[Option<MotionVector>; REF_FRAMES]; 5],
                             ref_rd_cost: f64
) -> (f64, Option<RDOPartitionOutput>) {
    let mut rd_cost = std::f64::MAX;
    let mut best_rd = std::f64::MAX;
    let mut best_pred_modes: Vec<RDOPartitionOutput> = Vec::new();

    if bo.x >= cw.bc.cols || bo.y >= cw.bc.rows {
        return (rd_cost, None);
    }

    let bsw = bsize.width_mi();
    let bsh = bsize.height_mi();
    let is_square = bsize.is_sqr();

    // Always split if the current partition is too large
    let must_split = (bo.x + bsw as usize > fi.w_in_b ||
        bo.y + bsh as usize > fi.h_in_b ||
        bsize.greater_than(BlockSize::BLOCK_64X64)) && is_square;

    // must_split overrides the minimum partition size when applicable
    let can_split = (bsize > fi.min_partition_size && is_square) || must_split;

    let mut best_partition = PartitionType::PARTITION_INVALID;
    let mut best_decision = RDOPartitionOutput {
        rd_cost,
        bo: bo.clone(),
        bsize: bsize,
        pred_mode_luma: PredictionMode::DC_PRED,
        pred_mode_chroma: PredictionMode::DC_PRED,
        pred_cfl_params: CFLParams::new(),
        ref_frames: [INTRA_FRAME, NONE_FRAME],
        mvs: [MotionVector { row: 0, col: 0}; 2],
        skip: false,
        tx_size: TxSize::TX_4X4,
        tx_type: TxType::DCT_DCT,
    }; // Best decision that is not PARTITION_SPLIT

    let cw_checkpoint = cw.checkpoint();
    let w_pre_checkpoint = w_pre_cdef.checkpoint();
    let w_post_checkpoint = w_post_cdef.checkpoint();

    // Code the whole block
    // TODO(yushin): Try move PARTITION_NONE to below partition loop
    if !must_split {
        let mut cost: f64 = 0.0;

        if bsize.gte(BlockSize::BLOCK_8X8) && is_square {
            let w: &mut dyn Writer = if cw.bc.cdef_coded {w_post_cdef} else {w_pre_cdef};
            let tell = w.tell_frac();
            cw.write_partition(w, bo, PartitionType::PARTITION_NONE, bsize);
            cost = (w.tell_frac() - tell) as f64 * get_lambda(fi)/ ((1 << OD_BITRES) as f64);
        }

        let pmv_idx = if bsize.greater_than(BlockSize::BLOCK_32X32) {
            0
        } else {
            ((bo.x & 32) >> 5) + ((bo.y & 32) >> 4) + 1
        };
        let spmvs = &pmvs[pmv_idx];

        let mode_decision = rdo_mode_decision(fi, fs, cw, bsize, bo, spmvs, false).part_modes[0].clone();

        rd_cost = mode_decision.rd_cost + cost;

        if rd_cost < ref_rd_cost {
            best_partition = PartitionType::PARTITION_NONE;
            best_rd = rd_cost;
            best_decision = mode_decision.clone();
            best_pred_modes.push(best_decision.clone());

            encode_block_with_modes(fi, fs, cw, w_pre_cdef, w_post_cdef, bsize, bo,
                                &mode_decision);
        }
    }

    // Test all partition types other than PARTITION_NONE by comparing their RD costs
    if can_split {
        for &partition in RAV1E_PARTITION_TYPES {
            if partition == PartitionType::PARTITION_NONE { continue; }

            assert!(bsw == bsh);

            if must_split {
                let cbw = (fi.w_in_b - bo.x).min(bsw); // clipped block width, i.e. having effective pixels
                let cbh = (fi.h_in_b - bo.y).min(bsh);
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
                let w: &mut dyn Writer = if cw.bc.cdef_coded {w_post_cdef} else {w_pre_cdef};
                let tell = w.tell_frac();
                cw.write_partition(w, bo, partition, bsize);
                rd_cost = (w.tell_frac() - tell) as f64 * get_lambda(fi)/ ((1 << OD_BITRES) as f64);
            }

            let four_partitions = [
                bo,
                &BlockOffset{ x: bo.x + hbsw as usize, y: bo.y },
                &BlockOffset{ x: bo.x, y: bo.y + hbsh as usize },
                &BlockOffset{ x: bo.x + hbsw as usize, y: bo.y + hbsh as usize }
            ];
            let partitions = get_sub_partitions(&four_partitions, partition);

            // If either of horz or vert partition types is being tested,
            // two partitioned rectangles, defined in 'partitions', of the current block
            // is passed to encode_partition_bottomup()
            for offset in partitions {
                if let (cost, Some(mode_decision)) = encode_partition_bottomup(
                    fi,
                    fs,
                    cw,
                    w_pre_cdef,
                    w_post_cdef,
                    subsize,
                    offset,
                    pmvs,//&best_decision.mvs[0]
                    best_rd
                ) {
                    rd_cost += cost;
                    if rd_cost > best_rd || rd_cost > ref_rd_cost { break; }
                    else { child_modes.push(mode_decision); }
                }
            };

            if rd_cost < best_rd && rd_cost < ref_rd_cost {
                best_rd = rd_cost;
                best_partition = partition;
                best_pred_modes = child_modes.clone();
            }
        }

        // If the best partition is not PARTITION_SPLIT or PARTITION_INVALID, recode it
        if best_partition != PartitionType::PARTITION_SPLIT &&
            best_partition != PartitionType::PARTITION_INVALID {
            cw.rollback(&cw_checkpoint);
            w_pre_cdef.rollback(&w_pre_checkpoint);
            w_post_cdef.rollback(&w_post_checkpoint);

            assert!(best_partition != PartitionType::PARTITION_NONE || !must_split);
            let subsize = bsize.subsize(best_partition);

            if bsize.gte(BlockSize::BLOCK_8X8) {
                let w: &mut dyn Writer = if cw.bc.cdef_coded {w_post_cdef} else {w_pre_cdef};
                cw.write_partition(w, bo, best_partition, bsize);
            }
            for mode in best_pred_modes {
                assert!(subsize == mode.bsize);
                let offset = mode.bo.clone();
                // FIXME: redundant block re-encode
                encode_block_with_modes(fi, fs, cw, w_pre_cdef, w_post_cdef,
                                        mode.bsize, &offset, &mode);
            }
        }
    }

    if best_partition != PartitionType::PARTITION_INVALID {
        let subsize = bsize.subsize(best_partition);

        if bsize.gte(BlockSize::BLOCK_8X8) &&
            (bsize == BlockSize::BLOCK_8X8 || best_partition != PartitionType::PARTITION_SPLIT) {
            cw.bc.update_partition_context(bo, subsize, bsize);
        }
    }
    (best_rd, Some(best_decision))
}

fn encode_partition_topdown(fi: &FrameInvariants, fs: &mut FrameState,
            cw: &mut ContextWriter, w_pre_cdef: &mut dyn Writer, w_post_cdef: &mut dyn Writer,
            bsize: BlockSize, bo: &BlockOffset, block_output: &Option<RDOOutput>,
            pmvs: &[[Option<MotionVector>; REF_FRAMES]; 5]
) {

    if bo.x >= cw.bc.cols || bo.y >= cw.bc.rows {
        return;
    }
    let bsw = bsize.width_mi();
    let bsh = bsize.height_mi();
    let is_square = bsize.is_sqr();

    // Always split if the current partition is too large
    let must_split = (bo.x + bsw as usize > fi.w_in_b ||
        bo.y + bsh as usize > fi.h_in_b ||
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
        let cbw = (fi.w_in_b - bo.x).min(bsw); // clipped block width, i.e. having effective pixels
        let cbh = (fi.h_in_b - bo.y).min(bsh);

        if cbw == bsw/2 && cbh == bsh { split_vert = true; }
        if cbh == bsh/2 && cbw == bsw { split_horz = true; }
    }

    if must_split && (!split_vert && !split_horz) {
        // Oversized blocks are split automatically
        partition = PartitionType::PARTITION_SPLIT;
    } else if must_split || (bsize > fi.min_partition_size && is_square) {
        // Blocks of sizes within the supported range are subjected to a partitioning decision
        let mut partition_types: Vec<PartitionType> = Vec::new();
        if must_split {
            partition_types.push(PartitionType::PARTITION_SPLIT);
            if split_horz { partition_types.push(PartitionType::PARTITION_HORZ); };
            if split_vert { partition_types.push(PartitionType::PARTITION_VERT); };
        }
        else {
            //partition_types.append(&mut RAV1E_PARTITION_TYPES.to_vec());
            partition_types.push(PartitionType::PARTITION_NONE);
            partition_types.push(PartitionType::PARTITION_SPLIT);
        }
        rdo_output = rdo_partition_decision(fi, fs, cw,
            w_pre_cdef, w_post_cdef, bsize, bo, &rdo_output, pmvs, &partition_types);
        partition = rdo_output.part_type;
    } else {
        // Blocks of sizes below the supported range are encoded directly
        partition = PartitionType::PARTITION_NONE;
    }

    assert!(PartitionType::PARTITION_NONE <= partition &&
            partition < PartitionType::PARTITION_INVALID);

    let subsize = bsize.subsize(partition);

    if bsize.gte(BlockSize::BLOCK_8X8) && is_square {
        let w: &mut dyn Writer = if cw.bc.cdef_coded {w_post_cdef} else {w_pre_cdef};
        cw.write_partition(w, bo, partition, bsize);
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
                        ((bo.x & 32) >> 5) + ((bo.y & 32) >> 4) + 1
                    };
                    let spmvs = &pmvs[pmv_idx];

                    // Make a prediction mode decision for blocks encoded with no rdo_partition_decision call (e.g. edges)
                    rdo_mode_decision(fi, fs, cw, bsize, bo, spmvs, false).part_modes[0].clone()
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
                rdo_tx_size_type(fi, fs, cw, bsize, bo, mode_luma, ref_frames, mvs, skip);

            let mut mv_stack = Vec::new();
            let is_compound = ref_frames[1] != NONE_FRAME;
            let mode_context = cw.find_mvrefs(bo, ref_frames, &mut mv_stack, bsize, fi, is_compound);

            // TODO proper remap when is_compound is true
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
                        mode_luma =
                            if mv_stack.len() == 0 { PredictionMode::NEARESTMV }
                            else if mv_stack.len() == 1 { PredictionMode::NEAR0MV }
                            else { PredictionMode::GLOBALMV };
                    }
                    mode_chroma = mode_luma;
                }
            }

            // FIXME: every final block that has gone through the RDO decision process is encoded twice
            cdef_coded = encode_block_a(&fi.sequence, fs, cw, if cdef_coded  {w_post_cdef} else {w_pre_cdef},
                         bsize, bo, skip);
            encode_block_b(fi, fs, cw, if cdef_coded  {w_post_cdef} else {w_pre_cdef},
                          mode_luma, mode_chroma, ref_frames, mvs, bsize, bo, skip, cfl,
                          tx_size, tx_type, mode_context, &mv_stack, false);
        },
        PARTITION_SPLIT |
        PARTITION_HORZ |
        PARTITION_VERT => {
            let num_modes = if partition == PARTITION_SPLIT { 1 }
                            else { 1 };

            if rdo_output.part_modes.len() >= num_modes {
                // The optimal prediction modes for each split block is known from an rdo_partition_decision() call
                assert!(subsize != BlockSize::BLOCK_INVALID);

                for mode in rdo_output.part_modes {
                    let offset = mode.bo.clone();

                    // Each block is subjected to a new splitting decision
                    encode_partition_topdown(fi, fs, cw, w_pre_cdef, w_post_cdef, subsize, &offset,
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
                bo,
                &BlockOffset{ x: bo.x + hbsw as usize, y: bo.y },
                &BlockOffset{ x: bo.x, y: bo.y + hbsh as usize },
                &BlockOffset{ x: bo.x + hbsw as usize, y: bo.y + hbsh as usize }
                ];
                let partitions = get_sub_partitions(&four_partitions, partition);

                partitions.iter().for_each(|&offset| {
                        encode_partition_topdown(
                            fi,
                            fs,
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
        _ => { assert!(false); },
    }

    if bsize.gte(BlockSize::BLOCK_8X8) &&
        (bsize == BlockSize::BLOCK_8X8 || partition != PartitionType::PARTITION_SPLIT) {
            cw.bc.update_partition_context(bo, subsize, bsize);
    }
}

fn encode_tile(fi: &FrameInvariants, fs: &mut FrameState) -> Vec<u8> {
    let mut w = WriterEncoder::new();

    let fc = if fi.primary_ref_frame == PRIMARY_REF_NONE {
      CDFContext::new(fi.base_q_idx)
    } else {
      match fi.rec_buffer.frames[fi.ref_frames[fi.primary_ref_frame as usize] as usize] {
        Some(ref rec) => rec.cdfs,
        None => CDFContext::new(fi.base_q_idx)
      }
    };

    let bc = BlockContext::new(fi.w_in_b, fi.h_in_b);
    // For now, restoration unit size is locked to superblock size.
    let mut cw = ContextWriter::new(fc, bc);

    // initial coarse ME loop
    let mut frame_pmvs = Vec::new();

    for sby in 0..fi.sb_height {
        for sbx in 0..fi.sb_width {
            let sbo = SuperBlockOffset { x: sbx, y: sby };
            let bo = sbo.block_offset(0, 0);
            let mut pmvs: [Option<MotionVector>; REF_FRAMES] = [None; REF_FRAMES];
            for i in 0..INTER_REFS_PER_FRAME {
                let r = fi.ref_frames[i] as usize;
                if pmvs[r].is_none() {
                    assert!(!fi.sequence.use_128x128_superblock);
                    pmvs[r] = estimate_motion_ss4(fi, fs, BlockSize::BLOCK_64X64, r, &bo);
                }
            }
            frame_pmvs.push(pmvs);
        }
    }

    // main loop
    for sby in 0..fi.sb_height {
        cw.bc.reset_left_contexts();

        for sbx in 0..fi.sb_width {
            let mut w_pre_cdef = WriterRecorder::new();
            let mut w_post_cdef = WriterRecorder::new();
            let mut cdef_index = 0;
            let sbo = SuperBlockOffset { x: sbx, y: sby };
            let bo = sbo.block_offset(0, 0);
            cw.bc.cdef_coded = false;
            cw.bc.code_deltas = fi.delta_q_present;

            // Do subsampled ME
            let mut pmvs: [[Option<MotionVector>; REF_FRAMES]; 5] = [[None; REF_FRAMES]; 5];
            for i in 0..INTER_REFS_PER_FRAME {
                let r = fi.ref_frames[i] as usize;
                if pmvs[0][r].is_none() {
                    pmvs[0][r] = frame_pmvs[sby * fi.sb_width + sbx][r];
                    if let Some(pmv) = pmvs[0][r] {
                        let pmv_w = if sbx > 0 {
                            frame_pmvs[sby * fi.sb_width + sbx - 1][r]
                        } else {
                            None
                        };
                        let pmv_e = if sbx < fi.sb_width - 1 {
                            frame_pmvs[sby * fi.sb_width + sbx + 1][r]
                        } else {
                            None
                        };
                        let pmv_n = if sby > 0 {
                            frame_pmvs[sby * fi.sb_width + sbx - fi.sb_width][r]
                        } else {
                            None
                        };
                        let pmv_s = if sby < fi.sb_height - 1 {
                            frame_pmvs[sby * fi.sb_width + sbx + fi.sb_width][r]
                        } else {
                            None
                        };

                        assert!(!fi.sequence.use_128x128_superblock);
                        pmvs[1][r] = estimate_motion_ss2(
                            fi, fs, BlockSize::BLOCK_32X32, r, &sbo.block_offset(0, 0), &[Some(pmv), pmv_w, pmv_n]
                        );
                        pmvs[2][r] = estimate_motion_ss2(
                            fi, fs, BlockSize::BLOCK_32X32, r, &sbo.block_offset(8, 0), &[Some(pmv), pmv_e, pmv_n]
                        );
                        pmvs[3][r] = estimate_motion_ss2(
                            fi, fs, BlockSize::BLOCK_32X32, r, &sbo.block_offset(0, 8), &[Some(pmv), pmv_w, pmv_s]
                        );
                        pmvs[4][r] = estimate_motion_ss2(
                            fi, fs, BlockSize::BLOCK_32X32, r, &sbo.block_offset(8, 8), &[Some(pmv), pmv_e, pmv_s]
                        );
                    }
                }
            }

            // Encode SuperBlock
            if fi.config.speed_settings.encode_bottomup {
                encode_partition_bottomup(fi, fs, &mut cw,
                                          &mut w_pre_cdef, &mut w_post_cdef,
                                          BlockSize::BLOCK_64X64, &bo, &pmvs, std::f64::MAX);
            }
            else {
                encode_partition_topdown(fi, fs, &mut cw,
                                         &mut w_pre_cdef, &mut w_post_cdef,
                                         BlockSize::BLOCK_64X64, &bo, &None, &pmvs);
            }

            // CDEF has to be decisded before loop restoration, but coded after
            if cw.bc.cdef_coded {
                cdef_index = rdo_cdef_decision(&sbo, fi, fs, &mut cw);
                cw.bc.set_cdef(&sbo, cdef_index);
            }

            // loop restoration must be decided last but coded before anything else
            if fi.sequence.enable_restoration {
                fs.restoration.lrf_optimize_superblock(&sbo, fi, &mut cw);
                cw.write_lrf(&mut w, fi, &mut fs.restoration, &sbo);
            }

            // Once loop restoration is coded, we can replay the initial block bits
            w_pre_cdef.replay(&mut w);

            if cw.bc.cdef_coded {
                // CDEF index must be written in the middle, we can code it now
                cw.write_cdef(&mut w, cdef_index, fi.cdef_bits);
                // ...and then finally code what comes after the CDEF index
                w_post_cdef.replay(&mut w);
            }
        }
    }
    /* TODO: Don't apply if lossless */
    deblock_filter_optimize(fi, fs, &mut cw.bc);
    if fs.deblock.levels[0] != 0 || fs.deblock.levels[1] != 0 {
        deblock_filter_frame(fs, &mut cw.bc, fi.sequence.bit_depth);
    }
    {
      // Until the loop filters are pipelined, we'll need to keep
      // around a copy of both the pre- and post-cdef frame.
      let pre_cdef_frame = fs.rec.clone();

      /* TODO: Don't apply if lossless */
      if fi.sequence.enable_cdef {
        cdef_filter_frame(fi, &mut fs.rec, &mut cw.bc);
      }
      /* TODO: Don't apply if lossless */
      if fi.sequence.enable_restoration {
        fs.restoration.lrf_filter_frame(&mut fs.rec, &pre_cdef_frame, &fi);
      }
    }

    fs.cdfs = cw.fc;
    fs.cdfs.reset_counts();

    let mut h = w.done();
    h.push(0); // superframe anti emulation
    h
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

pub fn encode_frame(fi: &mut FrameInvariants, fs: &mut FrameState) -> Vec<u8> {
    let mut packet = Vec::new();
    if fi.show_existing_frame {
        write_obus(&mut packet, fi, fs).unwrap();
        match fi.rec_buffer.frames[fi.frame_to_show_map_idx as usize] {
            Some(ref rec) => for p in 0..3 {
                fs.rec.planes[p].data.copy_from_slice(rec.frame.planes[p].data.as_slice());
            },
            None => (),
        }
    } else {
        if !fi.intra_only {
            for i in 0..INTER_REFS_PER_FRAME {
                fi.ref_frame_sign_bias[i] =
                if !fi.sequence.enable_order_hint {
                    false
                } else if let Some(ref rec) = fi.rec_buffer.frames[fi.ref_frames[i] as usize] {
                    let hint = rec.order_hint;
                    fi.sequence.get_relative_dist(hint, fi.order_hint) > 0
                } else {
                    false
                };
            }
        }

        fs.input_hres.downsample_from(&fs.input.planes[0]);
        fs.input_hres.pad(fi.width, fi.height);
        fs.input_qres.downsample_from(&fs.input_hres);
        fs.input_qres.pad(fi.width, fi.height);

        segmentation_optimize(fi, fs);

        let tile = encode_tile(fi, fs); // actually tile group

        write_obus(&mut packet, fi, fs).unwrap();
        let mut buf1 = Vec::new();
        {
            let mut bw1 = BitWriter::endian(&mut buf1, BigEndian);
            bw1.write_obu_header(OBU_Type::OBU_TILE_GROUP, 0).unwrap();
        }
        packet.write_all(&buf1).unwrap();
        buf1.clear();

        let obu_payload_size = tile.len() as u64;
        {
            let mut bw1 = BitWriter::endian(&mut buf1, BigEndian);
            // uleb128()
            let mut coded_payload_length = [0 as u8; 8];
            let leb_size = aom_uleb_encode(obu_payload_size, &mut coded_payload_length);
            for i in 0..leb_size {
                bw1.write(8, coded_payload_length[i]).unwrap();
            }
        }
        packet.write_all(&buf1).unwrap();
        buf1.clear();

      packet.write_all(&tile).unwrap();
    }
    packet
}

pub fn update_rec_buffer(fi: &mut FrameInvariants, fs: FrameState) {
  let rfs = Rc::new(
    ReferenceFrame {
      order_hint: fi.order_hint,
      frame: fs.rec,
      input_hres: fs.input_hres,
      input_qres: fs.input_qres,
      cdfs: fs.cdfs
    }
  );
  for i in 0..(REF_FRAMES as usize) {
    if (fi.refresh_frame_flags & (1 << i)) != 0 {
      fi.rec_buffer.frames[i] = Some(Rc::clone(&rfs));
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
