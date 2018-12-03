use std::fmt;
use std::io::Write;
use std::sync::Arc;

use bitstream_io::BigEndian;
use bitstream_io::BitWriter;

use api::EncoderConfig;
use context::CDFContext;
use context::SuperBlockOffset;
use context::MAX_SB_SIZE;
use context::PLANES;
use encoder::aom_uleb_encode;
use encoder::deblock::DeblockState;
use encoder::headers::UncompressedHeader;
use encoder::reference::ReferenceFramesSet;
use encoder::reference::ReferenceMode;
use encoder::reference::ALL_REF_FRAMES_MASK;
use encoder::segmentation::SegmentationState;
use encoder::sequence::Sequence;
use encoder::tile::encode_tile;
use encoder::write_obus;
use encoder::OBU_Type;
use encoder::Tune;
use encoder::PRIMARY_REF_NONE;
use lrf::RESTORE_NONE;
use partition::BlockSize;
use partition::GlobalMVMode;
use partition::ALTREF_FRAME;
use partition::INTER_REFS_PER_FRAME;
use partition::LAST2_FRAME;
use partition::LAST3_FRAME;
use partition::LAST_FRAME;
use partition::NONE_FRAME;
use partition::SUBPEL_FILTER_SIZE;
use plane::Plane;
use quantize::QuantizationContext;
use segmentation::segmentation_optimize;
use util::Fixed;

#[derive(Debug, Clone)]
pub struct Frame {
  pub planes: [Plane; 3]
}

const FRAME_MARGIN: usize = 16 + SUBPEL_FILTER_SIZE;

impl Frame {
  pub fn new(width: usize, height: usize) -> Frame {
    Frame {
      planes: [
        Plane::new(
          width,
          height,
          0,
          0,
          MAX_SB_SIZE + FRAME_MARGIN,
          MAX_SB_SIZE + FRAME_MARGIN
        ),
        Plane::new(
          width / 2,
          height / 2,
          1,
          1,
          MAX_SB_SIZE / 2 + FRAME_MARGIN,
          MAX_SB_SIZE / 2 + FRAME_MARGIN
        ),
        Plane::new(
          width / 2,
          height / 2,
          1,
          1,
          MAX_SB_SIZE / 2 + FRAME_MARGIN,
          MAX_SB_SIZE / 2 + FRAME_MARGIN
        )
      ]
    }
  }

  pub fn pad(&mut self, w: usize, h: usize) {
    for p in self.planes.iter_mut() {
      p.pad(w, h);
    }
  }

  pub fn window(&self, sbo: &SuperBlockOffset) -> Frame {
    Frame {
      planes: [
        self.planes[0].window(&sbo.plane_offset(&self.planes[0].cfg)),
        self.planes[1].window(&sbo.plane_offset(&self.planes[1].cfg)),
        self.planes[2].window(&sbo.plane_offset(&self.planes[2].cfg))
      ]
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
  x: usize
}

impl<'a> PixelIter<'a> {
  pub fn new(planes: &'a [Plane; 3]) -> Self {
    PixelIter { planes, y: 0, x: 0 }
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
      self.planes[2].p(self.x / 2, self.y / 2)
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

#[derive(Debug)]
pub struct FrameState {
  pub input: Arc<Frame>,
  pub input_hres: Plane, // half-resolution version of input luma
  pub input_qres: Plane, // quarter-resolution version of input luma
  pub rec: Frame,
  pub qc: QuantizationContext,
  pub cdfs: CDFContext,
  pub deblock: DeblockState,
  pub segmentation: SegmentationState
}

impl FrameState {
  pub fn new(fi: &FrameInvariants) -> FrameState {
    FrameState::new_with_frame(
      fi,
      Arc::new(Frame::new(fi.padded_w, fi.padded_h))
    )
  }

  pub fn new_with_frame(
    fi: &FrameInvariants, frame: Arc<Frame>
  ) -> FrameState {
    FrameState {
      input: frame,
      input_hres: Plane::new(
        fi.padded_w / 2,
        fi.padded_h / 2,
        1,
        1,
        (MAX_SB_SIZE + FRAME_MARGIN) / 2,
        (MAX_SB_SIZE + FRAME_MARGIN) / 2
      ),
      input_qres: Plane::new(
        fi.padded_w / 4,
        fi.padded_h / 4,
        2,
        2,
        (MAX_SB_SIZE + FRAME_MARGIN) / 4,
        (MAX_SB_SIZE + FRAME_MARGIN) / 4
      ),
      rec: Frame::new(fi.padded_w, fi.padded_h),
      qc: Default::default(),
      cdfs: CDFContext::new(0),
      deblock: Default::default(),
      segmentation: Default::default()
    }
  }

  pub fn window(&self, sbo: &SuperBlockOffset) -> FrameState {
    FrameState {
      input: Arc::new(self.input.window(sbo)),
      input_hres: self
        .input_hres
        .window(&sbo.plane_offset(&self.input_hres.cfg)),
      input_qres: self
        .input_qres
        .window(&sbo.plane_offset(&self.input_qres.cfg)),
      rec: self.rec.window(sbo),
      qc: self.qc,
      cdfs: self.cdfs,
      deblock: self.deblock,
      segmentation: self.segmentation
    }
  }
}

// Frame Invariants are invariant inside a frame
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct FrameInvariants {
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
  /// a bitmask that specifies which
  /// reference frame slots will be updated with the current frame
  /// after it is decoded.
  pub refresh_frame_flags: u32,
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
  pub lrf_types: [u8; PLANES],
  pub delta_q_present: bool,
  pub config: EncoderConfig,
  pub ref_frames: [u8; INTER_REFS_PER_FRAME],
  pub ref_frame_sign_bias: [bool; INTER_REFS_PER_FRAME],
  pub rec_buffer: ReferenceFramesSet,
  pub base_q_idx: u8,
  pub me_range_scale: u8,
  pub use_tx_domain_distortion: bool,
  pub inter_cfg: Option<InterPropsConfig>
}

impl FrameInvariants {
  pub fn new(
    width: usize, height: usize, config: EncoderConfig
  ) -> FrameInvariants {
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
    let use_tx_domain_distortion =
      config.tune == Tune::Psnr && config.speed_settings.tx_domain_distortion;

    FrameInvariants {
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
      cdef_y_strengths: [
        0 * 4 + 0,
        1 * 4 + 0,
        2 * 4 + 1,
        3 * 4 + 1,
        5 * 4 + 2,
        7 * 4 + 3,
        10 * 4 + 3,
        13 * 4 + 3
      ],
      cdef_uv_strengths: [
        0 * 4 + 0,
        1 * 4 + 0,
        2 * 4 + 1,
        3 * 4 + 1,
        5 * 4 + 2,
        7 * 4 + 3,
        10 * 4 + 3,
        13 * 4 + 3
      ],
      lrf_types: [RESTORE_NONE, RESTORE_NONE, RESTORE_NONE],
      delta_q_present: false,
      config,
      ref_frames: [0; INTER_REFS_PER_FRAME],
      ref_frame_sign_bias: [false; INTER_REFS_PER_FRAME],
      rec_buffer: ReferenceFramesSet::new(),
      base_q_idx: config.quantizer as u8,
      me_range_scale: 1,
      use_tx_domain_distortion,
      inter_cfg: None
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
    fi.base_q_idx = (fi.config.quantizer.max(1 + q_boost).min(255 + q_boost)
      - q_boost) as u8;
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
      group_idx
    })
  }

  /// Returns the created FrameInvariants along with a bool indicating success.
  /// This interface provides simpler usage, because we always need the produced
  /// FrameInvariants regardless of success or failure.
  pub fn new_inter_frame(
    previous_fi: &Self, segment_start_frame: u64, idx_in_segment: u64,
    next_keyframe: u64
  ) -> (Self, bool) {
    let mut fi = previous_fi.clone();
    fi.frame_type = FrameType::INTER;
    fi.intra_only = false;
    fi.apply_inter_props_cfg(idx_in_segment);
    let inter_cfg = fi.inter_cfg.unwrap();

    fi.order_hint = (inter_cfg.group_src_len * inter_cfg.group_idx
      + if inter_cfg.reorder
        && inter_cfg.idx_in_group < inter_cfg.pyramid_depth
      {
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
      pos_to_lvl(
        inter_cfg.idx_in_group - inter_cfg.pyramid_depth + 1,
        inter_cfg.pyramid_depth
      )
    };

    // Frames with lvl == 0 are stored in slots 0..4 and frames with higher values
    // of lvl in slots 4..8
    let slot_idx = if lvl == 0 {
      (fi.order_hint >> inter_cfg.pyramid_depth) % 4 as u32
    } else {
      3 + lvl as u32
    };
    fi.show_frame =
      !inter_cfg.reorder || inter_cfg.idx_in_group >= inter_cfg.pyramid_depth;
    fi.show_existing_frame = fi.show_frame
      && inter_cfg.reorder
      && (inter_cfg.idx_in_group - inter_cfg.pyramid_depth + 1).count_ones()
        == 1
      && inter_cfg.idx_in_group != inter_cfg.pyramid_depth;
    fi.frame_to_show_map_idx = slot_idx;
    fi.refresh_frame_flags =
      if fi.show_existing_frame { 0 } else { 1 << slot_idx };

    let q_drop = 15 * lvl as usize;
    fi.base_q_idx = (fi.config.quantizer.min(255 - q_drop) + q_drop) as u8;

    let second_ref_frame = if !inter_cfg.multiref {
      NONE_FRAME
    } else if !inter_cfg.reorder || inter_cfg.idx_in_group == 0 {
      LAST2_FRAME
    } else {
      ALTREF_FRAME
    };
    let ref_in_previous_group = LAST3_FRAME;

    // reuse probability estimates from previous frames only in top level frames
    fi.primary_ref_frame = if lvl > 0 {
      PRIMARY_REF_NONE
    } else {
      (ref_in_previous_group - LAST_FRAME) as u32
    };

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

    fi.reference_mode = if inter_cfg.multiref
      && inter_cfg.reorder
      && inter_cfg.idx_in_group != 0
    {
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
  pub group_idx: u64
}

#[allow(dead_code, non_camel_case_types)]
#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(C)]
pub enum FrameType {
  KEY,
  INTER,
  INTRA_ONLY,
  SWITCH
}

impl fmt::Display for FrameType {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      FrameType::KEY => write!(f, "Key frame"),
      FrameType::INTER => write!(f, "Inter frame"),
      FrameType::INTRA_ONLY => write!(f, "Intra only frame"),
      FrameType::SWITCH => write!(f, "Switching frame")
    }
  }
}

pub fn encode_frame(
  sequence: &mut Sequence, fi: &mut FrameInvariants, fs: &mut FrameState
) -> Vec<u8> {
  let mut packet = Vec::new();
  if fi.show_existing_frame {
    //write_uncompressed_header(&mut packet, sequence, fi).unwrap();
    write_obus(&mut packet, sequence, fi, fs).unwrap();
    match fi.rec_buffer.frames[fi.frame_to_show_map_idx as usize] {
      Some(ref rec) =>
        for p in 0..3 {
          fs.rec.planes[p]
            .data
            .copy_from_slice(rec.frame.planes[p].data.as_slice());
        },
      None => ()
    }
  } else {
    if !fi.intra_only {
      for i in 0..INTER_REFS_PER_FRAME {
        fi.ref_frame_sign_bias[i] = if !sequence.enable_order_hint {
          false
        } else if let Some(ref rec) =
          fi.rec_buffer.frames[fi.ref_frames[i] as usize]
        {
          let hint = rec.order_hint;
          sequence.get_relative_dist(hint, fi.order_hint) > 0
        } else {
          false
        };
      }
    }

    fs.input_hres.downsample_from(&fs.input.planes[0]);
    fs.input_hres.pad(fi.width, fi.height);
    fs.input_qres.downsample_from(&fs.input_hres);
    fs.input_qres.pad(fi.width, fi.height);

    let bit_depth = sequence.bit_depth;

    segmentation_optimize(fi, fs);

    let tile = encode_tile(sequence, fi, fs, bit_depth); // actually tile group

    //write_uncompressed_header(&mut packet, sequence, fi).unwrap();
    write_obus(&mut packet, sequence, fi, fs).unwrap();
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
      let leb_size =
        aom_uleb_encode(obu_payload_size, &mut coded_payload_length);
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
