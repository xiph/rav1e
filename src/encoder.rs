
use context::*;
use partition::*;
use transform::*;
use quantize::*;
use plane::*;
use rdo::*;
use ec::*;
use std::fmt;
use util::*;
use cdef::*;

use bitstream_io::{BE, LE, BitWriter};
use std::rc::Rc;
use std::io::*;
use std::io;
use std;

extern {
    pub fn av1_rtcd();
    pub fn aom_dsp_rtcd();
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub planes: [Plane; 3]
}

impl Frame {
    pub fn new(width: usize, height:usize) -> Frame {
        Frame {
            planes: [
                Plane::new(width, height, 0, 0),
                Plane::new(width/2, height/2, 1, 1),
                Plane::new(width/2, height/2, 1, 1)
            ]
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReferenceFrame {
  pub frame: Frame,
  pub cdfs: CDFContext,
}

#[derive(Debug)]
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

#[derive(Copy, Clone)]
pub struct Sequence {
  // OBU Sequence header of AV1
    pub profile: u8,
    pub num_bits_width: u32,
    pub num_bits_height: u32,
    pub bit_depth: usize,
    pub chroma_sampling: ChromaSampling,
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
    pub enable_filter_intra: bool,         // enables/disables filterintra
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
    pub fn new(width: usize, height: usize, bit_depth: usize, chroma_sampling: ChromaSampling) -> Sequence {
        let width_bits = 32 - (width as u32).leading_zeros();
        let height_bits = 32 - (height as u32).leading_zeros();
        assert!(width_bits <= 16);
        assert!(height_bits <= 16);

        let profile = if bit_depth == 12 {
            2
        } else if chroma_sampling == ChromaSampling::Cs444 {
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
            bit_depth: bit_depth,
            chroma_sampling: chroma_sampling,
            max_frame_width: width as u32,
            max_frame_height: height as u32,
            frame_id_numbers_present_flag: false,
            frame_id_length: 0,
            delta_frame_id_length: 0,
            use_128x128_superblock: false,
            order_hint_bits_minus_1: 0,
            force_screen_content_tools: 0,
            force_integer_mv: 2,
            still_picture: false,
            reduced_still_picture_hdr: false,
            monochrome: false,
            enable_filter_intra: true,
            enable_intra_edge_filter: true,
            enable_interintra_compound: false,
            enable_masked_compound: false,
            enable_dual_filter: false,
            enable_order_hint: false,
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
}

#[derive(Debug)]
pub struct FrameState {
    pub input: Frame,
    pub rec: Frame,
    pub qc: QuantizationContext,
    pub cdfs: CDFContext,
}

impl FrameState {
    pub fn new(fi: &FrameInvariants) -> FrameState {
        FrameState {
            input: Frame::new(fi.padded_w, fi.padded_h),
            rec: Frame::new(fi.padded_w, fi.padded_h),
            qc: Default::default(),
            cdfs: CDFContext::new(0),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DeblockState {
    pub levels: [u8; PLANES+1],
    pub sharpness: u8,
    pub deltas_enabled: bool,
    pub delta_updates_enabled: bool,
    pub ref_deltas_enabled: bool,
    pub ref_deltas: [i8; REF_FRAMES],
    pub mode_deltas_enabled: bool,
    pub mode_deltas: [i8; 2],
    pub block_deltas_enabled: bool,
    pub block_delta_shift: u8,
    pub block_delta_multi: bool,
}

impl Default for DeblockState {
    fn default() -> Self {
        DeblockState {
            levels: [0; PLANES+1],
            sharpness: 0,
            deltas_enabled: false,
            delta_updates_enabled: false,
            ref_deltas_enabled: false,
            ref_deltas: [1, 0, 0, 0, 0, -1, -1, -1],
            mode_deltas_enabled: false,
            mode_deltas: [0, 0],
            block_deltas_enabled: false,
            block_delta_shift: 0,
            block_delta_multi: false
        }
    }
}

// Frame Invariants are invariant inside a frame
#[allow(dead_code)]
#[derive(Debug)]
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
    pub show_frame: bool,
    pub showable_frame: bool,
    pub error_resilient: bool,
    pub intra_only: bool,
    pub allow_high_precision_mv: bool,
    pub frame_type: FrameType,
    pub show_existing_frame: bool,
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
    pub ref_frames: [usize; INTER_REFS_PER_FRAME],
    pub rec_buffer: ReferenceFramesSet,
    pub deblock: DeblockState
}

impl FrameInvariants {
    pub fn new(width: usize, height: usize, config: EncoderConfig) -> FrameInvariants {
        // Speed level decides the minimum partition size, i.e. higher speed --> larger min partition size,
        // with exception that SBs on right or bottom frame borders split down to BLOCK_4X4.
        // At speed = 0, RDO search is exhaustive.
        let mut min_partition_size = if config.speed <= 1 { BlockSize::BLOCK_4X4 }
                                 else if config.speed <= 2 { BlockSize::BLOCK_8X8 }
                                 else if config.speed <= 3 { BlockSize::BLOCK_16X16 }
                                 else { BlockSize::BLOCK_32X32 };

        if config.tune == Tune::Psychovisual {
            if min_partition_size < BlockSize::BLOCK_8X8 {
                // TODO: Display message that min partition size is enforced to 8x8
                min_partition_size = BlockSize::BLOCK_8X8;
                println!("If tune=Psychovisual is used, min partition size is enforced to 8x8");
            }
        }
        let use_reduced_tx_set = config.speed > 1;

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
            show_frame: true,
            showable_frame: true,
            error_resilient: false,
            intra_only: false,
            allow_high_precision_mv: false,
            frame_type: FrameType::KEY,
            show_existing_frame: false,
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
            rec_buffer: ReferenceFramesSet::new(),
            deblock: Default::default()
        }
    }

    pub fn new_frame_state(&self) -> FrameState {
        FrameState {
            input: Frame::new(self.padded_w, self.padded_h),
            rec: Frame::new(self.padded_w, self.padded_h),
            qc: Default::default(),
            cdfs: CDFContext::new(0),
        }
    }
}

impl fmt::Display for FrameInvariants{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Frame {} - {}", self.number, self.frame_type)
    }
}

#[allow(dead_code,non_camel_case_types)]
#[derive(Debug,PartialEq,Clone,Copy)]
pub enum FrameType {
    KEY,
    INTER,
    INTRA_ONLY,
    SWITCH,
}

//const REFERENCE_MODES: usize = 3;

#[allow(dead_code,non_camel_case_types)]
#[derive(Debug,PartialEq)]
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

#[derive(Copy, Clone, Debug)]
pub struct EncoderConfig {
    pub limit: u64,
    pub quantizer: usize,
    pub speed: usize,
    pub tune: Tune
}

impl Default for EncoderConfig {
    fn default() -> Self {
        EncoderConfig {
            limit: 0,
            quantizer: 100,
            speed: 0,
            tune: Tune::Psnr,
        }
    }
}

pub fn write_ivf_header(output_file: &mut dyn io::Write, width: usize, height: usize, num: usize, den: usize) {
    let mut bw = BitWriter::<LE>::new(output_file);
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
    let mut bw = BitWriter::<LE>::new(output_file);
    bw.write(32, data.len() as u32).unwrap();
    bw.write(64, pts).unwrap();
    bw.write_bytes(data).unwrap();
}

trait UncompressedHeader {
    // Start of OBU Headers
    fn write_obu_header(&mut self, obu_type: OBU_Type, obu_extension: u32)
            -> io::Result<()>;
    fn write_sequence_header_obu(&mut self, seq: &mut Sequence, fi: &FrameInvariants)
            -> io::Result<()>;
    fn write_frame_header_obu(&mut self, seq: &Sequence, fi: &FrameInvariants)
            -> io::Result<()>;
    fn write_sequence_header(&mut self, seq: &mut Sequence, fi: &FrameInvariants)
                                    -> io::Result<()>;
    fn write_color_config(&mut self, seq: &mut Sequence) -> io::Result<()>;
    // End of OBU Headers

    fn write_frame_size(&mut self, fi: &FrameInvariants) -> io::Result<()>;
    fn write_deblock_filter_a(&mut self, fi: &FrameInvariants) -> io::Result<()>;
    fn write_deblock_filter_b(&mut self, fi: &FrameInvariants) -> io::Result<()>;
    fn write_frame_cdef(&mut self, seq: &Sequence, fi: &FrameInvariants) -> io::Result<()>;
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

impl<'a> UncompressedHeader for BitWriter<'a, BE> {
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
            assert!(false);
            //self.write(8, obu_extension & 0xFF)?; size += 8;
        }

        Ok(())
    }

    fn write_sequence_header_obu(&mut self, seq: &mut Sequence, fi: &FrameInvariants)
        -> io::Result<()> {
        self.write(3, seq.profile)?; // profile, 3 bits
        self.write(1, 0)?; // still_picture
        self.write(1, 0)?; // reduced_still_picture
        self.write_bit(false)?; // display model present
        self.write_bit(false)?; // no timing info present
        self.write(5, 0)?; // one operating point
        self.write(12,0)?; // idc
        self.write(5, 31)?; // level
        self.write(1, 0)?; // tier
        if seq.reduced_still_picture_hdr {
            assert!(false);
        }

        self.write_sequence_header(seq, fi)?;

        self.write_color_config(seq)?;

        self.write_bit(seq.film_grain_params_present)?;

        self.write_bit(true)?; // add_trailing_bits

        Ok(())
    }

    fn write_sequence_header(&mut self, seq: &mut Sequence, fi: &FrameInvariants)
        -> io::Result<()> {
        self.write_frame_size(fi)?;

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
        self.write_bit(seq.enable_filter_intra)?;
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

        self.write_bit(high_bd)?; // high bit depth

        if seq.bit_depth == 12 {
            self.write_bit(true)?; // 12-bit
        }

        if seq.profile != 1 {
            self.write_bit(seq.monochrome)?; // monochrome?
        } else {
            unimplemented!(); // 4:4:4 sampling at 8 or 10 bits
        }

        self.write_bit(false)?; // No color description present

        if seq.monochrome {
            assert!(false);
        }

        self.write_bit(false)?; // color range

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

        self.write(2, 0)?; // chroma_sample_position == CSP_UNKNOWN

        self.write_bit(false)?; // separate uv delta q

        Ok(())
    }

#[allow(unused)]
    fn write_frame_header_obu(&mut self, seq: &Sequence, fi: &FrameInvariants)
        -> io::Result<()> {
      if seq.reduced_still_picture_hdr {
        assert!(fi.show_existing_frame);
        assert!(fi.frame_type == FrameType::KEY);
        assert!(fi.show_frame);
      } else {
        if fi.show_existing_frame {
          self.write_bit(true)?; // show_existing_frame=1
          self.write(3, 0)?; // show last frame

          //TODO:
          /* temporal_point_info();
            if seq.decoder_model_info_present_flag &&
              timing_info.equal_picture_interval == 0 {
            // write frame_presentation_delay;
          }
          if seq.frame_id_numbers_present_flag {
            // write display_frame_id;
          }*/

          self.byte_align()?;
          return Ok((()));
        }
        self.write_bit(false)?; // show_existing_frame=0
        self.write(2, fi.frame_type as u32)?;
        self.write_bit(fi.show_frame)?; // show frame

        if fi.show_frame {
          //TODO:
          /* temporal_point_info();
              if seq.decoder_model_info_present_flag &&
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

      if seq.force_screen_content_tools == 2 {
        self.write_bit(fi.allow_screen_content_tools != 0)?;
      } else {
        assert!(fi.allow_screen_content_tools ==
                seq.force_screen_content_tools);
      }

      if fi.allow_screen_content_tools == 2 {
        if seq.force_integer_mv == 2 {
          self.write_bit(fi.force_integer_mv != 0)?;
        } else {
          assert!(fi.force_integer_mv == seq.force_integer_mv);
        }
      } else {
        assert!(fi.allow_screen_content_tools ==
                seq.force_screen_content_tools);
      }

      if seq.frame_id_numbers_present_flag {
        assert!(false); // Not supported by rav1e yet!
        //TODO:
        //let frame_id_len = seq.frame_id_length;
        //self.write(frame_id_len, fi.current_frame_id);
      }

      let mut frame_size_override_flag = false;
      if fi.frame_type == FrameType::SWITCH {
        frame_size_override_flag = true;
      } else if seq.reduced_still_picture_hdr {
        frame_size_override_flag = false;
      } else {
        self.write_bit(frame_size_override_flag)?; // frame size overhead flag
      }

      if seq.enable_order_hint {
        assert!(false); // Not supported by rav1e yet!
      }
      if fi.error_resilient || fi.intra_only {
      } else {
        self.write(PRIMARY_REF_BITS, fi.primary_ref_frame)?;
      }

      if seq.decoder_model_info_present_flag {
        assert!(false); // Not supported by rav1e yet!
      }

      if fi.frame_type == FrameType::KEY {
        if !fi.show_frame {  // unshown keyframe (forward keyframe)
          assert!(false); // Not supported by rav1e yet!
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
        if (fi.error_resilient && seq.enable_order_hint) {
          assert!(false); // Not supported by rav1e yet!
          //for _ in 0..REF_FRAMES {
          //  self.write(order_hint_bits_minus_1,ref_order_hint[i])?; // order_hint
          //}
        }
      }

      // if KEY or INTRA_ONLY frame
      // FIXME: Not sure whether putting frame/render size here is good idea
      if fi.intra_only {
        if frame_size_override_flag {
          assert!(false); // Not supported by rav1e yet!
        }
        if seq.enable_superres {
          assert!(false); // Not supported by rav1e yet!
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
          if seq.enable_order_hint {
            assert!(false); // Not supported by rav1e yet!
            self.write_bit(frame_refs_short_signaling)?;
            if frame_refs_short_signaling {
              assert!(false); // Not supported by rav1e yet!
            }
          }

          for i in 0..7 {
            if !frame_refs_short_signaling {
              self.write(REF_FRAMES_LOG2 as u32, fi.ref_frames[i] as u8)?;
            }
            if seq.frame_id_numbers_present_flag {
              assert!(false); // Not supported by rav1e yet!
            }
          }
          if fi.error_resilient && frame_size_override_flag {
            assert!(false); // Not supported by rav1e yet!
          } else {
            if frame_size_override_flag {
               assert!(false); // Not supported by rav1e yet!
            }
            if seq.enable_superres {
              assert!(false); // Not supported by rav1e yet!
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
          if fi.error_resilient || !seq.enable_ref_frame_mvs {
          } else {
            self.write_bit(fi.use_ref_frame_mvs)?;
          }
        }
      }

      if !seq.reduced_still_picture_hdr && !fi.disable_cdf_update {
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
      assert!(fi.config.quantizer > 0);
      self.write(8,fi.config.quantizer as u8)?; // base_q_idx
      self.write_bit(false)?; // y dc delta q
      self.write_bit(false)?; // uv dc delta q
      self.write_bit(false)?; // uv ac delta q
      self.write_bit(false)?; // no qm

      // segmentation
      self.write_bit(false)?; // segmentation is disabled

      // delta_q
      self.write_bit(false)?; // delta_q_present_flag: no delta q

      // delta_lf_params in the spec
      self.write_deblock_filter_a(fi)?;

      // code for features not yet implemented....

      // loop_filter_params in the spec
      self.write_deblock_filter_b(fi)?;

      // cdef
      self.write_frame_cdef(seq, fi)?;
      // loop restoration
      if seq.enable_restoration {
          self.write(6,0)?; // no y, u or v loop restoration
      }
      self.write_bit(false)?; // tx mode == TX_MODE_SELECT ?

      let mut reference_select = false;
      if !fi.intra_only {
        reference_select = fi.reference_mode != ReferenceMode::SINGLE;
        self.write_bit(reference_select)?;
      }

      let skip_mode_allowed =
        !(fi.intra_only  || !reference_select || !seq.enable_order_hint);
      if skip_mode_allowed {
        unimplemented!();
        self.write_bit(false)?; // skip_mode_present
      }

      if fi.intra_only || fi.error_resilient || !seq.enable_warped_motion {
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

      if seq.film_grain_params_present && fi.show_frame {
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

    fn write_deblock_filter_a(&mut self, fi: &FrameInvariants) -> io::Result<()> {
        if fi.delta_q_present {
            if !fi.allow_intrabc {
                self.write_bit(fi.deblock.block_deltas_enabled)?;
            }
            if fi.deblock.block_deltas_enabled {
                self.write(2,fi.deblock.block_delta_shift)?;
                self.write_bit(fi.deblock.block_delta_multi)?;
            }
        }
        Ok(())
    }

    fn write_deblock_filter_b(&mut self, fi: &FrameInvariants) -> io::Result<()> {
        assert!(fi.deblock.levels[0] < 64);
        self.write(6, fi.deblock.levels[0])?; // loop deblocking filter level 0
        assert!(fi.deblock.levels[1] < 64);
        self.write(6, fi.deblock.levels[1])?; // loop deblocking filter level 1
        if PLANES > 1 && (fi.deblock.levels[0] > 0 || fi.deblock.levels[1] > 0) {
            assert!(fi.deblock.levels[2] < 64);
            self.write(6, fi.deblock.levels[2])?; // loop deblocking filter level 2
            assert!(fi.deblock.levels[3] < 64);
            self.write(6, fi.deblock.levels[3])?; // loop deblocking filter level 3
        }
        self.write(3,0)?; // deblocking filter sharpness
        self.write_bit(fi.deblock.deltas_enabled)?; // loop deblocking filter deltas enabled
        if fi.deblock.deltas_enabled {
            self.write_bit(fi.deblock.delta_updates_enabled)?; // deltas updates enabled
            if fi.deblock.delta_updates_enabled {
                // conditionally write ref delta updates
                let prev_ref_deltas = if fi.primary_ref_frame == PRIMARY_REF_NONE {
                    [1, 0, 0, 0, 0, -1, -1, -1]
                } else {
                    fi.rec_buffer.deblock[fi.ref_frames[fi.primary_ref_frame as usize]].ref_deltas
                };
                for i in 0..REF_FRAMES {
                    let update = fi.deblock.ref_deltas[i] != prev_ref_deltas[i];
                    self.write_bit(update)?;
                    if update {
                        self.write_signed(7,fi.deblock.ref_deltas[i])?;
                    }
                }
                // conditionally write mode delta updates
                let prev_mode_deltas = if fi.primary_ref_frame == PRIMARY_REF_NONE {
                    [0, 0]
                } else {
                    fi.rec_buffer.deblock[fi.ref_frames[fi.primary_ref_frame as usize]].mode_deltas
                };
                for i in 0..2 {
                    let update = fi.deblock.mode_deltas[i] != prev_mode_deltas[i];
                    self.write_bit(update)?;
                    if update {
                        self.write_signed(7,fi.deblock.mode_deltas[i])?;
                    }
                }
            }
        }
        Ok(())
    }

    fn write_frame_cdef(&mut self, seq: &Sequence, fi: &FrameInvariants) -> io::Result<()> {
        if seq.enable_cdef {
            assert!(fi.cdef_damping >= 3);
            assert!(fi.cdef_damping <= 6);
            self.write(2, fi.cdef_damping - 3)?;
            assert!(fi.cdef_bits < 4);
            self.write(2,fi.cdef_bits)?; // cdef bits
            for i in 0..(1<<fi.cdef_bits) {
                assert!(fi.cdef_y_strengths[i]<64);
                assert!(fi.cdef_uv_strengths[i]<64);
                self.write(6,fi.cdef_y_strengths[i])?; // cdef y strength
                self.write(6,fi.cdef_uv_strengths[i])?; // cdef uv strength
            }
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
  return size;
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

fn write_obus(packet: &mut dyn io::Write, sequence: &mut Sequence,
                            fi: &mut FrameInvariants) -> io::Result<()> {
    //let mut uch = BitWriter::<BE>::new(packet);
    let obu_extension = 0 as u32;

    let mut buf1 = Vec::new();
    {
        let mut bw1 = BitWriter::<BE>::new(&mut buf1);
      bw1.write_obu_header(OBU_Type::OBU_TEMPORAL_DELIMITER, obu_extension)?;
      bw1.write(8,0)?;	// size of payload == 0, one byte
    }
    packet.write(&buf1).unwrap();
    buf1.clear();

    // write sequence header obu if KEY_FRAME, preceded by 4-byte size
    if fi.frame_type == FrameType::KEY {
        let mut buf2 = Vec::new();
        {
            let mut bw2 = BitWriter::<BE>::new(&mut buf2);
            bw2.write_sequence_header_obu(sequence, fi)?;
            bw2.byte_align()?;
        }

        {
            let mut bw1 = BitWriter::<BE>::new(&mut buf1);
            bw1.write_obu_header(OBU_Type::OBU_SEQUENCE_HEADER, obu_extension)?;
        }
        packet.write(&buf1).unwrap();
        buf1.clear();

        let obu_payload_size = buf2.len() as u64;
        {
            let mut bw1 = BitWriter::<BE>::new(&mut buf1);
            // uleb128()
            let mut coded_payload_length = [0 as u8; 8];
            let leb_size = aom_uleb_encode(obu_payload_size, &mut coded_payload_length);
            for i in 0..leb_size {
                bw1.write(8, coded_payload_length[i])?;
            }
        }
        packet.write(&buf1).unwrap();
        buf1.clear();

        packet.write(&buf2).unwrap();
        buf2.clear();
    }

    let mut buf2 = Vec::new();
    {
        let mut bw2 = BitWriter::<BE>::new(&mut buf2);
        bw2.write_frame_header_obu(sequence, fi)?;
    }

    {
        let mut bw1 = BitWriter::<BE>::new(&mut buf1);
        bw1.write_obu_header(OBU_Type::OBU_FRAME_HEADER, obu_extension)?;
    }
    packet.write(&buf1).unwrap();
    buf1.clear();

    let obu_payload_size = buf2.len() as u64;
    {
        let mut bw1 = BitWriter::<BE>::new(&mut buf1);
        // uleb128()
        let mut coded_payload_length = [0 as u8; 8];
        let leb_size = aom_uleb_encode(obu_payload_size, &mut coded_payload_length);
        for i in 0..leb_size {
            bw1.write(8, coded_payload_length[i])?;
        }
    }
    packet.write(&buf1).unwrap();
    buf1.clear();

    packet.write(&buf2).unwrap();
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

// For a transform block,
// predict, transform, quantize, write coefficients to a bitstream,
// dequantize, inverse-transform.
pub fn encode_tx_block(
  fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter,
  w: &mut dyn Writer, p: usize, bo: &BlockOffset, mode: PredictionMode,
  tx_size: TxSize, tx_type: TxType, plane_bsize: BlockSize, po: &PlaneOffset,
  skip: bool, bit_depth: usize, ac: &[i16], alpha: i16
) -> bool {
    let rec = &mut fs.rec.planes[p];
    let PlaneConfig { stride, xdec, ydec, .. } = fs.input.planes[p].cfg;

    if mode.is_intra() {
      mode.predict_intra(&mut rec.mut_slice(po), tx_size, bit_depth, &ac, alpha);
    }

    if skip { return false; }

    let mut residual: AlignedArray<[i16; 64 * 64]> = UninitializedAlignedArray();
    let mut coeffs_storage: AlignedArray<[i32; 64 * 64]> = UninitializedAlignedArray();
    let mut rcoeffs: AlignedArray<[i32; 64 * 64]> = UninitializedAlignedArray();
    let coeffs = &mut coeffs_storage.array[..tx_size.area()];

    diff(&mut residual.array,
         &fs.input.planes[p].slice(po),
         &rec.slice(po),
         tx_size.width(),
         tx_size.height());

    forward_transform(&residual.array, coeffs, tx_size.width(), tx_size, tx_type, bit_depth);
    fs.qc.quantize(coeffs);

    let has_coeff = cw.write_coeffs_lv_map(w, p, bo, &coeffs, tx_size, tx_type, plane_bsize, xdec, ydec,
                            fi.use_reduced_tx_set);

    // Reconstruct
    dequantize(fi.config.quantizer, &coeffs, &mut rcoeffs.array, tx_size, bit_depth);

    inverse_transform_add(&rcoeffs.array, &mut rec.mut_slice(po).as_mut_slice(), stride, tx_size, tx_type, bit_depth);
    has_coeff
}

pub fn motion_compensate(fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter,
                         luma_mode: PredictionMode, ref_frame: usize, mv: MotionVector,
                         bsize: BlockSize, bo: &BlockOffset, bit_depth: usize) {
  if luma_mode.is_intra() { return; }

  let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;

  // Inter mode prediction can take place once for a whole partition,
  // instead of each tx-block.
  let num_planes = 1 + if has_chroma(bo, bsize, xdec, ydec) { 2 } else { 0 };
  for p in 0..num_planes {
    let plane_bsize = if p == 0 { bsize }
    else { get_plane_block_size(bsize, xdec, ydec) };

    let po = bo.plane_offset(&fs.input.planes[p].cfg);

    let rec = &mut fs.rec.planes[p];

    // TODO: make more generic to handle 2xN and Nx2 MC
    if p > 0 && bsize == BlockSize::BLOCK_4X4 {
      let mv0 = &cw.bc.at(&bo.with_offset(-1,-1)).mv[0];
      let mv1 = &cw.bc.at(&bo.with_offset(0,-1)).mv[0];
      let po1 = PlaneOffset { x: po.x+2, y: po.y };
      let mv2 = &cw.bc.at(&bo.with_offset(-1,0)).mv[0];
      let po2 = PlaneOffset { x: po.x, y: po.y+2 };
      let po3 = PlaneOffset { x: po.x+2, y: po.y+2 };
      let some_use_intra = cw.bc.at(&bo.with_offset(-1,-1)).mode.is_intra()
        || cw.bc.at(&bo.with_offset(0,-1)).mode.is_intra()
        || cw.bc.at(&bo.with_offset(-1,0)).mode.is_intra();

      if some_use_intra {
        luma_mode.predict_inter(fi, p, &po, &mut rec.mut_slice(&po), plane_bsize.width(),
        plane_bsize.height(), ref_frame, &mv, bit_depth);
      } else {
        luma_mode.predict_inter(fi, p, &po, &mut rec.mut_slice(&po), 2, 2, ref_frame, mv0, bit_depth);
        luma_mode.predict_inter(fi, p, &po1, &mut rec.mut_slice(&po1), 2, 2, ref_frame, mv1, bit_depth);
        luma_mode.predict_inter(fi, p, &po2, &mut rec.mut_slice(&po2), 2, 2, ref_frame, mv2, bit_depth);
        luma_mode.predict_inter(fi, p, &po3, &mut rec.mut_slice(&po3), 2, 2, ref_frame, &mv, bit_depth);
      }
    } else {
      luma_mode.predict_inter(fi, p, &po, &mut rec.mut_slice(&po), plane_bsize.width(),
      plane_bsize.height(), ref_frame, &mv, bit_depth);
    }
  }
}

pub fn encode_block_a(seq: &Sequence,
                 cw: &mut ContextWriter, w: &mut dyn Writer,
                 bsize: BlockSize, bo: &BlockOffset, skip: bool) -> bool {
    cw.bc.set_skip(bo, bsize, skip);
    cw.write_skip(w, bo, skip);
    if !skip && seq.enable_cdef {
        cw.bc.cdef_coded = true;
    }
    cw.bc.cdef_coded
}

pub fn encode_block_b(seq: &Sequence, fi: &FrameInvariants, fs: &mut FrameState,
                 cw: &mut ContextWriter, w: &mut dyn Writer,
                 luma_mode: PredictionMode, chroma_mode: PredictionMode,
                 ref_frame: usize, mv: MotionVector,
                 bsize: BlockSize, bo: &BlockOffset, skip: bool, bit_depth: usize,
                 cfl: CFLParams, tx_size: TxSize, tx_type: TxType,
                 mode_context: usize, mv_stack: &Vec<CandidateMV>) {
    let is_inter = !luma_mode.is_intra();
    if is_inter { assert!(luma_mode == chroma_mode); };
    let sb_size = if seq.use_128x128_superblock {
        BlockSize::BLOCK_128X128
    } else {
        BlockSize::BLOCK_64X64
    };
    cw.bc.set_block_size(bo, bsize);
    cw.bc.set_mode(bo, bsize, luma_mode);
    //write_q_deltas();
    if cw.bc.code_deltas && fi.deblock.block_deltas_enabled && (bsize < sb_size || !skip) {
        cw.write_block_deblock_deltas(w, bo, fi.deblock.block_delta_multi);
    }
    cw.bc.code_deltas = false;

    if fi.frame_type == FrameType::INTER {
        cw.write_is_inter(w, bo, is_inter);
        if is_inter {
            cw.fill_neighbours_ref_counts(bo);
            cw.bc.set_ref_frame(bo, bsize, ref_frame);
            cw.bc.set_motion_vector(bo, bsize, mv);
            cw.write_ref_frames(w, bo);

            //let mode_context = if bo.x == 0 && bo.y == 0 { 0 } else if bo.x ==0 || bo.y == 0 { 51 } else { 85 };
            // NOTE: Until rav1e supports other inter modes than GLOBALMV
            cw.write_inter_mode(w, luma_mode, mode_context);

            if luma_mode == PredictionMode::NEWMV || luma_mode == PredictionMode::NEW_NEWMV {
              let ref_mv_idx = 0;
              let num_mv_found = mv_stack.len();
              for idx in 0..2 {
                if num_mv_found > idx + 1 {
                  let drl_mode = ref_mv_idx > idx;
                  let ctx: usize = (mv_stack[idx].weight < REF_CAT_LEVEL) as usize
                    + (mv_stack[idx + 1].weight < REF_CAT_LEVEL) as usize;

                  cw.write_drl_mode(w, drl_mode, ctx);
                  if !drl_mode { break; }
                }
              }

              let ref_mv = if num_mv_found > 0 {
                mv_stack[ref_mv_idx].this_mv
              } else {
                MotionVector{ row: 0, col: 0 }
              };

              let mv_precision = if fi.force_integer_mv != 0 {
                MvSubpelPrecision::MV_SUBPEL_NONE
              } else if fi.allow_high_precision_mv {
                MvSubpelPrecision::MV_SUBPEL_HIGH_PRECISION
              } else {
                MvSubpelPrecision::MV_SUBPEL_LOW_PRECISION
              };
              cw.write_mv(w, &mv, &ref_mv, mv_precision);
            } else if luma_mode == PredictionMode::NEARESTMV {
              if mv_stack.len() > 0 {
                assert!(mv_stack[0].this_mv.row == mv.row);
                assert!(mv_stack[0].this_mv.col == mv.col);
              } else {
                assert!(0 == mv.row);
                assert!(0 == mv.col);
              }
            }
        } else {
            cw.write_intra_mode(w, bsize, luma_mode);
        }
    } else {
        cw.write_intra_mode_kf(w, bo, luma_mode);
    }

    let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;

    if luma_mode.is_directional() && bsize >= BlockSize::BLOCK_8X8 {
        cw.write_angle_delta(w, 0, luma_mode);
    }

    if has_chroma(bo, bsize, xdec, ydec) && !is_inter {
        cw.write_intra_uv_mode(w, chroma_mode, luma_mode, bsize);
        if chroma_mode.is_cfl() {
          assert!(bsize.cfl_allowed());
          cw.write_cfl_alphas(w, cfl);
        }
        if chroma_mode.is_directional() && bsize >= BlockSize::BLOCK_8X8 {
            cw.write_angle_delta(w, 0, chroma_mode);
        }
    }

    if skip {
        cw.bc.reset_skip_context(bo, bsize, xdec, ydec);
    }

    // TODO: Extra condition related to palette mode, see `read_filter_intra_mode_info` in decodemv.c
    if luma_mode == PredictionMode::DC_PRED && bsize.width() <= 32 && bsize.height() <= 32 {
        cw.write_use_filter_intra(w,false, bsize); // Always turn off FILTER_INTRA
    }

    motion_compensate(fi, fs, cw, luma_mode, ref_frame, mv, bsize, bo, bit_depth);

    if is_inter {
      write_tx_tree(fi, fs, cw, w, luma_mode, bo, bsize, tx_size, tx_type, skip, bit_depth, false); // i.e. var-tx if inter mode
    } else {
      write_tx_blocks(fi, fs, cw, w, luma_mode, chroma_mode, bo, bsize, tx_size, tx_type, skip, bit_depth, cfl, false);
    }
}

pub fn luma_ac(
  ac: &mut [i16], fs: &mut FrameState, bo: &BlockOffset, bsize: BlockSize
) {
  let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;
  let plane_bsize = get_plane_block_size(bsize, xdec, ydec);
  let po = if bsize.is_sub8x8() {
    bo.with_offset(-1, -1).plane_offset(&fs.input.planes[0].cfg)
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
      ac[sub_y * 32 + sub_x] = sample;
      sum += sample as i32;
    }
  }
  let shift = plane_bsize.width_log2() + plane_bsize.height_log2();
  let average = ((sum + (1 << (shift - 1))) >> shift) as i16;
  for sub_y in 0..plane_bsize.height() {
    for sub_x in 0..plane_bsize.width() {
      ac[sub_y * 32 + sub_x] -= average;
    }
  }
}

pub fn write_tx_blocks(fi: &FrameInvariants, fs: &mut FrameState,
                       cw: &mut ContextWriter, w: &mut dyn Writer,
                       luma_mode: PredictionMode, chroma_mode: PredictionMode, bo: &BlockOffset,
                       bsize: BlockSize, tx_size: TxSize, tx_type: TxType, skip: bool, bit_depth: usize,
                       cfl: CFLParams, luma_only: bool) {
    let bw = bsize.width_mi() / tx_size.width_mi();
    let bh = bsize.height_mi() / tx_size.height_mi();

    let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;
    let ac = &mut [0i16; 32 * 32];

    fs.qc.update(fi.config.quantizer, tx_size, luma_mode.is_intra(), bit_depth);

    for by in 0..bh {
        for bx in 0..bw {
            let tx_bo = BlockOffset {
                x: bo.x + bx * tx_size.width_mi(),
                y: bo.y + by * tx_size.height_mi()
            };

            let po = tx_bo.plane_offset(&fs.input.planes[0].cfg);
            encode_tx_block(
              fi, fs, cw, w, 0, &tx_bo, luma_mode, tx_size, tx_type, bsize, &po,
              skip, bit_depth, ac, 0,
            );
        }
    }

    if luma_only { return };

    // TODO: these are only valid for 4:2:0
    let uv_tx_size = match bsize {
        BlockSize::BLOCK_4X4 | BlockSize::BLOCK_8X8 => TxSize::TX_4X4,
        BlockSize::BLOCK_16X16 => TxSize::TX_8X8,
        BlockSize::BLOCK_32X32 => TxSize::TX_16X16,
        _ => TxSize::TX_32X32
    };

    let mut bw_uv = (bw * tx_size.width_mi()) >> xdec;
    let mut bh_uv = (bh * tx_size.height_mi()) >> ydec;

    if (bw_uv == 0 || bh_uv == 0) && has_chroma(bo, bsize, xdec, ydec) {
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
        let uv_tx_type = uv_intra_mode_to_tx_type_context(chroma_mode);
        fs.qc.update(fi.config.quantizer, uv_tx_size, true, bit_depth);

        for p in 1..3 {
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
                    po.x += bx * uv_tx_size.width();
                    po.y += by * uv_tx_size.height();

                    encode_tx_block(fi, fs, cw, w, p, &tx_bo, chroma_mode, uv_tx_size, uv_tx_type,
                                    plane_bsize, &po, skip, bit_depth, ac, alpha);
                }
            }
        }
    }
}

// FIXME: For now, assume tx_mode is LARGEST_TX, so var-tx is not implemented yet
// but only one tx block exist for a inter mode partition.
pub fn write_tx_tree(fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter, w: &mut dyn Writer,
                       luma_mode: PredictionMode, bo: &BlockOffset,
                       bsize: BlockSize, tx_size: TxSize, tx_type: TxType, skip: bool, bit_depth: usize,
                       luma_only: bool) {
    let bw = bsize.width_mi() / tx_size.width_mi();
    let bh = bsize.height_mi() / tx_size.height_mi();

    let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;
    let ac = &[0i16; 32 * 32];

    fs.qc.update(fi.config.quantizer, tx_size, luma_mode.is_intra(), bit_depth);

    let po = bo.plane_offset(&fs.input.planes[0].cfg);
    let has_coeff = encode_tx_block(
      fi, fs, cw, w, 0, &bo, luma_mode, tx_size, tx_type, bsize, &po, skip,
      bit_depth, ac, 0,
    );

    if luma_only { return };

    // TODO: these are only valid for 4:2:0
    let uv_tx_size = match bsize {
        BlockSize::BLOCK_4X4 | BlockSize::BLOCK_8X8 => TxSize::TX_4X4,
        BlockSize::BLOCK_16X16 => TxSize::TX_8X8,
        BlockSize::BLOCK_32X32 => TxSize::TX_16X16,
        _ => TxSize::TX_32X32
    };

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

        fs.qc.update(fi.config.quantizer, uv_tx_size, false, bit_depth);

        for p in 1..3 {
            let tx_bo = BlockOffset {
                x: bo.x  - ((bw * tx_size.width_mi() == 1) as usize),
                y: bo.y  - ((bh * tx_size.height_mi() == 1) as usize)
            };

            let po = bo.plane_offset(&fs.input.planes[p].cfg);

            encode_tx_block(fi, fs, cw, w, p, &tx_bo, luma_mode, uv_tx_size, uv_tx_type,
                            plane_bsize, &po, skip, bit_depth, ac, 0);
        }
    }
}

fn encode_partition_bottomup(seq: &Sequence, fi: &FrameInvariants, fs: &mut FrameState,
                             cw: &mut ContextWriter, w_pre_cdef: &mut dyn Writer, w_post_cdef: &mut dyn Writer,
                             bsize: BlockSize, bo: &BlockOffset) -> f64 {
    let mut rd_cost = std::f64::MAX;

    if bo.x >= cw.bc.cols || bo.y >= cw.bc.rows {
        return rd_cost;
    }

    let bs = bsize.width_mi();

    // Always split if the current partition is too large
    let must_split = bo.x + bs as usize > fi.w_in_b ||
        bo.y + bs as usize > fi.h_in_b ||
        bsize >= BlockSize::BLOCK_64X64;

    // must_split overrides the minimum partition size when applicable
    let can_split = bsize > fi.min_partition_size || must_split;

    let mut partition = PartitionType::PARTITION_NONE;
    let mut best_decision = RDOPartitionOutput {
        rd_cost,
        bo: bo.clone(),
        pred_mode_luma: PredictionMode::DC_PRED,
        pred_mode_chroma: PredictionMode::DC_PRED,
        pred_cfl_params: CFLParams::new(),
        ref_frame: INTRA_FRAME,
        mv: MotionVector { row: 0, col: 0},
        skip: false
    }; // Best decision that is not PARTITION_SPLIT

    let hbs = bs >> 1; // Half the block size in blocks
    let mut subsize: BlockSize;

    let cw_checkpoint = cw.checkpoint();
    let w_pre_checkpoint = w_pre_cdef.checkpoint();
    let w_post_checkpoint = w_post_cdef.checkpoint();

    // Code the whole block
    if !must_split {
        partition = PartitionType::PARTITION_NONE;

        let mut cost: f64 = 0.0;

        if bsize >= BlockSize::BLOCK_8X8 {
            let w: &mut dyn Writer = if cw.bc.cdef_coded {w_post_cdef} else {w_pre_cdef};
            let tell = w.tell_frac();
            cw.write_partition(w, bo, partition, bsize);
            cost = (w.tell_frac() - tell) as f64 * get_lambda(fi, seq.bit_depth)/ ((1 << OD_BITRES) as f64);
        }
        let mode_decision = rdo_mode_decision(seq, fi, fs, cw, bsize, bo).part_modes[0].clone();
        let (mode_luma, mode_chroma) = (mode_decision.pred_mode_luma, mode_decision.pred_mode_chroma);
        let cfl = mode_decision.pred_cfl_params;
        let ref_frame = mode_decision.ref_frame;
        let mv = mode_decision.mv;
        let skip = mode_decision.skip;
        let mut cdef_coded = cw.bc.cdef_coded;
        rd_cost = mode_decision.rd_cost + cost;

        let mut mv_stack = Vec::new();
        let mode_context = cw.find_mvrefs(bo, ref_frame, &mut mv_stack, bsize, false);

        let (tx_size, tx_type) =
          rdo_tx_size_type(seq, fi, fs, cw, bsize, bo, mode_luma, ref_frame, mv, skip);

        cdef_coded = encode_block_a(seq, cw, if cdef_coded  {w_post_cdef} else {w_pre_cdef},
                                   bsize, bo, skip);
        encode_block_b(seq, fi, fs, cw, if cdef_coded  {w_post_cdef} else {w_pre_cdef},
                       mode_luma, mode_chroma, ref_frame, mv, bsize, bo, skip, seq.bit_depth, cfl,
                       tx_size, tx_type, mode_context, &mv_stack);

        best_decision = mode_decision;
    }

    // Code a split partition and compare RD costs
    if can_split {
        cw.rollback(&cw_checkpoint);
        w_pre_cdef.rollback(&w_pre_checkpoint);
        w_post_cdef.rollback(&w_post_checkpoint);

        partition = PartitionType::PARTITION_SPLIT;
        subsize = get_subsize(bsize, partition);

        let nosplit_rd_cost = rd_cost;

        rd_cost = 0.0;

        if bsize >= BlockSize::BLOCK_8X8 {
            let w: &mut dyn Writer = if cw.bc.cdef_coded {w_post_cdef} else {w_pre_cdef};
            let tell = w.tell_frac();
            cw.write_partition(w, bo, partition, bsize);
            rd_cost = (w.tell_frac() - tell) as f64 * get_lambda(fi, seq.bit_depth)/ ((1 << OD_BITRES) as f64);
        }

        rd_cost += encode_partition_bottomup(seq, fi, fs, cw, w_pre_cdef, w_post_cdef, subsize,
                                             bo);
        rd_cost += encode_partition_bottomup(seq, fi, fs, cw, w_pre_cdef, w_post_cdef, subsize,
                                             &BlockOffset { x: bo.x + hbs as usize, y: bo.y });
        rd_cost += encode_partition_bottomup(seq, fi, fs, cw, w_pre_cdef, w_post_cdef, subsize,
                                             &BlockOffset { x: bo.x, y: bo.y + hbs as usize });
        rd_cost += encode_partition_bottomup(seq, fi, fs, cw, w_pre_cdef, w_post_cdef, subsize,
                                             &BlockOffset { x: bo.x + hbs as usize, y: bo.y + hbs as usize });

        // Recode the full block if it is more efficient
        if !must_split && nosplit_rd_cost < rd_cost {
            rd_cost = nosplit_rd_cost;

            cw.rollback(&cw_checkpoint);
            w_pre_cdef.rollback(&w_pre_checkpoint);
            w_post_cdef.rollback(&w_post_checkpoint);

            partition = PartitionType::PARTITION_NONE;

            if bsize >= BlockSize::BLOCK_8X8 {
                let w: &mut dyn Writer = if cw.bc.cdef_coded {w_post_cdef} else {w_pre_cdef};
                cw.write_partition(w, bo, partition, bsize);
            }

            // FIXME: redundant block re-encode
            let (mode_luma, mode_chroma) = (best_decision.pred_mode_luma, best_decision.pred_mode_chroma);
            let cfl = best_decision.pred_cfl_params;
            let ref_frame = best_decision.ref_frame;
            let mv = best_decision.mv;
            let skip = best_decision.skip;
            let mut cdef_coded = cw.bc.cdef_coded;

            let mut mv_stack = Vec::new();
            let mode_context = cw.find_mvrefs(bo, ref_frame, &mut mv_stack, bsize, false);

            let (tx_size, tx_type) =
                rdo_tx_size_type(seq, fi, fs, cw, bsize, bo, mode_luma, ref_frame, mv, skip);

            cdef_coded = encode_block_a(seq, cw, if cdef_coded {w_post_cdef} else {w_pre_cdef},
                                       bsize, bo, skip);
            encode_block_b(seq, fi, fs, cw, if cdef_coded {w_post_cdef} else {w_pre_cdef},
                          mode_luma, mode_chroma, ref_frame, mv, bsize, bo, skip, seq.bit_depth, cfl,
                          tx_size, tx_type, mode_context, &mv_stack);
        }
    }

    subsize = get_subsize(bsize, partition);

    if bsize >= BlockSize::BLOCK_8X8 &&
        (bsize == BlockSize::BLOCK_8X8 || partition != PartitionType::PARTITION_SPLIT) {
        cw.bc.update_partition_context(bo, subsize, bsize);
    }

    rd_cost
}

fn encode_partition_topdown(seq: &Sequence, fi: &FrameInvariants, fs: &mut FrameState,
            cw: &mut ContextWriter, w_pre_cdef: &mut dyn Writer, w_post_cdef: &mut dyn Writer,
            bsize: BlockSize, bo: &BlockOffset, block_output: &Option<RDOOutput>) {

    if bo.x >= cw.bc.cols || bo.y >= cw.bc.rows {
        return;
    }

    let bs = bsize.width_mi();

    // Always split if the current partition is too large
    let must_split = bo.x + bs as usize > fi.w_in_b ||
        bo.y + bs as usize > fi.h_in_b ||
        bsize >= BlockSize::BLOCK_64X64;

    let mut rdo_output = block_output.clone().unwrap_or(RDOOutput {
        part_type: PartitionType::PARTITION_INVALID,
        rd_cost: std::f64::MAX,
        part_modes: Vec::new()
    });
    let partition: PartitionType;

    if must_split {
        // Oversized blocks are split automatically
        partition = PartitionType::PARTITION_SPLIT;
    } else if bsize > fi.min_partition_size {
        // Blocks of sizes within the supported range are subjected to a partitioning decision
        rdo_output = rdo_partition_decision(seq, fi, fs, cw, bsize, bo, &rdo_output);
        partition = rdo_output.part_type;
    } else {
        // Blocks of sizes below the supported range are encoded directly
        partition = PartitionType::PARTITION_NONE;
    }

    assert!(bsize.width_mi() == bsize.height_mi());
    assert!(PartitionType::PARTITION_NONE <= partition &&
            partition < PartitionType::PARTITION_INVALID);

    let hbs = bs >> 1; // Half the block size in blocks
    let subsize = get_subsize(bsize, partition);

    if bsize >= BlockSize::BLOCK_8X8 {
        let w: &mut dyn Writer = if cw.bc.cdef_coded {w_post_cdef} else {w_pre_cdef};
        cw.write_partition(w, bo, partition, bsize);
    }

    match partition {
        PartitionType::PARTITION_NONE => {
            let part_decision = if !rdo_output.part_modes.is_empty() {
                    // The optimal prediction mode is known from a previous iteration
                    rdo_output.part_modes[0].clone()
                } else {
                    // Make a prediction mode decision for blocks encoded with no rdo_partition_decision call (e.g. edges)
                    rdo_mode_decision(seq, fi, fs, cw, bsize, bo).part_modes[0].clone()
                };

            let mut mode_luma = part_decision.pred_mode_luma;
            let mut mode_chroma = part_decision.pred_mode_chroma;

            let cfl = part_decision.pred_cfl_params;
            let skip = part_decision.skip;
            let ref_frame = part_decision.ref_frame;
            let mv = part_decision.mv;
            let mut cdef_coded = cw.bc.cdef_coded;

            let (tx_size, tx_type) =
                rdo_tx_size_type(seq, fi, fs, cw, bsize, bo, mode_luma, ref_frame, mv, skip);

            let mut mv_stack = Vec::new();
            let mode_context = cw.find_mvrefs(bo, ref_frame, &mut mv_stack, bsize, false);

            if mode_luma == PredictionMode::NEARESTMV &&
                (mv_stack.len() > 0 && (mv_stack[0].this_mv.row != mv.row || mv_stack[0].this_mv.col != mv.col) ||
                 mv_stack.len() == 0 && (0 != mv.row || 0 != mv.col)) {
              mode_luma = PredictionMode::NEWMV;
              mode_chroma = PredictionMode::NEWMV;
            }

            // FIXME: every final block that has gone through the RDO decision process is encoded twice
            cdef_coded = encode_block_a(seq, cw, if cdef_coded  {w_post_cdef} else {w_pre_cdef},
                         bsize, bo, skip);
            encode_block_b(seq, fi, fs, cw, if cdef_coded  {w_post_cdef} else {w_pre_cdef},
                          mode_luma, mode_chroma, ref_frame, mv, bsize, bo, skip, seq.bit_depth, cfl,
                          tx_size, tx_type, mode_context, &mv_stack);
        },
        PartitionType::PARTITION_SPLIT => {
            if rdo_output.part_modes.len() >= 4 {
                // The optimal prediction modes for each split block is known from an rdo_partition_decision() call
                assert!(subsize != BlockSize::BLOCK_INVALID);

                for mode in rdo_output.part_modes {
                    let offset = mode.bo.clone();

                    // Each block is subjected to a new splitting decision
                    encode_partition_topdown(seq, fi, fs, cw, w_pre_cdef, w_post_cdef, subsize, &offset,
                        &Some(RDOOutput {
                            rd_cost: mode.rd_cost,
                            part_type: PartitionType::PARTITION_NONE,
                            part_modes: vec![mode] }));
                }
            }
            else {
                encode_partition_topdown(seq, fi, fs, cw, w_pre_cdef, w_post_cdef, subsize,
                                         bo, &None);
                encode_partition_topdown(seq, fi, fs, cw, w_pre_cdef, w_post_cdef, subsize,
                                         &BlockOffset{x: bo.x + hbs as usize, y: bo.y}, &None);
                encode_partition_topdown(seq, fi, fs, cw, w_pre_cdef, w_post_cdef, subsize,
                                         &BlockOffset{x: bo.x, y: bo.y + hbs as usize}, &None);
                encode_partition_topdown(seq, fi, fs, cw, w_pre_cdef, w_post_cdef, subsize,
                                         &BlockOffset{x: bo.x + hbs as usize, y: bo.y + hbs as usize}, &None);
            }
        },
        _ => { assert!(false); },
    }

    if bsize >= BlockSize::BLOCK_8X8 &&
        (bsize == BlockSize::BLOCK_8X8 || partition != PartitionType::PARTITION_SPLIT) {
            cw.bc.update_partition_context(bo, subsize, bsize);
    }
}

fn encode_tile(sequence: &mut Sequence, fi: &FrameInvariants, fs: &mut FrameState, bit_depth: usize) -> Vec<u8> {
    let mut w = WriterEncoder::new();

    let fc = if fi.primary_ref_frame == PRIMARY_REF_NONE {
      CDFContext::new(fi.config.quantizer as u8)
    } else {
      match fi.rec_buffer.frames[fi.ref_frames[fi.primary_ref_frame as usize]] {
        Some(ref rec) => rec.cdfs.clone(),
        None => CDFContext::new(fi.config.quantizer as u8)
      }
    };

    let bc = BlockContext::new(fi.w_in_b, fi.h_in_b);
    let mut cw = ContextWriter::new(fc,  bc);

    for sby in 0..fi.sb_height {
        cw.bc.reset_left_contexts();

        for sbx in 0..fi.sb_width {
            let mut w_post_cdef = WriterRecorder::new();
            let sbo = SuperBlockOffset { x: sbx, y: sby };
            let bo = sbo.block_offset(0, 0);
            cw.bc.cdef_coded = false;
            cw.bc.code_deltas = fi.delta_q_present;

            // Encode SuperBlock
            if fi.config.speed == 0 {
                encode_partition_bottomup(sequence, fi, fs, &mut cw,
                                          &mut w, &mut w_post_cdef,
                                          BlockSize::BLOCK_64X64, &bo);
            }
            else {
                encode_partition_topdown(sequence, fi, fs, &mut cw,
                                         &mut w, &mut w_post_cdef,
                                         BlockSize::BLOCK_64X64, &bo, &None);
            }

            if cw.bc.cdef_coded {
                let cdef_index = rdo_cdef_decision(&sbo, fi, fs, &mut cw, bit_depth);
                // CDEF index must be written in the middle, we can code it now
                cw.write_cdef(&mut w, cdef_index, fi.cdef_bits);
                cw.bc.set_cdef(&sbo, cdef_index);
                // ...and then finally code what comes after the CDEF index
                w_post_cdef.replay(&mut w);
            }
        }
    }
    /* TODO: Don't apply if lossless */
    if sequence.enable_cdef {
        cdef_filter_frame(fi, &mut fs.rec, &mut cw.bc, bit_depth);
    }

    fs.cdfs = cw.fc.clone();
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
        let mut bw = BitWriter::<BE>::new(&mut buf);
        bw.write_bit(tile_start_and_end_present_flag).unwrap();
        bw.byte_align().unwrap();
    }
    buf.clone()
}

pub fn encode_frame(sequence: &mut Sequence, fi: &mut FrameInvariants, fs: &mut FrameState) -> Vec<u8> {
    let mut packet = Vec::new();
    //write_uncompressed_header(&mut packet, sequence, fi).unwrap();
    write_obus(&mut packet, sequence, fi).unwrap();
    if fi.show_existing_frame {
        match fi.rec_buffer.frames[0] {
            Some(ref rec) => for p in 0..3 {
                fs.rec.planes[p].data.copy_from_slice(rec.frame.planes[p].data.as_slice());
            },
            None => (),
        }
    } else {
        let bit_depth = sequence.bit_depth;
        let tile = encode_tile(sequence, fi, fs, bit_depth); // actually tile group

        let mut buf1 = Vec::new();
        {
            let mut bw1 = BitWriter::<BE>::new(&mut buf1);
            bw1.write_obu_header(OBU_Type::OBU_TILE_GROUP, 0).unwrap();
        }
        packet.write(&buf1).unwrap();
        buf1.clear();

        let obu_payload_size = tile.len() as u64;
        {
            let mut bw1 = BitWriter::<BE>::new(&mut buf1);
            // uleb128()
            let mut coded_payload_length = [0 as u8; 8];
            let leb_size = aom_uleb_encode(obu_payload_size, &mut coded_payload_length);
            for i in 0..leb_size {
                bw1.write(8, coded_payload_length[i]).unwrap();
            }
        }
        packet.write(&buf1).unwrap();
        buf1.clear();

      packet.write(&tile).unwrap();
    }
    packet
}

pub fn update_rec_buffer(fi: &mut FrameInvariants, fs: FrameState) {
  let rfs = Rc::new(ReferenceFrame { frame: fs.rec, cdfs: fs.cdfs } );
  for i in 0..(REF_FRAMES as usize) {
    if (fi.refresh_frame_flags & (1 << i)) != 0 {
      fi.rec_buffer.frames[i] = Some(Rc::clone(&rfs));
      fi.rec_buffer.deblock[i] = fi.deblock.clone();
    }
  }
}

