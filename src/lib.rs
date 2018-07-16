// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(safe_extern_statics)]
#![cfg_attr(feature = "cargo-clippy", allow(collapsible_if))]

extern crate bitstream_io;
extern crate backtrace;
extern crate clap;
extern crate libc;
extern crate rand;
extern crate y4m;

#[macro_use]
extern crate enum_iterator_derive;

use std::fs::File;
use std::io::prelude::*;
use bitstream_io::{BE, LE, BitWriter};
use clap::{App, Arg};

// for benchmarking purpose
pub mod ec;
pub mod partition;
pub mod plane;
pub mod context;
pub mod transform;
pub mod quantize;
pub mod predict;
pub mod rdo;

use context::*;
use partition::*;
use transform::*;
use quantize::*;
use plane::*;
use rdo::*;
use ec::*;
use std::fmt;

extern {
    pub fn av1_rtcd();
    pub fn aom_dsp_rtcd();
}

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

const MAX_NUM_TEMPORAL_LAYERS: usize = 8;
const MAX_NUM_SPATIAL_LAYERS: usize = 4;
const MAX_NUM_OPERATING_POINTS: usize = MAX_NUM_TEMPORAL_LAYERS * MAX_NUM_SPATIAL_LAYERS;

const PRIMARY_REF_NONE: u32 = 7;
const PRIMARY_REF_BITS: u32 = 3;

#[derive(Copy,Clone)]
pub struct Sequence {
  // OBU Sequence header of AV1
    pub profile: u8,
    pub num_bits_width: u32,
    pub num_bits_height: u32,
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
    pub monochrome: bool,                  // Monochorme video
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
    pub fn new(width: usize, height: usize) -> Sequence {
        let width_bits = 32 - (width as u32).leading_zeros();
        let height_bits = 32 - (height as u32).leading_zeros();
        assert!(width_bits <= 16);
        assert!(height_bits <= 16);

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
            profile: 0,
            num_bits_width: width_bits,
            num_bits_height: height_bits,
            max_frame_width: width as u32,
            max_frame_height: height as u32,
            frame_id_numbers_present_flag: false,
            frame_id_length: 0,
            delta_frame_id_length: 0,
            use_128x128_superblock: false,
            order_hint_bits_minus_1: 0,
            force_screen_content_tools: 2,  // 2: adaptive
            force_integer_mv: 2,            // 2: adaptive
            still_picture: false,
            reduced_still_picture_hdr: false,
            monochrome: false,
            enable_filter_intra: false,
            enable_intra_edge_filter: false,
            enable_interintra_compound: false,
            enable_masked_compound: false,
            enable_dual_filter: false,
            enable_order_hint: false,
            enable_jnt_comp: false,
            enable_ref_frame_mvs: false,
            enable_warped_motion: false,
            enable_superres: false,
            enable_cdef: false,
            enable_restoration: false,
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

pub struct FrameState {
    pub input: Frame,
    pub rec: Frame
}

impl FrameState {
    pub fn new(fi: &FrameInvariants) -> FrameState {
        FrameState {
            input: Frame::new(fi.padded_w, fi.padded_h),
            rec: Frame::new(fi.padded_w, fi.padded_h),
        }
    }
}

trait Fixed {
    fn floor_log2(&self, n: usize) -> usize;
    fn ceil_log2(&self, n: usize) -> usize;
    fn align_power_of_two(&self, n: usize) -> usize;
    fn align_power_of_two_and_shift(&self, n: usize) -> usize;
}

impl Fixed for usize {
    #[inline]
    fn floor_log2(&self, n: usize) -> usize {
        self & !((1 << n) - 1)
    }
    #[inline]
    fn ceil_log2(&self, n: usize) -> usize {
        (self + (1 << n) - 1).floor_log2(n)
    }
    #[inline]
    fn align_power_of_two(&self, n: usize) -> usize {
        self.ceil_log2(n)
    }
    #[inline]
    fn align_power_of_two_and_shift(&self, n: usize) -> usize {
        (self + (1 << n) - 1) >> n
    }
}

// Frame Invariants are invariant inside a frame
#[allow(dead_code)]
pub struct FrameInvariants {
    pub qindex: usize,
    pub speed: usize,
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
}

impl FrameInvariants {
    pub fn new(width: usize, height: usize, qindex: usize, speed: usize) -> FrameInvariants {
        // Speed level decides the minimum partition size, i.e. higher speed --> larger min partition size,
        // with exception that SBs on right or bottom frame borders split down to BLOCK_4X4.
        // At speed = 0, RDO search is exhaustive.
        let min_partition_size = if speed <= 1 { BlockSize::BLOCK_4X4 }
                                 else if speed <= 2 { BlockSize::BLOCK_8X8 }
                                 else if speed <= 3 { BlockSize::BLOCK_16X16 }
                                 else { BlockSize::BLOCK_32X32 };
        let use_reduced_tx_set = speed > 1;

        FrameInvariants {
            qindex,
            speed,
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
            error_resilient: true,
            intra_only: false,
            allow_high_precision_mv: true,
            frame_type: FrameType::KEY,
            show_existing_frame: false,
            use_reduced_tx_set,
            reference_mode: ReferenceMode::SINGLE,
            use_prev_frame_mvs: false,
            min_partition_size,
            globalmv_transformation_type: [GlobalMVMode::IDENTITY; ALTREF_FRAME + 1],
            num_tg: 1,
            large_scale_tile: false,
            disable_cdf_update: true,
            allow_screen_content_tools: 0,
            force_integer_mv: 0,
            primary_ref_frame: PRIMARY_REF_NONE,
            refresh_frame_flags: 0,
            allow_intrabc: false,
            use_ref_frame_mvs: false,
            is_filter_switchable: false,
            is_motion_mode_switchable: false, // 0: only the SIMPLE motion mode will be used.
            disable_frame_end_update_cdf: true,
            allow_warped_motion: true,
        }
    }
}

impl fmt::Display for FrameInvariants{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Frame {} - {}", self.number, self.frame_type)
    }
}

#[allow(dead_code,non_camel_case_types)]
#[derive(Debug,PartialEq,EnumIterator,Clone,Copy)]
pub enum FrameType {
    KEY,
    INTER,
    INTRA_ONLY,
    SWITCH,
}

//const REFERENCE_MODES: usize = 3;

#[allow(dead_code,non_camel_case_types)]
#[derive(Debug,PartialEq,EnumIterator)]
pub enum ReferenceMode {
  SINGLE = 0,
  COMPOUND = 1,
  SELECT = 2,
}

const REF_FRAMES: u32 = 8;
const REF_FRAMES_LOG2: u32 = 3;

/*const NONE_FRAME: isize = -1;
const INTRA_FRAME: usize = 0;*/
const LAST_FRAME: usize = 1;

/*const LAST2_FRAME: usize = 2;
const LAST3_FRAME: usize = 3;
const GOLDEN_FRAME: usize = 4;
const BWDREF_FRAME: usize = 5;
const ALTREF2_FRAME: usize = 6;*/
const ALTREF_FRAME: usize = 7;
/*const LAST_REF_FRAMES: usize = LAST3_FRAME - LAST_FRAME + 1;

const INTER_REFS_PER_FRAME: usize = ALTREF_FRAME - LAST_FRAME + 1;
const TOTAL_REFS_PER_FRAME: usize = ALTREF_FRAME - INTRA_FRAME + 1;

const FWD_REFS: usize = GOLDEN_FRAME - LAST_FRAME + 1;
//const FWD_RF_OFFSET(ref) (ref - LAST_FRAME)
const BWD_REFS: usize = ALTREF_FRAME - BWDREF_FRAME + 1;
//const BWD_RF_OFFSET(ref) (ref - BWDREF_FRAME)

const SINGLE_REFS: usize = FWD_REFS + BWD_REFS;
*/

impl fmt::Display for FrameType{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FrameType::KEY => write!(f, "Key frame"),
            FrameType::INTER => write!(f, "Inter frame"),
            FrameType::INTRA_ONLY => write!(f, "Intra only frame"),
            FrameType::SWITCH => write!(f, "Switching frame"),
        }
    }
}


pub struct EncoderConfig {
    pub input_file: Box<Read>,
    pub output_file: Box<Write>,
    pub rec_file: Option<Box<Write>>,
    pub limit: u64,
    pub quantizer: usize,
    pub speed: usize
}

impl EncoderConfig {
    pub fn from_cli() -> EncoderConfig {
        let matches = App::new("rav1e")
            .version("0.1.0")
            .about("AV1 video encoder")
           .arg(Arg::with_name("INPUT")
                .help("Uncompressed YUV4MPEG2 video input")
                .required(true)
                .index(1))
            .arg(Arg::with_name("OUTPUT")
                .help("Compressed AV1 in IVF video output")
                .short("o")
                .long("output")
                .required(true)
                .takes_value(true))
            .arg(Arg::with_name("RECONSTRUCTION")
                .short("r")
                .takes_value(true))
            .arg(Arg::with_name("LIMIT")
                .help("Maximum number of frames to encode")
                .short("l")
                .long("limit")
                .takes_value(true)
                .default_value("0"))
            .arg(Arg::with_name("QP")
                .help("Quantizer (0-255)")
                .long("quantizer")
                .takes_value(true)
                .default_value("100"))
            .arg(Arg::with_name("SPEED")
                .help("Speed level (0(slow)-10(fast))")
                .short("s")
                .long("speed")
                .takes_value(true)
                .default_value("3"))
            .get_matches();

        EncoderConfig {
            input_file: match matches.value_of("INPUT").unwrap() {
                "-" => Box::new(std::io::stdin()) as Box<Read>,
                f => Box::new(File::open(&f).unwrap()) as Box<Read>
            },
            output_file: match matches.value_of("OUTPUT").unwrap() {
                "-" => Box::new(std::io::stdout()) as Box<Write>,
                f => Box::new(File::create(&f).unwrap()) as Box<Write>
            },
            rec_file: matches.value_of("RECONSTRUCTION").map(|f| {
                Box::new(File::create(&f).unwrap()) as Box<Write>
            }),
            limit: matches.value_of("LIMIT").unwrap().parse().unwrap(),
            quantizer: matches.value_of("QP").unwrap().parse().unwrap(),
            speed: matches.value_of("SPEED").unwrap().parse().unwrap()
        }
    }
}

pub fn write_ivf_header(output_file: &mut Write, width: usize, height: usize, num: usize, den: usize) {
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

pub fn write_ivf_frame(output_file: &mut Write, pts: u64, data: &[u8]) {
    let mut bw = BitWriter::<LE>::new(output_file);
    bw.write(32, data.len() as u32).unwrap();
    bw.write(64, pts).unwrap();
    bw.write_bytes(data).unwrap();
}

trait UncompressedHeader {
    // Start of OBU Headers
    fn write_obu_header(&mut self, obu_type: OBU_Type, obu_extension: u32)
            -> Result<(), std::io::Error>;
    fn write_sequence_header_obu(&mut self, seq: &mut Sequence, fi: &FrameInvariants)
            -> Result<(), std::io::Error>;
    fn write_frame_header_obu(&mut self, seq: &Sequence, fi: &mut FrameInvariants)
            -> Result<(), std::io::Error>;
    fn write_sequence_header2(&mut self, seq: &mut Sequence, fi: &FrameInvariants)
                                    -> Result<(), std::io::Error>;
    fn write_color_config(&mut self, seq: &mut Sequence) -> Result<(), std::io::Error>;
    // End of OBU Headers

    fn write_frame_size(&mut self, fi: &FrameInvariants) -> Result<(), std::io::Error>;
    fn write_sequence_header(&mut self, fi: &FrameInvariants)
                                    -> Result<(), std::io::Error>;
    fn write_bitdepth_colorspace_sampling(&mut self) -> Result<(), std::io::Error>;
    fn write_frame_setup(&mut self) -> Result<(), std::io::Error>;
    fn write_loop_filter(&mut self) -> Result<(), std::io::Error>;
    fn write_cdef(&mut self) -> Result<(), std::io::Error>;
}

const OP_POINTS_IDC_BITS:usize = 12;
const LEVEL_MAJOR_MIN:usize = 2;
const LEVEL_MAJOR_BITS:usize = 3;
const LEVEL_MINOR_BITS:usize = 2;
const LEVEL_BITS:usize = LEVEL_MAJOR_BITS + LEVEL_MINOR_BITS;
const FRAME_ID_LENGTH: usize = 15;
const DELTA_FRAME_ID_LENGTH: usize = 14;

impl<'a> UncompressedHeader for BitWriter<'a, BE> {
    // Start of OBU Headers
    // Write OBU Header syntax
    fn write_obu_header(&mut self, obu_type: OBU_Type, obu_extension: u32)
            -> Result<(), std::io::Error>{
        self.write(1, 0)?; // forbidden bit.
        self.write(4, obu_type as u32)?;
        self.write_bit(obu_extension != 0)?;
        self.write(1, 1)?; // obu_has_payload_length_field
        self.write(1, 0)?; // reserved

        if obu_extension != 0 {
            assert!(false);
            //self.write(8, obu_extension & 0xFF)?; size += 8;
        }

        Ok(())
    }

#[allow(unused)]
    fn write_sequence_header_obu(&mut self, seq: &mut Sequence, fi: &FrameInvariants)
        -> Result<(), std::io::Error> {
        self.write(3, seq.profile)?; // profile 0, 3 bits
        self.write(1, 0)?; // still_picture
        self.write(1, 0)?; // reduced_still_picture

        if seq.reduced_still_picture_hdr {
            assert!(false);
        } else {
            self.write(1, 0)?; // timing_info_present_flag
            if false { // if timing_info_present_flag == true
                assert!(false);
            }

            self.write(1, 0)?; // display_model_info_present_flag
            self.write(5, 0)?; // operating_points_cnt_minus_1, 5 bits

            for i in 0..seq.operating_points_cnt_minus_1 + 1 {
                self.write(OP_POINTS_IDC_BITS as u32, seq.operating_point_idc[i])?;
                //let seq_level_idx = 1 as u16;	// NOTE: This comes from minor and major
                let seq_level_idx = 
                    ((seq.level[i][1] - LEVEL_MAJOR_MIN) << LEVEL_MINOR_BITS) + seq.level[i][0]; 
                self.write(LEVEL_BITS as u32, seq_level_idx as u16)?;

                if seq.level[i][1] > 3 {
                    assert!(false); // NOTE: Not supported yet.
                    //self.write(1, seq.tier[i])?;
                }
                if seq.decoder_model_info_present_flag {
                    assert!(false); // NOTE: Not supported yet.
                }
                if seq.display_model_info_present_flag {
                    assert!(false); // NOTE: Not supported yet.
                }
            }
        }

        self.write_sequence_header2(seq, fi);

        self.write_color_config(seq)?;

        self.write_sequence_header2(seq, fi);

        self.write_bit(seq.film_grain_params_present)?;

        self.write(1,1)?; // add_trailing_bits

        Ok(())
    }

#[allow(unused)]
    fn write_sequence_header2(&mut self, seq: &mut Sequence, fi: &FrameInvariants)
        -> Result<(), std::io::Error> {
        self.write(4, seq.num_bits_width - 1)?;
        self.write(4, seq.num_bits_height - 1)?;
        self.write(seq.num_bits_width, (seq.max_frame_width - 1) as u16)?;
        self.write(seq.num_bits_height, (seq.max_frame_height - 1) as u16)?;

        if !seq.reduced_still_picture_hdr {
            seq.frame_id_numbers_present_flag =
                if fi.large_scale_tile { false } else { fi.error_resilient };
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
              self.write(1, 1)?;
            } else {
              self.write(1, 0)?;
              self.write_bit(seq.force_screen_content_tools != 0)?;
            }
            if seq.force_screen_content_tools > 0 {
              if seq.force_integer_mv == 2 {
                self.write(1, 1)?;
              } else {
                self.write(1, 0)?;
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

#[allow(unused)]
    fn write_color_config(&mut self, seq: &mut Sequence) -> Result<(), std::io::Error> {
        self.write(1,0)?; // 8 bit video
        self.write_bit(seq.monochrome)?; 	// monochrome?
        self.write_bit(false)?;  					// No color description present

        if seq.monochrome {
            assert!(false);
        }
        self.write(1,0)?; // color range

        if true { // subsampling_x == 1 && cm->subsampling_y == 1
            self.write(2,0)?; // chroma_sample_position == AOM_CSP_UNKNOWN
        }
        self.write_bit(seq.separate_uv_delta_q)?;

        Ok(())
    }

#[allow(unused)]
    fn write_frame_header_obu(&mut self, seq: &Sequence, fi: &mut FrameInvariants)
        -> Result<(), std::io::Error> {
      if seq.reduced_still_picture_hdr {
        assert!(fi.show_existing_frame);
        assert!(fi.frame_type == FrameType::KEY);
        assert!(fi.show_frame);
      } else {
        if fi.show_existing_frame {
          self.write_bit(true)?; // show_existing_frame=1
          self.write(3,0)?; // show last frame

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
        //let frame_type = fi.frame_type;
        self.write(2, fi.frame_type as u32)?;
        fi.intra_only =
          if fi.frame_type == FrameType::KEY ||
            fi.frame_type == FrameType::INTRA_ONLY { true }
          else { false };
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

        // NOTE: DO this before encoding started atm.
        //fi.primary_ref_frame = PRIMARY_REF_NONE;
      } else {
        self.write(PRIMARY_REF_BITS, fi.primary_ref_frame)?;
      }

      if seq.decoder_model_info_present_flag {
        assert!(false); // Not supported by rav1e yet!
      }

      if fi.frame_type == FrameType::KEY {
        if !fi.show_frame {  // unshown keyframe (forward keyframe)
          assert!(false); // Not supported by rav1e yet!
          //self.write_bit(REF_FRAMES, fi.refresh_frame_flags)?;
        } else {
          //assert!(refresh_frame_mask == 0xFF);
        }
      } else { // Inter frame info goes here
        if fi.intra_only {
          self.write(REF_FRAMES,0)?; // refresh_frame_flags
        } else {
          // TODO: This should be set once inter mode is used
          self.write(REF_FRAMES,0)?; // refresh_frame_flags
        }
      };

      if (!fi.intra_only || fi.refresh_frame_flags != 0xFF) {
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

      let mut frame_refs_short_signaling = false;
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
          } else { frame_refs_short_signaling = true; }

          for i in LAST_FRAME..ALTREF_FRAME+1 {
            if !frame_refs_short_signaling {
              self.write(REF_FRAMES_LOG2, 0)?;
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
            //if render_and_frame_size_different { }
          }
           if fi.force_integer_mv != 0 {
            fi.allow_high_precision_mv = false;
          } else {
            self.write_bit(fi.allow_high_precision_mv);
          }
          self.write_bit(fi.is_filter_switchable)?;
          self.write_bit(fi.is_motion_mode_switchable)?;
          if fi.error_resilient || !seq.enable_ref_frame_mvs {
            fi.use_ref_frame_mvs = false;
          } else {
            self.write_bit(fi.use_ref_frame_mvs)?;
          }
        }
      }

      if seq.reduced_still_picture_hdr || fi.disable_cdf_update {
        fi.disable_frame_end_update_cdf = true;
      } else {
        self.write_bit(fi.disable_frame_end_update_cdf)?;
      }

      // tile
      self.write_bit(true)?; // uniform_tile_spacing_flag
      if fi.width > 64 {
        // TODO: if tile_cols > 1, write more increment_tile_cols_log2 bits
        self.write(1,0)?; // tile cols
      }
      if fi.height > 64 {
        // TODO: if tile_rows > 1, write increment_tile_rows_log2 bits
        self.write(1,0)?; // tile rows
      }
      // TODO: if tile_cols * tile_rows > 1 {
      // write context_update_tile_id and tile_size_bytes_minus_1 }

      // quantization
      assert!(fi.qindex > 0);
      self.write(8,fi.qindex as u8)?; // base_q_idx
      self.write_bit(false)?; // y dc delta q
      self.write_bit(false)?; // uv dc delta q
      self.write_bit(false)?; // uv ac delta q
      self.write_bit(false)?; // no qm

      // segmentation
      self.write_bit(false)?; // segmentation is disabled

      // delta_q
      self.write_bit(false)?; // delta_q_present_flag: no delta q

      // loop filter
      self.write_loop_filter()?;
      // cdef
      self.write_cdef()?;
      // loop restoration
      // If seq.enable_restoration is false, don't signal about loop restoration
      if seq.enable_restoration {
        //self.write(6,0)?; // no y, u or v loop restoration
      }
      self.write_bit(false)?; // tx mode == TX_MODE_SELECT ?

      // frame_reference_mode : reference_select?
      let mut reference_select = false;
      if !fi.intra_only {
        reference_select = fi.reference_mode != ReferenceMode::SINGLE;
        self.write_bit(reference_select)?;
      }

      let skip_mode_allowed =
        !(fi.intra_only  || !reference_select || !seq.enable_order_hint);
      if skip_mode_allowed {
        self.write_bit(false)?; // skip_mode_present
      }

      if fi.intra_only || fi.error_resilient || !seq.enable_warped_motion {
        fi.allow_warped_motion = false;
      } else {
        self.write_bit(fi.allow_warped_motion)?; // allow_warped_motion
      }

      self.write_bit(fi.use_reduced_tx_set)?; // reduced tx

      // global motion
      if fi.intra_only == false {
        for i in LAST_FRAME..ALTREF_FRAME+1 {
          let mode = fi.globalmv_transformation_type[i];
          self.write_bit(mode != GlobalMVMode::IDENTITY)?;
          if mode != GlobalMVMode::IDENTITY {
            self.write_bit(mode == GlobalMVMode::ROTZOOM)?;
            if mode != GlobalMVMode::ROTZOOM {
                self.write_bit(mode == GlobalMVMode::TRANSLATION)?;
            }
          }

          if mode >= GlobalMVMode::ROTZOOM {
            unimplemented!();
          }
          if mode >= GlobalMVMode::AFFINE {
            unimplemented!();
          }
          if mode >= GlobalMVMode::TRANSLATION {
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
          };
        }
      }

      if seq.film_grain_params_present && fi.show_frame {
        assert!(false); // Not supported by rav1e yet!
      }

      if fi.large_scale_tile {
        assert!(false); // Not supported by rav1e yet!
        // write ext_file info
      }
      self.byte_align()?;

      // TODO: update size
      // size +=

      Ok(())
    }
    // End of OBU Headers

    fn write_frame_size(&mut self, fi: &FrameInvariants) -> Result<(), std::io::Error> {
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
    fn write_sequence_header(&mut self, fi: &FrameInvariants)
        -> Result<(), std::io::Error> {
        self.write_frame_size(fi)?;
        self.write(1,0)?; // don't use frame ids
        self.write(1,0)?; // screen content tools forced
        self.write(1,0)?; // screen content tools forced off
        Ok(())
    }
    fn write_bitdepth_colorspace_sampling(&mut self) -> Result<(), std::io::Error> {
        self.write(1,0)?; // 8 bit video
        self.write(1,0)?; // not monochrome
        self.write(4,0)?; // colorspace
        self.write(1,0)?; // color range
        Ok(())
    }
    fn write_frame_setup(&mut self) -> Result<(), std::io::Error> {
        self.write_bit(false)?; // no superres
        self.write_bit(false)?; // scaling active
        Ok(())
    }
    fn write_loop_filter(&mut self) -> Result<(), std::io::Error> {
        self.write(6,0)?; // loop filter level 0
        self.write(6,0)?; // loop filter level 1
        self.write(3,0)?; // loop filter sharpness
        self.write_bit(false) // loop filter deltas enabled
    }
    fn write_cdef(&mut self) -> Result<(), std::io::Error> {
        self.write(2,0)?; // cdef clpf damping
        self.write(2,0)?; // cdef bits
        for _ in 0..1 {
            self.write(6,0)?; // cdef y strength
            self.write(6,0)?; // cdef uv strength
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
#[allow(unused)]
fn aom_uleb_size_in_bytes(mut value: u64) -> usize {
  let mut size = 0;
  loop {
    size += 1;
    value = value >> 7;
    if value == 0 { break; }
  }
  return size;
}

#[allow(unused)]
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

#[allow(unused)]
fn write_obus(packet: &mut Write, sequence: &mut Sequence,
                            fi: &mut FrameInvariants) -> Result<(), std::io::Error> {
    //let mut uch = BitWriter::<BE>::new(packet);
    let obu_extension = 0 as u32;

    let mut buf1 = Vec::new();
    {
      let mut bw1 = BitWriter::<BE>::new(&mut buf1);
      bw1.write_obu_header(OBU_Type::OBU_TEMPORAL_DELIMITER, obu_extension);
      bw1.write(8,0)?;	// size of payload == 0, one byte
    }
    packet.write(&buf1).unwrap();
    buf1.clear();

    // write sequence header obu if KEY_FRAME, preceded by 4-byte size
    if fi.frame_type == FrameType::KEY {
        {
            let mut bw1 = BitWriter::<BE>::new(&mut buf1);
            bw1.write_obu_header(OBU_Type::OBU_SEQUENCE_HEADER, obu_extension);
        }
        packet.write(&buf1).unwrap();
        buf1.clear();

        let mut buf2 = Vec::new();
        {
            let mut bw2 = BitWriter::<BE>::new(&mut buf2);
            bw2.write_sequence_header_obu(sequence, fi);
            bw2.byte_align()?;
        }
        let obu_payload_size = buf2.len() as u64;
        {
            let mut bw1 = BitWriter::<BE>::new(&mut buf1);
            // uleb128()
            let mut coded_payload_length = [0 as u8; 8];
            let leb_size = aom_uleb_encode(obu_payload_size, &mut coded_payload_length);
            for i in 0..leb_size {
                bw1.write(8, coded_payload_length[i]);
            }
        }
        packet.write(&buf1).unwrap();
        buf1.clear();

        packet.write(&buf2).unwrap();
        buf2.clear();
    }

    let write_frame_header = fi.num_tg > 1 || fi.show_existing_frame;

    if write_frame_header {
        // TODO: If # of tiles > 1 or show_existing_frame is true,
        // write Frame Header OBU here.
        {
            let mut bw1 = BitWriter::<BE>::new(&mut buf1);
            bw1.write_obu_header(OBU_Type::OBU_FRAME_HEADER, obu_extension);
        }
        packet.write(&buf1).unwrap();
        buf1.clear();

        let mut buf2 = Vec::new();
        {
            let mut bw2 = BitWriter::<BE>::new(&mut buf2);
            let error = bw2.write_frame_header_obu(sequence, fi);
            bw2.byte_align()?;
        }
        let obu_payload_size = buf2.len() as u64;
        {
            let mut bw1 = BitWriter::<BE>::new(&mut buf1);
            // uleb128()
            let mut coded_payload_length = [0 as u8; 8];
            let leb_size = aom_uleb_encode(obu_payload_size, &mut coded_payload_length);
            for i in 0..leb_size {
                bw1.write(8, coded_payload_length[i]);
            }
        }
        packet.write(&buf1).unwrap();
        buf1.clear();

        packet.write(&buf2).unwrap();
        buf2.clear();
    }

    // TODO: Below will be done in encode_tile() but not here.
/*
    if (cm->show_existing_frame) {
        data_size = 0;
    } else {
        //  Each tile group obu will be preceded by 4-byte size of the tile group
        //  obu
        data_size = write_tiles_in_tg_obus(cpi, data, &saved_wb,
                                        obu_extension_header, &fh_info);
    }
*/
    Ok(())
}

fn write_uncompressed_header(packet: &mut Write, sequence: &Sequence,
                            fi: &FrameInvariants) -> Result<(), std::io::Error> {
    let mut bw = BitWriter::<BE>::new(packet);
    bw.write(2,2)?; // AOM_FRAME_MARKER, 0x2
    bw.write(2,sequence.profile)?; // profile 0
    if fi.show_existing_frame {
        bw.write_bit(true)?; // show_existing_frame=1
        bw.write(3,0)?; // show last frame
        bw.byte_align()?;
        return Ok(());
    }
    bw.write_bit(false)?; // show_existing_frame=0
    bw.write_bit(fi.frame_type == FrameType::INTER)?; // keyframe : 0, inter: 1
    bw.write_bit(fi.show_frame)?; // show frame
    /*
    if fi.intra_only {
        bw.write_bit(true)?; // disable intra edge
    }
    */
    if fi.frame_type == FrameType::KEY || fi.frame_type == FrameType::INTRA_ONLY {
        assert!(fi.intra_only);
    }
    if fi.frame_type != FrameType::KEY {
        if fi.show_frame { assert!(!fi.intra_only); }
        else { bw.write_bit( fi.intra_only )?; };
    };
    bw.write_bit(fi.error_resilient)?; // error resilient

    if fi.frame_type == FrameType::KEY || fi.intra_only {
        bw.write_sequence_header(fi)?;
    }

    //bw.write(8+7,0)?; // frame id

    bw.write_bit(false)?; // no override frame size

    if fi.frame_type == FrameType::KEY {
        bw.write_bitdepth_colorspace_sampling()?;
        bw.write(1,0)?; // separate uv delta q
        bw.write_frame_setup()?;
    } else { // Inter frame info goes here
        if fi.intra_only {
            bw.write_bitdepth_colorspace_sampling()?;
            bw.write(1,0)?; // separate uv delta q
            bw.write(8,0)?; // refresh_frame_flags
            bw.write_frame_setup()?;
        } else {
            bw.write(8,0)?; // refresh_frame_flags
            // TODO: More Inter frame info goes here
            for _ in 0..7 {
                bw.write(3,0)?; // dummy ref_frame = 0 until real MC happens
            }
            bw.write_frame_setup()?;
            bw.write_bit(fi.allow_high_precision_mv)?;
            bw.write_bit(false)?; // frame_interp_filter is NOT switchable
            bw.write(2,0)?;	// EIGHTTAP_REGULAR
            if !fi.intra_only && !fi.error_resilient {
                bw.write_bit(false)?; // do not use_ref_frame_mvs
            }
        }
    };


    bw.write(3,0x0)?; // frame context
    bw.write_loop_filter()?;
    bw.write(8,fi.qindex as u8)?; // qindex
    bw.write_bit(false)?; // y dc delta q
    bw.write_bit(false)?; // uv dc delta q
    bw.write_bit(false)?; // uv ac delta q
    bw.write_bit(false)?; // no qm
    bw.write_bit(false)?; // segmentation off
    bw.write_bit(false)?; // no delta q
    bw.write_cdef()?;
    bw.write(6,0)?; // no y, u or v loop restoration
    bw.write_bit(false)?; // tx mode select

    //fi.reference_mode = ReferenceMode::SINGLE;

    if fi.reference_mode != ReferenceMode::SINGLE {
        // setup_compound_reference_mode();
    }

    if !fi.intra_only {
        bw.write_bit(false)?; } // do not use inter_intra
    if !fi.intra_only && fi.reference_mode != ReferenceMode::SINGLE {
        bw.write_bit(false)?; } // do not allow_masked_compound

    bw.write_bit(fi.use_reduced_tx_set)?; // reduced tx

    if !fi.intra_only {
        for i in LAST_FRAME..ALTREF_FRAME+1 {
            let mode = fi.globalmv_transformation_type[i];
            bw.write_bit(mode != GlobalMVMode::IDENTITY)?;
            if mode != GlobalMVMode::IDENTITY {
                bw.write_bit(mode == GlobalMVMode::ROTZOOM)?;
                if mode != GlobalMVMode::ROTZOOM {
                    bw.write_bit(mode == GlobalMVMode::TRANSLATION)?;
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
                    BCodeWriter::write_s_refsubexpfin(&mut bw, (1 << bits) + 1,
                                                      3, mv_x_ref >> bits_diff,
                                                      mv_x >> bits_diff)?;
                    BCodeWriter::write_s_refsubexpfin(&mut bw, (1 << bits) + 1,
                                                      3, mv_y_ref >> bits_diff,
                                                      mv_y >> bits_diff)?;
                }
                GlobalMVMode::ROTZOOM => unimplemented!(),
                GlobalMVMode::AFFINE => unimplemented!(),
            };
        }
    }

    bw.write_bit(true)?; // uniform tile spacing
    if fi.width > 64 {
        bw.write(1,0)?; // tile cols
    }
    if fi.height > 64 {
        bw.write(1,0)?; // tile rows
    }
    // if tile_cols * tile_rows > 1
    //.write_bit(true)?; // loop filter across tiles
    bw.write(2,3)?; // tile_size_bytes
    bw.byte_align()?;
    Ok(())
}

/// Write into `dst` the difference between the blocks at `src1` and `src2`
fn diff(dst: &mut [i16], src1: &PlaneSlice, src2: &PlaneSlice, width: usize, height: usize) {
    for j in 0..height {
        for i in 0..width {
            dst[j*width + i] = (src1.p(i, j) as i16) - (src2.p(i, j) as i16);
        }
    }
}

use std::mem::uninitialized;

// For a transform block,
// predict, transform, quantize, write coefficients to a bitstream,
// dequantize, inverse-transform.
pub fn encode_tx_block(fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter,
                  p: usize, bo: &BlockOffset, mode: PredictionMode, tx_size: TxSize, tx_type: TxType,
                  plane_bsize: BlockSize, po: &PlaneOffset, skip: bool) {
    let rec = &mut fs.rec.planes[p];
    let PlaneConfig { stride, xdec, ydec } = fs.input.planes[p].cfg;

    mode.predict(&mut rec.mut_slice(po), tx_size);

    if skip { return; }

    let mut residual: [i16; 64*64] = unsafe { uninitialized() };
    let mut coeffs_storage: [i32; 64*64] = unsafe { uninitialized() };
    let mut rcoeffs: [i32; 64*64] = unsafe { uninitialized() };

    let coeffs = &mut coeffs_storage[..tx_size.area()];

    diff(&mut residual,
         &fs.input.planes[p].slice(po),
         &rec.slice(po),
         tx_size.width(),
         tx_size.height());


    forward_transform(&residual, coeffs, tx_size.width(), tx_size, tx_type);
    quantize_in_place(fi.qindex, coeffs, tx_size);

    cw.write_coeffs_lv_map(p, bo, &coeffs, tx_size, tx_type, plane_bsize, xdec, ydec,
                            fi.use_reduced_tx_set);

    // Reconstruct
    dequantize(fi.qindex, &coeffs, &mut rcoeffs, tx_size);

    inverse_transform_add(&rcoeffs, &mut rec.mut_slice(po).as_mut_slice(), stride, tx_size, tx_type);
}

fn encode_block(fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter,
            luma_mode: PredictionMode, chroma_mode: PredictionMode,
            bsize: BlockSize, bo: &BlockOffset, skip: bool) {
    let is_inter = luma_mode >= PredictionMode::NEARESTMV;

    cw.bc.set_skip(bo, bsize, skip);
    cw.write_skip(bo, skip);

    if fi.frame_type == FrameType::INTER {
        cw.write_is_inter(bo, is_inter);
        if !is_inter {
            cw.write_intra_mode(bsize, luma_mode);
        }
    } else {
        cw.write_intra_mode_kf(bo, luma_mode);
    }

    cw.bc.set_mode(bo, bsize, luma_mode);

    let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;

    if luma_mode.is_directional() && bsize >= BlockSize::BLOCK_8X8 {
        cw.write_angle_delta(0, luma_mode);
    }

    if has_chroma(bo, bsize, xdec, ydec) {
        cw.write_intra_uv_mode(chroma_mode, luma_mode, bsize);
        if chroma_mode.is_directional() && bsize >= BlockSize::BLOCK_8X8 {
            cw.write_angle_delta(0, chroma_mode);
        }
    }

    if skip {
        cw.bc.reset_skip_context(bo, bsize, xdec, ydec);
    }

    // these rules follow TX_MODE_LARGEST
    let tx_size = match bsize {
        BlockSize::BLOCK_4X4 => TxSize::TX_4X4,
        BlockSize::BLOCK_8X8 => TxSize::TX_8X8,
        BlockSize::BLOCK_16X16 => TxSize::TX_16X16,
        _ => TxSize::TX_32X32
    };

    // Luma plane transform type decision
    let tx_set_type = get_ext_tx_set_type(tx_size, is_inter, fi.use_reduced_tx_set);

    let tx_type = if tx_set_type > TxSetType::EXT_TX_SET_DCTONLY && fi.speed <= 3 {
        // FIXME: there is one redundant transform type decision per encoded block
        rdo_tx_type_decision(fi, fs, cw, luma_mode, bsize, bo, tx_size, tx_set_type)
    } else {
        TxType::DCT_DCT
    };

    write_tx_blocks(fi, fs, cw, luma_mode, chroma_mode, bo, bsize, tx_size, tx_type, skip);
}

pub fn write_tx_blocks(fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter,
                       luma_mode: PredictionMode, chroma_mode: PredictionMode, bo: &BlockOffset,
                       bsize: BlockSize, tx_size: TxSize, tx_type: TxType, skip: bool) {
    let bw = bsize.width_mi() / tx_size.width_mi();
    let bh = bsize.height_mi() / tx_size.height_mi();

    let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;

    for by in 0..bh {
        for bx in 0..bw {
            let tx_bo = BlockOffset {
                x: bo.x + bx * tx_size.width_mi(),
                y: bo.y + by * tx_size.height_mi()
            };

            let po = tx_bo.plane_offset(&fs.input.planes[0].cfg);
            encode_tx_block(fi, fs, cw, 0, &tx_bo, luma_mode, tx_size, tx_type, bsize, &po, skip);
        }
    }

    // these are only valid for 4:2:0
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
        let uv_tx_type = uv_intra_mode_to_tx_type_context(chroma_mode);
        let partition_x = (bo.x & LOCAL_BLOCK_MASK) >> xdec << MI_SIZE_LOG2;
        let partition_y = (bo.y & LOCAL_BLOCK_MASK) >> ydec << MI_SIZE_LOG2;

        for p in 1..3 {
            let sb_offset = bo.sb_offset().plane_offset(&fs.input.planes[p].cfg);

            for by in 0..bh_uv {
                for bx in 0..bw_uv {
                    let tx_bo =
                        BlockOffset {
                            x: bo.x + ((bx * uv_tx_size.width_mi()) << xdec) -
                                ((bw * tx_size.width_mi() == 1) as usize),
                            y: bo.y + ((by * uv_tx_size.height_mi()) << ydec) -
                                ((bh * tx_size.height_mi() == 1) as usize)
                        };

                    let po = PlaneOffset {
                        x: sb_offset.x + partition_x + bx * uv_tx_size.width(),
                        y: sb_offset.y + partition_y + by * uv_tx_size.height()
                    };

                    encode_tx_block(fi, fs, cw, p, &tx_bo, chroma_mode, uv_tx_size, uv_tx_type,
                                    plane_bsize, &po, skip);
                }
            }
        }
    }
}

fn encode_partition_bottomup(fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter,
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
        skip: false
    }; // Best decision that is not PARTITION_SPLIT

    let hbs = bs >> 1; // Half the block size in blocks
    let mut subsize: BlockSize;

    let checkpoint = cw.checkpoint();

    // Code the whole block
    if !must_split {
        partition = PartitionType::PARTITION_NONE;

        if bsize >= BlockSize::BLOCK_8X8 {
            cw.write_partition(bo, partition, bsize);
        }

        let mode_decision = rdo_mode_decision(fi, fs, cw, bsize, bo).part_modes[0].clone();
        let (mode_luma, mode_chroma) = (mode_decision.pred_mode_luma, mode_decision.pred_mode_chroma);
        let skip = mode_decision.skip;
        rd_cost = mode_decision.rd_cost;

        encode_block(fi, fs, cw, mode_luma, mode_chroma, bsize, bo, skip);

        best_decision = mode_decision;
    }

    // Code a split partition and compare RD costs
    if can_split {
        cw.rollback(&checkpoint);

        partition = PartitionType::PARTITION_SPLIT;
        subsize = get_subsize(bsize, partition);

        let nosplit_rd_cost = rd_cost;

        if bsize >= BlockSize::BLOCK_8X8 {
            cw.write_partition(bo, partition, bsize);
        }

        rd_cost = encode_partition_bottomup(fi, fs, cw, subsize, bo);
        rd_cost += encode_partition_bottomup(fi, fs, cw, subsize, &BlockOffset { x: bo.x + hbs as usize, y: bo.y });
        rd_cost += encode_partition_bottomup(fi, fs, cw, subsize, &BlockOffset { x: bo.x, y: bo.y + hbs as usize });
        rd_cost += encode_partition_bottomup(fi, fs, cw, subsize, &BlockOffset { x: bo.x + hbs as usize, y: bo.y + hbs as usize });

        // Recode the full block if it is more efficient
        if !must_split && nosplit_rd_cost < rd_cost {
            cw.rollback(&checkpoint);

            partition = PartitionType::PARTITION_NONE;

            if bsize >= BlockSize::BLOCK_8X8 {
                cw.write_partition(bo, partition, bsize);
            }

            // FIXME: redundant block re-encode
            let (mode_luma, mode_chroma) = (best_decision.pred_mode_luma, best_decision.pred_mode_chroma);
            let skip = best_decision.skip;
            encode_block(fi, fs, cw, mode_luma, mode_chroma, bsize, bo, skip);
        }
    }

    subsize = get_subsize(bsize, partition);

    if bsize >= BlockSize::BLOCK_8X8 &&
        (bsize == BlockSize::BLOCK_8X8 || partition != PartitionType::PARTITION_SPLIT) {
        cw.bc.update_partition_context(bo, subsize, bsize);
    }

    rd_cost
}

fn encode_partition_topdown(fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter,
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
        part_modes: std::vec::Vec::new()
    });
    let partition: PartitionType;

    if must_split {
        // Oversized blocks are split automatically
        partition = PartitionType::PARTITION_SPLIT;
    } else if bsize > fi.min_partition_size {
        // Blocks of sizes within the supported range are subjected to a partitioning decision
        rdo_output = rdo_partition_decision(fi, fs, cw, bsize, bo, &rdo_output);
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
        cw.write_partition(bo, partition, bsize);
    }

    match partition {
        PartitionType::PARTITION_NONE => {
            let part_decision = if !rdo_output.part_modes.is_empty() {
                    // The optimal prediction mode is known from a previous iteration
                    rdo_output.part_modes[0].clone()
                } else {
                    // Make a prediction mode decision for blocks encoded with no rdo_partition_decision call (e.g. edges)
                    rdo_mode_decision(fi, fs, cw, bsize, bo).part_modes[0].clone()
                };

            let (mode_luma, mode_chroma) = (part_decision.pred_mode_luma, part_decision.pred_mode_chroma);
            let skip = part_decision.skip;

            // FIXME: every final block that has gone through the RDO decision process is encoded twice
            encode_block(fi, fs, cw, mode_luma, mode_chroma, bsize, bo, skip);
        },
        PartitionType::PARTITION_SPLIT => {
            if rdo_output.part_modes.len() >= 4 {
                // The optimal prediction modes for each split block is known from an rdo_partition_decision() call
                assert!(subsize != BlockSize::BLOCK_INVALID);

                for mode in rdo_output.part_modes {
                    let offset = mode.bo.clone();

                    // Each block is subjected to a new splitting decision
                    encode_partition_topdown(fi, fs, cw, subsize, &offset,
                        &Some(RDOOutput {
                            rd_cost: mode.rd_cost,
                            part_type: PartitionType::PARTITION_NONE,
                            part_modes: vec![mode] }));
                }
            }
            else {
                encode_partition_topdown(fi, fs, cw, subsize, bo, &None);
                encode_partition_topdown(fi, fs, cw, subsize, &BlockOffset{x: bo.x + hbs as usize, y: bo.y}, &None);
                encode_partition_topdown(fi, fs, cw, subsize, &BlockOffset{x: bo.x, y: bo.y + hbs as usize}, &None);
                encode_partition_topdown(fi, fs, cw, subsize, &BlockOffset{x: bo.x + hbs as usize, y: bo.y + hbs as usize}, &None);
            }
        },
        _ => { assert!(false); },
    }

    if bsize >= BlockSize::BLOCK_8X8 &&
        (bsize == BlockSize::BLOCK_8X8 || partition != PartitionType::PARTITION_SPLIT) {
            cw.bc.update_partition_context(bo, subsize, bsize);
    }
}

fn encode_tile(fi: &FrameInvariants, fs: &mut FrameState) -> Vec<u8> {
    let w = ec::Writer::new();
    let fc = CDFContext::new(fi.qindex as u8);
    let bc = BlockContext::new(fi.w_in_b, fi.h_in_b);
    let mut cw = ContextWriter::new(w, fc,  bc);

    for sby in 0..fi.sb_height {
        cw.bc.reset_left_contexts();

        for sbx in 0..fi.sb_width {
            let sbo = SuperBlockOffset { x: sbx, y: sby };
            let bo = sbo.block_offset(0, 0);

            // Encode SuperBlock
            if fi.speed == 0 {
                encode_partition_bottomup(fi, fs, &mut cw, BlockSize::BLOCK_64X64, &bo);
            }
            else {
                encode_partition_topdown(fi, fs, &mut cw, BlockSize::BLOCK_64X64, &bo, &None);
            }
        }
    }
    let mut h = cw.w.done();
    h.push(0); // superframe anti emulation
    h
}

fn encode_frame(sequence: &mut Sequence, fi: &mut FrameInvariants, fs: &mut FrameState, last_rec: &Option<Frame>) -> Vec<u8> {
    let mut packet = Vec::new();
    write_uncompressed_header(&mut packet, sequence, fi).unwrap();
    //write_obus(&mut packet, sequence, fi).unwrap();
    if fi.show_existing_frame {
        match last_rec {
            Some(ref rec) => for p in 0..3 {
                fs.rec.planes[p].data.copy_from_slice(rec.planes[p].data.as_slice());
            },
            None => (),
        }
    } else {
        let tile = encode_tile(fi, fs);
        packet.write(&tile).unwrap();
    }
    packet
}

/// Encode and write a frame.
pub fn process_frame(sequence: &mut Sequence, fi: &mut FrameInvariants,
                     output_file: &mut Write,
                     y4m_dec: &mut y4m::Decoder<Box<Read>>,
                     y4m_enc: Option<&mut y4m::Encoder<Box<Write>>>,
                     last_rec: &mut Option<Frame>) -> bool {
    unsafe {
        av1_rtcd();
        aom_dsp_rtcd();
    }
    let width = fi.width;
    let height = fi.height;
    let y4m_bits = y4m_dec.get_bit_depth();
    let y4m_bytes = y4m_dec.get_bytes_per_sample();
    let csp = y4m_dec.get_colorspace();
    match csp {
        y4m::Colorspace::C420 | 
        y4m::Colorspace::C420jpeg |
        y4m::Colorspace::C420paldv | 
        y4m::Colorspace::C420mpeg2 => {},
        _ => {
            panic!("Colorspace {:?} is not supported yet.", csp);
        },
    }
    match y4m_dec.read_frame() {
        Ok(y4m_frame) => {
            let y4m_y = y4m_frame.get_y_plane();
            let y4m_u = y4m_frame.get_u_plane();
            let y4m_v = y4m_frame.get_v_plane();
            eprintln!("{}", fi);
            let mut fs = FrameState::new(&fi);
            fs.input.planes[0].copy_from_raw_u8(&y4m_y, width*y4m_bytes, y4m_bytes);
            fs.input.planes[1].copy_from_raw_u8(&y4m_u, width*y4m_bytes/2, y4m_bytes);
            fs.input.planes[2].copy_from_raw_u8(&y4m_v, width*y4m_bytes/2, y4m_bytes);

            // We cannot currently encode > 8 bit input!
            match y4m_bits {
                8 => {},
                10 | 12 => {
                    for plane in 0..3 {
                        for row in fs.input.planes[plane].data.chunks_mut(fs.rec.planes[plane].cfg.stride) {
                            for col in row.iter_mut() { *col >>= y4m_bits-8 }
                        }
                    }
                },
                _ => panic! ("unknown input bit depth!"),
            }

            let packet = encode_frame(sequence, fi, &mut fs, &last_rec);
            write_ivf_frame(output_file, fi.number, packet.as_ref());
            if let Some(mut y4m_enc) = y4m_enc {
                let mut rec_y = vec![128 as u8; width*height];
                let mut rec_u = vec![128 as u8; width*height/4];
                let mut rec_v = vec![128 as u8; width*height/4];
                for (y, line) in rec_y.chunks_mut(width).enumerate() {
                    for (x, pixel) in line.iter_mut().enumerate() {
                        let stride = fs.rec.planes[0].cfg.stride;
                        *pixel = fs.rec.planes[0].data[y*stride+x] as u8;
                    }
                }
                for (y, line) in rec_u.chunks_mut(width/2).enumerate() {
                    for (x, pixel) in line.iter_mut().enumerate() {
                        let stride = fs.rec.planes[1].cfg.stride;
                        *pixel = fs.rec.planes[1].data[y*stride+x] as u8;
                    }
                }
                for (y, line) in rec_v.chunks_mut(width/2).enumerate() {
                    for (x, pixel) in line.iter_mut().enumerate() {
                        let stride = fs.rec.planes[2].cfg.stride;
                        *pixel = fs.rec.planes[2].data[y*stride+x] as u8;
                    }
                }
                let rec_frame = y4m::Frame::new([&rec_y, &rec_u, &rec_v], None);
                y4m_enc.write_frame(&rec_frame).unwrap();
            }
            *last_rec = Some(fs.rec);
            true
        },
        _ => false
    }
}
