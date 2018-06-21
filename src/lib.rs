// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(safe_extern_statics)]

extern crate bitstream_io;
extern crate backtrace;
extern crate byteorder;
extern crate clap;
extern crate libc;
extern crate rand;
extern crate y4m;

#[macro_use]
extern crate enum_iterator_derive;

use std::fs::File;
use std::io::prelude::*;
use bitstream_io::{BE, BitWriter};
use byteorder::*;
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

pub struct Sequence {
    pub profile: u8
}

impl Sequence {
    pub fn new() -> Sequence {
        Sequence {
            profile: 0
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
    pub error_resilient: bool,
    pub intra_only: bool,
    pub frame_type: FrameType,
    pub show_existing_frame: bool,
    pub use_reduced_tx_set: bool,
    pub reference_mode: ReferenceMode,
    pub use_prev_frame_mvs: bool,
    pub min_partition_size: BlockSize,
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
        FrameInvariants {
            qindex: qindex,
            speed: speed,
            width: width,
            height: height,
            padded_w: ((width+7)>>3)<<3,
            padded_h: ((height+7)>>3)<<3,
            sb_width: (width+63)/64,
            sb_height: (height+63)/64,
            w_in_b: 2 * ((width+7)>>3) ,	// MiCols, ((width+7)/8)<<3 >> MI_SIZE_LOG2
            h_in_b: 2 * ((height+7)>>3),	// MiRows, ((height+7)/8)<<3 >> MI_SIZE_LOG2
            number: 0,
            show_frame: true,
            error_resilient: true,
            intra_only: false,
            frame_type: FrameType::KEY,
            show_existing_frame: false,
            use_reduced_tx_set: true,
            reference_mode: ReferenceMode::SINGLE,
            use_prev_frame_mvs: false,
            min_partition_size: min_partition_size,
        }
    }
}

impl fmt::Display for FrameInvariants{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Frame {} - {}", self.number, self.frame_type)
    }
}

#[allow(dead_code,non_camel_case_types)]
#[derive(Debug,PartialEq,EnumIterator)]
pub enum FrameType {
    KEY,
    INTER,
    INTRA_ONLY,
    S,
}

//const REFERENCE_MODES: usize = 3;

#[allow(dead_code,non_camel_case_types)]
#[derive(Debug,PartialEq,EnumIterator)]
pub enum ReferenceMode {
  SINGLE = 0,
  COMPOUND = 1,
  SELECT = 2,
}

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
            &FrameType::KEY => write!(f, "Key frame"),
            &FrameType::INTER => write!(f, "Inter frame"),
            &FrameType::INTRA_ONLY => write!(f, "Intra only frame"),
            &FrameType::S => write!(f, "Switching frame"),
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
                f @ _ => Box::new(File::open(&f).unwrap()) as Box<Read>
            },
            output_file: match matches.value_of("OUTPUT").unwrap() {
                "-" => Box::new(std::io::stdout()) as Box<Write>,
                f @ _ => Box::new(File::create(&f).unwrap()) as Box<Write>
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

// TODO: possibly just use bitwriter instead of byteorder
pub fn write_ivf_header(output_file: &mut Write, width: usize, height: usize, num: usize, den: usize) {
    output_file.write(b"DKIF").unwrap();
    output_file.write_u16::<LittleEndian>(0).unwrap(); // version
    output_file.write_u16::<LittleEndian>(32).unwrap(); // header length
    output_file.write(b"AV01").unwrap();
    output_file.write_u16::<LittleEndian>(width as u16).unwrap();
    output_file.write_u16::<LittleEndian>(height as u16).unwrap();
    output_file.write_u32::<LittleEndian>(num as u32).unwrap();
    output_file.write_u32::<LittleEndian>(den as u32).unwrap();
    output_file.write_u32::<LittleEndian>(0).unwrap();
    output_file.write_u32::<LittleEndian>(0).unwrap();
}

pub fn write_ivf_frame(output_file: &mut Write, pts: u64, data: &[u8]) {
    output_file.write_u32::<LittleEndian>(data.len() as u32).unwrap();
    output_file.write_u64::<LittleEndian>(pts).unwrap();
    output_file.write(data).unwrap();
}

trait UncompressedHeader {
    fn write_frame_size(&mut self, fi: &FrameInvariants) -> Result<(), std::io::Error>;
    fn write_sequence_header(&mut self, fi: &FrameInvariants)
                                    -> Result<(), std::io::Error>;
    fn write_bitdepth_colorspace_sampling(&mut self) -> Result<(), std::io::Error>;
    fn write_frame_setup(&mut self) -> Result<(), std::io::Error>;
    fn write_loop_filter(&mut self) -> Result<(), std::io::Error>;
    fn write_cdef(&mut self) -> Result<(), std::io::Error>;
}

impl<'a> UncompressedHeader for BitWriter<'a, BE> {
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

fn write_uncompressed_header(packet: &mut Write, sequence: &Sequence,
                            fi: &FrameInvariants) -> Result<(), std::io::Error> {
    let mut uch = BitWriter::<BE>::new(packet);
    uch.write(2,2)?; // AOM_FRAME_MARKER, 0x2
    uch.write(2,sequence.profile)?; // profile 0
    if fi.show_existing_frame {
        uch.write_bit(true)?; // show_existing_frame=1
        uch.write(3,0)?; // show last frame
        uch.byte_align()?;
        return Ok(());
    }
    uch.write_bit(false)?; // show_existing_frame=0
    uch.write_bit(fi.frame_type == FrameType::INTER)?; // keyframe : 0, inter: 1
    uch.write_bit(fi.show_frame)?; // show frame

    if fi.frame_type == FrameType::KEY || fi.frame_type == FrameType::INTRA_ONLY {
        assert!(fi.intra_only == true);
    }
    if fi.frame_type != FrameType::KEY {
        if fi.show_frame { assert!(fi.intra_only == false); }
        else { uch.write_bit( fi.intra_only )?; };
    };
    uch.write_bit(fi.error_resilient)?; // error resilient

    if fi.frame_type == FrameType::KEY || fi.intra_only {
        uch.write_sequence_header(fi)?;
    }

    //uch.write(8+7,0)?; // frame id

    uch.write_bit(false)?; // no override frame size

    if fi.frame_type == FrameType::KEY {
        uch.write_bitdepth_colorspace_sampling()?;
        uch.write_frame_setup()?;
    } else { // Inter frame info goes here
        if fi.intra_only {
            uch.write_bitdepth_colorspace_sampling()?;
            uch.write(8,0)?; // refresh_frame_flags
            uch.write_frame_setup()?;
        } else {
            uch.write(8,0)?; // refresh_frame_flags
            // TODO: More Inter frame info goes here
            for _ in 0..7 {
                uch.write(3,0)?; // dummy ref_frame = 0 until real MC happens
            }
            uch.write_frame_setup()?;
            uch.write_bit(true)?; // allow_high_precision_mv
            uch.write_bit(false)?; // frame_interp_filter is NOT switchable
            uch.write(2,0)?;	// EIGHTTAP_REGULAR
            if !fi.intra_only && !fi.error_resilient {
                uch.write_bit(false)?; // do not use_ref_frame_mvs
            }
        }
    };


    uch.write(3,0x0)?; // frame context
    uch.write_loop_filter()?;
    uch.write(8,fi.qindex as u8)?; // qindex
    uch.write_bit(false)?; // y dc delta q
    uch.write_bit(false)?; // uv dc delta q
    uch.write_bit(false)?; // uv ac delta q
    uch.write_bit(false)?; // no qm
    uch.write_bit(false)?; // segmentation off
    uch.write_bit(false)?; // no delta q
    uch.write_cdef()?;
    uch.write(6,0)?; // no y, u or v loop restoration
    uch.write_bit(false)?; // tx mode select

    //fi.reference_mode = ReferenceMode::SINGLE;

    if fi.reference_mode != ReferenceMode::SINGLE {
        // setup_compound_reference_mode();
    }

    if !fi.intra_only {
        uch.write_bit(false)?; } // do not use inter_intra
    if !fi.intra_only && fi.reference_mode != ReferenceMode::SINGLE {
        uch.write_bit(false)?; } // do not allow_masked_compound

    uch.write_bit(fi.use_reduced_tx_set)?; // reduced tx

    if fi.intra_only == false {
        // write global motion info here
        for _ in LAST_FRAME..ALTREF_FRAME+1 {
            uch.write_bit(false)?; // TransformationType == IDENTITY
        }
    }

    uch.write_bit(true)?; // uniform tile spacing
    if fi.width > 64 {
        uch.write(1,0)?; // tile cols
    }
    if fi.height > 64 {
        uch.write(1,0)?; // tile rows
    }
    // if tile_cols * tile_rows > 1
    //uch.write_bit(true)?; // loop filter across tiles
    uch.write(2,3)?; // tile_size_bytes
    uch.byte_align()?;
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
    let stride = fs.input.planes[p].cfg.stride;
    let rec = &mut fs.rec.planes[p];
    let xdec = fs.input.planes[p].cfg.xdec;
    let ydec = fs.input.planes[p].cfg.ydec;

    mode.predict(&mut rec.mut_slice(po), tx_size);

    if skip { return; }

    let mut residual: [i16; 64*64] = unsafe { uninitialized() };
    let mut coeffs_storage: [i32; 64*64] = unsafe { uninitialized() };
    let mut rcoeffs: [i32; 64*64] = unsafe { uninitialized() };

    let coeffs = &mut coeffs_storage[..tx_size.width()*tx_size.height()];

    diff(&mut residual,
         &fs.input.planes[p].slice(po),
         &rec.slice(po),
         1<<tx_size_wide_log2[tx_size as usize],
         1<<tx_size_high_log2[tx_size as usize]);


    forward_transform(&residual, coeffs, 1<<tx_size_wide_log2[tx_size as usize], tx_size, tx_type);
    quantize_in_place(fi.qindex, coeffs, tx_size);

    cw.write_coeffs_lv_map(p, bo, &coeffs, tx_size, tx_type, plane_bsize, xdec, ydec,
                            fi.use_reduced_tx_set);

    //reconstruct
    dequantize(fi.qindex, &coeffs, &mut rcoeffs, tx_size);

    inverse_transform_add(&mut rcoeffs, &mut rec.mut_slice(po).as_mut_slice(), stride, tx_size, tx_type);
}

fn encode_block(fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter,
            mode: PredictionMode, bsize: BlockSize, bo: &BlockOffset) {
    let skip = false;
    let is_inter = mode >= PredictionMode::NEARESTMV;

    cw.bc.set_skip(bo, bsize, skip);
    cw.write_skip(bo, skip);

    if fi.frame_type == FrameType::INTER {
        cw.write_is_inter(bo, is_inter);
        if !is_inter {
            cw.write_intra_mode(bsize, mode);
        }
    } else {
        cw.write_intra_mode_kf(bo, mode);
    }

    cw.bc.set_mode(bo, bsize, mode);

    let xdec = fs.input.planes[1].cfg.xdec;
    let ydec = fs.input.planes[1].cfg.ydec;

    let uv_mode = mode;

    if mode.is_directional() && bsize >= BlockSize::BLOCK_8X8 {
        cw.write_angle_delta(0, mode);
    }

    if has_chroma(bo, bsize, xdec, ydec) {
        cw.write_intra_uv_mode(uv_mode, mode, bsize);
        if uv_mode.is_directional() && bsize >= BlockSize::BLOCK_8X8 {
            cw.write_angle_delta(0, uv_mode);
        }
    }

    // these rules follow TX_MODE_LARGEST
    let tx_type = TxType::DCT_DCT;
    let tx_size = match bsize {
        BlockSize::BLOCK_4X4 => TxSize::TX_4X4,
        BlockSize::BLOCK_8X8 => TxSize::TX_8X8,
        BlockSize::BLOCK_16X16 => TxSize::TX_16X16,
        _ => TxSize::TX_32X32
    };

    if skip {
        cw.bc.reset_skip_context(bo, bsize, xdec, ydec);
    }

    let bw = mi_size_wide[bsize as usize] as usize / tx_size.width_mi();
    let bh = mi_size_high[bsize as usize] as usize / tx_size.height_mi();

    // FIXME(you): consider factor out as write_tx_blocks()
    for p in 0..1 {
        for by in 0..bh {
            for bx in 0..bw {
                let tx_bo = BlockOffset{x: bo.x + bx*tx_size.width_mi(), y: bo.y + by*tx_size.height_mi()};
                let po = tx_bo.plane_offset(&fs.input.planes[p].cfg);
                encode_tx_block(fi, fs, cw, p, &tx_bo, mode, tx_size, tx_type, bsize, &po, skip);
            }
        }
    }

    // these are only valid for 4:2:0
    let uv_tx_size = match bsize {
        BlockSize::BLOCK_4X4 | BlockSize::BLOCK_8X8 => TxSize::TX_4X4,
        BlockSize::BLOCK_16X16 => TxSize::TX_8X8,
        BlockSize::BLOCK_32X32 => TxSize::TX_16X16,
        _ => TxSize::TX_32X32
    };

    let mut bw_uv = bw*tx_size.width_mi() >> xdec;
    let mut bh_uv = bh*tx_size.height_mi() >> ydec;

    if (bw_uv == 0 || bh_uv == 0) && has_chroma(bo, bsize, xdec, ydec) {
        bw_uv = 1;
        bh_uv = 1;
    }

    bw_uv /= uv_tx_size.width_mi();
    bh_uv /= uv_tx_size.height_mi();

    let plane_bsize = get_plane_block_size(bsize, xdec, ydec);

    if bw_uv > 0 && bh_uv > 0 {
        let uv_tx_type = uv_intra_mode_to_tx_type_context(uv_mode);
        let partition_x = (bo.x & LOCAL_BLOCK_MASK) >> xdec << MI_SIZE_LOG2;
        let partition_y = (bo.y & LOCAL_BLOCK_MASK) >> ydec << MI_SIZE_LOG2;

        for p in 1..3 {
            let sb_offset = bo.sb_offset().plane_offset(&fs.input.planes[p].cfg);

            for by in 0..bh_uv {
                for bx in 0..bw_uv {
                    let tx_bo =
                        BlockOffset{x: bo.x + (bx*uv_tx_size.width_mi() << xdec) - ((bw*tx_size.width_mi() == 1) as usize),
                                    y: bo.y + (by*uv_tx_size.height_mi() << ydec) - ((bh*tx_size.height_mi() == 1) as usize)};
                    let po = PlaneOffset {
                        x: sb_offset.x + partition_x + bx*uv_tx_size.width(),
                        y: sb_offset.y + partition_y + by*uv_tx_size.height()};

                    encode_tx_block(fi, fs, cw, p, &tx_bo, uv_mode, uv_tx_size, uv_tx_type, plane_bsize, &po, skip);
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

    let bs = mi_size_wide[bsize as usize];

    // Always split if the current partition is too large
    let must_split = bo.x + bs as usize > fi.w_in_b ||
        bo.y + bs as usize > fi.h_in_b ||
        bsize >= BlockSize::BLOCK_64X64;

    // must_split overrides the minimum partition size when applicable
    let can_split = bsize > fi.min_partition_size || must_split;

    let mut partition = PartitionType::PARTITION_NONE;
    let mut best_decision = RDOPartitionOutput {
        rd_cost: rd_cost,
        bo: bo.clone(),
        pred_mode: PredictionMode::DC_PRED
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
        let pred_mode = mode_decision.pred_mode;
        rd_cost = mode_decision.rd_cost;

        encode_block(fi, fs, cw, pred_mode, bsize, bo);

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
            let pred_mode = best_decision.pred_mode;
            encode_block(fi, fs, cw, pred_mode, bsize, bo);
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

    let bs = mi_size_wide[bsize as usize];

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

    assert!(mi_size_wide[bsize as usize] == mi_size_high[bsize as usize]);
    assert!(PartitionType::PARTITION_NONE <= partition &&
            partition < PartitionType::PARTITION_INVALID);

    let hbs = bs >> 1; // Half the block size in blocks
    let subsize = get_subsize(bsize, partition);

    if bsize >= BlockSize::BLOCK_8X8 {
        cw.write_partition(bo, partition, bsize);
    }

    match partition {
        PartitionType::PARTITION_NONE => {
            let pred_mode = if rdo_output.part_modes.len() > 0 {
                    // The optimal prediction mode is known from a previous iteration
                    rdo_output.part_modes[0].pred_mode
                } else {
                    // Make a prediction mode decision for blocks encoded with no rdo_partition_decision call (e.g. edges)
                    rdo_mode_decision(fi, fs, cw, bsize, bo).part_modes[0].pred_mode
                };

            // FIXME: every final block that has gone through the RDO decision process is encoded twice
            encode_block(fi, fs, cw, pred_mode, bsize, bo);
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

fn encode_frame(sequence: &Sequence, fi: &FrameInvariants, fs: &mut FrameState, last_rec: &Option<Frame>) -> Vec<u8> {
    let mut packet = Vec::new();
    write_uncompressed_header(&mut packet, sequence, fi).unwrap();
    if fi.show_existing_frame {
        match last_rec {
            &Some(ref rec) => for p in 0..3 {
                fs.rec.planes[p].data.copy_from_slice(rec.planes[p].data.as_slice());
            },
            &None => (),
        }
    } else {
        let tile = encode_tile(fi, fs);
        packet.write(&tile).unwrap();
    }
    packet
}

/// Encode and write a frame.
pub fn process_frame(sequence: &Sequence, fi: &FrameInvariants,
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

            let packet = encode_frame(&sequence, &fi, &mut fs, &last_rec);
            write_ivf_frame(output_file, fi.number, packet.as_ref());
            match y4m_enc {
                Some(mut y4m_enc) => {
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
                None => {}
            }
            *last_rec = Some(fs.rec);
            true
        },
        _ => false
    }
}
