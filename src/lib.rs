#![allow(safe_extern_statics)]

extern crate bitstream_io;
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
use predict::*;
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
    pub ftype: FrameType,
    pub show_existing_frame: bool,
    pub min_partition_size: BlockSize,
}

impl FrameInvariants {
    pub fn new(width: usize, height: usize, qindex: usize, speed: usize) -> FrameInvariants {
        // Speed level decides the minimum partition size, i.e. higher speed --> larger min partition size,
        // with exception that SBs on right or bottom frame borders split down to BLOCK_4X4.
        let min_partition_size = if speed <= 0 { BlockSize::BLOCK_4X4 } 
                                 else if speed <= 1 { BlockSize::BLOCK_8X8 }
                                 else if speed <= 2 { BlockSize::BLOCK_16X16 }
                                 else if speed <= 3 { BlockSize::BLOCK_32X32 }
                                 else { BlockSize::BLOCK_64X64 };
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
            ftype: FrameType::KEY,
            show_existing_frame: false,
            min_partition_size: min_partition_size,
        }
    }
}

impl fmt::Display for FrameInvariants{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Frame {} - {}", self.number, self.ftype)
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
                .default_value("0"))
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

fn write_uncompressed_header(packet: &mut Write, sequence: &Sequence, fi: &FrameInvariants) -> Result<(), std::io::Error> {
    let mut uch = BitWriter::<BE>::new(packet);
    uch.write(2,2)?; // frame type
    uch.write(2,sequence.profile)?; // profile 0
    if fi.show_existing_frame {
        uch.write_bit(true)?; // show_existing_frame=1
        uch.write(3,0)?; // show last frame
        uch.byte_align()?;
        return Ok(());
    }
    uch.write_bit(false)?; // show_existing_frame=0
    uch.write_bit(false)?; // keyframe
    uch.write_bit(true)?; // show frame
    uch.write_bit(true)?; // error resilient
    uch.write(1,0)?; // don't use frame ids
    //uch.write(8+7,0)?; // frame id
    uch.write(3,0)?; // colorspace
    uch.write(1,0)?; // color range
    uch.write(16,(fi.width-1) as u16)?; // width
    uch.write(16,(fi.height-1) as u16)?; // height
    uch.write_bit(false)?; // scaling active
    uch.write_bit(false)?; // screen content tools
    uch.write(3,0x0)?; // frame context
    uch.write(6,0)?; // loop filter level
    uch.write(3,0)?; // loop filter sharpness
    uch.write_bit(false)?; // loop filter deltas enabled
    uch.write(8,fi.qindex as u8)?; // qindex
    uch.write_bit(false)?; // y dc delta q
    uch.write_bit(false)?; // uv dc delta q
    uch.write_bit(false)?; // uv ac delta q
    //uch.write_bit(false)?; // using qmatrix
    uch.write_bit(false)?; // segmentation off
    uch.write_bit(false)?; // no delta q
    uch.write(2,0)?; // cdef clpf damping
    uch.write(2,0)?; // cdef bits
    for _ in 0..1 {
        uch.write(6,0)?; // cdef y strength
        uch.write(6,0)?; // cdef uv strength
    }
    uch.write(6,0)?; // no y, u or v loop restoration
    uch.write_bit(false)?; // tx mode select
    uch.write(2,0)?; // only 4x4 transforms
    //uch.write_bit(false)?; // use hybrid pred
    //uch.write_bit(false)?; // use compound pred
    uch.write_bit(true)?; // reduced tx
    if fi.width > 256 {
        uch.write(1,0)?; // tile cols
    }
    uch.write(1,0)?; // tile rows
    // if tile_cols * tile_rows > 1
    //uch.write_bit(true)?; // loop filter across tiles
    uch.write(2,3)?; // tile_size_bytes
    uch.byte_align()?;
    Ok(())
}

/// Write into `dst` the difference between the 4x4 blocks at `src1` and `src2`
fn diff(dst: &mut [i16; 16], src1: &PlaneSlice, src2: &PlaneSlice, width: usize, height: usize) {
    for j in 0..height {
        for i in 0..width {
            dst[j*width + i] = (src1.p(i, j) as i16) - (src2.p(i, j) as i16);
        }
    }
}

fn has_chroma(bo: &BlockOffset, bsize: BlockSize,
                       subsampling_x: usize, subsampling_y: usize) -> bool {
    let bw = mi_size_wide[bsize as usize] as u8;
    let bh = mi_size_high[bsize as usize] as u8;
    let ref_pos = ((bo.x & 0x01) == 1 || (bw & 0x01) == 0 || subsampling_x == 0) &&
                  ((bo.y & 0x01) == 1 || (bh & 0x01) == 0 || subsampling_y == 0);

    ref_pos
}

// For a transform block,
// predict, transform, quantize, write coefficients to a bitstream,
// dequantize, inverse-transform.
pub fn encode_tx_block(fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter,
                  p: usize, bo: &BlockOffset, mode: PredictionMode, tx_type: TxType,
                  po: &PlaneOffset, skip: bool) {
    let stride = fs.input.planes[p].cfg.stride;
    let rec = &mut fs.rec.planes[p];
    let tx_size = TxSize::TX_4X4;

    mode.predict(&mut rec.mut_slice(po), tx_size);

    if skip { return; }

    let mut residual = [0 as i16; 16];

    diff(&mut residual,
         &fs.input.planes[p].slice(po),
         &rec.slice(po),
         1<<tx_size_wide_log2[tx_size as usize],
         1<<tx_size_high_log2[tx_size as usize]);

    let mut coeffs = [0 as i32; 16];
    forward_transform(&residual, &mut coeffs, 4, tx_size, tx_type);
    quantize_in_place(fi.qindex, &mut coeffs);
    cw.write_coeffs(p, bo, &coeffs, TxSize::TX_4X4, tx_type);

    //reconstruct
    let mut rcoeffs = [0 as i32; 16];
    dequantize(fi.qindex, &coeffs, &mut rcoeffs);

    inverse_transform_add(&mut rcoeffs, &mut rec.mut_slice(po).as_mut_slice(), stride, tx_size, tx_type);
}

fn encode_block(fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter,
            mode: PredictionMode, bsize: BlockSize, bo: &BlockOffset) {
    let skip = false;

    cw.bc.set_skip(bo, bsize, skip);
    cw.write_skip(bo, skip);
    cw.bc.set_mode(bo, bsize, mode);
    cw.write_intra_mode_kf(bo, mode);

    let xdec = fs.input.planes[1].cfg.xdec;
    let ydec = fs.input.planes[1].cfg.ydec;

    let uv_mode = mode;

    if has_chroma(bo, bsize, xdec, ydec) {
        cw.write_intra_uv_mode(uv_mode, mode);
    }

    let tx_type = TxType::DCT_DCT;

    if skip == false { cw.write_tx_type(tx_type, mode); }

    let bw = mi_size_wide[bsize as usize] as usize;
    let bh = mi_size_high[bsize as usize] as usize;

    // FIXME(you): Loop for TX blocks. For now, fixed as a 4x4 TX only,
    // but consider factor out as write_tx_blocks()
    for p in 0..1 {
        for by in 0..bh {
            for bx in 0..bw {
                let tx_bo = BlockOffset{x: bo.x + bx, y: bo.y + by};
                let po = tx_bo.plane_offset(&fs.input.planes[p].cfg);
                encode_tx_block(fi, fs, cw, p, &tx_bo, mode, tx_type, &po, skip);
            }
        }
    }

    let mut bw_uv = bw >> xdec;
    let mut bh_uv = bh >> ydec;

    if (bw_uv == 0 || bh_uv == 0) && has_chroma(bo, bsize, xdec, ydec) {
        bw_uv = 1;
        bh_uv = 1;
    }

    if bw_uv > 0 && bh_uv > 0 {
        let uv_tx_type = uv_intra_mode_to_tx_type_context(uv_mode);
        let partition_x = (bo.x & LOCAL_BLOCK_MASK) >> xdec << MI_SIZE_LOG2;
        let partition_y = (bo.y & LOCAL_BLOCK_MASK) >> ydec << MI_SIZE_LOG2;

        for p in 1..3 {
            let sb_offset = bo.sb_offset().plane_offset(&fs.input.planes[p].cfg);

            for by in 0..bh_uv {
                for bx in 0..bw_uv {
                    let tx_bo =
                        BlockOffset{x: bo.x + (bx << xdec) - ((bw == 1) as usize),
                                    y: bo.y + (by << ydec) - ((bh == 1) as usize)};
                    let po = PlaneOffset {
                        x: sb_offset.x + partition_x + (bx << MI_SIZE_LOG2),
                        y: sb_offset.y + partition_y + (by << MI_SIZE_LOG2) };

                    encode_tx_block(fi, fs, cw, p, &tx_bo, uv_mode, uv_tx_type, &po, skip);
                }
            }
        }
    }
}

// RDO-based mode decision
fn rdo_mode_decision(fi: &FrameInvariants, fs: &mut FrameState,
                  cw: &mut ContextWriter,
                  bsize: BlockSize, bo: &BlockOffset) -> RDOOutput {
    let q = dc_q(fi.qindex) as f64;
    let q0 = q / 8.0_f64;	// Convert q into Q0 precision, given thatn libaom quantizers are Q3.

    // Lambda formula from doc/theoretical_results.lyx in the daala repo
    let lambda = q0*q0*2.0_f64.log2()/6.0;	// Use Q0 quantizer since lambda will be applied to Q0 pixel domain

    let mut best_mode = PredictionMode::DC_PRED;
    let mut best_rd = std::f64::MAX;
    let tell = cw.w.tell_frac();
    let w = block_size_wide[bsize as usize];
    let h = block_size_high[bsize as usize];

    for &mode in RAV1E_INTRA_MODES {
        let checkpoint = cw.checkpoint();

        encode_block(fi, fs, cw, mode, bsize, bo);
        let po = bo.plane_offset(&fs.input.planes[0].cfg);
        let d = sse_wxh(&fs.input.planes[0].slice(&po), &fs.rec.planes[0].slice(&po),
                        w as usize, h as usize);
        let r = ((cw.w.tell_frac() - tell) as f64)/8.0;

        let rd = (d as f64) + lambda*r;
        if rd < best_rd {
            best_rd = rd;
            best_mode = mode;
        }

        cw.rollback(checkpoint.clone());
    }

    assert!(best_rd as i64 >= 0);

    let rdo_output = RDOOutput { rd_cost: best_rd as u64,
                                pred_mode: best_mode};
    rdo_output
}

fn encode_partition(fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter,
            bsize: BlockSize, bo: &BlockOffset) {

    if bo.x >= cw.bc.cols || bo.y >= cw.bc.rows {
        return;
    }

    let is_sb_on_frame_border = (fi.sb_width-1) * 16 <= bo.x || (fi.sb_height-1) * 16 <= bo.y;

    // TODO(anyone): Until we have RDO-based block size decision,
    // split all the way down to 4x4 blocks, then do rdo_mode_decision() for each 4x4 block.
    let mut partition = PartitionType::PARTITION_NONE;

    if is_sb_on_frame_border {
        // SBs on right or bottom frame borders split down to BLOCK_4X4.
        if bsize > BlockSize::BLOCK_4X4 { partition = PartitionType::PARTITION_SPLIT; }
    } else {
        if bsize > fi.min_partition_size { partition = PartitionType::PARTITION_SPLIT; }
    };

    assert!(mi_size_wide[bsize as usize] == mi_size_high[bsize as usize]);

    assert!(PartitionType::PARTITION_NONE <= partition &&
            partition < PartitionType::PARTITION_INVALID);

    let bs = mi_size_wide[bsize as usize];
    let hbs = bs >> 1; // Half the block size in blocks
    let subsize = get_subsize(bsize, partition);

    if bsize >= BlockSize::BLOCK_8X8 {
        cw.write_partition(bo, partition, bsize);
    }

    match partition {
        PartitionType::PARTITION_NONE => {
            // TODO(anyone): Until we have RDO-based block size decision,
            // call rdo_mode_decision() for a partition.
            let rdo_none = rdo_mode_decision(fi, fs, cw, bsize, bo);
            // FIXME(anyone): Instead of calling set_mode() in encode_block() for each 4x4tx position,
            // it would be better to call set_mode() for each MI block position here.
            cw.bc.set_mode(bo, bsize, rdo_none.pred_mode);

            encode_block(fi, fs, cw, rdo_none.pred_mode, bsize, bo);
        },
        PartitionType::PARTITION_SPLIT => {
            assert!(subsize != BlockSize::BLOCK_INVALID);
            encode_partition(fi, fs, cw, subsize, bo);
            encode_partition(fi, fs, cw, subsize, &BlockOffset{x: bo.x + hbs as usize, y: bo.y});
            encode_partition(fi, fs, cw, subsize, &BlockOffset{x: bo.x, y: bo.y + hbs as usize});
            encode_partition(fi, fs, cw, subsize, &BlockOffset{x: bo.x + hbs as usize, y: bo.y + hbs as usize});
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
    let fc = CDFContext::new();
    let bc = BlockContext::new(fi.w_in_b, fi.h_in_b);
    let mut cw = ContextWriter {
        w: w,
        fc: fc,
        bc: bc,
    };

    for sby in 0..fi.sb_height {
        cw.bc.reset_left_contexts();

        for sbx in 0..fi.sb_width {
            let sbo = SuperBlockOffset { x: sbx, y: sby };
            let bo = sbo.block_offset(0, 0);

            // TODO(anyone):
            // RDO-based block size decision for each SuperBlock can be done here.

            // Encode SuperBlock
            encode_partition(fi, fs, &mut cw, BlockSize::BLOCK_64X64, &bo);
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
    match y4m_dec.read_frame() {
        Ok(y4m_frame) => {
            let y4m_y = y4m_frame.get_y_plane();
            let y4m_u = y4m_frame.get_u_plane();
            let y4m_v = y4m_frame.get_v_plane();
            eprintln!("{}", fi);
            let mut fs = FrameState::new(&fi);
            fs.input.planes[0].copy_from_raw_u8(&y4m_y, width);
            fs.input.planes[1].copy_from_raw_u8(&y4m_u, width/2);
            fs.input.planes[2].copy_from_raw_u8(&y4m_v, width/2);
            let packet = encode_frame(&sequence, &fi, &mut fs, &last_rec);
            write_ivf_frame(output_file, fi.number, packet.as_ref());
            match y4m_enc {
                Some(mut y4m_enc) => {
                    let mut rec_y = vec![128 as u8; width*height];
                    let mut rec_u = vec![128 as u8; width*height/4];
                    let mut rec_v = vec![128 as u8; width*height/4];
                    for y in 0..height {
                        for x in 0..width {
                            let stride = fs.rec.planes[0].cfg.stride;
                            rec_y[y*width+x] = fs.rec.planes[0].data[y*stride+x] as u8;
                        }
                    }
                    for y in 0..height/2 {
                        for x in 0..width/2 {
                            let stride = fs.rec.planes[1].cfg.stride;
                            rec_u[y*width/2+x] = fs.rec.planes[1].data[y*stride+x] as u8;
                        }
                    }
                    for y in 0..height/2 {
                        for x in 0..width/2 {
                            let stride = fs.rec.planes[2].cfg.stride;
                            rec_v[y*width/2+x] = fs.rec.planes[2].data[y*stride+x] as u8;
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
