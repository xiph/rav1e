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
pub mod context;
pub mod transform;
pub mod quantize;
pub mod predict;
pub mod rdo;

use context::*;
use partition::*;
use transform::*;
use quantize::*;
use predict::*;
use rdo::*;

pub struct Plane {
    pub data: Vec<u16>,
    pub cfg: PlaneConfig,
}

#[allow(dead_code)]
impl Plane {
    pub fn new(width: usize, height: usize, xdec: usize, ydec: usize) -> Plane {
        Plane {
            data: vec![128; width*height],
            cfg: PlaneConfig {
                stride: width,
                xdec,
                ydec,
            },
        }
    }
    pub fn slice<'a>(&'a mut self, x: usize, y: usize) -> PlaneMutSlice<'a> {
        PlaneMutSlice {
            p: self,
            x: x,
            y: y
        }
    }
    pub fn p(&self, x: usize, y: usize) -> u16 {
        self.data[y*self.cfg.stride+x]
    }
}

#[allow(dead_code)]
pub struct PlaneMutSlice<'a> {
    pub p: &'a mut Plane,
    pub x: usize,
    pub y: usize
}

#[allow(dead_code)]
impl<'a> PlaneMutSlice<'a> {
    pub fn as_slice(&mut self) -> &mut [u16] {
        let stride = self.p.cfg.stride;
        &mut self.p.data[self.y*stride+self.x..]
    }
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
            input: Frame::new(fi.sb_width*64, fi.sb_height*64),
            rec: Frame::new(fi.sb_width*64, fi.sb_height*64),
        }
    }
}

#[derive(Debug, EnumIterator)]
pub enum FrameType {
    Intra,
    Inter
}

#[allow(dead_code)]
pub struct FrameInvariants {
    pub qindex: usize,
    pub width: usize,
    pub height: usize,
    pub sb_width: usize,
    pub sb_height: usize,
    pub frame_type: FrameType,
}

impl FrameInvariants {
    pub fn new(width: usize, height: usize) -> FrameInvariants {
        FrameInvariants {
            qindex: 100,
            width: width,
            height: height,
            sb_width: (width+63)/64,
            sb_height: (height+63)/64,
            frame_type: FrameType::Intra,
        }
    }
}

pub struct EncoderConfig {
    pub input_file: Box<Read>,
    pub output_file: Box<Write>,
    pub rec_file: Option<Box<Write>>,
    pub limit: u64
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
            limit: matches.value_of("LIMIT").unwrap().parse().unwrap()
        }
    }
}

// TODO: possibly just use bitwriter instead of byteorder
pub fn write_ivf_header(output_file: &mut Write, width: usize, height: usize) {
    output_file.write(b"DKIF").unwrap();
    output_file.write_u16::<LittleEndian>(0).unwrap(); // version
    output_file.write_u16::<LittleEndian>(32).unwrap(); // header length
    output_file.write(b"AV01").unwrap();
    output_file.write_u16::<LittleEndian>(width as u16).unwrap();
    output_file.write_u16::<LittleEndian>(height as u16).unwrap();
    output_file.write_u32::<LittleEndian>(60).unwrap();
    output_file.write_u32::<LittleEndian>(0).unwrap();
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
    uch.write_bit(false)?; // show_existing_frame=0
    uch.write_bit(false)?; // keyframe
    uch.write_bit(true)?; // show frame
    uch.write_bit(true)?; // error resilient
    uch.write(1,0)?; // don't use frame ids
    //uch.write(8+7,0)?; // frame id
    uch.write(3,0)?; // colorspace
    uch.write(1,0)?; // color range
    uch.write(16,(fi.sb_width*64-1) as u16)?; // width
    uch.write(16,(fi.sb_height*64-1) as u16)?; // height
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
    uch.write(1,0)?; // cdef dering damping
    uch.write(2,0)?; // cdef clpf damping
    uch.write(2,0)?; // cdef bits
    for _ in 0..1 {
        uch.write(7,0)?; // cdef y strength
        uch.write(7,0)?; // cdef uv strength
    }
    uch.write_bit(false)?; // no delta q
    uch.write_bit(false)?; // tx mode select
    uch.write(2,0)?; // only 4x4 transforms
    //uch.write_bit(false)?; // use hybrid pred
    //uch.write_bit(false)?; // use compound pred
    uch.write_bit(true)?; // reduced tx
    if fi.sb_width*64-1 > 256 {
        uch.write(1,0)?; // tile cols
    }
    uch.write(1,0)?; // tile rows
    uch.write_bit(true)?; // loop filter across tiles
    uch.write(2,0)?; // tile_size_bytes
    uch.byte_align()?;
    Ok(())
}

fn write_b(cw: &mut ContextWriter, fi: &FrameInvariants, fs: &mut FrameState, p: usize, bo: &BlockOffset, mode: PredictionMode, tx_type: TxType) {
    let stride = fs.input.planes[p].cfg.stride;
    let mut above = [0 as u16; 4];
    let mut left = [0 as u16; 4];
    let po = bo.plane_offset(&fs.input.planes[p].cfg);

    if bo.y == 0 {
        above = [127; 4];
    } else {
        for i in 0..4 {
            above[i] = fs.rec.planes[p].data[(po.y - 1)*stride + po.x + i];
        }
    }
    if bo.x == 0 {
        left = [129; 4];
    } else {
        for i in 0..4 {
            left[i] = fs.rec.planes[p].data[(po.y + i)*stride + po.x - 1];
        }
    }
    match mode {
        PredictionMode::DC_PRED =>
            match (bo.x, bo.y) {
                (0, 0) =>
                    pred_dc_128(&mut fs.rec.planes[p].data[po.y*stride + po.x..], stride),
                (_, 0) =>
                    pred_dc_left_4x4(&mut fs.rec.planes[p].data[po.y*stride + po.x..], stride, &above, &left),
                (0, _) =>
                    pred_dc_top_4x4(&mut fs.rec.planes[p].data[po.y*stride + po.x..], stride, &above, &left),
                _ =>
                    pred_dc(&mut fs.rec.planes[p].data[po.y*stride + po.x..], stride, &above, &left),
            }
        PredictionMode::H_PRED =>
            pred_h(&mut fs.rec.planes[p].data[po.y*stride + po.x..], stride, &left, 4),
        PredictionMode::V_PRED =>
            pred_v(&mut fs.rec.planes[p].data[po.y*stride + po.x..], stride, &above, 4),
        _ =>
            panic!("Unimplemented prediction mode: {:?}", mode),
    }
    let mut coeffs = [0 as i32; 16];
    let mut residual = [0 as i16; 16];
    for j in 0..4 {
        for i in 0..4 {
            residual[j*4+i] = fs.input.planes[p].data[(po.y + j)*stride + po.x + i] as i16
                - fs.rec.planes[p].data[(po.y + j)*stride + po.x + i] as i16;
        }
    }
    fht4x4(&residual, &mut coeffs, 4, tx_type);
    quantize_in_place(fi.qindex, &mut coeffs);
    cw.write_coeffs(p, bo, &coeffs, TxSize::TX_4X4, tx_type);

    //reconstruct
    let mut rcoeffs = [0 as i32; 16];
    dequantize(fi.qindex, &coeffs, &mut rcoeffs);
    iht4x4_add(&mut rcoeffs, &mut fs.rec.planes[p].data[po.y*stride + po.x..], stride, tx_type);
}

fn write_sb(cw: &mut ContextWriter, fi: &FrameInvariants, fs: &mut FrameState, sbo: &SuperBlockOffset, mode: PredictionMode) {
    cw.write_partition(PartitionType::PARTITION_NONE);
    // The partition offset is represented using a BlockOffset
    let po = sbo.block_offset(0, 0);
    cw.write_skip(&po, false);
    cw.write_intra_mode_kf(&po, mode);
    let uv_mode = mode;
    cw.write_intra_uv_mode(uv_mode, mode);
    let tx_type = TxType::DCT_DCT;
    cw.write_tx_type(tx_type, mode);
    for p in 0..1 {
        for by in 0..16 {
            for bx in 0..16 {
                let bo = sbo.block_offset(bx, by);
                cw.bc.at(&bo).mode = mode;
                write_b(cw, fi, fs, p, &bo, mode, tx_type);
            }
        }
    }
    let uv_tx_type = exported_intra_mode_to_tx_type_context[uv_mode as usize];
    for p in 1..3 {
        for by in 0..8 {
            for bx in 0..8 {
                let bo = sbo.block_offset(bx, by);
                write_b(cw, fi, fs, p, &bo, uv_mode, uv_tx_type);
            }
        }
    }
}

fn encode_tile(fi: &FrameInvariants, fs: &mut FrameState) -> Vec<u8> {
    let w = ec::Writer::new();
    let fc = CDFContext::new();
    let bc = BlockContext::new(fi.sb_width*16, fi.sb_height*16);
    let mut cw = ContextWriter {
        w: w,
        fc: fc,
        bc: bc,
    };

    let stride = fs.input.planes[0].cfg.stride;
    let q = dc_q(fi.qindex) as f64;

    // Lambda formula from doc/theoretical_results.lyx in the daala repo
    let lambda = q*q*2.0_f64.log2()/6.0;

    for sby in 0..fi.sb_height {
        for p in 0..3 {
            cw.bc.reset_left_coeff_context(p);
        }
        for sbx in 0..fi.sb_width {
            let sbo = SuperBlockOffset { x: sbx, y: sby };
            let po = sbo.plane_offset(&fs.input.planes[0].cfg);
            let tell = cw.w.tell_frac();

            let mut best_mode = PredictionMode::DC_PRED;
            let mut best_rd = std::f64::MAX;

            for &mode in RAV1E_INTRA_MODES {
                let checkpoint = cw.checkpoint();

                write_sb(&mut cw, fi, fs, &sbo, mode);
                let d = sse_64x64(&fs.input.planes[0].data, &fs.rec.planes[0].data, po.x, po.y, stride);
                let r = ((cw.w.tell_frac() - tell) as f64)/8.0;

                let rd = (d as f64) + lambda*r;
                if rd < best_rd {
                    best_rd = rd;
                    best_mode = mode;
                }

                cw.rollback(checkpoint.clone());
            }

            write_sb(&mut cw, fi, fs, &sbo, best_mode);
        }
    }
    let mut h = cw.w.done();
    h.push(0); // superframe anti emulation
    h
}

fn encode_frame(sequence: &Sequence, fi: &FrameInvariants, fs: &mut FrameState) -> Vec<u8> {
    let mut packet = Vec::new();
    write_uncompressed_header(&mut packet, sequence, fi).unwrap();
    let tile = encode_tile(fi, fs);
    packet.write(&tile).unwrap();
    packet
}

/// Encode and write a frame.
pub fn process_frame(frame_number: u64, sequence: &Sequence, fi: &FrameInvariants,
                     output_file: &mut Write,
                     y4m_dec: &mut y4m::Decoder<Box<Read>>,
                     y4m_enc: Option<&mut y4m::Encoder<Box<Write>>>) -> bool {
    let width = fi.width;
    let height = fi.height;
    match y4m_dec.read_frame() {
        Ok(y4m_frame) => {
            let y4m_y = y4m_frame.get_y_plane();
            let y4m_u = y4m_frame.get_u_plane();
            let y4m_v = y4m_frame.get_v_plane();
            eprintln!("Frame {}", frame_number);
            let mut fs = FrameState::new(&fi);
            for y in 0..height {
                for x in 0..width {
                    let stride = fs.input.planes[0].cfg.stride;
                    fs.input.planes[0].data[y*stride+x] = y4m_y[y*width+x] as u16;
                }
            }
            for y in 0..height/2 {
                for x in 0..width/2 {
                    let stride = fs.input.planes[1].cfg.stride;
                    fs.input.planes[1].data[y*stride+x] = y4m_u[y*width/2+x] as u16;
                }
            }
            for y in 0..height/2 {
                for x in 0..width/2 {
                    let stride = fs.input.planes[2].cfg.stride;
                    fs.input.planes[2].data[y*stride+x] = y4m_v[y*width/2+x] as u16;
                }
            }
            let packet = encode_frame(&sequence, &fi, &mut fs);
            write_ivf_frame(output_file, frame_number, packet.as_ref());
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
            true
        },
        _ => false
    }
}
