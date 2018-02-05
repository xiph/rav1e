extern crate bitstream_io;
extern crate byteorder;
extern crate clap;
extern crate libc;
extern crate rand;
extern crate y4m;

use std::fs::File;
use std::io::prelude::*;
use bitstream_io::{BE, BitWriter};
use byteorder::*;
use clap::App;

mod ec;
mod partition;
mod context;
mod transform;
mod quantize;
mod predict;

use context::*;
use partition::*;
use transform::*;
use quantize::*;
use predict::*;

pub struct Plane {
    pub data: Vec<u16>,
    pub stride: usize,
    pub xdec: usize,
    pub ydec: usize,
}

#[allow(dead_code)]
impl Plane {
    pub fn new(width: usize, height: usize, xdec: usize, ydec: usize) -> Plane {
        Plane {
            data: vec![128; width*height],
            stride: width,
            xdec: xdec,
            ydec: ydec,
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
        self.data[y*self.stride+x]
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
        let stride = self.p.stride;
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

#[allow(dead_code)]
pub struct FrameInvariants {
    pub qindex: usize,
    pub width: usize,
    pub height: usize,
    pub sb_width: usize,
    pub sb_height: usize
}

impl FrameInvariants {
    pub fn new(width: usize, height: usize) -> FrameInvariants {
        FrameInvariants {
            qindex: 100,
            width: width,
            height: height,
            sb_width: (width+63)/64,
            sb_height: (height+63)/64,
        }
    }
}

pub struct EncoderFiles {
    pub input_file: Box<Read>,
    pub output_file: Box<Write>,
    pub rec_file: File,
}

impl EncoderFiles {
    pub fn from_cli() -> EncoderFiles {
        let matches = App::new("rav1e")
            .version("0.1.0")
            .about("AV1 video encoder")
            .args_from_usage(
                "<INPUT.y4m>              'Uncompressed YUV4MPEG2 video input'
                 <OUTPUT.ivf>             'Compressed AV1 in IVF video output'")
            .get_matches();
        let input = matches.value_of("INPUT.y4m").unwrap();
        let output = matches.value_of("OUTPUT.ivf").unwrap();

        let input_file = if input == "-" {
            Box::new(std::io::stdin()) as Box<Read>
        } else {
            Box::new(File::open(&input).unwrap()) as Box<Read>
        };
        let output_file = Box::new(File::create(&output).unwrap()) as Box<Write>;
        let rec_file = File::create("rec.y4m").unwrap();

        EncoderFiles {
            input_file: input_file,
            output_file: output_file,
            rec_file: rec_file
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

fn write_uncompressed_header(packet: &mut Write, sequence: &Sequence, fi: &FrameInvariants, compressed_len: u32) -> Result<(), std::io::Error> {
    let mut uch = BitWriter::<BE>::new(packet);
    uch.write(2,2)?; // frame type
    uch.write(2,sequence.profile)?; // profile 0
    uch.write_bit(false)?; // show_existing_frame=0
    uch.write_bit(false)?; // keyframe
    uch.write_bit(true)?; // show frame
    uch.write_bit(true)?; // error resilient
    uch.write(8+7,0)?; // frame id
    uch.write(8,0x49)?; // sync codes
    uch.write(8,0x83)?;
    uch.write(8,0x43)?;
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
    //println!("compressed header length: {}", compressed_len);
    uch.write(16,compressed_len)?; // compressed header length
    uch.byte_align()?;
    Ok(())
}

fn get_compressed_header() -> Vec<u8> {
    let mut h = Vec::new();
    let mut w = ec::Writer::new();
    // zero length compressed header is invalid, write 1 bit of garbage
    w.bool(false,1024);
    h.write(&w.done()).unwrap();
    h
}

fn write_b(cw: &mut ContextWriter, fi: &FrameInvariants, fs: &mut FrameState, p: usize, sbx: usize, sby: usize, bx: usize, by: usize) {
    let stride = fs.input.planes[p].stride;
    let xdec = fs.input.planes[p].xdec;
    let ydec = fs.input.planes[p].ydec;
    let mut above = [0 as u16; 4];
    let mut left = [0 as u16; 4];
    let x = (sbx*64>>xdec) + bx*4;
    let y = (sby*64>>ydec) + by*4;
    match (sby, by) {
        (0,0) => above = [127; 4],
        _ => {
            for i in 0..4 {
                above[i] = fs.rec.planes[p].data[(y-1)*stride+x+i];
            }
        }
    }
    match (sbx, bx) {
        (0,0) => left = [129; 4],
        _ => {
            for i in 0..4 {
                left[i] = fs.rec.planes[p].data[(y+i)*stride+x-1];
            }
        }
    }
    match (sbx, bx, sby, by) {
        (0,0,0,0) => pred_dc_128(&mut fs.rec.planes[p].data[y*stride+x..], stride),
        (_,_,0,0) => pred_dc_left_4x4(&mut fs.rec.planes[p].data[y*stride+x..], stride, &above, &left),
        (0,0,_,_) => pred_dc_top_4x4(&mut fs.rec.planes[p].data[y*stride+x..], stride, &above, &left),
        _ => pred_dc_4x4(&mut fs.rec.planes[p].data[y*stride+x..], stride, &above, &left),
    }
    let mut coeffs = [0 as i32; 16];
    let mut residual = [0 as i16; 16];
    for j in 0..4 {
        for i in 0..4 {
            residual[j*4+i] = fs.input.planes[p].data[(y+j)*stride+x+i] as i16 - fs.rec.planes[p].data[(j+y)*stride+x+i] as i16;
        }
    }
    fdct4x4(&residual, &mut coeffs, 4);
    quantize_in_place(fi.qindex, &mut coeffs);
    cw.mc.set_loc(sbx*16+bx, sby*16+by);
    cw.write_coeffs(p, &coeffs);
    //reconstruct
    let mut rcoeffs = [0 as i32; 16];
    dequantize(fi.qindex, &coeffs, &mut rcoeffs);
    idct4x4_add(&mut rcoeffs, &mut fs.rec.planes[p].data[y*stride+x..], stride);
}

fn write_sb(cw: &mut ContextWriter, fi: &FrameInvariants, fs: &mut FrameState, sbx: usize, sby: usize) {
    cw.write_partition(PartitionType::PARTITION_NONE);
    cw.write_skip(false);
    cw.write_intra_mode(PredictionMode::DC_PRED);
    cw.write_intra_uv_mode(PredictionMode::DC_PRED);
    cw.write_tx_type(TxType::DCT_DCT);
    for p in 0..1 {
        for by in 0..16 {
            for bx in 0..16 {
                write_b(cw, fi, fs, p, sbx, sby, bx, by);
            }
        }
    }
    for p in 1..3 {
        for by in 0..8 {
            for bx in 0..8 {
                write_b(cw, fi, fs, p, sbx, sby, bx, by);
            }
        }
    }
}

fn encode_tile(fi: &FrameInvariants, fs: &mut FrameState) -> Vec<u8> {
    let w = ec::Writer::new();
    let fc = CDFContext::new();
    let mc = MIContext::new(fi.sb_width*16, fi.sb_height*16);
    let mut cw = ContextWriter {
        w: w,
        fc: fc,
        mc: mc,
    };
    for sby in 0..fi.sb_height {
        cw.reset_left_coeff_context(0);
        for sbx in 0..fi.sb_width {
            write_sb(&mut cw, fi, fs, sbx, sby);
        }
    }
    let mut h = cw.w.done();
    h.push(0); // superframe anti emulation
    h
}

fn encode_frame(sequence: &Sequence, fi: &FrameInvariants, fs: &mut FrameState) -> Vec<u8> {
    let mut packet = Vec::new();
    let compressed_header = get_compressed_header();
    write_uncompressed_header(&mut packet, sequence, fi, compressed_header.len() as u32).unwrap();
    packet.write(&compressed_header).unwrap();
    let tile = encode_tile(fi, fs);
    packet.write(&tile).unwrap();
    packet
}

/// Encode and write a frame.
pub fn process_frame(frame_number: u64, sequence: &Sequence, fi: &FrameInvariants,
                     output_file: &mut Write,
                     y4m_dec: &mut y4m::Decoder<Box<Read>>,
                     y4m_enc: &mut y4m::Encoder<File>) -> bool {
    let width = fi.width;
    let height = fi.height;
    match y4m_dec.read_frame() {
        Ok(y4m_frame) => {
            let y4m_y = y4m_frame.get_y_plane();
            let y4m_u = y4m_frame.get_u_plane();
            let y4m_v = y4m_frame.get_v_plane();
            println!("Frame {}", frame_number);
            let mut fs = FrameState::new(&fi);
            for y in 0..height {
                for x in 0..width {
                    let stride = fs.input.planes[0].stride;
                    fs.input.planes[0].data[y*stride+x] = y4m_y[y*width+x] as u16;
                }
            }
            for y in 0..height/2 {
                for x in 0..width/2 {
                    let stride = fs.input.planes[1].stride;
                    fs.input.planes[1].data[y*stride+x] = y4m_u[y*width/2+x] as u16;
                }
            }
            for y in 0..height/2 {
                for x in 0..width/2 {
                    let stride = fs.input.planes[2].stride;
                    fs.input.planes[2].data[y*stride+x] = y4m_v[y*width/2+x] as u16;
                }
            }
            let packet = encode_frame(&sequence, &fi, &mut fs);
            write_ivf_frame(output_file, frame_number, packet.as_ref());
            let mut rec_y = vec![128 as u8; width*height];
            let mut rec_u = vec![128 as u8; width*height/4];
            let mut rec_v = vec![128 as u8; width*height/4];
            for y in 0..height {
                for x in 0..width {
                    let stride = fs.rec.planes[0].stride;
                    rec_y[y*width+x] = fs.rec.planes[0].data[y*stride+x] as u8;
                }
            }
            for y in 0..height/2 {
                for x in 0..width/2 {
                    let stride = fs.rec.planes[1].stride;
                    rec_u[y*width/2+x] = fs.rec.planes[1].data[y*stride+x] as u8;
                }
            }
            for y in 0..height/2 {
                for x in 0..width/2 {
                    let stride = fs.rec.planes[2].stride;
                    rec_v[y*width/2+x] = fs.rec.planes[2].data[y*stride+x] as u8;
                }
            }
            let rec_frame = y4m::Frame::new([&rec_y, &rec_u, &rec_v], None);
            y4m_enc.write_frame(&rec_frame).unwrap();
            true
        },
        _ => false
    }
}
