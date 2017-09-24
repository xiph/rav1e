extern crate bitstream_io;
extern crate byteorder;
extern crate libc;
extern crate rand;
extern crate y4m;

use std::env;
use std::fs::File;
use std::io::prelude::*;
use bitstream_io::{BE, BitWriter};
use byteorder::*;

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

struct Plane {
    data: Vec<u16>,
    stride: usize
}

#[allow(dead_code)]
impl Plane {
    pub fn new(width: usize, height: usize) -> Plane {
        Plane {
            data: vec![128; width*height],
            stride: width
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
struct PlaneMutSlice<'a> {
    p: &'a mut Plane,
    x: usize,
    y: usize
}

#[allow(dead_code)]
impl<'a> PlaneMutSlice<'a> {
    pub fn as_slice(&mut self) -> &mut [u16] {
        let stride = self.p.stride;
        &mut self.p.data[self.y*stride+self.x..]
    }
}

struct Frame {
    planes: [Plane; 3]
}

impl Frame {
    pub fn new(width: usize, height:usize) -> Frame {
        Frame {
            planes: [
                Plane::new(width, height),
                Plane::new(width/2, height/2),
                Plane::new(width/2, height/2)
            ]
        }
    }
}

struct Sequence {
    profile: u8
}

impl Sequence {
    pub fn new() -> Sequence {
        Sequence {
            profile: 0
        }
    }
}

struct FrameState {
    input: Frame,
    rec: Frame
}

impl FrameState {
    pub fn new(fi: &FrameInvariants) -> FrameState {
        FrameState {
            input: Frame::new(fi.sb_width*64, fi.sb_height*64),
            rec: Frame::new(fi.sb_width*64, fi.sb_height*64),
        }
    }
}

struct FrameInvariants {
    qindex: usize,
    width: usize,
    height: usize,
    sb_width: usize,
    sb_height: usize
}

impl FrameInvariants {
    pub fn new(width: usize, height: usize) -> FrameInvariants {
        FrameInvariants {
            qindex: 200,
            width: width,
            height: height,
            sb_width: (width+63)/64,
            sb_height: (height+63)/64,
        }
    }
}

// TODO: possibly just use bitwriter instead of byteorder
fn write_ivf_header(output_file: &mut File, width: usize, height: usize) {
    output_file.write(b"DKIF").unwrap();
    output_file.write_u16::<LittleEndian>(0).unwrap(); // version
    output_file.write_u16::<LittleEndian>(32).unwrap(); // header length
    output_file.write(b"AV10").unwrap();
    output_file.write_u16::<LittleEndian>(width as u16).unwrap();
    output_file.write_u16::<LittleEndian>(height as u16).unwrap();
    output_file.write_u32::<LittleEndian>(60).unwrap();
    output_file.write_u32::<LittleEndian>(0).unwrap();
    output_file.write_u32::<LittleEndian>(0).unwrap();
    output_file.write_u32::<LittleEndian>(0).unwrap();
}

fn write_ivf_frame(output_file: &mut File, pts: u64, data: &[u8]) {
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
    uch.write(16,(fi.width-1) as u16)?; // width
    uch.write(16,(fi.height-1) as u16)?; // height
    uch.write_bit(false)?; // scaling active
    uch.write_bit(false)?; // screen content tools
    uch.write(3,0x0)?; // frame context
    uch.write(6,0)?; // loop filter level
    uch.write(3,0)?; // loop filter sharpness
    uch.write_bit(false)?; // loop filter deltas enabled
    uch.write(1,0)?; // cdef dering damping
    uch.write(2,0)?; // cdef clpf damping
    uch.write(2,0)?; // cdef bits
    for _ in 0..1 {
        uch.write(7,127)?; // cdef y strength
        uch.write(7,0)?; // cdef uv strength
    }
    uch.write(8,fi.qindex as u8)?; // qindex
    uch.write_bit(false)?; // y dc delta q
    uch.write_bit(false)?; // uv dc delta q
    uch.write_bit(false)?; // uv ac delta q
    //uch.write_bit(false)?; // using qmatrix
    uch.write_bit(false)?; // segmentation off
    uch.write_bit(false)?; // no delta q
    uch.write_bit(false)?; // tx mode select
    uch.write(2,0)?; // only 4x4 transforms
    //uch.write_bit(false)?; // use hybrid pred
    //uch.write_bit(false)?; // use compound pred
    uch.write_bit(true)?; // reduced tx

    //uch.write(1,0)?; // tile cols
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
    h.write(w.done()).unwrap();
    h
}

fn write_sb(cw: &mut ContextWriter, fi: &FrameInvariants, fs: &mut FrameState, sbx: usize, sby: usize) {
    cw.write_partition(PartitionType::PARTITION_NONE);
    cw.write_skip(false);
    cw.write_intra_mode(PredictionMode::DC_PRED);
    cw.write_intra_uv_mode(PredictionMode::DC_PRED);
    cw.write_tx_type(TxType::DCT_DCT);
    for p in 0..1 {
        let stride = fs.input.planes[0].stride;
        for by in 0..16 {
            for bx in 0..16 {
                let mut above = [0 as u16; 4];
                let mut left = [0 as u16; 4];
                match (sby, by) {
                    (0,0) => above = [127; 4],
                    _ => {
                        for i in 0..4 {
                            above[i] = fs.rec.planes[0].data[(sby*64+by*4-1)*stride+sbx*64+bx*4+i];
                        }
                    }
                }
                match (sbx, bx) {
                    (0,0) => left = [129; 4],
                    _ => {
                        for i in 0..4 {
                            left[i] = fs.rec.planes[0].data[(sby*64+by*4+i)*stride+sbx*64+bx*4-1];
                        }
                    }
                }
                match (sbx, bx, sby, by) {
                    (_,_,0,0) => pred_dc_left_4x4(&mut fs.rec.planes[0].data[(sby*64+by*4)*stride+sbx*64+bx*4..fi.sb_width*fi.sb_height*64*64], stride, &above, &left),
                    (0,0,_,_) => pred_dc_top_4x4(&mut fs.rec.planes[0].data[(sby*64+by*4)*stride+sbx*64+bx*4..fi.sb_width*fi.sb_height*64*64], stride, &above, &left),
                    _ => pred_dc_4x4(&mut fs.rec.planes[0].data[(sby*64+by*4)*stride+sbx*64+bx*4..fi.sb_width*fi.sb_height*64*64], stride, &above, &left),
                }
                let mut coeffs = [0 as i32; 16];
                let mut residual = [0 as i16; 16];
                for y in 0..4 {
                    for x in 0..4 {
                        residual[y*4+x] = fs.input.planes[0].data[(sby*64+by*4+y)*stride+sbx*64+bx*4+x] as i16 - fs.rec.planes[0].data[(sby*64+by*4+y)*stride+sbx*64+bx*4+x] as i16;
                    }
                }
                fdct4x4(&residual, &mut coeffs, 4);
                quantize_in_place(fi.qindex, &mut coeffs);
                cw.mc.set_loc(sbx*16+bx, sby*16+by);
                cw.write_coeffs(p, &coeffs);
                //reconstruct
                let mut rcoeffs = [0 as i32; 16];
                dequantize(fi.qindex, &coeffs, &mut rcoeffs);
                idct4x4_add(&mut rcoeffs, &mut fs.rec.planes[0].data[(sby*64+by*4)*stride+sbx*64+bx*4..fi.sb_width*fi.sb_height*64*64], stride);
            }
        }
    }
    // chroma all zeroes
    for p in 1..3 {
        for by in 0..8 {
            for bx in 0..8 {
                cw.mc.set_loc(bx, by);
                cw.write_token_block_zero(p);
            }
        }
    }
}

fn encode_tile(fi: &FrameInvariants, fs: &mut FrameState) -> Vec<u8> {
    let mut h = Vec::new();
    let mut w = ec::Writer::new();
    let mut fc = CDFContext::new();
    let mut mc = MIContext::new(fi.sb_width*16, fi.sb_height*16);
    {
        let mut cw = ContextWriter {
            w: &mut w,
            fc: &mut fc,
            mc: &mut mc,
        };
        for sby in 0..fi.sb_height {
            cw.reset_left_coeff_context(0);
            for sbx in 0..fi.sb_width {
                write_sb(&mut cw, fi, fs, sbx, sby);
            }
        }
    }
    h.write(w.done()).unwrap();
    h.write(&[0]).unwrap(); // superframe anti emulation
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

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut input_file = File::open(&args[1]).unwrap();
    let mut output_file = File::create(&args[2]).unwrap();
    let mut rec_file = File::create("rec.y4m").unwrap();
    let mut y4m_dec = y4m::decode(&mut input_file).unwrap();
    let width = y4m_dec.get_width();
    let height = y4m_dec.get_height();
    let sequence = Sequence::new();
    let mut y4m_enc = y4m::encode(width,height,y4m::Ratio::new(30,1)).write_header(&mut rec_file).unwrap();
    println!("Writing file");
    write_ivf_header(&mut output_file, width, height);
    let mut i = 0;
    loop {
        match y4m_dec.read_frame() {
            Ok(y4m_frame) => {
                let fi = FrameInvariants::new(width, height);
                let y4m_y = y4m_frame.get_y_plane();
                println!("Frame {}", i);
                let mut fs = FrameState::new(&fi);
                for y in 0..height {
                    for x in 0..width {
                        let stride = fs.input.planes[0].stride;
                        fs.input.planes[0].data[y*stride+x] = y4m_y[y*width+x] as u16;
                    }
                }
                let packet = encode_frame(&sequence, &fi, &mut fs);
                write_ivf_frame(&mut output_file, i, packet.as_ref());
                let mut rec_y = vec![128 as u8; width*height];
                let rec_u = vec![128 as u8; width*height/4];
                let rec_v = vec![128 as u8; width*height/4];
                for y in 0..height {
                    for x in 0..width {
                        let stride = fs.rec.planes[0].stride;
                        rec_y[y*width+x] = fs.rec.planes[0].data[y*stride+x] as u8;
                    }
                }
                let rec_frame = y4m::Frame::new([&rec_y, &rec_u, &rec_v], None);
                y4m_enc.write_frame(&rec_frame).unwrap();
                i += 1;
            },
            _ => break
        }
    }
}
