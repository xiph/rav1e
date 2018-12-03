// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use std::io;
use std::rc::Rc;

use bitstream_io::{BigEndian, BitWriter, LittleEndian};

use context::*;
use encoder::headers::UncompressedHeader;
use partition::*;
use plane::*;
use util::*;

pub use self::block::*;
pub use self::deblock::*;
pub use self::frame::*;
pub use self::motion_comp::*;
pub use self::reference::*;
pub use self::segmentation::*;
pub use self::sequence::*;
pub use self::transform::*;

mod block;
mod deblock;
mod frame;
mod headers;
mod motion_comp;
mod partition;
mod reference;
mod segmentation;
mod sequence;
mod tile;
mod transform;

extern {
  pub fn av1_rtcd();
  pub fn aom_dsp_rtcd();
}

const MAX_NUM_TEMPORAL_LAYERS: usize = 8;
const MAX_NUM_SPATIAL_LAYERS: usize = 4;
const MAX_NUM_OPERATING_POINTS: usize =
  MAX_NUM_TEMPORAL_LAYERS * MAX_NUM_SPATIAL_LAYERS;

pub const PRIMARY_REF_NONE: u32 = 7;
const PRIMARY_REF_BITS: u32 = 3;

arg_enum! {
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

pub fn write_ivf_header(
  output_file: &mut dyn io::Write, width: usize, height: usize, num: usize,
  den: usize
) {
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

pub fn write_ivf_frame(
  output_file: &mut dyn io::Write, pts: u64, data: &[u8]
) {
  let mut bw = BitWriter::endian(output_file, LittleEndian);
  bw.write(32, data.len() as u32).unwrap();
  bw.write(64, pts).unwrap();
  bw.write_bytes(data).unwrap();
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
  OBU_PADDING = 15
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
    if value == 0 {
      break;
    }
  }
  size
}

fn aom_uleb_encode(mut value: u64, coded_value: &mut [u8]) -> usize {
  let leb_size = aom_uleb_size_in_bytes(value);

  for i in 0..leb_size {
    let mut byte = (value & 0x7f) as u8;
    value >>= 7;
    if value != 0 {
      byte |= 0x80
    }; // Signal that more bytes follow.
    coded_value[i] = byte;
  }

  leb_size
}

fn write_obus(
  packet: &mut dyn io::Write, sequence: &mut Sequence,
  fi: &mut FrameInvariants, fs: &FrameState
) -> io::Result<()> {
  //let mut uch = BitWriter::endian(packet, BigEndian);
  let obu_extension = 0 as u32;

  let mut buf1 = Vec::new();
  {
    let mut bw1 = BitWriter::endian(&mut buf1, BigEndian);
    bw1.write_obu_header(OBU_Type::OBU_TEMPORAL_DELIMITER, obu_extension)?;
    bw1.write(8, 0)?; // size of payload == 0, one byte
  }
  packet.write_all(&buf1).unwrap();
  buf1.clear();

  // write sequence header obu if KEY_FRAME, preceded by 4-byte size
  if fi.frame_type == FrameType::KEY {
    let mut buf2 = Vec::new();
    {
      let mut bw2 = BitWriter::endian(&mut buf2, BigEndian);
      bw2.write_sequence_header_obu(sequence, fi)?;
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
      let leb_size =
        aom_uleb_encode(obu_payload_size, &mut coded_payload_length);
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
    bw2.write_frame_header_obu(sequence, fi, fs)?;
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
    let leb_size =
      aom_uleb_encode(obu_payload_size, &mut coded_payload_length);
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
fn diff(
  dst: &mut [i16], src1: &PlaneSlice<'_>, src2: &PlaneSlice<'_>, width: usize,
  height: usize
) {
  let src1_stride = src1.plane.cfg.stride;
  let src2_stride = src2.plane.cfg.stride;

  for ((l, s1), s2) in dst
    .chunks_mut(width)
    .take(height)
    .zip(src1.as_slice().chunks(src1_stride))
    .zip(src2.as_slice().chunks(src2_stride))
  {
    for ((r, v1), v2) in l.iter_mut().zip(s1).zip(s2) {
      *r = *v1 as i16 - *v2 as i16;
    }
  }
}

fn get_qidx(
  fi: &FrameInvariants, fs: &FrameState, cw: &ContextWriter, bo: &BlockOffset
) -> u8 {
  let mut qidx = fi.base_q_idx;
  let sidx = cw.bc.at(bo).segmentation_idx as usize;
  if fs.segmentation.features[sidx][SegLvl::SEG_LVL_ALT_Q as usize] {
    let delta = fs.segmentation.data[sidx][SegLvl::SEG_LVL_ALT_Q as usize];
    qidx = clamp((qidx as i16) + delta, 0, 255) as u8;
  }
  qidx
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

pub fn update_rec_buffer(fi: &mut FrameInvariants, fs: FrameState) {
  let rfs = Rc::new(ReferenceFrame {
    order_hint: fi.order_hint,
    frame: fs.rec,
    input_hres: fs.input_hres,
    input_qres: fs.input_qres,
    cdfs: fs.cdfs
  });
  for i in 0..(REF_FRAMES as usize) {
    if (fi.refresh_frame_flags & (1 << i)) != 0 {
      fi.rec_buffer.frames[i] = Some(Rc::clone(&rfs));
      fi.rec_buffer.deblock[i] = fs.deblock;
    }
  }
}

#[cfg(test)]
mod test {
  use api::EncoderConfig;

  #[test]
  fn frame_state_window() {
    use super::*;
    let config = EncoderConfig { ..Default::default() };
    let fi = FrameInvariants::new(1024, 1024, config);
    let mut fs = FrameState::new(&fi);
    for p in fs.rec.planes.iter_mut() {
      for (i, v) in p
        .mut_slice(&PlaneOffset { x: 0, y: 0 })
        .as_mut_slice()
        .iter_mut()
        .enumerate()
      {
        *v = i as u16;
      }
    }
    let offset = BlockOffset { x: 56, y: 56 };
    let sbo = offset.sb_offset();
    let fs_ = fs.window(&sbo);
    for p in 0..3 {
      assert!(fs_.rec.planes[p].cfg.xorigin < 0);
      assert!(fs_.rec.planes[p].cfg.yorigin < 0);
      let po = offset.plane_offset(&fs.rec.planes[p].cfg);
      assert_eq!(
        fs.rec.planes[p].slice(&po).as_slice()[..32],
        fs_.rec.planes[p].slice(&po).as_slice()[..32]
      );
    }
  }
}
