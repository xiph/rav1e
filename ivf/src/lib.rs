// Copyright (c) 2001-2016, Alliance for Open Media. All rights reserved
// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

/// Simple ivf muxer
///
use bitstream_io::{BitRead, BitReader, BitWrite, BitWriter, LittleEndian};
use std::io;

pub fn write_ivf_header(
  output_file: &mut dyn io::Write, width: usize, height: usize,
  framerate_num: usize, framerate_den: usize,
) {
  let mut bw = BitWriter::endian(output_file, LittleEndian);
  bw.write_bytes(b"DKIF").unwrap();
  bw.write(16, 0).unwrap(); // version
  bw.write(16, 32).unwrap(); // version
  bw.write_bytes(b"AV01").unwrap();
  bw.write(16, width as u16).unwrap();
  bw.write(16, height as u16).unwrap();
  bw.write(32, framerate_num as u32).unwrap();
  bw.write(32, framerate_den as u32).unwrap();
  bw.write(32, 0).unwrap();
  bw.write(32, 0).unwrap();
}

pub fn write_ivf_frame(
  output_file: &mut dyn io::Write, pts: u64, data: &[u8],
) {
  let mut bw = BitWriter::endian(output_file, LittleEndian);
  bw.write(32, data.len() as u32).unwrap();
  bw.write(64, pts).unwrap();
  bw.write_bytes(data).unwrap();
}

#[derive(Debug, PartialEq)]
pub struct Header {
  pub tag: [u8; 4],
  pub w: u16,
  pub h: u16,
  pub timebase_num: u32,
  pub timebase_den: u32,
}

pub fn read_header(r: &mut dyn io::Read) -> io::Result<Header> {
  let mut br = BitReader::endian(r, LittleEndian);

  let mut signature = [0u8; 4];
  let mut tag = [0u8; 4];

  br.read_bytes(&mut signature)?;

  if &signature != b"DKIF" {
    return Err(io::ErrorKind::InvalidData.into());
  }

  let _v0: u16 = br.read(16)?;
  let _v1: u16 = br.read(16)?;
  br.read_bytes(&mut tag)?;

  let w: u16 = br.read(16)?;
  let h: u16 = br.read(16)?;

  let timebase_den: u32 = br.read(32)?;
  let timebase_num: u32 = br.read(32)?;

  let _: u32 = br.read(32)?;
  let _: u32 = br.read(32)?;

  Ok(Header { tag, w, h, timebase_num, timebase_den })
}

pub struct Packet {
  pub data: Box<[u8]>,
  pub pts: u64,
}

pub fn read_packet(r: &mut dyn io::Read) -> io::Result<Packet> {
  let mut br = BitReader::endian(r, LittleEndian);

  let len: u32 = br.read(32)?;
  let pts: u64 = br.read(64)?;
  let mut buf = vec![0u8; len as usize];

  br.read_bytes(&mut buf)?;

  Ok(Packet { data: buf.into_boxed_slice(), pts })
}
