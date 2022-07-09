// Copyright (c) 2001-2016, Alliance for Open Media. All rights reserved
// Copyright (c) 2017-2021, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![deny(bare_trait_objects)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_ptr_alignment)]
#![allow(clippy::cognitive_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::verbose_bit_mask)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::many_single_char_names)]
// Performance lints
#![warn(clippy::linkedlist)]
#![warn(clippy::missing_const_for_fn)]
#![warn(clippy::mutex_integer)]
#![warn(clippy::suboptimal_flops)]
// Correctness lints
#![warn(clippy::expl_impl_clone_on_copy)]
#![warn(clippy::mem_forget)]
#![warn(clippy::path_buf_push_overwrite)]
// Clarity/formatting lints
#![warn(clippy::map_flatten)]
#![warn(clippy::mut_mut)]
#![warn(clippy::needless_borrow)]
#![warn(clippy::needless_continue)]
#![warn(clippy::range_plus_one)]
// Documentation lints
#![warn(clippy::doc_markdown)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
// FIXME: Temporarily disabled due to https://github.com/rust-lang/rust-clippy/issues/9142
#![allow(clippy::undocumented_unsafe_blocks)]

/// Simple ivf muxer
///
use bitstream_io::{BitRead, BitReader, BitWrite, BitWriter, LittleEndian};
use std::io;

/// # Panics
///
/// - If header cannot be written to output file.
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

/// # Panics
///
/// - If frame cannot be written to output file.
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

/// # Errors
///
/// - Returns `io::Error` if packet cannot be read
/// - Returns `io::ErrorKind::InvalidData` if header signature is invalid
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

/// # Errors
///
/// - Returns `io::Error` if packet cannot be read
pub fn read_packet(r: &mut dyn io::Read) -> io::Result<Packet> {
  let mut br = BitReader::endian(r, LittleEndian);

  let len: u32 = br.read(32)?;
  let pts: u64 = br.read(64)?;
  let mut buf = vec![0u8; len as usize];

  br.read_bytes(&mut buf)?;

  Ok(Packet { data: buf.into_boxed_slice(), pts })
}
