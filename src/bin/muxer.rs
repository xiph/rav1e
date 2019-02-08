// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use bitstream_io::{BitWriter, LittleEndian};
use crate::decoder::VideoDetails;
use std::io;
use std::io::Write;
use std::slice;

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

pub fn write_y4m_frame(y4m_enc: &mut y4m::Encoder<'_, Box<dyn Write>>, rec: &rav1e::Frame, y4m_details: VideoDetails) {
  let pitch_y = if y4m_details.bit_depth > 8 { y4m_details.width * 2 } else { y4m_details.width };
  let chroma_sampling_period = y4m_details.chroma_sampling.sampling_period();
  let (pitch_uv, height_uv) = (
    pitch_y / chroma_sampling_period.0,
    y4m_details.height / chroma_sampling_period.1
  );

  let (mut rec_y, mut rec_u, mut rec_v) = (
    vec![128u8; pitch_y * y4m_details.height],
    vec![128u8; pitch_uv * height_uv],
    vec![128u8; pitch_uv * height_uv]
  );

  let (stride_y, stride_u, stride_v) = (
    rec.planes[0].cfg.stride,
    rec.planes[1].cfg.stride,
    rec.planes[2].cfg.stride
  );

  for (line, line_out) in rec.planes[0]
    .data_origin()
    .chunks(stride_y)
    .zip(rec_y.chunks_mut(pitch_y))
  {
    if y4m_details.bit_depth > 8 {
      unsafe {
        line_out.copy_from_slice(slice::from_raw_parts::<u8>(
          line.as_ptr() as (*const u8),
          pitch_y
        ));
      }
    } else {
      line_out.copy_from_slice(
        &line.iter().map(|&v| v as u8).collect::<Vec<u8>>()[..pitch_y]
      );
    }
  }
  for (line, line_out) in rec.planes[1]
    .data_origin()
    .chunks(stride_u)
    .zip(rec_u.chunks_mut(pitch_uv))
  {
    if y4m_details.bit_depth > 8 {
      unsafe {
        line_out.copy_from_slice(slice::from_raw_parts::<u8>(
          line.as_ptr() as (*const u8),
          pitch_uv
        ));
      }
    } else {
      line_out.copy_from_slice(
        &line.iter().map(|&v| v as u8).collect::<Vec<u8>>()[..pitch_uv]
      );
    }
  }
  for (line, line_out) in rec.planes[2]
    .data_origin()
    .chunks(stride_v)
    .zip(rec_v.chunks_mut(pitch_uv))
  {
    if y4m_details.bit_depth > 8 {
      unsafe {
        line_out.copy_from_slice(slice::from_raw_parts::<u8>(
          line.as_ptr() as (*const u8),
          pitch_uv
        ));
      }
    } else {
      line_out.copy_from_slice(
        &line.iter().map(|&v| v as u8).collect::<Vec<u8>>()[..pitch_uv]
      );
    }
  }

  let rec_frame = y4m::Frame::new([&rec_y, &rec_u, &rec_v], None);
  y4m_enc.write_frame(&rec_frame).unwrap();
}
