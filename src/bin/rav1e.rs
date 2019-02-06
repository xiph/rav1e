// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

extern crate clap;
extern crate rav1e;
extern crate y4m;
#[macro_use]
extern crate scan_fmt;

mod common;
mod decoder;
use common::*;
use rav1e::*;

use std::slice;
use std::io;
use std::io::Write;
use std::io::Read;
use std::sync::Arc;
use decoder::Decoder;
use decoder::VideoDetails;

fn read_frame_batch<D: Decoder>(ctx: &mut Context, decoder: &mut D, video_info: VideoDetails) {
  loop {
    if ctx.needs_more_lookahead() {
      match decoder.read_frame(&video_info) {
        Ok(frame) => {
          match video_info.bit_depth {
            8 | 10 | 12 => {}
            _ => panic!("unknown input bit depth!")
          }

          let _ = ctx.send_frame(Some(Arc::new(frame)));
          continue;
        }
        _ => {
          let frames_to_be_coded = ctx.get_frame_count();
          // This is a hack, instead when EOF is reached simply "close" the encoder to input (flag)
          ctx.set_limit(frames_to_be_coded);
          ctx.flush();
        }
      }
    } else if !ctx.needs_more_frames(ctx.get_frame_count()) {
      ctx.flush();
    }
    break;
  }
}

// Encode and write a frame.
// Returns frame information in a `Result`.
fn process_frame(
  ctx: &mut Context, output_file: &mut dyn Write,
  y4m_dec: &mut y4m::Decoder<'_, Box<dyn Read>>,
  mut y4m_enc: Option<&mut y4m::Encoder<'_, Box<dyn Write>>>
) -> Result<Vec<FrameSummary>, ()> {
  let y4m_details = y4m_dec.get_video_details();
  let mut frame_summaries = Vec::new();
  read_frame_batch(ctx, y4m_dec, y4m_details);
  let pkt_wrapped = ctx.receive_packet();
  if let Ok(pkt) = pkt_wrapped {
    write_ivf_frame(output_file, pkt.number as u64, pkt.data.as_ref());
    if let Some(y4m_enc_uw) = y4m_enc.as_mut() {
      if let Some(ref rec) = pkt.rec {
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
        y4m_enc_uw.write_frame(&rec_frame).unwrap();
      }
    }
    frame_summaries.push(pkt.into());
  }
  Ok(frame_summaries)
}

fn main() {
  let mut cli = parse_cli();
  let mut y4m_dec = y4m::decode(&mut cli.io.input).expect("input is not a y4m file");
  let video_info = y4m_dec.get_video_details();
  let mut y4m_enc = match cli.io.rec.as_mut() {
    Some(rec) => Some(
      y4m::encode(
        video_info.width,
        video_info.height,
        y4m::Ratio::new(video_info.time_base.den as usize, video_info.time_base.num as usize)
      ).with_colorspace(y4m_dec.get_colorspace())
        .write_header(rec)
        .unwrap()
    ),
    None => None
  };

  cli.enc.width = video_info.width;
  cli.enc.height = video_info.height;
  cli.enc.bit_depth = video_info.bit_depth;
  cli.enc.chroma_sampling = video_info.chroma_sampling;
  cli.enc.chroma_sample_position = video_info.chroma_sample_position;
  cli.enc.time_base = video_info.time_base;
  let cfg = Config {
    enc: cli.enc
  };

  let mut ctx = cfg.new_context();

  let stderr = io::stderr();
  let mut err = stderr.lock();

  let _ = writeln!(
    err,
    "{}x{} @ {}/{} fps",
    video_info.width,
    video_info.height,
    video_info.time_base.den,
    video_info.time_base.num
  );

  write_ivf_header(
    &mut cli.io.output,
    video_info.width,
    video_info.height,
    video_info.time_base.den as usize,
    video_info.time_base.num as usize
  );

  let mut progress = ProgressInfo::new(
    Rational { num: video_info.time_base.den, den: video_info.time_base.num },
    if cli.limit == 0 { None } else { Some(cli.limit) },
      cfg.enc.show_psnr
  );

  ctx.set_limit(cli.limit as u64);

  loop {
    match process_frame(&mut ctx, &mut cli.io.output, &mut y4m_dec, y4m_enc.as_mut()) {
      Ok(frame_info) => {
        for frame in frame_info {
          progress.add_frame(frame);
          let _ = if cli.verbose {
            writeln!(err, "{} - {}", frame, progress)
          } else {
            write!(err, "\r{}                    ", progress)
          };
        }
      },
      Err(_) => break,
    };

    if !ctx.needs_more_frames(progress.frames_encoded() as u64) {
      break;
    }

    cli.io.output.flush().unwrap();
  }

  let _ = write!(err, "\n{}\n", progress.print_stats());
}
