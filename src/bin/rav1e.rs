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

mod common;
use common::*;

use std::io;
use std::io::Write;
use rav1e::*;

fn main() {
  let mut cli = parse_cli();
  let mut y4m_dec = y4m::decode(&mut cli.io.input).unwrap();
  let width = y4m_dec.get_width();
  let height = y4m_dec.get_height();
  let framerate = y4m_dec.get_framerate();
  let color_space = y4m_dec.get_colorspace();

  let mut y4m_enc = match cli.io.rec.as_mut() {
    Some(rec) => Some(
      y4m::encode(width, height, framerate)
        .with_colorspace(color_space)
        .write_header(rec)
        .unwrap()
    ),
    None => None
  };

  let chroma_sampling = match color_space {
    y4m::Colorspace::C420
    | y4m::Colorspace::C420jpeg
    | y4m::Colorspace::C420paldv
    | y4m::Colorspace::C420mpeg2
    | y4m::Colorspace::C420p10
    | y4m::Colorspace::C420p12 => ChromaSampling::Cs420,
    y4m::Colorspace::C422
    | y4m::Colorspace::C422p10
    | y4m::Colorspace::C422p12 => ChromaSampling::Cs422,
    y4m::Colorspace::C444
    | y4m::Colorspace::C444p10
    | y4m::Colorspace::C444p12 => ChromaSampling::Cs444,
    _ => panic!("Chroma sampling unknown for the specified color space.")
  };

  let bit_depth = color_space.get_bit_depth();

  let cfg = Config {
    frame_info: FrameInfo { width, height, bit_depth, chroma_sampling },
    timebase: Rational::new(framerate.den as u64, framerate.num as u64),
    enc: cli.enc
  };

  let mut ctx = cfg.new_context();

  let stderr = io::stderr();
  let mut err = stderr.lock();

  let _ = writeln!(err, "{}x{} @ {}/{} fps", width, height, framerate.num, framerate.den);

  write_ivf_header(
    &mut cli.io.output,
    width,
    height,
    framerate.num,
    framerate.den
  );

  let mut progress = ProgressInfo::new(
    framerate,
    if cli.limit == 0 { None } else { Some(cli.limit) },
      cfg.enc.show_psnr
  );

  ctx.set_frames_to_be_coded(cli.limit as u64);

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
