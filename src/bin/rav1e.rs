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
mod decoder;
use common::*;
use rav1e::*;

use std::io;
use std::io::Write;
use decoder::Decoder;

fn main() {
  let mut cli = parse_cli();
  let mut y4m_dec = y4m::decode(&mut cli.io.input).unwrap();
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

  let cfg = Config {
    video_info,
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
