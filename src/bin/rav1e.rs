// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![deny(bare_trait_objects)]

mod common;
mod decoder;
mod muxer;
use crate::common::*;
use rav1e::prelude::*;

use std::io;
use std::io::Write;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use crate::decoder::Decoder;
use crate::decoder::VideoDetails;
use crate::muxer::*;
use std::fs::File;
use std::io::BufWriter;

struct Source<D: Decoder> {
 limit: usize,
 count: usize,
 input: D,
 #[cfg(all(unix, feature = "signal-hook"))]
 exit_requested: Arc<std::sync::atomic::AtomicBool>,
}

impl<D: Decoder> Source<D> {
  fn read_frame<T: Pixel>(&mut self, ctx: &mut Context<T>, video_info: VideoDetails) {
    if self.limit != 0 && self.count == self.limit {
      ctx.flush();
      return;
    }

    #[cfg(all(unix, feature = "signal-hook"))] {
      if self.exit_requested.load(std::sync::atomic::Ordering::SeqCst) {
        ctx.flush();
        return;
      }
    }

    match self.input.read_frame(&video_info) {
      Ok(frame) => {
        match video_info.bit_depth {
          8 | 10 | 12 => {}
          _ => panic!("unknown input bit depth!")
        }
        self.count += 1;
        let _ = ctx.send_frame(Some(Arc::new(frame)));
      }
      _ => {
        ctx.flush();
      }
    };
  }
}

// Encode and write a frame.
// Returns frame information in a `Result`.
fn process_frame<T: Pixel, D: Decoder>(
  ctx: &mut Context<T>, output_file: &mut dyn Muxer,
  source: &mut Source<D>,
  mut y4m_enc: Option<&mut y4m::Encoder<'_, Box<dyn Write>>>
) -> Option<Vec<FrameSummary>> {
  let y4m_details = source.input.get_video_details();
  let mut frame_summaries = Vec::new();
  let pkt_wrapped = ctx.receive_packet();
  match pkt_wrapped {
    Ok(pkt) => {
      output_file.write_frame(pkt.input_frameno as u64, pkt.data.as_ref());
      if let (Some(ref mut y4m_enc_uw), Some(ref rec)) = (y4m_enc.as_mut(), &pkt.rec) {
        write_y4m_frame(y4m_enc_uw, rec, y4m_details);
      }
      frame_summaries.push(pkt.into());
    }
    Err(EncoderStatus::NeedMoreData) => {
      source.read_frame(ctx, y4m_details);
    }
    Err(EncoderStatus::EnoughData) => {
      unreachable!();
    }
    Err(EncoderStatus::LimitReached) => {
      return None;
    }
    Err(EncoderStatus::Failure) => {
      panic!("Failed to encode video");
    }
    Err(EncoderStatus::Encoded) => {}
  }
  Some(frame_summaries)
}

fn write_stats_file<T: Pixel>(ctx: &Context<T>, filename: &Path) -> Result<(), io::Error> {
  let file = File::create(filename)?;
  let writer = BufWriter::new(file);
  serde_json::to_writer(writer, ctx.get_first_pass_data()).expect("Serialization should not fail");
  Ok(())
}

fn do_encode<T: Pixel, D: Decoder>(
  cfg: Config, verbose: bool, mut progress: ProgressInfo,
  output: &mut dyn Muxer,
  source: &mut Source<D>,
  mut y4m_enc: Option<y4m::Encoder<'_, Box<dyn Write>>>
) {
  let mut ctx: Context<T> = cfg.new_context();


  while let Some(frame_info) =
    process_frame(&mut ctx, &mut *output, source, y4m_enc.as_mut())
  {
    for frame in frame_info {
      progress.add_frame(frame);
      if verbose {
        eprintln!("{} - {}", frame, progress);
      } else {
        eprint!("\r{}                    ", progress);
      };
    }

    output.flush().unwrap();
  }

  if cfg.enc.pass == Some(1) {
    if let Err(e) =
      write_stats_file(&ctx, cfg.enc.stats_file.as_ref().unwrap())
    {
      eprintln!("\nError: Failed to write stats file! {}\n", e);
    }
  }
  eprint!("\n{}\n", progress.print_summary());
}

fn main() {
  let mut cli = parse_cli();
  let mut y4m_dec = y4m::decode(&mut cli.io.input).expect("input is not a y4m file");
  let video_info = y4m_dec.get_video_details();
  let y4m_enc = match cli.io.rec.as_mut() {
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
    enc: cli.enc,
    threads: cli.threads,
  };

  eprintln!("{}x{} @ {}/{} fps",
    video_info.width,
    video_info.height,
    video_info.time_base.den,
    video_info.time_base.num
  );

  cli.io.output.write_header(
    video_info.width,
    video_info.height,
    video_info.time_base.den as usize,
    video_info.time_base.num as usize
  );

  let progress = ProgressInfo::new(
    Rational { num: video_info.time_base.den, den: video_info.time_base.num },
    if cli.limit == 0 { None } else { Some(cli.limit) },
      cfg.enc.show_psnr
  );

  for _ in 0..cli.skip {
    y4m_dec.read_frame().expect("Skipped more frames than in the input");
  }

  #[cfg(all(unix, feature = "signal-hook"))]
  let exit_requested = {
    use std::sync::atomic::*;
    let e  = Arc::new(AtomicBool::from(false));

    fn setup_signal(sig: i32, e: Arc<AtomicBool>) {
      unsafe {
        signal_hook::register(sig, move || {
          if e.load(Ordering::SeqCst) {
            std::process::exit(128 + sig);
          }
          e.store(true, Ordering::SeqCst);
          eprintln!("\rExit requested, flushing.\n");
        }).expect("Cannot register the signal hooks");
      }
    }

    setup_signal(signal_hook::SIGTERM, e.clone());
    setup_signal(signal_hook::SIGQUIT, e.clone());
    setup_signal(signal_hook::SIGINT, e.clone());

    e
  };

  #[cfg(all(unix, feature = "signal-hook"))]
  let mut source = Source {
    limit: cli.limit,
    input: y4m_dec,
    count: 0,
    exit_requested
  };
  #[cfg(not(all(unix, feature = "signal-hook")))]
  let mut source = Source { limit: cli.limit, input: y4m_dec, count: 0 };

  if video_info.bit_depth == 8 {
    do_encode::<u8, y4m::Decoder<'_, Box<dyn Read>>>(
      cfg, cli.verbose, progress, &mut *cli.io.output, &mut source, y4m_enc
    )
  } else {
    do_encode::<u16, y4m::Decoder<'_, Box<dyn Read>>>(
      cfg, cli.verbose, progress, &mut *cli.io.output, &mut source, y4m_enc
    )
  }
}
