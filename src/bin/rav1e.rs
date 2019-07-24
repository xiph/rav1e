// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![deny(bare_trait_objects)]

#[macro_use]
extern crate err_derive;

mod common;
mod decoder;
mod error;
mod muxer;

use crate::common::*;
use crate::error::*;
use rav1e::prelude::*;

use std::io::Write;
use std::io::Read;
use std::io::Seek;
use std::sync::Arc;
use crate::decoder::Decoder;
use crate::decoder::VideoDetails;
use crate::muxer::*;
use std::fs::File;

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
  pass1file: Option<&mut File>,
  pass2file: Option<&mut File>,
  buffer: &mut [u8],
  buf_pos: &mut usize,
  mut y4m_enc: Option<&mut y4m::Encoder<'_, Box<dyn Write>>>
) -> Result<Option<Vec<FrameSummary>>, CliError> {
  let y4m_details = source.input.get_video_details();
  let mut frame_summaries = Vec::new();
  let mut pass1file = pass1file;
  let mut pass2file = pass2file;
  // Submit first pass data to pass 2.
  if let Some(passfile) = pass2file.as_mut() {
    loop {
      let mut bytes = ctx.twopass_bytes_needed();
      if bytes == 0 {
        break;
      }
      // Read in some more bytes, if necessary.
      bytes = bytes.min(buffer.len() - *buf_pos);
      if bytes > 0 {
        passfile.read_exact(&mut buffer[*buf_pos..*buf_pos + bytes])
         .expect("Could not read frame data from two-pass data file!");
      }
      // And pass them off.
      let consumed = ctx.twopass_in(&buffer[*buf_pos..*buf_pos + bytes])
       .expect("Error submitting pass data in second pass.");
      // If the encoder consumed the whole buffer, reset it.
      if consumed >= bytes {
        *buf_pos = 0;
      }
      else {
        *buf_pos += consumed;
      }
    }
  }
  // Extract first pass data from pass 1.
  // We call this before encoding any frames to ensure we are in 2-pass mode
  //  and to get the placeholder header data.
  if let Some(passfile) = pass1file.as_mut() {
    if let Some(outbuf) = ctx.twopass_out() {
      passfile.write_all(outbuf)
       .expect("Unable to write to two-pass data file.");
    }
  }

  let pkt_wrapped = ctx.receive_packet();
  match pkt_wrapped {
    Ok(pkt) => {
      output_file.write_frame(pkt.input_frameno as u64, pkt.data.as_ref(), pkt.frame_type);
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
      if let Some(passfile) = pass1file.as_mut() {
        if let Some(outbuf) = ctx.twopass_out() {
          // The last block of data we get is the summary data that needs to go
          //  at the start of the pass file.
          // Seek to the start so we can write it there.
          passfile.seek(std::io::SeekFrom::Start(0))
           .expect("Unable to seek in two-pass data file.");
          passfile.write_all(outbuf)
           .expect("Unable to write to two-pass data file.");
        }
      }
      return Ok(None);
    }
    e @ Err(EncoderStatus::Failure) => {
      let _ = e.map_err(|e| e.context("Failed to encode video"))?;
    }
    e @ Err(EncoderStatus::NotReady) => {
      let _ = e.map_err(|e| e.context("Mismanaged handling of two-pass stats data"))?;
    }
    Err(EncoderStatus::Encoded) => {}
  }
  Ok(Some(frame_summaries))
}

fn do_encode<T: Pixel, D: Decoder>(
  cfg: Config, verbose: bool, mut progress: ProgressInfo,
  output: &mut dyn Muxer,
  source: &mut Source<D>,
  pass1file_name: Option<&String>,
  pass2file_name: Option<&String>,
  mut y4m_enc: Option<y4m::Encoder<'_, Box<dyn Write>>>
) -> Result<(), CliError> {
  let mut ctx: Context<T> = cfg.new_context();

  let mut pass2file = pass2file_name.map(|f| {
    File::open(f)
     .unwrap_or_else(|_| panic!("Unable to open \"{}\" for reading two-pass data.", f))
  });
  let mut pass1file = pass1file_name.map(|f| {
    File::create(f)
     .unwrap_or_else(|_| panic!("Unable to open \"{}\" for writing two-pass data.", f))
  });

  let mut buffer: [u8; 80] = [0; 80];
  let mut buf_pos = 0;

  while let Some(frame_info) =
    process_frame(&mut ctx, &mut *output, source, pass1file.as_mut(),
     pass2file.as_mut(), &mut buffer, &mut buf_pos, y4m_enc.as_mut())?
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
  eprint!("\n{}\n", progress.print_summary());
  Ok(())
}

fn main() {
  better_panic::install();

  match run() {
    Ok(()) => {},
    Err(e) => error::print_error(&e),
  }
}

fn run() -> Result<(), error::CliError> {
  let mut cli = parse_cli()?;
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

  // If no pixel range is specified via CLI, assume limited,
  // as it is the default for the Y4M format.
  if !cli.color_range_specified {
    cli.enc.pixel_range = PixelRange::Limited;
  }

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
      cfg, cli.verbose, progress, &mut *cli.io.output, &mut source,
      cli.pass1file_name.as_ref(), cli.pass2file_name.as_ref(), y4m_enc
    )?
  } else {
    do_encode::<u16, y4m::Decoder<'_, Box<dyn Read>>>(
      cfg, cli.verbose, progress, &mut *cli.io.output, &mut source,
      cli.pass1file_name.as_ref(), cli.pass2file_name.as_ref(), y4m_enc
    )?
  }

  Ok(())
}
