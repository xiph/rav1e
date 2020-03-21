// Copyright (c) 2017-2019, The rav1e contributors. All rights reserved
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
#![warn(clippy::expl_impl_clone_on_copy)]
#![warn(clippy::linkedlist)]
#![warn(clippy::map_flatten)]
#![warn(clippy::mem_forget)]
#![warn(clippy::mut_mut)]
#![warn(clippy::mutex_integer)]
#![warn(clippy::needless_borrow)]
#![warn(clippy::needless_continue)]
#![warn(clippy::path_buf_push_overwrite)]
#![warn(clippy::range_plus_one)]

#[macro_use]
extern crate log;

mod common;
mod decoder;
mod error;
#[cfg(feature = "serialize")]
mod kv;
mod muxer;
mod stats;

use crate::common::*;
use crate::error::*;
use crate::stats::*;
use rav1e::prelude::*;

use crate::decoder::Decoder;
use crate::decoder::VideoDetails;
use crate::muxer::*;
use std::fs::File;
use std::io::Read;
use std::io::Seek;
use std::io::Write;
use std::sync::Arc;

struct Source<D: Decoder> {
  limit: usize,
  count: usize,
  input: D,
  #[cfg(all(unix, feature = "signal-hook"))]
  exit_requested: Arc<std::sync::atomic::AtomicBool>,
}

impl<D: Decoder> Source<D> {
  fn read_frame<T: Pixel>(
    &mut self, ctx: &mut Context<T>, video_info: VideoDetails,
  ) {
    if self.limit != 0 && self.count == self.limit {
      ctx.flush();
      return;
    }

    #[cfg(all(unix, feature = "signal-hook"))]
    {
      if self.exit_requested.load(std::sync::atomic::Ordering::SeqCst) {
        ctx.flush();
        return;
      }
    }

    match self.input.read_frame(&video_info) {
      Ok(frame) => {
        match video_info.bit_depth {
          8 | 10 | 12 => {}
          _ => panic!("unknown input bit depth!"),
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
  ctx: &mut Context<T>, output_file: &mut dyn Muxer, source: &mut Source<D>,
  pass1file: Option<&mut File>, pass2file: Option<&mut File>,
  buffer: &mut [u8], buf_pos: &mut usize,
  mut y4m_enc: Option<&mut y4m::Encoder<'_, Box<dyn Write>>>,
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
        passfile
          .read_exact(&mut buffer[*buf_pos..*buf_pos + bytes])
          .expect("Could not read frame data from two-pass data file!");
      }
      // And pass them off.
      let consumed = ctx
        .twopass_in(&buffer[*buf_pos..*buf_pos + bytes])
        .expect("Error submitting pass data in second pass.");
      // If the encoder consumed the whole buffer, reset it.
      if consumed >= bytes {
        *buf_pos = 0;
      } else {
        *buf_pos += consumed;
      }
    }
  }
  // Extract first pass data from pass 1.
  // We call this before encoding any frames to ensure we are in 2-pass mode
  //  and to get the placeholder header data.
  if let Some(passfile) = pass1file.as_mut() {
    if let Some(outbuf) = ctx.twopass_out() {
      passfile
        .write_all(outbuf)
        .expect("Unable to write to two-pass data file.");
    }
  }

  let pkt_wrapped = ctx.receive_packet();
  match pkt_wrapped {
    Ok(pkt) => {
      output_file.write_frame(
        pkt.input_frameno as u64,
        pkt.data.as_ref(),
        pkt.frame_type,
      );
      if let (Some(ref mut y4m_enc_uw), Some(ref rec)) =
        (y4m_enc.as_mut(), &pkt.rec)
      {
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
          passfile
            .seek(std::io::SeekFrom::Start(0))
            .expect("Unable to seek in two-pass data file.");
          passfile
            .write_all(outbuf)
            .expect("Unable to write to two-pass data file.");
        }
      }
      return Ok(None);
    }
    e @ Err(EncoderStatus::Failure) => {
      let _ = e.map_err(|e| e.context("Failed to encode video"))?;
    }
    e @ Err(EncoderStatus::NotReady) => {
      let _ = e.map_err(|e| {
        e.context("Mismanaged handling of two-pass stats data")
      })?;
    }
    Err(EncoderStatus::Encoded) => {}
  }
  Ok(Some(frame_summaries))
}

fn do_encode<T: Pixel, D: Decoder>(
  cfg: Config, verbose: Verbose, mut progress: ProgressInfo,
  output: &mut dyn Muxer, source: &mut Source<D>,
  pass1file_name: Option<&String>, pass2file_name: Option<&String>,
  mut y4m_enc: Option<y4m::Encoder<'_, Box<dyn Write>>>,
) -> Result<(), CliError> {
  let mut ctx: Context<T> =
    cfg.new_context().map_err(|e| e.context("Invalid encoder settings"))?;

  let mut pass2file = pass2file_name.map(|f| {
    File::open(f).unwrap_or_else(|_| {
      panic!("Unable to open \"{}\" for reading two-pass data.", f)
    })
  });
  let mut pass1file = pass1file_name.map(|f| {
    File::create(f).unwrap_or_else(|_| {
      panic!("Unable to open \"{}\" for writing two-pass data.", f)
    })
  });

  let mut buffer: [u8; 80] = [0; 80];
  let mut buf_pos = 0;

  while let Some(frame_info) = process_frame(
    &mut ctx,
    &mut *output,
    source,
    pass1file.as_mut(),
    pass2file.as_mut(),
    &mut buffer,
    &mut buf_pos,
    y4m_enc.as_mut(),
  )? {
    if verbose != Verbose::Quiet {
      for frame in frame_info {
        progress.add_frame(frame.clone());
        if verbose == Verbose::Verbose {
          info!("{} - {}", frame, progress);
        } else {
          // Print a one-line progress indicator that overrides itself with every update
          eprint!("\r{}                    ", progress);
        };
      }

      output.flush().unwrap();
    }
  }
  if verbose != Verbose::Quiet {
    if verbose == Verbose::Verbose {
      // Clear out the temporary progress indicator
      eprint!("\r");
    }
    progress.print_summary(verbose == Verbose::Verbose);
  }
  Ok(())
}

fn main() {
  #[cfg(feature = "tracing")]
  use rust_hawktracer::*;
  better_panic::install();
  init_logger();

  #[cfg(feature = "tracing")]
  let instance = HawktracerInstance::new();
  #[cfg(feature = "tracing")]
  let _listener = instance.create_listener(HawktracerListenerType::ToFile {
    file_path: "trace.bin".into(),
    buffer_size: 4096,
  });

  match run() {
    Ok(()) => {}
    Err(e) => error::print_error(&e),
  }
}

fn init_logger() {
  use std::str::FromStr;
  fn level_colored(l: log::Level) -> console::StyledObject<&'static str> {
    use console::style;
    use log::Level;
    match l {
      Level::Trace => style("??").dim(),
      Level::Debug => style("? ").dim(),
      Level::Info => style("> ").green(),
      Level::Warn => style("! ").yellow(),
      Level::Error => style("!!").red(),
    }
  }

  // this can be changed to flatten
  let level = std::env::var("RAV1E_LOG")
    .ok()
    .map(|l| log::LevelFilter::from_str(&l).ok())
    .unwrap_or(Some(log::LevelFilter::Info))
    .unwrap();

  fern::Dispatch::new()
    .format(move |out, message, record| {
      out.finish(format_args!(
        "{level} {message}",
        level = level_colored(record.level()),
        message = message,
      ));
    })
    // set the default log level. to filter out verbose log messages from dependencies, set
    // this to Warn and overwrite the log level for your crate.
    .level(log::LevelFilter::Warn)
    // change log levels for individual modules. Note: This looks for the record's target
    // field which defaults to the module path but can be overwritten with the `target`
    // parameter:
    // `info!(target="special_target", "This log message is about special_target");`
    .level_for("rav1e", level)
    // output to stdout
    .chain(std::io::stderr())
    .apply()
    .unwrap();
}

cfg_if::cfg_if! {
  if #[cfg(any(target_os = "windows", target_arch = "wasm32"))] {
    fn print_rusage() {
      eprintln!("Windows benchmarking is not supported currently.");
    }
  } else {
    fn print_rusage() {
      let (utime, stime, maxrss) = unsafe {
        let mut usage = std::mem::zeroed();
        let _ = libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        (usage.ru_utime, usage.ru_stime, usage.ru_maxrss)
      };
      eprintln!(
        "user time: {} s",
        utime.tv_sec as f64 + utime.tv_usec as f64 / 1_000_000f64
      );
      eprintln!(
        "system time: {} s",
        stime.tv_sec as f64 + stime.tv_usec as f64 / 1_000_000f64
      );
      eprintln!("maximum rss: {} KB", maxrss);
    }
  }
}

fn run() -> Result<(), error::CliError> {
  let mut cli = parse_cli()?;
  // Maximum frame size by specification + maximum y4m header
  let limit = y4m::Limits {
    // Use saturating operations to gracefully handle 32-bit architectures
    bytes: 64usize
      .saturating_mul(64)
      .saturating_mul(4096)
      .saturating_mul(2304)
      .saturating_add(1024),
  };
  let mut y4m_dec = y4m::Decoder::new_with_limits(&mut cli.io.input, limit)
    .expect("cannot decode the input");
  let video_info = y4m_dec.get_video_details();
  let y4m_enc = match cli.io.rec.as_mut() {
    Some(rec) => Some(
      y4m::encode(
        video_info.width,
        video_info.height,
        y4m::Ratio::new(
          video_info.time_base.den as usize,
          video_info.time_base.num as usize,
        ),
      )
      .with_colorspace(y4m_dec.get_colorspace())
      .write_header(rec)
      .unwrap(),
    ),
    None => None,
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

  if !cli.override_time_base {
    cli.enc.time_base = video_info.time_base;
  }

  let cfg = Config { enc: cli.enc, threads: cli.threads };

  #[cfg(feature = "serialize")]
  {
    if let Some(save_config) = cli.save_config {
      let mut out = File::create(save_config)
        .map_err(|e| e.context("Cannot create configuration file"))?;
      let s = toml::to_string(&cfg.enc).unwrap();
      out
        .write_all(s.as_bytes())
        .map_err(|e| e.context("Cannot write the configuration file"))?
    }
  }

  cli.io.output.write_header(
    video_info.width,
    video_info.height,
    cli.enc.time_base.den as usize,
    cli.enc.time_base.num as usize,
  );

  info!(
    "Using y4m decoder: {}x{}p @ {}/{} fps, {}, {}-bit",
    video_info.width,
    video_info.height,
    video_info.time_base.den,
    video_info.time_base.num,
    video_info.chroma_sampling,
    video_info.bit_depth
  );
  info!("Encoding settings: {}", cfg.enc);

  let progress = ProgressInfo::new(
    Rational { num: video_info.time_base.den, den: video_info.time_base.num },
    if cli.limit == 0 { None } else { Some(cli.limit) },
    cfg.enc.show_psnr,
  );

  for _ in 0..cli.skip {
    y4m_dec.read_frame().expect("Skipped more frames than in the input");
  }

  #[cfg(all(unix, feature = "signal-hook"))]
  let exit_requested = {
    use std::sync::atomic::*;
    let e = Arc::new(AtomicBool::from(false));

    fn setup_signal(sig: i32, e: Arc<AtomicBool>) {
      unsafe {
        signal_hook::register(sig, move || {
          if e.load(Ordering::SeqCst) {
            std::process::exit(128 + sig);
          }
          e.store(true, Ordering::SeqCst);
          info!("Exit requested, flushing.");
        })
        .expect("Cannot register the signal hooks");
      }
    }

    setup_signal(signal_hook::SIGTERM, e.clone());
    setup_signal(signal_hook::SIGQUIT, e.clone());
    setup_signal(signal_hook::SIGINT, e.clone());

    e
  };

  #[cfg(all(unix, feature = "signal-hook"))]
  let mut source =
    Source { limit: cli.limit, input: y4m_dec, count: 0, exit_requested };
  #[cfg(not(all(unix, feature = "signal-hook")))]
  let mut source = Source { limit: cli.limit, input: y4m_dec, count: 0 };

  if video_info.bit_depth == 8 {
    do_encode::<u8, y4m::Decoder<'_, Box<dyn Read>>>(
      cfg,
      cli.verbose,
      progress,
      &mut *cli.io.output,
      &mut source,
      cli.pass1file_name.as_ref(),
      cli.pass2file_name.as_ref(),
      y4m_enc,
    )?
  } else {
    do_encode::<u16, y4m::Decoder<'_, Box<dyn Read>>>(
      cfg,
      cli.verbose,
      progress,
      &mut *cli.io.output,
      &mut source,
      cli.pass1file_name.as_ref(),
      cli.pass2file_name.as_ref(),
      y4m_enc,
    )?
  }
  if cli.benchmark {
    print_rusage();
  }

  Ok(())
}
