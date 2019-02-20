// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::{ColorPrimaries, MatrixCoefficients, TransferCharacteristics};
use clap::{App, AppSettings, Arg, ArgMatches, SubCommand, Shell};
use rav1e::partition::BlockSize;
use rav1e::*;

use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;
use std::time::Instant;
use std::{fmt, io};

pub struct EncoderIO {
  pub input: Box<dyn Read>,
  pub output: Box<dyn Write>,
  pub rec: Option<Box<dyn Write>>
}

pub struct CliOptions {
  pub io: EncoderIO,
  pub enc: EncoderConfig,
  pub limit: usize,
  pub verbose: bool,
}

pub fn parse_cli() -> CliOptions {
  let mut app = App::new("rav1e")
    .version(env!("CARGO_PKG_VERSION"))
    .about("AV1 video encoder")
    .setting(AppSettings::DeriveDisplayOrder)
    .setting(AppSettings::SubcommandsNegateReqs)
    // THREADS
    .arg(
      Arg::with_name("THREADS")
        .help("Set the threadpool size")
        .long("threads")
        .takes_value(true)
        .default_value("0")
    )
    // INPUT/OUTPUT
    .arg(
      Arg::with_name("INPUT")
        .help("Uncompressed YUV4MPEG2 video input")
        .required(true)
        .index(1)
    )
    .arg(
      Arg::with_name("OUTPUT")
        .help("Compressed AV1 in IVF video output")
        .short("o")
        .long("output")
        .required(true)
        .takes_value(true)
    )
    .arg(
      Arg::with_name("STATS_FILE")
        .help("Custom location for first-pass stats file")
        .long("stats")
        .takes_value(true)
        .default_value("rav1e_stats.json")
    )
    // ENCODING SETTINGS
    .arg(
      Arg::with_name("PASS")
        .help("Specify first-pass or second-pass to run as a two-pass encode; If not provided, will run a one-pass encode")
        .short("p")
        .long("pass")
        .takes_value(true)
        .possible_values(&["1", "2"])
    )
    .arg(
      Arg::with_name("LIMIT")
        .help("Maximum number of frames to encode")
        .short("l")
        .long("limit")
        .takes_value(true)
        .default_value("0")
    )
    .arg(
      Arg::with_name("QP")
        .help("Quantizer (0-255), smaller values are higher quality [default: 100]")
        .long("quantizer")
        .takes_value(true)
    )
    .arg(
      Arg::with_name("BITRATE")
        .help("Bitrate (kbps)")
        .short("b")
        .long("bitrate")
        .takes_value(true)
    )
    .arg(
      Arg::with_name("SPEED")
        .help("Speed level (0 is best quality, 10 is fastest)")
        .short("s")
        .long("speed")
        .takes_value(true)
        .default_value("3")
    )
    .arg(
      Arg::with_name("MIN_KEYFRAME_INTERVAL")
        .help("Minimum interval between keyframes")
        .short("i")
        .long("min-keyint")
        .takes_value(true)
        .default_value("12")
    )
    .arg(
      Arg::with_name("KEYFRAME_INTERVAL")
        .help("Maximum interval between keyframes")
        .short("I")
        .long("keyint")
        .takes_value(true)
        .default_value("240")
    )
    .arg(
      Arg::with_name("LOW_LATENCY")
        .help("Low latency mode; disables frame reordering")
        .long("low_latency")
        .takes_value(true)
        .default_value("false")
    )
    .arg(
      Arg::with_name("TUNE")
        .help("Quality tuning")
        .long("tune")
        .possible_values(&Tune::variants())
        .default_value("Psychovisual")
        .case_insensitive(true)
    )
    // MASTERING
    .arg(
      Arg::with_name("PIXEL_RANGE")
        .help("Pixel range")
        .long("range")
        .possible_values(&PixelRange::variants())
        .default_value("unspecified")
        .case_insensitive(true)
    )
    .arg(
      Arg::with_name("COLOR_PRIMARIES")
        .help("Color primaries used to describe color parameters")
        .long("primaries")
        .possible_values(&ColorPrimaries::variants())
        .default_value("unspecified")
        .case_insensitive(true)
    )
    .arg(
      Arg::with_name("TRANSFER_CHARACTERISTICS")
        .help("Transfer characteristics used to describe color parameters")
        .long("transfer")
        .possible_values(&TransferCharacteristics::variants())
        .default_value("unspecified")
        .case_insensitive(true)
    )
    .arg(
      Arg::with_name("MATRIX_COEFFICIENTS")
        .help("Matrix coefficients used to describe color parameters")
        .long("matrix")
        .possible_values(&MatrixCoefficients::variants())
        .default_value("unspecified")
        .case_insensitive(true)
    )
    .arg(
      Arg::with_name("MASTERING_DISPLAY")
        .help("Mastering display primaries in the form of G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)")
        .long("mastering_display")
        .default_value("unspecified")
        .case_insensitive(true)
    )
    .arg(
      Arg::with_name("CONTENT_LIGHT")
        .help("Content light level used to describe content luminosity (cll,fall)")
        .long("content_light")
        .default_value("0,0")
        .case_insensitive(true)
    )
    // DEBUGGING
    .arg(
      Arg::with_name("VERBOSE")
        .help("Verbose logging; outputs info for every frame")
        .long("verbose")
        .short("v")
    )
    .arg(
      Arg::with_name("PSNR")
        .help("Calculate and display PSNR metrics")
        .long("psnr")
    )
    .arg(
      Arg::with_name("RECONSTRUCTION")
        .help("Outputs a Y4M file containing the output from the decoder")
        .short("r")
        .takes_value(true)
    )
    .arg(
      Arg::with_name("SPEED_TEST")
        .help("Run an encode using default encoding settings, manually adjusting only the settings specified; allows benchmarking settings in isolation")
        .hidden(true)
        .long("speed-test")
        .takes_value(true)
        .possible_values(&[
          "baseline",
          "min_block_size_4x4",
          "min_block_size_8x8",
          "min_block_size_32x32",
          "min_block_size_64x64",
          "multiref",
          "fast_deblock",
          "reduced_tx_set",
          "tx_domain_distortion",
          "encode_bottomup",
          "rdo_tx_decision",
          "prediction_modes_keyframes",
          "prediction_modes_all",
          "include_near_mvs",
          "no_scene_detection",
        ])
    )
    .subcommand(SubCommand::with_name("advanced")
                .about("Advanced features")
                .arg(Arg::with_name("SHELL")
                     .help("Output to stdout the completion definition for the shell")
                     .short("c")
                     .long("completion")
                     .takes_value(true)
                     .possible_values(&Shell::variants())
                )
    );

  let matches = app.clone().get_matches();

  if let Some(threads) = matches.value_of("THREADS").map(|v| v.parse().expect("Threads must be an integer")) {
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
  }

  if let Some(matches) = matches.subcommand_matches("advanced") {
    if let Some(shell) = matches.value_of("SHELL").map(|v| v.parse().unwrap()) {
      app.gen_completions_to("rav1e", shell, &mut std::io::stdout());
      std::process::exit(0);
    }
  }

  let io = EncoderIO {
    input: match matches.value_of("INPUT").unwrap() {
      "-" => Box::new(io::stdin()) as Box<dyn Read>,
      f => Box::new(File::open(&f).unwrap()) as Box<dyn Read>
    },
    output: match matches.value_of("OUTPUT").unwrap() {
      "-" => Box::new(io::stdout()) as Box<dyn Write>,
      f => Box::new(File::create(&f).unwrap()) as Box<dyn Write>
    },
    rec: matches
      .value_of("RECONSTRUCTION")
      .map(|f| Box::new(File::create(&f).unwrap()) as Box<dyn Write>)
  };

  CliOptions {
    io,
    enc: parse_config(&matches),
    limit: matches.value_of("LIMIT").unwrap().parse().unwrap(),
    verbose: matches.is_present("VERBOSE"),
  }
}

fn parse_config(matches: &ArgMatches<'_>) -> EncoderConfig {
  let maybe_quantizer = matches.value_of("QP").map(|qp| qp.parse().unwrap());
  let maybe_bitrate =
    matches.value_of("BITRATE").map(|bitrate| bitrate.parse().unwrap());
  let quantizer = maybe_quantizer.unwrap_or_else(|| {
    if maybe_bitrate.is_some() {
      // If a bitrate is specified, the quantizer is the maximum allowed (e.g.,
      //  the minimum quality allowed), which by default should be
      //  unconstrained.
      255
    } else {
      100
    }
  });
  let bitrate = maybe_bitrate.unwrap_or(0);
  if quantizer == 0 {
    unimplemented!("Lossless encoding not yet implemented");
  } else if quantizer > 255 {
    panic!("Quantizer must be between 0-255");
  }

  let mut cfg = if let Some(settings) = matches.value_of("SPEED_TEST") {
    eprintln!("Running in speed test mode--ignoring other settings");
    let mut cfg = EncoderConfig::default();
    settings
      .split_whitespace()
      .for_each(|setting| apply_speed_test_cfg(&mut cfg, setting));
    cfg
  } else {
    let speed = matches.value_of("SPEED").unwrap().parse().unwrap();
    let max_interval: u64 = matches.value_of("KEYFRAME_INTERVAL").unwrap().parse().unwrap();
    let mut min_interval: u64 = matches.value_of("MIN_KEYFRAME_INTERVAL").unwrap().parse().unwrap();

    if matches.occurrences_of("MIN_KEYFRAME_INTERVAL") == 0 {
      min_interval = min_interval.min(max_interval);
    }

    // Validate arguments
    if speed > 10 {
      panic!("Speed must be between 0-10");
    } else if min_interval > max_interval {
      panic!("Maximum keyframe interval must be greater than or equal to minimum keyframe interval");
    }

    let color_primaries =
      matches.value_of("COLOR_PRIMARIES").unwrap().parse().unwrap_or_default();
    let transfer_characteristics = matches
      .value_of("TRANSFER_CHARACTERISTICS")
      .unwrap()
      .parse()
      .unwrap_or_default();
    let matrix_coefficients = matches
      .value_of("MATRIX_COEFFICIENTS")
      .unwrap()
      .parse()
      .unwrap_or_default();

    let mut cfg = EncoderConfig::with_speed_preset(speed);
    cfg.max_key_frame_interval = min_interval;
    cfg.max_key_frame_interval = max_interval;

    cfg.pixel_range = matches.value_of("PIXEL_RANGE").unwrap().parse().unwrap_or_default();
    cfg.color_description = if color_primaries == ColorPrimaries::Unspecified &&
      transfer_characteristics == TransferCharacteristics::Unspecified &&
      matrix_coefficients == MatrixCoefficients::Unspecified {
      // No need to set a color description with all parameters unspecified.
      None
    } else {
      Some(ColorDescription {
        color_primaries,
        transfer_characteristics,
        matrix_coefficients
      })
    };

    let mastering_display_opt = matches.value_of("MASTERING_DISPLAY").unwrap();
    cfg.mastering_display = if mastering_display_opt == "unspecified" { None } else {
      let (g_x, g_y, b_x, b_y, r_x, r_y, wp_x, wp_y, max_lum, min_lum) = scan_fmt!(mastering_display_opt, "G({},{})B({},{})R({},{})WP({},{})L({},{})", f64, f64, f64, f64, f64, f64, f64, f64, f64, f64);
      Some(MasteringDisplay {
        primaries: [
          Point {
            x: (r_x.unwrap() * ((1 << 16) as f64)).round() as u16,
            y: (r_y.unwrap() * ((1 << 16) as f64)).round() as u16,
          },
          Point {
            x: (g_x.unwrap() * ((1 << 16) as f64)).round() as u16,
            y: (g_y.unwrap() * ((1 << 16) as f64)).round() as u16,
          },
          Point {
            x: (b_x.unwrap() * ((1 << 16) as f64)).round() as u16,
            y: (b_y.unwrap() * ((1 << 16) as f64)).round() as u16,
          }
        ],
        white_point: Point {
          x: (wp_x.unwrap() * ((1 << 16) as f64)).round() as u16,
          y: (wp_y.unwrap() * ((1 << 16) as f64)).round() as u16,
        },
        max_luminance: (max_lum.unwrap() * ((1 << 8) as f64)).round() as u32,
        min_luminance: (min_lum.unwrap() * ((1 << 14) as f64)).round() as u32,
      })
    };

    let content_light_opt = matches.value_of("CONTENT_LIGHT").unwrap();
    let (cll, fall) = scan_fmt!(content_light_opt, "{},{}", u16, u16);
    cfg.content_light = if cll.unwrap() == 0 && fall.unwrap() == 0 { None } else {
      Some(ContentLight {
        max_content_light_level: cll.unwrap(),
        max_frame_average_light_level: fall.unwrap()
      })
    };
    cfg
  };

  cfg.quantizer = quantizer;
  cfg.bitrate = bitrate;
  cfg.show_psnr = matches.is_present("PSNR");
  cfg.pass = matches.value_of("PASS").map(|pass| pass.parse().unwrap());
  cfg.stats_file = if cfg.pass.is_some() {
    Some(PathBuf::from(matches.value_of("STATS_FILE").unwrap()))
  } else {
    None
  };
  cfg.tune = matches.value_of("TUNE").unwrap().parse().unwrap();
  cfg.low_latency = matches.value_of("LOW_LATENCY").unwrap().parse().unwrap();

  cfg
}

fn apply_speed_test_cfg(cfg: &mut EncoderConfig, setting: &str) {
  match setting {
    "baseline" => {
      // Use default settings
    },
    "min_block_size_4x4" => {
      cfg.speed_settings.min_block_size = BlockSize::BLOCK_4X4;
    },
    "min_block_size_8x8" => {
      cfg.speed_settings.min_block_size = BlockSize::BLOCK_8X8;
    },
    "min_block_size_32x32" => {
      cfg.speed_settings.min_block_size = BlockSize::BLOCK_32X32;
    },
    "min_block_size_64x64" => {
      cfg.speed_settings.min_block_size = BlockSize::BLOCK_64X64;
    },
    "multiref" => {
      cfg.speed_settings.multiref = true;
    },
    "fast_deblock" => {
      cfg.speed_settings.fast_deblock = true;
    },
    "reduced_tx_set" => {
      cfg.speed_settings.reduced_tx_set = true;
    },
    "tx_domain_distortion" => {
      cfg.speed_settings.tx_domain_distortion = true;
    },
    "encode_bottomup" => {
      cfg.speed_settings.encode_bottomup = true;
    },
    "rdo_tx_decision" => {
      cfg.speed_settings.rdo_tx_decision = true;
    },
    "prediction_modes_keyframes" => {
      cfg.speed_settings.prediction_modes = PredictionModesSetting::ComplexKeyframes;
    },
    "prediction_modes_all" => {
      cfg.speed_settings.prediction_modes = PredictionModesSetting::ComplexAll;
    },
    "include_near_mvs" => {
      cfg.speed_settings.include_near_mvs = true;
    },
    "no_scene_detection" => {
      cfg.speed_settings.no_scene_detection = true;
    },
    setting => {
      panic!("Unrecognized speed test setting {}", setting);
    }
  };
}

#[derive(Debug, Clone, Copy)]
pub struct FrameSummary {
  // Frame size in bytes
  pub size: usize,
  pub number: u64,
  pub frame_type: FrameType,
  // PSNR for Y, U, and V planes
  pub psnr: Option<(f64, f64, f64)>,
}

impl From<Packet> for FrameSummary {
  fn from(packet: Packet) -> Self {
    Self {
      size: packet.data.len(),
      number: packet.number,
      frame_type: packet.frame_type,
      psnr: packet.psnr,
    }
  }
}

impl fmt::Display for FrameSummary {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "Frame {} - {} - {} bytes{}",
      self.number,
      self.frame_type,
      self.size,
      if let Some(psnr) = self.psnr {
        format!(" - PSNR: Y: {:.4}  Cb: {:.4}  Cr: {:.4}", psnr.0, psnr.1, psnr.2)
      } else { String::new() }
    )
  }
}

#[derive(Debug, Clone)]
pub struct ProgressInfo {
  // Frame rate of the video
  frame_rate: Rational,
  // The length of the whole video, in frames, if known
  total_frames: Option<usize>,
  // The time the encode was started
  time_started: Instant,
  // List of frames encoded so far
  frame_info: Vec<FrameSummary>,
  // Video size so far in bytes.
  //
  // This value will be updated in the CLI very frequently, so we cache the previous value
  // to reduce the overall complexity.
  encoded_size: usize,
  // Whether to display PSNR statistics during and at end of encode
  show_psnr: bool,
}

impl ProgressInfo {
  pub fn new(frame_rate: Rational, total_frames: Option<usize>, show_psnr: bool) -> Self {
    Self {
      frame_rate,
      total_frames,
      time_started: Instant::now(),
      frame_info: Vec::with_capacity(total_frames.unwrap_or_default()),
      encoded_size: 0,
      show_psnr,
    }
  }

  pub fn add_frame(&mut self, frame: FrameSummary) {
    self.encoded_size += frame.size;
    self.frame_info.push(frame);
  }

  pub fn frames_encoded(&self) -> usize {
    self.frame_info.len()
  }

  pub fn encoding_fps(&self) -> f64 {
    let duration = Instant::now().duration_since(self.time_started);
    self.frame_info.len() as f64 / (duration.as_secs() as f64 + duration.subsec_millis() as f64 / 1000f64)
  }

  pub fn video_fps(&self) -> f64 {
    self.frame_rate.num as f64 / self.frame_rate.den as f64
  }

  // Returns the bitrate of the frames so far, in bits/second
  pub fn bitrate(&self) -> usize {
    let bits = self.encoded_size * 8;
    let seconds = self.frame_info.len() as f64 / self.video_fps();
    (bits as f64 / seconds) as usize
  }

  // Estimates the final filesize in bytes, if the number of frames is known
  pub fn estimated_size(&self) -> usize {
    self.total_frames
      .map(|frames| self.encoded_size * frames / self.frames_encoded())
      .unwrap_or_default()
  }

  // Estimates the remaining encoding time in seconds, if the number of frames is known
  pub fn estimated_time(&self) -> f64 {
    self.total_frames
      .map(|frames| (frames - self.frames_encoded()) as f64 / self.encoding_fps())
      .unwrap_or_default()
  }

  // Number of frames of given type which appear in the video
  pub fn get_frame_type_count(&self, frame_type: FrameType) -> usize {
    self.frame_info.iter()
      .filter(|frame| frame.frame_type == frame_type)
      .count()
  }

  // Size in bytes of all frames of given frame type
  pub fn get_frame_type_size(&self, frame_type: FrameType) -> usize {
    self.frame_info.iter()
      .filter(|frame| frame.frame_type == frame_type)
      .map(|frame| frame.size)
      .sum()
  }

  pub fn print_summary(&self) -> String {
    let (key, key_size) = (
      self.get_frame_type_count(FrameType::KEY),
      self.get_frame_type_size(FrameType::KEY)
    );
    let (inter, inter_size) = (
      self.get_frame_type_count(FrameType::INTER),
      self.get_frame_type_size(FrameType::INTER)
    );
    let (ionly, ionly_size) = (
      self.get_frame_type_count(FrameType::INTRA_ONLY),
      self.get_frame_type_size(FrameType::INTRA_ONLY)
    );
    let (switch, switch_size) = (
      self.get_frame_type_count(FrameType::SWITCH),
      self.get_frame_type_size(FrameType::SWITCH)
    );
    format!("\
    Key Frames: {:>6}    avg size: {:>7} B\n\
    Inter:      {:>6}    avg size: {:>7} B\n\
    Intra Only: {:>6}    avg size: {:>7} B\n\
    Switch:     {:>6}    avg size: {:>7} B\
    {}",
      key, key_size / key,
      inter, inter_size.checked_div(inter).unwrap_or(0),
      ionly, ionly_size / key,
      switch, switch_size / key,
      if self.show_psnr {
        let psnr_y =
          self.frame_info.iter().map(|fi| fi.psnr.unwrap().0).sum::<f64>()
            / self.frame_info.len() as f64;
        let psnr_u =
          self.frame_info.iter().map(|fi| fi.psnr.unwrap().1).sum::<f64>()
            / self.frame_info.len() as f64;
        let psnr_v =
          self.frame_info.iter().map(|fi| fi.psnr.unwrap().2).sum::<f64>()
            / self.frame_info.len() as f64;
        format!("\nMean PSNR: Y: {:.4}  Cb: {:.4}  Cr: {:.4}  Avg: {:.4}",
                psnr_y, psnr_u, psnr_v,
                (psnr_y + psnr_u + psnr_v) / 3.0)
      } else { String::new() }
    )
  }
}

impl fmt::Display for ProgressInfo {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if let Some(total_frames) = self.total_frames {
      write!(
        f,
        "encoded {}/{} frames, {:.3} fps, {:.2} Kb/s, est. size: {:.2} MB, est. time: {:.0} s",
        self.frames_encoded(),
        total_frames,
        self.encoding_fps(),
        self.bitrate() as f64 / 1024f64,
        self.estimated_size() as f64 / (1024 * 1024) as f64,
        self.estimated_time()
      )
    } else {
      write!(
        f,
        "encoded {} frames, {:.3} fps, {:.2} Kb/s",
        self.frames_encoded(),
        self.encoding_fps(),
        self.bitrate() as f64 / 1024f64
      )
    }
  }
}
