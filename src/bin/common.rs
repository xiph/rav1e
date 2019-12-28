// Copyright (c) 2017-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::error::*;
use crate::muxer::{create_muxer, Muxer};
use crate::{ColorPrimaries, MatrixCoefficients, TransferCharacteristics};
use clap::{App, AppSettings, Arg, ArgMatches, Shell, SubCommand};
use rav1e::prelude::*;
use rav1e::version;
use scan_fmt::scan_fmt;

use std::fs::File;
use std::io;
use std::io::prelude::*;

pub struct EncoderIO {
  pub input: Box<dyn Read>,
  pub output: Box<dyn Muxer>,
  pub rec: Option<Box<dyn Write>>,
}

pub struct CliOptions {
  pub io: EncoderIO,
  pub enc: EncoderConfig,
  pub limit: usize,
  pub color_range_specified: bool,
  pub override_time_base: bool,
  pub skip: usize,
  pub verbose: bool,
  pub benchmark: bool,
  pub threads: usize,
  pub pass1file_name: Option<String>,
  pub pass2file_name: Option<String>,
  pub save_config: Option<String>,
}

#[cfg(feature = "serialize")]
fn build_speed_long_help() -> String {
  let levels = (0..=10)
    .map(|speed| {
      let s = SpeedSettings::from_preset(speed);
      let o = crate::kv::to_string(&s).unwrap().replace(", ", "\n    ");
      format!("{:2} :\n    {}", speed, o)
    })
    .collect::<Vec<String>>()
    .join("\n");

  format!(
    "Speed level (0 is best quality, 10 is fastest)\n\
     Speeds 10 and 0 are extremes and are generally not recommended\n\
     {}",
    levels
  )
}

#[cfg(not(feature = "serialize"))]
fn build_speed_long_help() -> String {
  "Speed level (0 is best quality, 10 is fastest)\n\
   Speeds 10 and 0 are extremes and are generally not recommended"
    .into()
}

#[allow(unused_mut)]
pub fn parse_cli() -> Result<CliOptions, CliError> {
  let ver_short = version::short();
  let ver_long = version::full();
  let ver = version::full();
  let speed_long_help = build_speed_long_help();
  let mut app = App::new("rav1e")
    .version(ver.as_str())
    .long_version(ver_long.as_str())
    .version_short(ver_short.as_str())
    .about("AV1 video encoder")
    .setting(AppSettings::DeriveDisplayOrder)
    .setting(AppSettings::SubcommandsNegateReqs)
    .arg(Arg::with_name("FULLHELP")
      .help("Prints more detailed help information")
      .long("fullhelp"))
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
        .required_unless("FULLHELP")
        .index(1)
    )
    .arg(
      Arg::with_name("OUTPUT")
        .help("Compressed AV1 in IVF video output")
        .short("o")
        .long("output")
        .required_unless("FULLHELP")
        .takes_value(true)
    )
    // ENCODING SETTINGS
    .arg(
      Arg::with_name("FIRST_PASS")
        .help("Perform the first pass of a two-pass encode, saving the pass data to the specified file for future passes")
        .long("first-pass")
        .takes_value(true)
    )
    .arg(
      Arg::with_name("SECOND_PASS")
        .help("Perform the second pass of a two-pass encode, reading the pass data saved from a previous pass from the specified file")
        .long("second-pass")
        .takes_value(true)
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
      Arg::with_name("SKIP")
        .help("Skip n number of frames and encode")
        .long("skip")
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
      Arg::with_name("MINQP")
        .help("Minimum quantizer (0-255) to use in bitrate mode [default: 0]")
        .long("min-quantizer")
        .alias("min_quantizer")
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
        .help("Speed level (0 is best quality, 10 is fastest)\n\
        Speeds 10 and 0 are extremes and are generally not recommended")
        .long_help(&speed_long_help)
        .short("s")
        .long("speed")
        .takes_value(true)
        .default_value("6")
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
        .help("Maximum interval between keyframes. When set to 0, disables fixed-interval keyframes.")
        .short("I")
        .long("keyint")
        .takes_value(true)
        .default_value("240")
    )
    .arg(
      Arg::with_name("SWITCH_FRAME_INTERVAL")
        .help("Maximum interval between switch frames. When set to 0, disables switch frames.")
        .short("S")
        .long("switch-frame-interval")
        .takes_value(true)
        .default_value("0")
    )
    .arg(
      Arg::with_name("RESERVOIR_FRAME_DELAY")
        .help("Number of frames over which rate control should distribute the reservoir [default: min(240, 1.5x keyint)]\n\
         A minimum value of 12 is enforced.")
        .long("reservoir-frame-delay")
        .alias("reservoir_frame_delay")
        .takes_value(true)
    )
    .arg(
      Arg::with_name("LOW_LATENCY")
        .help("Low latency mode; disables frame reordering\n\
            Has a significant speed-to-quality trade-off")
        .long("low-latency")
        .alias("low_latency")
    )
    .arg(
      Arg::with_name("RDO_LOOKAHEAD_FRAMES")
        .help("Number of frames encoder should lookahead for RDO purposes [default: 40]\n")
        .long("rdo-lookahead-frames")
        .takes_value(true)
    )
    .arg(
      Arg::with_name("TUNE")
        .help("Quality tuning")
        .long("tune")
        .possible_values(&Tune::variants())
        .default_value("Psychovisual")
        .case_insensitive(true)
    )
    .arg(
      Arg::with_name("TILE_ROWS")
        .help("Number of tile rows. Must be a power of 2. rav1e may override this based on video resolution.")
        .long("tile-rows")
        .takes_value(true)
        .default_value("0")
    )
    .arg(
      Arg::with_name("TILE_COLS")
        .help("Number of tile columns. Must be a power of 2. rav1e may override this based on video resolution.")
        .long("tile-cols")
        .takes_value(true)
        .default_value("0")
    )
    .arg(
      Arg::with_name("TILES")
        .help("Number of tiles. Tile-cols and tile-rows are overridden\n\
               so that the video has at least this many tiles.")
        .long("tiles")
        .takes_value(true)
        .default_value("0")
    )
    // MASTERING
    .arg(
      Arg::with_name("PIXEL_RANGE")
        .help("Pixel range")
        .long("range")
        .possible_values(&PixelRange::variants())
        .default_value("limited")
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
        .long("mastering-display")
        .alias("mastering_display")
        .default_value("unspecified")
        .case_insensitive(true)
    )
    .arg(
      Arg::with_name("CONTENT_LIGHT")
        .help("Content light level used to describe content luminosity (cll,fall)")
        .long("content-light")
        .alias("content_light")
        .default_value("0,0")
        .case_insensitive(true)
    )
    // TIMING INFO
    .arg(
      Arg::with_name("FRAME_RATE")
        .help("Constant frame rate to set at the output (inferred from input when omitted)")
        .long("frame-rate")
        .alias("frame_rate")
        .takes_value(true)
    )
    .arg(
      Arg::with_name("TIME_SCALE")
        .help("The time scale associated with the frame rate if provided (ignored otherwise)")
        .long("time-scale")
        .alias("time_scale")
        .default_value("1")
        .takes_value(true)
    )
    // STILL PICTURE
    .arg(
      Arg::with_name("STILL_PICTURE")
        .help("Still picture mode")
        .long("still-picture")
        .alias("still_picture")
    )
    // DEBUGGING
    .arg(
      Arg::with_name("BENCHMARK")
        .help("Provide a benchmark report at the end of the encoding")
        .long("benchmark")
    )
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
        .long("reconstruction")
        .short("r")
        .takes_value(true)
    )
    .arg(
      Arg::with_name("SPEED_TEST")
        .help("Run an encode using default encoding settings, manually adjusting only the settings specified; allows benchmarking settings in isolation")
        .hidden(true)
        .long("speed-test")
        .takes_value(true)
    )
    .subcommand(SubCommand::with_name("advanced")
                .setting(AppSettings::Hidden)
                .about("Advanced features")
                .arg(Arg::with_name("SHELL")
                     .help("Output to stdout the completion definition for the shell")
                     .short("c")
                     .long("completion")
                     .takes_value(true)
                     .possible_values(&Shell::variants())
                )
                .arg(Arg::with_name("SAVE_CONFIG")
                     .help("Save the current configuration in a toml file")
                     .short("s")
                     .long("save_config")
                     .takes_value(true)
                )
                .arg(Arg::with_name("LOAD_CONFIG")
                     .help("Load the encoder configuration from a toml file")
                     .short("l")
                     .long("load_config")
                     .takes_value(true)
                )
    );

  let matches = app.clone().get_matches();

  if matches.is_present("FULLHELP") {
    app.print_long_help().unwrap();
    std::process::exit(0);
  }

  let threads = matches
    .value_of("THREADS")
    .map(|v| v.parse().expect("Threads must be an integer"))
    .unwrap();

  let mut save_config = None;
  let mut enc = None;

  if let Some(matches) = matches.subcommand_matches("advanced") {
    if let Some(shell) = matches.value_of("SHELL").map(|v| v.parse().unwrap())
    {
      app.gen_completions_to("rav1e", shell, &mut std::io::stdout());
      std::process::exit(0);
    }

    #[cfg(feature = "serialize")]
    {
      save_config = matches.value_of("SAVE_CONFIG").map(|v| v.to_owned());
      if let Some(load_config) = matches.value_of("LOAD_CONFIG") {
        let mut config = String::new();
        File::open(load_config)
          .and_then(|mut f| f.read_to_string(&mut config))
          .map_err(|e| e.context("Cannot open the configuration file"))?;

        enc = Some(toml::from_str(&config).unwrap());
      }
    }
    #[cfg(not(feature = "serialize"))]
    {
      if matches.value_of("SAVE_CONFIG").is_some()
        || matches.value_of("LOAD_CONFIG").is_some()
      {
        let e: io::Error = io::ErrorKind::InvalidInput.into();
        return Err(e.context(
          "The load/save config advanced option requires the
            `serialize` feature, rebuild adding it.",
        ));
      }
    }
  }

  let rec = match matches.value_of("RECONSTRUCTION") {
    Some(f) => Some(Box::new(
      File::create(&f)
        .map_err(|e| e.context("Cannot create reconstruction file"))?,
    ) as Box<dyn Write>),
    None => None,
  };

  let io = EncoderIO {
    input: match matches.value_of("INPUT").unwrap() {
      "-" => Box::new(io::stdin()) as Box<dyn Read>,
      f => Box::new(
        File::open(&f).map_err(|e| e.context("Cannot open input file"))?,
      ) as Box<dyn Read>,
    },
    output: create_muxer(matches.value_of("OUTPUT").unwrap())?,
    rec,
  };

  let enc = enc.map_or_else(|| parse_config(&matches), Ok)?;

  Ok(CliOptions {
    io,
    enc,
    limit: matches.value_of("LIMIT").unwrap().parse().unwrap(),
    // Use `occurrences_of()` because `is_present()` is always true
    // if a parameter has a default value.
    color_range_specified: matches.occurrences_of("PIXEL_RANGE") > 0,
    override_time_base: matches.is_present("FRAME_RATE"),
    skip: matches.value_of("SKIP").unwrap().parse().unwrap(),
    benchmark: matches.is_present("BENCHMARK"),
    verbose: matches.is_present("VERBOSE"),
    threads,
    pass1file_name: matches.value_of("FIRST_PASS").map(|s| s.to_owned()),
    pass2file_name: matches.value_of("SECOND_PASS").map(|s| s.to_owned()),
    save_config,
  })
}

pub trait MatchGet {
  fn value_of_int(&self, name: &str) -> Option<Result<i32, CliError>>;
}

impl MatchGet for ArgMatches<'_> {
  fn value_of_int(&self, name: &str) -> Option<Result<i32, CliError>> {
    self
      .value_of(name)
      .map(|v| v.parse().map_err(|e: std::num::ParseIntError| e.context(name)))
  }
}

fn parse_config(matches: &ArgMatches<'_>) -> Result<EncoderConfig, CliError> {
  let maybe_quantizer = matches.value_of_int("QP");
  let maybe_bitrate = matches.value_of_int("BITRATE");
  let quantizer = maybe_quantizer.unwrap_or_else(|| {
    if maybe_bitrate.is_some() {
      // If a bitrate is specified, the quantizer is the maximum allowed (e.g.,
      //  the minimum quality allowed), which by default should be
      //  unconstrained.
      Ok(255)
    } else {
      Ok(100)
    }
  })? as usize;
  let bitrate: i32 = maybe_bitrate.unwrap_or(Ok(0))?;
  if quantizer == 0 {
    unimplemented!("Lossless encoding not yet implemented");
  } else if quantizer > 255 {
    panic!("Quantizer must be between 0-255");
  }

  let mut cfg = if let Some(settings) = matches.value_of("SPEED_TEST") {
    info!("Running in speed test mode--ignoring other settings");
    let mut cfg = EncoderConfig::default();
    settings
      .split_whitespace()
      .for_each(|setting| apply_speed_test_cfg(&mut cfg, setting));
    cfg
  } else {
    let speed = matches.value_of("SPEED").unwrap().parse().unwrap();
    let max_interval: u64 =
      matches.value_of("KEYFRAME_INTERVAL").unwrap().parse().unwrap();
    let mut min_interval: u64 =
      matches.value_of("MIN_KEYFRAME_INTERVAL").unwrap().parse().unwrap();

    if matches.occurrences_of("MIN_KEYFRAME_INTERVAL") == 0 {
      min_interval = min_interval.min(max_interval);
    }

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
    cfg.set_key_frame_interval(min_interval, max_interval);
    cfg.switch_frame_interval =
      matches.value_of("SWITCH_FRAME_INTERVAL").unwrap().parse().unwrap();

    cfg.pixel_range =
      matches.value_of("PIXEL_RANGE").unwrap().parse().unwrap_or_default();
    cfg.color_description = if color_primaries == ColorPrimaries::Unspecified
      && transfer_characteristics == TransferCharacteristics::Unspecified
      && matrix_coefficients == MatrixCoefficients::Unspecified
    {
      // No need to set a color description with all parameters unspecified.
      None
    } else {
      Some(ColorDescription {
        color_primaries,
        transfer_characteristics,
        matrix_coefficients,
      })
    };

    let mastering_display_opt = matches.value_of("MASTERING_DISPLAY").unwrap();
    cfg.mastering_display = if mastering_display_opt == "unspecified" {
      None
    } else {
      let (g_x, g_y, b_x, b_y, r_x, r_y, wp_x, wp_y, max_lum, min_lum) =
        scan_fmt!(
          mastering_display_opt,
          "G({},{})B({},{})R({},{})WP({},{})L({},{})",
          f64,
          f64,
          f64,
          f64,
          f64,
          f64,
          f64,
          f64,
          f64,
          f64
        )
        .expect("Cannot parse the mastering display option");
      Some(MasteringDisplay {
        primaries: [
          ChromaticityPoint {
            x: (r_x * ((1 << 16) as f64)).round() as u16,
            y: (r_y * ((1 << 16) as f64)).round() as u16,
          },
          ChromaticityPoint {
            x: (g_x * ((1 << 16) as f64)).round() as u16,
            y: (g_y * ((1 << 16) as f64)).round() as u16,
          },
          ChromaticityPoint {
            x: (b_x * ((1 << 16) as f64)).round() as u16,
            y: (b_y * ((1 << 16) as f64)).round() as u16,
          },
        ],
        white_point: ChromaticityPoint {
          x: (wp_x * ((1 << 16) as f64)).round() as u16,
          y: (wp_y * ((1 << 16) as f64)).round() as u16,
        },
        max_luminance: (max_lum * ((1 << 8) as f64)).round() as u32,
        min_luminance: (min_lum * ((1 << 14) as f64)).round() as u32,
      })
    };

    let content_light_opt = matches.value_of("CONTENT_LIGHT").unwrap();
    let (cll, fall) = scan_fmt!(content_light_opt, "{},{}", u16, u16)
      .expect("Cannot parse the content light option");
    cfg.content_light = if cll == 0 && fall == 0 {
      None
    } else {
      Some(ContentLight {
        max_content_light_level: cll,
        max_frame_average_light_level: fall,
      })
    };
    cfg
  };

  cfg.still_picture = matches.is_present("STILL_PICTURE");

  cfg.quantizer = quantizer;
  cfg.min_quantizer =
    matches.value_of("MINQP").unwrap_or("0").parse().unwrap();
  cfg.bitrate = bitrate.checked_mul(1000).expect("Bitrate too high");
  cfg.reservoir_frame_delay = matches
    .value_of("RESERVOIR_FRAME_DELAY")
    .map(|reservior_frame_delay| reservior_frame_delay.parse().unwrap());
  cfg.rdo_lookahead_frames =
    matches.value_of("RDO_LOOKAHEAD_FRAMES").unwrap_or("40").parse().unwrap();
  cfg.show_psnr = matches.is_present("PSNR");
  cfg.tune = matches.value_of("TUNE").unwrap().parse().unwrap();

  if cfg.tune == Tune::Psychovisual {
    cfg.speed_settings.tx_domain_distortion = false;
  }

  cfg.tile_cols = matches.value_of("TILE_COLS").unwrap().parse().unwrap();
  cfg.tile_rows = matches.value_of("TILE_ROWS").unwrap().parse().unwrap();

  cfg.tiles = matches.value_of("TILES").unwrap().parse().unwrap();

  if cfg.tile_cols > 64 || cfg.tile_rows > 64 {
    panic!("Tile columns and rows may not be greater than 64");
  }

  if let Some(frame_rate) = matches.value_of("FRAME_RATE") {
    cfg.time_base = Rational::new(
      matches.value_of("TIME_SCALE").unwrap().parse().unwrap(),
      frame_rate.parse().unwrap(),
    );
  }

  cfg.low_latency = matches.is_present("LOW_LATENCY");

  Ok(cfg)
}

fn apply_speed_test_cfg(cfg: &mut EncoderConfig, setting: &str) {
  match setting {
    "baseline" => {
      cfg.speed_settings = SpeedSettings::default();
    }
    "min_block_size_8x8" => {
      cfg.speed_settings.partition_range =
        PartitionRange::new(BlockSize::BLOCK_8X8, BlockSize::BLOCK_64X64);
    }
    "min_block_size_16x16" => {
      cfg.speed_settings.partition_range =
        PartitionRange::new(BlockSize::BLOCK_16X16, BlockSize::BLOCK_64X64);
    }
    "min_block_size_32x32" => {
      cfg.speed_settings.partition_range =
        PartitionRange::new(BlockSize::BLOCK_32X32, BlockSize::BLOCK_64X64);
    }
    "min_block_size_64x64" => {
      cfg.speed_settings.partition_range =
        PartitionRange::new(BlockSize::BLOCK_64X64, BlockSize::BLOCK_64X64);
    }
    "no_multiref" => {
      cfg.speed_settings.multiref = false;
    }
    "fast_deblock" => {
      cfg.speed_settings.fast_deblock = true;
    }
    "reduced_tx_set" => {
      cfg.speed_settings.reduced_tx_set = true;
    }
    "tx_domain_distortion" => {
      cfg.speed_settings.tx_domain_distortion = true;
    }
    "tx_domain_rate" => {
      cfg.speed_settings.tx_domain_rate = true;
    }
    "encode_topdown" => {
      cfg.speed_settings.encode_bottomup = false;
    }
    "no_rdo_tx_decision" => {
      cfg.speed_settings.rdo_tx_decision = false;
    }
    "prediction_modes_simple" => {
      cfg.speed_settings.prediction_modes = PredictionModesSetting::Simple;
    }
    "prediction_modes_keyframes" => {
      cfg.speed_settings.prediction_modes =
        PredictionModesSetting::ComplexKeyframes;
    }
    "exclude_near_mvs" => {
      cfg.speed_settings.include_near_mvs = false;
    }
    "no_scene_detection" => {
      cfg.speed_settings.no_scene_detection = true;
    }
    "fast_scene_detection" => {
      cfg.speed_settings.fast_scene_detection = true;
    }
    "full_search_me" => {
      cfg.speed_settings.diamond_me = false;
    }
    "no_cdef" => {
      cfg.speed_settings.cdef = false;
    }
    setting => {
      panic!("Unrecognized speed test setting {}", setting);
    }
  };
}
