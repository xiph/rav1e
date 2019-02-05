// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use clap::{App, Arg, ArgMatches};
use {ColorPrimaries, TransferCharacteristics, MatrixCoefficients};
use rav1e::*;

use std::{fmt, io, slice};
use std::fs::File;
use std::io::prelude::*;
use std::sync::Arc;
use std::time::Instant;
use y4m;
use decoder::Decoder;

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
  let matches = App::new("rav1e")
    .version("0.1.0")
    .about("AV1 video encoder")
    .arg(
      Arg::with_name("INPUT")
        .help("Uncompressed YUV4MPEG2 video input")
        .required(true)
        .index(1)
    ).arg(
      Arg::with_name("OUTPUT")
        .help("Compressed AV1 in IVF video output")
        .short("o")
        .long("output")
        .required(true)
        .takes_value(true)
    ).arg(
      Arg::with_name("RECONSTRUCTION")
        .short("r")
        .takes_value(true)
    ).arg(
      Arg::with_name("LIMIT")
        .help("Maximum number of frames to encode")
        .short("l")
        .long("limit")
        .takes_value(true)
        .default_value("0")
    ).arg(
      Arg::with_name("QP")
        .help("Quantizer (0-255)")
        .long("quantizer")
        .takes_value(true)
        .default_value("100")
    ).arg(
      Arg::with_name("SPEED")
        .help("Speed level (0(slow)-10(fast))")
        .short("s")
        .long("speed")
        .takes_value(true)
        .default_value("3")
    ).arg(
      Arg::with_name("MIN_KEYFRAME_INTERVAL")
        .help("Minimum interval between keyframes")
        .short("i")
        .long("min-keyint")
        .takes_value(true)
        .default_value("12")
    ).arg(
      Arg::with_name("KEYFRAME_INTERVAL")
        .help("Maximum interval between keyframes")
        .short("I")
        .long("keyint")
        .takes_value(true)
        .default_value("240")
    ).arg(
      Arg::with_name("LOW_LATENCY")
        .help("low latency mode. true or [false]")
        .long("low_latency")
        .takes_value(true)
        .default_value("false")
    ).arg(
      Arg::with_name("TUNE")
        .help("Quality tuning (Will enforce partition sizes >= 8x8)")
        .long("tune")
        .possible_values(&Tune::variants())
        .default_value("psnr")
        .case_insensitive(true)
    ).arg(
      Arg::with_name("PIXEL_RANGE")
      .help("Pixel range")
      .long("range")
      .possible_values(&PixelRange::variants())
      .default_value("unspecified")
      .case_insensitive(true)
    ).arg(
      Arg::with_name("COLOR_PRIMARIES")
      .help("Color primaries used to describe color parameters.")
      .long("primaries")
      .possible_values(&ColorPrimaries::variants())
      .default_value("unspecified")
      .case_insensitive(true)
    ).arg(
      Arg::with_name("TRANSFER_CHARACTERISTICS")
      .help("Transfer characteristics used to describe color parameters.")
      .long("transfer")
      .possible_values(&TransferCharacteristics::variants())
      .default_value("unspecified")
      .case_insensitive(true)
    ).arg(
      Arg::with_name("MATRIX_COEFFICIENTS")
      .help("Matrix coefficients used to describe color parameters.")
      .long("matrix")
      .possible_values(&MatrixCoefficients::variants())
      .default_value("unspecified")
      .case_insensitive(true)
    ).arg(
      Arg::with_name("MASTERING_DISPLAY")
      .help("Mastering display primaries in the form of G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min).")
      .long("mastering_display")
      .default_value("unspecified")
      .case_insensitive(true)
    ).arg(
      Arg::with_name("CONTENT_LIGHT")
      .help("Content light level used to describe content luminosity (cll,fall).")
      .long("content_light")
      .default_value("0,0")
      .case_insensitive(true)
    ).arg(
      Arg::with_name("VERBOSE")
        .help("verbose logging, output info for every frame")
        .long("verbose")
        .short("v")
    ).arg(
      Arg::with_name("PSNR")
        .help("calculate and display PSNR metrics")
        .long("psnr")
    ).get_matches();

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

fn parse_config(matches: &ArgMatches) -> EncoderConfig {
  let speed = matches.value_of("SPEED").unwrap().parse().unwrap();
  let quantizer = matches.value_of("QP").unwrap().parse().unwrap();
  let min_interval = matches.value_of("MIN_KEYFRAME_INTERVAL").unwrap().parse().unwrap();
  let max_interval = matches.value_of("KEYFRAME_INTERVAL").unwrap().parse().unwrap();

  // Validate arguments
  if quantizer == 0 {
    unimplemented!("Lossless encoding not yet implemented");
  } else if quantizer > 255 || speed > 10 {
    panic!("argument out of range");
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
  cfg.low_latency = matches.value_of("LOW_LATENCY").unwrap().parse().unwrap();
  cfg.tune = matches.value_of("TUNE").unwrap().parse().unwrap();

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
  cfg.quantizer = quantizer;
  cfg.show_psnr = matches.is_present("PSNR");

  let mastering_display_opt = matches.value_of("MASTERING_DISPLAY").unwrap();
  cfg.mastering_display = if mastering_display_opt == "unspecified" { None } else {
    let (g_x, g_y, b_x, b_y, r_x, r_y, wp_x, wp_y, max_lum, min_lum) = scan_fmt!(mastering_display_opt, "G({},{})B({},{})R({},{})WP({},{})L({},{})", f64, f64, f64, f64, f64, f64, f64, f64, f64, f64);
    Some(MasteringDisplay {
      primaries: [
        Point{
          x: (r_x.unwrap() * ((1 << 16) as f64)).round() as u16,
          y: (r_y.unwrap() * ((1 << 16) as f64)).round() as u16,
        },
        Point{
          x: (g_x.unwrap() * ((1 << 16) as f64)).round() as u16,
          y: (g_y.unwrap() * ((1 << 16) as f64)).round() as u16,
        },
        Point{
          x: (b_x.unwrap() * ((1 << 16) as f64)).round() as u16,
          y: (b_y.unwrap() * ((1 << 16) as f64)).round() as u16,
        }
      ],
      white_point: Point{
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
      max_content_light_level : cll.unwrap(),
      max_frame_average_light_level : fall.unwrap()
    })
  };

  cfg
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
pub fn process_frame(
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

  pub fn print_stats(&self) -> String {
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
