// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use clap::{App, Arg, ArgMatches};
use rav1e::*;
use std::fmt;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::slice;
use std::sync::Arc;
use std::time::Instant;
use y4m;

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
    ).arg(Arg::with_name("RECONSTRUCTION").short("r").takes_value(true))
    .arg(
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
        .help("low latency mode. true or false")
        .long("low_latency")
        .takes_value(true)
        .default_value("true")
    ).arg(
      Arg::with_name("TUNE")
        .help("Quality tuning (Will enforce partition sizes >= 8x8)")
        .long("tune")
        .possible_values(&Tune::variants())
        .default_value("psnr")
        .case_insensitive(true)
    ).arg(
      Arg::with_name("VERBOSE")
        .help("verbose logging, output info for every frame")
        .long("verbose")
        .short("v")
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

  let mut cfg = EncoderConfig::with_speed_preset(speed);
  cfg.max_key_frame_interval = min_interval;
  cfg.max_key_frame_interval = max_interval;
  cfg.low_latency = matches.value_of("LOW_LATENCY").unwrap().parse().unwrap();
  cfg.tune = matches.value_of("TUNE").unwrap().parse().unwrap();
  cfg.quantizer = quantizer;
  cfg
}

#[derive(Debug, Clone, Copy)]
pub struct FrameSummary {
  /// Frame size in bytes
  pub size: usize,
  pub number: u64,
  pub frame_type: FrameType
}

impl From<Packet> for FrameSummary {
  fn from(packet: Packet) -> Self {
    Self {
      size: packet.data.len(),
      number: packet.number,
      frame_type: packet.frame_type,
    }
  }
}

impl fmt::Display for FrameSummary {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "Frame {} - {} - {} bytes",
      self.number,
      self.frame_type,
      self.size
    )
  }
}

/// Encode and write a frame.
/// Returns frame information in a `Result`.
pub fn process_frame(
  ctx: &mut Context, output_file: &mut dyn Write,
  y4m_dec: &mut y4m::Decoder<'_, Box<dyn Read>>,
  mut y4m_enc: Option<&mut y4m::Encoder<'_, Box<dyn Write>>>
) -> Result<Vec<FrameSummary>, ()> {
  let width = y4m_dec.get_width();
  let height = y4m_dec.get_height();
  let y4m_bits = y4m_dec.get_bit_depth();
  let y4m_bytes = y4m_dec.get_bytes_per_sample();
  let bit_depth = y4m_dec.get_colorspace().get_bit_depth();

  if ctx.needs_more_frames(ctx.get_frame_count()) {
    match y4m_dec.read_frame() {
      Ok(y4m_frame) => {
        let y4m_y = y4m_frame.get_y_plane();
        let y4m_u = y4m_frame.get_u_plane();
        let y4m_v = y4m_frame.get_v_plane();
        let mut input = ctx.new_frame();
        {
          let input = Arc::get_mut(&mut input).unwrap();
          input.planes[0].copy_from_raw_u8(&y4m_y, width * y4m_bytes, y4m_bytes);
          input.planes[1].copy_from_raw_u8(
            &y4m_u,
            width * y4m_bytes / 2,
            y4m_bytes
          );
          input.planes[2].copy_from_raw_u8(
            &y4m_v,
            width * y4m_bytes / 2,
            y4m_bytes
          );
        }

        match y4m_bits {
          8 | 10 | 12 => {}
          _ => panic!("unknown input bit depth!")
        }

        let _ = ctx.send_frame(input);
      }
      _ => {
        let frames_to_be_coded = ctx.get_frame_count();
        ctx.set_frames_to_be_coded(frames_to_be_coded);
        ctx.flush();
      }
    }
  } else {
    ctx.flush();
  };

  let mut frame_summaries = Vec::new();
  loop {
    let pkt_wrapped = ctx.receive_packet();
    match pkt_wrapped {
      Ok(pkt) => {
        write_ivf_frame(output_file, pkt.number as u64, pkt.data.as_ref());
        if let Some(y4m_enc_uw) = y4m_enc.as_mut() {
          if let Some(ref rec) = pkt.rec {
            let pitch_y = if bit_depth > 8 { width * 2 } else { width };
            let pitch_uv = pitch_y / 2;

            let (mut rec_y, mut rec_u, mut rec_v) = (
              vec![128u8; pitch_y * height],
              vec![128u8; pitch_uv * (height / 2)],
              vec![128u8; pitch_uv * (height / 2)]
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
              if bit_depth > 8 {
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
              if bit_depth > 8 {
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
              if bit_depth > 8 {
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
      },
      _ => { break; }
    }
  }
  Ok(frame_summaries)
}

#[derive(Debug, Clone)]
pub struct ProgressInfo {
  /// Frame rate of the video
  frame_rate: y4m::Ratio,
  /// The length of the whole video, in frames, if known
  total_frames: Option<usize>,
  /// The time the encode was started
  time_started: Instant,
  /// List of frames encoded so far
  frame_info: Vec<FrameSummary>,
  /// Video size so far in bytes.
  ///
  /// This value will be updated in the CLI very frequently, so we cache the previous value
  /// to reduce the overall complexity.
  encoded_size: usize,
}

impl ProgressInfo {
  pub fn new(frame_rate: y4m::Ratio, total_frames: Option<usize>) -> Self {
    Self {
      frame_rate,
      total_frames,
      time_started: Instant::now(),
      frame_info: Vec::with_capacity(total_frames.unwrap_or_default()),
      encoded_size: 0,
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

  /// Returns the bitrate of the frames so far, in bits/second
  pub fn bitrate(&self) -> usize {
    let bits = self.encoded_size * 8;
    let seconds = self.frame_info.len() as f64 / self.video_fps();
    (bits as f64 / seconds) as usize
  }

  /// Estimates the final filesize in bytes, if the number of frames is known
  pub fn estimated_size(&self) -> usize {
    self.total_frames
      .map(|frames| self.encoded_size * frames / self.frames_encoded())
      .unwrap_or_default()
  }

  /// Number of frames of given type which appear in the video
  pub fn get_frame_type_count(&self, frame_type: FrameType) -> usize {
    self.frame_info.iter()
      .filter(|frame| frame.frame_type == frame_type)
      .count()
  }

  /// Size in bytes of all frames of given frame type
  pub fn get_frame_type_size(&self, frame_type: FrameType) -> usize {
    self.frame_info.iter()
      .filter(|frame| frame.frame_type == frame_type)
      .map(|frame| frame.size)
      .sum()
  }

  pub fn print_stats(&self) -> String {
    let (key, key_size) = (self.get_frame_type_count(FrameType::KEY), self.get_frame_type_size(FrameType::KEY));
    let (inter, inter_size) = (self.get_frame_type_count(FrameType::INTER), self.get_frame_type_size(FrameType::INTER));
    let (ionly, ionly_size) = (self.get_frame_type_count(FrameType::INTRA_ONLY), self.get_frame_type_size(FrameType::INTRA_ONLY));
    let (switch, switch_size) = (self.get_frame_type_count(FrameType::SWITCH), self.get_frame_type_size(FrameType::SWITCH));
    format!("\
    Key Frames: {:>6}    avg size: {:>7} B\n\
    Inter:      {:>6}    avg size: {:>7} B\n\
    Intra Only: {:>6}    avg size: {:>7} B\n\
    Switch:     {:>6}    avg size: {:>7} B",
      key, key_size / key,
      inter, inter_size.checked_div(inter).unwrap_or(0),
      ionly, ionly_size / key,
      switch, switch_size / key
    )
  }
}

impl fmt::Display for ProgressInfo {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if let Some(total_frames) = self.total_frames {
      write!(
        f,
        "encoded {}/{} frames, {:.2} fps, {:.2} Kb/s, est. size: {:.2} MB",
        self.frames_encoded(),
        total_frames,
        self.encoding_fps(),
        self.bitrate() as f64 / 1024f64,
        self.estimated_size() as f64 / (1024 * 1024) as f64
      )
    } else {
      write!(
        f,
        "encoded {} frames, {:.2} fps, {:.2} Kb/s",
        self.frames_encoded(),
        self.encoding_fps(),
        self.bitrate() as f64 / 1024f64
      )
    }
  }
}
