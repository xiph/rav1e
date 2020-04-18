// Copyright (c) 2001-2016, Alliance for Open Media. All rights reserved
// Copyright (c) 2017-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! # C API for rav1e
//!
//! [rav1e](https://github.com/xiph/rav1e/) is an [AV1](https://aomediacodec.github.io/av1-spec/)
//! encoder written in [Rust](https://rust-lang.org)
//!
//! This is the C-compatible API
#![deny(missing_docs)]

use std::slice;
use std::sync::Arc;

use std::ffi::CStr;
use std::ffi::CString;
use std::os::raw::c_char;
use std::os::raw::c_int;

use libc::ptrdiff_t;
use libc::size_t;

use num_derive::*;
use num_traits::cast::FromPrimitive;

use crate::prelude as rav1e;

type PixelRange = rav1e::PixelRange;
type ChromaSamplePosition = rav1e::ChromaSamplePosition;
type ChromaSampling = rav1e::ChromaSampling;
type MatrixCoefficients = rav1e::MatrixCoefficients;
type ColorPrimaries = rav1e::ColorPrimaries;
type TransferCharacteristics = rav1e::TransferCharacteristics;
type Rational = rav1e::Rational;
type FrameTypeOverride = rav1e::FrameTypeOverride;

#[derive(Clone)]
enum FrameInternal {
  U8(Arc<rav1e::Frame<u8>>),
  U16(Arc<rav1e::Frame<u16>>),
}

impl From<rav1e::Frame<u8>> for FrameInternal {
  fn from(f: rav1e::Frame<u8>) -> FrameInternal {
    FrameInternal::U8(Arc::new(f))
  }
}

impl From<rav1e::Frame<u16>> for FrameInternal {
  fn from(f: rav1e::Frame<u16>) -> FrameInternal {
    FrameInternal::U16(Arc::new(f))
  }
}

/// Raw video Frame
///
/// It can be allocated through rav1e_frame_new(), populated using rav1e_frame_fill_plane()
/// and freed using rav1e_frame_unref().
pub struct Frame {
  fi: FrameInternal,
  frame_type: FrameTypeOverride,
}

/// Status that can be returned by encoder functions.
#[repr(C)]
#[derive(Copy, Clone, Debug, FromPrimitive)]
pub enum EncoderStatus {
  /// Normal operation.
  Success = 0,
  /// The encoder needs more data to produce an output packet.
  ///
  /// May be emitted by `rav1e_receive_packet` when frame reordering is
  /// enabled.
  NeedMoreData,
  /// There are enough frames in the queue.
  ///
  /// May be emitted by `rav1e_send_frame` when trying to send a frame after
  /// the encoder has been flushed or the internal queue is full.
  EnoughData,
  /// The encoder has already produced the number of frames requested.
  ///
  /// May be emitted by `rav1e_receive_packet` after a flush request had been
  /// processed or the frame limit had been reached.
  LimitReached,
  /// A Frame had been encoded but not emitted yet.
  Encoded,
  /// Generic fatal error.
  Failure = -1,
  /// A frame was encoded in the first pass of a 2-pass encode, but its stats
  /// data was not retrieved with `rav1e_twopass_out`, or not enough stats data
  /// was provided in the second pass of a 2-pass encode to encode the next
  /// frame.
  NotReady = -2,
}

impl EncoderStatus {
  fn to_c(&self) -> *const u8 {
    use self::EncoderStatus::*;
    match self {
      Success => "Normal operation\0".as_ptr(),
      NeedMoreData => "The encoder needs more data to produce an output packet\0".as_ptr(),
      EnoughData => "There are enough frames in the queue\0".as_ptr(),
      LimitReached => "The encoder has already produced the number of frames requested\0".as_ptr(),
      Encoded => "A Frame had been encoded but not emitted yet\0".as_ptr(),
      Failure => "Generic fatal error\0".as_ptr(),
      NotReady => "First-pass stats data not retrieved or not enough second-pass data provided\0".as_ptr(),
    }
  }
}

impl From<Option<rav1e::EncoderStatus>> for EncoderStatus {
  fn from(status: Option<rav1e::EncoderStatus>) -> Self {
    match status {
      None => EncoderStatus::Success,
      Some(s) => match s {
        rav1e::EncoderStatus::NeedMoreData => EncoderStatus::NeedMoreData,
        rav1e::EncoderStatus::EnoughData => EncoderStatus::EnoughData,
        rav1e::EncoderStatus::LimitReached => EncoderStatus::LimitReached,
        rav1e::EncoderStatus::Encoded => EncoderStatus::Encoded,
        rav1e::EncoderStatus::Failure => EncoderStatus::Failure,
        rav1e::EncoderStatus::NotReady => EncoderStatus::NotReady,
      },
    }
  }
}

/// Encoder configuration
///
/// Instantiate it using rav1e_config_default() and fine-tune it using
/// rav1e_config_parse().
///
/// Use rav1e_config_unref() to free its memory.
pub struct Config {
  cfg: rav1e::Config,
}

enum EncContext {
  U8(rav1e::Context<u8>),
  U16(rav1e::Context<u16>),
}

impl EncContext {
  fn new_frame(&self) -> FrameInternal {
    match self {
      EncContext::U8(ctx) => ctx.new_frame().into(),
      EncContext::U16(ctx) => ctx.new_frame().into(),
    }
  }
  fn send_frame(
    &mut self, frame: Option<FrameInternal>, frame_type: FrameTypeOverride,
  ) -> Result<(), rav1e::EncoderStatus> {
    let info = rav1e::FrameParameters { frame_type_override: frame_type };
    if let Some(frame) = frame {
      match (self, frame) {
        (EncContext::U8(ctx), FrameInternal::U8(ref f)) => {
          ctx.send_frame((Arc::clone(f), info))
        }
        (EncContext::U16(ctx), FrameInternal::U16(ref f)) => {
          ctx.send_frame((Arc::clone(f), info))
        }
        _ => Err(rav1e::EncoderStatus::Failure),
      }
    } else {
      match self {
        EncContext::U8(ctx) => ctx.send_frame(None),
        EncContext::U16(ctx) => ctx.send_frame(None),
      }
    }
  }

  fn receive_packet(&mut self) -> Result<Packet, rav1e::EncoderStatus> {
    fn receive_packet<T: rav1e::Pixel>(
      ctx: &mut rav1e::Context<T>,
    ) -> Result<Packet, rav1e::EncoderStatus> {
      ctx.receive_packet().map(|p| {
        let rav1e::Packet { data, input_frameno, frame_type, .. } = p;
        let len = data.len();
        let data = Box::into_raw(data.into_boxed_slice()) as *const u8;
        Packet { data, len, input_frameno, frame_type }
      })
    }
    match self {
      EncContext::U8(ctx) => receive_packet(ctx),
      EncContext::U16(ctx) => receive_packet(ctx),
    }
  }

  fn container_sequence_header(&self) -> Vec<u8> {
    match self {
      EncContext::U8(ctx) => ctx.container_sequence_header(),
      EncContext::U16(ctx) => ctx.container_sequence_header(),
    }
  }

  fn twopass_bytes_needed(&mut self) -> usize {
    match self {
      EncContext::U8(ctx) => ctx.twopass_bytes_needed(),
      EncContext::U16(ctx) => ctx.twopass_bytes_needed(),
    }
  }

  fn twopass_in(&mut self, buf: &[u8]) -> Result<usize, rav1e::EncoderStatus> {
    match self {
      EncContext::U8(ctx) => ctx.twopass_in(buf),
      EncContext::U16(ctx) => ctx.twopass_in(buf),
    }
  }

  fn twopass_out(&mut self) -> Option<&[u8]> {
    match self {
      EncContext::U8(ctx) => ctx.twopass_out(),
      EncContext::U16(ctx) => ctx.twopass_out(),
    }
  }
}

/// Encoder context
///
/// Contains the encoding state, it is created by rav1e_context_new() using an
/// Encoder configuration.
///
/// Use rav1e_context_unref() to free its memory.
pub struct Context {
  ctx: EncContext,
  last_err: Option<rav1e::EncoderStatus>,
}

type FrameType = rav1e::FrameType;

/// Encoded Packet
///
/// The encoded packets are retrieved using rav1e_receive_packet().
///
/// Use rav1e_packet_unref() to free its memory.
#[repr(C)]
pub struct Packet {
  /// Encoded data buffer
  pub data: *const u8,
  /// Encoded data buffer size
  pub len: size_t,
  /// Frame sequence number
  pub input_frameno: u64,
  /// Frame type
  pub frame_type: FrameType,
}

/// Version information as presented in `[package]` `version`.
///
/// e.g. `0.1.0``
///
/// Can be parsed by [semver](https://crates.io/crates/semver).
/// This returns the version of the loaded library, regardless
/// of which version the library user was built against.
#[no_mangle]
pub unsafe extern fn rav1e_version_short() -> *const c_char {
  concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Version information with the information
/// provided by `git describe --tags`.
///
/// e.g. `0.1.0 (v0.1.0-1-g743d464)`
///
/// This returns the version of the loaded library, regardless
/// of which version the library user was built against.
#[no_mangle]
pub unsafe extern fn rav1e_version_full() -> *const c_char {
  concat!(
    env!("CARGO_PKG_VERSION"),
    " (",
    env!("VERGEN_SEMVER_LIGHTWEIGHT"),
    ")\0"
  )
  .as_ptr() as *const c_char
}

/// Simple Data
///
///
///
/// Use rav1e_data_unref() to free its memory.
#[repr(C)]
pub struct Data {
  /// Pointer to the data buffer
  pub data: *const u8,
  /// Data buffer size
  pub len: size_t,
}

/// Free a RaData buffer
#[no_mangle]
pub unsafe extern fn rav1e_data_unref(data: *mut Data) {
  if !data.is_null() {
    let data = Box::from_raw(data);
    let _ = Vec::from_raw_parts(
      data.data as *mut u8,
      data.len as usize,
      data.len as usize,
    );
  }
}

/// Create a RaConfig filled with default parameters.
#[no_mangle]
pub unsafe extern fn rav1e_config_default() -> *mut Config {
  let cfg = rav1e::Config { enc: rav1e::EncoderConfig::default(), threads: 0 };

  let c = Box::new(Config { cfg });

  Box::into_raw(c)
}

/// Set the time base of the stream
///
/// Needed for rate control.
#[no_mangle]
pub unsafe extern fn rav1e_config_set_time_base(
  cfg: *mut Config, time_base: Rational,
) {
  (*cfg).cfg.enc.time_base = time_base
}

/// Set pixel format of the stream.
///
/// Supported values for subsampling and chromapos are defined by the
/// enum types RaChromaSampling and RaChromaSamplePosition respectively.
/// Valid values for fullrange are 0 and 1.
///
/// Returns a negative value on error or 0.
#[no_mangle]
pub unsafe extern fn rav1e_config_set_pixel_format(
  cfg: *mut Config, bit_depth: u8, subsampling: ChromaSampling,
  chroma_pos: ChromaSamplePosition, pixel_range: PixelRange,
) -> c_int {
  if bit_depth != 8 && bit_depth != 10 && bit_depth != 12 {
    return -1;
  }
  (*cfg).cfg.enc.bit_depth = bit_depth as usize;

  let subsampling_val =
    std::mem::transmute::<ChromaSampling, i32>(subsampling);
  if ChromaSampling::from_i32(subsampling_val).is_none() {
    return -1;
  }
  (*cfg).cfg.enc.chroma_sampling = subsampling;

  let chroma_pos_val =
    std::mem::transmute::<ChromaSamplePosition, i32>(chroma_pos);
  if ChromaSamplePosition::from_i32(chroma_pos_val).is_none() {
    return -1;
  }
  (*cfg).cfg.enc.chroma_sample_position = chroma_pos;

  let pixel_range_val = std::mem::transmute::<PixelRange, i32>(pixel_range);
  if PixelRange::from_i32(pixel_range_val).is_none() {
    return -1;
  }
  (*cfg).cfg.enc.pixel_range = pixel_range;

  0
}

/// Set color properties of the stream.
///
/// Supported values are defined by the enum types
/// RaMatrixCoefficients, RaColorPrimaries, and RaTransferCharacteristics
/// respectively.
///
/// Return a negative value on error or 0.
#[no_mangle]
pub unsafe extern fn rav1e_config_set_color_description(
  cfg: *mut Config, matrix: MatrixCoefficients, primaries: ColorPrimaries,
  transfer: TransferCharacteristics,
) -> c_int {
  (*cfg).cfg.enc.color_description = Some(rav1e::ColorDescription {
    matrix_coefficients: matrix,
    color_primaries: primaries,
    transfer_characteristics: transfer,
  });

  if (*cfg).cfg.enc.color_description.is_some() {
    0
  } else {
    -1
  }
}

/// Set the content light level information for HDR10 streams.
///
/// Return a negative value on error or 0.
#[no_mangle]
pub unsafe extern fn rav1e_config_set_content_light(
  cfg: *mut Config, max_content_light_level: u16,
  max_frame_average_light_level: u16,
) -> c_int {
  (*cfg).cfg.enc.content_light = Some(rav1e::ContentLight {
    max_content_light_level,
    max_frame_average_light_level,
  });

  if (*cfg).cfg.enc.content_light.is_some() {
    0
  } else {
    -1
  }
}

/// Set the mastering display information for HDR10 streams.
///
/// primaries and white_point arguments are RaChromaticityPoint, containing 0.16 fixed point
/// values.
/// max_luminance is a 24.8 fixed point value.
/// min_luminance is a 18.14 fixed point value.
///
/// Returns a negative value on error or 0.
#[no_mangle]
pub unsafe extern fn rav1e_config_set_mastering_display(
  cfg: *mut Config, primaries: [rav1e::ChromaticityPoint; 3],
  white_point: rav1e::ChromaticityPoint, max_luminance: u32,
  min_luminance: u32,
) -> c_int {
  (*cfg).cfg.enc.mastering_display = Some(rav1e::MasteringDisplay {
    primaries,
    white_point,
    max_luminance,
    min_luminance,
  });

  if (*cfg).cfg.enc.mastering_display.is_some() {
    0
  } else {
    -1
  }
}

/// Free the RaConfig.
#[no_mangle]
pub unsafe extern fn rav1e_config_unref(cfg: *mut Config) {
  if !cfg.is_null() {
    let _ = Box::from_raw(cfg);
  }
}

fn tile_log2(blk_size: usize, target: usize) -> usize {
  let mut k = 0;
  while (blk_size << k) < target {
    k += 1;
  }
  k
}

fn check_tile_log2(n: Result<usize, ()>) -> Result<usize, ()> {
  match n {
    Ok(n) => {
      if ((1 << tile_log2(1, n)) - n) == 0 || n == 0 {
        Ok(n)
      } else {
        Err(())
      }
    }
    Err(e) => Err(e),
  }
}

fn check_frame_size(n: Result<usize, ()>) -> Result<usize, ()> {
  match n {
    Ok(n) => {
      if n >= 16 && n < u16::max_value().into() {
        Ok(n)
      } else {
        Err(())
      }
    }
    Err(e) => Err(e),
  }
}

unsafe fn option_match(
  cfg: *mut Config, key: *const c_char, value: *const c_char,
) -> Result<(), ()> {
  let key = CStr::from_ptr(key).to_str().map_err(|_| ())?;
  let value = CStr::from_ptr(value).to_str().map_err(|_| ())?;
  let enc = &mut (*cfg).cfg.enc;

  match key {
    "width" => enc.width = check_frame_size(value.parse().map_err(|_| ()))?,
    "height" => enc.height = check_frame_size(value.parse().map_err(|_| ()))?,
    "speed" => {
      enc.speed_settings =
        rav1e::SpeedSettings::from_preset(value.parse().map_err(|_| ())?)
    }

    "threads" => (*cfg).cfg.threads = value.parse().map_err(|_| ())?,

    "tiles" => enc.tiles = value.parse().map_err(|_| ())?,
    "tile_rows" => {
      enc.tile_rows = check_tile_log2(value.parse().map_err(|_| ()))?
    }
    "tile_cols" => {
      enc.tile_cols = check_tile_log2(value.parse().map_err(|_| ()))?
    }

    "tune" => enc.tune = value.parse().map_err(|_| ())?,
    "quantizer" => enc.quantizer = value.parse().map_err(|_| ())?,
    "min_quantizer" => enc.min_quantizer = value.parse().map_err(|_| ())?,
    "bitrate" => enc.bitrate = value.parse().map_err(|_| ())?,

    "key_frame_interval" => {
      enc.set_key_frame_interval(
        enc.min_key_frame_interval,
        value.parse().map_err(|_| ())?,
      );
    }
    "min_key_frame_interval" => {
      enc.set_key_frame_interval(
        value.parse().map_err(|_| ())?,
        enc.max_key_frame_interval,
      );
    }
    "switch_frame_interval" => {
      enc.switch_frame_interval = value.parse().map_err(|_| ())?
    }
    "reservoir_frame_delay" => {
      enc.reservoir_frame_delay = Some(value.parse().map_err(|_| ())?)
    }
    "rdo_lookahead_frames" => {
      enc.rdo_lookahead_frames = value.parse().map_err(|_| ())?
    }
    "low_latency" => enc.low_latency = value.parse().map_err(|_| ())?,
    "enable_timing_info" => {
      enc.enable_timing_info = value.parse().map_err(|_| ())?
    }
    "still_picture" => enc.still_picture = value.parse().map_err(|_| ())?,

    _ => return Err(()),
  }

  Ok(())
}

/// Set a configuration parameter using its key and value as string.
///
/// Available keys and values
/// - "quantizer": 0-255, default 100
/// - "speed": 0-10, default 6
/// - "tune": "psnr"-"psychovisual", default "psychovisual"
///
/// Return a negative value on error or 0.
#[no_mangle]
pub unsafe extern fn rav1e_config_parse(
  cfg: *mut Config, key: *const c_char, value: *const c_char,
) -> c_int {
  if option_match(cfg, key, value) == Ok(()) {
    0
  } else {
    -1
  }
}

/// Set a configuration parameter using its key and value as integer.
///
/// Available keys and values are the same as rav1e_config_parse()
///
/// Return a negative value on error or 0.
#[no_mangle]
pub unsafe extern fn rav1e_config_parse_int(
  cfg: *mut Config, key: *const c_char, value: c_int,
) -> c_int {
  let val = CString::new(value.to_string()).unwrap();
  if option_match(cfg, key, val.as_ptr()) == Ok(()) {
    0
  } else {
    -1
  }
}

/// Generate a new encoding context from a populated encoder configuration
///
/// Multiple contexts can be generated through it.
/// Returns Null if context creation failed, e.g. by passing
/// an invalid Config.
#[no_mangle]
pub unsafe extern fn rav1e_context_new(cfg: *const Config) -> *mut Context {
  let cfg = &(*cfg).cfg;
  let enc = &cfg.enc;

  let ctx = match enc.bit_depth {
    8 => cfg.new_context().map(EncContext::U8),
    _ => cfg.new_context().map(EncContext::U16),
  };

  if let Ok(ctx) = ctx {
    Box::into_raw(Box::new(Context { ctx, last_err: None }))
  } else {
    std::ptr::null_mut()
  }
}

/// Free the RaContext.
#[no_mangle]
pub unsafe extern fn rav1e_context_unref(ctx: *mut Context) {
  if !ctx.is_null() {
    let _ = Box::from_raw(ctx);
  }
}

/// Produce a new frame from the encoding context
///
/// It must be populated using rav1e_frame_fill_plane().
///
/// The frame is reference counted and must be released passing it to rav1e_frame_unref(),
/// see rav1e_send_frame().
#[no_mangle]
pub unsafe extern fn rav1e_frame_new(ctx: *const Context) -> *mut Frame {
  let fi = (*ctx).ctx.new_frame();
  let frame_type = rav1e::FrameTypeOverride::No;
  let f = Frame { fi, frame_type };
  let frame = Box::new(f.into());

  Box::into_raw(frame)
}

/// Free the RaFrame.
#[no_mangle]
pub unsafe extern fn rav1e_frame_unref(frame: *mut Frame) {
  if !frame.is_null() {
    let _ = Box::from_raw(frame);
  }
}

/// Overrides the encoders frame type decision for a frame
///
/// Must be called before rav1e_send_frame() if used.
#[no_mangle]
pub unsafe extern fn rav1e_frame_set_type(
  frame: *mut Frame, frame_type: FrameTypeOverride,
) -> c_int {
  let frame_type_val =
    std::mem::transmute::<FrameTypeOverride, i32>(frame_type);
  if FrameTypeOverride::from_i32(frame_type_val).is_none() {
    return -1;
  }
  (*frame).frame_type = frame_type;

  0
}

/// Retrieve the first-pass data of a two-pass encode for the frame that was
/// just encoded. This should be called BEFORE every call to rav1e_receive_packet()
/// (including the very first one), even if no packet was produced by the
/// last call to rav1e_receive_packet, if any (i.e., RA_ENCODER_STATUS_ENCODED
/// was returned). It needs to be called once more after
/// RA_ENCODER_STATUS_LIMIT_REACHED is returned, to retrieve the header that
/// should be written to the front of the stats file (overwriting the
/// placeholder header that was emitted at the start of encoding).
///
/// It is still safe to call this function when rav1e_receive_packet() returns any
/// other error. It will return NULL instead of returning a duplicate copy
/// of the previous frame's data.
///
/// Must be freed with rav1e_data_unref().
#[no_mangle]
pub unsafe extern fn rav1e_twopass_out(ctx: *mut Context) -> *mut Data {
  let buf = (*ctx).ctx.twopass_out();

  if buf.is_none() {
    return std::ptr::null_mut();
  }

  let v = buf.unwrap().to_vec();
  Box::into_raw(Box::new(Data {
    len: v.len(),
    data: Box::into_raw(v.into_boxed_slice()) as *mut u8,
  }))
}

/// Ask how many bytes of the stats file are needed before the next frame
/// of the second pass in a two-pass encode can be encoded. This is a lower
/// bound (more might be required), but if 0 is returned, then encoding can
/// proceed. This is just a hint to the application, and does not need to
/// be called for encoding the second pass to work, so long as the
/// application continues to provide more data to rav1e_twopass_in() in a loop
/// until rav1e_twopass_in() returns 0.
#[no_mangle]
pub unsafe extern fn rav1e_twopass_bytes_needed(ctx: *mut Context) -> size_t {
  (*ctx).ctx.twopass_bytes_needed() as size_t
}

/// Provide stats data produced in the first pass of a two-pass encode to the
/// second pass. On success this returns the number of bytes of that data
/// which were consumed. When encoding the second pass of a two-pass encode,
/// this should be called repeatedly in a loop before every call to
/// rav1e_receive_packet() (including the very first one) until no bytes are
/// consumed, or until twopass_bytes_needed() returns 0. Returns -1 on failure.
#[no_mangle]
pub unsafe extern fn rav1e_twopass_in(
  ctx: *mut Context, buf: *mut u8, buf_size: size_t,
) -> c_int {
  let buf_slice = slice::from_raw_parts(buf, buf_size as usize);
  let r = (*ctx).ctx.twopass_in(buf_slice);
  match r {
    Ok(v) => v as c_int,
    Err(v) => {
      (*ctx).last_err = Some(v);
      -1
    }
  }
}

/// Send the frame for encoding
///
/// The function increases the frame internal reference count and it can be passed multiple
/// times to different rav1e_send_frame().
///
/// Returns:
/// - `0` on success,
/// - `> 0` if the input queue is full
/// - `< 0` on unrecoverable failure
#[no_mangle]
pub unsafe extern fn rav1e_send_frame(
  ctx: *mut Context, frame: *const Frame,
) -> EncoderStatus {
  let frame_internal =
    if frame.is_null() { None } else { Some((*frame).fi.clone()) };
  let frame_type = if frame.is_null() {
    rav1e::FrameTypeOverride::No
  } else {
    (*frame).frame_type
  };

  let ret = (*ctx)
    .ctx
    .send_frame(frame_internal, frame_type)
    .map(|_v| None)
    .unwrap_or_else(|e| Some(e));

  (*ctx).last_err = ret;

  ret.into()
}

/// Return the last encoder status
#[no_mangle]
pub unsafe extern fn rav1e_last_status(ctx: *const Context) -> EncoderStatus {
  (*ctx).last_err.into()
}

/// Return a static string matching the EncoderStatus variant.
///
#[no_mangle]
pub unsafe extern fn rav1e_status_to_str(
  status: EncoderStatus,
) -> *const c_char {
  if EncoderStatus::from_i32(std::mem::transmute(status)).is_none() {
    return std::ptr::null();
  }

  status.to_c() as *const c_char
}

/// Receive encoded data
///
/// Returns:
/// - `0` on success
/// - `> 0` if additional frame data is required
/// - `< 0` on unrecoverable failure
#[no_mangle]
pub unsafe extern fn rav1e_receive_packet(
  ctx: *mut Context, pkt: *mut *mut Packet,
) -> EncoderStatus {
  let ret = (*ctx)
    .ctx
    .receive_packet()
    .map(|packet| {
      *pkt = Box::into_raw(Box::new(packet));
      None
    })
    .unwrap_or_else(|e| Some(e));

  (*ctx).last_err = ret;

  ret.into()
}

/// Free the RaPacket.
#[no_mangle]
pub unsafe extern fn rav1e_packet_unref(pkt: *mut Packet) {
  if !pkt.is_null() {
    let pkt = Box::from_raw(pkt);
    let _ = Vec::from_raw_parts(
      pkt.data as *mut u8,
      pkt.len as usize,
      pkt.len as usize,
    );
  }
}

/// Produce a sequence header matching the current encoding context
///
/// Its format is compatible with the AV1 Matroska and ISOBMFF specification.
///
/// Use rav1e_data_unref() to free it.
#[no_mangle]
pub unsafe extern fn rav1e_container_sequence_header(
  ctx: *const Context,
) -> *mut Data {
  let buf = (*ctx).ctx.container_sequence_header();

  Box::into_raw(Box::new(Data {
    len: buf.len(),
    data: Box::into_raw(buf.into_boxed_slice()) as *mut u8,
  }))
}

fn rav1e_frame_fill_plane_internal<T: rav1e::Pixel>(
  f: &mut Arc<rav1e::Frame<T>>, plane: c_int, data_slice: &[u8],
  stride: ptrdiff_t, bytewidth: c_int,
) {
  let input = Arc::make_mut(f);
  input.planes[plane as usize].copy_from_raw_u8(
    data_slice,
    stride as usize,
    bytewidth as usize,
  );
}

/// Fill a frame plane
///
/// Currently the frame contains 3 planes, the first is luminance followed by
/// chrominance.
///
/// The data is copied and this function has to be called for each plane.
///
/// frame: A frame provided by rav1e_frame_new()
/// plane: The index of the plane starting from 0
/// data: The data to be copied
/// data_len: Length of the buffer
/// stride: Plane line in bytes, including padding
/// bytewidth: Number of bytes per component, either 1 or 2
#[no_mangle]
pub unsafe extern fn rav1e_frame_fill_plane(
  frame: *mut Frame, plane: c_int, data: *const u8, data_len: size_t,
  stride: ptrdiff_t, bytewidth: c_int,
) {
  let data_slice = slice::from_raw_parts(data, data_len as usize);

  match (*frame).fi {
    FrameInternal::U8(ref mut f) => {
      rav1e_frame_fill_plane_internal(f, plane, data_slice, stride, bytewidth)
    }
    FrameInternal::U16(ref mut f) => {
      rav1e_frame_fill_plane_internal(f, plane, data_slice, stride, bytewidth)
    }
  }
}
