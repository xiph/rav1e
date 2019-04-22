//! # C API for rav1e
//!
//! [rav1e](https://github.com/xiph/rav1e/) is an [AV1](https://aomediacodec.github.io/av1-spec/)
//! encoder written in [Rust](https://rust-lang.org)
//!
//! This is the C-compatible API

extern crate rav1e;
extern crate libc;

use std::slice;
use std::sync::Arc;

use std::ffi::CStr;
use std::ffi::CString;
use std::os::raw::c_char;
use std::os::raw::c_int;

use libc::size_t;
use libc::ptrdiff_t;

/// Raw video Frame
///
/// It can be allocated throught rav1e_frame_new(), populated using rav1e_frame_fill_plane()
/// and freed using rav1e_frame_unref().
pub struct Frame(Arc<rav1e::Frame<u16>>);

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub enum EncoderStatus {
    /// Normal operation
    Success = 0,
    /// The encoder needs more data to produce an output Packet
    /// May be emitted by `Context::receive_packet`  when frame reordering is enabled.
    NeedMoreData,
    /// There are enough Frames queue
    /// May be emitted by `Context::send_frame` when the input queue is constrained
    EnoughData,
    /// The encoder already produced the number of frames requested
    /// May be emitted by `Context::receive_packet` after a flush request had been processed
    /// or the frame limit had been reached.
    LimitReached,
    /// Generic fatal error
    Failure = -1,
}

impl From<Option<rav1e::EncoderStatus>> for EncoderStatus {
    fn from(status: Option<rav1e::EncoderStatus>) -> Self {
        match status {
            None => EncoderStatus::Success,
            Some(s) => match s {
                rav1e::EncoderStatus::NeedMoreData => EncoderStatus::NeedMoreData,
                rav1e::EncoderStatus::EnoughData => EncoderStatus::EnoughData,
                rav1e::EncoderStatus::LimitReached => EncoderStatus::LimitReached,
                rav1e::EncoderStatus::Failure => EncoderStatus::Failure,
            }
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

/// Encoder context
///
/// Contains the encoding state, it is created by rav1e_context_new() using an
/// Encoder configuration.
///
/// Use rav1e_context_unref() to free its memory.
pub struct Context {
    ctx: rav1e::Context<u16>,
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
    pub number: u64,
    /// Frame type
    pub frame_type: FrameType,
}

type ChromaSamplePosition=rav1e::ChromaSamplePosition;
type ChromaSampling=rav1e::ChromaSampling;
type MatrixCoefficients=rav1e::MatrixCoefficients;
type ColorPrimaries=rav1e::ColorPrimaries;
type TransferCharacteristics=rav1e::TransferCharacteristics;
type Rational=rav1e::Rational;

#[no_mangle]
pub unsafe extern "C" fn rav1e_config_default() -> *mut Config {
    let cfg = rav1e::Config {
        enc: rav1e::EncoderConfig::default(),
        threads: 0,
    };

    let c = Box::new(Config {
        cfg,
    });

    Box::into_raw(c)
}

/// Set color properties of the stream.
///
/// Supported values are defined by the enum types
/// RaMatrixCoefficients, RaColorPrimaries, and RaTransferCharacteristics
/// respectively.
///
/// Return a negative value on error or 0.
#[no_mangle]
pub unsafe extern "C" fn rav1e_config_set_color_description(cfg: *mut Config,
                                                            matrix: MatrixCoefficients,
                                                            primaries: ColorPrimaries,
                                                            transfer: TransferCharacteristics
) -> c_int {
    (*cfg).cfg.enc.color_description = Some(rav1e::ColorDescription {
        matrix_coefficients: matrix,
        color_primaries: primaries,
        transfer_characteristics: transfer,
    });

    if (*cfg).cfg.enc.color_description.is_some() { 0 } else { -1 }
}

/// Set the content light level information for HDR10 streams.
///
/// Return a negative value on error or 0.
#[no_mangle]
pub unsafe extern "C" fn rav1e_config_set_content_light(cfg: *mut Config,
                                                        max_content_light_level: u16,
                                                        max_frame_average_light_level: u16
) -> c_int {
    (*cfg).cfg.enc.content_light = Some(rav1e::ContentLight {
        max_content_light_level,
        max_frame_average_light_level,
    });

    if (*cfg).cfg.enc.content_light.is_some() { 0 } else { -1 }
}

/// Set the mastering display information for HDR10 streams.
///
/// primaries and white_point arguments are RaPoint, containing 0.16 fixed point values.
/// max_luminance is a 24.8 fixed point value.
/// min_luminance is a 18.14 fixed point value.
///
/// Returns a negative value on error or 0.
#[no_mangle]
pub unsafe extern "C" fn rav1e_config_set_mastering_display(cfg: *mut Config,
                                                            primaries: [rav1e::Point; 3],
                                                            white_point: rav1e::Point,
                                                            max_luminance: u32,
                                                            min_luminance: u32
) -> c_int {
    (*cfg).cfg.enc.mastering_display = Some(rav1e::MasteringDisplay {
        primaries,
        white_point,
        max_luminance,
        min_luminance,
    });

    if (*cfg).cfg.enc.mastering_display.is_some() { 0 } else { -1 }
}

#[no_mangle]
pub unsafe extern "C" fn rav1e_config_unref(cfg: *mut Config) {
    if !cfg.is_null() {
        let _ = Box::from_raw(cfg);
    }
}

unsafe fn option_match(
    cfg: *mut Config,
    key: *const c_char,
    value: *const c_char
) -> Result<(), ()> {
    let key = CStr::from_ptr(key).to_str().map_err(|_| ())?;
    let value = CStr::from_ptr(value).to_str().map_err(|_| ())?;
    let enc = &mut(*cfg).cfg.enc;

    match key {
        "width" => enc.width = value.parse().map_err(|_| ())?,
        "height" => enc.height = value.parse().map_err(|_| ())?,
        "speed" => enc.speed_settings = rav1e::SpeedSettings::from_preset(value.parse().map_err(|_| ())?),

        "threads" => (*cfg).cfg.threads = value.parse().map_err(|_| ())?,

        "tile_rows_log2" => enc.tile_rows_log2 = value.parse().map_err(|_| ())?,
        "tile_cols_log2" => enc.tile_cols_log2 = value.parse().map_err(|_| ())?,

        "tune" => enc.tune = value.parse().map_err(|_| ())?,
        "quantizer" => enc.quantizer = value.parse().map_err(|_| ())?,
        "bitrate" => enc.bitrate = value.parse().map_err(|_| ())?,

        "key_frame_interval" => enc.max_key_frame_interval = value.parse().map_err(|_| ())?,
        "min_key_frame_interval" => enc.min_key_frame_interval = value.parse().map_err(|_| ())?,
        "low_latency" => enc.low_latency = value.parse().map_err(|_| ())?,

        _ => return Err(())
    }

    Ok(())
}

/// Set a configuration parameter using its key and value as string.
///
/// Available keys and values
/// - "quantizer": 0-255, default 100
/// - "speed": 0-10, default 3
/// - "tune": "psnr"-"psychovisual", default "psnr"
///
/// Return a negative value on error or 0.
#[no_mangle]
pub unsafe extern "C" fn rav1e_config_parse(
    cfg: *mut Config,
    key: *const c_char,
    value: *const c_char,
) -> c_int {
    if option_match(cfg, key, value) == Ok(()) { 0 } else { -1 }
}

/// Set a configuration parameter using its key and value as integer.
///
/// Available keys and values are the same as rav1e_config_parse()
///
/// Return a negative value on error or 0.
#[no_mangle]
pub unsafe extern "C" fn rav1e_config_parse_int(
    cfg: *mut Config,
    key: *const c_char,
    value: c_int,
) -> c_int {
    let val = CString::new(value.to_string()).unwrap();
    if option_match(cfg, key, val.as_ptr()) == Ok(()) { 0 } else { -1 }
}

/// Generate a new encoding context from a populated encoder configuration
///
/// Multiple contexts can be generated through it.
#[no_mangle]
pub unsafe extern "C" fn rav1e_context_new(cfg: *const Config) -> *mut Context {
    let ctx = Context {
        ctx: (*cfg).cfg.new_context(),
        last_err: None,
    };

    Box::into_raw(Box::new(ctx))
}

#[no_mangle]
pub unsafe extern "C" fn rav1e_context_unref(ctx: *mut Context) {
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
pub unsafe extern "C" fn rav1e_frame_new(ctx: *const Context) -> *mut Frame {
    let f = (*ctx).ctx.new_frame();
    let frame = Box::new(Frame(f));

    Box::into_raw(frame)
}

#[no_mangle]
pub unsafe extern "C" fn rav1e_frame_unref(frame: *mut Frame) {
    if !frame.is_null() {
        let _ = Box::from_raw(frame);
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
pub unsafe extern "C" fn rav1e_send_frame(ctx: *mut Context, frame: *const Frame) -> EncoderStatus {
    let frame = if frame.is_null() {
        None
    } else {
        Some((*frame).0.clone())
    };

    let ret = (*ctx)
        .ctx
        .send_frame(frame)
        .map(|_v| {
            None
        }).unwrap_or_else(|e| {
            Some(e)
        });

    (*ctx).last_err = ret;

    ret.into()
}

/// Return the last encoder status
#[no_mangle]
pub unsafe extern "C" fn rav1e_last_status(ctx: *const Context) -> EncoderStatus {
    (*ctx).last_err.into()
}

/// Return a string matching the EncooderStatus variant.
#[no_mangle]
pub unsafe extern "C" fn rav1e_status_to_str(status: EncoderStatus) -> *mut c_char {
    let status = format!("{:?}", status);
    let cptr = CString::new(status).unwrap().as_ptr();

    libc::strdup(cptr)
}

/// Receive encoded data
///
/// Returns:
/// - `0` on success
/// - `> 0` if additional frame data is required
/// - `< 0` on unrecoverable failure
#[no_mangle]
pub unsafe extern "C" fn rav1e_receive_packet(
    ctx: *mut Context,
    pkt: *mut *mut Packet,
) -> EncoderStatus {
    let ret = (*ctx)
        .ctx
        .receive_packet()
        .map(|p| {
            let rav1e::Packet { data, number, frame_type, .. } = p;
            let len  = data.len();
            let data = Box::into_raw(data.into_boxed_slice()) as *const u8;
            let packet = Packet {
                data,
                len,
                number,
                frame_type,
            };
            *pkt = Box::into_raw(Box::new(packet));
            None
        }).unwrap_or_else(|e| {
            Some(e)
        });

    (*ctx).last_err = ret;

    ret.into()
}

#[no_mangle]
pub unsafe extern fn rav1e_packet_unref(pkt: *mut Packet) {
    if !pkt.is_null() {
        let pkt = Box::from_raw(pkt);
        let _ = Box::from_raw(pkt.data as *mut u8);
    }
}

/// Produce a sequence header matching the current encoding context
///
/// Its format is compatible with the AV1 Matroska and ISOBMFF specification.
///
/// Use rav1e_container_sequence_header_unref() to free it.
#[no_mangle]
pub unsafe extern fn rav1e_container_sequence_header(ctx: *mut Context, buf_size: *mut size_t) -> *mut u8 {
    let buf = (*ctx).ctx.container_sequence_header();

    *buf_size = buf.len();
    Box::into_raw(buf.into_boxed_slice()) as *mut u8
}

#[no_mangle]
pub unsafe extern fn rav1e_container_sequence_header_unref(sequence: *mut u8) {
    if !sequence.is_null() {
        let _ = Box::from_raw(sequence);
    }
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
/// data_len: Lenght of the buffer
/// stride: Plane line in bytes, including padding
/// bytewidth: Number of bytes per component, either 1 or 2
#[no_mangle]
pub unsafe extern "C" fn rav1e_frame_fill_plane(
    frame: *mut Frame,
    plane: c_int,
    data: *const u8,
    data_len: size_t,
    stride: ptrdiff_t,
    bytewidth: c_int,
) {
    let f = &mut (*frame).0;
    let input = Arc::make_mut(f);
    let data_slice = slice::from_raw_parts(data, data_len as usize);

    input.planes[plane as usize].copy_from_raw_u8(data_slice, stride as usize, bytewidth as usize);
}
