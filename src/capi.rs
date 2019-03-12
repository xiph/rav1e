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
use std::os::raw::c_char;
use std::os::raw::c_int;

use libc::size_t;
use libc::ptrdiff_t;

/// Raw video Frame
///
/// It can be allocated throught rav1e_frame_new(), populated using rav1e_frame_fill_plane()
/// and freed using rav1e_frame_unref().
pub struct Frame(Arc<rav1e::Frame<u16>>);

type EncoderStatus=rav1e::EncoderStatus;

/// Encoder configuration
///
/// Instantiate it using rav1e_config_default() and fine-tune it using
/// rav1e_config_parse().
///
/// Use rav1e_config_unref() to free its memory.
pub struct Config {
    cfg: rav1e::Config,
    last_err: Option<EncoderStatus>,
}

/// Encoder context
///
/// Contains the encoding state, it is created by rav1e_context_new() using an
/// Encoder configuration.
///
/// Use rav1e_context_unref() to free its memory.
pub struct Context {
    ctx: rav1e::Context<u16>,
    last_err: Option<EncoderStatus>,
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
        last_err: None,
    });

    Box::into_raw(c)
}

#[no_mangle]
pub unsafe extern "C" fn rav1e_config_set_color_description(cfg: *mut Config,
                                                            matrix: MatrixCoefficients,
                                                            primaries: ColorPrimaries,
                                                            transfer: TransferCharacteristics)
{
    (*cfg).cfg.enc.color_description = Some(rav1e::ColorDescription {
        matrix_coefficients: matrix,
        color_primaries: primaries,
        transfer_characteristics: transfer,
    });
}

#[no_mangle]
pub unsafe extern "C" fn rav1e_config_set_content_light(cfg: *mut Config,
                                                        max_content_light_level: u16,
                                                        max_frame_average_light_level: u16)
{
    (*cfg).cfg.enc.content_light = Some(rav1e::ContentLight {
        max_content_light_level,
        max_frame_average_light_level,
    });
}

#[no_mangle]
pub unsafe extern "C" fn rav1e_config_set_mastering_display(cfg: *mut Config,
                                                            primaries: [rav1e::Point; 3],
                                                            white_point: rav1e::Point,
                                                            max_luminance: u32,
                                                            min_luminance: u32)
{
    (*cfg).cfg.enc.mastering_display = Some(rav1e::MasteringDisplay {
        primaries,
        white_point,
        max_luminance,
        min_luminance,
    });
}

#[no_mangle]
pub unsafe extern "C" fn rav1e_config_unref(cfg: *mut Config) {
    let _ = Box::from_raw(cfg);
}

/// Set a configuration parameter using its key and value
///
/// Available keys and values
/// - "quantizer": 0-255, default 100
/// - "speed": 0-10, default 3
/// - "tune": "psnr"-"psychovisual", default "psnr"
///
/// Returns a negative value on non-zero.
#[no_mangle]
pub unsafe extern "C" fn rav1e_config_parse(
    cfg: *mut Config,
    key: *const c_char,
    value: *const c_char,
) -> c_int {
    let key = CStr::from_ptr(key).to_string_lossy();
    let value = CStr::from_ptr(value).to_string_lossy();

    (*cfg)
        .cfg
        .parse(&key, &value)
        .map(|_v| {
            (*cfg).last_err = None;
            0
        }).map_err(|e| (*cfg).last_err = Some(e))
        .unwrap_or(-1)
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
    let _ = Box::from_raw(ctx);
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
    let _ = Arc::from_raw(frame);
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
pub unsafe extern "C" fn rav1e_send_frame(ctx: *mut Context, frame: *const Frame) -> c_int {
    let frame = if frame.is_null() {
        None
    } else {
        Some((*frame).0.clone())
    };

    (*ctx)
        .ctx
        .send_frame(frame)
        .map(|_v| {
            (*ctx).last_err = None;
            0
        }).unwrap_or_else(|e| {
            use rav1e::EncoderStatus::*;
            (*ctx).last_err = Some(e);
            match e {
                EnoughData => 1,
                NeedMoreData | NeedMoreFrames => unreachable!(),
                _ => -1,
            }
        })
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
) -> c_int {
    (*ctx)
        .ctx
        .receive_packet()
        .map(|p| {
            (*ctx).last_err = None;
            let packet = Packet {
                data: p.data.as_ptr(),
                len: p.data.len(),
                number: p.number,
                frame_type: p.frame_type,
            };
            *pkt = Box::into_raw(Box::new(packet));
            0
        }).unwrap_or_else(|e| {
            use rav1e::EncoderStatus::*;
            (*ctx).last_err = Some(e);
            match e {
                NeedMoreData |
                    NeedMoreFrames => 1,
                EnoughData => unreachable!(),
                _ => -1,
            }
        })
}

#[no_mangle]
pub unsafe extern fn rav1e_packet_unref(pkt: *mut Packet) {
    let _ = Box::from_raw(pkt);
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
    let _ = Box::from_raw(sequence);
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
    let input = Arc::get_mut(&mut (*frame).0).unwrap();
    let data_slice = slice::from_raw_parts(data, data_len as usize);

    input.planes[plane as usize].copy_from_raw_u8(data_slice, stride as usize, bytewidth as usize);
}
