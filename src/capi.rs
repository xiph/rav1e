//! # C API for rav1e
//!
//! [rav1e](https://github.com/xiph/rav1e/) is an [AV1](https://aomediacodec.github.io/av1-spec/)
//! encoder written in [Rust](https://rust-lang.org)
//!
//! This is the C-compatible API to it

extern crate rav1e;

use std::slice;
use std::sync::Arc;

use std::ffi::CStr;
use std::os::raw::c_char;
use std::os::raw::c_int;

/// Raw video Frame
///
/// It can be allocated throught rav1e_frame_new(), populated using rav1e_frame_fill_plane()
/// and freed using rav1e_frame_drop().
pub struct Frame(Arc<rav1e::Frame>);

type EncoderStatus=rav1e::EncoderStatus;

/// Encoder configuration
///
/// Instantiate it using rav1e_config_default() and fine-tune it using
/// rav1e_config_parse().
///
/// Use rav1e_config_drop() to free its memory.
pub struct Config {
    cfg: rav1e::Config,
    last_err: Option<EncoderStatus>,
}

/// Encoder context
///
/// Contains the encoding state, it is created by rav1e_context_new() using an
/// Encoder configuration.
///
/// Use rav1e_context_drop() to free its memory.
pub struct Context {
    ctx: rav1e::Context,
    last_err: Option<EncoderStatus>,
}

type FrameType = rav1e::FrameType;

/// Encoded Packet
///
/// The encoded packets are retrieved using rav1e_receive_packet().
///
/// Use rav1e_packet_drop() to free its memory.
#[repr(C)]
pub struct Packet {
    /// Encoded data buffer
    pub data: *const u8,
    /// Encoded data buffer size
    pub len: usize,
    /// Frame sequence number
    pub number: u64,
    /// Frame type
    pub frame_type: FrameType,
}

type ChromaSampling=rav1e::ChromaSampling;
type Ratio=rav1e::Ratio;

#[no_mangle]
pub unsafe extern "C" fn rav1e_config_default(
    width: usize,
    height: usize,
    bit_depth: usize,
    chroma_sampling: ChromaSampling,
    timebase: Ratio,
) -> *mut Config {
    let cfg = rav1e::Config {
        frame_info: rav1e::FrameInfo {
            width,
            height,
            bit_depth,
            chroma_sampling,
        },
        timebase,
        enc: Default::default(),
    };

    let c = Box::new(Config {
        cfg,
        last_err: None,
    });

    Box::into_raw(c)
}

#[no_mangle]
pub unsafe extern "C" fn rav1e_config_drop(cfg: *mut Config) {
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
pub unsafe extern "C" fn rav1e_context_drop(ctx: *mut Context) {
    let _ = Box::from_raw(ctx);
}

/// Produce a new frame from the encoding context
///
/// It must be populated using rav1e_frame_fill_plane().
///
/// The frame is reference counted and must be released passing it to rav1e_frame_drop(),
/// see rav1e_send_frame().
#[no_mangle]
pub unsafe extern "C" fn rav1e_frame_new(ctx: *const Context) -> *mut Frame {
    let f = (*ctx).ctx.new_frame();
    let frame = Box::new(Frame(f));

    Box::into_raw(frame)
}

#[no_mangle]
pub unsafe extern "C" fn rav1e_frame_drop(ctx: *mut Frame) {
    let _ = Arc::from_raw(ctx);
}

/// Send the frame for encoding
///
/// The function increases the frame internal reference count and it can be passed multiple
/// times to different rav1e_send_frame().
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
        }).map_err(|e| (*ctx).last_err = Some(e))
        .unwrap_or(-1)
}

/// Receive encoded data
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
        }).map_err(|e| (*ctx).last_err = Some(e))
        .unwrap_or(-1)
}

#[no_mangle]
pub unsafe extern fn rav1e_packet_drop(pkt: *mut Packet) {
    let _ = Box::from_raw(pkt);
}

/// Produce a sequence header matching the current encoding context
///
/// Its format is compatible with the AV1 Matroska and ISOBMFF specification.
#[no_mangle]
pub unsafe extern fn rav1e_container_sequence_header(ctx: *mut Context, len: *mut usize) -> *mut u8 {
    let buf = (*ctx).ctx.container_sequence_header();

    *len = buf.len();
    Box::into_raw(buf.into_boxed_slice()) as *mut u8
}

pub unsafe extern fn rav1e_container_sequence_header_drop(sequence: *mut u8) {
    let _ = Box::from_raw(sequence);
}


/// Fill a frame plane
///
/// Currently the frame contains 3 frames, the first is luminance followed by
/// chrominance.
#[no_mangle]
pub unsafe extern "C" fn rav1e_frame_fill_plane(
    frame: *mut Frame,
    plane: usize,
    data: *const u8,
    data_len: usize,
    stride: usize,
    bytewidth: usize,
) {
    let input = Arc::get_mut(&mut (*frame).0).unwrap();
    let data_slice = slice::from_raw_parts(data, data_len);

    input.planes[plane].copy_from_raw_u8(data_slice, stride, bytewidth);
}
