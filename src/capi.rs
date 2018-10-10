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
pub struct RaFrame(Arc<rav1e::Frame>);

pub struct RaConfig {
    cfg: rav1e::Config,
    last_err: Option<rav1e::EncoderStatus>,
}

pub struct RaContext {
    ctx: rav1e::Context,
    last_err: Option<rav1e::EncoderStatus>,
}

type RaFrameType = rav1e::FrameType;

#[repr(C)]
pub struct RaPacket {
    pub data: *const u8,
    pub len: usize,
    pub number: u64,
    pub frame_type: RaFrameType,
}


#[no_mangle]
pub unsafe extern "C" fn rav1e_config_default(
    width: usize,
    height: usize,
    bit_depth: usize,
    chroma_sampling: rav1e::ChromaSampling,
    timebase: rav1e::Ratio,
) -> *mut RaConfig {
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

    let c = Box::new(RaConfig {
        cfg,
        last_err: None,
    });

    Box::into_raw(c)
}

#[no_mangle]
pub unsafe extern "C" fn rav1e_config_drop(cfg: *mut RaConfig) {
    let _ = Box::from_raw(cfg);
}

#[no_mangle]
pub unsafe extern "C" fn rav1e_config_parse(
    cfg: *mut RaConfig,
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

#[no_mangle]
pub unsafe extern "C" fn rav1e_context_new(cfg: *const RaConfig) -> *mut RaContext {
    let ctx = RaContext {
        ctx: (*cfg).cfg.new_context(),
        last_err: None,
    };

    Box::into_raw(Box::new(ctx))
}

#[no_mangle]
pub unsafe extern "C" fn rav1e_context_drop(ctx: *mut RaContext) {
    let _ = Box::from_raw(ctx);
}

#[no_mangle]
pub unsafe extern "C" fn rav1e_frame_new(ctx: *const RaContext) -> *mut RaFrame {
    let f = (*ctx).ctx.new_frame();
    let frame = Box::new(RaFrame(f));

    Box::into_raw(frame)
}

#[no_mangle]
pub unsafe extern "C" fn rav1e_frame_drop(ctx: *mut RaFrame) {
    let _ = Arc::from_raw(ctx);
}

#[no_mangle]
pub unsafe extern "C" fn rav1e_send_frame(ctx: *mut RaContext, frame: *const RaFrame) -> c_int {
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

#[no_mangle]
pub unsafe extern "C" fn rav1e_receive_packet(
    ctx: *mut RaContext,
    pkt: *mut *mut RaPacket,
) -> c_int {
    (*ctx)
        .ctx
        .receive_packet()
        .map(|p| {
            (*ctx).last_err = None;
            let packet = RaPacket {
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
pub unsafe extern fn rav1e_packet_drop(pkt: *mut RaPacket) {
    let _ = Box::from_raw(pkt);
}

#[no_mangle]
pub unsafe extern fn rav1e_container_sequence_header(ctx: *mut RaContext, len: *mut usize) -> *mut u8 {
    let buf = (*ctx).ctx.container_sequence_header();

    *len = buf.len();
    Box::into_raw(buf.into_boxed_slice()) as *mut u8
}

pub unsafe extern fn rav1e_container_sequence_header_drop(sequence: *mut u8) {
    let _ = Box::from_raw(sequence);
}

#[no_mangle]
pub unsafe extern "C" fn rav1e_frame_fill_plane(
    frame: *mut RaFrame,
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
