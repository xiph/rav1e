// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::api::ContextInner;
use crate::quantize::ac_q;
use crate::quantize::dc_q;
use crate::quantize::select_ac_qi;
use crate::quantize::select_dc_qi;
use crate::util::clamp;
use crate::util::Pixel;

// The number of frame sub-types for which we track distinct parameters.
pub const FRAME_NSUBTYPES: usize = 4;

pub const FRAME_SUBTYPE_I: usize = 0;
pub const FRAME_SUBTYPE_P: usize = 1;
pub const FRAME_SUBTYPE_B0: usize = 2;
pub const FRAME_SUBTYPE_B1: usize = 3;

// The scale of AV1 quantizer tables (relative to the pixel domain), i.e., Q3.
const QSCALE: i32 = 3;

// We clamp the actual I and B frame delays to a minimum of 10 to work
//  within the range of values where later incrementing the delay works as
//  designed.
// 10 is not an exact choice, but rather a good working trade-off.
const INTER_DELAY_TARGET_MIN: i32 = 10;

// The base quantizer for a frame is adjusted based on the frame type using the
//  formula (log_qp*mqp + dqp), where log_qp is the base-2 logarithm of the
//  "linear" quantizer (the actual factor by which coefficients are divided).
// Because log_qp has an implicit offset built in based on the scale of the
//  coefficients (which depends on the pixel bit depth and the transform
//  scale), we normalize the quantizer to the equivalent for 8-bit pixels with
//  orthonormal transforms for the purposes of rate modeling.
const MQP_Q12: &[i32; FRAME_NSUBTYPES] = &[
  // TODO: Use a const function once f64 operations in const functions are
  //  stable.
  (1.0 * (1 << 12) as f64) as i32,
  (1.0 * (1 << 12) as f64) as i32,
  (1.0 * (1 << 12) as f64) as i32,
  (1.0 * (1 << 12) as f64) as i32
];

// The ratio 33_810_170.0 / 86_043_287.0 was derived by approximating the median
// of a change of 15 quantizer steps in the quantizer tables.
const DQP_Q57: &[i64; FRAME_NSUBTYPES] = &[
  (-(33_810_170.0 / 86_043_287.0) * (1i64 << 57) as f64) as i64,
  (0.0 * (1i64 << 57) as f64) as i64,
  ((33_810_170.0 / 86_043_287.0) * (1i64 << 57) as f64) as i64,
  (2.0 * (33_810_170.0 / 86_043_287.0) * (1i64 << 57) as f64) as i64
];

// Integer binary logarithm of a 64-bit value.
// v: A 64-bit value.
// Returns floor(log2(v)) + 1, or 0 if v == 0.
// This is the number of bits that would be required to represent v in two's
//  complement notation with all of the leading zeros stripped.
// TODO: Mark const once leading_zeros() as a constant function stabilizes.
fn ilog64(v: i64) -> i32 {
  64 - (v.leading_zeros() as i32)
}

// Convert an integer into a Q57 fixed-point fraction.
// The integer must be in the range -64 to 63, inclusive.
const fn q57(v: i32) -> i64 {
  // TODO: Add assert if it ever becomes possible to do in a const function.
  (v as i64) << 57
}

#[rustfmt::skip]
const ATANH_LOG2: &[i64; 32] = &[
  0x32B8_0347_3F7A_D0F4, 0x2F2A_71BD_4E25_E916, 0x2E68_B244_BB93_BA06,
  0x2E39_FB91_98CE_62E4, 0x2E2E_683F_6856_5C8F, 0x2E2B_850B_E207_7FC1,
  0x2E2A_CC58_FE7B_78DB, 0x2E2A_9E2D_E52F_D5F2, 0x2E2A_92A3_38D5_3EEC,
  0x2E2A_8FC0_8F5E_19B6, 0x2E2A_8F07_E51A_485E, 0x2E2A_8ED9_BA8A_F388,
  0x2E2A_8ECE_2FE7_384A, 0x2E2A_8ECB_4D3E_4B1A, 0x2E2A_8ECA_9494_0FE8,
  0x2E2A_8ECA_6669_811D, 0x2E2A_8ECA_5ADE_DD6A, 0x2E2A_8ECA_57FC_347E,
  0x2E2A_8ECA_5743_8A43, 0x2E2A_8ECA_5715_5FB4, 0x2E2A_8ECA_5709_D510,
  0x2E2A_8ECA_5706_F267, 0x2E2A_8ECA_5706_39BD, 0x2E2A_8ECA_5706_0B92,
  0x2E2A_8ECA_5706_0008, 0x2E2A_8ECA_5705_FD25, 0x2E2A_8ECA_5705_FC6C,
  0x2E2A_8ECA_5705_FC3E, 0x2E2A_8ECA_5705_FC33, 0x2E2A_8ECA_5705_FC30,
  0x2E2A_8ECA_5705_FC2F, 0x2E2A_8ECA_5705_FC2F
];

// Computes the binary exponential of logq57.
// input: a log base 2 in Q57 format.
// output: a 64 bit integer in Q0 (no fraction).
// TODO: Mark const once we can use local variables in a const function.
fn bexp64(logq57: i64) -> i64 {
  let ipart = (logq57 >> 57) as i32;
  if ipart < 0 {
    return 0;
  }
  if ipart >= 63 {
    return 0x7FFF_FFFF_FFFF_FFFF;
  }
  // z is the fractional part of the log in Q62 format.
  // We need 1 bit of headroom since the magnitude can get larger than 1
  //  during the iteration, and a sign bit.
  let mut z = logq57 - q57(ipart);
  let mut w: i64;
  if z != 0 {
    // Rust has 128 bit multiplies, so it should be possible to do this
    //  faster without losing accuracy.
    z <<= 5;
    // w is the exponential in Q61 format (since it also needs headroom and can
    //  get as large as 2.0); we could get another bit if we dropped the sign,
    //  but we'll recover that bit later anyway.
    // Ideally this should start out as
    //   \lim_{n->\infty} 2^{61}/\product_{i=1}^n \sqrt{1-2^{-2i}}
    //  but in order to guarantee convergence we have to repeat iterations 4,
    //  13 (=3*4+1), and 40 (=3*13+1, etc.), so it winds up somewhat larger.
    w = 0x26A3_D0E4_01DD_846D;
    let mut i: i64 = 0;
    loop {
      let mask = -((z < 0) as i64);
      w += ((w >> (i + 1)) + mask) ^ mask;
      z -= (ATANH_LOG2[i as usize] + mask) ^ mask;
      // Repeat iteration 4.
      if i >= 3 {
        break;
      }
      z *= 2;
      i += 1;
    }
    loop {
      let mask = -((z < 0) as i64);
      w += ((w >> (i + 1)) + mask) ^ mask;
      z -= (ATANH_LOG2[i as usize] + mask) ^ mask;
      // Repeat iteration 13.
      if i >= 12 {
        break;
      }
      z *= 2;
      i += 1;
    }
    while i < 32 {
      let mask = -((z < 0) as i64);
      w += ((w >> (i + 1)) + mask) ^ mask;
      z = (z - ((ATANH_LOG2[i as usize] + mask) ^ mask)) * 2;
      i += 1;
    }
    // Skip the remaining iterations unless we really require that much
    //  precision.
    // We could have bailed out earlier for smaller iparts, but that would
    //  require initializing w from a table, as the limit doesn't converge to
    //  61-bit precision until n=30.
    let mut wlo: i32 = 0;
    if ipart > 30 {
      // For these iterations, we just update the low bits, as the high bits
      //  can't possibly be affected.
      // OD_ATANH_LOG2 has also converged (it actually did so one iteration
      //  earlier, but that's no reason for an extra special case).
      loop {
        let mask = -((z < 0) as i64);
        wlo += (((w >> i) + mask) ^ mask) as i32;
        z -= (ATANH_LOG2[31] + mask) ^ mask;
        // Repeat iteration 40.
        if i >= 39 {
          break;
        }
        z *= 2;
        i += 1;
      }
      while i < 61 {
        let mask = -((z < 0) as i64);
        wlo += (((w >> i) + mask) ^ mask) as i32;
        z = (z - ((ATANH_LOG2[31] + mask) ^ mask)) * 2;
        i += 1;
      }
    }
    w = (w << 1) + (wlo as i64);
  } else {
    w = 1i64 << 62;
  }
  if ipart < 62 {
    w = ((w >> (61 - ipart)) + 1) >> 1;
  }
  w
}

// Computes the binary log of w.
// input: a 64-bit integer in Q0 (no fraction).
// output: a 64-bit log in Q57.
// TODO: Mark const once we can use local variables in a const function.
fn blog64(w: i64) -> i64 {
  let mut w = w;
  if w <= 0 {
    return -1;
  }
  let ipart = ilog64(w) - 1;
  if ipart > 61 {
    w >>= ipart - 61;
  } else {
    w <<= 61 - ipart;
  }
  // z is the fractional part of the log in Q61 format.
  let mut z: i64 = 0;
  if (w & (w - 1)) != 0 {
    // Rust has 128 bit multiplies, so it should be possible to do this
    //  faster without losing accuracy.
    // x and y are the cosh() and sinh(), respectively, in Q61 format.
    // We are computing z = 2*atanh(y/x) = 2*atanh((w - 1)/(w + 1)).
    let mut x = w + (1i64 << 61);
    let mut y = w - (1i64 << 61);
    for i in 0..4 {
      let mask = -((y < 0) as i64);
      z += ((ATANH_LOG2[i as usize] >> i) + mask) ^ mask;
      let u = x >> (i + 1);
      x -= ((y >> (i + 1)) + mask) ^ mask;
      y -= (u + mask) ^ mask;
    }
    // Repeat iteration 4.
    for i in 3..13 {
      let mask = -((y < 0) as i64);
      z += ((ATANH_LOG2[i as usize] >> i) + mask) ^ mask;
      let u = x >> (i + 1);
      x -= ((y >> (i + 1)) + mask) ^ mask;
      y -= (u + mask) ^ mask;
    }
    // Repeat iteration 13.
    for i in 12..32 {
      let mask = -((y < 0) as i64);
      z += ((ATANH_LOG2[i as usize] >> i) + mask) ^ mask;
      let u = x >> (i + 1);
      x -= ((y >> (i + 1)) + mask) ^ mask;
      y -= (u + mask) ^ mask;
    }
    // OD_ATANH_LOG2 has converged.
    for i in 32..40 {
      let mask = -((y < 0) as i64);
      z += ((ATANH_LOG2[31] >> i) + mask) ^ mask;
      let u = x >> (i + 1);
      x -= ((y >> (i + 1)) + mask) ^ mask;
      y -= (u + mask) ^ mask;
    }
    // Repeat iteration 40.
    for i in 39..62 {
      let mask = -((y < 0) as i64);
      z += ((ATANH_LOG2[31] >> i) + mask) ^ mask;
      let u = x >> (i + 1);
      x -= ((y >> (i + 1)) + mask) ^ mask;
      y -= (u + mask) ^ mask;
    }
    z = (z + 8) >> 4;
  }
  q57(ipart) + z
}

// Converts a Q57 fixed-point fraction to Q24 by rounding.
const fn q57_to_q24(v: i64) -> i32 {
  (((v >> 32) + 1) >> 1) as i32
}

// Converts a Q24 fixed-point fraction to Q57.
const fn q24_to_q57(v: i32) -> i64 {
  (v as i64) << 33
}

#[rustfmt::skip]
const ROUGH_TAN_LOOKUP: &[u16; 18] = &[
     0,   358,   722,  1098,  1491,  1910,
  2365,  2868,  3437,  4096,  4881,  5850,
  7094,  8784, 11254, 15286, 23230, 46817
];

// A digital approximation of a 2nd-order low-pass Bessel follower.
// We use this for rate control because it has fast reaction time, but is
//  critically damped.
pub struct IIRBessel2 {
  c: [i32; 2],
  g: i32,
  x: [i32; 2],
  y: [i32; 2]
}

// alpha is Q24 in the range [0,0.5).
// The return value is 5.12.
// TODO: Mark const once we can use local variables in a const function.
fn warp_alpha(alpha: i32) -> i32 {
  let i = ((alpha*36) >> 24).min(16);
  let t0 = ROUGH_TAN_LOOKUP[i as usize];
  let t1 = ROUGH_TAN_LOOKUP[i as usize + 1];
  let d = alpha*36 - (i << 24);
  ((((t0 as i64) << 32) + (((t1 - t0) << 8) as i64)*(d as i64)) >> 32) as i32
}

// Compute Bessel filter coefficients with the specified delay.
// Return: Filter parameters (c[0], c[1], g).
fn iir_bessel2_get_parameters(delay: i32) -> (i32, i32, i32) {
  // This borrows some code from an unreleased version of Postfish.
  // See the recipe at http://unicorn.us.com/alex/2polefilters.html for details
  //  on deriving the filter coefficients.
  // alpha is Q24
  let alpha = (1 << 24)/delay;
  // warp is 7.12 (5.12? the max value is 70386 in Q12).
  let warp = warp_alpha(alpha).max(1) as i64;
  // k1 is 9.12 (6.12?)
  let k1 = 3*warp;
  // k2 is 16.24 (11.24?)
  let k2 = k1*warp;
  // d is 16.15 (10.15?)
  let d = ((((1 << 12) + k1) << 12) + k2 + 256) >> 9;
  // a is 0.32, since d is larger than both 1.0 and k2
  let a = (k2 << 23)/d;
  // ik2 is 25.24
  let ik2 = (1i64 << 48)/k2;
  // b1 is Q56; in practice, the integer ranges between -2 and 2.
  let b1 = 2*a*(ik2 - (1i64 << 24));
  // b2 is Q56; in practice, the integer ranges between -2 and 2.
  let b2 = (1i64 << 56) - ((4*a) << 24) - b1;
  // All of the filter parameters are Q24.
  (
    ((b1 + (1i64 << 31)) >> 32) as i32,
    ((b2 + (1i64 << 31)) >> 32) as i32,
    ((a + 128) >> 8) as i32
  )
}

impl IIRBessel2 {
  pub fn new(delay: i32, value: i32) -> IIRBessel2 {
    let (c0, c1, g) = iir_bessel2_get_parameters(delay);
    IIRBessel2 { c: [c0, c1], g, x: [value, value], y: [value, value] }
  }

  // Re-initialize Bessel filter coefficients with the specified delay.
  // This does not alter the x/y state, but changes the reaction time of the
  //  filter.
  // Altering the time constant of a reactive filter without altering internal
  //  state is something that has to be done carefuly, but our design operates
  //  at high enough delays and with small enough time constant changes to make
  //  it safe.
  pub fn reinit(&mut self, delay: i32) {
    let (c0, c1, g) = iir_bessel2_get_parameters(delay);
    self.c[0] = c0;
    self.c[1] = c1;
    self.g = g;
  }

  pub fn update(&mut self, x: i32) -> i32 {
    let c0 = self.c[0] as i64;
    let c1 = self.c[1] as i64;
    let g = self.g as i64;
    let x0 = self.x[0] as i64;
    let x1 = self.x[1] as i64;
    let y0 = self.y[0] as i64;
    let y1 = self.y[1] as i64;
    let ya = ((((x as i64) + x0*2 + x1)*g + y0*c0 + y1*c1
      + (1i64 << 23)) >> 24) as i32;
    self.x[1] = self.x[0];
    self.x[0] = x;
    self.y[1] = self.y[0];
    self.y[0] = ya;
    ya
  }
}

pub struct RCState {
  // The target bit-rate in bits per second.
  target_bitrate: i32,
  // The number of frames over which to distribute the reservoir usage.
  reservoir_frame_delay: i32,
  // The maximum quantizer index to allow (for the luma AC coefficients, other
  //  quantizers will still be adjusted to match).
  maybe_ac_qi_max: Option<u8>,
  // Will we drop frames to meet bitrate requirements?
  drop_frames: bool,
  // Do we respect the maximum reservoir fullness?
  cap_overflow: bool,
  // Can the reservoir go negative?
  cap_underflow: bool,
  // Two-pass mode state.
  // 0 => 1-pass encoding.
  // 1 => 1st pass of 2-pass encoding.
  // 2 => 2nd pass of 2-pass encoding.
  twopass_state: i32,
  // The log of the number of pixels in a frame in Q57 format.
  log_npixels: i64,
  // The target average bits per frame.
  bits_per_frame: i64,
  // The current bit reservoir fullness (bits available to be used).
  reservoir_fullness: i64,
  // The target buffer fullness.
  // This is where we'd like to be by the last keyframe that appears in the
  //  next reservoir_frame_delay frames.
  reservoir_target: i64,
  // The maximum buffer fullness (total size of the buffer).
  reservoir_max: i64,
  // The log of estimated scale factor for the rate model in Q57 format.
  log_scale: [i64; FRAME_NSUBTYPES],
  // The exponent used in the rate model in Q6 format.
  exp: [u8; FRAME_NSUBTYPES],
  // The log of an estimated scale factor used to obtain the real framerate,
  //  for VFR sources or, e.g., 12 fps content doubled to 24 fps, etc.
  // TODO vfr: log_vfr_scale: i64,
  // Second-order lowpass filters to track scale and VFR.
  scalefilter: [IIRBessel2; FRAME_NSUBTYPES],
  // TODO vfr: vfrfilter: IIRBessel2,
  // The number of frames of each type we have seen, for filter adaptation
  //  purposes.
  nframes: [i32; FRAME_NSUBTYPES],
  inter_delay: [i32; FRAME_NSUBTYPES - 1],
  inter_delay_target: i32,
  // The total accumulated estimation bias.
  rate_bias: i64
}

// TODO: Separate qi values for each color plane.
pub struct QuantizerParameters {
  // The full-precision, unmodulated log quantizer upon which our modulated
  //  quantizer indices are based.
  // This is only used to limit sudden quality changes from frame to frame, and
  //  as such is not adjusted when we encounter buffer overrun or underrun.
  pub log_base_q: i64,
  // The full-precision log quantizer modulated by the current frame type upon
  //  which our quantizer indices are based (including any adjustments to
  //  prevent buffer overrun or underrun).
  // This is used when estimating the scale parameter once we know the actual
  //  bit usage of a frame.
  pub log_target_q: i64,
  pub dc_qi: [u8; 3],
  pub ac_qi: [u8; 3],
  pub lambda: f64
}

const Q57_SQUARE_EXP_SCALE: f64 =
  (2.0 * ::std::f64::consts::LN_2) / ((1i64 << 57) as f64);

// Daala style log-offset for chroma quantizers
fn chroma_offset(log_target_q: i64) -> (i64, i64) {
    let x = log_target_q.max(0);
    // Gradient 0.266 optimized for CIEDE2000+PSNR on subset3
    let y = (x >> 2) + (x >> 6);
    // blog64(7) - blog64(4); blog64(5) - blog64(4)
    (0x19D_5D9F_D501_0B37 - y, 0xA4_D3C2_5E68_DC58 - y)
}

impl QuantizerParameters {
  fn new_from_log_q(
    log_base_q: i64, log_target_q: i64, bit_depth: usize
  ) -> QuantizerParameters {
    let scale = q57(QSCALE + bit_depth as i32 - 8);
    let quantizer = bexp64(log_target_q + scale);
    let (offset_u, offset_v) = chroma_offset(log_target_q);
    let quantizer_u = bexp64(log_target_q + offset_u + scale);
    let quantizer_v = bexp64(log_target_q + offset_v + scale);
    QuantizerParameters {
      log_base_q,
      log_target_q,
      // TODO: Allow lossless mode; i.e. qi == 0.
      dc_qi: [
        select_dc_qi(quantizer, bit_depth).max(1),
        select_dc_qi(quantizer_u, bit_depth).max(1),
        select_dc_qi(quantizer_v, bit_depth).max(1)
      ],
      ac_qi: [
        select_ac_qi(quantizer, bit_depth).max(1),
        select_ac_qi(quantizer_u, bit_depth).max(1),
        select_ac_qi(quantizer_v, bit_depth).max(1)
      ],
      lambda: (::std::f64::consts::LN_2 / 6.0)
        * ((log_target_q as f64) * Q57_SQUARE_EXP_SCALE).exp()
    }
  }
}

impl RCState {
  pub fn new(
    frame_width: i32, frame_height: i32, framerate_num: i64,
    framerate_den: i64, target_bitrate: i32, maybe_ac_qi_max: Option<u8>,
    max_key_frame_interval: i32
  ) -> RCState {
    // The buffer size is set equal to 1.5x the keyframe interval, clamped to
    //  the range [12, 256] frames.
    // The interval is short enough to allow reaction, but long enough to allow
    //  looking into the next GOP (avoiding the case where the last frames
    //  before an I-frame get starved).
    // The 12 frame minimum gives us some chance to distribute bit estimation
    //  errors in the worst case.
    // The 256 frame maximum means we'll require 8-10 seconds of pre-buffering
    // at 24-30 fps, which is not unreasonable.
    let reservoir_frame_delay = clamp((max_key_frame_interval*3) >> 1, 12, 256);
    // TODO: What are the limits on these?
    let npixels = (frame_width as i64)*(frame_height as i64);
    // Insane framerates or frame sizes mean insane bitrates.
    // Let's not get carried away.
    // TODO: Support constraints imposed by levels.
    let bits_per_frame = clamp(
      (target_bitrate as i64)*framerate_den/framerate_num, 32, 0x4000_0000_0000
    );
    let reservoir_max = bits_per_frame*(reservoir_frame_delay as i64);
    // Start with a buffer fullness and fullness target of 50%.
    let reservoir_target = (reservoir_max + 1) >> 1;
    // Pick exponents and initial scales for quantizer selection.
    let ibpp = npixels/bits_per_frame;
    // All of these initial scale/exp values are from Theora, and have not yet
    //  been adapted to AV1, so they're certainly wrong.
    // The B-frame values especially are simply copies of the P-frame values.
    let i_exp: u8;
    let i_log_scale: i64;
    if ibpp < 1 {
      i_exp = 59;
      i_log_scale = blog64(1997) - q57(QSCALE);
    } else if ibpp < 2 {
      i_exp = 55;
      i_log_scale = blog64(1604) - q57(QSCALE);
    } else {
      i_exp = 48;
      i_log_scale = blog64(834) - q57(QSCALE);
    }
    let p_exp: u8;
    let p_log_scale: i64;
    if ibpp < 4 {
      p_exp = 100;
      p_log_scale = blog64(2249) - q57(QSCALE);
    } else if ibpp < 8 {
      p_exp = 95;
      p_log_scale = blog64(1751) - q57(QSCALE);
    } else {
      p_exp = 73;
      p_log_scale = blog64(1260) - q57(QSCALE);
    }
    // TODO: Add support for "golden" P frames.
    RCState {
      target_bitrate,
      reservoir_frame_delay,
      maybe_ac_qi_max,
      // By default, enforce hard buffer constraints.
      drop_frames: true,
      cap_overflow: true,
      cap_underflow: false,
      // TODO: Support multiple passes.
      twopass_state: 0,
      log_npixels: blog64(npixels),
      bits_per_frame,
      reservoir_fullness: reservoir_target,
      reservoir_target,
      reservoir_max,
      log_scale: [i_log_scale, p_log_scale, p_log_scale, p_log_scale],
      exp: [i_exp, p_exp, p_exp, p_exp],
      scalefilter: [
        IIRBessel2::new(4, q57_to_q24(i_log_scale)),
        IIRBessel2::new(INTER_DELAY_TARGET_MIN, q57_to_q24(p_log_scale)),
        IIRBessel2::new(INTER_DELAY_TARGET_MIN, q57_to_q24(p_log_scale)),
        IIRBessel2::new(INTER_DELAY_TARGET_MIN, q57_to_q24(p_log_scale))
      ],
      // TODO VFR
      nframes: [0; FRAME_NSUBTYPES],
      inter_delay: [INTER_DELAY_TARGET_MIN; FRAME_NSUBTYPES - 1],
      inter_delay_target: reservoir_frame_delay >> 1,
      rate_bias: 0
    }
  }

  // TODO: Separate quantizers for Cb and Cr.
  pub fn select_qi<T: Pixel>(
    &self, ctx: &ContextInner<T>, fti: usize, maybe_prev_log_base_q: Option<i64>
  ) -> QuantizerParameters {
    // Is rate control active?
    if self.target_bitrate <= 0 {
      // Rate control is not active.
      // Derive quantizer directly from frame type.
      // TODO: Rename "quantizer" something that indicates it is a quantizer
      //  index, and move it somewhere more sensible (or choose a better way to
      //  parameterize a "quality" configuration parameter).
      let base_qi = ctx.config.quantizer;
      let bit_depth = ctx.config.bit_depth;
      // We use the AC quantizer as the source quantizer since its quantizer
      //  tables have unique entries, while the DC tables do not.
      let ac_quantizer = ac_q(base_qi as u8, 0, bit_depth) as i64;
      // Pick the nearest DC entry since an exact match may be unavailable.
      let dc_qi = select_dc_qi(ac_quantizer, bit_depth);
      let dc_quantizer = dc_q(dc_qi as u8, 0, bit_depth) as i64;
      // Get the log quantizers as Q57.
      let log_ac_q = blog64(ac_quantizer) - q57(QSCALE + bit_depth as i32 - 8);
      let log_dc_q = blog64(dc_quantizer) - q57(QSCALE + bit_depth as i32 - 8);
      // Target the midpoint of the chosen entries.
      let log_base_q = (log_ac_q + log_dc_q + 1) >> 1;
      // Adjust the quantizer for the frame type, result is Q57:
      let log_q = ((log_base_q + (1i64 << 11)) >> 12) * (MQP_Q12[fti] as i64)
        + DQP_Q57[fti];
      QuantizerParameters::new_from_log_q(log_base_q, log_q, bit_depth)
    } else {
      match self.twopass_state {
        // Single pass only right now.
        _ => {
          // Figure out how to re-distribute bits so that we hit our fullness
          //  target before the last keyframe in our current buffer window
          //  (after the current frame), or the end of the buffer window,
          //  whichever comes first.
          // Count the various types and classes of frames.
          let mut nframes: [i32; FRAME_NSUBTYPES] = [0; FRAME_NSUBTYPES];
          let reservoir_frames = ctx.guess_frame_subtypes(&mut nframes,
            self.reservoir_frame_delay);
          // TODO: Scale for VFR.
          // If we've been missing our target, add a penalty term.
          let rate_bias =
            (self.rate_bias/(ctx.idx as i64 + 100))*(reservoir_frames as i64);
          // rate_total is the total bits available over the next
          //  reservoir_frames frames.
          let rate_total = self.reservoir_fullness - self.reservoir_target
            + rate_bias + (reservoir_frames as i64)*self.bits_per_frame;
          // Find a target quantizer that meets our rate target for the
          //  specific mix of frame types we'll have over the next
          //  reservoir_frame frames.
          // We model the rate<->quantizer relationship as
          //  rate = scale*(quantizer**-exp)
          // In this case, we have our desired rate, an exponent selected in
          //  setup, and a scale that's been measured over our frame history,
          //  so we're solving for the quantizer.
          // Exponentiation with arbitrary exponents is expensive, so we work
          //  in the binary log domain (binary exp and log aren't too bad):
          //  rate = exp2(log2(scale) - log2(quantizer)*exp)
          // There's no easy closed form solution, so we bisection searh for it.
          let bit_depth = ctx.config.bit_depth;
          // TODO: Proper handling of lossless.
          let mut log_qlo = blog64(dc_q(0, 0, bit_depth) as i64)
            - q57(QSCALE + bit_depth as i32 - 8);
          // The AC quantizer tables map to values larger than the DC quantizer
          //  tables, so we use that as the upper bound to make sure we can use
          //  the full table if needed.
          let mut log_qhi = blog64(ac_q(self.maybe_ac_qi_max.unwrap_or(255), 0,
            bit_depth) as i64) - q57(QSCALE + bit_depth as i32 - 8);
          let mut log_base_q = (log_qlo + log_qhi) >> 1;
          while log_qlo < log_qhi {
            // Count bits contributed by each frame type using the model.
            let mut bits = 0i64;
            for ftj in 0..FRAME_NSUBTYPES {
              // Modulate base quantizer by frame type.
              let log_q =
                ((log_base_q + (1i64 << 11)) >> 12)*(MQP_Q12[ftj] as i64)
                + DQP_Q57[ftj];
              // All the fields here are Q57 except for the exponent, which is
              //  Q6.
              bits += (nframes[ftj] as i64)*
                bexp64(self.log_scale[ftj] + self.log_npixels
                - ((log_q + 32) >> 6)*(self.exp[ftj] as i64));
            }
            let diff = bits - rate_total;
            if diff > 0 {
              log_qlo = log_base_q + 1;
            } else if diff < 0 {
              log_qhi = log_base_q - 1;
            } else {
              break;
            }
            log_base_q = (log_qlo + log_qhi) >> 1;
          }
          // If this was not one of the initial frames, limit the change in
          //  base quantizer to within [0.8*Q, 1.2*Q] where Q is the previous
          //  frame's base quantizer.
          if let Some(prev_log_base_q) = maybe_prev_log_base_q {
            log_base_q = clamp(
              log_base_q,
              prev_log_base_q - 0xA4_D3C2_5E68_DC58,
              prev_log_base_q + 0xA4_D3C2_5E68_DC58
            );
          }
          // Modulate base quantizer by frame type.
          let mut log_q =
            ((log_base_q + (1i64 << 11)) >> 12)*(MQP_Q12[fti] as i64)
            + DQP_Q57[fti];
          // The above allocation looks only at the total rate we'll accumulate
          //  in the next reservoir_frame_delay frames.
          // However, we could overflow the bit reservoir on the very next
          //  frame.
          // Check for that here if we're not using a soft target.
          if self.cap_overflow {
            // Allow 3% of the buffer for prediction error.
            // This should be plenty, and we don't mind if we go a bit over.
            // We only want to keep these bits from being completely wasted.
            let margin = (self.reservoir_max + 31) >> 5;
            // We want to use at least this many bits next frame.
            let soft_limit = self.reservoir_fullness + self.bits_per_frame
              - (self.reservoir_max - margin);
            let log_soft_limit = blog64(soft_limit);
            // If we're predicting we won't use that many bits...
            let log_scale_pixels = self.log_scale[fti] + self.log_npixels;
            let exp = self.exp[fti] as i64;
            let mut log_q_exp = ((log_q + 32) >> 6)*exp;
            if log_scale_pixels - log_q_exp < log_soft_limit {
              // Scale the adjustment based on how far into the margin we are.
              log_q_exp +=
                ((log_scale_pixels - log_soft_limit - log_q_exp) >> 32)*
                (margin.min(soft_limit) << 32)/margin;
              log_q = ((log_q_exp + (exp >> 1))/exp) << 6;
            }
          }
          // We just checked we don't overflow the reservoir next frame, now
          //  check we don't underflow and bust the budget (when not using a
          //  soft target).
          if self.maybe_ac_qi_max.is_none() {
            // Compute the maximum number of bits we can use in the next frame.
            // Allow 50% of the rate for a single frame for prediction error.
            // This may not be enough for keyframes or sudden changes in
            //  complexity.
            let log_hard_limit =
              blog64(self.reservoir_fullness + (self.bits_per_frame >> 1));
            // If we're predicting we'll use more than this...
            let log_scale_pixels = self.log_scale[fti] + self.log_npixels;
            let exp = self.exp[fti] as i64;
            let mut log_q_exp = ((log_q + 32) >> 6)*exp;
            if log_scale_pixels - log_q_exp > log_hard_limit {
              // Force the target to hit our limit exactly.
              log_q_exp = log_scale_pixels - log_hard_limit;
              log_q = ((log_q_exp + (exp >> 1))/exp) << 6;
              // If that target is unreasonable, oh well; we'll have to drop.
            }
          }
          QuantizerParameters::new_from_log_q(log_base_q, log_q, bit_depth)
        }
      }
    }
  }

  pub fn update_state(
    &mut self, bits: i64, fti: usize, log_target_q: i64, droppable: bool
  ) -> bool {
    let mut dropped = false;
    // Update rate control only if rate control is active.
    if self.target_bitrate > 0 {
      let log_q_exp = ((log_target_q + 32) >> 6)*(self.exp[fti] as i64);
      let prev_log_scale = self.log_scale[fti];
      let mut bits = bits;
      if bits <= 0 {
        // We didn't code any blocks in this frame.
        bits = 0;
        dropped = true;
        // TODO: Adjust VFR rate based on drop count.
      } else {
        // Compute the estimated scale factor for this frame type.
        let log_bits = blog64(bits);
        let log_scale = (log_bits - self.log_npixels + log_q_exp).min(q57(16));
        let log_scale_q24 = q57_to_q24(log_scale);
        // If this is the first example of the given frame type we've seen, we
        //  immediately replace the default scale factor guess with the
        //  estimate we just computed using the first frame.
        if self.nframes[fti] <= 0 {
          let f = &mut self.scalefilter[fti];
          let x = log_scale_q24;
          f.x[0] = x;
          f.x[1] = x;
          f.y[0] = x;
          f.y[1] = x;
          // TODO: Duplicate regular P frame state for first golden P frame.
        } else {
          // Lengthen the time constant for the inter filters as we collect
          //  more frame statistics, until we reach our target.
          if fti > 0
            && self.inter_delay[fti - 1] < self.inter_delay_target
            && self.nframes[fti] >= self.inter_delay[fti - 1]
          {
            self.inter_delay[fti - 1] += 1;
            self.scalefilter[fti].reinit(self.inter_delay[fti - 1]);
          }
          // Update the low-pass scale filter for this frame type regardless of
          //  whether or not we will ultimately drop this frame.
          self.log_scale[fti] =
            q24_to_q57(self.scalefilter[fti].update(log_scale_q24));
        }
        // If this frame busts our budget, it must be dropped.
        if droppable
          && self.drop_frames
          && self.reservoir_fullness + self.bits_per_frame < bits
        {
          // TODO: Adjust VFR rate based on drop count.
          bits = 0;
          dropped = true;
        } else {
          // TODO: Update a low-pass filter to estimate the "real" frame rate
          //  taking timestamps and drops into account.
          // This is only done if the frame is coded, as it needs the final
          //  count of dropped frames.
        }
        // Increment the frame count for filter adaptation purposes.
        if self.nframes[fti] < ::std::i32::MAX {
          self.nframes[fti] += 1;
        }
      }
      self.reservoir_fullness += self.bits_per_frame - bits;
      // If we're too quick filling the buffer and overflow is capped, that
      //  rate is lost forever.
      if self.cap_overflow {
        self.reservoir_fullness =
          self.reservoir_fullness.min(self.reservoir_max);
      }
      // If we're too quick draining the buffer and underflow is capped, don't
      //  try to make up that rate later.
      if self.cap_underflow {
        self.reservoir_fullness = self.reservoir_fullness.max(0);
      }
      // Adjust the bias for the real bits we've used.
      self.rate_bias +=
        bexp64(prev_log_scale + self.log_npixels - log_q_exp) - bits;
    }
    dropped
  }
}

#[cfg(test)]
mod test {
  use super::{bexp64, blog64};

  #[test]
  fn blog64_vectors() -> () {
    assert!(blog64(1793) == 0x159dc71e24d32daf);
    assert!(blog64(0x678dde6e5fd29f05) == 0x7d6373ad151ca685);
  }

  #[test]
  fn bexp64_vectors() -> () {
    assert!(bexp64(0x159dc71e24d32daf) == 1793);
    assert!((bexp64(0x7d6373ad151ca685) - 0x678dde6e5fd29f05).abs() < 29);
  }

  #[test]
  fn blog64_bexp64_round_trip() {
    for a in 1..=std::u16::MAX as i64 {
      let b = std::i64::MAX / a;
      let (log_a, log_b, log_ab) = (blog64(a), blog64(b), blog64(a * b));
      assert!((log_a + log_b - log_ab).abs() < 4);
      assert!(bexp64(log_a) == a);
      assert!((bexp64(log_b) - b).abs() < 128);
      assert!((bexp64(log_ab) - a * b).abs() < 128);
    }
  }
}
