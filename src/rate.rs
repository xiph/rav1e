// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::encoder::FrameInvariants;
use crate::quantize::ac_q;
use crate::quantize::select_ac_qi;
use crate::quantize::select_dc_qi;

// The number of frame sub-types for which we track distinct parameters.
pub const FRAME_NSUBTYPES: usize = 4;

pub const FRAME_SUBTYPE_I: usize = 0;
pub const FRAME_SUBTYPE_P: usize = 1;
pub const FRAME_SUBTYPE_B0: usize = 2;
pub const FRAME_SUBTYPE_B1: usize = 3;

// The scale of AV1 quantizer tables (relative to the pixel domain), i.e., Q3.
const QSCALE: i32 = 3;

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

#[cfg_attr(rustfmt, rustfmt_skip)]
const ATANH_LOG2: &[i64; 32] = &[
  0x32B803473F7AD0F4, 0x2F2A71BD4E25E916, 0x2E68B244BB93BA06,
  0x2E39FB9198CE62E4, 0x2E2E683F68565C8F, 0x2E2B850BE2077FC1,
  0x2E2ACC58FE7B78DB, 0x2E2A9E2DE52FD5F2, 0x2E2A92A338D53EEC,
  0x2E2A8FC08F5E19B6, 0x2E2A8F07E51A485E, 0x2E2A8ED9BA8AF388,
  0x2E2A8ECE2FE7384A, 0x2E2A8ECB4D3E4B1A, 0x2E2A8ECA94940FE8,
  0x2E2A8ECA6669811D, 0x2E2A8ECA5ADEDD6A, 0x2E2A8ECA57FC347E,
  0x2E2A8ECA57438A43, 0x2E2A8ECA57155FB4, 0x2E2A8ECA5709D510,
  0x2E2A8ECA5706F267, 0x2E2A8ECA570639BD, 0x2E2A8ECA57060B92,
  0x2E2A8ECA57060008, 0x2E2A8ECA5705FD25, 0x2E2A8ECA5705FC6C,
  0x2E2A8ECA5705FC3E, 0x2E2A8ECA5705FC33, 0x2E2A8ECA5705FC30,
  0x2E2A8ECA5705FC2F, 0x2E2A8ECA5705FC2F
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
    return 0x7FFFFFFFFFFFFFFF;
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
    w = 0x26A3D0E401DD846D;
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

pub struct RCState {
  // TODO: Add state needed for actual bitrate targeting.
}

// TODO: Separate qi values for each color plane.
pub struct QuantizerParameters {
  // The full-precision log quantizer modulated by the current frame type upon
  //  which our quantizer indices are based (including any adjustments to
  //  prevent buffer overrun or underrun).
  // This is used when estimating the scale parameter once we know the actual
  //  bit usage of a frame.
  pub log_target_q: i64,
  pub dc_qi: u8,
  pub ac_qi: u8,
  pub lambda: f64
}

const Q57_SQUARE_EXP_SCALE: f64 =
  (2.0 * ::std::f64::consts::LN_2) / ((1i64 << 57) as f64);

impl QuantizerParameters {
  fn new_from_log_q(log_target_q: i64, bit_depth: i32) -> QuantizerParameters {
    let quantizer = bexp64(log_target_q + q57(QSCALE + bit_depth - 8));
    QuantizerParameters {
      log_target_q,
      dc_qi: select_dc_qi(quantizer, bit_depth as usize),
      ac_qi: select_ac_qi(quantizer, bit_depth as usize),
      lambda: (::std::f64::consts::LN_2 / 6.0)
        * ((log_target_q as f64) * Q57_SQUARE_EXP_SCALE).exp()
    }
  }
}

impl RCState {
  pub fn new() -> RCState {
    RCState {}
  }

  // TODO: Separate quantizers for Cb and Cr.
  pub fn select_qi(
    &self, fi: &FrameInvariants, fti: usize
  ) -> QuantizerParameters {
    // Rate control is not active.
    // Derive quantizer directly from frame type.
    // TODO: Rename "quantizer" something that indicates it is a quantizer
    //  index, and move it somewhere more sensible (or choose a better way to
    //  parameterize a "quality" configuration parameter).
    let base_qi = fi.config.quantizer;
    let bit_depth = fi.sequence.bit_depth as i32;
    // Adjust base_qi for the frame type.
    // TODO: Adjust the quantizer (not the index) instead to avoid issues with
    //  the non-linearity of the AV1 quantizer tables.
    let qi = if fti == FRAME_SUBTYPE_I {
      let q_boost = 15;
      base_qi.max(1 + q_boost).min(255 + q_boost) - q_boost
    } else {
      let q_drop = 15 * (fti - FRAME_SUBTYPE_P);
      base_qi.min(255 - q_drop) + q_drop
    };
    // We use the AC quantizer as the source quantizer since its quantizer
    //  tables have unique entries, while the DC tables do not.
    let quantizer = ac_q(qi as u8, 0, bit_depth as usize) as i64;
    // Get the log quantizer as Q57.
    let log_q = blog64(quantizer) - q57(QSCALE + bit_depth - 8);
    QuantizerParameters::new_from_log_q(log_q, bit_depth)
  }
}
