// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(safe_extern_statics)]

use context::*;
use partition::PredictionMode::*;
use partition::*;
use plane::*;
use quantize::*;
use std::cmp;
use util::clamp;
use util::ILog;
use DeblockState;
use FrameInvariants;
use FrameState;
use FrameType;

fn deblock_adjusted_level(
  deblock: &DeblockState, block: &Block, pli: usize, vertical: bool
) -> usize {
  let idx = if pli == 0 {
    if vertical {
      0
    } else {
      1
    }
  } else {
    pli + 1
  };

  let level = if deblock.block_deltas_enabled {
    // By-block filter strength delta, if the feature is active.
    let block_delta = if deblock.block_delta_multi {
      block.deblock_deltas[idx] << deblock.block_delta_shift
    } else {
      block.deblock_deltas[0] << deblock.block_delta_shift
    };

    // Add to frame-specified filter strength (Y-vertical, Y-horizontal, U, V)
    clamp(block_delta + deblock.levels[idx] as i8, 0, MAX_LOOP_FILTER as i8)
      as u8
  } else {
    deblock.levels[idx]
  };

  // if fi.seg_feaure_active {
  // rav1e does not yet support segments or segment features
  // }

  // Are delta modifiers for specific references and modes active?  If so, add them too.
  if deblock.deltas_enabled {
    let mode = block.mode;
    let reference = block.ref_frames[0];
    let mode_type =
      if mode >= NEARESTMV && mode != GLOBALMV && mode != GLOBAL_GLOBALMV {
        1
      } else {
        0
      };
    let l5 = level >> 5;
    clamp(
      level as i32
        + ((deblock.ref_deltas[reference] as i32) << l5)
        + if reference == INTRA_FRAME {
          0
        } else {
          (deblock.mode_deltas[mode_type] as i32) << l5
        },
      0,
      MAX_LOOP_FILTER as i32
    ) as usize
  } else {
    level as usize
  }
}

fn deblock_left<'a>(
  bc: &'a BlockContext, in_bo: &BlockOffset, p: &Plane
) -> &'a Block {
  let xdec = p.cfg.xdec;
  let ydec = p.cfg.ydec;

  // This little bit of weirdness is straight out of the spec;
  // subsampled chroma uses odd mi row/col
  let bo = BlockOffset { x: in_bo.x | xdec, y: in_bo.y | ydec };

  // We already know we're not at the upper/left corner, so prev_block is in frame
  bc.at(&bo.with_offset(-1 << xdec, 0))
}

fn deblock_up<'a>(
  bc: &'a BlockContext, in_bo: &BlockOffset, p: &Plane
) -> &'a Block {
  let xdec = p.cfg.xdec;
  let ydec = p.cfg.ydec;

  // This little bit of weirdness is straight out of the spec;
  // subsampled chroma uses odd mi row/col
  let bo = BlockOffset { x: in_bo.x | xdec, y: in_bo.y | ydec };

  // We already know we're not at the upper/left corner, so prev_block is in frame
  bc.at(&bo.with_offset(0, -1 << ydec))
}

// Must be called on a tx edge, and not on a frame edge.  This is enforced above the call.
fn deblock_size(
  block: &Block, prev_block: &Block, p: &Plane, pli: usize, vertical: bool,
  block_edge: bool
) -> usize {
  let xdec = p.cfg.xdec;
  let ydec = p.cfg.ydec;

  // filter application is conditional on skip and block edge
  if !(block_edge
    || !block.skip
    || !prev_block.skip
    || block.ref_frames[0] <= INTRA_FRAME
    || prev_block.ref_frames[0] <= INTRA_FRAME)
  {
    0
  } else {
    let (tx_size, prev_tx_size) = if vertical {
      (cmp::max(block.tx_w >> xdec, 1), cmp::max(prev_block.tx_w >> xdec, 1))
    } else {
      (cmp::max(block.tx_h >> ydec, 1), cmp::max(prev_block.tx_h >> ydec, 1))
    };

    cmp::min(
      if pli == 0 { 14 } else { 6 },
      cmp::min(tx_size, prev_tx_size) << MI_SIZE_LOG2
    )
  }
}

// Must be called on a tx edge
fn deblock_level(
  deblock: &DeblockState, block: &Block, prev_block: &Block, pli: usize,
  vertical: bool
) -> usize {
  let level = deblock_adjusted_level(deblock, block, pli, vertical);
  if level == 0 {
    deblock_adjusted_level(deblock, prev_block, pli, vertical)
  } else {
    level
  }
}

// four taps, 4 outputs (two are trivial)
fn filter_narrow2_4(
  p1: i32, p0: i32, q0: i32, q1: i32, shift: usize
) -> [i32; 4] {
  let filter0 = clamp(p1 - q1, -128 << shift, (128 << shift) - 1);
  let filter1 =
    clamp(filter0 + 3 * (q0 - p0) + 4, -128 << shift, (128 << shift) - 1) >> 3;
  // be certain our optimization removing a clamp is sound
  debug_assert!({
    let base =
      clamp(filter0 + 3 * (q0 - p0), -128 << shift, (128 << shift) - 1);
    let test = clamp(base + 4, -128 << shift, (128 << shift) - 1) >> 3;
    filter1 == test
  });
  let filter2 =
    clamp(filter0 + 3 * (q0 - p0) + 3, -128 << shift, (128 << shift) - 1) >> 3;
  // be certain our optimization removing a clamp is sound
  debug_assert!({
    let base =
      clamp(filter0 + 3 * (q0 - p0), -128 << shift, (128 << shift) - 1);
    let test = clamp(base + 3, -128 << shift, (128 << shift) - 1) >> 3;
    filter2 == test
  });
  [
    p1,
    clamp(p0 + filter2, 0, (256 << shift) - 1),
    clamp(q0 - filter1, 0, (256 << shift) - 1),
    q1
  ]
}

// six taps, 6 outputs (four are trivial)
fn filter_narrow2_6(
  p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32, shift: usize
) -> [i32; 6] {
  let x = filter_narrow2_4(p1, p0, q0, q1, shift);
  [p2, x[0], x[1], x[2], x[3], q2]
}

// 12 taps, 12 outputs (ten are trivial)
fn filter_narrow2_12(
  p5: i32, p4: i32, p3: i32, p2: i32, p1: i32, p0: i32, q0: i32, q1: i32,
  q2: i32, q3: i32, q4: i32, q5: i32, shift: usize
) -> [i32; 12] {
  let x = filter_narrow2_4(p1, p0, q0, q1, shift);
  [p5, p4, p3, p2, x[0], x[1], x[2], x[3], q2, q3, q4, q5]
}

// four taps, 4 outputs
fn filter_narrow4_4(
  p1: i32, p0: i32, q0: i32, q1: i32, shift: usize
) -> [i32; 4] {
  let filter1 =
    clamp(3 * (q0 - p0) + 4, -128 << shift, (128 << shift) - 1) >> 3;
  // be certain our optimization removing a clamp is sound
  debug_assert!({
    let base = clamp(3 * (q0 - p0), -128 << shift, (128 << shift) - 1);
    let test = clamp(base + 4, -128 << shift, (128 << shift) - 1) >> 3;
    filter1 == test
  });
  let filter2 =
    clamp(3 * (q0 - p0) + 3, -128 << shift, (128 << shift) - 1) >> 3;
  // be certain our optimization removing a clamp is sound
  debug_assert!({
    let base = clamp(3 * (q0 - p0), -128 << shift, (128 << shift) - 1);
    let test = clamp(base + 3, -128 << shift, (128 << shift) - 1) >> 3;
    filter2 == test
  });
  let filter3 = filter1 + 1 >> 1;
  [
    clamp(p1 + filter3, 0, (256 << shift) - 1),
    clamp(p0 + filter2, 0, (256 << shift) - 1),
    clamp(q0 - filter1, 0, (256 << shift) - 1),
    clamp(q1 - filter3, 0, (256 << shift) - 1)
  ]
}

// six taps, 6 outputs (two are trivial)
fn filter_narrow4_6(
  p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32, shift: usize
) -> [i32; 6] {
  let x = filter_narrow4_4(p1, p0, q0, q1, shift);
  [p2, x[0], x[1], x[2], x[3], q2]
}

// 12 taps, 12 outputs (eight are trivial)
fn filter_narrow4_12(
  p5: i32, p4: i32, p3: i32, p2: i32, p1: i32, p0: i32, q0: i32, q1: i32,
  q2: i32, q3: i32, q4: i32, q5: i32, shift: usize
) -> [i32; 12] {
  let x = filter_narrow4_4(p1, p0, q0, q1, shift);
  [p5, p4, p3, p2, x[0], x[1], x[2], x[3], q2, q3, q4, q5]
}

// six taps, 4 outputs
#[cfg_attr(rustfmt, rustfmt_skip)]
fn filter_wide6_4(p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32) -> [i32; 4] {
    [p2*3 + p1*2 + p0*2 + q0   + (1<<2) >> 3,
     p2   + p1*2 + p0*2 + q0*2 + q1   + (1<<2) >> 3,
            p1   + p0*2 + q0*2 + q1*2 + q2   + (1<<2) >> 3,
                   p0   + q0*2 + q1*2 + q2*3 + (1<<2) >> 3]
}

// eight taps, 6 outputs
#[cfg_attr(rustfmt, rustfmt_skip)]
fn filter_wide8_6(p3: i32, p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32, q3: i32) -> [i32; 6] {
    [p3*3 + p2*2 + p1   + p0   + q0   + (1<<2) >> 3,
     p3*2 + p2   + p1*2 + p0   + q0   + q1   + (1<<2) >> 3,
     p3   + p2   + p1   + p0*2 + q0   + q1   + q2   +(1<<2) >> 3,
            p2   + p1   + p0   + q0*2 + q1   + q2   + q3   + (1<<2) >> 3,
                   p1   + p0   + q0   + q1*2 + q2   + q3*2 + (1<<2) >> 3,
                          p0   + q0   + q1   + q2*2 + q3*3 + (1<<2) >> 3]
}

// 12 taps, 12 outputs (six are trivial)
fn filter_wide8_12(
  p5: i32, p4: i32, p3: i32, p2: i32, p1: i32, p0: i32, q0: i32, q1: i32,
  q2: i32, q3: i32, q4: i32, q5: i32
) -> [i32; 12] {
  let x = filter_wide8_6(p3, p2, p1, p0, q0, q1, q2, q3);
  [p5, p4, p3, x[0], x[1], x[2], x[3], x[4], x[5], q3, q4, q5]
}

// fourteen taps, 12 outputs
#[cfg_attr(rustfmt, rustfmt_skip)]
fn filter_wide14_12(p6: i32, p5: i32, p4: i32, p3: i32, p2: i32, p1: i32, p0: i32,
                    q0: i32, q1: i32, q2: i32, q3: i32, q4: i32, q5: i32, q6: i32) -> [i32; 12] {
    [p6*7 + p5*2 + p4*2 + p3   + p2   + p1   + p0   + q0   + (1<<3) >> 4,
     p6*5 + p5*2 + p4*2 + p3*2 + p2   + p1   + p0   + q0   + q1   + (1<<3) >> 4,
     p6*4 + p5   + p4*2 + p3*2 + p2*2 + p1   + p0   + q0   + q1   + q2   + (1<<3) >> 4,
     p6*3 + p5   + p4   + p3*2 + p2*2 + p1*2 + p0   + q0   + q1   + q2   + q3   + (1<<3) >> 4,
     p6*2 + p5   + p4   + p3   + p2*2 + p1*2 + p0*2 + q0   + q1   + q2   + q3   + q4   + (1<<3) >> 4,
     p6   + p5   + p4   + p3   + p2   + p1*2 + p0*2 + q0*2 + q1   + q2   + q3   + q4   + q5   + (1<<3) >> 4,
            p5   + p4   + p3   + p2   + p1   + p0*2 + q0*2 + q1*2 + q2   + q3   + q4   + q5   + q6 + (1<<3) >> 4,
                   p4   + p3   + p2   + p1   + p0   + q0*2 + q1*2 + q2*2 + q3   + q4   + q5   + q6*2 + (1<<3) >> 4,
                          p3   + p2   + p1   + p0   + q0   + q1*2 + q2*2 + q3*2 + q4   + q5   + q6*3 + (1<<3) >> 4,
                                 p2   + p1   + p0   + q0   + q1   + q2*2 + q3*2 + q4*2 + q5   + q6*4 + (1<<3) >> 4,
                                        p1   + p0   + q0   + q1   + q2   + q3*2 + q4*2 + q5*2 + q6*5 + (1<<3) >> 4,
                                               p0   + q0   + q1   + q2   + q3   + q4*2 + q5*2 + q6*7 + (1<<3) >> 4]
}

fn stride_copy(dst: &mut [u16], src: &[i32], pitch: usize) {
  for (dst, src) in dst.iter_mut().step_by(pitch).take(src.len()).zip(src) {
    *dst = *src as u16
  }
}

fn stride_sse(a: &[u16], b: &[i32], pitch: usize) -> i64 {
  let mut acc: i32 = 0;
  for (a, b) in a.iter().step_by(pitch).take(b.len()).zip(b) {
    acc += (*a as i32 - *b) * (*a as i32 - *b)
  }
  acc as i64
}

fn _level_to_limit(level: i32, shift: usize) -> i32 {
  level << shift
}

fn limit_to_level(limit: i32, shift: usize) -> i32 {
  limit + (1 << shift) - 1 >> shift
}

fn _level_to_blimit(level: i32, shift: usize) -> i32 {
  3 * level + 4 << shift
}

fn blimit_to_level(blimit: i32, shift: usize) -> i32 {
  ((blimit + (1 << shift) - 1 >> shift) - 2) / 3
}

fn _level_to_thresh(level: i32, shift: usize) -> i32 {
  level >> 4 << shift
}

fn thresh_to_level(thresh: i32, shift: usize) -> i32 {
  thresh + (1 << shift) - 1 >> shift << 4
}

fn nhev4(p1: i32, p0: i32, q0: i32, q1: i32, shift: usize) -> usize {
  thresh_to_level(cmp::max((p1 - p0).abs(), (q1 - q0).abs()), shift) as usize
}

fn mask4(p1: i32, p0: i32, q0: i32, q1: i32, shift: usize) -> usize {
  cmp::max(
    limit_to_level(cmp::max((p1 - p0).abs(), (q1 - q0).abs()), shift),
    blimit_to_level((p0 - q0).abs() * 2 + (p1 - q1).abs() / 2, shift)
  ) as usize
}

// Assumes rec[0] is set 2 taps back from the edge
fn deblock_size4(
  rec: &mut [u16], pitch: usize, stride: usize, level: usize, bd: usize
) {
  let mut s = 0;
  for _i in 0..4 {
    let p = &mut rec[s..];
    let p1 = p[0] as i32;
    let p0 = p[pitch] as i32;
    let q0 = p[pitch * 2] as i32;
    let q1 = p[pitch * 3] as i32;
    if mask4(p1, p0, q0, q1, bd - 8) <= level {
      let x;
      if nhev4(p1, p0, q0, q1, bd - 8) <= level {
        x = filter_narrow4_4(p1, p0, q0, q1, bd - 8);
      } else {
        x = filter_narrow2_4(p1, p0, q0, q1, bd - 8);
      }
      stride_copy(p, &x, pitch);
    }
    s += stride;
  }
}

// Assumes rec[0] and src[0] are set 2 taps back from the edge.
// Accesses four taps, accumulates four pixels into the tally
fn sse_size4(
  rec: &[u16], src: &[u16], tally: &mut [i64; MAX_LOOP_FILTER + 2],
  rec_pitch: usize, src_pitch: usize, rec_stride: usize, src_stride: usize,
  bd: usize
) {
  let mut rec_s = 0;
  let mut src_s = 0;
  for _i in 0..4 {
    let p = &rec[rec_s..]; // four taps
    let a = &src[src_s..]; // four pixels to compare
    let p1 = p[0] as i32;
    let p0 = p[rec_pitch] as i32;
    let q0 = p[rec_pitch * 2] as i32;
    let q1 = p[rec_pitch * 3] as i32;

    // three possibilities: no filter, narrow2 and narrow4
    // All possibilities produce four outputs
    let none: [_; 4] = [p1, p0, q0, q1];
    let narrow2 = filter_narrow2_4(p1, p0, q0, q1, bd - 8);
    let narrow4 = filter_narrow4_4(p1, p0, q0, q1, bd - 8);

    // mask4 sets the dividing line for filter vs no filter
    // nhev4 sets the dividing line between narrow2 and narrow4
    let mask =
      clamp(mask4(p1, p0, q0, q1, bd - 8), 1, MAX_LOOP_FILTER + 1) as usize;
    let nhev =
      clamp(nhev4(p1, p0, q0, q1, bd - 8), mask, MAX_LOOP_FILTER + 1) as usize;

    // sse for each; short-circuit the 'special' no-op cases.
    let sse_none = stride_sse(a, &none, src_pitch);
    let sse_narrow2 =
      if nhev != mask { stride_sse(a, &narrow2, src_pitch) } else { sse_none };
    let sse_narrow4 = if nhev <= MAX_LOOP_FILTER {
      stride_sse(a, &narrow4, src_pitch)
    } else {
      sse_none
    };

    // accumulate possible filter values into the tally
    // level 0 is a special case
    tally[0] += sse_none;
    tally[mask] -= sse_none;
    tally[mask] += sse_narrow2;
    tally[nhev] -= sse_narrow2;
    tally[nhev] += sse_narrow4;

    rec_s += rec_stride;
    src_s += src_stride;
  }
}

fn mask6(
  p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32, shift: usize
) -> usize {
  cmp::max(
    limit_to_level(
      cmp::max(
        (p2 - p1).abs(),
        cmp::max((p1 - p0).abs(), cmp::max((q2 - q1).abs(), (q1 - q0).abs()))
      ),
      shift
    ),
    blimit_to_level((p0 - q0).abs() * 2 + (p1 - q1).abs() / 2, shift)
  ) as usize
}

fn flat6(p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32) -> usize {
  cmp::max(
    (p1 - p0).abs(),
    cmp::max((q1 - q0).abs(), cmp::max((p2 - p0).abs(), (q2 - q0).abs()))
  ) as usize
}

// Assumes slice[0] is set 3 taps back from the edge
fn deblock_size6(
  rec: &mut [u16], pitch: usize, stride: usize, level: usize, bd: usize
) {
  let mut s = 0;
  let flat = 1 << bd - 8;
  for _i in 0..4 {
    let p = &mut rec[s..];
    let p2 = p[0] as i32;
    let p1 = p[pitch] as i32;
    let p0 = p[pitch * 2] as i32;
    let q0 = p[pitch * 3] as i32;
    let q1 = p[pitch * 4] as i32;
    let q2 = p[pitch * 5] as i32;
    if mask6(p2, p1, p0, q0, q1, q2, bd - 8) <= level {
      let x;
      if flat6(p2, p1, p0, q0, q1, q2) <= flat {
        x = filter_wide6_4(p2, p1, p0, q0, q1, q2);
      } else if nhev4(p1, p0, q0, q1, bd - 8) <= level {
        x = filter_narrow4_4(p1, p0, q0, q1, bd - 8);
      } else {
        x = filter_narrow2_4(p1, p0, q0, q1, bd - 8);
      }
      stride_copy(&mut p[pitch..], &x, pitch);
    }
    s += stride;
  }
}

// Assumes rec[0] and src[0] are set 3 taps back from the edge.
// Accesses six taps, accumulates four pixels into the tally
fn sse_size6(
  rec: &[u16], src: &[u16], tally: &mut [i64; MAX_LOOP_FILTER + 2],
  rec_pitch: usize, src_pitch: usize, rec_stride: usize, src_stride: usize,
  bd: usize
) {
  let mut rec_s = 0;
  let mut src_s = 0;
  let flat = 1 << bd - 8;
  for _i in 0..4 {
    let p = &rec[rec_s..]; // six taps
    let a = &src[src_s + src_pitch..]; // four pixels to compare so offset one forward
    let p2 = p[0] as i32;
    let p1 = p[rec_pitch] as i32;
    let p0 = p[rec_pitch * 2] as i32;
    let q0 = p[rec_pitch * 3] as i32;
    let q1 = p[rec_pitch * 4] as i32;
    let q2 = p[rec_pitch * 5] as i32;

    // Four possibilities: no filter, wide6, narrow2 and narrow4
    // All possibilities produce four outputs
    let none: [_; 4] = [p1, p0, q0, q1];
    let wide6 = filter_wide6_4(p2, p1, p0, q0, q1, q2);
    let narrow2 = filter_narrow2_4(p1, p0, q0, q1, bd - 8);
    let narrow4 = filter_narrow4_4(p1, p0, q0, q1, bd - 8);

    // mask6 sets the dividing line for filter vs no filter
    // flat6 decides between wide and narrow filters (unrelated to level)
    // nhev4 sets the dividing line between narrow2 and narrow4
    let mask =
      clamp(mask6(p2, p1, p0, q0, q1, q2, bd - 8), 1, MAX_LOOP_FILTER + 1)
        as usize;
    let flatp = flat6(p2, p1, p0, q0, q1, q2) <= flat;
    let nhev =
      clamp(nhev4(p1, p0, q0, q1, bd - 8), mask, MAX_LOOP_FILTER + 1) as usize;

    // sse for each; short-circuit the 'special' no-op cases.
    let sse_none = stride_sse(a, &none, src_pitch);
    let sse_wide6 = if flatp && mask <= MAX_LOOP_FILTER {
      stride_sse(a, &wide6, src_pitch)
    } else {
      sse_none
    };
    let sse_narrow2 = if !flatp && nhev != mask {
      stride_sse(a, &narrow2, src_pitch)
    } else {
      sse_none
    };
    let sse_narrow4 = if !flatp && nhev <= MAX_LOOP_FILTER {
      stride_sse(a, &narrow4, src_pitch)
    } else {
      sse_none
    };

    // accumulate possible filter values into the tally
    tally[0] += sse_none;
    tally[mask] -= sse_none;
    if flatp {
      tally[mask] += sse_wide6;
    } else {
      tally[mask] += sse_narrow2;
      tally[nhev] -= sse_narrow2;
      tally[nhev] += sse_narrow4;
    }

    rec_s += rec_stride;
    src_s += src_stride;
  }
}

fn mask8(
  p3: i32, p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32, q3: i32,
  shift: usize
) -> usize {
  cmp::max(
    limit_to_level(
      cmp::max(
        (p3 - p2).abs(),
        cmp::max(
          (p2 - p1).abs(),
          cmp::max(
            (p1 - p0).abs(),
            cmp::max(
              (q3 - q2).abs(),
              cmp::max((q2 - q1).abs(), (q1 - q0).abs())
            )
          )
        )
      ),
      shift
    ),
    blimit_to_level((p0 - q0).abs() * 2 + (p1 - q1).abs() / 2, shift)
  ) as usize
}

fn flat8(
  p3: i32, p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32, q3: i32
) -> usize {
  cmp::max(
    (p1 - p0).abs(),
    cmp::max(
      (q1 - q0).abs(),
      cmp::max(
        (p2 - p0).abs(),
        cmp::max((q2 - q0).abs(), cmp::max((p3 - p0).abs(), (q3 - q0).abs()))
      )
    )
  ) as usize
}

// Assumes rec[0] is set 4 taps back from the edge
fn deblock_size8(
  rec: &mut [u16], pitch: usize, stride: usize, level: usize, bd: usize
) {
  let mut s = 0;
  let flat = 1 << bd - 8;
  for _i in 0..4 {
    let p = &mut rec[s..];
    let p3 = p[0] as i32;
    let p2 = p[pitch] as i32;
    let p1 = p[pitch * 2] as i32;
    let p0 = p[pitch * 3] as i32;
    let q0 = p[pitch * 4] as i32;
    let q1 = p[pitch * 5] as i32;
    let q2 = p[pitch * 6] as i32;
    let q3 = p[pitch * 7] as i32;
    if mask8(p3, p2, p1, p0, q0, q1, q2, q3, bd - 8) <= level {
      let x: [i32; 6];
      if flat8(p3, p2, p1, p0, q0, q1, q2, q3) <= flat {
        x = filter_wide8_6(p3, p2, p1, p0, q0, q1, q2, q3);
      } else {
        if nhev4(p1, p0, q0, q1, bd - 8) <= level {
          x = filter_narrow4_6(p2, p1, p0, q0, q1, q2, bd - 8);
        } else {
          x = filter_narrow2_6(p2, p1, p0, q0, q1, q2, bd - 8);
        }
      }
      stride_copy(&mut p[pitch..], &x, pitch);
    }
    s += stride;
  }
}

// Assumes rec[0] and src[0] are set 4 taps back from the edge.
// Accesses eight taps, accumulates six pixels into the tally
fn sse_size8(
  rec: &[u16], src: &[u16], tally: &mut [i64; MAX_LOOP_FILTER + 2],
  rec_pitch: usize, src_pitch: usize, rec_stride: usize, src_stride: usize,
  bd: usize
) {
  let mut rec_s = 0;
  let mut src_s = 0;
  let flat = 1 << bd - 8;
  for _i in 0..4 {
    let p = &rec[rec_s..]; // eight taps
    let a = &src[src_s + src_pitch..]; // six pixels to compare so offset one forward
    let p3 = p[0] as i32;
    let p2 = p[rec_pitch] as i32;
    let p1 = p[rec_pitch * 2] as i32;
    let p0 = p[rec_pitch * 3] as i32;
    let q0 = p[rec_pitch * 4] as i32;
    let q1 = p[rec_pitch * 5] as i32;
    let q2 = p[rec_pitch * 6] as i32;
    let q3 = p[rec_pitch * 7] as i32;

    // Four possibilities: no filter, wide8, narrow2 and narrow4
    let none: [_; 6] = [p2, p1, p0, q0, q1, q2];
    let wide8: [_; 6] = filter_wide8_6(p3, p2, p1, p0, q0, q1, q2, q3);
    let narrow2: [_; 6] = filter_narrow2_6(p2, p1, p0, q0, q1, q2, bd - 8);
    let narrow4: [_; 6] = filter_narrow4_6(p2, p1, p0, q0, q1, q2, bd - 8);

    // mask8 sets the dividing line for filter vs no filter
    // flat8 decides between wide and narrow filters (unrelated to level)
    // nhev4 sets the dividing line between narrow2 and narrow4
    let mask = clamp(
      mask8(p3, p2, p1, p0, q0, q1, q2, q3, bd - 8),
      1,
      MAX_LOOP_FILTER + 1
    ) as usize;
    let flatp = flat8(p3, p2, p1, p0, q0, q1, q2, q3) <= flat;
    let nhev =
      clamp(nhev4(p1, p0, q0, q1, bd - 8), mask, MAX_LOOP_FILTER + 1) as usize;

    // sse for each; short-circuit the 'special' no-op cases.
    let sse_none = stride_sse(a, &none, src_pitch);
    let sse_wide8 = if flatp && mask <= MAX_LOOP_FILTER {
      stride_sse(a, &wide8, src_pitch)
    } else {
      sse_none
    };
    let sse_narrow2 = if !flatp && nhev != mask {
      stride_sse(a, &narrow2, src_pitch)
    } else {
      sse_none
    };
    let sse_narrow4 = if !flatp && nhev <= MAX_LOOP_FILTER {
      stride_sse(a, &narrow4, src_pitch)
    } else {
      sse_none
    };

    // accumulate possible filter values into the tally
    tally[0] += sse_none;
    tally[mask] -= sse_none;
    if flatp {
      tally[mask] += sse_wide8;
    } else {
      tally[mask] += sse_narrow2;
      tally[nhev] -= sse_narrow2;
      tally[nhev] += sse_narrow4;
    }

    src_s += src_stride;
    rec_s += rec_stride;
  }
}

fn flat14_outer(
  p6: i32, p5: i32, p4: i32, p0: i32, q0: i32, q4: i32, q5: i32, q6: i32
) -> usize {
  cmp::max(
    (p4 - p0).abs(),
    cmp::max(
      (q4 - q0).abs(),
      cmp::max(
        (p5 - p0).abs(),
        cmp::max((q5 - q0).abs(), cmp::max((p6 - p0).abs(), (q6 - q0).abs()))
      )
    )
  ) as usize
}

// Assumes rec[0] is set 7 taps back from the edge
fn deblock_size14(
  rec: &mut [u16], pitch: usize, stride: usize, level: usize, bd: usize
) {
  let mut s = 0;
  let flat = 1 << bd - 8;
  for _i in 0..4 {
    let p = &mut rec[s..];
    let p6 = p[0] as i32;
    let p5 = p[pitch] as i32;
    let p4 = p[pitch * 2] as i32;
    let p3 = p[pitch * 3] as i32;
    let p2 = p[pitch * 4] as i32;
    let p1 = p[pitch * 5] as i32;
    let p0 = p[pitch * 6] as i32;
    let q0 = p[pitch * 7] as i32;
    let q1 = p[pitch * 8] as i32;
    let q2 = p[pitch * 9] as i32;
    let q3 = p[pitch * 10] as i32;
    let q4 = p[pitch * 11] as i32;
    let q5 = p[pitch * 12] as i32;
    let q6 = p[pitch * 13] as i32;
    // 'mask' test
    if mask8(p3, p2, p1, p0, q0, q1, q2, q3, bd - 8) <= level {
      let x: [i32; 12];
      // inner flatness test
      if flat8(p3, p2, p1, p0, q0, q1, q2, q3) <= flat {
        // outer flatness test
        if flat14_outer(p6, p5, p4, p0, q0, q4, q5, q6) <= flat {
          // sufficient flatness across 14 pixel width; run full-width filter
          x = filter_wide14_12(
            p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6,
          );
        } else {
          // only flat in inner area, run 8-tap
          x = filter_wide8_12(p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5);
        }
      } else {
        // not flat, run narrow filter
        if nhev4(p1, p0, q0, q1, bd - 8) <= level {
          x = filter_narrow4_12(
            p5,
            p4,
            p3,
            p2,
            p1,
            p0,
            q0,
            q1,
            q2,
            q3,
            q4,
            q5,
            bd - 8
          );
        } else {
          x = filter_narrow2_12(
            p5,
            p4,
            p3,
            p2,
            p1,
            p0,
            q0,
            q1,
            q2,
            q3,
            q4,
            q5,
            bd - 8
          );
        }
      }
      stride_copy(&mut p[pitch..], &x, pitch);
    }
    s += stride;
  }
}

// Assumes rec[0] and src[0] are set 7 taps back from the edge.
// Accesses fourteen taps, accumulates twelve pixels into the tally
fn sse_size14(
  rec: &[u16], src: &[u16], tally: &mut [i64; MAX_LOOP_FILTER + 2],
  rec_pitch: usize, src_pitch: usize, rec_stride: usize, src_stride: usize,
  bd: usize
) {
  let mut rec_s = 0;
  let mut src_s = 0;
  let flat = 1 << bd - 8;
  for _i in 0..4 {
    let p = &rec[rec_s..]; // 14 taps
    let a = &src[src_s + src_pitch..]; // 12 pixels to compare so offset one forward
    let p6 = p[0] as i32;
    let p5 = p[rec_pitch] as i32;
    let p4 = p[rec_pitch * 2] as i32;
    let p3 = p[rec_pitch * 3] as i32;
    let p2 = p[rec_pitch * 4] as i32;
    let p1 = p[rec_pitch * 5] as i32;
    let p0 = p[rec_pitch * 6] as i32;
    let q0 = p[rec_pitch * 7] as i32;
    let q1 = p[rec_pitch * 8] as i32;
    let q2 = p[rec_pitch * 9] as i32;
    let q3 = p[rec_pitch * 10] as i32;
    let q4 = p[rec_pitch * 11] as i32;
    let q5 = p[rec_pitch * 12] as i32;
    let q6 = p[rec_pitch * 13] as i32;

    // Five possibilities: no filter, wide14, wide8, narrow2 and narrow4
    let none: [i32; 12] = [p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5];
    let wide14 =
      filter_wide14_12(p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6);
    let wide8 =
      filter_wide8_12(p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5);
    let narrow2 = filter_narrow2_12(
      p5,
      p4,
      p3,
      p2,
      p1,
      p0,
      q0,
      q1,
      q2,
      q3,
      q4,
      q5,
      bd - 8
    );
    let narrow4 = filter_narrow4_12(
      p5,
      p4,
      p3,
      p2,
      p1,
      p0,
      q0,
      q1,
      q2,
      q3,
      q4,
      q5,
      bd - 8
    );

    // mask8 sets the dividing line for filter vs no filter
    // flat8 decides between wide and narrow filters (unrelated to level)
    // flat14 decides between wide14 and wide8 filters
    // nhev4 sets the dividing line between narrow2 and narrow4
    let mask = clamp(
      mask8(p3, p2, p1, p0, q0, q1, q2, q3, bd - 8),
      1,
      MAX_LOOP_FILTER + 1
    ) as usize;
    let flat8p = flat8(p3, p2, p1, p0, q0, q1, q2, q3) <= flat;
    let flat14p = flat14_outer(p6, p5, p4, p0, q0, q4, q5, q6) <= flat;
    let nhev =
      clamp(nhev4(p1, p0, q0, q1, bd - 8), mask, MAX_LOOP_FILTER + 1) as usize;

    // sse for each; short-circuit the 'special' no-op cases.
    let sse_none = stride_sse(a, &none, src_pitch);
    let sse_wide8 = if flat8p && !flat14p && mask <= MAX_LOOP_FILTER {
      stride_sse(a, &wide8, src_pitch)
    } else {
      sse_none
    };
    let sse_wide14 = if flat8p && flat14p && mask <= MAX_LOOP_FILTER {
      stride_sse(a, &wide14, src_pitch)
    } else {
      sse_none
    };
    let sse_narrow2 = if !flat8p && nhev != mask {
      stride_sse(a, &narrow2, src_pitch)
    } else {
      sse_none
    };
    let sse_narrow4 = if !flat8p && nhev <= MAX_LOOP_FILTER {
      stride_sse(a, &narrow4, src_pitch)
    } else {
      sse_none
    };

    // accumulate possible filter values into the tally
    tally[0] += sse_none;
    tally[mask] -= sse_none;
    if flat8p {
      if flat14p {
        tally[mask] += sse_wide14;
      } else {
        tally[mask] += sse_wide8;
      }
    } else {
      tally[mask] += sse_narrow2;
      tally[nhev] -= sse_narrow2;
      tally[nhev] += sse_narrow4;
    }

    rec_s += rec_stride;
    src_s += src_stride;
  }
}

fn filter_v_edge(
  deblock: &DeblockState, bc: &BlockContext, bo: &BlockOffset, p: &mut Plane,
  pli: usize, bd: usize
) {
  let block = bc.at(&bo);
  let tx_edge = bo.x & (block.tx_w - 1) == 0;
  if tx_edge {
    let prev_block = deblock_left(bc, bo, p);
    let block_edge = bo.x & (block.n4_w - 1) == 0;
    let filter_size =
      deblock_size(block, prev_block, p, pli, true, block_edge);
    if filter_size > 0 {
      let level = deblock_level(deblock, block, prev_block, pli, true);
      if level > 0 {
        let po = bo.plane_offset(&p.cfg);
        let stride = p.cfg.stride;
        let mut plane_slice = p.mut_slice(&po);
        plane_slice.x -= (filter_size >> 1) as isize;
        let slice = plane_slice.as_mut_slice();
        match filter_size {
          4 => {
            deblock_size4(slice, 1, stride, level, bd);
          }
          6 => {
            deblock_size6(slice, 1, stride, level, bd);
          }
          8 => {
            deblock_size8(slice, 1, stride, level, bd);
          }
          14 => {
            deblock_size14(slice, 1, stride, level, bd);
          }
          _ => unreachable!()
        }
      }
    }
  }
}

fn sse_v_edge(
  bc: &BlockContext, bo: &BlockOffset, rec_plane: &Plane, src_plane: &Plane,
  tally: &mut [i64; MAX_LOOP_FILTER + 2], pli: usize, bd: usize
) {
  let block = bc.at(&bo);
  let tx_edge = bo.x & (block.tx_w - 1) == 0;
  if tx_edge {
    let prev_block = deblock_left(bc, bo, rec_plane);
    let block_edge = bo.x & (block.n4_w - 1) == 0;
    let filter_size =
      deblock_size(block, prev_block, rec_plane, pli, true, block_edge);
    if filter_size > 0 {
      let po = bo.plane_offset(&rec_plane.cfg); // rec and src have identical subsampling
      let rec_slice = rec_plane.slice(&po);
      let src_slice = src_plane.slice(&po);
      let rec_tmp = rec_slice.go_left(filter_size >> 1);
      let src_tmp = src_slice.go_left(filter_size >> 1);
      let rec = rec_tmp.as_slice();
      let src = src_tmp.as_slice();
      match filter_size {
        4 => {
          sse_size4(
            rec,
            src,
            tally,
            1,
            1,
            rec_plane.cfg.stride,
            src_plane.cfg.stride,
            bd
          );
        }
        6 => {
          sse_size6(
            rec,
            src,
            tally,
            1,
            1,
            rec_plane.cfg.stride,
            src_plane.cfg.stride,
            bd
          );
        }
        8 => {
          sse_size8(
            rec,
            src,
            tally,
            1,
            1,
            rec_plane.cfg.stride,
            src_plane.cfg.stride,
            bd
          );
        }
        14 => {
          sse_size14(
            rec,
            src,
            tally,
            1,
            1,
            rec_plane.cfg.stride,
            src_plane.cfg.stride,
            bd
          );
        }
        _ => unreachable!()
      }
    }
  }
}

fn filter_h_edge(
  deblock: &DeblockState, bc: &BlockContext, bo: &BlockOffset, p: &mut Plane,
  pli: usize, bd: usize
) {
  let block = bc.at(&bo);
  let tx_edge = bo.y & (block.tx_h - 1) == 0;
  if tx_edge {
    let prev_block = deblock_up(bc, bo, p);
    let block_edge = bo.y & (block.n4_h - 1) == 0;
    let filter_size =
      deblock_size(block, prev_block, p, pli, false, block_edge);
    if filter_size > 0 {
      let level = deblock_level(deblock, block, prev_block, pli, false);
      if level > 0 {
        let po = bo.plane_offset(&p.cfg);
        let stride = p.cfg.stride;
        let mut plane_slice = p.mut_slice(&po);
        plane_slice.y -= (filter_size >> 1) as isize;
        let slice = plane_slice.as_mut_slice();
        match filter_size {
          4 => {
            deblock_size4(slice, stride, 1, level, bd);
          }
          6 => {
            deblock_size6(slice, stride, 1, level, bd);
          }
          8 => {
            deblock_size8(slice, stride, 1, level, bd);
          }
          14 => {
            deblock_size14(slice, stride, 1, level, bd);
          }
          _ => unreachable!()
        }
      }
    }
  }
}

fn sse_h_edge(
  bc: &BlockContext, bo: &BlockOffset, rec_plane: &Plane, src_plane: &Plane,
  tally: &mut [i64; MAX_LOOP_FILTER + 2], pli: usize, bd: usize
) {
  let block = bc.at(&bo);
  let tx_edge = bo.y & (block.tx_h - 1) == 0;
  if tx_edge {
    let prev_block = deblock_up(bc, bo, rec_plane);
    let block_edge = bo.y & (block.n4_h - 1) == 0;
    let filter_size =
      deblock_size(block, prev_block, rec_plane, pli, true, block_edge);
    if filter_size > 0 {
      let po = bo.plane_offset(&rec_plane.cfg); // rec and src have identical subsampling
      let rec_slice = rec_plane.slice(&po);
      let src_slice = src_plane.slice(&po);
      let rec_tmp = rec_slice.go_up(filter_size >> 1);
      let src_tmp = src_slice.go_up(filter_size >> 1);
      let rec = rec_tmp.as_slice();
      let src = src_tmp.as_slice();
      match filter_size {
        4 => {
          sse_size4(
            rec,
            src,
            tally,
            rec_plane.cfg.stride,
            src_plane.cfg.stride,
            1,
            1,
            bd
          );
        }
        6 => {
          sse_size6(
            rec,
            src,
            tally,
            rec_plane.cfg.stride,
            src_plane.cfg.stride,
            1,
            1,
            bd
          );
        }
        8 => {
          sse_size8(
            rec,
            src,
            tally,
            rec_plane.cfg.stride,
            src_plane.cfg.stride,
            1,
            1,
            bd
          );
        }
        14 => {
          sse_size14(
            rec,
            src,
            tally,
            rec_plane.cfg.stride,
            src_plane.cfg.stride,
            1,
            1,
            bd
          );
        }
        _ => unreachable!()
      }
    }
  }
}

// Deblocks all edges, vertical and horizontal, in a single plane
pub fn deblock_plane(
  deblock: &DeblockState, p: &mut Plane, pli: usize, bc: &mut BlockContext,
  bd: usize
) {
  let xdec = p.cfg.xdec;
  let ydec = p.cfg.ydec;

  match pli {
    0 =>
      if deblock.levels[0] == 0 && deblock.levels[1] == 0 {
        return;
      },
    1 =>
      if deblock.levels[2] == 0 {
        return;
      },
    2 =>
      if deblock.levels[3] == 0 {
        return;
      },
    _ => return
  }

  // vertical edge filtering leads horizonal by one full MI-sized
  // row (and horizontal filtering doesn't happen along the upper
  // edge).  Unroll to avoid corner-cases.
  if bc.rows > 0 {
    for x in (1 << xdec..bc.cols).step_by(1 << xdec) {
      filter_v_edge(deblock, bc, &BlockOffset { x, y: 0 }, p, pli, bd);
    }
    if bc.rows > 1 << ydec {
      for x in (1 << xdec..bc.cols).step_by(1 << xdec) {
        filter_v_edge(
          deblock,
          bc,
          &BlockOffset { x, y: 1 << ydec },
          p,
          pli,
          bd
        );
      }
    }
  }

  // filter rows where vertical and horizontal edge filtering both
  // happen (horizontal edge filtering lags vertical by one row).
  for y in ((2 << ydec)..bc.rows).step_by(1 << ydec) {
    // Check for vertical edge at first MI block boundary on this row
    if 1 << xdec < bc.cols {
      filter_v_edge(deblock, bc, &BlockOffset { x: 1 << xdec, y }, p, pli, bd);
    }
    // run the rest of the row with both vertical and horizontal edge filtering.
    // Horizontal lags vertical edge by one row and two columns.
    for x in (2 << xdec..bc.cols).step_by(1 << xdec) {
      filter_v_edge(deblock, bc, &BlockOffset { x, y }, p, pli, bd);
      filter_h_edge(
        deblock,
        bc,
        &BlockOffset { x: x - (2 << xdec), y: y - (1 << ydec) },
        p,
        pli,
        bd
      );
    }
    // ..and the last two horizontal edges for the row
    if bc.cols - (2 << xdec) > 0 {
      filter_h_edge(
        deblock,
        bc,
        &BlockOffset { x: bc.cols - (2 << xdec), y: y - (1 << ydec) },
        p,
        pli,
        bd
      );
      if bc.cols - (1 << xdec) > 0 {
        filter_h_edge(
          deblock,
          bc,
          &BlockOffset { x: bc.cols - (1 << xdec), y: y - (1 << ydec) },
          p,
          pli,
          bd
        );
      }
    }
  }

  // Last horizontal row, vertical is already complete
  if bc.rows > 1 << ydec {
    for x in (0..bc.cols).step_by(1 << xdec) {
      filter_h_edge(
        deblock,
        bc,
        &BlockOffset { x, y: bc.rows - (1 << ydec) },
        p,
        pli,
        bd
      );
    }
  }
}

// sse count of all edges in a single plane, accumulates into vertical and horizontal counts
fn sse_plane(
  rec: &Plane, src: &Plane, v_sse: &mut [i64; MAX_LOOP_FILTER + 2],
  h_sse: &mut [i64; MAX_LOOP_FILTER + 2], pli: usize, bc: &mut BlockContext,
  bd: usize
) {
  let xdec = rec.cfg.xdec;
  let ydec = rec.cfg.ydec;

  // No horizontal edge filtering along top of frame
  for x in (1 << xdec..bc.cols).step_by(1 << xdec) {
    sse_v_edge(bc, &BlockOffset { x, y: 0 }, rec, src, v_sse, pli, bd);
  }

  // Unlike actual filtering, we're counting horizontal and vertical
  // as separable cases.  No need to lag the horizontal processing
  // behind vertical.
  for y in (1 << ydec..bc.rows).step_by(1 << ydec) {
    // No vertical filtering along left edge of frame
    sse_h_edge(bc, &BlockOffset { x: 0, y }, rec, src, h_sse, pli, bd);
    for x in (1 << xdec..bc.cols).step_by(1 << xdec) {
      sse_v_edge(bc, &BlockOffset { x, y }, rec, src, v_sse, pli, bd);
      sse_h_edge(bc, &BlockOffset { x, y }, rec, src, h_sse, pli, bd);
    }
  }
}

// Deblocks all edges in all planes of a frame
pub fn deblock_filter_frame(
  fs: &mut FrameState, bc: &mut BlockContext, bit_depth: usize
) {
  for pli in 0..PLANES {
    deblock_plane(&fs.deblock, &mut fs.rec.planes[pli], pli, bc, bit_depth);
  }
}

fn sse_optimize(fs: &mut FrameState, bc: &mut BlockContext, bit_depth: usize) {
  assert!(MAX_LOOP_FILTER < 999);
  // i64 allows us to accumulate a total of ~ 35 bits worth of pixels
  assert!(
    fs.input.planes[0].cfg.width.ilog() + fs.input.planes[0].cfg.height.ilog()
      < 35
  );

  for pli in 0..PLANES {
    let mut v_tally: [i64; MAX_LOOP_FILTER + 2] = [0; MAX_LOOP_FILTER + 2];
    let mut h_tally: [i64; MAX_LOOP_FILTER + 2] = [0; MAX_LOOP_FILTER + 2];

    sse_plane(
      &fs.rec.planes[pli],
      &fs.input.planes[pli],
      &mut v_tally,
      &mut h_tally,
      pli,
      bc,
      bit_depth
    );

    for i in 1..=MAX_LOOP_FILTER {
      v_tally[i] += v_tally[i - 1];
      h_tally[i] += h_tally[i - 1];
    }

    match pli {
      0 => {
        let mut best_v = 999;
        let mut best_h = 999;
        for i in 0..=MAX_LOOP_FILTER {
          if best_v == 999 || v_tally[best_v] > v_tally[i] {
            best_v = i;
          };
          if best_h == 999 || h_tally[best_h] > h_tally[i] {
            best_h = i;
          };
        }
        fs.deblock.levels[0] = best_v as u8;
        fs.deblock.levels[1] = best_h as u8;
      }
      1 | 2 => {
        let mut best = 999;
        for i in 0..=MAX_LOOP_FILTER {
          if best == 999
            || v_tally[best] + h_tally[best] > v_tally[i] + h_tally[i]
          {
            best = i;
          };
        }
        fs.deblock.levels[pli + 1] = best as u8;
      }
      _ => unreachable!()
    }
  }
}

pub fn deblock_filter_optimize(
  fi: &FrameInvariants, fs: &mut FrameState, bc: &mut BlockContext,
  bit_depth: usize
) {
  if fi.config.speed > 3 {
    let q = ac_q(fi.base_q_idx, bit_depth) as i32;
    let level = clamp(
      match bit_depth {
        8 =>
          if fi.frame_type == FrameType::KEY {
            q * 17563 - 421574 + (1 << 18 >> 1) >> 18
          } else {
            q * 6017 + 650707 + (1 << 18 >> 1) >> 18
          },
        10 =>
          if fi.frame_type == FrameType::KEY {
            (q * 20723 + 4060632 + (1 << 20 >> 1) >> 20) - 4
          } else {
            q * 20723 + 4060632 + (1 << 20 >> 1) >> 20
          },
        12 =>
          if fi.frame_type == FrameType::KEY {
            (q * 20723 + 16242526 + (1 << 22 >> 1) >> 22) - 4
          } else {
            q * 20723 + 16242526 + (1 << 22 >> 1) >> 22
          },
        _ => {
          assert!(false);
          0
        }
      },
      0,
      MAX_LOOP_FILTER as i32
    ) as u8;

    fs.deblock.levels[0] = level;
    fs.deblock.levels[1] = level;
    fs.deblock.levels[2] = level;
    fs.deblock.levels[3] = level;
  } else {
    sse_optimize(fs, bc, bit_depth);
  }
}
