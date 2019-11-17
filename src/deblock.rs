// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::api::FrameType;
use crate::context::*;
use crate::encoder::FrameInvariants;
use crate::encoder::FrameState;
use crate::frame::*;
use crate::partition::RefType::*;
use crate::predict::PredictionMode::*;
use crate::quantize::*;
use crate::util::Pixel;
use crate::util::{clamp, ILog};
use crate::DeblockState;
use std::cmp;
use std::sync::Arc;

fn deblock_adjusted_level(
  deblock: &DeblockState, block: &Block, pli: usize, vertical: bool,
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
        + ((deblock.ref_deltas[reference.to_index()] as i32) << l5)
        + if reference == INTRA_FRAME {
          0
        } else {
          (deblock.mode_deltas[mode_type] as i32) << l5
        },
      0,
      MAX_LOOP_FILTER as i32,
    ) as usize
  } else {
    level as usize
  }
}

fn deblock_left<'a, T: Pixel>(
  blocks: &'a FrameBlocks, in_bo: PlaneBlockOffset, p: &Plane<T>,
) -> &'a Block {
  let xdec = p.cfg.xdec;
  let ydec = p.cfg.ydec;

  // This little bit of weirdness is straight out of the spec;
  // subsampled chroma uses odd mi row/col
  let bo =
    PlaneBlockOffset(BlockOffset { x: in_bo.0.x | xdec, y: in_bo.0.y | ydec });

  // We already know we're not at the upper/left corner, so prev_block is in frame
  &blocks[bo.with_offset(-1 << xdec, 0)]
}

fn deblock_up<'a, T: Pixel>(
  blocks: &'a FrameBlocks, in_bo: PlaneBlockOffset, p: &Plane<T>,
) -> &'a Block {
  let xdec = p.cfg.xdec;
  let ydec = p.cfg.ydec;

  // This little bit of weirdness is straight out of the spec;
  // subsampled chroma uses odd mi row/col
  let bo =
    PlaneBlockOffset(BlockOffset { x: in_bo.0.x | xdec, y: in_bo.0.y | ydec });

  // We already know we're not at the upper/left corner, so prev_block is in frame
  &blocks[bo.with_offset(0, -1 << ydec)]
}

// Must be called on a tx edge, and not on a frame edge.  This is enforced above the call.
fn deblock_size<T: Pixel>(
  block: &Block, prev_block: &Block, p: &Plane<T>, pli: usize, vertical: bool,
  block_edge: bool,
) -> usize {
  let xdec = p.cfg.xdec;
  let ydec = p.cfg.ydec;

  // filter application is conditional on skip and block edge
  if !(block_edge
    || !block.skip
    || !prev_block.skip
    || block.ref_frames[0] == INTRA_FRAME
    || prev_block.ref_frames[0] == INTRA_FRAME)
  {
    0
  } else {
    let (txsize, prev_txsize) = if pli == 0 {
      (block.txsize, prev_block.txsize)
    } else {
      (
        block.bsize.largest_chroma_tx_size(xdec, ydec),
        prev_block.bsize.largest_chroma_tx_size(xdec, ydec),
      )
    };
    let (tx_n, prev_tx_n) = if vertical {
      (cmp::max(txsize.width_mi(), 1), cmp::max(prev_txsize.width_mi(), 1))
    } else {
      (cmp::max(txsize.height_mi(), 1), cmp::max(prev_txsize.height_mi(), 1))
    };
    cmp::min(
      if pli == 0 { 14 } else { 6 },
      cmp::min(tx_n, prev_tx_n) << MI_SIZE_LOG2,
    )
  }
}

// Must be called on a tx edge
fn deblock_level(
  deblock: &DeblockState, block: &Block, prev_block: &Block, pli: usize,
  vertical: bool,
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
  p1: i32, p0: i32, q0: i32, q1: i32, shift: usize,
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
    q1,
  ]
}

// six taps, 6 outputs (four are trivial)
fn filter_narrow2_6(
  p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32, shift: usize,
) -> [i32; 6] {
  let x = filter_narrow2_4(p1, p0, q0, q1, shift);
  [p2, x[0], x[1], x[2], x[3], q2]
}

// 12 taps, 12 outputs (ten are trivial)
fn filter_narrow2_12(
  p5: i32, p4: i32, p3: i32, p2: i32, p1: i32, p0: i32, q0: i32, q1: i32,
  q2: i32, q3: i32, q4: i32, q5: i32, shift: usize,
) -> [i32; 12] {
  let x = filter_narrow2_4(p1, p0, q0, q1, shift);
  [p5, p4, p3, p2, x[0], x[1], x[2], x[3], q2, q3, q4, q5]
}

// four taps, 4 outputs
fn filter_narrow4_4(
  p1: i32, p0: i32, q0: i32, q1: i32, shift: usize,
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
  let filter3 = (filter1 + 1) >> 1;
  [
    clamp(p1 + filter3, 0, (256 << shift) - 1),
    clamp(p0 + filter2, 0, (256 << shift) - 1),
    clamp(q0 - filter1, 0, (256 << shift) - 1),
    clamp(q1 - filter3, 0, (256 << shift) - 1),
  ]
}

// six taps, 6 outputs (two are trivial)
fn filter_narrow4_6(
  p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32, shift: usize,
) -> [i32; 6] {
  let x = filter_narrow4_4(p1, p0, q0, q1, shift);
  [p2, x[0], x[1], x[2], x[3], q2]
}

// 12 taps, 12 outputs (eight are trivial)
fn filter_narrow4_12(
  p5: i32, p4: i32, p3: i32, p2: i32, p1: i32, p0: i32, q0: i32, q1: i32,
  q2: i32, q3: i32, q4: i32, q5: i32, shift: usize,
) -> [i32; 12] {
  let x = filter_narrow4_4(p1, p0, q0, q1, shift);
  [p5, p4, p3, p2, x[0], x[1], x[2], x[3], q2, q3, q4, q5]
}

// six taps, 4 outputs
#[rustfmt::skip]
const fn filter_wide6_4(
  p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32
) -> [i32; 4] {
  [
    (p2*3 + p1*2 + p0*2 + q0   + (1<<2)) >> 3,
    (p2   + p1*2 + p0*2 + q0*2 + q1   + (1<<2)) >> 3,
           (p1   + p0*2 + q0*2 + q1*2 + q2   + (1<<2)) >> 3,
                  (p0   + q0*2 + q1*2 + q2*3 + (1<<2)) >> 3
  ]
}

// eight taps, 6 outputs
#[rustfmt::skip]
const fn filter_wide8_6(
  p3: i32, p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32, q3: i32
) -> [i32; 6] {
  [
    (p3*3 + p2*2 + p1   + p0   + q0   + (1<<2)) >> 3,
    (p3*2 + p2   + p1*2 + p0   + q0   + q1   + (1<<2)) >> 3,
    (p3   + p2   + p1   + p0*2 + q0   + q1   + q2   +(1<<2)) >> 3,
           (p2   + p1   + p0   + q0*2 + q1   + q2   + q3   + (1<<2)) >> 3,
                  (p1   + p0   + q0   + q1*2 + q2   + q3*2 + (1<<2)) >> 3,
                         (p0   + q0   + q1   + q2*2 + q3*3 + (1<<2)) >> 3
  ]
}

// 12 taps, 12 outputs (six are trivial)
const fn filter_wide8_12(
  p5: i32, p4: i32, p3: i32, p2: i32, p1: i32, p0: i32, q0: i32, q1: i32,
  q2: i32, q3: i32, q4: i32, q5: i32,
) -> [i32; 12] {
  let x = filter_wide8_6(p3, p2, p1, p0, q0, q1, q2, q3);
  [p5, p4, p3, x[0], x[1], x[2], x[3], x[4], x[5], q3, q4, q5]
}

// fourteen taps, 12 outputs
#[rustfmt::skip]
const fn filter_wide14_12(
  p6: i32, p5: i32, p4: i32, p3: i32, p2: i32, p1: i32, p0: i32, q0: i32,
  q1: i32, q2: i32, q3: i32, q4: i32, q5: i32, q6: i32
) -> [i32; 12] {
  [
    (p6*7 + p5*2 + p4*2 + p3   + p2   + p1   + p0   + q0   + (1<<3)) >> 4,
    (p6*5 + p5*2 + p4*2 + p3*2 + p2   + p1   + p0   + q0   + q1   + (1<<3)) >> 4,
    (p6*4 + p5   + p4*2 + p3*2 + p2*2 + p1   + p0   + q0   + q1   + q2   + (1<<3)) >> 4,
    (p6*3 + p5   + p4   + p3*2 + p2*2 + p1*2 + p0   + q0   + q1   + q2   + q3   + (1<<3)) >> 4,
    (p6*2 + p5   + p4   + p3   + p2*2 + p1*2 + p0*2 + q0   + q1   + q2   + q3   + q4   + (1<<3)) >> 4,
    (p6   + p5   + p4   + p3   + p2   + p1*2 + p0*2 + q0*2 + q1   + q2   + q3   + q4   + q5   + (1<<3)) >> 4,
           (p5   + p4   + p3   + p2   + p1   + p0*2 + q0*2 + q1*2 + q2   + q3   + q4   + q5   + q6 + (1<<3)) >> 4,
                  (p4   + p3   + p2   + p1   + p0   + q0*2 + q1*2 + q2*2 + q3   + q4   + q5   + q6*2 + (1<<3)) >> 4,
                         (p3   + p2   + p1   + p0   + q0   + q1*2 + q2*2 + q3*2 + q4   + q5   + q6*3 + (1<<3)) >> 4,
                                (p2   + p1   + p0   + q0   + q1   + q2*2 + q3*2 + q4*2 + q5   + q6*4 + (1<<3)) >> 4,
                                       (p1   + p0   + q0   + q1   + q2   + q3*2 + q4*2 + q5*2 + q6*5 + (1<<3)) >> 4,
                                              (p0   + q0   + q1   + q2   + q3   + q4*2 + q5*2 + q6*7 + (1<<3)) >> 4
  ]
}

#[inline]
fn copy_horizontal<T: Pixel>(
  dst: &mut PlaneMutSlice<'_, T>, x: usize, y: usize, src: &[i32],
) {
  let row = &mut dst[y][x..];
  for (dst, src) in row.iter_mut().take(src.len()).zip(src) {
    *dst = T::cast_from(*src);
  }
}

#[inline]
fn copy_vertical<T: Pixel>(
  dst: &mut PlaneMutSlice<'_, T>, x: usize, y: usize, src: &[i32],
) {
  for (i, v) in src.iter().enumerate() {
    let p = &mut dst[y + i][x];
    *p = T::cast_from(*v);
  }
}

fn stride_sse<T: Pixel>(a: &[T], b: &[i32], pitch: usize) -> i64 {
  let mut acc: i32 = 0;
  for (a, b) in a.iter().step_by(pitch).take(b.len()).zip(b) {
    let v: i32 = (*a).as_();
    acc += (v - *b) * (v - *b)
  }
  acc as i64
}

const fn _level_to_limit(level: i32, shift: usize) -> i32 {
  level << shift
}

const fn limit_to_level(limit: i32, shift: usize) -> i32 {
  (limit + (1 << shift) - 1) >> shift
}

const fn _level_to_blimit(level: i32, shift: usize) -> i32 {
  (3 * level + 4) << shift
}

const fn blimit_to_level(blimit: i32, shift: usize) -> i32 {
  (((blimit + (1 << shift) - 1) >> shift) - 2) / 3
}

const fn _level_to_thresh(level: i32, shift: usize) -> i32 {
  level >> 4 << shift
}

const fn thresh_to_level(thresh: i32, shift: usize) -> i32 {
  (thresh + (1 << shift) - 1) >> shift << 4
}

fn nhev4(p1: i32, p0: i32, q0: i32, q1: i32, shift: usize) -> usize {
  thresh_to_level(cmp::max((p1 - p0).abs(), (q1 - q0).abs()), shift) as usize
}

fn mask4(p1: i32, p0: i32, q0: i32, q1: i32, shift: usize) -> usize {
  cmp::max(
    limit_to_level(cmp::max((p1 - p0).abs(), (q1 - q0).abs()), shift),
    blimit_to_level((p0 - q0).abs() * 2 + (p1 - q1).abs() / 2, shift),
  ) as usize
}

#[inline]
fn deblock_size4_inner(
  [p1, p0, q0, q1]: [i32; 4], level: usize, bd: usize,
) -> Option<[i32; 4]> {
  if mask4(p1, p0, q0, q1, bd - 8) <= level {
    let x = if nhev4(p1, p0, q0, q1, bd - 8) <= level {
      filter_narrow4_4(p1, p0, q0, q1, bd - 8)
    } else {
      filter_narrow2_4(p1, p0, q0, q1, bd - 8)
    };
    Some(x)
  } else {
    None
  }
}

// Assumes rec[0] is set 2 taps back from the edge
fn deblock_v_size4<T: Pixel>(
  rec: &mut PlaneMutSlice<'_, T>, level: usize, bd: usize,
) {
  for y in 0..4 {
    let p = &rec[y];
    let vals = [p[0].as_(), p[1].as_(), p[2].as_(), p[3].as_()];
    if let Some(data) = deblock_size4_inner(vals, level, bd) {
      copy_horizontal(rec, 0, y, &data);
    }
  }
}

// Assumes rec[0] is set 2 taps back from the edge
fn deblock_h_size4<T: Pixel>(
  rec: &mut PlaneMutSlice<'_, T>, level: usize, bd: usize,
) {
  for x in 0..4 {
    let vals =
      [rec[0][x].as_(), rec[1][x].as_(), rec[2][x].as_(), rec[3][x].as_()];
    if let Some(data) = deblock_size4_inner(vals, level, bd) {
      copy_vertical(rec, x, 0, &data);
    }
  }
}

// Assumes rec[0] and src[0] are set 2 taps back from the edge.
// Accesses four taps, accumulates four pixels into the tally
fn sse_size4<T: Pixel>(
  rec: &PlaneSlice<'_, T>, src: &PlaneSlice<'_, T>,
  tally: &mut [i64; MAX_LOOP_FILTER + 2], rec_pitch: usize, src_pitch: usize,
  bd: usize,
) {
  for y in 0..4 {
    let p = &rec[y]; // four taps
    let a = &src[y]; // four pixels to compare
    let p1: i32 = p[0].as_();
    let p0: i32 = p[rec_pitch].as_();
    let q0: i32 = p[rec_pitch * 2].as_();
    let q1: i32 = p[rec_pitch * 3].as_();

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
  }
}

fn mask6(
  p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32, shift: usize,
) -> usize {
  cmp::max(
    limit_to_level(
      cmp::max(
        (p2 - p1).abs(),
        cmp::max((p1 - p0).abs(), cmp::max((q2 - q1).abs(), (q1 - q0).abs())),
      ),
      shift,
    ),
    blimit_to_level((p0 - q0).abs() * 2 + (p1 - q1).abs() / 2, shift),
  ) as usize
}

fn flat6(p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32) -> usize {
  cmp::max(
    (p1 - p0).abs(),
    cmp::max((q1 - q0).abs(), cmp::max((p2 - p0).abs(), (q2 - q0).abs())),
  ) as usize
}

#[inline]
fn deblock_size6_inner(
  [p2, p1, p0, q0, q1, q2]: [i32; 6], level: usize, bd: usize,
) -> Option<[i32; 4]> {
  if mask6(p2, p1, p0, q0, q1, q2, bd - 8) <= level {
    let flat = 1 << (bd - 8);
    let x = if flat6(p2, p1, p0, q0, q1, q2) <= flat {
      filter_wide6_4(p2, p1, p0, q0, q1, q2)
    } else if nhev4(p1, p0, q0, q1, bd - 8) <= level {
      filter_narrow4_4(p1, p0, q0, q1, bd - 8)
    } else {
      filter_narrow2_4(p1, p0, q0, q1, bd - 8)
    };
    Some(x)
  } else {
    None
  }
}

// Assumes slice[0] is set 3 taps back from the edge
fn deblock_v_size6<T: Pixel>(
  rec: &mut PlaneMutSlice<'_, T>, level: usize, bd: usize,
) {
  for y in 0..4 {
    let p = &rec[y];
    let vals =
      [p[0].as_(), p[1].as_(), p[2].as_(), p[3].as_(), p[4].as_(), p[5].as_()];
    if let Some(data) = deblock_size6_inner(vals, level, bd) {
      copy_horizontal(rec, 1, y, &data);
    }
  }
}

// Assumes slice[0] is set 3 taps back from the edge
fn deblock_h_size6<T: Pixel>(
  rec: &mut PlaneMutSlice<'_, T>, level: usize, bd: usize,
) {
  for x in 0..4 {
    let vals = [
      rec[0][x].as_(),
      rec[1][x].as_(),
      rec[2][x].as_(),
      rec[3][x].as_(),
      rec[4][x].as_(),
      rec[5][x].as_(),
    ];
    if let Some(data) = deblock_size6_inner(vals, level, bd) {
      copy_vertical(rec, x, 1, &data);
    }
  }
}

// Assumes rec[0] and src[0] are set 3 taps back from the edge.
// Accesses six taps, accumulates four pixels into the tally
fn sse_size6<T: Pixel>(
  rec: &PlaneSlice<'_, T>, src: &PlaneSlice<'_, T>,
  tally: &mut [i64; MAX_LOOP_FILTER + 2], rec_pitch: usize, src_pitch: usize,
  bd: usize,
) {
  let flat = 1 << (bd - 8);
  for y in 0..4 {
    let p = &rec[y]; // six taps
    let a = &src[y][src_pitch..]; // four pixels to compare so offset one forward
    let p2: i32 = p[0].as_();
    let p1: i32 = p[rec_pitch].as_();
    let p0: i32 = p[rec_pitch * 2].as_();
    let q0: i32 = p[rec_pitch * 3].as_();
    let q1: i32 = p[rec_pitch * 4].as_();
    let q2: i32 = p[rec_pitch * 5].as_();

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
  }
}

fn mask8(
  p3: i32, p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32, q3: i32,
  shift: usize,
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
              cmp::max((q2 - q1).abs(), (q1 - q0).abs()),
            ),
          ),
        ),
      ),
      shift,
    ),
    blimit_to_level((p0 - q0).abs() * 2 + (p1 - q1).abs() / 2, shift),
  ) as usize
}

fn flat8(
  p3: i32, p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32, q3: i32,
) -> usize {
  cmp::max(
    (p1 - p0).abs(),
    cmp::max(
      (q1 - q0).abs(),
      cmp::max(
        (p2 - p0).abs(),
        cmp::max((q2 - q0).abs(), cmp::max((p3 - p0).abs(), (q3 - q0).abs())),
      ),
    ),
  ) as usize
}

#[inline]
fn deblock_size8_inner(
  [p3, p2, p1, p0, q0, q1, q2, q3]: [i32; 8], level: usize, bd: usize,
) -> Option<[i32; 6]> {
  if mask8(p3, p2, p1, p0, q0, q1, q2, q3, bd - 8) <= level {
    let flat = 1 << (bd - 8);
    let x = if flat8(p3, p2, p1, p0, q0, q1, q2, q3) <= flat {
      filter_wide8_6(p3, p2, p1, p0, q0, q1, q2, q3)
    } else if nhev4(p1, p0, q0, q1, bd - 8) <= level {
      filter_narrow4_6(p2, p1, p0, q0, q1, q2, bd - 8)
    } else {
      filter_narrow2_6(p2, p1, p0, q0, q1, q2, bd - 8)
    };
    Some(x)
  } else {
    None
  }
}

// Assumes rec[0] is set 4 taps back from the edge
fn deblock_v_size8<T: Pixel>(
  rec: &mut PlaneMutSlice<'_, T>, level: usize, bd: usize,
) {
  for y in 0..4 {
    let p = &rec[y];
    let vals = [
      p[0].as_(),
      p[1].as_(),
      p[2].as_(),
      p[3].as_(),
      p[4].as_(),
      p[5].as_(),
      p[6].as_(),
      p[7].as_(),
    ];
    if let Some(data) = deblock_size8_inner(vals, level, bd) {
      copy_horizontal(rec, 1, y, &data);
    }
  }
}

// Assumes rec[0] is set 4 taps back from the edge
fn deblock_h_size8<T: Pixel>(
  rec: &mut PlaneMutSlice<'_, T>, level: usize, bd: usize,
) {
  for x in 0..4 {
    let vals = [
      rec[0][x].as_(),
      rec[1][x].as_(),
      rec[2][x].as_(),
      rec[3][x].as_(),
      rec[4][x].as_(),
      rec[5][x].as_(),
      rec[6][x].as_(),
      rec[7][x].as_(),
    ];
    if let Some(data) = deblock_size8_inner(vals, level, bd) {
      copy_vertical(rec, x, 1, &data);
    }
  }
}

// Assumes rec[0] and src[0] are set 4 taps back from the edge.
// Accesses eight taps, accumulates six pixels into the tally
fn sse_size8<T: Pixel>(
  rec: &PlaneSlice<'_, T>, src: &PlaneSlice<'_, T>,
  tally: &mut [i64; MAX_LOOP_FILTER + 2], rec_pitch: usize, src_pitch: usize,
  bd: usize,
) {
  let flat = 1 << (bd - 8);
  for y in 0..4 {
    let p = &rec[y]; // eight taps
    let a = &src[y][src_pitch..]; // six pixels to compare so offset one forward
    let p3: i32 = p[0].as_();
    let p2: i32 = p[rec_pitch].as_();
    let p1: i32 = p[rec_pitch * 2].as_();
    let p0: i32 = p[rec_pitch * 3].as_();
    let q0: i32 = p[rec_pitch * 4].as_();
    let q1: i32 = p[rec_pitch * 5].as_();
    let q2: i32 = p[rec_pitch * 6].as_();
    let q3: i32 = p[rec_pitch * 7].as_();

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
      MAX_LOOP_FILTER + 1,
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
  }
}

fn flat14_outer(
  p6: i32, p5: i32, p4: i32, p0: i32, q0: i32, q4: i32, q5: i32, q6: i32,
) -> usize {
  cmp::max(
    (p4 - p0).abs(),
    cmp::max(
      (q4 - q0).abs(),
      cmp::max(
        (p5 - p0).abs(),
        cmp::max((q5 - q0).abs(), cmp::max((p6 - p0).abs(), (q6 - q0).abs())),
      ),
    ),
  ) as usize
}

#[inline]
fn deblock_size14_inner(
  [p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6]: [i32; 14],
  level: usize, bd: usize,
) -> Option<[i32; 12]> {
  // 'mask' test
  if mask8(p3, p2, p1, p0, q0, q1, q2, q3, bd - 8) <= level {
    let flat = 1 << (bd - 8);
    // inner flatness test
    let x = if flat8(p3, p2, p1, p0, q0, q1, q2, q3) <= flat {
      // outer flatness test
      if flat14_outer(p6, p5, p4, p0, q0, q4, q5, q6) <= flat {
        // sufficient flatness across 14 pixel width; run full-width filter
        filter_wide14_12(
          p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6,
        )
      } else {
        // only flat in inner area, run 8-tap
        filter_wide8_12(p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5)
      }
    } else if nhev4(p1, p0, q0, q1, bd - 8) <= level {
      // not flat, run narrow filter
      filter_narrow4_12(p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, bd - 8)
    } else {
      filter_narrow2_12(p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, bd - 8)
    };
    Some(x)
  } else {
    None
  }
}

// Assumes rec[0] is set 7 taps back from the edge
fn deblock_v_size14<T: Pixel>(
  rec: &mut PlaneMutSlice<'_, T>, level: usize, bd: usize,
) {
  for y in 0..4 {
    let p = &rec[y];
    let vals = [
      p[0].as_(),
      p[1].as_(),
      p[2].as_(),
      p[3].as_(),
      p[4].as_(),
      p[5].as_(),
      p[6].as_(),
      p[7].as_(),
      p[8].as_(),
      p[9].as_(),
      p[10].as_(),
      p[11].as_(),
      p[12].as_(),
      p[13].as_(),
    ];
    if let Some(data) = deblock_size14_inner(vals, level, bd) {
      copy_horizontal(rec, 1, y, &data);
    }
  }
}

// Assumes rec[0] is set 7 taps back from the edge
fn deblock_h_size14<T: Pixel>(
  rec: &mut PlaneMutSlice<'_, T>, level: usize, bd: usize,
) {
  for x in 0..4 {
    let vals = [
      rec[0][x].as_(),
      rec[1][x].as_(),
      rec[2][x].as_(),
      rec[3][x].as_(),
      rec[4][x].as_(),
      rec[5][x].as_(),
      rec[6][x].as_(),
      rec[7][x].as_(),
      rec[8][x].as_(),
      rec[9][x].as_(),
      rec[10][x].as_(),
      rec[11][x].as_(),
      rec[12][x].as_(),
      rec[13][x].as_(),
    ];
    if let Some(data) = deblock_size14_inner(vals, level, bd) {
      copy_vertical(rec, x, 1, &data);
    }
  }
}

// Assumes rec[0] and src[0] are set 7 taps back from the edge.
// Accesses fourteen taps, accumulates twelve pixels into the tally
fn sse_size14<T: Pixel>(
  rec: &PlaneSlice<'_, T>, src: &PlaneSlice<'_, T>,
  tally: &mut [i64; MAX_LOOP_FILTER + 2], rec_pitch: usize, src_pitch: usize,
  bd: usize,
) {
  let flat = 1 << (bd - 8);
  for y in 0..4 {
    let p = &rec[y]; // 14 taps
    let a = &src[y][src_pitch..]; // 12 pixels to compare so offset one forward
    let p6: i32 = p[0].as_();
    let p5: i32 = p[rec_pitch].as_();
    let p4: i32 = p[rec_pitch * 2].as_();
    let p3: i32 = p[rec_pitch * 3].as_();
    let p2: i32 = p[rec_pitch * 4].as_();
    let p1: i32 = p[rec_pitch * 5].as_();
    let p0: i32 = p[rec_pitch * 6].as_();
    let q0: i32 = p[rec_pitch * 7].as_();
    let q1: i32 = p[rec_pitch * 8].as_();
    let q2: i32 = p[rec_pitch * 9].as_();
    let q3: i32 = p[rec_pitch * 10].as_();
    let q4: i32 = p[rec_pitch * 11].as_();
    let q5: i32 = p[rec_pitch * 12].as_();
    let q6: i32 = p[rec_pitch * 13].as_();

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
      bd - 8,
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
      bd - 8,
    );

    // mask8 sets the dividing line for filter vs no filter
    // flat8 decides between wide and narrow filters (unrelated to level)
    // flat14 decides between wide14 and wide8 filters
    // nhev4 sets the dividing line between narrow2 and narrow4
    let mask = clamp(
      mask8(p3, p2, p1, p0, q0, q1, q2, q3, bd - 8),
      1,
      MAX_LOOP_FILTER + 1,
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
  }
}

fn filter_v_edge<T: Pixel>(
  deblock: &DeblockState, blocks: &FrameBlocks, bo: PlaneBlockOffset,
  p: &mut Plane<T>, pli: usize, bd: usize, xdec: usize, ydec: usize,
) {
  let block = &blocks[bo];
  let txsize = if pli == 0 {
    block.txsize
  } else {
    block.bsize.largest_chroma_tx_size(xdec, ydec)
  };
  let tx_edge = bo.0.x >> xdec & (txsize.width_mi() - 1) == 0;
  if tx_edge {
    let prev_block = deblock_left(blocks, bo, p);
    let block_edge = bo.0.x & (block.n4_w - 1) == 0;
    let filter_size =
      deblock_size(block, prev_block, p, pli, true, block_edge);
    if filter_size > 0 {
      let level = deblock_level(deblock, block, prev_block, pli, true);
      if level > 0 {
        let po = bo.plane_offset(&p.cfg);
        let mut plane_slice = p.mut_slice(po);
        plane_slice.x -= (filter_size >> 1) as isize;
        match filter_size {
          4 => {
            deblock_v_size4(&mut plane_slice, level, bd);
          }
          6 => {
            deblock_v_size6(&mut plane_slice, level, bd);
          }
          8 => {
            deblock_v_size8(&mut plane_slice, level, bd);
          }
          14 => {
            deblock_v_size14(&mut plane_slice, level, bd);
          }
          _ => unreachable!(),
        }
      }
    }
  }
}

fn sse_v_edge<T: Pixel>(
  blocks: &FrameBlocks, bo: PlaneBlockOffset, rec_plane: &Plane<T>,
  src_plane: &Plane<T>, tally: &mut [i64; MAX_LOOP_FILTER + 2], pli: usize,
  bd: usize, xdec: usize, ydec: usize,
) {
  let block = &blocks[bo];
  let txsize = if pli == 0 {
    block.txsize
  } else {
    block.bsize.largest_chroma_tx_size(xdec, ydec)
  };
  let tx_edge = bo.0.x >> xdec & (txsize.width_mi() - 1) == 0;
  if tx_edge {
    let prev_block = deblock_left(blocks, bo, rec_plane);
    let block_edge = bo.0.x & (block.n4_w - 1) == 0;
    let filter_size =
      deblock_size(block, prev_block, rec_plane, pli, true, block_edge);
    if filter_size > 0 {
      let po = {
        let mut po = bo.plane_offset(&rec_plane.cfg); // rec and src have identical subsampling
        po.x -= (filter_size >> 1) as isize;
        po
      };
      let rec_slice = rec_plane.slice(po);
      let src_slice = src_plane.slice(po);
      match filter_size {
        4 => {
          sse_size4(&rec_slice, &src_slice, tally, 1, 1, bd);
        }
        6 => {
          sse_size6(&rec_slice, &src_slice, tally, 1, 1, bd);
        }
        8 => {
          sse_size8(&rec_slice, &src_slice, tally, 1, 1, bd);
        }
        14 => {
          sse_size14(&rec_slice, &src_slice, tally, 1, 1, bd);
        }
        _ => unreachable!(),
      }
    }
  }
}

fn filter_h_edge<T: Pixel>(
  deblock: &DeblockState, blocks: &FrameBlocks, bo: PlaneBlockOffset,
  p: &mut Plane<T>, pli: usize, bd: usize, xdec: usize, ydec: usize,
) {
  let block = &blocks[bo];
  let txsize = if pli == 0 {
    block.txsize
  } else {
    block.bsize.largest_chroma_tx_size(xdec, ydec)
  };
  let tx_edge = bo.0.y >> ydec & (txsize.height_mi() - 1) == 0;
  if tx_edge {
    let prev_block = deblock_up(blocks, bo, p);
    let block_edge = bo.0.y & (block.n4_h - 1) == 0;
    let filter_size =
      deblock_size(block, prev_block, p, pli, false, block_edge);
    if filter_size > 0 {
      let level = deblock_level(deblock, block, prev_block, pli, false);
      if level > 0 {
        let po = bo.plane_offset(&p.cfg);
        let mut plane_slice = p.mut_slice(po);
        plane_slice.y -= (filter_size >> 1) as isize;
        match filter_size {
          4 => {
            deblock_h_size4(&mut plane_slice, level, bd);
          }
          6 => {
            deblock_h_size6(&mut plane_slice, level, bd);
          }
          8 => {
            deblock_h_size8(&mut plane_slice, level, bd);
          }
          14 => {
            deblock_h_size14(&mut plane_slice, level, bd);
          }
          _ => unreachable!(),
        }
      }
    }
  }
}

fn sse_h_edge<T: Pixel>(
  blocks: &FrameBlocks, bo: PlaneBlockOffset, rec_plane: &Plane<T>,
  src_plane: &Plane<T>, tally: &mut [i64; MAX_LOOP_FILTER + 2], pli: usize,
  bd: usize, xdec: usize, ydec: usize,
) {
  let block = &blocks[bo];
  let txsize = if pli == 0 {
    block.txsize
  } else {
    block.bsize.largest_chroma_tx_size(xdec, ydec)
  };
  let tx_edge = bo.0.y >> ydec & (txsize.height_mi() - 1) == 0;
  if tx_edge {
    let prev_block = deblock_up(blocks, bo, rec_plane);
    let block_edge = bo.0.y & (block.n4_h - 1) == 0;
    let filter_size =
      deblock_size(block, prev_block, rec_plane, pli, true, block_edge);
    if filter_size > 0 {
      let po = {
        let mut po = bo.plane_offset(&rec_plane.cfg); // rec and src have identical subsampling
        po.x -= (filter_size >> 1) as isize;
        po
      };
      let rec_slice = rec_plane.slice(po);
      let src_slice = src_plane.slice(po);
      match filter_size {
        4 => {
          sse_size4(&rec_slice, &src_slice, tally, 1, 1, bd);
        }
        6 => {
          sse_size6(&rec_slice, &src_slice, tally, 1, 1, bd);
        }
        8 => {
          sse_size8(&rec_slice, &src_slice, tally, 1, 1, bd);
        }
        14 => {
          sse_size14(&rec_slice, &src_slice, tally, 1, 1, bd);
        }
        _ => unreachable!(),
      }
    }
  }
}

// Deblocks all edges, vertical and horizontal, in a single plane
pub fn deblock_plane<T: Pixel>(
  fi: &FrameInvariants<T>, deblock: &DeblockState, p: &mut Plane<T>,
  pli: usize, blocks: &FrameBlocks,
) {
  let xdec = p.cfg.xdec;
  let ydec = p.cfg.ydec;
  let bd = fi.sequence.bit_depth;

  match pli {
    0 => {
      if deblock.levels[0] == 0 && deblock.levels[1] == 0 {
        return;
      }
    }
    1 => {
      if deblock.levels[2] == 0 {
        return;
      }
    }
    2 => {
      if deblock.levels[3] == 0 {
        return;
      }
    }
    _ => return,
  }

  // Deblocking happens in 4x4 (luma) units; luma x,y are clipped to
  // the *crop frame* by 4x4 block.  Rounding is to handle chroma
  // fenceposts here instead of throughout the code.
  let cols = (((fi.width + MI_SIZE - 1) >> MI_SIZE_LOG2) + (1 << xdec >> 1))
    >> xdec
    << xdec; // Clippy can go suck an egg
  let rows = (((fi.height + MI_SIZE - 1) >> MI_SIZE_LOG2) + (1 << ydec >> 1))
    >> ydec
    << ydec; // Clippy can go suck an egg

  // vertical edge filtering leads horizonal by one full MI-sized
  // row (and horizontal filtering doesn't happen along the upper
  // edge).  Unroll to avoid corner-cases.
  if rows > 0 {
    for x in (1 << xdec..cols).step_by(1 << xdec) {
      filter_v_edge(
        deblock,
        blocks,
        PlaneBlockOffset(BlockOffset { x, y: 0 }),
        p,
        pli,
        bd,
        xdec,
        ydec,
      );
    }
    if rows > 1 << ydec {
      for x in (1 << xdec..cols).step_by(1 << xdec) {
        filter_v_edge(
          deblock,
          blocks,
          PlaneBlockOffset(BlockOffset { x, y: 1 << ydec }),
          p,
          pli,
          bd,
          xdec,
          ydec,
        );
      }
    }
  }

  // filter rows where vertical and horizontal edge filtering both
  // happen (horizontal edge filtering lags vertical by one row).
  for y in ((2 << ydec)..rows).step_by(1 << ydec) {
    // Check for vertical edge at first MI block boundary on this row
    if cols > 1 << xdec {
      filter_v_edge(
        deblock,
        blocks,
        PlaneBlockOffset(BlockOffset { x: 1 << xdec, y }),
        p,
        pli,
        bd,
        xdec,
        ydec,
      );
    }
    // run the rest of the row with both vertical and horizontal edge filtering.
    // Horizontal lags vertical edge by one row and two columns.
    for x in (2 << xdec..cols).step_by(1 << xdec) {
      filter_v_edge(
        deblock,
        blocks,
        PlaneBlockOffset(BlockOffset { x, y }),
        p,
        pli,
        bd,
        xdec,
        ydec,
      );
      filter_h_edge(
        deblock,
        blocks,
        PlaneBlockOffset(BlockOffset {
          x: x - (2 << xdec),
          y: y - (1 << ydec),
        }),
        p,
        pli,
        bd,
        xdec,
        ydec,
      );
    }
    // ..and the last two horizontal edges for the row
    if cols > 2 << xdec {
      filter_h_edge(
        deblock,
        blocks,
        PlaneBlockOffset(BlockOffset {
          x: cols - (2 << xdec),
          y: y - (1 << ydec),
        }),
        p,
        pli,
        bd,
        xdec,
        ydec,
      );
      if cols > 1 << xdec {
        filter_h_edge(
          deblock,
          blocks,
          PlaneBlockOffset(BlockOffset {
            x: cols - (1 << xdec),
            y: y - (1 << ydec),
          }),
          p,
          pli,
          bd,
          xdec,
          ydec,
        );
      }
    }
  }

  // Last horizontal row, vertical is already complete
  if rows > 1 << ydec {
    for x in (0..cols).step_by(1 << xdec) {
      filter_h_edge(
        deblock,
        blocks,
        PlaneBlockOffset(BlockOffset { x, y: rows - (1 << ydec) }),
        p,
        pli,
        bd,
        xdec,
        ydec,
      );
    }
  }
}

// sse count of all edges in a single plane, accumulates into vertical and horizontal counts
fn sse_plane<T: Pixel>(
  fi: &FrameInvariants<T>, rec: &Plane<T>, src: &Plane<T>,
  v_sse: &mut [i64; MAX_LOOP_FILTER + 2],
  h_sse: &mut [i64; MAX_LOOP_FILTER + 2], pli: usize, blocks: &FrameBlocks,
) {
  let xdec = rec.cfg.xdec;
  let ydec = rec.cfg.ydec;

  // Deblocking happens in 4x4 (luma) units; luma x,y are clipped to
  // the *crop frame* by 4x4 block.
  let cols = cmp::min(blocks.cols, (fi.width + MI_SIZE - 1) >> MI_SIZE_LOG2);
  let rows = cmp::min(blocks.rows, (fi.height + MI_SIZE - 1) >> MI_SIZE_LOG2);

  let bd = fi.sequence.bit_depth;
  // No horizontal edge filtering along top of frame
  for x in (1 << xdec..cols).step_by(1 << xdec) {
    sse_v_edge(
      blocks,
      PlaneBlockOffset(BlockOffset { x, y: 0 }),
      rec,
      src,
      v_sse,
      pli,
      bd,
      xdec,
      ydec,
    );
  }

  // Unlike actual filtering, we're counting horizontal and vertical
  // as separable cases.  No need to lag the horizontal processing
  // behind vertical.
  for y in (1 << ydec..rows).step_by(1 << ydec) {
    // No vertical filtering along left edge of frame
    sse_h_edge(
      blocks,
      PlaneBlockOffset(BlockOffset { x: 0, y }),
      rec,
      src,
      h_sse,
      pli,
      bd,
      xdec,
      ydec,
    );
    for x in (1 << xdec..cols).step_by(1 << xdec) {
      sse_v_edge(
        blocks,
        PlaneBlockOffset(BlockOffset { x, y }),
        rec,
        src,
        v_sse,
        pli,
        bd,
        xdec,
        ydec,
      );
      sse_h_edge(
        blocks,
        PlaneBlockOffset(BlockOffset { x, y }),
        rec,
        src,
        h_sse,
        pli,
        bd,
        xdec,
        ydec,
      );
    }
  }
}

// Deblocks all edges in all planes of a frame
pub fn deblock_filter_frame<T: Pixel>(
  fi: &FrameInvariants<T>, fs: &mut FrameState<T>, blocks: &FrameBlocks,
) {
  let fs_rec = Arc::make_mut(&mut fs.rec);
  for pli in 0..PLANES {
    deblock_plane(fi, &fs.deblock, &mut fs_rec.planes[pli], pli, blocks);
  }
}

fn sse_optimize<T: Pixel>(
  fi: &FrameInvariants<T>, fs: &mut FrameState<T>, blocks: &FrameBlocks,
) {
  // i64 allows us to accumulate a total of ~ 35 bits worth of pixels
  assert!(
    fs.input.planes[0].cfg.width.ilog() + fs.input.planes[0].cfg.height.ilog()
      < 35
  );

  for pli in 0..PLANES {
    let mut v_tally: [i64; MAX_LOOP_FILTER + 2] = [0; MAX_LOOP_FILTER + 2];
    let mut h_tally: [i64; MAX_LOOP_FILTER + 2] = [0; MAX_LOOP_FILTER + 2];

    sse_plane(
      fi,
      &fs.rec.planes[pli],
      &fs.input.planes[pli],
      &mut v_tally,
      &mut h_tally,
      pli,
      blocks,
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
      _ => unreachable!(),
    }
  }
}

pub fn deblock_filter_optimize<T: Pixel>(
  fi: &FrameInvariants<T>, fs: &mut FrameState<T>, blocks: &FrameBlocks,
) {
  if fi.config.speed_settings.fast_deblock {
    let q = ac_q(fi.base_q_idx, 0, fi.sequence.bit_depth) as i32;
    let level = clamp(
      match fi.sequence.bit_depth {
        8 => {
          if fi.frame_type == FrameType::KEY {
            (q * 17563 - 421_574 + (1 << 18 >> 1)) >> 18
          } else {
            (q * 6017 + 650_707 + (1 << 18 >> 1)) >> 18
          }
        }
        10 => {
          if fi.frame_type == FrameType::KEY {
            ((q * 20723 + 4_060_632 + (1 << 20 >> 1)) >> 20) - 4
          } else {
            (q * 20723 + 4_060_632 + (1 << 20 >> 1)) >> 20
          }
        }
        12 => {
          if fi.frame_type == FrameType::KEY {
            ((q * 20723 + 16_242_526 + (1 << 22 >> 1)) >> 22) - 4
          } else {
            (q * 20723 + 16_242_526 + (1 << 22 >> 1)) >> 22
          }
        }
        _ => unreachable!(),
      },
      0,
      MAX_LOOP_FILTER as i32,
    ) as u8;

    fs.deblock.levels[0] = level;
    fs.deblock.levels[1] = level;
    fs.deblock.levels[2] = level;
    fs.deblock.levels[3] = level;
  } else {
    sse_optimize(fi, fs, blocks);
  }
}
