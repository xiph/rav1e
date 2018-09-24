// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(safe_extern_statics)]

use std::cmp;
use context::*;
use plane::*;
use quantize::*;
use partition::*;
use partition::PredictionMode::*;
use util::clamp;
use FrameInvariants;
use FrameType;
use FrameState;
use DeblockState;

fn deblock_adjusted_level(deblock: &DeblockState, block: &Block, pli: usize, vertical: bool) -> i32 {
    let idx = if pli == 0 { if vertical { 0 } else { 1 } } else { pli+1 };

    let level = if deblock.block_deltas_enabled {
        // By-block filter strength delta, if the feature is active.
        let block_delta = if deblock.block_delta_multi {
            block.deblock_deltas[ idx ] << deblock.block_delta_shift
        } else {
            block.deblock_deltas[ 0 ] << deblock.block_delta_shift
        };

        // Add to frame-specified filter strength (Y-vertical, Y-horizontal, U, V)
        clamp(block_delta + deblock.levels[idx] as i8, 0, MAX_LOOP_FILTER as i8) as u8
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
        let mode_type = if mode >= NEARESTMV && mode != GLOBALMV && mode!= GLOBAL_GLOBALMV {1} else {0};
        let l5 = level >> 5;
        clamp (level as i32 + ((deblock.ref_deltas[reference] as i32) << l5) +
               if reference == INTRA_FRAME {
                   0
               } else {
                   (deblock.mode_deltas[mode_type] as i32) << l5
               }, 0, MAX_LOOP_FILTER as i32)
    } else {
        level as i32
    }
}

fn deblock_left<'a>(bc: &'a BlockContext, in_bo: &BlockOffset, p: &mut Plane) -> &'a Block {
    let xdec = p.cfg.xdec;
    let ydec = p.cfg.ydec;

    // This little bit of weirdness is straight out of the spec;
    // subsampled chroma uses odd mi row/col
    let bo = BlockOffset{x: in_bo.x | xdec, y: in_bo.y | ydec};

    // We already know we're not at the upper/left corner, so prev_block is in frame
    bc.at(&bo.with_offset(-1 << xdec, 0))
}

fn deblock_up<'a>(bc: &'a BlockContext, in_bo: &BlockOffset, p: &mut Plane) -> &'a Block {
    let xdec = p.cfg.xdec;
    let ydec = p.cfg.ydec;

    // This little bit of weirdness is straight out of the spec;
    // subsampled chroma uses odd mi row/col
    let bo = BlockOffset{x: in_bo.x | xdec, y: in_bo.y | ydec};
    
    // We already know we're not at the upper/left corner, so prev_block is in frame
    bc.at(&bo.with_offset(0, -1 << ydec))
}

// Must be called on a tx edge, and not on a frame edge.  This is enforced above the call.
fn deblock_size(block: &Block, prev_block: &Block, p: &mut Plane, pli: usize,
                vertical: bool, block_edge: bool) -> i32 {
    let xdec = p.cfg.xdec;
    let ydec = p.cfg.ydec;

    // filter application is conditional on skip and block edge
    if !(block_edge || !block.skip || !prev_block.skip ||
         block.ref_frames[0] <= INTRA_FRAME || prev_block.ref_frames[0] <= INTRA_FRAME) {
        0
    } else {
        let (tx_size, prev_tx_size) = if vertical {
            (cmp::max(block.tx_w>>xdec, 1), cmp::max(prev_block.tx_w>>xdec, 1))
        } else {
            (cmp::max(block.tx_h>>ydec, 1), cmp::max(prev_block.tx_h>>ydec, 1))
        };

        cmp::min( if pli==0 {14} else {6}, cmp::min(tx_size, prev_tx_size) << MI_SIZE_LOG2) as i32
    }
}

// Must be called on a tx edge
fn deblock_level(deblock: &DeblockState, block: &Block, prev_block: &Block,
                 pli: usize, vertical: bool) -> i32 {

    let level = deblock_adjusted_level(deblock, block, pli, vertical);
    if level == 0 {
        deblock_adjusted_level(deblock, prev_block, pli, vertical)
    } else {
        level
    }
}

fn hev4(thresh: i32, p1: i32, p0: i32, q0: i32, q1: i32) -> bool {
    (p1 - p0).abs() > thresh || (q1 - q0).abs() > thresh
}

// four taps, 4 outputs (two are trivial)
fn filter_narrow2(p1: i32, p0: i32, q0: i32, q1: i32, shift: usize) -> [i32; 4] {
    let outer_filter = clamp(p1 - q1, -128 << shift, (128 << shift) - 1);
    let base_filter = clamp(outer_filter + 3 * (q0 - p0), -128 << shift, (128 << shift) - 1);
    let filter1 = clamp(base_filter + 4, -128 << shift, (128 << shift)-1) >> 3;
    let filter2 = clamp(base_filter + 3, -128 << shift, (128 << shift)-1) >> 3;
    [p1,
     clamp(p0 + filter2, 0, (256 << shift)-1),
     clamp(q0 - filter1, 0, (256 << shift)-1),
     q1]
}

// four taps, 4 outputs
fn filter_narrow4(p1: i32, p0: i32, q0: i32, q1: i32, shift: usize) -> [i32; 4] {
    let base_filter = clamp(3 * (q0 - p0), -128 << shift, (128 << shift) - 1);
    let filter1 = clamp(base_filter + 4, -128 << shift, (128 << shift)-1) >> 3;
    let filter2 = clamp(base_filter + 3, -128 << shift, (128 << shift)-1) >> 3;
    let filter3 = filter1 + 1 >> 1;
    [clamp(p1 + filter3, 0, (256 << shift)-1),
     clamp(p0 + filter2, 0, (256 << shift)-1),
     clamp(q0 - filter1, 0, (256 << shift)-1),
     clamp(q1 - filter3, 0, (256 << shift)-1)]
}

// six taps, 4 outputs
#[cfg_attr(rustfmt, rustfmt_skip)]
fn filter_wide6(p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32) -> [i32; 4] {
    [p2*3 + p1*2 + p0*2 + q0   + (1<<2) >> 3,
     p2   + p1*2 + p0*2 + q0*2 + q1   + (1<<2) >> 3,
            p1   + p0*2 + q0*2 + q1*2 + q2   + (1<<2) >> 3,
                   p0   + q0*2 + q1*2 + q2*3 + (1<<2) >> 3]
}

// eight taps, 6 outputs
#[cfg_attr(rustfmt, rustfmt_skip)]
fn filter_wide8(p3: i32, p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32, q3: i32) -> [i32; 6] {
    [p3*3 + p2*2 + p1   + p0   + q0   + (1<<2) >> 3,
     p3*2 + p2   + p1*2 + p0   + q0   + q1   + (1<<2) >> 3,
     p3   + p2   + p1   + p0*2 + q0   + q1   + q2   +(1<<2) >> 3,
            p2   + p1   + p0   + q0*2 + q1   + q2   + q3   + (1<<2) >> 3,
                   p1   + p0   + q0   + q1*2 + q2   + q3*2 + (1<<2) >> 3,
                          p0   + q0   + q1   + q2*2 + q3*3 + (1<<2) >> 3]
}

// fourteen taps, 12 outputs
#[cfg_attr(rustfmt, rustfmt_skip)]
fn filter_wide14(p6: i32, p5: i32, p4: i32, p3: i32, p2: i32, p1: i32, p0: i32,
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

fn mask4(blimit: i32, limit: i32, p1: i32, p0: i32, q0: i32, q1: i32) -> bool {
    (p1 - p0).abs() <= limit &&
        (q1 - q0).abs() <= limit &&
        (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 <= blimit
}

fn stride_copy(dst: &mut [u16], src: &[i32], pitch: usize) {
    for (dst, src) in dst.iter_mut().step_by(pitch).take(src.len()).zip(src) { *dst = *src as u16 };
}

// Assumes rec[0] is set 2 taps back from the edge
fn deblock_size4(rec: &mut[u16], pitch: usize, stride: usize, level: i32, bd: usize) {
    let mut s = 0;
    let blimit = 3 * level + 4 << bd - 8;
    let limit = level << bd - 8;
    let thresh = level >> 4 << bd - 8;
    for _i in 0..4 {
        let p = &mut rec[s..];
        let p1 = p[0] as i32;
        let p0 = p[pitch] as i32;
        let q0 = p[pitch*2] as i32;
        let q1 = p[pitch*3] as i32;
        if mask4(blimit, limit, p1, p0, q0, q1) {
            let x;
            if hev4(thresh, p1, p0, q0, q1) {
                x = filter_narrow2(p1, p0, q0, q1, bd - 8);
            } else {
                x = filter_narrow4(p1, p0, q0, q1, bd - 8);
            }
            stride_copy(p, &x, pitch);
        }
        s += stride;
    }
}

fn mask6(blimit: i32, limit: i32, p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32) -> bool {
    (p2 - p1).abs() <= limit && (p1 - p0).abs() <= limit &&
        (q2 - q1).abs() <= limit && (q1 - q0).abs() <= limit &&
        (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 <= blimit
}

fn flat6(flat: i32, p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32) -> bool {
    (p1 - p0).abs() <= flat && (q1 - q0).abs() <= flat &&
        (p2 - p0).abs() <= flat && (q2 - q0).abs() <= flat
}

// Assumes slice[0] is set 3 taps back from the edge
fn deblock_size6(rec: &mut[u16], pitch: usize, stride: usize, level: i32, bd: usize) {
    let mut s = 0;
    let blimit = 3 * level + 4 << bd - 8;
    let limit = level << bd - 8;
    let thresh = level >> 4 << bd - 8;
    let flat = 1 << bd - 8;
    for _i in 0..4 {
        let p = &mut rec[s..];
        let p2 = p[0] as i32;
        let p1 = p[pitch] as i32;
        let p0 = p[pitch*2] as i32;
        let q0 = p[pitch*3] as i32;
        let q1 = p[pitch*4] as i32;
        let q2 = p[pitch*5] as i32;
        if mask6(blimit, limit, p2, p1, p0, q0, q1, q2) {
            let x;
            if flat6(flat, p2, p1, p0, q0, q1, q2) {
                x = filter_wide6(p2, p1, p0, q0, q1, q2);
            } else if hev4(thresh, p1, p0, q0, q1) {
                x = filter_narrow2(p1, p0, q0, q1, bd - 8);
            } else {
                x = filter_narrow4(p1, p0, q0, q1, bd - 8);
            }
            stride_copy(&mut p[pitch..], &x, pitch);
        }
        s += stride;
    }
}

fn mask8(blimit: i32, limit: i32, p3: i32, p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32, q3: i32) -> bool {
    (p3 - p2).abs() <= limit && (p2 - p1).abs() <= limit && (p1 - p0).abs() <= limit &&
        (q3 - q2).abs() <= limit && (q2 - q1).abs() <= limit && (q1 - q0).abs() <= limit &&
        (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 <= blimit
}

fn flat8(flat: i32, p3:i32, p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32, q3: i32) -> bool {
    (p1 - p0).abs() <= flat && (q1 - q0).abs() <= flat &&
        (p2 - p0).abs() <= flat && (q2 - q0).abs() <= flat &&
        (p3 - p0).abs() <= flat && (q3 - q0).abs() <= flat
}

// Assumes rec[0] is set 4 taps back from the edge
fn deblock_size8(rec: &mut[u16], pitch: usize, stride: usize, level: i32, bd: usize) {
    let mut s = 0;
    let blimit = 3 * level + 4 << bd - 8;
    let limit = level << bd - 8;
    let thresh = level >> 4 << bd - 8;
    let flat = 1 << bd - 8;
    for _i in 0..4 {
        let p = &mut rec[s..];
        let p3 = p[0] as i32;
        let p2 = p[pitch] as i32;
        let p1 = p[pitch*2] as i32;
        let p0 = p[pitch*3] as i32;
        let q0 = p[pitch*4] as i32;
        let q1 = p[pitch*5] as i32;
        let q2 = p[pitch*6] as i32;
        let q3 = p[pitch*7] as i32;
        if mask8(blimit, limit, p3, p2, p1, p0, q0, q1, q2, q3) {
            if flat8(flat, p3, p2, p1, p0, q0, q1, q2, q3) {
                let x = filter_wide8(p3, p2, p1, p0, q0, q1, q2, q3);
                stride_copy(&mut p[pitch..], &x, pitch);
            } else {
                let x;
                if hev4(thresh, p1, p0, q0, q1) {
                    x = filter_narrow2(p1, p0, q0, q1, bd - 8);
                } else {
                    x = filter_narrow4(p1, p0, q0, q1, bd - 8);
                }
                stride_copy(&mut p[pitch*2..], &x, pitch);
            }
        }
        s += stride;
    }
}

fn flat14_outer(flat: i32, p6: i32, p5: i32, p4: i32, p0: i32, q0: i32, q4: i32, q5: i32, q6: i32) -> bool {
    (p4 - p0).abs() <= flat && (q4 - q0).abs() <= flat &&
        (p5 - p0).abs() <= flat && (q5 - q0).abs() <= flat &&
        (p6 - p0).abs() <= flat && (q6 - q0).abs() <= flat
}

// Assumes slice[0] is set 7 taps back from the edge, accumulates 12 pixels
fn deblock_size14(rec: &mut[u16], pitch: usize, stride: usize, level: i32, bd: usize) {
    let mut s = 0;
    let blimit = 3 * level + 4 << bd - 8;
    let limit = level << bd - 8;
    let thresh = level >> 4 << bd - 8;
    let flat = 1 << bd - 8;
    for _i in 0..4 {
        let p = &mut rec[s..];
        let p6 = p[0] as i32;
        let p5 = p[pitch] as i32;
        let p4 = p[pitch*2] as i32;
        let p3 = p[pitch*3] as i32;
        let p2 = p[pitch*4] as i32;
        let p1 = p[pitch*5] as i32;
        let p0 = p[pitch*6] as i32;
        let q0 = p[pitch*7] as i32;
        let q1 = p[pitch*8] as i32;
        let q2 = p[pitch*9] as i32;
        let q3 = p[pitch*10] as i32;
        let q4 = p[pitch*11] as i32;
        let q5 = p[pitch*12] as i32;
        let q6 = p[pitch*13] as i32;
        // 'mask' test
        if mask8(blimit, limit, p3, p2, p1, p0, q0, q1, q2, q3) {
            // inner flatness test
            if flat8(flat, p3, p2, p1, p0, q0, q1, q2, q3) {
                // outer flatness test
                if flat14_outer(flat, p6, p5, p4, p0, q0, q4, q5, q6) {
                    // sufficient flatness across 14 pixel width; run full-width filter
                    let x = filter_wide14(p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6);
                    stride_copy(&mut p[pitch..], &x, pitch);
                } else {
                    // only flat in inner area, run 8-tap
                    let x = filter_wide8(p3, p2, p1, p0, q0, q1, q2, q3);
                    stride_copy(&mut p[pitch*4..], &x, pitch);
                }
            } else {
                // not flat, run narrow filter
                let x;
                if hev4(thresh, p1, p0, q0, q1) {
                    x = filter_narrow2(p1, p0, q0, q1, bd - 8);
                } else {
                    x = filter_narrow4(p1, p0, q0, q1, bd - 8);
                }
                stride_copy(&mut p[pitch*5..], &x, pitch);
            }
        }
        s += stride;
    }
}

fn filter_v_edge(deblock: &DeblockState,
                 bc: &BlockContext,
                 bo: &BlockOffset,
                 p: &mut Plane,
                 pli: usize,
                 bd: usize) {
    let block = bc.at(&bo);
    let tx_edge = bo.x & (block.tx_w - 1) == 0;
    if tx_edge {
        let prev_block = deblock_left(bc, bo, p);
        let block_edge = bo.x & (block.n4_w - 1) == 0;
        let filter_size = deblock_size(block, prev_block, p, pli, true, block_edge);
        if filter_size > 0 {
            let level = deblock_level(deblock, block, prev_block, pli, true);
            let po = bo.plane_offset(&p.cfg);
            let stride = p.cfg.stride;
            let mut plane_slice = p.mut_slice(&po);
            plane_slice.x -= (filter_size>>1) as isize;
            let slice = plane_slice.as_mut_slice();
            match filter_size {
                4 => { deblock_size4(slice, 1, stride, level, bd); },
                6 => { deblock_size6(slice, 1, stride, level, bd); },
                8 => { deblock_size8(slice, 1, stride, level, bd); },
                14 => { deblock_size14(slice, 1, stride, level, bd); },
                _ => {unreachable!()}
            }
        }
    }
}

fn filter_h_edge(deblock: &DeblockState,
                 bc: &BlockContext,
                 bo: &BlockOffset,
                 p: &mut Plane,
                 pli: usize,
                 bd: usize) {
    let block = bc.at(&bo);
    let tx_edge = bo.y & (block.tx_h - 1) == 0;
    if tx_edge {
        let prev_block = deblock_up(bc, bo, p);
        let block_edge = bo.y & (block.n4_h - 1) == 0;
        let filter_size = deblock_size(block, prev_block, p, pli, false, block_edge);
        if filter_size > 0 {
            let level = deblock_level(deblock, block, prev_block, pli, false);
            let po = bo.plane_offset(&p.cfg);
            let stride = p.cfg.stride;
            let mut plane_slice = p.mut_slice(&po);
            plane_slice.y -= (filter_size>>1) as isize;
            let slice = plane_slice.as_mut_slice(); 
            match filter_size {
                4 => { deblock_size4(slice, stride, 1, level, bd); },
                6 => { deblock_size6(slice, stride, 1, level, bd); },
                8 => { deblock_size8(slice, stride, 1, level, bd); },
                14 => { deblock_size14(slice, stride, 1, level, bd); },
                _ => {unreachable!()}
            }
        }
    }
}

// Deblocks all edges, vertical and horizontal, in a single plane
pub fn deblock_plane(deblock: &DeblockState, p: &mut Plane,
                     pli: usize, bc: &mut BlockContext, bd: usize) {

    let xdec = p.cfg.xdec;
    let ydec = p.cfg.ydec;

    match pli {
        0 => if deblock.levels[0] == 0 && deblock.levels[1] == 0 {return},
        1 => if deblock.levels[2] == 0 {return},
        2 => if deblock.levels[3] == 0 {return},
        _ => {return}
    }

    // vertical edge filtering leads horizonal by one full MI-sized
    // row (and horizontal filtering doesn't happen along the upper
    // edge).  Unroll to avoid corner-cases.
    if bc.rows > 0 {
        for x in (1<<xdec..bc.cols).step_by(1 << xdec) {
            filter_v_edge(deblock, bc, &BlockOffset{x: x, y: 0}, p, pli, bd);
        }
        if bc.rows > 1 << ydec {
            for x in (1<<xdec..bc.cols).step_by(1 << xdec) {
                filter_v_edge(deblock, bc, &BlockOffset{x: x, y: 1 << ydec}, p, pli, bd);
            }
        }
    }
    
    // filter rows where vertical and horizontal edge filtering both
    // happen (horizontal edge filtering lags vertical by one row).
    for y in ((2 << ydec)..bc.rows).step_by(1 << ydec) {
        // Check for vertical edge at first MI block boundary on this row
        if 1 << xdec < bc.cols { 
            filter_v_edge(deblock, bc, &BlockOffset{x: 1 << xdec, y: y}, p, pli, bd);
        }
        // run the rest of the row with both vertical and horizontal edge filtering.
        // Horizontal lags vertical edge by one row and two columns.
        for x in (2 << xdec..bc.cols).step_by(1 << xdec){
            filter_v_edge(deblock, bc, &BlockOffset{x: x, y: y}, p, pli, bd);
            filter_h_edge(deblock, bc, &BlockOffset{x: x - (2 << xdec), y: y - (1 << ydec)}, p, pli, bd);
        }
        // ..and the last two horizontal edges for the row
        if bc.cols - (2 << xdec) > 0 {
            filter_h_edge(deblock, bc, &BlockOffset{x: bc.cols - (2 << xdec), y: y - (1 << ydec)}, p, pli, bd);
            if bc.cols - (1 << xdec) > 0 {
                filter_h_edge(deblock, bc, &BlockOffset{x: bc.cols - (1 << xdec), y: y - (1 << ydec)}, p, pli, bd);
            }
        }
    }

    // Last horizontal row, vertical is already complete
    if bc.rows > 1 << ydec {
        for x in (0..bc.cols).step_by(1 << xdec) {
            filter_h_edge(deblock, bc, &BlockOffset{x: x, y: bc.rows - (1 << ydec)}, p, pli, bd);
        }
    }
}

// Deblocks all edges in all planes of a frame
pub fn deblock_filter_frame(fs: &mut FrameState,
                            bc: &mut BlockContext, bit_depth: usize) {
    for pli in 0..PLANES {
        deblock_plane(&fs.deblock, &mut fs.rec.planes[pli], pli, bc, bit_depth);
    }
}

pub fn deblock_filter_optimize(fi: &FrameInvariants, fs: &mut FrameState,
                               _bc: &mut BlockContext, bit_depth: usize) {
    let q = ac_q(fi.base_q_idx, bit_depth) as i32;
    let level = clamp (match bit_depth {
        8 => {
            if fi.frame_type == FrameType::KEY {
                q * 17563 - 421574 + (1<<18>>1) >> 18
            } else {
                q * 6017 + 650707 + (1<<18>>1) >> 18
            }
        }
        10 => {
            if fi.frame_type == FrameType::KEY {
                (q * 20723 + 4060632 + (1<<20>>1) >> 20) - 4
            } else {
                q * 20723 + 4060632 + (1<<20>>1) >> 20
            }
        }
        12 => {
            if fi.frame_type == FrameType::KEY {
                (q * 20723 + 16242526 + (1<<22>>1) >> 22) - 4
            } else {
                q * 20723 + 16242526 + (1<<22>>1) >> 22
            }
        }
        _ => {assert!(false); 0}
    }, 0, MAX_LOOP_FILTER as i32) as u8;

    fs.deblock.levels[0] = level;
    fs.deblock.levels[1] = level;
    fs.deblock.levels[2] = level;
    fs.deblock.levels[3] = level;
}
