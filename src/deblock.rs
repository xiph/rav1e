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

fn deblock_adjusted_level(deblock: &DeblockState, block: &Block, pli: usize, pass: usize) -> u8 {
    let idx = if pli == 0 { pass } else { pli+1 };

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
               }, 0, MAX_LOOP_FILTER as i32) as u8
    } else {
        level
    }
}

// Must be called on a tx edge; returns filter setup, location of next
// tx edge for loop index advancement, and current size along loop
// axis in the event block size != tx size
fn deblock_params(fi: &FrameInvariants, deblock: &DeblockState, bc: &mut BlockContext, in_bo: &BlockOffset,
                  p: &mut Plane, pli: usize, pass: usize, block_edge: bool, bd: usize) ->
    (u8, u16, u16, u16) {
    let xdec = p.cfg.xdec;
    let ydec = p.cfg.ydec;
    let w = fi.width as isize >> xdec;
    let h = fi.height as isize >> ydec;
    let po = in_bo.plane_offset(&p.cfg);

    // at or past edge of the frame?  Don't deblock, signal next loop.
    if (po.x >= w) || (po.y >= h) {
        return (0, 0, 0, 0)
    }

    // This little bit of weirdness is straight out of the spec;
    // subsampled chroma uses odd mi row/col
    let bo = BlockOffset{x: in_bo.x | xdec, y: in_bo.y | ydec};
    let block = bc.at(&bo);
    
    // upper/left edge skipped
    if (pass == 0 && po.x == 0) || (pass == 1 && po.y == 0) {
        return (0, 0, 0, 0)
    }

    // We already know we're not at the upper/left corner, so prev_block is in frame
    let prev_block = bc.at(&bo.with_offset(if pass==0 { -(1 << xdec) } else { 0 },
                                           if pass==0 { 0 } else { -(1 << ydec) }));


    // filter application is conditional on skip and block edge
    if !(block_edge || !block.skip || !prev_block.skip ||
         block.ref_frames[0] <= INTRA_FRAME || prev_block.ref_frames[0] <= INTRA_FRAME) {
        return (0, 0, 0, 0)
    }

    let mut level = deblock_adjusted_level(deblock, block, pli, pass) as u16;
    if level == 0 { level = deblock_adjusted_level(deblock, prev_block, pli, pass) as u16; }
    if level <= 0 {
        // When level == 0, the filter is a no-op even if it runs
        (0, 0, 0, 0)
    } else {
        // Filter active; set up the rest
        let (tx_size, prev_tx_size) = if pass == 0 {
            (cmp::max(block.tx_w>>xdec, 1), cmp::max(prev_block.tx_w>>xdec, 1))
        } else {
            (cmp::max(block.tx_h>>ydec, 1), cmp::max(prev_block.tx_h>>ydec, 1))
        };

        // Rather than computing a filterSize and then later a
        // filter_length, we simply construct a filter_length
        // directly.
        let filter_len = cmp::min( if pli==0 {14} else {6}, cmp::min(tx_size, prev_tx_size)<<MI_SIZE_LOG2);
        let blimit = 3 * level + 4;
        let thresh = level >> 4;

        (filter_len as u8, blimit << (bd - 8), level << (bd - 8), thresh << (bd - 8))
    }
}

fn filter_narrow(thresh: u16, bd: usize, data: &mut [u16], pitch: usize,
            p1: i32, p0: i32, q0: i32, q1: i32) {
    let shift = bd - 8;
    let hev =  (p1 - p0).abs() as u16 > thresh || (q1 - q0).abs() as u16 > thresh;
    let base_filter = clamp (if hev { clamp(p1 - q1, -128<<shift, (128<<shift) - 1) } else { 0 } +
                             3 * (q0 - p0), -128<<shift, (128<<shift) - 1);
    
    // Inner taps
    let filter1 = clamp(base_filter + 4, -128<<shift, (128<<shift)-1) >> 3;
    let filter2 = clamp(base_filter + 3, -128<<shift, (128<<shift)-1) >> 3;
    data[pitch] = clamp(p0 + filter2, 0, (256<<shift)-1) as u16;
    data[pitch*2] = clamp(q0 - filter1, 0, (256<<shift)-1) as u16;
    if !hev {
        // Outer taps
        let filter3 = filter1 + 1 >> 1;
        data[0] = clamp(p1 + filter3, 0, (256<<shift)-1) as u16;
        data[pitch*3] = clamp(q1 - filter3, 0, (256<<shift)-1) as u16;
    }
}

fn filter_wide6(data: &mut [u16], pitch: usize,
            p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32) {
    data[pitch]   = (p2*3 + p1*2 + p0*2 + q0 + (1<<2) >> 3) as u16;
    data[pitch*2] = (p2 + p1*2 + p0*2 + q0*2 + q1 + (1<<2) >> 3) as u16;
    data[pitch*3] = (p1 + p0*2 + q0*2 + q1*2 + q2 + (1<<2) >> 3) as u16;
    data[pitch*4] = (p0 + q0*2 + q1*2 + q2*3 + (1<<2) >> 3) as u16;
}

fn filter_wide8(data: &mut [u16], pitch: usize,
            p3: i32, p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32, q3: i32) {
    data[pitch]   = (p3*3 + p2*2 + p1 + p0 + q0 + (1<<2) >> 3) as u16;
    data[pitch*2] = (p3*2 + p2 + p1*2 + p0 + q0 + q1 + (1<<2) >> 3) as u16;
    data[pitch*3] = (p3 + p2 + p1 + p0*2 + q0 + q1 + q2 + (1<<2) >> 3) as u16;
    data[pitch*4] = (p2 + p1 + p0 + q0*2 + q1 + q2 + q3 + (1<<2) >> 3) as u16;
    data[pitch*5] = (p1 + p0 + q0 + q1*2 + q2 + q3*2 + (1<<2) >> 3) as u16;
    data[pitch*6] = (p0 + q0 + q1 + q2*2 + q3*3 + (1<<2) >> 3) as u16;
}

fn filter_wide14(data: &mut [u16], pitch: usize,
             p6: i32, p5: i32, p4: i32, p3: i32, p2: i32, p1: i32, p0: i32,
             q0: i32, q1: i32, q2: i32, q3: i32, q4: i32, q5: i32, q6: i32) {
    data[pitch]   = (p6*7 + p5*2 + p4*2 + p3 + p2 + p1 + p0 + q0 + (1<<3) >> 4) as u16;
    data[pitch*2] = (p6*5 + p5*2 + p4*2 + p3*2 + p2 + p1 + p0 + q0 + q1 + (1<<3) >> 4) as u16;
    data[pitch*3] = (p6*4 + p5 + p4*2 + p3*2 + p2*2 + p1 + p0 + q0 + q1 + q2 + (1<<3) >> 4) as u16;
    data[pitch*4] = (p6*3 + p5 + p4 + p3*2 + p2*2 + p1*2 + p0 + q0 + q1 + q2 + q3 + (1<<3) >> 4) as u16;
    data[pitch*5] = (p6*2 + p5 + p4 + p3 + p2*2 + p1*2 + p0*2 + q0 + q1 + q2 + q3 + q4 + (1<<3) >> 4) as u16;
    data[pitch*6] = (p6 + p5 + p4 + p3 + p2 + p1*2 + p0*2 + q0*2 + q1 + q2 + q3 + q4 + q5 + (1<<3) >> 4) as u16;
    data[pitch*7] = (p5 + p4 + p3 + p2 + p1 + p0*2 + q0*2 + q1*2 + q2 + q3 + q4 + q5 + q6 + (1<<3) >> 4) as u16;
    data[pitch*8] = (p4 + p3 + p2 + p1 + p0 + q0*2 + q1*2 + q2*2 + q3 + q4 + q5 + q6*2 + (1<<3) >> 4) as u16;
    data[pitch*9] = (p3 + p2 + p1 + p0 + q0 + q1*2 + q2*2 + q3*2 + q4 + q5 + q6*3 + (1<<3) >> 4) as u16;
    data[pitch*10] = (p2 + p1 + p0 + q0 + q1 + q2*2 + q3*2 + q4*2 + q5 + q6*4 + (1<<3) >> 4) as u16;
    data[pitch*11] = (p1 + p0 + q0 + q1 + q2 + q3*2 + q4*2 + q5*2 + q6*5 + (1<<3) >> 4) as u16;
    data[pitch*12] = (p0 + q0 + q1 + q2 + q3 + q4*2 + q5*2 + q6*7 + (1<<3) >> 4) as u16;
}   

// Assumes slice[0] is set 2 taps back from the edge
fn deblock_len4<'a>(slice: &'a mut PlaneMutSlice<'a>, pitch: usize, stride: usize,
                 blimit: u16, limit: u16, thresh: u16, bd: usize) {
    let mut s = 0;
    let data = slice.as_mut_slice();
    for _i in 0..4 {
        let p = &mut data[s..];
        let p1 = p[0] as i32;
        let p0 = p[pitch] as i32;
        let q0 = p[pitch*2] as i32;
        let q1 = p[pitch*3] as i32;
        // 'mask' test
        if (p1 - p0).abs() as u16 <= limit &&
            (q1 - q0).abs() as u16 <= limit &&
            (p0 - q0).abs() as u16 * 2 + (p1 - q1).abs() as u16 / 2 <= blimit {
                filter_narrow(thresh, bd, p, pitch, p1, p0, q0, q1);
            }
        s += stride;
    }
}

// Assumes slice[0] is set 3 taps back from the edge
fn deblock_len6<'a>(slice: &'a mut PlaneMutSlice<'a>, pitch: usize, stride: usize,
                 blimit: u16, limit: u16, thresh: u16, bd: usize) {
    let mut s = 0;
    let flat = 1 << bd - 8;
    let data = slice.as_mut_slice();
    for _i in 0..4 {
        let p = &mut data[s..];
        let p2 = p[0] as i32;
        let p1 = p[pitch] as i32;
        let p0 = p[pitch*2] as i32;
        let q0 = p[pitch*3] as i32;
        let q1 = p[pitch*4] as i32;
        let q2 = p[pitch*5] as i32;
        // 'mask' test
        if (p2 - p1).abs() as u16 <= limit &&
            (p1 - p0).abs() as u16 <= limit &&
            (q1 - q0).abs() as u16 <= limit &&
            (q2 - q1).abs() as u16 <= limit &&
            (p0 - q0).abs() as u16 * 2 + (p1 - q1).abs() as u16 / 2 <= blimit {
                // 'flat' test
                if (p1 - p0).abs() as u16 <= flat &&
                    (q1 - q0).abs() as u16 <= flat &&
                    (p2 - p0).abs() as u16 <= flat &&
                    (q2 - q0).abs() as u16 <= flat {
                        // sufficiently flat, run wide filter
                        filter_wide6(p, pitch, p2, p1, p0, q0, q1, q2);
                    } else {
                        // insufficiently flat, run narrow filter
                        filter_narrow(thresh, bd, &mut p[pitch..], pitch, p1, p0, q0, q1);
                    }
            }
        s += stride;
    }
}

// Assumes slice[0] is set 4 taps back from the edge
fn deblock_len8<'a>(slice: &'a mut PlaneMutSlice<'a>, pitch: usize, stride: usize,
                 blimit: u16, limit: u16, thresh: u16, bd: usize) {
    let mut s = 0;
    let flat = 1 << bd - 8;
    let data = slice.as_mut_slice();
    for _i in 0..4 {
        let p = &mut data[s..];
        let p3 = p[0] as i32;
        let p2 = p[pitch] as i32;
        let p1 = p[pitch*2] as i32;
        let p0 = p[pitch*3] as i32;
        let q0 = p[pitch*4] as i32;
        let q1 = p[pitch*5] as i32;
        let q2 = p[pitch*6] as i32;
        let q3 = p[pitch*7] as i32;
        // 'mask' test
        if (p3 - p2).abs() as u16 <= limit &&
            (p2 - p1).abs() as u16 <= limit &&
            (p1 - p0).abs() as u16 <= limit &&
            (q1 - q0).abs() as u16 <= limit &&
            (q2 - q1).abs() as u16 <= limit &&
            (q3 - q2).abs() as u16 <= limit &&
            (p0 - q0).abs() as u16 * 2 + (p1 - q1).abs() as u16 / 2 <= blimit {
                // 'flat' test
                if (p1 - p0).abs() as u16 <= flat &&
                    (q1 - q0).abs() as u16 <= flat &&
                    (p2 - p0).abs() as u16 <= flat &&
                    (q2 - q0).abs() as u16 <= flat &&
                    (p3 - p0).abs() as u16 <= flat &&
                    (q3 - q0).abs() as u16 <= flat {
                        // sufficiently flat, run wide filter
                        filter_wide8(p, pitch, p3, p2, p1, p0, q0, q1, q2, q3);
                    } else {
                        // insufficiently flat, run narrow filter
                        filter_narrow(thresh, bd, &mut p[pitch*2..], pitch, p1, p0, q0, q1);
                    }
            }
        s += stride;
    }
}

// Assumes slice[0] is set 7 taps back from the edge
fn deblock_len14<'a>(slice: &'a mut PlaneMutSlice<'a>, pitch: usize, stride: usize,
                 blimit: u16, limit: u16, thresh: u16, bd: usize) {
    let mut s = 0;
    let flat = 1 << bd - 8;
    let data = slice.as_mut_slice();
    for _i in 0..4 {
        let p = &mut data[s..];
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
        if (p3 - p2).abs() as u16 <= limit &&
            (p2 - p1).abs() as u16 <= limit &&
            (p1 - p0).abs() as u16 <= limit &&
            (q1 - q0).abs() as u16 <= limit &&
            (q2 - q1).abs() as u16 <= limit &&
            (q3 - q2).abs() as u16 <= limit &&
            (p0 - q0).abs() as u16 * 2 + (p1 - q1).abs() as u16 / 2 <= blimit {
                // 'flat' test (inner pixel flatness)
                if (p1 - p0).abs() as u16 <= flat &&
                    (q1 - q0).abs() as u16 <= flat &&
                    (p2 - p0).abs() as u16 <= flat &&
                    (q2 - q0).abs() as u16 <= flat &&
                    (p3 - p0).abs() as u16 <= flat &&
                    (q3 - q0).abs() as u16 <= flat {
                        // 'flat2' test (outer pixel flatness)
                        if (p4 - p0).abs() as u16 <= flat &&
                            (q4 - q0).abs() as u16 <= flat &&
                            (p5 - p0).abs() as u16 <= flat &&
                            (q5 - q0).abs() as u16 <= flat &&
                            (p6 - p0).abs() as u16 <= flat &&
                            (q6 - q0).abs() as u16 <= flat {
                                // sufficient flatness across 14 pixel width; run full-width filter
                                filter_wide14(p, pitch, p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6);
                            } else {
                                // Only sufficient flatness across inner 8 pixels; run 8-tap wide filter
                                filter_wide8(&mut p[pitch*3..], pitch, p3, p2, p1, p0, q0, q1, q2, q3);
                            }
                    } else {
                        // insufficient flatness; run narrow filter
                        filter_narrow(thresh, bd, &mut p[pitch*5..], pitch, p1, p0, q0, q1);
                    }
            }
        s += stride;
    }
}

// Deblock edges in a single plane of a single 64x64 superblock
// Works in-place

// deblock_superblock uses a different prcessing boundary convention
// with respect to the superblock, as compared to libaom, in order to
// get the vertical and horizontal passes more tightly coupled in
// memory.

// In libaom, the edges that are deblocked for a given block offset
// are the left-hand vertical edge (pass 0) or the upper horizontal
// edge (pass 2).  Rav1e uses that convention internally when dealing
// with individual transform blocks.

// Libaom follows the same convention within superblocks; in eeach
// pass, the uppermost or leftmost edge of a superblock are deblocked,
// leaving the lower/right hand edge unprocessed (to be handled by the
// next superblock deblocking call).  The version of deblocking in
// libaom that runs V and H 'simultaneously' has the horizontal
// deblocking lag the vertical by a full superblock.

// Vertical edge deblocking must always preceed the horizontal
// deblocking for any given filtered pixel. Rav1e deblocks
// simultaneously at a sub-superblock level, thus it alters the
// convention to deblocking to the righthand-most and upper-most edges
// of a superblock, and leaving the left-hand and lower edges
// unprocessed.  The left-hand edge would have been deblocked by the
// previous superblock deblocking call, and the lower edge will be
// deblocked by the next call, as in libaom.

fn deblock_superblock(fi: &FrameInvariants,
                      deblock: &DeblockState,
                      plane: &mut Plane,
                      pli: usize,
                      bc: &mut BlockContext,
                      sbo: &SuperBlockOffset,
                      bit_depth: usize) {
    
    let xdec = plane.cfg.xdec;
    let ydec = plane.cfg.ydec;
    let stride = plane.cfg.stride;
    let bo0 = sbo.block_offset(0, 0);

    // row loop is oversized by 1 iteration; vertical edge processing
    // ignores the last, horizontal edge processing ignores the first.
    for yi in (0..MAX_MIB_SIZE + (1 << ydec)).step_by(1 << ydec) {
        let y_v = yi + bo0.y;
        // Vertical edges along one MISIZE row
        if yi < MAX_MIB_SIZE && y_v < bc.rows {
            let mut bx = bo0.x;
            let mut tx = bo0.x;
            // for vertical edges, the col iteration is also
            // oversized; we skip over the left edge and process the
            // right edge.
            while tx < cmp::min(bo0.x + MAX_MIB_SIZE + 1, bc.cols) {
                let bo = BlockOffset{x: tx, y: y_v};
                let (block_adv, tx_adv) = {
                    let block = bc.at(&bo);
                    (cmp::max(block.n4_w, 1<<xdec), cmp::max(block.tx_w, 1<<xdec))
                };
                if tx > bo0.x { // skip processing the left edge
                    let (filter_len, blimit, limit, thresh) =
                        deblock_params(fi, deblock, bc, &bo, plane, pli, 0, bx == tx, bit_depth);            
                    if filter_len > 0 {
                        let po = bo.plane_offset(&plane.cfg);
                        let mut slice = plane.mut_slice(&po);
                        slice.x -= (filter_len>>1) as isize;
                        match filter_len {
                            4 => { deblock_len4(&mut slice, 1, stride, blimit, limit, thresh, bit_depth); },
                            6 => { deblock_len6(&mut slice, 1, stride, blimit, limit, thresh, bit_depth); },
                            8 => { deblock_len8(&mut slice, 1, stride, blimit, limit, thresh, bit_depth); },
                            14 => { deblock_len14(&mut slice, 1, stride, blimit, limit, thresh, bit_depth); },
                            _ => {}
                        }
                    }
                }
                tx += tx_adv;
                if bx + block_adv == tx {bx = tx};
            }
        }
        // Horizontal edges along one MI_SIZE row, lagging the
        // vertical by two rows to start, one row to end to ensure we have a minimum of 7
        // filtered rows of pixels (in fact, we have 8).
        if yi > 0 {
            let y_h = y_v - (1 << ydec);
            if y_h < bc.rows {
                let mut bx = bo0.x;
                for tx in (bo0.x..cmp::min(bo0.x + MAX_MIB_SIZE, bc.cols)).step_by(1 << xdec) {
                    let bo = BlockOffset{x: tx, y: y_h};
                    let (block_adv, block_edge, tx_edge) = {
                        let block = bc.at(&bo);
                        (cmp::max(block.n4_w, 1<<xdec),
                         bo.y & (block.n4_h - 1) == 0,
                         bo.y & (block.tx_h - 1) == 0)
                    };
                    // Are we actually on a transform edge?
                    if tx_edge {
                        let (filter_len, blimit, limit, thresh) =
                            deblock_params(fi, deblock, bc, &bo, plane, pli, 1, block_edge, bit_depth);
                        if filter_len > 0 {
                            let po = bo.plane_offset(&plane.cfg);
                            let mut slice = plane.mut_slice(&po);
                            slice.y -= (filter_len>>1) as isize;
                            match filter_len {
                                4 => { deblock_len4(&mut slice, stride, 1, blimit, limit, thresh, bit_depth); },
                                6 => { deblock_len6(&mut slice, stride, 1, blimit, limit, thresh, bit_depth); },
                                8 => { deblock_len8(&mut slice, stride, 1, blimit, limit, thresh, bit_depth); },
                                14 => { deblock_len14(&mut slice, stride, 1, blimit, limit, thresh, bit_depth); },
                                _ => {}
                            }
                        }
                    }
                    if bx + block_adv == tx {bx = tx};
                }
            }
        }
    }
}

// Deblocks all edges, vertical and horizontal, in a single plane
pub fn deblock_plane(fi: &FrameInvariants, deblock: &DeblockState, plane: &mut Plane,
                     pli: usize, bc: &mut BlockContext, bit_depth: usize) {
    // Each filter block is 64x64, except right and/or bottom for non-multiple-of-64 sizes.
    // FIXME: 128x128 SB support will break this, we need FilterBlockOffset etc.
    let fb_height = (fi.padded_h + 63) / 64;
    let fb_width = (fi.padded_w + 63) / 64;

    match pli {
        0 => if deblock.levels[0] == 0 && deblock.levels[1] == 0 {return},
        1 => if deblock.levels[2] == 0 {return},
        2 => if deblock.levels[3] == 0 {return},
        _ => {return}
    }
      
    // filter vertical and horizontal edges by super block.
    for row in 0..fb_height {
        for col in 0..fb_width {
            let sbo = SuperBlockOffset { x: col, y: row };
            deblock_superblock(fi, deblock, plane, pli, bc, &sbo, bit_depth);
        }
    }
}

// Deblocks all edges in all planes of a frame
pub fn deblock_filter_frame(fi: &FrameInvariants, fs: &mut FrameState,
                            bc: &mut BlockContext, bit_depth: usize) {
    for p in 0..PLANES {
        deblock_plane(fi, &fs.deblock, &mut fs.rec.planes[p], p, bc, bit_depth);
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
