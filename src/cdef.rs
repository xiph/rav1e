// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
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
use util::clamp;
use FrameInvariants;
use Frame;

const CDEF_VERY_LARGE: u16 = 30000;
const CDEF_SEC_STRENGTHS: u8 = 4;

fn msb(x: i32) -> i32 {
    31 ^ (x.leading_zeros() as i32)
}

// Instead of dividing by n between 2 and 8, we multiply by 3*5*7*8/n.
// The output is then 840 times larger, but we don't care for finding
// the max. */
const CDEF_DIV_TABLE: [i32; 9] = [ 0, 840, 420, 280, 210, 168, 140, 120, 105 ];

// Detect direction. 0 means 45-degree up-right, 2 is horizontal, and so on.
// The search minimizes the weighted variance along all the lines in a
// particular direction, i.e. the squared error between the input and a
// "predicted" block where each pixel is replaced by the average along a line
// in a particular direction. Since each direction have the same sum(x^2) term,
// that term is never computed. See Section 2, step 2, of:
// http://jmvalin.ca/notes/intra_paint.pdf
fn cdef_find_dir(img: &[u16], stride: usize, var: &mut i32, coeff_shift: i32) -> i32 {
    let mut cost: [i32; 8] = [0; 8];
    let mut partial: [[i32; 15]; 8] = [[0; 15]; 8];
    let mut best_cost: i32 = 0;
    let mut best_dir = 0;
    for i in 0..8 {
        for j in 0..8 {
            // We subtract 128 here to reduce the maximum range of the squared
            // partial sums. 
            debug_assert!((img[i * stride + j] >> coeff_shift) <= 255);
            let x = (img[i * stride + j] as i32 >> coeff_shift) - 128;
            partial[0][i + j] += x;
            partial[1][i + j / 2] += x;
            partial[2][i] += x;
            partial[3][3 + i - j / 2] += x;
            partial[4][7 + i - j] += x;
            partial[5][3 - i / 2 + j] += x;
            partial[6][j] += x;
            partial[7][i / 2 + j] += x;
        }
    }
    for i in 0..8 {
        cost[2] += partial[2][i] * partial[2][i];
        cost[6] += partial[6][i] * partial[6][i];
    }
    cost[2] *= CDEF_DIV_TABLE[8];
    cost[6] *= CDEF_DIV_TABLE[8];
    for i in 0..7 {
        cost[0] += (partial[0][i]*partial[0][i] +
                    partial[0][14-i]*partial[0][14-i]) * CDEF_DIV_TABLE[i + 1];
        cost[4] += (partial[4][i]*partial[4][i] +
                    partial[4][14-i]*partial[4][14-i]) * CDEF_DIV_TABLE[i + 1];
    }
    cost[0] += partial[0][7] * partial[0][7] * CDEF_DIV_TABLE[8];
    cost[4] += partial[4][7] * partial[4][7] * CDEF_DIV_TABLE[8];
    let mut i = 1;
    while i<8 {
        for j in 0..5 {
            cost[i] += partial[i][3 + j] * partial[i][3 + j];
        }
        cost[i] *= CDEF_DIV_TABLE[8];
        for j in 0..3 {
            cost[i] += (partial[i][j]*partial[i][j] +
                        partial[i][10-j]*partial[i][10-j]) * CDEF_DIV_TABLE[2 * j + 2];
        }
        i+=2;
    }
    for i in 0..8 {
        if cost[i] > best_cost {
            best_cost = cost[i];
            best_dir = i;
        }
    }
    // Difference between the optimal variance and the variance along the
    // orthogonal direction. Again, the sum(x^2) terms cancel out. 
    // We'd normally divide by 840, but dividing by 1024 is close enough
    // for what we're going to do with this. */
    *var = (best_cost - cost[(best_dir + 4) & 7]) >> 10;
        
    best_dir as i32
}

fn constrain(diff: i32, threshold: i32, damping: i32) -> i32 {
    if threshold != 0 {
        let shift = cmp::max(0, damping - msb(threshold));
        let magnitude = cmp::min(diff.abs(), cmp::max(0, threshold - (diff.abs() >> shift)));
        if diff < 0 {
            -1 * magnitude
        } else {
            magnitude
        }   
    } else {
        0
    }
}

// Unlike the AOM code, our block addressing points to the UL corner
// of the 2-pixel padding around the block, not the block itself.
// The destination is unpadded.
fn cdef_filter_block(dst: &mut [u16], dstride: i32, input: &[u16], istride: i32,
                     pri_strength: i32, sec_strength: i32, dir: usize, pri_damping: i32,
                     sec_damping: i32, xsize: i32, ysize: i32, coeff_shift: i32) {

    let cdef_pri_taps = [[4, 2], [3, 3]];
    let cdef_sec_taps = [[2, 1], [2, 1]];
    let pri_taps = cdef_pri_taps[((pri_strength >> coeff_shift) & 1) as usize];
    let sec_taps = cdef_sec_taps[((pri_strength >> coeff_shift) & 1) as usize];
    let cdef_directions = [[-1 * istride + 1, -2 * istride + 2 ],
                           [ 0 * istride + 1, -1 * istride + 2 ],
                           [ 0 * istride + 1,  0 * istride + 2 ],
                           [ 0 * istride + 1,  1 * istride + 2 ],
                           [ 1 * istride + 1,  2 * istride + 2 ],
                           [ 1 * istride + 0,  2 * istride + 1 ],
                           [ 1 * istride + 0,  2 * istride + 0 ],
                           [ 1 * istride + 0,  2 * istride - 1 ]];
    for i in 0..ysize {
        for j in 0..xsize {
            let x = input[((i+2) * istride + j+2) as usize];
            let mut sum = 0 as i32;
            let mut max = x;
            let mut min = x;
            for k in 0..2usize {
                let p0 = input[((i+2)*istride + j+2 + cdef_directions[dir][k]) as usize];
                let p1 = input[((i+2)*istride + j+2 - cdef_directions[dir][k]) as usize];
                sum += pri_taps[k] * constrain(p0 as i32 - x as i32, pri_strength, pri_damping);
                sum += pri_taps[k] * constrain(p1 as i32 - x as i32, pri_strength, pri_damping);
                if p0 != CDEF_VERY_LARGE {
                    max = cmp::max(p0, max);
                }
                if p1 != CDEF_VERY_LARGE {
                    max = cmp::max(p1, max);
                }
                min = cmp::min(p0, min);
                min = cmp::min(p1, min);
                let s0 = input[((i+2) * istride + j+2 + cdef_directions[(dir + 2) & 7][k]) as usize];
                let s1 = input[((i+2) * istride + j+2 - cdef_directions[(dir + 2) & 7][k]) as usize];
                let s2 = input[((i+2) * istride + j+2 + cdef_directions[(dir + 6) & 7][k]) as usize];
                let s3 = input[((i+2) * istride + j+2 - cdef_directions[(dir + 6) & 7][k]) as usize];
                if s0 != CDEF_VERY_LARGE {
                    max = cmp::max(s0, max);
                }
                if s1 != CDEF_VERY_LARGE {
                    max = cmp::max(s1, max);
                }
                if s2 != CDEF_VERY_LARGE {
                    max = cmp::max(s2, max);
                }
                if s3 != CDEF_VERY_LARGE {
                    max = cmp::max(s3, max);
                }
                min = cmp::min(s0, min);
                min = cmp::min(s1, min);
                min = cmp::min(s2, min);
                min = cmp::min(s3, min);
                sum += sec_taps[k] * constrain(s0 as i32 - x as i32, sec_strength, sec_damping);
                sum += sec_taps[k] * constrain(s1 as i32 - x as i32, sec_strength, sec_damping);
                sum += sec_taps[k] * constrain(s2 as i32 - x as i32, sec_strength, sec_damping);
                sum += sec_taps[k] * constrain(s3 as i32 - x as i32, sec_strength, sec_damping);
            }
            dst[(i * dstride + j) as usize] = clamp(x as i32 + ((8 + sum - (sum < 0) as i32) >> 4),
                                                    min as i32, max as i32) as u16;
        }
    }
}

// We use the variance of an 8x8 block to adjust the effective filter strength.
fn adjust_strength(strength: i32, var: i32) -> i32 {
    let i = if (var >> 6) != 0 {cmp::min(msb(var >> 6), 12)} else {0};
    if var!=0 {strength * (4 + i) + 8 >> 4} else {0}
}

// Input to this process is the array CurrFrame of reconstructed samples.
// Output from this process is the array CdefFrame containing deringed samples.
// The purpose of CDEF is to perform deringing based on the detected direction of blocks.
// CDEF parameters are stored for each 64 by 64 block of pixels.
// The CDEF filter is applied on each 8 by 8 block of pixels.
// Reference: http://av1-spec.argondesign.com/av1-spec/av1-spec.html#cdef-process
pub fn cdef_frame(fi: &FrameInvariants, rec: &mut Frame, bc: &mut BlockContext, bit_depth: usize) {
    let coeff_shift = bit_depth as i32 - 8;
    let cdef_damping = fi.cdef_damping as i32;

    // Each filter block is 64x64, except right and/or bottom for non-multiple-of-64 sizes.
    // FIXME: 128x128 SB support will break this, we need FilterBlockOffset etc.
    let fb_height = (fi.padded_h + 63) / 64;
    let fb_width = (fi.padded_w + 63) / 64;

    // Construct a padded copy of the reconstructed frame.
    let mut padded_px: [[usize; 2]; 3] = [[0; 2]; 3];
    for p in 0..3 {
        padded_px[p][0] =  (fb_width*64 >> rec.planes[p].cfg.xdec) + 4;
        padded_px[p][1] =  (fb_height*64 >> rec.planes[p].cfg.ydec) + 4;
    }
    let mut cdef_frame = Frame {
        planes: [
            Plane::new(padded_px[0][0], padded_px[0][1], rec.planes[0].cfg.xdec, rec.planes[0].cfg.ydec),
            Plane::new(padded_px[1][0], padded_px[1][1], rec.planes[1].cfg.xdec, rec.planes[1].cfg.ydec),
            Plane::new(padded_px[2][0], padded_px[2][1], rec.planes[2].cfg.xdec, rec.planes[2].cfg.ydec)
        ]
    };
    for p in 0..3 {
        let rec_w = fi.padded_w >> rec.planes[p].cfg.xdec;
        let rec_h = fi.padded_h >> rec.planes[p].cfg.ydec;
        for row in 0..padded_px[p][1] {
            // pad first two elements of current row
            {
                let mut cdef_slice = cdef_frame.planes[p].mut_slice(&PlaneOffset { x: 0, y: row});
                let mut cdef_row = &mut cdef_slice.as_mut_slice()[..2];
                cdef_row[0] = CDEF_VERY_LARGE;
                cdef_row[1] = CDEF_VERY_LARGE;
            }
            // pad out end of current row
            {
                let mut cdef_slice = cdef_frame.planes[p].mut_slice(&PlaneOffset { x: rec_w+2, y: row });
                let mut cdef_row = &mut cdef_slice.as_mut_slice()[..padded_px[p][0]-rec_w-2];
                for x in cdef_row {
                    *x = CDEF_VERY_LARGE;
                }
            }
            // copy current row from rec if we're in data, or pad if we're in first two rows/last N rows
            {
                let mut cdef_slice = cdef_frame.planes[p].mut_slice(&PlaneOffset { x: 2, y: row });
                let mut cdef_row = &mut cdef_slice.as_mut_slice()[..rec_w];
                if row < 2 || row >= rec_h+2 {
                    for x in cdef_row {
                        *x = CDEF_VERY_LARGE;
                    }
                } else {
                    let rec_stride = rec.planes[p].cfg.stride;
                    cdef_row.copy_from_slice(&rec.planes[p].data[(row-2)*rec_stride..(row-1)*rec_stride][..rec_w]);
                }
            }
        }
    }

    // Perform actual CDEF, using the padded copy as source, and the input rec vector as destination.
    for fby in 0..fb_height {
        for fbx in 0..fb_width {
            let sbo = SuperBlockOffset { x: fbx, y: fby };

            // Each direction block is 8x8 in y, potentially smaller if subsampled in chroma
            for by in 0..8 {
                for bx in 0..8 {
                    let block_offset = sbo.block_offset(bx, by);
                    if block_offset.x < bc.cols && block_offset.y < bc.rows {
                        let skip = bc.at(&block_offset).skip;
                        if !skip {
                            let mut dir = 0;
                            let mut var: i32 = 0;
                            let cdef_index = bc.at(&block_offset).cdef_index;
                            let cdef_y_strength = fi.cdef_y_strengths[cdef_index as usize];
                            let cdef_uv_strength = fi.cdef_uv_strengths[cdef_index as usize];
                            let cdef_pri_y_strength = (cdef_y_strength / CDEF_SEC_STRENGTHS) as i32;
                            let mut cdef_sec_y_strength = (cdef_y_strength % CDEF_SEC_STRENGTHS) as i32;
                            let cdef_pri_uv_strength = (cdef_uv_strength / CDEF_SEC_STRENGTHS) as i32;
                            let mut cdef_sec_uv_strength = (cdef_uv_strength % CDEF_SEC_STRENGTHS) as i32;
                            if cdef_sec_y_strength == 3 {
                                cdef_sec_y_strength += 1;
                            }
                            if cdef_sec_uv_strength == 3 {
                                cdef_sec_uv_strength += 1;
                            }
                            for p in 0..3 {
                                let mut rec_plane = &mut rec.planes[p];
                                let rec_po = sbo.plane_offset(&rec_plane.cfg);
                                let mut cdef_plane = &mut cdef_frame.planes[p];
                                let xdec = cdef_plane.cfg.xdec;
                                let ydec = cdef_plane.cfg.ydec;
                                let mut xsize = (fi.padded_w as i32 - 8*bx as i32 >> xdec as i32) - rec_po.x as i32;
                                let mut ysize = (fi.padded_h as i32 - 8*by as i32 >> ydec as i32) - rec_po.y as i32;
                                if xsize > (8>>xdec) {
                                    xsize = 8 >> xdec;
                                }
                                if ysize > (8>>ydec) {
                                    ysize = 8 >> ydec;
                                }
                                if xsize > 0 && ysize > 0 {
                                    let rec_stride = rec_plane.cfg.stride;
                                    let mut rec_slice = &mut rec_plane.mut_slice(&rec_po);
                                    let cdef_stride = cdef_plane.cfg.stride;
                                    let cdef_po = sbo.plane_offset(&cdef_plane.cfg);
                                    let cdef_slice = &cdef_plane.mut_slice(&cdef_po);

                                    let mut local_pri_strength;
                                    let mut local_sec_strength;
                                    let mut local_damping: i32 = cdef_damping + coeff_shift;
                                    let mut local_dir: usize;

                                    if p==0 {
                                        dir = cdef_find_dir(cdef_slice.offset((8*bx>>xdec)+2,(8*by>>ydec)+2),
                                                            cdef_stride, &mut var, coeff_shift);
                                        local_pri_strength = adjust_strength(cdef_pri_y_strength << coeff_shift, var);
                                        local_sec_strength = cdef_sec_y_strength << coeff_shift;
                                        local_dir = if cdef_pri_y_strength != 0 {dir as usize} else {0};
                                    } else {
                                        local_pri_strength = cdef_pri_uv_strength << coeff_shift;
                                        local_sec_strength = cdef_sec_uv_strength << coeff_shift;
                                        local_damping -= 1;
                                        local_dir = if cdef_pri_uv_strength != 0 {dir as usize} else {0};
                                    }

                                    cdef_filter_block(rec_slice.offset_as_mutable(8*bx>>xdec,8*by>>ydec), rec_stride as i32,
                                                      cdef_slice.offset(8*bx>>xdec,8*by>>ydec), cdef_stride as i32, 
                                                      local_pri_strength, local_sec_strength, local_dir,
                                                      local_damping, local_damping,
                                                      xsize, ysize,
                                                      coeff_shift as i32);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
