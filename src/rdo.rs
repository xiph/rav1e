// Copyright (c) 2001-2016, Alliance for Open Media. All rights reserved
// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(non_camel_case_types)]

use BlockSize;
use FrameInvariants;
use FrameState;
use FrameType;
use encode_block;
use write_tx_blocks;
use context::*;
use ec::OD_BITRES;
use partition::*;
use plane::*;
use predict::RAV1E_INTRA_MODES;
use quantize::dc_q;
use std;
use std::vec::Vec;

#[derive(Clone)]
pub struct RDOOutput {
    pub rd_cost: f64,
    pub part_type: PartitionType,
    pub part_modes: Vec<RDOPartitionOutput>
}

#[derive(Clone)]
pub struct RDOPartitionOutput {
    pub rd_cost: f64,
    pub bo: BlockOffset,
    pub pred_mode: PredictionMode
}

// Sum of Squared Error for a wxh block
fn sse_wxh(src1: &PlaneSlice, src2: &PlaneSlice, w: usize, h: usize) -> u64 {
    let mut sse: u64 = 0;
    for j in 0..h {
        for i in 0..w {
            let dist = (src1.p(i, j) as i16 - src2.p(i, j) as i16) as i64;
            sse += (dist * dist) as u64;
        }
    }
    sse
}

// Compute the rate-distortion cost for an encode
fn compute_rd_cost(fi: &FrameInvariants, fs: &FrameState,
                   w_y: u8, h_y: u8, w_uv: u8, h_uv: u8,
                   partition_start_x: usize, partition_start_y: usize,
                   bo: &BlockOffset, bit_cost: u32) -> f64 {
    let q = dc_q(fi.qindex) as f64;

    // Convert q into Q0 precision, given that libaom quantizers are Q3
    let q0 = q / 8.0_f64;

    // Lambda formula from doc/theoretical_results.lyx in the daala repo
    // Use Q0 quantizer since lambda will be applied to Q0 pixel domain
    let lambda = q0 * q0 * std::f64::consts::LN_2 / 6.0;

    // Compute distortion
    let po = bo.plane_offset(&fs.input.planes[0].cfg);
    let mut distortion = sse_wxh(&fs.input.planes[0].slice(&po),
                                 &fs.rec.planes[0].slice(&po),
                                 w_y as usize, h_y as usize);

    // Add chroma distortion only when it is available
    if w_uv > 0 && h_uv > 0 {
        for p in 1..3 {
            let sb_offset = bo.sb_offset().plane_offset(&fs.input.planes[p].cfg);
            let po = PlaneOffset {
                x: sb_offset.x + partition_start_x,
                y: sb_offset.y + partition_start_y
            };

            distortion += sse_wxh(&fs.input.planes[p].slice(&po),
                                  &fs.rec.planes[p].slice(&po),
                                  w_uv as usize, h_uv as usize);
        }
    };

    // Compute rate
    let rate = (bit_cost as f64) / ((1 << OD_BITRES) as f64);

    let rd_cost = (distortion as f64) + lambda * rate;

    rd_cost
}

// RDO-based mode decision
pub fn rdo_mode_decision(fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter,
                         bsize: BlockSize, bo: &BlockOffset) -> RDOOutput {
    let mut best_mode = PredictionMode::DC_PRED;
    let mut best_rd = std::f64::MAX;
    let tell = cw.w.tell_frac();

    // Get block luma and chroma dimensions
    let w = block_size_wide[bsize as usize];
    let h = block_size_high[bsize as usize];

    let xdec = fs.input.planes[1].cfg.xdec;
    let ydec = fs.input.planes[1].cfg.ydec;
    let mut w_uv = w >> xdec;
    let mut h_uv = h >> ydec;

    if (w_uv == 0 || h_uv == 0) && has_chroma(bo, bsize, xdec, ydec) {
        w_uv = 4;
        h_uv = 4;
    }

    let partition_start_x = (bo.x & LOCAL_BLOCK_MASK) >> xdec << MI_SIZE_LOG2;
    let partition_start_y = (bo.y & LOCAL_BLOCK_MASK) >> ydec << MI_SIZE_LOG2;

    for &mode in RAV1E_INTRA_MODES {
        if fi.frame_type == FrameType::KEY && mode >= PredictionMode::NEARESTMV {
            break;
        }

        let checkpoint = cw.checkpoint();

        encode_block(fi, fs, cw, mode, bsize, bo);

        let cost = cw.w.tell_frac() - tell;
        let rd = compute_rd_cost(fi, fs, w, h, w_uv, h_uv,
                                 partition_start_x, partition_start_y, bo, cost);

        if rd < best_rd {
            best_rd = rd;
            best_mode = mode;
        }

        cw.rollback(&checkpoint);
    }

    assert!(best_rd >= 0_f64);

    let rdo_output = RDOOutput {
        rd_cost: best_rd,
        part_type: PartitionType::PARTITION_NONE,
        part_modes: vec![RDOPartitionOutput { bo: bo.clone(), pred_mode: best_mode, rd_cost: best_rd }]
    };

    rdo_output
}

// RDO-based intra frame transform type decision
pub fn rdo_tx_type_decision(fi: &FrameInvariants, fs: &mut FrameState,
                                   cw: &mut ContextWriter, mode: PredictionMode,
                                   bsize: BlockSize, bo: &BlockOffset, tx_size: TxSize) -> TxType {
    let mut best_type = TxType::DCT_DCT;
    let mut best_rd = std::f64::MAX;
    let tell = cw.w.tell_frac();

    // Get block luma and chroma dimensions
    let w = block_size_wide[bsize as usize];
    let h = block_size_high[bsize as usize];

    let xdec = fs.input.planes[1].cfg.xdec;
    let ydec = fs.input.planes[1].cfg.ydec;
    let mut w_uv = w >> xdec;
    let mut h_uv = h >> ydec;

    if (w_uv == 0 || h_uv == 0) && has_chroma(bo, bsize, xdec, ydec) {
        w_uv = 4;
        h_uv = 4;
    }

    let partition_start_x = (bo.x & LOCAL_BLOCK_MASK) >> xdec << MI_SIZE_LOG2;
    let partition_start_y = (bo.y & LOCAL_BLOCK_MASK) >> ydec << MI_SIZE_LOG2;

    for &tx_type in RAV1E_INTRA_TX_TYPES {
        if tx_type == TxType::IDTX && tx_size >= TxSize::TX_32X32 {
            continue;
        }

        let checkpoint = cw.checkpoint();

        write_tx_blocks(fi, fs, cw, mode, bo, bsize, tx_size, tx_type, false);

        let cost = cw.w.tell_frac() - tell;
        let rd = compute_rd_cost(fi, fs, w, h, w_uv, h_uv,
                                 partition_start_x, partition_start_y, bo, cost);

        if rd < best_rd {
            best_rd = rd;
            best_type = tx_type;
        }

        cw.rollback(&checkpoint);
    }

    assert!(best_rd >= 0_f64);

    best_type
}

// RDO-based single level partitioning decision
pub fn rdo_partition_decision(fi: &FrameInvariants, fs: &mut FrameState,
                          cw: &mut ContextWriter, bsize: BlockSize, bo: &BlockOffset,
                          cached_block: &RDOOutput) -> RDOOutput {
    let max_rd = std::f64::MAX;

    let mut best_partition = cached_block.part_type;
    let mut best_rd = cached_block.rd_cost;
    let mut best_pred_modes = cached_block.part_modes.clone();

    for &partition in RAV1E_PARTITION_TYPES {
        // Do not re-encode results we already have
        if partition == cached_block.part_type && cached_block.rd_cost < max_rd {
            continue;
        }

        let checkpoint = cw.checkpoint();

        let mut rd: f64;
        let mut child_modes = std::vec::Vec::new();

        match partition {
            PartitionType::PARTITION_NONE => {
                if bsize > BlockSize::BLOCK_32X32 {
                    continue;
                }

                let mode_decision = cached_block.part_modes.get(0)
                    .unwrap_or(&rdo_mode_decision(fi, fs, cw, bsize, bo).part_modes[0]).clone();
                child_modes.push(mode_decision);
            },
            PartitionType::PARTITION_SPLIT => {
                let subsize = get_subsize(bsize, partition);

                if subsize == BlockSize::BLOCK_INVALID {
                    continue;
                }

                let bs = mi_size_wide[bsize as usize];
                let hbs = bs >> 1; // Half the block size in blocks

                let offset = BlockOffset { x: bo.x, y: bo.y };
                let mode_decision = rdo_mode_decision(fi, fs, cw, subsize, &offset).part_modes[0].clone();
                child_modes.push(mode_decision);

                let offset = BlockOffset { x: bo.x + hbs as usize, y: bo.y };
                let mode_decision = rdo_mode_decision(fi, fs, cw, subsize, &offset).part_modes[0].clone();
                child_modes.push(mode_decision);

                let offset = BlockOffset { x: bo.x, y: bo.y + hbs as usize };
                let mode_decision = rdo_mode_decision(fi, fs, cw, subsize, &offset).part_modes[0].clone();
                child_modes.push(mode_decision);

                let offset = BlockOffset { x: bo.x + hbs as usize, y: bo.y + hbs as usize };
                let mode_decision = rdo_mode_decision(fi, fs, cw, subsize, &offset).part_modes[0].clone();
                child_modes.push(mode_decision);
            },
            _ => { assert!(false); },
        }

        rd = child_modes.iter().map(|m| m.rd_cost).sum::<f64>();

        if rd < best_rd {
            best_rd = rd;
            best_partition = partition;
            best_pred_modes = child_modes.clone();
        }

        cw.rollback(&checkpoint);
    }

    assert!(best_rd >= 0_f64);

    let rdo_output = RDOOutput { rd_cost: best_rd,
        part_type: best_partition,
        part_modes: best_pred_modes };

    rdo_output
}
