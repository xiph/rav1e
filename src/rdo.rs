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
#![cfg_attr(feature = "cargo-clippy", allow(cast_lossless))]

use context::*;
use ec::OD_BITRES;
use ec::Writer;
use ec::WriterCounter;
use encode_block_a;
use encode_block_b;
use partition::*;
use plane::*;
use cdef::*;
use predict::{RAV1E_INTRA_MODES, RAV1E_INTRA_MODES_MINIMAL, RAV1E_INTER_MODES};
use quantize::dc_q;
use std;
use std::f64;
use std::vec::Vec;
use write_tx_blocks;
use write_tx_tree;
use BlockSize;
use Frame;
use FrameInvariants;
use FrameState;
use FrameType;
use Tune;
use Sequence;

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
  pub pred_mode_luma: PredictionMode,
  pub pred_mode_chroma: PredictionMode,
  pub skip: bool
}

#[allow(unused)]
fn cdef_dist_wxh_8x8(src1: &PlaneSlice, src2: &PlaneSlice) -> u64 {
  //TODO: Handle high bit-depth here by setting coeff_shift
  let coeff_shift = 0;
  let mut sum_s: i32 = 0;
  let mut sum_d: i32 = 0;
  let mut sum_s2: i64 = 0;
  let mut sum_d2: i64 = 0;
  let mut sum_sd: i64 = 0;
  for j in 0..8 {
    for i in 0..8 {
      let s = src1.p(i, j) as i32;
      let d = src2.p(i, j) as i32;
      sum_s += s;
      sum_d += d;
      sum_s2 += (s * s) as i64;
      sum_d2 += (d * d) as i64;
      sum_sd += (s * d) as i64;
    }
  }
  let svar = (sum_s2 - ((sum_s as i64 * sum_s as i64 + 32) >> 6)) as f64;
  let dvar = (sum_d2 - ((sum_d as i64 * sum_d as i64 + 32) >> 6)) as f64;
  let sse = (sum_d2 + sum_s2 - 2 * sum_sd) as f64;
  //The two constants were tuned for CDEF, but can probably be better tuned for use in general RDO
  let ssim_boost = 0.5_f64 * (svar + dvar + (400 << 2 * coeff_shift) as f64)
    / f64::sqrt((20000 << 4 * coeff_shift) as f64 + svar * dvar);
  (sse * ssim_boost + 0.5_f64) as u64
}

#[allow(unused)]
fn cdef_dist_wxh(
  src1: &PlaneSlice, src2: &PlaneSlice, w: usize, h: usize
) -> u64 {
  assert!(w & 0x7 == 0);
  assert!(h & 0x7 == 0);

  let mut sum: u64 = 0;
  for j in 0..h / 8 {
    for i in 0..w / 8 {
      sum += cdef_dist_wxh_8x8(
        &src1.subslice(i * 8, j * 8),
        &src2.subslice(i * 8, j * 8)
      )
    }
  }
  sum
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
fn compute_rd_cost(
  fi: &FrameInvariants, fs: &FrameState, w_y: usize, h_y: usize, w_uv: usize,
  h_uv: usize, bo: &BlockOffset, bit_cost: u32
) -> f64 {
  let q = dc_q(fi.config.quantizer) as f64;

  // Convert q into Q0 precision, given that libaom quantizers are Q3
  let q0 = q / 8.0_f64;

  // Lambda formula from doc/theoretical_results.lyx in the daala repo
  // Use Q0 quantizer since lambda will be applied to Q0 pixel domain
  let lambda = q0 * q0 * std::f64::consts::LN_2 / 6.0;

  // Compute distortion
  let po = bo.plane_offset(&fs.input.planes[0].cfg);
  let mut distortion = if fi.config.tune == Tune::Psnr {
    sse_wxh(
      &fs.input.planes[0].slice(&po),
      &fs.rec.planes[0].slice(&po),
      w_y,
      h_y
    )
  } else if fi.config.tune == Tune::Psychovisual {
    cdef_dist_wxh(
      &fs.input.planes[0].slice(&po),
      &fs.rec.planes[0].slice(&po),
      w_y,
      h_y
    )
  } else {
    unimplemented!();
  };

  // Add chroma distortion only when it is available
  if w_uv > 0 && h_uv > 0 {
    for p in 1..3 {
      let po = bo.plane_offset(&fs.input.planes[p].cfg);

      distortion += sse_wxh(
        &fs.input.planes[p].slice(&po),
        &fs.rec.planes[p].slice(&po),
        w_uv,
        h_uv
      );
    }
  };

  // Compute rate
  let rate = (bit_cost as f64) / ((1 << OD_BITRES) as f64);

  (distortion as f64) + lambda * rate
}

// RDO-based mode decision
pub fn rdo_mode_decision(
  seq: &Sequence, fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter,
  bsize: BlockSize, bo: &BlockOffset) -> RDOOutput {
  let mut best_mode_luma = PredictionMode::DC_PRED;
  let mut best_mode_chroma = PredictionMode::DC_PRED;
  let mut best_skip = false;
  let mut best_rd = std::f64::MAX;

  // Get block luma and chroma dimensions
  let w = bsize.width();
  let h = bsize.height();

  let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;

  let mut w_uv = w >> xdec;
  let mut h_uv = h >> ydec;

  let is_chroma_block = has_chroma(bo, bsize, xdec, ydec);

  if (w_uv == 0 || h_uv == 0) && is_chroma_block {
    w_uv = 4;
    h_uv = 4;
  }

  let skip = false;

  let cw_checkpoint = cw.checkpoint();

  // Exclude complex prediction modes at higher speed levels
  let mode_set = if fi.config.speed <= 3 {
    (if fi.frame_type == FrameType::INTER { RAV1E_INTER_MODES }
      else { RAV1E_INTRA_MODES })
  } else {
    (if fi.frame_type == FrameType::INTER { RAV1E_INTER_MODES }
    else { RAV1E_INTRA_MODES_MINIMAL })
  };

  for &luma_mode in mode_set {
    assert!(fi.frame_type == FrameType::INTER || luma_mode.is_intra());

    if is_chroma_block && fi.config.speed <= 3 && luma_mode.is_intra() {
      // Find the best chroma prediction mode for the current luma prediction mode
      for &chroma_mode in RAV1E_INTRA_MODES {
        let mut wr: &mut Writer = &mut WriterCounter::new();
        let tell = wr.tell_frac();
        
        encode_block_a(seq, cw, wr, bsize, bo, skip);
        encode_block_b(fi, fs, cw, wr, luma_mode, chroma_mode, bsize, bo, skip);

        let cost = wr.tell_frac() - tell;
        let rd = compute_rd_cost(
          fi,
          fs,
          w,
          h,
          w_uv,
          h_uv,
          bo,
          cost
        );

        if rd < best_rd {
          best_rd = rd;
          best_mode_luma = luma_mode;
          best_mode_chroma = chroma_mode;
          best_skip = skip;
        }

        cw.rollback(&cw_checkpoint);
      }
    } else {
      let mut wr: &mut Writer = &mut WriterCounter::new();
      let tell = wr.tell_frac();
      encode_block_a(seq, cw, wr, bsize, bo, skip);
      encode_block_b(fi, fs, cw, wr, luma_mode, luma_mode, bsize, bo, skip);

      let cost = wr.tell_frac() - tell;
      let rd = compute_rd_cost(
        fi,
        fs,
        w,
        h,
        w_uv,
        h_uv,
        bo,
        cost
      );

      if rd < best_rd {
        best_rd = rd;
        best_mode_luma = luma_mode;
        best_mode_chroma = luma_mode;
        best_skip = skip;
      }

      cw.rollback(&cw_checkpoint);
    }
  }

  assert!(best_rd >= 0_f64);

  RDOOutput {
    rd_cost: best_rd,
    part_type: PartitionType::PARTITION_NONE,
    part_modes: vec![RDOPartitionOutput {
      bo: bo.clone(),
      pred_mode_luma: best_mode_luma,
      pred_mode_chroma: best_mode_chroma,
      rd_cost: best_rd,
      skip: best_skip
    }]
  }
}

// RDO-based intra frame transform type decision
pub fn rdo_tx_type_decision(
  fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter,
  mode: PredictionMode, bsize: BlockSize, bo: &BlockOffset, tx_size: TxSize,
  tx_set: TxSet
) -> TxType {
  let mut best_type = TxType::DCT_DCT;
  let mut best_rd = std::f64::MAX;

  // Get block luma and chroma dimensions
  let w = bsize.width();
  let h = bsize.height();

  let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;

  let mut w_uv = w >> xdec;
  let mut h_uv = h >> ydec;

  if (w_uv == 0 || h_uv == 0) && has_chroma(bo, bsize, xdec, ydec) {
    w_uv = 4;
    h_uv = 4;
  }

  let is_inter = mode >= PredictionMode::NEARESTMV;

  let cw_checkpoint = cw.checkpoint();

  for &tx_type in RAV1E_TX_TYPES {
    // Skip unsupported transform types
    if av1_tx_used[tx_set as usize][tx_type as usize] == 0 {
      continue;
    }

    let mut wr: &mut Writer = &mut WriterCounter::new();
    let tell = wr.tell_frac();
    if is_inter {
      write_tx_tree(
        fi, fs, cw, wr, mode, mode, bo, bsize, tx_size, tx_type, false,
      );
    }  else {
      write_tx_blocks(
        fi, fs, cw, wr, mode, mode, bo, bsize, tx_size, tx_type, false,
      );
    }

    let cost = wr.tell_frac() - tell;
    let rd = compute_rd_cost(
      fi,
      fs,
      w,
      h,
      w_uv,
      h_uv,
      bo,
      cost
    );

    if rd < best_rd {
      best_rd = rd;
      best_type = tx_type;
    }

    cw.rollback(&cw_checkpoint);
  }

  assert!(best_rd >= 0_f64);

  best_type
}

// RDO-based single level partitioning decision
pub fn rdo_partition_decision(
  seq: &Sequence, fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter,
  bsize: BlockSize, bo: &BlockOffset, cached_block: &RDOOutput) -> RDOOutput {
  let max_rd = std::f64::MAX;

  let mut best_partition = cached_block.part_type;
  let mut best_rd = cached_block.rd_cost;
  let mut best_pred_modes = cached_block.part_modes.clone();

  let cw_checkpoint = cw.checkpoint();

  for &partition in RAV1E_PARTITION_TYPES {
    // Do not re-encode results we already have
    if partition == cached_block.part_type && cached_block.rd_cost < max_rd {
      continue;
    }

    let mut rd: f64;
    let mut child_modes = std::vec::Vec::new();

    match partition {
      PartitionType::PARTITION_NONE => {
        if bsize > BlockSize::BLOCK_32X32 {
          continue;
        }

        let mode_decision = cached_block
          .part_modes
          .get(0)
          .unwrap_or(&rdo_mode_decision(seq, fi, fs, cw, bsize, bo).part_modes[0])
          .clone();
        child_modes.push(mode_decision);
      }
      PartitionType::PARTITION_SPLIT => {
        let subsize = get_subsize(bsize, partition);

        if subsize == BlockSize::BLOCK_INVALID {
          continue;
        }

        let bs = bsize.width_mi();
        let hbs = bs >> 1; // Half the block size in blocks

        let offset = BlockOffset { x: bo.x, y: bo.y };
        let mode_decision = rdo_mode_decision(seq, fi, fs, cw, subsize, &offset)
          .part_modes[0]
          .clone();
        child_modes.push(mode_decision);

        let offset = BlockOffset { x: bo.x + hbs as usize, y: bo.y };
        let mode_decision = rdo_mode_decision(seq, fi, fs, cw, subsize, &offset)
          .part_modes[0]
          .clone();
        child_modes.push(mode_decision);

        let offset = BlockOffset { x: bo.x, y: bo.y + hbs as usize };
        let mode_decision = rdo_mode_decision(seq, fi, fs, cw, subsize, &offset)
          .part_modes[0]
          .clone();
        child_modes.push(mode_decision);

        let offset =
          BlockOffset { x: bo.x + hbs as usize, y: bo.y + hbs as usize };
        let mode_decision = rdo_mode_decision(seq, fi, fs, cw, subsize, &offset)
          .part_modes[0]
          .clone();
        child_modes.push(mode_decision);
      }
      _ => {
        assert!(false);
      }
    }

    rd = child_modes.iter().map(|m| m.rd_cost).sum::<f64>();

    if rd < best_rd {
      best_rd = rd;
      best_partition = partition;
      best_pred_modes = child_modes.clone();
    }

    cw.rollback(&cw_checkpoint);
  }

  assert!(best_rd >= 0_f64);

  RDOOutput {
    rd_cost: best_rd,
    part_type: best_partition,
    part_modes: best_pred_modes
  }
}

pub fn rdo_cdef_decision(sbo: &SuperBlockOffset, fi: &FrameInvariants,
                         fs: &FrameState, cw: &mut ContextWriter, bit_depth: usize) -> u8 {
    // FIXME: 128x128 SB support will break this, we need FilterBlockOffset etc.
    // Construct a single-superblock-sized frame to test-filter into
    let sbo_0 = SuperBlockOffset { x: 0, y: 0 };
    let bc = &mut cw.bc;
    let mut cdef_output = Frame {
        planes: [
            Plane::new(64 >> fs.rec.planes[0].cfg.xdec, 64 >> fs.rec.planes[0].cfg.ydec,
                       fs.rec.planes[0].cfg.xdec, fs.rec.planes[0].cfg.ydec),
            Plane::new(64 >> fs.rec.planes[1].cfg.xdec, 64 >> fs.rec.planes[1].cfg.ydec,
                       fs.rec.planes[1].cfg.xdec, fs.rec.planes[1].cfg.ydec),
            Plane::new(64 >> fs.rec.planes[2].cfg.xdec, 64 >> fs.rec.planes[2].cfg.ydec,
                       fs.rec.planes[2].cfg.xdec, fs.rec.planes[2].cfg.ydec),
        ]
    };
    // Construct a padded input
    let mut rec_input = Frame {
        planes: [
            Plane::new((64 >> fs.rec.planes[0].cfg.xdec)+4, (64 >> fs.rec.planes[0].cfg.ydec)+4,
                       fs.rec.planes[0].cfg.xdec, fs.rec.planes[0].cfg.ydec),
            Plane::new((64 >> fs.rec.planes[1].cfg.xdec)+4, (64 >> fs.rec.planes[1].cfg.ydec)+4,
                       fs.rec.planes[1].cfg.xdec, fs.rec.planes[1].cfg.ydec),
            Plane::new((64 >> fs.rec.planes[2].cfg.xdec)+4, (64 >> fs.rec.planes[2].cfg.ydec)+4,
                       fs.rec.planes[2].cfg.xdec, fs.rec.planes[2].cfg.ydec),
        ]
    };
    // Copy reconstructed data into padded input
    for p in 0..3 {
        let xdec = fs.rec.planes[p].cfg.xdec;
        let ydec = fs.rec.planes[p].cfg.ydec;
        let h = fi.padded_h >> ydec;
        let w = fi.padded_w >> xdec;
        let offset = sbo.plane_offset(&fs.rec.planes[p].cfg);
        for y in 0..(64>>ydec)+4 {
            let mut rec_slice = rec_input.planes[p].mut_slice(&PlaneOffset {x:0, y:y});
            let mut rec_row = rec_slice.as_mut_slice();
            if offset.y+y < 2 || offset.y+y >= h+2 {
                // above or below the frame, fill with flag
                for x in 0..(64>>xdec)+4 { rec_row[x] = CDEF_VERY_LARGE; }
            } else {
                let mut in_slice = fs.rec.planes[p].slice(&PlaneOffset {x:0, y:offset.y+y-2});
                let mut in_row = in_slice.as_slice();
                // are we guaranteed to be all in frame this row?
                if offset.x < 2 || offset.x+(64>>xdec)+2 >= w {
                    // No; do it the hard way.  off left or right edge, fill with flag.
                    for x in 0..(64>>xdec)+4 {
                        if offset.x+x >= 2 && offset.x+x < w+2 {
                            rec_row[x] = in_row[offset.x+x-2]
                        } else {
                            rec_row[x] = CDEF_VERY_LARGE;
                        }
                    }
                }  else  {
                    // Yes, do it the easy way: just copy
                    rec_row[0..(64>>xdec)+4].copy_from_slice(&in_row[offset.x-2..offset.x+(64>>xdec)+2]);
                }
            }
        }
    }

    // RDO comparisons
    let mut best_index: u8 = 0;
    let mut best_err: u64 = 0;
    for cdef_index in 0..(1<<fi.cdef_bits) {
        //for p in 0..3 {
        //    for i in 0..cdef_output.planes[p].data.len() { cdef_output.planes[p].data[i] = CDEF_VERY_LARGE; }
        //}
        // TODO: Don't repeat find_direction over and over; split filter_superblock to run it separately
        cdef_filter_superblock(fi, &mut rec_input, &mut cdef_output, bc, &sbo_0, &sbo, bit_depth, cdef_index);


        // Rate is constant, compute just distortion
        // Computation is block by block, paying attention to skip flag

        // Each direction block is 8x8 in y, potentially smaller if subsampled in chroma
        // We're dealing only with in-frmae and unpadded planes now
        let mut err:u64 = 0;
        for by in 0..8 {
            for bx in 0..8 {
                let bo = sbo.block_offset(bx, by);
                if bo.x+bx < bc.cols && bo.y+by < bc.rows {
                    let skip = bc.at(&bo).skip;
                    if !skip {
                        for p in 0..3 {
                            let mut in_plane = &fs.input.planes[p];
                            let in_po = sbo.block_offset(bx<<1, by<<1).plane_offset(&in_plane.cfg);
                            let in_slice = in_plane.slice(&in_po);

                            let mut out_plane = &mut cdef_output.planes[p];
                            let out_po = sbo_0.block_offset(bx<<1, by<<1).plane_offset(&out_plane.cfg);
                            let out_slice = &out_plane.slice(&out_po);
                            
                            let xdec = in_plane.cfg.xdec;
                            let ydec = in_plane.cfg.ydec;

                            if p==0 {
                                err += cdef_dist_wxh_8x8(&in_slice, &out_slice);
                            } else {
                                err += sse_wxh(&in_slice, &out_slice, 8>>xdec, 8>>ydec);
                            }
                        }
                    }
                }
            }
        }

        if cdef_index == 0 || err < best_err {
            best_err = err;
            best_index = cdef_index;
        }
        
    }
    best_index
}

