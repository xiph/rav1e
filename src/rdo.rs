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

use api::PredictionModesSetting;
use cdef::*;
use context::*;
use ec::{OD_BITRES, Writer, WriterCounter};
use encoder::{ChromaSampling, ReferenceMode};
use encode_block_a;
use encode_block_b;
use encode_block_with_modes;
use Frame;
use FrameInvariants;
use FrameState;
use FrameType;
use luma_ac;
use me::*;
use motion_compensate;
use partition::*;
use plane::*;
use predict::{RAV1E_INTRA_MODES, RAV1E_INTER_MODES_MINIMAL, RAV1E_INTER_COMPOUND_MODES};
use quantize::dc_q;
use Tune;
use write_tx_blocks;
use write_tx_tree;

use std;
use std::vec::Vec;
use partition::PartitionType::*;

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
  pub bsize: BlockSize,
  pub pred_mode_luma: PredictionMode,
  pub pred_mode_chroma: PredictionMode,
  pub pred_cfl_params: CFLParams,
  pub ref_frames: [usize; 2],
  pub mvs: [MotionVector; 2],
  pub skip: bool,
  pub tx_size: TxSize,
  pub tx_type: TxType,
}

#[allow(unused)]
fn cdef_dist_wxh_8x8(
  src1: &PlaneSlice<'_>, src2: &PlaneSlice<'_>, bit_depth: usize
) -> u64 {
  let coeff_shift = bit_depth - 8;

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
  src1: &PlaneSlice<'_>, src2: &PlaneSlice<'_>, w: usize, h: usize,
  bit_depth: usize
) -> u64 {
  assert!(w & 0x7 == 0);
  assert!(h & 0x7 == 0);

  let mut sum: u64 = 0;
  for j in 0..h / 8 {
    for i in 0..w / 8 {
      sum += cdef_dist_wxh_8x8(
        &src1.subslice(i * 8, j * 8),
        &src2.subslice(i * 8, j * 8),
        bit_depth
      )
    }
  }
  sum
}

// Sum of Squared Error for a wxh block
pub fn sse_wxh(
  src1: &PlaneSlice<'_>, src2: &PlaneSlice<'_>, w: usize, h: usize
) -> u64 {
  assert!(w & (MI_SIZE - 1) == 0);
  assert!(h & (MI_SIZE - 1) == 0);

  let mut sse: u64 = 0;
  for j in 0..h {
    let src1j = src1.subslice(0, j);
    let src2j = src2.subslice(0, j);
    let s1 = src1j.as_slice_w_width(w);
    let s2 = src2j.as_slice_w_width(w);

    let row_sse = s1
      .iter()
      .zip(s2)
      .map(|(&a, &b)| {
        let c = (a as i16 - b as i16) as i32;
        (c * c) as u32
      }).sum::<u32>();
    sse += row_sse as u64;
  }
  sse
}

pub fn get_lambda(fi: &FrameInvariants) -> f64 {
  let q = dc_q(fi.base_q_idx, fi.dc_delta_q[0], fi.sequence.bit_depth) as f64;

  // Convert q into Q0 precision, given that libaom quantizers are Q3
  let q0 = q / 8.0_f64;

  // Lambda formula from doc/theoretical_results.lyx in the daala repo
  // Use Q0 quantizer since lambda will be applied to Q0 pixel domain
  q0 * q0 * std::f64::consts::LN_2 / 6.0
}

pub fn get_lambda_sqrt(fi: &FrameInvariants) -> f64 {
  let q = dc_q(fi.base_q_idx, fi.dc_delta_q[0], fi.sequence.bit_depth) as f64;

  // Convert q into Q0 precision, given that libaom quantizers are Q3
  let q0 = q / 8.0_f64;

  // Lambda formula from doc/theoretical_results.lyx in the daala repo
  // Use Q0 quantizer since lambda will be applied to Q0 pixel domain
  q0 * (std::f64::consts::LN_2 / 6.0).sqrt()
}

// Compute the rate-distortion cost for an encode
fn compute_rd_cost(
  fi: &FrameInvariants, fs: &FrameState, w_y: usize, h_y: usize,
  is_chroma_block: bool, bo: &BlockOffset, bit_cost: u32,
  luma_only: bool
) -> f64 {
  let lambda = get_lambda(fi);

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
      h_y,
      fi.sequence.bit_depth
    )
  } else {
    unimplemented!();
  };

  if !luma_only {
  let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;

  let mask = !(MI_SIZE - 1);
  let mut w_uv = (w_y >> xdec) & mask;
  let mut h_uv = (h_y >> ydec) & mask;

  if (w_uv == 0 || h_uv == 0) && is_chroma_block {
    w_uv = MI_SIZE;
    h_uv = MI_SIZE;
  }

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
  }
  // Compute rate
  let rate = (bit_cost as f64) / ((1 << OD_BITRES) as f64);

  (distortion as f64) + lambda * rate
}

// Compute the rate-distortion cost for an encode
fn compute_tx_rd_cost(
  fi: &FrameInvariants, fs: &FrameState, w_y: usize, h_y: usize,
  is_chroma_block: bool, bo: &BlockOffset, bit_cost: u32, tx_dist: i64,
  skip: bool, luma_only: bool
) -> f64 {
  assert!(fi.config.tune == Tune::Psnr);

  let lambda = get_lambda(fi);

  // Compute distortion
  let mut distortion = if skip {
    let po = bo.plane_offset(&fs.input.planes[0].cfg);

    sse_wxh(
      &fs.input.planes[0].slice(&po),
      &fs.rec.planes[0].slice(&po),
      w_y,
      h_y
    )
  } else {
    assert!(tx_dist >= 0);
    tx_dist as u64
  };

  if !luma_only && skip {
    let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;

    let mask = !(MI_SIZE - 1);
    let mut w_uv = (w_y >> xdec) & mask;
    let mut h_uv = (h_y >> ydec) & mask;

    if (w_uv == 0 || h_uv == 0) && is_chroma_block {
      w_uv = MI_SIZE;
      h_uv = MI_SIZE;
    }

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
    }
  }
  // Compute rate
  let rate = (bit_cost as f64) / ((1 << OD_BITRES) as f64);

  (distortion as f64) + lambda * rate
}

pub fn rdo_tx_size_type(
  fi: &FrameInvariants, fs: &mut FrameState,
  cw: &mut ContextWriter, bsize: BlockSize, bo: &BlockOffset,
  luma_mode: PredictionMode, ref_frames: [usize; 2], mvs: [MotionVector; 2], skip: bool
) -> (TxSize, TxType) {
  // these rules follow TX_MODE_LARGEST
  let tx_size = match bsize {
    BlockSize::BLOCK_4X4 => TxSize::TX_4X4,
    BlockSize::BLOCK_8X8 => TxSize::TX_8X8,
    BlockSize::BLOCK_16X16 => TxSize::TX_16X16,
    BlockSize::BLOCK_4X8 => TxSize::TX_4X8,
    BlockSize::BLOCK_8X4 => TxSize::TX_8X4,
    BlockSize::BLOCK_8X16 => TxSize::TX_8X16,
    BlockSize::BLOCK_16X8 => TxSize::TX_16X8,
    BlockSize::BLOCK_16X32 => TxSize::TX_16X32,
    BlockSize::BLOCK_32X16 => TxSize::TX_32X16,
    BlockSize::BLOCK_32X32 => TxSize::TX_32X32,
    BlockSize::BLOCK_32X64 => TxSize::TX_32X64,
    BlockSize::BLOCK_64X32 => TxSize::TX_64X32,
    BlockSize::BLOCK_64X64 => TxSize::TX_64X64,
    _ => unimplemented!()
  };
  cw.bc.set_tx_size(bo, tx_size);
  // Were we not hardcoded to TX_MODE_LARGEST, block tx size would be written here

  // Luma plane transform type decision
  let is_inter = !luma_mode.is_intra();
  let tx_set = get_tx_set(tx_size, is_inter, fi.use_reduced_tx_set);

  let tx_type =
    if tx_set > TxSet::TX_SET_DCTONLY && fi.config.speed_settings.rdo_tx_decision && !skip {
      rdo_tx_type_decision(
        fi,
        fs,
        cw,
        luma_mode,
        ref_frames,
        mvs,
        bsize,
        bo,
        tx_size,
        tx_set,
      )
    } else {
      TxType::DCT_DCT
    };

  assert!(tx_size.sqr() <= TxSize::TX_32X32 || tx_type == TxType::DCT_DCT);

  (tx_size, tx_type)
}

struct EncodingSettings {
  mode_luma: PredictionMode,
  mode_chroma: PredictionMode,
  cfl_params: CFLParams,
  skip: bool,
  rd: f64,
  ref_frames: [usize; 2],
  mvs: [MotionVector; 2],
  tx_size: TxSize,
  tx_type: TxType
}

impl Default for EncodingSettings {
  fn default() -> Self {
    EncodingSettings {
      mode_luma: PredictionMode::DC_PRED,
      mode_chroma: PredictionMode::DC_PRED,
      cfl_params: CFLParams::new(),
      skip: false,
      rd: std::f64::MAX,
      ref_frames: [INTRA_FRAME, NONE_FRAME],
      mvs: [MotionVector { row: 0, col: 0 }; 2],
      tx_size: TxSize::TX_4X4,
      tx_type: TxType::DCT_DCT
    }
  }
}
// RDO-based mode decision
pub fn rdo_mode_decision(fi: &FrameInvariants, fs: &mut FrameState,
  cw: &mut ContextWriter, bsize: BlockSize, bo: &BlockOffset,
  pmvs: &[Option<MotionVector>], needs_rec: bool
) -> RDOOutput {
  let mut best = EncodingSettings::default();

  // Get block luma and chroma dimensions
  let w = bsize.width();
  let h = bsize.height();

  let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;
  let is_chroma_block = has_chroma(bo, bsize, xdec, ydec);

  let cw_checkpoint = cw.checkpoint();

  let mut ref_frames_set = Vec::new();
  let mut ref_slot_set = Vec::new();
  let mut mvs_from_me = Vec::new();
  let mut fwdref = None;
  let mut bwdref = None;

  if fi.frame_type == FrameType::INTER {
    for i in LAST_FRAME..NONE_FRAME {
      // Don't search LAST3 since it's used only for probs
      if i == LAST3_FRAME { continue; }
      if !ref_slot_set.contains(&fi.ref_frames[i - LAST_FRAME]) {
        if fwdref == None && i < BWDREF_FRAME {
          fwdref = Some(ref_frames_set.len());
        }
        if bwdref == None && i >= BWDREF_FRAME {
          bwdref = Some(ref_frames_set.len());
        }
        ref_frames_set.push([i, NONE_FRAME]);
        let slot_idx = fi.ref_frames[i - LAST_FRAME];
        ref_slot_set.push(slot_idx);
      }
    }
    assert!(ref_frames_set.len() != 0);
  }

  let mut mode_set: Vec<(PredictionMode, usize)> = Vec::new();
  let mut mv_stacks = Vec::new();
  let mut mode_contexts = Vec::new();

  for (i, &ref_frames) in ref_frames_set.iter().enumerate() {
    let mut mv_stack: Vec<CandidateMV> = Vec::new();
    mode_contexts.push(cw.find_mvrefs(bo, ref_frames, &mut mv_stack, bsize, fi, false));

    if fi.frame_type == FrameType::INTER {
      let mut pmv = [MotionVector{ row: 0, col: 0 }; 2];
      if mv_stack.len() > 0 { pmv[0] = mv_stack[0].this_mv; }
      if mv_stack.len() > 1 { pmv[1] = mv_stack[1].this_mv; }
      let cmv = pmvs[ref_slot_set[i] as usize].unwrap();
      mvs_from_me.push([
        motion_estimation(fi, fs, bsize, bo, ref_frames[0], cmv, &pmv),
        MotionVector { row: 0, col: 0 }
      ]);

      for &x in RAV1E_INTER_MODES_MINIMAL {
        mode_set.push((x, i));
      }
      if mv_stack.len() >= 1 {
        mode_set.push((PredictionMode::NEAR0MV, i));
      }
      if mv_stack.len() >= 2 {
        mode_set.push((PredictionMode::GLOBALMV, i));
      }
      let include_near_mvs = fi.config.speed_settings.include_near_mvs;
      if include_near_mvs {
        if mv_stack.len() >= 3 {
          mode_set.push((PredictionMode::NEAR1MV, i));
        }
        if mv_stack.len() >= 4 {
          mode_set.push((PredictionMode::NEAR2MV, i));
        }
      }
      if !mv_stack.iter().take(if include_near_mvs {4} else {2})
        .any(|ref x| x.this_mv.row == mvs_from_me[i][0].row && x.this_mv.col == mvs_from_me[i][0].col)
        && (mvs_from_me[i][0].row != 0 || mvs_from_me[i][0].col != 0) {
        mode_set.push((PredictionMode::NEWMV, i));
      }
    }
    mv_stacks.push(mv_stack);
  }

  let sz = bsize.width_mi().min(bsize.height_mi());

  if fi.frame_type == FrameType::INTER && fi.reference_mode != ReferenceMode::SINGLE && sz >= 2 {
    // Adding compound candidate
    if let Some(r0) = fwdref {
      if let Some(r1) = bwdref {
        let ref_frames = [ref_frames_set[r0][0], ref_frames_set[r1][0]];
        ref_frames_set.push(ref_frames);
        let mv0 = mvs_from_me[r0][0];
        let mv1 = mvs_from_me[r1][0];
        mvs_from_me.push([mv0, mv1]);
        let mut mv_stack: Vec<CandidateMV> = Vec::new();
        mode_contexts.push(cw.find_mvrefs(bo, ref_frames, &mut mv_stack, bsize, fi, true));
        for &x in RAV1E_INTER_COMPOUND_MODES {
          mode_set.push((x, ref_frames_set.len() - 1));
        }
        mv_stacks.push(mv_stack);
      }
    }
  }

  let luma_rdo = |luma_mode: PredictionMode, fs: &mut FrameState, cw: &mut ContextWriter, best: &mut EncodingSettings,
    mvs: [MotionVector; 2], ref_frames: [usize; 2], mode_set_chroma: &[PredictionMode], luma_mode_is_intra: bool,
    mode_context: usize, mv_stack: &Vec<CandidateMV>| {
    let (tx_size, mut tx_type) = rdo_tx_size_type(
        fi, fs, cw, bsize, bo, luma_mode, ref_frames, mvs, false,
    );

    // Find the best chroma prediction mode for the current luma prediction mode
    let mut chroma_rdo = |skip: bool| {
      mode_set_chroma.iter().for_each(|&chroma_mode| {
        let wr: &mut dyn Writer = &mut WriterCounter::new();
        let tell = wr.tell_frac();

        if skip { tx_type = TxType::DCT_DCT; };

        if bsize >= BlockSize::BLOCK_8X8 && bsize.is_sqr() {
          cw.write_partition(wr, bo, PartitionType::PARTITION_NONE, bsize);
        }

        encode_block_a(&fi.sequence, fs, cw, wr, bsize, bo, skip);
        let tx_dist =
        encode_block_b(
          fi,
          fs,
          cw,
          wr,
          luma_mode,
          chroma_mode,
          ref_frames,
          mvs,
          bsize,
          bo,
          skip,
          CFLParams::new(),
          tx_size,
          tx_type,
          mode_context,
          mv_stack,
          !needs_rec
        );

        let cost = wr.tell_frac() - tell;
        let rd = if fi.use_tx_domain_distortion && !needs_rec {
          compute_tx_rd_cost(
            fi,
            fs,
            w,
            h,
            is_chroma_block,
            bo,
            cost,
            tx_dist,
            skip,
            false
          )
        } else {
          compute_rd_cost(
            fi,
            fs,
            w,
            h,
            is_chroma_block,
            bo,
            cost,
            false
          )
        };
        if rd < best.rd {
        //if rd < best.rd || luma_mode == PredictionMode::NEW_NEWMV {
          best.rd = rd;
          best.mode_luma = luma_mode;
          best.mode_chroma = chroma_mode;
          best.ref_frames = ref_frames;
          best.mvs = mvs;
          best.skip = skip;
          best.tx_size = tx_size;
          best.tx_type = tx_type;
        }

        cw.rollback(&cw_checkpoint);
      });
    };

    chroma_rdo(false);
    // Don't skip when using intra modes
    if !luma_mode_is_intra {
        chroma_rdo(true);
    };
  };

  if fi.frame_type != FrameType::INTER {
    assert!(mode_set.len() == 0);
  }

  mode_set.iter().for_each(|&(luma_mode, i)| {
    let mvs = match luma_mode {
      PredictionMode::NEWMV | PredictionMode::NEW_NEWMV => mvs_from_me[i],
      PredictionMode::NEARESTMV | PredictionMode::NEAREST_NEARESTMV => if mv_stacks[i].len() > 0 {
        [mv_stacks[i][0].this_mv, mv_stacks[i][0].comp_mv]
      } else {
        [MotionVector { row: 0, col: 0 }; 2]
      },
      PredictionMode::NEAR0MV => if mv_stacks[i].len() > 1 {
        [mv_stacks[i][1].this_mv, mv_stacks[i][1].comp_mv]
      } else {
        [MotionVector { row: 0, col: 0 }; 2]
      },
      PredictionMode::NEAR1MV | PredictionMode::NEAR2MV =>
          [mv_stacks[i][luma_mode as usize - PredictionMode::NEAR0MV as usize + 1].this_mv,
          mv_stacks[i][luma_mode as usize - PredictionMode::NEAR0MV as usize + 1].comp_mv],
      PredictionMode::NEAREST_NEWMV => [mv_stacks[i][0].this_mv, mvs_from_me[i][1]],
      PredictionMode::NEW_NEARESTMV => [mvs_from_me[i][0], mv_stacks[i][0].comp_mv],
      _ => [MotionVector { row: 0, col: 0 }; 2]
    };
    let mode_set_chroma = vec![luma_mode];

    luma_rdo(luma_mode, fs, cw, &mut best, mvs, ref_frames_set[i], &mode_set_chroma, false,
             mode_contexts[i], &mv_stacks[i]);
  });

  if !best.skip {
    let tx_size = bsize.tx_size();

    // Reduce number of prediction modes at higher speed levels
    let num_modes_rdo = if (fi.frame_type == FrameType::KEY
      && fi.config.speed_settings.prediction_modes >= PredictionModesSetting::ComplexKeyframes)
      || (fi.frame_type == FrameType::INTER && fi.config.speed_settings.prediction_modes >= PredictionModesSetting::ComplexAll)
    {
      7
    } else {
      3
    };

    let intra_mode_set = RAV1E_INTRA_MODES;
    let mut sads = {
      let edge_buf = {
        let rec = &mut fs.rec.planes[0];
        let po = bo.plane_offset(&rec.cfg);
        get_intra_edges(&rec.slice(&po), tx_size, fi.sequence.bit_depth, 0, fi.w_in_b, fi.h_in_b, None)
      };
      intra_mode_set.iter().map(|&luma_mode| {
        let rec = &mut fs.rec.planes[0];
        let po = bo.plane_offset(&rec.cfg);
        luma_mode.predict_intra(&mut rec.mut_slice(&po), tx_size, fi.sequence.bit_depth, &[0i16; 2], 0, &edge_buf);

        let plane_org = fs.input.planes[0].slice(&po);
        let plane_ref = rec.slice(&po);

        (luma_mode, get_sad(&plane_org, &plane_ref, tx_size.height(), tx_size.width(), fi.sequence.bit_depth))
      }).collect::<Vec<_>>()
    };

    sads.sort_by_key(|a| a.1);

    // Find mode with lowest rate cost
    let mut z = 32768;
    let probs_all = if fi.frame_type == FrameType::INTER {
      cw.get_cdf_intra_mode(bsize)
    } else {
      cw.get_cdf_intra_mode_kf(bo)
    }.iter().take(INTRA_MODES).map(|&a| { let d = z - a; z = a; d }).collect::<Vec<_>>();


    let mut probs = intra_mode_set.iter().map(|&a| (a, probs_all[a as usize])).collect::<Vec<_>>();
    probs.sort_by_key(|a| !a.1);

    let mut modes = Vec::new();
    probs.iter().take(num_modes_rdo / 2).for_each(|&(luma_mode, _prob)| modes.push(luma_mode));
    sads.iter().take(num_modes_rdo).for_each(|&(luma_mode, _sad)| if !modes.contains(&luma_mode) { modes.push(luma_mode) } );

    modes.iter().take(num_modes_rdo).for_each(|&luma_mode| {
      let mvs = [MotionVector { row: 0, col: 0 }; 2];
      let ref_frames = [INTRA_FRAME, NONE_FRAME];
      let mut mode_set_chroma = vec![luma_mode];
      if is_chroma_block && luma_mode != PredictionMode::DC_PRED {
        mode_set_chroma.push(PredictionMode::DC_PRED);
      }
      luma_rdo(luma_mode, fs, cw, &mut best, mvs, ref_frames, &mode_set_chroma, true,
               0, &Vec::new());
    });
  }

  if best.mode_luma.is_intra() && is_chroma_block && bsize.cfl_allowed() {
    let chroma_mode = PredictionMode::UV_CFL_PRED;
    let cw_checkpoint = cw.checkpoint();
    let wr: &mut dyn Writer = &mut WriterCounter::new();
    write_tx_blocks(
      fi,
      fs,
      cw,
      wr,
      best.mode_luma,
      best.mode_luma,
      bo,
      bsize,
      best.tx_size,
      best.tx_type,
      false,
      CFLParams::new(),
      true,
      false
    );
    cw.rollback(&cw_checkpoint);
    if let Some(cfl) = rdo_cfl_alpha(fs, bo, bsize, fi.sequence.bit_depth, fi.sequence.chroma_sampling) {
      let mut wr: &mut dyn Writer = &mut WriterCounter::new();
      let tell = wr.tell_frac();

      encode_block_a(&fi.sequence, fs, cw, wr, bsize, bo, best.skip);
      encode_block_b(
        fi,
        fs,
        cw,
        wr,
        best.mode_luma,
        chroma_mode,
        best.ref_frames,
        best.mvs,
        bsize,
        bo,
        best.skip,
        cfl,
        best.tx_size,
        best.tx_type,
        0,
        &Vec::new(),
        false // For CFL, luma should be always reconstructed.
      );

      let cost = wr.tell_frac() - tell;

      // For CFL, tx-domain distortion is not an option.
      let rd =
        compute_rd_cost(
          fi,
          fs,
          w,
          h,
          is_chroma_block,
          bo,
          cost,
          false
        );

      if rd < best.rd {
        best.rd = rd;
        best.mode_chroma = chroma_mode;
        best.cfl_params = cfl;
      }

      cw.rollback(&cw_checkpoint);
    }
  }

  cw.bc.set_mode(bo, bsize, best.mode_luma);
  cw.bc.set_ref_frames(bo, bsize, best.ref_frames);
  cw.bc.set_motion_vectors(bo, bsize, best.mvs);

  assert!(best.rd >= 0_f64);

  RDOOutput {
    rd_cost: best.rd,
    part_type: PartitionType::PARTITION_NONE,
    part_modes: vec![RDOPartitionOutput {
      bo: bo.clone(),
      bsize: bsize,
      pred_mode_luma: best.mode_luma,
      pred_mode_chroma: best.mode_chroma,
      pred_cfl_params: best.cfl_params,
      ref_frames: best.ref_frames,
      mvs: best.mvs,
      rd_cost: best.rd,
      skip: best.skip,
      tx_size: best.tx_size,
      tx_type: best.tx_type,
    }]
  }
}

pub fn rdo_cfl_alpha(
  fs: &mut FrameState, bo: &BlockOffset, bsize: BlockSize, bit_depth: usize,
  chroma_sampling: ChromaSampling) -> Option<CFLParams> {
  let uv_tx_size = bsize.largest_uv_tx_size(chroma_sampling);

  let mut ac = [0i16; 32 * 32];
  luma_ac(&mut ac, fs, bo, bsize);
  let best_alpha: Vec<i16> = (1..3)
    .map(|p| {
      let rec = &mut fs.rec.planes[p];
      let input = &fs.input.planes[p];
      let po = bo.plane_offset(&fs.input.planes[p].cfg);
      (-16i16..17i16)
        .min_by_key(|&alpha| {
          let edge_buf = get_intra_edges(&rec.slice(&po), uv_tx_size, bit_depth, p, 0, 0, Some(PredictionMode::UV_CFL_PRED));
          PredictionMode::UV_CFL_PRED.predict_intra(
            &mut rec.mut_slice(&po),
            uv_tx_size,
            bit_depth,
            &ac,
            alpha,
            &edge_buf
          );
          sse_wxh(
            &input.slice(&po),
            &rec.slice(&po),
            uv_tx_size.width(),
            uv_tx_size.height()
          )
        }).unwrap()
    }).collect();

  if best_alpha[0] == 0 && best_alpha[1] == 0 {
    None
  } else {
    Some(CFLParams::from_alpha(best_alpha[0], best_alpha[1]))
  }
}

// RDO-based transform type decision
pub fn rdo_tx_type_decision(
  fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter,
  mode: PredictionMode, ref_frames: [usize; 2], mvs: [MotionVector; 2], bsize: BlockSize, bo: &BlockOffset, tx_size: TxSize,
  tx_set: TxSet) -> TxType {
  let mut best_type = TxType::DCT_DCT;
  let mut best_rd = std::f64::MAX;

  // Get block luma and chroma dimensions
  let w = bsize.width();
  let h = bsize.height();

  let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;
  let is_chroma_block = has_chroma(bo, bsize, xdec, ydec);

  let is_inter = !mode.is_intra();

  let cw_checkpoint = cw.checkpoint();

  for &tx_type in RAV1E_TX_TYPES {
    // Skip unsupported transform types
    if av1_tx_used[tx_set as usize][tx_type as usize] == 0 {
      continue;
    }

    if is_inter {
      motion_compensate(fi, fs, cw, mode, ref_frames, mvs, bsize, bo, true);
    }

    let mut wr: &mut dyn Writer = &mut WriterCounter::new();
    let tell = wr.tell_frac();
    let tx_dist = if is_inter {
      write_tx_tree(
        fi, fs, cw, wr, mode, bo, bsize, tx_size, tx_type, false, true, true
      )
    }  else {
      let cfl = CFLParams::new(); // Unused
      write_tx_blocks(
        fi, fs, cw, wr, mode, mode, bo, bsize, tx_size, tx_type, false, cfl, true, true
      )
    };

    let cost = wr.tell_frac() - tell;
      let rd = if fi.use_tx_domain_distortion {
        compute_tx_rd_cost(
          fi,
          fs,
          w,
          h,
          is_chroma_block,
          bo,
          cost,
          tx_dist,
          false,
          true
        )
      } else {
        compute_rd_cost(
          fi,
          fs,
          w,
          h,
          is_chroma_block,
          bo,
          cost,
          true
        )
    };
    if rd < best_rd {
      best_rd = rd;
      best_type = tx_type;
    }

    cw.rollback(&cw_checkpoint);
  }

  assert!(best_rd >= 0_f64);

  best_type
}

pub fn get_sub_partitions<'a>(four_partitions: &[&'a BlockOffset; 4],
   partition: PartitionType) -> Vec<&'a BlockOffset> {
  let mut partitions = vec![ four_partitions[0] ];

  if partition == PARTITION_NONE {
      return partitions;
  }
  if partition == PARTITION_VERT || partition == PARTITION_SPLIT {
     partitions.push(four_partitions[1]);
  };
  if partition == PARTITION_HORZ || partition == PARTITION_SPLIT {
     partitions.push(four_partitions[2]);
  };
  if partition == PARTITION_SPLIT {
     partitions.push(four_partitions[3]);
  };

  partitions
}

pub fn get_sub_partitions_with_border_check<'a>(four_partitions: &[&'a BlockOffset; 4],
   partition: PartitionType, fi: &FrameInvariants, subsize: BlockSize) -> Vec<&'a BlockOffset> {
  let mut partitions = vec![ four_partitions[0] ];

  if partition == PARTITION_NONE {
      return partitions;
  }
  let hbsw = subsize.width_mi(); // Half the block size width in blocks
  let hbsh = subsize.height_mi(); // Half the block size height in blocks

  if partition == PARTITION_VERT || partition == PARTITION_SPLIT {
    if four_partitions[1].x + hbsw as usize <= fi.w_in_b &&
      four_partitions[1].y + hbsh as usize <= fi.h_in_b {
        partitions.push(four_partitions[1]); }
  };
  if partition == PARTITION_HORZ || partition == PARTITION_SPLIT {
    if four_partitions[2].x + hbsw as usize <= fi.w_in_b &&
      four_partitions[2].y + hbsh as usize <= fi.h_in_b {
        partitions.push(four_partitions[2]); }
  };
  if partition == PARTITION_SPLIT {
    if four_partitions[3].x + hbsw as usize <= fi.w_in_b &&
      four_partitions[3].y + hbsh as usize <= fi.h_in_b {
        partitions.push(four_partitions[3]); }
  };

  partitions
}

// RDO-based single level partitioning decision
pub fn rdo_partition_decision(
  fi: &FrameInvariants, fs: &mut FrameState,
  cw: &mut ContextWriter, w_pre_cdef: &mut dyn Writer, w_post_cdef: &mut dyn Writer,
  bsize: BlockSize, bo: &BlockOffset,
  cached_block: &RDOOutput, pmvs: &[[Option<MotionVector>; REF_FRAMES]; 5],
  partition_types: &Vec<PartitionType>,
) -> RDOOutput {
  let mut best_partition = cached_block.part_type;
  let mut best_rd = cached_block.rd_cost;
  let mut best_pred_modes = cached_block.part_modes.clone();

  for &partition in partition_types {
    // Do not re-encode results we already have
    if partition == cached_block.part_type {
      continue;
    }

    let mut cost: f64 = 0.0;
    let mut child_modes = std::vec::Vec::new();

    match partition {
      PartitionType::PARTITION_NONE => {
        if bsize > BlockSize::BLOCK_64X64 {
          continue;
        }

        let pmv_idx = if bsize > BlockSize::BLOCK_32X32 {
          0
        } else {
          ((bo.x & 32) >> 5) + ((bo.y & 32) >> 4) + 1
        };

        let spmvs = &pmvs[pmv_idx];

        let mode_decision = rdo_mode_decision(fi, fs, cw, bsize, bo, spmvs, false).part_modes[0].clone();
        child_modes.push(mode_decision);
      }
      PARTITION_SPLIT |
      PARTITION_HORZ |
      PARTITION_VERT => {
        let subsize = bsize.subsize(partition);

        if subsize == BlockSize::BLOCK_INVALID {
          continue;
        }

        //pmv = best_pred_modes[0].mvs[0];

        assert!(best_pred_modes.len() <= 4);

        let hbsw = subsize.width_mi(); // Half the block size width in blocks
        let hbsh = subsize.height_mi(); // Half the block size height in blocks
        let four_partitions = [
          bo,
          &BlockOffset{ x: bo.x + hbsw as usize, y: bo.y },
          &BlockOffset{ x: bo.x, y: bo.y + hbsh as usize },
          &BlockOffset{ x: bo.x + hbsw as usize, y: bo.y + hbsh as usize }
        ];
        let partitions = get_sub_partitions_with_border_check(&four_partitions, partition, fi, subsize);

        let pmv_idxs = partitions.iter().map(|&offset| {
          if subsize.greater_than(BlockSize::BLOCK_32X32) {
              0
          } else {
              ((offset.x & 32) >> 5) + ((offset.y & 32) >> 4) + 1
          }
        }).collect::<Vec<_>>();

        let cw_checkpoint = cw.checkpoint();
        let w_pre_checkpoint = w_pre_cdef.checkpoint();
        let w_post_checkpoint = w_post_cdef.checkpoint();

        if bsize >= BlockSize::BLOCK_8X8 {
          let w: &mut dyn Writer = if cw.bc.cdef_coded {w_post_cdef} else {w_pre_cdef};
          let tell = w.tell_frac();
          cw.write_partition(w, bo, partition, bsize);
          cost = (w.tell_frac() - tell) as f64 * get_lambda(fi)/ ((1 << OD_BITRES) as f64);
        }

        child_modes.extend(
          partitions
            .iter().zip(pmv_idxs)
            .map(|(&offset, pmv_idx)| {
              let mode_decision =
              rdo_mode_decision(fi, fs, cw, subsize, &offset,
                &pmvs[pmv_idx], true)
                .part_modes[0]
                .clone();

                if subsize >= BlockSize::BLOCK_8X8 && subsize.is_sqr() {
                  let w: &mut dyn Writer = if cw.bc.cdef_coded {w_post_cdef} else {w_pre_cdef};
                  cw.write_partition(w, offset, PartitionType::PARTITION_NONE, subsize);
                }

                encode_block_with_modes(fi, fs, cw, w_pre_cdef, w_post_cdef, subsize,
                                    offset, &mode_decision);
                mode_decision
            }).collect::<Vec<_>>()
        );
        cw.rollback(&cw_checkpoint);
        w_pre_cdef.rollback(&w_pre_checkpoint);
        w_post_cdef.rollback(&w_post_checkpoint);
      }
      _ => {
        assert!(false);
      }
    }

    let rd = cost + child_modes.iter().map(|m| m.rd_cost).sum::<f64>();

    if rd < best_rd {
      best_rd = rd;
      best_partition = partition;
      best_pred_modes = child_modes.clone();
    }
  }

  assert!(best_rd >= 0_f64);

  RDOOutput {
    rd_cost: best_rd,
    part_type: best_partition,
    part_modes: best_pred_modes
  }
}

pub fn rdo_cdef_decision(sbo: &SuperBlockOffset, fi: &FrameInvariants,
                         fs: &FrameState, cw: &mut ContextWriter) -> u8 {
    // FIXME: 128x128 SB support will break this, we need FilterBlockOffset etc.
    // Construct a single-superblock-sized frame to test-filter into
    let sbo_0 = SuperBlockOffset { x: 0, y: 0 };
    let bc = &mut cw.bc;
    let mut cdef_output = Frame {
        planes: [
            Plane::new(64 >> fs.rec.planes[0].cfg.xdec, 64 >> fs.rec.planes[0].cfg.ydec,
                       fs.rec.planes[0].cfg.xdec, fs.rec.planes[0].cfg.ydec, 0, 0),
            Plane::new(64 >> fs.rec.planes[1].cfg.xdec, 64 >> fs.rec.planes[1].cfg.ydec,
                       fs.rec.planes[1].cfg.xdec, fs.rec.planes[1].cfg.ydec, 0, 0),
            Plane::new(64 >> fs.rec.planes[2].cfg.xdec, 64 >> fs.rec.planes[2].cfg.ydec,
                       fs.rec.planes[2].cfg.xdec, fs.rec.planes[2].cfg.ydec, 0, 0),
        ]
    };
    // Construct a padded input
    let mut rec_input = Frame {
        planes: [
            Plane::new((64 >> fs.rec.planes[0].cfg.xdec)+4, (64 >> fs.rec.planes[0].cfg.ydec)+4,
                       fs.rec.planes[0].cfg.xdec, fs.rec.planes[0].cfg.ydec, 0, 0),
            Plane::new((64 >> fs.rec.planes[1].cfg.xdec)+4, (64 >> fs.rec.planes[1].cfg.ydec)+4,
                       fs.rec.planes[1].cfg.xdec, fs.rec.planes[1].cfg.ydec, 0, 0),
            Plane::new((64 >> fs.rec.planes[2].cfg.xdec)+4, (64 >> fs.rec.planes[2].cfg.ydec)+4,
                       fs.rec.planes[2].cfg.xdec, fs.rec.planes[2].cfg.ydec, 0, 0),
        ]
    };
    // Copy reconstructed data into padded input
    for p in 0..3 {
        let xdec = fs.rec.planes[p].cfg.xdec;
        let ydec = fs.rec.planes[p].cfg.ydec;
        let h = fi.padded_h as isize >> ydec;
        let w = fi.padded_w as isize >> xdec;
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
                            rec_row[x as usize] = in_row[(offset.x+x-2) as usize]
                        } else {
                            rec_row[x as usize] = CDEF_VERY_LARGE;
                        }
                    }
                }  else  {
                    // Yes, do it the easy way: just copy
                    rec_row[0..(64>>xdec)+4].copy_from_slice(&in_row[(offset.x-2) as usize..(offset.x+(64>>xdec)+2) as usize]);
                }
            }
        }
    }

    // RDO comparisons
    let mut best_index: u8 = 0;
    let mut best_err: u64 = 0;
    let cdef_dirs = cdef_analyze_superblock(&mut rec_input, bc, &sbo_0, &sbo, fi.sequence.bit_depth);
    for cdef_index in 0..(1<<fi.cdef_bits) {
        //for p in 0..3 {
        //    for i in 0..cdef_output.planes[p].data.len() { cdef_output.planes[p].data[i] = CDEF_VERY_LARGE; }
        //}
        // TODO: Don't repeat find_direction over and over; split filter_superblock to run it separately
        cdef_filter_superblock(fi, &mut rec_input, &mut cdef_output,
                               bc, &sbo_0, &sbo, cdef_index, &cdef_dirs);

        // Rate is constant, compute just distortion
        // Computation is block by block, paying attention to skip flag

        // Each direction block is 8x8 in y, potentially smaller if subsampled in chroma
        // We're dealing only with in-frmae and unpadded planes now
        let mut err:u64 = 0;
        for by in 0..8 {
            for bx in 0..8 {
                let bo = sbo.block_offset(bx<<1, by<<1);
                if bo.x < bc.cols && bo.y < bc.rows {
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
                                err += cdef_dist_wxh_8x8(&in_slice, &out_slice, fi.sequence.bit_depth);
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
