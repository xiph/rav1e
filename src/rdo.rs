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

use crate::api::*;
use crate::cdef::*;
use crate::context::*;
use crate::dist::*;
use crate::ec::{Writer, WriterCounter, OD_BITRES};
use crate::encode_block_with_modes;
use crate::encoder::FrameInvariants;
use crate::frame::Frame;
use crate::frame::*;
use crate::header::ReferenceMode;
use crate::lrf::*;
use crate::luma_ac;
use crate::mc::MotionVector;
use crate::me::*;
use crate::motion_compensate;
use crate::partition::RefType::*;
use crate::partition::*;
use crate::predict::{
  PredictionMode, RAV1E_INTER_COMPOUND_MODES, RAV1E_INTER_MODES_MINIMAL,
  RAV1E_INTRA_MODES,
};
use crate::rdo_tables::*;
use crate::tiling::*;
use crate::transform::{TxSet, TxSize, TxType, RAV1E_TX_TYPES};
use crate::util::{AlignedArray, CastFromPrimitive, Pixel};
use crate::write_tx_blocks;
use crate::write_tx_tree;
use crate::Tune;
use crate::{encode_block_post_cdef, encode_block_pre_cdef};

use crate::partition::PartitionType::*;
use arrayvec::*;
use serde_derive::{Deserialize, Serialize};
use std;
use std::vec::Vec;

#[derive(Copy, Clone, PartialEq)]
pub enum RDOType {
  PixelDistRealRate,
  TxDistRealRate,
  TxDistEstRate,
  Train,
}

impl RDOType {
  pub fn needs_tx_dist(self) -> bool {
    match self {
      // Pixel-domain distortion and exact ec rate
      RDOType::PixelDistRealRate => false,
      // Tx-domain distortion and exact ec rate
      RDOType::TxDistRealRate => true,
      // Tx-domain distortion and txdist-based rate
      RDOType::TxDistEstRate => true,
      RDOType::Train => true,
    }
  }
  pub fn needs_coeff_rate(self) -> bool {
    match self {
      RDOType::PixelDistRealRate => true,
      RDOType::TxDistRealRate => true,
      RDOType::TxDistEstRate => false,
      RDOType::Train => true,
    }
  }
}

#[derive(Clone)]
pub struct RDOOutput {
  pub rd_cost: f64,
  pub part_type: PartitionType,
  pub part_modes: Vec<RDOPartitionOutput>,
}

#[derive(Clone)]
pub struct RDOPartitionOutput {
  pub rd_cost: f64,
  pub bo: TileBlockOffset,
  pub bsize: BlockSize,
  pub pred_mode_luma: PredictionMode,
  pub pred_mode_chroma: PredictionMode,
  pub pred_cfl_params: CFLParams,
  pub ref_frames: [RefType; 2],
  pub mvs: [MotionVector; 2],
  pub skip: bool,
  pub tx_size: TxSize,
  pub tx_type: TxType,
  pub sidx: u8,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct RDOTracker {
  rate_bins: Vec<Vec<Vec<u64>>>,
  rate_counts: Vec<Vec<Vec<u64>>>,
}

impl RDOTracker {
  pub fn new() -> RDOTracker {
    RDOTracker {
      rate_bins: vec![
        vec![vec![0; RDO_NUM_BINS]; TxSize::TX_SIZES_ALL];
        RDO_QUANT_BINS
      ],
      rate_counts: vec![
        vec![vec![0; RDO_NUM_BINS]; TxSize::TX_SIZES_ALL];
        RDO_QUANT_BINS
      ],
    }
  }
  fn merge_array(new: &mut Vec<u64>, old: &[u64]) {
    for (n, o) in new.iter_mut().zip(old.iter()) {
      *n += o;
    }
  }
  fn merge_2d_array(new: &mut Vec<Vec<u64>>, old: &[Vec<u64>]) {
    for (n, o) in new.iter_mut().zip(old.iter()) {
      RDOTracker::merge_array(n, o);
    }
  }
  fn merge_3d_array(new: &mut Vec<Vec<Vec<u64>>>, old: &[Vec<Vec<u64>>]) {
    for (n, o) in new.iter_mut().zip(old.iter()) {
      RDOTracker::merge_2d_array(n, o);
    }
  }
  pub fn merge_in(&mut self, input: &RDOTracker) {
    RDOTracker::merge_3d_array(&mut self.rate_bins, &input.rate_bins);
    RDOTracker::merge_3d_array(&mut self.rate_counts, &input.rate_counts);
  }
  pub fn add_rate(
    &mut self, qindex: u8, ts: TxSize, fast_distortion: u64, rate: u64,
  ) {
    if fast_distortion != 0 {
      let bs_index = ts as usize;
      let q_bin_idx = (qindex as usize) / RDO_QUANT_DIV;
      let bin_idx_tmp = ((fast_distortion as i64
        - (RATE_EST_BIN_SIZE as i64) / 2) as u64
        / RATE_EST_BIN_SIZE) as usize;
      let bin_idx = if bin_idx_tmp >= RDO_NUM_BINS {
        RDO_NUM_BINS - 1
      } else {
        bin_idx_tmp
      };
      self.rate_counts[q_bin_idx][bs_index][bin_idx] += 1;
      self.rate_bins[q_bin_idx][bs_index][bin_idx] += rate;
    }
  }
  pub fn print_code(&self) {
    println!("pub static RDO_RATE_TABLE: [[[u64; RDO_NUM_BINS]; TxSize::TX_SIZES_ALL]; RDO_QUANT_BINS] = [");
    for q_bin in 0..RDO_QUANT_BINS {
      print!("[");
      for bs_index in 0..TxSize::TX_SIZES_ALL {
        print!("[");
        for (rate_total, rate_count) in self.rate_bins[q_bin][bs_index]
          .iter()
          .zip(self.rate_counts[q_bin][bs_index].iter())
        {
          if *rate_count > 100 {
            print!("{},", rate_total / rate_count);
          } else {
            print!("99999,");
          }
        }
        println!("],");
      }
      println!("],");
    }
    println!("];");
  }
}

pub fn estimate_rate(qindex: u8, ts: TxSize, fast_distortion: u64) -> u64 {
  let bs_index = ts as usize;
  let q_bin_idx = (qindex as usize) / RDO_QUANT_DIV;
  let bin_idx_down =
    ((fast_distortion) / RATE_EST_BIN_SIZE).min((RDO_NUM_BINS - 2) as u64);
  let bin_idx_up = (bin_idx_down + 1).min((RDO_NUM_BINS - 1) as u64);
  let x0 = (bin_idx_down * RATE_EST_BIN_SIZE) as i64;
  let x1 = (bin_idx_up * RATE_EST_BIN_SIZE) as i64;
  let y0 = RDO_RATE_TABLE[q_bin_idx][bs_index][bin_idx_down as usize] as i64;
  let y1 = RDO_RATE_TABLE[q_bin_idx][bs_index][bin_idx_up as usize] as i64;
  let slope = ((y1 - y0) << 8) / (x1 - x0);
  (y0 + (((fast_distortion as i64 - x0) * slope) >> 8)).max(0) as u64
}

#[allow(unused)]
fn cdef_dist_wxh_8x8<T: Pixel>(
  src1: &PlaneRegion<'_, T>, src2: &PlaneRegion<'_, T>, bit_depth: usize,
) -> u64 {
  debug_assert!(src1.plane_cfg.xdec == 0);
  debug_assert!(src1.plane_cfg.ydec == 0);
  debug_assert!(src2.plane_cfg.xdec == 0);
  debug_assert!(src2.plane_cfg.ydec == 0);

  let coeff_shift = bit_depth - 8;

  let mut sum_s: i32 = 0;
  let mut sum_d: i32 = 0;
  let mut sum_s2: i64 = 0;
  let mut sum_d2: i64 = 0;
  let mut sum_sd: i64 = 0;
  for j in 0..8 {
    for i in 0..8 {
      let s: i32 = src1[j][i].as_();
      let d: i32 = src2[j][i].as_();
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
  let ssim_boost = (4033_f64 / 16_384_f64)
    * (svar + dvar + (16_384 << (2 * coeff_shift)) as f64)
    / f64::sqrt((16_265_089u64 << (4 * coeff_shift)) as f64 + svar * dvar);
  (sse * ssim_boost + 0.5_f64) as u64
}

#[allow(unused)]
fn cdef_dist_wxh<T: Pixel, F: Fn(Area, BlockSize) -> f64>(
  src1: &PlaneRegion<'_, T>, src2: &PlaneRegion<'_, T>, w: usize, h: usize,
  bit_depth: usize, compute_bias: F,
) -> u64 {
  assert!(w & 0x7 == 0);
  assert!(h & 0x7 == 0);
  debug_assert!(src1.plane_cfg.xdec == 0);
  debug_assert!(src1.plane_cfg.ydec == 0);
  debug_assert!(src2.plane_cfg.xdec == 0);
  debug_assert!(src2.plane_cfg.ydec == 0);

  let mut sum: u64 = 0;
  for j in 0isize..h as isize / 8 {
    for i in 0isize..w as isize / 8 {
      let area = Area::StartingAt { x: i * 8, y: j * 8 };
      let value = cdef_dist_wxh_8x8(
        &src1.subregion(area),
        &src2.subregion(area),
        bit_depth,
      );

      // cdef is always called on non-subsampled planes, so BLOCK_8X8 is
      // correct here.
      let bias = compute_bias(area, BlockSize::BLOCK_8X8);
      sum += (value as f64 * bias) as u64;
    }
  }
  sum
}

// Sum of Squared Error for a wxh block
pub fn sse_wxh<T: Pixel, F: Fn(Area, BlockSize) -> f64>(
  src1: &PlaneRegion<'_, T>, src2: &PlaneRegion<'_, T>, w: usize, h: usize,
  compute_bias: F,
) -> u64 {
  assert!(w & (MI_SIZE - 1) == 0);
  assert!(h & (MI_SIZE - 1) == 0);

  // To bias the distortion correctly, compute it in blocks which are
  // equivalent to 4×4 blocks in a non-subsampled plane.
  let block_w = MI_SIZE >> src1.plane_cfg.xdec;
  let block_h = MI_SIZE >> src1.plane_cfg.ydec;

  let mut sse: u64 = 0;
  for block_y in 0..h / block_h {
    for block_x in 0..w / block_w {
      let mut value = 0;

      for j in 0..block_h {
        let s1 = &src1[block_y * block_h + j]
          [block_x * block_w..(block_x + 1) * block_w];
        let s2 = &src2[block_y * block_h + j]
          [block_x * block_w..(block_x + 1) * block_w];

        let row_sse = s1
          .iter()
          .zip(s2)
          .map(|(&a, &b)| {
            let c = (i16::cast_from(a) - i16::cast_from(b)) as i32;
            (c * c) as u32
          })
          .sum::<u32>();
        value += row_sse as u64;
      }

      let bias = compute_bias(
        // StartingAt gives the correct block offset.
        Area::StartingAt {
          x: (block_x * block_w) as isize,
          y: (block_y * block_h) as isize,
        },
        // Our block width and height are chosen in such a way as to match 4×4
        // non-subsampled blocks.
        BlockSize::BLOCK_4X4,
      );
      sse += (value as f64 * bias) as u64;
    }
  }
  sse
}

// Compute the pixel-domain distortion for an encode
fn compute_distortion<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>, bsize: BlockSize,
  is_chroma_block: bool, tile_bo: TileBlockOffset, luma_only: bool,
) -> u64 {
  let area = Area::BlockStartingAt { bo: tile_bo.0 };
  let input_region = ts.input_tile.planes[0].subregion(area);
  let rec_region = ts.rec.planes[0].subregion(area);
  let mut distortion = match fi.config.tune {
    Tune::Psychovisual if bsize.width() >= 8 && bsize.height() >= 8 => {
      cdef_dist_wxh(
        &input_region,
        &rec_region,
        bsize.width(),
        bsize.height(),
        fi.sequence.bit_depth,
        |bias_area, bsize| {
          compute_distortion_bias(
            fi,
            input_region.subregion(bias_area).frame_block_offset(),
            bsize,
          )
        },
      )
    }
    Tune::Psnr | Tune::Psychovisual => sse_wxh(
      &input_region,
      &rec_region,
      bsize.width(),
      bsize.height(),
      |bias_area, bsize| {
        compute_distortion_bias(
          fi,
          input_region.subregion(bias_area).frame_block_offset(),
          bsize,
        )
      },
    ),
  };

  if !luma_only {
    let PlaneConfig { xdec, ydec, .. } = ts.input.planes[1].cfg;

    let mask = !(MI_SIZE - 1);
    let mut w_uv = (bsize.width() >> xdec) & mask;
    let mut h_uv = (bsize.height() >> ydec) & mask;

    if (w_uv == 0 || h_uv == 0) && is_chroma_block {
      w_uv = MI_SIZE;
      h_uv = MI_SIZE;
    }

    // Add chroma distortion only when it is available
    if w_uv > 0 && h_uv > 0 {
      for p in 1..3 {
        let input_region = ts.input_tile.planes[p].subregion(area);
        let rec_region = ts.rec.planes[p].subregion(area);
        distortion += sse_wxh(
          &input_region,
          &rec_region,
          w_uv,
          h_uv,
          |bias_area, bsize| {
            compute_distortion_bias(
              fi,
              input_region.subregion(bias_area).frame_block_offset(),
              bsize,
            )
          },
        );
      }
    };
  }
  distortion
}

// Compute the transform-domain distortion for an encode
fn compute_tx_distortion<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>, bsize: BlockSize,
  is_chroma_block: bool, tile_bo: TileBlockOffset, tx_dist: i64, skip: bool,
  luma_only: bool,
) -> u64 {
  assert!(fi.config.tune == Tune::Psnr);
  let area = Area::BlockStartingAt { bo: tile_bo.0 };
  let input_region = ts.input_tile.planes[0].subregion(area);
  let rec_region = ts.rec.planes[0].subregion(area);
  let mut distortion = if skip {
    sse_wxh(
      &input_region,
      &rec_region,
      bsize.width(),
      bsize.height(),
      |bias_area, bsize| {
        compute_distortion_bias(
          fi,
          input_region.subregion(bias_area).frame_block_offset(),
          bsize,
        )
      },
    )
  } else {
    assert!(tx_dist >= 0);
    let bias =
      compute_distortion_bias(fi, ts.to_frame_block_offset(tile_bo), bsize);
    (tx_dist as f64 * bias) as u64
  };

  if !luma_only && skip {
    let PlaneConfig { xdec, ydec, .. } = ts.input.planes[1].cfg;

    let mask = !(MI_SIZE - 1);
    let mut w_uv = (bsize.width() >> xdec) & mask;
    let mut h_uv = (bsize.height() >> ydec) & mask;

    if (w_uv == 0 || h_uv == 0) && is_chroma_block {
      w_uv = MI_SIZE;
      h_uv = MI_SIZE;
    }

    // Add chroma distortion only when it is available
    if w_uv > 0 && h_uv > 0 {
      for p in 1..3 {
        let input_region = ts.input_tile.planes[p].subregion(area);
        let rec_region = ts.rec.planes[p].subregion(area);
        distortion += sse_wxh(
          &input_region,
          &rec_region,
          w_uv,
          h_uv,
          |bias_area, bsize| {
            compute_distortion_bias(
              fi,
              input_region.subregion(bias_area).frame_block_offset(),
              bsize,
            )
          },
        );
      }
    }
  }
  distortion
}

fn compute_mean_importance<T: Pixel>(
  fi: &FrameInvariants<T>, frame_bo: PlaneBlockOffset, bsize: BlockSize,
) -> f32 {
  let x1 = frame_bo.0.x;
  let y1 = frame_bo.0.y;
  let x2 = (x1 + bsize.width_mi()).min(fi.w_in_b);
  let y2 = (y1 + bsize.height_mi()).min(fi.h_in_b);

  let mut total_importance = 0.;
  for y in y1..y2 {
    for x in x1..x2 {
      total_importance += fi.block_importances[y / 2 * fi.w_in_imp_b + x / 2];
    }
  }
  // Divide by the full area even though some blocks were outside.
  total_importance / (bsize.width_mi() * bsize.height_mi()) as f32
}

fn compute_distortion_bias<T: Pixel>(
  fi: &FrameInvariants<T>, frame_bo: PlaneBlockOffset, bsize: BlockSize,
) -> f64 {
  let mean_importance = compute_mean_importance(fi, frame_bo, bsize);

  // Chosen based on a series of AWCY runs.
  const FACTOR: f32 = 3.;
  const ADDEND: f64 = 0.65;

  let bias = (mean_importance / FACTOR) as f64 + ADDEND;
  debug_assert!(bias.is_finite());

  bias
}

pub fn compute_rd_cost<T: Pixel>(
  fi: &FrameInvariants<T>, rate: u32, distortion: u64,
) -> f64 {
  let rate_in_bits = (rate as f64) / ((1 << OD_BITRES) as f64);
  distortion as f64 + fi.lambda * rate_in_bits
}

pub fn rdo_tx_size_type<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &mut TileStateMut<'_, T>,
  cw: &mut ContextWriter, bsize: BlockSize, tile_bo: TileBlockOffset,
  luma_mode: PredictionMode, ref_frames: [RefType; 2], mvs: [MotionVector; 2],
  skip: bool,
) -> (TxSize, TxType) {
  let mut tx_size = max_txsize_rect_lookup[bsize as usize];
  let mut best_tx_type = TxType::DCT_DCT;
  let mut best_tx_size = tx_size;
  let mut best_rd = std::f64::MAX;
  let is_inter = !luma_mode.is_intra();

  let do_rdo_tx_size = fi.tx_mode_select
    && fi.config.speed_settings.rdo_tx_decision
    && luma_mode.is_intra();
  let rdo_tx_depth = if do_rdo_tx_size { 2 } else { 0 };
  let cw_checkpoint = cw.checkpoint();

  for _ in 0..=rdo_tx_depth {
    let tx_set = get_tx_set(tx_size, is_inter, fi.use_reduced_tx_set);

    let do_rdo_tx_type = tx_set > TxSet::TX_SET_DCTONLY
      && fi.config.speed_settings.rdo_tx_decision
      && !skip;

    if !do_rdo_tx_size && !do_rdo_tx_type {
      return (best_tx_size, best_tx_type);
    };

    let tx_types =
      if do_rdo_tx_type { RAV1E_TX_TYPES } else { &[TxType::DCT_DCT] };

    // Luma plane transform type decision
    let (tx_type, rd_cost) = rdo_tx_type_decision(
      fi, ts, cw, luma_mode, ref_frames, mvs, bsize, tile_bo, tx_size, tx_set,
      tx_types,
    );

    if rd_cost < best_rd {
      best_tx_size = tx_size;
      best_tx_type = tx_type;
      best_rd = rd_cost;
    }

    debug_assert!(tx_size.width_log2() <= bsize.width_log2());
    debug_assert!(tx_size.height_log2() <= bsize.height_log2());
    debug_assert!(
      tx_size.sqr() <= TxSize::TX_32X32 || tx_type == TxType::DCT_DCT
    );

    let next_tx_size = sub_tx_size_map[tx_size as usize];
    cw.rollback(&cw_checkpoint);

    if next_tx_size == tx_size {
      break;
    } else {
      tx_size = next_tx_size;
    };
  }

  (best_tx_size, best_tx_type)
}

struct EncodingSettings {
  mode_luma: PredictionMode,
  mode_chroma: PredictionMode,
  cfl_params: CFLParams,
  skip: bool,
  rd: f64,
  ref_frames: [RefType; 2],
  mvs: [MotionVector; 2],
  tx_size: TxSize,
  tx_type: TxType,
  sidx: u8,
}

impl Default for EncodingSettings {
  fn default() -> Self {
    Self {
      mode_luma: PredictionMode::DC_PRED,
      mode_chroma: PredictionMode::DC_PRED,
      cfl_params: CFLParams::default(),
      skip: false,
      rd: std::f64::MAX,
      ref_frames: [INTRA_FRAME, NONE_FRAME],
      mvs: [MotionVector::default(); 2],
      tx_size: TxSize::TX_4X4,
      tx_type: TxType::DCT_DCT,
      sidx: 0,
    }
  }
}

#[inline]
fn luma_chroma_mode_rdo<T: Pixel>(
  luma_mode: PredictionMode, fi: &FrameInvariants<T>, bsize: BlockSize,
  tile_bo: TileBlockOffset, ts: &mut TileStateMut<'_, T>,
  cw: &mut ContextWriter, rdo_type: RDOType,
  cw_checkpoint: &ContextWriterCheckpoint, best: &mut EncodingSettings,
  mvs: [MotionVector; 2], ref_frames: [RefType; 2],
  mode_set_chroma: &[PredictionMode], luma_mode_is_intra: bool,
  mode_context: usize, mv_stack: &ArrayVec<[CandidateMV; 9]>,
) {
  let PlaneConfig { xdec, ydec, .. } = ts.input.planes[1].cfg;

  let is_chroma_block = has_chroma(tile_bo, bsize, xdec, ydec);

  // Find the best chroma prediction mode for the current luma prediction mode
  let mut chroma_rdo = |skip: bool| -> bool {
    let mut zero_distortion = false;

    // If skip is true, sidx is not coded.
    // If quantizer RDO is disabled, sidx isn't coded either.
    let sidx_range = if skip {
      0..=0
    } else if !fi.config.speed_settings.quantizer_rdo {
      let importance =
        compute_mean_importance(fi, ts.to_frame_block_offset(tile_bo), bsize);
      // Chosen based on the RDO segment ID statistics for speed 2 on the DOTA2
      // clip. More precisely:
      // - Mean importance and the corresponding best sidx chosen by RDO were
      //   dumped from encoding the DOTA2 clip on --speed 2.
      // - The values were plotted in a logarithmic 2D histogram.
      // - Based on that, the value below were chosen.
      let heuristic_sidx = match importance {
        x if x >= 0. && x < 2. => 1,
        x if x >= 2. && x < 4. => 0,
        x if x >= 4. => 2,
        _ => unreachable!(),
      };
      heuristic_sidx..=heuristic_sidx
    } else {
      0..=2
    };

    for sidx in sidx_range {
      cw.bc.blocks.set_segmentation_idx(tile_bo, bsize, sidx);

      let (tx_size, tx_type) = rdo_tx_size_type(
        fi, ts, cw, bsize, tile_bo, luma_mode, ref_frames, mvs, skip,
      );
      for &chroma_mode in mode_set_chroma.iter() {
        let wr = &mut WriterCounter::new();
        let tell = wr.tell_frac();

        if bsize >= BlockSize::BLOCK_8X8 && bsize.is_sqr() {
          cw.write_partition(
            wr,
            tile_bo,
            PartitionType::PARTITION_NONE,
            bsize,
          );
        }

        // TODO(yushin): luma and chroma would have different decision based on chroma format
        let need_recon_pixel =
          luma_mode_is_intra && tx_size.block_size() != bsize;

        encode_block_pre_cdef(&fi.sequence, ts, cw, wr, bsize, tile_bo, skip);
        let tx_dist = encode_block_post_cdef(
          fi,
          ts,
          cw,
          wr,
          luma_mode,
          chroma_mode,
          ref_frames,
          mvs,
          bsize,
          tile_bo,
          skip,
          CFLParams::default(),
          tx_size,
          tx_type,
          mode_context,
          &mv_stack,
          rdo_type,
          need_recon_pixel,
          false,
        );

        let rate = wr.tell_frac() - tell;
        let distortion = if fi.use_tx_domain_distortion && !need_recon_pixel {
          compute_tx_distortion(
            fi,
            ts,
            bsize,
            is_chroma_block,
            tile_bo,
            tx_dist,
            skip,
            false,
          )
        } else {
          compute_distortion(fi, ts, bsize, is_chroma_block, tile_bo, false)
        };
        let rd = compute_rd_cost(fi, rate, distortion);
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
          best.sidx = sidx;
          zero_distortion = distortion == 0;
        }

        cw.rollback(cw_checkpoint);
      }
    }

    zero_distortion
  };

  // Don't skip when using intra modes
  let zero_distortion =
    if !luma_mode_is_intra { chroma_rdo(true) } else { false };
  // early skip
  if !zero_distortion {
    chroma_rdo(false);
  }
}

// RDO-based mode decision
pub fn rdo_mode_decision<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &mut TileStateMut<'_, T>,
  cw: &mut ContextWriter, bsize: BlockSize, tile_bo: TileBlockOffset,
  pmvs: &mut [Option<MotionVector>],
) -> RDOPartitionOutput {
  let mut best = EncodingSettings::default();

  let PlaneConfig { xdec, ydec, .. } = ts.input.planes[1].cfg;
  let is_chroma_block = has_chroma(tile_bo, bsize, xdec, ydec);

  let cw_checkpoint = cw.checkpoint();

  // we can never have more than 7 reference frame sets
  let mut ref_frames_set = ArrayVec::<[_; 7]>::new();
  // again, max of 7 ref slots
  let mut ref_slot_set = ArrayVec::<[_; 7]>::new();
  // our implementation never returns more than 3 at the moment
  let mut mvs_from_me = ArrayVec::<[_; 3]>::new();
  let mut fwdref = None;
  let mut bwdref = None;

  let rdo_type = if fi.config.train_rdo {
    RDOType::Train
  } else if fi.use_tx_domain_rate {
    RDOType::TxDistEstRate
  } else if fi.use_tx_domain_distortion {
    RDOType::TxDistRealRate
  } else {
    RDOType::PixelDistRealRate
  };

  if fi.frame_type == FrameType::INTER {
    for i in ALL_INTER_REFS.iter() {
      // Don't search LAST3 since it's used only for probs
      if *i == LAST3_FRAME {
        continue;
      }
      if !ref_slot_set.contains(&fi.ref_frames[i.to_index()]) {
        if fwdref == None && i.is_fwd_ref() {
          fwdref = Some(ref_frames_set.len());
        }
        if bwdref == None && i.is_bwd_ref() {
          bwdref = Some(ref_frames_set.len());
        }
        ref_frames_set.push([*i, NONE_FRAME]);
        let slot_idx = fi.ref_frames[i.to_index()];
        ref_slot_set.push(slot_idx);
      }
    }
    assert!(!ref_frames_set.is_empty());
  }

  let mut inter_mode_set = ArrayVec::<[(PredictionMode, usize); 20]>::new();
  let mut mv_stacks = ArrayVec::<[_; 20]>::new();
  let mut mode_contexts = ArrayVec::<[_; 7]>::new();

  let motion_estimation = if fi.config.speed_settings.diamond_me {
    crate::me::DiamondSearch::motion_estimation
  } else {
    crate::me::FullSearch::motion_estimation
  };

  for (i, &ref_frames) in ref_frames_set.iter().enumerate() {
    let mut mv_stack = ArrayVec::<[CandidateMV; 9]>::new();
    mode_contexts.push(cw.find_mvrefs(
      tile_bo,
      ref_frames,
      &mut mv_stack,
      bsize,
      fi,
      false,
    ));

    if fi.frame_type == FrameType::INTER {
      let mut pmv = [MotionVector::default(); 2];
      if !mv_stack.is_empty() {
        pmv[0] = mv_stack[0].this_mv;
      }
      if mv_stack.len() > 1 {
        pmv[1] = mv_stack[1].this_mv;
      }
      let ref_slot = ref_slot_set[i] as usize;
      let cmv = pmvs[ref_slot].unwrap_or_else(Default::default);

      let b_me =
        motion_estimation(fi, ts, bsize, tile_bo, ref_frames[0], cmv, pmv);

      if !fi.config.speed_settings.encode_bottomup
        && (bsize == BlockSize::BLOCK_32X32 || bsize == BlockSize::BLOCK_64X64)
      {
        pmvs[ref_slot] = Some(b_me);
      };

      mvs_from_me.push([b_me, MotionVector::default()]);

      for &x in RAV1E_INTER_MODES_MINIMAL {
        inter_mode_set.push((x, i));
      }
      if !mv_stack.is_empty() {
        inter_mode_set.push((PredictionMode::NEAR0MV, i));
      }
      if mv_stack.len() >= 2 {
        inter_mode_set.push((PredictionMode::GLOBALMV, i));
      }
      let include_near_mvs = fi.config.speed_settings.include_near_mvs;
      if include_near_mvs {
        if mv_stack.len() >= 3 {
          inter_mode_set.push((PredictionMode::NEAR1MV, i));
        }
        if mv_stack.len() >= 4 {
          inter_mode_set.push((PredictionMode::NEAR2MV, i));
        }
      }
      let same_row_col = |x: &CandidateMV| {
        x.this_mv.row == mvs_from_me[i][0].row
          && x.this_mv.col == mvs_from_me[i][0].col
      };
      if !mv_stack
        .iter()
        .take(if include_near_mvs { 4 } else { 2 })
        .any(same_row_col)
        && (mvs_from_me[i][0].row != 0 || mvs_from_me[i][0].col != 0)
      {
        inter_mode_set.push((PredictionMode::NEWMV, i));
      }
    }
    mv_stacks.push(mv_stack);
  }

  let sz = bsize.width_mi().min(bsize.height_mi());

  if fi.frame_type == FrameType::INTER
    && fi.reference_mode != ReferenceMode::SINGLE
    && sz >= 2
  {
    // Adding compound candidate
    if let Some(r0) = fwdref {
      if let Some(r1) = bwdref {
        let ref_frames = [ref_frames_set[r0][0], ref_frames_set[r1][0]];
        ref_frames_set.push(ref_frames);
        let mv0 = mvs_from_me[r0][0];
        let mv1 = mvs_from_me[r1][0];
        mvs_from_me.push([mv0, mv1]);
        let mut mv_stack = ArrayVec::<[CandidateMV; 9]>::new();
        mode_contexts.push(cw.find_mvrefs(
          tile_bo,
          ref_frames,
          &mut mv_stack,
          bsize,
          fi,
          true,
        ));
        for &x in RAV1E_INTER_COMPOUND_MODES {
          inter_mode_set.push((x, ref_frames_set.len() - 1));
        }
        mv_stacks.push(mv_stack);
      }
    }
  }

  if fi.frame_type != FrameType::INTER {
    assert!(inter_mode_set.is_empty());
  }

  inter_mode_set.iter().for_each(|&(luma_mode, i)| {
    let mvs = match luma_mode {
      PredictionMode::NEWMV | PredictionMode::NEW_NEWMV => mvs_from_me[i],
      PredictionMode::NEARESTMV | PredictionMode::NEAREST_NEARESTMV => {
        if !mv_stacks[i].is_empty() {
          [mv_stacks[i][0].this_mv, mv_stacks[i][0].comp_mv]
        } else {
          [MotionVector::default(); 2]
        }
      }
      PredictionMode::NEAR0MV | PredictionMode::NEAR_NEARMV => {
        if mv_stacks[i].len() > 1 {
          [mv_stacks[i][1].this_mv, mv_stacks[i][1].comp_mv]
        } else {
          [MotionVector::default(); 2]
        }
      }
      PredictionMode::NEAR1MV | PredictionMode::NEAR2MV => [
        mv_stacks[i]
          [luma_mode as usize - PredictionMode::NEAR0MV as usize + 1]
          .this_mv,
        mv_stacks[i]
          [luma_mode as usize - PredictionMode::NEAR0MV as usize + 1]
          .comp_mv,
      ],
      PredictionMode::NEAREST_NEWMV => {
        [mv_stacks[i][0].this_mv, mvs_from_me[i][1]]
      }
      PredictionMode::NEW_NEARESTMV => {
        [mvs_from_me[i][0], mv_stacks[i][0].comp_mv]
      }
      PredictionMode::GLOBALMV | PredictionMode::GLOBAL_GLOBALMV => {
        [MotionVector::default(); 2]
      }
      _ => {
        unimplemented!();
      }
    };
    let mode_set_chroma = ArrayVec::from([luma_mode]);

    luma_chroma_mode_rdo(
      luma_mode,
      fi,
      bsize,
      tile_bo,
      ts,
      cw,
      rdo_type,
      &cw_checkpoint,
      &mut best,
      mvs,
      ref_frames_set[i],
      &mode_set_chroma,
      false,
      mode_contexts[i],
      &mv_stacks[i],
    );
  });

  if !best.skip {
    let num_modes_rdo: usize;
    let mut modes = ArrayVec::<[_; INTRA_MODES]>::new();

    // If tx partition (i.e. fi.tx_mode_select) is enabled, don't use below intra prediction screening
    if !fi.tx_mode_select {
      let tx_size = bsize.tx_size();

      // Reduce number of prediction modes at higher speed levels
      num_modes_rdo = if (fi.frame_type == FrameType::KEY
        && fi.config.speed_settings.prediction_modes
          >= PredictionModesSetting::ComplexKeyframes)
        || (fi.frame_type == FrameType::INTER
          && fi.config.speed_settings.prediction_modes
            >= PredictionModesSetting::ComplexAll)
      {
        7
      } else {
        3
      };

      let intra_mode_set = RAV1E_INTRA_MODES;
      let mut sads = {
        // FIXME: If tx partition is used, this whole sads block should be fixed
        debug_assert!(bsize == tx_size.block_size());
        let edge_buf = {
          let rec = &ts.rec.planes[0].as_const();
          let po = tile_bo.plane_offset(&rec.plane_cfg);
          // FIXME: If tx partition is used, get_intra_edges() should be called for each tx block
          get_intra_edges(
            rec,
            tile_bo,
            0,
            0,
            bsize,
            po,
            tx_size,
            fi.sequence.bit_depth,
            None,
          )
        };
        intra_mode_set
          .iter()
          .map(|&luma_mode| {
            let tile_rect = ts.tile_rect();
            let rec = &mut ts.rec.planes[0];
            let mut rec_region =
              rec.subregion_mut(Area::BlockStartingAt { bo: tile_bo.0 });
            // FIXME: If tx partition is used, luma_mode.predict_intra() should be called for each tx block
            luma_mode.predict_intra(
              tile_rect,
              &mut rec_region,
              tx_size,
              fi.sequence.bit_depth,
              &[0i16; 2],
              0,
              &edge_buf,
            );

            let plane_org = ts.input_tile.planes[0]
              .subregion(Area::BlockStartingAt { bo: tile_bo.0 });
            let plane_ref = rec_region.as_const();

            (
              luma_mode,
              get_sad(
                &plane_org,
                &plane_ref,
                tx_size.block_size(),
                fi.sequence.bit_depth,
                fi.cpu_feature_level,
              ),
            )
          })
          .collect::<Vec<_>>()
      };

      sads.sort_by_key(|a| a.1);

      // Find mode with lowest rate cost
      let mut z = 32768;
      let probs_all = if fi.frame_type == FrameType::INTER {
        cw.get_cdf_intra_mode(bsize)
      } else {
        cw.get_cdf_intra_mode_kf(tile_bo)
      }
      .iter()
      .take(INTRA_MODES)
      .map(|&a| {
        let d = z - a;
        z = a;
        d
      })
      .collect::<Vec<_>>();

      let mut probs = intra_mode_set
        .iter()
        .map(|&a| (a, probs_all[a as usize]))
        .collect::<Vec<_>>();
      probs.sort_by_key(|a| !a.1);

      probs
        .iter()
        .take(num_modes_rdo / 2)
        .for_each(|&(luma_mode, _prob)| modes.push(luma_mode));
      sads.iter().take(num_modes_rdo).for_each(|&(luma_mode, _sad)| {
        if !modes.contains(&luma_mode) {
          modes.push(luma_mode)
        }
      });
    } else {
      modes.extend(RAV1E_INTRA_MODES.iter().copied());
      num_modes_rdo = modes.len();
      debug_assert!(num_modes_rdo == RAV1E_INTRA_MODES.len());
    }

    debug_assert!(num_modes_rdo >= 1);

    modes.iter().take(num_modes_rdo).for_each(|&luma_mode| {
      let mvs = [MotionVector::default(); 2];
      let ref_frames = [INTRA_FRAME, NONE_FRAME];
      let mut mode_set_chroma = vec![luma_mode];
      //let mut mode_set_chroma = vec![PredictionMode::DC_PRED];
      if is_chroma_block && luma_mode != PredictionMode::DC_PRED {
        mode_set_chroma.push(PredictionMode::DC_PRED);
      }
      luma_chroma_mode_rdo(
        luma_mode,
        fi,
        bsize,
        tile_bo,
        ts,
        cw,
        rdo_type,
        &cw_checkpoint,
        &mut best,
        mvs,
        ref_frames,
        &mode_set_chroma,
        true,
        0,
        &ArrayVec::<[CandidateMV; 9]>::new(),
      );
    });
  }

  if best.mode_luma.is_intra() && is_chroma_block && bsize.cfl_allowed() {
    cw.bc.blocks.set_segmentation_idx(tile_bo, bsize, best.sidx);

    let chroma_mode = PredictionMode::UV_CFL_PRED;
    let cw_checkpoint = cw.checkpoint();
    let wr: &mut dyn Writer = &mut WriterCounter::new();
    write_tx_blocks(
      fi,
      ts,
      cw,
      wr,
      best.mode_luma,
      best.mode_luma,
      tile_bo,
      bsize,
      best.tx_size,
      best.tx_type,
      false,
      CFLParams::default(),
      true,
      rdo_type,
      true,
    );
    cw.rollback(&cw_checkpoint);
    if let Some(cfl) = rdo_cfl_alpha(ts, tile_bo, bsize, fi.sequence.bit_depth)
    {
      let wr: &mut dyn Writer = &mut WriterCounter::new();
      let tell = wr.tell_frac();

      encode_block_pre_cdef(
        &fi.sequence,
        ts,
        cw,
        wr,
        bsize,
        tile_bo,
        best.skip,
      );
      let _ = encode_block_post_cdef(
        fi,
        ts,
        cw,
        wr,
        best.mode_luma,
        chroma_mode,
        best.ref_frames,
        best.mvs,
        bsize,
        tile_bo,
        best.skip,
        cfl,
        best.tx_size,
        best.tx_type,
        0,
        &Vec::new(),
        rdo_type,
        true, // For CFL, luma should be always reconstructed.
        false,
      );

      let rate = wr.tell_frac() - tell;

      // For CFL, tx-domain distortion is not an option.
      let distortion =
        compute_distortion(fi, ts, bsize, is_chroma_block, tile_bo, false);
      let rd = compute_rd_cost(fi, rate, distortion);
      if rd < best.rd {
        best.rd = rd;
        best.mode_chroma = chroma_mode;
        best.cfl_params = cfl;
      }

      cw.rollback(&cw_checkpoint);
    }
  }

  cw.bc.blocks.set_mode(tile_bo, bsize, best.mode_luma);
  cw.bc.blocks.set_ref_frames(tile_bo, bsize, best.ref_frames);
  cw.bc.blocks.set_motion_vectors(tile_bo, bsize, best.mvs);

  assert!(best.rd >= 0_f64);

  RDOPartitionOutput {
    bo: tile_bo,
    bsize,
    pred_mode_luma: best.mode_luma,
    pred_mode_chroma: best.mode_chroma,
    pred_cfl_params: best.cfl_params,
    ref_frames: best.ref_frames,
    mvs: best.mvs,
    rd_cost: best.rd,
    skip: best.skip,
    tx_size: best.tx_size,
    tx_type: best.tx_type,
    sidx: best.sidx,
  }
}

pub fn rdo_cfl_alpha<T: Pixel>(
  ts: &mut TileStateMut<'_, T>, tile_bo: TileBlockOffset, bsize: BlockSize,
  bit_depth: usize,
) -> Option<CFLParams> {
  let PlaneConfig { xdec, ydec, .. } = ts.input.planes[1].cfg;
  let uv_tx_size = bsize.largest_chroma_tx_size(xdec, ydec);
  debug_assert!(bsize.subsampled_size(xdec, ydec) == uv_tx_size.block_size());

  let mut ac: AlignedArray<[i16; 32 * 32]> = AlignedArray::uninitialized();
  luma_ac(&mut ac.array, ts, tile_bo, bsize);
  let best_alpha: Vec<i16> = (1..3)
    .map(|p| {
      let &PlaneConfig { xdec, ydec, .. } = ts.rec.planes[p].plane_cfg;
      let tile_rect = ts.tile_rect().decimated(xdec, ydec);
      let rec = &mut ts.rec.planes[p];
      let input = &ts.input_tile.planes[p];
      let po = tile_bo.plane_offset(rec.plane_cfg);
      let edge_buf = get_intra_edges(
        &rec.as_const(),
        tile_bo,
        0,
        0,
        bsize,
        po,
        uv_tx_size,
        bit_depth,
        Some(PredictionMode::UV_CFL_PRED),
      );
      let mut alpha_cost = |alpha: i16| -> u64 {
        let mut rec_region =
          rec.subregion_mut(Area::BlockStartingAt { bo: tile_bo.0 });
        PredictionMode::UV_CFL_PRED.predict_intra(
          tile_rect,
          &mut rec_region,
          uv_tx_size,
          bit_depth,
          &ac.array,
          alpha,
          &edge_buf,
        );
        sse_wxh(
          &input.subregion(Area::BlockStartingAt { bo: tile_bo.0 }),
          &rec_region.as_const(),
          uv_tx_size.width(),
          uv_tx_size.height(),
          |_, _| 1., // We're not doing RDO here.
        )
      };
      let mut best = (alpha_cost(0), 0);
      let mut count = 2;
      for alpha in 1i16..=16i16 {
        let cost = (alpha_cost(alpha), alpha_cost(-alpha));
        if cost.0 < best.0 {
          best = (cost.0, alpha);
          count += 2;
        }
        if cost.1 < best.0 {
          best = (cost.1, -alpha);
          count += 2;
        }
        if count < alpha {
          break;
        }
      }
      best.1
    })
    .collect();

  if best_alpha[0] == 0 && best_alpha[1] == 0 {
    None
  } else {
    Some(CFLParams::from_alpha(best_alpha[0], best_alpha[1]))
  }
}

// RDO-based transform type decision
pub fn rdo_tx_type_decision<T: Pixel>(
  fi: &FrameInvariants<T>, ts: &mut TileStateMut<'_, T>,
  cw: &mut ContextWriter, mode: PredictionMode, ref_frames: [RefType; 2],
  mvs: [MotionVector; 2], bsize: BlockSize, tile_bo: TileBlockOffset,
  tx_size: TxSize, tx_set: TxSet, tx_types: &[TxType],
) -> (TxType, f64) {
  let mut best_type = TxType::DCT_DCT;
  let mut best_rd = std::f64::MAX;

  let PlaneConfig { xdec, ydec, .. } = ts.input.planes[1].cfg;
  let is_chroma_block = has_chroma(tile_bo, bsize, xdec, ydec);

  let is_inter = !mode.is_intra();

  let cw_checkpoint = cw.checkpoint();

  let rdo_type = if fi.use_tx_domain_distortion {
    RDOType::TxDistRealRate
  } else {
    RDOType::PixelDistRealRate
  };
  let need_recon_pixel = tx_size.block_size() != bsize;

  for &tx_type in tx_types {
    // Skip unsupported transform types
    if av1_tx_used[tx_set as usize][tx_type as usize] == 0 {
      continue;
    }

    if is_inter {
      motion_compensate(
        fi, ts, cw, mode, ref_frames, mvs, bsize, tile_bo, true,
      );
    }

    let wr: &mut dyn Writer = &mut WriterCounter::new();
    let tell = wr.tell_frac();
    let tx_dist = if is_inter {
      write_tx_tree(
        fi,
        ts,
        cw,
        wr,
        mode,
        tile_bo,
        bsize,
        tx_size,
        tx_type,
        false,
        true,
        rdo_type,
        need_recon_pixel,
      )
    } else {
      write_tx_blocks(
        fi,
        ts,
        cw,
        wr,
        mode,
        mode,
        tile_bo,
        bsize,
        tx_size,
        tx_type,
        false,
        CFLParams::default(), // Unused.
        true,
        rdo_type,
        need_recon_pixel,
      )
    };

    let rate = wr.tell_frac() - tell;
    let distortion = if fi.use_tx_domain_distortion {
      compute_tx_distortion(
        fi,
        ts,
        bsize,
        is_chroma_block,
        tile_bo,
        tx_dist,
        false,
        true,
      )
    } else {
      compute_distortion(fi, ts, bsize, is_chroma_block, tile_bo, true)
    };
    let rd = compute_rd_cost(fi, rate, distortion);
    if rd < best_rd {
      best_rd = rd;
      best_type = tx_type;
    }

    cw.rollback(&cw_checkpoint);
  }

  assert!(best_rd >= 0_f64);

  (best_type, best_rd)
}

pub fn get_sub_partitions(
  four_partitions: &[TileBlockOffset; 4], partition: PartitionType,
) -> Vec<TileBlockOffset> {
  let mut partitions = vec![four_partitions[0]];

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

pub fn get_sub_partitions_with_border_check(
  four_partitions: &[TileBlockOffset; 4], partition: PartitionType,
  mi_width: usize, mi_height: usize, subsize: BlockSize,
) -> Vec<TileBlockOffset> {
  let mut partitions = vec![four_partitions[0]];

  if partition == PARTITION_NONE {
    return partitions;
  }

  let hbsw = subsize.width_mi(); // Half the block size width in blocks
  let hbsh = subsize.height_mi(); // Half the block size height in blocks

  if (partition == PARTITION_VERT || partition == PARTITION_SPLIT)
    && four_partitions[1].0.x + hbsw <= mi_width
    && four_partitions[1].0.y + hbsh <= mi_height
  {
    partitions.push(four_partitions[1]);
  };

  if (partition == PARTITION_HORZ || partition == PARTITION_SPLIT)
    && four_partitions[2].0.x + hbsw <= mi_width
    && four_partitions[2].0.y + hbsh <= mi_height
  {
    partitions.push(four_partitions[2]);
  };

  if partition == PARTITION_SPLIT
    && four_partitions[3].0.x + hbsw <= mi_width
    && four_partitions[3].0.y + hbsh <= mi_height
  {
    partitions.push(four_partitions[3]);
  };

  partitions
}

// RDO-based single level partitioning decision
pub fn rdo_partition_decision<T: Pixel, W: Writer>(
  fi: &FrameInvariants<T>, ts: &mut TileStateMut<'_, T>,
  cw: &mut ContextWriter, w_pre_cdef: &mut W, w_post_cdef: &mut W,
  bsize: BlockSize, tile_bo: TileBlockOffset, cached_block: &RDOOutput,
  pmvs: &mut [[Option<MotionVector>; REF_FRAMES]; 5],
  partition_types: &[PartitionType], rdo_type: RDOType,
) -> RDOOutput {
  let mut best_partition = cached_block.part_type;
  let mut best_rd = cached_block.rd_cost;
  let mut best_pred_modes = cached_block.part_modes.clone();

  let cw_checkpoint = cw.checkpoint();
  let w_pre_checkpoint = w_pre_cdef.checkpoint();
  let w_post_checkpoint = w_post_cdef.checkpoint();

  for &partition in partition_types {
    // Do not re-encode results we already have
    if partition == cached_block.part_type {
      continue;
    }
    let mut cost: f64 = 0.0;
    let mut child_modes = std::vec::Vec::new();
    let mut early_exit = false;

    match partition {
      PartitionType::PARTITION_NONE => {
        if bsize > BlockSize::BLOCK_64X64 {
          continue;
        }

        let pmv_idx = if bsize > BlockSize::BLOCK_32X32 {
          0
        } else {
          ((tile_bo.0.x & 32) >> 5) + ((tile_bo.0.y & 32) >> 4) + 1
        };

        let spmvs = &mut pmvs[pmv_idx];

        let mode_decision =
          rdo_mode_decision(fi, ts, cw, bsize, tile_bo, spmvs);
        child_modes.push(mode_decision);
      }
      PARTITION_SPLIT | PARTITION_HORZ | PARTITION_VERT => {
        let subsize = bsize.subsize(partition);

        if subsize == BlockSize::BLOCK_INVALID {
          continue;
        }

        //pmv = best_pred_modes[0].mvs[0];

        assert!(best_pred_modes.len() <= 4);

        let hbsw = subsize.width_mi(); // Half the block size width in blocks
        let hbsh = subsize.height_mi(); // Half the block size height in blocks
        let four_partitions = [
          tile_bo,
          TileBlockOffset(BlockOffset {
            x: tile_bo.0.x + hbsw as usize,
            y: tile_bo.0.y,
          }),
          TileBlockOffset(BlockOffset {
            x: tile_bo.0.x,
            y: tile_bo.0.y + hbsh as usize,
          }),
          TileBlockOffset(BlockOffset {
            x: tile_bo.0.x + hbsw as usize,
            y: tile_bo.0.y + hbsh as usize,
          }),
        ];
        let partitions = get_sub_partitions_with_border_check(
          &four_partitions,
          partition,
          ts.mi_width,
          ts.mi_height,
          subsize,
        );

        let pmv_idxs = partitions
          .iter()
          .map(|&offset| {
            if subsize.greater_than(BlockSize::BLOCK_32X32) {
              0
            } else {
              ((offset.0.x & 32) >> 5) + ((offset.0.y & 32) >> 4) + 1
            }
          })
          .collect::<Vec<_>>();

        if bsize >= BlockSize::BLOCK_8X8 {
          let w: &mut W =
            if cw.bc.cdef_coded { w_post_cdef } else { w_pre_cdef };
          let tell = w.tell_frac();
          cw.write_partition(w, tile_bo, partition, bsize);
          cost = compute_rd_cost(fi, w.tell_frac() - tell, 0);
        }
        let mut rd_cost_sum = 0.0;

        for (&offset, pmv_idx) in partitions.iter().zip(pmv_idxs) {
          let mode_decision =
            rdo_mode_decision(fi, ts, cw, subsize, offset, &mut pmvs[pmv_idx]);

          rd_cost_sum += mode_decision.rd_cost;

          if fi.enable_early_exit && rd_cost_sum > best_rd {
            early_exit = true;
            break;
          }

          if subsize >= BlockSize::BLOCK_8X8 && subsize.is_sqr() {
            let w: &mut W =
              if cw.bc.cdef_coded { w_post_cdef } else { w_pre_cdef };
            cw.write_partition(
              w,
              offset,
              PartitionType::PARTITION_NONE,
              subsize,
            );
          }
          encode_block_with_modes(
            fi,
            ts,
            cw,
            w_pre_cdef,
            w_post_cdef,
            subsize,
            offset,
            &mode_decision,
            rdo_type,
            false,
          );
          child_modes.push(mode_decision);
        }
      }
      _ => {
        unreachable!();
      }
    }

    if !early_exit {
      let rd = cost + child_modes.iter().map(|m| m.rd_cost).sum::<f64>();

      if rd < best_rd {
        best_rd = rd;
        best_partition = partition;
        best_pred_modes = child_modes.clone();
      }
    }
    cw.rollback(&cw_checkpoint);
    w_pre_cdef.rollback(&w_pre_checkpoint);
    w_post_cdef.rollback(&w_post_checkpoint);
  }

  assert!(best_rd >= 0_f64);

  RDOOutput {
    rd_cost: best_rd,
    part_type: best_partition,
    part_modes: best_pred_modes,
  }
}

fn rdo_loop_plane_error<T: Pixel>(
  sbo: TileSuperBlockOffset, tile_sbo: TileSuperBlockOffset, sb_wh: usize,
  fi: &FrameInvariants<T>, ts: &TileStateMut<'_, T>, blocks: &TileBlocks<'_>,
  test: &Frame<T>, pli: usize,
) -> u64 {
  let sb_blocks =
    if fi.sequence.use_128x128_superblock { 16 } else { 8 } * sb_wh;
  // Each direction block is 8x8 in y, potentially smaller if subsampled in chroma
  // accumulating in-frame and unpadded
  let mut err: u64 = 0;
  for by in 0..sb_blocks {
    for bx in 0..sb_blocks {
      let bo = tile_sbo.block_offset(bx << 1, by << 1);
      if bo.0.x < blocks.cols() && bo.0.y < blocks.rows() {
        let in_plane = &ts.input_tile.planes[pli];
        let test_plane = &test.planes[pli];
        let &PlaneConfig { xdec, ydec, .. } = in_plane.plane_cfg;
        debug_assert_eq!(xdec, test_plane.cfg.xdec);
        debug_assert_eq!(ydec, test_plane.cfg.ydec);

        let in_bo = tile_sbo.block_offset(bx << 1, by << 1);
        let in_region =
          in_plane.subregion(Area::BlockStartingAt { bo: in_bo.0 });

        let test_bo = sbo.block_offset(bx << 1, by << 1);
        let test_region =
          test_plane.region(Area::BlockStartingAt { bo: test_bo.0 });

        let value = if pli == 0 {
          cdef_dist_wxh_8x8(&in_region, &test_region, fi.sequence.bit_depth)
        } else {
          // The closure returns 1. because we bias the distortion right
          // below.
          sse_wxh(&in_region, &test_region, 8 >> xdec, 8 >> ydec, |_, _| 1.)
        };

        let bias = compute_distortion_bias(
          fi,
          ts.to_frame_block_offset(bo),
          BlockSize::BLOCK_8X8,
        );
        err += (value as f64 * bias) as u64;
      }
    }
  }
  err
}

// Passed in a superblock offset representing the upper left corner of
// the LRU area we're optimizing.  This area covers the largest LRU in
// any of the present planes, but may consist of a number of
// superblocks and full, smaller LRUs in the other planes
pub fn rdo_loop_decision<T: Pixel>(
  tile_sbo: TileSuperBlockOffset, fi: &FrameInvariants<T>,
  ts: &mut TileStateMut<'_, T>, cw: &mut ContextWriter, w: &mut dyn Writer,
) {
  assert!(fi.sequence.enable_cdef || fi.sequence.enable_restoration);
  // Determine area of optimization: Which plane has the largest LRUs?
  // How many LRUs for each?
  let mut sb_wh = 1; // how many superblocks wide the largest LRU
                     // is/how much we're processing (same thing)
  let mut lru_wh = [0; PLANES]; // how manu LRUs we're processing
  for pli in 0..PLANES {
    let sb_shift = ts.restoration.planes[pli].rp_cfg.sb_shift;
    if sb_wh < (1 << sb_shift) {
      sb_wh = 1 << sb_shift;
    }
  }
  for pli in 0..PLANES {
    let sb_shift = ts.restoration.planes[pli].rp_cfg.sb_shift;
    let wh = 1 << sb_shift;
    lru_wh[pli] = sb_wh / wh;
  }

  let mut best_index = vec![vec![-1; sb_wh]; sb_wh];
  let mut best_lrf: Vec<Vec<Vec<RestorationFilter>>> = Vec::new();
  let sbo_0 = TileSuperBlockOffset(SuperBlockOffset { x: 0, y: 0 });

  for pli in 0..PLANES {
    best_lrf.push(Vec::new());
    for lrfy in 0..lru_wh[pli] {
      best_lrf[pli].push(Vec::new());
      for _lrfx in 0..lru_wh[pli] {
        best_lrf[pli][lrfy].push(RestorationFilter::None);
      }
    }
  }

  // Construct a largest-LRU-sized padded frame to filter from,
  // and a largest-LRU-sized padded frame to test-filter into
  // all stages; reconstruction goes to cdef so it must be additionally padded
  let mut cdef_input = None;
  let const_rec = ts.rec.as_const();
  let mut lrf_input = cdef_sb_frame(fi, sb_wh, &const_rec);
  let mut lrf_output = cdef_sb_frame(fi, sb_wh, &const_rec);
  if fi.sequence.enable_cdef {
    // min sized temporary frame; sb_wh number of superblocks with padding
    cdef_input =
      Some(cdef_sb_padded_frame_copy(fi, tile_sbo, sb_wh, &const_rec, 2));
  } else {
    // No cdef; copy lrf input from reconstructiton
    for pli in 0..PLANES {
      let po = tile_sbo.plane_offset(&ts.rec.planes[pli].plane_cfg);
      let rec_region =
        ts.rec.planes[pli].subregion(Area::StartingAt { x: po.x, y: po.y });
      let width = lrf_input.planes[pli].cfg.width.min(rec_region.rect().width);
      let height =
        lrf_input.planes[pli].cfg.height.min(rec_region.rect().height);
      for (rec, inp) in rec_region
        .rows_iter()
        .zip(lrf_input.planes[pli].as_region_mut().rows_iter_mut())
        .take(height)
      {
        inp[..width].copy_from_slice(&rec[..width]);
      }
      lrf_input.planes[pli].pad(width, height);
    }
  }

  // CDEF/LRF decision iteration
  // Start with a default of CDEF 0 and RestorationFilter::None
  // Try all CDEF options for each sb with current LRF; if new CDEF+LRF choice is better, select it.
  // Then try all LRF options with current CDEFs; if new CDEFs+LRF choice is better, select it.
  // If LRF choice changed for any plane, repeat until no changes
  // Limit iterations and where we break based on speed setting (in the TODO list ;-)
  let bd = fi.sequence.bit_depth;

  let cdef_data = cdef_input.as_ref().map(|input| {
    (
      input,
      cdef_analyze_superblock_range(
        input,
        &cw.bc.blocks.as_const(),
        sbo_0,
        tile_sbo,
        sb_wh,
        bd,
      ),
    )
  });

  let mut cdef_change = true;
  let mut lrf_change = true;
  while cdef_change || lrf_change {
    // check for [new] cdef indices if cdef is enabled.
    if let Some((cdef_input, cdef_dirs)) = cdef_data.as_ref() {
      for sby in 0..sb_wh {
        for sbx in 0..sb_wh {
          let prev_best_index = best_index[sby][sbx];
          let loop_sbo =
            TileSuperBlockOffset(SuperBlockOffset { x: sbx, y: sby });
          let loop_tile_sbo = TileSuperBlockOffset(SuperBlockOffset {
            x: tile_sbo.0.x + sbx,
            y: tile_sbo.0.y + sby,
          });
          let mut best_cost = -1.;
          let mut best_new_index = -1i8;

          for cdef_index in 0..(1 << fi.cdef_bits) {
            let mut err = 0;
            let mut rate = 0;
            cdef_filter_superblock(
              fi,
              &cdef_input,
              &mut lrf_input,
              &cw.bc.blocks.as_const(),
              loop_sbo,
              loop_tile_sbo,
              cdef_index,
              &cdef_dirs[sby][sbx],
            );
            // apply LRF if any
            for pli in 0..PLANES {
              let wh =
                if fi.sequence.use_128x128_superblock { 128 } else { 64 };
              let xdec = lrf_input.planes[pli].cfg.xdec;
              let ydec = lrf_input.planes[pli].cfg.ydec;
              let width = (wh + (1 << xdec >> 1)) >> xdec;
              let height = (wh + (1 << ydec >> 1)) >> ydec;
              // which LRU are we currently testing against?
              let rp = &ts.restoration.planes[pli];
              if let (
                Some((tile_lru_x, tile_lru_y)),
                Some((loop_tile_lru_x, loop_tile_lru_y)),
              ) = (
                rp.restoration_unit_index(tile_sbo, false),
                rp.restoration_unit_index(loop_tile_sbo, false),
              ) {
                let lru_x = loop_tile_lru_x - tile_lru_x;
                let lru_y = loop_tile_lru_y - tile_lru_y;

                match best_lrf[pli][lru_y][lru_x] {
                  RestorationFilter::None {} => {
                    err += rdo_loop_plane_error(
                      loop_sbo,
                      loop_tile_sbo,
                      1,
                      fi,
                      ts,
                      &cw.bc.blocks.as_const(),
                      &lrf_input,
                      pli,
                    );

                    rate += if fi.sequence.enable_restoration {
                      cw.count_lrf_switchable(
                        w,
                        &ts.restoration.as_const(),
                        best_lrf[pli][lru_y][lru_x],
                        pli,
                      )
                    } else {
                      0 // no relative cost differeneces to different
                        // CDEF params.  If cdef is on, it's a wash.
                    };
                  }
                  RestorationFilter::Sgrproj { set, xqd } => {
                    // only run on this superblock
                    // if height is 128x128, we'll need to run two stripes
                    let loop_po =
                      loop_sbo.plane_offset(&lrf_input.planes[pli].cfg);
                    if height > 64 {
                      let loop_po2 =
                        PlaneOffset { x: loop_po.x, y: loop_po.y + 64 };
                      sgrproj_stripe_filter(
                        set,
                        xqd,
                        fi,
                        &mut ts.stripe_filter_buffer,
                        width,
                        64,
                        width,
                        64,
                        &lrf_input.planes[pli].slice(loop_po),
                        &lrf_input.planes[pli].slice(loop_po),
                        &mut lrf_output.planes[pli].mut_slice(loop_po),
                      );
                      sgrproj_stripe_filter(
                        set,
                        xqd,
                        fi,
                        &mut ts.stripe_filter_buffer,
                        width,
                        64,
                        width,
                        64,
                        &lrf_input.planes[pli].slice(loop_po2),
                        &lrf_input.planes[pli].slice(loop_po2),
                        &mut lrf_output.planes[pli].mut_slice(loop_po2),
                      );
                    } else {
                      sgrproj_stripe_filter(
                        set,
                        xqd,
                        fi,
                        &mut ts.stripe_filter_buffer,
                        width,
                        height,
                        width,
                        height,
                        &lrf_input.planes[pli].slice(loop_po),
                        &lrf_input.planes[pli].slice(loop_po),
                        &mut lrf_output.planes[pli].mut_slice(loop_po),
                      );
                    }
                  }
                  RestorationFilter::Wiener { .. } => unreachable!(), // coming soon
                }
                let this_err = rdo_loop_plane_error(
                  loop_sbo,
                  loop_tile_sbo,
                  1,
                  fi,
                  ts,
                  &cw.bc.blocks.as_const(),
                  &lrf_output,
                  pli,
                );
                err += this_err;
                let this_rate = cw.count_lrf_switchable(
                  w,
                  &ts.restoration.as_const(),
                  best_lrf[pli][lru_y][lru_x],
                  pli,
                );
                rate += this_rate;
              } else {
                // No LRU here, compute error directly from CDEF output.
                err += rdo_loop_plane_error(
                  loop_sbo,
                  loop_tile_sbo,
                  1,
                  fi,
                  ts,
                  &cw.bc.blocks.as_const(),
                  &lrf_input,
                  pli,
                );
                // no aelative cost differeneces to different
                // CDEF params.  If cdef is on, it's a wash.
                // rate += 0;
              }
            }

            let cost = compute_rd_cost(fi, rate, err);
            if best_cost < 0. || cost < best_cost {
              best_cost = cost;
              best_new_index = cdef_index as i8;
            }
          }

          if best_new_index != prev_best_index {
            cdef_change = true;
            best_index[sby][sbx] = best_new_index;
            cw.bc.blocks.set_cdef(loop_tile_sbo, best_new_index as u8);
          }
        }
      }
    }

    if !cdef_change {
      break;
    }
    cdef_change = false;
    lrf_change = false;

    // check for new best restoration filter if enabled
    if fi.sequence.enable_restoration {
      // need cdef output from best index, not just last iteration

      if let Some((cdef_input, cdef_dirs)) = cdef_data.as_ref() {
        for sby in 0..sb_wh {
          for sbx in 0..sb_wh {
            let loop_sbo =
              TileSuperBlockOffset(SuperBlockOffset { x: sbx, y: sby });
            let loop_tile_sbo = TileSuperBlockOffset(SuperBlockOffset {
              x: tile_sbo.0.x + sbx,
              y: tile_sbo.0.y + sby,
            });
            cdef_filter_superblock(
              fi,
              &cdef_input,
              &mut lrf_input,
              &cw.bc.blocks.as_const(),
              loop_sbo,
              loop_tile_sbo,
              best_index[sby][sbx] as u8,
              &cdef_dirs[sby][sbx],
            );
          }
        }
      }

      for pli in 0..PLANES {
        let sb_shift = ts.restoration.planes[pli].rp_cfg.sb_shift;
        let stripe_height = ts.restoration.planes[pli].rp_cfg.stripe_height;
        let unit_size = ts.restoration.planes[pli].rp_cfg.unit_size;
        let wh = 1 << sb_shift; // width/height, in sb, of this LRU in this plane
        for lru_y in 0..lru_wh[pli] {
          // number of LRUs vertically
          for lru_x in 0..lru_wh[pli] {
            // number of LRUs horizontally
            let loop_sbo = TileSuperBlockOffset(SuperBlockOffset {
              x: lru_x * wh,
              y: lru_y * wh,
            });
            let loop_tile_sbo = TileSuperBlockOffset(SuperBlockOffset {
              x: tile_sbo.0.x + loop_sbo.0.x,
              y: tile_sbo.0.y + loop_sbo.0.y,
            });
            if fi.sequence.enable_restoration
              && ts.restoration.has_restoration_unit(loop_tile_sbo, pli, false)
            {
              let ref_plane = &ts.input.planes[pli]; // reference
              let lrf_in_plane = &lrf_input.planes[pli];
              let loop_po = loop_sbo.plane_offset(&lrf_in_plane.cfg);
              let loop_tile_po = loop_tile_sbo.plane_offset(&ref_plane.cfg);
              let mut best_new_lrf = RestorationFilter::None;
              let err = rdo_loop_plane_error(
                loop_sbo,
                loop_tile_sbo,
                wh,
                fi,
                ts,
                &cw.bc.blocks.as_const(),
                &lrf_input,
                pli,
              );
              let rate = cw.count_lrf_switchable(
                w,
                &ts.restoration.as_const(),
                best_new_lrf,
                pli,
              );
              let mut best_cost = compute_rd_cost(fi, rate, err);

              for set in 0..16 {
                // clip to encoded area
                let unit_width =
                  unit_size.min(ref_plane.cfg.width - loop_tile_po.x as usize);
                let unit_height = unit_size
                  .min(ref_plane.cfg.height - loop_tile_po.y as usize);
                let (xqd0, xqd1) = sgrproj_solve(
                  set,
                  fi,
                  &mut ts.solve_buffer,
                  &ref_plane.slice(loop_tile_po),
                  &lrf_in_plane.slice(loop_po),
                  unit_width,
                  unit_height,
                );
                let current_lrf =
                  RestorationFilter::Sgrproj { set, xqd: [xqd0, xqd1] };
                if let RestorationFilter::Sgrproj { set, xqd } = current_lrf {
                  // one stripe at a time
                  // do not consider any stretch; we might fall off the tile
                  for y in (0..unit_height).step_by(stripe_height) {
                    let stripe_po =
                      PlaneOffset { x: loop_po.x, y: loop_po.y + y as isize };
                    let stripe_h = unit_size.min(stripe_height);
                    sgrproj_stripe_filter(
                      set,
                      xqd,
                      fi,
                      &mut ts.solve_buffer,
                      unit_width,
                      stripe_h,
                      unit_width,
                      stripe_h,
                      &lrf_input.planes[pli].slice(stripe_po),
                      &lrf_input.planes[pli].slice(stripe_po),
                      &mut lrf_output.planes[pli].mut_slice(stripe_po),
                    );
                  }
                }
                let err = rdo_loop_plane_error(
                  loop_sbo,
                  loop_tile_sbo,
                  wh,
                  fi,
                  ts,
                  &cw.bc.blocks.as_const(),
                  &lrf_output,
                  pli,
                );
                let rate = cw.count_lrf_switchable(
                  w,
                  &ts.restoration.as_const(),
                  current_lrf,
                  pli,
                );
                let cost = compute_rd_cost(fi, rate, err);

                if cost < best_cost {
                  best_cost = cost;
                  best_new_lrf = current_lrf;
                }
              }

              if best_lrf[pli][lru_y][lru_x].notequal(best_new_lrf) {
                best_lrf[pli][lru_y][lru_x] = best_new_lrf;
                lrf_change = true;
                if let Some(ru) = ts.restoration.planes[pli]
                  .restoration_unit_mut(loop_tile_sbo)
                {
                  ru.filter = best_new_lrf;
                }
              }
            }
          }
        }
      }
    }
  }
}

#[test]
fn estimate_rate_test() {
  assert_eq!(estimate_rate(0, TxSize::TX_4X4, 0), RDO_RATE_TABLE[0][0][0]);
}
