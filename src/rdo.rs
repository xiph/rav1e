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
use me::*;
use ec::OD_BITRES;
use ec::Writer;
use ec::WriterCounter;
use luma_ac;
use encode_block_a;
use encode_block_b;
use motion_compensate;
use partition::*;
use plane::*;
use cdef::*;
use predict::{RAV1E_INTRA_MODES, RAV1E_INTRA_MODES_MINIMAL, RAV1E_INTER_MODES_MINIMAL};
use quantize::dc_q;
use std;
use std::f64;
use std::vec::Vec;
use std::iter::*;
use write_tx_blocks;
use write_tx_tree;
use partition::BlockSize;
use Frame;
use FrameInvariants;
use FrameState;
use FrameType;
use Tune;
use Sequence;
#[derive(Clone, Copy, PartialEq)]
pub enum RDOType {
  Fast,
  Accurate
}
pub static RDO_DISTORTION_TABLE: [[u64; rdo_num_bins]; TxSize::TX_SIZES_ALL] = [
[0,247,280,297,314,328,337,332,332,334,322,314,313,296,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,71,],
[0,483,606,686,746,790,820,841,857,871,879,885,888,892,895,893,892,896,900,899,899,897,895,883,879,884,876,879,884,876,872,856,873,854,875,855,853,838,816,805,797,821,767,757,748,753,747,99999,99999,95,],
[0,521,917,1130,1277,1472,1637,1786,1924,2043,2170,2278,2391,2490,2566,2644,2709,2773,2837,2907,2968,3017,3077,3113,3173,3208,3249,3282,3319,3353,3380,3417,3431,3452,3476,3499,3526,3528,3546,3553,3584,3588,3599,3599,3608,3630,3640,3630,3640,3658,],
[0,99999,671,945,1169,1528,1787,1821,2305,2812,3092,3533,3957,4271,4294,4353,4440,4515,4599,4854,4911,5051,5248,5481,5593,5879,5992,6015,6233,6364,6559,6626,6864,6927,7044,7154,7347,7400,7555,7664,7571,7816,8027,8235,8166,8320,8475,8505,8695,13003,],
[0,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[0,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[0,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[0,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[0,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[0,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[0,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[0,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[0,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[0,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[0,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[0,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[0,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[0,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[0,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
];
pub static RDO_RATE_TABLE: [[u64; rdo_num_bins]; TxSize::TX_SIZES_ALL] = [
[411,614,716,748,755,727,710,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,136,],
[406,827,1144,1363,1513,1623,1707,1763,1789,1803,1788,1750,1729,1718,1700,1652,1658,1627,1599,1526,1507,1404,1374,1418,1408,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,150,],
[176,397,780,1166,1528,1885,2232,2531,2817,3110,3395,3672,3931,4155,4379,4573,4710,4880,4999,5102,5237,5324,5424,5498,5570,5646,5704,5779,5858,5866,5958,5949,5984,5959,5988,6036,6023,5935,5939,5980,5866,5840,5921,5791,5786,5815,5703,5678,5676,2246,],
[135,169,218,315,442,582,833,1134,1387,1699,1972,2366,2698,2919,3231,3557,3924,4195,4496,4724,4968,5363,5614,5831,6139,6437,6704,6879,7284,7630,7889,7951,8274,8481,8746,8992,9252,9434,9739,9763,10311,10438,10524,10963,11005,11341,11644,11565,11825,17430,],
[99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
[99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,],
];



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
  pub pred_cfl_params: CFLParams,
  pub ref_frame: usize,
  pub mv: MotionVector,
  pub skip: bool
}

const rdo_num_bins: usize =  50;
const rdo_max_bin: usize = 10000;
const DIST_EST_MAX_BIN: usize = 10000;
const RATE_EST_MAX_BIN: usize = 20000;
const rdo_bin_size: u64 = (rdo_max_bin / rdo_num_bins) as u64;
const DIST_EST_BIN_SIZE: u64 = (DIST_EST_MAX_BIN / rdo_num_bins) as u64;
const RATE_EST_BIN_SIZE: u64 = (RATE_EST_MAX_BIN / rdo_num_bins) as u64;


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RDOTracker {
  rate_bins: Vec<Vec<u64>>,
  rate_counts: Vec<Vec<u64>>,
  dist_bins: Vec
        <Vec<u64>>,
  dist_counts: Vec<Vec<u64>>
}

impl RDOTracker {
  pub fn new() -> RDOTracker {
    RDOTracker {
      rate_bins: vec![vec![0; rdo_num_bins]; TxSize::TX_SIZES_ALL],
      rate_counts: vec![vec![0; rdo_num_bins]; TxSize::TX_SIZES_ALL],
      dist_bins: vec![vec![0; rdo_num_bins]; TxSize::TX_SIZES_ALL],
      dist_counts: vec![vec![0; rdo_num_bins]; TxSize::TX_SIZES_ALL]
    }
  }
  fn merge_array(new: &mut Vec<u64>, old: &Vec<u64>) {
    for (n, o) in new.iter_mut().zip(old.iter()) {
      *n += o;
    }
  }
  fn merge_2d_array(new: &mut Vec<Vec<u64>>, old: &Vec<Vec<u64>>) {
    for (n, o) in new.iter_mut().zip(old.iter()) {
      RDOTracker::merge_array(n, o);
    }
  }
  pub fn merge_in(&mut self, input: &RDOTracker) {
    RDOTracker::merge_2d_array(&mut self.rate_bins, &input.rate_bins);
    RDOTracker::merge_2d_array(&mut self.rate_counts, &input.rate_counts);
    RDOTracker::merge_2d_array(&mut self.dist_bins, &input.dist_bins);
    RDOTracker::merge_2d_array(&mut self.dist_counts, &input.dist_counts);
  }
  pub fn add_rate(&mut self, ts: TxSize, fast_distortion: u64, rate: u64) {
    if fast_distortion != 0 {
      let bs_index = ts as usize;
      let bin_idx_tmp = (((fast_distortion as i64 - (RATE_EST_BIN_SIZE as i64) / 2)) as u64 / RATE_EST_BIN_SIZE) as usize;
      let bin_idx = if bin_idx_tmp >= rdo_num_bins {
        rdo_num_bins - 1
      } else {
        bin_idx_tmp
      };
      self.rate_counts[bs_index][bin_idx] += 1;
      self.rate_bins[bs_index][bin_idx] += rate;
    }
  }
  pub fn estimate_rate(&self, ts: TxSize, fast_distortion: u64) -> u64 {
    let bs_index = ts as usize;
    let bin_idx_down = ((fast_distortion) / RATE_EST_BIN_SIZE).min((rdo_num_bins - 2) as u64);
    let bin_idx_up = (bin_idx_down + 1).min((rdo_num_bins - 1) as u64);
    let x0 = (bin_idx_down * RATE_EST_BIN_SIZE) as i64;
    let x1 = (bin_idx_up * RATE_EST_BIN_SIZE) as i64;
    let y0 = RDO_RATE_TABLE[bs_index][bin_idx_down as usize] as i64;
    let y1 = RDO_RATE_TABLE[bs_index][bin_idx_up as usize] as i64;
    let slope = ((y1 - y0) << 8) / (x1 - x0);
    (y0 + (((fast_distortion as i64 - x0) * slope) >> 8)) as u64
  }
  pub fn add_distortion(&mut self, ts: TxSize, fast_distortion: u64, distortion: u64) {
    if fast_distortion != 0 {
      let bs_index = ts as usize;
      let bin_idx_tmp = (((fast_distortion as i64 - (DIST_EST_BIN_SIZE as i64) / 2)) as u64 / DIST_EST_BIN_SIZE) as usize;
      let bin_idx = if bin_idx_tmp >= rdo_num_bins {
        rdo_num_bins - 1
      } else {
        bin_idx_tmp
      };
      self.dist_counts[bs_index][bin_idx] += 1;
      self.dist_bins[bs_index][bin_idx] += distortion;
    }
  }
  pub fn estimate_distortion(&self, ts: TxSize, fast_distortion: u64) -> u64 {
    let bs_index = ts as usize;
    let bin_idx_down = ((fast_distortion) / DIST_EST_BIN_SIZE).min((rdo_num_bins - 2) as u64);
    let bin_idx_up = (bin_idx_down + 1).min((rdo_num_bins - 1) as u64);
    let x0 = (bin_idx_down * DIST_EST_BIN_SIZE) as i64;
    let x1 = (bin_idx_up * DIST_EST_BIN_SIZE) as i64;
    let y0 = RDO_DISTORTION_TABLE[bs_index][bin_idx_down as usize] as i64;
    let y1 = RDO_DISTORTION_TABLE[bs_index][bin_idx_up as usize] as i64;
    let slope = ((y1 - y0) << 8) / (x1 - x0);
    (y0 + (((fast_distortion as i64 - x0) * slope) >> 8)) as u64
  }
  pub fn print_distortion(&self) {
    let bs_index = TxSize::TX_32X32 as usize;
    for (bin_idx, (dist_total, dist_count)) in self.dist_bins[bs_index].iter().zip(self.dist_counts[bs_index].iter()).enumerate() {
      if *dist_count != 0 {
        println!("{} {}", bin_idx, dist_total / dist_count);
      }
    }
  }
  pub fn print_rate(&self) {
    let bs_index = 0;
    for (bin_idx, (rate_total, rate_count)) in self.rate_bins[bs_index].iter().zip(self.rate_counts[bs_index].iter()).enumerate() {
      if *rate_count != 0 {
        println!("{} {}", bin_idx, rate_total / rate_count);
      }
    }
  }
  pub fn print_code(&self) {
    println!("pub static RDO_DISTORTION_TABLE: [[u64; rdo_num_bins]; TxSize::TX_SIZES_ALL] = [");
    for bs_index in 0..TxSize::TX_SIZES_ALL {
      print!("[");
      for (bin_idx, (dist_total, dist_count)) in self.dist_bins[bs_index].iter().zip(self.dist_counts[bs_index].iter()).enumerate() {
        if bin_idx == 0 {
          print!("0,"); // we know zero SAD equals zero distortion
        } else if *dist_count > 100 {
          print!("{},", dist_total / dist_count);
        } else {
          print!("99999,"); // ensure mode isn't selected
        }
      }
      println!("],");
    }
    println!("];");
    println!("pub static RDO_RATE_TABLE: [[u64; rdo_num_bins]; TxSize::TX_SIZES_ALL] = [");
    for bs_index in 0..TxSize::TX_SIZES_ALL {
        print!("[");
        for (bin_idx, (rate_total, rate_count)) in self.rate_bins[bs_index].iter().zip(self.rate_counts[bs_index].iter()).enumerate() {
            if *rate_count > 100 {
                print!("{},", rate_total / rate_count);
            } else {
                print!("99999,");
            }
        }
        println!("],");
    }
    println!("];");
  }
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
fn sse_wxh(
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

pub fn compute_fast_distortion(
  refr: PlaneSlice, pred: PlaneSlice, w_y: usize, h_y: usize) -> u64 {
    let mut sad = 0 as u32;
    let mut plane_org = pred;
    let mut plane_ref = refr;

    for _r in 0..h_y {
        {
            let slice_org = plane_org.as_slice_w_width(w_y);
            let slice_ref = plane_ref.as_slice_w_width(w_y);
            sad += slice_org.iter().zip(slice_ref).map(|(&a, &b)| (a as i32 - b as i32).abs() as u32).sum::<u32>();
        }
        plane_org.y += 1;
        plane_ref.y += 1;
    }
  sad as u64
}

fn estimate_rd_cost(fi: &FrameInvariants, bit_depth: usize,
  bit_cost: u32, estimated_distortion: u64
) -> (u64, f64) {
  let q = dc_q(fi.config.quantizer as u8, bit_depth) as f64;

  // Convert q into Q0 precision, given that libaom quantizers are Q3
  let q0 = q / 8.0_f64;

  // Lambda formula from doc/theoretical_results.lyx in the daala repo
  // Use Q0 quantizer since lambda will be applied to Q0 pixel domain
  let lambda = q0 * q0 * std::f64::consts::LN_2 / 6.0;
  // Compute rate
  let rate = (bit_cost as f64) / ((1 << OD_BITRES) as f64);
  (estimated_distortion, (estimated_distortion as f64) + lambda * rate)
}

pub fn get_lambda(fi: &FrameInvariants, bit_depth: usize) -> f64 {
  let q = dc_q(fi.base_q_idx, bit_depth) as f64;

  // Convert q into Q0 precision, given that libaom quantizers are Q3
  let q0 = q / 8.0_f64;

  // Lambda formula from doc/theoretical_results.lyx in the daala repo
  // Use Q0 quantizer since lambda will be applied to Q0 pixel domain
  q0 * q0 * std::f64::consts::LN_2 / 6.0
}

// Compute the rate-distortion cost for an encode
fn compute_rd_cost(
  fi: &FrameInvariants, fs: &FrameState, w_y: usize, h_y: usize,
  is_chroma_block: bool, bo: &BlockOffset, bit_cost: u32, bit_depth: usize,
  luma_only: bool
) -> (u64, f64) {
  let lambda = get_lambda(fi, bit_depth);

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
      bit_depth
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

  (distortion,(distortion as f64) + lambda * rate)
}

pub fn rdo_tx_size_type(
  seq: &Sequence, fi: &FrameInvariants, fs: &mut FrameState,
  cw: &mut ContextWriter, bsize: BlockSize, bo: &BlockOffset,
  luma_mode: PredictionMode, ref_frame: usize, mv: MotionVector, skip: bool
) -> (TxSize, TxType) {
  // these rules follow TX_MODE_LARGEST
  let tx_size = match bsize {
    BlockSize::BLOCK_4X4 => TxSize::TX_4X4,
    BlockSize::BLOCK_8X8 => TxSize::TX_8X8,
    BlockSize::BLOCK_16X16 => TxSize::TX_16X16,
    _ => TxSize::TX_32X32
  };
  cw.bc.set_tx_size(bo, tx_size);
  // Were we not hardcoded to TX_MODE_LARGEST, block tx size would be written here

  // Luma plane transform type decision
  let is_inter = !luma_mode.is_intra();
  let tx_set = get_tx_set(tx_size, is_inter, fi.use_reduced_tx_set);

  let tx_type =
    if tx_set > TxSet::TX_SET_DCTONLY && fi.config.speed <= 3 && !skip {
      rdo_tx_type_decision(
        fi,
        fs,
        cw,
        luma_mode,
        ref_frame,
        mv,
        bsize,
        bo,
        tx_size,
        tx_set,
        seq.bit_depth
      )
    } else {
      TxType::DCT_DCT
    };

  (tx_size, tx_type)
}

struct EncodingSettings {
  mode_luma: PredictionMode,
  mode_chroma: PredictionMode,
  cfl_params: CFLParams,
  skip: bool,
  rd: f64,
  ref_frame: usize,
  mv: MotionVector,
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
      ref_frame: INTRA_FRAME,
      mv: MotionVector { row: 0, col: 0 },
      tx_size: TxSize::TX_4X4,
      tx_type: TxType::DCT_DCT
    }
  }
}
// RDO-based mode decision
pub fn rdo_mode_decision(
  seq: &Sequence, fi: &FrameInvariants, fs: &mut FrameState,
  cw: &mut ContextWriter, bsize: BlockSize, bo: &BlockOffset,
  pmv: &MotionVector
) -> RDOOutput {
  let mut best = EncodingSettings::default();
  let rdo_type = if fi.config.speed == 0 {
    RDOType::Accurate
  } else { RDOType::Fast };

  // Get block luma and chroma dimensions
  let w = bsize.width();
  let h = bsize.height();

  let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;
  let is_chroma_block = has_chroma(bo, bsize, xdec, ydec);

  let cw_checkpoint = cw.checkpoint();

  // Exclude complex prediction modes at higher speed levels
  let intra_mode_set = if (fi.frame_type == FrameType::KEY
    && fi.config.speed <= 3)
    || (fi.frame_type == FrameType::INTER && fi.config.speed <= 1)
  {
    RAV1E_INTRA_MODES
  } else {
    RAV1E_INTRA_MODES_MINIMAL
  };

  let mut ref_frame_set = Vec::new();
  let mut ref_slot_set = Vec::new();

  if fi.frame_type == FrameType::INTER {
    for i in LAST_FRAME..NONE_FRAME {
      if !ref_slot_set.contains(&fi.ref_frames[i - LAST_FRAME]) {
        ref_frame_set.push(i);
        ref_slot_set.push(fi.ref_frames[i - LAST_FRAME]);
      }
    }
    assert!(ref_frame_set.len() != 0);
  }

  let mut mode_set: Vec<(PredictionMode, usize)> = Vec::new();
  let mut mv_stacks = Vec::new();
  let mut mode_contexts = Vec::new();

  for (i, &ref_frame) in ref_frame_set.iter().enumerate() {
    let mut mvs: Vec<CandidateMV> = Vec::new();
    mode_contexts.push(cw.find_mvrefs(bo, ref_frame, &mut mvs, bsize, false));

    if fi.frame_type == FrameType::INTER {
      for &x in RAV1E_INTER_MODES_MINIMAL {
        mode_set.push((x, i));
      }
      if fi.config.speed <= 2 {
        if mvs.len() >= 3 {
          mode_set.push((PredictionMode::NEAR1MV, i));
        }
        if mvs.len() >= 4 {
          mode_set.push((PredictionMode::NEAR2MV, i));
        }
      }
    }
    mv_stacks.push(mvs);
  }

  let luma_rdo = |luma_mode: PredictionMode, fs: &mut FrameState, cw: &mut ContextWriter, best: &mut EncodingSettings,
    mv: MotionVector, ref_frame: usize, mode_set_chroma: &[PredictionMode], luma_mode_is_intra: bool,
    mode_context: usize, mv_stack: &Vec<CandidateMV>| {
    let (tx_size, tx_type) = rdo_tx_size_type(
      seq, fi, fs, cw, bsize, bo, luma_mode, ref_frame, mv, false,
    );

    // Find the best chroma prediction mode for the current luma prediction mode
    let mut chroma_rdo = |skip: bool| {
      mode_set_chroma.iter().for_each(|&chroma_mode| {
        let wr: &mut dyn Writer = &mut WriterCounter::new();
        let tell = wr.tell_frac();

        encode_block_a(seq, cw, wr, bsize, bo, skip);
        let tell_coeffs = wr.tell_frac();
        let (fast_distortion, estimated_distortion) = encode_block_b(
          seq,
          fi,
          fs,
          cw,
          wr,
          luma_mode,
          chroma_mode,
          ref_frame,
          mv,
          bsize,
          bo,
          skip,
          seq.bit_depth,
          CFLParams::new(),
          tx_size,
          tx_type,
          mode_context,
          mv_stack,
          rdo_type,
        );
        let cost_coeffs = wr.tell_frac() - tell_coeffs;
        let cost = wr.tell_frac() - tell;
        let (distortion, rd) = match rdo_type {
          RDOType::Accurate => compute_rd_cost(
            fi,
            fs,
            w,
            h,
            is_chroma_block,
            bo,
            cost,
            seq.bit_depth,
            false
          ),
          RDOType::Fast => estimate_rd_cost(fi, seq.bit_depth, cost, estimated_distortion)
        };
        if rd < best.rd {
          best.rd = rd;
          best.mode_luma = luma_mode;
          best.mode_chroma = chroma_mode;
          best.ref_frame = ref_frame;
          best.mv = mv;
          best.skip = skip;
          best.tx_size = tx_size;
          best.tx_type = tx_type;
        }
        //let (distortion2, rd2) = estimate_rd_cost(fi, seq.bit_depth, cost, estimated_distortion);
        //println!("{} {}", distortion, estimated_distortion);
        fs.t.add_distortion(tx_size, fast_distortion, distortion);
        fs.t.add_rate(tx_size, fast_distortion, cost_coeffs as u64);

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
    let mv = match luma_mode {
      PredictionMode::NEWMV => motion_estimation(fi, fs, bsize, bo, ref_frame_set[i], pmv),
      PredictionMode::NEARESTMV => if mv_stacks[i].len() > 0 {
        mv_stacks[i][0].this_mv
      } else {
        MotionVector { row: 0, col: 0 }
      },
      PredictionMode::NEAR0MV => if mv_stacks[i].len() > 1 {
        mv_stacks[i][1].this_mv
      } else {
        MotionVector { row: 0, col: 0 }
      },
      PredictionMode::NEAR1MV | PredictionMode::NEAR2MV =>
          mv_stacks[i][luma_mode as usize - PredictionMode::NEAR0MV as usize + 1].this_mv,
      _ => MotionVector { row: 0, col: 0 }
    };
    let mode_set_chroma = vec![luma_mode];

    luma_rdo(luma_mode, fs, cw, &mut best, mv, ref_frame_set[i], &mode_set_chroma, false,
             mode_contexts[i], &mv_stacks[i]);
  });

  if !best.skip {
    intra_mode_set.iter().for_each(|&luma_mode| {
      let mv = MotionVector { row: 0, col: 0 };
      let mut mode_set_chroma = vec![luma_mode];
      if is_chroma_block && luma_mode != PredictionMode::DC_PRED {
        mode_set_chroma.push(PredictionMode::DC_PRED);
      }
      luma_rdo(luma_mode, fs, cw, &mut best, mv, INTRA_FRAME, &mode_set_chroma, true,
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
      seq.bit_depth,
      CFLParams::new(),
      true,
      rdo_type
    );
    cw.rollback(&cw_checkpoint);
    if let Some(cfl) = rdo_cfl_alpha(fs, bo, bsize, seq.bit_depth) {
      let mut wr: &mut dyn Writer = &mut WriterCounter::new();
      let tell = wr.tell_frac();

      encode_block_a(seq, cw, wr, bsize, bo, best.skip);
      let (fast_distortion, estimated_distortion) = encode_block_b(
        seq,
        fi,
        fs,
        cw,
        wr,
        best.mode_luma,
        chroma_mode,
        best.ref_frame,
        best.mv,
        bsize,
        bo,
        best.skip,
        seq.bit_depth,
        cfl,
        best.tx_size,
        best.tx_type,
        0,
        &Vec::new(),
        rdo_type
      );

      let cost = wr.tell_frac() - tell;
      let (_, rd) = compute_rd_cost(
        fi,
        fs,
        w,
        h,
        is_chroma_block,
        bo,
        cost,
        seq.bit_depth,
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
  cw.bc.set_ref_frame(bo, bsize, best.ref_frame);
  cw.bc.set_motion_vector(bo, bsize, best.mv);

  assert!(best.rd >= 0_f64);

  RDOOutput {
    rd_cost: best.rd,
    part_type: PartitionType::PARTITION_NONE,
    part_modes: vec![RDOPartitionOutput {
      bo: bo.clone(),
      pred_mode_luma: best.mode_luma,
      pred_mode_chroma: best.mode_chroma,
      pred_cfl_params: best.cfl_params,
      ref_frame: best.ref_frame,
      mv: best.mv,
      rd_cost: best.rd,
      skip: best.skip
    }]
  }
}

pub fn rdo_cfl_alpha(
  fs: &mut FrameState, bo: &BlockOffset, bsize: BlockSize, bit_depth: usize
) -> Option<CFLParams> {
  // TODO: these are only valid for 4:2:0
  let uv_tx_size = match bsize {
    BlockSize::BLOCK_4X4 | BlockSize::BLOCK_8X8 => TxSize::TX_4X4,
    BlockSize::BLOCK_16X16 => TxSize::TX_8X8,
    BlockSize::BLOCK_32X32 => TxSize::TX_16X16,
    _ => TxSize::TX_32X32
  };

  let mut ac = [0i16; 32 * 32];
  luma_ac(&mut ac, fs, bo, bsize);
  let best_alpha: Vec<i16> = (1..3)
    .map(|p| {
      let rec = &mut fs.rec.planes[p];
      let input = &fs.input.planes[p];
      let po = bo.plane_offset(&fs.input.planes[p].cfg);
      (-16i16..17i16)
        .min_by_key(|&alpha| {
          PredictionMode::UV_CFL_PRED.predict_intra(
            &mut rec.mut_slice(&po),
            uv_tx_size,
            bit_depth,
            &ac,
            alpha
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
  mode: PredictionMode, ref_frame: usize, mv: MotionVector, bsize: BlockSize, bo: &BlockOffset, tx_size: TxSize,
  tx_set: TxSet, bit_depth: usize
) -> TxType {
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

    motion_compensate(fi, fs, cw, mode, ref_frame, mv, bsize, bo, bit_depth, true);

    let mut wr: &mut dyn Writer = &mut WriterCounter::new();
    let tell = wr.tell_frac();
    if is_inter {
      write_tx_tree(
        fi, fs, cw, wr, mode, bo, bsize, tx_size, tx_type, false, bit_depth, true
      );
    }  else {
      let cfl = CFLParams::new(); // Unused
      write_tx_blocks(
        fi, fs, cw, wr, mode, mode, bo, bsize, tx_size, tx_type, false, bit_depth, cfl, true, RDOType::Accurate
      );
    }

    let cost = wr.tell_frac() - tell;
    let (_, rd) = compute_rd_cost(
      fi,
      fs,
      w,
      h,
      is_chroma_block,
      bo,
      cost,
      bit_depth,
      true
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
  seq: &Sequence, fi: &FrameInvariants, fs: &mut FrameState,
  cw: &mut ContextWriter, bsize: BlockSize, bo: &BlockOffset,
  cached_block: &RDOOutput
) -> RDOOutput {
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
    let mut pmv =  MotionVector { row: 0, col: 0 };

    match partition {
      PartitionType::PARTITION_NONE => {
        if bsize > BlockSize::BLOCK_32X32 {
          continue;
        }

        let mode_decision = cached_block
          .part_modes
          .get(0)
          .unwrap_or(
            &rdo_mode_decision(seq, fi, fs, cw, bsize, bo, &pmv).part_modes[0]
          ).clone();
        child_modes.push(mode_decision);
      }
      PartitionType::PARTITION_SPLIT => {
        let subsize = bsize.subsize(partition);

        if subsize == BlockSize::BLOCK_INVALID {
          continue;
        }
        pmv = best_pred_modes[0].mv;

        assert!(best_pred_modes.len() <= 4);
        let bs = bsize.width_mi();
        let hbs = bs >> 1; // Half the block size in blocks
        let partitions = [
          bo,
          &BlockOffset{ x: bo.x + hbs as usize, y: bo.y },
          &BlockOffset{ x: bo.x, y: bo.y + hbs as usize },
          &BlockOffset{ x: bo.x + hbs as usize, y: bo.y + hbs as usize }
        ];
        child_modes.extend(
          partitions
            .iter()
            .map(|&offset| {
              rdo_mode_decision(seq, fi, fs, cw, subsize, &offset, &pmv)
                .part_modes[0]
                .clone()
            }).collect::<Vec<_>>()
        );
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
    let cdef_dirs = cdef_analyze_superblock(&mut rec_input, bc, &sbo_0, &sbo, bit_depth);
    for cdef_index in 0..(1<<fi.cdef_bits) {
        //for p in 0..3 {
        //    for i in 0..cdef_output.planes[p].data.len() { cdef_output.planes[p].data[i] = CDEF_VERY_LARGE; }
        //}
        // TODO: Don't repeat find_direction over and over; split filter_superblock to run it separately
        cdef_filter_superblock(fi, &mut rec_input, &mut cdef_output,
                               bc, &sbo_0, &sbo, bit_depth, cdef_index, &cdef_dirs);

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
                                err += cdef_dist_wxh_8x8(&in_slice, &out_slice, bit_depth);
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

pub fn get_fast_distortion_tx_block(
  _fi: &FrameInvariants, fs: &mut FrameState, _cw: &mut ContextWriter,
  w: &mut dyn Writer, p: usize, _bo: &BlockOffset, mode: PredictionMode,
  tx_size: TxSize, _tx_type: TxType, _plane_bsize: BlockSize, po: &PlaneOffset,
  skip: bool, bit_depth: usize, ac: &[i16], alpha: i16
) -> u64 {
  let rec = &mut fs.rec.planes[p];

  if mode.is_intra() {
    mode.predict_intra(&mut rec.mut_slice(po), tx_size, bit_depth, &ac, alpha);
  }

  let fast_distortion = compute_fast_distortion(fs.input.planes[p].slice(po), rec.slice(po), tx_size.width(), tx_size.height());

  fast_distortion
}

#[test]
fn estimate_rate_test() {
    let t = RDOTracker::new();
    assert_eq!(t.estimate_rate(TxSize::TX_4X4, 0), 595);
    assert_eq!(t.estimate_rate(TxSize::TX_4X4, RATE_EST_BIN_SIZE*1), 746);
    assert_eq!(t.estimate_rate(TxSize::TX_4X4, RATE_EST_BIN_SIZE*2), 691);
    assert_eq!(t.estimate_rate(TxSize::TX_4X4, RATE_EST_BIN_SIZE/2), 643);
}
