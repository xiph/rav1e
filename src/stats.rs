// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::partition::BlockSize;
use crate::prelude::PredictionMode;
use crate::transform::TxType;
use std::collections::BTreeMap;
use std::ops::{Add, AddAssign};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderStats {
  /// Stores count of pixels belonging to each block size in this frame
  pub block_size_counts: BTreeMap<BlockSize, usize>,
  /// Stores count of pixels belonging to skip blocks in this frame
  pub skip_block_count: usize,
  /// Stores count of pixels belonging to each transform type in this frame
  pub tx_type_counts: BTreeMap<TxType, usize>,
  /// Stores count of pixels belonging to each luma prediction mode in this frame
  pub luma_pred_mode_counts: BTreeMap<PredictionMode, usize>,
  /// Stores count of pixels belonging to each chroma prediction mode in this frame
  pub chroma_pred_mode_counts: BTreeMap<PredictionMode, usize>,
}

impl Default for EncoderStats {
  fn default() -> Self {
    EncoderStats {
      block_size_counts: BTreeMap::new(),
      skip_block_count: 0,
      tx_type_counts: BTreeMap::new(),
      luma_pred_mode_counts: BTreeMap::new(),
      chroma_pred_mode_counts: BTreeMap::new(),
    }
  }
}

impl Add<&Self> for EncoderStats {
  type Output = Self;

  fn add(self, rhs: &EncoderStats) -> Self::Output {
    let mut lhs = self;
    lhs += rhs;
    lhs
  }
}

impl AddAssign<&Self> for EncoderStats {
  fn add_assign(&mut self, rhs: &EncoderStats) {
    rhs.block_size_counts.iter().for_each(|(&k, &v)| {
      *self.block_size_counts.entry(k).or_insert(0) += v;
    });
    rhs.chroma_pred_mode_counts.iter().for_each(|(&k, &v)| {
      *self.chroma_pred_mode_counts.entry(k).or_insert(0) += v;
    });
    rhs.luma_pred_mode_counts.iter().for_each(|(&k, &v)| {
      *self.luma_pred_mode_counts.entry(k).or_insert(0) += v;
    });
    rhs.tx_type_counts.iter().for_each(|(&k, &v)| {
      *self.tx_type_counts.entry(k).or_insert(0) += v;
    });
    self.skip_block_count += rhs.skip_block_count;
  }
}
