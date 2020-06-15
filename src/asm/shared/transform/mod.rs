// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

pub mod inverse;

use crate::transform::*;

#[inline]
pub const fn get_tx_size_idx(tx_size: TxSize) -> usize {
  (tx_size as usize) & 31
}

#[inline]
pub const fn get_tx_type_idx(tx_type: TxType) -> usize {
  // TX_TYPES is 2^4 or 16
  (tx_type as usize) & (TX_TYPES - 1)
}
