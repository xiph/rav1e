// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

/// Information provided to the progress callback
#[derive(Debug, Clone)]
pub struct ProgressData {}

/// Progress callback
pub trait GranularProgress: std::fmt::Debug + Sync + Send {
  /// Return if the encoding process should continue or not
  fn progress(&self, info: &ProgressData) -> bool;
}

#[derive(Debug)]
pub(crate) struct DefaultProgress {}

impl GranularProgress for DefaultProgress {
  fn progress(&self, _info: &ProgressData) -> bool {
    true
  }
}
