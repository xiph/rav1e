// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

extern crate libc;

mod predict;

benchmark_group!(
  intra_prediction,
  predict::intra_dc_4x4_native,
  predict::intra_dc_4x4_aom,
  predict::intra_h_4x4_native,
  predict::intra_h_4x4_aom,
  predict::intra_v_4x4_native,
  predict::intra_v_4x4_aom,
  predict::intra_paeth_4x4_native,
  predict::intra_paeth_4x4_aom,
  predict::intra_smooth_4x4_native,
  predict::intra_smooth_4x4_aom,
  predict::intra_smooth_h_4x4_native,
  predict::intra_smooth_h_4x4_aom,
  predict::intra_smooth_v_4x4_native,
  predict::intra_smooth_v_4x4_aom,
  predict::intra_cfl_4x4_native,
  predict::intra_cfl_4x4_aom
);
