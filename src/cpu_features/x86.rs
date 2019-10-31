// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use arg_enum_proc_macro::ArgEnum;
use std::env;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, ArgEnum)]
pub enum CpuFeatureLevel {
  NATIVE,
  SSE2,
  SSSE3,
  AVX2,
}

impl CpuFeatureLevel {
  pub const fn len() -> usize {
    CpuFeatureLevel::AVX2 as usize + 1
  }

  #[inline(always)]
  pub fn as_index(self) -> usize {
    const LEN: usize = CpuFeatureLevel::len();
    assert_eq!(LEN & (LEN - 1), 0);
    self as usize & (LEN - 1)
  }
}

impl Default for CpuFeatureLevel {
  fn default() -> CpuFeatureLevel {
    let detected: CpuFeatureLevel = if is_x86_feature_detected!("avx2") {
      CpuFeatureLevel::AVX2
    } else if is_x86_feature_detected!("ssse3") {
      CpuFeatureLevel::SSSE3
    } else if is_x86_feature_detected!("sse2") {
      CpuFeatureLevel::SSE2
    } else {
      CpuFeatureLevel::NATIVE
    };
    let manual: CpuFeatureLevel = match env::var("RAV1E_CPU_TARGET") {
      Ok(feature) => match feature.as_ref() {
        "rust" | "native" => CpuFeatureLevel::NATIVE,
        "avx2" => CpuFeatureLevel::AVX2,
        "ssse3" => CpuFeatureLevel::SSSE3,
        "sse2" => CpuFeatureLevel::SSE2,
        _ => detected,
      },
      Err(_e) => detected,
    };
    if manual > detected {
      detected
    } else {
      manual
    }
  }
}
