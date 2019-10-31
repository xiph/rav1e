// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use std::env;
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd)]
pub enum CpuFeatureLevel {
  NATIVE,
  SSE2,
  SSSE3,
  SSE4_1,
  AVX2,
}

impl FromStr for CpuFeatureLevel {
  type Err = ();

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    Ok(match s.to_lowercase().as_str() {
      "rust" | "native" => CpuFeatureLevel::NATIVE,
      "avx2" => CpuFeatureLevel::AVX2,
      "sse4" | "sse4_1" | "sse4.1" => CpuFeatureLevel::SSE4_1,
      "ssse3" => CpuFeatureLevel::SSSE3,
      "sse2" => CpuFeatureLevel::SSE2,
      _ => {
        return Err(());
      }
    })
  }
}

impl fmt::Display for CpuFeatureLevel {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "{}",
      match self {
        CpuFeatureLevel::NATIVE => "Native",
        CpuFeatureLevel::SSE2 => "SSE2",
        CpuFeatureLevel::SSSE3 => "SSSE3",
        CpuFeatureLevel::SSE4_1 => "SSE4.1",
        CpuFeatureLevel::AVX2 => "AVX2",
      }
    )
  }
}

impl CpuFeatureLevel {
  pub const fn len() -> usize {
    CpuFeatureLevel::AVX2 as usize + 1
  }

  #[inline(always)]
  pub fn as_index(self) -> usize {
    self as usize
  }
}

impl Default for CpuFeatureLevel {
  fn default() -> CpuFeatureLevel {
    let detected: CpuFeatureLevel = if is_x86_feature_detected!("avx2") {
      CpuFeatureLevel::AVX2
    } else if is_x86_feature_detected!("sse4.1") {
      CpuFeatureLevel::SSE4_1
    } else if is_x86_feature_detected!("ssse3") {
      CpuFeatureLevel::SSSE3
    } else if is_x86_feature_detected!("sse2") {
      CpuFeatureLevel::SSE2
    } else {
      CpuFeatureLevel::NATIVE
    };
    let manual: CpuFeatureLevel = match env::var("RAV1E_CPU_TARGET") {
      Ok(feature) => CpuFeatureLevel::from_str(&feature).unwrap_or(detected),
      Err(_e) => detected,
    };
    if manual > detected {
      detected
    } else {
      manual
    }
  }
}
