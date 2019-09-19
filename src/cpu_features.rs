// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub use native::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use x86::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
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
      } else {
        CpuFeatureLevel::NATIVE
      };
      let manual: CpuFeatureLevel = match env::var("RAV1E_CPU_TARGET") {
        Ok(feature) => match feature.as_ref() {
          "rust" => CpuFeatureLevel::NATIVE,
          "avx2" => CpuFeatureLevel::AVX2,
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
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
mod native {
  use arg_enum_proc_macro::ArgEnum;

  #[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, ArgEnum)]
  pub enum CpuFeatureLevel {
    NATIVE,
  }

  impl Default for CpuFeatureLevel {
    fn default() -> CpuFeatureLevel {
      CpuFeatureLevel::NATIVE
    }
  }
}
