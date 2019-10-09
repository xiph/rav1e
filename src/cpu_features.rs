// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use arrayvec::ArrayVec;

// This macro helps reduce some code duplication
macro_rules! detect_feature {
  ($platform:ident, $feature:literal, $arr:ident) => {
    paste::expr! {
      if [<is_ $platform _feature_detected>]!($feature) {
        $arr.push($feature);
      }
    }
  };
}

pub fn get_detected_cpu_features() -> ArrayVec<[&'static str; 3]> {
  let mut features = ArrayVec::new();

  #[cfg(all(
    feature = "nasm",
    any(target_arch = "x86", target_arch = "x86_64")
  ))]
  {
    detect_feature!(x86, "sse2", features);
    detect_feature!(x86, "ssse3", features);
    detect_feature!(x86, "avx2", features);
  }

  features
}
