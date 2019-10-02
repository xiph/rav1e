// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::cpu_features::CpuFeatureLevel;
use crate::ec::native;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn update_cdf(cdf: &mut [u16], val: u32, cpu: CpuFeatureLevel) {
  if cdf.len() == 5 && cpu >= CpuFeatureLevel::SSE2 {
    return unsafe {
      update_cdf_4_sse2(cdf, val);
    };
  }

  native::update_cdf(cdf, val, cpu);
}

#[target_feature(enable = "sse2")]
unsafe fn update_cdf_4_sse2(cdf: &mut [u16], val: u32) {
  let nsymbs = 4;
  let rate = 3 + (nsymbs >> 1).min(2) + (cdf[nsymbs] >> 4) as usize;
  cdf[nsymbs] += 1 - (cdf[nsymbs] >> 5);

  let val_idx = _mm_set_epi16(0, 0, 0, 0, 3, 2, 1, 0);
  let val_splat = _mm_set1_epi16(val as i16);
  let mask = _mm_cmpgt_epi16(val_splat, val_idx);
  let v = _mm_loadl_epi64(cdf.as_ptr() as *const __m128i);
  // *v -= *v >> rate;
  let shrunk_v =
    _mm_sub_epi16(v, _mm_srl_epi16(v, _mm_cvtsi64_si128(rate as i64)));
  // *v += (32768 - *v) >> rate;
  let expanded_v = _mm_add_epi16(
    v,
    _mm_srl_epi16(
      _mm_sub_epi16(_mm_set1_epi16(-32768), v),
      _mm_cvtsi64_si128(rate as i64),
    ),
  );
  let out_v = _mm_or_si128(
    _mm_andnot_si128(mask, shrunk_v),
    _mm_and_si128(mask, expanded_v),
  );
  _mm_storel_epi64(cdf.as_mut_ptr() as *mut __m128i, out_v);
}

#[cfg(test)]
mod test {
  use crate::cpu_features::CpuFeatureLevel;
  use crate::ec::native;

  #[test]
  fn update_cdf_4_sse2() {
    let mut cdf = [7296, 3819, 1616, 0, 0];
    let mut cdf2 = [7296, 3819, 1616, 0, 0];
    for i in 0..4 {
      native::update_cdf(&mut cdf, i, CpuFeatureLevel::default());
      unsafe {
        super::update_cdf_4_sse2(&mut cdf2, i);
      }
      assert_eq!(cdf, cdf2);
    }
  }
}
