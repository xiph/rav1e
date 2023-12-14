// Copyright (c) 2021-2022, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

cfg_if::cfg_if! {
  if #[cfg(nasm_x86_64)] {
    use crate::asm::x86::sad_plane::*;
  } else {
    use self::rust::*;
  }
}

use v_frame::plane::Plane;

use crate::cpu_features::CpuFeatureLevel;
use crate::util::{CastFromPrimitive, Pixel};

pub(crate) mod rust {
  use super::*;
  use crate::cpu_features::CpuFeatureLevel;

  #[inline]
  pub(crate) fn sad_plane_internal<T: Pixel>(
    src: &Plane<T>, dst: &Plane<T>, _cpu: CpuFeatureLevel,
  ) -> u64 {
    debug_assert!(src.cfg.width == dst.cfg.width);
    debug_assert!(src.cfg.height == dst.cfg.height);

    src
      .rows_iter()
      .zip(dst.rows_iter())
      .map(|(src, dst)| {
        src
          .iter()
          .zip(dst.iter())
          .map(|(&p1, &p2)| i32::cast_from(p1).abs_diff(i32::cast_from(p2)))
          .sum::<u32>() as u64
      })
      .sum()
  }
}

/// Compute the sum of absolute differences (SADs) on 2 rows of pixels
///
/// This differs from other SAD functions in that it operates over a row
/// (or line) of unknown length rather than a `PlaneRegion<T>`.
pub(crate) fn sad_plane<T: Pixel>(
  src: &Plane<T>, dst: &Plane<T>, cpu: CpuFeatureLevel,
) -> u64 {
  sad_plane_internal(src, dst, cpu)
}
