// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::cpu_features::CpuFeatureLevel;
use crate::dist::*;
use crate::partition::BlockSize;
use crate::tiling::*;
use crate::util::*;

type SadFn = unsafe extern fn(
  src: *const u8,
  src_stride: isize,
  dst: *const u8,
  dst_stride: isize,
) -> u32;

macro_rules! declare_asm_dist_fn {
  ($(($name: ident, $T: ident)),+) => (
    $(
      extern { fn $name (
        src: *const $T, src_stride: isize, dst: *const $T, dst_stride: isize
      ) -> u32; }
    )+
  )
}

declare_asm_dist_fn![
  (rav1e_sad4x4_neon, u8),
  (rav1e_sad4x8_neon, u8),
  (rav1e_sad4x16_neon, u8),
  (rav1e_sad8x4_neon, u8),
  (rav1e_sad8x8_neon, u8),
  (rav1e_sad8x16_neon, u8),
  (rav1e_sad8x32_neon, u8),
  (rav1e_sad16x4_neon, u8),
  (rav1e_sad16x8_neon, u8),
  (rav1e_sad16x16_neon, u8),
  (rav1e_sad16x32_neon, u8),
  (rav1e_sad16x64_neon, u8),
  (rav1e_sad32x8_neon, u8),
  (rav1e_sad32x16_neon, u8),
  (rav1e_sad32x32_neon, u8),
  (rav1e_sad32x64_neon, u8),
  (rav1e_sad64x16_neon, u8),
  (rav1e_sad64x32_neon, u8),
  (rav1e_sad64x64_neon, u8),
  (rav1e_sad64x128_neon, u8),
  (rav1e_sad128x64_neon, u8),
  (rav1e_sad128x128_neon, u8)
];

// BlockSize::BLOCK_SIZES.next_power_of_two();
const DIST_FNS_LENGTH: usize = 32;

fn to_index(bsize: BlockSize) -> usize {
  bsize as usize & (DIST_FNS_LENGTH - 1)
}

#[inline(always)]
#[allow(clippy::let_and_return)]
pub fn get_sad<T: Pixel>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, bsize: BlockSize,
  bit_depth: usize, cpu: CpuFeatureLevel,
) -> u32 {
  let call_rust = || -> u32 { rust::get_sad(dst, src, bsize, bit_depth, cpu) };

  #[cfg(feature = "check_asm")]
  let ref_dist = call_rust();

  let dist = match T::type_enum() {
    PixelType::U8 => match SAD_FNS[cpu.as_index()][to_index(bsize)] {
      Some(func) => unsafe {
        (func)(
          src.data_ptr() as *const _,
          T::to_asm_stride(src.plane_cfg.stride),
          dst.data_ptr() as *const _,
          T::to_asm_stride(dst.plane_cfg.stride),
        )
      },
      None => call_rust(),
    },
    PixelType::U16 => call_rust(),
  };

  #[cfg(feature = "check_asm")]
  assert_eq!(dist, ref_dist);

  dist
}

static SAD_FNS_NEON: [Option<SadFn>; DIST_FNS_LENGTH] = {
  let mut out: [Option<SadFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

  use BlockSize::*;

  out[BLOCK_4X4 as usize] = Some(rav1e_sad4x4_neon);
  out[BLOCK_4X8 as usize] = Some(rav1e_sad4x8_neon);
  out[BLOCK_4X16 as usize] = Some(rav1e_sad4x16_neon);

  out[BLOCK_8X4 as usize] = Some(rav1e_sad8x4_neon);
  out[BLOCK_8X8 as usize] = Some(rav1e_sad8x8_neon);
  out[BLOCK_8X16 as usize] = Some(rav1e_sad8x16_neon);
  out[BLOCK_8X32 as usize] = Some(rav1e_sad8x32_neon);

  out[BLOCK_16X4 as usize] = Some(rav1e_sad16x4_neon);
  out[BLOCK_16X8 as usize] = Some(rav1e_sad16x8_neon);
  out[BLOCK_16X16 as usize] = Some(rav1e_sad16x16_neon);
  out[BLOCK_16X32 as usize] = Some(rav1e_sad16x32_neon);
  out[BLOCK_16X64 as usize] = Some(rav1e_sad16x64_neon);

  out[BLOCK_32X8 as usize] = Some(rav1e_sad32x8_neon);
  out[BLOCK_32X16 as usize] = Some(rav1e_sad32x16_neon);
  out[BLOCK_32X32 as usize] = Some(rav1e_sad32x32_neon);
  out[BLOCK_32X64 as usize] = Some(rav1e_sad32x64_neon);

  out[BLOCK_64X16 as usize] = Some(rav1e_sad64x16_neon);
  out[BLOCK_64X32 as usize] = Some(rav1e_sad64x32_neon);
  out[BLOCK_64X64 as usize] = Some(rav1e_sad64x64_neon);
  out[BLOCK_64X128 as usize] = Some(rav1e_sad64x128_neon);

  out[BLOCK_128X64 as usize] = Some(rav1e_sad128x64_neon);
  out[BLOCK_128X128 as usize] = Some(rav1e_sad128x128_neon);

  out
};

cpu_function_lookup_table!(
  SAD_FNS: [[Option<SadFn>; DIST_FNS_LENGTH]],
  default: [None; DIST_FNS_LENGTH],
  [NEON]
);

#[cfg(test)]
mod test {
  use super::*;
  use crate::frame::{AsRegion, Plane};
  use rand::random;
  use std::str::FromStr;

  macro_rules! test_dist_fns {
    ($(($W:expr, $H:expr)),*, $DIST_TY:ident, $BD:expr, $OPT:ident, $OPTLIT:tt) => {
      $(
        paste::item! {
          #[test]
          fn [<get_ $DIST_TY _ $W x $H _bd_ $BD _ $OPT>]() {
            let bsize = BlockSize::[<BLOCK_ $W X $H>];
            if $BD > 8 {
              // dynamic allocation: test
              let mut src = Plane::from_slice(&vec![0u16; $W * $H], $W);
              // dynamic allocation: test
              let mut dst = Plane::from_slice(&vec![0u16; $W * $H], $W);
              for (s, d) in src.data.iter_mut().zip(dst.data.iter_mut()) {
                *s = random::<u8>() as u16 * $BD / 8;
                *d = random::<u8>() as u16 * $BD / 8;
              }
              let result = [<get_ $DIST_TY>](&src.as_region(), &dst.as_region(), bsize, $BD, CpuFeatureLevel::from_str($OPTLIT).unwrap());
              let rust_result = [<get_ $DIST_TY>](&src.as_region(), &dst.as_region(), bsize, $BD, CpuFeatureLevel::RUST);

              assert_eq!(rust_result, result);
            } else {
              // dynamic allocation: test
              let mut src = Plane::from_slice(&vec![0u8; $W * $H], $W);
              // dynamic allocation: test
              let mut dst = Plane::from_slice(&vec![0u8; $W * $H], $W);
              for (s, d) in src.data.iter_mut().zip(dst.data.iter_mut()) {
                *s = random::<u8>();
                *d = random::<u8>();
              }
              let result = [<get_ $DIST_TY>](&src.as_region(), &dst.as_region(), bsize, $BD, CpuFeatureLevel::from_str($OPTLIT).unwrap());
              let rust_result = [<get_ $DIST_TY>](&src.as_region(), &dst.as_region(), bsize, $BD, CpuFeatureLevel::RUST);

              assert_eq!(rust_result, result);
            }
          }
        }
      )*
    }
  }

  test_dist_fns!(
    (4, 4),
    (4, 8),
    (4, 16),
    (8, 4),
    (8, 8),
    (8, 16),
    (8, 32),
    (16, 4),
    (16, 8),
    (16, 16),
    (16, 32),
    (16, 64),
    (32, 8),
    (32, 16),
    (32, 32),
    (32, 64),
    (64, 16),
    (64, 32),
    (64, 64),
    (64, 128),
    (128, 64),
    (128, 128),
    sad,
    8,
    neon,
    "neon"
  );
}
