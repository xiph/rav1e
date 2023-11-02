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

type SatdFn = SadFn;

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
  (rav1e_sad128x128_neon, u8),
  /* SATD */
  (rav1e_satd4x4_neon, u8),
  (rav1e_satd4x8_neon, u8),
  (rav1e_satd4x16_neon, u8),
  (rav1e_satd8x4_neon, u8),
  (rav1e_satd16x4_neon, u8)
];

// BlockSize::BLOCK_SIZES.next_power_of_two();
const DIST_FNS_LENGTH: usize = 32;

#[inline]
const fn to_index(bsize: BlockSize) -> usize {
  bsize as usize & (DIST_FNS_LENGTH - 1)
}

#[inline(always)]
#[allow(clippy::let_and_return)]
pub fn get_sad<T: Pixel>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, w: usize, h: usize,
  bit_depth: usize, cpu: CpuFeatureLevel,
) -> u32 {
  let bsize_opt = BlockSize::from_width_and_height_opt(w, h);

  let call_rust = || -> u32 { rust::get_sad(src, dst, w, h, bit_depth, cpu) };

  #[cfg(feature = "check_asm")]
  let ref_dist = call_rust();

  let dist = match (bsize_opt, T::type_enum()) {
    (Err(_), _) => call_rust(),
    (Ok(bsize), PixelType::U8) => {
      match SAD_FNS[cpu.as_index()][to_index(bsize)] {
        Some(func) => unsafe {
          (func)(
            src.data_ptr() as *const _,
            T::to_asm_stride(src.plane_cfg.stride),
            dst.data_ptr() as *const _,
            T::to_asm_stride(dst.plane_cfg.stride),
          )
        },
        None => call_rust(),
      }
    }
    (Ok(_bsize), PixelType::U16) => call_rust(),
  };

  #[cfg(feature = "check_asm")]
  assert_eq!(dist, ref_dist);

  dist
}

macro_rules! impl_satd_fn {
  ($(($name: ident, $T: ident, $LOG_W:expr, $log_h:expr)),+) => (
    $(
      #[no_mangle]
      unsafe extern fn $name (
        src: *const $T, src_stride: isize, dst: *const $T, dst_stride: isize
      ) -> u32 {
        rav1e_satd8wx8h_neon::<$LOG_W>(src, src_stride, dst, dst_stride, $log_h)
      }
    )+
  )
}

impl_satd_fn![
  (rav1e_satd8x8_neon, u8, 0, 0),
  (rav1e_satd8x16_neon, u8, 0, 1),
  (rav1e_satd8x32_neon, u8, 0, 2),
  (rav1e_satd16x8_neon, u8, 1, 0),
  (rav1e_satd16x16_neon, u8, 1, 1),
  (rav1e_satd16x32_neon, u8, 1, 2),
  (rav1e_satd16x64_neon, u8, 1, 3),
  (rav1e_satd32x8_neon, u8, 2, 0),
  (rav1e_satd32x16_neon, u8, 2, 1),
  (rav1e_satd32x32_neon, u8, 2, 2),
  (rav1e_satd32x64_neon, u8, 2, 3),
  (rav1e_satd64x16_neon, u8, 3, 1),
  (rav1e_satd64x32_neon, u8, 3, 2),
  (rav1e_satd64x64_neon, u8, 3, 3),
  (rav1e_satd64x128_neon, u8, 3, 4),
  (rav1e_satd128x64_neon, u8, 4, 3),
  (rav1e_satd128x128_neon, u8, 4, 4)
];

unsafe fn rav1e_satd8wx8h_neon<const LOG_W: usize>(
  mut src: *const u8, src_stride: isize, mut dst: *const u8,
  dst_stride: isize, log_h: usize,
) -> u32 {
  let mut sum = 0;
  for _ in 0..(1 << log_h) {
    let (mut src_off, mut dst_off) = (src, dst);
    for _ in 0..(1 << LOG_W) {
      sum +=
        rav1e_satd8x8_internal_neon(src_off, src_stride, dst_off, dst_stride);
      src_off = src_off.add(8);
      dst_off = dst_off.add(8);
    }
    src = src.offset(src_stride << 3);
    dst = dst.offset(dst_stride << 3);
  }
  (sum + 4) >> 3
}

unsafe fn rav1e_satd8x8_internal_neon(
  src: *const u8, src_stride: isize, dst: *const u8, dst_stride: isize,
) -> u32 {
  use core::arch::aarch64::*;

  let load_row = |src: *const u8, dst: *const u8| -> int16x8_t {
    vreinterpretq_s16_u16(vsubl_u8(vld1_u8(src), vld1_u8(dst)))
  };
  let butterfly = |a: int16x8_t, b: int16x8_t| -> int16x8x2_t {
    int16x8x2_t(vaddq_s16(a, b), vsubq_s16(a, b))
  };
  let zip1 = |v: int16x8x2_t| -> int16x8x2_t {
    int16x8x2_t(vzip1q_s16(v.0, v.1), vzip2q_s16(v.0, v.1))
  };
  let zip2 = |v: int16x8x2_t| -> int16x8x2_t {
    let v =
      int32x4x2_t(vreinterpretq_s32_s16(v.0), vreinterpretq_s32_s16(v.1));
    let v = int32x4x2_t(vzip1q_s32(v.0, v.1), vzip2q_s32(v.0, v.1));
    int16x8x2_t(vreinterpretq_s16_s32(v.0), vreinterpretq_s16_s32(v.1))
  };
  let zip4 = |v: int16x8x2_t| -> int16x8x2_t {
    let v =
      int64x2x2_t(vreinterpretq_s64_s16(v.0), vreinterpretq_s64_s16(v.1));
    let v = int64x2x2_t(vzip1q_s64(v.0, v.1), vzip2q_s64(v.0, v.1));
    int16x8x2_t(vreinterpretq_s16_s64(v.0), vreinterpretq_s16_s64(v.1))
  };

  let (src_stride2, dst_stride2) = (src_stride << 1, dst_stride << 1);
  let int16x8x2_t(r0, r1) = zip1(butterfly(
    load_row(src, dst),
    load_row(src.offset(src_stride), dst.offset(dst_stride)),
  ));
  let (src, dst) = (src.offset(src_stride2), dst.offset(dst_stride2));
  let int16x8x2_t(r2, r3) = zip1(butterfly(
    load_row(src, dst),
    load_row(src.offset(src_stride), dst.offset(dst_stride)),
  ));
  let (src, dst) = (src.offset(src_stride2), dst.offset(dst_stride2));
  let int16x8x2_t(r4, r5) = zip1(butterfly(
    load_row(src, dst),
    load_row(src.offset(src_stride), dst.offset(dst_stride)),
  ));
  let (src, dst) = (src.offset(src_stride2), dst.offset(dst_stride2));
  let int16x8x2_t(r6, r7) = zip1(butterfly(
    load_row(src, dst),
    load_row(src.offset(src_stride), dst.offset(dst_stride)),
  ));

  let int16x8x2_t(r0, r2) = zip2(butterfly(r0, r2));
  let int16x8x2_t(r1, r3) = zip2(butterfly(r1, r3));
  let int16x8x2_t(r4, r6) = zip2(butterfly(r4, r6));
  let int16x8x2_t(r5, r7) = zip2(butterfly(r5, r7));

  let int16x8x2_t(r0, r4) = zip4(butterfly(r0, r4));
  let int16x8x2_t(r1, r5) = zip4(butterfly(r1, r5));
  let int16x8x2_t(r2, r6) = zip4(butterfly(r2, r6));
  let int16x8x2_t(r3, r7) = zip4(butterfly(r3, r7));

  let int16x8x2_t(r0, r1) = butterfly(r0, r1);
  let int16x8x2_t(r2, r3) = butterfly(r2, r3);
  let int16x8x2_t(r4, r5) = butterfly(r4, r5);
  let int16x8x2_t(r6, r7) = butterfly(r6, r7);

  let int16x8x2_t(r0, r2) = butterfly(r0, r2);
  let int16x8x2_t(r1, r3) = butterfly(r1, r3);
  let int16x8x2_t(r4, r6) = butterfly(r4, r6);
  let int16x8x2_t(r5, r7) = butterfly(r5, r7);

  let int16x8x2_t(r0, r4) = butterfly(r0, r4);
  let int16x8x2_t(r1, r5) = butterfly(r1, r5);
  let int16x8x2_t(r2, r6) = butterfly(r2, r6);
  let int16x8x2_t(r3, r7) = butterfly(r3, r7);

  let r0 = vabsq_s16(r0);
  let r1 = vabsq_s16(r1);
  let r2 = vabsq_s16(r2);
  let r3 = vabsq_s16(r3);
  let r4 = vabsq_s16(r4);
  let r5 = vabsq_s16(r5);
  let r6 = vabsq_s16(r6);
  let r7 = vabsq_s16(r7);

  let (t0, t1) = (vmovl_s16(vget_low_s16(r0)), vmovl_s16(vget_high_s16(r0)));
  let (t0, t1) = (vaddw_s16(t0, vget_low_s16(r1)), vaddw_high_s16(t1, r1));
  let (t0, t1) = (vaddw_s16(t0, vget_low_s16(r2)), vaddw_high_s16(t1, r2));
  let (t0, t1) = (vaddw_s16(t0, vget_low_s16(r3)), vaddw_high_s16(t1, r3));
  let (t0, t1) = (vaddw_s16(t0, vget_low_s16(r4)), vaddw_high_s16(t1, r4));
  let (t0, t1) = (vaddw_s16(t0, vget_low_s16(r5)), vaddw_high_s16(t1, r5));
  let (t0, t1) = (vaddw_s16(t0, vget_low_s16(r6)), vaddw_high_s16(t1, r6));
  let (t0, t1) = (vaddw_s16(t0, vget_low_s16(r7)), vaddw_high_s16(t1, r7));

  vaddvq_s32(vaddq_s32(t0, t1)) as u32
}

#[inline(always)]
#[allow(clippy::let_and_return)]
pub fn get_satd<T: Pixel>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, w: usize, h: usize,
  bit_depth: usize, cpu: CpuFeatureLevel,
) -> u32 {
  let bsize_opt = BlockSize::from_width_and_height_opt(w, h);

  let call_rust = || -> u32 { rust::get_satd(src, dst, w, h, bit_depth, cpu) };

  #[cfg(feature = "check_asm")]
  let ref_dist = call_rust();

  let dist = match (bsize_opt, T::type_enum()) {
    (Err(_), _) => call_rust(),
    (Ok(bsize), PixelType::U8) => {
      match SATD_FNS[cpu.as_index()][to_index(bsize)] {
        // SAFETY: Calls Assembly code.
        Some(func) => unsafe {
          (func)(
            src.data_ptr() as *const _,
            T::to_asm_stride(src.plane_cfg.stride),
            dst.data_ptr() as *const _,
            T::to_asm_stride(dst.plane_cfg.stride),
          )
        },
        None => call_rust(),
      }
    }
    _ => call_rust(),
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

static SATD_FNS_NEON: [Option<SatdFn>; DIST_FNS_LENGTH] = {
  let mut out: [Option<SatdFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

  use BlockSize::*;

  out[BLOCK_4X4 as usize] = Some(rav1e_satd4x4_neon);
  out[BLOCK_4X8 as usize] = Some(rav1e_satd4x8_neon);
  out[BLOCK_4X16 as usize] = Some(rav1e_satd4x16_neon);
  out[BLOCK_8X4 as usize] = Some(rav1e_satd8x4_neon);
  out[BLOCK_16X4 as usize] = Some(rav1e_satd16x4_neon);

  out[BLOCK_8X8 as usize] = Some(rav1e_satd8x8_neon);
  out[BLOCK_8X16 as usize] = Some(rav1e_satd8x16_neon);
  out[BLOCK_8X32 as usize] = Some(rav1e_satd8x32_neon);
  out[BLOCK_16X8 as usize] = Some(rav1e_satd16x8_neon);
  out[BLOCK_16X16 as usize] = Some(rav1e_satd16x16_neon);
  out[BLOCK_16X32 as usize] = Some(rav1e_satd16x32_neon);
  out[BLOCK_16X64 as usize] = Some(rav1e_satd16x64_neon);
  out[BLOCK_32X8 as usize] = Some(rav1e_satd32x8_neon);
  out[BLOCK_32X16 as usize] = Some(rav1e_satd32x16_neon);
  out[BLOCK_32X32 as usize] = Some(rav1e_satd32x32_neon);
  out[BLOCK_32X64 as usize] = Some(rav1e_satd32x64_neon);
  out[BLOCK_64X16 as usize] = Some(rav1e_satd64x16_neon);
  out[BLOCK_64X32 as usize] = Some(rav1e_satd64x32_neon);
  out[BLOCK_64X64 as usize] = Some(rav1e_satd64x64_neon);
  out[BLOCK_64X128 as usize] = Some(rav1e_satd64x128_neon);
  out[BLOCK_128X64 as usize] = Some(rav1e_satd128x64_neon);
  out[BLOCK_128X128 as usize] = Some(rav1e_satd128x128_neon);

  out
};

cpu_function_lookup_table!(
  SAD_FNS: [[Option<SadFn>; DIST_FNS_LENGTH]],
  default: [None; DIST_FNS_LENGTH],
  [NEON]
);

cpu_function_lookup_table!(
  SATD_FNS: [[Option<SatdFn>; DIST_FNS_LENGTH]],
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
            if $BD > 8 {
              // dynamic allocation: test
              let mut src = Plane::from_slice(&vec![0u16; $W * $H], $W);
              // dynamic allocation: test
              let mut dst = Plane::from_slice(&vec![0u16; $W * $H], $W);
              for (s, d) in src.data.iter_mut().zip(dst.data.iter_mut()) {
                *s = random::<u8>() as u16 * $BD / 8;
                *d = random::<u8>() as u16 * $BD / 8;
              }
              let result = [<get_ $DIST_TY>](&src.as_region(), &dst.as_region(), $W, $H, $BD, CpuFeatureLevel::from_str($OPTLIT).unwrap());
              let rust_result = [<get_ $DIST_TY>](&src.as_region(), &dst.as_region(), $W, $H, $BD, CpuFeatureLevel::RUST);

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
              let result = [<get_ $DIST_TY>](&src.as_region(), &dst.as_region(), $W, $H, $BD, CpuFeatureLevel::from_str($OPTLIT).unwrap());
              let rust_result = [<get_ $DIST_TY>](&src.as_region(), &dst.as_region(), $W, $H, $BD, CpuFeatureLevel::RUST);

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
    satd,
    8,
    neon,
    "neon"
  );
}
