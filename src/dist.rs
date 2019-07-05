// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
pub use self::nasm::get_sad;
#[cfg(any(not(target_arch = "x86_64"), not(feature = "nasm")))]
pub use self::native::get_sad;

#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
pub use self::nasm::get_satd;
#[cfg(any(not(target_arch = "x86_64"), not(feature = "nasm")))]
pub use self::native::get_satd;

#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
mod nasm {
  use crate::tiling::*;
  use crate::util::*;
  use std::mem;

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
    // SSSE3
    (rav1e_sad_4x4_hbd_ssse3, u16),
    (rav1e_sad_8x8_hbd10_ssse3, u16),
    (rav1e_sad_16x16_hbd_ssse3, u16),
    (rav1e_sad_32x32_hbd10_ssse3, u16),
    (rav1e_sad_64x64_hbd10_ssse3, u16),
    (rav1e_sad_128x128_hbd10_ssse3, u16),
    // SSE2
    (rav1e_sad4x4_sse2, u8),
    (rav1e_sad4x8_sse2, u8),
    (rav1e_sad4x16_sse2, u8),
    (rav1e_sad8x4_sse2, u8),
    (rav1e_sad8x8_sse2, u8),
    (rav1e_sad8x16_sse2, u8),
    (rav1e_sad8x32_sse2, u8),
    (rav1e_sad16x16_sse2, u8),
    (rav1e_sad32x32_sse2, u8),
    (rav1e_sad64x64_sse2, u8),
    (rav1e_sad128x128_sse2, u8),
    // AVX
    (rav1e_sad16x4_avx2, u8),
    (rav1e_sad16x8_avx2, u8),
    (rav1e_sad16x16_avx2, u8),
    (rav1e_sad16x32_avx2, u8),
    (rav1e_sad16x64_avx2, u8),
    (rav1e_sad32x8_avx2, u8),
    (rav1e_sad32x16_avx2, u8),
    (rav1e_sad32x32_avx2, u8),
    (rav1e_sad32x64_avx2, u8),
    (rav1e_sad64x16_avx2, u8),
    (rav1e_sad64x32_avx2, u8),
    (rav1e_sad64x64_avx2, u8),
    (rav1e_sad64x128_avx2, u8),
    (rav1e_sad128x64_avx2, u8),
    (rav1e_sad128x128_avx2, u8),

    (rav1e_satd_4x4_avx2, u8),
    (rav1e_satd_8x8_avx2, u8),
    (rav1e_satd_16x16_avx2, u8),
    (rav1e_satd_32x32_avx2, u8),
    (rav1e_satd_64x64_avx2, u8),
    (rav1e_satd_128x128_avx2, u8),

    (rav1e_satd_4x8_avx2, u8),
    (rav1e_satd_8x4_avx2, u8),
    (rav1e_satd_8x16_avx2, u8),
    (rav1e_satd_16x8_avx2, u8),
    (rav1e_satd_16x32_avx2, u8),
    (rav1e_satd_32x16_avx2, u8),
    (rav1e_satd_32x64_avx2, u8),
    (rav1e_satd_64x32_avx2, u8),
    (rav1e_satd_64x128_avx2, u8),
    (rav1e_satd_128x64_avx2, u8),

    (rav1e_satd_4x16_avx2, u8),
    (rav1e_satd_16x4_avx2, u8),
    (rav1e_satd_8x32_avx2, u8),
    (rav1e_satd_32x8_avx2, u8),
    (rav1e_satd_16x64_avx2, u8),
    (rav1e_satd_64x16_avx2, u8),
    (rav1e_satd_32x128_avx2, u8),
    (rav1e_satd_128x32_avx2, u8)
  ];

  #[target_feature(enable = "ssse3")]
  unsafe fn sad_hbd_ssse3(
    plane_org: &PlaneRegion<'_, u16>, plane_ref: &PlaneRegion<'_, u16>,
    blk_w: usize, blk_h: usize, bit_depth: usize
  ) -> u32 {
    let mut sum = 0 as u32;
    let org_stride = (plane_org.plane_cfg.stride * 2) as isize;
    let ref_stride = (plane_ref.plane_cfg.stride * 2) as isize;
    assert!(blk_h >= 4 && blk_w >= 4);
    let step_size =
      blk_h.min(blk_w).min(if bit_depth <= 10 { 128 } else { 4 });
    let func = match step_size.ilog() {
      3 => rav1e_sad_4x4_hbd_ssse3,
      4 => rav1e_sad_8x8_hbd10_ssse3,
      5 => rav1e_sad_16x16_hbd_ssse3,
      6 => rav1e_sad_32x32_hbd10_ssse3,
      7 => rav1e_sad_64x64_hbd10_ssse3,
      8 => rav1e_sad_128x128_hbd10_ssse3,
      _ => rav1e_sad_128x128_hbd10_ssse3
    };
    for r in (0..blk_h).step_by(step_size) {
      for c in (0..blk_w).step_by(step_size) {
        // FIXME for now, T == u16
        let org_ptr = &plane_org[r][c] as *const u16;
        let ref_ptr = &plane_ref[r][c] as *const u16;
        sum += func(org_ptr, org_stride, ref_ptr, ref_stride);
      }
    }
    sum
  }

  #[target_feature(enable = "sse2")]
  unsafe fn sad_sse2(
    plane_org: &PlaneRegion<'_, u8>, plane_ref: &PlaneRegion<'_, u8>,
    blk_w: usize, blk_h: usize
  ) -> u32 {
    let org_ptr = plane_org.data_ptr();
    let ref_ptr = plane_ref.data_ptr();
    let org_stride = plane_org.plane_cfg.stride as isize;
    let ref_stride = plane_ref.plane_cfg.stride as isize;
    if blk_w == 16 && blk_h == 16 && (org_ptr as usize & 15) == 0 {
      return rav1e_sad16x16_sse2(org_ptr, org_stride, ref_ptr, ref_stride);
    }
    // Note: unaligned blocks come from hres/qres ME search
    let ptr_align_log2 = (org_ptr as usize).trailing_zeros() as usize;
    // The largest unaligned-safe function is for 8x8
    let ptr_align = 1 << ptr_align_log2.max(3);
    let step_size = blk_h.min(blk_w).min(ptr_align);
    let func = match step_size.ilog() {
      3 => rav1e_sad4x4_sse2,
      4 => rav1e_sad8x8_sse2,
      5 => rav1e_sad16x16_sse2,
      6 => rav1e_sad32x32_sse2,
      7 => rav1e_sad64x64_sse2,
      8 => rav1e_sad128x128_sse2,
      _ => rav1e_sad128x128_sse2
    };
    let mut sum = 0 as u32;
    for r in (0..blk_h).step_by(step_size) {
      for c in (0..blk_w).step_by(step_size) {
        let org_ptr = &plane_org[r][c] as *const u8;
        let ref_ptr = &plane_ref[r][c] as *const u8;
        sum += func(org_ptr, org_stride, ref_ptr, ref_stride);
      }
    }
    sum
  }

  #[target_feature(enable = "avx2")]
  unsafe fn sad_avx2(
    plane_org: &PlaneRegion<'_, u8>, plane_ref: &PlaneRegion<'_, u8>,
    blk_w: usize, blk_h: usize
  ) -> u32 {
    let org_ptr = plane_org.data_ptr();
    let ref_ptr = plane_ref.data_ptr();
    let org_stride = plane_org.plane_cfg.stride as isize;
    let ref_stride = plane_ref.plane_cfg.stride as isize;

    let func = match (blk_w, blk_h) {
      (4, 4) => rav1e_sad4x4_sse2,
      (4, 8) => rav1e_sad4x8_sse2,
      (4, 16) => rav1e_sad4x16_sse2,

      (8, 4) => rav1e_sad8x4_sse2,
      (8, 8) => rav1e_sad8x8_sse2,
      (8, 16) => rav1e_sad8x16_sse2,
      (8, 32) => rav1e_sad8x32_sse2,

      (16, 4) => rav1e_sad16x4_avx2,
      (16, 8) => rav1e_sad16x8_avx2,
      (16, 16) => rav1e_sad16x16_avx2,
      (16, 32) => rav1e_sad16x32_avx2,
      (16, 64) => rav1e_sad16x64_avx2,

      (32, 8) => rav1e_sad32x8_avx2,
      (32, 16) => rav1e_sad32x16_avx2,
      (32, 32) => rav1e_sad32x32_avx2,
      (32, 64) => rav1e_sad32x64_avx2,

      (64, 16) => rav1e_sad64x16_avx2,
      (64, 32) => rav1e_sad64x32_avx2,
      (64, 64) => rav1e_sad64x64_avx2,
      (64, 128) => rav1e_sad64x128_avx2,

      (128, 64) => rav1e_sad128x64_avx2,
      (128, 128) => rav1e_sad128x128_avx2,

      _ => unreachable!()
    };
    func(org_ptr, org_stride, ref_ptr, ref_stride)
  }

  #[inline(always)]
  pub fn get_sad<T: Pixel>(
    plane_org: &PlaneRegion<'_, T>, plane_ref: &PlaneRegion<'_, T>,
    blk_w: usize, blk_h: usize, bit_depth: usize
  ) -> u32 {
    #[cfg(all(target_arch = "x86_64", feature = "nasm"))]
    {
      if mem::size_of::<T>() == 2
        && is_x86_feature_detected!("ssse3")
        && blk_h >= 4
        && blk_w >= 4
      {
        return unsafe {
          let plane_org =
            &*(plane_org as *const _ as *const PlaneRegion<'_, u16>);
          let plane_ref =
            &*(plane_ref as *const _ as *const PlaneRegion<'_, u16>);
          sad_hbd_ssse3(plane_org, plane_ref, blk_w, blk_h, bit_depth)
        };
      }
      if mem::size_of::<T>() == 1
        && is_x86_feature_detected!("avx2")
        && blk_h >= 4
        && blk_w >= 4
      {
        return unsafe {
          let plane_org =
            &*(plane_org as *const _ as *const PlaneRegion<'_, u8>);
          let plane_ref =
            &*(plane_ref as *const _ as *const PlaneRegion<'_, u8>);
          sad_avx2(plane_org, plane_ref, blk_w, blk_h)
        };
      }
      if mem::size_of::<T>() == 1
        && is_x86_feature_detected!("sse2")
        && blk_h >= 4
        && blk_w >= 4
      {
        return unsafe {
          let plane_org =
            &*(plane_org as *const _ as *const PlaneRegion<'_, u8>);
          let plane_ref =
            &*(plane_ref as *const _ as *const PlaneRegion<'_, u8>);
          sad_sse2(plane_org, plane_ref, blk_w, blk_h)
        };
      }
    }
    super::native::get_sad(plane_org, plane_ref, blk_w, blk_h, bit_depth)
  }

  #[target_feature(enable = "avx2")]
  unsafe fn satd_avx2(
    plane_org: &PlaneRegion<'_, u8>, plane_ref: &PlaneRegion<'_, u8>,
    blk_w: usize, blk_h: usize
  ) -> u32 {
    let org_ptr = plane_org.data_ptr();
    let ref_ptr = plane_ref.data_ptr();
    let org_stride = plane_org.plane_cfg.stride as isize;
    let ref_stride = plane_ref.plane_cfg.stride as isize;

    let func = match (blk_w, blk_h) {
      (4, 4) => rav1e_satd_4x4_avx2,
      (8, 8) => rav1e_satd_8x8_avx2,
      (16, 16) => rav1e_satd_16x16_avx2,
      (32, 32) => rav1e_satd_32x32_avx2,
      (64, 64) => rav1e_satd_64x64_avx2,
      (128, 128) => rav1e_satd_128x128_avx2,

      (4, 8) => rav1e_satd_4x8_avx2,
      (8, 4) => rav1e_satd_8x4_avx2,
      (8, 16) => rav1e_satd_8x16_avx2,
      (16, 8) => rav1e_satd_16x8_avx2,
      (16, 32) => rav1e_satd_16x32_avx2,
      (32, 16) => rav1e_satd_32x16_avx2,
      (32, 64) => rav1e_satd_32x64_avx2,
      (64, 32) => rav1e_satd_64x32_avx2,
      (64, 128) => rav1e_satd_64x128_avx2,
      (128, 64) => rav1e_satd_128x64_avx2,

      (4, 16) => rav1e_satd_4x16_avx2,
      (16, 4) => rav1e_satd_16x4_avx2,
      (8, 32) => rav1e_satd_8x32_avx2,
      (32, 8) => rav1e_satd_32x8_avx2,
      (16, 64) => rav1e_satd_16x64_avx2,
      (64, 16) => rav1e_satd_64x16_avx2,
      (32, 128) => rav1e_satd_32x128_avx2,
      (128, 32) => rav1e_satd_128x32_avx2,

      _ => unreachable!()
    };
    func(org_ptr, org_stride, ref_ptr, ref_stride)
  }

  #[allow(unused)]
  #[inline(always)]
  pub fn get_satd<T: Pixel>(
    plane_org: &PlaneRegion<'_, T>, plane_ref: &PlaneRegion<'_, T>,
    blk_w: usize, blk_h: usize, bit_depth: usize
  ) -> u32 {
    #[cfg(all(target_arch = "x86_64", feature = "nasm"))]
    {
      if mem::size_of::<T>() == 1
        && is_x86_feature_detected!("avx2")
        && blk_h >= 4
        && blk_w >= 4
      {
        return unsafe {
          let plane_org =
            &*(plane_org as *const _ as *const PlaneRegion<'_, u8>);
          let plane_ref =
            &*(plane_ref as *const _ as *const PlaneRegion<'_, u8>);
          satd_avx2(plane_org, plane_ref, blk_w, blk_h)
        };
      }
    }
    super::native::get_satd(plane_org, plane_ref, blk_w, blk_h, bit_depth)
  }
}

mod native {
  use crate::tiling::*;
  use crate::util::*;

  #[inline(always)]
  pub fn get_sad<T: Pixel>(
    plane_org: &PlaneRegion<'_, T>, plane_ref: &PlaneRegion<'_, T>,
    blk_w: usize, blk_h: usize, _bit_depth: usize
  ) -> u32 {
    let mut sum = 0 as u32;

    for (slice_org, slice_ref) in
      plane_org.rows_iter().take(blk_h).zip(plane_ref.rows_iter())
    {
      sum += slice_org
        .iter()
        .take(blk_w)
        .zip(slice_ref)
        .map(|(&a, &b)| (i32::cast_from(a) - i32::cast_from(b)).abs() as u32)
        .sum::<u32>();
    }

    sum
  }

  #[inline(always)]
  fn butterfly(a: i32, b: i32) -> (i32, i32) {
    ((a + b), (a - b))
  }

  #[inline(always)]
  #[allow(clippy::identity_op, clippy::erasing_op)]
  fn hadamard4_1d(data: &mut [i32], n: usize, stride0: usize, stride1: usize) {
    for i in 0..n {
      let sub: &mut [i32] = &mut data[i * stride0..];
      let (a0, a1) = butterfly(sub[0 * stride1], sub[1 * stride1]);
      let (a2, a3) = butterfly(sub[2 * stride1], sub[3 * stride1]);
      let (b0, b2) = butterfly(a0, a2);
      let (b1, b3) = butterfly(a1, a3);
      sub[0 * stride1] = b0;
      sub[1 * stride1] = b1;
      sub[2 * stride1] = b2;
      sub[3 * stride1] = b3;
    }
  }

  #[inline(always)]
  #[allow(clippy::identity_op, clippy::erasing_op)]
  fn hadamard8_1d(data: &mut [i32], n: usize, stride0: usize, stride1: usize) {
    for i in 0..n {
      let sub: &mut [i32] = &mut data[i * stride0..];

      let (a0, a1) = butterfly(sub[0 * stride1], sub[1 * stride1]);
      let (a2, a3) = butterfly(sub[2 * stride1], sub[3 * stride1]);
      let (a4, a5) = butterfly(sub[4 * stride1], sub[5 * stride1]);
      let (a6, a7) = butterfly(sub[6 * stride1], sub[7 * stride1]);

      let (b0, b2) = butterfly(a0, a2);
      let (b1, b3) = butterfly(a1, a3);
      let (b4, b6) = butterfly(a4, a6);
      let (b5, b7) = butterfly(a5, a7);

      let (c0, c4) = butterfly(b0, b4);
      let (c1, c5) = butterfly(b1, b5);
      let (c2, c6) = butterfly(b2, b6);
      let (c3, c7) = butterfly(b3, b7);

      sub[0 * stride1] = c0;
      sub[1 * stride1] = c1;
      sub[2 * stride1] = c2;
      sub[3 * stride1] = c3;
      sub[4 * stride1] = c4;
      sub[5 * stride1] = c5;
      sub[6 * stride1] = c6;
      sub[7 * stride1] = c7;
    }
  }

  #[inline(always)]
  fn hadamard2d(data: &mut [i32], (w, h): (usize, usize)) {
    /*Vertical transform.*/
    let vert_func = if h == 4 { hadamard4_1d } else { hadamard8_1d };
    vert_func(data, w, 1, h);
    /*Horizontal transform.*/
    let horz_func = if w == 4 { hadamard4_1d } else { hadamard8_1d };
    horz_func(data, h, w, 1);
  }

  fn hadamard4x4(data: &mut [i32]) {
    hadamard2d(data, (4, 4));
  }

  fn hadamard8x8(data: &mut [i32]) {
    hadamard2d(data, (8, 8));
  }

  /// Sum of absolute transformed differences
  /// Use the sum of 4x4 and 8x8 hadamard transforms for the transform. 4x* and
  /// *x4 blocks use 4x4 and all others use 8x8.
  #[allow(unused)]
  #[inline(always)]
  pub fn get_satd<T: Pixel>(
    plane_org: &PlaneRegion<'_, T>, plane_ref: &PlaneRegion<'_, T>,
    blk_w: usize, blk_h: usize, _bit_depth: usize
  ) -> u32 {
    // Size of hadamard transform should be 4x4 or 8x8
    // 4x* and *x4 use 4x4 and all other use 8x8
    let size: usize = blk_w.min(blk_h).min(8);
    let tx2d = if size == 4 { hadamard4x4 } else { hadamard8x8 };

    let mut sum = 0 as u64;

    // Loop over chunks the size of the chosen transform
    for chunk_y in (0..blk_h).step_by(size) {
      for chunk_x in (0..blk_w).step_by(size) {
        let chunk_area: Area = Area::Rect {
          x: chunk_x as isize,
          y: chunk_y as isize,
          width: size,
          height: size
        };
        let chunk_org = plane_org.subregion(chunk_area);
        let chunk_ref = plane_ref.subregion(chunk_area);
        let buf: &mut [i32] = &mut [0; 8 * 8][..size * size];

        // Move the difference of the transforms to a buffer
        for (row_diff, (row_org, row_ref)) in buf
          .chunks_mut(size)
          .zip(chunk_org.rows_iter().zip(chunk_ref.rows_iter()))
        {
          for (diff, (a, b)) in
            row_diff.iter_mut().zip(row_org.iter().zip(row_ref.iter()))
          {
            *diff = i32::cast_from(*a) - i32::cast_from(*b);
          }
        }

        // Perform the hadamard transform on the differences
        tx2d(buf);

        // Sum the absolute values of the transformed differences
        sum += buf.iter().map(|a| a.abs() as u64).sum::<u64>();
      }
    }

    // Normalize the results
    let ln = msb(size as i32) as u64;
    ((sum + (1 << ln >> 1)) >> ln) as u32
  }
}

#[cfg(test)]
pub mod test {
  use super::*;
  use crate::frame::*;
  use crate::partition::BlockSize;
  use crate::partition::BlockSize::*;
  use crate::tiling::Area;
  use crate::util::Pixel;

  // Generate plane data for get_sad_same()
  fn setup_planes<T: Pixel>() -> (Plane<T>, Plane<T>) {
    // Two planes with different strides
    let mut input_plane = Plane::new(640, 480, 0, 0, 128 + 8, 128 + 8);
    let mut rec_plane = Plane::new(640, 480, 0, 0, 2 * 128 + 8, 2 * 128 + 8);

    // Make the test pattern robust to data alignment
    let xpad_off =
      (input_plane.cfg.xorigin - input_plane.cfg.xpad) as i32 - 8i32;

    for (i, row) in
      input_plane.data.chunks_mut(input_plane.cfg.stride).enumerate()
    {
      for (j, pixel) in row.into_iter().enumerate() {
        let val = (j + i) as i32 - xpad_off & 255i32;
        assert!(
          val >= u8::min_value().into() && val <= u8::max_value().into()
        );
        *pixel = T::cast_from(val);
      }
    }

    for (i, row) in rec_plane.data.chunks_mut(rec_plane.cfg.stride).enumerate()
    {
      for (j, pixel) in row.into_iter().enumerate() {
        let val = j as i32 - i as i32 - xpad_off & 255i32;
        assert!(
          val >= u8::min_value().into() && val <= u8::max_value().into()
        );
        *pixel = T::cast_from(val);
      }
    }

    (input_plane, rec_plane)
  }

  // Regression and validation test for SAD computation
  fn get_sad_same_inner<T: Pixel>() {
    let blocks: Vec<(BlockSize, u32)> = vec![
      (BLOCK_4X4, 1912),
      (BLOCK_4X8, 4296),
      (BLOCK_8X4, 3496),
      (BLOCK_8X8, 7824),
      (BLOCK_8X16, 16592),
      (BLOCK_16X8, 14416),
      (BLOCK_16X16, 31136),
      (BLOCK_16X32, 60064),
      (BLOCK_32X16, 59552),
      (BLOCK_32X32, 120128),
      (BLOCK_32X64, 186688),
      (BLOCK_64X32, 250176),
      (BLOCK_64X64, 438912),
      (BLOCK_64X128, 654272),
      (BLOCK_128X64, 1016768),
      (BLOCK_128X128, 1689792),
      (BLOCK_4X16, 8680),
      (BLOCK_16X4, 6664),
      (BLOCK_8X32, 31056),
      (BLOCK_32X8, 27600),
      (BLOCK_16X64, 93344),
      (BLOCK_64X16, 116384),
    ];

    let bit_depth: usize = 8;
    let (input_plane, rec_plane) = setup_planes::<T>();

    for block in blocks {
      let bsw = block.0.width();
      let bsh = block.0.height();
      let area = Area::StartingAt { x: 32, y: 40 };

      let mut input_region = input_plane.region(area);
      let mut rec_region = rec_plane.region(area);

      assert_eq!(
        block.1,
        get_sad(&mut input_region, &mut rec_region, bsw, bsh, bit_depth)
      );
    }
  }

  #[test]
  fn get_sad_same_u8() {
    get_sad_same_inner::<u8>();
  }

  #[test]
  fn get_sad_same_u16() {
    get_sad_same_inner::<u16>();
  }

  fn get_satd_same_inner<T: Pixel>() {
    let blocks: Vec<(BlockSize, u32)> = vec![
      (BLOCK_4X4, 1408),
      (BLOCK_4X8, 2016),
      (BLOCK_8X4, 1816),
      (BLOCK_8X8, 3984),
      (BLOCK_8X16, 5136),
      (BLOCK_16X8, 4864),
      (BLOCK_16X16, 9984),
      (BLOCK_16X32, 13824),
      (BLOCK_32X16, 13760),
      (BLOCK_32X32, 27952),
      (BLOCK_32X64, 37168),
      (BLOCK_64X32, 45104),
      (BLOCK_64X64, 84176),
      (BLOCK_64X128, 127920),
      (BLOCK_128X64, 173680),
      (BLOCK_128X128, 321456),
      (BLOCK_4X16, 3136),
      (BLOCK_16X4, 2632),
      (BLOCK_8X32, 7056),
      (BLOCK_32X8, 6624),
      (BLOCK_16X64, 18432),
      (BLOCK_64X16, 21312),
    ];

    let bit_depth: usize = 8;
    let (input_plane, rec_plane) = setup_planes::<T>();

    for block in blocks {
      let bsw = block.0.width();
      let bsh = block.0.height();
      let area = Area::StartingAt { x: 32, y: 40 };

      let mut input_region = input_plane.region(area);
      let mut rec_region = rec_plane.region(area);

      assert_eq!(
        block.1,
        get_satd(&mut input_region, &mut rec_region, bsw, bsh, bit_depth)
      );
    }
  }

  #[test]
  fn get_satd_same_u8() {
    get_satd_same_inner::<u8>();
  }

  #[test]
  fn get_satd_same_u16() {
    get_satd_same_inner::<u16>();
  }
}
