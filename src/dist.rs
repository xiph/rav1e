// Copyright (c) 2019-2022, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

cfg_if::cfg_if! {
  if #[cfg(nasm_x86_64)] {
    pub use crate::asm::x86::dist::*;
  } else if #[cfg(asm_neon)] {
    pub use crate::asm::aarch64::dist::*;
    pub use self::rust::get_weighted_sse;
    pub use self::rust::cdef_dist_kernel;
  } else {
    pub use self::rust::*;
  }
}

pub(crate) mod rust {
  use crate::activity::apply_ssim_boost;
  use crate::cpu_features::CpuFeatureLevel;
  use crate::tiling::*;
  use crate::util::*;

  use crate::encoder::IMPORTANCE_BLOCK_SIZE;
  use crate::rdo::DistortionScale;

  /// Compute the sum of absolute differences over a block.
  /// w and h can be at most 128, the size of the largest block.
  pub fn get_sad<T: Pixel>(
    plane_org: &PlaneRegion<'_, T>, plane_ref: &PlaneRegion<'_, T>, w: usize,
    h: usize, _bit_depth: usize, _cpu: CpuFeatureLevel,
  ) -> u32 {
    assert!(w <= 128 && h <= 128);
    assert!(plane_org.rect().width >= w && plane_org.rect().height >= h);
    assert!(plane_ref.rect().width >= w && plane_ref.rect().height >= h);

    let mut sum: u32 = 0;

    for (slice_org, slice_ref) in
      plane_org.rows_iter().take(h).zip(plane_ref.rows_iter())
    {
      let mut iter_org = slice_org[..w].chunks_exact(4);
      let mut iter_ref = slice_ref[..w].chunks_exact(4);
      for (chunk_org, chunk_ref) in iter_org.by_ref().zip(iter_ref.by_ref()) {
        let chunk_org = wide::i32x4::from([
          i32::cast_from(chunk_org[0]),
          i32::cast_from(chunk_org[1]),
          i32::cast_from(chunk_org[2]),
          i32::cast_from(chunk_org[3]),
        ]);
        let chunk_ref = wide::i32x4::from([
          i32::cast_from(chunk_ref[0]),
          i32::cast_from(chunk_ref[1]),
          i32::cast_from(chunk_ref[2]),
          i32::cast_from(chunk_ref[3]),
        ]);
        let sad = (chunk_org - chunk_ref).abs();
        for i in sad.to_array() {
          sum += i as u32;
        }
      }
      sum += iter_org
        .remainder()
        .iter()
        .zip(iter_ref.remainder().iter())
        .map(|(&a, &b)| (i32::cast_from(a) - i32::cast_from(b)).unsigned_abs())
        .sum::<u32>();
    }

    sum
  }

  #[inline(always)]
  const fn butterfly(a: i32, b: i32) -> (i32, i32) {
    ((a + b), (a - b))
  }

  #[inline(always)]
  #[allow(clippy::identity_op, clippy::erasing_op)]
  fn hadamard4_1d<
    const LEN: usize,
    const N: usize,
    const STRIDE0: usize,
    const STRIDE1: usize,
  >(
    data: &mut [i32; LEN],
  ) {
    for i in 0..N {
      let sub: &mut [i32] = &mut data[i * STRIDE0..];
      let (a0, a1) = butterfly(sub[0 * STRIDE1], sub[1 * STRIDE1]);
      let (a2, a3) = butterfly(sub[2 * STRIDE1], sub[3 * STRIDE1]);
      let (b0, b2) = butterfly(a0, a2);
      let (b1, b3) = butterfly(a1, a3);
      sub[0 * STRIDE1] = b0;
      sub[1 * STRIDE1] = b1;
      sub[2 * STRIDE1] = b2;
      sub[3 * STRIDE1] = b3;
    }
  }

  #[inline(always)]
  #[allow(clippy::identity_op, clippy::erasing_op)]
  fn hadamard8_1d<
    const LEN: usize,
    const N: usize,
    const STRIDE0: usize,
    const STRIDE1: usize,
  >(
    data: &mut [i32; LEN],
  ) {
    for i in 0..N {
      let sub: &mut [i32] = &mut data[i * STRIDE0..];

      let (a0, a1) = butterfly(sub[0 * STRIDE1], sub[1 * STRIDE1]);
      let (a2, a3) = butterfly(sub[2 * STRIDE1], sub[3 * STRIDE1]);
      let (a4, a5) = butterfly(sub[4 * STRIDE1], sub[5 * STRIDE1]);
      let (a6, a7) = butterfly(sub[6 * STRIDE1], sub[7 * STRIDE1]);

      let (b0, b2) = butterfly(a0, a2);
      let (b1, b3) = butterfly(a1, a3);
      let (b4, b6) = butterfly(a4, a6);
      let (b5, b7) = butterfly(a5, a7);

      let (c0, c4) = butterfly(b0, b4);
      let (c1, c5) = butterfly(b1, b5);
      let (c2, c6) = butterfly(b2, b6);
      let (c3, c7) = butterfly(b3, b7);

      sub[0 * STRIDE1] = c0;
      sub[1 * STRIDE1] = c1;
      sub[2 * STRIDE1] = c2;
      sub[3 * STRIDE1] = c3;
      sub[4 * STRIDE1] = c4;
      sub[5 * STRIDE1] = c5;
      sub[6 * STRIDE1] = c6;
      sub[7 * STRIDE1] = c7;
    }
  }

  #[inline(always)]
  fn hadamard2d<const LEN: usize, const W: usize, const H: usize>(
    data: &mut [i32; LEN],
  ) {
    /*Vertical transform.*/
    let vert_func = if H == 4 {
      hadamard4_1d::<LEN, W, 1, H>
    } else {
      hadamard8_1d::<LEN, W, 1, H>
    };
    vert_func(data);
    /*Horizontal transform.*/
    let horz_func = if W == 4 {
      hadamard4_1d::<LEN, H, W, 1>
    } else {
      hadamard8_1d::<LEN, H, W, 1>
    };
    horz_func(data);
  }

  // SAFETY: The length of data must be 16.
  unsafe fn hadamard4x4(data: &mut [i32]) {
    hadamard2d::<{ 4 * 4 }, 4, 4>(&mut *(data.as_mut_ptr() as *mut [i32; 16]));
  }

  // SAFETY: The length of data must be 64.
  unsafe fn hadamard8x8(data: &mut [i32]) {
    hadamard2d::<{ 8 * 8 }, 8, 8>(&mut *(data.as_mut_ptr() as *mut [i32; 64]));
  }

  /// Sum of absolute transformed differences over a block.
  /// w and h can be at most 128, the size of the largest block.
  /// Use the sum of 4x4 and 8x8 hadamard transforms for the transform, but
  /// revert to sad on edges when these transforms do not fit into w and h.
  /// 4x4 transforms instead of 8x8 transforms when width or height < 8.
  pub fn get_satd<T: Pixel>(
    plane_org: &PlaneRegion<'_, T>, plane_ref: &PlaneRegion<'_, T>, w: usize,
    h: usize, _bit_depth: usize, _cpu: CpuFeatureLevel,
  ) -> u32 {
    assert!(w <= 128 && h <= 128);
    assert!(plane_org.rect().width >= w && plane_org.rect().height >= h);
    assert!(plane_ref.rect().width >= w && plane_ref.rect().height >= h);

    // Size of hadamard transform should be 4x4 or 8x8
    // 4x* and *x4 use 4x4 and all other use 8x8
    let size: usize = w.min(h).min(8);
    let tx2d = if size == 4 { hadamard4x4 } else { hadamard8x8 };

    let mut sum: u64 = 0;

    // Loop over chunks the size of the chosen transform
    for chunk_y in (0..h).step_by(size) {
      let chunk_h = (h - chunk_y).min(size);
      for chunk_x in (0..w).step_by(size) {
        let chunk_w = (w - chunk_x).min(size);
        let chunk_area: Area = Area::Rect {
          x: chunk_x as isize,
          y: chunk_y as isize,
          width: chunk_w,
          height: chunk_h,
        };
        let chunk_org = plane_org.subregion(chunk_area);
        let chunk_ref = plane_ref.subregion(chunk_area);

        // Revert to sad on edge blocks (frame edges)
        if chunk_w != size || chunk_h != size {
          sum += get_sad(
            &chunk_org, &chunk_ref, chunk_w, chunk_h, _bit_depth, _cpu,
          ) as u64;
          continue;
        }

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
        // SAFETY: A sufficient number elements exist for the size of the transform.
        unsafe {
          tx2d(buf);
        }

        // Sum the absolute values of the transformed differences
        sum += buf.iter().map(|a| a.unsigned_abs() as u64).sum::<u64>();
      }
    }

    // Normalize the results
    let ln = msb(size as i32) as u64;
    ((sum + (1 << ln >> 1)) >> ln) as u32
  }

  /// Number of bits rounded off before summing in `get_weighted_sse`
  pub const GET_WEIGHTED_SSE_SHIFT: u8 = 8;

  /// Computes weighted sum of squared error.
  ///
  /// Each scale is applied to a 4x4 region in the provided inputs. Each scale
  /// value is a fixed point number, currently [`DistortionScale`].
  ///
  /// Implementations can require alignment (`bw` (block width) for [`src1`] and
  /// [`src2`] and `bw/4` for `scale`).
  #[inline(never)]
  pub fn get_weighted_sse<T: Pixel>(
    src1: &PlaneRegion<'_, T>, src2: &PlaneRegion<'_, T>, scale: &[u32],
    scale_stride: usize, w: usize, h: usize, _bit_depth: usize,
    _cpu: CpuFeatureLevel,
  ) -> u64 {
    // Always chunk and apply scaling on the sse of squares the size of
    // decimated/sub-sampled importance block sizes.
    // Warning: Changing this will require changing/disabling assembly.
    let chunk_size = IMPORTANCE_BLOCK_SIZE >> 1;

    let mut sse: u64 = 0;

    for block_y in 0..(h + chunk_size - 1) / chunk_size {
      for block_x in 0..(w + chunk_size - 1) / chunk_size {
        let mut block_sse: u32 = 0;

        for j in 0..chunk_size.min(h - block_y * chunk_size) {
          let s1 = &src1[block_y * chunk_size + j]
            [block_x * chunk_size..((block_x + 1) * chunk_size).min(w)];
          let s2 = &src2[block_y * chunk_size + j]
            [block_x * chunk_size..((block_x + 1) * chunk_size).min(w)];

          block_sse += s1
            .iter()
            .zip(s2)
            .map(|(&a, &b)| {
              let c = (i16::cast_from(a) - i16::cast_from(b)) as i32;
              (c * c) as u32
            })
            .sum::<u32>();
        }

        sse += (block_sse as u64
          * scale[block_y * scale_stride + block_x] as u64
          + (1 << GET_WEIGHTED_SSE_SHIFT >> 1))
          >> GET_WEIGHTED_SSE_SHIFT;
      }
    }
    let den = DistortionScale::new(1, 1 << GET_WEIGHTED_SSE_SHIFT).0 as u64;
    (sse + (den >> 1)) / den
  }

  /// Number of bits of precision used in `AREA_DIVISORS`
  const AREA_DIVISOR_BITS: u8 = 14;

  /// Lookup table for 2^`AREA_DIVISOR_BITS` / (1 + x)
  #[rustfmt::skip]
  const AREA_DIVISORS: [u16; 64] = [
    16384, 8192, 5461, 4096, 3277, 2731, 2341, 2048, 1820, 1638, 1489, 1365,
     1260, 1170, 1092, 1024,  964,  910,  862,  819,  780,  745,  712,  683,
      655,  630,  607,  585,  565,  546,  529,  512,  496,  482,  468,  455,
      443,  431,  420,  410,  400,  390,  381,  372,  364,  356,  349,  341,
      334,  328,  321,  315,  309,  303,  298,  293,  287,  282,  278,  273,
      269,  264,  260,  256,
  ];

  /// Computes a distortion metric of the sum of squares weighted by activity.
  /// w and h should be <= 8.
  #[inline(never)]
  pub fn cdef_dist_kernel<T: Pixel>(
    src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, w: usize, h: usize,
    bit_depth: usize, _cpu: CpuFeatureLevel,
  ) -> u32 {
    // TODO: Investigate using different constants in ssim boost for block sizes
    // smaller than 8x8.

    debug_assert!(src.plane_cfg.xdec == 0);
    debug_assert!(src.plane_cfg.ydec == 0);
    debug_assert!(dst.plane_cfg.xdec == 0);
    debug_assert!(dst.plane_cfg.ydec == 0);

    // Limit kernel to 8x8
    debug_assert!(w <= 8);
    debug_assert!(h <= 8);

    // Compute the following summations.
    let mut sum_s: u32 = 0; // sum(src_{i,j})
    let mut sum_d: u32 = 0; // sum(dst_{i,j})
    let mut sum_s2: u32 = 0; // sum(src_{i,j}^2)
    let mut sum_d2: u32 = 0; // sum(dst_{i,j}^2)
    let mut sum_sd: u32 = 0; // sum(src_{i,j} * dst_{i,j})
    for j in 0..h {
      let row1 = &src[j][0..w];
      let row2 = &dst[j][0..w];
      for (s, d) in row1.iter().zip(row2) {
        let s: u32 = u32::cast_from(*s);
        let d: u32 = u32::cast_from(*d);
        sum_s += s;
        sum_d += d;

        sum_s2 += s * s;
        sum_d2 += d * d;
        sum_sd += s * d;
      }
    }

    // To get the distortion, compute sum of squared error and apply a weight
    // based on the variance of the two planes.
    let sse = sum_d2 + sum_s2 - 2 * sum_sd;

    // Convert to 64-bits to avoid overflow when squaring
    let sum_s = sum_s as u64;
    let sum_d = sum_d as u64;

    // Calculate the variance (more accurately variance*area) of each plane.
    // var[iance] = avg(X^2) - avg(X)^2 = sum(X^2) / n - sum(X)^2 / n^2
    //    (n = # samples i.e. area)
    // var * n = sum(X^2) - sum(X)^2 / n
    // When w and h are powers of two, this can be done via shifting.
    let div = AREA_DIVISORS[w * h - 1] as u64;
    let div_shift = AREA_DIVISOR_BITS;
    // Due to rounding, negative values can occur when w or h aren't powers of
    // two. Saturate to avoid underflow.
    let mut svar = sum_s2.saturating_sub(
      ((sum_s * sum_s * div + (1 << div_shift >> 1)) >> div_shift) as u32,
    );
    let mut dvar = sum_d2.saturating_sub(
      ((sum_d * sum_d * div + (1 << div_shift >> 1)) >> div_shift) as u32,
    );

    // Scale variances up to 8x8 size.
    //   scaled variance = var * (8x8) / wxh
    // For 8x8, this is a nop. For powers of 2, this is doable with shifting.
    // TODO: It should be possible and faster to do this adjustment in ssim boost
    let scale_shift = AREA_DIVISOR_BITS - 6;
    svar =
      ((svar as u64 * div + (1 << scale_shift >> 1)) >> scale_shift) as u32;
    dvar =
      ((dvar as u64 * div + (1 << scale_shift >> 1)) >> scale_shift) as u32;

    apply_ssim_boost(sse, svar, dvar, bit_depth)
  }
}

#[cfg(test)]
pub mod test {
  use super::*;
  use crate::cpu_features::CpuFeatureLevel;
  use crate::frame::*;
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
      for (j, pixel) in row.iter_mut().enumerate() {
        let val = ((j + i) as i32 - xpad_off) & 255i32;
        assert!(
          val >= u8::min_value().into() && val <= u8::max_value().into()
        );
        *pixel = T::cast_from(val);
      }
    }

    for (i, row) in rec_plane.data.chunks_mut(rec_plane.cfg.stride).enumerate()
    {
      for (j, pixel) in row.iter_mut().enumerate() {
        let val = (j as i32 - i as i32 - xpad_off) & 255i32;
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
    // dynamic allocation: test
    let blocks: Vec<(usize, usize, u32)> = vec![
      (4, 4, 1912),
      (4, 8, 4296),
      (8, 4, 3496),
      (8, 8, 7824),
      (8, 16, 16592),
      (16, 8, 14416),
      (16, 16, 31136),
      (16, 32, 60064),
      (32, 16, 59552),
      (32, 32, 120128),
      (32, 64, 186688),
      (64, 32, 250176),
      (64, 64, 438912),
      (64, 128, 654272),
      (128, 64, 1016768),
      (128, 128, 1689792),
      (4, 16, 8680),
      (16, 4, 6664),
      (8, 32, 31056),
      (32, 8, 27600),
      (16, 64, 93344),
      (64, 16, 116384),
    ];

    let bit_depth: usize = 8;
    let (input_plane, rec_plane) = setup_planes::<T>();

    for (w, h, distortion) in blocks {
      let area = Area::StartingAt { x: 32, y: 40 };

      let input_region = input_plane.region(area);
      let rec_region = rec_plane.region(area);

      assert_eq!(
        distortion,
        get_sad(
          &input_region,
          &rec_region,
          w,
          h,
          bit_depth,
          CpuFeatureLevel::default()
        )
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
    let blocks: Vec<(usize, usize, u32)> = vec![
      (4, 4, 1408),
      (4, 8, 2016),
      (8, 4, 1816),
      (8, 8, 3984),
      (8, 16, 5136),
      (16, 8, 4864),
      (16, 16, 9984),
      (16, 32, 13824),
      (32, 16, 13760),
      (32, 32, 27952),
      (32, 64, 37168),
      (64, 32, 45104),
      (64, 64, 84176),
      (64, 128, 127920),
      (128, 64, 173680),
      (128, 128, 321456),
      (4, 16, 3136),
      (16, 4, 2632),
      (8, 32, 7056),
      (32, 8, 6624),
      (16, 64, 18432),
      (64, 16, 21312),
    ];

    let bit_depth: usize = 8;
    let (input_plane, rec_plane) = setup_planes::<T>();

    for (w, h, distortion) in blocks {
      let area = Area::StartingAt { x: 32, y: 40 };

      let input_region = input_plane.region(area);
      let rec_region = rec_plane.region(area);

      assert_eq!(
        distortion,
        get_satd(
          &input_region,
          &rec_region,
          w,
          h,
          bit_depth,
          CpuFeatureLevel::default()
        )
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
