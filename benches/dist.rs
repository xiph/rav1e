// Copyright (c) 2017-2022, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(clippy::trivially_copy_pass_by_ref)]

use criterion::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use rav1e::bench::cpu_features::*;
use rav1e::bench::dist;
use rav1e::bench::frame::*;
use rav1e::bench::partition::BlockSize::*;
use rav1e::bench::partition::*;
use rav1e::bench::rdo::DistortionScale;
use rav1e::bench::tiling::*;
use rav1e::bench::util::Aligned;
use rav1e::Pixel;

const DIST_BENCH_SET: &[(BlockSize, usize)] = &[
  (BLOCK_4X4, 8),
  (BLOCK_4X8, 8),
  (BLOCK_8X4, 8),
  (BLOCK_8X8, 8),
  (BLOCK_8X16, 8),
  (BLOCK_16X8, 8),
  (BLOCK_16X16, 8),
  (BLOCK_16X32, 8),
  (BLOCK_32X16, 8),
  (BLOCK_32X32, 8),
  (BLOCK_32X64, 8),
  (BLOCK_64X32, 8),
  (BLOCK_64X64, 8),
  (BLOCK_64X128, 8),
  (BLOCK_128X64, 8),
  (BLOCK_128X128, 8),
  (BLOCK_4X16, 8),
  (BLOCK_16X4, 8),
  (BLOCK_8X32, 8),
  (BLOCK_32X8, 8),
  (BLOCK_16X64, 8),
  (BLOCK_64X16, 8),
  (BLOCK_4X4, 10),
  (BLOCK_4X8, 10),
  (BLOCK_8X4, 10),
  (BLOCK_8X8, 10),
  (BLOCK_8X16, 10),
  (BLOCK_16X8, 10),
  (BLOCK_16X16, 10),
  (BLOCK_16X32, 10),
  (BLOCK_32X16, 10),
  (BLOCK_32X32, 10),
  (BLOCK_32X64, 10),
  (BLOCK_64X32, 10),
  (BLOCK_64X64, 10),
  (BLOCK_64X128, 10),
  (BLOCK_128X64, 10),
  (BLOCK_128X128, 10),
  (BLOCK_4X16, 10),
  (BLOCK_16X4, 10),
  (BLOCK_8X32, 10),
  (BLOCK_32X8, 10),
  (BLOCK_16X64, 10),
  (BLOCK_64X16, 10),
];

fn fill_plane<T: Pixel>(ra: &mut ChaChaRng, plane: &mut Plane<T>) {
  let stride = plane.cfg.stride;
  for row in plane.data_origin_mut().chunks_mut(stride) {
    for pixel in row {
      let v: u8 = ra.random();
      *pixel = T::cast_from(v);
    }
  }
}

fn new_plane<T: Pixel>(
  ra: &mut ChaChaRng, width: usize, height: usize,
) -> Plane<T> {
  let mut p = Plane::new(width, height, 0, 0, 128 + 8, 128 + 8);

  fill_plane(ra, &mut p);

  p
}

type DistFn<T> = fn(
  plane_org: &PlaneRegion<'_, T>,
  plane_ref: &PlaneRegion<'_, T>,
  w: usize,
  h: usize,
  bit_depth: usize,
  cpu: CpuFeatureLevel,
) -> u32;

fn run_dist_bench<T: Pixel>(
  b: &mut Bencher, &(bs, bit_depth): &(BlockSize, usize), func: DistFn<T>,
) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let cpu = CpuFeatureLevel::default();
  let w = 640;
  let h = 480;
  let input_plane = new_plane::<T>(&mut ra, w, h);
  let rec_plane = new_plane::<T>(&mut ra, w, h);

  let plane_org = input_plane.as_region();
  let plane_ref = rec_plane.as_region();

  let blk_w = bs.width();
  let blk_h = bs.height();

  b.iter(|| {
    let _ =
      black_box(func(&plane_org, &plane_ref, blk_w, blk_h, bit_depth, cpu));
  })
}

fn bench_get_sad(b: &mut Bencher, &(bs, bit_depth): &(BlockSize, usize)) {
  if bit_depth <= 8 {
    run_dist_bench::<u8>(b, &(bs, bit_depth), dist::get_sad::<u8>)
  } else {
    run_dist_bench::<u16>(b, &(bs, bit_depth), dist::get_sad::<u16>)
  }
}

pub fn get_sad(c: &mut Criterion) {
  let mut b = c.benchmark_group("get_sad");

  for i in DIST_BENCH_SET.iter() {
    b.bench_with_input(
      BenchmarkId::new(i.0.to_string(), i.1),
      i,
      bench_get_sad,
    );
  }
}

fn bench_get_satd(b: &mut Bencher, &(bs, bit_depth): &(BlockSize, usize)) {
  if bit_depth <= 8 {
    run_dist_bench::<u8>(b, &(bs, bit_depth), dist::get_satd::<u8>)
  } else {
    run_dist_bench::<u16>(b, &(bs, bit_depth), dist::get_satd::<u16>)
  }
}

pub fn get_satd(c: &mut Criterion) {
  let mut b = c.benchmark_group("get_satd");

  for i in DIST_BENCH_SET.iter() {
    b.bench_with_input(
      BenchmarkId::new(i.0.to_string(), i.1),
      i,
      bench_get_satd,
    );
  }
}

/// Fill data for scaling of one
fn fill_scaling(ra: &mut ChaChaRng, scales: &mut [u32]) {
  for a in scales.iter_mut() {
    *a = ra.random_range(
      DistortionScale::from(0.5).0..DistortionScale::from(1.5).0,
    );
  }
}

type WeightedSseFn<T> = fn(
  plane_org: &PlaneRegion<'_, T>,
  plane_ref: &PlaneRegion<'_, T>,
  scale: &[u32],
  scale_stride: usize,
  w: usize,
  h: usize,
  bit_depth: usize,
  cpu: CpuFeatureLevel,
) -> u64;

fn run_weighted_sse_bench<T: Pixel>(
  b: &mut Bencher, &(bs, bit_depth): &(BlockSize, usize),
  func: WeightedSseFn<T>,
) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let cpu = CpuFeatureLevel::default();
  let w = 640;
  let h = 480;
  let input_plane = new_plane::<T>(&mut ra, w, h);
  let rec_plane = new_plane::<T>(&mut ra, w, h);

  const SCALE_STRIDE: usize = 256;
  let mut scaling_storage = Aligned::new([0u32; 256 * SCALE_STRIDE]);
  let scaling = &mut scaling_storage.data;
  fill_scaling(&mut ra, scaling);

  let plane_org = input_plane.as_region();
  let plane_ref = rec_plane.as_region();

  let blk_w = bs.width();
  let blk_h = bs.width();

  b.iter(|| {
    let _ = black_box(func(
      &plane_org,
      &plane_ref,
      scaling,
      SCALE_STRIDE,
      blk_w,
      blk_h,
      bit_depth,
      cpu,
    ));
  })
}

fn bench_get_weighted_sse(
  b: &mut Bencher, &(bs, bit_depth): &(BlockSize, usize),
) {
  if bit_depth <= 8 {
    run_weighted_sse_bench::<u8>(
      b,
      &(bs, bit_depth),
      dist::get_weighted_sse::<u8>,
    )
  } else {
    run_weighted_sse_bench::<u16>(
      b,
      &(bs, bit_depth),
      dist::get_weighted_sse::<u16>,
    )
  }
}

pub fn get_weighted_sse(c: &mut Criterion) {
  let mut b = c.benchmark_group("get_weighted_sse");

  for i in DIST_BENCH_SET.iter() {
    b.bench_with_input(
      BenchmarkId::new(i.0.to_string(), i.1),
      i,
      bench_get_weighted_sse,
    );
  }
}
