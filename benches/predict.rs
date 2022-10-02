// Copyright (c) 2017-2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use criterion::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use rav1e::bench::cpu_features::CpuFeatureLevel;
use rav1e::bench::frame::*;
use rav1e::bench::partition::BlockSize;
use rav1e::bench::predict::*;
use rav1e::bench::transform::TxSize;
use rav1e::bench::util::*;

pub const BLOCK_SIZE: BlockSize = BlockSize::BLOCK_32X32;

pub fn generate_block<T: Pixel>(
  rng: &mut ChaChaRng, edge_buf: &mut Aligned<[T; 257]>,
) -> (Plane<T>, Vec<i16>) {
  let block = Plane::from_slice(
    &vec![T::cast_from(0); BLOCK_SIZE.width() * BLOCK_SIZE.height()],
    BLOCK_SIZE.width(),
  );
  let ac: Vec<i16> = (0..(32 * 32)).map(|_| rng.gen()).collect();
  for v in edge_buf.data.iter_mut() {
    *v = T::cast_from(rng.gen::<u8>());
  }

  (block, ac)
}

pub fn bench_pred_fn<F>(c: &mut Criterion, id: &str, f: F)
where
  F: FnMut(&mut Bencher) + 'static,
{
  let mut b = c.benchmark_group(id);

  if id.ends_with("_4x4_u8") {
    b.throughput(Throughput::Bytes(16));
  } else if id.ends_with("_4x4") {
    b.throughput(Throughput::Bytes(32));
  }

  b.bench_function(BenchmarkId::from_parameter(id), f);
}

pub fn pred_bench(c: &mut Criterion) {
  bench_pred_fn(c, "intra_dc_4x4", |b: &mut Bencher| {
    intra_bench::<u16>(b, PredictionMode::DC_PRED, PredictionVariant::BOTH)
  });
  bench_pred_fn(c, "intra_dc_128_4x4", |b: &mut Bencher| {
    intra_bench::<u16>(b, PredictionMode::DC_PRED, PredictionVariant::NONE)
  });
  bench_pred_fn(c, "intra_dc_left_4x4", |b: &mut Bencher| {
    intra_bench::<u16>(b, PredictionMode::DC_PRED, PredictionVariant::LEFT)
  });
  bench_pred_fn(c, "intra_dc_top_4x4", |b: &mut Bencher| {
    intra_bench::<u16>(b, PredictionMode::DC_PRED, PredictionVariant::TOP)
  });
  bench_pred_fn(c, "intra_v_4x4", |b: &mut Bencher| {
    intra_bench::<u16>(b, PredictionMode::V_PRED, PredictionVariant::BOTH)
  });
  bench_pred_fn(c, "intra_h_4x4", |b: &mut Bencher| {
    intra_bench::<u16>(b, PredictionMode::H_PRED, PredictionVariant::BOTH)
  });
  bench_pred_fn(c, "intra_smooth_4x4", |b: &mut Bencher| {
    intra_bench::<u16>(b, PredictionMode::SMOOTH_PRED, PredictionVariant::BOTH)
  });
  bench_pred_fn(c, "intra_smooth_v_4x4", |b: &mut Bencher| {
    intra_bench::<u16>(
      b,
      PredictionMode::SMOOTH_V_PRED,
      PredictionVariant::BOTH,
    )
  });
  bench_pred_fn(c, "intra_smooth_h_4x4", |b: &mut Bencher| {
    intra_bench::<u16>(
      b,
      PredictionMode::SMOOTH_H_PRED,
      PredictionVariant::BOTH,
    )
  });
  bench_pred_fn(c, "intra_paeth_4x4", |b: &mut Bencher| {
    intra_bench::<u16>(b, PredictionMode::PAETH_PRED, PredictionVariant::BOTH)
  });
  bench_pred_fn(c, "intra_dc_4x4_u8", |b: &mut Bencher| {
    intra_bench::<u8>(b, PredictionMode::DC_PRED, PredictionVariant::BOTH)
  });
  bench_pred_fn(c, "intra_dc_128_4x4_u8", |b: &mut Bencher| {
    intra_bench::<u8>(b, PredictionMode::DC_PRED, PredictionVariant::NONE)
  });
  bench_pred_fn(c, "intra_dc_left_4x4_u8", |b: &mut Bencher| {
    intra_bench::<u8>(b, PredictionMode::DC_PRED, PredictionVariant::LEFT)
  });
  bench_pred_fn(c, "intra_dc_top_4x4_u8", |b: &mut Bencher| {
    intra_bench::<u8>(b, PredictionMode::DC_PRED, PredictionVariant::TOP)
  });
  bench_pred_fn(c, "intra_v_4x4_u8", |b: &mut Bencher| {
    intra_bench::<u8>(b, PredictionMode::V_PRED, PredictionVariant::BOTH)
  });
  bench_pred_fn(c, "intra_h_4x4_u8", |b: &mut Bencher| {
    intra_bench::<u8>(b, PredictionMode::H_PRED, PredictionVariant::BOTH)
  });
  bench_pred_fn(c, "intra_smooth_4x4_u8", |b: &mut Bencher| {
    intra_bench::<u8>(b, PredictionMode::SMOOTH_PRED, PredictionVariant::BOTH)
  });
  bench_pred_fn(c, "intra_smooth_v_4x4_u8", |b: &mut Bencher| {
    intra_bench::<u8>(
      b,
      PredictionMode::SMOOTH_V_PRED,
      PredictionVariant::BOTH,
    )
  });
  bench_pred_fn(c, "intra_smooth_h_4x4_u8", |b: &mut Bencher| {
    intra_bench::<u8>(
      b,
      PredictionMode::SMOOTH_H_PRED,
      PredictionVariant::BOTH,
    )
  });
  bench_pred_fn(c, "intra_paeth_4x4_u8", |b: &mut Bencher| {
    intra_bench::<u8>(b, PredictionMode::PAETH_PRED, PredictionVariant::BOTH)
  });
}

pub fn intra_bench<T: Pixel>(
  b: &mut Bencher, mode: PredictionMode, variant: PredictionVariant,
) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let mut edge_buf = unsafe { Aligned::uninitialized() };
  let (mut block, ac) = generate_block::<T>(&mut rng, &mut edge_buf);
  let cpu = CpuFeatureLevel::default();
  let bitdepth = match T::type_enum() {
    PixelType::U8 => 8,
    PixelType::U16 => 10,
  };
  let angle = match mode {
    PredictionMode::V_PRED => 90,
    PredictionMode::H_PRED => 180,
    _ => 0,
  };
  b.iter(|| {
    dispatch_predict_intra::<T>(
      mode,
      variant,
      &mut block.as_region_mut(),
      TxSize::TX_4X4,
      bitdepth,
      &ac,
      angle,
      None,
      &edge_buf,
      cpu,
    );
  })
}
