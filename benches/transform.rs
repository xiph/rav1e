// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
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
use rav1e::bench::cpu_features::*;
use rav1e::bench::transform;
use rav1e::bench::transform::{forward_transform, TxSize, TxType};

fn init_buffers(size: usize) -> (Vec<i32>, Vec<i32>) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let input: Vec<i32> = (0..size).map(|_| ra.gen()).collect();
  let output = vec![0i32; size];

  (input, output)
}

pub fn av1_idct4(c: &mut Criterion) {
  let (input, mut output) = init_buffers(4);

  c.bench_function("av1_idct4_8", move |b| {
    b.iter(|| transform::inverse::av1_idct4(&input[..], &mut output[..], 16))
  });
}

pub fn av1_idct8(c: &mut Criterion) {
  let (input, mut output) = init_buffers(8);

  c.bench_function("av1_idct8_8", move |b| {
    b.iter(|| transform::inverse::av1_idct8(&input[..], &mut output[..], 16))
  });
}

pub fn av1_iidentity4(c: &mut Criterion) {
  let (input, mut output) = init_buffers(4);

  c.bench_function("av1_iidentity4_8", move |b| {
    b.iter(|| {
      transform::inverse::av1_iidentity4(&input[..], &mut output[..], 16)
    })
  });
}

pub fn av1_iidentity8(c: &mut Criterion) {
  let (input, mut output) = init_buffers(8);

  c.bench_function("av1_iidentity8_8", move |b| {
    b.iter(|| {
      transform::inverse::av1_iidentity8(&input[..], &mut output[..], 16)
    })
  });
}

pub fn av1_iadst4(c: &mut Criterion) {
  let (input, mut output) = init_buffers(4);

  c.bench_function("av1_iadst4_8", move |b| {
    b.iter(|| transform::inverse::av1_iadst4(&input[..], &mut output[..], 16))
  });
}

pub fn av1_iadst8(c: &mut Criterion) {
  let (input, mut output) = init_buffers(8);

  c.bench_function("av1_iadst8_8", move |b| {
    b.iter(|| transform::inverse::av1_iadst8(&input[..], &mut output[..], 16))
  });
}

fn get_valid_txfm_types(tx_size: TxSize) -> &'static [TxType] {
  let size_sq = tx_size.sqr_up();
  use TxType::*;
  if size_sq == TxSize::TX_64X64 {
    &[DCT_DCT]
  } else if size_sq == TxSize::TX_32X32 {
    &[DCT_DCT, IDTX]
  } else {
    &[
      DCT_DCT,
      ADST_DCT,
      DCT_ADST,
      ADST_ADST,
      FLIPADST_DCT,
      DCT_FLIPADST,
      FLIPADST_FLIPADST,
      ADST_FLIPADST,
      FLIPADST_ADST,
      IDTX,
      V_DCT,
      H_DCT,
      V_ADST,
      H_ADST,
      V_FLIPADST,
      H_FLIPADST,
    ]
  }
}

pub fn bench_forward_transforms(c: &mut Criterion) {
  let mut group = c.benchmark_group("forward_transform");

  let mut rng = rand::thread_rng();
  let cpu = CpuFeatureLevel::default();

  let tx_sizes = {
    use TxSize::*;
    [
      TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_64X64, TX_4X8, TX_8X4, TX_8X16,
      TX_16X8, TX_16X32, TX_32X16, TX_32X64, TX_64X32, TX_4X16, TX_16X4,
      TX_8X32, TX_32X8, TX_16X64, TX_64X16,
    ]
  };

  for &tx_size in &tx_sizes {
    let area = tx_size.area();

    let input: Vec<i16> =
      (0..area).map(|_| rng.gen_range(-255, 256)).collect();
    let mut output = vec![0i16; area];

    for &tx_type in get_valid_txfm_types(tx_size) {
      group.bench_function(
        format!("{:?}_{:?}", tx_size, tx_type).as_str(),
        |b| {
          b.iter(|| {
            forward_transform(
              &input[..],
              &mut output[..],
              tx_size.width(),
              tx_size,
              tx_type,
              8,
              cpu,
            )
          })
        },
      );
    }
  }

  group.finish();
}

criterion_group!(
  inverse_transforms,
  av1_idct4,
  av1_idct8,
  av1_iidentity4,
  av1_iidentity8,
  av1_iadst4,
  av1_iadst8
);

criterion_group!(forward_transforms, bench_forward_transforms);
