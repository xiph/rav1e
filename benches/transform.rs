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
use rav1e::bench::transform;

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

pub fn daala_fdct4(c: &mut Criterion) {
  let (input, mut output) = init_buffers(4);

  c.bench_function("daala_fdct4", move |b| {
    b.iter(|| {
      transform::forward::native::daala_fdct4(&input[..], &mut output[..])
    })
  });
}

pub fn daala_fdct8(c: &mut Criterion) {
  let (input, mut output) = init_buffers(8);

  c.bench_function("daala_fdct8", move |b| {
    b.iter(|| {
      transform::forward::native::daala_fdct8(&input[..], &mut output[..])
    })
  });
}

pub fn fidentity4(c: &mut Criterion) {
  let (input, mut output) = init_buffers(4);

  c.bench_function("fidentity4", move |b| {
    b.iter(|| {
      transform::forward::native::fidentity4(&input[..], &mut output[..])
    })
  });
}

pub fn fidentity8(c: &mut Criterion) {
  let (input, mut output) = init_buffers(8);

  c.bench_function("fidentity8", move |b| {
    b.iter(|| {
      transform::forward::native::fidentity8(&input[..], &mut output[..])
    })
  });
}

pub fn daala_fdst_vii_4(c: &mut Criterion) {
  let (input, mut output) = init_buffers(4);

  c.bench_function("daala_fdst_vii_4", move |b| {
    b.iter(|| {
      transform::forward::native::daala_fdst_vii_4(&input[..], &mut output[..])
    })
  });
}

pub fn daala_fdst8(c: &mut Criterion) {
  let (input, mut output) = init_buffers(8);

  c.bench_function("daala_fdst8", move |b| {
    b.iter(|| {
      transform::forward::native::daala_fdst8(&input[..], &mut output[..])
    })
  });
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

criterion_group!(
  forward_transforms,
  daala_fdct4,
  daala_fdct8,
  fidentity4,
  fidentity8,
  daala_fdst_vii_4,
  daala_fdst8
);
