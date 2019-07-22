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

  c.bench_function("av1_idct4_8", move |b| b.iter(
    || transform::av1_idct4(&input[..], &mut output[..], 16)));
}

pub fn av1_idct8(c: &mut Criterion) {
  let (input, mut output) = init_buffers(8);

  c.bench_function("av1_idct8_8", move |b| b.iter(
    || transform::av1_idct8(&input[..], &mut output[..], 16)));
}

criterion_group!(transform, av1_idct4, av1_idct8);
