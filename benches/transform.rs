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
use rav1e::transform;

fn bench_idct4(b: &mut Bencher, bit_depth: &usize) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let input: [i32; 4] = ra.gen();
  let mut output = [0i32; 4];
  let range = bit_depth + 8;

  b.iter(|| {
    transform::av1_idct4(&input[..], &mut output[..], range);
  });
}

pub fn av1_idct4(c: &mut Criterion) {
  let plain = Fun::new("plain", bench_idct4);
  let funcs = vec![plain];

  c.bench_functions("av1_idct4_8", funcs, 8);
}

fn bench_idct8(b: &mut Bencher, bit_depth: &usize) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let input: [i32; 8] = ra.gen();
  let mut output = [0i32; 8];
  let range = bit_depth + 8;

  b.iter(|| {
    transform::av1_idct8(&input[..], &mut output[..], range);
  });
}

pub fn av1_idct8(c: &mut Criterion) {
  let plain = Fun::new("plain", bench_idct8);
  let funcs = vec![plain];

  c.bench_functions("av1_idct8_8", funcs, 8);
}

criterion_group!(transform, av1_idct4, av1_idct8);
