// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use criterion::*;
use crate::partition::*;
use crate::partition::BlockSize::*;
use crate::plane::*;
use rand::{ChaChaRng, Rng, SeedableRng};
use rav1e::me;
use rav1e::Pixel;

fn fill_plane<T: Pixel>(ra: &mut ChaChaRng, plane: &mut Plane<T>) {
  let stride = plane.cfg.stride;
  for row in plane.data.chunks_mut(stride) {
    for pixel in row {
      let v: u8 = ra.gen();
      *pixel = T::cast_from(v);
    }
  }
}

fn new_plane<T: Pixel>(ra: &mut ChaChaRng, width: usize, height: usize) -> Plane<T> {
  let mut p = Plane::new(width, height, 0, 0, 128 + 8, 128 + 8);

  fill_plane(ra, &mut p);

  p
}

fn run_sad_bench<T: Pixel>(b: &mut Bencher, &(bs, bit_depth): &(BlockSize, usize)) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let bsw = bs.width();
  let bsh = bs.height();
  let w = 640;
  let h = 480;
  let input_plane = new_plane::<T>(&mut ra, w, h);
  let rec_plane = new_plane::<T>(&mut ra, w, h);

  let plane_org = input_plane.as_region();
  let plane_ref = rec_plane.as_region();

  b.iter(|| {
    let _ =
      black_box(me::get_sad(&plane_org, &plane_ref, bsw, bsh, bit_depth));
  })
}

fn bench_get_sad(b: &mut Bencher, &(bs, bit_depth): &(BlockSize, usize)) {
  if bit_depth <= 8 {
    run_sad_bench::<u8>(b, &(bs, bit_depth))
  }
  else {
    run_sad_bench::<u16>(b, &(bs, bit_depth))
  }
}

pub fn get_sad(c: &mut Criterion) {
  let blocks = vec![
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
    (BLOCK_64X16, 10)
  ];

  c.bench_function_over_inputs("get_sad", bench_get_sad, blocks);
}
