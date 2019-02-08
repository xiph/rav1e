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

fn fill_plane(ra: &mut ChaChaRng, plane: &mut Plane) {
  let stride = plane.cfg.stride;
  for row in plane.data.chunks_mut(stride) {
    for pixel in row {
      let v: u8 = ra.gen();
      *pixel = v as u16;
    }
  }
}

fn new_plane(ra: &mut ChaChaRng, width: usize, height: usize) -> Plane {
  let mut p = Plane::new(width, height, 0, 0, 128 + 8, 128 + 8);

  fill_plane(ra, &mut p);

  p
}

fn bench_get_sad(b: &mut Bencher, bs: &BlockSize) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let bsw = bs.width();
  let bsh = bs.height();
  let w = 640;
  let h = 480;
  let bit_depth = 10;
  let input_plane = new_plane(&mut ra, w, h);
  let rec_plane = new_plane(&mut ra, w, h);
  let po = PlaneOffset { x: 0, y: 0 };

  let plane_org = input_plane.slice(&po);
  let plane_ref = rec_plane.slice(&po);

  b.iter(|| {
    let _ =
      black_box(me::get_sad(&plane_org, &plane_ref, bsw, bsh, bit_depth));
  })
}

pub fn get_sad(c: &mut Criterion) {
  let blocks = vec![
    BLOCK_4X4,
    BLOCK_4X8,
    BLOCK_8X4,
    BLOCK_8X8,
    BLOCK_8X16,
    BLOCK_16X8,
    BLOCK_16X16,
    BLOCK_16X32,
    BLOCK_32X16,
    BLOCK_32X32,
    BLOCK_32X64,
    BLOCK_64X32,
    BLOCK_64X64,
    BLOCK_64X128,
    BLOCK_128X64,
    BLOCK_128X128,
    BLOCK_4X16,
    BLOCK_16X4,
    BLOCK_8X32,
    BLOCK_32X8,
    BLOCK_16X64,
    BLOCK_64X16,
  ];

  c.bench_function_over_inputs("get_sad", bench_get_sad, blocks);
}
