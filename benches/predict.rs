// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use criterion::*;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaChaRng;
use rav1e::bench::frame::*;
use rav1e::bench::partition::BlockSize;
use rav1e::bench::predict::{Block4x4, Intra};
use rav1e::bench::util::*;

pub const BLOCK_SIZE: BlockSize = BlockSize::BLOCK_32X32;

pub fn generate_block(
  rng: &mut ChaChaRng,
) -> (Plane<u16>, Vec<u16>, Vec<u16>) {
  let block = Plane::wrap(
    vec![0u16; BLOCK_SIZE.width() * BLOCK_SIZE.height()],
    BLOCK_SIZE.width(),
  );
  let above_context: Vec<u16> =
    (0..BLOCK_SIZE.height()).map(|_| rng.gen()).collect();
  let left_context: Vec<u16> =
    (0..BLOCK_SIZE.width()).map(|_| rng.gen()).collect();

  (block, above_context, left_context)
}

pub fn generate_block_u8<'a>(
  rng: &mut ChaChaRng, edge_buf: &'a mut AlignedArray<[u8; 65]>,
) -> (Plane<u8>, &'a [u8], &'a [u8]) {
  let block = Plane::wrap(
    vec![0u8; BLOCK_SIZE.width() * BLOCK_SIZE.height()],
    BLOCK_SIZE.width(),
  );
  rng.fill_bytes(&mut edge_buf.array);
  let above_context = &edge_buf.array[33..];
  let left_context = &edge_buf.array[..32];

  (block, above_context, left_context)
}

pub fn bench_pred_fn<F>(c: &mut Criterion, id: &str, f: F)
where
  F: FnMut(&mut Bencher) + 'static,
{
  let b = Benchmark::new(id, f);
  c.bench(
    id,
    if id.ends_with("_4x4_u8") {
      b.throughput(Throughput::Bytes(16))
    } else if id.ends_with("_4x4") {
      b.throughput(Throughput::Bytes(32))
    } else {
      b
    },
  );
}

pub fn pred_bench(c: &mut Criterion) {
  bench_pred_fn(c, "intra_dc_4x4", intra_dc_4x4);
  bench_pred_fn(c, "intra_dc_left_4x4", intra_dc_left_4x4);
  bench_pred_fn(c, "intra_dc_top_4x4", intra_dc_top_4x4);
  bench_pred_fn(c, "intra_h_4x4", intra_h_4x4);
  bench_pred_fn(c, "intra_v_4x4", intra_v_4x4);
  bench_pred_fn(c, "intra_paeth_4x4", intra_paeth_4x4);
  bench_pred_fn(c, "intra_smooth_4x4", intra_smooth_4x4);
  bench_pred_fn(c, "intra_smooth_h_4x4", intra_smooth_h_4x4);
  bench_pred_fn(c, "intra_smooth_v_4x4", intra_smooth_v_4x4);
  bench_pred_fn(c, "intra_cfl_4x4", intra_cfl_4x4);
  bench_pred_fn(c, "intra_dc_4x4_u8", intra_dc_4x4_u8);
  bench_pred_fn(c, "intra_dc_128_4x4_u8", intra_dc_128_4x4_u8);
  bench_pred_fn(c, "intra_dc_left_4x4_u8", intra_dc_left_4x4_u8);
  bench_pred_fn(c, "intra_dc_top_4x4_u8", intra_dc_top_4x4_u8);
  bench_pred_fn(c, "intra_h_4x4_u8", intra_h_4x4_u8);
  bench_pred_fn(c, "intra_v_4x4_u8", intra_v_4x4_u8);
  bench_pred_fn(c, "intra_paeth_4x4_u8", intra_paeth_4x4_u8);
  bench_pred_fn(c, "intra_smooth_4x4_u8", intra_smooth_4x4_u8);
  bench_pred_fn(c, "intra_smooth_h_4x4_u8", intra_smooth_h_4x4_u8);
  bench_pred_fn(c, "intra_smooth_v_4x4_u8", intra_smooth_v_4x4_u8);
}

pub fn intra_dc_4x4(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let (mut block, above, left) = generate_block(&mut rng);

  b.iter(|| {
    Block4x4::pred_dc(&mut block.as_region_mut(), &above[..4], &left[..4]);
  })
}

pub fn intra_dc_left_4x4(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let (mut block, above, left) = generate_block(&mut rng);

  b.iter(|| {
    Block4x4::pred_dc_left(
      &mut block.as_region_mut(),
      &above[..4],
      &left[..4],
    );
  })
}

pub fn intra_dc_top_4x4(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let (mut block, above, left) = generate_block(&mut rng);

  b.iter(|| {
    Block4x4::pred_dc_top(&mut block.as_region_mut(), &above[..4], &left[..4]);
  })
}

pub fn intra_h_4x4(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let (mut block, _above, left) = generate_block(&mut rng);

  b.iter(|| {
    Block4x4::pred_h(&mut block.as_region_mut(), &left[..4]);
  })
}

pub fn intra_v_4x4(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let (mut block, above, _left) = generate_block(&mut rng);

  b.iter(|| {
    Block4x4::pred_v(&mut block.as_region_mut(), &above[..4]);
  })
}

pub fn intra_paeth_4x4(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let (mut block, above, left) = generate_block(&mut rng);
  let above_left = unsafe { *above.as_ptr().offset(-1) };

  b.iter(|| {
    Block4x4::pred_paeth(
      &mut block.as_region_mut(),
      &above[..4],
      &left[..4],
      above_left,
    );
  })
}

pub fn intra_smooth_4x4(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let (mut block, above, left) = generate_block(&mut rng);

  b.iter(|| {
    Block4x4::pred_smooth(&mut block.as_region_mut(), &above[..4], &left[..4]);
  })
}

pub fn intra_smooth_h_4x4(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let (mut block, above, left) = generate_block(&mut rng);

  b.iter(|| {
    Block4x4::pred_smooth_h(
      &mut block.as_region_mut(),
      &above[..4],
      &left[..4],
    );
  })
}

pub fn intra_smooth_v_4x4(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let (mut block, above, left) = generate_block(&mut rng);

  b.iter(|| {
    Block4x4::pred_smooth_v(
      &mut block.as_region_mut(),
      &above[..4],
      &left[..4],
    );
  })
}

pub fn intra_cfl_4x4(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let (mut block, above, left) = generate_block(&mut rng);
  let ac: Vec<i16> = (0..(32 * 32)).map(|_| rng.gen()).collect();
  let alpha = -1 as i16;

  b.iter(|| {
    Block4x4::pred_cfl(
      &mut block.as_region_mut(),
      &ac,
      alpha,
      8,
      &above,
      &left,
    );
  })
}

pub fn intra_dc_4x4_u8(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let mut edge_buf = AlignedArray::uninitialized();
  let (mut block, above, left) = generate_block_u8(&mut rng, &mut edge_buf);

  b.iter(|| {
    Block4x4::pred_dc(
      &mut block.as_region_mut(),
      &above[..4],
      &left[32 - 4..],
    );
  })
}

pub fn intra_dc_128_4x4_u8(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let mut edge_buf = AlignedArray::uninitialized();
  let (mut block, _above, _left) = generate_block_u8(&mut rng, &mut edge_buf);

  b.iter(|| {
    Block4x4::pred_dc_128(&mut block.as_region_mut(), 8);
  })
}

pub fn intra_dc_left_4x4_u8(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let mut edge_buf = AlignedArray::uninitialized();
  let (mut block, above, left) = generate_block_u8(&mut rng, &mut edge_buf);

  b.iter(|| {
    Block4x4::pred_dc_left(
      &mut block.as_region_mut(),
      &above[..4],
      &left[32 - 4..],
    );
  })
}

pub fn intra_dc_top_4x4_u8(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let mut edge_buf = AlignedArray::uninitialized();
  let (mut block, above, left) = generate_block_u8(&mut rng, &mut edge_buf);

  b.iter(|| {
    Block4x4::pred_dc_top(
      &mut block.as_region_mut(),
      &above[..4],
      &left[32 - 4..],
    );
  })
}

pub fn intra_h_4x4_u8(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let mut edge_buf = AlignedArray::uninitialized();
  let (mut block, _above, left) = generate_block_u8(&mut rng, &mut edge_buf);

  b.iter(|| {
    Block4x4::pred_h(&mut block.as_region_mut(), &left[32 - 4..]);
  })
}

pub fn intra_v_4x4_u8(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let mut edge_buf = AlignedArray::uninitialized();
  let (mut block, above, _left) = generate_block_u8(&mut rng, &mut edge_buf);

  b.iter(|| {
    Block4x4::pred_v(&mut block.as_region_mut(), &above[..4]);
  })
}

pub fn intra_paeth_4x4_u8(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let mut edge_buf = AlignedArray::uninitialized();
  let (mut block, above, left) = generate_block_u8(&mut rng, &mut edge_buf);
  let above_left = unsafe { *above.as_ptr().offset(-1) };

  b.iter(|| {
    Block4x4::pred_paeth(
      &mut block.as_region_mut(),
      &above[..4],
      &left[32 - 4..],
      above_left,
    );
  })
}

pub fn intra_smooth_4x4_u8(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let mut edge_buf = AlignedArray::uninitialized();
  let (mut block, above, left) = generate_block_u8(&mut rng, &mut edge_buf);

  b.iter(|| {
    Block4x4::pred_smooth(
      &mut block.as_region_mut(),
      &above[..4],
      &left[32 - 4..],
    );
  })
}

pub fn intra_smooth_h_4x4_u8(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let mut edge_buf = AlignedArray::uninitialized();
  let (mut block, above, left) = generate_block_u8(&mut rng, &mut edge_buf);

  b.iter(|| {
    Block4x4::pred_smooth_h(
      &mut block.as_region_mut(),
      &above[..4],
      &left[32 - 4..],
    );
  })
}

pub fn intra_smooth_v_4x4_u8(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let mut edge_buf = AlignedArray::uninitialized();
  let (mut block, above, left) = generate_block_u8(&mut rng, &mut edge_buf);

  b.iter(|| {
    Block4x4::pred_smooth_v(
      &mut block.as_region_mut(),
      &above[..4],
      &left[32 - 4..],
    );
  })
}
