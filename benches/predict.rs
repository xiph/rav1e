// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use bencher::*;
use rand::{ChaChaRng, Rng, SeedableRng};
use rav1e::partition::BlockSize;
use rav1e::predict::{Block4x4, Intra};

pub const MAX_ITER: usize = 50000;
pub const BLOCK_SIZE: BlockSize = BlockSize::BLOCK_32X32;

pub fn generate_block(rng: &mut ChaChaRng) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let block = vec![0u16; BLOCK_SIZE.width() * BLOCK_SIZE.height()];
  let above_context: Vec<u16> = (0..BLOCK_SIZE.height()).map(|_| rng.gen()).collect();
  let left_context: Vec<u16> = (0..BLOCK_SIZE.width()).map(|_| rng.gen()).collect();

  (block, above_context, left_context)
}

pub fn intra_dc_4x4(b: &mut Bencher) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let (mut block, above, left) = generate_block(&mut ra);

  b.iter(|| {
    for _ in 0..MAX_ITER {
      Block4x4::pred_dc(&mut block, BLOCK_SIZE.width(), &above[..4], &left[..4]);
    }
  })
}

pub fn intra_h_4x4(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let (mut block, _above, left) = generate_block(&mut rng);

  b.iter(|| {
    for _ in 0..MAX_ITER {
      Block4x4::pred_h(&mut block, BLOCK_SIZE.width(), &left[..4]);
    }
  })
}

pub fn intra_v_4x4(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let (mut block, above, _left) = generate_block(&mut rng);

  b.iter(|| {
    for _ in 0..MAX_ITER {
      Block4x4::pred_v(&mut block, BLOCK_SIZE.width(), &above[..4]);
    }
  })
}

pub fn intra_paeth_4x4(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let (mut block, above, left) = generate_block(&mut rng);
  let above_left = unsafe { *above.as_ptr().offset(-1) };

  b.iter(|| {
    for _ in 0..MAX_ITER {
      Block4x4::pred_paeth(&mut block, BLOCK_SIZE.width(), &above[..4], &left[..4], above_left);
    }
  })
}

pub fn intra_smooth_4x4(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let (mut block, above, left) = generate_block(&mut rng);

  b.iter(|| {
    for _ in 0..MAX_ITER {
      Block4x4::pred_smooth(&mut block, BLOCK_SIZE.width(), &above[..4], &left[..4]);
    }
  })
}

pub fn intra_smooth_h_4x4(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let (mut block, above, left) = generate_block(&mut rng);

  b.iter(|| {
    for _ in 0..MAX_ITER {
      Block4x4::pred_smooth_h(&mut block, BLOCK_SIZE.width(), &above[..4], &left[..4]);
    }
  })
}

pub fn intra_smooth_v_4x4(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let (mut block, above, left) = generate_block(&mut rng);

  b.iter(|| {
    for _ in 0..MAX_ITER {
      Block4x4::pred_smooth_v(&mut block, BLOCK_SIZE.width(), &above[..4], &left[..4]);
    }
  })
}

pub fn intra_cfl_4x4(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let (mut block, _above, _left) = generate_block(&mut rng);
  let ac: Vec<i16> = (0..(32 * 32)).map(|_| rng.gen()).collect();
  let alpha = -1 as i16;

  b.iter(|| {
    for _ in 0..MAX_ITER {
      Block4x4::pred_cfl(&mut block, BLOCK_SIZE.width(), &ac, alpha, 8);
    }
  })
}
