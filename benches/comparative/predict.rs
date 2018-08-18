// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use bencher::*;
use comparative::libc;
use predict as predict_native;
use predict::*;
use rand::{ChaChaRng, Rng, SeedableRng};

extern {
  fn highbd_dc_predictor(
    dst: *mut u16, stride: libc::ptrdiff_t, bw: libc::c_int, bh: libc::c_int,
    above: *const u16, left: *const u16, bd: libc::c_int
  );

  fn highbd_h_predictor(
    dst: *mut u16, stride: libc::ptrdiff_t, bw: libc::c_int, bh: libc::c_int,
    above: *const u16, left: *const u16, bd: libc::c_int
  );

  fn highbd_v_predictor(
    dst: *mut u16, stride: libc::ptrdiff_t, bw: libc::c_int, bh: libc::c_int,
    above: *const u16, left: *const u16, bd: libc::c_int
  );

  fn highbd_paeth_predictor(
    dst: *mut u16, stride: libc::ptrdiff_t, bw: libc::c_int, bh: libc::c_int,
    above: *const u16, left: *const u16, bd: libc::c_int
  );

  fn highbd_smooth_predictor(
    dst: *mut u16, stride: libc::ptrdiff_t, bw: libc::c_int, bh: libc::c_int,
    above: *const u16, left: *const u16, bd: libc::c_int
  );

  fn highbd_smooth_h_predictor(
    dst: *mut u16, stride: libc::ptrdiff_t, bw: libc::c_int, bh: libc::c_int,
    above: *const u16, left: *const u16, bd: libc::c_int
  );

  fn highbd_smooth_v_predictor(
    dst: *mut u16, stride: libc::ptrdiff_t, bw: libc::c_int, bh: libc::c_int,
    above: *const u16, left: *const u16, bd: libc::c_int
  );

  fn cfl_predict_hbd_c(
    ac_buf_q3: *const i16, dst: *mut u16, stride: libc::ptrdiff_t,
    alpha_q3: libc::c_int, bd: libc::c_int, bw: libc::c_int, bh: libc::c_int
  );
}

fn predict_intra_4x4_aom(
  b: &mut Bencher, 
  predictor: unsafe extern "C" fn(*mut u16, libc::ptrdiff_t, libc::c_int, libc::c_int, *const u16, *const u16, libc::c_int)) 
{
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let (mut block, above_context, left_context) = generate_block(&mut rng);

  b.iter(|| {
    for _ in 0..MAX_ITER {
      unsafe {
        predictor(block.as_mut_ptr(), BLOCK_SIZE.width() as libc::ptrdiff_t, 
          4, 4, above_context.as_ptr(), left_context.as_ptr(), 8);
      }
    }
  })
}

pub fn intra_dc_4x4_native(b: &mut Bencher) {
  predict_native::intra_dc_4x4(b);
}

pub fn intra_dc_4x4_aom(b: &mut Bencher) {
  predict_intra_4x4_aom(b, highbd_dc_predictor);
}

pub fn intra_h_4x4_native(b: &mut Bencher) {
  predict_native::intra_h_4x4(b);
}

pub fn intra_h_4x4_aom(b: &mut Bencher) {
  predict_intra_4x4_aom(b, highbd_h_predictor);
}

pub fn intra_v_4x4_native(b: &mut Bencher) {
  predict_native::intra_v_4x4(b);
}

pub fn intra_v_4x4_aom(b: &mut Bencher) {
  predict_intra_4x4_aom(b, highbd_v_predictor);
}

pub fn intra_paeth_4x4_native(b: &mut Bencher) {
  predict_native::intra_paeth_4x4(b);
}

pub fn intra_paeth_4x4_aom(b: &mut Bencher) {
  predict_intra_4x4_aom(b, highbd_paeth_predictor);
}

pub fn intra_smooth_4x4_native(b: &mut Bencher) {
  predict_native::intra_smooth_4x4(b);
}

pub fn intra_smooth_4x4_aom(b: &mut Bencher) {
  predict_intra_4x4_aom(b, highbd_smooth_predictor);
}

pub fn intra_smooth_h_4x4_native(b: &mut Bencher) {
  predict_native::intra_smooth_h_4x4(b);
}

pub fn intra_smooth_h_4x4_aom(b: &mut Bencher) {
  predict_intra_4x4_aom(b, highbd_smooth_h_predictor);
}

pub fn intra_smooth_v_4x4_native(b: &mut Bencher) {
  predict_native::intra_smooth_v_4x4(b);
}

pub fn intra_smooth_v_4x4_aom(b: &mut Bencher) {
  predict_intra_4x4_aom(b, highbd_smooth_v_predictor);
}

pub fn intra_cfl_4x4_native(b: &mut Bencher) {
  predict_native::intra_cfl_4x4(b);
}

pub fn intra_cfl_4x4_aom(b: &mut Bencher) {
  let mut rng = ChaChaRng::from_seed([0; 32]);
  let (mut block, _above_context, _left_context) = generate_block(&mut rng);
  let ac: Vec<i16> = (0..(32 * 32)).map(|_| rng.gen()).collect();
  let alpha = -1 as i16;

  b.iter(|| {
    for _ in 0..MAX_ITER {
      unsafe {
        cfl_predict_hbd_c(ac.as_ptr(), block.as_mut_ptr(),
          BLOCK_SIZE.width() as libc::ptrdiff_t, alpha as libc::c_int, 8, 4, 4);
      }
    }
  })
}
