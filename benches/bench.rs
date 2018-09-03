// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#[macro_use]
extern crate criterion;
extern crate rand;
extern crate rav1e;

mod predict;

use criterion::*;
use rav1e::*;
use rav1e::context::*;
use rav1e::cdef::cdef_filter_frame;
use rav1e::ec;
use rav1e::partition::*;
use rav1e::predict::*;

#[cfg(feature = "comparative_bench")]
mod comparative;

fn write_b(c: &mut Criterion) {
  for &tx_size in &[TxSize::TX_4X4, TxSize::TX_8X8] {
    for &qi in &[20, 55] {
      let n = format!("write_b_bench({:?}, {})", tx_size, qi);
      c.bench_function(&n, move |b| write_b_bench(b, tx_size, qi));
    }
  }
}

fn write_b_bench(b: &mut Bencher, tx_size: TxSize, qindex: usize) {
  unsafe {
    av1_rtcd();
    aom_dsp_rtcd();
  }
  let config =
    EncoderConfig { quantizer: qindex, speed: 10, ..Default::default() };
  let mut fi = FrameInvariants::new(1024, 1024, config);
  let mut w = ec::WriterEncoder::new();
  let fc = CDFContext::new(fi.config.quantizer as u8);
  let bc = BlockContext::new(fi.sb_width * 16, fi.sb_height * 16);
  let mut fs = FrameState::new(&fi);
  let mut cw = ContextWriter::new(fc, bc);

  let tx_type = TxType::DCT_DCT;

  let sbx = 0;
  let sby = 0;
  let ac = &[0i16; 32 * 32];

  b.iter(|| {
    for &mode in RAV1E_INTRA_MODES {
      let sbo = SuperBlockOffset { x: sbx, y: sby };
      fs.qc.update(fi.config.quantizer, tx_size, mode.is_intra(), 8);
      for p in 1..3 {
        for by in 0..8 {
          for bx in 0..8 {
            // For ex, 8x8 tx should be applied to even numbered (bx,by)
            if (tx_size.width_mi() >> 1) & bx != 0 ||
              (tx_size.height_mi() >> 1) & by != 0 { continue; };
            let bo = sbo.block_offset(bx, by);
            let tx_bo = BlockOffset { x: bo.x + bx, y: bo.y + by };
            let po = tx_bo.plane_offset(&fs.input.planes[p].cfg);
            encode_tx_block(
              &mut fi,
              &mut fs,
              &mut cw,
              &mut w,
              p,
              &bo,
              mode,
              tx_size,
              tx_type,
              tx_size.block_size(),
              &po,
              false,
              8,
              ac,
              0
            );
          }
        }
      }
    }
  });
}

fn cdef_frame(c: &mut Criterion) {
  let w = 128;
  let h = 128;
  let n = format!("cdef_frame({}, {})", w, h);
  c.bench_function(&n, move |b| cdef_frame_bench(b, w, h));
}

fn cdef_frame_bench(b: &mut Bencher, w: usize, h: usize) {
  let config =
    EncoderConfig { quantizer: 100, speed: 10, ..Default::default() };
  let fi = FrameInvariants::new(w, h, config);
  let mut bc = BlockContext::new(fi.sb_width * 16, fi.sb_height * 16);
  let mut fs = FrameState::new(&fi);

  b.iter(|| cdef_filter_frame(&fi, &mut fs.rec, &mut bc, 8));
}

criterion_group!(
  intra_prediction,
  predict::pred_bench,
);

criterion_group!(cdef, cdef_frame);
criterion_group!(write_block, write_b);

#[cfg(feature = "comparative_bench")]
criterion_main!(comparative::intra_prediction);

#[cfg(not(feature = "comparative_bench"))]
criterion_main!(write_block, intra_prediction, cdef);
