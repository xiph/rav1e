// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

mod predict;
mod transform;
mod me;

use rav1e::*;
use rav1e::cdef::cdef_filter_frame;
use rav1e::context::*;
use rav1e::partition::*;
use rav1e::predict::*;
use rav1e::rdo::rdo_cfl_alpha;
use rav1e::rdo::RDOType;
use rav1e::ec::*;
use crate::transform::transform;

use criterion::*;
use std::time::Duration;

fn write_b(c: &mut Criterion) {
  for &tx_size in &[TxSize::TX_4X4, TxSize::TX_8X8] {
    for &qi in &[20, 55] {
      let n = format!("write_b_bench({:?}, {})", tx_size, qi);
      c.bench_function(&n, move |b| write_b_bench(b, tx_size, qi));
    }
  }
}

fn write_b_bench(b: &mut Bencher, tx_size: TxSize, qindex: usize) {
  let config = EncoderConfig {
    width: 1024,
    height: 1024,
    quantizer: qindex,
    speed_settings: SpeedSettings::from_preset(10),
    ..Default::default()
  };
  let sequence = Sequence::new(&Default::default());
  let mut fi = FrameInvariants::<u16>::new(config, sequence);
  let mut w = ec::WriterEncoder::new();
  let mut fc = CDFContext::new(fi.base_q_idx);
  let mut fb = FrameBlocks::new(fi.sb_width * 16, fi.sb_height * 16);
  let bc = BlockContext::new(&mut fb);
  let mut fs = FrameState::new(&fi);
  let mut ts = fs.as_tile_state_mut();
  // For now, restoration unit size is locked to superblock size.
  let mut cw = ContextWriter::new(&mut fc, bc);

  let tx_type = TxType::DCT_DCT;

  let sbx = 0;
  let sby = 0;
  let ac = &[0i16; 32 * 32];

  b.iter(|| {
    for &mode in RAV1E_INTRA_MODES {
      let sbo = SuperBlockOffset { x: sbx, y: sby };
      for p in 1..3 {
        ts.qc.update(fi.base_q_idx, tx_size, mode.is_intra(), 8, fi.dc_delta_q[p], fi.ac_delta_q[p]);
        for by in 0..8 {
          for bx in 0..8 {
            // For ex, 8x8 tx should be applied to even numbered (bx,by)
            if (tx_size.width_mi() >> 1) & bx != 0
              || (tx_size.height_mi() >> 1) & by != 0
            {
              continue;
            };
            let bo = sbo.block_offset(bx, by);
            let tx_bo = BlockOffset { x: bo.x + bx, y: bo.y + by };
            let po = tx_bo.plane_offset(&ts.input.planes[p].cfg);
            encode_tx_block(
              &mut fi,
              &mut ts,
              &mut cw,
              &mut w,
              p,
              bo,
              mode,
              tx_size,
              tx_type,
              tx_size.block_size(),
              po,
              false,
              ac,
              0,
              RDOType::PixelDistRealRate,
              false
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

fn cdef_frame_bench(b: &mut Bencher, width: usize, height: usize) {
  let config = EncoderConfig {
    width,
    height,
    quantizer: 100,
    speed_settings: SpeedSettings::from_preset(10),
    ..Default::default()
  };
  let sequence = Sequence::new(&Default::default());
  let fi = FrameInvariants::<u16>::new(config, sequence);
  let mut fb = FrameBlocks::new(fi.sb_width * 16, fi.sb_height * 16);
  let bc = BlockContext::new(&mut fb);
  let mut fs = FrameState::new(&fi);

  b.iter(|| cdef_filter_frame(&fi, &mut fs.rec, &bc.blocks));
}

fn cfl_rdo(c: &mut Criterion) {
  for &bsize in &[
    BlockSize::BLOCK_4X4,
    BlockSize::BLOCK_8X8,
    BlockSize::BLOCK_16X16,
    BlockSize::BLOCK_32X32
  ] {
    let n = format!("cfl_rdo({:?})", bsize);
    c.bench_function(&n, move |b| cfl_rdo_bench(b, bsize));
  }
}

fn cfl_rdo_bench(b: &mut Bencher, bsize: BlockSize) {
  let config = EncoderConfig {
    width: 1024,
    height: 1024,
    quantizer: 100,
    speed_settings: SpeedSettings::from_preset(10),
    ..Default::default()
  };
  let sequence = Sequence::new(&Default::default());
  let fi = FrameInvariants::<u16>::new(config, sequence);
  let mut fs = FrameState::new(&fi);
  let mut ts = fs.as_tile_state_mut();
  let offset = BlockOffset { x: 1, y: 1 };
  b.iter(|| rdo_cfl_alpha(&mut ts, offset, bsize, fi.sequence.bit_depth))
}

fn ec_bench(c: &mut Criterion) {
    c.bench_function("update_cdf_4_native", update_cdf_4_native);
    c.bench_function("update_cdf_4_sse2", update_cdf_4_sse2);
}

fn update_cdf_4_native(b: &mut Bencher) {
    let mut cdf = [7296, 3819, 1616, 0, 0];
    b.iter(|| {
        for i in 0..1000 {
            WriterBase::<WriterRecorder>::update_cdf(&mut cdf, i & 3);
            black_box(cdf);
        }
    });
}

fn update_cdf_4_sse2(b: &mut Bencher) {
    let mut cdf = [7296, 3819, 1616, 0, 0];
    b.iter(|| {
        for i in 0..1000 {
            WriterBase::<WriterRecorder>::update_cdf_4_sse2(&mut cdf, i & 3);
            black_box(cdf);
        }
    });
}

criterion_group!(intra_prediction, predict::pred_bench,);

criterion_group!(cfl, cfl_rdo);
criterion_group!(cdef, cdef_frame);
criterion_group!(write_block, write_b);
criterion_group!{ name = me;
                  config = Criterion::default().warm_up_time(Duration::new(1,0));
                  targets = me::get_sad
}

criterion_group!(ec, ec_bench);

criterion_main!(write_block, intra_prediction, cdef, cfl, me, transform, ec);

