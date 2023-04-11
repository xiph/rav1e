#![allow(clippy::unit_arg)]

use criterion::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use rav1e::bench::context::BlockContext;
use rav1e::bench::context::CDFContext;
use rav1e::bench::context::ContextWriter;
use rav1e::bench::context::FrameBlocks;
use rav1e::bench::cpu_features::*;
use rav1e::bench::encoder::FrameState;
use rav1e::bench::encoder::{motion_compensate, FrameInvariants};
use rav1e::bench::frame::{AsRegion, PlaneOffset, PlaneSlice};
use rav1e::bench::mc::*;
use rav1e::bench::partition::RefType;
use rav1e::bench::util::Aligned;
use rav1e::context::{BlockOffset, TileBlockOffset};
use rav1e::prelude::*;
use std::sync::Arc;

fn bench_put_8tap_top_left_lbd(c: &mut Criterion) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let cpu = CpuFeatureLevel::default();
  let w = 640;
  let h = 480;
  let input_plane = new_plane::<u8>(&mut ra, w, h);
  let mut dst_plane = new_plane::<u8>(&mut ra, w, h);

  let (row_frac, col_frac, src) = get_params(
    &input_plane,
    PlaneOffset { x: 0, y: 0 },
    MotionVector { row: 0, col: 0 },
  );
  c.bench_function("put_8tap_top_left_lbd", |b| {
    b.iter(|| {
      let _ = black_box(put_8tap(
        &mut dst_plane.as_region_mut(),
        src,
        8,
        8,
        col_frac,
        row_frac,
        FilterMode::REGULAR,
        FilterMode::REGULAR,
        8,
        cpu,
      ));
    })
  });
}

fn bench_put_8tap_top_lbd(c: &mut Criterion) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let cpu = CpuFeatureLevel::default();
  let w = 640;
  let h = 480;
  let input_plane = new_plane::<u8>(&mut ra, w, h);
  let mut dst_plane = new_plane::<u8>(&mut ra, w, h);

  let (row_frac, col_frac, src) = get_params(
    &input_plane,
    PlaneOffset { x: 0, y: 0 },
    MotionVector { row: 0, col: 4 },
  );
  c.bench_function("put_8tap_top_lbd", |b| {
    b.iter(|| {
      let _ = black_box(put_8tap(
        &mut dst_plane.as_region_mut(),
        src,
        8,
        8,
        col_frac,
        row_frac,
        FilterMode::REGULAR,
        FilterMode::REGULAR,
        8,
        cpu,
      ));
    })
  });
}

fn bench_put_8tap_left_lbd(c: &mut Criterion) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let cpu = CpuFeatureLevel::default();
  let w = 640;
  let h = 480;
  let input_plane = new_plane::<u8>(&mut ra, w, h);
  let mut dst_plane = new_plane::<u8>(&mut ra, w, h);

  let (row_frac, col_frac, src) = get_params(
    &input_plane,
    PlaneOffset { x: 0, y: 0 },
    MotionVector { row: 4, col: 0 },
  );
  c.bench_function("put_8tap_left_lbd", |b| {
    b.iter(|| {
      let _ = black_box(put_8tap(
        &mut dst_plane.as_region_mut(),
        src,
        8,
        8,
        col_frac,
        row_frac,
        FilterMode::REGULAR,
        FilterMode::REGULAR,
        8,
        cpu,
      ));
    })
  });
}

fn bench_put_8tap_center_lbd(c: &mut Criterion) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let cpu = CpuFeatureLevel::default();
  let w = 640;
  let h = 480;
  let input_plane = new_plane::<u8>(&mut ra, w, h);
  let mut dst_plane = new_plane::<u8>(&mut ra, w, h);

  let (row_frac, col_frac, src) = get_params(
    &input_plane,
    PlaneOffset { x: 0, y: 0 },
    MotionVector { row: 4, col: 4 },
  );
  c.bench_function("put_8tap_center_lbd", |b| {
    b.iter(|| {
      let _ = black_box(put_8tap(
        &mut dst_plane.as_region_mut(),
        src,
        8,
        8,
        col_frac,
        row_frac,
        FilterMode::REGULAR,
        FilterMode::REGULAR,
        8,
        cpu,
      ));
    })
  });
}

fn bench_put_8tap_top_left_hbd(c: &mut Criterion) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let cpu = CpuFeatureLevel::default();
  let w = 640;
  let h = 480;
  let input_plane = new_plane::<u16>(&mut ra, w, h);
  let mut dst_plane = new_plane::<u16>(&mut ra, w, h);

  let (row_frac, col_frac, src) = get_params(
    &input_plane,
    PlaneOffset { x: 0, y: 0 },
    MotionVector { row: 0, col: 0 },
  );
  c.bench_function("put_8tap_top_left_hbd", |b| {
    b.iter(|| {
      let _ = black_box(put_8tap(
        &mut dst_plane.as_region_mut(),
        src,
        8,
        8,
        col_frac,
        row_frac,
        FilterMode::REGULAR,
        FilterMode::REGULAR,
        10,
        cpu,
      ));
    })
  });
}

fn bench_put_8tap_top_hbd(c: &mut Criterion) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let cpu = CpuFeatureLevel::default();
  let w = 640;
  let h = 480;
  let input_plane = new_plane::<u16>(&mut ra, w, h);
  let mut dst_plane = new_plane::<u16>(&mut ra, w, h);

  let (row_frac, col_frac, src) = get_params(
    &input_plane,
    PlaneOffset { x: 0, y: 0 },
    MotionVector { row: 0, col: 4 },
  );
  c.bench_function("put_8tap_top_hbd", |b| {
    b.iter(|| {
      let _ = black_box(put_8tap(
        &mut dst_plane.as_region_mut(),
        src,
        8,
        8,
        col_frac,
        row_frac,
        FilterMode::REGULAR,
        FilterMode::REGULAR,
        10,
        cpu,
      ));
    })
  });
}

fn bench_put_8tap_left_hbd(c: &mut Criterion) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let cpu = CpuFeatureLevel::default();
  let w = 640;
  let h = 480;
  let input_plane = new_plane::<u16>(&mut ra, w, h);
  let mut dst_plane = new_plane::<u16>(&mut ra, w, h);

  let (row_frac, col_frac, src) = get_params(
    &input_plane,
    PlaneOffset { x: 0, y: 0 },
    MotionVector { row: 4, col: 0 },
  );
  c.bench_function("put_8tap_left_hbd", |b| {
    b.iter(|| {
      let _ = black_box(put_8tap(
        &mut dst_plane.as_region_mut(),
        src,
        8,
        8,
        col_frac,
        row_frac,
        FilterMode::REGULAR,
        FilterMode::REGULAR,
        10,
        cpu,
      ));
    })
  });
}

fn bench_put_8tap_center_hbd(c: &mut Criterion) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let cpu = CpuFeatureLevel::default();
  let w = 640;
  let h = 480;
  let input_plane = new_plane::<u16>(&mut ra, w, h);
  let mut dst_plane = new_plane::<u16>(&mut ra, w, h);

  let (row_frac, col_frac, src) = get_params(
    &input_plane,
    PlaneOffset { x: 0, y: 0 },
    MotionVector { row: 4, col: 4 },
  );
  c.bench_function("put_8tap_center_hbd", |b| {
    b.iter(|| {
      let _ = black_box(put_8tap(
        &mut dst_plane.as_region_mut(),
        src,
        8,
        8,
        col_frac,
        row_frac,
        FilterMode::REGULAR,
        FilterMode::REGULAR,
        10,
        cpu,
      ));
    })
  });
}

fn bench_prep_8tap_top_left_lbd(c: &mut Criterion) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let cpu = CpuFeatureLevel::default();
  let w = 640;
  let h = 480;
  let input_plane = new_plane::<u8>(&mut ra, w, h);
  let mut dst = Aligned::<[i16; 128 * 128]>::from_fn(|_| 0);

  let (row_frac, col_frac, src) = get_params(
    &input_plane,
    PlaneOffset { x: 0, y: 0 },
    MotionVector { row: 0, col: 0 },
  );
  c.bench_function("prep_8tap_top_left_lbd", |b| {
    b.iter(|| {
      let _ = black_box(prep_8tap(
        &mut dst.data,
        src,
        8,
        8,
        col_frac,
        row_frac,
        FilterMode::REGULAR,
        FilterMode::REGULAR,
        8,
        cpu,
      ));
    })
  });
}

fn bench_prep_8tap_top_lbd(c: &mut Criterion) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let cpu = CpuFeatureLevel::default();
  let w = 640;
  let h = 480;
  let input_plane = new_plane::<u8>(&mut ra, w, h);
  let mut dst = Aligned::<[i16; 128 * 128]>::from_fn(|_| 0);

  let (row_frac, col_frac, src) = get_params(
    &input_plane,
    PlaneOffset { x: 0, y: 0 },
    MotionVector { row: 0, col: 4 },
  );
  c.bench_function("prep_8tap_top_lbd", |b| {
    b.iter(|| {
      let _ = black_box(prep_8tap(
        &mut dst.data,
        src,
        8,
        8,
        col_frac,
        row_frac,
        FilterMode::REGULAR,
        FilterMode::REGULAR,
        8,
        cpu,
      ));
    })
  });
}

fn bench_prep_8tap_left_lbd(c: &mut Criterion) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let cpu = CpuFeatureLevel::default();
  let w = 640;
  let h = 480;
  let input_plane = new_plane::<u8>(&mut ra, w, h);
  let mut dst = Aligned::<[i16; 128 * 128]>::from_fn(|_| 0);

  let (row_frac, col_frac, src) = get_params(
    &input_plane,
    PlaneOffset { x: 0, y: 0 },
    MotionVector { row: 4, col: 0 },
  );
  c.bench_function("prep_8tap_left_lbd", |b| {
    b.iter(|| {
      let _ = black_box(prep_8tap(
        &mut dst.data,
        src,
        8,
        8,
        col_frac,
        row_frac,
        FilterMode::REGULAR,
        FilterMode::REGULAR,
        8,
        cpu,
      ));
    })
  });
}

fn bench_prep_8tap_center_lbd(c: &mut Criterion) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let cpu = CpuFeatureLevel::default();
  let w = 640;
  let h = 480;
  let input_plane = new_plane::<u8>(&mut ra, w, h);
  let mut dst = Aligned::<[i16; 128 * 128]>::from_fn(|_| 0);

  let (row_frac, col_frac, src) = get_params(
    &input_plane,
    PlaneOffset { x: 0, y: 0 },
    MotionVector { row: 4, col: 4 },
  );
  c.bench_function("prep_8tap_center_lbd", |b| {
    b.iter(|| {
      let _ = black_box(prep_8tap(
        &mut dst.data,
        src,
        8,
        8,
        col_frac,
        row_frac,
        FilterMode::REGULAR,
        FilterMode::REGULAR,
        8,
        cpu,
      ));
    })
  });
}

fn bench_prep_8tap_top_left_hbd(c: &mut Criterion) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let cpu = CpuFeatureLevel::default();
  let w = 640;
  let h = 480;
  let input_plane = new_plane::<u16>(&mut ra, w, h);
  let mut dst = Aligned::<[i16; 128 * 128]>::from_fn(|_| 0);

  let (row_frac, col_frac, src) = get_params(
    &input_plane,
    PlaneOffset { x: 0, y: 0 },
    MotionVector { row: 0, col: 0 },
  );
  c.bench_function("prep_8tap_top_left_hbd", |b| {
    b.iter(|| {
      let _ = black_box(prep_8tap(
        &mut dst.data,
        src,
        8,
        8,
        col_frac,
        row_frac,
        FilterMode::REGULAR,
        FilterMode::REGULAR,
        10,
        cpu,
      ));
    })
  });
}

fn bench_prep_8tap_top_hbd(c: &mut Criterion) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let cpu = CpuFeatureLevel::default();
  let w = 640;
  let h = 480;
  let input_plane = new_plane::<u16>(&mut ra, w, h);
  let mut dst = Aligned::<[i16; 128 * 128]>::from_fn(|_| 0);

  let (row_frac, col_frac, src) = get_params(
    &input_plane,
    PlaneOffset { x: 0, y: 0 },
    MotionVector { row: 0, col: 4 },
  );
  c.bench_function("prep_8tap_top_hbd", |b| {
    b.iter(|| {
      let _ = black_box(prep_8tap(
        &mut dst.data,
        src,
        8,
        8,
        col_frac,
        row_frac,
        FilterMode::REGULAR,
        FilterMode::REGULAR,
        10,
        cpu,
      ));
    })
  });
}

fn bench_prep_8tap_left_hbd(c: &mut Criterion) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let cpu = CpuFeatureLevel::default();
  let w = 640;
  let h = 480;
  let input_plane = new_plane::<u16>(&mut ra, w, h);
  let mut dst = Aligned::<[i16; 128 * 128]>::from_fn(|_| 0);

  let (row_frac, col_frac, src) = get_params(
    &input_plane,
    PlaneOffset { x: 0, y: 0 },
    MotionVector { row: 4, col: 0 },
  );
  c.bench_function("prep_8tap_left_hbd", |b| {
    b.iter(|| {
      let _ = black_box(prep_8tap(
        &mut dst.data,
        src,
        8,
        8,
        col_frac,
        row_frac,
        FilterMode::REGULAR,
        FilterMode::REGULAR,
        10,
        cpu,
      ));
    })
  });
}

fn bench_prep_8tap_center_hbd(c: &mut Criterion) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let cpu = CpuFeatureLevel::default();
  let w = 640;
  let h = 480;
  let input_plane = new_plane::<u16>(&mut ra, w, h);
  let mut dst = Aligned::<[i16; 128 * 128]>::from_fn(|_| 0);

  let (row_frac, col_frac, src) = get_params(
    &input_plane,
    PlaneOffset { x: 0, y: 0 },
    MotionVector { row: 4, col: 4 },
  );
  c.bench_function("prep_8tap_center_hbd", |b| {
    b.iter(|| {
      let _ = black_box(prep_8tap(
        &mut dst.data,
        src,
        8,
        8,
        col_frac,
        row_frac,
        FilterMode::REGULAR,
        FilterMode::REGULAR,
        10,
        cpu,
      ));
    })
  });
}

fn bench_motion_compensate(c: &mut Criterion) {
  let config = Arc::new(EncoderConfig {
    width: 640,
    height: 480,
    bit_depth: 8,
    speed_settings: SpeedSettings::from_preset(2),
    ..Default::default()
  });
  let sequence = Arc::new(Sequence::new(&config));
  let fi = FrameInvariants::<u8>::new(config, sequence);
  let mut fc = CDFContext::new(fi.base_q_idx);
  let mut fb = FrameBlocks::new(fi.sb_width * 16, fi.sb_height * 16);
  let mut tb = fb.as_tile_blocks_mut();
  let bc = BlockContext::new(&mut tb);
  let mut fs = FrameState::new(&fi);
  let mut cw = ContextWriter::new(&mut fc, bc);
  fs.apply_tile_state_mut(|ts| {
    c.bench_function("motion_compensate", |b| {
      b.iter(|| {
        let _ = black_box(motion_compensate(
          &fi,
          ts,
          &mut cw,
          PredictionMode::NEAREST_NEARESTMV,
          [RefType::LAST_FRAME, RefType::ALTREF_FRAME],
          [MotionVector { row: 0, col: 0 }, MotionVector { row: 0, col: 0 }],
          BlockSize::BLOCK_32X16,
          TileBlockOffset(BlockOffset { x: 152, y: 88 }),
          false,
        ));
      })
    });
  });
}

fn bench_motion_compensate_hbd(c: &mut Criterion) {
  let config = Arc::new(EncoderConfig {
    width: 640,
    height: 480,
    bit_depth: 10,
    speed_settings: SpeedSettings::from_preset(2),
    ..Default::default()
  });
  let sequence = Arc::new(Sequence::new(&config));
  let fi = FrameInvariants::<u16>::new(config, sequence);
  let mut fc = CDFContext::new(fi.base_q_idx);
  let mut fb = FrameBlocks::new(fi.sb_width * 16, fi.sb_height * 16);
  let mut tb = fb.as_tile_blocks_mut();
  let bc = BlockContext::new(&mut tb);
  let mut fs = FrameState::new(&fi);
  let mut cw = ContextWriter::new(&mut fc, bc);
  fs.apply_tile_state_mut(|ts| {
    c.bench_function("motion_compensate_hbd", |b| {
      b.iter(|| {
        let _ = black_box(motion_compensate(
          &fi,
          ts,
          &mut cw,
          PredictionMode::NEAREST_NEARESTMV,
          [RefType::LAST_FRAME, RefType::ALTREF_FRAME],
          [MotionVector { row: 0, col: 0 }, MotionVector { row: 0, col: 0 }],
          BlockSize::BLOCK_32X16,
          TileBlockOffset(BlockOffset { x: 152, y: 88 }),
          false,
        ));
      })
    });
  });
}

criterion_group!(
  mc,
  bench_put_8tap_top_left_lbd,
  bench_put_8tap_top_lbd,
  bench_put_8tap_left_lbd,
  bench_put_8tap_center_lbd,
  bench_put_8tap_top_left_hbd,
  bench_put_8tap_top_hbd,
  bench_put_8tap_left_hbd,
  bench_put_8tap_center_hbd,
  bench_prep_8tap_top_left_lbd,
  bench_prep_8tap_top_lbd,
  bench_prep_8tap_left_lbd,
  bench_prep_8tap_center_lbd,
  bench_prep_8tap_top_left_hbd,
  bench_prep_8tap_top_hbd,
  bench_prep_8tap_left_hbd,
  bench_prep_8tap_center_hbd,
  bench_motion_compensate,
  bench_motion_compensate_hbd,
);

fn fill_plane<T: Pixel>(ra: &mut ChaChaRng, plane: &mut Plane<T>) {
  let stride = plane.cfg.stride;
  for row in plane.data_origin_mut().chunks_mut(stride) {
    for pixel in row {
      let v: u8 = ra.gen();
      *pixel = T::cast_from(v);
    }
  }
}

fn new_plane<T: Pixel>(
  ra: &mut ChaChaRng, width: usize, height: usize,
) -> Plane<T> {
  let mut p = Plane::new(width, height, 0, 0, 128 + 8, 128 + 8);

  fill_plane(ra, &mut p);

  p
}

fn get_params<T: Pixel>(
  rec_plane: &Plane<T>, po: PlaneOffset, mv: MotionVector,
) -> (i32, i32, PlaneSlice<T>) {
  let rec_cfg = &rec_plane.cfg;
  let shift_row = 3 + rec_cfg.ydec;
  let shift_col = 3 + rec_cfg.xdec;
  let row_offset = mv.row as i32 >> shift_row;
  let col_offset = mv.col as i32 >> shift_col;
  let row_frac =
    (mv.row as i32 - (row_offset << shift_row)) << (4 - shift_row);
  let col_frac =
    (mv.col as i32 - (col_offset << shift_col)) << (4 - shift_col);
  let qo = PlaneOffset {
    x: po.x + col_offset as isize - 3,
    y: po.y + row_offset as isize - 3,
  };
  (row_frac, col_frac, rec_plane.slice(qo).clamp().subslice(3, 3))
}
