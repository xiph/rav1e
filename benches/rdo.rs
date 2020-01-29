use criterion::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;

use rav1e::bench::frame::AsRegion;
use rav1e::bench::rdo;
use rav1e::bench::tiling::Area;
use rav1e::prelude::Plane;

fn init_plane_u8(width: usize, height: usize, seed: u8) -> Plane<u8> {
  let mut ra = ChaChaRng::from_seed([seed; 32]);
  let data: Vec<u8> = (0..(width * height)).map(|_| ra.gen()).collect();
  Plane::wrap(data, width)
}

pub fn cdef_dist_wxh_8x8(c: &mut Criterion) {
  let src1 = init_plane_u8(8, 8, 1);
  let src2 = init_plane_u8(8, 8, 2);

  c.bench_function("cdef_dist_wxh_8x8", move |b| {
    b.iter(|| {
      rdo::cdef_dist_wxh(
        &src1.region(Area::Rect { x: 0, y: 0, width: 8, height: 8 }),
        &src2.region(Area::Rect { x: 0, y: 0, width: 8, height: 8 }),
        8,
        8,
        8,
        |_, _| 1.0,
      )
    })
  });
}

pub fn sse_wxh_8x8(c: &mut Criterion) {
  let src1 = init_plane_u8(8, 8, 1);
  let src2 = init_plane_u8(8, 8, 2);

  c.bench_function("sse_wxh_8x8", move |b| {
    b.iter(|| {
      rdo::sse_wxh(
        &src1.region(Area::Rect { x: 0, y: 0, width: 8, height: 8 }),
        &src2.region(Area::Rect { x: 0, y: 0, width: 8, height: 8 }),
        8,
        8,
        |_, _| 1.0,
      )
    })
  });
}

pub fn sse_wxh_4x4(c: &mut Criterion) {
  let src1 = init_plane_u8(8, 8, 1);
  let src2 = init_plane_u8(8, 8, 2);

  c.bench_function("sse_wxh_4z4", move |b| {
    b.iter(|| {
      rdo::sse_wxh(
        &src1.region(Area::Rect { x: 0, y: 0, width: 4, height: 4 }),
        &src2.region(Area::Rect { x: 0, y: 0, width: 4, height: 4 }),
        4,
        4,
        |_, _| 1.0,
      )
    })
  });
}

pub fn sse_wxh_2x2(c: &mut Criterion) {
  let mut src1 = init_plane_u8(8, 8, 1);
  let mut src2 = init_plane_u8(8, 8, 2);
  src1.cfg.xdec = 1;
  src1.cfg.ydec = 1;
  src2.cfg.xdec = 1;
  src2.cfg.ydec = 1;

  c.bench_function("sse_wxh_2x2", move |b| {
    b.iter(|| {
      rdo::sse_wxh(
        &src1.region(Area::Rect { x: 0, y: 0, width: 4, height: 4 }),
        &src2.region(Area::Rect { x: 0, y: 0, width: 4, height: 4 }),
        4,
        4,
        |_, _| 1.0,
      )
    })
  });
}

criterion_group!(
  rdo,
  cdef_dist_wxh_8x8,
  sse_wxh_8x8,
  sse_wxh_4x4,
  sse_wxh_2x2,
);
