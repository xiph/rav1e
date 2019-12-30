use criterion::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use rav1e::bench::context::{BlockOffset, PlaneBlockOffset};
use rav1e::bench::encoder::{
  FrameInvariants, Sequence, IMPORTANCE_BLOCK_SIZE,
};
use rav1e::bench::frame::AsRegion;
use rav1e::bench::partition::BlockSize;
use rav1e::bench::rdo;
use rav1e::bench::tiling::Area;
use rav1e::prelude::{EncoderConfig, Plane};

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

fn init_block_importances(width: usize, height: usize) -> Box<[f32]> {
  let mut ra = ChaChaRng::from_seed([1; 32]);
  let w_in_b = width / IMPORTANCE_BLOCK_SIZE;
  let h_in_b = height / IMPORTANCE_BLOCK_SIZE;
  (0..(w_in_b * h_in_b))
    .map(|_| ra.gen_range(0.0f32, 4000.0f32))
    .collect::<Vec<_>>()
    .into_boxed_slice()
}

fn init_intra_costs(width: usize, height: usize) -> Box<[u32]> {
  let mut ra = ChaChaRng::from_seed([2; 32]);
  let w_in_b = width / IMPORTANCE_BLOCK_SIZE;
  let h_in_b = height / IMPORTANCE_BLOCK_SIZE;
  (0..(w_in_b * h_in_b))
    .map(|_| ra.gen_range(0, 4000))
    .collect::<Vec<_>>()
    .into_boxed_slice()
}

pub fn compute_distortion_scale_8x8(c: &mut Criterion) {
  let mut cfg = EncoderConfig::with_speed_preset(6);
  cfg.width = 128;
  cfg.height = 128;
  let seq = Sequence::new(&cfg);
  let mut fi = FrameInvariants::new_key_frame(cfg, seq, 0);
  fi.block_importances = init_block_importances(128, 128);
  fi.lookahead_intra_costs = init_intra_costs(128, 128);
  c.bench_function("compute_distortion_scale_8x8", move |b| {
    b.iter(|| {
      rdo::compute_distortion_scale::<u8>(
        &fi,
        PlaneBlockOffset(BlockOffset { x: 0, y: 0 }),
        BlockSize::BLOCK_8X8,
      )
    })
  });
}

pub fn compute_distortion_scale_64x64(c: &mut Criterion) {
  let mut cfg = EncoderConfig::with_speed_preset(6);
  cfg.width = 128;
  cfg.height = 128;
  let seq = Sequence::new(&cfg);
  let mut fi = FrameInvariants::new_key_frame(cfg, seq, 0);
  fi.block_importances = init_block_importances(128, 128);
  fi.lookahead_intra_costs = init_intra_costs(128, 128);
  c.bench_function("compute_distortion_scale_64x64", move |b| {
    b.iter(|| {
      rdo::compute_distortion_scale::<u8>(
        &fi,
        PlaneBlockOffset(BlockOffset { x: 0, y: 0 }),
        BlockSize::BLOCK_64X64,
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
  compute_distortion_scale_8x8,
  compute_distortion_scale_64x64
);
