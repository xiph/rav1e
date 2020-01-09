use criterion::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use rav1e::prelude::Plane;

fn init_plane_u8(width: usize, height: usize) -> Plane<u8> {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let data: Vec<u8> = (0..(width * height)).map(|_| ra.gen()).collect();
  Plane::wrap(data, width)
}

fn init_plane_u16(width: usize, height: usize) -> Plane<u16> {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let data: Vec<u16> = (0..(width * height)).map(|_| ra.gen()).collect();
  Plane::wrap(data, width)
}

pub fn downsample_8bit(c: &mut Criterion) {
  let input = init_plane_u8(1920, 1080);
  let mut output = Plane::new(
    input.cfg.width / 2,
    input.cfg.height / 2,
    1,
    1,
    input.cfg.xpad / 2,
    input.cfg.ypad / 2,
  );
  c.bench_function("downsample_8bit", move |b| {
    b.iter(|| {
      output.downsample_from(&input);
    })
  });
}

pub fn downsample_10bit(c: &mut Criterion) {
  let input = init_plane_u16(1920, 1080);
  let mut output = Plane::new(
    input.cfg.width / 2,
    input.cfg.height / 2,
    1,
    1,
    input.cfg.xpad / 2,
    input.cfg.ypad / 2,
  );
  c.bench_function("downsample_10bit", move |b| {
    b.iter(|| {
      output.downsample_from(&input);
    })
  });
}

criterion_group!(plane, downsample_8bit, downsample_10bit);
