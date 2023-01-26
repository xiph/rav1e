use super::new_plane;
use criterion::*;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use rav1e::bench::denoise::*;
use rav1e::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;

fn bench_dft_denoiser_8b(c: &mut Criterion) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let w = 640;
  let h = 480;
  let mut frame_queue = BTreeMap::new();
  for i in 0..3 {
    frame_queue.insert(
      i,
      Some(Arc::new(Frame {
        planes: [
          new_plane::<u8>(&mut ra, w, h),
          new_plane::<u8>(&mut ra, w / 2, h / 2),
          new_plane::<u8>(&mut ra, w / 2, h / 2),
        ],
      })),
    );
  }
  frame_queue.insert(3, None);

  c.bench_function("dft_denoiser_8b", |b| {
    b.iter_with_setup(
      || DftDenoiser::new(2.0, w, h, 8, ChromaSampling::Cs420),
      |mut denoiser| {
        for _ in 0..3 {
          let _ = black_box(denoiser.filter_frame(&frame_queue));
        }
      },
    )
  });
}

fn bench_dft_denoiser_10b(c: &mut Criterion) {
  let mut ra = ChaChaRng::from_seed([0; 32]);
  let w = 640;
  let h = 480;
  let mut frame_queue = BTreeMap::new();
  for i in 0..3 {
    let mut frame = Frame {
      planes: [
        new_plane::<u16>(&mut ra, w, h),
        new_plane::<u16>(&mut ra, w / 2, h / 2),
        new_plane::<u16>(&mut ra, w / 2, h / 2),
      ],
    };
    for p in 0..3 {
      // Shift from 16-bit to 10-bit
      frame.planes[p].data.iter_mut().for_each(|pix| {
        *pix = *pix >> 6;
      });
    }
    frame_queue.insert(i, Some(Arc::new(frame)));
  }
  frame_queue.insert(3, None);

  c.bench_function("dft_denoiser_10b", |b| {
    b.iter_with_setup(
      || DftDenoiser::new(2.0, w, h, 10, ChromaSampling::Cs420),
      |mut denoiser| {
        for _ in 0..3 {
          let _ = black_box(denoiser.filter_frame(&frame_queue));
        }
      },
    )
  });
}

criterion_group!(denoise, bench_dft_denoiser_8b, bench_dft_denoiser_10b);
