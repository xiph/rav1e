// Copyright (c) 2018-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::encoder::FrameInvariants;
use crate::prelude::*;

use std::sync::Arc;

use interpolate_name::interpolate_test;

fn setup_encoder<T: Pixel>(
  w: usize, h: usize, speed: usize, quantizer: usize, bit_depth: usize,
  chroma_sampling: ChromaSampling, min_keyint: u64, max_keyint: u64,
  bitrate: i32, low_latency: bool, switch_frame_interval: u64,
  no_scene_detection: bool, rdo_lookahead_frames: usize,
) -> Context<T> {
  assert!(bit_depth == 8 || std::mem::size_of::<T>() > 1);
  let mut enc = EncoderConfig::with_speed_preset(speed);
  enc.quantizer = quantizer;
  enc.min_key_frame_interval = min_keyint;
  enc.max_key_frame_interval = max_keyint;
  enc.low_latency = low_latency;
  enc.switch_frame_interval = switch_frame_interval;
  enc.width = w;
  enc.height = h;
  enc.bit_depth = bit_depth;
  enc.chroma_sampling = chroma_sampling;
  enc.bitrate = bitrate;
  enc.speed_settings.no_scene_detection = no_scene_detection;
  enc.rdo_lookahead_frames = rdo_lookahead_frames;

  let cfg = Config::new().with_encoder_config(enc).with_threads(1);

  cfg.new_context().unwrap()
}

/*
fn fill_frame<T: Pixel>(ra: &mut ChaChaRng, frame: &mut Frame<T>) {
  for plane in frame.planes.iter_mut() {
    let stride = plane.cfg.stride;
    for row in plane.data.chunks_mut(stride) {
      for pixel in row {
        let v: u8 = ra.gen();
        *pixel = T::cast_from(v);
      }
    }
  }
}
*/

fn fill_frame_const<T: Pixel>(frame: &mut Frame<T>, value: T) {
  for plane in frame.planes.iter_mut() {
    let stride = plane.cfg.stride;
    for row in plane.data.chunks_mut(stride) {
      for pixel in row {
        *pixel = value;
      }
    }
  }
}

#[interpolate_test(low_latency_no_scene_change, true, true)]
#[interpolate_test(reorder_no_scene_change, false, true)]
#[interpolate_test(low_latency_scene_change_detection, true, false)]
#[interpolate_test(reorder_scene_change_detection, false, false)]
fn flush(low_lantency: bool, no_scene_detection: bool) {
  let mut ctx = setup_encoder::<u8>(
    64,
    80,
    10,
    100,
    8,
    ChromaSampling::Cs420,
    150,
    200,
    0,
    low_lantency,
    0,
    no_scene_detection,
    10,
  );
  let limit = 41;

  for _ in 0..limit {
    let input = ctx.new_frame();
    let _ = ctx.send_frame(input);
  }

  ctx.flush();

  let mut count = 0;

  'out: for _ in 0..limit {
    loop {
      match ctx.receive_packet() {
        Ok(_) => {
          eprintln!("Packet Received {}/{}", count, limit);
          count += 1;
        }
        Err(EncoderStatus::EnoughData) => {
          eprintln!("{:?}", EncoderStatus::EnoughData);

          break 'out;
        }
        Err(e) => {
          eprintln!("{:?}", e);
          break;
        }
      }
    }
  }

  assert_eq!(limit, count);
}

#[interpolate_test(low_latency_no_scene_change, true, true)]
#[interpolate_test(reorder_no_scene_change, false, true)]
#[interpolate_test(low_latency_scene_change_detection, true, false)]
#[interpolate_test(reorder_scene_change_detection, false, false)]
fn flush_unlimited(low_lantency: bool, no_scene_detection: bool) {
  let mut ctx = setup_encoder::<u8>(
    64,
    80,
    10,
    100,
    8,
    ChromaSampling::Cs420,
    150,
    200,
    0,
    low_lantency,
    0,
    no_scene_detection,
    10,
  );
  let limit = 41;

  for _ in 0..limit {
    let input = ctx.new_frame();
    let _ = ctx.send_frame(input);
  }

  ctx.flush();

  let mut count = 0;

  'out: for _ in 0..limit {
    loop {
      match ctx.receive_packet() {
        Ok(_) => {
          eprintln!("Packet Received {}/{}", count, limit);
          count += 1;
        }
        Err(EncoderStatus::EnoughData) => {
          eprintln!("{:?}", EncoderStatus::EnoughData);

          break 'out;
        }
        Err(e) => {
          eprintln!("{:?}", e);
          break;
        }
      }
    }
  }

  assert_eq!(limit, count);
}

fn send_frames<T: Pixel>(
  ctx: &mut Context<T>, limit: u64, scene_change_at: u64,
) {
  for i in 0..limit {
    if i < scene_change_at {
      send_test_frame(ctx, T::min_value());
    } else {
      send_test_frame(ctx, T::max_value());
    }
  }
}

fn send_test_frame<T: Pixel>(ctx: &mut Context<T>, content_value: T) {
  let mut input = ctx.new_frame();
  fill_frame_const(&mut input, content_value);
  let _ = ctx.send_frame(Arc::new(input));
}

fn get_frame_invariants<T: Pixel>(
  ctx: Context<T>,
) -> impl Iterator<Item = FrameInvariants<T>> {
  ctx.inner.frame_data.into_iter().map(|(_, v)| v.fi)
}

#[interpolate_test(0, 0)]
#[interpolate_test(1, 1)]
fn output_frameno_low_latency_minus(missing: u64) {
  // Test output_frameno configurations when there are <missing> less frames
  // than the perfect subgop size, in no-reorder mode.

  let mut ctx = setup_encoder::<u8>(
    64,
    80,
    10,
    100,
    8,
    ChromaSampling::Cs420,
    5,
    5,
    0,
    true,
    0,
    true,
    10,
  );
  let limit = 10 - missing;
  send_frames(&mut ctx, limit, 0);
  ctx.flush();

  // data[output_frameno] = (input_frameno, !invalid)
  let data = get_frame_invariants(ctx)
    .map(|fi| (fi.input_frameno, !fi.invalid))
    .collect::<Vec<_>>();

  assert_eq!(
    &data[..],
    match missing {
      0 => {
        &[
          (0, true), // I-frame
          (1, true), // P-frame
          (2, true), // P-frame
          (3, true), // P-frame
          (4, true), // P-frame
          (5, true), // I-frame
          (6, true), // P-frame
          (7, true), // P-frame
          (8, true), // P-frame
          (9, true), // P-frame
        ][..]
      }
      1 => {
        &[
          (0, true), // I-frame
          (1, true), // P-frame
          (2, true), // P-frame
          (3, true), // P-frame
          (4, true), // P-frame
          (5, true), // I-frame
          (6, true), // P-frame
          (7, true), // P-frame
          (8, true), // P-frame
        ][..]
      }
      _ => unreachable!(),
    }
  );
}

#[test]
fn switch_frame_interval() {
  // Test output_frameno configurations when there are <missing> less frames
  // than the perfect subgop size, in no-reorder mode.

  let mut ctx = setup_encoder::<u8>(
    64,
    80,
    10,
    100,
    8,
    ChromaSampling::Cs420,
    5,
    5,
    0,
    true,
    2,
    true,
    10,
  );
  let limit = 10;
  send_frames(&mut ctx, limit, 0);
  ctx.flush();

  let data = get_frame_invariants(ctx)
    .map(|fi| (fi.input_frameno, fi.frame_type))
    .collect::<Vec<_>>();

  assert_eq!(
    &data[..],
    &[
      (0, FrameType::KEY),
      (1, FrameType::INTER),
      (2, FrameType::SWITCH),
      (3, FrameType::INTER),
      (4, FrameType::SWITCH),
      (5, FrameType::KEY),
      (6, FrameType::INTER),
      (7, FrameType::SWITCH),
      (8, FrameType::INTER),
      (9, FrameType::SWITCH),
    ][..]
  );
}

#[test]
fn minimum_frame_delay() {
  let mut ctx = setup_encoder::<u8>(
    64,
    80,
    10,
    100,
    8,
    ChromaSampling::Cs420,
    5,
    5,
    0,
    true,
    0,
    true,
    1,
  );

  let limit = 4; // 4 frames in for 1 frame out (delay of 3 frames)
  send_frames(&mut ctx, limit, 0);

  let data = get_frame_invariants(ctx)
    .map(|fi| (fi.input_frameno, fi.frame_type))
    .collect::<Vec<_>>();

  assert_eq!(&data[..], &[(0, FrameType::KEY),][..]);
}

#[interpolate_test(0, 0)]
#[interpolate_test(1, 1)]
fn pyramid_level_low_latency_minus(missing: u64) {
  // Test pyramid_level configurations when there are <missing> less frames
  // than the perfect subgop size, in no-reorder mode.

  let mut ctx = setup_encoder::<u8>(
    64,
    80,
    10,
    100,
    8,
    ChromaSampling::Cs420,
    5,
    5,
    0,
    true,
    0,
    true,
    10,
  );
  let limit = 10 - missing;
  send_frames(&mut ctx, limit, 0);
  ctx.flush();

  // data[output_frameno] = pyramid_level
  let data =
    get_frame_invariants(ctx).map(|fi| fi.pyramid_level).collect::<Vec<_>>();

  assert!(data.into_iter().all(|pyramid_level| pyramid_level == 0));
}

#[interpolate_test(0, 0)]
#[interpolate_test(1, 1)]
#[interpolate_test(2, 2)]
#[interpolate_test(3, 3)]
#[interpolate_test(4, 4)]
fn output_frameno_reorder_minus(missing: u64) {
  // Test output_frameno configurations when there are <missing> less frames
  // than the perfect subgop size.

  let mut ctx = setup_encoder::<u8>(
    64,
    80,
    10,
    100,
    8,
    ChromaSampling::Cs420,
    5,
    5,
    0,
    false,
    0,
    true,
    10,
  );

  // TODO: when we support more pyramid depths, this test will need tweaks.
  assert_eq!(ctx.inner.inter_cfg.pyramid_depth, 2);

  let limit = 10 - missing;
  send_frames(&mut ctx, limit, 0);
  ctx.flush();

  // data[output_frameno] = (input_frameno, !invalid)
  let data = get_frame_invariants(ctx)
    .map(|fi| (fi.input_frameno, !fi.invalid))
    .collect::<Vec<_>>();

  assert_eq!(
    &data[..],
    match missing {
      0 => {
        &[
          (0, true), // I-frame
          (4, true), // P-frame
          (2, true), // B0-frame
          (1, true), // B1-frame (first)
          (2, true), // B0-frame (show existing)
          (3, true), // B1-frame (second)
          (4, true), // P-frame (show existing)
          (5, true), // I-frame
          (9, true), // P-frame
          (7, true), // B0-frame
          (6, true), // B1-frame (first)
          (7, true), // B0-frame (show existing)
          (8, true), // B1-frame (second)
          (9, true), // P-frame (show existing)
        ][..]
      }
      1 => {
        &[
          (0, true),  // I-frame
          (4, true),  // P-frame
          (2, true),  // B0-frame
          (1, true),  // B1-frame (first)
          (2, true),  // B0-frame (show existing)
          (3, true),  // B1-frame (second)
          (4, true),  // P-frame (show existing)
          (5, true),  // I-frame
          (5, false), // Last frame (missing)
          (7, true),  // B0-frame
          (6, true),  // B1-frame (first)
          (7, true),  // B0-frame (show existing)
          (8, true),  // B1-frame (second)
          (8, false), // Last frame (missing)
        ][..]
      }
      2 => {
        &[
          (0, true),  // I-frame
          (4, true),  // P-frame
          (2, true),  // B0-frame
          (1, true),  // B1-frame (first)
          (2, true),  // B0-frame (show existing)
          (3, true),  // B1-frame (second)
          (4, true),  // P-frame (show existing)
          (5, true),  // I-frame
          (5, false), // Last frame (missing)
          (7, true),  // B0-frame
          (6, true),  // B1-frame (first)
          (7, true),  // B0-frame (show existing)
          (7, false), // 2nd last (missing)
          (7, false), // Last frame (missing)
        ][..]
      }
      3 => {
        &[
          (0, true),  // I-frame
          (4, true),  // P-frame
          (2, true),  // B0-frame
          (1, true),  // B1-frame (first)
          (2, true),  // B0-frame (show existing)
          (3, true),  // B1-frame (second)
          (4, true),  // P-frame (show existing)
          (5, true),  // I-frame
          (5, false), // Last frame (missing)
          (5, false), // 3rd last (missing)
          (6, true),  // B1-frame (first)
          (6, false), // 3rd last (missing)
          (6, false), // 2nd last (missing)
          (6, false), // Last frame (missing)
        ][..]
      }
      4 => {
        &[
          (0, true), // I-frame
          (4, true), // P-frame
          (2, true), // B0-frame
          (1, true), // B1-frame (first)
          (2, true), // B0-frame (show existing)
          (3, true), // B1-frame (second)
          (4, true), // P-frame (show existing)
          (5, true), // I-frame
        ][..]
      }
      _ => unreachable!(),
    }
  );
}

#[interpolate_test(0, 0)]
#[interpolate_test(1, 1)]
#[interpolate_test(2, 2)]
#[interpolate_test(3, 3)]
#[interpolate_test(4, 4)]
fn pyramid_level_reorder_minus(missing: u64) {
  // Test pyramid_level configurations when there are <missing> less frames
  // than the perfect subgop size.

  let mut ctx = setup_encoder::<u8>(
    64,
    80,
    10,
    100,
    8,
    ChromaSampling::Cs420,
    5,
    5,
    0,
    false,
    0,
    true,
    10,
  );

  // TODO: when we support more pyramid depths, this test will need tweaks.
  assert_eq!(ctx.inner.inter_cfg.pyramid_depth, 2);

  let limit = 10 - missing;
  send_frames(&mut ctx, limit, 0);
  ctx.flush();

  // data[output_frameno] = pyramid_level
  let data =
    get_frame_invariants(ctx).map(|fi| fi.pyramid_level).collect::<Vec<_>>();

  assert_eq!(
    &data[..],
    match missing {
      0 => {
        &[
          0, // I-frame
          0, // P-frame
          1, // B0-frame
          2, // B1-frame (first)
          1, // B0-frame (show existing)
          2, // B1-frame (second)
          0, // P-frame (show existing)
          0, // I-frame
          0, // P-frame
          1, // B0-frame
          2, // B1-frame (first)
          1, // B0-frame (show existing)
          2, // B1-frame (second)
          0, // P-frame (show existing)
        ][..]
      }
      1 => {
        &[
          0, // I-frame
          0, // P-frame
          1, // B0-frame
          2, // B1-frame (first)
          1, // B0-frame (show existing)
          2, // B1-frame (second)
          0, // P-frame (show existing)
          0, // I-frame
          0, // Last frame (missing)
          1, // B0-frame
          2, // B1-frame (first)
          1, // B0-frame (show existing)
          2, // B1-frame (second)
          2, // Last frame (missing)
        ][..]
      }
      2 => {
        &[
          0, // I-frame
          0, // P-frame
          1, // B0-frame
          2, // B1-frame (first)
          1, // B0-frame (show existing)
          2, // B1-frame (second)
          0, // P-frame (show existing)
          0, // I-frame
          0, // Last frame (missing)
          1, // B0-frame
          2, // B1-frame (first)
          1, // B0-frame (show existing)
          1, // 2nd last (missing)
          1, // Last frame (missing)
        ][..]
      }
      3 => {
        &[
          0, // I-frame
          0, // P-frame
          1, // B0-frame
          2, // B1-frame (first)
          1, // B0-frame (show existing)
          2, // B1-frame (second)
          0, // P-frame (show existing)
          0, // I-frame
          0, // Last frame (missing)
          0, // 3rd last (missing)
          2, // B1-frame (first)
          2, // 3rd last (missing)
          2, // 2nd last (missing)
          2, // Last frame (missing)
        ][..]
      }
      4 => {
        &[
          0, // I-frame
          0, // P-frame
          1, // B0-frame
          2, // B1-frame (first)
          1, // B0-frame (show existing)
          2, // B1-frame (second)
          0, // P-frame (show existing)
          0, // I-frame
        ][..]
      }
      _ => unreachable!(),
    }
  );
}

#[interpolate_test(0, 0)]
#[interpolate_test(1, 1)]
#[interpolate_test(2, 2)]
#[interpolate_test(3, 3)]
#[interpolate_test(4, 4)]
fn output_frameno_reorder_scene_change_at(scene_change_at: u64) {
  // Test output_frameno configurations when there's a scene change at the
  // <scene_change_at>th frame.

  let mut ctx = setup_encoder::<u8>(
    64,
    80,
    10,
    100,
    8,
    ChromaSampling::Cs420,
    0,
    5,
    0,
    false,
    0,
    false,
    10,
  );

  // TODO: when we support more pyramid depths, this test will need tweaks.
  assert_eq!(ctx.inner.inter_cfg.pyramid_depth, 2);

  let limit = 10;
  send_frames(&mut ctx, limit, scene_change_at);
  ctx.flush();

  // data[output_frameno] = (input_frameno, !invalid)
  let data = get_frame_invariants(ctx)
    .map(|fi| (fi.input_frameno, !fi.invalid))
    .filter(|&(frameno, _)| frameno < 5)
    .collect::<Vec<_>>();

  assert_eq!(
    &data[..],
    match scene_change_at {
      0 => {
        &[
          (0, true), // I-frame
          (4, true), // P-frame
          (2, true), // B0-frame
          (1, true), // B1-frame (first)
          (2, true), // B0-frame (show existing)
          (3, true), // B1-frame (second)
          (4, true), // P-frame (show existing)
        ][..]
      }
      1 => {
        &[
          (0, true), // I-frame
          (1, true), // I-frame
          (3, true), // B0-frame
          (2, true), // B1-frame (first)
          (3, true), // B0-frame (show existing)
          (4, true), // B1-frame (second)
        ][..]
      }
      2 => {
        &[
          (0, true),  // I-frame
          (0, false), // Missing
          (0, false), // Missing
          (1, true),  // B1-frame (first)
          (1, false), // Missing
          (1, false), // Missing
          (1, false), // Missing
          (2, true),  // I-frame
          (4, true),  // B0-frame
          (3, true),  // B1-frame (first)
          (4, true),  // B0-frame (show existing)
        ][..]
      }
      3 => {
        &[
          (0, true),  // I-frame
          (0, false), // Missing
          (2, true),  // B0-frame
          (1, true),  // B1-frame (first)
          (2, true),  // B0-frame (show existing)
          (2, false), // Missing
          (2, false), // Missing
          (3, true),  // I-frame
          (4, true),  // B1-frame (first)
        ][..]
      }
      4 => {
        &[
          (0, true),  // I-frame
          (0, false), // Missing
          (2, true),  // B0-frame
          (1, true),  // B1-frame (first)
          (2, true),  // B0-frame (show existing)
          (3, true),  // B1-frame (second)
          (3, false), // Missing
          (4, true),  // I-frame
        ][..]
      }
      _ => unreachable!(),
    }
  );
}

#[interpolate_test(0, 0)]
#[interpolate_test(1, 1)]
#[interpolate_test(2, 2)]
#[interpolate_test(3, 3)]
#[interpolate_test(4, 4)]
fn pyramid_level_reorder_scene_change_at(scene_change_at: u64) {
  // Test pyramid_level configurations when there's a scene change at the
  // <scene_change_at>th frame.

  let mut ctx = setup_encoder::<u8>(
    64,
    80,
    10,
    100,
    8,
    ChromaSampling::Cs420,
    0,
    5,
    0,
    false,
    0,
    false,
    10,
  );

  // TODO: when we support more pyramid depths, this test will need tweaks.
  assert_eq!(ctx.inner.inter_cfg.pyramid_depth, 2);

  let limit = 10;
  send_frames(&mut ctx, limit, scene_change_at);
  ctx.flush();

  // data[output_frameno] = pyramid_level
  let data = get_frame_invariants(ctx)
    .filter(|fi| fi.input_frameno < 5)
    .map(|fi| fi.pyramid_level)
    .collect::<Vec<_>>();

  assert_eq!(
    &data[..],
    match scene_change_at {
      0 => {
        &[
          0, // I-frame
          0, // P-frame
          1, // B0-frame
          2, // B1-frame (first)
          1, // B0-frame (show existing)
          2, // B1-frame (second)
          0, // P-frame (show existing)
        ][..]
      }
      1 => {
        &[
          0, // I-frame
          0, // I-frame
          1, // B0-frame
          2, // B1-frame (first)
          1, // B0-frame (show existing)
          2, // B1-frame (second)
        ][..]
      }
      2 => {
        &[
          0, // I-frame
          0, // Missing
          0, // Missing
          2, // B1-frame (first)
          2, // Missing
          2, // Missing
          2, // Missing
          0, // I-frame
          1, // B0-frame
          2, // B1-frame (first)
          1, // B0-frame (show existing)
        ][..]
      }
      3 => {
        &[
          0, // I-frame
          0, // Missing
          1, // B0-frame
          2, // B1-frame (first)
          1, // B0-frame (show existing)
          1, // Missing
          1, // Missing
          0, // I-frame
          2, // B1-frame (first)
        ][..]
      }
      4 => {
        &[
          0, // I-frame
          0, // Missing
          1, // B0-frame
          2, // B1-frame (first)
          1, // B0-frame (show existing)
          2, // B1-frame (second)
          2, // Missing
          0, // I-frame
        ][..]
      }
      _ => unreachable!(),
    }
  );
}

#[interpolate_test(0, 0)]
#[interpolate_test(1, 1)]
#[interpolate_test(2, 2)]
#[interpolate_test(3, 3)]
#[interpolate_test(4, 4)]
fn output_frameno_incremental_reorder_minus(missing: u64) {
  // Test output_frameno configurations when there are <missing> less frames
  // than the perfect subgop size, computing the lookahead data incrementally.

  let mut ctx = setup_encoder::<u8>(
    64,
    80,
    10,
    100,
    8,
    ChromaSampling::Cs420,
    5,
    5,
    0,
    false,
    0,
    true,
    10,
  );

  // TODO: when we support more pyramid depths, this test will need tweaks.
  assert_eq!(ctx.inner.inter_cfg.pyramid_depth, 2);

  let limit = 10 - missing;
  for _ in 0..limit {
    send_frames(&mut ctx, 1, 0);
  }
  ctx.flush();

  // data[output_frameno] = (input_frameno, !invalid)
  let data = get_frame_invariants(ctx)
    .map(|fi| (fi.input_frameno, !fi.invalid))
    .collect::<Vec<_>>();

  assert_eq!(
    &data[..],
    match missing {
      0 => {
        &[
          (0, true), // I-frame
          (4, true), // P-frame
          (2, true), // B0-frame
          (1, true), // B1-frame (first)
          (2, true), // B0-frame (show existing)
          (3, true), // B1-frame (second)
          (4, true), // P-frame (show existing)
          (5, true), // I-frame
          (9, true), // P-frame
          (7, true), // B0-frame
          (6, true), // B1-frame (first)
          (7, true), // B0-frame (show existing)
          (8, true), // B1-frame (second)
          (9, true), // P-frame (show existing)
        ][..]
      }
      1 => {
        &[
          (0, true),  // I-frame
          (4, true),  // P-frame
          (2, true),  // B0-frame
          (1, true),  // B1-frame (first)
          (2, true),  // B0-frame (show existing)
          (3, true),  // B1-frame (second)
          (4, true),  // P-frame (show existing)
          (5, true),  // I-frame
          (5, false), // Last frame (missing)
          (7, true),  // B0-frame
          (6, true),  // B1-frame (first)
          (7, true),  // B0-frame (show existing)
          (8, true),  // B1-frame (second)
          (8, false), // Last frame (missing)
        ][..]
      }
      2 => {
        &[
          (0, true),  // I-frame
          (4, true),  // P-frame
          (2, true),  // B0-frame
          (1, true),  // B1-frame (first)
          (2, true),  // B0-frame (show existing)
          (3, true),  // B1-frame (second)
          (4, true),  // P-frame (show existing)
          (5, true),  // I-frame
          (5, false), // Last frame (missing)
          (7, true),  // B0-frame
          (6, true),  // B1-frame (first)
          (7, true),  // B0-frame (show existing)
          (7, false), // 2nd last (missing)
          (7, false), // Last frame (missing)
        ][..]
      }
      3 => {
        &[
          (0, true),  // I-frame
          (4, true),  // P-frame
          (2, true),  // B0-frame
          (1, true),  // B1-frame (first)
          (2, true),  // B0-frame (show existing)
          (3, true),  // B1-frame (second)
          (4, true),  // P-frame (show existing)
          (5, true),  // I-frame
          (5, false), // Last frame (missing)
          (5, false), // 3rd last (missing)
          (6, true),  // B1-frame (first)
          (6, false), // 3rd last (missing)
          (6, false), // 2nd last (missing)
          (6, false), // Last frame (missing)
        ][..]
      }
      4 => {
        &[
          (0, true), // I-frame
          (4, true), // P-frame
          (2, true), // B0-frame
          (1, true), // B1-frame (first)
          (2, true), // B0-frame (show existing)
          (3, true), // B1-frame (second)
          (4, true), // P-frame (show existing)
          (5, true), // I-frame
        ][..]
      }
      _ => unreachable!(),
    }
  );
}

#[interpolate_test(0, 0)]
#[interpolate_test(1, 1)]
#[interpolate_test(2, 2)]
#[interpolate_test(3, 3)]
#[interpolate_test(4, 4)]
fn output_frameno_incremental_reorder_scene_change_at(scene_change_at: u64) {
  // Test output_frameno configurations when there's a scene change at the
  // <scene_change_at>th frame, computing the lookahead data incrementally.

  let mut ctx = setup_encoder::<u8>(
    64,
    80,
    10,
    100,
    8,
    ChromaSampling::Cs420,
    0,
    5,
    0,
    false,
    0,
    false,
    10,
  );

  // TODO: when we support more pyramid depths, this test will need tweaks.
  assert_eq!(ctx.inner.inter_cfg.pyramid_depth, 2);

  let limit = 10;
  for i in 0..limit {
    send_frames(&mut ctx, 1, scene_change_at.saturating_sub(i));
  }
  ctx.flush();

  // data[output_frameno] = (input_frameno, !invalid)
  let data = get_frame_invariants(ctx)
    .map(|fi| (fi.input_frameno, !fi.invalid))
    .filter(|&(frameno, _)| frameno < 5)
    .collect::<Vec<_>>();

  assert_eq!(
    &data[..],
    match scene_change_at {
      0 => {
        &[
          (0, true), // I-frame
          (4, true), // P-frame
          (2, true), // B0-frame
          (1, true), // B1-frame (first)
          (2, true), // B0-frame (show existing)
          (3, true), // B1-frame (second)
          (4, true), // P-frame (show existing)
        ][..]
      }
      1 => {
        &[
          (0, true), // I-frame
          (1, true), // I-frame
          (3, true), // B0-frame
          (2, true), // B1-frame (first)
          (3, true), // B0-frame (show existing)
          (4, true), // B1-frame (second)
        ][..]
      }
      2 => {
        &[
          (0, true),  // I-frame
          (0, false), // Missing
          (0, false), // Missing
          (1, true),  // B1-frame (first)
          (1, false), // Missing
          (1, false), // Missing
          (1, false), // Missing
          (2, true),  // I-frame
          (4, true),  // B0-frame
          (3, true),  // B1-frame (first)
          (4, true),  // B0-frame (show existing)
        ][..]
      }
      3 => {
        &[
          (0, true),  // I-frame
          (0, false), // Missing
          (2, true),  // B0-frame
          (1, true),  // B1-frame (first)
          (2, true),  // B0-frame (show existing)
          (2, false), // Missing
          (2, false), // Missing
          (3, true),  // I-frame
          (4, true),  // B1-frame (first)
        ][..]
      }
      4 => {
        &[
          (0, true),  // I-frame
          (0, false), // Missing
          (2, true),  // B0-frame
          (1, true),  // B1-frame (first)
          (2, true),  // B0-frame (show existing)
          (3, true),  // B1-frame (second)
          (3, false), // Missing
          (4, true),  // I-frame
        ][..]
      }
      _ => unreachable!(),
    }
  );
}

fn send_frame_kf<T: Pixel>(ctx: &mut Context<T>, keyframe: bool) {
  let input = ctx.new_frame();

  let frame_type_override =
    if keyframe { FrameTypeOverride::Key } else { FrameTypeOverride::No };

  let opaque = Some(Box::new(keyframe) as Box<dyn std::any::Any + Send>);

  let fp = FrameParameters { frame_type_override, opaque };

  let _ = ctx.send_frame((input, fp));
}

#[test]
fn test_opaque_delivery() {
  let mut ctx = setup_encoder::<u8>(
    64,
    80,
    10,
    100,
    8,
    ChromaSampling::Cs420,
    0,
    5,
    0,
    false,
    0,
    false,
    10,
  );

  let kf_at = 3;

  let limit = 5;
  for i in 0..limit {
    send_frame_kf(&mut ctx, kf_at == i);
  }
  ctx.flush();

  while let Ok(pkt) = ctx.receive_packet() {
    let Packet { opaque, input_frameno, .. } = pkt;
    if let Some(opaque) = opaque {
      let kf = opaque.downcast::<bool>().unwrap();
      assert_eq!(kf, Box::new(input_frameno == kf_at));
    }
  }
}

#[interpolate_test(0, 0)]
#[interpolate_test(1, 1)]
#[interpolate_test(2, 2)]
#[interpolate_test(3, 3)]
#[interpolate_test(4, 4)]
fn output_frameno_incremental_reorder_keyframe_at(kf_at: u64) {
  // Test output_frameno configurations when there's a forced keyframe at the
  // <kf_at>th frame, computing the lookahead data incrementally.

  let mut ctx = setup_encoder::<u8>(
    64,
    80,
    10,
    100,
    8,
    ChromaSampling::Cs420,
    0,
    5,
    0,
    false,
    0,
    false,
    10,
  );

  // TODO: when we support more pyramid depths, this test will need tweaks.
  assert_eq!(ctx.inner.inter_cfg.pyramid_depth, 2);

  let limit = 5;
  for i in 0..limit {
    send_frame_kf(&mut ctx, kf_at == i);
  }
  ctx.flush();

  // data[output_frameno] = (input_frameno, !invalid)
  let data = get_frame_invariants(ctx)
    .map(|fi| (fi.input_frameno, !fi.invalid))
    .collect::<Vec<_>>();

  assert_eq!(
    &data[..],
    match kf_at {
      0 => {
        &[
          (0, true), // I-frame
          (4, true), // P-frame
          (2, true), // B0-frame
          (1, true), // B1-frame (first)
          (2, true), // B0-frame (show existing)
          (3, true), // B1-frame (second)
          (4, true), // P-frame (show existing)
        ][..]
      }
      1 => {
        &[
          (0, true),  // I-frame
          (1, true),  // I-frame
          (1, false), // Missing
          (3, true),  // B0-frame
          (2, true),  // B1-frame (first)
          (3, true),  // B0-frame (show existing)
          (4, true),  // B1-frame (second)
          (4, false), // Missing
        ][..]
      }
      2 => {
        &[
          (0, true),  // I-frame
          (0, false), // Missing
          (0, false), // Missing
          (1, true),  // B1-frame (first)
          (1, false), // Missing
          (1, false), // Missing
          (1, false), // Missing
          (2, true),  // I-frame
          (2, false), // Missing
          (4, true),  // B0-frame
          (3, true),  // B1-frame (first)
          (4, true),  // B0-frame (show existing)
          (4, false), // Missing
          (4, false), // Missing
        ][..]
      }
      3 => {
        &[
          (0, true),  // I-frame
          (0, false), // Missing
          (2, true),  // B0-frame
          (1, true),  // B1-frame (first)
          (2, true),  // B0-frame (show existing)
          (2, false), // Missing
          (2, false), // Missing
          (3, true),  // I-frame
          (3, false), // Missing
          (3, false), // Missing
          (4, true),  // B1-frame (first)
          (4, false), // Missing
          (4, false), // Missing
          (4, false), // Missing
        ][..]
      }
      4 => {
        &[
          (0, true),  // I-frame
          (0, false), // Missing
          (2, true),  // B0-frame
          (1, true),  // B1-frame (first)
          (2, true),  // B0-frame (show existing)
          (3, true),  // B1-frame (second)
          (3, false), // Missing
          (4, true),  // I-frame
        ][..]
      }
      _ => unreachable!(),
    }
  );
}

#[interpolate_test(1, 1)]
#[interpolate_test(2, 2)]
#[interpolate_test(3, 3)]
fn output_frameno_no_scene_change_at_short_flash(flash_at: u64) {
  // Test output_frameno configurations when there's a single-frame flash at the
  // <flash_at>th frame.
  let mut ctx = setup_encoder::<u8>(
    64,
    80,
    10,
    100,
    8,
    ChromaSampling::Cs420,
    0,
    5,
    0,
    false,
    0,
    false,
    10,
  );

  // TODO: when we support more pyramid depths, this test will need tweaks.
  assert_eq!(ctx.inner.inter_cfg.pyramid_depth, 2);

  let limit = 5;
  for i in 0..limit {
    if i == flash_at {
      send_test_frame(&mut ctx, u8::min_value());
    } else {
      send_test_frame(&mut ctx, u8::max_value());
    }
  }
  ctx.flush();

  // data[output_frameno] = (input_frameno, !invalid)
  let data = get_frame_invariants(ctx)
    .map(|fi| (fi.input_frameno, !fi.invalid))
    .collect::<Vec<_>>();

  assert_eq!(
    &data[..],
    &[
      (0, true), // I-frame
      (4, true), // P-frame
      (2, true), // B0-frame
      (1, true), // B1-frame (first)
      (2, true), // B0-frame (show existing)
      (3, true), // B1-frame (second)
      (4, true), // P-frame (show existing)
    ]
  );
}

#[test]
fn output_frameno_no_scene_change_at_max_len_flash() {
  // Test output_frameno configurations when there's a multi-frame flash
  // with length equal to the max flash length

  let mut ctx = setup_encoder::<u8>(
    64,
    80,
    10,
    100,
    8,
    ChromaSampling::Cs420,
    0,
    10,
    0,
    false,
    0,
    false,
    10,
  );

  // TODO: when we support more pyramid depths, this test will need tweaks.
  assert_eq!(ctx.inner.inter_cfg.pyramid_depth, 2);
  assert_eq!(ctx.inner.inter_cfg.group_input_len, 4);

  send_test_frame(&mut ctx, u8::min_value());
  send_test_frame(&mut ctx, u8::min_value());
  send_test_frame(&mut ctx, u8::max_value());
  send_test_frame(&mut ctx, u8::max_value());
  send_test_frame(&mut ctx, u8::max_value());
  send_test_frame(&mut ctx, u8::max_value());
  send_test_frame(&mut ctx, u8::min_value());
  send_test_frame(&mut ctx, u8::min_value());
  ctx.flush();

  let data = get_frame_invariants(ctx)
    .map(|fi| (fi.input_frameno, !fi.invalid))
    .collect::<Vec<_>>();

  assert_eq!(
    &data[..],
    &[
      (0, true),  // I-frame
      (4, true),  // P-frame
      (2, true),  // B0-frame
      (1, true),  // B1-frame (first)
      (2, true),  // B0-frame (show existing)
      (3, true),  // B1-frame (second)
      (4, true),  // P-frame (show existing)
      (4, false), // invalid
      (6, true),  // B0-frame
      (5, true),  // B1-frame (first)
      (6, true),  // B0-frame (show existing)
      (7, true),  // B1-frame (second)
      (7, false), // invalid
    ]
  );
}

#[test]
fn output_frameno_scene_change_past_max_len_flash() {
  // Test output_frameno configurations when there's a multi-frame flash
  // with length greater than the max flash length

  let mut ctx = setup_encoder::<u8>(
    64,
    80,
    10,
    100,
    8,
    ChromaSampling::Cs420,
    0,
    10,
    0,
    false,
    0,
    false,
    10,
  );

  // TODO: when we support more pyramid depths, this test will need tweaks.
  assert_eq!(ctx.inner.inter_cfg.pyramid_depth, 2);
  assert_eq!(ctx.inner.inter_cfg.group_input_len, 4);

  send_test_frame(&mut ctx, u8::min_value());
  send_test_frame(&mut ctx, u8::min_value());
  send_test_frame(&mut ctx, u8::max_value());
  send_test_frame(&mut ctx, u8::max_value());
  send_test_frame(&mut ctx, u8::max_value());
  send_test_frame(&mut ctx, u8::max_value());
  send_test_frame(&mut ctx, u8::max_value());
  send_test_frame(&mut ctx, u8::min_value());
  send_test_frame(&mut ctx, u8::min_value());
  send_test_frame(&mut ctx, u8::min_value());
  send_test_frame(&mut ctx, u8::min_value());
  send_test_frame(&mut ctx, u8::min_value());
  ctx.flush();

  let data = get_frame_invariants(ctx)
    .map(|fi| (fi.input_frameno, !fi.invalid))
    .filter(|&(frameno, _)| frameno <= 7)
    .collect::<Vec<_>>();

  assert_eq!(
    &data[..],
    &[
      (0, true),  // I-frame
      (0, false), // invalid
      (0, false), // invalid
      (1, true),  // B1-frame (first)
      (1, false), // invalid
      (1, false), // invalid
      (1, false), // invalid
      (2, true),  // I-frame
      (6, true),  // P-frame
      (4, true),  // B0-frame
      (3, true),  // B1-frame (first)
      (4, true),  // B0-frame (show existing)
      (5, true),  // B1-frame (second)
      (6, true),  // P-frame (show existing)
      (7, true),  // I-frame
    ]
  );
}

#[test]
fn output_frameno_no_scene_change_at_multiple_flashes() {
  // Test output_frameno configurations when there are multiple consecutive flashes

  let mut ctx = setup_encoder::<u8>(
    64,
    80,
    10,
    100,
    8,
    ChromaSampling::Cs420,
    0,
    10,
    0,
    false,
    0,
    false,
    10,
  );

  // TODO: when we support more pyramid depths, this test will need tweaks.
  assert_eq!(ctx.inner.inter_cfg.pyramid_depth, 2);
  assert_eq!(ctx.inner.inter_cfg.group_input_len, 4);

  send_test_frame(&mut ctx, u8::min_value());
  send_test_frame(&mut ctx, u8::min_value());
  send_test_frame(&mut ctx, 40);
  send_test_frame(&mut ctx, 100);
  send_test_frame(&mut ctx, 160);
  send_test_frame(&mut ctx, 240);
  send_test_frame(&mut ctx, 240);
  send_test_frame(&mut ctx, 240);
  send_test_frame(&mut ctx, 240);
  send_test_frame(&mut ctx, 240);
  send_test_frame(&mut ctx, 240);
  ctx.flush();

  let data = get_frame_invariants(ctx)
    .map(|fi| (fi.input_frameno, !fi.invalid))
    .filter(|&(frameno, _)| frameno <= 7)
    .collect::<Vec<_>>();

  assert_eq!(
    &data[..],
    &[
      (0, true), // I-frame
      (4, true), // P-frame
      (2, true), // B0-frame
      (1, true), // B1-frame (first)
      (2, true), // B0-frame (show existing)
      (3, true), // B1-frame (second)
      (4, true), // P-frame (show existing),
      (5, true), // P-frame
      (7, true), // B0-frame
      (6, true), // B1-frame (first)
      (7, true), // B0-frame (show existing)
    ]
  );
}

#[derive(Clone, Copy)]
struct LookaheadTestExpectations {
  pre_receive_frame_q_lens: [usize; 60],
  pre_receive_fi_lens: [usize; 60],
  post_receive_frame_q_lens: [usize; 60],
  post_receive_fi_lens: [usize; 60],
}

#[test]
fn lookahead_size_properly_bounded_8() {
  const LOOKAHEAD_SIZE: usize = 8;
  const EXPECTATIONS: LookaheadTestExpectations = LookaheadTestExpectations {
    pre_receive_frame_q_lens: [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 19, 20, 20, 21, 19, 20, 20, 21, 19, 20, 20, 21, 19, 20, 20, 21, 19,
      20, 20, 21, 19, 20, 20, 21, 19, 20, 20, 21, 19, 20, 20, 21, 19, 20, 20,
      21, 19, 20, 20,
    ],
    pre_receive_fi_lens: [
      0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 7, 7, 7, 7, 13, 13, 13, 13, 19, 19, 19,
      14, 20, 19, 19, 14, 20, 19, 19, 14, 20, 19, 19, 14, 20, 19, 19, 14, 20,
      19, 19, 14, 20, 19, 19, 14, 20, 19, 19, 14, 20, 19, 19, 14, 20, 19, 19,
      14, 20, 19,
    ],
    post_receive_frame_q_lens: [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      18, 19, 19, 20, 18, 19, 19, 20, 18, 19, 19, 20, 18, 19, 19, 20, 18, 19,
      19, 20, 18, 19, 19, 20, 18, 19, 19, 20, 18, 19, 19, 20, 18, 19, 19, 20,
      18, 19, 19, 20,
    ],
    post_receive_fi_lens: [
      0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 7, 7, 7, 7, 13, 13, 13, 13, 19, 19, 14,
      14, 19, 19, 14, 14, 19, 19, 14, 14, 19, 19, 14, 14, 19, 19, 14, 14, 19,
      19, 14, 14, 19, 19, 14, 14, 19, 19, 14, 14, 19, 19, 14, 14, 19, 19, 14,
      14, 19, 19,
    ],
  };
  lookahead_size_properly_bounded(LOOKAHEAD_SIZE, false, &EXPECTATIONS);
}

#[test]
fn lookahead_size_properly_bounded_10() {
  const LOOKAHEAD_SIZE: usize = 10;
  const EXPECTATIONS: LookaheadTestExpectations = LookaheadTestExpectations {
    pre_receive_frame_q_lens: [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 20, 21, 22, 23, 20, 21, 22, 23, 20, 21, 22, 23, 20, 21, 22,
      23, 20, 21, 22, 23, 20, 21, 22, 23, 20, 21, 22, 23, 20, 21, 22, 23, 20,
      21, 22, 23, 20,
    ],
    pre_receive_fi_lens: [
      0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 7, 7, 7, 7, 13, 13, 13, 13, 19, 19, 19,
      19, 25, 19, 19, 19, 25, 19, 19, 19, 25, 19, 19, 19, 25, 19, 19, 19, 25,
      19, 19, 19, 25, 19, 19, 19, 25, 19, 19, 19, 25, 19, 19, 19, 25, 19, 19,
      19, 25, 19,
    ],
    post_receive_frame_q_lens: [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 19, 20, 21, 22, 19, 20, 21, 22, 19, 20, 21, 22, 19, 20, 21, 22,
      19, 20, 21, 22, 19, 20, 21, 22, 19, 20, 21, 22, 19, 20, 21, 22, 19, 20,
      21, 22, 19, 20,
    ],
    post_receive_fi_lens: [
      0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 7, 7, 7, 7, 13, 13, 13, 13, 19, 19, 19,
      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
      19, 19, 19,
    ],
  };
  lookahead_size_properly_bounded(LOOKAHEAD_SIZE, false, &EXPECTATIONS);
}

#[test]
fn lookahead_size_properly_bounded_16() {
  const LOOKAHEAD_SIZE: usize = 16;
  const EXPECTATIONS: LookaheadTestExpectations = LookaheadTestExpectations {
    pre_receive_frame_q_lens: [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26, 27, 28, 29, 27, 28, 28, 29, 27, 28, 28, 29, 27,
      28, 28, 29, 27, 28, 28, 29, 27, 28, 28, 29, 27, 28, 28, 29, 27, 28, 28,
      29, 27, 28, 28,
    ],
    pre_receive_fi_lens: [
      0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 7, 7, 7, 7, 13, 13, 13, 13, 19, 19, 19,
      19, 25, 25, 25, 25, 31, 31, 31, 26, 32, 31, 31, 26, 32, 31, 31, 26, 32,
      31, 31, 26, 32, 31, 31, 26, 32, 31, 31, 26, 32, 31, 31, 26, 32, 31, 31,
      26, 32, 31,
    ],
    post_receive_frame_q_lens: [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26, 27, 28, 26, 27, 27, 28, 26, 27, 27, 28, 26, 27,
      27, 28, 26, 27, 27, 28, 26, 27, 27, 28, 26, 27, 27, 28, 26, 27, 27, 28,
      26, 27, 27, 28,
    ],
    post_receive_fi_lens: [
      0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 7, 7, 7, 7, 13, 13, 13, 13, 19, 19, 19,
      19, 25, 25, 25, 25, 31, 31, 26, 26, 31, 31, 26, 26, 31, 31, 26, 26, 31,
      31, 26, 26, 31, 31, 26, 26, 31, 31, 26, 26, 31, 31, 26, 26, 31, 31, 26,
      26, 31, 31,
    ],
  };
  lookahead_size_properly_bounded(LOOKAHEAD_SIZE, false, &EXPECTATIONS);
}

#[test]
fn lookahead_size_properly_bounded_lowlatency_8() {
  const LOOKAHEAD_SIZE: usize = 8;
  const EXPECTATIONS: LookaheadTestExpectations = LookaheadTestExpectations {
    pre_receive_frame_q_lens: [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 13, 13, 13, 13, 13, 13,
      13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
      13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
      13, 13, 13, 13,
    ],
    pre_receive_fi_lens: [
      0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10,
      10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
      10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
      10, 10, 10,
    ],
    post_receive_frame_q_lens: [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12,
      12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
      12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
      12, 12, 12, 12,
    ],
    post_receive_fi_lens: [
      0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
      9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
      9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    ],
  };
  lookahead_size_properly_bounded(LOOKAHEAD_SIZE, true, &EXPECTATIONS);
}

#[test]
fn lookahead_size_properly_bounded_lowlatency_1() {
  const LOOKAHEAD_SIZE: usize = 1;
  const EXPECTATIONS: LookaheadTestExpectations = LookaheadTestExpectations {
    pre_receive_frame_q_lens: [
      1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
      6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
      6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    ],
    pre_receive_fi_lens: [
      0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    ],
    post_receive_frame_q_lens: [
      1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    ],
    post_receive_fi_lens: [
      0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    ],
  };
  lookahead_size_properly_bounded(LOOKAHEAD_SIZE, true, &EXPECTATIONS);
}

fn lookahead_size_properly_bounded(
  rdo_lookahead: usize, low_latency: bool,
  expectations: &LookaheadTestExpectations,
) {
  // Test that lookahead reads in the proper number of frames at once

  let mut ctx = setup_encoder::<u8>(
    64,
    80,
    10,
    100,
    8,
    ChromaSampling::Cs420,
    0,
    100,
    0,
    low_latency,
    0,
    true,
    rdo_lookahead,
  );

  const LIMIT: usize = 60;

  let mut pre_receive_frame_q_lens = [0; LIMIT];
  let mut pre_receive_fi_lens = [0; LIMIT];
  let mut post_receive_frame_q_lens = [0; LIMIT];
  let mut post_receive_fi_lens = [0; LIMIT];

  for i in 0..LIMIT {
    let input = ctx.new_frame();
    let _ = ctx.send_frame(input);
    pre_receive_frame_q_lens[i] = ctx.inner.frame_q.len();
    pre_receive_fi_lens[i] = ctx.inner.frame_data.len();
    while ctx.receive_packet().is_ok() {
      // Receive packets until lookahead consumed, due to pyramids receiving frames in groups
    }
    post_receive_frame_q_lens[i] = ctx.inner.frame_q.len();
    post_receive_fi_lens[i] = ctx.inner.frame_data.len();
  }

  assert_eq!(
    &pre_receive_frame_q_lens[..],
    &expectations.pre_receive_frame_q_lens[..]
  );
  assert_eq!(&pre_receive_fi_lens[..], &expectations.pre_receive_fi_lens[..]);
  assert_eq!(
    &post_receive_frame_q_lens[..],
    &expectations.post_receive_frame_q_lens[..]
  );
  assert_eq!(
    &post_receive_fi_lens[..],
    &expectations.post_receive_fi_lens[..]
  );

  ctx.flush();
  let end = ctx.inner.frame_q.get(&(LIMIT as u64));
  assert!(end.is_some());
  assert!(end.unwrap().is_none());

  loop {
    match ctx.receive_packet() {
      Ok(_) | Err(EncoderStatus::Encoded) => {
        // Receive packets until all frames consumed
      }
      _ => {
        break;
      }
    }
  }
  assert_eq!(ctx.inner.frames_processed, LIMIT as u64);
}

#[test]
fn zero_frames() {
  let config = Config::default();
  let mut ctx: Context<u8> = config.new_context().unwrap();
  ctx.flush();
  assert_eq!(ctx.receive_packet(), Err(EncoderStatus::LimitReached));
}

#[test]
fn tile_cols_overflow() {
  let mut enc = EncoderConfig::default();
  enc.tile_cols = usize::max_value();
  let config = Config::new().with_encoder_config(enc);
  let _: Result<Context<u8>, _> = config.new_context();
}

#[test]
fn max_key_frame_interval_overflow() {
  let mut enc = EncoderConfig::default();
  enc.max_key_frame_interval = i32::max_value() as u64;
  enc.reservoir_frame_delay = None;
  let config = Config::new().with_encoder_config(enc);
  let _: Result<Context<u8>, _> = config.new_context();
}

#[test]
fn target_bitrate_overflow() {
  let mut enc = EncoderConfig::default();
  enc.bitrate = i32::max_value();
  enc.time_base = Rational::new(i64::max_value() as u64, 1);
  let config = Config::new().with_encoder_config(enc);
  let _: Result<Context<u8>, _> = config.new_context();
}

#[test]
fn time_base_den_divide_by_zero() {
  let mut enc = EncoderConfig::default();
  enc.time_base = Rational::new(1, 0);
  let config = Config::new().with_encoder_config(enc);
  let _: Result<Context<u8>, _> = config.new_context();
}

#[test]
fn large_width_assert() {
  let mut enc = EncoderConfig::default();
  enc.width = u32::max_value() as usize;
  let config = Config::new().with_encoder_config(enc);
  let _: Result<Context<u8>, _> = config.new_context();
}

#[test]
fn reservoir_max_overflow() {
  let mut enc = EncoderConfig::default();
  enc.reservoir_frame_delay = Some(i32::max_value());
  enc.bitrate = i32::max_value();
  enc.time_base = Rational::new(i32::max_value() as u64 * 2, 1);
  let config = Config::new().with_encoder_config(enc);
  let _: Result<Context<u8>, _> = config.new_context();
}

#[test]
fn zero_width() {
  let mut enc = EncoderConfig::default();
  enc.width = 0;
  let config = Config::new().with_encoder_config(enc);
  let res: Result<Context<u8>, _> = config.new_context();
  assert!(res.is_err());
}

#[test]
fn rdo_lookahead_frames_overflow() {
  let mut enc = EncoderConfig::default();
  enc.rdo_lookahead_frames = usize::max_value();
  let config = Config::new().with_encoder_config(enc);
  let res: Result<Context<u8>, _> = config.new_context();
  assert!(res.is_err());
}

#[test]
fn log_q_exp_overflow() {
  let enc = EncoderConfig {
    width: 16,
    height: 16,
    bit_depth: 8,
    chroma_sampling: ChromaSampling::Cs420,
    chroma_sample_position: ChromaSamplePosition::Unknown,
    pixel_range: PixelRange::Limited,
    color_description: None,
    mastering_display: None,
    content_light: None,
    enable_timing_info: false,
    still_picture: false,
    error_resilient: false,
    switch_frame_interval: 0,
    time_base: Rational { num: 1, den: 25 },
    min_key_frame_interval: 12,
    max_key_frame_interval: 240,
    reservoir_frame_delay: None,
    low_latency: false,
    quantizer: 100,
    min_quantizer: 64,
    bitrate: 1,
    tune: Tune::Psychovisual,
    tile_cols: 0,
    tile_rows: 0,
    tiles: 0,
    rdo_lookahead_frames: 40,
    speed_settings: SpeedSettings {
      partition_range: PartitionRange::new(
        BlockSize::BLOCK_64X64,
        BlockSize::BLOCK_64X64,
      ),
      multiref: false,
      fast_deblock: true,
      reduced_tx_set: true,
      tx_domain_distortion: true,
      tx_domain_rate: false,
      encode_bottomup: false,
      rdo_tx_decision: false,
      prediction_modes: PredictionModesSetting::Simple,
      include_near_mvs: false,
      no_scene_detection: true,
      fast_scene_detection: false,
      diamond_me: true,
      cdef: true,
      lrf: true,
      use_satd_subpel: false,
      non_square_partition: false,
      ..Default::default()
    },
  };
  let config = Config::new().with_encoder_config(enc).with_threads(1);

  let mut ctx: Context<u8> = config.new_context().unwrap();
  for _ in 0..2 {
    ctx.send_frame(ctx.new_frame()).unwrap();
  }
  ctx.flush();

  ctx.receive_packet().unwrap();
  let _ = ctx.receive_packet();
}

#[test]
fn guess_frame_subtypes_assert() {
  let enc = EncoderConfig {
    width: 16,
    height: 16,
    bit_depth: 8,
    chroma_sampling: ChromaSampling::Cs420,
    chroma_sample_position: ChromaSamplePosition::Unknown,
    pixel_range: PixelRange::Limited,
    color_description: None,
    mastering_display: None,
    content_light: None,
    enable_timing_info: false,
    still_picture: false,
    error_resilient: false,
    switch_frame_interval: 0,
    time_base: Rational { num: 1, den: 25 },
    min_key_frame_interval: 0,
    max_key_frame_interval: 1,
    reservoir_frame_delay: None,
    low_latency: false,
    quantizer: 100,
    min_quantizer: 0,
    bitrate: 16384,
    tune: Tune::Psychovisual,
    tile_cols: 0,
    tile_rows: 0,
    tiles: 0,
    rdo_lookahead_frames: 40,
    speed_settings: SpeedSettings {
      partition_range: PartitionRange::new(
        BlockSize::BLOCK_64X64,
        BlockSize::BLOCK_64X64,
      ),
      multiref: false,
      fast_deblock: true,
      reduced_tx_set: true,
      tx_domain_distortion: true,
      tx_domain_rate: false,
      encode_bottomup: false,
      rdo_tx_decision: false,
      prediction_modes: PredictionModesSetting::Simple,
      include_near_mvs: false,
      no_scene_detection: true,
      fast_scene_detection: false,
      diamond_me: true,
      cdef: true,
      lrf: true,
      use_satd_subpel: false,
      non_square_partition: false,
      ..Default::default()
    },
  };
  let config = Config::new().with_encoder_config(enc).with_threads(1);

  let mut ctx: Context<u8> = config.new_context().unwrap();
  ctx.send_frame(ctx.new_frame()).unwrap();
  ctx.flush();

  ctx.receive_packet().unwrap();
}
