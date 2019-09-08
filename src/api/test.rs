use super::*;

use interpolate_name::interpolate_test;

fn setup_encoder<T: Pixel>(
  w: usize, h: usize, speed: usize, quantizer: usize, bit_depth: usize,
  chroma_sampling: ChromaSampling, min_keyint: u64, max_keyint: u64,
  bitrate: i32, low_latency: bool, no_scene_detection: bool,
  rdo_lookahead_frames: usize,
) -> Context<T> {
  assert!(bit_depth == 8 || std::mem::size_of::<T>() > 1);
  let mut enc = EncoderConfig::with_speed_preset(speed);
  enc.quantizer = quantizer;
  enc.min_key_frame_interval = min_keyint;
  enc.max_key_frame_interval = max_keyint;
  enc.low_latency = low_latency;
  enc.width = w;
  enc.height = h;
  enc.bit_depth = bit_depth;
  enc.chroma_sampling = chroma_sampling;
  enc.bitrate = bitrate;
  enc.speed_settings.no_scene_detection = no_scene_detection;
  enc.rdo_lookahead_frames = rdo_lookahead_frames;

  let cfg = Config { enc, threads: 0 };

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
  ctx.inner.frame_invariants.into_iter().map(|(_, v)| v)
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
    true,
    10,
  );
  let limit = 10 - missing;
  send_frames(&mut ctx, limit, 0);
  ctx.flush();

  let data = get_frame_invariants(ctx)
    .map(|fi| {
      (
        fi.input_frameno,
        fi.frame_type,
        fi.pyramid_level,
        fi.show_existing_frame,
        !fi.invalid,
      )
    })
    .collect::<Vec<_>>();

  assert_eq!(
    &data[..],
    match missing {
      0 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (1, FrameType::INTER, 0, false, true),
          (2, FrameType::INTER, 0, false, true),
          (3, FrameType::INTER, 0, false, true),
          (4, FrameType::INTER, 0, false, true),
          (5, FrameType::KEY, 0, false, true),
          (6, FrameType::INTER, 0, false, true),
          (7, FrameType::INTER, 0, false, true),
          (8, FrameType::INTER, 0, false, true),
          (9, FrameType::INTER, 0, false, true),
        ][..]
      }
      1 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (1, FrameType::INTER, 0, false, true),
          (2, FrameType::INTER, 0, false, true),
          (3, FrameType::INTER, 0, false, true),
          (4, FrameType::INTER, 0, false, true),
          (5, FrameType::KEY, 0, false, true),
          (6, FrameType::INTER, 0, false, true),
          (7, FrameType::INTER, 0, false, true),
          (8, FrameType::INTER, 0, false, true),
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
    true,
    10,
  );

  // TODO: when we support more pyramid depths, this test will need tweaks.
  assert_eq!(ctx.inner.inter_cfg.pyramid_depth, 2);

  let limit = 10 - missing;
  send_frames(&mut ctx, limit, 0);
  ctx.flush();

  let data = get_frame_invariants(ctx)
    .map(|fi| {
      (
        fi.input_frameno,
        fi.frame_type,
        fi.pyramid_level,
        fi.show_existing_frame,
        !fi.invalid,
      )
    })
    .collect::<Vec<_>>();

  assert_eq!(
    &data[..],
    match missing {
      0 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (4, FrameType::INTER, 0, false, true),
          (2, FrameType::INTER, 1, false, true),
          (1, FrameType::INTER, 2, false, true),
          (2, FrameType::INTER, 1, true, true),
          (3, FrameType::INTER, 2, false, true),
          (4, FrameType::INTER, 0, true, true),
          (5, FrameType::KEY, 0, false, true),
          (9, FrameType::INTER, 0, false, true),
          (7, FrameType::INTER, 1, false, true),
          (6, FrameType::INTER, 2, false, true),
          (7, FrameType::INTER, 1, true, true),
          (8, FrameType::INTER, 2, false, true),
          (9, FrameType::INTER, 0, true, true),
        ][..]
      }
      1 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (4, FrameType::INTER, 0, false, true),
          (2, FrameType::INTER, 1, false, true),
          (1, FrameType::INTER, 2, false, true),
          (2, FrameType::INTER, 1, true, true),
          (3, FrameType::INTER, 2, false, true),
          (4, FrameType::INTER, 0, true, true),
          (5, FrameType::KEY, 0, false, true),
          (5, FrameType::INTER, 0, false, false),
          (7, FrameType::INTER, 1, false, true),
          (6, FrameType::INTER, 2, false, true),
          (7, FrameType::INTER, 1, true, true),
          (8, FrameType::INTER, 2, false, true),
          (8, FrameType::INTER, 2, false, false),
        ][..]
      }
      2 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (4, FrameType::INTER, 0, false, true),
          (2, FrameType::INTER, 1, false, true),
          (1, FrameType::INTER, 2, false, true),
          (2, FrameType::INTER, 1, true, true),
          (3, FrameType::INTER, 2, false, true),
          (4, FrameType::INTER, 0, true, true),
          (5, FrameType::KEY, 0, false, true),
          (5, FrameType::INTER, 0, false, false),
          (7, FrameType::INTER, 1, false, true),
          (6, FrameType::INTER, 2, false, true),
          (7, FrameType::INTER, 1, true, true),
          (7, FrameType::INTER, 1, false, false),
          (7, FrameType::INTER, 1, false, false),
        ][..]
      }
      3 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (4, FrameType::INTER, 0, false, true),
          (2, FrameType::INTER, 1, false, true),
          (1, FrameType::INTER, 2, false, true),
          (2, FrameType::INTER, 1, true, true),
          (3, FrameType::INTER, 2, false, true),
          (4, FrameType::INTER, 0, true, true),
          (5, FrameType::KEY, 0, false, true),
          (5, FrameType::INTER, 0, false, false),
          (5, FrameType::INTER, 0, false, false),
          (6, FrameType::INTER, 2, false, true),
          (6, FrameType::INTER, 2, false, false),
          (6, FrameType::INTER, 2, false, false),
          (6, FrameType::INTER, 2, false, false),
        ][..]
      }
      4 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (4, FrameType::INTER, 0, false, true),
          (2, FrameType::INTER, 1, false, true),
          (1, FrameType::INTER, 2, false, true),
          (2, FrameType::INTER, 1, true, true),
          (3, FrameType::INTER, 2, false, true),
          (4, FrameType::INTER, 0, true, true),
          (5, FrameType::KEY, 0, false, true),
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
    false,
    10,
  );

  // TODO: when we support more pyramid depths, this test will need tweaks.
  assert_eq!(ctx.inner.inter_cfg.pyramid_depth, 2);

  let limit = 5;
  send_frames(&mut ctx, limit, scene_change_at);
  ctx.flush();

  let data = get_frame_invariants(ctx)
    .map(|fi| {
      (
        fi.input_frameno,
        fi.frame_type,
        fi.pyramid_level,
        fi.show_existing_frame,
        !fi.invalid,
      )
    })
    .collect::<Vec<_>>();

  assert_eq!(
    &data[..],
    match scene_change_at {
      0 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (4, FrameType::INTER, 0, false, true),
          (2, FrameType::INTER, 1, false, true),
          (1, FrameType::INTER, 2, false, true),
          (2, FrameType::INTER, 1, true, true),
          (3, FrameType::INTER, 2, false, true),
          (4, FrameType::INTER, 0, true, true),
        ][..]
      }
      1 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (1, FrameType::KEY, 0, false, true),
          (1, FrameType::INTER, 0, false, false),
          (3, FrameType::INTER, 1, false, true),
          (2, FrameType::INTER, 2, false, true),
          (3, FrameType::INTER, 1, true, true),
          (4, FrameType::INTER, 2, false, true),
          (4, FrameType::INTER, 2, false, false),
        ][..]
      }
      2 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (0, FrameType::INTER, 0, false, false),
          (0, FrameType::INTER, 0, false, false),
          (1, FrameType::INTER, 2, false, true),
          (1, FrameType::INTER, 2, false, false),
          (1, FrameType::INTER, 2, false, false),
          (1, FrameType::INTER, 2, false, false),
          (2, FrameType::KEY, 0, false, true),
          (2, FrameType::INTER, 0, false, false),
          (4, FrameType::INTER, 1, false, true),
          (3, FrameType::INTER, 2, false, true),
          (4, FrameType::INTER, 1, true, true),
          (4, FrameType::INTER, 1, false, false),
          (4, FrameType::INTER, 1, false, false),
        ][..]
      }
      3 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (0, FrameType::INTER, 0, false, false),
          (2, FrameType::INTER, 1, false, true),
          (1, FrameType::INTER, 2, false, true),
          (2, FrameType::INTER, 1, true, true),
          (2, FrameType::INTER, 1, false, false),
          (2, FrameType::INTER, 1, false, false),
          (3, FrameType::KEY, 0, false, true),
          (3, FrameType::INTER, 0, false, false),
          (3, FrameType::INTER, 0, false, false),
          (4, FrameType::INTER, 2, false, true),
          (4, FrameType::INTER, 2, false, false),
          (4, FrameType::INTER, 2, false, false),
          (4, FrameType::INTER, 2, false, false),
        ][..]
      }
      4 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (0, FrameType::INTER, 0, false, false),
          (2, FrameType::INTER, 1, false, true),
          (1, FrameType::INTER, 2, false, true),
          (2, FrameType::INTER, 1, true, true),
          (3, FrameType::INTER, 2, false, true),
          (3, FrameType::INTER, 2, false, false),
          (4, FrameType::KEY, 0, false, true),
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

  let data = get_frame_invariants(ctx)
    .map(|fi| {
      (
        fi.input_frameno,
        fi.frame_type,
        fi.pyramid_level,
        fi.show_existing_frame,
        !fi.invalid,
      )
    })
    .collect::<Vec<_>>();

  assert_eq!(
    &data[..],
    match missing {
      0 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (4, FrameType::INTER, 0, false, true),
          (2, FrameType::INTER, 1, false, true),
          (1, FrameType::INTER, 2, false, true),
          (2, FrameType::INTER, 1, true, true),
          (3, FrameType::INTER, 2, false, true),
          (4, FrameType::INTER, 0, true, true),
          (5, FrameType::KEY, 0, false, true),
          (9, FrameType::INTER, 0, false, true),
          (7, FrameType::INTER, 1, false, true),
          (6, FrameType::INTER, 2, false, true),
          (7, FrameType::INTER, 1, true, true),
          (8, FrameType::INTER, 2, false, true),
          (9, FrameType::INTER, 0, true, true),
        ][..]
      }
      1 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (4, FrameType::INTER, 0, false, true),
          (2, FrameType::INTER, 1, false, true),
          (1, FrameType::INTER, 2, false, true),
          (2, FrameType::INTER, 1, true, true),
          (3, FrameType::INTER, 2, false, true),
          (4, FrameType::INTER, 0, true, true),
          (5, FrameType::KEY, 0, false, true),
          (5, FrameType::INTER, 0, false, false),
          (7, FrameType::INTER, 1, false, true),
          (6, FrameType::INTER, 2, false, true),
          (7, FrameType::INTER, 1, true, true),
          (8, FrameType::INTER, 2, false, true),
          (8, FrameType::INTER, 2, false, false),
        ][..]
      }
      2 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (4, FrameType::INTER, 0, false, true),
          (2, FrameType::INTER, 1, false, true),
          (1, FrameType::INTER, 2, false, true),
          (2, FrameType::INTER, 1, true, true),
          (3, FrameType::INTER, 2, false, true),
          (4, FrameType::INTER, 0, true, true),
          (5, FrameType::KEY, 0, false, true),
          (5, FrameType::INTER, 0, false, false),
          (7, FrameType::INTER, 1, false, true),
          (6, FrameType::INTER, 2, false, true),
          (7, FrameType::INTER, 1, true, true),
          (7, FrameType::INTER, 1, false, false),
          (7, FrameType::INTER, 1, false, false),
        ][..]
      }
      3 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (4, FrameType::INTER, 0, false, true),
          (2, FrameType::INTER, 1, false, true),
          (1, FrameType::INTER, 2, false, true),
          (2, FrameType::INTER, 1, true, true),
          (3, FrameType::INTER, 2, false, true),
          (4, FrameType::INTER, 0, true, true),
          (5, FrameType::KEY, 0, false, true),
          (5, FrameType::INTER, 0, false, false),
          (5, FrameType::INTER, 0, false, false),
          (6, FrameType::INTER, 2, false, true),
          (6, FrameType::INTER, 2, false, false),
          (6, FrameType::INTER, 2, false, false),
          (6, FrameType::INTER, 2, false, false),
        ][..]
      }
      4 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (4, FrameType::INTER, 0, false, true),
          (2, FrameType::INTER, 1, false, true),
          (1, FrameType::INTER, 2, false, true),
          (2, FrameType::INTER, 1, true, true),
          (3, FrameType::INTER, 2, false, true),
          (4, FrameType::INTER, 0, true, true),
          (5, FrameType::KEY, 0, false, true),
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
    false,
    10,
  );

  // TODO: when we support more pyramid depths, this test will need tweaks.
  assert_eq!(ctx.inner.inter_cfg.pyramid_depth, 2);

  let limit = 5;
  for i in 0..limit {
    send_frames(&mut ctx, 1, scene_change_at.saturating_sub(i));
  }
  ctx.flush();

  let data = get_frame_invariants(ctx)
    .map(|fi| {
      (
        fi.input_frameno,
        fi.frame_type,
        fi.pyramid_level,
        fi.show_existing_frame,
        !fi.invalid,
      )
    })
    .collect::<Vec<_>>();

  assert_eq!(
    &data[..],
    match scene_change_at {
      0 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (4, FrameType::INTER, 0, false, true),
          (2, FrameType::INTER, 1, false, true),
          (1, FrameType::INTER, 2, false, true),
          (2, FrameType::INTER, 1, true, true),
          (3, FrameType::INTER, 2, false, true),
          (4, FrameType::INTER, 0, true, true),
        ][..]
      }
      1 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (1, FrameType::KEY, 0, false, true),
          (1, FrameType::INTER, 0, false, false),
          (3, FrameType::INTER, 1, false, true),
          (2, FrameType::INTER, 2, false, true),
          (3, FrameType::INTER, 1, true, true),
          (4, FrameType::INTER, 2, false, true),
          (4, FrameType::INTER, 2, false, false),
        ][..]
      }
      2 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (0, FrameType::INTER, 0, false, false),
          (0, FrameType::INTER, 0, false, false),
          (1, FrameType::INTER, 2, false, true),
          (1, FrameType::INTER, 2, false, false),
          (1, FrameType::INTER, 2, false, false),
          (1, FrameType::INTER, 2, false, false),
          (2, FrameType::KEY, 0, false, true),
          (2, FrameType::INTER, 0, false, false),
          (4, FrameType::INTER, 1, false, true),
          (3, FrameType::INTER, 2, false, true),
          (4, FrameType::INTER, 1, true, true),
          (4, FrameType::INTER, 1, false, false),
          (4, FrameType::INTER, 1, false, false),
        ][..]
      }
      3 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (0, FrameType::INTER, 0, false, false),
          (2, FrameType::INTER, 1, false, true),
          (1, FrameType::INTER, 2, false, true),
          (2, FrameType::INTER, 1, true, true),
          (2, FrameType::INTER, 1, false, false),
          (2, FrameType::INTER, 1, false, false),
          (3, FrameType::KEY, 0, false, true),
          (3, FrameType::INTER, 0, false, false),
          (3, FrameType::INTER, 0, false, false),
          (4, FrameType::INTER, 2, false, true),
          (4, FrameType::INTER, 2, false, false),
          (4, FrameType::INTER, 2, false, false),
          (4, FrameType::INTER, 2, false, false),
        ][..]
      }
      4 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (0, FrameType::INTER, 0, false, false),
          (2, FrameType::INTER, 1, false, true),
          (1, FrameType::INTER, 2, false, true),
          (2, FrameType::INTER, 1, true, true),
          (3, FrameType::INTER, 2, false, true),
          (3, FrameType::INTER, 2, false, false),
          (4, FrameType::KEY, 0, false, true),
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

  let fp = FrameParameters { frame_type_override };

  let _ = ctx.send_frame((input, fp));
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

  let data = get_frame_invariants(ctx)
    .map(|fi| {
      (
        fi.input_frameno,
        fi.frame_type,
        fi.pyramid_level,
        fi.show_existing_frame,
        !fi.invalid,
      )
    })
    .collect::<Vec<_>>();

  assert_eq!(
    &data[..],
    match kf_at {
      0 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (4, FrameType::INTER, 0, false, true),
          (2, FrameType::INTER, 1, false, true),
          (1, FrameType::INTER, 2, false, true),
          (2, FrameType::INTER, 1, true, true),
          (3, FrameType::INTER, 2, false, true),
          (4, FrameType::INTER, 0, true, true),
        ][..]
      }
      1 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (1, FrameType::KEY, 0, false, true),
          (1, FrameType::INTER, 0, false, false),
          (3, FrameType::INTER, 1, false, true),
          (2, FrameType::INTER, 2, false, true),
          (3, FrameType::INTER, 1, true, true),
          (4, FrameType::INTER, 2, false, true),
          (4, FrameType::INTER, 2, false, false),
        ][..]
      }
      2 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (0, FrameType::INTER, 0, false, false),
          (0, FrameType::INTER, 0, false, false),
          (1, FrameType::INTER, 2, false, true),
          (1, FrameType::INTER, 2, false, false),
          (1, FrameType::INTER, 2, false, false),
          (1, FrameType::INTER, 2, false, false),
          (2, FrameType::KEY, 0, false, true),
          (2, FrameType::INTER, 0, false, false),
          (4, FrameType::INTER, 1, false, true),
          (3, FrameType::INTER, 2, false, true),
          (4, FrameType::INTER, 1, true, true),
          (4, FrameType::INTER, 1, false, false),
          (4, FrameType::INTER, 1, false, false),
        ][..]
      }
      3 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (0, FrameType::INTER, 0, false, false),
          (2, FrameType::INTER, 1, false, true),
          (1, FrameType::INTER, 2, false, true),
          (2, FrameType::INTER, 1, true, true),
          (2, FrameType::INTER, 1, false, false),
          (2, FrameType::INTER, 1, false, false),
          (3, FrameType::KEY, 0, false, true),
          (3, FrameType::INTER, 0, false, false),
          (3, FrameType::INTER, 0, false, false),
          (4, FrameType::INTER, 2, false, true),
          (4, FrameType::INTER, 2, false, false),
          (4, FrameType::INTER, 2, false, false),
          (4, FrameType::INTER, 2, false, false),
        ][..]
      }
      4 => {
        &[
          (0, FrameType::KEY, 0, false, true),
          (0, FrameType::INTER, 0, false, false),
          (2, FrameType::INTER, 1, false, true),
          (1, FrameType::INTER, 2, false, true),
          (2, FrameType::INTER, 1, true, true),
          (3, FrameType::INTER, 2, false, true),
          (3, FrameType::INTER, 2, false, false),
          (4, FrameType::KEY, 0, false, true),
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

  let data = get_frame_invariants(ctx)
    .map(|fi| {
      (
        fi.input_frameno,
        fi.frame_type,
        fi.pyramid_level,
        fi.show_existing_frame,
        !fi.invalid,
      )
    })
    .collect::<Vec<_>>();

  assert_eq!(
    &data[..],
    &[
      (0, FrameType::KEY, 0, false, true),
      (4, FrameType::INTER, 0, false, true),
      (2, FrameType::INTER, 1, false, true),
      (1, FrameType::INTER, 2, false, true),
      (2, FrameType::INTER, 1, true, true),
      (3, FrameType::INTER, 2, false, true),
      (4, FrameType::INTER, 0, true, true),
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
    .map(|fi| {
      (
        fi.input_frameno,
        fi.frame_type,
        fi.pyramid_level,
        fi.show_existing_frame,
        !fi.invalid,
      )
    })
    .collect::<Vec<_>>();

  assert_eq!(
    &data[..],
    &[
      (0, FrameType::KEY, 0, false, true),
      (4, FrameType::INTER, 0, false, true),
      (2, FrameType::INTER, 1, false, true),
      (1, FrameType::INTER, 2, false, true),
      (2, FrameType::INTER, 1, true, true),
      (3, FrameType::INTER, 2, false, true),
      (4, FrameType::INTER, 0, true, true),
      (4, FrameType::INTER, 0, false, false),
      (6, FrameType::INTER, 1, false, true),
      (5, FrameType::INTER, 2, false, true),
      (6, FrameType::INTER, 1, true, true),
      (7, FrameType::INTER, 2, false, true),
      (7, FrameType::INTER, 2, false, false),
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
  ctx.flush();

  let data = get_frame_invariants(ctx)
    .map(|fi| {
      (
        fi.input_frameno,
        fi.frame_type,
        fi.pyramid_level,
        fi.show_existing_frame,
        !fi.invalid,
      )
    })
    .collect::<Vec<_>>();

  assert_eq!(
    &data[..],
    &[
      (0, FrameType::KEY, 0, false, true),
      (0, FrameType::INTER, 0, false, false),
      (0, FrameType::INTER, 0, false, false),
      (1, FrameType::INTER, 2, false, true),
      (1, FrameType::INTER, 2, false, false),
      (1, FrameType::INTER, 2, false, false),
      (1, FrameType::INTER, 2, false, false),
      (2, FrameType::KEY, 0, false, true),
      (6, FrameType::INTER, 0, false, true),
      (4, FrameType::INTER, 1, false, true),
      (3, FrameType::INTER, 2, false, true),
      (4, FrameType::INTER, 1, true, true),
      (5, FrameType::INTER, 2, false, true),
      (6, FrameType::INTER, 0, true, true),
      (6, FrameType::INTER, 0, false, false),
      (6, FrameType::INTER, 0, false, false),
      (6, FrameType::INTER, 0, false, false),
      (6, FrameType::INTER, 0, false, false),
      (6, FrameType::INTER, 0, false, false),
      (6, FrameType::INTER, 0, false, false),
      (7, FrameType::KEY, 0, false, true),
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
  ctx.flush();

  let data = get_frame_invariants(ctx)
    .map(|fi| {
      (
        fi.input_frameno,
        fi.frame_type,
        fi.pyramid_level,
        fi.show_existing_frame,
        !fi.invalid,
      )
    })
    .collect::<Vec<_>>();

  assert_eq!(
    &data[..],
    &[
      (0, FrameType::KEY, 0, false, true),
      (4, FrameType::INTER, 0, false, true),
      (2, FrameType::INTER, 1, false, true),
      (1, FrameType::INTER, 2, false, true),
      (2, FrameType::INTER, 1, true, true),
      (3, FrameType::INTER, 2, false, true),
      (4, FrameType::INTER, 0, true, true),
      (4, FrameType::INTER, 0, false, false),
      (6, FrameType::INTER, 1, false, true),
      (5, FrameType::INTER, 2, false, true),
      (6, FrameType::INTER, 1, true, true),
      (7, FrameType::INTER, 2, false, true),
      (7, FrameType::INTER, 2, false, false),
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
      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 29, 29, 30, 30, 29, 29, 30, 30,
      29, 29, 30, 30, 29, 29, 30, 30, 29, 29, 30, 30, 29, 29, 30, 30, 29, 29,
      30, 30, 29, 29,
    ],
    pre_receive_fi_lens: [
      1, 1, 1, 1, 7, 7, 7, 7, 13, 13, 13, 13, 19, 19, 19, 19, 20, 21, 22, 24,
      26, 27, 28, 30, 32, 33, 34, 36, 38, 36, 35, 36, 38, 36, 35, 36, 38, 36,
      35, 36, 38, 36, 35, 36, 38, 36, 35, 36, 38, 36, 35, 36, 38, 36, 35, 36,
      38, 36, 35, 36,
    ],
    post_receive_frame_q_lens: [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26, 27, 28, 29, 28, 28, 29, 29, 28, 28, 29, 29, 28,
      28, 29, 29, 28, 28, 29, 29, 28, 28, 29, 29, 28, 28, 29, 29, 28, 28, 29,
      29, 28, 28, 29,
    ],
    post_receive_fi_lens: [
      1, 1, 1, 1, 7, 7, 7, 7, 13, 13, 13, 13, 19, 19, 19, 19, 20, 21, 22, 24,
      26, 27, 28, 30, 32, 33, 34, 36, 35, 34, 34, 36, 35, 34, 34, 36, 35, 34,
      34, 36, 35, 34, 34, 36, 35, 34, 34, 36, 35, 34, 34, 36, 35, 34, 34, 36,
      35, 34, 34, 36,
    ],
  };
  lookahead_size_properly_bounded(LOOKAHEAD_SIZE, &EXPECTATIONS);
}

#[test]
fn lookahead_size_properly_bounded_10() {
  const LOOKAHEAD_SIZE: usize = 10;
  const EXPECTATIONS: LookaheadTestExpectations = LookaheadTestExpectations {
    pre_receive_frame_q_lens: [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 31, 31, 32, 32, 31, 31,
      32, 32, 31, 31, 32, 32, 31, 31, 32, 32, 31, 31, 32, 32, 31, 31, 32, 32,
      31, 31, 32, 32,
    ],
    pre_receive_fi_lens: [
      1, 1, 1, 1, 7, 7, 7, 7, 13, 13, 13, 13, 19, 19, 19, 19, 22, 24, 25, 25,
      26, 27, 28, 30, 32, 33, 34, 36, 38, 39, 40, 39, 39, 39, 40, 39, 39, 39,
      40, 39, 39, 39, 40, 39, 39, 39, 40, 39, 39, 39, 40, 39, 39, 39, 40, 39,
      39, 39, 40, 39,
    ],
    post_receive_frame_q_lens: [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 30, 30, 31, 31, 30, 30, 31,
      31, 30, 30, 31, 31, 30, 30, 31, 31, 30, 30, 31, 31, 30, 30, 31, 31, 30,
      30, 31, 31, 30,
    ],
    post_receive_fi_lens: [
      1, 1, 1, 1, 7, 7, 7, 7, 13, 13, 13, 13, 19, 19, 19, 19, 22, 24, 25, 25,
      26, 27, 28, 30, 32, 33, 34, 36, 38, 39, 37, 37, 38, 39, 37, 37, 38, 39,
      37, 37, 38, 39, 37, 37, 38, 39, 37, 37, 38, 39, 37, 37, 38, 39, 37, 37,
      38, 39, 37, 37,
    ],
  };
  lookahead_size_properly_bounded(LOOKAHEAD_SIZE, &EXPECTATIONS);
}

#[test]
fn lookahead_size_properly_bounded_16() {
  const LOOKAHEAD_SIZE: usize = 16;
  const EXPECTATIONS: LookaheadTestExpectations = LookaheadTestExpectations {
    pre_receive_frame_q_lens: [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
      37, 37, 38, 38, 37, 37, 38, 38, 37, 37, 38, 38, 37, 37, 38, 38, 37, 37,
      38, 38, 37, 37,
    ],
    pre_receive_fi_lens: [
      1, 1, 1, 1, 7, 7, 7, 7, 13, 13, 13, 13, 19, 19, 19, 19, 25, 25, 25, 25,
      31, 31, 31, 31, 32, 33, 34, 36, 38, 39, 40, 42, 44, 45, 46, 48, 50, 48,
      47, 48, 50, 48, 47, 48, 50, 48, 47, 48, 50, 48, 47, 48, 50, 48, 47, 48,
      50, 48, 47, 48,
    ],
    post_receive_frame_q_lens: [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 36,
      36, 37, 37, 36, 36, 37, 37, 36, 36, 37, 37, 36, 36, 37, 37, 36, 36, 37,
      37, 36, 36, 37,
    ],
    post_receive_fi_lens: [
      1, 1, 1, 1, 7, 7, 7, 7, 13, 13, 13, 13, 19, 19, 19, 19, 25, 25, 25, 25,
      31, 31, 31, 31, 32, 33, 34, 36, 38, 39, 40, 42, 44, 45, 46, 48, 47, 46,
      46, 48, 47, 46, 46, 48, 47, 46, 46, 48, 47, 46, 46, 48, 47, 46, 46, 48,
      47, 46, 46, 48,
    ],
  };
  lookahead_size_properly_bounded(LOOKAHEAD_SIZE, &EXPECTATIONS);
}

fn lookahead_size_properly_bounded(
  rdo_lookahead: usize, expectations: &LookaheadTestExpectations,
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
    false,
    true,
    rdo_lookahead,
  );

  const LIMIT: usize = 60;

  for i in 0..LIMIT {
    let input = ctx.new_frame();
    let _ = ctx.send_frame(input);
    assert_eq!(
      ctx.inner.frame_q.len(),
      expectations.pre_receive_frame_q_lens[i]
    );
    assert_eq!(
      ctx.inner.frame_invariants.len(),
      expectations.pre_receive_fi_lens[i]
    );
    while ctx.receive_packet().is_ok() {
      // Receive packets until lookahead consumed, due to pyramids receiving frames in groups
    }
    assert_eq!(
      ctx.inner.frame_q.len(),
      expectations.post_receive_frame_q_lens[i]
    );
    assert_eq!(
      ctx.inner.frame_invariants.len(),
      expectations.post_receive_fi_lens[i]
    );
  }

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
  let mut config = Config::default();
  config.enc.tile_cols = usize::max_value();
  let _: Result<Context<u8>, _> = config.new_context();
}

#[test]
fn max_key_frame_interval_overflow() {
  let mut config = Config::default();
  config.enc.max_key_frame_interval = i32::max_value() as u64;
  config.enc.reservoir_frame_delay = None;
  let _: Result<Context<u8>, _> = config.new_context();
}

#[test]
fn target_bitrate_overflow() {
  let mut config = Config::default();
  config.enc.bitrate = i32::max_value();
  config.enc.time_base = Rational::new(i64::max_value() as u64, 1);
  let _: Result<Context<u8>, _> = config.new_context();
}

#[test]
fn time_base_den_divide_by_zero() {
  let mut config = Config::default();
  config.enc.time_base = Rational::new(1, 0);
  let _: Result<Context<u8>, _> = config.new_context();
}

#[test]
fn large_width_assert() {
  let mut config = Config::default();
  config.enc.width = u32::max_value() as usize;
  let _: Result<Context<u8>, _> = config.new_context();
}

#[test]
fn reservoir_max_overflow() {
  let mut config = Config::default();
  config.enc.reservoir_frame_delay = Some(i32::max_value());
  config.enc.bitrate = i32::max_value();
  config.enc.time_base = Rational::new(i32::max_value() as u64 * 2, 1);
  let _: Result<Context<u8>, _> = config.new_context();
}

#[test]
fn zero_width() {
  let mut config = Config::default();
  config.enc.width = 0;
  let res: Result<Context<u8>, _> = config.new_context();
  assert!(res.is_err());
}

#[test]
fn rdo_lookahead_frames_overflow() {
  let mut config = Config::default();
  config.enc.rdo_lookahead_frames = usize::max_value();
  let res: Result<Context<u8>, _> = config.new_context();
  assert!(res.is_err());
}

#[test]
fn log_q_exp_overflow() {
  let config = Config {
    enc: EncoderConfig {
      width: 1,
      height: 1,
      bit_depth: 8,
      chroma_sampling: ChromaSampling::Cs420,
      chroma_sample_position: ChromaSamplePosition::Unknown,
      pixel_range: PixelRange::Limited,
      color_description: None,
      mastering_display: None,
      content_light: None,
      still_picture: false,
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
        min_block_size: BlockSize::BLOCK_64X64,
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
        diamond_me: true,
        cdef: true,
        quantizer_rdo: false,
        use_satd_subpel: false,
      },
      show_psnr: false,
      train_rdo: false,
    },
    threads: 1,
  };

  let mut ctx: Context<u8> = config.new_context().unwrap();
  for _ in 0..2 {
    ctx.send_frame(ctx.new_frame()).unwrap();
  }
  ctx.flush();

  ctx.receive_packet().unwrap();
  let _ = ctx.receive_packet();
}
