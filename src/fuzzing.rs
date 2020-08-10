// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use std::sync::Arc;

use arbitrary::{Arbitrary, Unstructured};

use crate::prelude::*;

// Adding new fuzz targets
//
// 1. Add a function to this file which looks like this:
//
//    pub fn fuzz_something(data: &[u8]) {
//      let mut g = create_generator!();
//
//      // Invoke everything you need.
//      //
//      // You should use g.g() to get an arbitrary value of any type that
//      // implements Arbitrary [1]. This is how fuzzer affects the
//      // execution—by feeding in different bytes, which result in different
//      // arbitrary values being generated.
//      // [1]: https://docs.rs/arbitrary/0.2.0/arbitrary/trait.Arbitrary.html
//      //
//      // Print out the structures you create with arbitrary data with
//      // debug!().
//    }
//
// 2. cargo fuzz add something
// 3. Copy the contents of any other .rs file from fuzz/fuzz_targets/ into the
//    newly created fuzz/fuzz_targets/something.rs and change the function
//    being called to fuzz_something.
//
// Now you can fuzz the new target with cargo fuzz.

// A helper for generating arbitrary data.
struct Generator<'a> {
  buffer: Unstructured<'a>,
}

impl<'a> Generator<'a> {
  fn new(data: &'a [u8]) -> Self {
    Self { buffer: Unstructured::new(data) }
  }

  fn g<T: Arbitrary>(&mut self) -> T {
    <T as Arbitrary>::arbitrary(&mut self.buffer).unwrap()
  }
}

macro_rules! create_generator {
  ($data:expr) => {{
    Generator::new($data)
  }};
}

pub fn fuzz_construct_context(data: &[u8]) {
  let mut g = create_generator!(data);

  let mut config = Config::default();
  config.threads = 1;
  config.enc.width = g.g();
  config.enc.height = g.g();
  config.enc.bit_depth = (g.g::<u8>() % 17) as usize;
  config.enc.still_picture = g.g();
  config.enc.time_base = Rational::new(g.g(), g.g());
  config.enc.min_key_frame_interval = g.g();
  config.enc.max_key_frame_interval = g.g();
  config.enc.reservoir_frame_delay = g.g();
  config.enc.low_latency = g.g();
  config.enc.quantizer = g.g();
  config.enc.min_quantizer = g.g();
  config.enc.bitrate = g.g();
  config.enc.tile_cols = g.g();
  config.enc.tile_rows = g.g();
  config.enc.tiles = g.g();
  config.enc.rdo_lookahead_frames = g.g();
  config.enc.speed_settings = SpeedSettings::from_preset(g.g());

  debug!("config = {:#?}", config);

  let _: Result<Context<u16>, _> = config.new_context();
}

fn encode_frames(
  ctx: &mut Context<u8>, mut frames: impl Iterator<Item = Frame<u8>>,
) -> Result<(), EncoderStatus> {
  loop {
    let rv = ctx.receive_packet();
    debug!("ctx.receive_packet() = {:#?}", rv);

    match rv {
      Ok(_packet) => {}
      Err(EncoderStatus::Encoded) => {}
      Err(EncoderStatus::LimitReached) => {
        break;
      }
      Err(EncoderStatus::NeedMoreData) => {
        ctx.send_frame(frames.next().map(Arc::new))?;
      }
      Err(EncoderStatus::EnoughData) => {
        unreachable!();
      }
      Err(EncoderStatus::NotReady) => {
        unreachable!();
      }
      Err(EncoderStatus::Failure) => {
        return Err(EncoderStatus::Failure);
      }
    }
  }

  Ok(())
}

pub fn fuzz_encode(data: &[u8]) {
  let mut g = create_generator!(data);

  let mut config = Config::default();
  config.threads = 1;
  config.enc.width = g.g::<u8>() as usize + 1;
  config.enc.height = g.g::<u8>() as usize + 1;
  config.enc.still_picture = g.g();
  config.enc.time_base = Rational::new(g.g(), g.g());
  config.enc.min_key_frame_interval = (g.g::<u8>() % 4) as u64;
  config.enc.max_key_frame_interval = (g.g::<u8>() % 4) as u64 + 1;
  config.enc.low_latency = g.g();
  config.enc.quantizer = g.g();
  config.enc.min_quantizer = g.g();
  config.enc.bitrate = g.g();
  // config.enc.tile_cols = g.g();
  // config.enc.tile_rows = g.g();
  // config.enc.tiles = g.g();
  config.enc.rdo_lookahead_frames = g.g();
  config.enc.speed_settings = SpeedSettings::from_preset(10);

  debug!("config = {:#?}", config);

  let res = config.new_context();
  if res.is_err() {
    return;
  }
  let mut context: Context<u8> = res.unwrap();

  let frame_count = g.g::<u8>() % 3 + 1;
  let mut frame = context.new_frame();
  let frames = (0..frame_count).map(|_| {
    for plane in &mut frame.planes {
      let stride = plane.cfg.stride;
      for row in plane.data_origin_mut().chunks_mut(stride) {
        for pixel in row {
          *pixel = g.g();
        }
      }
    }

    frame.clone()
  });

  let _ = encode_frames(&mut context, frames);
}

#[cfg(feature = "decode_test_dav1d")]
pub fn fuzz_encode_decode(data: &[u8]) {
  use crate::test_encode_decode::*;

  let mut g = create_generator!(data);

  let w = g.g::<u8>() as usize + 16;
  let h = g.g::<u8>() as usize + 16;
  let speed = 10;
  let q = g.g::<u8>() as usize;
  let limit = (g.g::<u8>() % 3) as usize + 1;
  let min_keyint = g.g::<u64>() % 4;
  let max_keyint = g.g::<u64>() % 4 + 1;
  let switch_frame_interval = 0;
  let low_latency = g.g();
  let error_resilient = false;
  let bitrate = g.g();
  let still_picture = false;

  debug!(
    "w = {:#?}\n\
     h = {:#?}\n\
     speed = {:#?}\n\
     q = {:#?}\n\
     limit = {:#?}\n\
     min_keyint = {:#?}\n\
     max_keyint = {:#?}\n\
     low_latency = {:#?}\n\
     bitrate = {:#?}",
    w, h, speed, q, limit, min_keyint, max_keyint, low_latency, bitrate
  );

  let mut dec = get_decoder::<u8>("dav1d", w, h);
  dec.encode_decode(
    w,
    h,
    speed,
    q,
    limit,
    8,
    Default::default(),
    min_keyint,
    max_keyint,
    switch_frame_interval,
    low_latency,
    error_resilient,
    bitrate,
    1,
    1,
    still_picture,
  );
}
