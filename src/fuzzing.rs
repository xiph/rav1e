// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use std::sync::Arc;

use libfuzzer_sys::arbitrary::{Arbitrary, Error, Unstructured};

use crate::prelude::*;

// Adding new fuzz targets
//
// 1. Add a function to this file which looks like this:
//
//    pub fn fuzz_something(data: Data) {
//      // Invoke everything you need.
//      //
//      // Your function may accept a value of any type that implements
//      // Arbitrary [1]. This is how fuzzer affects the execution—by
//      // feeding in different bytes, which result in different
//      // arbitrary values being generated.
//      // [1]: https://docs.rs/arbitrary/0.2.0/arbitrary/trait.Arbitrary.html
//      //
//      // Derive Debug for the structures you create with arbitrary data
//    }
//
// 2. cargo fuzz add something
// 3. Copy the contents of any other .rs file from fuzz/fuzz_targets/ into the
//    newly created fuzz/fuzz_targets/something.rs and change the function
//    being called to fuzz_something.
//
// Now you can fuzz the new target with cargo fuzz.

#[derive(Debug)]
pub struct ArbitraryConfig {
  config: Config,
}

impl Arbitrary for ArbitraryConfig {
  fn arbitrary(u: &mut Unstructured<'_>) -> Result<Self, Error> {
    let mut config = Config::default();
    config.threads = 1;
    config.enc.width = Arbitrary::arbitrary(u)?;
    config.enc.height = Arbitrary::arbitrary(u)?;
    config.enc.bit_depth = (u8::arbitrary(u)? % 17) as usize;
    config.enc.still_picture = Arbitrary::arbitrary(u)?;
    config.enc.time_base =
      Rational::new(Arbitrary::arbitrary(u)?, Arbitrary::arbitrary(u)?);
    config.enc.min_key_frame_interval = Arbitrary::arbitrary(u)?;
    config.enc.max_key_frame_interval = Arbitrary::arbitrary(u)?;
    config.enc.reservoir_frame_delay = Arbitrary::arbitrary(u)?;
    config.enc.low_latency = Arbitrary::arbitrary(u)?;
    config.enc.quantizer = Arbitrary::arbitrary(u)?;
    config.enc.min_quantizer = Arbitrary::arbitrary(u)?;
    config.enc.bitrate = Arbitrary::arbitrary(u)?;
    config.enc.tile_cols = Arbitrary::arbitrary(u)?;
    config.enc.tile_rows = Arbitrary::arbitrary(u)?;
    config.enc.tiles = Arbitrary::arbitrary(u)?;
    config.enc.rdo_lookahead_frames = Arbitrary::arbitrary(u)?;
    config.enc.speed_settings =
      SpeedSettings::from_preset(Arbitrary::arbitrary(u)?);
    Ok(Self { config })
  }
}

pub fn fuzz_construct_context(arbitrary: ArbitraryConfig) {
  let _: Result<Context<u16>, _> = arbitrary.config.new_context();
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

pub fn fuzz_encode(arbitrary: ArbitraryConfig) {
  let res = arbitrary.config.new_context();
  if res.is_err() {
    return;
  }
  let mut context: Context<u8> = res.unwrap();

  let frame_count = 3; // g.g::<u8>() % 3 + 1;
  let mut frame = context.new_frame();
  let frames = (0..frame_count).map(|_| {
    for plane in &mut frame.planes {
      let stride = plane.cfg.stride;
      for row in plane.data_origin_mut().chunks_mut(stride) {
        for pixel in row {
          *pixel = 42; // g.g();
        }
      }
    }

    frame.clone()
  });

  let _ = encode_frames(&mut context, frames);
}

#[derive(Debug)]
pub struct DecodeTestParameters {
  w: usize,
  h: usize,
  speed: usize,
  q: usize,
  limit: usize,
  bit_depth: usize,
  chroma_sampling: ChromaSampling,
  min_keyint: u64,
  max_keyint: u64,
  switch_frame_interval: u64,
  low_latency: bool,
  error_resilient: bool,
  bitrate: i32,
  tile_cols_log2: usize,
  tile_rows_log2: usize,
  still_picture: bool,
  seed: [u8; 32],
}

impl Arbitrary for DecodeTestParameters {
  fn arbitrary(u: &mut Unstructured<'_>) -> Result<Self, Error> {
    let valid_samplings =
      [ChromaSampling::Cs420, ChromaSampling::Cs422, ChromaSampling::Cs444];
    Ok(Self {
      w: u8::arbitrary(u)? as usize + 16,
      h: u8::arbitrary(u)? as usize + 16,
      speed: u8::arbitrary(u)? as usize % 11,
      q: u8::arbitrary(u)? as usize,
      limit: (u8::arbitrary(u)? % 3) as usize + 1,
      bit_depth: 8,
      chroma_sampling: valid_samplings[(u8::arbitrary(u)? % 3) as usize],
      min_keyint: (u8::arbitrary(u)? % 4) as u64,
      max_keyint: (u8::arbitrary(u)? % 4) as u64,
      switch_frame_interval: 0,
      low_latency: bool::arbitrary(u)?,
      error_resilient: bool::arbitrary(u)?,
      bitrate: u16::arbitrary(u)? as i32,
      tile_cols_log2: bool::arbitrary(u)? as usize + 1,
      tile_rows_log2: bool::arbitrary(u)? as usize + 1,
      still_picture: bool::arbitrary(u)?,
      seed: Arbitrary::arbitrary(u)?,
    })
  }
}

#[cfg(feature = "decode_test_dav1d")]
pub fn fuzz_encode_decode(p: DecodeTestParameters) {
  use crate::test_encode_decode::*;

  let mut dec = get_decoder::<u8>("dav1d", p.w, p.h);
  dec.encode_decode(
    p.w,
    p.h,
    p.speed,
    p.q,
    if p.still_picture { 1 } else { p.limit },
    p.bit_depth,
    p.chroma_sampling,
    p.min_keyint,
    p.max_keyint,
    p.switch_frame_interval,
    p.low_latency,
    p.error_resilient,
    p.bitrate,
    p.tile_cols_log2,
    p.tile_rows_log2,
    p.still_picture,
    p.seed,
  );
}
