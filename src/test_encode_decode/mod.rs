// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

// Fuzzing only uses a subset of these.
#![cfg_attr(fuzzing, allow(unused))]

use crate::color::ChromaSampling;
use crate::config::*;
use crate::util::Pixel;
use crate::*;
use interpolate_name::interpolate_test;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use std::collections::VecDeque;

#[cfg(feature = "decode_test")]
mod aom;
#[cfg(feature = "decode_test_dav1d")]
mod dav1d;

#[cfg(feature = "decode_test")]
use aom::AomDecoder;
#[cfg(feature = "decode_test_dav1d")]
use dav1d::Dav1dDecoder;

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

fn read_frame_batch<T: Pixel>(
  ctx: &mut Context<T>, ra: &mut ChaChaRng, limit: usize,
) {
  for _ in 0..limit {
    let mut input = ctx.new_frame();
    fill_frame(ra, &mut input);

    let _ = ctx.send_frame(input);
  }

  ctx.flush();
}

pub(crate) enum DecodeResult {
  Done,
  NotDone,
  Corrupted(usize),
}

pub(crate) trait TestDecoder<T: Pixel> {
  fn setup_decoder(w: usize, h: usize) -> Self
  where
    Self: Sized;
  fn encode_decode(
    &mut self, w: usize, h: usize, speed: usize, quantizer: usize,
    limit: usize, bit_depth: usize, chroma_sampling: ChromaSampling,
    min_keyint: u64, max_keyint: u64, switch_frame_interval: u64,
    low_latency: bool, error_resilient: bool, bitrate: i32,
    tile_cols_log2: usize, tile_rows_log2: usize, still_picture: bool,
  ) {
    let mut ra = ChaChaRng::from_seed([0; 32]);

    let mut ctx: Context<T> = setup_encoder(
      w,
      h,
      speed,
      quantizer,
      bit_depth,
      chroma_sampling,
      min_keyint,
      max_keyint,
      switch_frame_interval,
      low_latency,
      error_resilient,
      bitrate,
      tile_cols_log2,
      tile_rows_log2,
      still_picture,
    );

    debug!(
      "Encoding {}x{} speed {} quantizer {} bit-depth {} bitrate {}",
      w, h, speed, quantizer, bit_depth, bitrate
    );
    #[cfg(feature = "dump_ivf")]
    let mut out = std::fs::File::create(&format!(
      "out-{}x{}-s{}-q{}-r{}-{:?}.ivf",
      w, h, speed, quantizer, bitrate, chroma_sampling
    ))
    .unwrap();
    #[cfg(feature = "dump_ivf")]
    ivf::write_ivf_header(&mut out, w, h, 30, 1);

    let mut rec_fifo = VecDeque::new();
    read_frame_batch(&mut ctx, &mut ra, limit);

    for _ in 0..limit {
      let mut corrupted_count = 0;
      loop {
        let res = ctx.receive_packet();
        if let Ok(pkt) = res {
          debug!("Encoded packet {}", pkt.input_frameno);

          #[cfg(feature = "dump_ivf")]
          ivf::write_ivf_frame(&mut out, pkt.input_frameno, &pkt.data);

          if let Some(pkt_rec) = pkt.rec {
            rec_fifo.push_back((*pkt_rec).clone());
          }
          let packet = pkt.data;
          debug!("Decoding frame {}", pkt.input_frameno);
          match self.decode_packet(
            &packet,
            &mut rec_fifo,
            w,
            h,
            chroma_sampling,
            bit_depth,
          ) {
            DecodeResult::Done => {
              break;
            }
            DecodeResult::NotDone => {}
            DecodeResult::Corrupted(corrupted) => {
              corrupted_count += corrupted;
            }
          }
        } else {
          break;
        }
      }
      assert_eq!(corrupted_count, 0);
    }
  }
  fn decode_packet(
    &mut self, packet: &[u8], rec_fifo: &mut VecDeque<Frame<T>>, w: usize,
    h: usize, chroma_sampling: ChromaSampling, bit_depth: usize,
  ) -> DecodeResult;
}

pub fn compare_plane<T: Ord + std::fmt::Debug>(
  rec: &[T], rec_stride: usize, dec: &[T], dec_stride: usize, width: usize,
  height: usize, pli: usize,
) {
  for (row, line) in
    rec.chunks(rec_stride).zip(dec.chunks(dec_stride)).take(height).enumerate()
  {
    assert_eq!(
      &line.0[..width],
      &line.1[..width],
      "at row {} of plane {}",
      row,
      pli
    );
  }
}

fn setup_encoder<T: Pixel>(
  w: usize, h: usize, speed: usize, quantizer: usize, bit_depth: usize,
  chroma_sampling: ChromaSampling, min_keyint: u64, max_keyint: u64,
  switch_frame_interval: u64, low_latency: bool, error_resilient: bool,
  bitrate: i32, tile_cols_log2: usize, tile_rows_log2: usize,
  still_picture: bool,
) -> Context<T> {
  assert!(bit_depth == 8 || std::mem::size_of::<T>() > 1);
  let mut enc = EncoderConfig::with_speed_preset(speed);
  enc.quantizer = quantizer;
  enc.min_key_frame_interval = min_keyint;
  enc.max_key_frame_interval = max_keyint;
  enc.switch_frame_interval = switch_frame_interval;
  enc.low_latency = low_latency;
  enc.error_resilient = error_resilient;
  enc.width = w;
  enc.height = h;
  enc.bit_depth = bit_depth;
  enc.chroma_sampling = chroma_sampling;
  enc.bitrate = bitrate;
  enc.tile_cols = 1 << tile_cols_log2;
  enc.tile_rows = 1 << tile_rows_log2;
  enc.still_picture = still_picture;

  let cfg = Config { enc, threads: 0 };

  cfg.new_context().unwrap()
}

// TODO: support non-multiple-of-16 dimensions
static DIMENSION_OFFSETS: &[(usize, usize)] =
  &[(0, 0), (4, 4), (8, 8), (16, 16)];

fn speed(s: usize, decoder: &str) {
  let quantizer = 100;
  let limit = 5;
  let w = 64;
  let h = 80;

  for b in DIMENSION_OFFSETS.iter() {
    let w = w + b.0;
    let h = h + b.1;
    let mut dec = get_decoder::<u8>(decoder, w as usize, h as usize);
    dec.encode_decode(
      w,
      h,
      s,
      quantizer,
      limit,
      8,
      Default::default(),
      15,
      15,
      0,
      true,
      false,
      0,
      0,
      0,
      false,
    );
  }
}

macro_rules! test_speeds {
  ($($S:expr),+) => {
    $(
        paste::item!{
            #[cfg_attr(feature = "decode_test", interpolate_test(aom, "aom"))]
            #[cfg_attr(feature = "decode_test_dav1d", interpolate_test(dav1d, "dav1d"))]
            #[ignore]
            fn [<speed_ $S>](decoder: &str) {
                speed($S, decoder)
            }
        }
    )*
  }
}

test_speeds! { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }

macro_rules! test_dimensions {
  ($(($W:expr, $H:expr)),+) => {
    $(
        paste::item!{
            #[cfg_attr(feature = "decode_test", interpolate_name::interpolate_test(aom, "aom"))]
            #[cfg_attr(feature = "decode_test_dav1d", interpolate_name::interpolate_test(dav1d, "dav1d"))]
            fn [<dimension_ $W x $H>](decoder: &str) {
                super::dimension($W, $H, decoder)
            }
        }
    )*
  }
}

#[cfg(not(feature = "quick_test"))]
mod large_dimension {
  test_dimensions! {
    (512, 512),
    (1024, 1024),
    (2048, 2048)
  }
}

mod small_dimension {
  test_dimensions! {
    (256, 256),
    (258, 258),
    (260, 260),
    (262, 262),
    (264, 264),
    (265, 265)
  }
}

mod tiny_dimension {
  test_dimensions! {
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128)
  }
}

fn dimension(w: usize, h: usize, decoder: &str) {
  let quantizer = 100;
  let limit = 1;
  let speed = 10;

  let mut dec = get_decoder::<u8>(decoder, w as usize, h as usize);
  dec.encode_decode(
    w,
    h,
    speed,
    quantizer,
    limit,
    8,
    Default::default(),
    15,
    15,
    0,
    true,
    false,
    0,
    0,
    0,
    false,
  );
}

fn quantizer(decoder: &str, q: usize) {
  let limit = 5;
  let w = 64;
  let h = 80;
  let speed = 10;

  for b in DIMENSION_OFFSETS.iter() {
    let mut dec = get_decoder::<u8>(decoder, b.0, b.1);
    dec.encode_decode(
      w + b.0,
      h + b.1,
      speed,
      q,
      limit,
      8,
      Default::default(),
      15,
      15,
      0,
      true,
      false,
      0,
      0,
      0,
      false,
    );
  }
}

macro_rules! test_quantizer {
  ($($Q:expr),+) => {
    $(
      paste::item!{
        #[cfg_attr(feature = "decode_test", interpolate_test(aom, "aom"))]
        #[cfg_attr(feature = "decode_test_dav1d", interpolate_test(dav1d, "dav1d"))]
        fn [<quantizer_ $Q>](decoder: &str) {
          quantizer(decoder, $Q);
        }
      }
    )*
  }
}

test_quantizer! {60, 80, 100, 120}

#[cfg_attr(feature = "decode_test", interpolate_test(aom, "aom"))]
#[cfg_attr(feature = "decode_test_dav1d", interpolate_test(dav1d, "dav1d"))]
#[ignore]
fn bitrate(decoder: &str) {
  let limit = 5;
  let w = 64;
  let h = 80;
  let speed = 10;

  for &q in [172, 220, 252, 255].iter() {
    for &r in [100, 1000, 10_000].iter() {
      let mut dec = get_decoder::<u8>(decoder, w as usize, h as usize);
      dec.encode_decode(
        w,
        h,
        speed,
        q,
        limit,
        8,
        Default::default(),
        15,
        15,
        0,
        true,
        false,
        r,
        0,
        0,
        false,
      );
    }
  }
}

#[cfg_attr(feature = "decode_test", interpolate_test(aom, "aom"))]
#[cfg_attr(feature = "decode_test_dav1d", interpolate_test(dav1d, "dav1d"))]
fn keyframes(decoder: &str) {
  let limit = 12;
  let w = 64;
  let h = 80;
  let speed = 9;
  let q = 100;

  let mut dec = get_decoder::<u8>(decoder, w as usize, h as usize);
  dec.encode_decode(
    w,
    h,
    speed,
    q,
    limit,
    8,
    Default::default(),
    6,
    6,
    0,
    true,
    false,
    0,
    0,
    0,
    false,
  );
}

#[cfg_attr(feature = "decode_test", interpolate_test(aom, "aom"))]
#[cfg_attr(feature = "decode_test_dav1d", interpolate_test(dav1d, "dav1d"))]
fn reordering(decoder: &str) {
  let limit = 12;
  let w = 64;
  let h = 80;
  let speed = 10;
  let q = 100;

  for keyint in &[4, 5, 6] {
    let mut dec = get_decoder::<u8>(decoder, w as usize, h as usize);
    dec.encode_decode(
      w,
      h,
      speed,
      q,
      limit,
      8,
      Default::default(),
      *keyint,
      *keyint,
      0,
      false,
      false,
      0,
      0,
      0,
      false,
    );
  }
}

#[cfg_attr(feature = "decode_test", interpolate_test(aom, "aom"))]
#[cfg_attr(feature = "decode_test_dav1d", interpolate_test(dav1d, "dav1d"))]
fn reordering_short_video(decoder: &str) {
  // Regression test for https://github.com/xiph/rav1e/issues/890
  let limit = 2;
  let w = 64;
  let h = 80;
  let speed = 10;
  let q = 100;
  let keyint = 12;

  let mut dec = get_decoder::<u8>(decoder, w as usize, h as usize);
  dec.encode_decode(
    w,
    h,
    speed,
    q,
    limit,
    8,
    Default::default(),
    keyint,
    keyint,
    0,
    false,
    false,
    0,
    0,
    0,
    false,
  );
}

#[cfg_attr(feature = "decode_test", interpolate_test(aom, "aom"))]
#[cfg_attr(feature = "decode_test_dav1d", interpolate_test(dav1d, "dav1d"))]
fn error_resilient(decoder: &str) {
  let limit = 2;
  let w = 64;
  let h = 80;
  let speed = 10;
  let q = 100;
  let keyint = 12;

  let mut dec = get_decoder::<u8>(decoder, w as usize, h as usize);
  dec.encode_decode(
    w,
    h,
    speed,
    q,
    limit,
    8,
    Default::default(),
    keyint,
    keyint,
    0,
    true,
    true,
    0,
    0,
    0,
    false,
  );
}

#[cfg_attr(feature = "decode_test", interpolate_test(aom, "aom"))]
#[cfg_attr(feature = "decode_test_dav1d", interpolate_test(dav1d, "dav1d"))]
fn error_resilient_reordering(decoder: &str) {
  let limit = 6;
  let w = 64;
  let h = 80;
  let speed = 10;
  let q = 100;

  for keyint in &[4, 5, 6] {
    let mut dec = get_decoder::<u8>(decoder, w as usize, h as usize);
    dec.encode_decode(
      w,
      h,
      speed,
      q,
      limit,
      8,
      Default::default(),
      *keyint,
      *keyint,
      0,
      false,
      true,
      0,
      0,
      0,
      false,
    );
  }
}

#[cfg_attr(feature = "decode_test", interpolate_test(aom, "aom"))]
#[cfg_attr(feature = "decode_test_dav1d", interpolate_test(dav1d, "dav1d"))]
fn switch_frame(decoder: &str) {
  let limit = 3;
  let w = 64;
  let h = 80;
  let speed = 10;
  let q = 100;
  let keyint = 12;

  let mut dec = get_decoder::<u8>(decoder, w as usize, h as usize);
  dec.encode_decode(
    w,
    h,
    speed,
    q,
    limit,
    8,
    Default::default(),
    keyint,
    keyint,
    1,
    true,
    true,
    0,
    0,
    0,
    false,
  );
}

#[cfg_attr(feature = "decode_test", interpolate_test(aom, "aom"))]
#[cfg_attr(feature = "decode_test_dav1d", interpolate_test(dav1d, "dav1d"))]
#[ignore]
fn odd_size_frame_with_full_rdo(decoder: &str) {
  let limit = 3;
  let w = 512 + 32 + 16 + 5;
  let h = 512 + 16 + 5;
  let speed = 0;
  let qindex = 100;

  let mut dec = get_decoder::<u8>(decoder, w as usize, h as usize);
  dec.encode_decode(
    w,
    h,
    speed,
    qindex,
    limit,
    8,
    Default::default(),
    15,
    15,
    0,
    true,
    false,
    0,
    0,
    0,
    false,
  );
}

#[cfg_attr(feature = "decode_test", interpolate_test(aom, "aom"))]
#[cfg_attr(feature = "decode_test_dav1d", interpolate_test(dav1d, "dav1d"))]
#[ignore]
fn low_bit_depth(decoder: &str) {
  let quantizer = 100;
  let limit = 3; // Include inter frames
  let speed = 0; // Test as many tools as possible
  let w = 64;
  let h = 80;

  // 8-bit
  let mut dec = get_decoder::<u8>(decoder, w as usize, h as usize);
  dec.encode_decode(
    w,
    h,
    speed,
    quantizer,
    limit,
    8,
    Default::default(),
    15,
    15,
    0,
    true,
    false,
    0,
    0,
    0,
    false,
  );
}

fn high_bit_depth(decoder: &str, depth: usize) {
  let quantizer = 100;
  let limit = 3; // Include inter frames
  let speed = 0; // Test as many tools as possible
  let w = 64;
  let h = 80;

  let mut dec = get_decoder::<u16>(decoder, w as usize, h as usize);
  dec.encode_decode(
    w,
    h,
    speed,
    quantizer,
    limit,
    depth,
    Default::default(),
    15,
    15,
    0,
    true,
    false,
    0,
    0,
    0,
    false,
  );
}

macro_rules! test_high_bit_depth {
  ($($B:expr),+) => {
    $(
      paste::item_with_macros!{
        #[cfg_attr(feature = "decode_test", interpolate_test(aom, "aom"))]
        #[cfg_attr(feature = "decode_test_dav1d", interpolate_test(dav1d, "dav1d"))]
        #[ignore]
        fn [<high_bit_depth_ $B>](decoder: &str) {
          high_bit_depth(decoder, $B);
        }
      }
    )*
  }
}

test_high_bit_depth! {10, 12}

fn chroma_sampling(decoder: &str, cs: ChromaSampling) {
  let quantizer = 100;
  let limit = 3; // Include inter frames
  let speed = 0; // Test as many tools as possible
  let w = 64;
  let h = 80;

  let mut dec = get_decoder::<u8>(decoder, w as usize, h as usize);
  dec.encode_decode(
    w, h, speed, quantizer, limit, 8, cs, 15, 15, 0, true, false, 0, 0, 0,
    false,
  );
}

macro_rules! test_chroma_sampling {
  ($(($S:expr, $I:expr)),+) => {
    $(
      paste::item_with_macros!{
        #[cfg_attr(feature = "decode_test", interpolate_test(aom, "aom"))]
        #[cfg_attr(feature = "decode_test_dav1d", interpolate_test(dav1d, "dav1d"))]
        #[ignore]
        fn [<chroma_sampling_ $S>](decoder: &str) {
          chroma_sampling(decoder, $I);
        }
      }
    )*
  }
}

test_chroma_sampling! {(400, ChromaSampling::Cs400), (420, ChromaSampling::Cs420),
(422, ChromaSampling::Cs422), (444, ChromaSampling::Cs444)}

#[cfg_attr(feature = "decode_test", interpolate_test(aom, "aom"))]
#[cfg_attr(feature = "decode_test_dav1d", interpolate_test(dav1d, "dav1d"))]
fn tile_encoding_with_stretched_restoration_units(decoder: &str) {
  let limit = 5;
  let w = 256;
  // the bottom tiles are small (their height is 140-128=12), so they will use stretched
  // restoration units from their above neighbours
  let h = 140;
  let speed = 10;
  let q = 100;

  let mut dec = get_decoder::<u8>(decoder, w as usize, h as usize);
  dec.encode_decode(
    w,
    h,
    speed,
    q,
    limit,
    8,
    Default::default(),
    15,
    15,
    0,
    true,
    false,
    0,
    2,
    2,
    false,
  );
}

#[cfg_attr(feature = "decode_test", interpolate_test(aom, "aom"))]
#[cfg_attr(feature = "decode_test_dav1d", interpolate_test(dav1d, "dav1d"))]
fn still_picture_mode(decoder: &str) {
  let limit = 1;
  let w = 480;
  let h = 304;
  let speed = 6;
  let qindex = 100;

  let mut dec = get_decoder::<u8>(decoder, w as usize, h as usize);
  dec.encode_decode(
    w,
    h,
    speed,
    qindex,
    limit,
    8,
    Default::default(),
    0,
    0,
    0,
    false,
    false,
    0,
    0,
    0,
    true,
  );
}

pub(crate) fn get_decoder<T: Pixel>(
  decoder: &str, w: usize, h: usize,
) -> Box<dyn TestDecoder<T>> {
  match decoder {
    #[cfg(feature = "decode_test")]
    "aom" => Box::new(AomDecoder::<T>::setup_decoder(w, h)),
    #[cfg(feature = "decode_test_dav1d")]
    "dav1d" => Box::new(Dav1dDecoder::<T>::setup_decoder(w, h)),
    _ => unimplemented!(),
  }
}

#[cfg_attr(feature = "decode_test", interpolate_test(aom, "aom"))]
#[cfg_attr(feature = "decode_test_dav1d", interpolate_test(dav1d, "dav1d"))]
#[ignore]
fn rdo_loop_decision_lrf_sanity(decoder: &str) {
  let limit = 2;
  let w = 936;
  let h = 1404;
  let speed = 9;
  let q = 240;

  let mut dec = get_decoder::<u8>(decoder, w as usize, h as usize);
  dec.encode_decode(
    w,
    h,
    speed,
    q,
    limit,
    8,
    Default::default(),
    6,
    6,
    0,
    true,
    false,
    0,
    0,
    0,
    false,
  );
}

#[cfg_attr(feature = "decode_test", interpolate_test(aom, "aom"))]
#[cfg_attr(feature = "decode_test_dav1d", interpolate_test(dav1d, "dav1d"))]
#[ignore]
fn rdo_loop_decision_cdef_sanity(decoder: &str) {
  let limit = 2;
  let w = 1404;
  let h = 936;
  let speed = 9;
  let q = 240;

  let mut dec = get_decoder::<u8>(decoder, w as usize, h as usize);
  dec.encode_decode(
    w,
    h,
    speed,
    q,
    limit,
    8,
    Default::default(),
    6,
    6,
    0,
    true,
    false,
    0,
    0,
    0,
    false,
  );
}
