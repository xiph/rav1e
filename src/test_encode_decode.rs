// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use super::*;
use rand::{ChaChaRng, Rng, SeedableRng};
use std::sync::Arc;
use crate::util::Pixel;
#[cfg(feature="decode_test")]
use crate::test_encode_decode_aom::AomDecoder;
#[cfg(feature="decode_test_dav1d")]
use crate::test_encode_decode_dav1d::Dav1dDecoder;
use std::collections::VecDeque;

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

pub(crate) fn read_frame_batch<T: Pixel>(ctx: &mut Context<T>, ra: &mut ChaChaRng) {
  while ctx.needs_more_lookahead() {
    let mut input = ctx.new_frame();
    fill_frame(ra, Arc::get_mut(&mut input).unwrap());

    let _ = ctx.send_frame(Some(input));
  }
  if !ctx.needs_more_frames(ctx.get_frame_count()) {
    ctx.flush();
  }
}

pub(crate) enum DecodeResult {
  Done,
  NotDone,
  Corrupted(usize),
}

pub(crate) trait TestDecoder<T: Pixel> {
  fn setup_decoder(w: usize, h: usize) -> Self where Self: Sized;
  fn encode_decode(
    &mut self, w: usize, h: usize, speed: usize, quantizer: usize,
    limit: usize, bit_depth: usize, chroma_sampling: ChromaSampling,
    min_keyint: u64, max_keyint: u64, low_latency: bool, bitrate: i32
  ) {
    let mut ra = ChaChaRng::from_seed([0; 32]);

    let mut ctx: Context<T> =
      setup_encoder(w, h, speed, quantizer, bit_depth, chroma_sampling,
                    min_keyint, max_keyint, low_latency, bitrate);
    ctx.set_limit(limit as u64);

    println!("Encoding {}x{} speed {} quantizer {} bit-depth {}", w, h, speed, quantizer, bit_depth);
    #[cfg(feature="dump_ivf")]
    let mut out = std::fs::File::create(&format!("out-{}x{}-s{}-q{}-{:?}.ivf",
                                                   w, h, speed, quantizer, chroma_sampling)).unwrap();
    #[cfg(feature="dump_ivf")]
    ivf::write_ivf_header(&mut out, w, h, 30, 1);

    let mut rec_fifo = VecDeque::new();
    for _ in 0..limit {
      read_frame_batch(&mut ctx, &mut ra);

      let mut corrupted_count = 0;
      loop {
        let res = ctx.receive_packet();
        if let Ok(pkt) = res {
          println!("Encoded packet {}", pkt.number);

          #[cfg(feature="dump_ivf")]
          ivf::write_ivf_frame(&mut out, pkt.number, &pkt.data);

          if let Some(pkt_rec) = pkt.rec {
            rec_fifo.push_back(pkt_rec.clone());
          }
          let packet = pkt.data;
          println!("Decoding frame {}", pkt.number);
          match self.decode_packet(&packet, &mut rec_fifo, w, h, bit_depth) {
            DecodeResult::Done => { break; }
            DecodeResult::NotDone => {}
            DecodeResult::Corrupted(corrupted) => { corrupted_count += corrupted; }
          }
        } else {
          break;
        }
      }
      assert_eq!(corrupted_count, 0);
    }
  }
  fn decode_packet(&mut self, packet: &[u8], rec_fifo: &mut VecDeque<Frame<T>>, w: usize, h: usize, bit_depth: usize) -> DecodeResult;
}

pub(crate) fn compare_plane<T: Ord + std::fmt::Debug>(
  rec: &[T], rec_stride: usize, dec: &[T], dec_stride: usize, width: usize,
  height: usize
) {
  for line in rec.chunks(rec_stride).zip(dec.chunks(dec_stride)).take(height) {
    assert_eq!(&line.0[..width], &line.1[..width]);
  }
}

pub(crate) fn setup_encoder<T: Pixel>(
  w: usize, h: usize, speed: usize, quantizer: usize, bit_depth: usize,
  chroma_sampling: ChromaSampling, min_keyint: u64, max_keyint: u64,
  low_latency: bool, bitrate: i32
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

  let cfg = Config {
    enc
  };

  cfg.new_context()
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
      dec.encode_decode(w, h, s, quantizer, limit, 8, Default::default(), 15, 15, true, 0);
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

test_speeds!{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }

macro_rules! test_dimensions {
  ($(($W:expr, $H:expr)),+) => {
    $(
        paste::item!{
            #[cfg_attr(feature = "decode_test", interpolate_test(aom, "aom"))]
            #[cfg_attr(feature = "decode_test_dav1d", interpolate_test(dav1d, "dav1d"))]
            fn [<dimension_ $W x $H>](decoder: &str) {
                dimension($W, $H, decoder)
            }
        }
    )*
  }
}

test_dimensions!{
  (8, 8),
  (16, 16),
  (32, 32),
  (64, 64),
  (128, 128),
  (256, 256),
  (512, 512),
  (1024, 1024),
  (2048, 2048),
  (258, 258),
  (260, 260),
  (262, 262),
  (264, 264),
  (265, 265)
}

fn dimension(w: usize, h: usize, decoder: &str) {
  let quantizer = 100;
  let limit = 1;
  let speed = 10;

  let mut dec = get_decoder::<u8>(decoder, w as usize, h as usize);
  dec.encode_decode(w, h, speed, quantizer, limit, 8, Default::default(), 15, 15, true, 0);
}

#[cfg_attr(feature = "decode_test", interpolate_test(aom, "aom"))]
#[cfg_attr(feature = "decode_test_dav1d", interpolate_test(dav1d, "dav1d"))]
fn quantizer(decoder: &str) {
  let limit = 5;
  let w = 64;
  let h = 80;
  let speed = 10;

  for b in DIMENSION_OFFSETS.iter() {
    for &q in [80, 100, 120].iter() {
      let mut dec = get_decoder::<u8>(decoder, b.0, b.1);
      dec.encode_decode(w + b.0, h + b.1, speed, q, limit, 8, Default::default(), 15, 15, true, 0);
    }
  }
}

#[cfg_attr(feature = "decode_test", interpolate_test(aom, "aom"))]
#[cfg_attr(feature = "decode_test_dav1d", interpolate_test(dav1d, "dav1d"))]
fn bitrate(decoder: &str) {
  let limit = 5;
  let w = 64;
  let h = 80;
  let speed = 10;

  for &q in [172, 220, 252, 255].iter() {
    for &r in [100, 1000, 10_000].iter() {
      let mut dec = get_decoder::<u8>(decoder, w as usize, h as usize);
      dec.encode_decode(w, h, speed, q, limit, 8, Default::default(), 15, 15, true, r);
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
  dec.encode_decode(w, h, speed, q, limit, 8, Default::default(), 6, 6, true, 0);
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
    dec.encode_decode(w, h, speed, q, limit, 8, Default::default(), *keyint, *keyint, false, 0);
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
  dec.encode_decode(w, h, speed, q, limit, 8, Default::default(), keyint, keyint, false, 0);
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
  dec.encode_decode(w, h, speed, qindex, limit, 8, Default::default(), 15, 15, true, 0);
}

#[cfg_attr(feature = "decode_test", interpolate_test(aom, "aom"))]
#[cfg_attr(feature = "decode_test_dav1d", interpolate_test(dav1d, "dav1d"))]
fn all_bit_depths(decoder: &str) {
  let quantizer = 100;
  let limit = 3; // Include inter frames
  let speed = 0; // Test as many tools as possible
  let w = 64;
  let h = 80;

  // 8-bit
  let mut dec = get_decoder::<u8>(decoder, w as usize, h as usize);
  dec.encode_decode(w, h, speed, quantizer, limit, 8, Default::default(), 15, 15, true, 0);

  // 10-bit
  let mut dec = get_decoder::<u16>(decoder, w as usize, h as usize);
  dec.encode_decode(w, h, speed, quantizer, limit, 10, Default::default(), 15, 15, true, 0);

  // 12-bit
  let mut dec = get_decoder::<u16>(decoder, w as usize, h as usize);
  dec.encode_decode(w, h, speed, quantizer, limit, 12, Default::default(), 15, 15, true, 0);
}

#[cfg_attr(feature = "decode_test", interpolate_test(aom, "aom"))]
#[cfg_attr(feature = "decode_test_dav1d", interpolate_test(dav1d, "dav1d"))]
fn chroma_sampling(decoder: &str) {
  let quantizer = 100;
  let limit = 3; // Include inter frames
  let speed = 0; // Test as many tools as possible
  let w = 64;
  let h = 80;

  // TODO: bump keyint when inter is supported

  // 4:2:0
  let mut dec = get_decoder::<u8>(decoder, w as usize, h as usize);
  dec.encode_decode(w, h, speed, quantizer, limit, 8, ChromaSampling::Cs420, 1, 1, true, 0);

  // 4:2:2
  let mut dec = get_decoder::<u8>(decoder, w as usize, h as usize);
  dec.encode_decode(w, h, speed, quantizer, limit, 8, ChromaSampling::Cs422, 1, 1, true, 0);

  // 4:4:4
  let mut dec = get_decoder::<u8>(decoder, w as usize, h as usize);
  dec.encode_decode(w, h, speed, quantizer, limit, 8, ChromaSampling::Cs444, 1, 1, true, 0);
}

fn get_decoder<T: Pixel>(decoder: &str, w: usize, h: usize) -> Box<dyn TestDecoder<T>> {
  match decoder {
    #[cfg(feature="decode_test")]
    "aom" => Box::new(AomDecoder::<T>::setup_decoder(w, h)),
    #[cfg(feature="decode_test_dav1d")]
    "dav1d" => Box::new(Dav1dDecoder::<T>::setup_decoder(w, h)),
    _ => unimplemented!()
  }
}