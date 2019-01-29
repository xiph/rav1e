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
use std::{mem, ptr, slice};
use std::collections::VecDeque;
use std::sync::Arc;

use dav1d_sys::*;

fn fill_frame(ra: &mut ChaChaRng, frame: &mut Frame) {
  for plane in frame.planes.iter_mut() {
    let stride = plane.cfg.stride;
    for row in plane.data.chunks_mut(stride) {
      for mut pixel in row {
        let v: u8 = ra.gen();
        *pixel = v as u16;
      }
    }
  }
}

struct Decoder {
  dec: *mut Dav1dContext
}

fn setup_decoder() -> Decoder {
  unsafe {
    let mut settings = mem::uninitialized();
    let mut dec: Decoder = mem::uninitialized();

    dav1d_default_settings(&mut settings);

    let ret = dav1d_open(&mut dec.dec, &settings);

    if ret != 0 {
      panic!("Cannot instantiate the decoder {}", ret);
    }

    dec
  }
}

impl Drop for Decoder {
  fn drop(&mut self) {
    unsafe { dav1d_close(&mut self.dec) };
  }
}

fn setup_encoder(
  w: usize, h: usize, speed: usize, quantizer: usize, bit_depth: usize,
  chroma_sampling: ChromaSampling, min_keyint: u64, max_keyint: u64,
  low_latency: bool
) -> Context {

  let mut enc = EncoderConfig::with_speed_preset(speed);
  enc.quantizer = quantizer;
  enc.min_key_frame_interval = min_keyint;
  enc.max_key_frame_interval = max_keyint;
  enc.low_latency = low_latency;

  let cfg = Config {
    frame_info: FrameInfo {
      width: w,
      height: h,
      bit_depth,
      chroma_sampling,
      ..Default::default()
    },
    timebase: Rational::new(1, 1000),
    enc
  };

  cfg.new_context()
}

// TODO: support non-multiple-of-16 dimensions
static DIMENSION_OFFSETS: &[(usize, usize)] =
  &[(0, 0), (4, 4), (8, 8), (16, 16)];

fn speed(s: usize) {
  let quantizer = 100;
  let limit = 5;
  let w = 64;
  let h = 80;

  for b in DIMENSION_OFFSETS.iter() {
      encode_decode(w + b.0, h + b.1, s, quantizer, limit, 8, 15, 15, true);
  }
}

macro_rules! test_speeds {
  ($($S:expr),+) => {
    $(
        paste::item!{
            #[test]
            #[ignore]
            fn [<speed_ $S>]() {
                speed($S)
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
            #[test]
            fn [<dimension_ $W x $H>]() {
                dimension($W, $H)
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

fn dimension(w: usize, h: usize) {
  let quantizer = 100;
  let limit = 1;
  let speed = 4;

  encode_decode(w, h, speed, quantizer, limit, 8, 15, 15, true);
}

#[test]
fn quantizer() {
  let limit = 5;
  let w = 64;
  let h = 80;
  let speed = 4;

  for b in DIMENSION_OFFSETS.iter() {
    for &q in [80, 100, 120].iter() {
      encode_decode(w + b.0, h + b.1, speed, q, limit, 8, 15, 15, true);
    }
  }
}

#[test]
fn keyframes() {
  let limit = 12;
  let w = 64;
  let h = 80;
  let speed = 10;
  let q = 100;

  encode_decode(w, h, speed, q, limit, 8, 6, 6, true);
}

#[test]
fn reordering() {
  let limit = 12;
  let w = 64;
  let h = 80;
  let speed = 10;
  let q = 100;

  for keyint in &[4, 5, 6] {
    encode_decode(w, h, speed, q, limit, 8, *keyint, *keyint, false);
  }
}

#[test]
#[ignore]
fn odd_size_frame_with_full_rdo() {
  let limit = 3;
  let w = 512 + 32 + 16 + 5;
  let h = 512 + 16 + 5;
  let speed = 0;
  let qindex = 100;

  encode_decode(w, h, speed, qindex, limit, 8, 15, 15, true);
}

fn high_bd(bits: usize) {
  let quantizer = 100;
  let limit = 3; // Include inter frames
  let speed = 0; // Test as many tools as possible
  let w = 64;
  let h = 80;

  encode_decode(w, h, speed, quantizer, limit, bits, 15, 15, true);
}

#[test]
fn high_bd_10() {
  high_bd(10);
}

#[test]
fn high_bd_12() {
  high_bd(12);
}

fn compare_plane<T: Ord + std::fmt::Debug>(
  rec: &[T], rec_stride: usize, dec: &[T], dec_stride: usize, width: usize,
  height: usize
) {
  for line in rec.chunks(rec_stride).zip(dec.chunks(dec_stride)).take(height) {
    assert_eq!(&line.0[..width], &line.1[..width]);
  }
}

fn compare_pic(pic: &Dav1dPicture, frame: &Frame, bit_depth: usize, width: usize, height: usize) {
  use plane::Plane;

  let cmp_plane = |data, stride, frame_plane: &Plane| {
    let w = width >> frame_plane.cfg.xdec;
    let h = height >> frame_plane.cfg.ydec;
    let rec_stride = frame_plane.cfg.stride;

    if bit_depth > 8 {
      let stride = stride / 2;
      let dec = unsafe {
        let data = data as *const u16;
        let size = stride * h;

        slice::from_raw_parts(data, size)
      };

      let rec: Vec<u16> =
        frame_plane.data_origin().iter().map(|&v| v).collect();

      compare_plane::<u16>(&rec[..], rec_stride, dec, stride, w, h);
    } else {
      let dec = unsafe {
        let data = data as *const u8;
        let size = stride * h;

        slice::from_raw_parts(data, size)
      };

      let rec: Vec<u8> =
        frame_plane.data_origin().iter().map(|&v| v as u8).collect();

      compare_plane::<u8>(&rec[..], rec_stride, dec, stride, w, h);
    }
  };

  let lstride = pic.stride[0] as usize;
  let cstride = pic.stride[1] as usize;

  cmp_plane(pic.data[0], lstride, &frame.planes[0]);
  cmp_plane(pic.data[1], cstride, &frame.planes[1]);
  cmp_plane(pic.data[2], cstride, &frame.planes[2]);
}

fn encode_decode(
  w: usize, h: usize, speed: usize, quantizer: usize, limit: usize,
  bit_depth: usize, min_keyint: u64, max_keyint: u64, low_latency: bool
) {
  let mut ra = ChaChaRng::from_seed([0; 32]);

  let dec = setup_decoder();
  let mut ctx =
    setup_encoder(w, h, speed, quantizer, bit_depth, ChromaSampling::Cs420,
                  min_keyint, max_keyint, low_latency);

  println!("Encoding {}x{} speed {} quantizer {}", w, h, speed, quantizer);

  let mut rec_fifo = VecDeque::new();

  for _ in 0..limit {
    let mut input = ctx.new_frame();
    fill_frame(&mut ra, Arc::get_mut(&mut input).unwrap());

    let _ = ctx.send_frame(input);

    let mut done = false;
    let mut corrupted_count = 0;
    while !done {
      let res = ctx.receive_packet();
      if let Ok(pkt) = res {
        println!("Encoded packet {}", pkt.number);

        if let Some(pkt_rec) = pkt.rec {
          rec_fifo.push_back(pkt_rec.clone());
        }

        let packet = pkt.data;

        unsafe {
          let mut data: Dav1dData = mem::zeroed();
          println!("Decoding frame {}", pkt.number);
          let mut ptr = dav1d_data_create(&mut data, packet.len());
          ptr::copy_nonoverlapping(packet.as_ptr(), ptr, packet.len());
          let ret = dav1d_send_data(
            dec.dec, &mut data
          );
          println!("Decoded. -> {}", ret);
          if ret != 0 { /*
            let error_msg = aom_codec_error(&mut dec.dec);
            println!(
              "  Decode codec_decode failed: {}",
              CStr::from_ptr(error_msg).to_string_lossy()
            );
            let detail = aom_codec_error_detail(&mut dec.dec);
            if !detail.is_null() {
              println!(
                "  Decode codec_decode failed {}",
                CStr::from_ptr(detail).to_string_lossy()
              );
            } */

            corrupted_count += 1;
          }

          if ret == 0 {
            loop {
              let mut pic: Dav1dPicture = mem::zeroed();
              println!("Retrieving frame");
              let ret = dav1d_get_picture(dec.dec, &mut pic);
              println!("Retrieved.");
              if ret == -(EAGAIN as i32) {
                break;
              }
              if ret != 0 {
                /* let detail = aom_codec_error_detail(&mut dec.dec);
                panic!(
                  "Decode codec_control failed {}",
                  CStr::from_ptr(detail).to_string_lossy()
                );
                */
                panic!("Decode fail");
              }

              let rec = rec_fifo.pop_front().unwrap();
              compare_pic(&pic, &rec, bit_depth, w, h);
            }
          }
        }
      } else {
        done = true;
      }
    }
    assert_eq!(corrupted_count, 0);
  }
}
