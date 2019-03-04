// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.


use super::*;
use rand::{ChaChaRng, SeedableRng};
use std::{mem, ptr, slice};
use std::collections::VecDeque;
use std::ffi::CStr;
use crate::util::Pixel;
use crate::test_encode_decode::{compare_plane, read_frame_batch, setup_encoder, TestDecoder};
use aom_sys::*;

pub(crate) struct AomDecoder {
  dec: aom_codec_ctx
}

impl TestDecoder for AomDecoder {
  fn setup_decoder(w: usize, h: usize) -> Self {
    unsafe {
      let interface = aom_codec_av1_dx();
      let mut dec: AomDecoder = mem::uninitialized();
      let cfg = aom_codec_dec_cfg_t {
        threads: 1,
        w: w as u32,
        h: h as u32,
        allow_lowbitdepth: 1,
        cfg: cfg_options { ext_partition: 1 }
      };

      let ret = aom_codec_dec_init_ver(
        &mut dec.dec,
        interface,
        &cfg,
        0,
        AOM_DECODER_ABI_VERSION as i32
      );
      if ret != 0 {
        panic!("Cannot instantiate the decoder {}", ret);
      }

      dec
    }
  }

  fn encode_decode(
    &mut self, w: usize, h: usize, speed: usize, quantizer: usize,
    limit: usize, bit_depth: usize, chroma_sampling: ChromaSampling,
    min_keyint: u64, max_keyint: u64, low_latency: bool, bitrate: i32
  ) {

  fn encode_decode_inner<T: Pixel>(
    dec: &mut aom_codec_ctx, w: usize, h: usize, speed: usize, quantizer: usize,
    limit: usize, bit_depth: usize, chroma_sampling: ChromaSampling,
    min_keyint: u64, max_keyint: u64, low_latency: bool, bitrate: i32
  ) {
    let mut ra = ChaChaRng::from_seed([0; 32]);

    let mut ctx = setup_encoder::<T>(
        w, h, speed, quantizer, bit_depth, chroma_sampling,
        min_keyint, max_keyint, low_latency, bitrate);
    ctx.set_limit(limit as u64);

    println!("Encoding {}x{} speed {} quantizer {} bit-depth {}", w, h, speed, quantizer, bit_depth);

    let mut iter: aom_codec_iter_t = ptr::null_mut();

    let mut rec_fifo = VecDeque::new();
    for _ in 0..limit {
      read_frame_batch(&mut ctx, &mut ra);

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
            println!("Decoding frame {}", pkt.number);
            let ret = aom_codec_decode(
              dec,
              packet.as_ptr(),
              packet.len(),
              ptr::null_mut()
            );
            println!("Decoded. -> {}", ret);
            if ret != 0 {
              let error_msg = aom_codec_error(dec);
              println!(
                "  Decode codec_decode failed: {}",
                CStr::from_ptr(error_msg).to_string_lossy()
              );
              let detail = aom_codec_error_detail(dec);
              if !detail.is_null() {
                println!(
                  "  Decode codec_decode failed {}",
                  CStr::from_ptr(detail).to_string_lossy()
                );
              }

              corrupted_count += 1;
            }

            if ret == 0 {
              loop {
                println!("Retrieving frame");
                let img = aom_codec_get_frame(dec, &mut iter);
                println!("Retrieved.");
                if img.is_null() {
                  done = true;
                  break;
                }
                let mut corrupted = 0;
                let ret = aom_codec_control_(
                  dec,
                  aom_dec_control_id::AOMD_GET_FRAME_CORRUPTED as i32,
                  &mut corrupted
                );
                if ret != 0 {
                  let detail = aom_codec_error_detail(dec);
                  panic!(
                    "Decode codec_control failed {}",
                    CStr::from_ptr(detail).to_string_lossy()
                  );
                }
                corrupted_count += corrupted;

                let rec = rec_fifo.pop_front().unwrap();
                compare_img(img, &rec, bit_depth, w, h);
              }
            }
          }
        } else {
          done = true;
        }
      }
      assert_eq!(corrupted_count, 0);
    }
  };
    if bit_depth == 8 {
      encode_decode_inner::<u8>(&mut self.dec, w, h, speed, quantizer, limit, bit_depth, chroma_sampling,
                          min_keyint, max_keyint, low_latency, bitrate)
    } else {
      encode_decode_inner::<u16>(&mut self.dec, w, h, speed, quantizer, limit, bit_depth, chroma_sampling,
                           min_keyint, max_keyint, low_latency, bitrate)
    };
  }
}

impl Drop for AomDecoder {
  fn drop(&mut self) {
    unsafe { aom_codec_destroy(&mut self.dec) };
  }
}

fn compare_img<T: Pixel>(img: *const aom_image_t, frame: &Frame<T>, bit_depth: usize, width: usize, height: usize) {
  let img = unsafe { *img };
  let img_iter = img.planes.iter().zip(img.stride.iter());

  for (img_plane, frame_plane) in img_iter.zip(frame.planes.iter()) {
    let w = width >> frame_plane.cfg.xdec;
    let h = height >> frame_plane.cfg.ydec;
    let rec_stride = frame_plane.cfg.stride;

    if bit_depth > 8 {
      let dec_stride = *img_plane.1 as usize / 2;

      let dec = unsafe {
        let data = *img_plane.0 as *const u16;
        let size = dec_stride * h;

        slice::from_raw_parts(data, size)
      };

      let rec: Vec<u16> =
        frame_plane.data_origin().iter().map(|&v| u16::cast_from(v)).collect();

      compare_plane::<u16>(&rec[..], rec_stride, dec, dec_stride, w, h);
    } else {
      let dec_stride = *img_plane.1 as usize;

      let dec = unsafe {
        let data = *img_plane.0 as *const u8;
        let size = dec_stride * h;

        slice::from_raw_parts(data, size)
      };

      let rec: Vec<u8> =
        frame_plane.data_origin().iter().map(|&v| u8::cast_from(v)).collect();

      compare_plane::<u8>(&rec[..], rec_stride, dec, dec_stride, w, h);
    }
  }
}
