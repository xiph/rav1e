// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use super::*;
use std::marker::PhantomData;
use std::{mem, ptr, slice};
use std::collections::VecDeque;
use crate::util::{Pixel, CastFromPrimitive};
use crate::test_encode_decode::{compare_plane, TestDecoder, DecodeResult};

use dav1d_sys::*;

pub(crate) struct Dav1dDecoder<T: Pixel> {
  dec: *mut Dav1dContext,
  pixel: PhantomData<T>
}

impl<T: Pixel> TestDecoder<T> for Dav1dDecoder<T> {
  fn setup_decoder(_w: usize, _h: usize) -> Self {
    unsafe {
      let mut settings = mem::uninitialized();
      let mut dec: Dav1dDecoder<T> = mem::uninitialized();

      dav1d_default_settings(&mut settings);

      let ret = dav1d_open(&mut dec.dec, &settings);

      if ret != 0 {
        panic!("Cannot instantiate the decoder {}", ret);
      }

      dec
    }
  }

  fn decode_packet(&mut self, packet: &[u8], rec_fifo: &mut VecDeque<Frame<T>>, w: usize, h: usize, bit_depth: usize) -> DecodeResult {
    let mut corrupted_count = 0;
    unsafe {
      let mut data: Dav1dData = mem::zeroed();
      let ptr = dav1d_data_create(&mut data, packet.len());
      ptr::copy_nonoverlapping(packet.as_ptr(), ptr, packet.len());
      let ret = dav1d_send_data(
        self.dec, &mut data
      );
      println!("Decoded. -> {}", ret);
      if ret != 0 {
        corrupted_count += 1;
      }

      if ret == 0 {
        loop {
          let mut pic: Dav1dPicture = mem::zeroed();
          println!("Retrieving frame");
          let ret = dav1d_get_picture(self.dec, &mut pic);
          println!("Retrieved.");
          if ret == -(EAGAIN as i32) {
            return DecodeResult::Done;
          }
          if ret != 0 {
            panic!("Decode fail");
          }

          let rec = rec_fifo.pop_front().unwrap();
          compare_pic(&pic, &rec, bit_depth, w, h);
        }
      }
    }
    if corrupted_count > 0 {
      DecodeResult::Corrupted(corrupted_count)
    } else {
      DecodeResult::NotDone
    }
  }
}

impl<T: Pixel> Drop for Dav1dDecoder<T> {
  fn drop(&mut self) {
    unsafe { dav1d_close(&mut self.dec) };
  }
}

fn compare_pic<T: Pixel>(pic: &Dav1dPicture, frame: &Frame<T>, bit_depth: usize, width: usize, height: usize) {
  use plane::Plane;

  let cmp_plane = |data, stride, frame_plane: &Plane<T>| {
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
        frame_plane.data_origin().iter().map(|&v| u16::cast_from(v)).collect();

      compare_plane::<u16>(&rec[..], rec_stride, dec, stride, w, h);
    } else {
      let dec = unsafe {
        let data = data as *const u8;
        let size = stride * h;

        slice::from_raw_parts(data, size)
      };

      let rec: Vec<u8> =
        frame_plane.data_origin().iter().map(|&v| u8::cast_from(v)).collect();

      compare_plane::<u8>(&rec[..], rec_stride, dec, stride, w, h);
    }
  };

  let lstride = pic.stride[0] as usize;
  let cstride = pic.stride[1] as usize;

  cmp_plane(pic.data[0], lstride, &frame.planes[0]);
  cmp_plane(pic.data[1], cstride, &frame.planes[1]);
  cmp_plane(pic.data[2], cstride, &frame.planes[2]);
}
