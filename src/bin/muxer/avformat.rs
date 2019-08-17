// Copyright (c) 2017-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#[cfg(feature = "avformat-sys")]
use avformat_sys::*;

#[allow(unused_imports)]
use super::Muxer;
#[allow(unused_imports)]
use rav1e::prelude::*;
#[allow(unused_imports)]
use std::ffi::CString;
#[allow(unused_imports)]
use std::io;
#[allow(unused_imports)]
use std::mem;
#[allow(unused_imports)]
use std::ptr;

#[cfg(feature = "avformat-sys")]
pub struct AvformatMuxer {
  context: *mut AVFormatContext,
  stream_time_base: AVRational, //time base get from container
  time_base: AVRational,        //set by muxer caller
  duration: i64,
}

#[cfg(feature = "avformat-sys")]
impl AvformatMuxer {
  pub fn open(path: &str) -> Box<dyn Muxer> {
    unsafe {
      let mut context = ptr::null_mut();
      let p = CString::new(path).expect("convert to cstring failed");
      match avformat_alloc_output_context2(
        &mut context,
        ptr::null_mut(),
        ptr::null(),
        p.as_ptr() as *const i8,
      ) {
        0 => {
          let mut io = ptr::null_mut();
          let ret = avio_open(
            &mut io,
            p.as_ptr() as *const i8,
            AVIO_FLAG_WRITE as i32,
          );
          if ret < 0 {
            panic!("open ouput file {} failed", path);
          }
          (*context).pb = io;
          let default_time_base = AVRational { num: 1, den: 1 };
          Box::new(AvformatMuxer {
            context,
            stream_time_base: default_time_base,
            time_base: default_time_base,
            duration: 0,
          })
        }
        e => panic!("open ouput failed, error = {}", e),
      }
    }
  }
}

#[cfg(feature = "avformat-sys")]
impl Muxer for AvformatMuxer {
  fn write_header(
    &mut self, width: usize, height: usize, framerate_num: usize,
    framerate_den: usize,
  ) {
    unsafe {
      let stream = avformat_new_stream(self.context, ptr::null_mut());
      if stream.is_null() {
        panic!("new stream failed");
      }
      let param = (*stream).codecpar;
      (*param).codec_type = AVMediaType::AVMEDIA_TYPE_VIDEO;
      (*param).codec_id = AVCodecID::AV_CODEC_ID_AV1;
      (*param).width = width as i32;
      (*param).height = height as i32;
      let ret = avformat_write_header(self.context, ptr::null_mut());
      if ret < 0 {
        panic!("write header failed error = {}", ret);
      }
      self.stream_time_base = (*stream).time_base;

      //set timebase and duration base on fps.
      self.time_base.den = framerate_num as i32;
      self.time_base.num = framerate_den as i32;
      self.duration = av_rescale_q(1, self.time_base, self.stream_time_base);
    }
  }

  fn write_frame(&mut self, pts: u64, data: &[u8], frame_type: FrameType) {
    unsafe {
      let mut pkt: AVPacket = mem::zeroed();
      av_init_packet(&mut pkt);
      pkt.data = data.as_ptr() as *mut _;
      pkt.size = data.len() as i32;
      if frame_type == FrameType::KEY {
        pkt.flags = AV_PKT_FLAG_KEY as i32;
      }

      let pts =
        av_rescale_q(pts as i64, self.time_base, self.stream_time_base);
      pkt.pts = pts;
      //no b frame, pts equals dts
      pkt.dts = pts;
      pkt.duration = self.duration;
      let ret = av_write_frame(self.context, &mut pkt);
      if ret < 0 {
        panic!("write frame failed error = {}", ret);
      }
    }
  }

  fn flush(&mut self) -> io::Result<()> {
    unsafe {
      loop {
        let ret = av_write_frame(self.context, ptr::null_mut());
        if ret < 0 {
          panic!("write frame failed error = {}", ret)
        }
        if ret == 1 {
          break;
        }
      }
    }
    Ok(())
  }
}

#[cfg(feature = "avformat-sys")]
impl Drop for AvformatMuxer {
  fn drop(&mut self) {
    unsafe {
      av_write_trailer(self.context);
      avio_close((*self.context).pb);
      avformat_free_context(self.context);
    }
  }
}
