// Copyright (c) 2017-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

mod ivf;
use self::ivf::IvfMuxer;

mod mp4;
use mp4::Mp4Muxer;

mod y4m;
pub use self::y4m::write_y4m_frame;

#[cfg(feature = "avformat-sys")]
mod avformat;

use std::io;
use std::ffi::OsStr;
use std::path::Path;


pub trait Muxer {
  fn write_header(
    &mut self, width: usize, height: usize, framerate_num: usize,
    framerate_den: usize
  );

  fn write_frame(&mut self, pts: u64, data: &[u8]);

  fn flush(&mut self) -> io::Result<()>;
}

pub fn create_muxer(path: &str) -> Box<dyn Muxer> {
  if path == "-" {
    return IvfMuxer::open(path);
  }

  let ext = Path::new(path)
    .extension()
    .and_then(OsStr::to_str)
    .map(str::to_lowercase)
    .unwrap_or_else(|| "ivf".into());

  match &ext[..] {
    "mp4" => {
      Mp4Muxer::open(path)
    }
    "ivf" => {
      IvfMuxer::open(path)
    }
    _e => {
      panic!("{} is not a supported extension, please change to .ivf", ext);
    }
  }
}
