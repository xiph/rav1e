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

mod y4m;
pub use self::y4m::write_y4m_frame;

use std::io;

pub trait Muxer {
  fn write_header(
    &mut self, width: usize, height: usize, framerate_num: usize,
    framerate_den: usize
  );

  fn write_frame(&mut self, pts: u64, data: &[u8]);

  fn flush(&mut self) -> io::Result<()>;
}

pub fn create_muxer(path: &str) -> Box<dyn Muxer> {
  IvfMuxer::open(path)
}
