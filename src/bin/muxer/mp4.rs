// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use super::Muxer;

#[cfg(feature = "avformat-sys")]
use super::avformat::AvformatMuxer;

pub struct Mp4Muxer {}

#[allow(unused_variables)]
impl Mp4Muxer {
  pub fn open(path: &str) -> Box<dyn Muxer> {
    #[cfg(feature = "avformat-sys")]
    {
      AvformatMuxer::open(path)
    }
    #[cfg(not(feature = "avformat-sys"))]
    {
      panic!("need avformat-sys for .mp4, please build with --features=\"avformat-sys\", or you can use .ivf extension");
    }
  }
}
