// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

extern crate rav1e;
extern crate y4m;

use rav1e::*;

fn main() {
  let (mut io, config) = EncoderConfig::from_cli();
  let mut y4m_dec = y4m::decode(&mut io.input).unwrap();
  let width = y4m_dec.get_width();
  let height = y4m_dec.get_height();
  let framerate = y4m_dec.get_framerate();
  let mut y4m_enc = match io.rec.as_mut() {
    Some(rec) =>
      Some(y4m::encode(width, height, framerate).write_header(rec).unwrap()),
    None => None
  };

  let mut fi = FrameInvariants::new(width, height, config);
  let mut sequence = Sequence::new(width, height);
  write_ivf_header(
    &mut io.output,
    width,
    height,
    framerate.num,
    framerate.den
  );

  loop {
    //fi.frame_type = FrameType::KEY;
    fi.frame_type =
      if fi.number % 30 == 0 { FrameType::KEY } else { FrameType::INTER };

    fi.refresh_frame_flags =
      if fi.frame_type == FrameType::KEY { ALL_REF_FRAMES_MASK } else { 1 };
    fi.intra_only = fi.frame_type == FrameType::KEY
      || fi.frame_type == FrameType::INTRA_ONLY;
    fi.use_prev_frame_mvs = !(fi.intra_only || fi.error_resilient);

    if !process_frame(
      &mut sequence,
      &mut fi,
      &mut io.output,
      &mut y4m_dec,
      y4m_enc.as_mut()
    ) {
      break;
    }
    fi.number += 1;
    //fi.show_existing_frame = fi.number % 2 == 1;
    if fi.number == config.limit {
      break;
    }
    io.output.flush().unwrap();
  }
}
