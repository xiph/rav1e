// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use rav1e::config::SpeedSettings;
use rav1e::prelude::*;

/// Encode the same tiny blank frame 30 times
pub fn simple_encoding(repeats: u32) {
  let mut enc = EncoderConfig::default();
  enc.width = 64;
  enc.height = 96;
  enc.speed_settings = SpeedSettings::from_preset(9);

  let cfg = Config::new().with_encoder_config(enc);
  let mut ctx: Context<u16> = cfg.new_context().unwrap();

  let f = ctx.new_frame();

  for i in 0..repeats {
    match ctx.send_frame(f.clone()) {
      Ok(_) => {}
      Err(e) => match e {
        EncoderStatus::EnoughData => {}
        _ => {
          panic!("Unable to send frame {}", i);
        }
      },
    }
  }
  ctx.flush();

  loop {
    match ctx.receive_packet() {
      Ok(_packet) => {}
      Err(EncoderStatus::Encoded) => {}
      Err(EncoderStatus::LimitReached) => break,
      Err(e) => panic!(e),
    }
  }
}
