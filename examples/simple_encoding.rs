// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

// Encode the same tiny blank frame 30 times
use rav1e::config::SpeedSettings;
use rav1e::*;

fn main() {
  let enc = EncoderConfig {
    width: 64,
    height: 96,
    speed_settings: SpeedSettings::from_preset(9),
    ..Default::default()
  };

  let cfg = Config::new().with_encoder_config(enc.clone());

  let mut ctx: Context<u16> = cfg.new_context().unwrap();

  let mut f = ctx.new_frame();

  let pixels = vec![42; enc.width * enc.height];

  for p in &mut f.planes {
    let stride = (enc.width + p.cfg.xdec) >> p.cfg.xdec;
    p.copy_from_raw_u8(&pixels, stride, 1);
  }

  let limit = 30;

  for i in 0..limit {
    println!("Sending frame {}", i);
    match ctx.send_frame(f.clone()) {
      Ok(_) => {}
      Err(e) => match e {
        EncoderStatus::EnoughData => {
          println!("Unable to append frame {} to the internal queue", i);
        }
        _ => {
          panic!("Unable to send frame {}", i);
        }
      },
    }
  }

  ctx.flush();

  // Test that we cleanly exit once we hit the limit
  let mut i = 0;
  while i < limit + 5 {
    match ctx.receive_packet() {
      Ok(pkt) => {
        println!("Packet {}", pkt.input_frameno);
        i += 1;
      }
      Err(e) => match e {
        EncoderStatus::LimitReached => {
          println!("Limit reached");
          break;
        }
        EncoderStatus::Encoded => println!("  Encoded"),
        EncoderStatus::NeedMoreData => println!("  Need more data"),
        _ => {
          panic!("Unable to receive packet {}", i);
        }
      },
    }
  }
}
