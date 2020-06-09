// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

mod utils;

use rav1e::config::SpeedSettings;
use rav1e::prelude::*;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern {
  #[wasm_bindgen(catch)]
  fn alert(s: &str) -> Result<(), JsValue>;
}

/// Encode the same tiny blank frame 30 times
#[wasm_bindgen]
pub fn simple_encoding(repeats: u32) {
  let msg = simple_encoding_without_alert(repeats);

  match alert(&*msg) {
    Ok(_) => {}
    Err(_) => log!("Error calling alert()"),
  };
}

pub fn simple_encoding_without_alert(repeats: u32) -> String {
  // Enable logging for panics
  utils::set_panic_hook();

  let mut enc = EncoderConfig::default();
  enc.width = 64;
  enc.height = 96;
  enc.speed_settings = SpeedSettings::from_preset(9);

  let cfg = Config::new().with_encoder_config(enc);
  let mut ctx: Context<u16> = cfg.new_context().unwrap();
  log!("Instantiated new encoder (config: {:?})", cfg);

  let f = ctx.new_frame();
  log!("Generated new frame: {:?}", f);

  for i in 0..repeats {
    log!("Sending frame {}", i);
    match ctx.send_frame(f.clone()) {
      Ok(_) => {}
      Err(e) => match e {
        EncoderStatus::EnoughData => {
          log!("Warn: Unable to append frame {} to the internal queue", i);
        }
        _ => {
          panic!("Unable to send frame {}", i);
        }
      },
    }
  }
  log!("Successfully send {} frames to the encoder", repeats);

  ctx.flush();
  log!("Flush encoder");

  loop {
    match ctx.receive_packet() {
      Ok(packet) => {
        // Mux the packet.
        log!("Received packet: {}", packet)
      }
      Err(EncoderStatus::Encoded) => {
        // A frame was encoded without emitting a packet. This is normal,
        // just proceed as usual.
      }
      Err(EncoderStatus::LimitReached) => {
        // All frames have been encoded. Time to break out of the loop.
        break;
      }
      Err(EncoderStatus::NeedMoreData) => {
        // The encoder has requested additional frames. Push the next frame
        // in, or flush the encoder if there are no frames left (on None).
        ctx.flush();
      }
      Err(e) => {
        panic!(e);
      }
    }
  }
  format!("Done encoding the same tiny blank frame {} times.", repeats)
}
