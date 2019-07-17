// Encode the same tiny blank frame 30 times
use rav1e::config::SpeedSettings;
use rav1e::*;

fn main() {
  let mut cfg = Config::default();

  cfg.enc.width = 64;
  cfg.enc.height = 96;

  cfg.enc.speed_settings = SpeedSettings::from_preset(9);

  let mut ctx: Context<u16> = cfg.new_context().unwrap();

  let f = ctx.new_frame();

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
