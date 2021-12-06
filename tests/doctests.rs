use rav1e::prelude::*;

#[test]
fn send_frame() -> Result<(), Box<dyn std::error::Error>> {
  let cfg = Config::default();
  let mut ctx: Context<u8> = cfg.new_context().unwrap();
  let f1 = ctx.new_frame();
  let f2 = f1.clone();
  let info = FrameParameters {
    frame_type_override: FrameTypeOverride::Key,
    opaque: None,
  };

  // Send the plain frame data
  ctx.send_frame(f1)?;
  // Send the data and the per-frame parameters
  // In this case the frame is forced to be a keyframe.
  ctx.send_frame((f2, info))?;
  // Flush the encoder, it is equivalent to a call to `flush()`
  ctx.send_frame(None)?;
  Ok(())
}

#[test]
fn receive_packet() -> Result<(), Box<dyn std::error::Error>> {
  let cfg = Config::default();
  let mut ctx: Context<u8> = cfg.new_context()?;
  let frame = ctx.new_frame();

  ctx.send_frame(frame)?;
  ctx.flush();

  loop {
    match ctx.receive_packet() {
      Ok(_packet) => { /* Mux the packet. */ }
      Err(EncoderStatus::Encoded) => (),
      Err(EncoderStatus::LimitReached) => break,
      Err(err) => Err(err)?,
    }
  }
  Ok(())
}

use std::sync::Arc;

fn encode_frames(
  ctx: &mut Context<u8>, mut frames: impl Iterator<Item = Frame<u8>>,
) -> Result<(), EncoderStatus> {
  // This is a slightly contrived example, intended to showcase the
  // various statuses that can be returned from receive_packet().
  // Assume that, for example, there are a lot of frames in the
  // iterator, which are produced lazily, so you don't want to send
  // them all in at once as to not exhaust the memory.
  loop {
    match ctx.receive_packet() {
      Ok(_packet) => { /* Mux the packet. */ }
      Err(EncoderStatus::Encoded) => {
        // A frame was encoded without emitting a packet. This is
        // normal, just proceed as usual.
      }
      Err(EncoderStatus::LimitReached) => {
        // All frames have been encoded. Time to break out of the
        // loop.
        break;
      }
      Err(EncoderStatus::NeedMoreData) => {
        // The encoder has requested additional frames. Push the
        // next frame in, or flush the encoder if there are no
        // frames left (on None).
        ctx.send_frame(frames.next().map(Arc::new))?;
      }
      Err(EncoderStatus::EnoughData) => {
        // Since we aren't trying to push frames after flushing,
        // this should never happen in this example.
        unreachable!();
      }
      Err(EncoderStatus::NotReady) => {
        // We're not doing two-pass encoding, so this can never
        // occur.
        unreachable!();
      }
      Err(EncoderStatus::Failure) => {
        return Err(EncoderStatus::Failure);
      }
    }
  }

  Ok(())
}

#[test]
fn encoding() -> Result<(), Box<dyn std::error::Error>> {
  let mut enc = EncoderConfig::default();
  // So it runs faster.
  enc.width = 16;
  enc.height = 16;
  let cfg = Config::new().with_encoder_config(enc);
  let mut ctx: Context<u8> = cfg.new_context()?;

  let frames = vec![ctx.new_frame(); 4].into_iter();
  encode_frames(&mut ctx, frames)?;

  Ok(())
}
