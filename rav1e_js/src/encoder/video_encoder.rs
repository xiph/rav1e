// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(non_snake_case)]

use rav1e::prelude::*;
use std::cell::RefCell;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use web_sys::{Event, HtmlVideoElement};

use crate::encoder::Encoder;
use crate::utils::construct_js_err;
use crate::web;
use crate::web::{Canvas, Video};
use crate::{log, EncoderConfig, Frame, Packet};

/// Contains the encoding state
#[wasm_bindgen]
pub struct VideoEncoder {
  ctx: Rc<RefCell<Context<u8>>>,
  canvas: Rc<RefCell<Canvas>>,
}

impl Encoder for VideoEncoder {
  fn ctx<'a>(&'a self) -> Box<dyn Deref<Target = Context<u8>> + 'a> {
    Box::new(self.ctx.borrow())
  }

  fn ctx_mut<'a>(
    &'a mut self,
  ) -> Box<dyn DerefMut<Target = Context<u8>> + 'a> {
    Box::new(self.ctx.borrow_mut())
  }
}

#[wasm_bindgen]
impl VideoEncoder {
  #[wasm_bindgen(constructor)]
  pub fn fromEncoderConfig(
    conf: EncoderConfig,
  ) -> Result<VideoEncoder, JsValue> {
    let cfg = Config::new().with_encoder_config(conf.conf);

    match cfg.new_context() {
      Ok(ctx) => Ok(Self {
        ctx: Rc::new(RefCell::new(ctx)),
        canvas: Rc::new(RefCell::new(Canvas::new(
          conf.conf.width as u32,
          conf.conf.height as u32,
        ))),
      }),
      Err(e) => Err(construct_js_err(e, "Invalid EncoderConfig")),
    }
  }

  pub fn default() -> Result<VideoEncoder, JsValue> {
    Self::fromEncoderConfig(EncoderConfig::new())
  }

  /// Capture data of `HtmlVideoElement`, while it's playing.
  ///
  /// This process is done as soon `HtmlVideoElement.ended === true`.
  pub fn sendVideo(&mut self, video: &HtmlVideoElement) {
    let video = Video::new(video.clone());

    {
      let canvas = Rc::clone(&self.canvas);
      let ctx = Rc::clone(&self.ctx);
      let onplay = Box::new(move |event: Event| {
        let f = Rc::new(RefCell::new(None));
        let g = Rc::clone(&f);

        // cloning is needed, because they will get moved into Closure 'g'
        let ctx_1 = Rc::clone(&ctx);
        let canvas_1 = Rc::clone(&canvas);
        let video_1 = Video::from(event);

        // TODO: add time param to closure?
        *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
          if video_1.paused() {
            return;
          } else if video_1.ended() {
            f.borrow_mut().take();
            return;
          } else if video_1.ready() {
            canvas_1.borrow().draw_video_frame(&video_1.html);
            let frame = canvas_1.borrow().create_frame();
            match ctx_1.borrow_mut().send_frame(frame) {
              Ok(_) => {}
              Err(e) => match e {
                EncoderStatus::EnoughData => log!("{}", e),
                _ => panic!(e),
              },
            }
            log!("send");
          }

          web::request_animation_frame(f.borrow().as_ref().unwrap());
        }) as Box<dyn FnMut()>));
        web::request_animation_frame(g.borrow().as_ref().unwrap());
      });
      video.add_event_listener("play", onplay);
    }
  }

  pub fn debug(&self) -> String {
    self.debug_msg()
  }

  /// Allocates and returns a new frame.
  pub fn newFrame(&self) -> Frame {
    self.new_frame()
  }

  /// Sends the frame for encoding.
  ///
  /// This method adds the frame into the frame queue and runs the first passes of the look-ahead computation.
  pub fn sendFrame(&mut self, frame: &Frame) -> Result<(), JsValue> {
    self.send_frame(frame)
  }

  /// Flushes the encoder.
  ///
  /// Flushing signals the end of the video. After the encoder has been flushed, no additional frames are accepted.
  pub fn flush(&mut self) {
    self.flush_it();
  }

  /// Encodes the next frame and returns the encoded data.
  ///
  /// This method is where the main encoding work is done.
  pub fn receivePacket(&mut self) -> Result<Packet, JsValue> {
    self.receive_packet()
  }
}
