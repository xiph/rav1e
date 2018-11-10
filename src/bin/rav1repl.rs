// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

extern crate clap;
extern crate rustyline;
extern crate y4m;

extern crate rav1e;

mod common;
use common::*;

use rav1e::*;

use rustyline::error::ReadlineError;
use rustyline::Editor;

fn main() {
  let (mut io, enc, _) = parse_cli();
  let mut y4m_dec = y4m::decode(&mut io.input).unwrap();
  let width = y4m_dec.get_width();
  let height = y4m_dec.get_height();
  let framerate = y4m_dec.get_framerate();
  let color_space = y4m_dec.get_colorspace();

  let mut y4m_enc = match io.rec.as_mut() {
    Some(rec) =>
      Some(y4m::encode(width, height, framerate).write_header(rec).unwrap()),
    None => None
  };
  write_ivf_header(
    &mut io.output,
    width,
    height,
    framerate.num,
    framerate.den
  );

  let chroma_sampling = match color_space {
    y4m::Colorspace::C420
    | y4m::Colorspace::C420jpeg
    | y4m::Colorspace::C420paldv
    | y4m::Colorspace::C420mpeg2
    | y4m::Colorspace::C420p10
    | y4m::Colorspace::C420p12 => ChromaSampling::Cs420,
    y4m::Colorspace::C422
    | y4m::Colorspace::C422p10
    | y4m::Colorspace::C422p12 => ChromaSampling::Cs422,
    y4m::Colorspace::C444
    | y4m::Colorspace::C444p10
    | y4m::Colorspace::C444p12 => ChromaSampling::Cs444,
    _ => panic!("Chroma sampling unknown for the specified color space.")
  };
  let bit_depth = color_space.get_bit_depth();

  let cfg = Config {
    frame_info: FrameInfo { width, height, bit_depth, chroma_sampling },
    timebase: Rational::new(framerate.den as u64, framerate.num as u64),
    enc
  };
  let mut ctx = cfg.new_context();

  let mut rl = Editor::<()>::new();
  let _ = rl.load_history(".rav1e-history");
  let mut last_rec: Option<Frame> = None;
  loop {
    let readline = rl.readline(">> ");
    match readline {
      Ok(line) => {
        rl.add_history_entry(&line);
        match line.split_whitespace().next() {
          Some("process_frame") => {
            match process_frame(&mut ctx, &mut io.output, &mut y4m_dec, y4m_enc.as_mut()) {
              Ok(Some(frame_info)) => frame_info.iter().for_each(|frame| eprintln!("{}", frame)),
              Err(_) => break,
            };

            io.output.flush().unwrap();
          }
          Some("quit") => break,
          Some("exit") => break,
          Some(cmd) => {
            println!("Unrecognized command: {:?}", cmd);
          }
          None => {}
        }
      }
      Err(ReadlineError::Eof) => break,
      _ => {}
    }
  }
  rl.save_history(".rav1e-history").unwrap();
}
