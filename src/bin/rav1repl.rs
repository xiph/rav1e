// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

extern crate rustyline;
extern crate y4m;

extern crate rav1e;
use rav1e::*;

use rustyline::error::ReadlineError;
use rustyline::Editor;

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
  let mut fi = FrameInvariants::new(width, height, &config);
  let mut sequence = Sequence::new(width, height);
  write_ivf_header(
    &mut io.output,
    fi.padded_w,
    fi.padded_h,
    framerate.num,
    framerate.den
  );

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
            process_frame(
              &mut sequence,
              &mut fi,
              &mut io.output,
              &mut y4m_dec,
              y4m_enc.as_mut(),
              &mut last_rec
            );
            fi.number += 1;
            if fi.number == config.limit {
              break;
            }
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
