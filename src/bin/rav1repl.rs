extern crate rustyline;
extern crate y4m;

extern crate rav1e;
use rav1e::*;

use rustyline::error::ReadlineError;
use rustyline::Editor;

fn main() {
    let mut files = EncoderConfig::from_cli();
    let mut y4m_dec = y4m::decode(&mut files.input_file).unwrap();
    let width = y4m_dec.get_width();
    let height = y4m_dec.get_height();
    let framerate = y4m_dec.get_framerate();
    let mut y4m_enc = match files.rec_file.as_mut() {
        Some(rec_file) => Some(y4m::encode(width, height, framerate).write_header(rec_file).unwrap()),
        None => None
    };
    let mut fi = FrameInvariants::new(width, height, files.quantizer);
    let sequence = Sequence::new();
    write_ivf_header(&mut files.output_file, fi.sb_width*64, fi.sb_height*64, framerate.num, framerate.den);

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
                        process_frame(&sequence, &fi, &mut files.output_file, &mut y4m_dec, y4m_enc.as_mut(), &mut last_rec);
                        fi.number += 1;
                        if fi.number == files.limit {
                            break;
                        }
                    },
                    Some("quit") => break,
                    Some("exit") => break,
                    Some(cmd) => {
                        println!("Unrecognized command: {:?}", cmd);
                    },
                    None => {}
                }
            },
            Err(ReadlineError::Eof) => break,
            _ => {}
        }
    }
    rl.save_history(".rav1e-history").unwrap();
}
