extern crate rustyline;
extern crate y4m;

extern crate rav1e;
use rav1e::*;

use rustyline::error::ReadlineError;
use rustyline::Editor;

fn main() {
    let mut files = EncoderFiles::from_cli();
    let mut y4m_dec = y4m::decode(&mut files.input_file).unwrap();
    let width = y4m_dec.get_width();
    let height = y4m_dec.get_height();
    let mut y4m_enc = match files.rec_file {
        Some(f) => Some(y4m::encode(width,height,y4m::Ratio::new(30,1)).write_header(&mut f).unwrap()),
        None => None
    };
    let fi = FrameInvariants::new(width, height);
    let sequence = Sequence::new();
    write_ivf_header(&mut files.output_file, fi.sb_width*64, fi.sb_height*64);

    let mut frame_number = 0;
    let mut rl = Editor::<()>::new();
    let _ = rl.load_history(".rav1e-history");
    loop {
        let readline = rl.readline(">> ");
        match readline {
            Ok(line) => {
                rl.add_history_entry(&line);
                match line.split_whitespace().next() {
                    Some("process_frame") => {
                        process_frame(frame_number, &sequence, &fi,
                                      &mut files.output_file, &mut y4m_dec, y4m_enc.as_mut());
                        frame_number += 1;
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
