extern crate rav1e;
extern crate y4m;

use rav1e::*;

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
    let mut fi = FrameInvariants::new(width, height);
    let sequence = Sequence::new();
    write_ivf_header(&mut files.output_file, fi.sb_width*64, fi.sb_height*64, framerate.num, framerate.den);

    let mut last_rec: Option<Frame> = None;
    loop {
        if !process_frame(&sequence, &fi, &mut files.output_file, &mut y4m_dec, y4m_enc.as_mut(), &mut last_rec) {
            break;
        }
        fi.number += 1;
        fi.show_existing_frame = fi.number % 2 == 1;
        if fi.number == files.limit {
            break;
        }
        files.output_file.flush().unwrap();
    }
}
