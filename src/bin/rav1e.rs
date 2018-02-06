extern crate rav1e;
extern crate y4m;

use rav1e::*;

fn main() {
    let mut files = EncoderConfig::from_cli();
    let mut y4m_dec = y4m::decode(&mut files.input_file).unwrap();
    let width = y4m_dec.get_width();
    let height = y4m_dec.get_height();
    let mut y4m_enc = y4m::encode(width,height,y4m::Ratio::new(30,1)).write_header(&mut files.rec_file).unwrap();
    let fi = FrameInvariants::new(width, height);
    let sequence = Sequence::new();
    write_ivf_header(&mut files.output_file, fi.sb_width*64, fi.sb_height*64);

    let mut frame_number = 0;
    loop {
        if !process_frame(frame_number, &sequence, &fi,
                          &mut files.output_file, &mut y4m_dec, &mut y4m_enc) {
            break;
        }
        frame_number += 1;
    }
}
