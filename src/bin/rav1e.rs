// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

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
        Some(rec_file) => Some(
            y4m::encode(width, height, framerate)
                .write_header(rec_file)
                .unwrap()
        ),
        None => None
    };

    let mut fi =
        FrameInvariants::new(width, height, files.quantizer, files.speed);
    let sequence = Sequence::new();
    write_ivf_header(
        &mut files.output_file,
        width,
        height,
        framerate.num,
        framerate.den
    );

    let mut last_rec: Option<Frame> = None;
    loop {
        //fi.frame_type = FrameType::KEY;
        fi.frame_type = if fi.number % 30 == 0 {
            FrameType::KEY
        } else {
            FrameType::INTER
        };

        fi.intra_only = fi.frame_type == FrameType::KEY
            || fi.frame_type == FrameType::INTRA_ONLY;
        fi.use_prev_frame_mvs = !(fi.intra_only || fi.error_resilient);

        if !process_frame(
            &sequence,
            &fi,
            &mut files.output_file,
            &mut y4m_dec,
            y4m_enc.as_mut(),
            &mut last_rec
        ) {
            break;
        }
        fi.number += 1;
        //fi.show_existing_frame = fi.number % 2 == 1;
        if fi.number == files.limit {
            break;
        }
        files.output_file.flush().unwrap();
    }
}
