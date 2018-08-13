use y4m;
use rav1e::*;
use std::io::prelude::*;
use std::slice;

/// Encode and write a frame.
pub fn process_frame(sequence: &mut Sequence, fi: &mut FrameInvariants,
                     output_file: &mut Write,
                     y4m_dec: &mut y4m::Decoder<Box<Read>>,
                     y4m_enc: Option<&mut y4m::Encoder<Box<Write>>>) -> bool {
    unsafe {
        av1_rtcd();
        aom_dsp_rtcd();
    }
    let width = fi.width;
    let height = fi.height;
    let y4m_bits = y4m_dec.get_bit_depth();
    let y4m_bytes = y4m_dec.get_bytes_per_sample();
    let csp = y4m_dec.get_colorspace();

    match csp {
        y4m::Colorspace::C420 |
        y4m::Colorspace::C420jpeg |
        y4m::Colorspace::C420paldv |
        y4m::Colorspace::C420mpeg2 |
        y4m::Colorspace::C420p10 |
        y4m::Colorspace::C420p12 => {},
        _ => {
            panic!("Colorspace {:?} is not supported yet.", csp);
        },
    }
    match y4m_dec.read_frame() {
        Ok(y4m_frame) => {
            let y4m_y = y4m_frame.get_y_plane();
            let y4m_u = y4m_frame.get_u_plane();
            let y4m_v = y4m_frame.get_v_plane();
            eprintln!("{}", fi);
            let mut fs = FrameState::new(&fi);
            fs.input.planes[0].copy_from_raw_u8(&y4m_y, width * y4m_bytes, y4m_bytes);
            fs.input.planes[1].copy_from_raw_u8(&y4m_u, width * y4m_bytes / 2, y4m_bytes);
            fs.input.planes[2].copy_from_raw_u8(&y4m_v, width * y4m_bytes / 2, y4m_bytes);

            match y4m_bits {
                8 | 10 | 12 => {},
                _ => panic! ("unknown input bit depth!"),
            }

            let packet = encode_frame(sequence, fi, &mut fs);
            write_ivf_frame(output_file, fi.number, packet.as_ref());
            if let Some(mut y4m_enc) = y4m_enc {
                let pitch_y = if sequence.bit_depth > 8 {
                    width * 2
                } else {
                    width
                };
                let pitch_uv = pitch_y / 2;

                let (mut rec_y, mut rec_u, mut rec_v) = (
                    vec![128u8; pitch_y * height],
                    vec![128u8; pitch_uv * (height / 2)],
                    vec![128u8; pitch_uv * (height / 2)]);

                let (stride_y, stride_u, stride_v) = (
                    fs.rec.planes[0].cfg.stride,
                    fs.rec.planes[1].cfg.stride,
                    fs.rec.planes[2].cfg.stride);

                for (line, line_out) in fs.rec.planes[0].data.chunks(stride_y).zip(rec_y.chunks_mut(pitch_y)) {
                    if sequence.bit_depth > 8 {
                        unsafe {
                            line_out.copy_from_slice(
                                slice::from_raw_parts::<u8>(line.as_ptr() as (*const u8), pitch_y));
                        }
                    } else {
                        line_out.copy_from_slice(
                            &line.iter().map(|&v| v as u8).collect::<Vec<u8>>()[..pitch_y]);
                    }
                }
                for (line, line_out) in fs.rec.planes[1].data.chunks(stride_u).zip(rec_u.chunks_mut(pitch_uv)) {
                    if sequence.bit_depth > 8 {
                        unsafe {
                            line_out.copy_from_slice(
                                slice::from_raw_parts::<u8>(line.as_ptr() as (*const u8), pitch_uv));
                        }
                    } else {
                        line_out.copy_from_slice(
                            &line.iter().map(|&v| v as u8).collect::<Vec<u8>>()[..pitch_uv]);
                    }
                }
                for (line, line_out) in fs.rec.planes[2].data.chunks(stride_v).zip(rec_v.chunks_mut(pitch_uv)) {
                    if sequence.bit_depth > 8 {
                        unsafe {
                            line_out.copy_from_slice(
                                slice::from_raw_parts::<u8>(line.as_ptr() as (*const u8), pitch_uv));
                        }
                    } else {
                        line_out.copy_from_slice(
                            &line.iter().map(|&v| v as u8).collect::<Vec<u8>>()[..pitch_uv]);
                    }
                }

                let rec_frame = y4m::Frame::new([&rec_y, &rec_u, &rec_v], None);
                y4m_enc.write_frame(&rec_frame).unwrap();
            }

            update_rec_buffer(fi, fs);
            true
        },
        _ => false
    }
}
