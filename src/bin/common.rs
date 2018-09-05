use y4m;
use rav1e::*;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::slice;
use clap::{App, Arg};

pub struct EncoderIO {
    pub input: Box<dyn Read>,
    pub output: Box<dyn Write>,
    pub rec: Option<Box<dyn Write>>,
}

pub trait FromCli {
    fn from_cli() -> (EncoderIO, EncoderConfig);
}

impl FromCli for EncoderConfig {
    fn from_cli() -> (EncoderIO, EncoderConfig) {
        let matches = App::new("rav1e")
            .version("0.1.0")
            .about("AV1 video encoder")
           .arg(Arg::with_name("INPUT")
                .help("Uncompressed YUV4MPEG2 video input")
                .required(true)
                .index(1))
            .arg(Arg::with_name("OUTPUT")
                .help("Compressed AV1 in IVF video output")
                .short("o")
                .long("output")
                .required(true)
                .takes_value(true))
            .arg(Arg::with_name("RECONSTRUCTION")
                .short("r")
                .takes_value(true))
            .arg(Arg::with_name("LIMIT")
                .help("Maximum number of frames to encode")
                .short("l")
                .long("limit")
                .takes_value(true)
                .default_value("0"))
            .arg(Arg::with_name("QP")
                .help("Quantizer (0-255)")
                .long("quantizer")
                .takes_value(true)
                .default_value("100"))
            .arg(Arg::with_name("SPEED")
                .help("Speed level (0(slow)-10(fast))")
                .short("s")
                .long("speed")
                .takes_value(true)
                .default_value("3"))
            .arg(Arg::with_name("TUNE")
                .help("Quality tuning (Will enforce partition sizes >= 8x8)")
                .long("tune")
                .possible_values(&Tune::variants())
                .default_value("psnr")
                .case_insensitive(true))
            .get_matches();


        let io = EncoderIO {
            input: match matches.value_of("INPUT").unwrap() {
                "-" => Box::new(io::stdin()) as Box<dyn Read>,
                f => Box::new(File::open(&f).unwrap()) as Box<dyn Read>
            },
            output: match matches.value_of("OUTPUT").unwrap() {
                "-" => Box::new(io::stdout()) as Box<dyn Write>,
                f => Box::new(File::create(&f).unwrap()) as Box<dyn Write>
            },
            rec: matches.value_of("RECONSTRUCTION").map(|f| {
                Box::new(File::create(&f).unwrap()) as Box<dyn Write>
            })
        };

        let config = EncoderConfig {
            limit: matches.value_of("LIMIT").unwrap().parse().unwrap(),
            quantizer: matches.value_of("QP").unwrap().parse().unwrap(),
            speed: matches.value_of("SPEED").unwrap().parse().unwrap(),
            tune: matches.value_of("TUNE").unwrap().parse().unwrap()
        };

        // Validate arguments
        if config.quantizer == 0 {
            unimplemented!();
        } else if config.quantizer > 255 || config.speed > 10 {
            panic!("argument out of range");
        }

        (io, config)
    }
}

/// Encode and write a frame.
pub fn process_frame(sequence: &mut Sequence, fi: &mut FrameInvariants,
                     output_file: &mut dyn Write,
                     y4m_dec: &mut y4m::Decoder<'_, Box<dyn Read>>,
                     y4m_enc: Option<&mut y4m::Encoder<'_, Box<dyn Write>>>) -> bool {
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

                for (line, line_out) in fs.rec.planes[0].data_origin().chunks(stride_y).zip(rec_y.chunks_mut(pitch_y)) {
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
                for (line, line_out) in fs.rec.planes[1].data_origin().chunks(stride_u).zip(rec_u.chunks_mut(pitch_uv)) {
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
                for (line, line_out) in fs.rec.planes[2].data_origin().chunks(stride_v).zip(rec_v.chunks_mut(pitch_uv)) {
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

            fs.rec.pad();
            update_rec_buffer(fi, fs);
            true
        },
        _ => false
    }
}
