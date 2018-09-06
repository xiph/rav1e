#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

include!(concat!(env!("OUT_DIR"), "/aom.rs"));

use super::*;

use rand::{ChaChaRng, Rng, SeedableRng};
use std::mem;
use std::collections::VecDeque;

    fn fill_frame(ra: &mut ChaChaRng, frame: &mut Frame) {
        for plane in frame.planes.iter_mut() {
            let stride = plane.cfg.stride;
            for row in plane.data.chunks_mut(stride) {
                for mut pixel in row {
                    let v: u8 = ra.gen();
                    *pixel = v as u16;
                }
            }
        }
    }

    struct AomDecoder {
        dec: aom_codec_ctx,
    }

    fn setup_decoder(w: usize, h: usize) -> AomDecoder {
        unsafe {
            let interface = aom_codec_av1_dx();
            let mut dec: AomDecoder = mem::uninitialized();
            let cfg = aom_codec_dec_cfg_t  {
                threads: 1,
                w: w as u32,
                h: h as u32,
                allow_lowbitdepth: 1,
                cfg: cfg_options { ext_partition: 1 }
            };

            let ret = aom_codec_dec_init_ver(&mut dec.dec, interface, &cfg, 0, AOM_DECODER_ABI_VERSION as i32);
            if ret != 0 {
                panic!("Cannot instantiate the decoder {}", ret);
            }

            dec
        }
    }

    impl Drop for AomDecoder {
        fn drop(&mut self) {
            unsafe { aom_codec_destroy(&mut self.dec) };
        }

    }

    fn setup_encoder(w: usize, h: usize, speed: usize, quantizer: usize,
        bit_depth: usize, chroma_sampling: ChromaSampling) -> (FrameInvariants, Sequence) {
        unsafe {
            av1_rtcd();
            aom_dsp_rtcd();
        }

        let config = EncoderConfig {
            quantizer: quantizer,
            speed: speed,
            ..Default::default()
        };
        let mut fi = FrameInvariants::new(w, h, config);

        fi.use_reduced_tx_set = true;
        // fi.min_partition_size =
        let seq = Sequence::new(w, h, bit_depth, chroma_sampling);

        (fi, seq)
    }

    // TODO: support non-multiple-of-16 dimensions
    static DIMENSION_OFFSETS: &[(usize, usize)] = &[(0, 0), (4, 4), (8, 8), (16, 16)];

    #[test]
    #[ignore]
    fn speed() {
        let quantizer = 100;
        let limit = 5;
        let w = 64;
        let h = 80;

        for b in DIMENSION_OFFSETS.iter() {
            for s in 0 .. 10 {
                encode_decode(w + b.0, h + b.1, s, quantizer, limit, 8);
            }
        }
    }

    static DIMENSIONS: &[(usize, usize)] = &[(8, 8), (16, 16), (32, 32), (64, 64),
        (128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048),
        (258, 258), (260, 260), (262, 262), (264, 264), (265, 265)];

    #[test]
    #[ignore]
    fn dimensions() {
        let quantizer = 100;
        let limit = 1;
        let speed = 4;

        for (w, h) in DIMENSIONS.iter() {
            encode_decode(*w, *h, speed, quantizer, limit, 8);
        }
    }

    #[test]
    #[ignore]
    fn quantizer() {
        let limit = 5;
        let w = 64;
        let h = 80;
        let speed = 4;

        for b in DIMENSION_OFFSETS.iter() {
            for &q in [80, 100, 120].iter() {
                encode_decode(w + b.0, h + b.1, speed, q, limit, 8);
            }
        }
    }

    #[test]
    #[ignore]
    fn odd_size_frame_with_full_rdo() {
        let limit = 3;
        let w = 512 + 32 + 16 + 5;
        let h = 512 + 16 + 5;
        let speed = 0;
        let qindex = 100;

        encode_decode(w, h, speed, qindex, limit, 8);
    }

    #[test]
    #[ignore]
    fn high_bd() {
        let quantizer = 100;
        let limit = 3; // Include inter frames
        let speed = 0; // Test as many tools as possible
        let w = 64;
        let h = 80;

        // 10-bit
        encode_decode(w, h, speed, quantizer, limit, 10);

        // 12-bit
        encode_decode(w, h, speed, quantizer, limit, 12);
    }

    fn compare_plane<T: Ord + std::fmt::Debug>(rec: &[T], rec_stride: usize,
                     dec: &[T], dec_stride: usize,
                     width: usize, height: usize) {
        for line in rec.chunks(rec_stride)
            .zip(dec.chunks(dec_stride)).take(height) {
            assert_eq!(&line.0[..width], &line.1[..width]);
        }
    }

    fn compare_img(img: *const aom_image_t, frame: &Frame, bit_depth: usize) {
        use std::slice;
        let img = unsafe { *img };
        let img_iter = img.planes.iter().zip(img.stride.iter());

        for (img_plane, frame_plane) in img_iter.zip(frame.planes.iter()) {
            let w = frame_plane.cfg.width;
            let h = frame_plane.cfg.height;
            let rec_stride = frame_plane.cfg.stride;

            if bit_depth > 8 {
                let dec_stride = *img_plane.1 as usize / 2;

                let dec = unsafe {
                    let data = *img_plane.0 as *const u16;
                    let size = dec_stride * h;

                    slice::from_raw_parts(data, size)
                };

                let rec: Vec<u16> = frame_plane.data_origin().iter().map(|&v| v).collect();

                compare_plane::<u16>(&rec[..], rec_stride, dec, dec_stride, w, h);
            } else {
                let dec_stride = *img_plane.1 as usize;

                let dec = unsafe {
                    let data = *img_plane.0 as *const u8;
                    let size = dec_stride * h;

                    slice::from_raw_parts(data, size)
                };

                let rec: Vec<u8> = frame_plane.data_origin().iter().map(|&v| v as u8).collect();

                compare_plane::<u8>(&rec[..], rec_stride, dec, dec_stride, w, h);
            }
        }
    }

    fn encode_decode(w: usize, h: usize, speed: usize, quantizer: usize, limit: usize, bit_depth: usize) {
        use std::ptr;
        let mut ra = ChaChaRng::from_seed([0; 32]);

        let mut dec = setup_decoder(w, h);
        let (mut fi, mut seq) = setup_encoder(w, h, speed, quantizer, bit_depth, ChromaSampling::Cs420);

        println!("Encoding {}x{} speed {} quantizer {}", w, h, speed, quantizer);

        let mut iter: aom_codec_iter_t = ptr::null_mut();

        let mut rec_fifo = VecDeque::new();

        for _ in 0 .. limit {
            let mut fs = fi.new_frame_state();
            fill_frame(&mut ra, &mut fs.input);

            fi.frame_type = if fi.number % 30 == 0 { FrameType::KEY } else { FrameType::INTER };
            fi.refresh_frame_flags = if fi.frame_type == FrameType::KEY { ALL_REF_FRAMES_MASK } else { 1 };

            fi.intra_only = fi.frame_type == FrameType::KEY || fi.frame_type == FrameType::INTRA_ONLY;
            fi.use_prev_frame_mvs = !(fi.intra_only || fi.error_resilient);
            println!("Encoding frame {}", fi.number);
            let packet = encode_frame(&mut seq, &mut fi, &mut fs);
            println!("Encoded.");

            fs.rec.pad();

            rec_fifo.push_back(fs.rec.clone());

            update_rec_buffer(&mut fi, fs);

            let mut corrupted_count = 0;
            unsafe {
                println!("Decoding frame {}", fi.number);
                let ret = aom_codec_decode(&mut dec.dec, packet.as_ptr(), packet.len(), ptr::null_mut());
                println!("Decoded. -> {}", ret);
                if ret != 0 {
                    use std::ffi::CStr;
                    let error_msg = aom_codec_error(&mut dec.dec);
                    println!("  Decode codec_decode failed: {}", CStr::from_ptr(error_msg).to_string_lossy());
                    let detail = aom_codec_error_detail(&mut dec.dec);
                    if !detail.is_null() {
                        println!("  Decode codec_decode failed {}", CStr::from_ptr(detail).to_string_lossy());
                    }

                    corrupted_count += 1;
                }

                if ret == 0 {
                    loop {
                        println!("Retrieving frame");
                        let img = aom_codec_get_frame(&mut dec.dec, &mut iter);
                        println!("Retrieved.");
                        if img.is_null() {
                            break;
                        }
                        let mut corrupted = 0;
                        let ret = aom_codec_control_(&mut dec.dec, aom_dec_control_id_AOMD_GET_FRAME_CORRUPTED as i32, &mut corrupted);
                        if ret != 0 {
                            use std::ffi::CStr;
                            let detail = aom_codec_error_detail(&mut dec.dec);
                            panic!("Decode codec_control failed {}", CStr::from_ptr(detail).to_string_lossy());
                        }
                        corrupted_count += corrupted;

                        let rec = rec_fifo.pop_front().unwrap();
                        compare_img(img, &rec, bit_depth);
                    }
                }
            }

            assert_eq!(corrupted_count, 0);

            fi.number += 1;
        }
    }
