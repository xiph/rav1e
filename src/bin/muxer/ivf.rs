// Copyright (c) 2001-2016, Alliance for Open Media. All rights reserved
// Copyright (c) 2017-2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use super::Muxer;
use crate::error::*;
use anyhow::Result;
use av_data::packet::ReadPacket;
use av_data::params::{CodecParams, MediaKind, VideoInfo};
use av_data::pixel::{
  formats, ColorModel, FromPrimitive, TrichromaticEncodingSystem, YUVRange,
  YUVSystem,
};
use av_data::rational::Rational64;
use av_format::stream::Stream;
use av_ivf::common::GlobalInfo;
use av_ivf::muxer::{Context, IvfMuxer as AvIvfMuxer};
use rav1e::prelude::*;
use rust_hawktracer::*;
use std::fs::File;
use std::io::{stdout, BufWriter};
use std::io::{Cursor, Write};
use std::path::Path;
use std::sync::Arc;

pub struct IvfMuxer<W: Write + Send> {
  context: Context<AvIvfMuxer, W>,
}

impl<W: Write + Send> Muxer for IvfMuxer<W> {
  fn write_header(
    &mut self, width: usize, height: usize, framerate_num: usize,
    framerate_den: usize, subsampling: (usize, usize), bit_depth: u8,
    primaries: ColorPrimaries, xfer: TransferCharacteristics,
    matrix: MatrixCoefficients, full_range: bool,
  ) -> Result<()> {
    let timebase = Rational64::new(framerate_den as i64, framerate_num as i64);
    self.context.set_global_info(GlobalInfo {
      duration: None,
      timebase: Some(timebase),
      streams: vec![Stream::from_params(
        &CodecParams {
          kind: Some(MediaKind::Video(VideoInfo {
            width,
            height,
            format: Some(Arc::new(
              match (subsampling.0 + subsampling.1, bit_depth) {
                (2, 8) => {
                  let mut fmt = *formats::YUV420;
                  if full_range {
                    fmt.model = ColorModel::Trichromatic(
                      TrichromaticEncodingSystem::YUV(YUVSystem::YCbCr(
                        YUVRange::Full,
                      )),
                    );
                  }
                  fmt.primaries =
                    av_data::pixel::ColorPrimaries::from_u8(primaries as u8)
                      .unwrap();
                  fmt.xfer = av_data::pixel::TransferCharacteristic::from_u8(
                    xfer as u8,
                  )
                  .unwrap();
                  fmt.matrix =
                    av_data::pixel::MatrixCoefficients::from_u8(matrix as u8)
                      .unwrap();
                  fmt
                }
                (1, 8) => {
                  let mut fmt = *formats::YUV422;
                  if full_range {
                    fmt.model = ColorModel::Trichromatic(
                      TrichromaticEncodingSystem::YUV(YUVSystem::YCbCr(
                        YUVRange::Full,
                      )),
                    );
                  }
                  fmt.primaries =
                    av_data::pixel::ColorPrimaries::from_u8(primaries as u8)
                      .unwrap();
                  fmt.xfer = av_data::pixel::TransferCharacteristic::from_u8(
                    xfer as u8,
                  )
                  .unwrap();
                  fmt.matrix =
                    av_data::pixel::MatrixCoefficients::from_u8(matrix as u8)
                      .unwrap();
                  fmt
                }
                (0, 8) => {
                  let mut fmt = *formats::YUV444;
                  if full_range {
                    fmt.model = ColorModel::Trichromatic(
                      TrichromaticEncodingSystem::YUV(YUVSystem::YCbCr(
                        YUVRange::Full,
                      )),
                    );
                  }
                  fmt.primaries =
                    av_data::pixel::ColorPrimaries::from_u8(primaries as u8)
                      .unwrap();
                  fmt.xfer = av_data::pixel::TransferCharacteristic::from_u8(
                    xfer as u8,
                  )
                  .unwrap();
                  fmt.matrix =
                    av_data::pixel::MatrixCoefficients::from_u8(matrix as u8)
                      .unwrap();
                  fmt
                }
                // High bit depth
                (2, bd) => {
                  let mut fmt = *formats::YUV420_10;
                  if full_range {
                    fmt.model = ColorModel::Trichromatic(
                      TrichromaticEncodingSystem::YUV(YUVSystem::YCbCr(
                        YUVRange::Full,
                      )),
                    );
                  }
                  fmt.primaries =
                    av_data::pixel::ColorPrimaries::from_u8(primaries as u8)
                      .unwrap();
                  fmt.xfer = av_data::pixel::TransferCharacteristic::from_u8(
                    xfer as u8,
                  )
                  .unwrap();
                  fmt.matrix =
                    av_data::pixel::MatrixCoefficients::from_u8(matrix as u8)
                      .unwrap();
                  fmt.comp_info[0].as_mut().unwrap().depth = bd;
                  fmt.comp_info[1].as_mut().unwrap().comp_offs = bd;
                  fmt.comp_info[2].as_mut().unwrap().comp_offs = bd;
                  fmt
                }
                (1, bd) => {
                  let mut fmt = *formats::YUV422_10;
                  if full_range {
                    fmt.model = ColorModel::Trichromatic(
                      TrichromaticEncodingSystem::YUV(YUVSystem::YCbCr(
                        YUVRange::Full,
                      )),
                    );
                  }
                  fmt.primaries =
                    av_data::pixel::ColorPrimaries::from_u8(primaries as u8)
                      .unwrap();
                  fmt.xfer = av_data::pixel::TransferCharacteristic::from_u8(
                    xfer as u8,
                  )
                  .unwrap();
                  fmt.matrix =
                    av_data::pixel::MatrixCoefficients::from_u8(matrix as u8)
                      .unwrap();
                  fmt.comp_info[0].as_mut().unwrap().depth = bd;
                  fmt.comp_info[1].as_mut().unwrap().comp_offs = bd;
                  fmt.comp_info[2].as_mut().unwrap().comp_offs = bd;
                  fmt
                }
                (0, bd) => {
                  let mut fmt = *formats::YUV444_10;
                  if full_range {
                    fmt.model = ColorModel::Trichromatic(
                      TrichromaticEncodingSystem::YUV(YUVSystem::YCbCr(
                        YUVRange::Full,
                      )),
                    );
                  }
                  fmt.primaries =
                    av_data::pixel::ColorPrimaries::from_u8(primaries as u8)
                      .unwrap();
                  fmt.xfer = av_data::pixel::TransferCharacteristic::from_u8(
                    xfer as u8,
                  )
                  .unwrap();
                  fmt.matrix =
                    av_data::pixel::MatrixCoefficients::from_u8(matrix as u8)
                      .unwrap();
                  fmt.comp_info[0].as_mut().unwrap().depth = bd;
                  fmt.comp_info[1].as_mut().unwrap().comp_offs = bd;
                  fmt.comp_info[2].as_mut().unwrap().comp_offs = bd;
                  fmt
                }
                _ => unreachable!(),
              },
            )),
          })),
          codec_id: Some("av1".to_string()),
          extradata: None,
          bit_rate: 0,
          convergence_window: 0,
          delay: 0,
        },
        timebase,
      )],
    })?;
    self.context.configure()?;
    self.context.write_header()?;
    Ok(())
  }

  #[hawktracer(write_frame)]
  fn write_frame(
    &mut self, pts: u64, data: &[u8], _frame_type: FrameType,
  ) -> Result<()> {
    let mut packet = Cursor::new(data).get_packet(data.len())?;
    packet.pos = Some(pts as usize);
    self.context.write_packet(Arc::new(packet))?;
    Ok(())
  }

  fn flush(&mut self) -> Result<()> {
    self.context.write_trailer()?;
    Ok(())
  }
}

impl<W: Write + Send> IvfMuxer<W> {
  pub fn open<P: AsRef<Path>>(
    path: P,
  ) -> Result<Box<dyn Muxer + Send>, CliError> {
    Ok(match path.as_ref().to_str() {
      Some("-") => Box::new(IvfMuxer {
        context: Context::new(
          AvIvfMuxer::new(),
          av_ivf::muxer::Writer::new(stdout()),
        ),
      }),
      _ => {
        let file = BufWriter::new(
          File::create(path)
            .map_err(|e| e.context("Cannot open output file"))?,
        );
        Box::new(IvfMuxer {
          context: Context::new(
            AvIvfMuxer::new(),
            av_ivf::muxer::Writer::new(file),
          ),
        })
      }
    })
  }
}
