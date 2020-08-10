// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(non_snake_case)]

use rav1e::prelude::*;
use serde_json;
use v_frame::pixel::ChromaSampling;
use wasm_bindgen::prelude::*;

/// Encoder settings which impact the produced bitstream.
#[wasm_bindgen]
#[derive(Clone, Default)]
pub struct EncoderConfig {
  pub(crate) conf: rav1e::EncoderConfig,
}

#[wasm_bindgen]
impl EncoderConfig {
  #[wasm_bindgen(constructor)]
  pub fn new() -> Self {
    Self::default()
  }

  pub fn debug(&self) -> String {
    format!("{:?}", self.conf)
  }

  pub fn toJSON(&self) -> String {
    match serde_json::to_string(&self.conf) {
      Ok(a) => a,
      Err(e) => panic!(e),
    }
  }

  /// Width and height of the frames in pixels.
  pub fn setDim(&mut self, width: usize, height: usize) -> Self {
    self.conf.height = height;
    self.conf.width = width;
    self.clone()
  }

  /// Video time base.
  pub fn setTimeBase(&mut self, numerator: u64, denominator: u64) -> Self {
    self.conf.time_base = Rational { num: numerator, den: denominator };
    self.clone()
  }

  /// Bit depth.
  pub fn setBitDepth(&mut self, bit_depth: usize) -> Self {
    self.conf.bit_depth = bit_depth;
    self.clone()
  }

  /// Chroma subsampling.
  ///
  /// Pass enum `ChromaSampling` or a number in range 0 - 3:
  /// * `1`: `Cs420` (Both vertically and horizontally subsampled)
  /// * `2`: `Cs422` (Horizontally subsampled)
  /// * `3`: `Cs444` (Not subsampled)
  /// * `4`: `Cs400` (Monochrome)
  pub fn setChromaSampling(
    &mut self, chroma_sampling: ChromaSampling,
  ) -> Self {
    self.conf.chroma_sampling = chroma_sampling;
    self.clone()
  }

  /// Chroma sample position.
  ///
  /// Pass enum `ChromaSamplePosition` or number in range 0 - 2:
  /// * `0`: `Unknown` (The source video transfer function must be signaled outside
  /// the AV1 bitstream)
  /// * `1`: `Vertical` (Horizontally co-located with (0, 0) luma sample, vertically
  /// positioned in the middle between two luma samples)
  /// * `2`: `Collocated` (Co-located with (0, 0) luma sample)
  pub fn setChromaSamplePosition(
    &mut self, chroma_sample_position: ChromaSamplePosition,
  ) -> Self {
    self.conf.chroma_sample_position = chroma_sample_position;
    self.clone()
  }

  /// Pixel value range.
  ///
  /// C.f. VideoFullRangeFlag variable specified in ISO/IEC 23091-4/ITU-T H.273
  ///
  /// Pass enum `PixelRange` or number in range 0 - 1:
  /// * `0`: `Limited` (Studio swing representation)
  /// * `1`: `Full` (Full swing representation)
  pub fn setPixelRange(&mut self, pixel_range: PixelRange) -> Self {
    self.conf.pixel_range = pixel_range;
    self.clone()
  }

  /// Content color description (primaries, transfer characteristics, matrix).
  ///
  /// Please supply following enums as input:
  /// * `color_primaries`: `ColorPrimaries`
  /// * `transfer_characteristics`: `TransferCharacteristics`
  /// * `matrix_coefficients`: `MatrixCoefficients`
  pub fn setColorDescription(
    &mut self, color_primaries: ColorPrimaries,
    transfer_characteristics: TransferCharacteristics,
    matrix_coefficients: MatrixCoefficients,
  ) -> Self {
    // these huge match statements are a workaround, they can be removed as
    // soon https://github.com/rustwasm/wasm-bindgen/issues/2273 gets solved
    let color_primaries = match color_primaries {
      ColorPrimaries::BT709 => rav1e::color::ColorPrimaries::BT2020,
      ColorPrimaries::Unspecified => rav1e::color::ColorPrimaries::Unspecified,
      ColorPrimaries::BT470M => rav1e::color::ColorPrimaries::BT470M,
      ColorPrimaries::BT470BG => rav1e::color::ColorPrimaries::BT470BG,
      ColorPrimaries::BT601 => rav1e::color::ColorPrimaries::BT601,
      ColorPrimaries::SMPTE240 => rav1e::color::ColorPrimaries::SMPTE240,
      ColorPrimaries::GenericFilm => rav1e::color::ColorPrimaries::GenericFilm,
      ColorPrimaries::BT2020 => rav1e::color::ColorPrimaries::BT2020,
      ColorPrimaries::XYZ => rav1e::color::ColorPrimaries::XYZ,
      ColorPrimaries::SMPTE431 => rav1e::color::ColorPrimaries::SMPTE431,
      ColorPrimaries::SMPTE432 => rav1e::color::ColorPrimaries::SMPTE432,
      ColorPrimaries::EBU3213 => rav1e::color::ColorPrimaries::EBU3213,
    };
    let transfer_characteristics = match transfer_characteristics {
      TransferCharacteristics::BT709 => {
        rav1e::color::TransferCharacteristics::BT709
      }
      TransferCharacteristics::Unspecified => {
        rav1e::color::TransferCharacteristics::Unspecified
      }
      TransferCharacteristics::BT470M => {
        rav1e::color::TransferCharacteristics::BT470M
      }
      TransferCharacteristics::BT470BG => {
        rav1e::color::TransferCharacteristics::BT470BG
      }
      TransferCharacteristics::BT601 => {
        rav1e::color::TransferCharacteristics::BT601
      }
      TransferCharacteristics::SMPTE240 => {
        rav1e::color::TransferCharacteristics::SMPTE240
      }
      TransferCharacteristics::Linear => {
        rav1e::color::TransferCharacteristics::Linear
      }
      TransferCharacteristics::Log100 => {
        rav1e::color::TransferCharacteristics::Log100
      }
      TransferCharacteristics::Log100Sqrt10 => {
        rav1e::color::TransferCharacteristics::Log100Sqrt10
      }
      TransferCharacteristics::IEC61966 => {
        rav1e::color::TransferCharacteristics::IEC61966
      }
      TransferCharacteristics::BT1361 => {
        rav1e::color::TransferCharacteristics::BT1361
      }
      TransferCharacteristics::SRGB => {
        rav1e::color::TransferCharacteristics::SRGB
      }
      TransferCharacteristics::BT2020_10Bit => {
        rav1e::color::TransferCharacteristics::BT2020_10Bit
      }
      TransferCharacteristics::BT2020_12Bit => {
        rav1e::color::TransferCharacteristics::BT2020_12Bit
      }
      TransferCharacteristics::SMPTE2084 => {
        rav1e::color::TransferCharacteristics::SMPTE2084
      }
      TransferCharacteristics::SMPTE428 => {
        rav1e::color::TransferCharacteristics::SMPTE428
      }
      TransferCharacteristics::HLG => {
        rav1e::color::TransferCharacteristics::HLG
      }
    };
    let matrix_coefficients = match matrix_coefficients {
      MatrixCoefficients::Identity => {
        rav1e::color::MatrixCoefficients::Identity
      }
      MatrixCoefficients::BT709 => rav1e::color::MatrixCoefficients::BT709,
      MatrixCoefficients::Unspecified => {
        rav1e::color::MatrixCoefficients::Unspecified
      }
      MatrixCoefficients::FCC => rav1e::color::MatrixCoefficients::FCC,
      MatrixCoefficients::BT470BG => rav1e::color::MatrixCoefficients::BT470BG,
      MatrixCoefficients::BT601 => rav1e::color::MatrixCoefficients::BT601,
      MatrixCoefficients::SMPTE240 => {
        rav1e::color::MatrixCoefficients::SMPTE240
      }
      MatrixCoefficients::YCgCo => rav1e::color::MatrixCoefficients::YCgCo,
      MatrixCoefficients::BT2020NCL => {
        rav1e::color::MatrixCoefficients::BT2020NCL
      }
      MatrixCoefficients::BT2020CL => {
        rav1e::color::MatrixCoefficients::BT2020CL
      }
      MatrixCoefficients::SMPTE2085 => {
        rav1e::color::MatrixCoefficients::SMPTE2085
      }
      MatrixCoefficients::ChromatNCL => {
        rav1e::color::MatrixCoefficients::ChromatNCL
      }
      MatrixCoefficients::ChromatCL => {
        rav1e::color::MatrixCoefficients::ChromatCL
      }
      MatrixCoefficients::ICtCp => rav1e::color::MatrixCoefficients::ICtCp,
    };

    self.conf.color_description = Some(ColorDescription {
      color_primaries,
      transfer_characteristics,
      matrix_coefficients,
    });
    self.clone()
  }

  /// High dynamic range mastering display color volume
  ///
  /// As defined by CIE 1931
  ///
  /// * primaries (red, green, blue): `ChromaticityPoint`
  /// * `white_point`: `ChromaticityPoint`,
  /// * `max_luminance`: 24.8 fixed-point maximum luminance in candelas per square meter
  /// * `min_luminance`: 18.14 fixed-point minimum luminance in candelas per square meter
  pub fn setMasteringDisplay(
    &mut self, primaries_red: ChromaticityPoint,
    primaries_green: ChromaticityPoint, primaries_blue: ChromaticityPoint,
    white_point: ChromaticityPoint, max_luminance: u32, min_luminance: u32,
  ) -> Self {
    self.conf.mastering_display = Some(MasteringDisplay {
      primaries: [primaries_red, primaries_green, primaries_blue],
      white_point,
      max_luminance,
      min_luminance,
    });
    self.clone()
  }

  /// HDR content light parameters.
  ///
  /// As defined by CEA-861.3, Appendix A.
  pub fn setContentLight(
    &mut self, max_content_light_level: u16,
    max_frame_average_light_level: u16,
  ) -> Self {
    self.conf.content_light = Some(ContentLight {
      max_content_light_level,
      max_frame_average_light_level,
    });
    self.clone()
  }

  /// Enable signaling timing info in the bitstream.
  pub fn setEnableTimingInfo(&mut self, enable_timing_info: bool) -> Self {
    self.conf.enable_timing_info = enable_timing_info;
    self.clone()
  }

  /// Still picture mode flag.
  pub fn setStillPicture(&mut self, still_picture: bool) -> Self {
    self.conf.still_picture = still_picture;
    self.clone()
  }

  /// Flag to force all frames to be error resilient.
  pub fn setErrorResilient(&mut self, error_resilient: bool) -> Self {
    self.conf.error_resilient = error_resilient;
    self.clone()
  }

  /// Interval between switch frames (0 to disable)
  pub fn setSwitchFrameInterval(
    &mut self, switch_frame_interval: u64,
  ) -> Self {
    self.conf.switch_frame_interval = switch_frame_interval;
    self.clone()
  }

  /// Sets the minimum and maximum keyframe interval, handling special cases as needed.
  pub fn setKeyFrameInterval(
    &mut self, min_interval: u64, max_interval: u64,
  ) -> Self {
    self.conf.set_key_frame_interval(min_interval, max_interval);
    self.clone()
  }

  /// The number of temporal units over which to distribute the reservoir usage.
  pub fn setReservoirFrameDelay(
    &mut self, reservoir_frame_delay: i32,
  ) -> Self {
    self.conf.reservoir_frame_delay = Some(reservoir_frame_delay);
    self.clone()
  }

  /// Flag to enable low latency mode.
  ///
  /// In this mode the frame reordering is disabled.
  pub fn setLowLatency(&mut self, low_latency: bool) -> Self {
    self.conf.low_latency = low_latency;
    self.clone()
  }

  /// The base quantizer to use.
  pub fn setQuantizer(&mut self, quantizer: usize) -> Self {
    self.conf.quantizer = quantizer;
    self.clone()
  }

  /// The minimum allowed base quantizer to use in bitrate mode.
  pub fn setMinWQuantizer(&mut self, min_quantizer: u8) -> Self {
    self.conf.min_quantizer = min_quantizer;
    self.clone()
  }

  /// The target bitrate for the bitrate mode.
  pub fn setBitrate(&mut self, bitrate: i32) -> Self {
    self.conf.bitrate = bitrate;
    self.clone()
  }

  /// Metric to tune the quality for.
  ///
  /// Pass enum `Tune` or number in range 0 - 1:
  /// * `0`: `Psnr`
  /// * `1`: `Psychovisual`
  pub fn setTune(&mut self, tune: Tune) -> Self {
    self.conf.tune = tune;
    self.clone()
  }

  /// Number of tiles horizontally. Must be a power of two.
  ///
  /// Overridden by [`tiles`], if present.
  ///
  /// [`tiles`]: #structfield.tiles
  pub fn setTileCols(&mut self, tile_cols: usize) -> Self {
    self.conf.tile_cols = tile_cols;
    self.clone()
  }

  /// Number of tiles vertically. Must be a power of two.
  ///
  /// Overridden by [`tiles`], if present.
  ///
  /// [`tiles`]: #structfield.tiles
  pub fn setTileRows(&mut self, tile_rows: usize) -> Self {
    self.conf.tile_rows = tile_rows;
    self.clone()
  }

  /// Total number of tiles desired.
  ///
  /// Encoder will try to optimally split to reach this number of tiles,
  /// rounded up. Overrides [`tile_cols`] and [`tile_rows`].
  ///
  /// [`tile_cols`]: #structfield.tile_cols
  /// [`tile_rows`]: #structfield.tile_rows
  pub fn setTiles(&mut self, tiles: usize) -> Self {
    self.conf.tiles = tiles;
    self.clone()
  }

  /// Number of frames to read ahead for the RDO lookahead computation.
  pub fn setRdoLookaheadFrame(&mut self, rdo_lookahead_frames: usize) -> Self {
    self.conf.rdo_lookahead_frames = rdo_lookahead_frames;
    self.clone()
  }

  /// Set the speed setting according to a numeric speed preset.
  ///
  /// The speed settings vary depending on speed value from 0 to 10.
  /// * `10` (fastest): min block size 64x64, reduced TX set, fast deblock, fast scenechange detection.
  /// * `9`: min block size 32x32, reduced TX set, fast deblock.
  /// * `8`: min block size 8x8, reduced TX set, fast deblock.
  /// * `7`: min block size 8x8, reduced TX set.
  /// * `6` (default): min block size 8x8, reduced TX set, complex pred modes for keyframes.
  /// * `5`: min block size 8x8, complex pred modes for keyframes, RDO TX decision.
  /// * `4`: min block size 8x8, complex pred modes for keyframes, RDO TX decision, full SGR search.
  /// * `3`: min block size 8x8, complex pred modes for keyframes, RDO TX decision, include near MVs,
  ///        full SGR search.
  /// * `2`: min block size 4x4, complex pred modes, RDO TX decision, include near MVs,
  ///        full SGR search, coarse directions.
  /// * `1`: min block size 4x4, complex pred modes, RDO TX decision, include near MVs,
  ///        bottom-up encoding, full SGR search.
  /// * `0` (slowest): min block size 4x4, complex pred modes, RDO TX decision, include near MVs,
  ///        bottom-up encoding with non-square partitions everywhere, full SGR search.
  pub fn setSpeed(&mut self, speed: usize) -> Self {
    self.conf.speed_settings = SpeedSettings::from_preset(speed);
    self.clone()
  }
}

/// Supported Color Primaries
///
/// As defined by “Color primaries” section of ISO/IEC 23091-4/ITU-T H.273
#[wasm_bindgen]
pub enum ColorPrimaries {
  /// BT.709
  BT709,
  /// Unspecified, must be signaled or inferred outside of the bitstream
  Unspecified,
  /// BT.470 System M (historical)
  BT470M,
  /// BT.470 System B, G (historical)
  BT470BG,
  /// BT.601-7 525 (SMPTE 170 M)
  BT601,
  /// SMPTE 240M (historical)
  SMPTE240,
  /// Generic film
  GenericFilm,
  /// BT.2020, BT.2100
  BT2020,
  /// SMPTE 248 (CIE 1921 XYZ)
  XYZ,
  /// SMPTE RP 431-2
  SMPTE431,
  /// SMPTE EG 432-1
  SMPTE432,
  /// EBU Tech. 3213-E
  EBU3213,
}

/// Supported Transfer Characteristics
///
/// As defined by “Transfer characteristics” section of ISO/IEC 23091-4/ITU-TH.273.
#[wasm_bindgen]
pub enum TransferCharacteristics {
  /// BT.709
  BT709,
  /// Unspecified, must be signaled or inferred outside of the bitstream
  Unspecified,
  /// BT.470 System M (historical)
  BT470M,
  /// BT.470 System B, G (historical)
  BT470BG,
  /// BT.601-7 525 (SMPTE 170 M)
  BT601,
  /// SMPTE 240 M
  SMPTE240,
  /// Linear
  Linear,
  /// Logarithmic (100:1 range)
  Log100,
  /// Logarithmic ((100 * √10):1 range)
  Log100Sqrt10,
  /// IEC 61966-2-4
  IEC61966,
  /// BT.1361 extended color gamut system (historical)
  BT1361,
  /// sRGB or sYCC
  SRGB,
  /// BT.2020 10-bit systems
  BT2020_10Bit,
  /// BT.2020 12-bit systems
  BT2020_12Bit,
  /// SMPTE ST 2084, ITU BT.2100 PQ
  SMPTE2084,
  /// SMPTE ST 428
  SMPTE428,
  /// BT.2100 HLG (Hybrid Log Gamma), ARIB STD-B67
  HLG,
}
/// Matrix coefficients
///
/// As defined by the “Matrix coefficients” section of ISO/IEC 23091-4/ITU-TH.273.
#[wasm_bindgen]
pub enum MatrixCoefficients {
  /// Identity matrix
  Identity,
  /// BT.709
  BT709,
  /// Unspecified, must be signaled or inferred outside of the bitstream.
  Unspecified,
  /// US FCC 73.628
  FCC,
  /// BT.470 System B, G (historical)
  BT470BG,
  /// BT.601-7 525 (SMPTE 170 M)
  BT601,
  /// SMPTE 240 M
  SMPTE240,
  /// YCgCo
  YCgCo,
  /// BT.2020 non-constant luminance, BT.2100 YCbCr
  BT2020NCL,
  /// BT.2020 constant luminance
  BT2020CL,
  /// SMPTE ST 2085 YDzDx
  SMPTE2085,
  /// Chromaticity-derived non-constant luminance
  ChromatNCL,
  /// Chromaticity-derived constant luminance
  ChromatCL,
  /// BT.2020 ICtCp
  ICtCp,
}
