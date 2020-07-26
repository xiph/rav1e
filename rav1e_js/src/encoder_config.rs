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
#[derive(Clone)]
pub struct EncoderConfig {
  pub(crate) conf: rav1e::EncoderConfig,
}

#[wasm_bindgen]
impl EncoderConfig {
  #[wasm_bindgen(constructor)]
  pub fn new() -> Self {
    EncoderConfig { conf: rav1e::EncoderConfig::default() }
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

  // /// Content color description (primaries, transfer characteristics, matrix).
  // pub fn setColorDescription(
  //   &mut self, color_description: ColorDescription,
  // ) -> Self {
  //   self.conf.color_description = Some(color_description);
  //   self.clone()
  // }

  // /// High dynamic range mastering display color volume
  // ///
  // /// As defined by CIE 1931
  // pub fn setMasteringDisplay(
  //   &mut self, mastering_display: MasteringDisplay,
  // ) -> Self {
  //   self.conf.mastering_display = Some(mastering_display);
  //   self.clone()
  // }

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
