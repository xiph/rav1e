// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use err_derive::Error;
use itertools::Itertools;
use num_derive::*;

use crate::api::color::*;
use crate::api::Rational;
use crate::api::{Context, ContextInner};
use crate::cpu_features::CpuFeatureLevel;
use crate::encoder::Tune;
use crate::partition::BlockSize;
use crate::serialize::{Deserialize, Serialize};
use crate::tiling::TilingInfo;
use crate::util::Pixel;

use std::fmt;

// We add 1 to rdo_lookahead_frames in a bunch of places.
const MAX_RDO_LOOKAHEAD_FRAMES: usize = usize::max_value() - 1;
// Due to the math in RCState::new() regarding the reservoir frame delay.
const MAX_MAX_KEY_FRAME_INTERVAL: u64 = i32::max_value() as u64 / 3;

/// Encoder settings which impact the produced bitstream.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct EncoderConfig {
  // output size
  /// Width of the frames in pixels.
  pub width: usize,
  /// Height of the frames in pixels.
  pub height: usize,
  /// Video time base.
  pub time_base: Rational,

  // data format and ancillary color information
  /// Bit depth.
  pub bit_depth: usize,
  /// Chroma subsampling.
  pub chroma_sampling: ChromaSampling,
  /// Chroma sample position.
  pub chroma_sample_position: ChromaSamplePosition,
  /// Pixel value range.
  pub pixel_range: PixelRange,
  /// Content color description (primaries, transfer characteristics, matrix).
  pub color_description: Option<ColorDescription>,
  /// HDR mastering display parameters.
  pub mastering_display: Option<MasteringDisplay>,
  /// HDR content light parameters.
  pub content_light: Option<ContentLight>,

  /// Enable signaling timing info in the bitstream.
  pub enable_timing_info: bool,

  /// Still picture mode flag.
  pub still_picture: bool,

  /// Flag to force all frames to be error resilient.
  pub error_resilient: bool,

  /// Interval between switch frames (0 to disable)
  pub switch_frame_interval: u64,

  // encoder configuration
  /// The *minimum* interval between two keyframes
  pub min_key_frame_interval: u64,
  /// The *maximum* interval between two keyframes
  pub max_key_frame_interval: u64,
  /// The number of temporal units over which to distribute the reservoir
  /// usage.
  pub reservoir_frame_delay: Option<i32>,
  /// Flag to enable low latency mode.
  ///
  /// In this mode the frame reordering is disabled.
  pub low_latency: bool,
  /// The base quantizer to use.
  pub quantizer: usize,
  /// The minimum allowed base quantizer to use in bitrate mode.
  pub min_quantizer: u8,
  /// The target bitrate for the bitrate mode.
  pub bitrate: i32,
  /// Metric to tune the quality for.
  pub tune: Tune,
  /// Number of tiles horizontally. Must be a power of two.
  ///
  /// Overridden by [`tiles`], if present.
  ///
  /// [`tiles`]: #structfield.tiles
  pub tile_cols: usize,
  /// Number of tiles vertically. Must be a power of two.
  ///
  /// Overridden by [`tiles`], if present.
  ///
  /// [`tiles`]: #structfield.tiles
  pub tile_rows: usize,
  /// Total number of tiles desired.
  ///
  /// Encoder will try to optimally split to reach this number of tiles,
  /// rounded up. Overrides [`tile_cols`] and [`tile_rows`].
  ///
  /// [`tile_cols`]: #structfield.tile_cols
  /// [`tile_rows`]: #structfield.tile_rows
  pub tiles: usize,
  /// Number of frames to read ahead for the RDO lookahead computation.
  pub rdo_lookahead_frames: usize,
  /// If enabled, computes the PSNR values and stores them in [`Packet`].
  ///
  /// [`Packet`]: struct.Packet.html#structfield.psnr
  pub show_psnr: bool,

  /// Settings which affect the enconding speed vs. quality trade-off.
  pub speed_settings: SpeedSettings,
}

/// Default preset for EncoderConfig: it is a balance between quality and
/// speed. See [`with_speed_preset()`].
///
/// [`with_speed_preset()`]: struct.EncoderConfig.html#method.with_speed_preset
impl Default for EncoderConfig {
  fn default() -> Self {
    const DEFAULT_SPEED: usize = 6;
    Self::with_speed_preset(DEFAULT_SPEED)
  }
}

impl EncoderConfig {
  /// This is a preset which provides default settings according to a speed
  /// value in the specific range 0–10. Each speed value corresponds to a
  /// different preset. See [`from_preset()`]. If the input value is greater
  /// than 10, it will result in the same settings as 10.
  ///
  /// [`from_preset()`]: struct.SpeedSettings.html#method.from_preset
  pub fn with_speed_preset(speed: usize) -> Self {
    EncoderConfig {
      width: 640,
      height: 480,

      bit_depth: 8,
      chroma_sampling: ChromaSampling::Cs420,
      chroma_sample_position: ChromaSamplePosition::Unknown,
      pixel_range: Default::default(),
      color_description: None,
      mastering_display: None,
      content_light: None,

      enable_timing_info: false,

      still_picture: false,

      error_resilient: false,
      switch_frame_interval: 0,

      time_base: Rational { num: 1, den: 30 },

      min_key_frame_interval: 12,
      max_key_frame_interval: 240,
      min_quantizer: 0,
      reservoir_frame_delay: None,
      low_latency: false,
      quantizer: 100,
      bitrate: 0,
      tune: Tune::default(),
      tile_cols: 0,
      tile_rows: 0,
      tiles: 0,
      rdo_lookahead_frames: 40,
      speed_settings: SpeedSettings::from_preset(speed),
      show_psnr: false,
    }
  }

  /// Sets the minimum and maximum keyframe interval, handling special cases as needed.
  pub fn set_key_frame_interval(
    &mut self, min_interval: u64, max_interval: u64,
  ) {
    self.min_key_frame_interval = min_interval;

    // Map an input value of 0 to an infinite interval
    self.max_key_frame_interval = if max_interval == 0 {
      MAX_MAX_KEY_FRAME_INTERVAL
    } else {
      max_interval
    };
  }

  /// Returns the video frame rate computed from [`time_base`].
  ///
  /// [`time_base`]: #structfield.time_base
  pub fn frame_rate(&self) -> f64 {
    Rational::from_reciprocal(self.time_base).as_f64()
  }
}

impl fmt::Display for EncoderConfig {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    let pairs = [
      ("keyint_min", self.min_key_frame_interval.to_string()),
      ("keyint_max", self.max_key_frame_interval.to_string()),
      ("quantizer", self.quantizer.to_string()),
      ("bitrate", self.bitrate.to_string()),
      ("min_quantizer", self.min_quantizer.to_string()),
      ("low_latency", self.low_latency.to_string()),
      ("tune", self.tune.to_string()),
      ("rdo_lookahead_frames", self.rdo_lookahead_frames.to_string()),
      ("min_block_size", self.speed_settings.min_block_size.to_string()),
      (
        "multiref",
        (!self.low_latency || self.speed_settings.multiref).to_string(),
      ),
      ("fast_deblock", self.speed_settings.fast_deblock.to_string()),
      ("reduced_tx_set", self.speed_settings.reduced_tx_set.to_string()),
      (
        "tx_domain_distortion",
        self.speed_settings.tx_domain_distortion.to_string(),
      ),
      ("tx_domain_rate", self.speed_settings.tx_domain_rate.to_string()),
      ("encode_bottomup", self.speed_settings.encode_bottomup.to_string()),
      ("rdo_tx_decision", self.speed_settings.rdo_tx_decision.to_string()),
      ("prediction_modes", self.speed_settings.prediction_modes.to_string()),
      ("include_near_mvs", self.speed_settings.include_near_mvs.to_string()),
      (
        "no_scene_detection",
        self.speed_settings.no_scene_detection.to_string(),
      ),
      ("diamond_me", self.speed_settings.diamond_me.to_string()),
      ("cdef", self.speed_settings.cdef.to_string()),
      ("use_satd_subpel", self.speed_settings.use_satd_subpel.to_string()),
      (
        "non_square_partition",
        self.speed_settings.non_square_partition.to_string(),
      ),
      ("enable_timing_info", self.enable_timing_info.to_string()),
    ];
    write!(
      f,
      "{}",
      pairs.iter().map(|pair| format!("{}={}", pair.0, pair.1)).join(" ")
    )
  }
}

/// Contains the speed settings.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SpeedSettings {
  /// Minimum block size. Smaller is slower.
  ///
  /// Must be a square block size, so e.g. 8×4 isn't allowed here.
  pub min_block_size: BlockSize,
  /// Enables inter-frames to have multiple reference frames.
  ///
  /// Enabled is slower.
  pub multiref: bool,
  /// Enables fast deblocking filter.
  pub fast_deblock: bool,
  /// Enables reduced transform set.
  ///
  /// Enabled is faster.
  pub reduced_tx_set: bool,
  /// Enables using transform-domain distortion instead of pixel-domain.
  ///
  /// Enabled is faster.
  pub tx_domain_distortion: bool,
  /// Enables using transform-domain rate estimation.
  ///
  /// Enabled is faster.
  pub tx_domain_rate: bool,
  /// Enables bottom-up encoding, rather than top-down.
  ///
  /// Enabled is slower.
  pub encode_bottomup: bool,
  /// Enables searching transform size and type with RDO.
  ///
  /// Enabled is slower.
  pub rdo_tx_decision: bool,
  /// Prediction modes to search.
  ///
  /// Complex settings are slower.
  pub prediction_modes: PredictionModesSetting,
  /// Enables searching near motion vectors during RDO.
  ///
  /// Enabled is slower.
  pub include_near_mvs: bool,
  /// Disables scene-cut detection.
  ///
  /// Enabled is faster.
  pub no_scene_detection: bool,
  /// Fast scene detection mode, ignoring chroma planes.
  pub fast_scene_detection: bool,
  /// Enables diamond motion vector search rather than full search.
  pub diamond_me: bool,
  /// Enables CDEF.
  pub cdef: bool,
  /// Enables LRF.
  pub lrf: bool,
  /// The amount of search done for self guided restoration.
  pub sgr_complexity: SGRComplexityLevel,
  /// Use SATD instead of SAD for subpixel search.
  ///
  /// Enabled is slower.
  pub use_satd_subpel: bool,
  /// Use non-square partition type everywhere
  ///
  /// Enabled is slower.
  pub non_square_partition: bool,

  /// Use segmentation.
  pub enable_segmentation: bool,
}

impl Default for SpeedSettings {
  /// This is currently used exclusively for feature testing and comparison.
  /// It is set to the slowest settings possible.
  fn default() -> Self {
    SpeedSettings {
      min_block_size: BlockSize::BLOCK_4X4,
      multiref: true,
      fast_deblock: false,
      reduced_tx_set: false,
      tx_domain_distortion: false,
      tx_domain_rate: false,
      encode_bottomup: true,
      rdo_tx_decision: true,
      prediction_modes: PredictionModesSetting::ComplexAll,
      include_near_mvs: true,
      no_scene_detection: false,
      fast_scene_detection: false,
      diamond_me: true,
      cdef: true,
      lrf: false,
      sgr_complexity: SGRComplexityLevel::Full,
      use_satd_subpel: true,
      non_square_partition: true,
      enable_segmentation: true,
    }
  }
}

impl SpeedSettings {
  /// Set the speed setting according to a numeric speed preset.
  ///
  /// The speed settings vary depending on speed value from 0 to 10.
  /// - 10 (fastest): min block size 64x64, reduced TX set, fast deblock, fast scenechange detection.
  /// - 9: min block size 32x32, reduced TX set, fast deblock.
  /// - 8: min block size 8x8, reduced TX set, fast deblock.
  /// - 7: min block size 8x8, reduced TX set.
  /// - 6 (default): min block size 8x8, reduced TX set, complex pred modes for keyframes.
  /// - 5: min block size 8x8, complex pred modes for keyframes, reduced TX set, RDO TX decision.
  /// - 4: min block size 8x8, complex pred modes for keyframes, RDO TX decision.
  /// - 3: min block size 8x8, complex pred modes for keyframes, RDO TX decision, include near MVs.
  /// - 2: min block size 4x4, complex pred modes, RDO TX decision, include near MVs.
  /// - 1: min block size 4x4, complex pred modes, RDO TX decision, include near MVs, bottom-up encoding.
  /// - 0 (slowest): min block size 4x4, complex pred modes, RDO TX decision, include near MVs, bottom-up encoding with non-square partitions everywhere
  pub fn from_preset(speed: usize) -> Self {
    SpeedSettings {
      min_block_size: Self::min_block_size_preset(speed),
      multiref: Self::multiref_preset(speed),
      fast_deblock: Self::fast_deblock_preset(speed),
      reduced_tx_set: Self::reduced_tx_set_preset(speed),
      tx_domain_distortion: Self::tx_domain_distortion_preset(speed),
      tx_domain_rate: Self::tx_domain_rate_preset(speed),
      encode_bottomup: Self::encode_bottomup_preset(speed),
      rdo_tx_decision: Self::rdo_tx_decision_preset(speed),
      prediction_modes: Self::prediction_modes_preset(speed),
      include_near_mvs: Self::include_near_mvs_preset(speed),
      no_scene_detection: Self::no_scene_detection_preset(speed),
      fast_scene_detection: Self::fast_scene_detection_preset(speed),
      diamond_me: Self::diamond_me_preset(speed),
      cdef: Self::cdef_preset(speed),
      lrf: Self::lrf_preset(speed),
      sgr_complexity: Self::sgr_complexity_preset(speed),
      use_satd_subpel: Self::use_satd_subpel(speed),
      non_square_partition: Self::non_square_partition_preset(speed),
      enable_segmentation: Self::enable_segmentation_preset(speed),
    }
  }

  /// This preset is set this way because 8x8 with reduced TX set is faster but with equivalent
  /// or better quality compared to 16x16 (to which reduced TX set does not apply).
  fn min_block_size_preset(speed: usize) -> BlockSize {
    let min_block_size = if speed <= 2 {
      BlockSize::BLOCK_4X4
    } else if speed <= 8 {
      BlockSize::BLOCK_8X8
    } else if speed <= 9 {
      BlockSize::BLOCK_32X32
    } else {
      BlockSize::BLOCK_64X64
    };

    // Topdown search checks min_block_size for PARTITION_SPLIT only, so min_block_size must be square.
    assert!(min_block_size.is_sqr());
    min_block_size
  }

  /// Multiref is enabled automatically if low_latency is false.
  ///
  /// If low_latency is true, enabling multiref allows using multiple
  /// backwards references. low_latency false enables both forward and
  /// backwards references.
  const fn multiref_preset(speed: usize) -> bool {
    speed <= 7
  }

  const fn fast_deblock_preset(speed: usize) -> bool {
    speed >= 8
  }

  const fn reduced_tx_set_preset(speed: usize) -> bool {
    speed >= 5
  }

  /// TX domain distortion is always faster, with no significant quality change
  const fn tx_domain_distortion_preset(_speed: usize) -> bool {
    true
  }

  const fn tx_domain_rate_preset(_speed: usize) -> bool {
    false
  }

  const fn encode_bottomup_preset(speed: usize) -> bool {
    speed <= 1
  }

  const fn rdo_tx_decision_preset(speed: usize) -> bool {
    speed <= 5
  }

  fn prediction_modes_preset(speed: usize) -> PredictionModesSetting {
    if speed <= 2 {
      PredictionModesSetting::ComplexAll
    } else if speed <= 6 {
      PredictionModesSetting::ComplexKeyframes
    } else {
      PredictionModesSetting::Simple
    }
  }

  const fn include_near_mvs_preset(speed: usize) -> bool {
    speed <= 3
  }

  const fn no_scene_detection_preset(_speed: usize) -> bool {
    false
  }

  const fn fast_scene_detection_preset(speed: usize) -> bool {
    speed == 10
  }

  /// Currently Diamond ME gives better quality than full search on most videos,
  /// in addition to being faster.
  // There are a few outliers, such as the Wikipedia test clip.
  // TODO: Revisit this setting if full search quality improves in the future.
  const fn diamond_me_preset(_speed: usize) -> bool {
    true
  }

  const fn cdef_preset(_speed: usize) -> bool {
    true
  }

  const fn lrf_preset(_speed: usize) -> bool {
    true
  }

  fn sgr_complexity_preset(speed: usize) -> SGRComplexityLevel {
    if speed <= 9 {
      SGRComplexityLevel::Full
    } else {
      SGRComplexityLevel::Reduced
    }
  }

  const fn use_satd_subpel(speed: usize) -> bool {
    speed <= 9
  }

  const fn non_square_partition_preset(speed: usize) -> bool {
    speed == 0
  }

  // FIXME: this is currently only enabled at speed 0 because choosing a segment
  // requires doing RDO, but once that is replaced by a less bruteforce
  // solution we should be able to enable segmentation at all speeds.
  const fn enable_segmentation_preset(speed: usize) -> bool {
    speed == 0
  }
}

/// Prediction modes to search.
#[derive(
  Clone,
  Copy,
  Debug,
  PartialOrd,
  PartialEq,
  FromPrimitive,
  Serialize,
  Deserialize,
)]
pub enum PredictionModesSetting {
  /// Only simple prediction modes.
  Simple,
  /// Search all prediction modes on key frames and simple modes on other
  /// frames.
  ComplexKeyframes,
  /// Search all prediction modes on all frames.
  ComplexAll,
}

impl fmt::Display for PredictionModesSetting {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    write!(
      f,
      "{}",
      match self {
        PredictionModesSetting::Simple => "Simple",
        PredictionModesSetting::ComplexKeyframes => "Complex-KFs",
        PredictionModesSetting::ComplexAll => "Complex-All",
      }
    )
  }
}

/// Search level for self guided restoration
#[derive(
  Clone,
  Copy,
  Debug,
  PartialOrd,
  PartialEq,
  FromPrimitive,
  Serialize,
  Deserialize,
)]
pub enum SGRComplexityLevel {
  /// Search all sgr parameters
  Full,
  /// Search a reduced set of sgr parameters
  Reduced,
}

impl fmt::Display for SGRComplexityLevel {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    write!(
      f,
      "{}",
      match self {
        SGRComplexityLevel::Full => "Full",
        SGRComplexityLevel::Reduced => "Reduced",
      }
    )
  }
}

/// Enumeration of possible invalid configuration errors.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Error)]
pub enum InvalidConfig {
  /// The width is invalid.
  #[error(display = "invalid width {} (expected >= 16, <= 32767)", _0)]
  InvalidWidth(usize),
  /// The height is invalid.
  #[error(display = "invalid height {} (expected >= 16, <= 32767)", _0)]
  InvalidHeight(usize),
  /// RDO lookahead frame count is invalid.
  #[error(
    display = "invalid rdo lookahead frames {} (expected <= {})",
    actual,
    max
  )]
  InvalidRdoLookaheadFrames {
    /// The actual value.
    actual: usize,
    /// The maximal supported value.
    max: usize,
  },
  /// Maximal keyframe interval is invalid.
  #[error(
    display = "invalid max keyframe interval {} (expected <= {})",
    actual,
    max
  )]
  InvalidMaxKeyFrameInterval {
    /// The actual value.
    actual: u64,
    /// The maximal supported value.
    max: u64,
  },
  /// Tile columns is invalid.
  #[error(display = "invalid tile cols {} (expected power of 2)", _0)]
  InvalidTileCols(usize),
  /// Tile rows is invalid.
  #[error(display = "invalid tile rows {} (expected power of 2)", _0)]
  InvalidTileRows(usize),
  /// Framerate numerator is invalid.
  #[error(
    display = "invalid framerate numerator {} (expected > 0, <= {})",
    actual,
    max
  )]
  InvalidFrameRateNum {
    /// The actual value.
    actual: u64,
    /// The maximal supported value.
    max: u64,
  },
  /// Framerate denominator is invalid.
  #[error(
    display = "invalid framerate denominator {} (expected > 0, <= {})",
    actual,
    max
  )]
  InvalidFrameRateDen {
    /// The actual value.
    actual: u64,
    /// The maximal supported value.
    max: u64,
  },
  /// Reservoir frame delay is invalid.
  #[error(
    display = "invalid reservoir frame delay {} (expected >= 12, <= 131072)",
    _0
  )]
  InvalidReservoirFrameDelay(i32),
  /// Reservoir frame delay is invalid.
  #[error(
    display = "invalid switch frame interval {} (must only be used with low latency mode)",
    _0
  )]
  InvalidSwitchFrameInterval(u64),

  // This variant prevents people from exhaustively matching on this enum,
  // which allows us to add more variants without it being a breaking change.
  // This can be replaced with #[non_exhaustive] when it's stable:
  // https://github.com/rust-lang/rust/issues/44109
  #[doc(hidden)]
  #[error(display = "")]
  __NonExhaustive,
}

/// Contains the encoder configuration.
#[derive(Clone, Debug, Default)]
pub struct Config {
  /// Settings which impact the produced bitstream.
  pub enc: EncoderConfig,
  /// The number of threads in the threadpool.
  pub threads: usize,
}

fn check_tile_log2(n: usize) -> bool {
  let tile_log2 = TilingInfo::tile_log2(1, n);
  if tile_log2.is_none() {
    return false;
  }
  let tile_log2 = tile_log2.unwrap();

  ((1 << tile_log2) - n) == 0 || n == 0
}

impl Config {
  /// Creates a [`Context`] with this configuration.
  ///
  /// # Examples
  ///
  /// ```
  /// use rav1e::prelude::*;
  ///
  /// # fn main() -> Result<(), InvalidConfig> {
  /// let cfg = Config::default();
  /// let ctx: Context<u8> = cfg.new_context()?;
  /// # Ok(())
  /// # }
  /// ```
  ///
  /// [`Context`]: struct.Context.html
  pub fn new_context<T: Pixel>(&self) -> Result<Context<T>, InvalidConfig> {
    assert!(
      8 * std::mem::size_of::<T>() >= self.enc.bit_depth,
      "The Pixel u{} does not match the Config bit_depth {}",
      8 * std::mem::size_of::<T>(),
      self.enc.bit_depth
    );

    self.validate()?;

    // Because we don't have a FrameInvariants yet,
    // this is the only way to get the CpuFeatureLevel in use.
    // Since we only call this once, this shouldn't cause
    // performance issues.
    info!("CPU Feature Level: {}", CpuFeatureLevel::default());

    let pool = rayon::ThreadPoolBuilder::new()
      .num_threads(self.threads)
      .build()
      .unwrap();

    let mut config = self.enc;
    config.set_key_frame_interval(
      config.min_key_frame_interval,
      config.max_key_frame_interval,
    );

    // FIXME: inter unsupported with 4:2:2 and 4:4:4 chroma sampling
    let chroma_sampling = config.chroma_sampling;

    // FIXME: tx partition for intra not supported for chroma 422
    if chroma_sampling == ChromaSampling::Cs422 {
      config.speed_settings.rdo_tx_decision = false;
    }

    let inner = ContextInner::new(&config);

    Ok(Context { is_flushing: false, inner, pool, config })
  }

  /// Validates the configuration.
  pub fn validate(&self) -> Result<(), InvalidConfig> {
    use InvalidConfig::*;

    let config = &self.enc;

    if config.width < 16 || config.width > u16::max_value() as usize {
      return Err(InvalidWidth(config.width));
    }
    if config.height < 16 || config.height > u16::max_value() as usize {
      return Err(InvalidHeight(config.height));
    }

    if config.rdo_lookahead_frames > MAX_RDO_LOOKAHEAD_FRAMES {
      return Err(InvalidRdoLookaheadFrames {
        actual: config.rdo_lookahead_frames,
        max: MAX_RDO_LOOKAHEAD_FRAMES,
      });
    }
    if config.max_key_frame_interval > MAX_MAX_KEY_FRAME_INTERVAL {
      return Err(InvalidMaxKeyFrameInterval {
        actual: config.max_key_frame_interval,
        max: MAX_MAX_KEY_FRAME_INTERVAL,
      });
    }

    if !check_tile_log2(config.tile_cols) {
      return Err(InvalidTileCols(config.tile_cols));
    }
    if !check_tile_log2(config.tile_rows) {
      return Err(InvalidTileRows(config.tile_rows));
    }

    if config.time_base.num == 0
      || config.time_base.num > u32::max_value() as u64
    {
      return Err(InvalidFrameRateNum {
        actual: config.time_base.num,
        max: u32::max_value() as u64,
      });
    }
    if config.time_base.den == 0
      || config.time_base.den > u32::max_value() as u64
    {
      return Err(InvalidFrameRateDen {
        actual: config.time_base.den,
        max: u32::max_value() as u64,
      });
    }

    if let Some(delay) = config.reservoir_frame_delay {
      if delay < 12 || delay > 131_072 {
        return Err(InvalidReservoirFrameDelay(delay));
      }
    }

    if config.switch_frame_interval > 0 && !config.low_latency {
      return Err(InvalidSwitchFrameInterval(config.switch_frame_interval));
    }

    Ok(())
  }
}
