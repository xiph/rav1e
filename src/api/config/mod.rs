// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use thiserror::Error;

use std::sync::Arc;

use crate::api::{ChromaSampling, Context, ContextInner};
use crate::cpu_features::CpuFeatureLevel;
use crate::rayon::{ThreadPool, ThreadPoolBuilder};
use crate::tiling::TilingInfo;
use crate::util::Pixel;

mod encoder;
pub use encoder::*;

mod rate;
pub use rate::Error as RateControlError;
pub use rate::{RateControlConfig, RateControlSummary};

mod speedsettings;
pub use speedsettings::*;

/// Enumeration of possible invalid configuration errors.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Error)]
pub enum InvalidConfig {
  /// The width is invalid.
  #[error("invalid width {0} (expected >= 16, <= 32767)")]
  InvalidWidth(usize),
  /// The height is invalid.
  #[error("invalid height {0} (expected >= 16, <= 32767)")]
  InvalidHeight(usize),
  /// RDO lookahead frame count is invalid.
  #[error(
    "invalid rdo lookahead frames {actual} (expected <= {max} and >= {min})"
  )]
  InvalidRdoLookaheadFrames {
    /// The actual value.
    actual: usize,
    /// The maximal supported value.
    max: usize,
    /// The minimal supported value.
    min: usize,
  },
  /// Maximal keyframe interval is invalid.
  #[error("invalid max keyframe interval {actual} (expected <= {max})")]
  InvalidMaxKeyFrameInterval {
    /// The actual value.
    actual: u64,
    /// The maximal supported value.
    max: u64,
  },
  /// Tile columns is invalid.
  #[error("invalid tile cols {0} (expected power of 2)")]
  InvalidTileCols(usize),
  /// Tile rows is invalid.
  #[error("invalid tile rows {0} (expected power of 2)")]
  InvalidTileRows(usize),
  /// Framerate numerator is invalid.
  #[error("invalid framerate numerator {actual} (expected > 0, <= {max})")]
  InvalidFrameRateNum {
    /// The actual value.
    actual: u64,
    /// The maximal supported value.
    max: u64,
  },
  /// Framerate denominator is invalid.
  #[error("invalid framerate denominator {actual} (expected > 0, <= {max})")]
  InvalidFrameRateDen {
    /// The actual value.
    actual: u64,
    /// The maximal supported value.
    max: u64,
  },
  /// Reservoir frame delay is invalid.
  #[error("invalid reservoir frame delay {0} (expected >= 12, <= 131072)")]
  InvalidReservoirFrameDelay(i32),
  /// Reservoir frame delay is invalid.
  #[error(
    "invalid switch frame interval {0} (must only be used with low latency mode)"
  )]
  InvalidSwitchFrameInterval(u64),

  /// The rate control needs a target bitrate in order to produce results
  #[error("The rate control requires a target bitrate")]
  TargetBitrateNeeded,

  /// The configuration
  #[error("Mismatch in the rate control configuration")]
  RateControlConfigurationMismatch,

  // This variant prevents people from exhaustively matching on this enum,
  // which allows us to add more variants without it being a breaking change.
  // This can be replaced with #[non_exhaustive] when it's stable:
  // https://github.com/rust-lang/rust/issues/44109
  #[doc(hidden)]
  #[error("")]
  __NonExhaustive,
}

/// Contains the encoder configuration.
#[derive(Clone, Debug, Default)]
pub struct Config {
  /// Settings which impact the produced bitstream.
  pub(crate) enc: EncoderConfig,
  /// Rate control configuration
  pub(crate) rate_control: RateControlConfig,
  /// The number of threads in the threadpool.
  pub(crate) threads: usize,
  /// Shared thread pool
  pub(crate) pool: Option<Arc<ThreadPool>>,
}

impl Config {
  /// Create a default configuration
  ///
  /// same as Default::default()
  pub fn new() -> Self {
    Config::default()
  }

  /// Set the encoder configuration
  ///
  /// EncoderConfig contains the settings impacting the
  /// codec features used in the produced bitstream.
  pub fn with_encoder_config(mut self, enc: EncoderConfig) -> Self {
    self.enc = enc;
    self
  }

  /// Set the number of workers in the threadpool
  ///
  /// The threadpool is shared across all the different parallel
  /// components in the encoder.
  pub fn with_threads(mut self, threads: usize) -> Self {
    self.threads = threads;
    self
  }

  /// Set the rate control configuration
  ///
  /// The default configuration is single pass
  pub fn with_rate_control(mut self, rate_control: RateControlConfig) -> Self {
    self.rate_control = rate_control;
    self
  }

  #[cfg(features = "unstable")]
  /// Use the provided threadpool
  pub fn with_thread_pool(mut self, pool: Arc<ThreadPool>) -> Self {
    self.threadpool = Some(pool);
    self
  }
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
  pub(crate) fn new_inner<T: Pixel>(
    &self,
  ) -> Result<ContextInner<T>, InvalidConfig> {
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

    let mut inner = ContextInner::new(&config);

    if self.rate_control.emit_pass_data {
      let params = inner.rc_state.get_twopass_out_params(&inner, 0);
      inner.rc_state.init_first_pass(params.pass1_log_base_q);
    }

    if let Some(ref s) = self.rate_control.summary {
      inner.rc_state.init_second_pass();
      inner.rc_state.setup_second_pass(s);
    }

    Ok(inner)
  }
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
    let inner = self.new_inner()?;
    let config = inner.config;
    let pool = if let Some(ref p) = self.pool {
      p.clone()
    } else {
      let pool =
        ThreadPoolBuilder::new().num_threads(self.threads).build().unwrap();
      Arc::new(pool)
    };

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

    if config.rdo_lookahead_frames > MAX_RDO_LOOKAHEAD_FRAMES
      || config.rdo_lookahead_frames < 1
    {
      return Err(InvalidRdoLookaheadFrames {
        actual: config.rdo_lookahead_frames,
        max: MAX_RDO_LOOKAHEAD_FRAMES,
        min: 1,
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

    // TODO: add more validation
    let rc = &self.rate_control;

    if (rc.emit_pass_data || rc.summary.is_some()) && config.bitrate == 0 {
      return Err(TargetBitrateNeeded);
    }

    Ok(())
  }
}
