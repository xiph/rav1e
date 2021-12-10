// Copyright (c) 2020-2021, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use num_derive::*;

use crate::partition::BlockSize;
use crate::serialize::{Deserialize, Serialize};

use std::fmt;

// NOTE: Add Structures at the end.
/// Contains the speed settings.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SpeedSettings {
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

  /// The number of lookahead frames to be used for temporal RDO.
  ///
  /// Higher is slower.
  pub rdo_lookahead_frames: usize,

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

  /// Which scene detection mode to use. Standard is slower, but best.
  pub scene_detection_mode: SceneDetectionSpeed,

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

  /// Search level for segmentation.
  ///
  /// Full search is at least twice as slow.
  pub segmentation: SegmentationLevel,

  /// Enable tx split for inter mode block.
  pub enable_inter_tx_split: bool,

  /// Use fine directional intra prediction
  pub fine_directional_intra: bool,

  /// Enable full search in some parts of motion estimation. Allowing full
  /// search is slower.
  pub me_allow_full_search: bool,

  // NOTE: put enums and basic type fields above
  /// Range of partition sizes that can be used. Larger ranges are slower.
  ///
  /// Must be based on square block sizes, so e.g. 8×4 isn't allowed here.
  pub partition_range: PartitionRange,
}

impl Default for SpeedSettings {
  /// The default settings are equivalent to speed 0
  fn default() -> Self {
    SpeedSettings {
      partition_range: PartitionRange::new(
        BlockSize::BLOCK_4X4,
        BlockSize::BLOCK_64X64,
      ),
      multiref: true,
      fast_deblock: false,
      reduced_tx_set: false,
      // TX domain distortion is always faster, with no significant quality change,
      // although it will be ignored when Tune == Psychovisual.
      tx_domain_distortion: true,
      tx_domain_rate: false,
      encode_bottomup: true,
      rdo_tx_decision: true,
      rdo_lookahead_frames: 40,
      prediction_modes: PredictionModesSetting::ComplexAll,
      include_near_mvs: true,
      scene_detection_mode: SceneDetectionSpeed::Standard,
      cdef: true,
      lrf: true,
      sgr_complexity: SGRComplexityLevel::Full,
      use_satd_subpel: true,
      non_square_partition: true,
      segmentation: SegmentationLevel::Full,
      enable_inter_tx_split: false,
      fine_directional_intra: true,
      me_allow_full_search: true,
    }
  }
}

impl SpeedSettings {
  /// Set the speed setting according to a numeric speed preset.
  pub fn from_preset(speed: usize) -> Self {
    // The default settings are equivalent to speed 0
    let mut settings = SpeedSettings::default();

    if speed >= 1 {
      settings.segmentation = SegmentationLevel::Simple;
    }

    if speed >= 2 {
      settings.non_square_partition = false;
    }

    if speed >= 3 {
      settings.partition_range =
        PartitionRange::new(BlockSize::BLOCK_8X8, BlockSize::BLOCK_64X64);
      settings.rdo_lookahead_frames = 30;
      settings.prediction_modes = PredictionModesSetting::ComplexKeyframes;
    }

    if speed >= 4 {
      settings.encode_bottomup = false;
    }

    if speed >= 5 {
      settings.include_near_mvs = false;
      settings.sgr_complexity = SGRComplexityLevel::Reduced;
    }

    if speed >= 6 {
      settings.reduced_tx_set = true;
      settings.rdo_lookahead_frames = 20;
      settings.rdo_tx_decision = false;
      settings.me_allow_full_search = false;
    }

    if speed >= 7 {
      settings.prediction_modes = PredictionModesSetting::Simple;
    }

    if speed >= 8 {
      // Multiref is enabled automatically if low_latency is false.
      //
      // If low_latency is true, enabling multiref allows using multiple
      // backwards references. low_latency false enables both forward and
      // backwards references.
      settings.multiref = false;
      settings.fast_deblock = true;
    }

    if speed >= 9 {
      // 8x8 is fast enough to use until very high speed levels,
      // because 8x8 with reduced TX set is faster but with equivalent
      // or better quality compared to 16x16 (to which reduced TX set does not apply).
      settings.partition_range =
        PartitionRange::new(BlockSize::BLOCK_32X32, BlockSize::BLOCK_64X64);
      settings.rdo_lookahead_frames = 10;
      // FIXME: With unknown reasons, inter_tx_split does not work if reduced_tx_set is false
      settings.enable_inter_tx_split = true;
    }

    if speed >= 10 {
      settings.partition_range =
        PartitionRange::new(BlockSize::BLOCK_32X32, BlockSize::BLOCK_32X32);
      settings.scene_detection_mode = SceneDetectionSpeed::Fast;
      settings.lrf = false;
      settings.use_satd_subpel = false;
    }

    settings
  }
}

/// Range of block sizes to use.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PartitionRange {
  pub(crate) min: BlockSize,
  pub(crate) max: BlockSize,
}

impl PartitionRange {
  /// Creates a new partition range with min and max partition sizes.
  pub fn new(min: BlockSize, max: BlockSize) -> Self {
    assert!(max >= min);
    // Topdown search checks the min block size for PARTITION_SPLIT only, so
    // the min block size must be square.
    assert!(min.is_sqr());
    // Rectangular max partition sizes have not been tested.
    assert!(max.is_sqr());

    Self { min, max }
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
pub enum SceneDetectionSpeed {
  /// Fastest scene detection using pixel-wise comparison
  Fast,
  /// Scene detection using motion vectors and cost estimates
  Standard,
  /// Completely disable scene detection and only place keyframes
  /// at fixed intervals.
  None,
}

impl fmt::Display for SceneDetectionSpeed {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    write!(
      f,
      "{}",
      match self {
        SceneDetectionSpeed::Fast => "Fast",
        SceneDetectionSpeed::Standard => "Standard",
        SceneDetectionSpeed::None => "None",
      }
    )
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

/// Search level for segmentation
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
pub enum SegmentationLevel {
  /// No segmentation is signalled.
  Disabled,
  /// Segmentation index is derived from source statistics.
  Simple,
  /// Search all segmentation indices.
  Full,
}

impl fmt::Display for SegmentationLevel {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    write!(
      f,
      "{}",
      match self {
        SegmentationLevel::Disabled => "Disabled",
        SegmentationLevel::Simple => "Simple",
        SegmentationLevel::Full => "Full",
      }
    )
  }
}
