// Copyright (c) 2020, The rav1e contributors. All rights reserved
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
  /// Fast scene detection mode, uses simple SAD instead of encoder cost estimates.
  pub fast_scene_detection: bool,
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

  // NOTE: put enums and basic type fields above
  /// Range of partition sizes that can be used. Larger ranges are slower.
  ///
  /// Must be based on square block sizes, so e.g. 8×4 isn't allowed here.
  pub partition_range: PartitionRange,
}

impl Default for SpeedSettings {
  /// This is currently used exclusively for feature testing and comparison.
  /// It is set to the slowest settings possible.
  fn default() -> Self {
    SpeedSettings {
      partition_range: PartitionRange::new(
        BlockSize::BLOCK_4X4,
        BlockSize::BLOCK_64X64,
      ),
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
      cdef: true,
      lrf: false,
      sgr_complexity: SGRComplexityLevel::Full,
      use_satd_subpel: true,
      non_square_partition: true,
      segmentation: SegmentationLevel::Full,
      enable_inter_tx_split: false,
      fine_directional_intra: false,
    }
  }
}

impl SpeedSettings {
  /// Set the speed setting according to a numeric speed preset.
  ///
  /// The speed settings vary depending on speed value from 0 to 10.
  /// - 10 (fastest): fixed block size 32x32, reduced TX set, fast deblock, fast scenechange detection.
  /// - 9: min block size 32x32, reduced TX set, fast deblock.
  /// - 8: min block size 8x8, reduced TX set, fast deblock.
  /// - 7: min block size 8x8, reduced TX set.
  /// - 6 (default): min block size 8x8, reduced TX set, complex pred modes for keyframes.
  /// - 5: min block size 8x8, complex pred modes for keyframes, RDO TX decision.
  /// - 4: min block size 8x8, complex pred modes for keyframes, RDO TX decision, full SGR search.
  /// - 3: min block size 8x8, complex pred modes for keyframes, RDO TX decision, include near MVs,
  ///        full SGR search.
  /// - 2: min block size 8x8, complex pred modes for keyframes, RDO TX decision, include near MVs,
  ///        bottom-up encoding, full SGR search.
  /// - 1: min block size 4x4, complex pred modes, RDO TX decision, include near MVs,
  ///        bottom-up encoding, full SGR search.
  /// - 0 (slowest): min block size 4x4, complex pred modes, RDO TX decision, include near MVs,
  ///        bottom-up encoding with non-square partitions everywhere, full SGR search.
  pub fn from_preset(speed: usize) -> Self {
    SpeedSettings {
      partition_range: Self::partition_range_preset(speed),
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
      cdef: Self::cdef_preset(speed),
      lrf: Self::lrf_preset(speed),
      sgr_complexity: Self::sgr_complexity_preset(speed),
      use_satd_subpel: Self::use_satd_subpel(speed),
      non_square_partition: Self::non_square_partition_preset(speed),
      segmentation: Self::segmentation_preset(speed),
      enable_inter_tx_split: Self::enable_inter_tx_split_preset(speed),
      fine_directional_intra: Self::fine_directional_intra_preset(speed),
    }
  }

  /// This preset is set this way because 8x8 with reduced TX set is faster but with equivalent
  /// or better quality compared to 16x16 (to which reduced TX set does not apply).
  fn partition_range_preset(speed: usize) -> PartitionRange {
    if speed <= 1 {
      PartitionRange::new(BlockSize::BLOCK_4X4, BlockSize::BLOCK_64X64)
    } else if speed <= 8 {
      PartitionRange::new(BlockSize::BLOCK_8X8, BlockSize::BLOCK_64X64)
    } else if speed <= 9 {
      PartitionRange::new(BlockSize::BLOCK_32X32, BlockSize::BLOCK_64X64)
    } else {
      PartitionRange::new(BlockSize::BLOCK_32X32, BlockSize::BLOCK_32X32)
    }
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
    speed >= 6
  }

  /// TX domain distortion is always faster, with no significant quality change
  const fn tx_domain_distortion_preset(_speed: usize) -> bool {
    true
  }

  const fn tx_domain_rate_preset(_speed: usize) -> bool {
    false
  }

  const fn encode_bottomup_preset(speed: usize) -> bool {
    speed <= 2
  }

  /// Set default rdo-lookahead-frames for different speed settings
  pub fn rdo_lookahead_frames(speed: usize) -> usize {
    match speed {
      9..=10 => 10,
      6..=8 => 20,
      2..=5 => 30,
      0..=1 => 40,
      _ => 40,
    }
  }

  const fn rdo_tx_decision_preset(speed: usize) -> bool {
    speed <= 5
  }

  fn prediction_modes_preset(speed: usize) -> PredictionModesSetting {
    if speed <= 1 {
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

  const fn cdef_preset(_speed: usize) -> bool {
    true
  }

  const fn lrf_preset(speed: usize) -> bool {
    speed <= 9
  }

  fn sgr_complexity_preset(speed: usize) -> SGRComplexityLevel {
    if speed <= 4 {
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

  fn segmentation_preset(speed: usize) -> SegmentationLevel {
    if speed == 0 {
      SegmentationLevel::Full
    } else {
      SegmentationLevel::Simple
    }
  }

  // FIXME: With unknown reasons, inter_tx_split does not work if reduced_tx_set is false
  const fn enable_inter_tx_split_preset(speed: usize) -> bool {
    speed >= 9
  }

  fn fine_directional_intra_preset(_speed: usize) -> bool {
    true
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
