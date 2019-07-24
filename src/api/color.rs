// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use num_derive::FromPrimitive;
use arg_enum_proc_macro::ArgEnum;

/// Sample position for subsampled chroma
#[derive(Copy, Clone, Debug, PartialEq, FromPrimitive)]
#[repr(C)]
pub enum ChromaSamplePosition {
  /// The source video transfer function must be signaled
  /// outside the AV1 bitstream.
  Unknown,
  /// Horizontally co-located with (0, 0) luma sample, vertically positioned
  /// in the middle between two luma samples.
  Vertical,
  /// Co-located with (0, 0) luma sample.
  Colocated
}

impl Default for ChromaSamplePosition {
  fn default() -> Self {
    ChromaSamplePosition::Unknown
  }
}

/// Chroma subsampling format
#[derive(Copy, Clone, Debug, PartialEq, FromPrimitive)]
#[repr(C)]
pub enum ChromaSampling {
  /// Both vertically and horizontally subsampled.
  Cs420,
  /// Horizontally subsampled.
  Cs422,
  /// Not subsampled.
  Cs444,
  /// Monochrome.
  Cs400
}

impl Default for ChromaSampling {
  fn default() -> Self {
    ChromaSampling::Cs420
  }
}

impl ChromaSampling {
  /// Provide the sampling period in the horizontal and vertical axes.
  pub fn sampling_period(self) -> (usize, usize) {
    use self::ChromaSampling::*;
    match self {
      Cs420 => (2, 2),
      Cs422 => (2, 1),
      Cs444 => (1, 1),
      Cs400 => (2, 2)
    }
  }
}

/// Supported Color Primaries
///
/// As defined by “Color primaries” section of ISO/IEC 23091-4/ITU-T H.273
#[derive(ArgEnum, Debug, Clone, Copy, PartialEq, FromPrimitive)]
#[repr(C)]
pub enum ColorPrimaries {
  BT709 = 1,
  Unspecified,
  BT470M = 4,
  BT470BG,
  ST170M,
  ST240M,
  Film,
  BT2020,
  ST428,
  P3DCI,
  P3Display,
  Tech3213 = 22
}

impl Default for ColorPrimaries {
  fn default() -> Self {
    ColorPrimaries::Unspecified
  }
}

/// Supported Transfer Characteristics
///
/// As defined by “Transfer characteristics” section of ISO/IEC 23091-4/ITU-TH.273.
#[derive(ArgEnum, Debug, Clone, Copy, PartialEq, FromPrimitive)]
#[repr(C)]
pub enum TransferCharacteristics {
  BT1886 = 1,
  Unspecified,
  BT470M = 4,
  BT470BG,
  ST170M,
  ST240M,
  Linear,
  Logarithmic100,
  Logarithmic316,
  XVYCC,
  BT1361E,
  SRGB,
  BT2020Ten,
  BT2020Twelve,
  PerceptualQuantizer,
  ST428,
  HybridLogGamma
}

impl Default for TransferCharacteristics {
  fn default() -> Self {
    TransferCharacteristics::Unspecified
  }
}

/// Matrix coefficients
///
/// As defined by the “Matrix coefficients” section of ISO/IEC 23091-4/ITU-TH.273.
#[derive(ArgEnum, Debug, Clone, Copy, PartialEq, FromPrimitive)]
#[repr(C)]
pub enum MatrixCoefficients {
  Identity = 0,
  BT709,
  Unspecified,
  BT470M = 4,
  BT470BG,
  ST170M,
  ST240M,
  YCgCo,
  BT2020NonConstantLuminance,
  BT2020ConstantLuminance,
  ST2085,
  ChromaticityDerivedNonConstantLuminance,
  ChromaticityDerivedConstantLuminance,
  ICtCp
}

impl Default for MatrixCoefficients {
  fn default() -> Self {
    MatrixCoefficients::Unspecified
  }
}

/// Signal the content color description
#[derive(Copy, Clone, Debug)]
pub struct ColorDescription {
  pub color_primaries: ColorPrimaries,
  pub transfer_characteristics: TransferCharacteristics,
  pub matrix_coefficients: MatrixCoefficients
}

/// Allowed pixel value range
///
/// C.f. VideoFullRangeFlag variable specified in ISO/IEC 23091-4/ITU-T H.273
#[derive(ArgEnum, Debug, Clone, Copy, PartialEq, FromPrimitive)]
#[repr(C)]
pub enum PixelRange {
  /// Studio swing representation
  Limited,
  /// Full swing representation
  Full
}

impl Default for PixelRange {
  fn default() -> Self {
    PixelRange::Limited
  }
}

/// High dynamic range content light level
///
/// As defined by CEA-861.3, Appendix A.
#[derive(Copy, Clone, Debug)]
pub struct ContentLight {
  /// Maximum content light level
  pub max_content_light_level: u16,
  /// Maximum frame-average light level
  pub max_frame_average_light_level: u16
}

/// Chromaticity coordinates expressed as 0.16 fixed-point values
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Point {
  pub x: u16,
  pub y: u16
}

/// High dynamic range mastering display color volume
///
/// As defined by CIE 1931
#[derive(Copy, Clone, Debug)]
pub struct MasteringDisplay {
  /// Chromaticity coordinates in Red, Green, Blue order
  /// expressed as 0.16 fixed-point
  pub primaries: [Point; 3],
  /// Chromaticity coordinates expressed as 0.16 fixed-point
  pub white_point: Point,
  /// 24.8 fixed-point maximum luminance in candelas per square meter
  pub max_luminance: u32,
  /// 18.14 fixed-point minimum luminance in candelas per square meter
  pub min_luminance: u32
}


