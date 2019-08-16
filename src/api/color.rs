// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use arg_enum_proc_macro::ArgEnum;
use num_derive::FromPrimitive;

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
  Colocated,
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
  Cs400,
}

impl Default for ChromaSampling {
  fn default() -> Self {
    ChromaSampling::Cs420
  }
}

impl ChromaSampling {
  /// Provides the amount to right shift the luma plane dimensions to get the
  ///  chroma plane dimensions.
  /// Only values 0 or 1 are ever returned.
  /// The plane dimensions must also be rounded up to accommodate odd luma plane
  ///  sizes.
  /// Cs400 returns None, as there are no chroma planes.
  pub fn get_decimation(self) -> Option<(usize, usize)> {
    use self::ChromaSampling::*;
    match self {
      Cs420 => Some((1, 1)),
      Cs422 => Some((1, 0)),
      Cs444 => Some((0, 0)),
      Cs400 => None,
    }
  }

  /// Calculates the size of a chroma plane for this sampling type, given the luma plane dimensions.
  pub fn get_chroma_dimensions(
    self, luma_width: usize, luma_height: usize,
  ) -> (usize, usize) {
    if let Some((ss_x, ss_y)) = self.get_decimation() {
      ((luma_width + ss_x) >> ss_x, (luma_height + ss_y) >> ss_y)
    } else {
      (0, 0)
    }
  }
}

/// Supported Color Primaries
///
/// As defined by “Color primaries” section of ISO/IEC 23091-4/ITU-T H.273
#[derive(ArgEnum, Debug, Clone, Copy, PartialEq, FromPrimitive)]
#[repr(C)]
pub enum ColorPrimaries {
  /// BT.709
  BT709 = 1,
  /// Unspecified, must be signaled or inferred outside of the bitstream
  Unspecified,
  /// BT.470 System M (historical)
  BT470M = 4,
  /// BT.470 System B, G (historical)
  BT470BG,
  /// BT.601-7 525 (SMPTE 170 M)
  ST170M,
  /// SMPTE 240M (historical)
  ST240M,
  /// Generic film
  Film,
  /// BT.2020, BT.2100
  BT2020,
  /// SMPTE 248 (CIE 1921 XYZ)
  ST428,
  /// SMPTE RP 431-2
  P3DCI,
  /// SMPTE EG 432-1
  P3Display,
  /// EBU Tech. 3213-E
  Tech3213 = 22,
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
  /// BT.709
  BT1886 = 1,
  /// Unspecified, must be signaled or inferred outside of the bitstream
  Unspecified,
  /// BT.470 System M (historical)
  BT470M = 4,
  /// BT.470 System B, G (historical)
  BT470BG,
  /// BT.601-7 525 (SMPTE 170 M)
  ST170M,
  /// SMPTE 240 M
  ST240M,
  /// Linear
  Linear,
  /// Logarithmic (100:1 range)
  Logarithmic100,
  /// Logarithmic ((100 * √10):1 range)
  Logarithmic316,
  /// IEC 61966-2-4
  XVYCC,
  /// BT.1361 extended color gamut system (historical)
  BT1361E,
  /// sRGB or sYCC
  SRGB,
  /// BT.2020 10-bit systems
  BT2020Ten,
  /// BT.2020 12-bit systems
  BT2020Twelve,
  /// SMPTE ST 2084, ITU BT.2100 PQ
  PerceptualQuantizer,
  /// SMPTE ST 428
  ST428,
  /// BT.2100 HLG, ARIB STD-B67
  HybridLogGamma,
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
  /// Identity matrix
  Identity = 0,
  /// BT.709
  BT709,
  /// Unspecified, must be signaled or inferred outside of the bitstream.
  Unspecified,
  /// US FCC 73.628
  BT470M = 4,
  /// BT.470 System B, G (historical)
  BT470BG,
  /// BT.601-7 525 (SMPTE 170 M)
  ST170M,
  /// SMPTE 240 M
  ST240M,
  /// YCgCo
  YCgCo,
  /// BT.2020 non-constant luminance, BT.2100 YCbCr
  BT2020NonConstantLuminance,
  /// BT.2020 constant luminance
  BT2020ConstantLuminance,
  /// SMPTE ST 2085 YDzDx
  ST2085,
  /// Chromaticity-derived non-constant luminance
  ChromaticityDerivedNonConstantLuminance,
  /// Chromaticity-derived constant luminance
  ChromaticityDerivedConstantLuminance,
  /// BT.2020 ICtCp
  ICtCp,
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
  pub matrix_coefficients: MatrixCoefficients,
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
  Full,
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
  pub max_frame_average_light_level: u16,
}

/// Chromaticity coordinates expressed as 0.16 fixed-point values
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Point {
  pub x: u16,
  pub y: u16,
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
  pub min_luminance: u32,
}
