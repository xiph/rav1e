// Copyright (c) 2017-2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::serialize::{Deserialize, Serialize};
use crate::wasm_bindgen::*;

use num_derive::FromPrimitive;
use num_traits::{AsPrimitive, PrimInt, Signed};

use std::fmt;
use std::fmt::{Debug, Display};
use std::mem::size_of;
use std::ops::AddAssign;

/// Trait for casting between primitive types.
pub trait CastFromPrimitive<T>: Copy + 'static {
  /// Casts the given value into `Self`.
  fn cast_from(v: T) -> Self;
}

macro_rules! impl_cast_from_primitive {
  ( $T:ty => $U:ty ) => {
    impl CastFromPrimitive<$U> for $T {
      #[inline(always)]
      fn cast_from(v: $U) -> Self { v as Self }
    }
  };
  ( $T:ty => { $( $U:ty ),* } ) => {
    $( impl_cast_from_primitive!($T => $U); )*
  };
}

// casts to { u8, u16 } are implemented separately using Pixel, so that the
// compiler understands that CastFromPrimitive<T: Pixel> is always implemented
impl_cast_from_primitive!(u8 => { u32, u64, usize });
impl_cast_from_primitive!(u8 => { i8, i64, isize });
impl_cast_from_primitive!(u16 => { u32, u64, usize });
impl_cast_from_primitive!(u16 => { i8, i64, isize });
impl_cast_from_primitive!(i16 => { u32, u64, usize });
impl_cast_from_primitive!(i16 => { i8, i64, isize });
impl_cast_from_primitive!(i32 => { u32, u64, usize });
impl_cast_from_primitive!(i32 => { i8, i64, isize });

pub trait RegisteredPrimitive:
  PrimInt
  + AsPrimitive<u8>
  + AsPrimitive<i16>
  + AsPrimitive<u16>
  + AsPrimitive<i32>
  + AsPrimitive<u32>
  + AsPrimitive<usize>
  + CastFromPrimitive<u8>
  + CastFromPrimitive<i16>
  + CastFromPrimitive<u16>
  + CastFromPrimitive<i32>
  + CastFromPrimitive<u32>
  + CastFromPrimitive<usize>
{
}

impl RegisteredPrimitive for u8 {}
impl RegisteredPrimitive for u16 {}
impl RegisteredPrimitive for i16 {}
impl RegisteredPrimitive for i32 {}

macro_rules! impl_cast_from_pixel_to_primitive {
  ( $T:ty ) => {
    impl<T: RegisteredPrimitive> CastFromPrimitive<T> for $T {
      #[inline(always)]
      fn cast_from(v: T) -> Self {
        v.as_()
      }
    }
  };
}

impl_cast_from_pixel_to_primitive!(u8);
impl_cast_from_pixel_to_primitive!(i16);
impl_cast_from_pixel_to_primitive!(u16);
impl_cast_from_pixel_to_primitive!(i32);
impl_cast_from_pixel_to_primitive!(u32);

/// Types that can be used as pixel types.
pub enum PixelType {
  /// 8 bits per pixel, stored in a `u8`.
  U8,
  /// 10 or 12 bits per pixel, stored in a `u16`.
  U16,
}

/// A type that can be used as a pixel type.
pub trait Pixel:
  RegisteredPrimitive
  + Into<u32>
  + Into<i32>
  + Debug
  + Display
  + Send
  + Sync
  + 'static
{
  type Coeff: Coefficient;

  /// Returns a [`PixelType`] variant corresponding to this type.
  ///
  /// [`PixelType`]: enum.PixelType.html
  fn type_enum() -> PixelType;

  /// Converts stride in pixels to stride in bytes.
  #[inline]
  #[allow(clippy::wrong_self_convention)]
  fn to_asm_stride(in_stride: usize) -> isize {
    (in_stride * size_of::<Self>()) as isize
  }
}

impl Pixel for u8 {
  type Coeff = i16;

  #[inline]
  fn type_enum() -> PixelType {
    PixelType::U8
  }
}

impl Pixel for u16 {
  type Coeff = i32;

  #[inline]
  fn type_enum() -> PixelType {
    PixelType::U16
  }
}

pub trait Coefficient:
  RegisteredPrimitive + Into<i32> + AddAssign + Signed + Debug + 'static
{
  type Pixel: Pixel;
}

impl Coefficient for i16 {
  type Pixel = u8;
}
impl Coefficient for i32 {
  type Pixel = u16;
}

/// Chroma subsampling format
#[wasm_bindgen]
#[derive(
  Copy, Clone, Debug, PartialEq, FromPrimitive, Serialize, Deserialize,
)]
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

impl fmt::Display for ChromaSampling {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    write!(
      f,
      "{}",
      match self {
        ChromaSampling::Cs420 => "4:2:0",
        ChromaSampling::Cs422 => "4:2:2",
        ChromaSampling::Cs444 => "4:4:4",
        ChromaSampling::Cs400 => "Monochrome",
      }
    )
  }
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
