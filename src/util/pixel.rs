// Copyright (c) 2017-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use num_traits::{AsPrimitive, PrimInt, Signed};
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
  fn to_asm_stride(in_stride: usize) -> isize {
    (in_stride * size_of::<Self>()) as isize
  }
}

impl Pixel for u8 {
  type Coeff = i16;

  fn type_enum() -> PixelType {
    PixelType::U8
  }
}

impl Pixel for u16 {
  type Coeff = i32;

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
