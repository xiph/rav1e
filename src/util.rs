// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use num_traits::*;
use std::fmt::{Debug, Display};
use std::mem::{size_of, MaybeUninit};
use std::ops::AddAssign;

//TODO: Nice to have (although I wasn't able to find a way to do it yet in rust): zero-fill arrays that are
// shorter than required.  Need const fn (Rust Issue #24111) or const generics (Rust RFC #2000)
macro_rules! cdf {
    ($($x:expr),+) =>  {[$(32768 - $x),+, 0, 0]}
}

macro_rules! cdf_size {
  ($x:expr) => {
    $x + 1
  };
}

#[repr(align(32))]
pub struct Align32;

// A 16 byte aligned array.
// # Examples
// ```
// let mut x: AlignedArray<[i16; 64 * 64]> = AlignedArray::new([0; 64 * 64]);
// assert!(x.array.as_ptr() as usize % 16 == 0);
//
// let mut x: AlignedArray<[i16; 64 * 64]> = AlignedArray::uninitialized();
// assert!(x.array.as_ptr() as usize % 16 == 0);
// ```
pub struct AlignedArray<ARRAY> {
  _alignment: [Align32; 0],
  pub array: ARRAY,
}

impl<A> AlignedArray<A> {
  pub const fn new(array: A) -> Self {
    AlignedArray { _alignment: [], array }
  }
  #[allow(clippy::uninit_assumed_init)]
  pub fn uninitialized() -> Self {
    Self::new(unsafe { MaybeUninit::uninit().assume_init() })
  }
}

#[test]
fn sanity() {
  fn is_aligned<T>(ptr: *const T, n: usize) -> bool {
    ((ptr as usize) & ((1 << n) - 1)) == 0
  }

  let a: AlignedArray<_> = AlignedArray::new([0u8; 3]);
  assert!(is_aligned(a.array.as_ptr(), 4));
}

pub trait Fixed {
  fn floor_log2(&self, n: usize) -> usize;
  fn ceil_log2(&self, n: usize) -> usize;
  fn align_power_of_two(&self, n: usize) -> usize;
  fn align_power_of_two_and_shift(&self, n: usize) -> usize;
}

impl Fixed for usize {
  #[inline]
  fn floor_log2(&self, n: usize) -> usize {
    self & !((1 << n) - 1)
  }
  #[inline]
  fn ceil_log2(&self, n: usize) -> usize {
    (self + (1 << n) - 1).floor_log2(n)
  }
  #[inline]
  fn align_power_of_two(&self, n: usize) -> usize {
    self.ceil_log2(n)
  }
  #[inline]
  fn align_power_of_two_and_shift(&self, n: usize) -> usize {
    (self + (1 << n) - 1) >> n
  }
}

pub fn clamp<T: PartialOrd>(input: T, min: T, max: T) -> T {
  if input < min {
    min
  } else if input > max {
    max
  } else {
    input
  }
}

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
  RegisteredPrimitive + Into<i32> + AddAssign + Signed + 'static
{
  type Pixel: Pixel;
}

impl Coefficient for i16 {
  type Pixel = u8;
}
impl Coefficient for i32 {
  type Pixel = u16;
}

pub trait ILog: PrimInt {
  // Integer binary logarithm of an integer value.
  // Returns floor(log2(self)) + 1, or 0 if self == 0.
  // This is the number of bits that would be required to represent self in two's
  //  complement notation with all of the leading zeros stripped.
  // TODO: Mark const once trait functions can be constant
  fn ilog(self) -> usize {
    size_of::<Self>() * 8 - self.leading_zeros() as usize
  }
}

impl<T> ILog for T where T: PrimInt {}

#[inline(always)]
pub fn msb(x: i32) -> i32 {
  debug_assert!(x > 0);
  31 ^ (x.leading_zeros() as i32)
}

#[inline(always)]
pub const fn round_shift(value: i32, bit: usize) -> i32 {
  (value + (1 << bit >> 1)) >> bit
}

pub trait Dim {
  const W: usize;
  const H: usize;
}

macro_rules! blocks_dimension {
  ($(($W:expr, $H:expr)),+) => {
    paste::item! {
      $(
        pub struct [<Block $W x $H>];

        impl Dim for [<Block $W x $H>] {
          const W: usize = $W;
          const H: usize = $H;
        }
      )*
    }
  };
}

blocks_dimension! {
  (4, 4), (8, 8), (16, 16), (32, 32), (64, 64),
  (4, 8), (8, 16), (16, 32), (32, 64),
  (8, 4), (16, 8), (32, 16), (64, 32),
  (4, 16), (8, 32), (16, 64),
  (16, 4), (32, 8), (64, 16)
}
