// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use num_traits::*;
use std::mem;
use std::mem::size_of;
use std::fmt::{Debug, Display};
#[allow(deprecated, unused_imports)]
use ::std::ascii::AsciiExt;

// Imported from clap, to avoid to depend directly to the crate
macro_rules! _clap_count_exprs {
    () => { 0 };
    ($e:expr) => { 1 };
    ($e:expr, $($es:expr),+) => { 1 + _clap_count_exprs!($($es),*) };
}
macro_rules! arg_enum {
    (@as_item $($i:item)*) => ($($i)*);
    (@impls ( $($tts:tt)* ) -> ($e:ident, $($v:ident),+)) => {
        arg_enum!(@as_item
        $($tts)*

        impl ::std::str::FromStr for $e {
            type Err = String;

            fn from_str(s: &str) -> ::std::result::Result<Self,Self::Err> {
                match s {
                    $(stringify!($v) |
                    _ if s.eq_ignore_ascii_case(stringify!($v)) => Ok($e::$v)),+,
                    _ => Err({
                        let v = vec![
                            $(stringify!($v),)+
                        ];
                        format!("valid values: {}",
                            v.join(" ,"))
                    }),
                }
            }
        }
        impl ::std::fmt::Display for $e {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                match *self {
                    $($e::$v => write!(f, stringify!($v)),)+
                }
            }
        }
        impl $e {
            #[allow(dead_code)]
            pub fn variants() -> [&'static str; _clap_count_exprs!($(stringify!($v)),+)] {
                [
                    $(stringify!($v),)+
                ]
            }
        });
    };
    ($(#[$($m:meta),+])+ pub enum $e:ident { $($v:ident $(=$val:expr)*,)+ } ) => {
        arg_enum!(@impls
            ($(#[$($m),+])+
            pub enum $e {
                $($v$(=$val)*),+
            }) -> ($e, $($v),+)
        );
    };
    ($(#[$($m:meta),+])+ pub enum $e:ident { $($v:ident $(=$val:expr)*),+ } ) => {
        arg_enum!(@impls
            ($(#[$($m),+])+
            pub enum $e {
                $($v$(=$val)*),+
            }) -> ($e, $($v),+)
        );
    };
    ($(#[$($m:meta),+])+ enum $e:ident { $($v:ident $(=$val:expr)*,)+ } ) => {
        arg_enum!(@impls
            ($(#[$($m),+])+
             enum $e {
                 $($v$(=$val)*),+
             }) -> ($e, $($v),+)
        );
    };
    ($(#[$($m:meta),+])+ enum $e:ident { $($v:ident $(=$val:expr)*),+ } ) => {
        arg_enum!(@impls
            ($(#[$($m),+])+
            enum $e {
                $($v$(=$val)*),+
            }) -> ($e, $($v),+)
        );
    };
    (pub enum $e:ident { $($v:ident $(=$val:expr)*,)+ } ) => {
        arg_enum!(@impls
            (pub enum $e {
                $($v$(=$val)*),+
            }) -> ($e, $($v),+)
        );
    };
    (pub enum $e:ident { $($v:ident $(=$val:expr)*),+ } ) => {
        arg_enum!(@impls
            (pub enum $e {
                $($v$(=$val)*),+
            }) -> ($e, $($v),+)
        );
    };
    (enum $e:ident { $($v:ident $(=$val:expr)*,)+ } ) => {
        arg_enum!(@impls
            (enum $e {
                $($v$(=$val)*),+
            }) -> ($e, $($v),+)
        );
    };
    (enum $e:ident { $($v:ident $(=$val:expr)*),+ } ) => {
        arg_enum!(@impls
            (enum $e {
                $($v$(=$val)*),+
            }) -> ($e, $($v),+)
        );
    };
}

//TODO: Nice to have (although I wasnt able to find a way to do it yet in rust): zero-fill arrays that are
// shorter than required.  Need const fn (Rust Issue #24111) or const generics (Rust RFC #2000)
macro_rules! cdf {
    ($($x:expr),+) =>  {[$(32768 - $x),+, 0, 0]}
}

macro_rules! cdf_size {
    ($x:expr) => ($x+1);
}

#[repr(align(32))]
#[derive(Clone)]
pub struct Align32;

/// A 16 byte aligned array.
/// # Examples
/// ```
/// extern crate rav1e;
/// use rav1e::util::*;
///
/// let mut x: AlignedArray<[i16; 64 * 64]> = AlignedArray([0; 64 * 64]);
/// assert!(x.array.as_ptr() as usize % 16 == 0);
///
/// let mut x: AlignedArray<[i16; 64 * 64]> = UninitializedAlignedArray();
/// assert!(x.array.as_ptr() as usize % 16 == 0);
/// ```
#[derive(Clone)]
pub struct AlignedArray<ARRAY>
{
  _alignment: [Align32; 0],
  pub array: ARRAY
}

#[allow(non_snake_case)]
pub fn AlignedArray<ARRAY>(array: ARRAY) -> AlignedArray<ARRAY> {
  AlignedArray { _alignment: [], array }
}

#[allow(non_snake_case)]
pub fn UninitializedAlignedArray<ARRAY>() -> AlignedArray<ARRAY> {
  AlignedArray(unsafe { mem::uninitialized() })
}

#[test]
fn sanity() {
  let a: AlignedArray<_> = AlignedArray([0u8; 3]);
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

/// Check alignment.
pub fn is_aligned<T>(ptr: *const T, n: usize) -> bool {
  ((ptr as usize) & ((1 << n) - 1)) == 0
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

pub trait CastFromPrimitive<T> : Copy + 'static {
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
impl_cast_from_primitive!(u8 => { i8, i16, i32, i64, isize });
impl_cast_from_primitive!(u16 => { u32, u64, usize });
impl_cast_from_primitive!(u16 => { i8, i16, i32, i64, isize });
impl_cast_from_primitive!(i16 => { u32, u64, usize });
impl_cast_from_primitive!(i16 => { i8, i16, i32, i64, isize });
impl_cast_from_primitive!(i32 => { u32, u64, usize });
impl_cast_from_primitive!(i32 => { i8, i16, i32, i64, isize });

pub trait Pixel:
  PrimInt
  + Into<u32>
  + Into<i32>
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
  + Debug
  + Display
  + Send
  + Sync
  + 'static
{}

impl Pixel for u8 {}
impl Pixel for u16 {}

macro_rules! impl_cast_from_pixel_to_primitive {
  ( $T:ty ) => {
    impl<T: Pixel> CastFromPrimitive<T> for $T {
      #[inline(always)]
      fn cast_from(v: T) -> Self { v.as_() }
    }
  };
}

impl_cast_from_pixel_to_primitive!(u8);
impl_cast_from_pixel_to_primitive!(i16);
impl_cast_from_pixel_to_primitive!(u16);
impl_cast_from_pixel_to_primitive!(i32);
impl_cast_from_pixel_to_primitive!(u32);

pub trait ILog: PrimInt {
  fn ilog(self) -> Self {
    Self::from(size_of::<Self>() * 8 - self.leading_zeros() as usize).unwrap()
  }
}

impl<T> ILog for T where T: PrimInt {}

pub fn msb(x: i32) -> i32 {
  debug_assert!(x > 0);
  31 ^ (x.leading_zeros() as i32)
}

pub fn round_shift(value: i32, bit: usize) -> i32 {
  (value + (1 << bit >> 1)) >> bit
}

pub unsafe fn convert_slice_2d<NEW, OLD>(
  dst: *mut NEW, dst_stride: usize, src: *const OLD, src_stride: usize,
  width: usize, height: usize
)
where
  NEW: CastFromPrimitive<OLD> + Copy + 'static,
  OLD: Copy + 'static,
{
  for y in 0..height {
    for x in 0..width {
      let p_dst = dst.add(y * dst_stride + x);
      let p_src = src.add(y * src_stride + x);
      *p_dst = NEW::cast_from(*p_src);
    }
  }
}
