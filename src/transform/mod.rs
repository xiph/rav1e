// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(non_camel_case_types)]
#![allow(dead_code)]

pub use self::forward::*;
pub use self::inverse::*;

pub use self::forward::txfm_types::Detail as FwTxDetail;
pub use self::inverse::txfm_types::Detail as InvTxDetail;
pub use self::tx_1d_types::Detail as Tx1DDetail;
pub use self::tx_2d_types::Detail as Tx2DDetail;
pub use self::tx_sizes::Detail as TxSizeDetail;

use crate::context::*;
use crate::partition::BlockSize::*;
use crate::partition::*;
use crate::tiling::*;
use crate::util::{self, *};

use crate::cpu_features::CpuFeatureLevel;
use TxSize::*;

mod forward;
mod inverse;

pub static RAV1E_TX_TYPES: &[TxType] = &[
  TxType::DCT_DCT,
  TxType::ADST_DCT,
  TxType::DCT_ADST,
  TxType::ADST_ADST,
  // TODO: Add a speed setting for FLIPADST
  // TxType::FLIPADST_DCT,
  // TxType::DCT_FLIPADST,
  // TxType::FLIPADST_FLIPADST,
  // TxType::ADST_FLIPADST,
  // TxType::FLIPADST_ADST,
  TxType::IDTX,
  TxType::V_DCT,
  TxType::H_DCT,
  //TxType::V_FLIPADST,
  //TxType::H_FLIPADST,
];

static SQRT2_BITS: usize = 12;
static SQRT2: i32 = 5793; // 2^12 * sqrt(2)
static INV_SQRT2: i32 = 2896; // 2^12 / sqrt(2)

pub const TX_TYPES: usize = 16;

/// Take a function generic over the transform tuple (Size, Type)
/// and call the correct one based on the *values* of (Size, Type).
///
/// It is assumed the `T: Transform<_>` parameter is first.
/// Errors in this macro will cause *GREAT* amounts of error message
/// spewage. Sorry :(
#[macro_export]
macro_rules! specialize_f {
  (@foreach_block ($tx_size:expr),
   ($tx_ty:ty, ($(($b_x:expr, $b_y:expr),)*),),
   $f:ident $args:tt ) =>
  {
    // generate match arms for each block with the specified TxType
    // we have to create the match arms (ie `$pattern => $expression`)
    // without using a macro. See https://github.com/rust-lang/rust/issues/12832
    paste::expr! {
      match $tx_size {
      $(
        <$crate::util::[<Block $b_x x $b_y>] as $crate::transform::tx_sizes::Detail>::TX_SIZE => {
          // call the function with this tuple.
          $f::<($crate::util::[<Block $b_x x $b_y>], $tx_ty), _> $args
        },
      )*
        _ => {
          unreachable!("unhandled (type, size): ({:?}, {:?})", $tx_ty, $tx_size);
        },
      }
    }
  };
  (@foreach_type ($tx_size:expr, $tx_type:expr),
   ($($tx_ty:ident,)*),
   $f:ident $args:tt ) =>
  {
    match $tx_type {
      $(<$crate::transform::tx_2d_types::$tx_ty as $crate::transform::tx_2d_types::Detail>::TX_TYPE => {
        $crate::specialize_f! { @foreach_block
          ($tx_size),
          ($crate::transform::tx_2d_types::$tx_ty,
            ((4, 4), (8, 8), (16, 16), (32, 32), (64, 64),
             (4, 8), (8, 16), (16, 32), (32, 64),
             (8, 4), (16, 8), (32, 16), (64, 32),
             (4, 16), (8, 32), (16, 64),
             (16, 4), (32, 8), (64, 16), ),
          ),
          $f $args
        }
      },)*
    }
  };
  // Instantiate $f for every (Size, Type) possible. This is a lot, and may not
  // be available due to additional trait bounds.
  (@forall $tx_size:expr, $tx_type:expr, $f:ident $args:tt) => {
      $crate::specialize_f! { @foreach_type
        ($tx_size, $tx_type),
        (DctDct, AdstDct, DctAdst, AdstAdst, FlipAdstDct,
         DctFlipAdst, FlipAdstFlipAdst, AdstFlipAdst,
         FlipAdstAdst, IdId, DctId, IdDct, AdstId, IdAdst,
         FlipAdstId, IdFlipAdst, ),
        $f $args
      }
  };
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord)]
#[repr(C)]
pub enum TxType {
  DCT_DCT = 0,   // DCT  in both horizontal and vertical
  ADST_DCT = 1,  // ADST in vertical, DCT in horizontal
  DCT_ADST = 2,  // DCT  in vertical, ADST in horizontal
  ADST_ADST = 3, // ADST in both directions
  FLIPADST_DCT = 4,
  DCT_FLIPADST = 5,
  FLIPADST_FLIPADST = 6,
  ADST_FLIPADST = 7,
  FLIPADST_ADST = 8,
  IDTX = 9,
  V_DCT = 10,
  H_DCT = 11,
  V_ADST = 12,
  H_ADST = 13,
  V_FLIPADST = 14,
  H_FLIPADST = 15,
}
pub mod tx_1d_types {
  pub trait Detail {
    const FLIPPED: bool;
    const TBL_IDX: usize;
  }

  macro_rules! d {
    ($($name:ident,)*) => {
      $(
        #[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
        pub struct $name;
      )*
    };
  }

  d!(Dct, Adst, FlipAdst, Id,);

  impl Detail for Dct {
    const FLIPPED: bool = false;
    const TBL_IDX: usize = 1;
  }
  impl Detail for Adst {
    const FLIPPED: bool = false;
    const TBL_IDX: usize = 2;
  }
  impl Detail for FlipAdst {
    const FLIPPED: bool = true;
    const TBL_IDX: usize = 3;
  }
  impl Detail for Id {
    const FLIPPED: bool = false;
    const TBL_IDX: usize = 0;
  }
}
pub mod tx_2d_types {
  use super::tx_1d_types;
  use crate::transform::TxType;

  pub trait Detail {
    const TX_TYPE: TxType;

    type Col: tx_1d_types::Detail;
    type Row: tx_1d_types::Detail;
  }

  macro_rules! d {
    ($(($name:ident, $tx_type:ident, $col:ident, $row:ident, ),)*) => {
      $(
        #[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
        pub struct $name;

        impl Detail for $name {
          const TX_TYPE: TxType = TxType::$tx_type;

          type Col = tx_1d_types::$col;
          type Row = tx_1d_types::$row;
        }
      )*
    };
  }

  d!(
    (DctDct, DCT_DCT, Dct, Dct,),
    (AdstDct, ADST_DCT, Adst, Dct,),
    (DctAdst, DCT_ADST, Dct, Adst,),
    (AdstAdst, ADST_ADST, Adst, Adst,),
    (FlipAdstDct, FLIPADST_DCT, FlipAdst, Dct,),
    (DctFlipAdst, DCT_FLIPADST, Dct, FlipAdst,),
    (FlipAdstFlipAdst, FLIPADST_FLIPADST, FlipAdst, FlipAdst,),
    (AdstFlipAdst, ADST_FLIPADST, Adst, FlipAdst,),
    (FlipAdstAdst, FLIPADST_ADST, FlipAdst, Adst,),
    (IdId, IDTX, Id, Id,),
    (DctId, V_DCT, Dct, Id,),
    (IdDct, H_DCT, Id, Dct,),
    (AdstId, V_ADST, Adst, Id,),
    (IdAdst, H_ADST, Id, Adst,),
    (FlipAdstId, V_FLIPADST, FlipAdst, Id,),
    (IdFlipAdst, H_FLIPADST, Id, FlipAdst,),
  );
}

/// Transform Size
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Eq, Ord)]
#[repr(C)]
pub enum TxSize {
  TX_4X4,
  TX_8X8,
  TX_16X16,
  TX_32X32,
  TX_64X64,

  TX_4X8,
  TX_8X4,
  TX_8X16,
  TX_16X8,
  TX_16X32,
  TX_32X16,
  TX_32X64,
  TX_64X32,

  TX_4X16,
  TX_16X4,
  TX_8X32,
  TX_32X8,
  TX_16X64,
  TX_64X16,
}

impl TxSize {
  /// Number of square transform sizes
  pub const TX_SIZES: usize = 5;

  /// Number of transform sizes (including non-square sizes)
  pub const TX_SIZES_ALL: usize = 14 + 5;

  pub fn width(self) -> usize {
    1 << self.width_log2()
  }

  pub fn width_log2(self) -> usize {
    match self {
      TX_4X4 | TX_4X8 | TX_4X16 => 2,
      TX_8X8 | TX_8X4 | TX_8X16 | TX_8X32 => 3,
      TX_16X16 | TX_16X8 | TX_16X32 | TX_16X4 | TX_16X64 => 4,
      TX_32X32 | TX_32X16 | TX_32X64 | TX_32X8 => 5,
      TX_64X64 | TX_64X32 | TX_64X16 => 6,
    }
  }

  pub fn width_index(self) -> usize {
    self.width_log2() - TX_4X4.width_log2()
  }

  pub fn height(self) -> usize {
    1 << self.height_log2()
  }

  pub fn height_log2(self) -> usize {
    match self {
      TX_4X4 | TX_8X4 | TX_16X4 => 2,
      TX_8X8 | TX_4X8 | TX_16X8 | TX_32X8 => 3,
      TX_16X16 | TX_8X16 | TX_32X16 | TX_4X16 | TX_64X16 => 4,
      TX_32X32 | TX_16X32 | TX_64X32 | TX_8X32 => 5,
      TX_64X64 | TX_32X64 | TX_16X64 => 6,
    }
  }

  pub fn height_index(self) -> usize {
    self.height_log2() - TX_4X4.height_log2()
  }

  pub fn width_mi(self) -> usize {
    self.width() >> MI_SIZE_LOG2
  }

  pub fn area(self) -> usize {
    1 << self.area_log2()
  }

  pub fn area_log2(self) -> usize {
    self.width_log2() + self.height_log2()
  }

  pub fn height_mi(self) -> usize {
    self.height() >> MI_SIZE_LOG2
  }

  pub fn block_size(self) -> BlockSize {
    match self {
      TX_4X4 => BLOCK_4X4,
      TX_8X8 => BLOCK_8X8,
      TX_16X16 => BLOCK_16X16,
      TX_32X32 => BLOCK_32X32,
      TX_64X64 => BLOCK_64X64,
      TX_4X8 => BLOCK_4X8,
      TX_8X4 => BLOCK_8X4,
      TX_8X16 => BLOCK_8X16,
      TX_16X8 => BLOCK_16X8,
      TX_16X32 => BLOCK_16X32,
      TX_32X16 => BLOCK_32X16,
      TX_32X64 => BLOCK_32X64,
      TX_64X32 => BLOCK_64X32,
      TX_4X16 => BLOCK_4X16,
      TX_16X4 => BLOCK_16X4,
      TX_8X32 => BLOCK_8X32,
      TX_32X8 => BLOCK_32X8,
      TX_16X64 => BLOCK_16X64,
      TX_64X16 => BLOCK_64X16,
    }
  }

  pub fn sqr(self) -> TxSize {
    match self {
      TX_4X4 | TX_4X8 | TX_8X4 | TX_4X16 | TX_16X4 => TX_4X4,
      TX_8X8 | TX_8X16 | TX_16X8 | TX_8X32 | TX_32X8 => TX_8X8,
      TX_16X16 | TX_16X32 | TX_32X16 | TX_16X64 | TX_64X16 => TX_16X16,
      TX_32X32 | TX_32X64 | TX_64X32 => TX_32X32,
      TX_64X64 => TX_64X64,
    }
  }

  pub fn sqr_up(self) -> TxSize {
    match self {
      TX_4X4 => TX_4X4,
      TX_8X8 | TX_4X8 | TX_8X4 => TX_8X8,
      TX_16X16 | TX_8X16 | TX_16X8 | TX_4X16 | TX_16X4 => TX_16X16,
      TX_32X32 | TX_16X32 | TX_32X16 | TX_8X32 | TX_32X8 => TX_32X32,
      TX_64X64 | TX_32X64 | TX_64X32 | TX_16X64 | TX_64X16 => TX_64X64,
    }
  }

  pub fn by_dims(w: usize, h: usize) -> TxSize {
    match (w, h) {
      (4, 4) => TX_4X4,
      (8, 8) => TX_8X8,
      (16, 16) => TX_16X16,
      (32, 32) => TX_32X32,
      (64, 64) => TX_64X64,
      (4, 8) => TX_4X8,
      (8, 4) => TX_8X4,
      (8, 16) => TX_8X16,
      (16, 8) => TX_16X8,
      (16, 32) => TX_16X32,
      (32, 16) => TX_32X16,
      (32, 64) => TX_32X64,
      (64, 32) => TX_64X32,
      (4, 16) => TX_4X16,
      (16, 4) => TX_16X4,
      (8, 32) => TX_8X32,
      (32, 8) => TX_32X8,
      (16, 64) => TX_16X64,
      (64, 16) => TX_64X16,
      _ => unreachable!(),
    }
  }

  pub fn is_rect(self) -> bool {
    self.width_log2() != self.height_log2()
  }
}

pub mod tx_sizes {
  use crate::context::MI_SIZE_LOG2;
  use crate::predict::Dim;
  use crate::transform::TxSize;
  use crate::util::*;

  pub trait Detail: Dim + Block {
    const TX_SIZE: TxSize;

    const WIDTH_MI: usize = Self::WIDTH >> MI_SIZE_LOG2;
    const WIDTH_INDEX: usize =
      Self::WIDTH_LOG2 - <Block4x4 as Block>::WIDTH_LOG2;
    const HEIGHT_MI: usize = Self::HEIGHT >> MI_SIZE_LOG2;
    const HEIGHT_INDEX: usize =
      Self::HEIGHT_LOG2 - <Block4x4 as Block>::HEIGHT_LOG2;

    fn width_index(&self) -> usize {
      Self::WIDTH_INDEX
    }
    fn width_mi(&self) -> usize {
      Self::WIDTH_MI
    }
    fn height_index(&self) -> usize {
      Self::HEIGHT_INDEX
    }
    fn height_mi(&self) -> usize {
      Self::HEIGHT_MI
    }
  }

  // manually impl because by default `*_INDEX` depends on this.
  impl Detail for Block4x4 {
    const TX_SIZE: TxSize = TxSize::TX_4X4;

    const WIDTH_INDEX: usize = 0;
    const HEIGHT_INDEX: usize = 0;
  }

  macro_rules! block_detail {
    ($W:expr, $H:expr) => {
      paste::item! {
        impl Detail for [<Block $W x $H>] {
          const TX_SIZE: TxSize = TxSize::[<TX_ $W X $H>];
        }
      }
    };
  }

  macro_rules! blocks_detail {
    ($(($W:expr, $H:expr)),+) => {
      $(
        block_detail! { $W, $H }
      )*
    };
  }

  blocks_detail! { (8, 8), (16, 16), (32, 32), (64, 64) }
  blocks_detail! { (4, 8), (8, 16), (16, 32), (32, 64) }
  blocks_detail! { (8, 4), (16, 8), (32, 16), (64, 32) }
  blocks_detail! { (4, 16), (8, 32), (16, 64) }
  blocks_detail! { (16, 4), (32, 8), (64, 16) }
}

#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub enum TxSet {
  // DCT only
  TX_SET_DCTONLY,
  // DCT + Identity only
  TX_SET_DCT_IDTX,
  // Discrete Trig transforms w/o flip (4) + Identity (1)
  TX_SET_DTT4_IDTX,
  // Discrete Trig transforms w/o flip (4) + Identity (1) + 1D Hor/vert DCT (2)
  // for 16x16 only
  TX_SET_DTT4_IDTX_1DDCT_16X16,
  // Discrete Trig transforms w/o flip (4) + Identity (1) + 1D Hor/vert DCT (2)
  TX_SET_DTT4_IDTX_1DDCT,
  // Discrete Trig transforms w/ flip (9) + Identity (1)
  TX_SET_DTT9_IDTX,
  // Discrete Trig transforms w/ flip (9) + Identity (1) + 1D Hor/Ver DCT (2)
  TX_SET_DTT9_IDTX_1DDCT,
  // Discrete Trig transforms w/ flip (9) + Identity (1) + 1D Hor/Ver (6)
  // for 16x16 only
  TX_SET_ALL16_16X16,
  // Discrete Trig transforms w/ flip (9) + Identity (1) + 1D Hor/Ver (6)
  TX_SET_ALL16,
}

/// Utility function that returns the log of the ratio of the col and row sizes.
#[inline]
pub fn get_rect_tx_log_ratio(col: usize, row: usize) -> i8 {
  debug_assert!(col > 0 && row > 0);
  col.ilog() as i8 - row.ilog() as i8
}

// performs half a butterfly
#[inline]
fn half_btf(w0: i32, in0: i32, w1: i32, in1: i32, bit: usize) -> i32 {
  // Ensure defined behaviour for when w0*in0 + w1*in1 is negative and
  //   overflows, but w0*in0 + w1*in1 + rounding isn't.
  let result = (w0 * in0).wrapping_add(w1 * in1);
  // Implement a version of round_shift with wrapping
  if bit == 0 {
    result
  } else {
    result.wrapping_add(1 << (bit - 1)) >> bit
  }
}

// clamps value to a signed integer type of bit bits
#[inline]
fn clamp_value(value: i32, bit: usize) -> i32 {
  let max_value: i32 = ((1i64 << (bit - 1)) - 1) as i32;
  let min_value: i32 = (-(1i64 << (bit - 1))) as i32;
  clamp(value, min_value, max_value)
}

#[inline]
fn round_shift_array<T>(arr: &mut [i32], size: usize, bit: i8)
where
  T: ISimd<i32>,
{
  if bit == 0 {
    return;
  }
  debug_assert!(size <= arr.len(), "{} > {}", size, arr.len());
  debug_assert_eq!(size % T::LANES, 0);

  let arr = T::slice_cast_mut(arr);

  if bit > 0 {
    let bit = bit as u32;
    for v in arr.iter_mut() {
      *v = v.round_shift(bit);
    }
  } else {
    for v in arr.iter_mut() {
      *v <<= T::_splat(-bit as _);
    }
  }
}

#[derive(Debug, Clone, Copy)]
enum TxType1D {
  DCT,
  ADST,
  FLIPADST,
  IDTX,
}

pub trait Transform<P>
where
  P: Pixel,
{
  type Size: tx_sizes::Detail;
  type Type: tx_2d_types::Detail;

  type Col: tx_1d_types::Detail;
  type Row: tx_1d_types::Detail;
}

impl<S, T, P> Transform<P> for (S, T)
where
  P: Pixel,
  S: tx_sizes::Detail,
  T: tx_2d_types::Detail,
{
  type Size = S;
  type Type = T;

  type Col = <T as tx_2d_types::Detail>::Col;
  type Row = <T as tx_2d_types::Detail>::Row;
}

#[inline(never)]
pub fn forward_transform<T, P>(
  residual: &mut [i16], coeffs: &mut [i32], bit_depth: usize, _: P,
) where
  T: FwdTxfm2D<P> + InvTxfm2D<P>,
  P: Pixel,
  T::Size: FwdBlock + InvBlock,
  (T::Col, <T::Size as util::Block>::Vert): FwTxDetail + InvTxDetail,
  (T::Row, <T::Size as util::Block>::Hori): FwTxDetail + InvTxDetail,
{
  T::fht(residual, coeffs, bit_depth);
}
#[inline(never)]
pub fn inverse_transform_add<T, P>(
  rcoeffs: &mut [i32], plane: &mut PlaneRegionMut<'_, P>,
  bit_depth: usize,
  cpu: CpuFeatureLevel,
) where
  T: FwdTxfm2D<P> + InvTxfm2D<P>,
  P: Pixel,
  T::Size: FwdBlock + InvBlock,
  (T::Col, <T::Size as util::Block>::Vert): FwTxDetail + InvTxDetail,
  (T::Row, <T::Size as util::Block>::Hori): FwTxDetail + InvTxDetail,
{
  <T as InvTxfm2D<P>>::inv_txfm2d_add(rcoeffs, plane, bit_depth, cpu);
}

#[cfg(test)]
mod test {
  use super::TxType::*;
  use super::*;
  use crate::frame::*;
  use rand::random;

  fn test_roundtrip<T: Pixel>(
    tx_size: TxSize, tx_type: TxType, tolerance: i16,
  ) {
    let mut src_storage = AlignedArray::new([T::cast_from(0); 64 * 64]);
    let src = &mut src_storage[..tx_size.area()];
    let mut dst =
      Plane::wrap(vec![T::cast_from(0); tx_size.area()], tx_size.width());
    let mut res_storage = AlignedArray::new([0i16; 64 * 64]);
    let res = &mut res_storage[..tx_size.area()];
    let mut freq_storage = AlignedArray::new([0i32; 64 * 64]);
    let freq = &mut freq_storage[..tx_size.area()];
    for ((r, s), d) in
      res.iter_mut().zip(src.iter_mut()).zip(dst.data.iter_mut())
    {
      *s = T::cast_from(random::<u8>());
      *d = T::cast_from(random::<u8>());
      *r = i16::cast_from(*s) - i16::cast_from(*d);
    }
    specialize_f!(@forall tx_size, tx_type, forward_transform
                  (res, freq, 8, T::cast_from(0)));
    specialize_f!(@forall tx_size, tx_type, inverse_transform_add
                  (freq, &mut dst.as_region_mut(), 8,
                   CpuFeatureLevel::default()));

    let ne = src
      .iter()
      .zip(dst.data.iter())
      .enumerate()
      .filter(|(_, (&s, &d))| {
        i16::abs(i16::cast_from(s) - i16::cast_from(d)) > tolerance
      })
      .map(|v| format!("{:?}", v))
      .collect::<Vec<_>>();
    if ne.len() != 0 {
      eprintln!(
        "tx_size = {:?}, tx_type = {:?}, tolerance = {}",
        tx_size, tx_type, tolerance
      );
      eprintln!("roundtrip mismatch: {:#?}", ne);
      panic!();
    }
  }

  #[test]
  fn log_tx_ratios() {
    let combinations = [
      (TxSize::TX_4X4, 0),
      (TxSize::TX_8X8, 0),
      (TxSize::TX_16X16, 0),
      (TxSize::TX_32X32, 0),
      (TxSize::TX_64X64, 0),
      (TxSize::TX_4X8, -1),
      (TxSize::TX_8X4, 1),
      (TxSize::TX_8X16, -1),
      (TxSize::TX_16X8, 1),
      (TxSize::TX_16X32, -1),
      (TxSize::TX_32X16, 1),
      (TxSize::TX_32X64, -1),
      (TxSize::TX_64X32, 1),
      (TxSize::TX_4X16, -2),
      (TxSize::TX_16X4, 2),
      (TxSize::TX_8X32, -2),
      (TxSize::TX_32X8, 2),
      (TxSize::TX_16X64, -2),
      (TxSize::TX_64X16, 2),
    ];

    for &(tx_size, expected) in combinations.iter() {
      println!(
        "Testing combination {:?}, {:?}",
        tx_size.width(),
        tx_size.height()
      );
      assert!(
        get_rect_tx_log_ratio(tx_size.width(), tx_size.height()) == expected
      );
    }
  }

  fn roundtrips<T: Pixel>() {
    let combinations = [
      (TX_4X4, DCT_DCT, 0),
      (TX_4X4, ADST_DCT, 0),
      (TX_4X4, DCT_ADST, 0),
      (TX_4X4, ADST_ADST, 0),
      (TX_4X4, IDTX, 0),
      (TX_4X4, V_DCT, 0),
      (TX_4X4, H_DCT, 0),
      (TX_4X4, V_ADST, 0),
      (TX_4X4, H_ADST, 0),
      (TX_8X8, DCT_DCT, 1),
      (TX_8X8, ADST_DCT, 1),
      (TX_8X8, DCT_ADST, 1),
      (TX_8X8, ADST_ADST, 1),
      (TX_8X8, IDTX, 0),
      (TX_8X8, V_DCT, 0),
      (TX_8X8, H_DCT, 0),
      (TX_8X8, V_ADST, 0),
      (TX_8X8, H_ADST, 1),
      (TX_16X16, DCT_DCT, 1),
      (TX_16X16, ADST_DCT, 1),
      (TX_16X16, DCT_ADST, 1),
      (TX_16X16, ADST_ADST, 1),
      (TX_16X16, IDTX, 0),
      (TX_16X16, V_DCT, 1),
      (TX_16X16, H_DCT, 1),
      // 32x tranforms only use DCT_DCT and IDTX
      (TX_32X32, DCT_DCT, 2),
      (TX_32X32, IDTX, 0),
      // 64x tranforms only use DCT_DCT and IDTX
      //(TX_64X64, DCT_DCT, 0),
      (TX_4X8, DCT_DCT, 1),
      (TX_8X4, DCT_DCT, 1),
      (TX_4X16, DCT_DCT, 1),
      (TX_16X4, DCT_DCT, 1),
      (TX_8X16, DCT_DCT, 1),
      (TX_16X8, DCT_DCT, 1),
      (TX_8X32, DCT_DCT, 2),
      (TX_32X8, DCT_DCT, 2),
      (TX_16X32, DCT_DCT, 2),
      (TX_32X16, DCT_DCT, 2),
    ];
    for &(tx_size, tx_type, tolerance) in combinations.iter() {
      println!("Testing combination {:?}, {:?}", tx_size, tx_type);
      test_roundtrip::<T>(tx_size, tx_type, tolerance);
    }
  }

  #[test]
  fn roundtrips_u8() {
    roundtrips::<u8>();
  }

  #[test]
  fn roundtrips_u16() {
    roundtrips::<u16>();
  }
}
