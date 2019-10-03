//! Simd helpers

use std::mem::size_of;
use std::ops::*;

use packed_simd::*;

/// Some functions which "all" simd types implement
pub trait ISimd<E>: Sized + Copy
where
  Self: Add<Output = Self> + Add<E, Output = Self>,
  Self: Div<Output = Self> + Div<E, Output = Self>,
  Self: Mul<Output = Self> + Mul<E, Output = Self>,
  Self: Sub<Output = Self> + Sub<E, Output = Self>,
  Self: Shl<Output = Self> + Shl<u32, Output = Self>,
  Self: ShlAssign,
  Self: Shr<Output = Self> + Shr<u32, Output = Self>,
  Self: ShrAssign,
{
  const LANES: usize;
  const LOG2_LANES: usize;

  fn _splat(v: E) -> Self;

  // assumes the slice is properly aligned and has enough elements.
  fn load_from_slice(slice: &[E]) -> Self;
  fn load_from_slice_u(slice: &[E]) -> Self; // unaligned
  fn store_to_slice(self, slice: &mut [E]);
  fn store_to_slice_u(self, slice: &mut [E]);

  fn slice_cast_ref(slice: &[E]) -> &[Self] {
    let ptr = slice.as_ptr();
    debug_assert_eq!(ptr as usize % (size_of::<E>() * Self::LANES), 0);
    let len = slice.len() >> Self::LOG2_LANES;

    unsafe { ::std::slice::from_raw_parts(ptr as *const Self, len) }
  }
  fn slice_cast_mut(slice: &mut [E]) -> &mut [Self] {
    let ptr = slice.as_ptr();
    debug_assert_eq!(ptr as usize % (size_of::<E>() * Self::LANES), 0);
    let len = slice.len() >> Self::LOG2_LANES;

    unsafe { ::std::slice::from_raw_parts_mut(ptr as *mut Self, len) }
  }

  fn abs(self) -> Self;
  fn clamp(self, min: E, max: E) -> Self;
  fn round_shift(self, bit: u32) -> Self;
}
/// ugh. none of these functions are generic on anything.
macro_rules! impl_isimd {
  ($(($elem:ident, $signed:expr, ($($lanes:expr, )*),),)*) => {$($(

impl ISimd<$elem> for Simd<[$elem; 1 << $lanes]> {
  const LANES: usize = 1 << $lanes;
  const LOG2_LANES: usize = $lanes;

  #[inline]
  fn _splat(v: $elem) -> Self {
    Self::splat(v)
  }

  #[inline]
  fn load_from_slice(slice: &[$elem]) -> Self {
    Self::from_slice_aligned(slice)
  }
  #[inline]
  fn load_from_slice_u(slice: &[$elem]) -> Self {
    Self::from_slice_unaligned(slice)
  }
  #[inline]
  fn store_to_slice(self, slice: &mut [$elem]) {
    self.write_to_slice_aligned(slice)
  }
  #[inline]
  fn store_to_slice_u(self, slice: &mut [$elem]) {
    self.write_to_slice_unaligned(slice)
  }

  #[inline(always)]
  fn abs(self) -> Self {
    const SIGNED_BITS: u32 = (size_of::<$elem>() * 8 - 1) as _;
    let mut t = self;

    if $signed {
      let m = t >> SIGNED_BITS;
      t = (t + m) ^ m;
    }

    t
  }
  #[inline(always)]
  fn clamp(self, min: $elem, max: $elem) -> Self {
    self.max(Self::splat(min))
      .min(Self::splat(max))
  }
  #[inline(always)]
  fn round_shift(self, bit: u32) -> Self {
    (self + Self::splat((1 << bit >> 1) as _)) >> bit
  }
}

  )*)*}
}
impl_isimd! {
  (i8,    true, (1, 2, 3, 4, 5, 6, ), ),
  (i16,   true, (1, 2, 3, 4, 5, ), ),
  (i32,   true, (1, 2, 3, 4, ), ),
  (i64,   true, (1, 2, 3, ), ),
  (isize, true, (1, 2, 3, ), ),

  (u8,    false, (1, 2, 3, 4, 5, 6, ), ),
  (u16,   false, (1, 2, 3, 4, 5, ), ),
  (u32,   false, (1, 2, 3, 4, ), ),
  (u64,   false, (1, 2, 3, ), ),
  (usize, false, (1, 2, 3, ), ),
}

pub trait SGSimd<E>: ISimd<E>
where
  E: Sized,
{
  /// Needed because of compiler limitations.
  #[doc(hidden)]
  type SgOffsets;
  /// Needed because of compiler limitations.
  #[doc(hidden)]
  type SgMask;

  unsafe fn gather_ptr(
    ptr: *const E, len: usize, offsets: Self::SgOffsets,
  ) -> Self {
    let slice = ::std::slice::from_raw_parts(ptr, len);
    Self::gather(slice, offsets)
  }

  fn gather(slice: &[E], offsets: Self::SgOffsets) -> Self;
  fn masked_gather(
    self, slice: &[E], mask: Self::SgMask, offsets: Self::SgOffsets,
  ) -> Self;

  fn scatter(self, slice: &mut [E], offsets: Self::SgOffsets);
}
macro_rules! impl_sgsimd {
  ($(($elem:ident, $signed:expr, ($($lanes:expr, )*),),)*) => {$($(

impl SGSimd<$elem> for Simd<[$elem; 1 << $lanes]> {
  #[doc(hidden)]
  type SgOffsets = Simd<[usize; 1 << $lanes]>;
  #[doc(hidden)]
  type SgMask = Simd<[m8; 1 << $lanes]>;
  #[inline]
  fn gather(slice: &[$elem], offsets: Simd<[usize; 1 << $lanes]>)
    -> Self
  {
    type SimdPtrs = Simd<[*const $elem; 1 << $lanes]>;
    type SimdUsize = Simd<[usize; 1 << $lanes]>;

    let ptrs = SimdPtrs::splat(slice.as_ptr());
    let ptrs = unsafe { ptrs.add(offsets) };
    // create the mask:
    let len = SimdUsize::splat(slice.len());
    let mask = offsets.lt(len);

    unsafe { ptrs.read(mask, Default::default()) }
  }
  #[inline]
  fn masked_gather(self, slice: &[$elem], mask: Simd<[m8; 1 << $lanes]>,
                   offsets: Simd<[usize; 1 << $lanes]>)
    -> Self
  {
    type SimdPtrs = Simd<[*const $elem; 1 << $lanes]>;
    type SimdUsize = Simd<[usize; 1 << $lanes]>;

    let ptrs = SimdPtrs::splat(slice.as_ptr());
    let ptrs = unsafe { ptrs.add(offsets) };
    // create the mask:
    let len = SimdUsize::splat(slice.len());
    let offsets_mask: Simd<[m8; 1 << $lanes]> = offsets
      .lt(len)
      .cast();
    let mask = mask & offsets_mask;

    unsafe { ptrs.read(mask, self) }
  }
  #[inline]
  fn scatter(self, slice: &mut [$elem],
             offsets: Simd<[usize; 1 << $lanes]>) {
    type SimdPtrs = Simd<[*mut $elem; 1 << $lanes]>;
    type SimdUsize = Simd<[usize; 1 << $lanes]>;

    let ptrs = SimdPtrs::splat(slice.as_mut_ptr());
    let ptrs = unsafe { ptrs.add(offsets) };
    // create the mask:
    let len = SimdUsize::splat(slice.len());
    let mask = offsets.lt(len);

    unsafe { ptrs.write(mask, self) }
  }
}

  )*)*}
}
impl_sgsimd! {
  (i8,    true, (1, 2, 3, ), ),
  (i16,   true, (1, 2, 3, ), ),
  (i32,   true, (1, 2, 3, ), ),
  (i64,   true, (1, 2, 3, ), ),
  (isize, true, (1, 2, 3, ), ),

  (u8,    false, (1, 2, 3, ), ),
  (u16,   false, (1, 2, 3, ), ),
  (u32,   false, (1, 2, 3, ), ),
  (u64,   false, (1, 2, 3, ), ),
  (usize, false, (1, 2, 3, ), ),
}
