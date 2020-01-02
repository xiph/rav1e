// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::tiling::PlaneRegionMut;
use crate::util::*;

// Note: Input coeffs are mutable since the assembly uses them as a scratchpad
pub type InvTxfmFunc =
  unsafe extern fn(*mut u8, libc::ptrdiff_t, *mut i16, i32);

pub fn call_inverse_func<T: Pixel>(
  func: InvTxfmFunc, input: &[T::Coeff], output: &mut PlaneRegionMut<'_, T>,
  width: usize, height: usize, bd: usize,
) {
  debug_assert!(bd == 8);

  // 64x only uses 32 coeffs
  let area = width.min(32) * height.min(32);

  // Only use at most 32 columns and 32 rows of input coefficients.
  let input: &[T::Coeff] = &input[..area];

  let mut copied: AlignedArray<[T::Coeff; 32 * 32]> =
    AlignedArray::uninitialized();

  // Convert input to 16-bits.
  // TODO: Remove by changing inverse assembly to not overwrite its input
  for (a, b) in copied.array.iter_mut().zip(input) {
    *a = *b;
  }

  // perform the inverse transform
  unsafe {
    func(
      output.data_ptr_mut() as *mut _,
      output.plane_cfg.stride as isize,
      copied.array.as_mut_ptr() as *mut _,
      area as i32,
    );
  }
}

#[cfg(test)]
pub mod test {
  use super::*;
  use crate::cpu_features::CpuFeatureLevel;
  use crate::frame::{AsRegion, Plane};
  use crate::transform::*;
  use rand::random;

  pub fn test_transform(
    tx_size: TxSize, tx_type: TxType, cpu: CpuFeatureLevel,
  ) {
    let mut src_storage = [0u8; 64 * 64];
    let src = &mut src_storage[..tx_size.area()];
    let mut dst = Plane::wrap(vec![0u8; tx_size.area()], tx_size.width());
    let mut res_storage: AlignedArray<[i16; 64 * 64]> =
      AlignedArray::uninitialized();
    let res = &mut res_storage.array[..tx_size.area()];
    let mut freq_storage: AlignedArray<[i16; 64 * 64]> =
      AlignedArray::uninitialized();
    let freq = &mut freq_storage.array[..tx_size.area()];
    for ((r, s), d) in
      res.iter_mut().zip(src.iter_mut()).zip(dst.data.iter_mut())
    {
      *s = random::<u8>();
      *d = random::<u8>();
      *r = i16::from(*s) - i16::from(*d);
    }
    forward_transform(
      res,
      freq,
      tx_size.width(),
      tx_size,
      tx_type,
      8,
      CpuFeatureLevel::NATIVE,
    );
    let mut native_dst = dst.clone();

    inverse_transform_add(
      freq,
      &mut dst.as_region_mut(),
      tx_size,
      tx_type,
      8,
      cpu,
    );
    inverse_transform_add(
      freq,
      &mut native_dst.as_region_mut(),
      tx_size,
      tx_type,
      8,
      CpuFeatureLevel::NATIVE,
    );
    assert_eq!(native_dst.data_origin(), dst.data_origin());
  }
}
