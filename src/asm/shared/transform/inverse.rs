// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#[cfg(test)]
pub mod test {
  use crate::cpu_features::CpuFeatureLevel;
  use crate::frame::{AsRegion, Plane};
  use crate::tiling::PlaneRegionMut;
  use crate::transform::*;
  use crate::util::*;
  use rand::random;

  pub type TestedInverseFn = unsafe fn(
    input: &[i16],
    output: &mut PlaneRegionMut<'_, u8>,
    tx_type: TxType,
    bd: usize,
  );

  pub type RefInverseFn = fn(
    input: &[i16],
    output: &mut PlaneRegionMut<'_, u8>,
    tx_type: TxType,
    bd: usize,
    cpu: CpuFeatureLevel,
  );

  pub fn test_transform(
    tx_size: TxSize, tx_type: TxType, ref_func: RefInverseFn,
    test_func: TestedInverseFn,
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

    unsafe {
      test_func(freq, &mut dst.as_region_mut(), tx_type, 8);
    }
    ref_func(
      freq,
      &mut native_dst.as_region_mut(),
      tx_type,
      8,
      CpuFeatureLevel::NATIVE,
    );
    assert_eq!(native_dst.data_origin(), dst.data_origin());
  }
}
