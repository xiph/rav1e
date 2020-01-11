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
  eob: usize, width: usize, height: usize, bd: usize,
) {
  debug_assert!(bd == 8);

  // Only use at most 32 columns and 32 rows of input coefficients.
  let input: &[T::Coeff] = &input[..width.min(32) * height.min(32)];

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
      eob as i32,
    );
  }
}

#[cfg(test)]
pub mod test {
  use super::*;
  use crate::context::av1_get_coded_tx_size;
  use crate::cpu_features::CpuFeatureLevel;
  use crate::frame::{AsRegion, Plane};
  use crate::scan_order::av1_scan_orders;
  use crate::transform::*;
  use rand::{random, thread_rng, Rng};

  pub fn pick_eob<T: Coefficient>(
    coeffs: &mut [T], tx_size: TxSize, tx_type: TxType, sub_h: usize,
  ) -> usize {
    /* From dav1d
     * copy the topleft coefficients such that the return value (being the
     * coefficient scantable index for the eob token) guarantees that only
     * the topleft $sub out of $sz (where $sz >= $sub) coefficients in both
     * dimensions are non-zero. This leads to braching to specific optimized
     * simd versions (e.g. dc-only) so that we get full asm coverage in this
     * test */
    let coeff_h = av1_get_coded_tx_size(tx_size).height();
    let sub_high: usize = if sub_h > 0 { sub_h * 8 - 1 } else { 0 };
    let sub_low: usize = if sub_h > 1 { sub_high - 8 } else { 0 };
    let mut eob = 0;
    let mut exit = 0;

    let scan = av1_scan_orders[tx_size as usize][tx_type as usize].scan;

    for (i, &pos) in scan.iter().enumerate() {
      exit = i;

      let rc = pos as usize;
      let rcx = rc % coeff_h;
      let rcy = rc / coeff_h;

      if rcx > sub_high || rcy > sub_high {
        break;
      } else if eob == 0 && (rcx > sub_low || rcy > sub_low) {
        eob = i;
      }
    }

    if eob != 0 {
      eob += thread_rng().gen_range(0, (exit - eob).min(1));
    }
    for &pos in scan.iter().skip(eob + 1) {
      coeffs[pos as usize] = T::cast_from(0);
    }

    eob
  }

  pub fn test_transform(
    tx_size: TxSize, tx_type: TxType, cpu: CpuFeatureLevel,
  ) {
    let sub_h_iterations: usize = match tx_size.height().max(tx_size.width()) {
      4 => 2,
      8 => 2,
      16 => 3,
      32 | 64 => 4,
      _ => unreachable!(),
    };

    for sub_h in 0..sub_h_iterations {
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

      let eob: usize = pick_eob(freq, tx_size, tx_type, sub_h);
      let mut native_dst = dst.clone();

      inverse_transform_add(
        freq,
        &mut dst.as_region_mut(),
        eob,
        tx_size,
        tx_type,
        8,
        cpu,
      );
      inverse_transform_add(
        freq,
        &mut native_dst.as_region_mut(),
        eob,
        tx_size,
        tx_type,
        8,
        CpuFeatureLevel::NATIVE,
      );
      assert_eq!(native_dst.data_origin(), dst.data_origin());
    }
  }
}
