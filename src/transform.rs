// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

extern crate libc;

use partition::TxSize;
use partition::TxType;

// In libaom, functions that have more than one specialization use function
// pointers, so we need to declare them as static fields and call them
// indirectly. Otherwise, we call SSE or C variants directly. To fully
// understand what's going on here you should first understand the Perl code
// in libaom that generates these function bindings.

#[repr(C)]
struct TxfmParam {
  // for both forward and inverse transforms
  tx_type: TxType,
  tx_size: TxSize,
  lossless: libc::c_int,
  bd: libc::c_int,
  // are the pixel buffers octets or shorts?  This should collapse to
  // bd==8 implies !is_hbd, but that's not certain right now.
  is_hbd: libc::c_int,
  //tx_set_type: TxSetType,
  tx_set_type: libc::c_int, // hack: I don't know why we need set type?
  // for inverse transforms only
  eob: libc::c_int
}

#[cfg(target_feature = "sse2")]
extern {
  fn av1_fht4x4_sse2(
    input: *const i16, output: *mut i32, stride: libc::c_int,
    tx_type: *const libc::c_int
  );
  fn av1_fht8x8_sse2(
    input: *const i16, output: *mut i32, stride: libc::c_int,
    tx_type: *const libc::c_int
  );
}

#[cfg(target_feature = "sse2")]
use self::av1_fht4x4_sse2 as av1_fht4x4;
#[cfg(target_feature = "sse2")]
use self::av1_fht8x8_sse2 as av1_fht8x8;

#[cfg(not(target_feature = "sse2"))]
extern {
  fn av1_fht4x4_c(
    input: *const i16, output: *mut i32, stride: libc::c_int,
    tx_type: *const libc::c_int
  );
  fn av1_fht8x8_c(
    input: *const i16, output: *mut i32, stride: libc::c_int,
    tx_type: *const libc::c_int
  );
}

#[cfg(not(target_feature = "sse2"))]
use self::av1_fht4x4_c as av1_fht4x4;
#[cfg(not(target_feature = "sse2"))]
use self::av1_fht8x8_c as av1_fht8x8;

extern "C" {
  static av1_inv_txfm2d_add_4x4: extern fn(
    input: *const i32,
    output: *mut u16,
    stride: libc::c_int,
    tx_type: libc::c_int,
    bd: libc::c_int
  );
  fn av1_inv_txfm2d_add_4x4_c(
    input: *const i32,
    output: *mut u16,
    stride: libc::c_int,
    tx_type: libc::c_int,
    bd: libc::c_int
  );
  static av1_inv_txfm2d_add_8x8: extern fn(
    input: *const i32,
    output: *mut u16,
    stride: libc::c_int,
    tx_type: libc::c_int,
    bd: libc::c_int
  ) -> ();
  fn av1_inv_txfm2d_add_8x8_c(
    input: *const i32,
    output: *mut u16,
    stride: libc::c_int,
    tx_type: libc::c_int,
    bd: libc::c_int
  ) -> ();
  fn av1_fht16x16_c(
    input: *const i16,
    output: *mut i32,
    stride: libc::c_int,
    txfm_param: *mut TxfmParam
  );
  static av1_inv_txfm2d_add_16x16: extern fn(
    input: *const i32,
    output: *mut u16,
    stride: libc::c_int,
    tx_type: libc::c_int,
    bd: libc::c_int
  );
  fn av1_inv_txfm2d_add_16x16_c(
    input: *const i32, output: *mut u16, stride: libc::c_int,
    tx_type: libc::c_int, bd: libc::c_int
  );
  fn av1_fht32x32_sse2(
    input: *const i16,
    output: *mut i32,
    stride: libc::c_int,
    txfm_pram: *mut TxfmParam
  );
  static av1_inv_txfm2d_add_32x32: extern fn(
    input: *const i32,
    output: *mut u16,
    stride: libc::c_int,
    tx_type: libc::c_int,
    bd: libc::c_int
  );
  fn av1_inv_txfm2d_add_32x32_c(
    input: *const i32,
    output: *mut u16,
    stride: libc::c_int,
    tx_type: libc::c_int,
    bd: libc::c_int
  );
}

pub fn forward_transform(
  input: &[i16], output: &mut [i32], stride: usize, tx_size: TxSize,
  tx_type: TxType
) {
  match tx_size {
    TxSize::TX_4X4 => fht4x4(input, output, stride, tx_type),
    TxSize::TX_8X8 => fht8x8(input, output, stride, tx_type),
    TxSize::TX_16X16 => fht16x16(input, output, stride, tx_type),
    TxSize::TX_32X32 => fht32x32(input, output, stride, tx_type),
    _ => panic!("unimplemented tx size")
  }
}

pub fn inverse_transform_add(
  input: &[i32], output: &mut [u16], stride: usize, tx_size: TxSize,
  tx_type: TxType
) {
  match tx_size {
    TxSize::TX_4X4 => iht4x4_add(input, output, stride, tx_type),
    TxSize::TX_8X8 => iht8x8_add(input, output, stride, tx_type),
    TxSize::TX_16X16 => iht16x16_add(input, output, stride, tx_type),
    TxSize::TX_32X32 => iht32x32_add(input, output, stride, tx_type),
    _ => panic!("unimplemented tx size")
  }
}

fn fht4x4(input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType) {
  unsafe {
    av1_fht4x4(
      input.as_ptr(),
      output.as_mut_ptr(),
      stride as libc::c_int,
      &(tx_type as i32) as *const libc::c_int
    );
  }
}

fn iht4x4_add(
  input: &[i32], output: &mut [u16], stride: usize, tx_type: TxType
) {
  // SIMD code may assert for transform types beyond TxType::IDTX.
  if tx_type < TxType::IDTX {
    unsafe {
      av1_inv_txfm2d_add_4x4(
        input.as_ptr(),
        output.as_mut_ptr(),
        stride as libc::c_int,
        tx_type as libc::c_int,
        8
      );
    }
  } else {
    unsafe {
      av1_inv_txfm2d_add_4x4_c(
        input.as_ptr(),
        output.as_mut_ptr(),
        stride as libc::c_int,
        tx_type as libc::c_int,
        8
      );
    }
  }
}

fn fht8x8(input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType) {
  unsafe {
    av1_fht8x8(
      input.as_ptr(),
      output.as_mut_ptr(),
      stride as libc::c_int,
      &(tx_type as i32) as *const libc::c_int
    );
  }
}

fn iht8x8_add(
  input: &[i32], output: &mut [u16], stride: usize, tx_type: TxType
) {
  // SIMD code may assert for transform types beyond TxType::IDTX.
  if tx_type < TxType::IDTX {
    unsafe {
      av1_inv_txfm2d_add_8x8(
        input.as_ptr(),
        output.as_mut_ptr(),
        stride as libc::c_int,
        tx_type as libc::c_int,
        8
      );
    }
  } else {
    unsafe {
      av1_inv_txfm2d_add_8x8_c(
        input.as_ptr(),
        output.as_mut_ptr(),
        stride as libc::c_int,
        tx_type as libc::c_int,
        8
      );
    }
  }
}

fn fht16x16(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType
) {
  let mut p = TxfmParam {
    tx_type,
    tx_size: TxSize::TX_16X16,
    lossless: 0,
    bd: 8,
    is_hbd: 1,
    tx_set_type: 0,
    eob: 0
  };
  unsafe {
    av1_fht16x16_c(
      input.as_ptr(),
      output.as_mut_ptr(),
      stride as libc::c_int,
      &mut p
    );
  }
}

fn iht16x16_add(
  input: &[i32], output: &mut [u16], stride: usize, tx_type: TxType
) {
  unsafe {
    if tx_type < TxType::IDTX {
      // SSE C code asserts for transform types beyond TxType::IDTX.
      av1_inv_txfm2d_add_16x16(
        input.as_ptr(),
        output.as_mut_ptr(),
        stride as libc::c_int,
        tx_type as libc::c_int,
        8
      );
    } else {
      av1_inv_txfm2d_add_16x16_c(
        input.as_ptr(),
        output.as_mut_ptr(),
        stride as libc::c_int,
        tx_type as libc::c_int,
        8
      );
    }
  }
}

fn fht32x32(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType
) {
  let mut p = TxfmParam {
    tx_type,
    tx_size: TxSize::TX_32X32,
    lossless: 0,
    bd: 8,
    is_hbd: 1,
    tx_set_type: 0,
    eob: 0
  };
  unsafe {
    av1_fht32x32_sse2(
      input.as_ptr(),
      output.as_mut_ptr(),
      stride as libc::c_int,
      &mut p
    );
  }
}

fn iht32x32_add(
  input: &[i32], output: &mut [u16], stride: usize, tx_type: TxType
) {
  unsafe {
    if tx_type < TxType::IDTX {
      // SIMDI code may assert for transform types beyond TxType::IDTX.
      av1_inv_txfm2d_add_32x32(
        input.as_ptr(),
        output.as_mut_ptr(),
        stride as libc::c_int,
        tx_type as libc::c_int,
        8
      );
    } else {
      av1_inv_txfm2d_add_32x32_c(
        input.as_ptr(),
        output.as_mut_ptr(),
        stride as libc::c_int,
        tx_type as libc::c_int,
        8
      );
    }
  }
}
