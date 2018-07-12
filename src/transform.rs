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

extern {
  fn av1_fht4x4_c(
    input: *const i16, output: *mut i32, stride: libc::c_int,
    tx_type: *const libc::c_int
  );
  fn av1_inv_txfm2d_add_4x4_c(
    input: *const i32, output: *mut u16, stride: libc::c_int,
    tx_type: libc::c_int, bd: libc::c_int
  );
  fn av1_fht8x8_c(
    input: *const i16, output: *mut i32, stride: libc::c_int,
    tx_type: *const libc::c_int
  );
  fn av1_inv_txfm2d_add_8x8_c(
    input: *const i32, output: *mut u16, stride: libc::c_int,
    tx_type: libc::c_int, bd: libc::c_int
  );
  fn av1_fht16x16_c(
    input: *const i16, output: *mut i32, stride: libc::c_int,
    tx_type: *const libc::c_int
  );
  fn av1_inv_txfm2d_add_16x16_c(
    input: *const i32, output: *mut u16, stride: libc::c_int,
    tx_type: libc::c_int, bd: libc::c_int
  );
  fn av1_fht32x32_c(
    input: *const i16, output: *mut i32, stride: libc::c_int,
    tx_type: *const libc::c_int
  );
  fn av1_inv_txfm2d_add_32x32_c(
    input: *const i32, output: *mut u16, stride: libc::c_int,
    tx_type: libc::c_int, bd: libc::c_int
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
    av1_fht4x4_c(
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

fn fht8x8(input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType) {
  unsafe {
    av1_fht8x8_c(
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

fn fht16x16(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType
) {
  unsafe {
    av1_fht16x16_c(
      input.as_ptr(),
      output.as_mut_ptr(),
      stride as libc::c_int,
      &(tx_type as i32) as *const libc::c_int
    );
  }
}

fn iht16x16_add(
  input: &[i32], output: &mut [u16], stride: usize, tx_type: TxType
) {
  unsafe {
    av1_inv_txfm2d_add_16x16_c(
      input.as_ptr(),
      output.as_mut_ptr(),
      stride as libc::c_int,
      tx_type as libc::c_int,
      8
    );
  }
}

fn fht32x32(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType
) {
  unsafe {
    av1_fht32x32_c(
      input.as_ptr(),
      output.as_mut_ptr(),
      stride as libc::c_int,
      &(tx_type as i32) as *const libc::c_int
    );
  }
}

fn iht32x32_add(
  input: &[i32], output: &mut [u16], stride: usize, tx_type: TxType
) {
  unsafe {
    av1_inv_txfm2d_add_32x32_c(
      input.as_ptr(),
      output.as_mut_ptr(),
      stride as libc::c_int,
      tx_type as libc::c_int,
      8
    );
  }
}
