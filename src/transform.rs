extern crate libc;

use partition::TxSize::*;

extern {
    fn aom_fdct4x4_c(input: *const i16, output: *mut i32, stride: libc::c_int);
    fn av1_inv_txfm2d_add_4x4_c(input: *const i32, output: *mut u16, stride: libc::c_int,
                             tx_type: libc::c_int, bd: libc::c_int);
}

pub fn fdct4x4(input: &[i16], output: &mut [i32], stride: usize) {
    unsafe {
        aom_fdct4x4_c(input.as_ptr(), output.as_mut_ptr(), stride as libc::c_int);
    }
}

pub fn idct4x4_add(input: &[i32], output: &mut [u16], stride: usize) {
    unsafe {
        av1_inv_txfm2d_add_4x4_c(input.as_ptr(), output.as_mut_ptr(), stride as libc::c_int,
                                 TX_4X4 as libc::c_int, 8);
    }
}

