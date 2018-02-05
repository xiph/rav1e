extern crate libc;

use partition::TxType;

extern {
    fn av1_fht4x4_c(input: *const i16, output: *mut i32, stride: libc::c_int, tx_type: *const libc::c_int);
    fn av1_inv_txfm2d_add_4x4_c(input: *const i32, output: *mut u16, stride: libc::c_int,
                                tx_type: libc::c_int, bd: libc::c_int);
}

pub fn fht4x4(input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType) {
    unsafe {
        av1_fht4x4_c(input.as_ptr(), output.as_mut_ptr(), stride as libc::c_int, &(tx_type as i32) as *const libc::c_int);
    }
}

pub fn iht4x4_add(input: &[i32], output: &mut [u16], stride: usize, tx_type: TxType) {
    unsafe {
        av1_inv_txfm2d_add_4x4_c(input.as_ptr(), output.as_mut_ptr(), stride as libc::c_int,
                                 tx_type as libc::c_int, 8);
    }
}

