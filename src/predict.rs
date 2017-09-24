use libc;

extern {
    fn highbd_dc_predictor(dst: *mut u16, stride: libc::ptrdiff_t, bw: libc::c_int,
                                           bh: libc::c_int, above: *const u16,
                           left: *const u16, bd: libc::c_int);
    fn highbd_dc_left_predictor(dst: *mut u16, stride: libc::ptrdiff_t, bw: libc::c_int,
                           bh: libc::c_int, above: *const u16,
                           left: *const u16, bd: libc::c_int);
    fn highbd_dc_top_predictor(dst: *mut u16, stride: libc::ptrdiff_t, bw: libc::c_int,
                           bh: libc::c_int, above: *const u16,
                           left: *const u16, bd: libc::c_int);
    fn highbd_h_predictor(dst: *mut u16, stride: libc::ptrdiff_t, bw: libc::c_int,
                           bh: libc::c_int, above: *const u16,
                           left: *const u16, bd: libc::c_int);
}

pub fn pred_dc_4x4(output: &mut [u16], stride: usize, above: &[u16], left: &[u16]) {
    unsafe {
        highbd_dc_predictor(output.as_mut_ptr(), stride as libc::ptrdiff_t, 4, 4, above.as_ptr(), left.as_ptr(), 8);
    }
}

pub fn pred_dc_left_4x4(output: &mut [u16], stride: usize, above: &[u16], left: &[u16]) {
    unsafe {
        highbd_dc_left_predictor(output.as_mut_ptr(), stride as libc::ptrdiff_t, 4, 4, above.as_ptr(), left.as_ptr(), 8);
    }
}

pub fn pred_dc_top_4x4(output: &mut [u16], stride: usize, above: &[u16], left: &[u16]) {
    unsafe {
        highbd_dc_top_predictor(output.as_mut_ptr(), stride as libc::ptrdiff_t, 4, 4, above.as_ptr(), left.as_ptr(), 8);
    }
}

#[allow(dead_code)]
pub fn pred_h_4x4(output: &mut [u16], stride: usize, above: &[u16], left: &[u16]) {
    unsafe {
        highbd_h_predictor(output.as_mut_ptr(), stride as libc::ptrdiff_t, 4, 4, above.as_ptr(), left.as_ptr(), 8);
    }
}
