extern crate libc;

use std::mem::transmute;
use std::mem::uninitialized;
use std::slice;

pub struct Writer {
    enc: [u8; 100]
}

extern {
    fn od_ec_enc_init(enc: *mut libc::c_void, size: u32);
    fn od_ec_enc_done(enc: *mut libc::c_void, nbytes: *mut u32) -> *const u8;
    fn od_ec_encode_cdf_q15(enc: *mut libc::c_void, s: libc::c_int, cdf: *const u16,
                            nsyms: libc::c_int);
    fn od_ec_encode_bool_q15(enc: *mut libc::c_void, val: libc::c_int, f: libc::c_uint);
}

impl Writer {
    pub fn new() -> Writer {
        unsafe {
            let enc = uninitialized();
            od_ec_enc_init(transmute(&enc), 1024);
            Writer { enc: enc }
        }
    }
    pub fn done(&mut self) -> &[u8] {
        let mut nbytes: u32 = 0;
        unsafe {
            let b = od_ec_enc_done(transmute(&self.enc), &mut nbytes);
            slice::from_raw_parts(b, nbytes as usize)
        }
    }
    pub fn cdf(&mut self, s: u32, cdf: &[u16]) {
        unsafe {
            od_ec_encode_cdf_q15(transmute(&self.enc), s as libc::c_int, cdf.as_ptr(), cdf.len() as libc::c_int);
        }
    }
    pub fn bool(&mut self, val: bool, f: u16) {
        unsafe {
            od_ec_encode_bool_q15(transmute(&self.enc), val as libc::c_int, f as libc::c_uint);
        }
    }
    fn update_cdf(cdf: &mut [u16], val: u32, nsymbs: usize) {
        let rate = 4 + if cdf[nsymbs] > 31 { 1 } else { 0 } + (31 ^ (nsymbs as u32).leading_zeros());
        let rate2 = 5;
        let mut tmp: i32;
        let diff: i32;
        let tmp0 = 1 << rate2;
        tmp = 32768 - tmp0;
        diff = ((32768 - ((nsymbs as i32) << rate2)) >> rate) << rate;
        for i in 0..(nsymbs - 1) {
            if i as u32 == val {
                tmp -= diff;
            }
            cdf[i as usize] = ((cdf[i as usize] as i32) + ((tmp - (cdf[i as usize] as i32)) >> rate)) as u16;
            tmp -= tmp0;
        }
        if cdf[nsymbs] < 32 {
            cdf[nsymbs] += 1;
        }
    }
    pub fn symbol(&mut self, s: u32, cdf: &mut [u16], nsymbs: usize) {
        self.cdf(s, &cdf[..nsymbs]);
        Writer::update_cdf(cdf, s, nsymbs);
    }
}
