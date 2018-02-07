#[macro_use]
extern crate bencher;
extern crate rav1e;
extern crate rand;
extern crate libc;

use bencher::Bencher;
use rand::{ChaChaRng, Rng};
use rav1e::predict::*;

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
    fn highbd_v_predictor(dst: *mut u16, stride: libc::ptrdiff_t, bw: libc::c_int,
        bh: libc::c_int, above: *const u16,
        left: *const u16, bd: libc::c_int);
}

#[inline(always)]
fn pred_dc_4x4(output: &mut [u16], stride: usize, above: &[u16], left: &[u16]) {
    unsafe {
        highbd_dc_predictor(output.as_mut_ptr(), stride as libc::ptrdiff_t, 4, 4, above.as_ptr(), left.as_ptr(), 8);
    }
}

#[inline(always)]
fn pred_h_4x4(output: &mut [u16], stride: usize, above: &[u16], left: &[u16]) {
    unsafe {
        highbd_h_predictor(output.as_mut_ptr(), stride as libc::ptrdiff_t, 4, 4, above.as_ptr(), left.as_ptr(), 8);
    }
}

#[inline(always)]
fn pred_v_4x4(output: &mut [u16], stride: usize, above: &[u16], left: &[u16]) {
    unsafe {
        highbd_v_predictor(output.as_mut_ptr(), stride as libc::ptrdiff_t, 4, 4, above.as_ptr(), left.as_ptr(), 8);
    }
}

const MAX_ITER: usize = 50000;

fn setup_pred(ra: &mut ChaChaRng) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let o1 = vec![0u16; 32 * 32];
    let above: Vec<u16> = (0..32).map(|_| ra.gen()).collect();
    let left: Vec<u16> = (0..32).map(|_| ra.gen()).collect();

    (above, left, o1)
}

fn native(b: &mut Bencher) {
    let mut ra = ChaChaRng::new_unseeded();
    let (above, left, mut o2) = setup_pred(&mut ra);

    b.iter(|| {
        for _ in 0..MAX_ITER {
            pred_dc(&mut o2, 32, &above[..4], &left[..4]);
        }
    })
}

fn native_trait(b: &mut Bencher) {
    let mut ra = ChaChaRng::new_unseeded();
    let (above, left, mut o2) = setup_pred(&mut ra);

    b.iter(|| {
        for _ in 0..MAX_ITER {
            pred_dc_trait::<Block4x4>(&mut o2, 32, &above[..4], &left[..4]);
        }
    })
}

fn aom(b: &mut Bencher) {
    let mut ra = ChaChaRng::new_unseeded();
    let (above, left, mut o2) = setup_pred(&mut ra);

    b.iter(|| {
        for _ in 0..MAX_ITER {
            pred_dc_4x4(&mut o2, 32, &above[..4], &left[..4]);
        }
    })
}

benchmark_group!(predict, aom, native_trait, native);
benchmark_main!(predict);
