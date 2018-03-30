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

fn intra_dc_pred_native(b: &mut Bencher) {
    let mut ra = ChaChaRng::new_unseeded();
    let (above, left, mut o2) = setup_pred(&mut ra);

    b.iter(|| {
        for _ in 0..MAX_ITER {
            Block4x4::pred_dc(&mut o2, 32, &above[..4], &left[..4]);
        }
    })
}

fn intra_dc_pred_aom(b: &mut Bencher) {
    let mut ra = ChaChaRng::new_unseeded();
    let (above, left, mut o2) = setup_pred(&mut ra);

    b.iter(|| {
        for _ in 0..MAX_ITER {
            pred_dc_4x4(&mut o2, 32, &above[..4], &left[..4]);
        }
    })
}

fn intra_h_pred_native(b: &mut Bencher) {
    let mut ra = ChaChaRng::new_unseeded();
    let (_above, left, mut o2) = setup_pred(&mut ra);

    b.iter(|| {
        for _ in 0..MAX_ITER {
            Block4x4::pred_h(&mut o2, 32, &left[..4]);
        }
    })
}

fn intra_h_pred_aom(b: &mut Bencher) {
    let mut ra = ChaChaRng::new_unseeded();
    let (above, left, mut o2) = setup_pred(&mut ra);

    b.iter(|| {
        for _ in 0..MAX_ITER {
            pred_h_4x4(&mut o2, 32, &above[..4], &left[..4]);
        }
    })
}

fn intra_v_pred_native(b: &mut Bencher) {
    let mut ra = ChaChaRng::new_unseeded();
    let (above, _left, mut o2) = setup_pred(&mut ra);

    b.iter(|| {
        for _ in 0..MAX_ITER {
            Block4x4::pred_v(&mut o2, 32, &above[..4]);
        }
    })
}

fn intra_v_pred_aom(b: &mut Bencher) {
    let mut ra = ChaChaRng::new_unseeded();
    let (above, left, mut o2) = setup_pred(&mut ra);

    b.iter(|| {
        for _ in 0..MAX_ITER {
            pred_v_4x4(&mut o2, 32, &above[..4], &left[..4]);
        }
    })
}

use rav1e::*;
use rav1e::context::*;
use rav1e::partition::*;
use rav1e::ec;

fn write_b_bench_4x4(b: &mut Bencher) {
    write_b_bench(b, TxSize::TX_4X4);
}

fn write_b_bench_8x8(b: &mut Bencher) {
    write_b_bench(b, TxSize::TX_8X8);
}

fn write_b_bench(b: &mut Bencher, tx_size: TxSize) {
    unsafe {
        av1_rtcd();
        aom_dsp_rtcd();
    }
    let mut fi = FrameInvariants::new(1024, 1024, 100, 10);
    let w = ec::Writer::new();
    let fc = CDFContext::new(fi.qindex as u8);
    let bc = BlockContext::new(fi.sb_width * 16, fi.sb_height * 16);
    let mut fs = FrameState::new(&fi);
    let mut cw = ContextWriter {
        w: w,
        fc: fc,
        bc: bc,
    };

    let tx_type = TxType::DCT_DCT;

    let sbx = 0;
    let sby = 0;

    b.iter(|| {
        for &mode in RAV1E_INTRA_MODES {
            let sbo = SuperBlockOffset { x: sbx, y: sby };
            for p in 1..3 {
                for by in 0..8 {
                    for bx in 0..8 {
                        let bo = sbo.block_offset(bx, by);
                            let tx_bo = BlockOffset{x: bo.x + bx, y: bo.y + by};
                            let po = tx_bo.plane_offset(&fs.input.planes[p].cfg);
                            encode_tx_block(&mut fi, &mut fs, &mut cw, p, &bo, mode,
                                            tx_size, tx_type, &po, false);
                    }
                }
            }
        }
    });
}

benchmark_group!(intra, intra_dc_pred_native, intra_dc_pred_aom,
    intra_h_pred_native, intra_h_pred_aom, intra_v_pred_native,
    intra_v_pred_aom);
benchmark_group!(ec, write_b_bench_4x4, write_b_bench_8x8);
benchmark_main!(intra, ec);
