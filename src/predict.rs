use libc;

extern {
    #[cfg(test)]
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

pub fn pred_dc_128(output: &mut [u16], stride: usize) {
    for y in 0..4 {
        for x in 0..4 {
            output[y*stride+x] = 128;
        }
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

pub fn pred_h_4x4(output: &mut [u16], stride: usize, above: &[u16], left: &[u16]) {
    unsafe {
        highbd_h_predictor(output.as_mut_ptr(), stride as libc::ptrdiff_t, 4, 4, above.as_ptr(), left.as_ptr(), 8);
    }
}

pub fn pred_dc(output: &mut [u16], stride: usize, above: &[u16], left: &[u16]) {
    let edges = left.iter().chain(above.iter());
    let len = (left.len() + above.len()) as u32;
    let bw = above.len();
    let bh = left.len();
    let avg = (edges.fold(0, |acc, &v| acc + v as u32) + (len >> 1)) / len;

    for line in output.chunks_mut(stride).take(bh) {
        for v in &mut line[..bw] {
            *v = avg as u16;
        }
    }
}

#[cfg(test)]
pub fn pred_h(output: &mut [u16], stride: usize, left: &[u16], bw: usize) {
  for (line, l) in output.chunks_mut(stride).zip(left) {
    for v in &mut line[..bw] {
      *v = *l;
    }
  }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::{ChaChaRng, Rng};

    const MAX_ITER: usize = 1000;

    fn setup_pred(ra: &mut ChaChaRng) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
        let output = vec![0u16; 32 * 32];
        let above: Vec<u16> = (0..32).map(|_| ra.gen()).collect();
        let left: Vec<u16> = (0..32).map(|_| ra.gen()).collect();

        let o1 = output.clone();
        let o2 = output.clone();

        (above, left, o1, o2)
    }

    fn pred_dc_4x4(output: &mut [u16], stride: usize, above: &[u16], left: &[u16]) {
        unsafe {
            highbd_dc_predictor(output.as_mut_ptr(), stride as libc::ptrdiff_t, 4, 4, above.as_ptr(), left.as_ptr(), 8);
        }
    }

    fn do_dc_pred(ra: &mut ChaChaRng) -> (Vec<u16>, Vec<u16>) {
        let (above, left, mut o1, mut o2) = setup_pred(ra);

        pred_dc_4x4(&mut o1, 32, &above[..4], &left[..4]);
        pred_dc(&mut o2, 32, &above[..4], &left[..4]);

        (o1, o2)
    }

    fn do_h_pred(ra: &mut ChaChaRng) -> (Vec<u16>, Vec<u16>) {
        let (above, left, mut o1, mut o2) = setup_pred(ra);

        pred_h_4x4(&mut o1, 32, &above[..4], &left[..4]);
        pred_h(&mut o2, 32, &left[..4], 4);

        (o1, o2)
    }

    fn assert_same(o2: Vec<u16>) {
      for l in o2.chunks(32).take(4) {
        for v in l[..4].windows(2) {
          assert_eq!(v[0], v[1]);
        }
      }
    }

    #[test]
    fn pred_matches() {
        let mut ra = ChaChaRng::new_unseeded();
        for _ in 0..MAX_ITER {
            let (o1, o2) = do_dc_pred(&mut ra);
            assert_eq!(o1, o2);

            let (o1, o2) = do_h_pred(&mut ra);
            assert_eq!(o1, o2);
        }
    }

    #[test]
    fn pred_same() {
        let mut ra = ChaChaRng::new_unseeded();
        for _ in 0..MAX_ITER {
            let (_, o2) = do_dc_pred(&mut ra);

            assert_same(o2)
        }
    }

    #[test]
    fn pred_max() {
        let max12bit = 4096 - 1;
        let above = [max12bit; 32];
        let left = [max12bit; 32];

        let mut o = vec![0u16; 32 * 32];

        pred_dc(&mut o, 32, &above[..4], &left[..4]);

        for l in o.chunks(32).take(4) {
            for v in l[..4].iter() {
                assert_eq!(*v, max12bit);
            }
        }
    }
}
