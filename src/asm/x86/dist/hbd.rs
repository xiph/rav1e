use crate::util::{msb, Aligned};
use std::arch::x86_64::*;

macro_rules! satd_hbd_avx2 {
  ($(($W:expr, $H:expr)),*) => {
    $(
      paste::item! {
        #[target_feature(enable = "avx2,bmi1,bmi2")]
        pub(crate) unsafe extern fn [<rav1e_satd_ $W x $H _hbd_avx2>](
          src: *const u16, src_stride: isize, dst: *const u16, dst_stride: isize,
        ) -> u32 {
          // Size of hadamard transform should be 4x4 or 8x8.
          // 4x* and *x4 use 4x4 and all other use 8x8.
          // Because these are constants via the macro system,
          // the branches here will be optimized out.
          let size: usize = $W.min($H).min(8);

          let mut sum = 0u64;

          // Loop over chunks the size of the chosen transform
          for chunk_y in (0isize..$H).step_by(size) {
            for chunk_x in (0isize..$W).step_by(size) {
              let src = src.offset(chunk_y * src_stride + chunk_x);
              let dst = dst.offset(chunk_y * dst_stride + chunk_x);
              if size == 4 {
                sum += satd_kernel_4x4_hbd_avx2(src, src_stride, dst, dst_stride);
              } else {
                sum += satd_kernel_8x8_hbd_avx2(src, src_stride, dst, dst_stride);
              }
            }
          }

          // Normalize the results
          let ln = msb(size as i32) as u64;
          ((sum + (1 << ln >> 1)) >> ln) as u32
        }
      }
    )*
  }
}

macro_rules! satd_kernel_hbd_avx2 {
  ($(($W:expr, $H:expr)),*) => {
    $(
      paste::item! {
        #[target_feature(enable = "avx2,bmi1,bmi2")]
        unsafe extern fn [<satd_kernel_ $W x $H _hbd_avx2>](
          src: *const u16, src_stride: isize, dst: *const u16, dst_stride: isize,
        ) -> u64 {
          // Size of hadamard kernel should be 4x4 or 8x8.
          // Because these are constants via the macro system,
          // the branches here will be optimized out.
          let size: usize = $W.min($H).min(8);
          let sizei = size as isize;

          let buf = &mut Aligned::<[i32; 8 * 8]>::uninitialized().data[..size * size];
          let input1 = &mut Aligned::<[i32; 8 * 8]>::uninitialized().data[..size * size];
          let input2 = &mut Aligned::<[i32; 8 * 8]>::uninitialized().data[..size * size];
          for y in 0..(sizei) {
            for x in 0..(sizei) {
              input1.as_mut_ptr().offset(y * sizei + x).write(src.offset(y * src_stride + x).read() as i32);
              input2.as_mut_ptr().offset(y * sizei + x).write(dst.offset(y * dst_stride + x).read() as i32);
            }
          }

          // Move the difference of the transforms to a buffer
          for y in 0..(sizei) {
            if size == 4 {
              let row_src = _mm_load_si128(input1.as_ptr().offset(y * sizei) as *const __m128i);
              let row_dst = _mm_load_si128(input2.as_ptr().offset(y * sizei) as *const __m128i);
              let diff = _mm_sub_epi32(row_src, row_dst);
              _mm_store_si128(buf.as_mut_ptr().offset(y * sizei) as *mut __m128i, diff);
            } else {
              let row_src = _mm256_load_si256(input1.as_ptr().offset(y * sizei) as *const __m256i);
              let row_dst = _mm256_load_si256(input2.as_ptr().offset(y * sizei) as *const __m256i);
              let diff = _mm256_sub_epi32(row_src, row_dst);
              _mm256_store_si256(buf.as_mut_ptr().offset(y * sizei) as *mut __m256i, diff);
            }
          }

          // Perform the hadamard transform on the differences,
          // Then sum the absolute values of the transformed differences
          if size == 4 {
            hadamard4x4(buf.as_mut_ptr());
            let row1 = _mm256_abs_epi32(_mm256_load_si256(buf.as_ptr().add(0) as *const __m256i));
            let row2 = _mm256_abs_epi32(_mm256_load_si256(buf.as_ptr().add(8) as *const __m256i));
            let sum_t = _mm256_add_epi32(row1, row2);
            let mut abs = Aligned::<[u32; 8]>::uninitialized();
            _mm256_store_si256(abs.data.as_mut_ptr() as *mut __m256i, sum_t);
            abs.data.iter().copied().map(|a| a as u64).sum::<u64>()
          } else {
            hadamard8x8(buf.as_mut_ptr());
            let row1 = _mm256_abs_epi32(_mm256_load_si256(buf.as_ptr().add(0) as *const __m256i));
            let row2 = _mm256_abs_epi32(_mm256_load_si256(buf.as_ptr().add(8) as *const __m256i));
            let row3 = _mm256_abs_epi32(_mm256_load_si256(buf.as_ptr().add(16) as *const __m256i));
            let row4 = _mm256_abs_epi32(_mm256_load_si256(buf.as_ptr().add(24) as *const __m256i));
            let row5 = _mm256_abs_epi32(_mm256_load_si256(buf.as_ptr().add(32) as *const __m256i));
            let row6 = _mm256_abs_epi32(_mm256_load_si256(buf.as_ptr().add(40) as *const __m256i));
            let row7 = _mm256_abs_epi32(_mm256_load_si256(buf.as_ptr().add(48) as *const __m256i));
            let row8 = _mm256_abs_epi32(_mm256_load_si256(buf.as_ptr().add(56) as *const __m256i));
            let sum1 = _mm256_add_epi32(row1, row2);
            let sum2 = _mm256_add_epi32(row3, row4);
            let sum3 = _mm256_add_epi32(row5, row6);
            let sum4 = _mm256_add_epi32(row7, row8);
            let sum5 = _mm256_add_epi32(sum1, sum2);
            let sum6 = _mm256_add_epi32(sum3, sum4);
            let sum_t = _mm256_add_epi32(sum5, sum6);
            let mut abs_sums = Aligned::<[u32; 8]>::uninitialized();
            _mm256_store_si256(abs_sums.data.as_mut_ptr() as *mut __m256i, sum_t);
            abs_sums.data.iter().copied().map(|a| a as u64).sum::<u64>()
          }
        }
      }
    )*
  }
}

satd_kernel_hbd_avx2!((4, 4), (8, 8));

satd_hbd_avx2!(
  (4, 4),
  (4, 8),
  (4, 16),
  (8, 4),
  (8, 8),
  (8, 16),
  (8, 32),
  (16, 4),
  (16, 8),
  (16, 16),
  (16, 32),
  (16, 64),
  (32, 8),
  (32, 16),
  (32, 32),
  (32, 64),
  (64, 16),
  (64, 32),
  (64, 64),
  (64, 128),
  (128, 64),
  (128, 128)
);

#[inline(always)]
const fn butterfly(a: i32, b: i32) -> (i32, i32) {
  ((a + b), (a - b))
}

#[inline(always)]
#[allow(clippy::identity_op, clippy::erasing_op)]
unsafe fn hadamard4_1d(
  data: *mut i32, n: usize, stride0: usize, stride1: usize,
) {
  for i in 0..n {
    let sub = data.add(i * stride0);
    let (a0, a1) =
      butterfly(sub.add(0 * stride1).read(), sub.add(1 * stride1).read());
    let (a2, a3) =
      butterfly(sub.add(2 * stride1).read(), sub.add(3 * stride1).read());
    let (b0, b2) = butterfly(a0, a2);
    let (b1, b3) = butterfly(a1, a3);
    sub.add(0 * stride1).write(b0);
    sub.add(1 * stride1).write(b1);
    sub.add(2 * stride1).write(b2);
    sub.add(3 * stride1).write(b3);
  }
}

#[inline(always)]
#[allow(clippy::identity_op, clippy::erasing_op)]
unsafe fn hadamard8_1d(
  data: *mut i32, n: usize, stride0: usize, stride1: usize,
) {
  for i in 0..n {
    let sub = data.add(i * stride0);

    let (a0, a1) =
      butterfly(sub.add(0 * stride1).read(), sub.add(1 * stride1).read());
    let (a2, a3) =
      butterfly(sub.add(2 * stride1).read(), sub.add(3 * stride1).read());
    let (a4, a5) =
      butterfly(sub.add(4 * stride1).read(), sub.add(5 * stride1).read());
    let (a6, a7) =
      butterfly(sub.add(6 * stride1).read(), sub.add(7 * stride1).read());

    let (b0, b2) = butterfly(a0, a2);
    let (b1, b3) = butterfly(a1, a3);
    let (b4, b6) = butterfly(a4, a6);
    let (b5, b7) = butterfly(a5, a7);

    let (c0, c4) = butterfly(b0, b4);
    let (c1, c5) = butterfly(b1, b5);
    let (c2, c6) = butterfly(b2, b6);
    let (c3, c7) = butterfly(b3, b7);

    sub.add(0 * stride1).write(c0);
    sub.add(1 * stride1).write(c1);
    sub.add(2 * stride1).write(c2);
    sub.add(3 * stride1).write(c3);
    sub.add(4 * stride1).write(c4);
    sub.add(5 * stride1).write(c5);
    sub.add(6 * stride1).write(c6);
    sub.add(7 * stride1).write(c7);
  }
}

#[inline(always)]
fn hadamard2d(data: *mut i32, (w, h): (usize, usize)) {
  /*Vertical transform.*/
  let vert_func = if h == 4 { hadamard4_1d } else { hadamard8_1d };
  unsafe {
    vert_func(data, w, 1, h);
  }
  /*Horizontal transform.*/
  let horz_func = if w == 4 { hadamard4_1d } else { hadamard8_1d };
  unsafe {
    horz_func(data, h, w, 1);
  }
}

#[inline(always)]
fn hadamard4x4(data: *mut i32) {
  hadamard2d(data, (4, 4));
}

#[inline(always)]
fn hadamard8x8(data: *mut i32) {
  hadamard2d(data, (8, 8));
}
