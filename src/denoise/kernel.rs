use std::{
  mem::{size_of, transmute},
  ptr::copy_nonoverlapping,
};

use crate::util::cast;

use super::{
  bitblt, calc_pad_size, f32x16, BLOCK_SIZE, TEMPORAL_RADIUS, TEMPORAL_SIZE,
};
use av_metrics::video::Pixel;
use num_traits::clamp;
use wide::{f32x8, u16x8, u8x16};

pub fn reflection_padding<T: Pixel>(
  output: &mut [T], input: &[T], width: usize, height: usize, stride: usize,
) {
  let pad_width = calc_pad_size(width);
  let pad_height = calc_pad_size(height);

  let offset_y = (pad_height - height) / 2;
  let offset_x = (pad_width - width) / 2;

  bitblt(
    &mut output[(offset_y * pad_width + offset_x)..],
    pad_width,
    input,
    stride,
    width,
    height,
  );

  // copy left and right regions
  for y in offset_y..(offset_y + height) {
    let dst_line = &mut output[(y * pad_width)..];

    for x in 0..offset_x {
      dst_line[x] = dst_line[offset_x * 2 - x];
    }

    for x in (offset_x + width)..pad_width {
      dst_line[x] = dst_line[2 * (offset_x + width) - 2 - x];
    }
  }

  // copy top region
  for y in 0..offset_y {
    let dst = output[(y * pad_width)..][..pad_width].as_mut_ptr();
    let src = output[((offset_y * 2 - y) * pad_width)..][..pad_width].as_ptr();
    // SAFETY: We check the start and end bounds above.
    // We have to use `copy_nonoverlapping` because the borrow checker is not happy with
    // `copy_from_slice`.
    unsafe {
      copy_nonoverlapping(src, dst, pad_width);
    }
  }

  // copy bottom region
  for y in (offset_y + height)..pad_height {
    let dst = output[(y * pad_width)..][..pad_width].as_mut_ptr();
    let src = output[((2 * (offset_y + height) - 2 - y) * pad_width)..]
      [..pad_width]
      .as_ptr();
    // SAFETY: We check the start and end bounds above.
    // We have to use `copy_nonoverlapping` because the borrow checker is not happy with
    // `copy_from_slice`.
    unsafe {
      copy_nonoverlapping(src, dst, pad_width);
    }
  }
}

pub fn load_block<T: Pixel>(
  block: &mut [f32x16], shifted_src: &[T], width: usize, height: usize,
  bit_depth: usize, window: &[f32x16],
) {
  let scale = 1.0f32 / (1 << (bit_depth - 8)) as f32;
  let offset_x = calc_pad_size(width);
  let offset_y = calc_pad_size(height);
  for i in 0..TEMPORAL_SIZE {
    for j in 0..BLOCK_SIZE {
      // The compiler will optimize away these branches
      let vec_input = if size_of::<T>() == 1 {
        // SAFETY: We know that T is u8
        let u8s: [u8; 16] = unsafe {
          *transmute::<_, &[u8; 16]>(cast::<16, _>(
            &shifted_src[((i * offset_y + j) * offset_x)..][..16],
          ))
        };
        let f32_upper = u8x16::new(u8s).to_u32x8().to_f32x8();
        let f32_lower = u8x16::new([
          u8s[8], u8s[9], u8s[10], u8s[11], u8s[12], u8s[13], u8s[14],
          u8s[15], 0, 0, 0, 0, 0, 0, 0, 0,
        ])
        .to_u32x8()
        .to_f32x8();
        [f32_upper, f32_lower]
      } else {
        // SAFETY: We know that T is u8
        let u16s: [u16; 16] = unsafe {
          *transmute::<_, &[u16; 16]>(cast::<16, _>(
            &shifted_src[((i * offset_y + j) * offset_x)..][..16],
          ))
        };
        let f32_upper = u16x8::new(*cast(&u16s[..8])).to_u32x8().to_f32x8();
        let f32_lower = u16x8::new(*cast(&u16s[8..])).to_u32x8().to_f32x8();
        [f32_upper, f32_lower]
      };
      let window = &window[i * BLOCK_SIZE + j];
      let result_upper = scale * window[0] * vec_input[0];
      let result_lower = scale * window[1] * vec_input[1];
      block[i * BLOCK_SIZE * 2 + j] = [result_upper, result_lower];
    }
  }
}

pub fn fused(
  block: &mut [f32x16], sigma: f32, pmin: f32, pmax: f32,
  window_freq: &[f32x16],
) {
  for i in 0..TEMPORAL_SIZE {
    transpose_16x16(&mut block[(i * 32)..]);
    rdft::<16>(&mut block[(i * 32)..]);
    transpose_32x16(&mut block[(i * 32)..]);
    dft::<16>(&mut block[(i * 32)..]);
  }
  for i in 0..16 {
    dft::<3>(&mut block[(i * 2)..], 16);
  }

  let gf = block[0].extract(0) / window_freq[0].extract(0);
  remove_mean(block, gf, window_freq);

  frequency_filtering(block, sigma, pmin, pmax);

  add_mean(block, gf, window_freq);

  for i in 0..16 {
    idft::<3>(&mut block[(i * 2)..], 16);
  }
  idft::<16>(&mut block[(TEMPORAL_RADIUS * 32)..]);
  transpose_32x16(&mut block[(TEMPORAL_RADIUS * 32)..]);
  irdft::<16>(&mut block[(TEMPORAL_RADIUS * 32)..]);
  post_irdft::<16>(&mut block[(TEMPORAL_RADIUS * 32)..]);
  transpose_16x16(&mut block[(TEMPORAL_RADIUS * 32)..]);
}

pub fn store_block(
  shifted_dst: &mut [f32], shifted_block: &[f32x16], width: usize,
  height: usize, shifted_window: &[f32x16],
) {
  let pad_size = calc_pad_size(width);
  for i in 0..BLOCK_SIZE {
    let acc = &mut shifted_dst[(i * pad_size)..];
    let mut acc_simd =
      [f32x8::new(*cast(&acc[..8])), f32x8::new(*cast(&acc[8..16]))];
    acc_simd[0] =
      shifted_block[i][0].mul_add(shifted_window[i][0], acc_simd[0]);
    acc_simd[1] =
      shifted_block[i][1].mul_add(shifted_window[i][1], acc_simd[1]);
    acc[..8].copy_from_slice(acc_simd[0].as_array_ref());
    acc[8..16].copy_from_slice(acc_simd[1].as_array_ref());
  }
}

pub fn store_frame<T: Pixel>(
  output: &mut [T], shifted_src: &[f32], width: usize, height: usize,
  bit_depth: usize, dst_stride: usize, src_stride: usize,
) {
  let scale = 1.0f32 / (1 << (bit_depth - 8)) as f32;
  let peak = (1u32 << bit_depth) - 1;
  for y in 0..height {
    for x in 0..width {
      // SAFETY: We know the bounds of the planes for src and dest
      unsafe {
        let clamped = clamp(
          (*shifted_src.get_unchecked(y * src_stride + x) / scale + 0.5f32)
            as u32,
          0u32,
          peak,
        );
        *output.get_unchecked_mut(y * dst_stride + x) = T::cast_from(clamped);
      }
    }
  }
}

fn transpose_16x16() {
  todo!()
}

fn transpose_32x16() {
  todo!()
}

fn remove_mean() {
  todo!()
}

fn frequency_filtering() {
  todo!()
}

fn add_mean() {
  todo!()
}

fn rdft<const SIZE: usize>() {
  todo!()
}

fn dft<const SIZE: usize>() {
  todo!()
}

fn idft<const SIZE: usize>() {
  todo!()
}

fn irdft<const SIZE: usize>() {
  todo!()
}

fn post_irdft<const SIZE: usize>() {
  todo!()
}
