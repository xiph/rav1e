use std::{
  mem::{size_of, transmute},
  ptr::copy_nonoverlapping,
};

use crate::util::cast;

use super::{
  bitblt, calc_pad_size, f32x16, BLOCK_SIZE, BLOCK_STEP, TEMPORAL_SIZE,
};
use av_metrics::video::Pixel;
use num_complex::Complex32;
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
  bit_depth: usize, window: &[f32],
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
      let window = &window[(i * BLOCK_SIZE + j)..];
      let window_upper = f32x8::new(*cast(&window[..8]));
      let window_lower = f32x8::new(*cast(&window[8..16]));
      let result_upper = scale * window_upper * vec_input[0];
      let result_lower = scale * window_lower * vec_input[1];
      block[i * BLOCK_SIZE * 2 + j] = [result_upper, result_lower];
    }
  }
}

pub fn fused(
  output: &mut [f32x16], sigma: f32, pmin: f32, pmax: f32,
  window_freq: &[Complex32],
) {
  todo!()
}

pub fn store_block(
  output: &mut [f32], input: &[f32x16], width: usize, height: usize,
  window: &[f32],
) {
  todo!()
}

pub fn store_frame<T: Pixel>(
  output: &mut [T], shifted_src: &[f32], width: usize, height: usize,
  dst_stride: usize, src_stride: usize,
) {
  todo!()
}
