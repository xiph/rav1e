mod kernel;

use crate::api::FrameQueue;
use crate::util::{cast, Aligned};
use crate::EncoderStatus;
use arrayvec::ArrayVec;
use kernel::*;
use num_complex::Complex32;
use num_traits::Zero;
use std::f32::consts::PI;
use std::iter::once;
use std::mem::{size_of, transmute};
use std::ptr::copy_nonoverlapping;
use std::sync::Arc;
use v_frame::frame::Frame;
use v_frame::pixel::Pixel;
use wide::f32x8;

pub const TEMPORAL_RADIUS: usize = 1;
const TEMPORAL_SIZE: usize = TEMPORAL_RADIUS * 2 + 1;
const BLOCK_SIZE: usize = 16;
const BLOCK_STEP: usize = 12;
const REAL_SIZE: usize = TEMPORAL_SIZE * BLOCK_SIZE * BLOCK_SIZE;
const COMPLEX_SIZE: usize = TEMPORAL_SIZE * BLOCK_SIZE * (BLOCK_SIZE / 2 + 1);

// The C implementation uses this f32x16 type, which is implemented the same way
// for non-avx512 systems. `wide` doesn't have a f32x16 type, so we mimic it.
#[allow(non_camel_case_types)]
type f32x16 = [f32x8; 2];

/// This denoiser is based on the DFTTest2 plugin from Vapoursynth.
/// This type of denoising was chosen because it provides
/// high quality while not being too slow. The DFTTest2 implementation
/// is much faster than the original due to its custom FFT kernel.
pub struct DftDenoiser<T>
where
  T: Pixel,
{
  // External values
  prev_frame: Option<Arc<Frame<T>>>,
  pub(crate) cur_frameno: u64,
  bit_depth: usize,
  // Local values
  sigma: f32,
  window: Aligned<[f32; REAL_SIZE]>,
  window_freq: Aligned<[Complex32; COMPLEX_SIZE]>,
  pmin: f32,
  pmax: f32,
  padded: Vec<T>,
  padded2: Vec<f32>,
}

impl<T> DftDenoiser<T>
where
  T: Pixel,
{
  // This should only need to run once per video.
  pub fn new(
    sigma: f32, width: usize, height: usize, bit_depth: usize,
  ) -> Self {
    if size_of::<T>() == 1 {
      assert!(bit_depth <= 8);
    } else {
      assert!(bit_depth > 8);
    }

    let window = build_window();
    let wscale = window.iter().copied().map(|w| w * w).sum::<f32>();
    let sigma = sigma as f32 * wscale;
    let pmin = 0.0f32;
    let pmax = 500.0f32 * wscale;
    let mut window_freq_real = Aligned::new([0f32; REAL_SIZE]);
    window_freq_real.iter_mut().zip(window.iter()).for_each(|(freq, w)| {
      *freq = *w as f32 * 255.0;
    });
    let window_freq = rdft(&window_freq_real);
    let w_pad_size = calc_pad_size(width);
    let h_pad_size = calc_pad_size(height);
    let pad_size = w_pad_size * h_pad_size;
    let padded = vec![T::zero(); pad_size * TEMPORAL_SIZE];
    let padded2 = vec![0f32; pad_size];

    Self {
      prev_frame: None,
      cur_frameno: 0,
      bit_depth,
      sigma,
      window,
      window_freq,
      pmin,
      pmax,
      padded,
      padded2,
    }
  }

  pub fn filter_frame(
    &mut self, frame_q: &FrameQueue<T>,
  ) -> Result<Frame<T>, EncoderStatus> {
    let next_frame = frame_q.get(&(self.cur_frameno + 1));
    if next_frame.is_none() {
      // We also need to have the next unfiltered frame,
      // unless we are at the end of the video.
      return Err(EncoderStatus::NeedMoreData);
    }

    let next_frame = next_frame.cloned().flatten();
    let orig_frame = frame_q.get(&self.cur_frameno).unwrap().as_ref().unwrap();
    let frames =
      once(self.prev_frame.clone().unwrap_or_else(|| Arc::clone(orig_frame)))
        .chain(once(Arc::clone(orig_frame)))
        .chain(once(next_frame.unwrap_or_else(|| Arc::clone(orig_frame))))
        .collect::<ArrayVec<_, 3>>();

    let mut dest = (**orig_frame).clone();
    for p in 0..3 {
      let width = frames[0].planes[p].cfg.width;
      let height = frames[0].planes[p].cfg.height;
      let stride = frames[0].planes[p].cfg.stride;
      let w_pad_size = calc_pad_size(width);
      let h_pad_size = calc_pad_size(height);
      let pad_size_spatial = w_pad_size * h_pad_size;
      for i in 0..TEMPORAL_SIZE {
        let src = &frames[i].planes[p];
        reflection_padding(
          &mut self.padded[(i * pad_size_spatial)..],
          src.data_origin(),
          width,
          height,
          stride,
        )
      }

      for i in 0..h_pad_size {
        for j in 0..w_pad_size {
          let mut block = [f32x16::default(); 7 * BLOCK_SIZE * 2];
          let offset_x = w_pad_size;
          load_block(
            &mut block,
            &self.padded[((i * offset_x + j) * BLOCK_STEP)..],
            width,
            height,
            self.bit_depth,
            // SAFETY: We know that the window size is a multiple of 16
            unsafe { transmute(&self.window[..]) },
          );
          fused(
            &mut block,
            self.sigma,
            self.pmin,
            self.pmax,
            // SAFETY: We know that the window size is a multiple of 16
            unsafe { transmute(&self.window_freq[..]) },
          );
          store_block(
            &mut self.padded2[((i * offset_x + j) * BLOCK_STEP)..],
            &block[(TEMPORAL_RADIUS * BLOCK_SIZE * 2)..],
            width,
            height,
            // SAFETY: We know that the window size is a multiple of 16
            unsafe {
              transmute(
                &self.window[(TEMPORAL_RADIUS * BLOCK_SIZE * 2 * 16)..],
              )
            },
          );
          todo!()
        }
      }

      let offset_y = (h_pad_size - height) / 2;
      let offset_x = (w_pad_size - width) / 2;
      let dest_plane = &mut dest.planes[p];
      store_frame(
        dest_plane.data_origin_mut(),
        &self.padded2[(offset_y * w_pad_size + offset_x)..],
        width,
        height,
        self.bit_depth,
        stride,
        w_pad_size,
      );
    }

    self.prev_frame = Some(Arc::clone(orig_frame));
    self.cur_frameno += 1;

    Ok(dest)
  }
}

#[inline(always)]
// Hanning windowing
fn spatial_window_value(n: f32) -> f32 {
  let temp = PI * n / BLOCK_SIZE as f32;
  0.5 * (1.0 - (2.0 * temp).cos())
}

#[inline(always)]
// Simple rectangular windowing
const fn temporal_window_value() -> f32 {
  1.0
}

fn build_window() -> Aligned<[f32; REAL_SIZE]> {
  let temporal_window = [temporal_window_value(); TEMPORAL_SIZE];

  let mut spatial_window = [0f32; BLOCK_SIZE];
  spatial_window.iter_mut().enumerate().for_each(|(i, val)| {
    *val = spatial_window_value(i as f32 + 0.5);
  });
  let spatial_window = normalize(&spatial_window);

  let mut window = Aligned::new([0f32; REAL_SIZE]);
  let mut i = 0;
  for t_val in temporal_window {
    for s_val1 in spatial_window {
      for s_vals2 in spatial_window.chunks_exact(8) {
        let s_val2 = f32x8::new(*cast::<8, _>(s_vals2));
        let mut value = t_val * s_val1 * s_val2;
        // normalize for unnormalized FFT implementation
        value /=
          f32x8::from((TEMPORAL_SIZE as f32).sqrt() * BLOCK_SIZE as f32);
        // SAFETY: We know the slices are valid sizes
        unsafe {
          copy_nonoverlapping(
            value.as_array_ref().as_ptr(),
            window.as_mut_ptr().add(i),
            8usize,
          )
        };
        i += 8;
      }
    }
  }
  window
}

fn normalize(window: &[f32; BLOCK_SIZE]) -> [f32; BLOCK_SIZE] {
  let mut new_window = [0f32; BLOCK_SIZE];
  // SAFETY: We know all of the sizes, so bound checks are not needed.
  unsafe {
    for q in 0..BLOCK_SIZE {
      let nw = new_window.get_unchecked_mut(q);
      for h in (0..=q).rev().step_by(BLOCK_STEP) {
        *nw += window.get_unchecked(h).powi(2);
      }
      for h in ((q + BLOCK_STEP)..BLOCK_SIZE).step_by(BLOCK_STEP) {
        *nw += window.get_unchecked(h).powi(2);
      }
    }
  }
  for (w, nw) in window.iter().zip(new_window.iter_mut()) {
    *nw = *w / nw.sqrt();
  }
  new_window
}

// Identical to Vapoursynth's implementation `vs_bitblt`
// which basically copies the pixels in a plane.
pub fn bitblt<T: Pixel>(
  mut dest: &mut [T], dest_stride: usize, mut src: &[T], src_stride: usize,
  width: usize, height: usize,
) {
  if src_stride == dest_stride && src_stride == width {
    dest[..(width * height)].copy_from_slice(&src[..(width * height)]);
  } else {
    for _ in 0..height {
      dest[..width].copy_from_slice(&src[..width]);
      src = &src[src_stride..];
      dest = &mut dest[dest_stride..];
    }
  }
}

fn rdft(
  input: &Aligned<[f32; REAL_SIZE]>,
) -> Aligned<[Complex32; COMPLEX_SIZE]> {
  const SHAPE: [usize; 3] = [TEMPORAL_SIZE, BLOCK_SIZE, BLOCK_SIZE];

  let mut output = Aligned::new([Complex32::zero(); COMPLEX_SIZE]);

  for i in 0..(SHAPE[0] * SHAPE[1]) {
    dft(
      &mut output[(i * (SHAPE[1] / 2 + 1))..],
      DftInput::Real(&input[(i * SHAPE[1])..]),
      SHAPE[2],
      1,
    );
  }

  let mut output2 = Aligned::new([Complex32::zero(); COMPLEX_SIZE]);

  let stride = SHAPE[2] / 2 + 1;
  for i in 0..SHAPE[0] {
    for j in 0..stride {
      dft(
        &mut output2[(i * SHAPE[1] * stride + j)..],
        DftInput::Complex(&output[(i * SHAPE[1] * stride + j)..]),
        SHAPE[1],
        stride,
      );
    }
  }

  let stride = SHAPE[1] * stride;
  for i in 0..stride {
    dft(&mut output[i..], DftInput::Complex(&output2[i..]), SHAPE[0], stride);
  }

  output
}

enum DftInput<'a> {
  Real(&'a [f32]),
  Complex(&'a [Complex32]),
}

#[inline(always)]
fn dft(output: &mut [Complex32], input: DftInput, n: usize, stride: usize) {
  match input {
    DftInput::Real(input) => {
      let out_num = n / 2 + 1;
      for i in 0..out_num {
        let mut sum = Complex32::zero();
        for j in 0..n {
          let imag = -2f32 * i as f32 * j as f32 * PI / n as f32;
          let weight = Complex32::new(imag.cos(), imag.sin());
          sum += input[j * stride] * weight;
        }
        output[i * stride] = sum;
      }
    }
    DftInput::Complex(input) => {
      let out_num = n;
      for i in 0..out_num {
        let mut sum = Complex32::zero();
        for j in 0..n {
          let imag = -2f32 * i as f32 * j as f32 * PI / n as f32;
          let weight = Complex32::new(imag.cos(), imag.sin());
          sum += input[j * stride] * weight;
        }
        output[i * stride] = sum;
      }
    }
  }
}

#[inline(always)]
fn calc_pad_size(size: usize) -> usize {
  size
    + if size % BLOCK_SIZE > 0 { BLOCK_SIZE - size % BLOCK_SIZE } else { 0 }
    + BLOCK_SIZE
    - BLOCK_STEP.max(BLOCK_STEP) * 2
}
