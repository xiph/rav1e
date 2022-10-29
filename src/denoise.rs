use crate::api::FrameQueue;
use crate::util::{cast, Aligned};
use crate::EncoderStatus;
use arrayvec::ArrayVec;
use num_complex::Complex64;
use num_traits::Zero;
use std::f32::consts::PI;
use std::f64::consts::PI as PI64;
use std::iter::once;
use std::ptr::copy_nonoverlapping;
use std::sync::Arc;
use v_frame::frame::Frame;
use v_frame::pixel::Pixel;
use v_frame::plane::Plane;
use wide::f32x8;

pub const TEMPORAL_RADIUS: usize = 1;
const TEMPORAL_SIZE: usize = TEMPORAL_RADIUS * 2 + 1;
const BLOCK_SIZE: usize = 16;
const BLOCK_STEP: usize = 12;
const REAL_SIZE: usize = TEMPORAL_SIZE * BLOCK_SIZE * BLOCK_SIZE;
const COMPLEX_SIZE: usize = TEMPORAL_SIZE * BLOCK_SIZE * (BLOCK_SIZE / 2 + 1);

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
  // Local values
  sigma: Aligned<[f32; COMPLEX_SIZE]>,
  window: Aligned<[f32; REAL_SIZE]>,
  window_freq: Aligned<[Complex64; COMPLEX_SIZE]>,
  pmin: f32,
  pmax: f32,
}

impl<T> DftDenoiser<T>
where
  T: Pixel,
{
  // This should only need to run once per video.
  pub fn new(sigma: f32) -> Self {
    let window = build_window();
    let wscale = window.iter().copied().map(|w| w * w).sum::<f32>();
    let sigma = sigma as f32 * wscale;
    let pmin = 0.0f32;
    let pmax = 500.0f32 * wscale;
    let mut window_freq_real = Aligned::new([0f64; REAL_SIZE]);
    window_freq_real.iter_mut().zip(window.iter()).for_each(|(freq, w)| {
      *freq = *w as f64 * 255.0;
    });
    let window_freq = rdft(&window_freq_real);
    let sigma = Aligned::new([sigma; COMPLEX_SIZE]);

    Self {
      prev_frame: None,
      cur_frameno: 0,
      sigma,
      window,
      window_freq,
      pmin,
      pmax,
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

    let orig_frame = frame_q.get(&self.cur_frameno).unwrap().as_ref().unwrap();
    let frames = once(self.prev_frame.clone())
      .chain(once(Some(Arc::clone(orig_frame))))
      .chain(next_frame.cloned())
      .collect::<ArrayVec<Option<_>, 3>>();

    todo!();
    // let mut dest = (**orig_frame).clone();
    // let mut pad = ArrayVec::<_, TB_SIZE>::new();
    // for i in 0..TB_SIZE {
    //   let dec = self.chroma_sampling.get_decimation().unwrap_or((0, 0));
    //   let mut pad_frame = [
    //     Plane::new(
    //       self.pad_dimensions[0].0,
    //       self.pad_dimensions[0].1,
    //       0,
    //       0,
    //       0,
    //       0,
    //     ),
    //     Plane::new(
    //       self.pad_dimensions[1].0,
    //       self.pad_dimensions[1].1,
    //       dec.0,
    //       dec.1,
    //       0,
    //       0,
    //     ),
    //     Plane::new(
    //       self.pad_dimensions[2].0,
    //       self.pad_dimensions[2].1,
    //       dec.0,
    //       dec.1,
    //       0,
    //       0,
    //     ),
    //   ];

    //   let frame = frames.get(&i).unwrap_or(&frames[&TEMP_RADIUS]);
    //   self.copy_pad(frame, &mut pad_frame);
    //   pad.push(pad_frame);
    // }
    // self.do_filtering(&pad, &mut dest);

    self.prev_frame = Some(Arc::clone(orig_frame));
    self.cur_frameno += 1;

    // Ok(dest)
  }

  fn do_filtering(&mut self, src: &[[Plane<T>; 3]], dest: &mut Frame<T>) {
    todo!();
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
fn bitblt<T: Pixel>(
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
  input: &Aligned<[f64; REAL_SIZE]>,
) -> Aligned<[Complex64; COMPLEX_SIZE]> {
  const SHAPE: [usize; 3] = [TEMPORAL_SIZE, BLOCK_SIZE, BLOCK_SIZE];

  let mut output = Aligned::new([Complex64::zero(); COMPLEX_SIZE]);

  for i in 0..(SHAPE[0] * SHAPE[1]) {
    dft(
      &mut output[(i * (SHAPE[1] / 2 + 1))..],
      DftInput::Real(&input[(i * SHAPE[1])..]),
      SHAPE[2],
      1,
    );
  }

  let mut output2 = Aligned::new([Complex64::zero(); COMPLEX_SIZE]);

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
  Real(&'a [f64]),
  Complex(&'a [Complex64]),
}

#[inline(always)]
fn dft(output: &mut [Complex64], input: DftInput, n: usize, stride: usize) {
  match input {
    DftInput::Real(input) => {
      let out_num = n / 2 + 1;
      for i in 0..out_num {
        let mut sum = Complex64::zero();
        for j in 0..n {
          let imag = -2f64 * i as f64 * j as f64 * PI64 / n as f64;
          let weight = Complex64::new(imag.cos(), imag.sin());
          sum += input[j * stride] * weight;
        }
        output[i * stride] = sum;
      }
    }
    DftInput::Complex(input) => {
      let out_num = n;
      for i in 0..out_num {
        let mut sum = Complex64::zero();
        for j in 0..n {
          let imag = -2f64 * i as f64 * j as f64 * PI64 / n as f64;
          let weight = Complex64::new(imag.cos(), imag.sin());
          sum += input[j * stride] * weight;
        }
        output[i * stride] = sum;
      }
    }
  }
}
