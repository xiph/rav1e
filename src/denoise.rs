use crate::api::FrameQueue;
use crate::util::Aligned;
use crate::EncoderStatus;
use arrayvec::ArrayVec;
use std::f32::consts::PI;
use std::iter::once;
use std::mem::{size_of, transmute, MaybeUninit};
use std::sync::Arc;
use v_frame::frame::Frame;
use v_frame::pixel::{ChromaSampling, Pixel};
use v_frame::plane::Plane;

pub const TEMPORAL_RADIUS: usize = 1;
const TEMPORAL_SIZE: usize = TEMPORAL_RADIUS * 2 + 1;
const BLOCK_SIZE: usize = 16;
const BLOCK_STEP: usize = 12;
const REAL_TOTAL: usize = TEMPORAL_SIZE * BLOCK_SIZE * BLOCK_SIZE;
const COMPLEX_TOTAL: usize = TEMPORAL_SIZE * BLOCK_SIZE * (BLOCK_SIZE / 2 + 1);

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
  sigma: Aligned<[f32; COMPLEX_TOTAL]>,
  window: Aligned<[f32; REAL_TOTAL]>,
  window_freq: Aligned<[f32; REAL_TOTAL]>,
  pmin: f32,
  pmax: f32,
}

impl<T> DftDenoiser<T>
where
  T: Pixel,
{
  // This should only need to run once per video.
  pub fn new(
    sigma: f32, width: usize, height: usize, bit_depth: u8,
    chroma_sampling: ChromaSampling,
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
    // SAFETY: The `assume_init` is safe because the type we are claiming to have
    // initialized here is a bunch of `MaybeUninit`s, which do not require initialization.
    let mut window_freq_temp: Aligned<[MaybeUninit<f32>; REAL_TOTAL]> =
      Aligned::new(unsafe { MaybeUninit::uninit().assume_init() });
    window_freq_temp.iter_mut().zip(window.iter()).for_each(|(freq, w)| {
      freq.write(*w * 255.0);
    });
    // SAFETY: Everything is initialized. Transmute the array to the
    // initialized type.
    let window_freq_temp: Aligned<[f32; REAL_TOTAL]> =
      unsafe { transmute(window_freq_temp) };
    let window_freq = rdft(&window_freq_temp);
    let sigma = Aligned::new([sigma; COMPLEX_TOTAL]);

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

pub fn build_window() -> Aligned<[f32; REAL_TOTAL]> {
  let temporal_window = [temporal_window_value(); TEMPORAL_SIZE];

  // SAFETY: The `assume_init` is safe because the type we are claiming to have
  // initialized here is a bunch of `MaybeUninit`s, which do not require initialization.
  let mut spatial_window: [MaybeUninit<f32>; BLOCK_SIZE] =
    unsafe { MaybeUninit::uninit().assume_init() };
  spatial_window.iter_mut().enumerate().for_each(|(i, val)| {
    val.write(spatial_window_value(i as f32 + 0.5));
  });
  // SAFETY: Everything is initialized. Transmute the array to the
  // initialized type.
  let spatial_window: [f32; BLOCK_SIZE] = unsafe { transmute(spatial_window) };
  let spatial_window = normalize(&spatial_window);

  let mut window: Aligned<[MaybeUninit<f32>; REAL_TOTAL]> =
    Aligned::new(unsafe { MaybeUninit::uninit().assume_init() });
  let mut window_iter = window.iter_mut();
  for t_val in temporal_window {
    for s_val1 in spatial_window {
      for s_val2 in spatial_window {
        let mut value = t_val * s_val1 * s_val2;
        // normalize for unnormalized FFT implementation
        value /= (TEMPORAL_SIZE as f32).sqrt() * BLOCK_SIZE as f32;
        window_iter.next().unwrap().write(value);
      }
    }
  }
  // SAFETY: Everything is initialized. Transmute the array to the
  // initialized type.
  unsafe { transmute(window) }
}

pub fn normalize(window: &[f32; BLOCK_SIZE]) -> [f32; BLOCK_SIZE] {
  let mut new_window = [0f32; BLOCK_SIZE];
  for q in 0..BLOCK_SIZE {
    for h in (0..=q).rev().step_by(BLOCK_STEP) {
      new_window[q] += window[h].powi(2);
    }
    for h in ((q + BLOCK_STEP)..BLOCK_SIZE).step_by(BLOCK_STEP) {
      new_window[q] += window[h].powi(2);
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

fn rdft(data: &Aligned<[f32; REAL_TOTAL]>) -> Aligned<[f32; REAL_TOTAL]> {
  todo!();
}
