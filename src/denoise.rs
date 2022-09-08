use crate::api::FrameQueue;
use crate::util::Aligned;
use crate::EncoderStatus;
use arrayvec::ArrayVec;
use ndarray::{ArrayView3, ArrayViewMut3};
use ndrustfft::{
  ndfft, ndfft_r2c, ndifft, ndifft_r2c, Complex, FftHandler, R2cFftHandler,
};
use std::collections::{BTreeMap, VecDeque};
use std::f64::consts::PI;
use std::iter::once;
use std::mem::size_of;
use std::ptr::copy_nonoverlapping;
use std::sync::Arc;
use v_frame::frame::Frame;
use v_frame::math::clamp;
use v_frame::pixel::{CastFromPrimitive, ChromaSampling, Pixel};
use v_frame::plane::Plane;

const SB_SIZE: usize = 16;
const SO_SIZE: usize = 12;
const TB_SIZE: usize = 3;
pub(crate) const TB_MIDPOINT: usize = TB_SIZE / 2;
const BLOCK_AREA: usize = SB_SIZE * SB_SIZE;
const BLOCK_VOLUME: usize = BLOCK_AREA * TB_SIZE;
const COMPLEX_COUNT: usize = (SB_SIZE / 2 + 1) * SB_SIZE * TB_SIZE;
const CCNT2: usize = COMPLEX_COUNT * 2;
const INC: usize = SB_SIZE - SO_SIZE;

/// This denoiser is based on the DFTTest plugin from Vapoursynth.
/// This type of denoising was chosen because it provides
/// high quality while not being too slow.
pub struct DftDenoiser<T>
where
  T: Pixel,
{
  chroma_sampling: ChromaSampling,
  dest_scale: f32,
  src_scale: f32,
  peak: T,

  // These indices refer to planes of the input
  pad_dimensions: ArrayVec<(usize, usize), 3>,
  effective_heights: ArrayVec<usize, 3>,

  hw: Aligned<[f32; BLOCK_VOLUME]>,
  dftgc: Aligned<[Complex<f32>; COMPLEX_COUNT]>,
  fft: (R2cFftHandler<f32>, FftHandler<f32>, FftHandler<f32>),
  sigmas: Aligned<[f32; CCNT2]>,

  // This stores a copy of the unfiltered previous frame,
  // since in `frame_q` it will be filtered already.
  // We only have one frame, but it's left as a Vec so that
  // TB_SIZE could potentially be tweaked without any
  // code changes.
  frame_buffer: VecDeque<Arc<Frame<T>>>,
  pub(crate) cur_frameno: u64,
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

    let dest_scale = (1 << (bit_depth - 8)) as f32;
    let src_scale = 1.0 / dest_scale;
    let peak = T::cast_from((1u16 << bit_depth) - 1);

    let mut pad_dimensions = ArrayVec::<_, 3>::new();
    let mut effective_heights = ArrayVec::<_, 3>::new();
    for plane in 0..3 {
      let ae = (SB_SIZE - SO_SIZE).max(SO_SIZE) * 2;
      let (width, height) = if plane == 0 {
        (width, height)
      } else {
        chroma_sampling.get_chroma_dimensions(width, height)
      };
      let pad_w = width + extra(width, SB_SIZE) + ae;
      let pad_h = height + extra(height, SB_SIZE) + ae;
      let e_h =
        ((pad_h - SO_SIZE) / (SB_SIZE - SO_SIZE)) * (SB_SIZE - SO_SIZE);
      pad_dimensions.push((pad_w, pad_h));
      effective_heights.push(e_h);
    }

    let hw = Aligned::new(Self::create_window());
    let mut dftgr = Aligned::new([0f32; BLOCK_VOLUME]);

    let fft = (
      R2cFftHandler::new(SB_SIZE),
      FftHandler::new(SB_SIZE),
      FftHandler::new(TB_SIZE),
    );

    let mut wscale = 0.0f32;
    for k in 0..BLOCK_VOLUME {
      dftgr[k] = 255.0 * hw[k];
      wscale += hw[k].powi(2);
    }
    let wscale = 1.0 / wscale;

    let mut sigmas = Aligned::new([0f32; CCNT2]);
    sigmas.fill(sigma / wscale);

    let mut denoiser = DftDenoiser {
      chroma_sampling,
      dest_scale,
      src_scale,
      peak,
      pad_dimensions,
      effective_heights,
      hw,
      fft,
      sigmas,
      dftgc: Aligned::new([Complex::default(); COMPLEX_COUNT]),
      frame_buffer: VecDeque::with_capacity(TB_MIDPOINT),
      cur_frameno: 0,
    };

    let mut dftgc = Aligned::new([Complex::default(); COMPLEX_COUNT]);
    denoiser.real_to_complex_3d(&dftgr, &mut dftgc);
    denoiser.dftgc = dftgc;

    denoiser
  }

  pub fn filter_frame(
    &mut self, frame_q: &FrameQueue<T>,
  ) -> Result<Frame<T>, EncoderStatus> {
    if self.frame_buffer.len() < TB_MIDPOINT.min(self.cur_frameno as usize) {
      // We need to have the previous unfiltered frame
      // in the buffer for temporal filtering.
      return Err(EncoderStatus::NeedMoreData);
    }
    let future_frames = frame_q
      .range((self.cur_frameno + 1)..)
      .take(TB_MIDPOINT)
      .map(|(_, f)| f)
      .collect::<ArrayVec<_, TB_MIDPOINT>>();
    if future_frames.len() != TB_MIDPOINT
      && !future_frames.iter().any(|f| f.is_none())
    {
      // We also need to have the next unfiltered frame,
      // unless we are at the end of the video.
      return Err(EncoderStatus::NeedMoreData);
    }

    let orig_frame = frame_q.get(&self.cur_frameno).unwrap().as_ref().unwrap();
    let frames = self
      .frame_buffer
      .iter()
      .cloned()
      .enumerate()
      .chain(once(((TB_MIDPOINT), Arc::clone(orig_frame))))
      .chain(
        future_frames
          .into_iter()
          .flatten()
          .cloned()
          .enumerate()
          .map(|(i, f)| (i + 1 + TB_MIDPOINT, f)),
      )
      .collect::<BTreeMap<_, _>>();

    let mut dest = (**orig_frame).clone();
    let mut pad = ArrayVec::<_, TB_SIZE>::new();
    for i in 0..TB_SIZE {
      let dec = self.chroma_sampling.get_decimation().unwrap_or((0, 0));
      let mut pad_frame = [
        Plane::new(
          self.pad_dimensions[0].0,
          self.pad_dimensions[0].1,
          0,
          0,
          0,
          0,
        ),
        Plane::new(
          self.pad_dimensions[1].0,
          self.pad_dimensions[1].1,
          dec.0,
          dec.1,
          0,
          0,
        ),
        Plane::new(
          self.pad_dimensions[2].0,
          self.pad_dimensions[2].1,
          dec.0,
          dec.1,
          0,
          0,
        ),
      ];

      let frame = frames.get(&i).unwrap_or(&frames[&TB_MIDPOINT]);
      self.copy_pad(frame, &mut pad_frame);
      pad.push(pad_frame);
    }
    self.do_filtering(&pad, &mut dest);

    if self.frame_buffer.len() == TB_MIDPOINT {
      self.frame_buffer.pop_front();
    }
    self.frame_buffer.push_back(Arc::clone(orig_frame));
    self.cur_frameno += 1;

    Ok(dest)
  }

  fn do_filtering(&mut self, src: &[[Plane<T>; 3]], dest: &mut Frame<T>) {
    let mut dftr = [0f32; BLOCK_VOLUME];
    let mut dftc = [Complex::<f32>::default(); COMPLEX_COUNT];
    let mut means = [Complex::<f32>::default(); COMPLEX_COUNT];

    for p in 0..3 {
      let (pad_width, pad_height) = self.pad_dimensions[p];
      let mut ebuff = vec![0f32; pad_width * pad_height];
      let effective_height = self.effective_heights[p];
      let src_stride = src[0][p].cfg.stride;
      let ebuff_stride = pad_width;

      let mut src_planes = src
        .iter()
        .map(|f| f[p].data_origin())
        .collect::<ArrayVec<_, TB_SIZE>>();

      // SAFETY: We know the size of the planes we're working on,
      // so we can safely ensure we are not out of bounds.
      // There are a fair number of unsafe function calls here
      // which are unsafe for optimization purposes.
      // All are safe as long as we do not pass out-of-bounds parameters.
      unsafe {
        for y in (0..effective_height).step_by(INC) {
          for x in (0..=(pad_width - SB_SIZE)).step_by(INC) {
            for z in 0..TB_SIZE {
              self.proc0(
                &src_planes[z][x..],
                &self.hw[(BLOCK_AREA * z)..],
                &mut dftr[(BLOCK_AREA * z)..],
                src_stride,
                SB_SIZE,
                self.src_scale,
              );
            }

            self.real_to_complex_3d(&dftr, &mut dftc);
            self.remove_mean(&mut dftc, &self.dftgc, &mut means);

            self.filter_coeffs(&mut dftc);

            self.add_mean(&mut dftc, &means);
            self.complex_to_real_3d(&dftc, &mut dftr);

            self.proc1(
              &dftr[(TB_MIDPOINT * BLOCK_AREA)..],
              &self.hw[(TB_MIDPOINT * BLOCK_AREA)..],
              &mut ebuff[(y * ebuff_stride + x)..],
              SB_SIZE,
              ebuff_stride,
            );
          }

          for q in 0..TB_SIZE {
            src_planes[q] = &src_planes[q][(INC * src_stride)..];
          }
        }
      }

      let dest_width = dest.planes[p].cfg.width;
      let dest_height = dest.planes[p].cfg.height;
      let dest_stride = dest.planes[p].cfg.stride;
      let dest_plane = dest.planes[p].data_origin_mut();
      let ebp_offset = ebuff_stride * ((pad_height - dest_height) / 2)
        + (pad_width - dest_width) / 2;
      let ebp = &ebuff[ebp_offset..];

      self.cast(
        ebp,
        dest_plane,
        dest_width,
        dest_height,
        dest_stride,
        ebuff_stride,
      );
    }
  }

  fn create_window() -> [f32; BLOCK_VOLUME] {
    let mut hw = [0f32; BLOCK_VOLUME];
    let mut tw = [0f64; TB_SIZE];
    let mut sw = [0f64; SB_SIZE];

    tw.fill_with(Self::temporal_window);
    sw.iter_mut().enumerate().for_each(|(j, sw)| {
      *sw = Self::spatial_window(j as f64 + 0.5);
    });
    Self::normalize_for_overlap_add(&mut sw);

    let nscale = 1.0 / (BLOCK_VOLUME as f64).sqrt();
    for j in 0..TB_SIZE {
      for k in 0..SB_SIZE {
        for q in 0..SB_SIZE {
          hw[(j * SB_SIZE + k) * SB_SIZE + q] =
            (tw[j] * sw[k] * sw[q] * nscale) as f32;
        }
      }
    }

    hw
  }

  #[inline(always)]
  // Hanning windowing
  fn spatial_window(n: f64) -> f64 {
    0.5 - 0.5 * (2.0 * PI * n / SB_SIZE as f64).cos()
  }

  #[inline(always)]
  // Simple rectangular windowing
  fn temporal_window() -> f64 {
    1.0
  }

  // Accounts for spatial block overlap
  fn normalize_for_overlap_add(hw: &mut [f64]) {
    let inc = SB_SIZE - SO_SIZE;

    let mut nw = [0f64; SB_SIZE];
    let hw = &mut hw[..SB_SIZE];

    for q in 0..SB_SIZE {
      for h in (0..=q).rev().step_by(inc) {
        nw[q] += hw[h].powi(2);
      }
      for h in ((q + inc)..SB_SIZE).step_by(inc) {
        nw[q] += hw[h].powi(2);
      }
    }

    for q in 0..SB_SIZE {
      hw[q] /= nw[q].sqrt();
    }
  }

  #[inline]
  unsafe fn proc0(
    &self, s0: &[T], s1: &[f32], dest: &mut [f32], p0: usize, p1: usize,
    src_scale: f32,
  ) {
    let s0 = s0.as_ptr();
    let s1 = s1.as_ptr();
    let dest = dest.as_mut_ptr();

    for u in 0..p1 {
      for v in 0..p1 {
        let s0 = s0.add(u * p0 + v);
        let s1 = s1.add(u * p1 + v);
        let dest = dest.add(u * p1 + v);
        dest.write(u16::cast_from(s0.read()) as f32 * src_scale * s1.read())
      }
    }
  }

  #[inline]
  unsafe fn proc1(
    &self, s0: &[f32], s1: &[f32], dest: &mut [f32], p0: usize, p1: usize,
  ) {
    let s0 = s0.as_ptr();
    let s1 = s1.as_ptr();
    let dest = dest.as_mut_ptr();

    for u in 0..p0 {
      for v in 0..p0 {
        let s0 = s0.add(u * p0 + v);
        let s1 = s1.add(u * p0 + v);
        let dest = dest.add(u * p1 + v);
        dest.write(s0.read().mul_add(s1.read(), dest.read()));
      }
    }
  }

  #[inline]
  fn remove_mean(
    &self, dftc: &mut [Complex<f32>; COMPLEX_COUNT],
    dftgc: &[Complex<f32>; COMPLEX_COUNT],
    means: &mut [Complex<f32>; COMPLEX_COUNT],
  ) {
    let gf = dftc[0].re / dftgc[0].re;

    for h in 0..COMPLEX_COUNT {
      means[h].re = gf * dftgc[h].re;
      means[h].im = gf * dftgc[h].im;
      dftc[h].re -= means[h].re;
      dftc[h].im -= means[h].im;
    }
  }

  #[inline]
  fn add_mean(
    &self, dftc: &mut [Complex<f32>; COMPLEX_COUNT],
    means: &[Complex<f32>; COMPLEX_COUNT],
  ) {
    for h in 0..COMPLEX_COUNT {
      dftc[h].re += means[h].re;
      dftc[h].im += means[h].im;
    }
  }

  #[inline]
  // Applies a generalized wiener filter
  fn filter_coeffs(&self, dftc: &mut [Complex<f32>; COMPLEX_COUNT]) {
    for h in 0..COMPLEX_COUNT {
      let psd = dftc[h].re.mul_add(dftc[h].re, dftc[h].im.powi(2));
      let mult = ((psd - self.sigmas[h]) / (psd + 1e-15)).max(0.0);
      dftc[h].re *= mult;
      dftc[h].im *= mult;
    }
  }

  fn copy_pad(&self, src: &Frame<T>, dest: &mut [Plane<T>; 3]) {
    for p in 0..src.planes.len() {
      let src_width = src.planes[p].cfg.width;
      let dest_width = dest[p].cfg.width;
      let src_height = src.planes[p].cfg.height;
      let dest_height = dest[p].cfg.height;
      let src_stride = src.planes[p].cfg.stride;
      let dest_stride = dest[p].cfg.stride;

      let offy = (dest_height - src_height) / 2;
      let offx = (dest_width - src_width) / 2;

      bitblt(
        &mut dest[p].data_origin_mut()[(dest_stride * offy + offx)..],
        dest_stride,
        src.planes[p].data_origin(),
        src_stride,
        src_width,
        src_height,
      );

      let mut dest_ptr =
        &mut dest[p].data_origin_mut()[(dest_stride * offy)..];
      for _ in offy..(src_height + offy) {
        let dest_slice = &mut dest_ptr[..dest_width];

        let mut w = offx * 2;
        for x in 0..offx {
          dest_slice[x] = dest_slice[w];
          w -= 1;
        }

        w = offx + src_width - 2;
        for x in (offx + src_width)..dest_width {
          dest_slice[x] = dest_slice[w];
          w -= 1;
        }

        dest_ptr = &mut dest_ptr[dest_stride..];
      }

      let dest_origin = dest[p].data_origin_mut();
      let mut w = offy * 2;
      for y in 0..offy {
        // SAFETY: `copy_from_slice` has borrow checker issues here
        // because we are copying from `dest` to `dest`, but we manually
        // know that the two slices will not overlap. We still slice
        // the start and end as a safety check.
        unsafe {
          copy_nonoverlapping(
            dest_origin[(dest_stride * w)..][..dest_width].as_ptr(),
            dest_origin[(dest_stride * y)..][..dest_width].as_mut_ptr(),
            dest_width,
          );
        }
        w -= 1;
      }

      w = offy + src_height - 2;
      for y in (offy + src_height)..dest_height {
        // SAFETY: `copy_from_slice` has borrow checker issues here
        // because we are copying from `dest` to `dest`, but we manually
        // know that the two slices will not overlap. We still slice
        // the start and end as a safety check.
        unsafe {
          copy_nonoverlapping(
            dest_origin[(dest_stride * w)..][..dest_width].as_ptr(),
            dest_origin[(dest_stride * y)..][..dest_width].as_mut_ptr(),
            dest_width,
          );
        }
        w -= 1;
      }
    }
  }

  fn cast(
    &self, ebuff: &[f32], dest: &mut [T], dest_width: usize,
    dest_height: usize, dest_stride: usize, ebp_stride: usize,
  ) {
    let ebuff = ebuff.chunks(ebp_stride);
    let dest = dest.chunks_mut(dest_stride);

    for (ebuff, dest) in ebuff.zip(dest).take(dest_height) {
      for x in 0..dest_width {
        let fval = ebuff[x].mul_add(self.dest_scale, 0.5);
        dest[x] =
          clamp(T::cast_from(fval as u16), T::cast_from(0u16), self.peak);
      }
    }
  }

  // Applies a real-to-complex 3-dimensional FFT to `real`
  fn real_to_complex_3d(
    &mut self, real: &[f32; BLOCK_VOLUME],
    output: &mut [Complex<f32>; COMPLEX_COUNT],
  ) {
    let mut temp1_data = [Complex::default(); COMPLEX_COUNT];
    let mut temp2_data = [Complex::default(); COMPLEX_COUNT];
    let input =
      ArrayView3::from_shape((TB_SIZE, SB_SIZE, SB_SIZE), real).unwrap();
    let mut temp1 = ArrayViewMut3::from_shape(
      (TB_SIZE, SB_SIZE, SB_SIZE / 2 + 1),
      &mut temp1_data,
    )
    .unwrap();
    let mut temp2 = ArrayViewMut3::from_shape(
      (TB_SIZE, SB_SIZE, SB_SIZE / 2 + 1),
      &mut temp2_data,
    )
    .unwrap();
    let mut output =
      ArrayViewMut3::from_shape((TB_SIZE, SB_SIZE, SB_SIZE / 2 + 1), output)
        .unwrap();

    ndfft_r2c(&input, &mut temp1, &mut self.fft.0, 2);
    ndfft(&temp1, &mut temp2, &mut self.fft.1, 1);
    ndfft(&temp2, &mut output, &mut self.fft.2, 0);
  }

  // Applies a complex-to-real 3-dimensional FFT to `complex`
  fn complex_to_real_3d(
    &mut self, complex: &[Complex<f32>; COMPLEX_COUNT],
    output: &mut [f32; BLOCK_VOLUME],
  ) {
    let mut temp0_data = [Complex::default(); COMPLEX_COUNT];
    let mut temp1_data = [Complex::default(); COMPLEX_COUNT];
    let input =
      ArrayView3::from_shape((TB_SIZE, SB_SIZE, SB_SIZE / 2 + 1), complex)
        .unwrap();
    let mut temp0 = ArrayViewMut3::from_shape(
      (TB_SIZE, SB_SIZE, SB_SIZE / 2 + 1),
      &mut temp0_data,
    )
    .unwrap();
    let mut temp1 = ArrayViewMut3::from_shape(
      (TB_SIZE, SB_SIZE, SB_SIZE / 2 + 1),
      &mut temp1_data,
    )
    .unwrap();
    let mut output =
      ArrayViewMut3::from_shape((TB_SIZE, SB_SIZE, SB_SIZE), output).unwrap();

    ndifft(&input, &mut temp0, &mut self.fft.2, 0);
    ndifft(&temp0, &mut temp1, &mut self.fft.1, 1);
    ndifft_r2c(&temp1, &mut output, &mut self.fft.0, 2);
    output.iter_mut().for_each(|d| {
      *d *= BLOCK_VOLUME as f32;
    });
  }
}

#[inline(always)]
fn extra(a: usize, b: usize) -> usize {
  if a % b > 0 {
    b - (a % b)
  } else {
    0
  }
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
