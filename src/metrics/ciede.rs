use crate::util::CastFromPrimitive;
use crate::{Frame, Pixel};
use ndarray::{arr2, Array2, Array3, ArrayBase, Axis, Data, Ix2, Zip};
use palette::encoding::srgb::Srgb;
use palette::white_point::D65;
use std::f64;
use std::f64::consts::PI;

type Lab = palette::Lab<D65, f64>;
type Rgb = palette::rgb::Rgb<Srgb, f64>;

pub(crate) fn calculate_frame_ciede<T: Pixel>(
  frame1: &Frame<T>, frame2: &Frame<T>, bit_depth: usize,
) -> f64 {
  let frame1_rgb = frame_yuv_to_rgb(frame1, bit_depth);
  let frame2_rgb = frame_yuv_to_rgb(frame2, bit_depth);
  let frame1_lab = frame_rgb_to_lab(frame1_rgb);
  let frame2_lab = frame_rgb_to_lab(frame2_rgb);
  let ciede = ciede2000(frame1_lab, frame2_lab, 0.65, 1., 4.);
  45. - 20. * ciede.mean_axis(Axis(1)).mean_axis(Axis(0)).into_scalar().log10()
}

fn frame_yuv_to_rgb<T: Pixel>(
  frame: &Frame<T>, bit_depth: usize,
) -> Array3<f64> {
  // Assuming BT.709
  let yuv2rgb =
    arr2(&[[1., 0., 1.28033], [1., -0.21482, -0.38059], [1., 2.12798, 0.]]);

  let yuv444 = frame_to_yuv444_matrix(frame, bit_depth);
  dot_3d(yuv444, yuv2rgb.t())
}

/// If `a` is an N-D array and `b` is an M-D array (where `M>=2`), it is a
/// sum product over the last axis of `a` and the second-to-last axis of `b`::
///
/// `dot(a, b)[i,j,k] = sum(a[i,j,:] * b[:,k])`
///
/// This currently only supports the use case needed for converting YUV444 to RGB here.
fn dot_3d<S: Data<Elem = f64>>(
  a: Array3<f64>, b: ArrayBase<S, Ix2>,
) -> Array3<f64> {
  let mut result =
    Array3::zeros((a.len_of(Axis(0)), a.len_of(Axis(1)), a.len_of(Axis(2))));
  for i in 0..a.len_of(Axis(0)) {
    for j in 0..a.len_of(Axis(1)) {
      for k in 0..b.len_of(Axis(1)) {
        let prod = a.index_axis(Axis(0), i).index_axis(Axis(0), j).to_owned()
          * b.index_axis(Axis(1), k);
        result[[i, j, k]] = prod.sum();
      }
    }
  }
  result
}

/// This is more accurate than the Python implementation,
/// because scikit rounds factors used in conversion.
/// The `palette` crate uses exact factors calculated by dividing
/// whole numbers (e.g. `(6.0 / 29.0).powi(3)` instead of `0.008856`).
fn frame_rgb_to_lab(frame: Array3<f64>) -> Array2<Lab> {
  frame.map_axis(Axis(2), |rgb| {
    let rgb: Rgb = Rgb::new(rgb[0], rgb[1], rgb[2]);
    Lab::from(rgb)
  })
}

/// Converts the `Frame` representation of a YUV video frame,
/// with an arbitrary color sampling, into a YUV444 3-dim matrix
/// scaled for conversion to RGB.
fn frame_to_yuv444_matrix<T: Pixel>(
  frame: &Frame<T>, bit_depth: usize,
) -> Array3<f64> {
  let width = frame.planes[0].cfg.width;
  let height = frame.planes[0].cfg.height;
  let mut yuv444 = Array3::zeros((height, width, 3));
  for y in 0..height {
    for x in 0..width {
      let pix_y = (i16::cast_from(frame.planes[0].p(x, y)) * 8
        / bit_depth as i16) as f64;
      let pix_u = (i16::cast_from(
        frame.planes[1]
          .p(x >> frame.planes[1].cfg.xdec, y >> frame.planes[1].cfg.ydec),
      ) * 8
        / bit_depth as i16) as f64;
      let pix_v = (i16::cast_from(
        frame.planes[2]
          .p(x >> frame.planes[2].cfg.xdec, y >> frame.planes[2].cfg.ydec),
      ) * 8
        / bit_depth as i16) as f64;
      yuv444[[y, x, 0]] = (pix_y - 16.) / 219.;
      yuv444[[y, x, 1]] = (pix_u - 128.) / 224.;
      yuv444[[y, x, 2]] = (pix_v - 128.) / 224.;
    }
  }
  yuv444
}

/// Color difference as given by the CIEDE 2000 standard.
fn ciede2000(
  lab1: Array2<Lab>, lab2: Array2<Lab>, light_k: f64, chroma_k: f64,
  hue_k: f64,
) -> Array2<f64> {
  let l1 = lab1.map(|lab| lab.l);
  let a1 = lab1.map(|lab| lab.a);
  let b1 = lab1.map(|lab| lab.b);
  let l2 = lab2.map(|lab| lab.l);
  let a2 = lab2.map(|lab| lab.a);
  let b2 = lab2.map(|lab| lab.b);

  // distort `a` based on average chroma
  // then convert to lch coordinates from distorted `a`
  // all subsequence calculations are in the new coordinates
  // (often denoted "prime" in the literature)
  let cbar = 0.5 * (hypotenuse(&a1, &b1) + hypotenuse(&a2, &b2));
  let c7 = cbar.map(|val| val.powi(7));
  let g = 0.5
    * (1. - (c7.clone() / (&c7 + 25u64.pow(7) as f64)).map(|val| val.sqrt()));
  let scale = 1. + &g;
  let (c1, h1) = cart2polar_2pi(&(&a1 * &scale), &b1);
  let (c2, h2) = cart2polar_2pi(&(&a2 * &scale), &b2);

  // recall that c, h are polar coordinates.  c==r, h==theta
  //
  // ciede2000 has four terms to delta_e:
  // 1) Luminance term
  // 2) Hue term
  // 3) Chroma term
  // 4) hue Rotation term

  // lightness term
  let lbar = 0.5 * (&l1 + &l2);
  let tmp = (&lbar - 50.).map(|&val| val.powi(2));
  let sl = 1. + 0.015 * &tmp / (20. + &tmp).map(|&val| val.sqrt());
  let l_term = (&l2 - &l1) / (light_k * &sl);

  // chroma term
  let cbar = 0.5 * (&c1 + &c2); // new coordinates
  let sc = 1. + 0.045 * &cbar;
  let c_term = (&c2 - &c1) / (chroma_k * &sc);

  // hue term
  let h_diff = &h2 - &h1;
  let h_sum = &h1 + &h2;
  let cc = &c1 * &c2;

  let mut dh = h_diff.clone();
  Zip::from(&mut dh).and(&cc).apply(|val, &cc| {
    if *val > PI {
      *val -= 2. * PI;
    }
    if *val < -PI {
      *val += 2. * PI;
    }
    if cc.abs() < f64::EPSILON {
      // if r == 0, dtheta == 0
      *val = 0.;
    }
  });
  let dh_term = 2. * cc.map(|cc| cc.sqrt()) * (dh / 2.).map(|dh| dh.sin());

  let mut hbar = h_sum.clone();
  let mut mask = cc.map(|_| false);
  Zip::from(&mut mask).and(&cc).and(&h_diff).apply(|val, &cc, &h_diff| {
    *val = cc.abs() > f64::EPSILON && h_diff.abs() > PI;
  });
  Zip::from(&mut hbar).and(&h_sum).and(&cc).and(&mask).apply(
    |val, &h_sum, &cc, &mask| {
      if mask && h_sum < 2. * PI {
        *val += 2. * PI;
      }
      if mask && h_sum >= 2. * PI {
        *val -= 2. * PI;
      }
      if cc.abs() < f64::EPSILON {
        *val *= 2.;
      }
      *val *= 0.5;
    },
  );

  let t = 1. - 0.17 * (&hbar - 30f64.to_radians()).map(|val| val.cos())
    + 0.24 * (2. * &hbar).map(|val| val.cos())
    + 0.32 * (3. * &hbar + 6f64.to_radians()).map(|val| val.cos())
    - 0.20 * (4. * &hbar - 63f64.to_radians()).map(|val| val.cos());
  let sh = 1. + 0.015 * &cbar * t;

  let h_term = dh_term / (hue_k * sh);

  // hue rotation
  let c7 = cbar.map(|val| val.powi(7));
  let rc =
    2. * (c7.clone() / (c7 + 25u64.pow(7) as f64)).map(|val| val.sqrt());
  let dtheta = 30f64.to_radians()
    * (-1.
      * ((hbar.map(|val| val.to_degrees()) - 275.) / 25.)
        .map(|val| val.powi(2)))
    .map(|val| val.exp());
  let r_term =
    -1. * (2. * dtheta).map(|val| val.sin()) * &rc * &c_term * &h_term;

  // put it all together
  let mut de2 = l_term.map(|val| val.powi(2));
  Zip::from(&mut de2).and(&c_term).and(&h_term).and(&r_term).apply(
    |val, &c_term, &h_term, &r_term| {
      *val += c_term.powi(2);
      *val += h_term.powi(2);
      *val += r_term;
    },
  );
  de2.map(|val| val.sqrt())
}

/// convert cartesian coordinates to polar (uses non-standard theta range!)
///
/// NON-STANDARD RANGE! Maps to `(0, 2*pi)` rather than usual `(-pi, +pi)`
fn cart2polar_2pi(
  x: &Array2<f64>, y: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>) {
  let r = hypotenuse(x, y);
  let t = arctan2(y, x).map(|&val| val + if val < 0. { 2. * PI } else { 0. });
  (r, t)
}

fn hypotenuse(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
  let mut results = a.clone();
  Zip::from(&mut results).and(b).apply(|a, &b| *a = a.hypot(b));
  results
}

fn arctan2(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
  let mut results = a.clone();
  Zip::from(&mut results).and(b).apply(|a, &b| *a = a.atan2(b));
  results
}
