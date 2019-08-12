use crate::api::ChromaSampling;
use crate::frame::Plane;
use crate::metrics::FrameMetrics;
use crate::{CastFromPrimitive, Frame, Pixel};

pub(crate) fn calculate_frame_psnr_hvs<T: Pixel>(
  frame1: &Frame<T>, frame2: &Frame<T>, bit_depth: usize, cs: ChromaSampling,
) -> FrameMetrics {
  let cweight = cs.get_chroma_weight();
  let y = calculate_plane_psnr_hvs(
    &frame1.planes[0],
    &frame2.planes[0],
    0,
    bit_depth,
  );
  let u = calculate_plane_psnr_hvs(
    &frame1.planes[1],
    &frame2.planes[1],
    1,
    bit_depth,
  );
  let v = calculate_plane_psnr_hvs(
    &frame1.planes[2],
    &frame2.planes[2],
    2,
    bit_depth,
  );
  FrameMetrics {
    y: log10_convert(y, 1.0),
    u: log10_convert(u, 1.0),
    v: log10_convert(v, 1.0),
    weighted_avg: log10_convert(y + cweight * (u + v), 1.0 + 2.0 * cweight),
  }
}

// Normalized inverse quantization matrix for 8x8 DCT at the point of transparency.
// This is not the JPEG based matrix from the paper,
// this one gives a slightly higher MOS agreement.
#[rustfmt::skip]
const CSF_Y: [[f64; 8]; 8] = [
  [1.6193873005, 2.2901594831, 2.08509755623, 1.48366094411, 1.00227514334, 0.678296995242, 0.466224900598, 0.3265091542],
  [2.2901594831, 1.94321815382, 2.04793073064, 1.68731108984, 1.2305666963, 0.868920337363, 0.61280991668, 0.436405793551],
  [2.08509755623, 2.04793073064, 1.34329019223, 1.09205635862, 0.875748795257, 0.670882927016, 0.501731932449, 0.372504254596],
  [1.48366094411, 1.68731108984, 1.09205635862, 0.772819797575, 0.605636379554, 0.48309405692, 0.380429446972, 0.295774038565],
  [1.00227514334, 1.2305666963, 0.875748795257, 0.605636379554, 0.448996256676, 0.352889268808, 0.283006984131, 0.226951348204],
  [0.678296995242, 0.868920337363, 0.670882927016, 0.48309405692, 0.352889268808, 0.27032073436, 0.215017739696, 0.17408067321],
  [0.466224900598, 0.61280991668, 0.501731932449, 0.380429446972, 0.283006984131, 0.215017739696, 0.168869545842, 0.136153931001],
  [0.3265091542, 0.436405793551, 0.372504254596, 0.295774038565, 0.226951348204, 0.17408067321, 0.136153931001, 0.109083846276]
];

#[rustfmt::skip]
const CSF_CB420: [[f64; 8]; 8] = [
  [1.91113096927, 2.46074210438, 1.18284184739, 1.14982565193, 1.05017074788, 0.898018824055, 0.74725392039, 0.615105596242],
  [2.46074210438, 1.58529308355, 1.21363250036, 1.38190029285, 1.33100189972, 1.17428548929, 0.996404342439, 0.830890433625],
  [1.18284184739, 1.21363250036, 0.978712413627, 1.02624506078, 1.03145147362, 0.960060382087, 0.849823426169, 0.731221236837],
  [1.14982565193, 1.38190029285, 1.02624506078, 0.861317501629, 0.801821139099, 0.751437590932, 0.685398513368, 0.608694761374],
  [1.05017074788, 1.33100189972, 1.03145147362, 0.801821139099, 0.676555426187, 0.605503172737, 0.55002013668, 0.495804539034],
  [0.898018824055, 1.17428548929, 0.960060382087, 0.751437590932, 0.605503172737, 0.514674450957, 0.454353482512, 0.407050308965],
  [0.74725392039, 0.996404342439, 0.849823426169, 0.685398513368, 0.55002013668, 0.454353482512, 0.389234902883, 0.342353999733],
  [0.615105596242, 0.830890433625, 0.731221236837, 0.608694761374, 0.495804539034, 0.407050308965, 0.342353999733, 0.295530605237]
];

#[rustfmt::skip]
const CSF_CR420: [[f64; 8]; 8] = [
  [1.91113096927, 2.46074210438, 1.18284184739, 1.14982565193, 1.05017074788, 0.898018824055, 0.74725392039, 0.615105596242],
  [2.46074210438, 1.58529308355, 1.21363250036, 1.38190029285, 1.33100189972, 1.17428548929, 0.996404342439, 0.830890433625],
  [1.18284184739, 1.21363250036, 0.978712413627, 1.02624506078, 1.03145147362, 0.960060382087, 0.849823426169, 0.731221236837],
  [1.14982565193, 1.38190029285, 1.02624506078, 0.861317501629, 0.801821139099, 0.751437590932, 0.685398513368, 0.608694761374],
  [1.05017074788, 1.33100189972, 1.03145147362, 0.801821139099, 0.676555426187, 0.605503172737, 0.55002013668, 0.495804539034],
  [0.898018824055, 1.17428548929, 0.960060382087, 0.751437590932, 0.605503172737, 0.514674450957, 0.454353482512, 0.407050308965],
  [0.74725392039, 0.996404342439, 0.849823426169, 0.685398513368, 0.55002013668, 0.454353482512, 0.389234902883, 0.342353999733],
  [0.615105596242, 0.830890433625, 0.731221236837, 0.608694761374, 0.495804539034, 0.407050308965, 0.342353999733, 0.295530605237]
];

fn calculate_plane_psnr_hvs<T: Pixel>(
  plane1: &Plane<T>, plane2: &Plane<T>, plane_idx: usize, bit_depth: usize,
) -> f64 {
  const STEP: usize = 7;
  let mut result = 0.0;
  let mut pixels = 0usize;
  let csf = match plane_idx {
    0 => &CSF_Y,
    1 => &CSF_CB420,
    2 => &CSF_CR420,
    _ => unreachable!(),
  };

  // In the PSNR-HVS-M paper[1] the authors describe the construction of
  // their masking table as "we have used the quantization table for the
  // color component Y of JPEG [6] that has been also obtained on the
  // basis of CSF. Note that the values in quantization table JPEG have
  // been normalized and then squared." Their CSF matrix (from PSNR-HVS)
  // was also constructed from the JPEG matrices. I can not find any obvious
  // scheme of normalizing to produce their table, but if I multiply their
  // CSF by 0.38857 and square the result I get their masking table.
  // I have no idea where this constant comes from, but deviating from it
  // too greatly hurts MOS agreement.
  //
  // [1] Nikolay Ponomarenko, Flavia Silvestri, Karen Egiazarian, Marco Carli,
  //     Jaakko Astola, Vladimir Lukin, "On between-coefficient contrast masking
  //     of DCT basis functions", CD-ROM Proceedings of the Third
  //     International Workshop on Video Processing and Quality Metrics for Consumer
  //     Electronics VPQM-07, Scottsdale, Arizona, USA, 25-26 January, 2007, 4 p.
  const CSF_MULTIPLIER: f64 = 0.3885746225901003;
  let mut mask = [[0.0; 8]; 8];
  for x in 0..8 {
    for y in 0..8 {
      mask[x][y] = (csf[x][y] * CSF_MULTIPLIER).powi(2);
    }
  }

  let height = plane1.cfg.height;
  let width = plane1.cfg.width;
  let mut p1 = [0i16; 8 * 8];
  let mut p2 = [0i16; 8 * 8];
  let mut dct_p1 = [0i32; 8 * 8];
  let mut dct_p2 = [0i32; 8 * 8];
  for y in (0..(height - STEP)).step_by(STEP) {
    for x in (0..(width - STEP)).step_by(STEP) {
      let mut p1_means = [0.0; 4];
      let mut p2_means = [0.0; 4];
      let mut p1_vars = [0.0; 4];
      let mut p2_vars = [0.0; 4];
      let mut p1_gmean = 0.0;
      let mut p2_gmean = 0.0;
      let mut p1_gvar = 0.0;
      let mut p2_gvar = 0.0;
      let mut p1_mask = 0.0;
      let mut p2_mask = 0.0;

      for i in 0..8 {
        for j in 0..8 {
          p1[i * 8 + j] = i16::cast_from(plane1.p(x + j, y + i));
          p2[i * 8 + j] = i16::cast_from(plane2.p(x + j, y + i));

          let sub = ((i & 12) >> 2) + ((j & 12) >> 1);
          p1_gmean += p1[i * 8 + j] as f64;
          p2_gmean += p2[i * 8 + j] as f64;
          p1_means[sub] += p1[i * 8 + j] as f64;
          p2_means[sub] += p2[i * 8 + j] as f64;
        }
      }
      p1_gmean /= 64.0;
      p2_gmean /= 64.0;
      for i in 0..4 {
        p1_means[i] /= 16.0;
        p2_means[i] /= 16.0;
      }

      for i in 0..8 {
        for j in 0..8 {
          let sub = ((i & 12) >> 2) + ((j & 12) >> 1);
          p1_gvar += (p1[i * 8 + j] as f64 - p1_gmean)
            * (p1[i * 8 + j] as f64 - p1_gmean);
          p2_gvar += (p2[i * 8 + j] as f64 - p2_gmean)
            * (p2[i * 8 + j] as f64 - p2_gmean);
          p1_vars[sub] += (p1[i * 8 + j] as f64 - p1_means[sub])
            * (p1[i * 8 + j] as f64 - p1_means[sub]);
          p2_vars[sub] += (p2[i * 8 + j] as f64 - p2_means[sub])
            * (p2[i * 8 + j] as f64 - p2_means[sub]);
        }
      }
      p1_gvar *= 64.0 / 63.0;
      p2_gvar *= 64.0 / 63.0;
      for i in 0..4 {
        p1_vars[i] *= 16.0 / 15.0;
        p2_vars[i] *= 16.0 / 15.0;
      }
      if p1_gvar > 0.0 {
        p1_gvar = p1_vars.iter().sum::<f64>() / p1_gvar;
      }
      if p2_gvar > 0.0 {
        p2_gvar = p2_vars.iter().sum::<f64>() / p2_gvar;
      }

      p1.iter().copied().enumerate().for_each(|(i, v)| {
        dct_p1[i] = v as i32;
      });
      p2.iter().copied().enumerate().for_each(|(i, v)| {
        dct_p2[i] = v as i32;
      });
      od_bin_fdct8x8(&mut dct_p1, 8);
      od_bin_fdct8x8(&mut dct_p2, 8);
      for i in 0..8 {
        for j in (i == 0) as usize..8 {
          p1_mask += dct_p1[i * 8 + j].pow(2) as f64 * mask[i][j];
          p2_mask += dct_p2[i * 8 + j].pow(2) as f64 * mask[i][j];
        }
      }
      p1_mask = (p1_mask * p1_gvar).sqrt() / 32.0;
      p2_mask = (p2_mask * p2_gvar).sqrt() / 32.0;
      if p2_mask > p1_mask {
        p1_mask = p2_mask;
      }
      for i in 0..8 {
        for j in 0..8 {
          let mut err = (dct_p1[i * 8 + j] - dct_p2[i * 8 + j]).abs() as f64;
          if i != 0 || j != 0 {
            let err_mask = p1_mask / mask[i][j];
            err = if err < err_mask { 0.0 } else { err - err_mask };
          }
          result += (err * csf[i][j]).powi(2);
          pixels += 1;
        }
      }
    }
  }

  result /= pixels as f64;
  let sample_max: usize = (1 << bit_depth) - 1;
  result /= sample_max.pow(2) as f64;
  result
}

fn log10_convert(score: f64, weight: f64) -> f64 {
  10.0 * (-1.0 * (weight * score).log10())
}

// Based on daala's version. It is different from the 8x8 DCT we use during encoding.
fn od_bin_fdct8x8(data: &mut [i32], stride: usize) {
  let mut z = [0; 64];
  for i in 0..8 {
    od_bin_fdct8(&mut z[(8 * i)..], &data[i..], stride);
  }
  for i in 0..8 {
    od_bin_fdct8(&mut data[(stride * i)..], &z[i..], stride);
  }
}

fn od_bin_fdct8(y: &mut [i32], x: &[i32], x_stride: usize) {
  assert!(y.len() >= 8);
  assert!(x.len() >= 7 * x_stride);
  let mut t = [0; 8];
  let mut th = [0; 8];
  // Initial permutation
  t[0] = x[0];
  t[4] = x[x_stride];
  t[2] = x[2 * x_stride];
  t[6] = x[3 * x_stride];
  t[7] = x[4 * x_stride];
  t[3] = x[5 * x_stride];
  t[5] = x[6 * x_stride];
  t[1] = x[7 * x_stride];
  // +1/-1 butterflies
  t[1] = t[0] - t[1];
  th[1] = od_dct_rshift(t[1], 1);
  t[0] -= th[1];
  t[4] += t[5];
  th[4] = od_dct_rshift(t[4], 1);
  t[5] -= th[4];
  t[3] = t[2] - t[3];
  t[2] -= od_dct_rshift(t[3], 1);
  t[6] += t[7];
  th[6] = od_dct_rshift(t[6], 1);
  t[7] = th[6] - t[7];
  // + Embedded 4-point type-II DCT
  t[0] += th[6];
  t[6] = t[0] - t[6];
  t[2] = th[4] - t[2];
  t[4] = t[2] - t[4];
  // |-+ Embedded 2-point type-II DCT
  t[0] -= (t[4] * 13573 + 16384) >> 15;
  t[4] += (t[0] * 11585 + 8192) >> 14;
  t[0] -= (t[4] * 13573 + 16384) >> 15;
  // |-+ Embedded 2-point type-IV DST
  t[6] -= (t[2] * 21895 + 16384) >> 15;
  t[2] += (t[6] * 15137 + 8192) >> 14;
  t[6] -= (t[2] * 21895 + 16384) >> 15;
  // + Embedded 4-point type-IV DST
  t[3] += (t[5] * 19195 + 16384) >> 15;
  t[5] += (t[3] * 11585 + 8192) >> 14;
  t[3] -= (t[5] * 7489 + 4096) >> 13;
  t[7] = od_dct_rshift(t[5], 1) - t[7];
  t[5] -= t[7];
  t[3] = th[1] - t[3];
  t[1] -= t[3];
  t[7] += (t[1] * 3227 + 16384) >> 15;
  t[1] -= (t[7] * 6393 + 16384) >> 15;
  t[7] += (t[1] * 3227 + 16384) >> 15;
  t[5] += (t[3] * 2485 + 4096) >> 13;
  t[3] -= (t[5] * 18205 + 16384) >> 15;
  t[5] += (t[3] * 2485 + 4096) >> 13;
  y[0] = t[0];
  y[1] = t[1];
  y[2] = t[2];
  y[3] = t[3];
  y[4] = t[4];
  y[5] = t[5];
  y[6] = t[6];
  y[7] = t[7];
}

/// This is the strength reduced version of `a / (1 << b)`.
/// This will not work for `b == 0`, however currently this is only used for
/// `b == 1` anyway.
#[inline(always)]
fn od_dct_rshift(a: i32, b: u32) -> i32 {
  debug_assert!(b > 0);
  debug_assert!(b <= 32);

  ((a as u32 >> (32 - b)) as i32 + a) >> b
}
