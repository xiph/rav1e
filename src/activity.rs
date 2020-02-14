// Copyright (c) 2017-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::frame::*;
use crate::hawktracer::*;
use crate::tiling::*;
use crate::util::*;
use crate::transform::*;
use crate::SegmentationState;
use crate::FrameInvariants;
use TxSize::*;

#[derive(Debug, Default, Clone)]
pub struct ActivityMask {
  variances: Vec<f64>,
  width: usize,
  granularity: usize,

  pub tot_bins: usize,
  pub seg_bins: [usize; 8],
  pub avg_var: f64,
  pub var_scale: f64,
}

impl ActivityMask {
  #[hawktracer(activity_mask_from_plane)]
  pub fn from_plane<T: Pixel>(fi: &FrameInvariants<T>, luma_plane: &Plane<T>) -> ActivityMask {
    let granularity: usize = 3; /* Granularity of the map */
    let act_granularity: usize = 4; /* Granularity of the activity data, if != granularity it repeats */
    let tot_pix: usize = (1 << granularity) * (1 << granularity);

    /* Aligned width and height to the activity data */
    let width = (((luma_plane.cfg.width >> act_granularity) << act_granularity)) + (1 << act_granularity);
    let height = (((luma_plane.cfg.height >> act_granularity) << act_granularity)) + (1 << act_granularity);

    let mut variances = Vec::with_capacity((width >> granularity) * (height >> granularity));
    variances.resize((width >> granularity) * (height >> granularity), 0f64);

    let aligned_luma = Rect {
      x: 0_isize,
      y: 0_isize,
      width: width,
      height: height,
    };
    let luma = PlaneRegion::new(luma_plane, aligned_luma);

    let mut freq_storage: Aligned<[T::Coeff; 64 * 64]> = Aligned::uninitialized();
    let mut src_storage: Aligned<[i16; 64 * 64]> = Aligned::uninitialized();

    let tx_type = TxType::DCT_DCT;
    let tx_size = match act_granularity {
            2 => TX_4X4,
            3 => TX_8X8,
            4 => TX_16X16,
            5 => TX_32X32,
            6 => TX_64X64,
            _ => unreachable!(),
        };

    for y in 0..height >> act_granularity {
      for x in 0..width >> act_granularity {
        let block_rect = Area::Rect {
          x: (x << act_granularity) as isize,
          y: (y << act_granularity) as isize,
          width: 1 << act_granularity,
          height: 1 << act_granularity,
        };

        let block = luma.subregion(block_rect);

        let src = &mut src_storage.data[..tx_size.area()];
        let freq = &mut freq_storage.data[..tx_size.area()];

        for y in 0..(1 << act_granularity) {
            let l = &block[y];
            for x in 0..(1 << act_granularity) {
                src[y * (1 << act_granularity) + x] = l[x].as_();
            }
        }

        forward_transform(src, freq, tx_size.width(), tx_size, tx_type,
                          fi.sequence.bit_depth, fi.cpu_feature_level);

        let mut sum_f = 0f64;

        for y in 0..(1 << act_granularity) {
            for x in 0..(1 << act_granularity) {
                if (x < 4) && (y < 4) { continue };
                let coeff = freq[y*(1 << act_granularity) + x];
                sum_f += i32::cast_from(coeff).abs() as f64;
            }
        }

        sum_f /= (tot_pix - 4*4) as f64;

        /* Copy down to granularity */
        for i in 0..(1 << (act_granularity - granularity)) {
            for j in 0..(1 << (act_granularity - granularity)) {
                let loc = variances.get_mut((((x << act_granularity) >> granularity) + j) + (width >> granularity) * (((y << act_granularity) >> granularity) + i));
                match loc {
                    Some(val) => *val = sum_f,
                    None => unreachable!(),
                }
            }
        }

      }
    }

    let mut avg_var = 0f64;
    let mut max = 0f64;

    /* Merge temporal activity */
    for y in 0..fi.h_in_imp_b {
        for x in 0..fi.w_in_imp_b {

            let propagate_cost = fi.block_importances[y * fi.w_in_imp_b + x] as f64;
            let intra_cost = fi.lookahead_intra_costs[y * fi.w_in_imp_b + x] as f64;

            let temporal_act =
                if intra_cost == 0. {
                    1.0f64
                } else {
                    let strength = 1.0; // empirical, see comment above
                    let frac = (intra_cost + propagate_cost) / intra_cost;
                    frac.powf(strength / 3.0) * 1.0f64
                };

            let element = variances.get_mut(y * (width >> granularity) + x);
            match element {
                Some(x) => {
                    *x = *x + temporal_act;
                    avg_var += *x;
                    max = max.max(*x);
                }
                None => unreachable!(),
            }
        }
    }

//    println!("Avg var = {}", avg_var);

    let mut seg_bins = [0usize; 8];
    for i in 0..variances.len() {
        let element = variances.get_mut(i);
        match element {
            Some(x) => {
                *x = (*x / max) * 7f64;
                seg_bins[(*x).round() as usize] += 1;
            }
            None => unreachable!(),
        }
    }

    let tot_bins = variances.len();

    avg_var /= tot_bins as f64;
    avg_var /= max;
    avg_var *= 7f64;

    let var_scale = max / 7f64;

    ActivityMask { variances, width, granularity, tot_bins, seg_bins, avg_var, var_scale }
  }

  pub fn segid_at(&self, segmentation: &SegmentationState, x: usize, y: usize) -> u8 {
    let dec_width = self.width >> self.granularity;
    let res = self.variances.get((x >> self.granularity) + dec_width * (y >> self.granularity));
    match res {
        Some(val) => return segmentation.act_lut[(*val).round() as usize] as u8,
        None => unreachable!(),
    }
  }

  pub fn variance_at(&self, x: usize, y: usize) -> f64 {
    let dec_width = self.width >> self.granularity;
    let res = self.variances.get((x >> self.granularity) + dec_width * (y >> self.granularity));
    match res {
        /* Tuned to 16x16 varience from 3976852114 samples of a variety of pictures */
        Some(val) => return (*val * self.var_scale) * 60f64,
        None => unreachable!(),
    }
  }
}
