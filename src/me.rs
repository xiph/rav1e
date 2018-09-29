// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use context::BlockOffset;
use context::BLOCK_TO_PLANE_SHIFT;
use context::MI_SIZE;
use partition::*;
use plane::*;
use FrameInvariants;
use FrameState;

#[inline(always)]
pub fn get_sad(
  plane_org: &mut PlaneSlice, plane_ref: &mut PlaneSlice, blk_h: usize,
  blk_w: usize
) -> u32 {
  let mut sum = 0 as u32;

  for _r in 0..blk_h {
    {
      let slice_org = plane_org.as_slice_w_width(blk_w);
      let slice_ref = plane_ref.as_slice_w_width(blk_w);
      sum += slice_org
        .iter()
        .zip(slice_ref)
        .map(|(&a, &b)| (a as i32 - b as i32).abs() as u32)
        .sum::<u32>();
    }
    plane_org.y += 1;
    plane_ref.y += 1;
  }

  sum
}

#[inline(always)]
pub fn get_sad_iter(
  plane_org: &mut PlaneSlice, plane_ref: &mut PlaneSlice, blk_h: usize,
  blk_w: usize
) -> u32 {
  let mut sum = 0 as u32;

  let org_iter = plane_org.iter_width(blk_w);
  let ref_iter = plane_ref.iter_width(blk_w);

  for (slice_org, slice_ref) in org_iter.take(blk_h).zip(ref_iter) {
      sum += slice_org
        .iter()
        .zip(slice_ref)
        .map(|(&a, &b)| (a as i32 - b as i32).abs() as u32)
        .sum::<u32>();
  }

  sum
}
pub fn motion_estimation(
  fi: &FrameInvariants, fs: &FrameState, bsize: BlockSize,
  bo: &BlockOffset, ref_frame: usize, pmv: &MotionVector
) -> MotionVector {
  match fi.rec_buffer.frames[fi.ref_frames[ref_frame - LAST_FRAME]] {
    Some(ref rec) => {
      let po = PlaneOffset {
        x: (bo.x as isize) << BLOCK_TO_PLANE_SHIFT,
        y: (bo.y as isize) << BLOCK_TO_PLANE_SHIFT
      };
      let range = 32 as isize;
      let blk_w = bsize.width();
      let blk_h = bsize.height();
      let border_w = 128 + blk_w as isize * 8;
      let border_h = 128 + blk_h as isize * 8;
      let mvx_min = -(bo.x as isize) * (8 * MI_SIZE) as isize - border_w;
      let mvx_max = (fi.w_in_b - bo.x - blk_w / MI_SIZE) as isize * (8 * MI_SIZE) as isize + border_w;
      let mvy_min = -(bo.y as isize) * (8 * MI_SIZE) as isize - border_h;
      let mvy_max = (fi.h_in_b - bo.y - blk_h / MI_SIZE) as isize * (8 * MI_SIZE) as isize + border_h;
      let x_lo = po.x + ((-range + (pmv.col / 8) as isize).max(mvx_min / 8));
      let x_hi = po.x + ((range + (pmv.col / 8) as isize).min(mvx_max / 8));
      let y_lo = po.y + ((-range + (pmv.row / 8) as isize).max(mvy_min / 8));
      let y_hi = po.y + ((range + (pmv.row / 8) as isize).min(mvy_max / 8));

      let mut lowest_sad = 128 * 128 * 4096 as u32;
      let mut best_mv = MotionVector { row: 0, col: 0 };

      for y in (y_lo..y_hi).step_by(2) {
        for x in (x_lo..x_hi).step_by(2) {
          let mut plane_org = fs.input.planes[0].slice(&po);
          let mut plane_ref = rec.frame.planes[0].slice(&PlaneOffset { x, y });

          let sad = get_sad(&mut plane_org, &mut plane_ref, blk_h, blk_w);

          if sad < lowest_sad {
            lowest_sad = sad;
            best_mv = MotionVector {
              row: 8 * (y as i16 - po.y as i16),
              col: 8 * (x as i16 - po.x as i16)
            }
          }
        }
      }

      let mode = PredictionMode::NEWMV;
      let mut tmp_plane = Plane::new(blk_w, blk_h, 0, 0, 0, 0);

      let mut steps = vec![8, 4, 2];
      if fi.allow_high_precision_mv {
        steps.push(1);
      }

      for step in steps {
        let center_mv_h = best_mv;
        for i in 0..3 {
          for j in 0..3 {
            // Skip the center point that was already tested
            if i == 1 && j == 1 {
              continue;
            }

            let cand_mv = MotionVector {
              row: center_mv_h.row + step * (i as i16 - 1),
              col: center_mv_h.col + step * (j as i16 - 1)
            };

            if (cand_mv.col as isize) < mvx_min || (cand_mv.col as isize) > mvx_max {
              continue;
            }
            if (cand_mv.row as isize) < mvy_min || (cand_mv.row as isize) > mvy_max {
              continue;
            }

            {
              let tmp_slice =
                &mut tmp_plane.mut_slice(&PlaneOffset { x: 0, y: 0 });

              mode.predict_inter(
                fi, 0, &po, tmp_slice, blk_w, blk_h, ref_frame, &cand_mv, 8,
              );
            }

            let mut plane_org = fs.input.planes[0].slice(&po);
            let mut plane_ref = tmp_plane.slice(&PlaneOffset { x: 0, y: 0 });

            let sad = get_sad(&mut plane_org, &mut plane_ref, blk_h, blk_w);

            if sad < lowest_sad {
              lowest_sad = sad;
              best_mv = cand_mv;
            }
          }
        }
      }

      best_mv
    }

    None => MotionVector { row: 0, col: 0 }
  }
}

#[cfg(test)]
pub mod test {
  use super::*;

  // Generate plane data for get_sad_same()
  fn setup_sad() -> (Plane, Plane) {
    let mut input_plane = Plane::new(640, 480, 0, 0, 128 + 8, 128 + 8);
    let mut rec_plane = input_plane.clone();
    
    for (i, row) in input_plane.data.chunks_mut(input_plane.cfg.stride).enumerate() {
      for (j, mut pixel) in row.into_iter().enumerate() {
        *pixel = ((j + i) % 256) as u16;
      }
    }

    for (i, row) in rec_plane.data.chunks_mut(rec_plane.cfg.stride).enumerate() {
      for (j, mut pixel) in row.into_iter().enumerate() {
        *pixel = ((j as i32 - i as i32) % 256) as u16;
      }
    }

    (input_plane, rec_plane)
  }

  // Regression and validation test for SAD computation
  #[test]
  fn get_sad_same() {
    use partition::BlockSize;
    use partition::BlockSize::*;

    let blocks: Vec<(BlockSize, u32)> = vec![
      (BLOCK_4X4, 393592),
      (BLOCK_4X8, 395176),
      (BLOCK_8X4, 1440456),
      (BLOCK_8X8, 1835664),
      (BLOCK_8X16, 1842256),
      (BLOCK_16X8, 6022352),
      (BLOCK_16X16, 7864736),
      (BLOCK_16X32, 7893152),
      (BLOCK_32X16, 24605344),
      (BLOCK_32X32, 32499008),
      (BLOCK_32X64, 32629056),
      (BLOCK_64X32, 99412288),
      (BLOCK_64X64, 132043392),
      (BLOCK_64X128, 132621248),
      (BLOCK_128X64, 399430272),
      (BLOCK_128X128, 532067552),
      (BLOCK_4X16, 398344),
      (BLOCK_16X4, 3533800),
      (BLOCK_8X32, 1855440),
      (BLOCK_32X8, 14392656),
      (BLOCK_16X64, 7949984),
      (BLOCK_64X16, 58061984),
    ];

    let (input_plane, rec_plane) = setup_sad();

    for block in blocks {
        let bsw = block.0.width();
        let bsh = block.0.height();
        let po = PlaneOffset { x: 40, y: 40 };

        let mut input_slice = input_plane.slice(&po);
        let mut rec_slice = rec_plane.slice(&po);

        assert_eq!(block.1, get_sad(&mut input_slice, &mut rec_slice, bsw, bsh));
    }
  }
}
