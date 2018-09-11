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

pub fn motion_estimation(
  fi: &FrameInvariants, fs: &mut FrameState, bsize: BlockSize,
  bo: &BlockOffset, ref_frame: usize
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
      let cols = (rec.frame.planes[0].cfg.width + MI_SIZE - 1) / MI_SIZE;
      let rows = (rec.frame.planes[0].cfg.height + MI_SIZE - 1) / MI_SIZE;
      let x_min = -(bo.x as isize) * (8 * MI_SIZE) as isize - border_w;
      let x_max = (cols - bo.x - blk_w / MI_SIZE) as isize * (8 * MI_SIZE) as isize + border_w;
      let y_min = -(bo.y as isize) * (8 * MI_SIZE) as isize - border_h;
      let y_max = (rows - bo.y - blk_h / MI_SIZE) as isize * (8 * MI_SIZE) as isize + border_h;
      let x_lo = po.x + ((-range).max(x_min / 8));
      let x_hi = po.x + (range.min(x_max / 8));
      let y_lo = po.y + ((-range).max(y_min / 8));
      let y_hi = po.y + (range.min(y_max / 8));

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

            if (cand_mv.col as isize) < x_min || (cand_mv.col as isize) > x_max {
              continue;
            }
            if (cand_mv.row as isize) < y_min || (cand_mv.row as isize) > y_max {
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
