// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use std::cmp;
use FrameInvariants;
use FrameState;
use partition::BlockSize;
use context::BlockOffset;
use partition::MotionVector;
use partition::LAST_FRAME;
use plane::PlaneOffset;
use context::BLOCK_TO_PLANE_SHIFT;

pub fn motion_estimation(fi: &FrameInvariants, fs: &mut FrameState, bsize: BlockSize,
                         bo: &BlockOffset, ref_frame: usize) -> MotionVector {

  match fi.rec_buffer.frames[fi.ref_frames[ref_frame - LAST_FRAME]] {
    Some(ref rec) => {
      let po = PlaneOffset { x: bo.x << BLOCK_TO_PLANE_SHIFT, y: bo.y << BLOCK_TO_PLANE_SHIFT };
      let range = 16 as usize;
      let blk_w = bsize.width();
      let blk_h = bsize.height();
      let x_lo = cmp::max(0, po.x as isize - range as isize) as usize;
      let x_hi = cmp::min(fs.input.planes[0].cfg.width - blk_w, po.x + range);
      let y_lo = cmp::max(0, po.y as isize - range as isize) as usize;
      let y_hi = cmp::min(fs.input.planes[0].cfg.height - blk_h, po.y + range);

      let stride_org = fs.input.planes[0].cfg.stride;
      let stride_ref = rec.planes[0].cfg.stride;

      let mut lowest_sad = 128*128*4096 as u32;
      let mut best_mv = MotionVector { row: 0, col: 0 };

      for y in (y_lo..y_hi).step_by(2) {
        for x in (x_lo..x_hi).step_by(2) {
          let mut sad = 0 as u32;
          let mut plane_org = fs.input.planes[0].slice(&po);
          let mut plane_ref = rec.planes[0].slice(&PlaneOffset { x: x, y: y });

          for _r in 0..blk_h {
            {
              let slice_org = plane_org.as_slice_w_width(blk_w);
              let slice_ref = plane_ref.as_slice_w_width(blk_w);
              sad += slice_org.iter().zip(slice_ref).map(|(&a, &b)| (a as i32 - b as i32).abs() as u32).sum::<u32>();
            }
            plane_org.y += 1;
            plane_ref.y += 1;
          }

          if sad < lowest_sad {
            lowest_sad = sad;
            best_mv = MotionVector { row: 8*(y as i16 - po.y as i16), col: 8*(x as i16 - po.x as i16) }
          }

        }
      }
      best_mv
    },

    None => MotionVector { row: 0, col : 0 }
  }
}
