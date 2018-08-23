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

      let sorg = fs.input.planes[0].slice(&po);
      let slice_org = sorg.as_slice();
      let stride_org = fs.input.planes[0].cfg.stride;
      let stride_ref = rec.planes[0].cfg.stride;

      let mut lowest_sad = 128*128*4096 as usize;
      let mut best_mv = MotionVector { row: 0, col: 0 };

      for y in (y_lo..y_hi).step_by(2) {
        for x in (x_lo..x_hi).step_by(2) {

          let mut sad = 0;
          let sref = rec.planes[0].slice(&PlaneOffset { x: x_lo, y: y_lo });
          let slice_ref = sref.as_slice();

          for r in 0..blk_h {
            for c in 0..blk_w {
              let org_index = r * stride_org + c;
              let ref_index = r * stride_ref + c;
              let a = slice_org[org_index];
              let b = slice_ref[ref_index];
              let delta = b as isize - a as isize;
              sad += delta.abs() as usize;
            }
          }

          if sad < lowest_sad {
            lowest_sad = sad;
            best_mv = MotionVector { row: 8*(y as i16 - bo.y as i16), col: 8*(x as i16 - bo.x as i16) }
          }

        }
      }
      best_mv
    }

    None => MotionVector { row: 0, col : 0 }
  }
}
