use context::get_plane_block_size;
use context::has_chroma;
use context::BlockOffset;
use context::ContextWriter;
use encoder::frame::FrameInvariants;
use encoder::frame::FrameState;
use partition::BlockSize;
use partition::MotionVector;
use partition::PredictionMode;
use plane::PlaneConfig;
use plane::PlaneOffset;

pub fn motion_compensate(
  fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter,
  luma_mode: PredictionMode, ref_frames: [usize; 2], mvs: [MotionVector; 2],
  bsize: BlockSize, bo: &BlockOffset, bit_depth: usize, luma_only: bool
) {
  if luma_mode.is_intra() {
    return;
  }

  let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;

  // Inter mode prediction can take place once for a whole partition,
  // instead of each tx-block.
  let num_planes =
    1 + if !luma_only && has_chroma(bo, bsize, xdec, ydec) { 2 } else { 0 };

  for p in 0..num_planes {
    let plane_bsize =
      if p == 0 { bsize } else { get_plane_block_size(bsize, xdec, ydec) };

    let po = bo.plane_offset(&fs.input.planes[p].cfg);
    let rec = &mut fs.rec.planes[p];
    // TODO: make more generic to handle 2xN and Nx2 MC
    if p > 0 && bsize == BlockSize::BLOCK_4X4 {
      let some_use_intra = cw.bc.at(&bo.with_offset(-1, -1)).mode.is_intra()
        || cw.bc.at(&bo.with_offset(0, -1)).mode.is_intra()
        || cw.bc.at(&bo.with_offset(-1, 0)).mode.is_intra();

      if some_use_intra {
        luma_mode.predict_inter(
          fi,
          p,
          &po,
          &mut rec.mut_slice(&po),
          plane_bsize.width(),
          plane_bsize.height(),
          ref_frames,
          mvs,
          bit_depth
        );
      } else {
        assert!(xdec == 1 && ydec == 1);
        // TODO: these are only valid for 4:2:0
        let mv0 = cw.bc.at(&bo.with_offset(-1, -1)).mv;
        let rf0 = cw.bc.at(&bo.with_offset(-1, -1)).ref_frames;
        let mv1 = cw.bc.at(&bo.with_offset(0, -1)).mv;
        let rf1 = cw.bc.at(&bo.with_offset(0, -1)).ref_frames;
        let po1 = PlaneOffset { x: po.x + 2, y: po.y };
        let mv2 = cw.bc.at(&bo.with_offset(-1, 0)).mv;
        let rf2 = cw.bc.at(&bo.with_offset(-1, 0)).ref_frames;
        let po2 = PlaneOffset { x: po.x, y: po.y + 2 };
        let po3 = PlaneOffset { x: po.x + 2, y: po.y + 2 };
        luma_mode.predict_inter(
          fi,
          p,
          &po,
          &mut rec.mut_slice(&po),
          2,
          2,
          rf0,
          mv0,
          bit_depth
        );
        luma_mode.predict_inter(
          fi,
          p,
          &po1,
          &mut rec.mut_slice(&po1),
          2,
          2,
          rf1,
          mv1,
          bit_depth
        );
        luma_mode.predict_inter(
          fi,
          p,
          &po2,
          &mut rec.mut_slice(&po2),
          2,
          2,
          rf2,
          mv2,
          bit_depth
        );
        luma_mode.predict_inter(
          fi,
          p,
          &po3,
          &mut rec.mut_slice(&po3),
          2,
          2,
          ref_frames,
          mvs,
          bit_depth
        );
      }
    } else {
      luma_mode.predict_inter(
        fi,
        p,
        &po,
        &mut rec.mut_slice(&po),
        plane_bsize.width(),
        plane_bsize.height(),
        ref_frames,
        mvs,
        bit_depth
      );
    }
  }
}
