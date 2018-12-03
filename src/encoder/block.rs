use context::has_chroma;
use context::BlockOffset;
use context::CFLParams;
use context::CandidateMV;
use context::ContextWriter;
use context::REF_CAT_LEVEL;
use ec::Writer;
use encoder::frame::FrameInvariants;
use encoder::frame::FrameState;
use encoder::frame::FrameType;
use encoder::motion_comp::motion_compensate;
use encoder::sequence::Sequence;
use encoder::transform::write_tx_blocks;
use encoder::transform::write_tx_tree;
use partition::BlockSize;
use partition::MotionVector;
use partition::MvSubpelPrecision;
use partition::PredictionMode;
use partition::TxSize;
use partition::TxType;
use plane::PlaneConfig;

pub fn encode_block_a(
  seq: &Sequence, fs: &FrameState, cw: &mut ContextWriter, w: &mut dyn Writer,
  bsize: BlockSize, bo: &BlockOffset, skip: bool
) -> bool {
  cw.bc.set_skip(bo, bsize, skip);
  if fs.segmentation.enabled
    && fs.segmentation.update_map
    && fs.segmentation.preskip
  {
    cw.write_segmentation(
      w,
      bo,
      bsize,
      false,
      fs.segmentation.last_active_segid
    );
  }
  cw.write_skip(w, bo, skip);
  if fs.segmentation.enabled
    && fs.segmentation.update_map
    && !fs.segmentation.preskip
  {
    cw.write_segmentation(
      w,
      bo,
      bsize,
      skip,
      fs.segmentation.last_active_segid
    );
  }
  if !skip && seq.enable_cdef {
    cw.bc.cdef_coded = true;
  }
  cw.bc.cdef_coded
}

pub fn encode_block_b(
  seq: &Sequence, fi: &FrameInvariants, fs: &mut FrameState,
  cw: &mut ContextWriter, w: &mut dyn Writer, luma_mode: PredictionMode,
  chroma_mode: PredictionMode, ref_frames: [usize; 2], mvs: [MotionVector; 2],
  bsize: BlockSize, bo: &BlockOffset, skip: bool, bit_depth: usize,
  cfl: CFLParams, tx_size: TxSize, tx_type: TxType, mode_context: usize,
  mv_stack: &[CandidateMV], for_rdo_use: bool
) -> i64 {
  let is_inter = !luma_mode.is_intra();
  if is_inter {
    assert!(luma_mode == chroma_mode);
  };
  let sb_size = if seq.use_128x128_superblock {
    BlockSize::BLOCK_128X128
  } else {
    BlockSize::BLOCK_64X64
  };
  let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;
  if skip {
    cw.bc.reset_skip_context(bo, bsize, xdec, ydec);
  }
  cw.bc.set_block_size(bo, bsize);
  cw.bc.set_mode(bo, bsize, luma_mode);
  cw.bc.set_ref_frames(bo, bsize, ref_frames);
  cw.bc.set_motion_vectors(bo, bsize, mvs);

  //write_q_deltas();
  if cw.bc.code_deltas
    && fs.deblock.block_deltas_enabled
    && (bsize < sb_size || !skip)
  {
    cw.write_block_deblock_deltas(w, bo, fs.deblock.block_delta_multi);
  }
  cw.bc.code_deltas = false;

  if fi.frame_type == FrameType::INTER {
    cw.write_is_inter(w, bo, is_inter);
    if is_inter {
      cw.fill_neighbours_ref_counts(bo);
      cw.write_ref_frames(w, fi, bo);

      // NOTE: Until rav1e supports other inter modes than GLOBALMV
      if luma_mode >= PredictionMode::NEAREST_NEARESTMV {
        cw.write_compound_mode(w, luma_mode, mode_context);
      } else {
        cw.write_inter_mode(w, luma_mode, mode_context);
      }

      let ref_mv_idx = 0;
      let num_mv_found = mv_stack.len();

      if luma_mode == PredictionMode::NEWMV
        || luma_mode == PredictionMode::NEW_NEWMV
      {
        if luma_mode == PredictionMode::NEW_NEWMV {
          assert!(num_mv_found >= 2);
        }
        for idx in 0..2 {
          if num_mv_found > idx + 1 {
            let drl_mode = ref_mv_idx > idx;
            let ctx: usize = (mv_stack[idx].weight < REF_CAT_LEVEL) as usize
              + (mv_stack[idx + 1].weight < REF_CAT_LEVEL) as usize;
            cw.write_drl_mode(w, drl_mode, ctx);
            if !drl_mode {
              break;
            }
          }
        }
      }

      let ref_mvs = if num_mv_found > 0 {
        [mv_stack[ref_mv_idx].this_mv, mv_stack[ref_mv_idx].comp_mv]
      } else {
        [MotionVector { row: 0, col: 0 }; 2]
      };

      let mv_precision = if fi.force_integer_mv != 0 {
        MvSubpelPrecision::MV_SUBPEL_NONE
      } else if fi.allow_high_precision_mv {
        MvSubpelPrecision::MV_SUBPEL_HIGH_PRECISION
      } else {
        MvSubpelPrecision::MV_SUBPEL_LOW_PRECISION
      };

      if luma_mode == PredictionMode::NEWMV
        || luma_mode == PredictionMode::NEW_NEWMV
        || luma_mode == PredictionMode::NEW_NEARESTMV
      {
        cw.write_mv(w, mvs[0], ref_mvs[0], mv_precision);
      }
      if luma_mode == PredictionMode::NEW_NEWMV
        || luma_mode == PredictionMode::NEAREST_NEWMV
      {
        cw.write_mv(w, mvs[1], ref_mvs[1], mv_precision);
      }

      if luma_mode >= PredictionMode::NEAR0MV
        && luma_mode <= PredictionMode::NEAR2MV
      {
        let ref_mv_idx =
          luma_mode as usize - PredictionMode::NEAR0MV as usize + 1;
        if luma_mode != PredictionMode::NEAR0MV {
          assert!(num_mv_found > ref_mv_idx);
        }

        for idx in 1..3 {
          if num_mv_found > idx + 1 {
            let drl_mode = ref_mv_idx > idx;
            let ctx: usize = (mv_stack[idx].weight < REF_CAT_LEVEL) as usize
              + (mv_stack[idx + 1].weight < REF_CAT_LEVEL) as usize;

            cw.write_drl_mode(w, drl_mode, ctx);
            if !drl_mode {
              break;
            }
          }
        }
        if mv_stack.len() > 1 {
          assert!(mv_stack[ref_mv_idx].this_mv.row == mvs[0].row);
          assert!(mv_stack[ref_mv_idx].this_mv.col == mvs[0].col);
        } else {
          assert!(0 == mvs[0].row);
          assert!(0 == mvs[0].col);
        }
      } else if luma_mode == PredictionMode::NEARESTMV {
        if mv_stack.len() > 0 {
          assert!(mv_stack[0].this_mv.row == mvs[0].row);
          assert!(mv_stack[0].this_mv.col == mvs[0].col);
        } else {
          assert!(0 == mvs[0].row);
          assert!(0 == mvs[0].col);
        }
      }
    } else {
      cw.write_intra_mode(w, bsize, luma_mode);
    }
  } else {
    cw.write_intra_mode_kf(w, bo, luma_mode);
  }

  if !is_inter {
    if luma_mode.is_directional() && bsize >= BlockSize::BLOCK_8X8 {
      cw.write_angle_delta(w, 0, luma_mode);
    }
    if has_chroma(bo, bsize, xdec, ydec) {
      cw.write_intra_uv_mode(w, chroma_mode, luma_mode, bsize);
      if chroma_mode.is_cfl() {
        assert!(bsize.cfl_allowed());
        cw.write_cfl_alphas(w, cfl);
      }
      if chroma_mode.is_directional() && bsize >= BlockSize::BLOCK_8X8 {
        cw.write_angle_delta(w, 0, chroma_mode);
      }
    }
    // TODO: Extra condition related to palette mode, see `read_filter_intra_mode_info` in decodemv.c
    if luma_mode == PredictionMode::DC_PRED
      && bsize.width() <= 32
      && bsize.height() <= 32
    {
      cw.write_use_filter_intra(w, false, bsize); // Always turn off FILTER_INTRA
    }
  }

  motion_compensate(
    fi, fs, cw, luma_mode, ref_frames, mvs, bsize, bo, bit_depth, false,
  );

  if is_inter {
    write_tx_tree(
      fi,
      fs,
      cw,
      w,
      luma_mode,
      bo,
      bsize,
      tx_size,
      tx_type,
      skip,
      bit_depth,
      false,
      for_rdo_use
    )
  } else {
    write_tx_blocks(
      fi,
      fs,
      cw,
      w,
      luma_mode,
      chroma_mode,
      bo,
      bsize,
      tx_size,
      tx_type,
      skip,
      bit_depth,
      cfl,
      false,
      for_rdo_use
    )
  }
}
