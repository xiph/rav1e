use context::BlockOffset;
use context::CFLParams;
use context::ContextWriter;
use ec::Writer;
use ec::OD_BITRES;
use encoder::block::encode_block_a;
use encoder::block::encode_block_b;
use encoder::frame::FrameInvariants;
use encoder::frame::FrameState;
use encoder::sequence::Sequence;
use partition::BlockSize;
use partition::MotionVector;
use partition::PartitionType;
use partition::PredictionMode;
use partition::TxSize;
use partition::TxType;
use partition::INTRA_FRAME;
use partition::NONE_FRAME;
use partition::REF_FRAMES;
use rdo::get_lambda;
use rdo::rdo_mode_decision;
use rdo::rdo_partition_decision;
use rdo::rdo_tx_size_type;
use rdo::RDOOutput;
use rdo::RDOPartitionOutput;

pub fn encode_partition_bottomup(
  seq: &Sequence, fi: &FrameInvariants, fs: &mut FrameState,
  cw: &mut ContextWriter, w_pre_cdef: &mut dyn Writer,
  w_post_cdef: &mut dyn Writer, bsize: BlockSize, bo: &BlockOffset,
  pmvs: &[[Option<MotionVector>; REF_FRAMES]; 5]
) -> f64 {
  let mut rd_cost = std::f64::MAX;

  if bo.x >= cw.bc.cols || bo.y >= cw.bc.rows {
    return rd_cost;
  }

  let bs = bsize.width_mi();

  // Always split if the current partition is too large
  let must_split = bo.x + bs as usize > fi.w_in_b
    || bo.y + bs as usize > fi.h_in_b
    || bsize > BlockSize::BLOCK_64X64;

  // must_split overrides the minimum partition size when applicable
  let can_split = bsize > fi.min_partition_size || must_split;

  let mut partition = PartitionType::PARTITION_NONE;
  let mut best_decision = RDOPartitionOutput {
    rd_cost,
    bo: bo.clone(),
    pred_mode_luma: PredictionMode::DC_PRED,
    pred_mode_chroma: PredictionMode::DC_PRED,
    pred_cfl_params: CFLParams::new(),
    ref_frames: [INTRA_FRAME, NONE_FRAME],
    mvs: [MotionVector { row: 0, col: 0 }; 2],
    skip: false,
    tx_size: TxSize::TX_4X4,
    tx_type: TxType::DCT_DCT
  }; // Best decision that is not PARTITION_SPLIT

  let hbs = bs >> 1; // Half the block size in blocks
  let mut subsize: BlockSize;

  let cw_checkpoint = cw.checkpoint();
  let w_pre_checkpoint = w_pre_cdef.checkpoint();
  let w_post_checkpoint = w_post_cdef.checkpoint();

  // Code the whole block
  if !must_split {
    partition = PartitionType::PARTITION_NONE;

    let mut cost: f64 = 0.0;

    if bsize >= BlockSize::BLOCK_8X8 {
      let w: &mut dyn Writer =
        if cw.bc.cdef_coded { w_post_cdef } else { w_pre_cdef };
      let tell = w.tell_frac();
      cw.write_partition(w, bo, partition, bsize);
      cost = (w.tell_frac() - tell) as f64 * get_lambda(fi, seq.bit_depth)
        / ((1 << OD_BITRES) as f64);
    }

    let pmv_idx = if bsize > BlockSize::BLOCK_32X32 {
      0
    } else {
      ((bo.x & 32) >> 5) + ((bo.y & 32) >> 4) + 1
    };
    let spmvs = &pmvs[pmv_idx];

    let mode_decision =
      rdo_mode_decision(seq, fi, fs, cw, bsize, bo, spmvs, false).part_modes
        [0]
        .clone();
    let (mode_luma, mode_chroma) =
      (mode_decision.pred_mode_luma, mode_decision.pred_mode_chroma);
    let cfl = mode_decision.pred_cfl_params;
    {
      let ref_frames = mode_decision.ref_frames;
      let mvs = mode_decision.mvs;
      let skip = mode_decision.skip;
      let mut cdef_coded = cw.bc.cdef_coded;
      let (tx_size, tx_type) = (mode_decision.tx_size, mode_decision.tx_type);

      debug_assert!(
        (tx_size, tx_type)
          == rdo_tx_size_type(
            seq, fi, fs, cw, bsize, bo, mode_luma, ref_frames, mvs, skip
          )
      );
      cw.bc.set_tx_size(bo, tx_size);

      rd_cost = mode_decision.rd_cost + cost;

      let mut mv_stack = Vec::new();
      let is_compound = ref_frames[1] != NONE_FRAME;
      let mode_context = cw.find_mvrefs(
        bo,
        ref_frames,
        &mut mv_stack,
        bsize,
        false,
        fi,
        is_compound
      );

      cdef_coded = encode_block_a(
        seq,
        fs,
        cw,
        if cdef_coded { w_post_cdef } else { w_pre_cdef },
        bsize,
        bo,
        skip
      );
      encode_block_b(
        seq,
        fi,
        fs,
        cw,
        if cdef_coded { w_post_cdef } else { w_pre_cdef },
        mode_luma,
        mode_chroma,
        ref_frames,
        mvs,
        bsize,
        bo,
        skip,
        seq.bit_depth,
        cfl,
        tx_size,
        tx_type,
        mode_context,
        &mv_stack,
        false
      );
    }
    best_decision = mode_decision;
  }

  // Code a split partition and compare RD costs
  if can_split {
    cw.rollback(&cw_checkpoint);
    w_pre_cdef.rollback(&w_pre_checkpoint);
    w_post_cdef.rollback(&w_post_checkpoint);

    partition = PartitionType::PARTITION_SPLIT;
    subsize = bsize.subsize(partition);

    let nosplit_rd_cost = rd_cost;

    rd_cost = 0.0;

    if bsize >= BlockSize::BLOCK_8X8 {
      let w: &mut dyn Writer =
        if cw.bc.cdef_coded { w_post_cdef } else { w_pre_cdef };
      let tell = w.tell_frac();
      cw.write_partition(w, bo, partition, bsize);
      rd_cost = (w.tell_frac() - tell) as f64 * get_lambda(fi, seq.bit_depth)
        / ((1 << OD_BITRES) as f64);
    }

    let partitions = [
      bo,
      &BlockOffset { x: bo.x + hbs as usize, y: bo.y },
      &BlockOffset { x: bo.x, y: bo.y + hbs as usize },
      &BlockOffset { x: bo.x + hbs as usize, y: bo.y + hbs as usize }
    ];
    rd_cost += partitions
      .iter()
      .map(|&offset| {
        encode_partition_bottomup(
          seq,
          fi,
          fs,
          cw,
          w_pre_cdef,
          w_post_cdef,
          subsize,
          offset,
          pmvs //&best_decision.mvs[0]
        )
      })
      .sum::<f64>();

    // Recode the full block if it is more efficient
    if !must_split && nosplit_rd_cost < rd_cost {
      rd_cost = nosplit_rd_cost;

      cw.rollback(&cw_checkpoint);
      w_pre_cdef.rollback(&w_pre_checkpoint);
      w_post_cdef.rollback(&w_post_checkpoint);

      partition = PartitionType::PARTITION_NONE;

      if bsize >= BlockSize::BLOCK_8X8 {
        let w: &mut dyn Writer =
          if cw.bc.cdef_coded { w_post_cdef } else { w_pre_cdef };
        cw.write_partition(w, bo, partition, bsize);
      }

      // FIXME: redundant block re-encode
      let (mode_luma, mode_chroma) =
        (best_decision.pred_mode_luma, best_decision.pred_mode_chroma);
      let cfl = best_decision.pred_cfl_params;
      let ref_frames = best_decision.ref_frames;
      let mvs = best_decision.mvs;
      let skip = best_decision.skip;
      let mut cdef_coded = cw.bc.cdef_coded;
      let (tx_size, tx_type) = (best_decision.tx_size, best_decision.tx_type);

      debug_assert!(
        (tx_size, tx_type)
          == rdo_tx_size_type(
            seq, fi, fs, cw, bsize, bo, mode_luma, ref_frames, mvs, skip
          )
      );
      cw.bc.set_tx_size(bo, tx_size);

      let mut mv_stack = Vec::new();
      let is_compound = ref_frames[1] != NONE_FRAME;
      let mode_context = cw.find_mvrefs(
        bo,
        ref_frames,
        &mut mv_stack,
        bsize,
        false,
        fi,
        is_compound
      );

      cdef_coded = encode_block_a(
        seq,
        fs,
        cw,
        if cdef_coded { w_post_cdef } else { w_pre_cdef },
        bsize,
        bo,
        skip
      );
      encode_block_b(
        seq,
        fi,
        fs,
        cw,
        if cdef_coded { w_post_cdef } else { w_pre_cdef },
        mode_luma,
        mode_chroma,
        ref_frames,
        mvs,
        bsize,
        bo,
        skip,
        seq.bit_depth,
        cfl,
        tx_size,
        tx_type,
        mode_context,
        &mv_stack,
        false
      );
    }
  }

  subsize = bsize.subsize(partition);

  if bsize >= BlockSize::BLOCK_8X8
    && (bsize == BlockSize::BLOCK_8X8
      || partition != PartitionType::PARTITION_SPLIT)
  {
    cw.bc.update_partition_context(bo, subsize, bsize);
  }

  rd_cost
}

pub fn encode_partition_topdown(
  seq: &Sequence, fi: &FrameInvariants, fs: &mut FrameState,
  cw: &mut ContextWriter, w_pre_cdef: &mut dyn Writer,
  w_post_cdef: &mut dyn Writer, bsize: BlockSize, bo: &BlockOffset,
  block_output: &Option<RDOOutput>,
  pmvs: &[[Option<MotionVector>; REF_FRAMES]; 5]
) {
  if bo.x >= cw.bc.cols || bo.y >= cw.bc.rows {
    return;
  }

  let bs = bsize.width_mi();

  // Always split if the current partition is too large
  let must_split = bo.x + bs as usize > fi.w_in_b
    || bo.y + bs as usize > fi.h_in_b
    || bsize > BlockSize::BLOCK_64X64;

  let mut rdo_output = block_output.clone().unwrap_or(RDOOutput {
    part_type: PartitionType::PARTITION_INVALID,
    rd_cost: std::f64::MAX,
    part_modes: Vec::new()
  });
  let partition: PartitionType;

  if must_split {
    // Oversized blocks are split automatically
    partition = PartitionType::PARTITION_SPLIT;
  } else if bsize > fi.min_partition_size {
    // Blocks of sizes within the supported range are subjected to a partitioning decision
    rdo_output = rdo_partition_decision(
      seq,
      fi,
      fs,
      cw,
      w_pre_cdef,
      w_post_cdef,
      bsize,
      bo,
      &rdo_output,
      pmvs
    );
    partition = rdo_output.part_type;
  } else {
    // Blocks of sizes below the supported range are encoded directly
    partition = PartitionType::PARTITION_NONE;
  }

  assert!(bsize.width_mi() == bsize.height_mi());
  assert!(
    PartitionType::PARTITION_NONE <= partition
      && partition < PartitionType::PARTITION_INVALID
  );

  let hbs = bs >> 1; // Half the block size in blocks
  let subsize = bsize.subsize(partition);

  if bsize >= BlockSize::BLOCK_8X8 {
    let w: &mut dyn Writer =
      if cw.bc.cdef_coded { w_post_cdef } else { w_pre_cdef };
    cw.write_partition(w, bo, partition, bsize);
  }

  match partition {
    PartitionType::PARTITION_NONE => {
      let part_decision = if !rdo_output.part_modes.is_empty() {
        // The optimal prediction mode is known from a previous iteration
        rdo_output.part_modes[0].clone()
      } else {
        let pmv_idx = if bsize > BlockSize::BLOCK_32X32 {
          0
        } else {
          ((bo.x & 32) >> 5) + ((bo.y & 32) >> 4) + 1
        };
        let spmvs = &pmvs[pmv_idx];

        // Make a prediction mode decision for blocks encoded with no rdo_partition_decision call (e.g. edges)
        rdo_mode_decision(seq, fi, fs, cw, bsize, bo, spmvs, false).part_modes
          [0]
          .clone()
      };

      let mut mode_luma = part_decision.pred_mode_luma;
      let mut mode_chroma = part_decision.pred_mode_chroma;

      let cfl = part_decision.pred_cfl_params;
      let skip = part_decision.skip;
      let ref_frames = part_decision.ref_frames;
      let mvs = part_decision.mvs;
      let mut cdef_coded = cw.bc.cdef_coded;

      // NOTE: Cannot avoid calling rdo_tx_size_type() here again,
      // because, with top-down partition RDO, the neighnoring contexts
      // of current partition can change, i.e. neighboring partitions can split down more.
      let (tx_size, tx_type) = rdo_tx_size_type(
        seq, fi, fs, cw, bsize, bo, mode_luma, ref_frames, mvs, skip,
      );

      let mut mv_stack = Vec::new();
      let is_compound = ref_frames[1] != NONE_FRAME;
      let mode_context = cw.find_mvrefs(
        bo,
        ref_frames,
        &mut mv_stack,
        bsize,
        false,
        fi,
        is_compound
      );

      // TODO proper remap when is_compound is true
      if !mode_luma.is_intra() {
        if is_compound && mode_luma != PredictionMode::GLOBAL_GLOBALMV {
          let match0 = mv_stack[0].this_mv.row == mvs[0].row
            && mv_stack[0].this_mv.col == mvs[0].col;
          let match1 = mv_stack[0].comp_mv.row == mvs[1].row
            && mv_stack[0].comp_mv.col == mvs[1].col;

          mode_luma = if match0 && match1 {
            PredictionMode::NEAREST_NEARESTMV
          } else if match0 {
            PredictionMode::NEAREST_NEWMV
          } else if match1 {
            PredictionMode::NEW_NEARESTMV
          } else {
            PredictionMode::NEW_NEWMV
          };
          if mode_luma != PredictionMode::NEAREST_NEARESTMV
            && mvs[0].row == 0
            && mvs[0].col == 0
            && mvs[1].row == 0
            && mvs[1].col == 0
          {
            mode_luma = PredictionMode::GLOBAL_GLOBALMV;
          }
          mode_chroma = mode_luma;
        } else if !is_compound && mode_luma != PredictionMode::GLOBALMV {
          mode_luma = PredictionMode::NEWMV;
          for (c, m) in mv_stack.iter().take(4).zip(
            [
              PredictionMode::NEARESTMV,
              PredictionMode::NEAR0MV,
              PredictionMode::NEAR1MV,
              PredictionMode::NEAR2MV
            ]
            .iter()
          ) {
            if c.this_mv.row == mvs[0].row && c.this_mv.col == mvs[0].col {
              mode_luma = *m;
            }
          }
          if mode_luma == PredictionMode::NEWMV
            && mvs[0].row == 0
            && mvs[0].col == 0
          {
            mode_luma = if mv_stack.len() == 0 {
              PredictionMode::NEARESTMV
            } else if mv_stack.len() == 1 {
              PredictionMode::NEAR0MV
            } else {
              PredictionMode::GLOBALMV
            };
          }
          mode_chroma = mode_luma;
        }
      }

      // FIXME: every final block that has gone through the RDO decision process is encoded twice
      cdef_coded = encode_block_a(
        seq,
        fs,
        cw,
        if cdef_coded { w_post_cdef } else { w_pre_cdef },
        bsize,
        bo,
        skip
      );
      encode_block_b(
        seq,
        fi,
        fs,
        cw,
        if cdef_coded { w_post_cdef } else { w_pre_cdef },
        mode_luma,
        mode_chroma,
        ref_frames,
        mvs,
        bsize,
        bo,
        skip,
        seq.bit_depth,
        cfl,
        tx_size,
        tx_type,
        mode_context,
        &mv_stack,
        false
      );
    }
    PartitionType::PARTITION_SPLIT => {
      if rdo_output.part_modes.len() >= 4 {
        // The optimal prediction modes for each split block is known from an rdo_partition_decision() call
        assert!(subsize != BlockSize::BLOCK_INVALID);

        for mode in rdo_output.part_modes {
          let offset = mode.bo.clone();

          // Each block is subjected to a new splitting decision
          encode_partition_topdown(
            seq,
            fi,
            fs,
            cw,
            w_pre_cdef,
            w_post_cdef,
            subsize,
            &offset,
            &Some(RDOOutput {
              rd_cost: mode.rd_cost,
              part_type: PartitionType::PARTITION_NONE,
              part_modes: vec![mode]
            }),
            pmvs
          );
        }
      } else {
        let partitions = [
          bo,
          &BlockOffset { x: bo.x + hbs as usize, y: bo.y },
          &BlockOffset { x: bo.x, y: bo.y + hbs as usize },
          &BlockOffset { x: bo.x + hbs as usize, y: bo.y + hbs as usize }
        ];
        partitions.iter().for_each(|&offset| {
          encode_partition_topdown(
            seq,
            fi,
            fs,
            cw,
            w_pre_cdef,
            w_post_cdef,
            subsize,
            offset,
            &None,
            pmvs
          );
        });
      }
    }
    _ => {
      assert!(false);
    }
  }

  if bsize >= BlockSize::BLOCK_8X8
    && (bsize == BlockSize::BLOCK_8X8
      || partition != PartitionType::PARTITION_SPLIT)
  {
    cw.bc.update_partition_context(bo, subsize, bsize);
  }
}
