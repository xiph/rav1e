use context::av1_get_coded_tx_size;
use context::get_plane_block_size;
use context::has_chroma;
use context::uv_intra_mode_to_tx_type_context;
use context::BlockOffset;
use context::CFLParams;
use context::ContextWriter;
use ec::Writer;
use encoder::diff;
use encoder::frame::FrameInvariants;
use encoder::frame::FrameState;
use encoder::get_qidx;
use encoder::luma_ac;
use partition::BlockSize;
use partition::PredictionMode;
use partition::TxSize;
use partition::TxType;
use plane::PlaneConfig;
use plane::PlaneOffset;
use quantize::dequantize;
use quantize::get_log_tx_scale;
use transform::forward_transform;
use transform::inverse_transform_add;
use util::AlignedArray;
use util::UninitializedAlignedArray;

// For a transform block,
// predict, transform, quantize, write coefficients to a bitstream,
// dequantize, inverse-transform.
pub fn encode_tx_block(
  fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter,
  w: &mut dyn Writer, p: usize, bo: &BlockOffset, mode: PredictionMode,
  tx_size: TxSize, tx_type: TxType, plane_bsize: BlockSize, po: &PlaneOffset,
  skip: bool, bit_depth: usize, ac: &[i16], alpha: i16, for_rdo_use: bool
) -> (bool, i64) {
  let qidx = get_qidx(fi, fs, cw, bo);
  let rec = &mut fs.rec.planes[p];
  let PlaneConfig { stride, xdec, ydec, .. } = fs.input.planes[p].cfg;

  if mode.is_intra() {
    mode.predict_intra(&mut rec.mut_slice(po), tx_size, bit_depth, &ac, alpha);
  }

  if skip {
    return (false, -1);
  }

  let mut residual_storage: AlignedArray<[i16; 64 * 64]> =
    UninitializedAlignedArray();
  let mut coeffs_storage: AlignedArray<[i32; 64 * 64]> =
    UninitializedAlignedArray();
  let mut qcoeffs_storage: AlignedArray<[i32; 64 * 64]> =
    UninitializedAlignedArray();
  let mut rcoeffs_storage: AlignedArray<[i32; 64 * 64]> =
    UninitializedAlignedArray();
  let residual = &mut residual_storage.array[..tx_size.area()];
  let coeffs = &mut coeffs_storage.array[..tx_size.area()];
  let qcoeffs = &mut qcoeffs_storage.array[..tx_size.area()];
  let rcoeffs = &mut rcoeffs_storage.array[..tx_size.area()];

  diff(
    residual,
    &fs.input.planes[p].slice(po),
    &rec.slice(po),
    tx_size.width(),
    tx_size.height()
  );

  forward_transform(
    residual,
    coeffs,
    tx_size.width(),
    tx_size,
    tx_type,
    bit_depth
  );

  let coded_tx_size = av1_get_coded_tx_size(tx_size).area();
  fs.qc.quantize(coeffs, qcoeffs, coded_tx_size);

  let has_coeff = cw.write_coeffs_lv_map(
    w,
    p,
    bo,
    &qcoeffs,
    mode,
    tx_size,
    tx_type,
    plane_bsize,
    xdec,
    ydec,
    fi.use_reduced_tx_set
  );

  // Reconstruct
  dequantize(qidx, qcoeffs, rcoeffs, tx_size, bit_depth);

  let mut tx_dist: i64 = -1;

  if !fi.use_tx_domain_distortion || !for_rdo_use {
    inverse_transform_add(
      rcoeffs,
      &mut rec.mut_slice(po).as_mut_slice(),
      stride,
      tx_size,
      tx_type,
      bit_depth
    );
  } else {
    // Store tx-domain distortion of this block
    tx_dist = coeffs
      .iter()
      .zip(rcoeffs)
      .map(|(a, b)| {
        let c = *a as i32 - *b as i32;
        (c * c) as u64
      })
      .sum::<u64>() as i64;

    let tx_dist_scale_bits = 2 * (3 - get_log_tx_scale(tx_size));
    let tx_dist_scale_rounding_offset = 1 << (tx_dist_scale_bits - 1);
    tx_dist = (tx_dist + tx_dist_scale_rounding_offset) >> tx_dist_scale_bits;
  }
  (has_coeff, tx_dist)
}

pub fn write_tx_blocks(
  fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter,
  w: &mut dyn Writer, luma_mode: PredictionMode, chroma_mode: PredictionMode,
  bo: &BlockOffset, bsize: BlockSize, tx_size: TxSize, tx_type: TxType,
  skip: bool, bit_depth: usize, cfl: CFLParams, luma_only: bool,
  for_rdo_use: bool
) -> i64 {
  let bw = bsize.width_mi() / tx_size.width_mi();
  let bh = bsize.height_mi() / tx_size.height_mi();
  let qidx = get_qidx(fi, fs, cw, bo);

  let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;
  let ac = &mut [0i16; 32 * 32];
  let mut tx_dist: i64 = 0;
  let do_chroma = has_chroma(bo, bsize, xdec, ydec);

  fs.qc.update(qidx, tx_size, luma_mode.is_intra(), bit_depth);

  for by in 0..bh {
    for bx in 0..bw {
      let tx_bo = BlockOffset {
        x: bo.x + bx * tx_size.width_mi(),
        y: bo.y + by * tx_size.height_mi()
      };

      let po = tx_bo.plane_offset(&fs.input.planes[0].cfg);
      let (_, dist) = encode_tx_block(
        fi,
        fs,
        cw,
        w,
        0,
        &tx_bo,
        luma_mode,
        tx_size,
        tx_type,
        bsize,
        &po,
        skip,
        bit_depth,
        ac,
        0,
        for_rdo_use
      );
      assert!(
        !fi.use_tx_domain_distortion || !for_rdo_use || skip || dist >= 0
      );
      tx_dist += dist;
    }
  }

  if luma_only {
    return tx_dist;
  };

  // TODO: these are only valid for 4:2:0
  let uv_tx_size = match bsize {
    BlockSize::BLOCK_4X4 | BlockSize::BLOCK_8X8 => TxSize::TX_4X4,
    BlockSize::BLOCK_16X16 => TxSize::TX_8X8,
    BlockSize::BLOCK_32X32 => TxSize::TX_16X16,
    _ => TxSize::TX_32X32
  };

  let mut bw_uv = (bw * tx_size.width_mi()) >> xdec;
  let mut bh_uv = (bh * tx_size.height_mi()) >> ydec;

  if (bw_uv == 0 || bh_uv == 0) && do_chroma {
    bw_uv = 1;
    bh_uv = 1;
  }

  bw_uv /= uv_tx_size.width_mi();
  bh_uv /= uv_tx_size.height_mi();

  let plane_bsize = get_plane_block_size(bsize, xdec, ydec);

  if chroma_mode.is_cfl() {
    luma_ac(ac, fs, bo, bsize);
  }

  if bw_uv > 0 && bh_uv > 0 {
    let uv_tx_type = if uv_tx_size.width() >= 32 || uv_tx_size.height() >= 32 {
      TxType::DCT_DCT
    } else {
      uv_intra_mode_to_tx_type_context(chroma_mode)
    };
    fs.qc.update(fi.base_q_idx, uv_tx_size, true, bit_depth);

    for p in 1..3 {
      let alpha = cfl.alpha(p - 1);
      for by in 0..bh_uv {
        for bx in 0..bw_uv {
          let tx_bo = BlockOffset {
            x: bo.x + ((bx * uv_tx_size.width_mi()) << xdec)
              - ((bw * tx_size.width_mi() == 1) as usize),
            y: bo.y + ((by * uv_tx_size.height_mi()) << ydec)
              - ((bh * tx_size.height_mi() == 1) as usize)
          };

          let mut po = bo.plane_offset(&fs.input.planes[p].cfg);
          po.x += (bx * uv_tx_size.width()) as isize;
          po.y += (by * uv_tx_size.height()) as isize;
          let (_, dist) = encode_tx_block(
            fi,
            fs,
            cw,
            w,
            p,
            &tx_bo,
            chroma_mode,
            uv_tx_size,
            uv_tx_type,
            plane_bsize,
            &po,
            skip,
            bit_depth,
            ac,
            alpha,
            for_rdo_use
          );
          assert!(
            !fi.use_tx_domain_distortion || !for_rdo_use || skip || dist >= 0
          );
          tx_dist += dist;
        }
      }
    }
  }

  tx_dist
}

// FIXME: For now, assume tx_mode is LARGEST_TX, so var-tx is not implemented yet
// but only one tx block exist for a inter mode partition.
pub fn write_tx_tree(
  fi: &FrameInvariants, fs: &mut FrameState, cw: &mut ContextWriter,
  w: &mut dyn Writer, luma_mode: PredictionMode, bo: &BlockOffset,
  bsize: BlockSize, tx_size: TxSize, tx_type: TxType, skip: bool,
  bit_depth: usize, luma_only: bool, for_rdo_use: bool
) -> i64 {
  let bw = bsize.width_mi() / tx_size.width_mi();
  let bh = bsize.height_mi() / tx_size.height_mi();
  let qidx = get_qidx(fi, fs, cw, bo);

  let PlaneConfig { xdec, ydec, .. } = fs.input.planes[1].cfg;
  let ac = &[0i16; 32 * 32];
  let mut tx_dist: i64 = 0;

  fs.qc.update(qidx, tx_size, luma_mode.is_intra(), bit_depth);

  let po = bo.plane_offset(&fs.input.planes[0].cfg);
  let (has_coeff, dist) = encode_tx_block(
    fi,
    fs,
    cw,
    w,
    0,
    &bo,
    luma_mode,
    tx_size,
    tx_type,
    bsize,
    &po,
    skip,
    bit_depth,
    ac,
    0,
    for_rdo_use
  );
  assert!(!fi.use_tx_domain_distortion || !for_rdo_use || skip || dist >= 0);
  tx_dist += dist;

  if luma_only {
    return tx_dist;
  };

  // TODO: these are only valid for 4:2:0
  let uv_tx_size = match bsize {
    BlockSize::BLOCK_4X4 | BlockSize::BLOCK_8X8 => TxSize::TX_4X4,
    BlockSize::BLOCK_16X16 => TxSize::TX_8X8,
    BlockSize::BLOCK_32X32 => TxSize::TX_16X16,
    _ => TxSize::TX_32X32
  };

  let mut bw_uv = (bw * tx_size.width_mi()) >> xdec;
  let mut bh_uv = (bh * tx_size.height_mi()) >> ydec;

  if (bw_uv == 0 || bh_uv == 0) && has_chroma(bo, bsize, xdec, ydec) {
    bw_uv = 1;
    bh_uv = 1;
  }

  bw_uv /= uv_tx_size.width_mi();
  bh_uv /= uv_tx_size.height_mi();

  let plane_bsize = get_plane_block_size(bsize, xdec, ydec);

  if bw_uv > 0 && bh_uv > 0 {
    let uv_tx_type = if has_coeff { tx_type } else { TxType::DCT_DCT }; // if inter mode, uv_tx_type == tx_type

    fs.qc.update(qidx, uv_tx_size, false, bit_depth);

    for p in 1..3 {
      let tx_bo = BlockOffset {
        x: bo.x - ((bw * tx_size.width_mi() == 1) as usize),
        y: bo.y - ((bh * tx_size.height_mi() == 1) as usize)
      };

      let po = bo.plane_offset(&fs.input.planes[p].cfg);
      let (_, dist) = encode_tx_block(
        fi,
        fs,
        cw,
        w,
        p,
        &tx_bo,
        luma_mode,
        uv_tx_size,
        uv_tx_type,
        plane_bsize,
        &po,
        skip,
        bit_depth,
        ac,
        0,
        for_rdo_use
      );
      assert!(
        !fi.use_tx_domain_distortion || !for_rdo_use || skip || dist >= 0
      );
      tx_dist += dist;
    }
  }

  tx_dist
}
