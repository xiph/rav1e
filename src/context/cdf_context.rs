// Copyright (c) 2017-2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use super::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CDFContext {
  pub partition_cdf: [[u16; EXT_PARTITION_TYPES + 1]; PARTITION_CONTEXTS],
  pub kf_y_cdf: [[[u16; INTRA_MODES + 1]; KF_MODE_CONTEXTS]; KF_MODE_CONTEXTS],
  pub y_mode_cdf: [[u16; INTRA_MODES + 1]; BLOCK_SIZE_GROUPS],
  pub uv_mode_cdf: [[[u16; UV_INTRA_MODES + 1]; INTRA_MODES]; 2],
  pub cfl_sign_cdf: [u16; CFL_JOINT_SIGNS + 1],
  pub cfl_alpha_cdf: [[u16; CFL_ALPHABET_SIZE + 1]; CFL_ALPHA_CONTEXTS],
  pub newmv_cdf: [[u16; 2 + 1]; NEWMV_MODE_CONTEXTS],
  pub zeromv_cdf: [[u16; 2 + 1]; GLOBALMV_MODE_CONTEXTS],
  pub refmv_cdf: [[u16; 2 + 1]; REFMV_MODE_CONTEXTS],
  pub intra_tx_cdf: [[[[u16; TX_TYPES + 1]; INTRA_MODES];
    TX_SIZE_SQR_CONTEXTS]; TX_SETS_INTRA],
  pub inter_tx_cdf:
    [[[u16; TX_TYPES + 1]; TX_SIZE_SQR_CONTEXTS]; TX_SETS_INTER],
  pub tx_size_cdf:
    [[[u16; MAX_TX_DEPTH + 1 + 1]; TX_SIZE_CONTEXTS]; MAX_TX_CATS],
  pub txfm_partition_cdf: [[u16; 2 + 1]; TXFM_PARTITION_CONTEXTS],
  pub skip_cdfs: [[u16; 3]; SKIP_CONTEXTS],
  pub intra_inter_cdfs: [[u16; 3]; INTRA_INTER_CONTEXTS],
  pub angle_delta_cdf: [[u16; 2 * MAX_ANGLE_DELTA + 1 + 1]; DIRECTIONAL_MODES],
  pub filter_intra_cdfs: [[u16; 3]; BlockSize::BLOCK_SIZES_ALL],
  pub palette_y_mode_cdfs:
    [[[u16; 3]; PALETTE_Y_MODE_CONTEXTS]; PALETTE_BSIZE_CTXS],
  pub palette_uv_mode_cdfs: [[u16; 3]; PALETTE_UV_MODE_CONTEXTS],
  pub comp_mode_cdf: [[u16; 3]; COMP_INTER_CONTEXTS],
  pub comp_ref_type_cdf: [[u16; 3]; COMP_REF_TYPE_CONTEXTS],
  pub comp_ref_cdf: [[[u16; 3]; FWD_REFS - 1]; REF_CONTEXTS],
  pub comp_bwd_ref_cdf: [[[u16; 3]; BWD_REFS - 1]; REF_CONTEXTS],
  pub single_ref_cdfs: [[[u16; 2 + 1]; SINGLE_REFS - 1]; REF_CONTEXTS],
  pub drl_cdfs: [[u16; 2 + 1]; DRL_MODE_CONTEXTS],
  pub compound_mode_cdf:
    [[u16; INTER_COMPOUND_MODES + 1]; INTER_MODE_CONTEXTS],
  pub nmv_context: NMVContext,
  pub deblock_delta_multi_cdf: [[u16; DELTA_LF_PROBS + 1 + 1]; FRAME_LF_COUNT],
  pub deblock_delta_cdf: [u16; DELTA_LF_PROBS + 1 + 1],
  pub spatial_segmentation_cdfs: [[u16; 8 + 1]; 3],
  pub lrf_switchable_cdf: [u16; 3 + 1],
  pub lrf_sgrproj_cdf: [u16; 2 + 1],
  pub lrf_wiener_cdf: [u16; 2 + 1],

  // lv_map
  pub txb_skip_cdf: [[[u16; 3]; TXB_SKIP_CONTEXTS]; TxSize::TX_SIZES],
  pub dc_sign_cdf: [[[u16; 3]; DC_SIGN_CONTEXTS]; PLANE_TYPES],
  pub eob_extra_cdf:
    [[[[u16; 3]; EOB_COEF_CONTEXTS]; PLANE_TYPES]; TxSize::TX_SIZES],

  pub eob_flag_cdf16: [[[u16; 5 + 1]; 2]; PLANE_TYPES],
  pub eob_flag_cdf32: [[[u16; 6 + 1]; 2]; PLANE_TYPES],
  pub eob_flag_cdf64: [[[u16; 7 + 1]; 2]; PLANE_TYPES],
  pub eob_flag_cdf128: [[[u16; 8 + 1]; 2]; PLANE_TYPES],
  pub eob_flag_cdf256: [[[u16; 9 + 1]; 2]; PLANE_TYPES],
  pub eob_flag_cdf512: [[[u16; 10 + 1]; 2]; PLANE_TYPES],
  pub eob_flag_cdf1024: [[[u16; 11 + 1]; 2]; PLANE_TYPES],

  pub coeff_base_eob_cdf:
    [[[[u16; 3 + 1]; SIG_COEF_CONTEXTS_EOB]; PLANE_TYPES]; TxSize::TX_SIZES],
  pub coeff_base_cdf:
    [[[[u16; 4 + 1]; SIG_COEF_CONTEXTS]; PLANE_TYPES]; TxSize::TX_SIZES],
  pub coeff_br_cdf: [[[[u16; BR_CDF_SIZE + 1]; LEVEL_CONTEXTS]; PLANE_TYPES];
    TxSize::TX_SIZES],
}

impl CDFContext {
  pub fn new(quantizer: u8) -> CDFContext {
    let qctx = match quantizer {
      0..=20 => 0,
      21..=60 => 1,
      61..=120 => 2,
      _ => 3,
    };
    CDFContext {
      partition_cdf: default_partition_cdf,
      kf_y_cdf: default_kf_y_mode_cdf,
      y_mode_cdf: default_if_y_mode_cdf,
      uv_mode_cdf: default_uv_mode_cdf,
      cfl_sign_cdf: default_cfl_sign_cdf,
      cfl_alpha_cdf: default_cfl_alpha_cdf,
      newmv_cdf: default_newmv_cdf,
      zeromv_cdf: default_zeromv_cdf,
      refmv_cdf: default_refmv_cdf,
      intra_tx_cdf: default_intra_ext_tx_cdf,
      inter_tx_cdf: default_inter_ext_tx_cdf,
      tx_size_cdf: default_tx_size_cdf,
      txfm_partition_cdf: default_txfm_partition_cdf,
      skip_cdfs: default_skip_cdfs,
      intra_inter_cdfs: default_intra_inter_cdf,
      angle_delta_cdf: default_angle_delta_cdf,
      filter_intra_cdfs: default_filter_intra_cdfs,
      palette_y_mode_cdfs: default_palette_y_mode_cdfs,
      palette_uv_mode_cdfs: default_palette_uv_mode_cdfs,
      comp_mode_cdf: default_comp_mode_cdf,
      comp_ref_type_cdf: default_comp_ref_type_cdf,
      comp_ref_cdf: default_comp_ref_cdf,
      comp_bwd_ref_cdf: default_comp_bwdref_cdf,
      single_ref_cdfs: default_single_ref_cdf,
      drl_cdfs: default_drl_cdf,
      compound_mode_cdf: default_compound_mode_cdf,
      nmv_context: default_nmv_context,
      deblock_delta_multi_cdf: default_delta_lf_multi_cdf,
      deblock_delta_cdf: default_delta_lf_cdf,
      spatial_segmentation_cdfs: default_spatial_pred_seg_tree_cdf,
      lrf_switchable_cdf: default_switchable_restore_cdf,
      lrf_sgrproj_cdf: default_sgrproj_restore_cdf,
      lrf_wiener_cdf: default_wiener_restore_cdf,

      // lv_map
      txb_skip_cdf: av1_default_txb_skip_cdfs[qctx],
      dc_sign_cdf: av1_default_dc_sign_cdfs[qctx],
      eob_extra_cdf: av1_default_eob_extra_cdfs[qctx],

      eob_flag_cdf16: av1_default_eob_multi16_cdfs[qctx],
      eob_flag_cdf32: av1_default_eob_multi32_cdfs[qctx],
      eob_flag_cdf64: av1_default_eob_multi64_cdfs[qctx],
      eob_flag_cdf128: av1_default_eob_multi128_cdfs[qctx],
      eob_flag_cdf256: av1_default_eob_multi256_cdfs[qctx],
      eob_flag_cdf512: av1_default_eob_multi512_cdfs[qctx],
      eob_flag_cdf1024: av1_default_eob_multi1024_cdfs[qctx],

      coeff_base_eob_cdf: av1_default_coeff_base_eob_multi_cdfs[qctx],
      coeff_base_cdf: av1_default_coeff_base_multi_cdfs[qctx],
      coeff_br_cdf: av1_default_coeff_lps_multi_cdfs[qctx],
    }
  }

  pub fn reset_counts(&mut self) {
    macro_rules! reset_1d {
      ($field:expr) => {
        let r = $field.last_mut().unwrap();
        *r = 0;
      };
    }
    macro_rules! reset_2d {
      ($field:expr) => {
        for x in $field.iter_mut() {
          reset_1d!(x);
        }
      };
    }
    macro_rules! reset_3d {
      ($field:expr) => {
        for x in $field.iter_mut() {
          reset_2d!(x);
        }
      };
    }
    macro_rules! reset_4d {
      ($field:expr) => {
        for x in $field.iter_mut() {
          reset_3d!(x);
        }
      };
    }

    for i in 0..4 {
      self.partition_cdf[i][4] = 0;
    }
    for i in 4..16 {
      self.partition_cdf[i][10] = 0;
    }
    for i in 16..20 {
      self.partition_cdf[i][8] = 0;
    }

    reset_3d!(self.kf_y_cdf);
    reset_2d!(self.y_mode_cdf);

    for i in 0..INTRA_MODES {
      self.uv_mode_cdf[0][i][UV_INTRA_MODES - 1] = 0;
      self.uv_mode_cdf[1][i][UV_INTRA_MODES] = 0;
    }
    reset_1d!(self.cfl_sign_cdf);
    reset_2d!(self.cfl_alpha_cdf);
    reset_2d!(self.newmv_cdf);
    reset_2d!(self.zeromv_cdf);
    reset_2d!(self.refmv_cdf);

    for i in 0..TX_SIZE_SQR_CONTEXTS {
      for j in 0..INTRA_MODES {
        self.intra_tx_cdf[1][i][j][7] = 0;
        self.intra_tx_cdf[2][i][j][5] = 0;
      }
      self.inter_tx_cdf[1][i][16] = 0;
      self.inter_tx_cdf[2][i][12] = 0;
      self.inter_tx_cdf[3][i][2] = 0;
    }

    for i in 0..TX_SIZE_CONTEXTS {
      self.tx_size_cdf[0][i][MAX_TX_DEPTH] = 0;
    }
    reset_2d!(self.tx_size_cdf[1]);
    reset_2d!(self.tx_size_cdf[2]);
    reset_2d!(self.tx_size_cdf[3]);

    for i in 0..TXFM_PARTITION_CONTEXTS {
      self.txfm_partition_cdf[i][2] = 0;
    }

    reset_2d!(self.skip_cdfs);
    reset_2d!(self.intra_inter_cdfs);
    reset_2d!(self.angle_delta_cdf);
    reset_2d!(self.filter_intra_cdfs);
    reset_3d!(self.palette_y_mode_cdfs);
    reset_2d!(self.palette_uv_mode_cdfs);
    reset_2d!(self.comp_mode_cdf);
    reset_2d!(self.comp_ref_type_cdf);
    reset_3d!(self.comp_ref_cdf);
    reset_3d!(self.comp_bwd_ref_cdf);
    reset_3d!(self.single_ref_cdfs);
    reset_2d!(self.drl_cdfs);
    reset_2d!(self.compound_mode_cdf);
    reset_2d!(self.deblock_delta_multi_cdf);
    reset_1d!(self.deblock_delta_cdf);
    reset_2d!(self.spatial_segmentation_cdfs);
    reset_1d!(self.lrf_switchable_cdf);
    reset_1d!(self.lrf_sgrproj_cdf);
    reset_1d!(self.lrf_wiener_cdf);

    reset_1d!(self.nmv_context.joints_cdf);
    for i in 0..2 {
      reset_1d!(self.nmv_context.comps[i].classes_cdf);
      reset_2d!(self.nmv_context.comps[i].class0_fp_cdf);
      reset_1d!(self.nmv_context.comps[i].fp_cdf);
      reset_1d!(self.nmv_context.comps[i].sign_cdf);
      reset_1d!(self.nmv_context.comps[i].class0_hp_cdf);
      reset_1d!(self.nmv_context.comps[i].hp_cdf);
      reset_1d!(self.nmv_context.comps[i].class0_cdf);
      reset_2d!(self.nmv_context.comps[i].bits_cdf);
    }

    // lv_map
    reset_3d!(self.txb_skip_cdf);
    reset_3d!(self.dc_sign_cdf);
    reset_4d!(self.eob_extra_cdf);

    reset_3d!(self.eob_flag_cdf16);
    reset_3d!(self.eob_flag_cdf32);
    reset_3d!(self.eob_flag_cdf64);
    reset_3d!(self.eob_flag_cdf128);
    reset_3d!(self.eob_flag_cdf256);
    reset_3d!(self.eob_flag_cdf512);
    reset_3d!(self.eob_flag_cdf1024);

    reset_4d!(self.coeff_base_eob_cdf);
    reset_4d!(self.coeff_base_cdf);
    reset_4d!(self.coeff_br_cdf);
  }

  pub fn build_map(&self) -> HashMap<&'static str, (usize, usize)> {
    use std::mem::size_of_val;

    let partition_cdf_start =
      self.partition_cdf.first().unwrap().as_ptr() as usize;
    let partition_cdf_end =
      partition_cdf_start + size_of_val(&self.partition_cdf);
    let kf_y_cdf_start = self.kf_y_cdf.first().unwrap().as_ptr() as usize;
    let kf_y_cdf_end = kf_y_cdf_start + size_of_val(&self.kf_y_cdf);
    let y_mode_cdf_start = self.y_mode_cdf.first().unwrap().as_ptr() as usize;
    let y_mode_cdf_end = y_mode_cdf_start + size_of_val(&self.y_mode_cdf);
    let uv_mode_cdf_start =
      self.uv_mode_cdf.first().unwrap().as_ptr() as usize;
    let uv_mode_cdf_end = uv_mode_cdf_start + size_of_val(&self.uv_mode_cdf);
    let cfl_sign_cdf_start = self.cfl_sign_cdf.as_ptr() as usize;
    let cfl_sign_cdf_end =
      cfl_sign_cdf_start + size_of_val(&self.cfl_sign_cdf);
    let cfl_alpha_cdf_start =
      self.cfl_alpha_cdf.first().unwrap().as_ptr() as usize;
    let cfl_alpha_cdf_end =
      cfl_alpha_cdf_start + size_of_val(&self.cfl_alpha_cdf);
    let newmv_cdf_start = self.newmv_cdf.first().unwrap().as_ptr() as usize;
    let newmv_cdf_end = newmv_cdf_start + size_of_val(&self.newmv_cdf);
    let zeromv_cdf_start = self.zeromv_cdf.first().unwrap().as_ptr() as usize;
    let zeromv_cdf_end = zeromv_cdf_start + size_of_val(&self.zeromv_cdf);
    let refmv_cdf_start = self.refmv_cdf.first().unwrap().as_ptr() as usize;
    let refmv_cdf_end = refmv_cdf_start + size_of_val(&self.refmv_cdf);
    let intra_tx_cdf_start =
      self.intra_tx_cdf.first().unwrap().as_ptr() as usize;
    let intra_tx_cdf_end =
      intra_tx_cdf_start + size_of_val(&self.intra_tx_cdf);
    let inter_tx_cdf_start =
      self.inter_tx_cdf.first().unwrap().as_ptr() as usize;
    let inter_tx_cdf_end =
      inter_tx_cdf_start + size_of_val(&self.inter_tx_cdf);
    let tx_size_cdf_start =
      self.tx_size_cdf.first().unwrap().as_ptr() as usize;
    let tx_size_cdf_end = tx_size_cdf_start + size_of_val(&self.tx_size_cdf);
    let txfm_partition_cdf_start =
      self.txfm_partition_cdf.first().unwrap().as_ptr() as usize;
    let txfm_partition_cdf_end =
      txfm_partition_cdf_start + size_of_val(&self.txfm_partition_cdf);
    let skip_cdfs_start = self.skip_cdfs.first().unwrap().as_ptr() as usize;
    let skip_cdfs_end = skip_cdfs_start + size_of_val(&self.skip_cdfs);
    let intra_inter_cdfs_start =
      self.intra_inter_cdfs.first().unwrap().as_ptr() as usize;
    let intra_inter_cdfs_end =
      intra_inter_cdfs_start + size_of_val(&self.intra_inter_cdfs);
    let angle_delta_cdf_start =
      self.angle_delta_cdf.first().unwrap().as_ptr() as usize;
    let angle_delta_cdf_end =
      angle_delta_cdf_start + size_of_val(&self.angle_delta_cdf);
    let filter_intra_cdfs_start =
      self.filter_intra_cdfs.first().unwrap().as_ptr() as usize;
    let filter_intra_cdfs_end =
      filter_intra_cdfs_start + size_of_val(&self.filter_intra_cdfs);
    let palette_y_mode_cdfs_start =
      self.palette_y_mode_cdfs.first().unwrap().as_ptr() as usize;
    let palette_y_mode_cdfs_end =
      palette_y_mode_cdfs_start + size_of_val(&self.palette_y_mode_cdfs);
    let palette_uv_mode_cdfs_start =
      self.palette_uv_mode_cdfs.first().unwrap().as_ptr() as usize;
    let palette_uv_mode_cdfs_end =
      palette_uv_mode_cdfs_start + size_of_val(&self.palette_uv_mode_cdfs);
    let comp_mode_cdf_start =
      self.comp_mode_cdf.first().unwrap().as_ptr() as usize;
    let comp_mode_cdf_end =
      comp_mode_cdf_start + size_of_val(&self.comp_mode_cdf);
    let comp_ref_type_cdf_start =
      self.comp_ref_type_cdf.first().unwrap().as_ptr() as usize;
    let comp_ref_type_cdf_end =
      comp_ref_type_cdf_start + size_of_val(&self.comp_ref_type_cdf);
    let comp_ref_cdf_start =
      self.comp_ref_cdf.first().unwrap().as_ptr() as usize;
    let comp_ref_cdf_end =
      comp_ref_cdf_start + size_of_val(&self.comp_ref_cdf);
    let comp_bwd_ref_cdf_start =
      self.comp_bwd_ref_cdf.first().unwrap().as_ptr() as usize;
    let comp_bwd_ref_cdf_end =
      comp_bwd_ref_cdf_start + size_of_val(&self.comp_bwd_ref_cdf);
    let single_ref_cdfs_start =
      self.single_ref_cdfs.first().unwrap().as_ptr() as usize;
    let single_ref_cdfs_end =
      single_ref_cdfs_start + size_of_val(&self.single_ref_cdfs);
    let drl_cdfs_start = self.drl_cdfs.first().unwrap().as_ptr() as usize;
    let drl_cdfs_end = drl_cdfs_start + size_of_val(&self.drl_cdfs);
    let compound_mode_cdf_start =
      self.compound_mode_cdf.first().unwrap().as_ptr() as usize;
    let compound_mode_cdf_end =
      compound_mode_cdf_start + size_of_val(&self.compound_mode_cdf);
    let nmv_context_start = &self.nmv_context as *const NMVContext as usize;
    let nmv_context_end = nmv_context_start + size_of_val(&self.nmv_context);
    let deblock_delta_multi_cdf_start =
      self.deblock_delta_multi_cdf.first().unwrap().as_ptr() as usize;
    let deblock_delta_multi_cdf_end = deblock_delta_multi_cdf_start
      + size_of_val(&self.deblock_delta_multi_cdf);
    let deblock_delta_cdf_start = self.deblock_delta_cdf.as_ptr() as usize;
    let deblock_delta_cdf_end =
      deblock_delta_cdf_start + size_of_val(&self.deblock_delta_cdf);
    let spatial_segmentation_cdfs_start =
      self.spatial_segmentation_cdfs.first().unwrap().as_ptr() as usize;
    let spatial_segmentation_cdfs_end = spatial_segmentation_cdfs_start
      + size_of_val(&self.spatial_segmentation_cdfs);
    let lrf_switchable_cdf_start = self.lrf_switchable_cdf.as_ptr() as usize;
    let lrf_switchable_cdf_end =
      lrf_switchable_cdf_start + size_of_val(&self.lrf_switchable_cdf);
    let lrf_sgrproj_cdf_start = self.lrf_sgrproj_cdf.as_ptr() as usize;
    let lrf_sgrproj_cdf_end =
      lrf_sgrproj_cdf_start + size_of_val(&self.lrf_sgrproj_cdf);
    let lrf_wiener_cdf_start = self.lrf_wiener_cdf.as_ptr() as usize;
    let lrf_wiener_cdf_end =
      lrf_wiener_cdf_start + size_of_val(&self.lrf_wiener_cdf);

    let txb_skip_cdf_start =
      self.txb_skip_cdf.first().unwrap().as_ptr() as usize;
    let txb_skip_cdf_end =
      txb_skip_cdf_start + size_of_val(&self.txb_skip_cdf);
    let dc_sign_cdf_start =
      self.dc_sign_cdf.first().unwrap().as_ptr() as usize;
    let dc_sign_cdf_end = dc_sign_cdf_start + size_of_val(&self.dc_sign_cdf);
    let eob_extra_cdf_start =
      self.eob_extra_cdf.first().unwrap().as_ptr() as usize;
    let eob_extra_cdf_end =
      eob_extra_cdf_start + size_of_val(&self.eob_extra_cdf);
    let eob_flag_cdf16_start =
      self.eob_flag_cdf16.first().unwrap().as_ptr() as usize;
    let eob_flag_cdf16_end =
      eob_flag_cdf16_start + size_of_val(&self.eob_flag_cdf16);
    let eob_flag_cdf32_start =
      self.eob_flag_cdf32.first().unwrap().as_ptr() as usize;
    let eob_flag_cdf32_end =
      eob_flag_cdf32_start + size_of_val(&self.eob_flag_cdf32);
    let eob_flag_cdf64_start =
      self.eob_flag_cdf64.first().unwrap().as_ptr() as usize;
    let eob_flag_cdf64_end =
      eob_flag_cdf64_start + size_of_val(&self.eob_flag_cdf64);
    let eob_flag_cdf128_start =
      self.eob_flag_cdf128.first().unwrap().as_ptr() as usize;
    let eob_flag_cdf128_end =
      eob_flag_cdf128_start + size_of_val(&self.eob_flag_cdf128);
    let eob_flag_cdf256_start =
      self.eob_flag_cdf256.first().unwrap().as_ptr() as usize;
    let eob_flag_cdf256_end =
      eob_flag_cdf256_start + size_of_val(&self.eob_flag_cdf256);
    let eob_flag_cdf512_start =
      self.eob_flag_cdf512.first().unwrap().as_ptr() as usize;
    let eob_flag_cdf512_end =
      eob_flag_cdf512_start + size_of_val(&self.eob_flag_cdf512);
    let eob_flag_cdf1024_start =
      self.eob_flag_cdf1024.first().unwrap().as_ptr() as usize;
    let eob_flag_cdf1024_end =
      eob_flag_cdf1024_start + size_of_val(&self.eob_flag_cdf1024);
    let coeff_base_eob_cdf_start =
      self.coeff_base_eob_cdf.first().unwrap().as_ptr() as usize;
    let coeff_base_eob_cdf_end =
      coeff_base_eob_cdf_start + size_of_val(&self.coeff_base_eob_cdf);
    let coeff_base_cdf_start =
      self.coeff_base_cdf.first().unwrap().as_ptr() as usize;
    let coeff_base_cdf_end =
      coeff_base_cdf_start + size_of_val(&self.coeff_base_cdf);
    let coeff_br_cdf_start =
      self.coeff_br_cdf.first().unwrap().as_ptr() as usize;
    let coeff_br_cdf_end =
      coeff_br_cdf_start + size_of_val(&self.coeff_br_cdf);

    [
      ("partition_cdf", (partition_cdf_start, partition_cdf_end)),
      ("kf_y_cdf", (kf_y_cdf_start, kf_y_cdf_end)),
      ("y_mode_cdf", (y_mode_cdf_start, y_mode_cdf_end)),
      ("uv_mode_cdf", (uv_mode_cdf_start, uv_mode_cdf_end)),
      ("cfl_sign_cdf", (cfl_sign_cdf_start, cfl_sign_cdf_end)),
      ("cfl_alpha_cdf", (cfl_alpha_cdf_start, cfl_alpha_cdf_end)),
      ("newmv_cdf", (newmv_cdf_start, newmv_cdf_end)),
      ("zeromv_cdf", (zeromv_cdf_start, zeromv_cdf_end)),
      ("refmv_cdf", (refmv_cdf_start, refmv_cdf_end)),
      ("intra_tx_cdf", (intra_tx_cdf_start, intra_tx_cdf_end)),
      ("inter_tx_cdf", (inter_tx_cdf_start, inter_tx_cdf_end)),
      ("tx_size_cdf", (tx_size_cdf_start, tx_size_cdf_end)),
      (
        "txfm_partition_cdf",
        (txfm_partition_cdf_start, txfm_partition_cdf_end),
      ),
      ("skip_cdfs", (skip_cdfs_start, skip_cdfs_end)),
      ("intra_inter_cdfs", (intra_inter_cdfs_start, intra_inter_cdfs_end)),
      ("angle_delta_cdf", (angle_delta_cdf_start, angle_delta_cdf_end)),
      ("filter_intra_cdfs", (filter_intra_cdfs_start, filter_intra_cdfs_end)),
      (
        "palette_y_mode_cdfs",
        (palette_y_mode_cdfs_start, palette_y_mode_cdfs_end),
      ),
      (
        "palette_uv_mode_cdfs",
        (palette_uv_mode_cdfs_start, palette_uv_mode_cdfs_end),
      ),
      ("comp_mode_cdf", (comp_mode_cdf_start, comp_mode_cdf_end)),
      ("comp_ref_type_cdf", (comp_ref_type_cdf_start, comp_ref_type_cdf_end)),
      ("comp_ref_cdf", (comp_ref_cdf_start, comp_ref_cdf_end)),
      ("comp_bwd_ref_cdf", (comp_bwd_ref_cdf_start, comp_bwd_ref_cdf_end)),
      ("single_ref_cdfs", (single_ref_cdfs_start, single_ref_cdfs_end)),
      ("drl_cdfs", (drl_cdfs_start, drl_cdfs_end)),
      ("compound_mode_cdf", (compound_mode_cdf_start, compound_mode_cdf_end)),
      ("nmv_context", (nmv_context_start, nmv_context_end)),
      (
        "deblock_delta_multi_cdf",
        (deblock_delta_multi_cdf_start, deblock_delta_multi_cdf_end),
      ),
      ("deblock_delta_cdf", (deblock_delta_cdf_start, deblock_delta_cdf_end)),
      (
        "spatial_segmentation_cdfs",
        (spatial_segmentation_cdfs_start, spatial_segmentation_cdfs_end),
      ),
      (
        "lrf_switchable_cdf",
        (lrf_switchable_cdf_start, lrf_switchable_cdf_end),
      ),
      ("lrf_sgrproj_cdf", (lrf_sgrproj_cdf_start, lrf_sgrproj_cdf_end)),
      ("lrf_wiener_cdf", (lrf_wiener_cdf_start, lrf_wiener_cdf_end)),
      ("txb_skip_cdf", (txb_skip_cdf_start, txb_skip_cdf_end)),
      ("dc_sign_cdf", (dc_sign_cdf_start, dc_sign_cdf_end)),
      ("eob_extra_cdf", (eob_extra_cdf_start, eob_extra_cdf_end)),
      ("eob_flag_cdf16", (eob_flag_cdf16_start, eob_flag_cdf16_end)),
      ("eob_flag_cdf32", (eob_flag_cdf32_start, eob_flag_cdf32_end)),
      ("eob_flag_cdf64", (eob_flag_cdf64_start, eob_flag_cdf64_end)),
      ("eob_flag_cdf128", (eob_flag_cdf128_start, eob_flag_cdf128_end)),
      ("eob_flag_cdf256", (eob_flag_cdf256_start, eob_flag_cdf256_end)),
      ("eob_flag_cdf512", (eob_flag_cdf512_start, eob_flag_cdf512_end)),
      ("eob_flag_cdf1024", (eob_flag_cdf1024_start, eob_flag_cdf1024_end)),
      (
        "coeff_base_eob_cdf",
        (coeff_base_eob_cdf_start, coeff_base_eob_cdf_end),
      ),
      ("coeff_base_cdf", (coeff_base_cdf_start, coeff_base_cdf_end)),
      ("coeff_br_cdf", (coeff_br_cdf_start, coeff_br_cdf_end)),
    ]
    .iter()
    .cloned()
    .collect()
  }
}

/*
impl fmt::Debug for CDFContext {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "CDFContext contains too many numbers to print :-(")
  }
}
*/

#[derive(Debug, Default)]
pub struct FieldMap {
  map: HashMap<&'static str, (usize, usize)>,
  log: HashMap<usize, (&'static str, usize, usize)>,
}

impl FieldMap {
  fn new(fc: &CDFContext) -> FieldMap {
    Self { map: fc.build_map(), log: Default::default() }
  }

  /// Print the field the address belong to
  pub(crate) fn lookup(&self, addr: usize) -> (&'static str, usize, usize) {
    for (name, (start, end)) in &self.map {
      if addr >= *start && addr < *end {
        return (name, *start, *end);
      }
    }

    panic!("  CDF address not found {:x}", addr);
  }

  pub(crate) fn update(&mut self, addr: usize) {
    let (name, start, end) = self.lookup(addr);
    #[cfg(feature = "desync_finder")]
    {
      println!(" CDF {}", name);
      println!();
    }

    self.log.entry(start).and_modify(|v| v.1 += 1).or_insert((
      name,
      1,
      end - start,
    ));
  }

  fn summary(&self, ctx: usize) {
    println!("Summary for {:x}", ctx);
    let mut sum = 0;
    for (k, v) in self.log.iter() {
      println!(" {:x} {:x} {}: {} {}b", ctx, k, v.0, v.1, v.2);
      sum += v.2;
    }
    println!("total: {}", sum);
  }
}

#[macro_use]
macro_rules! symbol_with_update {
  ($self:ident, $w:ident, $s:expr, $cdf:expr) => {
    $w.symbol_with_update($s, $cdf);
    {
      let cdf: &[_] = $cdf;
      let map = &mut $self.fc_map;
      map.update(cdf.as_ptr() as usize);
    }
  };
}

pub struct ContextWriterCheckpoint {
  pub fc: CDFContext,
  pub bc: BlockContextCheckpoint,
  pub fc_map: FieldMap,
}

pub struct ContextWriter<'a> {
  pub bc: BlockContext<'a>,
  pub fc: &'a mut CDFContext,
  pub fc_map: FieldMap,
  pub debug: bool,
}

impl<'a> ContextWriter<'a> {
  pub fn new(fc: &'a mut CDFContext, bc: BlockContext<'a>) -> Self {
    let fc_map = FieldMap::new(fc);
    let debug = std::env::var_os("RAV1E_DEBUG").is_some();

    ContextWriter { fc, bc, fc_map, debug }
  }

  pub fn cdf_element_prob(cdf: &[u16], element: usize) -> u16 {
    (if element > 0 { cdf[element - 1] } else { 32768 }) - cdf[element]
  }

  pub fn checkpoint(&self) -> ContextWriterCheckpoint {
    let mut cc = ContextWriterCheckpoint {
      fc: *self.fc,
      bc: self.bc.checkpoint(),
      fc_map: Default::default(),
    };

    cc.fc_map = FieldMap::new(&cc.fc);

    cc
  }

  pub fn rollback(&mut self, checkpoint: &ContextWriterCheckpoint) {
    for (d_start, (name, _, len)) in self.fc_map.log.iter() {
      let (s_start, _s_end) = checkpoint.fc_map.map[name];
      unsafe {
        std::ptr::copy_nonoverlapping(
          s_start as *const u8,
          (*d_start) as *mut u8,
          *len,
        );
      }
    }
    // *self.fc = checkpoint.fc;
    self.bc.rollback(&checkpoint.bc);
    if self.debug {
      self.fc_map.summary(self as *const Self as usize);
      self.fc_map.log.clear();
    }
    pretty_assertions::assert_eq!(self.fc, &checkpoint.fc);
  }
}
