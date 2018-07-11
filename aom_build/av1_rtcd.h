// This file is generated. Do not edit.
#ifndef AV1_RTCD_H_
#define AV1_RTCD_H_

#ifdef RTCD_C
#define RTCD_EXTERN
#else
#define RTCD_EXTERN extern
#endif

/*
 * AV1
 */

#include "aom/aom_integer.h"
#include "aom_dsp/txfm_common.h"
#include "av1/common/common.h"
#include "av1/common/enums.h"
#include "av1/common/quant_common.h"
#include "av1/common/filter.h"
#include "av1/common/convolve.h"
#include "av1/common/av1_txfm.h"
#include "av1/common/odintrin.h"

#if CONFIG_LOOP_RESTORATION
#include "av1/common/restoration.h"
#endif

#if CONFIG_CFL
#include "av1/common/cfl.h"
#endif

struct macroblockd;

/* Encoder forward decls */
struct macroblock;
struct txfm_param;
struct aom_variance_vtable;
struct search_site_config;
struct mv;
union int_mv;
struct yv12_buffer_config;

#ifdef __cplusplus
extern "C" {
#endif

void apply_selfguided_restoration_c(const uint8_t *dat, int width, int height, int stride, int eps, const int *xqd, uint8_t *dst, int dst_stride, int32_t *tmpbuf, int bit_depth, int highbd);
#define apply_selfguided_restoration apply_selfguided_restoration_c

void av1_convolve_2d_c(const uint8_t *src, int src_stride, uint8_t *dst, int dst_stride, int w, int h, InterpFilterParams *filter_params_x, InterpFilterParams *filter_params_y, const int subpel_x_q4, const int subpel_y_q4, ConvolveParams *conv_params);
#define av1_convolve_2d av1_convolve_2d_c

void av1_convolve_2d_copy_c(const uint8_t *src, int src_stride, uint8_t *dst, int dst_stride, int w, int h, InterpFilterParams *filter_params_x, InterpFilterParams *filter_params_y, const int subpel_x_q4, const int subpel_y_q4, ConvolveParams *conv_params);
#define av1_convolve_2d_copy av1_convolve_2d_copy_c

void av1_convolve_2d_copy_sr_c(const uint8_t *src, int src_stride, uint8_t *dst, int dst_stride, int w, int h, InterpFilterParams *filter_params_x, InterpFilterParams *filter_params_y, const int subpel_x_q4, const int subpel_y_q4, ConvolveParams *conv_params);
#define av1_convolve_2d_copy_sr av1_convolve_2d_copy_sr_c

void av1_convolve_2d_scale_c(const uint8_t *src, int src_stride, CONV_BUF_TYPE *dst, int dst_stride, int w, int h, InterpFilterParams *filter_params_x, InterpFilterParams *filter_params_y, const int subpel_x_qn, const int x_step_qn, const int subpel_y_q4, const int y_step_qn, ConvolveParams *conv_params);
#define av1_convolve_2d_scale av1_convolve_2d_scale_c

void av1_convolve_2d_sr_c(const uint8_t *src, int src_stride, uint8_t *dst, int dst_stride, int w, int h, InterpFilterParams *filter_params_x, InterpFilterParams *filter_params_y, const int subpel_x_q4, const int subpel_y_q4, ConvolveParams *conv_params);
#define av1_convolve_2d_sr av1_convolve_2d_sr_c

void av1_convolve_horiz_c(const uint8_t *src, int src_stride, uint8_t *dst, int dst_stride, int w, int h, const InterpFilterParams fp, const int subpel_x_q4, int x_step_q4, ConvolveParams *conv_params);
#define av1_convolve_horiz av1_convolve_horiz_c

void av1_convolve_horiz_rs_c(const uint8_t *src, int src_stride, uint8_t *dst, int dst_stride, int w, int h, const int16_t *x_filters, const int x0_qn, const int x_step_qn);
#define av1_convolve_horiz_rs av1_convolve_horiz_rs_c

void av1_convolve_rounding_c(const int32_t *src, int src_stride, uint8_t *dst, int dst_stride, int w, int h, int bits);
#define av1_convolve_rounding av1_convolve_rounding_c

void av1_convolve_vert_c(const uint8_t *src, int src_stride, uint8_t *dst, int dst_stride, int w, int h, const InterpFilterParams fp, const int subpel_x_q4, int x_step_q4, ConvolveParams *conv_params);
#define av1_convolve_vert av1_convolve_vert_c

void av1_convolve_x_c(const uint8_t *src, int src_stride, uint8_t *dst, int dst_stride, int w, int h, InterpFilterParams *filter_params_x, InterpFilterParams *filter_params_y, const int subpel_x_q4, const int subpel_y_q4, ConvolveParams *conv_params);
#define av1_convolve_x av1_convolve_x_c

void av1_convolve_x_sr_c(const uint8_t *src, int src_stride, uint8_t *dst, int dst_stride, int w, int h, InterpFilterParams *filter_params_x, InterpFilterParams *filter_params_y, const int subpel_x_q4, const int subpel_y_q4, ConvolveParams *conv_params);
#define av1_convolve_x_sr av1_convolve_x_sr_c

void av1_convolve_y_c(const uint8_t *src, int src_stride, uint8_t *dst, int dst_stride, int w, int h, InterpFilterParams *filter_params_x, InterpFilterParams *filter_params_y, const int subpel_x_q4, const int subpel_y_q4, ConvolveParams *conv_params);
#define av1_convolve_y av1_convolve_y_c

void av1_convolve_y_sr_c(const uint8_t *src, int src_stride, uint8_t *dst, int dst_stride, int w, int h, InterpFilterParams *filter_params_x, InterpFilterParams *filter_params_y, const int subpel_x_q4, const int subpel_y_q4, ConvolveParams *conv_params);
#define av1_convolve_y_sr av1_convolve_y_sr_c

void av1_filter_intra_edge_c(uint8_t *p, int sz, int strength);
#define av1_filter_intra_edge av1_filter_intra_edge_c

void av1_filter_intra_edge_high_c(uint16_t *p, int sz, int strength);
#define av1_filter_intra_edge_high av1_filter_intra_edge_high_c

void av1_get_br_level_counts_c(const uint8_t *const levels, const int width, const int height, uint8_t *const level_counts);
#define av1_get_br_level_counts av1_get_br_level_counts_c

void av1_highbd_convolve8_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h, int bps);
#define av1_highbd_convolve8 av1_highbd_convolve8_c

void av1_highbd_convolve8_avg_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h, int bps);
#define av1_highbd_convolve8_avg av1_highbd_convolve8_avg_c

void av1_highbd_convolve8_avg_horiz_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h, int bps);
#define av1_highbd_convolve8_avg_horiz av1_highbd_convolve8_avg_horiz_c

void av1_highbd_convolve8_avg_vert_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h, int bps);
#define av1_highbd_convolve8_avg_vert av1_highbd_convolve8_avg_vert_c

void av1_highbd_convolve8_horiz_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h, int bps);
#define av1_highbd_convolve8_horiz av1_highbd_convolve8_horiz_c

void av1_highbd_convolve8_vert_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h, int bps);
#define av1_highbd_convolve8_vert av1_highbd_convolve8_vert_c

void av1_highbd_convolve_2d_c(const uint16_t *src, int src_stride, CONV_BUF_TYPE *dst, int dst_stride, int w, int h, InterpFilterParams *filter_params_x, InterpFilterParams *filter_params_y, const int subpel_x_q4, const int subpel_y_q4, ConvolveParams *conv_params, int bd);
#define av1_highbd_convolve_2d av1_highbd_convolve_2d_c

void av1_highbd_convolve_2d_scale_c(const uint16_t *src, int src_stride, CONV_BUF_TYPE *dst, int dst_stride, int w, int h, InterpFilterParams *filter_params_x, InterpFilterParams *filter_params_y, const int subpel_x_q4, const int x_step_qn, const int subpel_y_q4, const int y_step_qn, ConvolveParams *conv_params, int bd);
#define av1_highbd_convolve_2d_scale av1_highbd_convolve_2d_scale_c

void av1_highbd_convolve_avg_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h, int bps);
#define av1_highbd_convolve_avg av1_highbd_convolve_avg_c

void av1_highbd_convolve_copy_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h, int bps);
#define av1_highbd_convolve_copy av1_highbd_convolve_copy_c

void av1_highbd_convolve_horiz_c(const uint16_t *src, int src_stride, uint16_t *dst, int dst_stride, int w, int h, const InterpFilterParams fp, const int subpel_x_q4, int x_step_q4, int avg, int bd);
#define av1_highbd_convolve_horiz av1_highbd_convolve_horiz_c

void av1_highbd_convolve_horiz_rs_c(const uint16_t *src, int src_stride, uint16_t *dst, int dst_stride, int w, int h, const int16_t *x_filters, const int x0_qn, const int x_step_qn, int bd);
#define av1_highbd_convolve_horiz_rs av1_highbd_convolve_horiz_rs_c

void av1_highbd_convolve_rounding_c(const int32_t *src, int src_stride, uint8_t *dst, int dst_stride, int w, int h, int bits, int bd);
#define av1_highbd_convolve_rounding av1_highbd_convolve_rounding_c

void av1_highbd_convolve_vert_c(const uint16_t *src, int src_stride, uint16_t *dst, int dst_stride, int w, int h, const InterpFilterParams fp, const int subpel_x_q4, int x_step_q4, int avg, int bd);
#define av1_highbd_convolve_vert av1_highbd_convolve_vert_c

void av1_highbd_iht16x16_256_add_c(const tran_low_t *input, uint8_t *output, int pitch, const struct txfm_param *param);
#define av1_highbd_iht16x16_256_add av1_highbd_iht16x16_256_add_c

void av1_highbd_iht16x32_512_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_highbd_iht16x32_512_add av1_highbd_iht16x32_512_add_c

void av1_highbd_iht16x4_64_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_highbd_iht16x4_64_add av1_highbd_iht16x4_64_add_c

void av1_highbd_iht16x8_128_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_highbd_iht16x8_128_add av1_highbd_iht16x8_128_add_c

void av1_highbd_iht32x16_512_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_highbd_iht32x16_512_add av1_highbd_iht32x16_512_add_c

void av1_highbd_iht32x8_256_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_highbd_iht32x8_256_add av1_highbd_iht32x8_256_add_c

void av1_highbd_iht4x16_64_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_highbd_iht4x16_64_add av1_highbd_iht4x16_64_add_c

void av1_highbd_iht4x4_16_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_highbd_iht4x4_16_add av1_highbd_iht4x4_16_add_c

void av1_highbd_iht4x8_32_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_highbd_iht4x8_32_add av1_highbd_iht4x8_32_add_c

void av1_highbd_iht8x16_128_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_highbd_iht8x16_128_add av1_highbd_iht8x16_128_add_c

void av1_highbd_iht8x32_256_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_highbd_iht8x32_256_add av1_highbd_iht8x32_256_add_c

void av1_highbd_iht8x4_32_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_highbd_iht8x4_32_add av1_highbd_iht8x4_32_add_c

void av1_highbd_iht8x8_64_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_highbd_iht8x8_64_add av1_highbd_iht8x8_64_add_c

void av1_highbd_warp_affine_c(const int32_t *mat, const uint16_t *ref, int width, int height, int stride, uint16_t *pred, int p_col, int p_row, int p_width, int p_height, int p_stride, int subsampling_x, int subsampling_y, int bd, ConvolveParams *conv_params, int16_t alpha, int16_t beta, int16_t gamma, int16_t delta);
#define av1_highbd_warp_affine av1_highbd_warp_affine_c

void av1_iht16x16_256_add_c(const tran_low_t *input, uint8_t *output, int pitch, const struct txfm_param *param);
#define av1_iht16x16_256_add av1_iht16x16_256_add_c

void av1_iht16x32_512_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_iht16x32_512_add av1_iht16x32_512_add_c

void av1_iht16x4_64_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_iht16x4_64_add av1_iht16x4_64_add_c

void av1_iht16x64_1024_add_c(const tran_low_t *input, uint8_t *output, int pitch, const struct txfm_param *param);
#define av1_iht16x64_1024_add av1_iht16x64_1024_add_c

void av1_iht16x8_128_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_iht16x8_128_add av1_iht16x8_128_add_c

void av1_iht32x16_512_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_iht32x16_512_add av1_iht32x16_512_add_c

void av1_iht32x32_1024_add_c(const tran_low_t *input, uint8_t *output, int pitch, const struct txfm_param *param);
#define av1_iht32x32_1024_add av1_iht32x32_1024_add_c

void av1_iht32x64_2048_add_c(const tran_low_t *input, uint8_t *output, int pitch, const struct txfm_param *param);
#define av1_iht32x64_2048_add av1_iht32x64_2048_add_c

void av1_iht32x8_256_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_iht32x8_256_add av1_iht32x8_256_add_c

void av1_iht4x16_64_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_iht4x16_64_add av1_iht4x16_64_add_c

void av1_iht4x4_16_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_iht4x4_16_add av1_iht4x4_16_add_c

void av1_iht4x8_32_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_iht4x8_32_add av1_iht4x8_32_add_c

void av1_iht64x16_1024_add_c(const tran_low_t *input, uint8_t *output, int pitch, const struct txfm_param *param);
#define av1_iht64x16_1024_add av1_iht64x16_1024_add_c

void av1_iht64x32_2048_add_c(const tran_low_t *input, uint8_t *output, int pitch, const struct txfm_param *param);
#define av1_iht64x32_2048_add av1_iht64x32_2048_add_c

void av1_iht64x64_4096_add_c(const tran_low_t *input, uint8_t *output, int pitch, const struct txfm_param *param);
#define av1_iht64x64_4096_add av1_iht64x64_4096_add_c

void av1_iht8x16_128_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_iht8x16_128_add av1_iht8x16_128_add_c

void av1_iht8x32_256_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_iht8x32_256_add av1_iht8x32_256_add_c

void av1_iht8x4_32_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_iht8x4_32_add av1_iht8x4_32_add_c

void av1_iht8x8_64_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride, const struct txfm_param *param);
#define av1_iht8x8_64_add av1_iht8x8_64_add_c

void av1_inv_txfm2d_add_16x16_c(const int32_t *input, uint16_t *output, int stride, TX_TYPE tx_type, int bd);
#define av1_inv_txfm2d_add_16x16 av1_inv_txfm2d_add_16x16_c

void av1_inv_txfm2d_add_16x32_c(const int32_t *input, uint16_t *output, int stride, TX_TYPE tx_type, int bd);
#define av1_inv_txfm2d_add_16x32 av1_inv_txfm2d_add_16x32_c

void av1_inv_txfm2d_add_16x4_c(const int32_t *input, uint16_t *output, int stride, TX_TYPE tx_type, int bd);
#define av1_inv_txfm2d_add_16x4 av1_inv_txfm2d_add_16x4_c

void av1_inv_txfm2d_add_16x64_c(const int32_t *input, uint16_t *output, int stride, TX_TYPE tx_type, int bd);
#define av1_inv_txfm2d_add_16x64 av1_inv_txfm2d_add_16x64_c

void av1_inv_txfm2d_add_16x8_c(const int32_t *input, uint16_t *output, int stride, TX_TYPE tx_type, int bd);
#define av1_inv_txfm2d_add_16x8 av1_inv_txfm2d_add_16x8_c

void av1_inv_txfm2d_add_32x16_c(const int32_t *input, uint16_t *output, int stride, TX_TYPE tx_type, int bd);
#define av1_inv_txfm2d_add_32x16 av1_inv_txfm2d_add_32x16_c

void av1_inv_txfm2d_add_32x32_c(const int32_t *input, uint16_t *output, int stride, TX_TYPE tx_type, int bd);
#define av1_inv_txfm2d_add_32x32 av1_inv_txfm2d_add_32x32_c

void av1_inv_txfm2d_add_32x64_c(const int32_t *input, uint16_t *output, int stride, TX_TYPE tx_type, int bd);
#define av1_inv_txfm2d_add_32x64 av1_inv_txfm2d_add_32x64_c

void av1_inv_txfm2d_add_32x8_c(const int32_t *input, uint16_t *output, int stride, TX_TYPE tx_type, int bd);
#define av1_inv_txfm2d_add_32x8 av1_inv_txfm2d_add_32x8_c

void av1_inv_txfm2d_add_4x16_c(const int32_t *input, uint16_t *output, int stride, TX_TYPE tx_type, int bd);
#define av1_inv_txfm2d_add_4x16 av1_inv_txfm2d_add_4x16_c

void av1_inv_txfm2d_add_4x4_c(const int32_t *input, uint16_t *output, int stride, TX_TYPE tx_type, int bd);
#define av1_inv_txfm2d_add_4x4 av1_inv_txfm2d_add_4x4_c

void av1_inv_txfm2d_add_4x8_c(const int32_t *input, uint16_t *output, int stride, TX_TYPE tx_type, int bd);
#define av1_inv_txfm2d_add_4x8 av1_inv_txfm2d_add_4x8_c

void av1_inv_txfm2d_add_64x16_c(const int32_t *input, uint16_t *output, int stride, TX_TYPE tx_type, int bd);
#define av1_inv_txfm2d_add_64x16 av1_inv_txfm2d_add_64x16_c

void av1_inv_txfm2d_add_64x32_c(const int32_t *input, uint16_t *output, int stride, TX_TYPE tx_type, int bd);
#define av1_inv_txfm2d_add_64x32 av1_inv_txfm2d_add_64x32_c

void av1_inv_txfm2d_add_64x64_c(const int32_t *input, uint16_t *output, int stride, TX_TYPE tx_type, int bd);
#define av1_inv_txfm2d_add_64x64 av1_inv_txfm2d_add_64x64_c

void av1_inv_txfm2d_add_8x16_c(const int32_t *input, uint16_t *output, int stride, TX_TYPE tx_type, int bd);
#define av1_inv_txfm2d_add_8x16 av1_inv_txfm2d_add_8x16_c

void av1_inv_txfm2d_add_8x32_c(const int32_t *input, uint16_t *output, int stride, TX_TYPE tx_type, int bd);
#define av1_inv_txfm2d_add_8x32 av1_inv_txfm2d_add_8x32_c

void av1_inv_txfm2d_add_8x4_c(const int32_t *input, uint16_t *output, int stride, TX_TYPE tx_type, int bd);
#define av1_inv_txfm2d_add_8x4 av1_inv_txfm2d_add_8x4_c

void av1_inv_txfm2d_add_8x8_c(const int32_t *input, uint16_t *output, int stride, TX_TYPE tx_type, int bd);
#define av1_inv_txfm2d_add_8x8 av1_inv_txfm2d_add_8x8_c

void av1_inv_txfm_add_c(const tran_low_t *dqcoeff, uint8_t *dst, int stride, const TxfmParam *txfm_param);
#define av1_inv_txfm_add av1_inv_txfm_add_c

void av1_selfguided_restoration_c(const uint8_t *dgd, int width, int height, int stride, int32_t *flt1, int32_t *flt2, int flt_stride, const sgr_params_type *params, int bit_depth, int highbd);
#define av1_selfguided_restoration av1_selfguided_restoration_c

void av1_upsample_intra_edge_c(uint8_t *p, int sz);
#define av1_upsample_intra_edge av1_upsample_intra_edge_c

void av1_upsample_intra_edge_high_c(uint16_t *p, int sz, int bd);
#define av1_upsample_intra_edge_high av1_upsample_intra_edge_high_c

void av1_warp_affine_c(const int32_t *mat, const uint8_t *ref, int width, int height, int stride, uint8_t *pred, int p_col, int p_row, int p_width, int p_height, int p_stride, int subsampling_x, int subsampling_y, ConvolveParams *conv_params, int16_t alpha, int16_t beta, int16_t gamma, int16_t delta);
#define av1_warp_affine av1_warp_affine_c

void cdef_filter_block_c(uint8_t *dst8, uint16_t *dst16, int dstride, const uint16_t *in, int pri_strength, int sec_strength, int dir, int pri_damping, int sec_damping, int bsize, int max, int coeff_shift);
#define cdef_filter_block cdef_filter_block_c

int cdef_find_dir_c(const uint16_t *img, int stride, int32_t *var, int coeff_shift);
#define cdef_find_dir cdef_find_dir_c

void copy_rect8_16bit_to_16bit_c(uint16_t *dst, int dstride, const uint16_t *src, int sstride, int v, int h);
#define copy_rect8_16bit_to_16bit copy_rect8_16bit_to_16bit_c

void copy_rect8_8bit_to_16bit_c(uint16_t *dst, int dstride, const uint8_t *src, int sstride, int v, int h);
#define copy_rect8_8bit_to_16bit copy_rect8_8bit_to_16bit_c

cfl_predict_hbd_fn get_predict_hbd_fn_c(TX_SIZE tx_size);
#define get_predict_hbd_fn get_predict_hbd_fn_c

cfl_predict_lbd_fn get_predict_lbd_fn_c(TX_SIZE tx_size);
#define get_predict_lbd_fn get_predict_lbd_fn_c

cfl_subsample_hbd_fn get_subsample_hbd_fn_c(int sub_x, int sub_y);
#define get_subsample_hbd_fn get_subsample_hbd_fn_c

cfl_subsample_lbd_fn get_subsample_lbd_fn_c(int sub_x, int sub_y);
#define get_subsample_lbd_fn get_subsample_lbd_fn_c

cfl_subtract_average_fn get_subtract_average_fn_c(TX_SIZE tx_size);
#define get_subtract_average_fn get_subtract_average_fn_c

void av1_rtcd(void);

#include "aom_config.h"

#ifdef RTCD_C
static void setup_rtcd_internal(void)
{
}
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
