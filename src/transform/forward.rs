use super::*;

use std::cmp;

use partition::TxSize;
use partition::TxType;

const MAX_TXFM_STAGE_NUM: usize = 12;
const MAX_TXWH_IDX: usize = 5;

type TxfmShift = [i8; 3];

const FWD_SHIFT_4X4: TxfmShift = [2, 0, 0];
const FWD_SHIFT_8X8: TxfmShift = [2, -1, 0];
const FWD_SHIFT_16X16: TxfmShift = [2, -2, 0];
const FWD_SHIFT_32X32: TxfmShift = [2, -4, 0];
const FWD_SHIFT_64X64: TxfmShift = [0, -2, -2];
const FWD_SHIFT_4X8: TxfmShift = [2, -1, 0];
const FWD_SHIFT_8X4: TxfmShift = [2, -1, 0];
const FWD_SHIFT_8X16: TxfmShift = [2, -2, 0];
const FWD_SHIFT_16X8: TxfmShift = [2, -2, 0];
const FWD_SHIFT_16X32: TxfmShift = [2, -4, 0];
const FWD_SHIFT_32X16: TxfmShift = [2, -4, 0];
const FWD_SHIFT_32X64: TxfmShift = [0, -2, -2];
const FWD_SHIFT_64X32: TxfmShift = [2, -4, -2];
const FWD_SHIFT_4X16: TxfmShift = [2, -1, 0];
const FWD_SHIFT_16X4: TxfmShift = [2, -1, 0];
const FWD_SHIFT_8X32: TxfmShift = [2, -2, 0];
const FWD_SHIFT_32X8: TxfmShift = [2, -2, 0];
const FWD_SHIFT_16X64: TxfmShift = [0, -2, 0];
const FWD_SHIFT_64X16: TxfmShift = [2, -4, 0];

const FWD_TXFM_SHIFT_LS: [TxfmShift; TxSize::TX_SIZES_ALL] = [
  FWD_SHIFT_4X4,
  FWD_SHIFT_8X8,
  FWD_SHIFT_16X16,
  FWD_SHIFT_32X32,
  FWD_SHIFT_64X64,
  FWD_SHIFT_4X8,
  FWD_SHIFT_8X4,
  FWD_SHIFT_8X16,
  FWD_SHIFT_16X8,
  FWD_SHIFT_16X32,
  FWD_SHIFT_32X16,
  FWD_SHIFT_32X64,
  FWD_SHIFT_64X32,
  FWD_SHIFT_4X16,
  FWD_SHIFT_16X4,
  FWD_SHIFT_8X32,
  FWD_SHIFT_32X8,
  FWD_SHIFT_16X64,
  FWD_SHIFT_64X16
];

const FWD_COS_BIT_COL: [[i8; MAX_TXWH_IDX]; MAX_TXWH_IDX] = [
  [13, 13, 13, 0, 0],
  [13, 13, 13, 12, 0],
  [13, 13, 13, 12, 13],
  [0, 13, 13, 12, 13],
  [0, 0, 13, 12, 13]
];

const FWD_COS_BIT_ROW: [[i8; MAX_TXWH_IDX]; MAX_TXWH_IDX] = [
  [13, 13, 12, 0, 0],
  [13, 13, 13, 12, 0],
  [13, 13, 12, 13, 12],
  [0, 12, 13, 12, 11],
  [0, 0, 12, 11, 10]
];

const FDCT4_RANGE_MULT2: [i8; 4] = [0, 2, 3, 3];
const FDCT8_RANGE_MULT2: [i8; 6] = [0, 2, 4, 5, 5, 5];
const FDCT16_RANGE_MULT2: [i8; 8] = [0, 2, 4, 6, 7, 7, 7, 7];
const FDCT32_RANGE_MULT2: [i8; 10] = [0, 2, 4, 6, 8, 9, 9, 9, 9, 9];
const FDCT64_RANGE_MULT2: [i8; 12] =
  [0, 2, 4, 6, 8, 10, 11, 11, 11, 11, 11, 11];

const FADST4_RANGE_MULT2: [i8; 7] = [0, 2, 4, 3, 3, 3, 3];
const FADST8_RANGE_MULT2: [i8; 8] = [0, 0, 1, 3, 3, 5, 5, 5];
const FADST16_RANGE_MULT2: [i8; 10] = [0, 0, 1, 3, 3, 5, 5, 7, 7, 7];

const MAX_FWD_RANGE_MULT2_COL: [i8; 5] = [3, 5, 7, 9, 11];

const FIDTX4_RANGE_MULT2: [i8; 1] = [1];
const FIDTX8_RANGE_MULT2: [i8; 1] = [2];
const FIDTX16_RANGE_MULT2: [i8; 1] = [3];
const FIDTX32_RANGE_MULT2: [i8; 1] = [4];

const FWD_TXFM_RANGE_MULT2_LIST: [&[i8]; TxfmType::TXFM_TYPES] = [
  &FDCT4_RANGE_MULT2,
  &FDCT8_RANGE_MULT2,
  &FDCT16_RANGE_MULT2,
  &FDCT32_RANGE_MULT2,
  &FDCT64_RANGE_MULT2,
  &FADST4_RANGE_MULT2,
  &FADST8_RANGE_MULT2,
  &FADST16_RANGE_MULT2,
  &FIDTX4_RANGE_MULT2,
  &FIDTX8_RANGE_MULT2,
  &FIDTX16_RANGE_MULT2,
  &FIDTX32_RANGE_MULT2
];

const AV1_TXFM_STAGE_NUM_LIST: [i8; TxfmType::TXFM_TYPES] = [
  4,  // TXFM_TYPE_DCT4
  6,  // TXFM_TYPE_DCT8
  8,  // TXFM_TYPE_DCT16
  10, // TXFM_TYPE_DCT32
  12, // TXFM_TYPE_DCT64
  7,  // TXFM_TYPE_ADST4
  8,  // TXFM_TYPE_ADST8
  10, // TXFM_TYPE_ADST16
  1,  // TXFM_TYPE_IDENTITY4
  1,  // TXFM_TYPE_IDENTITY8
  1,  // TXFM_TYPE_IDENTITY16
  1,  // TXFM_TYPE_IDENTITY32
];

fn av1_gen_fwd_stage_range(
  stage_range_col: &mut [i8], stage_range_row: &mut [i8], cfg: &Txfm2DFlipCfg,
  bd: i8
) {
  for i in 0..cmp::min(cfg.stage_num_col as usize, MAX_TXFM_STAGE_NUM) {
    stage_range_col[i] = cfg.stage_range_col[i] + cfg.shift[0] + bd + 1;
  }
  for i in 0..cmp::min(cfg.stage_num_row as usize, MAX_TXFM_STAGE_NUM) {
    stage_range_row[i] =
      cfg.stage_range_row[i] + cfg.shift[0] + cfg.shift[1] + bd + 1;
  }
}

type TxfmFunc = Fn(&[i32], &mut [i32], usize, &[i8]);

fn av1_fdct4_new(
  input: &[i32], output: &mut [i32], cos_bit: usize, _stage_range: &[i8]
) {
  let mut step = [0i32; 4];
  let cospi = cospi_arr(cos_bit);

  // stage 1
  output[0] = input[0] + input[3];
  output[1] = input[1] + input[2];
  output[2] = -input[2] + input[1];
  output[3] = -input[3] + input[0];

  // stage 2
  step[0] = half_btf(cospi[32], output[0], cospi[32], output[1], cos_bit);
  step[1] = half_btf(-cospi[32], output[1], cospi[32], output[0], cos_bit);
  step[2] = half_btf(cospi[48], output[2], cospi[16], output[3], cos_bit);
  step[3] = half_btf(cospi[48], output[3], -cospi[16], output[2], cos_bit);

  // stage 3
  output[0] = step[0];
  output[1] = step[2];
  output[2] = step[1];
  output[3] = step[3];
}

fn av1_fdct8_new(
  input: &[i32], output: &mut [i32], cos_bit: usize, _stage_range: &[i8]
) {
  let mut step = [0i32; 8];
  let cospi = cospi_arr(cos_bit);

  // stage 1;
  output[0] = input[0] + input[7];
  output[1] = input[1] + input[6];
  output[2] = input[2] + input[5];
  output[3] = input[3] + input[4];
  output[4] = -input[4] + input[3];
  output[5] = -input[5] + input[2];
  output[6] = -input[6] + input[1];
  output[7] = -input[7] + input[0];

  // stage 2
  step[0] = output[0] + output[3];
  step[1] = output[1] + output[2];
  step[2] = -output[2] + output[1];
  step[3] = -output[3] + output[0];
  step[4] = output[4];
  step[5] = half_btf(-cospi[32], output[5], cospi[32], output[6], cos_bit);
  step[6] = half_btf(cospi[32], output[6], cospi[32], output[5], cos_bit);
  step[7] = output[7];

  // stage 3
  output[0] = half_btf(cospi[32], step[0], cospi[32], step[1], cos_bit);
  output[1] = half_btf(-cospi[32], step[1], cospi[32], step[0], cos_bit);
  output[2] = half_btf(cospi[48], step[2], cospi[16], step[3], cos_bit);
  output[3] = half_btf(cospi[48], step[3], -cospi[16], step[2], cos_bit);
  output[4] = step[4] + step[5];
  output[5] = -step[5] + step[4];
  output[6] = -step[6] + step[7];
  output[7] = step[7] + step[6];

  // stage 4
  step[0] = output[0];
  step[1] = output[1];
  step[2] = output[2];
  step[3] = output[3];
  step[4] = half_btf(cospi[56], output[4], cospi[8], output[7], cos_bit);
  step[5] = half_btf(cospi[24], output[5], cospi[40], output[6], cos_bit);
  step[6] = half_btf(cospi[24], output[6], -cospi[40], output[5], cos_bit);
  step[7] = half_btf(cospi[56], output[7], -cospi[8], output[4], cos_bit);

  // stage 5
  output[0] = step[0];
  output[1] = step[4];
  output[2] = step[2];
  output[3] = step[6];
  output[4] = step[1];
  output[5] = step[5];
  output[6] = step[3];
  output[7] = step[7];
}

fn av1_fdct16_new(
  input: &[i32], output: &mut [i32], cos_bit: usize, _stage_range: &[i8]
) {
  let mut step = [0i32; 16];
  let cospi = cospi_arr(cos_bit);

  // stage 1;
  output[0] = input[0] + input[15];
  output[1] = input[1] + input[14];
  output[2] = input[2] + input[13];
  output[3] = input[3] + input[12];
  output[4] = input[4] + input[11];
  output[5] = input[5] + input[10];
  output[6] = input[6] + input[9];
  output[7] = input[7] + input[8];
  output[8] = -input[8] + input[7];
  output[9] = -input[9] + input[6];
  output[10] = -input[10] + input[5];
  output[11] = -input[11] + input[4];
  output[12] = -input[12] + input[3];
  output[13] = -input[13] + input[2];
  output[14] = -input[14] + input[1];
  output[15] = -input[15] + input[0];

  // stage 2
  step[0] = output[0] + output[7];
  step[1] = output[1] + output[6];
  step[2] = output[2] + output[5];
  step[3] = output[3] + output[4];
  step[4] = -output[4] + output[3];
  step[5] = -output[5] + output[2];
  step[6] = -output[6] + output[1];
  step[7] = -output[7] + output[0];
  step[8] = output[8];
  step[9] = output[9];
  step[10] = half_btf(-cospi[32], output[10], cospi[32], output[13], cos_bit);
  step[11] = half_btf(-cospi[32], output[11], cospi[32], output[12], cos_bit);
  step[12] = half_btf(cospi[32], output[12], cospi[32], output[11], cos_bit);
  step[13] = half_btf(cospi[32], output[13], cospi[32], output[10], cos_bit);
  step[14] = output[14];
  step[15] = output[15];

  // stage 3
  output[0] = step[0] + step[3];
  output[1] = step[1] + step[2];
  output[2] = -step[2] + step[1];
  output[3] = -step[3] + step[0];
  output[4] = step[4];
  output[5] = half_btf(-cospi[32], step[5], cospi[32], step[6], cos_bit);
  output[6] = half_btf(cospi[32], step[6], cospi[32], step[5], cos_bit);
  output[7] = step[7];
  output[8] = step[8] + step[11];
  output[9] = step[9] + step[10];
  output[10] = -step[10] + step[9];
  output[11] = -step[11] + step[8];
  output[12] = -step[12] + step[15];
  output[13] = -step[13] + step[14];
  output[14] = step[14] + step[13];
  output[15] = step[15] + step[12];

  // stage 4
  step[0] = half_btf(cospi[32], output[0], cospi[32], output[1], cos_bit);
  step[1] = half_btf(-cospi[32], output[1], cospi[32], output[0], cos_bit);
  step[2] = half_btf(cospi[48], output[2], cospi[16], output[3], cos_bit);
  step[3] = half_btf(cospi[48], output[3], -cospi[16], output[2], cos_bit);
  step[4] = output[4] + output[5];
  step[5] = -output[5] + output[4];
  step[6] = -output[6] + output[7];
  step[7] = output[7] + output[6];
  step[8] = output[8];
  step[9] = half_btf(-cospi[16], output[9], cospi[48], output[14], cos_bit);
  step[10] = half_btf(-cospi[48], output[10], -cospi[16], output[13], cos_bit);
  step[11] = output[11];
  step[12] = output[12];
  step[13] = half_btf(cospi[48], output[13], -cospi[16], output[10], cos_bit);
  step[14] = half_btf(cospi[16], output[14], cospi[48], output[9], cos_bit);
  step[15] = output[15];

  // stage 5
  output[0] = step[0];
  output[1] = step[1];
  output[2] = step[2];
  output[3] = step[3];
  output[4] = half_btf(cospi[56], step[4], cospi[8], step[7], cos_bit);
  output[5] = half_btf(cospi[24], step[5], cospi[40], step[6], cos_bit);
  output[6] = half_btf(cospi[24], step[6], -cospi[40], step[5], cos_bit);
  output[7] = half_btf(cospi[56], step[7], -cospi[8], step[4], cos_bit);
  output[8] = step[8] + step[9];
  output[9] = -step[9] + step[8];
  output[10] = -step[10] + step[11];
  output[11] = step[11] + step[10];
  output[12] = step[12] + step[13];
  output[13] = -step[13] + step[12];
  output[14] = -step[14] + step[15];
  output[15] = step[15] + step[14];

  // stage 6
  step[0] = output[0];
  step[1] = output[1];
  step[2] = output[2];
  step[3] = output[3];
  step[4] = output[4];
  step[5] = output[5];
  step[6] = output[6];
  step[7] = output[7];
  step[8] = half_btf(cospi[60], output[8], cospi[4], output[15], cos_bit);
  step[9] = half_btf(cospi[28], output[9], cospi[36], output[14], cos_bit);
  step[10] = half_btf(cospi[44], output[10], cospi[20], output[13], cos_bit);
  step[11] = half_btf(cospi[12], output[11], cospi[52], output[12], cos_bit);
  step[12] = half_btf(cospi[12], output[12], -cospi[52], output[11], cos_bit);
  step[13] = half_btf(cospi[44], output[13], -cospi[20], output[10], cos_bit);
  step[14] = half_btf(cospi[28], output[14], -cospi[36], output[9], cos_bit);
  step[15] = half_btf(cospi[60], output[15], -cospi[4], output[8], cos_bit);

  // stage 7
  output[0] = step[0];
  output[1] = step[8];
  output[2] = step[4];
  output[3] = step[12];
  output[4] = step[2];
  output[5] = step[10];
  output[6] = step[6];
  output[7] = step[14];
  output[8] = step[1];
  output[9] = step[9];
  output[10] = step[5];
  output[11] = step[13];
  output[12] = step[3];
  output[13] = step[11];
  output[14] = step[7];
  output[15] = step[15];
}

fn av1_fdct32_new(
  input: &[i32], output: &mut [i32], cos_bit: usize, _stage_range: &[i8]
) {
  let mut step = [0i32; 32];
  let cospi = cospi_arr(cos_bit);

  // stage 1;
  output[0] = input[0] + input[31];
  output[1] = input[1] + input[30];
  output[2] = input[2] + input[29];
  output[3] = input[3] + input[28];
  output[4] = input[4] + input[27];
  output[5] = input[5] + input[26];
  output[6] = input[6] + input[25];
  output[7] = input[7] + input[24];
  output[8] = input[8] + input[23];
  output[9] = input[9] + input[22];
  output[10] = input[10] + input[21];
  output[11] = input[11] + input[20];
  output[12] = input[12] + input[19];
  output[13] = input[13] + input[18];
  output[14] = input[14] + input[17];
  output[15] = input[15] + input[16];
  output[16] = -input[16] + input[15];
  output[17] = -input[17] + input[14];
  output[18] = -input[18] + input[13];
  output[19] = -input[19] + input[12];
  output[20] = -input[20] + input[11];
  output[21] = -input[21] + input[10];
  output[22] = -input[22] + input[9];
  output[23] = -input[23] + input[8];
  output[24] = -input[24] + input[7];
  output[25] = -input[25] + input[6];
  output[26] = -input[26] + input[5];
  output[27] = -input[27] + input[4];
  output[28] = -input[28] + input[3];
  output[29] = -input[29] + input[2];
  output[30] = -input[30] + input[1];
  output[31] = -input[31] + input[0];

  // stage 2
  step[0] = output[0] + output[15];
  step[1] = output[1] + output[14];
  step[2] = output[2] + output[13];
  step[3] = output[3] + output[12];
  step[4] = output[4] + output[11];
  step[5] = output[5] + output[10];
  step[6] = output[6] + output[9];
  step[7] = output[7] + output[8];
  step[8] = -output[8] + output[7];
  step[9] = -output[9] + output[6];
  step[10] = -output[10] + output[5];
  step[11] = -output[11] + output[4];
  step[12] = -output[12] + output[3];
  step[13] = -output[13] + output[2];
  step[14] = -output[14] + output[1];
  step[15] = -output[15] + output[0];
  step[16] = output[16];
  step[17] = output[17];
  step[18] = output[18];
  step[19] = output[19];
  step[20] = half_btf(-cospi[32], output[20], cospi[32], output[27], cos_bit);
  step[21] = half_btf(-cospi[32], output[21], cospi[32], output[26], cos_bit);
  step[22] = half_btf(-cospi[32], output[22], cospi[32], output[25], cos_bit);
  step[23] = half_btf(-cospi[32], output[23], cospi[32], output[24], cos_bit);
  step[24] = half_btf(cospi[32], output[24], cospi[32], output[23], cos_bit);
  step[25] = half_btf(cospi[32], output[25], cospi[32], output[22], cos_bit);
  step[26] = half_btf(cospi[32], output[26], cospi[32], output[21], cos_bit);
  step[27] = half_btf(cospi[32], output[27], cospi[32], output[20], cos_bit);
  step[28] = output[28];
  step[29] = output[29];
  step[30] = output[30];
  step[31] = output[31];

  // stage 3
  output[0] = step[0] + step[7];
  output[1] = step[1] + step[6];
  output[2] = step[2] + step[5];
  output[3] = step[3] + step[4];
  output[4] = -step[4] + step[3];
  output[5] = -step[5] + step[2];
  output[6] = -step[6] + step[1];
  output[7] = -step[7] + step[0];
  output[8] = step[8];
  output[9] = step[9];
  output[10] = half_btf(-cospi[32], step[10], cospi[32], step[13], cos_bit);
  output[11] = half_btf(-cospi[32], step[11], cospi[32], step[12], cos_bit);
  output[12] = half_btf(cospi[32], step[12], cospi[32], step[11], cos_bit);
  output[13] = half_btf(cospi[32], step[13], cospi[32], step[10], cos_bit);
  output[14] = step[14];
  output[15] = step[15];
  output[16] = step[16] + step[23];
  output[17] = step[17] + step[22];
  output[18] = step[18] + step[21];
  output[19] = step[19] + step[20];
  output[20] = -step[20] + step[19];
  output[21] = -step[21] + step[18];
  output[22] = -step[22] + step[17];
  output[23] = -step[23] + step[16];
  output[24] = -step[24] + step[31];
  output[25] = -step[25] + step[30];
  output[26] = -step[26] + step[29];
  output[27] = -step[27] + step[28];
  output[28] = step[28] + step[27];
  output[29] = step[29] + step[26];
  output[30] = step[30] + step[25];
  output[31] = step[31] + step[24];

  // stage 4
  step[0] = output[0] + output[3];
  step[1] = output[1] + output[2];
  step[2] = -output[2] + output[1];
  step[3] = -output[3] + output[0];
  step[4] = output[4];
  step[5] = half_btf(-cospi[32], output[5], cospi[32], output[6], cos_bit);
  step[6] = half_btf(cospi[32], output[6], cospi[32], output[5], cos_bit);
  step[7] = output[7];
  step[8] = output[8] + output[11];
  step[9] = output[9] + output[10];
  step[10] = -output[10] + output[9];
  step[11] = -output[11] + output[8];
  step[12] = -output[12] + output[15];
  step[13] = -output[13] + output[14];
  step[14] = output[14] + output[13];
  step[15] = output[15] + output[12];
  step[16] = output[16];
  step[17] = output[17];
  step[18] = half_btf(-cospi[16], output[18], cospi[48], output[29], cos_bit);
  step[19] = half_btf(-cospi[16], output[19], cospi[48], output[28], cos_bit);
  step[20] = half_btf(-cospi[48], output[20], -cospi[16], output[27], cos_bit);
  step[21] = half_btf(-cospi[48], output[21], -cospi[16], output[26], cos_bit);
  step[22] = output[22];
  step[23] = output[23];
  step[24] = output[24];
  step[25] = output[25];
  step[26] = half_btf(cospi[48], output[26], -cospi[16], output[21], cos_bit);
  step[27] = half_btf(cospi[48], output[27], -cospi[16], output[20], cos_bit);
  step[28] = half_btf(cospi[16], output[28], cospi[48], output[19], cos_bit);
  step[29] = half_btf(cospi[16], output[29], cospi[48], output[18], cos_bit);
  step[30] = output[30];
  step[31] = output[31];

  // stage 5
  output[0] = half_btf(cospi[32], step[0], cospi[32], step[1], cos_bit);
  output[1] = half_btf(-cospi[32], step[1], cospi[32], step[0], cos_bit);
  output[2] = half_btf(cospi[48], step[2], cospi[16], step[3], cos_bit);
  output[3] = half_btf(cospi[48], step[3], -cospi[16], step[2], cos_bit);
  output[4] = step[4] + step[5];
  output[5] = -step[5] + step[4];
  output[6] = -step[6] + step[7];
  output[7] = step[7] + step[6];
  output[8] = step[8];
  output[9] = half_btf(-cospi[16], step[9], cospi[48], step[14], cos_bit);
  output[10] = half_btf(-cospi[48], step[10], -cospi[16], step[13], cos_bit);
  output[11] = step[11];
  output[12] = step[12];
  output[13] = half_btf(cospi[48], step[13], -cospi[16], step[10], cos_bit);
  output[14] = half_btf(cospi[16], step[14], cospi[48], step[9], cos_bit);
  output[15] = step[15];
  output[16] = step[16] + step[19];
  output[17] = step[17] + step[18];
  output[18] = -step[18] + step[17];
  output[19] = -step[19] + step[16];
  output[20] = -step[20] + step[23];
  output[21] = -step[21] + step[22];
  output[22] = step[22] + step[21];
  output[23] = step[23] + step[20];
  output[24] = step[24] + step[27];
  output[25] = step[25] + step[26];
  output[26] = -step[26] + step[25];
  output[27] = -step[27] + step[24];
  output[28] = -step[28] + step[31];
  output[29] = -step[29] + step[30];
  output[30] = step[30] + step[29];
  output[31] = step[31] + step[28];

  // stage 6
  step[0] = output[0];
  step[1] = output[1];
  step[2] = output[2];
  step[3] = output[3];
  step[4] = half_btf(cospi[56], output[4], cospi[8], output[7], cos_bit);
  step[5] = half_btf(cospi[24], output[5], cospi[40], output[6], cos_bit);
  step[6] = half_btf(cospi[24], output[6], -cospi[40], output[5], cos_bit);
  step[7] = half_btf(cospi[56], output[7], -cospi[8], output[4], cos_bit);
  step[8] = output[8] + output[9];
  step[9] = -output[9] + output[8];
  step[10] = -output[10] + output[11];
  step[11] = output[11] + output[10];
  step[12] = output[12] + output[13];
  step[13] = -output[13] + output[12];
  step[14] = -output[14] + output[15];
  step[15] = output[15] + output[14];
  step[16] = output[16];
  step[17] = half_btf(-cospi[8], output[17], cospi[56], output[30], cos_bit);
  step[18] = half_btf(-cospi[56], output[18], -cospi[8], output[29], cos_bit);
  step[19] = output[19];
  step[20] = output[20];
  step[21] = half_btf(-cospi[40], output[21], cospi[24], output[26], cos_bit);
  step[22] = half_btf(-cospi[24], output[22], -cospi[40], output[25], cos_bit);
  step[23] = output[23];
  step[24] = output[24];
  step[25] = half_btf(cospi[24], output[25], -cospi[40], output[22], cos_bit);
  step[26] = half_btf(cospi[40], output[26], cospi[24], output[21], cos_bit);
  step[27] = output[27];
  step[28] = output[28];
  step[29] = half_btf(cospi[56], output[29], -cospi[8], output[18], cos_bit);
  step[30] = half_btf(cospi[8], output[30], cospi[56], output[17], cos_bit);
  step[31] = output[31];

  // stage 7
  output[0] = step[0];
  output[1] = step[1];
  output[2] = step[2];
  output[3] = step[3];
  output[4] = step[4];
  output[5] = step[5];
  output[6] = step[6];
  output[7] = step[7];
  output[8] = half_btf(cospi[60], step[8], cospi[4], step[15], cos_bit);
  output[9] = half_btf(cospi[28], step[9], cospi[36], step[14], cos_bit);
  output[10] = half_btf(cospi[44], step[10], cospi[20], step[13], cos_bit);
  output[11] = half_btf(cospi[12], step[11], cospi[52], step[12], cos_bit);
  output[12] = half_btf(cospi[12], step[12], -cospi[52], step[11], cos_bit);
  output[13] = half_btf(cospi[44], step[13], -cospi[20], step[10], cos_bit);
  output[14] = half_btf(cospi[28], step[14], -cospi[36], step[9], cos_bit);
  output[15] = half_btf(cospi[60], step[15], -cospi[4], step[8], cos_bit);
  output[16] = step[16] + step[17];
  output[17] = -step[17] + step[16];
  output[18] = -step[18] + step[19];
  output[19] = step[19] + step[18];
  output[20] = step[20] + step[21];
  output[21] = -step[21] + step[20];
  output[22] = -step[22] + step[23];
  output[23] = step[23] + step[22];
  output[24] = step[24] + step[25];
  output[25] = -step[25] + step[24];
  output[26] = -step[26] + step[27];
  output[27] = step[27] + step[26];
  output[28] = step[28] + step[29];
  output[29] = -step[29] + step[28];
  output[30] = -step[30] + step[31];
  output[31] = step[31] + step[30];

  // stage 8
  step[0] = output[0];
  step[1] = output[1];
  step[2] = output[2];
  step[3] = output[3];
  step[4] = output[4];
  step[5] = output[5];
  step[6] = output[6];
  step[7] = output[7];
  step[8] = output[8];
  step[9] = output[9];
  step[10] = output[10];
  step[11] = output[11];
  step[12] = output[12];
  step[13] = output[13];
  step[14] = output[14];
  step[15] = output[15];
  step[16] = half_btf(cospi[62], output[16], cospi[2], output[31], cos_bit);
  step[17] = half_btf(cospi[30], output[17], cospi[34], output[30], cos_bit);
  step[18] = half_btf(cospi[46], output[18], cospi[18], output[29], cos_bit);
  step[19] = half_btf(cospi[14], output[19], cospi[50], output[28], cos_bit);
  step[20] = half_btf(cospi[54], output[20], cospi[10], output[27], cos_bit);
  step[21] = half_btf(cospi[22], output[21], cospi[42], output[26], cos_bit);
  step[22] = half_btf(cospi[38], output[22], cospi[26], output[25], cos_bit);
  step[23] = half_btf(cospi[6], output[23], cospi[58], output[24], cos_bit);
  step[24] = half_btf(cospi[6], output[24], -cospi[58], output[23], cos_bit);
  step[25] = half_btf(cospi[38], output[25], -cospi[26], output[22], cos_bit);
  step[26] = half_btf(cospi[22], output[26], -cospi[42], output[21], cos_bit);
  step[27] = half_btf(cospi[54], output[27], -cospi[10], output[20], cos_bit);
  step[28] = half_btf(cospi[14], output[28], -cospi[50], output[19], cos_bit);
  step[29] = half_btf(cospi[46], output[29], -cospi[18], output[18], cos_bit);
  step[30] = half_btf(cospi[30], output[30], -cospi[34], output[17], cos_bit);
  step[31] = half_btf(cospi[62], output[31], -cospi[2], output[16], cos_bit);

  // stage 9
  output[0] = step[0];
  output[1] = step[16];
  output[2] = step[8];
  output[3] = step[24];
  output[4] = step[4];
  output[5] = step[20];
  output[6] = step[12];
  output[7] = step[28];
  output[8] = step[2];
  output[9] = step[18];
  output[10] = step[10];
  output[11] = step[26];
  output[12] = step[6];
  output[13] = step[22];
  output[14] = step[14];
  output[15] = step[30];
  output[16] = step[1];
  output[17] = step[17];
  output[18] = step[9];
  output[19] = step[25];
  output[20] = step[5];
  output[21] = step[21];
  output[22] = step[13];
  output[23] = step[29];
  output[24] = step[3];
  output[25] = step[19];
  output[26] = step[11];
  output[27] = step[27];
  output[28] = step[7];
  output[29] = step[23];
  output[30] = step[15];
  output[31] = step[31];
}

fn av1_fdct64_new(
  input: &[i32], output: &mut [i32], cos_bit: usize, _stage_range: &[i8]
) {
  let mut step = [0i32; 64];
  let cospi = cospi_arr(cos_bit);

  // stage 1;
  output[0] = input[0] + input[63];
  output[1] = input[1] + input[62];
  output[2] = input[2] + input[61];
  output[3] = input[3] + input[60];
  output[4] = input[4] + input[59];
  output[5] = input[5] + input[58];
  output[6] = input[6] + input[57];
  output[7] = input[7] + input[56];
  output[8] = input[8] + input[55];
  output[9] = input[9] + input[54];
  output[10] = input[10] + input[53];
  output[11] = input[11] + input[52];
  output[12] = input[12] + input[51];
  output[13] = input[13] + input[50];
  output[14] = input[14] + input[49];
  output[15] = input[15] + input[48];
  output[16] = input[16] + input[47];
  output[17] = input[17] + input[46];
  output[18] = input[18] + input[45];
  output[19] = input[19] + input[44];
  output[20] = input[20] + input[43];
  output[21] = input[21] + input[42];
  output[22] = input[22] + input[41];
  output[23] = input[23] + input[40];
  output[24] = input[24] + input[39];
  output[25] = input[25] + input[38];
  output[26] = input[26] + input[37];
  output[27] = input[27] + input[36];
  output[28] = input[28] + input[35];
  output[29] = input[29] + input[34];
  output[30] = input[30] + input[33];
  output[31] = input[31] + input[32];
  output[32] = -input[32] + input[31];
  output[33] = -input[33] + input[30];
  output[34] = -input[34] + input[29];
  output[35] = -input[35] + input[28];
  output[36] = -input[36] + input[27];
  output[37] = -input[37] + input[26];
  output[38] = -input[38] + input[25];
  output[39] = -input[39] + input[24];
  output[40] = -input[40] + input[23];
  output[41] = -input[41] + input[22];
  output[42] = -input[42] + input[21];
  output[43] = -input[43] + input[20];
  output[44] = -input[44] + input[19];
  output[45] = -input[45] + input[18];
  output[46] = -input[46] + input[17];
  output[47] = -input[47] + input[16];
  output[48] = -input[48] + input[15];
  output[49] = -input[49] + input[14];
  output[50] = -input[50] + input[13];
  output[51] = -input[51] + input[12];
  output[52] = -input[52] + input[11];
  output[53] = -input[53] + input[10];
  output[54] = -input[54] + input[9];
  output[55] = -input[55] + input[8];
  output[56] = -input[56] + input[7];
  output[57] = -input[57] + input[6];
  output[58] = -input[58] + input[5];
  output[59] = -input[59] + input[4];
  output[60] = -input[60] + input[3];
  output[61] = -input[61] + input[2];
  output[62] = -input[62] + input[1];
  output[63] = -input[63] + input[0];

  // stage 2
  step[0] = output[0] + output[31];
  step[1] = output[1] + output[30];
  step[2] = output[2] + output[29];
  step[3] = output[3] + output[28];
  step[4] = output[4] + output[27];
  step[5] = output[5] + output[26];
  step[6] = output[6] + output[25];
  step[7] = output[7] + output[24];
  step[8] = output[8] + output[23];
  step[9] = output[9] + output[22];
  step[10] = output[10] + output[21];
  step[11] = output[11] + output[20];
  step[12] = output[12] + output[19];
  step[13] = output[13] + output[18];
  step[14] = output[14] + output[17];
  step[15] = output[15] + output[16];
  step[16] = -output[16] + output[15];
  step[17] = -output[17] + output[14];
  step[18] = -output[18] + output[13];
  step[19] = -output[19] + output[12];
  step[20] = -output[20] + output[11];
  step[21] = -output[21] + output[10];
  step[22] = -output[22] + output[9];
  step[23] = -output[23] + output[8];
  step[24] = -output[24] + output[7];
  step[25] = -output[25] + output[6];
  step[26] = -output[26] + output[5];
  step[27] = -output[27] + output[4];
  step[28] = -output[28] + output[3];
  step[29] = -output[29] + output[2];
  step[30] = -output[30] + output[1];
  step[31] = -output[31] + output[0];
  step[32] = output[32];
  step[33] = output[33];
  step[34] = output[34];
  step[35] = output[35];
  step[36] = output[36];
  step[37] = output[37];
  step[38] = output[38];
  step[39] = output[39];
  step[40] = half_btf(-cospi[32], output[40], cospi[32], output[55], cos_bit);
  step[41] = half_btf(-cospi[32], output[41], cospi[32], output[54], cos_bit);
  step[42] = half_btf(-cospi[32], output[42], cospi[32], output[53], cos_bit);
  step[43] = half_btf(-cospi[32], output[43], cospi[32], output[52], cos_bit);
  step[44] = half_btf(-cospi[32], output[44], cospi[32], output[51], cos_bit);
  step[45] = half_btf(-cospi[32], output[45], cospi[32], output[50], cos_bit);
  step[46] = half_btf(-cospi[32], output[46], cospi[32], output[49], cos_bit);
  step[47] = half_btf(-cospi[32], output[47], cospi[32], output[48], cos_bit);
  step[48] = half_btf(cospi[32], output[48], cospi[32], output[47], cos_bit);
  step[49] = half_btf(cospi[32], output[49], cospi[32], output[46], cos_bit);
  step[50] = half_btf(cospi[32], output[50], cospi[32], output[45], cos_bit);
  step[51] = half_btf(cospi[32], output[51], cospi[32], output[44], cos_bit);
  step[52] = half_btf(cospi[32], output[52], cospi[32], output[43], cos_bit);
  step[53] = half_btf(cospi[32], output[53], cospi[32], output[42], cos_bit);
  step[54] = half_btf(cospi[32], output[54], cospi[32], output[41], cos_bit);
  step[55] = half_btf(cospi[32], output[55], cospi[32], output[40], cos_bit);
  step[56] = output[56];
  step[57] = output[57];
  step[58] = output[58];
  step[59] = output[59];
  step[60] = output[60];
  step[61] = output[61];
  step[62] = output[62];
  step[63] = output[63];

  // stage 3
  output[0] = step[0] + step[15];
  output[1] = step[1] + step[14];
  output[2] = step[2] + step[13];
  output[3] = step[3] + step[12];
  output[4] = step[4] + step[11];
  output[5] = step[5] + step[10];
  output[6] = step[6] + step[9];
  output[7] = step[7] + step[8];
  output[8] = -step[8] + step[7];
  output[9] = -step[9] + step[6];
  output[10] = -step[10] + step[5];
  output[11] = -step[11] + step[4];
  output[12] = -step[12] + step[3];
  output[13] = -step[13] + step[2];
  output[14] = -step[14] + step[1];
  output[15] = -step[15] + step[0];
  output[16] = step[16];
  output[17] = step[17];
  output[18] = step[18];
  output[19] = step[19];
  output[20] = half_btf(-cospi[32], step[20], cospi[32], step[27], cos_bit);
  output[21] = half_btf(-cospi[32], step[21], cospi[32], step[26], cos_bit);
  output[22] = half_btf(-cospi[32], step[22], cospi[32], step[25], cos_bit);
  output[23] = half_btf(-cospi[32], step[23], cospi[32], step[24], cos_bit);
  output[24] = half_btf(cospi[32], step[24], cospi[32], step[23], cos_bit);
  output[25] = half_btf(cospi[32], step[25], cospi[32], step[22], cos_bit);
  output[26] = half_btf(cospi[32], step[26], cospi[32], step[21], cos_bit);
  output[27] = half_btf(cospi[32], step[27], cospi[32], step[20], cos_bit);
  output[28] = step[28];
  output[29] = step[29];
  output[30] = step[30];
  output[31] = step[31];
  output[32] = step[32] + step[47];
  output[33] = step[33] + step[46];
  output[34] = step[34] + step[45];
  output[35] = step[35] + step[44];
  output[36] = step[36] + step[43];
  output[37] = step[37] + step[42];
  output[38] = step[38] + step[41];
  output[39] = step[39] + step[40];
  output[40] = -step[40] + step[39];
  output[41] = -step[41] + step[38];
  output[42] = -step[42] + step[37];
  output[43] = -step[43] + step[36];
  output[44] = -step[44] + step[35];
  output[45] = -step[45] + step[34];
  output[46] = -step[46] + step[33];
  output[47] = -step[47] + step[32];
  output[48] = -step[48] + step[63];
  output[49] = -step[49] + step[62];
  output[50] = -step[50] + step[61];
  output[51] = -step[51] + step[60];
  output[52] = -step[52] + step[59];
  output[53] = -step[53] + step[58];
  output[54] = -step[54] + step[57];
  output[55] = -step[55] + step[56];
  output[56] = step[56] + step[55];
  output[57] = step[57] + step[54];
  output[58] = step[58] + step[53];
  output[59] = step[59] + step[52];
  output[60] = step[60] + step[51];
  output[61] = step[61] + step[50];
  output[62] = step[62] + step[49];
  output[63] = step[63] + step[48];

  // stage 4
  step[0] = output[0] + output[7];
  step[1] = output[1] + output[6];
  step[2] = output[2] + output[5];
  step[3] = output[3] + output[4];
  step[4] = -output[4] + output[3];
  step[5] = -output[5] + output[2];
  step[6] = -output[6] + output[1];
  step[7] = -output[7] + output[0];
  step[8] = output[8];
  step[9] = output[9];
  step[10] = half_btf(-cospi[32], output[10], cospi[32], output[13], cos_bit);
  step[11] = half_btf(-cospi[32], output[11], cospi[32], output[12], cos_bit);
  step[12] = half_btf(cospi[32], output[12], cospi[32], output[11], cos_bit);
  step[13] = half_btf(cospi[32], output[13], cospi[32], output[10], cos_bit);
  step[14] = output[14];
  step[15] = output[15];
  step[16] = output[16] + output[23];
  step[17] = output[17] + output[22];
  step[18] = output[18] + output[21];
  step[19] = output[19] + output[20];
  step[20] = -output[20] + output[19];
  step[21] = -output[21] + output[18];
  step[22] = -output[22] + output[17];
  step[23] = -output[23] + output[16];
  step[24] = -output[24] + output[31];
  step[25] = -output[25] + output[30];
  step[26] = -output[26] + output[29];
  step[27] = -output[27] + output[28];
  step[28] = output[28] + output[27];
  step[29] = output[29] + output[26];
  step[30] = output[30] + output[25];
  step[31] = output[31] + output[24];
  step[32] = output[32];
  step[33] = output[33];
  step[34] = output[34];
  step[35] = output[35];
  step[36] = half_btf(-cospi[16], output[36], cospi[48], output[59], cos_bit);
  step[37] = half_btf(-cospi[16], output[37], cospi[48], output[58], cos_bit);
  step[38] = half_btf(-cospi[16], output[38], cospi[48], output[57], cos_bit);
  step[39] = half_btf(-cospi[16], output[39], cospi[48], output[56], cos_bit);
  step[40] = half_btf(-cospi[48], output[40], -cospi[16], output[55], cos_bit);
  step[41] = half_btf(-cospi[48], output[41], -cospi[16], output[54], cos_bit);
  step[42] = half_btf(-cospi[48], output[42], -cospi[16], output[53], cos_bit);
  step[43] = half_btf(-cospi[48], output[43], -cospi[16], output[52], cos_bit);
  step[44] = output[44];
  step[45] = output[45];
  step[46] = output[46];
  step[47] = output[47];
  step[48] = output[48];
  step[49] = output[49];
  step[50] = output[50];
  step[51] = output[51];
  step[52] = half_btf(cospi[48], output[52], -cospi[16], output[43], cos_bit);
  step[53] = half_btf(cospi[48], output[53], -cospi[16], output[42], cos_bit);
  step[54] = half_btf(cospi[48], output[54], -cospi[16], output[41], cos_bit);
  step[55] = half_btf(cospi[48], output[55], -cospi[16], output[40], cos_bit);
  step[56] = half_btf(cospi[16], output[56], cospi[48], output[39], cos_bit);
  step[57] = half_btf(cospi[16], output[57], cospi[48], output[38], cos_bit);
  step[58] = half_btf(cospi[16], output[58], cospi[48], output[37], cos_bit);
  step[59] = half_btf(cospi[16], output[59], cospi[48], output[36], cos_bit);
  step[60] = output[60];
  step[61] = output[61];
  step[62] = output[62];
  step[63] = output[63];

  // stage 5
  output[0] = step[0] + step[3];
  output[1] = step[1] + step[2];
  output[2] = -step[2] + step[1];
  output[3] = -step[3] + step[0];
  output[4] = step[4];
  output[5] = half_btf(-cospi[32], step[5], cospi[32], step[6], cos_bit);
  output[6] = half_btf(cospi[32], step[6], cospi[32], step[5], cos_bit);
  output[7] = step[7];
  output[8] = step[8] + step[11];
  output[9] = step[9] + step[10];
  output[10] = -step[10] + step[9];
  output[11] = -step[11] + step[8];
  output[12] = -step[12] + step[15];
  output[13] = -step[13] + step[14];
  output[14] = step[14] + step[13];
  output[15] = step[15] + step[12];
  output[16] = step[16];
  output[17] = step[17];
  output[18] = half_btf(-cospi[16], step[18], cospi[48], step[29], cos_bit);
  output[19] = half_btf(-cospi[16], step[19], cospi[48], step[28], cos_bit);
  output[20] = half_btf(-cospi[48], step[20], -cospi[16], step[27], cos_bit);
  output[21] = half_btf(-cospi[48], step[21], -cospi[16], step[26], cos_bit);
  output[22] = step[22];
  output[23] = step[23];
  output[24] = step[24];
  output[25] = step[25];
  output[26] = half_btf(cospi[48], step[26], -cospi[16], step[21], cos_bit);
  output[27] = half_btf(cospi[48], step[27], -cospi[16], step[20], cos_bit);
  output[28] = half_btf(cospi[16], step[28], cospi[48], step[19], cos_bit);
  output[29] = half_btf(cospi[16], step[29], cospi[48], step[18], cos_bit);
  output[30] = step[30];
  output[31] = step[31];
  output[32] = step[32] + step[39];
  output[33] = step[33] + step[38];
  output[34] = step[34] + step[37];
  output[35] = step[35] + step[36];
  output[36] = -step[36] + step[35];
  output[37] = -step[37] + step[34];
  output[38] = -step[38] + step[33];
  output[39] = -step[39] + step[32];
  output[40] = -step[40] + step[47];
  output[41] = -step[41] + step[46];
  output[42] = -step[42] + step[45];
  output[43] = -step[43] + step[44];
  output[44] = step[44] + step[43];
  output[45] = step[45] + step[42];
  output[46] = step[46] + step[41];
  output[47] = step[47] + step[40];
  output[48] = step[48] + step[55];
  output[49] = step[49] + step[54];
  output[50] = step[50] + step[53];
  output[51] = step[51] + step[52];
  output[52] = -step[52] + step[51];
  output[53] = -step[53] + step[50];
  output[54] = -step[54] + step[49];
  output[55] = -step[55] + step[48];
  output[56] = -step[56] + step[63];
  output[57] = -step[57] + step[62];
  output[58] = -step[58] + step[61];
  output[59] = -step[59] + step[60];
  output[60] = step[60] + step[59];
  output[61] = step[61] + step[58];
  output[62] = step[62] + step[57];
  output[63] = step[63] + step[56];

  // stage 6
  step[0] = half_btf(cospi[32], output[0], cospi[32], output[1], cos_bit);
  step[1] = half_btf(-cospi[32], output[1], cospi[32], output[0], cos_bit);
  step[2] = half_btf(cospi[48], output[2], cospi[16], output[3], cos_bit);
  step[3] = half_btf(cospi[48], output[3], -cospi[16], output[2], cos_bit);
  step[4] = output[4] + output[5];
  step[5] = -output[5] + output[4];
  step[6] = -output[6] + output[7];
  step[7] = output[7] + output[6];
  step[8] = output[8];
  step[9] = half_btf(-cospi[16], output[9], cospi[48], output[14], cos_bit);
  step[10] = half_btf(-cospi[48], output[10], -cospi[16], output[13], cos_bit);
  step[11] = output[11];
  step[12] = output[12];
  step[13] = half_btf(cospi[48], output[13], -cospi[16], output[10], cos_bit);
  step[14] = half_btf(cospi[16], output[14], cospi[48], output[9], cos_bit);
  step[15] = output[15];
  step[16] = output[16] + output[19];
  step[17] = output[17] + output[18];
  step[18] = -output[18] + output[17];
  step[19] = -output[19] + output[16];
  step[20] = -output[20] + output[23];
  step[21] = -output[21] + output[22];
  step[22] = output[22] + output[21];
  step[23] = output[23] + output[20];
  step[24] = output[24] + output[27];
  step[25] = output[25] + output[26];
  step[26] = -output[26] + output[25];
  step[27] = -output[27] + output[24];
  step[28] = -output[28] + output[31];
  step[29] = -output[29] + output[30];
  step[30] = output[30] + output[29];
  step[31] = output[31] + output[28];
  step[32] = output[32];
  step[33] = output[33];
  step[34] = half_btf(-cospi[8], output[34], cospi[56], output[61], cos_bit);
  step[35] = half_btf(-cospi[8], output[35], cospi[56], output[60], cos_bit);
  step[36] = half_btf(-cospi[56], output[36], -cospi[8], output[59], cos_bit);
  step[37] = half_btf(-cospi[56], output[37], -cospi[8], output[58], cos_bit);
  step[38] = output[38];
  step[39] = output[39];
  step[40] = output[40];
  step[41] = output[41];
  step[42] = half_btf(-cospi[40], output[42], cospi[24], output[53], cos_bit);
  step[43] = half_btf(-cospi[40], output[43], cospi[24], output[52], cos_bit);
  step[44] = half_btf(-cospi[24], output[44], -cospi[40], output[51], cos_bit);
  step[45] = half_btf(-cospi[24], output[45], -cospi[40], output[50], cos_bit);
  step[46] = output[46];
  step[47] = output[47];
  step[48] = output[48];
  step[49] = output[49];
  step[50] = half_btf(cospi[24], output[50], -cospi[40], output[45], cos_bit);
  step[51] = half_btf(cospi[24], output[51], -cospi[40], output[44], cos_bit);
  step[52] = half_btf(cospi[40], output[52], cospi[24], output[43], cos_bit);
  step[53] = half_btf(cospi[40], output[53], cospi[24], output[42], cos_bit);
  step[54] = output[54];
  step[55] = output[55];
  step[56] = output[56];
  step[57] = output[57];
  step[58] = half_btf(cospi[56], output[58], -cospi[8], output[37], cos_bit);
  step[59] = half_btf(cospi[56], output[59], -cospi[8], output[36], cos_bit);
  step[60] = half_btf(cospi[8], output[60], cospi[56], output[35], cos_bit);
  step[61] = half_btf(cospi[8], output[61], cospi[56], output[34], cos_bit);
  step[62] = output[62];
  step[63] = output[63];

  // stage 7
  output[0] = step[0];
  output[1] = step[1];
  output[2] = step[2];
  output[3] = step[3];
  output[4] = half_btf(cospi[56], step[4], cospi[8], step[7], cos_bit);
  output[5] = half_btf(cospi[24], step[5], cospi[40], step[6], cos_bit);
  output[6] = half_btf(cospi[24], step[6], -cospi[40], step[5], cos_bit);
  output[7] = half_btf(cospi[56], step[7], -cospi[8], step[4], cos_bit);
  output[8] = step[8] + step[9];
  output[9] = -step[9] + step[8];
  output[10] = -step[10] + step[11];
  output[11] = step[11] + step[10];
  output[12] = step[12] + step[13];
  output[13] = -step[13] + step[12];
  output[14] = -step[14] + step[15];
  output[15] = step[15] + step[14];
  output[16] = step[16];
  output[17] = half_btf(-cospi[8], step[17], cospi[56], step[30], cos_bit);
  output[18] = half_btf(-cospi[56], step[18], -cospi[8], step[29], cos_bit);
  output[19] = step[19];
  output[20] = step[20];
  output[21] = half_btf(-cospi[40], step[21], cospi[24], step[26], cos_bit);
  output[22] = half_btf(-cospi[24], step[22], -cospi[40], step[25], cos_bit);
  output[23] = step[23];
  output[24] = step[24];
  output[25] = half_btf(cospi[24], step[25], -cospi[40], step[22], cos_bit);
  output[26] = half_btf(cospi[40], step[26], cospi[24], step[21], cos_bit);
  output[27] = step[27];
  output[28] = step[28];
  output[29] = half_btf(cospi[56], step[29], -cospi[8], step[18], cos_bit);
  output[30] = half_btf(cospi[8], step[30], cospi[56], step[17], cos_bit);
  output[31] = step[31];
  output[32] = step[32] + step[35];
  output[33] = step[33] + step[34];
  output[34] = -step[34] + step[33];
  output[35] = -step[35] + step[32];
  output[36] = -step[36] + step[39];
  output[37] = -step[37] + step[38];
  output[38] = step[38] + step[37];
  output[39] = step[39] + step[36];
  output[40] = step[40] + step[43];
  output[41] = step[41] + step[42];
  output[42] = -step[42] + step[41];
  output[43] = -step[43] + step[40];
  output[44] = -step[44] + step[47];
  output[45] = -step[45] + step[46];
  output[46] = step[46] + step[45];
  output[47] = step[47] + step[44];
  output[48] = step[48] + step[51];
  output[49] = step[49] + step[50];
  output[50] = -step[50] + step[49];
  output[51] = -step[51] + step[48];
  output[52] = -step[52] + step[55];
  output[53] = -step[53] + step[54];
  output[54] = step[54] + step[53];
  output[55] = step[55] + step[52];
  output[56] = step[56] + step[59];
  output[57] = step[57] + step[58];
  output[58] = -step[58] + step[57];
  output[59] = -step[59] + step[56];
  output[60] = -step[60] + step[63];
  output[61] = -step[61] + step[62];
  output[62] = step[62] + step[61];
  output[63] = step[63] + step[60];

  // stage 8
  step[0] = output[0];
  step[1] = output[1];
  step[2] = output[2];
  step[3] = output[3];
  step[4] = output[4];
  step[5] = output[5];
  step[6] = output[6];
  step[7] = output[7];
  step[8] = half_btf(cospi[60], output[8], cospi[4], output[15], cos_bit);
  step[9] = half_btf(cospi[28], output[9], cospi[36], output[14], cos_bit);
  step[10] = half_btf(cospi[44], output[10], cospi[20], output[13], cos_bit);
  step[11] = half_btf(cospi[12], output[11], cospi[52], output[12], cos_bit);
  step[12] = half_btf(cospi[12], output[12], -cospi[52], output[11], cos_bit);
  step[13] = half_btf(cospi[44], output[13], -cospi[20], output[10], cos_bit);
  step[14] = half_btf(cospi[28], output[14], -cospi[36], output[9], cos_bit);
  step[15] = half_btf(cospi[60], output[15], -cospi[4], output[8], cos_bit);
  step[16] = output[16] + output[17];
  step[17] = -output[17] + output[16];
  step[18] = -output[18] + output[19];
  step[19] = output[19] + output[18];
  step[20] = output[20] + output[21];
  step[21] = -output[21] + output[20];
  step[22] = -output[22] + output[23];
  step[23] = output[23] + output[22];
  step[24] = output[24] + output[25];
  step[25] = -output[25] + output[24];
  step[26] = -output[26] + output[27];
  step[27] = output[27] + output[26];
  step[28] = output[28] + output[29];
  step[29] = -output[29] + output[28];
  step[30] = -output[30] + output[31];
  step[31] = output[31] + output[30];
  step[32] = output[32];
  step[33] = half_btf(-cospi[4], output[33], cospi[60], output[62], cos_bit);
  step[34] = half_btf(-cospi[60], output[34], -cospi[4], output[61], cos_bit);
  step[35] = output[35];
  step[36] = output[36];
  step[37] = half_btf(-cospi[36], output[37], cospi[28], output[58], cos_bit);
  step[38] = half_btf(-cospi[28], output[38], -cospi[36], output[57], cos_bit);
  step[39] = output[39];
  step[40] = output[40];
  step[41] = half_btf(-cospi[20], output[41], cospi[44], output[54], cos_bit);
  step[42] = half_btf(-cospi[44], output[42], -cospi[20], output[53], cos_bit);
  step[43] = output[43];
  step[44] = output[44];
  step[45] = half_btf(-cospi[52], output[45], cospi[12], output[50], cos_bit);
  step[46] = half_btf(-cospi[12], output[46], -cospi[52], output[49], cos_bit);
  step[47] = output[47];
  step[48] = output[48];
  step[49] = half_btf(cospi[12], output[49], -cospi[52], output[46], cos_bit);
  step[50] = half_btf(cospi[52], output[50], cospi[12], output[45], cos_bit);
  step[51] = output[51];
  step[52] = output[52];
  step[53] = half_btf(cospi[44], output[53], -cospi[20], output[42], cos_bit);
  step[54] = half_btf(cospi[20], output[54], cospi[44], output[41], cos_bit);
  step[55] = output[55];
  step[56] = output[56];
  step[57] = half_btf(cospi[28], output[57], -cospi[36], output[38], cos_bit);
  step[58] = half_btf(cospi[36], output[58], cospi[28], output[37], cos_bit);
  step[59] = output[59];
  step[60] = output[60];
  step[61] = half_btf(cospi[60], output[61], -cospi[4], output[34], cos_bit);
  step[62] = half_btf(cospi[4], output[62], cospi[60], output[33], cos_bit);
  step[63] = output[63];

  // stage 9
  output[0] = step[0];
  output[1] = step[1];
  output[2] = step[2];
  output[3] = step[3];
  output[4] = step[4];
  output[5] = step[5];
  output[6] = step[6];
  output[7] = step[7];
  output[8] = step[8];
  output[9] = step[9];
  output[10] = step[10];
  output[11] = step[11];
  output[12] = step[12];
  output[13] = step[13];
  output[14] = step[14];
  output[15] = step[15];
  output[16] = half_btf(cospi[62], step[16], cospi[2], step[31], cos_bit);
  output[17] = half_btf(cospi[30], step[17], cospi[34], step[30], cos_bit);
  output[18] = half_btf(cospi[46], step[18], cospi[18], step[29], cos_bit);
  output[19] = half_btf(cospi[14], step[19], cospi[50], step[28], cos_bit);
  output[20] = half_btf(cospi[54], step[20], cospi[10], step[27], cos_bit);
  output[21] = half_btf(cospi[22], step[21], cospi[42], step[26], cos_bit);
  output[22] = half_btf(cospi[38], step[22], cospi[26], step[25], cos_bit);
  output[23] = half_btf(cospi[6], step[23], cospi[58], step[24], cos_bit);
  output[24] = half_btf(cospi[6], step[24], -cospi[58], step[23], cos_bit);
  output[25] = half_btf(cospi[38], step[25], -cospi[26], step[22], cos_bit);
  output[26] = half_btf(cospi[22], step[26], -cospi[42], step[21], cos_bit);
  output[27] = half_btf(cospi[54], step[27], -cospi[10], step[20], cos_bit);
  output[28] = half_btf(cospi[14], step[28], -cospi[50], step[19], cos_bit);
  output[29] = half_btf(cospi[46], step[29], -cospi[18], step[18], cos_bit);
  output[30] = half_btf(cospi[30], step[30], -cospi[34], step[17], cos_bit);
  output[31] = half_btf(cospi[62], step[31], -cospi[2], step[16], cos_bit);
  output[32] = step[32] + step[33];
  output[33] = -step[33] + step[32];
  output[34] = -step[34] + step[35];
  output[35] = step[35] + step[34];
  output[36] = step[36] + step[37];
  output[37] = -step[37] + step[36];
  output[38] = -step[38] + step[39];
  output[39] = step[39] + step[38];
  output[40] = step[40] + step[41];
  output[41] = -step[41] + step[40];
  output[42] = -step[42] + step[43];
  output[43] = step[43] + step[42];
  output[44] = step[44] + step[45];
  output[45] = -step[45] + step[44];
  output[46] = -step[46] + step[47];
  output[47] = step[47] + step[46];
  output[48] = step[48] + step[49];
  output[49] = -step[49] + step[48];
  output[50] = -step[50] + step[51];
  output[51] = step[51] + step[50];
  output[52] = step[52] + step[53];
  output[53] = -step[53] + step[52];
  output[54] = -step[54] + step[55];
  output[55] = step[55] + step[54];
  output[56] = step[56] + step[57];
  output[57] = -step[57] + step[56];
  output[58] = -step[58] + step[59];
  output[59] = step[59] + step[58];
  output[60] = step[60] + step[61];
  output[61] = -step[61] + step[60];
  output[62] = -step[62] + step[63];
  output[63] = step[63] + step[62];

  // stage 10
  step[0] = output[0];
  step[1] = output[1];
  step[2] = output[2];
  step[3] = output[3];
  step[4] = output[4];
  step[5] = output[5];
  step[6] = output[6];
  step[7] = output[7];
  step[8] = output[8];
  step[9] = output[9];
  step[10] = output[10];
  step[11] = output[11];
  step[12] = output[12];
  step[13] = output[13];
  step[14] = output[14];
  step[15] = output[15];
  step[16] = output[16];
  step[17] = output[17];
  step[18] = output[18];
  step[19] = output[19];
  step[20] = output[20];
  step[21] = output[21];
  step[22] = output[22];
  step[23] = output[23];
  step[24] = output[24];
  step[25] = output[25];
  step[26] = output[26];
  step[27] = output[27];
  step[28] = output[28];
  step[29] = output[29];
  step[30] = output[30];
  step[31] = output[31];
  step[32] = half_btf(cospi[63], output[32], cospi[1], output[63], cos_bit);
  step[33] = half_btf(cospi[31], output[33], cospi[33], output[62], cos_bit);
  step[34] = half_btf(cospi[47], output[34], cospi[17], output[61], cos_bit);
  step[35] = half_btf(cospi[15], output[35], cospi[49], output[60], cos_bit);
  step[36] = half_btf(cospi[55], output[36], cospi[9], output[59], cos_bit);
  step[37] = half_btf(cospi[23], output[37], cospi[41], output[58], cos_bit);
  step[38] = half_btf(cospi[39], output[38], cospi[25], output[57], cos_bit);
  step[39] = half_btf(cospi[7], output[39], cospi[57], output[56], cos_bit);
  step[40] = half_btf(cospi[59], output[40], cospi[5], output[55], cos_bit);
  step[41] = half_btf(cospi[27], output[41], cospi[37], output[54], cos_bit);
  step[42] = half_btf(cospi[43], output[42], cospi[21], output[53], cos_bit);
  step[43] = half_btf(cospi[11], output[43], cospi[53], output[52], cos_bit);
  step[44] = half_btf(cospi[51], output[44], cospi[13], output[51], cos_bit);
  step[45] = half_btf(cospi[19], output[45], cospi[45], output[50], cos_bit);
  step[46] = half_btf(cospi[35], output[46], cospi[29], output[49], cos_bit);
  step[47] = half_btf(cospi[3], output[47], cospi[61], output[48], cos_bit);
  step[48] = half_btf(cospi[3], output[48], -cospi[61], output[47], cos_bit);
  step[49] = half_btf(cospi[35], output[49], -cospi[29], output[46], cos_bit);
  step[50] = half_btf(cospi[19], output[50], -cospi[45], output[45], cos_bit);
  step[51] = half_btf(cospi[51], output[51], -cospi[13], output[44], cos_bit);
  step[52] = half_btf(cospi[11], output[52], -cospi[53], output[43], cos_bit);
  step[53] = half_btf(cospi[43], output[53], -cospi[21], output[42], cos_bit);
  step[54] = half_btf(cospi[27], output[54], -cospi[37], output[41], cos_bit);
  step[55] = half_btf(cospi[59], output[55], -cospi[5], output[40], cos_bit);
  step[56] = half_btf(cospi[7], output[56], -cospi[57], output[39], cos_bit);
  step[57] = half_btf(cospi[39], output[57], -cospi[25], output[38], cos_bit);
  step[58] = half_btf(cospi[23], output[58], -cospi[41], output[37], cos_bit);
  step[59] = half_btf(cospi[55], output[59], -cospi[9], output[36], cos_bit);
  step[60] = half_btf(cospi[15], output[60], -cospi[49], output[35], cos_bit);
  step[61] = half_btf(cospi[47], output[61], -cospi[17], output[34], cos_bit);
  step[62] = half_btf(cospi[31], output[62], -cospi[33], output[33], cos_bit);
  step[63] = half_btf(cospi[63], output[63], -cospi[1], output[32], cos_bit);

  // stage 11
  output[0] = step[0];
  output[1] = step[32];
  output[2] = step[16];
  output[3] = step[48];
  output[4] = step[8];
  output[5] = step[40];
  output[6] = step[24];
  output[7] = step[56];
  output[8] = step[4];
  output[9] = step[36];
  output[10] = step[20];
  output[11] = step[52];
  output[12] = step[12];
  output[13] = step[44];
  output[14] = step[28];
  output[15] = step[60];
  output[16] = step[2];
  output[17] = step[34];
  output[18] = step[18];
  output[19] = step[50];
  output[20] = step[10];
  output[21] = step[42];
  output[22] = step[26];
  output[23] = step[58];
  output[24] = step[6];
  output[25] = step[38];
  output[26] = step[22];
  output[27] = step[54];
  output[28] = step[14];
  output[29] = step[46];
  output[30] = step[30];
  output[31] = step[62];
  output[32] = step[1];
  output[33] = step[33];
  output[34] = step[17];
  output[35] = step[49];
  output[36] = step[9];
  output[37] = step[41];
  output[38] = step[25];
  output[39] = step[57];
  output[40] = step[5];
  output[41] = step[37];
  output[42] = step[21];
  output[43] = step[53];
  output[44] = step[13];
  output[45] = step[45];
  output[46] = step[29];
  output[47] = step[61];
  output[48] = step[3];
  output[49] = step[35];
  output[50] = step[19];
  output[51] = step[51];
  output[52] = step[11];
  output[53] = step[43];
  output[54] = step[27];
  output[55] = step[59];
  output[56] = step[7];
  output[57] = step[39];
  output[58] = step[23];
  output[59] = step[55];
  output[60] = step[15];
  output[61] = step[47];
  output[62] = step[31];
  output[63] = step[63];
}

fn av1_fadst4_new(
  input: &[i32], output: &mut [i32], cos_bit: usize, _stage_range: &[i8]
) {
  let sinpi = sinpi_arr(cos_bit);
  let mut x0;
  let mut x1;
  let mut x2;
  let mut x3;
  let mut s0;
  let mut s1;
  let mut s2;
  let mut s3;
  let s4;
  let s5;
  let s6;
  let mut s7;

  // stage 0
  x0 = input[0];
  x1 = input[1];
  x2 = input[2];
  x3 = input[3];

  if (x0 | x1 | x2 | x3) == 0 {
    output[0] = 0;
    output[1] = 0;
    output[2] = 0;
    output[3] = 0;
    return;
  }

  // stage 1
  s0 = sinpi[1] * x0;
  s1 = sinpi[4] * x0;
  s2 = sinpi[2] * x1;
  s3 = sinpi[1] * x1;
  s4 = sinpi[3] * x2;
  s5 = sinpi[4] * x3;
  s6 = sinpi[2] * x3;
  s7 = x0 + x1;

  // stage 2
  s7 = s7 - x3;

  // stage 3
  x0 = s0 + s2;
  x1 = sinpi[3] * s7;
  x2 = s1 - s3;
  x3 = s4;

  // stage 4
  x0 = x0 + s5;
  x2 = x2 + s6;

  // stage 5
  s0 = x0 + x3;
  s1 = x1;
  s2 = x2 - x3;
  s3 = x2 - x0;

  // stage 6
  s3 = s3 + x3;

  // 1-D transform scaling factor is sqrt(2).
  output[0] = round_shift(s0, cos_bit);
  output[1] = round_shift(s1, cos_bit);
  output[2] = round_shift(s2, cos_bit);
  output[3] = round_shift(s3, cos_bit);
}

fn av1_fadst8_new(
  input: &[i32], output: &mut [i32], cos_bit: usize, _stage_range: &[i8]
) {
  let mut step = [0i32; 8];
  let cospi = cospi_arr(cos_bit);

  // stage 1;
  output[0] = input[0];
  output[1] = -input[7];
  output[2] = -input[3];
  output[3] = input[4];
  output[4] = -input[1];
  output[5] = input[6];
  output[6] = input[2];
  output[7] = -input[5];

  // stage 2
  step[0] = output[0];
  step[1] = output[1];
  step[2] = half_btf(cospi[32], output[2], cospi[32], output[3], cos_bit);
  step[3] = half_btf(cospi[32], output[2], -cospi[32], output[3], cos_bit);
  step[4] = output[4];
  step[5] = output[5];
  step[6] = half_btf(cospi[32], output[6], cospi[32], output[7], cos_bit);
  step[7] = half_btf(cospi[32], output[6], -cospi[32], output[7], cos_bit);

  // stage 3
  output[0] = step[0] + step[2];
  output[1] = step[1] + step[3];
  output[2] = step[0] - step[2];
  output[3] = step[1] - step[3];
  output[4] = step[4] + step[6];
  output[5] = step[5] + step[7];
  output[6] = step[4] - step[6];
  output[7] = step[5] - step[7];

  // stage 4
  step[0] = output[0];
  step[1] = output[1];
  step[2] = output[2];
  step[3] = output[3];
  step[4] = half_btf(cospi[16], output[4], cospi[48], output[5], cos_bit);
  step[5] = half_btf(cospi[48], output[4], -cospi[16], output[5], cos_bit);
  step[6] = half_btf(-cospi[48], output[6], cospi[16], output[7], cos_bit);
  step[7] = half_btf(cospi[16], output[6], cospi[48], output[7], cos_bit);

  // stage 5
  output[0] = step[0] + step[4];
  output[1] = step[1] + step[5];
  output[2] = step[2] + step[6];
  output[3] = step[3] + step[7];
  output[4] = step[0] - step[4];
  output[5] = step[1] - step[5];
  output[6] = step[2] - step[6];
  output[7] = step[3] - step[7];

  // stage 6
  step[0] = half_btf(cospi[4], output[0], cospi[60], output[1], cos_bit);
  step[1] = half_btf(cospi[60], output[0], -cospi[4], output[1], cos_bit);
  step[2] = half_btf(cospi[20], output[2], cospi[44], output[3], cos_bit);
  step[3] = half_btf(cospi[44], output[2], -cospi[20], output[3], cos_bit);
  step[4] = half_btf(cospi[36], output[4], cospi[28], output[5], cos_bit);
  step[5] = half_btf(cospi[28], output[4], -cospi[36], output[5], cos_bit);
  step[6] = half_btf(cospi[52], output[6], cospi[12], output[7], cos_bit);
  step[7] = half_btf(cospi[12], output[6], -cospi[52], output[7], cos_bit);

  // stage 7
  output[0] = step[1];
  output[1] = step[6];
  output[2] = step[3];
  output[3] = step[4];
  output[4] = step[5];
  output[5] = step[2];
  output[6] = step[7];
  output[7] = step[0];
}

fn av1_fadst16_new(
  input: &[i32], output: &mut [i32], cos_bit: usize, _stage_range: &[i8]
) {
  let mut step = [0i32; 16];
  let cospi = cospi_arr(cos_bit);

  // stage 1;
  output[0] = input[0];
  output[1] = -input[15];
  output[2] = -input[7];
  output[3] = input[8];
  output[4] = -input[3];
  output[5] = input[12];
  output[6] = input[4];
  output[7] = -input[11];
  output[8] = -input[1];
  output[9] = input[14];
  output[10] = input[6];
  output[11] = -input[9];
  output[12] = input[2];
  output[13] = -input[13];
  output[14] = -input[5];
  output[15] = input[10];

  // stage 2
  step[0] = output[0];
  step[1] = output[1];
  step[2] = half_btf(cospi[32], output[2], cospi[32], output[3], cos_bit);
  step[3] = half_btf(cospi[32], output[2], -cospi[32], output[3], cos_bit);
  step[4] = output[4];
  step[5] = output[5];
  step[6] = half_btf(cospi[32], output[6], cospi[32], output[7], cos_bit);
  step[7] = half_btf(cospi[32], output[6], -cospi[32], output[7], cos_bit);
  step[8] = output[8];
  step[9] = output[9];
  step[10] = half_btf(cospi[32], output[10], cospi[32], output[11], cos_bit);
  step[11] = half_btf(cospi[32], output[10], -cospi[32], output[11], cos_bit);
  step[12] = output[12];
  step[13] = output[13];
  step[14] = half_btf(cospi[32], output[14], cospi[32], output[15], cos_bit);
  step[15] = half_btf(cospi[32], output[14], -cospi[32], output[15], cos_bit);

  // stage 3
  output[0] = step[0] + step[2];
  output[1] = step[1] + step[3];
  output[2] = step[0] - step[2];
  output[3] = step[1] - step[3];
  output[4] = step[4] + step[6];
  output[5] = step[5] + step[7];
  output[6] = step[4] - step[6];
  output[7] = step[5] - step[7];
  output[8] = step[8] + step[10];
  output[9] = step[9] + step[11];
  output[10] = step[8] - step[10];
  output[11] = step[9] - step[11];
  output[12] = step[12] + step[14];
  output[13] = step[13] + step[15];
  output[14] = step[12] - step[14];
  output[15] = step[13] - step[15];

  // stage 4
  step[0] = output[0];
  step[1] = output[1];
  step[2] = output[2];
  step[3] = output[3];
  step[4] = half_btf(cospi[16], output[4], cospi[48], output[5], cos_bit);
  step[5] = half_btf(cospi[48], output[4], -cospi[16], output[5], cos_bit);
  step[6] = half_btf(-cospi[48], output[6], cospi[16], output[7], cos_bit);
  step[7] = half_btf(cospi[16], output[6], cospi[48], output[7], cos_bit);
  step[8] = output[8];
  step[9] = output[9];
  step[10] = output[10];
  step[11] = output[11];
  step[12] = half_btf(cospi[16], output[12], cospi[48], output[13], cos_bit);
  step[13] = half_btf(cospi[48], output[12], -cospi[16], output[13], cos_bit);
  step[14] = half_btf(-cospi[48], output[14], cospi[16], output[15], cos_bit);
  step[15] = half_btf(cospi[16], output[14], cospi[48], output[15], cos_bit);

  // stage 5
  output[0] = step[0] + step[4];
  output[1] = step[1] + step[5];
  output[2] = step[2] + step[6];
  output[3] = step[3] + step[7];
  output[4] = step[0] - step[4];
  output[5] = step[1] - step[5];
  output[6] = step[2] - step[6];
  output[7] = step[3] - step[7];
  output[8] = step[8] + step[12];
  output[9] = step[9] + step[13];
  output[10] = step[10] + step[14];
  output[11] = step[11] + step[15];
  output[12] = step[8] - step[12];
  output[13] = step[9] - step[13];
  output[14] = step[10] - step[14];
  output[15] = step[11] - step[15];

  // stage 6
  step[0] = output[0];
  step[1] = output[1];
  step[2] = output[2];
  step[3] = output[3];
  step[4] = output[4];
  step[5] = output[5];
  step[6] = output[6];
  step[7] = output[7];
  step[8] = half_btf(cospi[8], output[8], cospi[56], output[9], cos_bit);
  step[9] = half_btf(cospi[56], output[8], -cospi[8], output[9], cos_bit);
  step[10] = half_btf(cospi[40], output[10], cospi[24], output[11], cos_bit);
  step[11] = half_btf(cospi[24], output[10], -cospi[40], output[11], cos_bit);
  step[12] = half_btf(-cospi[56], output[12], cospi[8], output[13], cos_bit);
  step[13] = half_btf(cospi[8], output[12], cospi[56], output[13], cos_bit);
  step[14] = half_btf(-cospi[24], output[14], cospi[40], output[15], cos_bit);
  step[15] = half_btf(cospi[40], output[14], cospi[24], output[15], cos_bit);

  // stage 7
  output[0] = step[0] + step[8];
  output[1] = step[1] + step[9];
  output[2] = step[2] + step[10];
  output[3] = step[3] + step[11];
  output[4] = step[4] + step[12];
  output[5] = step[5] + step[13];
  output[6] = step[6] + step[14];
  output[7] = step[7] + step[15];
  output[8] = step[0] - step[8];
  output[9] = step[1] - step[9];
  output[10] = step[2] - step[10];
  output[11] = step[3] - step[11];
  output[12] = step[4] - step[12];
  output[13] = step[5] - step[13];
  output[14] = step[6] - step[14];
  output[15] = step[7] - step[15];

  // stage 8
  step[0] = half_btf(cospi[2], output[0], cospi[62], output[1], cos_bit);
  step[1] = half_btf(cospi[62], output[0], -cospi[2], output[1], cos_bit);
  step[2] = half_btf(cospi[10], output[2], cospi[54], output[3], cos_bit);
  step[3] = half_btf(cospi[54], output[2], -cospi[10], output[3], cos_bit);
  step[4] = half_btf(cospi[18], output[4], cospi[46], output[5], cos_bit);
  step[5] = half_btf(cospi[46], output[4], -cospi[18], output[5], cos_bit);
  step[6] = half_btf(cospi[26], output[6], cospi[38], output[7], cos_bit);
  step[7] = half_btf(cospi[38], output[6], -cospi[26], output[7], cos_bit);
  step[8] = half_btf(cospi[34], output[8], cospi[30], output[9], cos_bit);
  step[9] = half_btf(cospi[30], output[8], -cospi[34], output[9], cos_bit);
  step[10] = half_btf(cospi[42], output[10], cospi[22], output[11], cos_bit);
  step[11] = half_btf(cospi[22], output[10], -cospi[42], output[11], cos_bit);
  step[12] = half_btf(cospi[50], output[12], cospi[14], output[13], cos_bit);
  step[13] = half_btf(cospi[14], output[12], -cospi[50], output[13], cos_bit);
  step[14] = half_btf(cospi[58], output[14], cospi[6], output[15], cos_bit);
  step[15] = half_btf(cospi[6], output[14], -cospi[58], output[15], cos_bit);

  // stage 9
  output[0] = step[1];
  output[1] = step[14];
  output[2] = step[3];
  output[3] = step[12];
  output[4] = step[5];
  output[5] = step[10];
  output[6] = step[7];
  output[7] = step[8];
  output[8] = step[9];
  output[9] = step[6];
  output[10] = step[11];
  output[11] = step[4];
  output[12] = step[13];
  output[13] = step[2];
  output[14] = step[15];
  output[15] = step[0];
}

fn av1_fidentity4_c(
  input: &[i32], output: &mut [i32], _cos_bit: usize, stage_range: &[i8]
) {
  for i in 0..4 {
    output[i] = round_shift(input[i] * SQRT2, SQRT2_BITS);
  }
  assert!(stage_range[0] + SQRT2_BITS as i8 <= 32);
}

fn av1_fidentity8_c(
  input: &[i32], output: &mut [i32], _cos_bit: usize, _stage_range: &[i8]
) {
  for i in 0..8 {
    output[i] = input[i] * 2;
  }
}

fn av1_fidentity16_c(
  input: &[i32], output: &mut [i32], _cos_bit: usize, stage_range: &[i8]
) {
  for i in 0..16 {
    output[i] = round_shift(input[i] * 2 * SQRT2, SQRT2_BITS);
  }
  assert!(stage_range[0] + SQRT2_BITS as i8 <= 32);
}

fn av1_fidentity32_c(
  input: &[i32], output: &mut [i32], _cos_bit: usize, _stage_range: &[i8]
) {
  for i in 0..32 {
    output[i] = input[i] * 4;
  }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TxfmType {
  DCT4,
  DCT8,
  DCT16,
  DCT32,
  DCT64,
  ADST4,
  ADST8,
  ADST16,
  Identity4,
  Identity8,
  Identity16,
  Identity32,
  Invalid
}

impl TxfmType {
  const TXFM_TYPES: usize = 12;
  const TX_TYPES_1D: usize = 4;
  const AV1_TXFM_TYPE_LS: [[TxfmType; Self::TX_TYPES_1D]; 5] = [
    [TxfmType::DCT4, TxfmType::ADST4, TxfmType::ADST4, TxfmType::Identity4],
    [TxfmType::DCT8, TxfmType::ADST8, TxfmType::ADST8, TxfmType::Identity8],
    [
      TxfmType::DCT16,
      TxfmType::ADST16,
      TxfmType::ADST16,
      TxfmType::Identity16
    ],
    [
      TxfmType::DCT32,
      TxfmType::Invalid,
      TxfmType::Invalid,
      TxfmType::Identity32
    ],
    [TxfmType::DCT64, TxfmType::Invalid, TxfmType::Invalid, TxfmType::Invalid]
  ];

  fn get_func(self) -> &'static TxfmFunc {
    match self {
      TxfmType::DCT4 => &av1_fdct4_new,
      TxfmType::DCT8 => &av1_fdct8_new,
      TxfmType::DCT16 => &av1_fdct16_new,
      TxfmType::DCT32 => &av1_fdct32_new,
      TxfmType::DCT64 => &av1_fdct64_new,
      TxfmType::ADST4 => &av1_fadst4_new,
      TxfmType::ADST8 => &av1_fadst8_new,
      TxfmType::ADST16 => &av1_fadst16_new,
      TxfmType::Identity4 => &av1_fidentity4_c,
      TxfmType::Identity8 => &av1_fidentity8_c,
      TxfmType::Identity16 => &av1_fidentity16_c,
      TxfmType::Identity32 => &av1_fidentity32_c,
      _ => unreachable!()
    }
  }
}

#[derive(Debug, Clone, Copy)]
struct Txfm2DFlipCfg {
  tx_size: TxSize,
  /// Flip upside down
  ud_flip: bool,
  /// Flip left to right
  lr_flip: bool,
  shift: TxfmShift,
  cos_bit_col: i8,
  cos_bit_row: i8,
  stage_range_col: [i8; MAX_TXFM_STAGE_NUM],
  stage_range_row: [i8; MAX_TXFM_STAGE_NUM],
  txfm_type_col: TxfmType,
  txfm_type_row: TxfmType,
  stage_num_col: usize,
  stage_num_row: usize
}

impl Txfm2DFlipCfg {
  fn fwd(tx_type: TxType, tx_size: TxSize) -> Self {
    let tx_type_1d_col = VTX_TAB[tx_type as usize];
    let tx_type_1d_row = HTX_TAB[tx_type as usize];
    let txw_idx = tx_size.width_index();
    let txh_idx = tx_size.height_index();
    let txfm_type_col =
      TxfmType::AV1_TXFM_TYPE_LS[txh_idx][tx_type_1d_col as usize];
    let txfm_type_row =
      TxfmType::AV1_TXFM_TYPE_LS[txw_idx][tx_type_1d_row as usize];
    assert_ne!(txfm_type_col, TxfmType::Invalid);
    assert_ne!(txfm_type_row, TxfmType::Invalid);
    let (ud_flip, lr_flip) = Self::get_flip_cfg(tx_type);
    let mut cfg = Txfm2DFlipCfg {
      tx_size,
      ud_flip,
      lr_flip,
      shift: FWD_TXFM_SHIFT_LS[tx_size as usize],
      cos_bit_col: FWD_COS_BIT_COL[txw_idx][txh_idx],
      cos_bit_row: FWD_COS_BIT_ROW[txw_idx][txh_idx],
      stage_range_col: [0; MAX_TXFM_STAGE_NUM],
      stage_range_row: [0; MAX_TXFM_STAGE_NUM],
      txfm_type_col,
      txfm_type_row,
      stage_num_col: AV1_TXFM_STAGE_NUM_LIST[txfm_type_col as usize] as usize,
      stage_num_row: AV1_TXFM_STAGE_NUM_LIST[txfm_type_row as usize] as usize
    };
    cfg.set_non_scale_range();
    cfg
  }

  fn set_non_scale_range(&mut self) {
    let txh_idx = self.tx_size.height_index();

    let range_mult2_col =
      FWD_TXFM_RANGE_MULT2_LIST[self.txfm_type_col as usize];
    for i in 0..self.stage_num_col {
      self.stage_range_col[i] = (range_mult2_col[i] + 1) >> 1;
    }

    let range_mult2_row =
      FWD_TXFM_RANGE_MULT2_LIST[self.txfm_type_row as usize];
    for i in 0..self.stage_num_row {
      self.stage_range_row[i] =
        (MAX_FWD_RANGE_MULT2_COL[txh_idx] + range_mult2_row[i] + 1) >> 1;
    }
  }

  /// Determine the flip config, returning (ud_flip, lr_flip)
  fn get_flip_cfg(tx_type: TxType) -> (bool, bool) {
    match tx_type {
      TxType::DCT_DCT
      | TxType::ADST_DCT
      | TxType::DCT_ADST
      | TxType::ADST_ADST
      | TxType::IDTX
      | TxType::V_DCT
      | TxType::H_DCT
      | TxType::V_ADST
      | TxType::H_ADST => (false, false),
      TxType::FLIPADST_DCT | TxType::FLIPADST_ADST | TxType::V_FLIPADST =>
        (true, false),
      TxType::DCT_FLIPADST | TxType::ADST_FLIPADST | TxType::H_FLIPADST =>
        (false, true),
      TxType::FLIPADST_FLIPADST => (true, true)
    }
  }
}

trait FwdTxfm2D: Dim {
  fn fwd_txfm2d(
    input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
    bd: usize
  ) {
    // TODO: Implement SSE version
    Self::fwd_txfm2d_rs(input, output, stride, tx_type, bd);
  }

  fn fwd_txfm2d_rs(
    input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
    bd: usize
  ) {
    let buf = &mut [0i32; 64 * 64][..Self::W * Self::H];
    let cfg = Txfm2DFlipCfg::fwd(tx_type, TxSize::by_dims(Self::W, Self::H));

    // Note when assigning txfm_size_col, we use the txfm_size from the
    // row configuration and vice versa. This is intentionally done to
    // accurately perform rectangular transforms. When the transform is
    // rectangular, the number of columns will be the same as the
    // txfm_size stored in the row cfg struct. It will make no difference
    // for square transforms.
    let txfm_size_col = TxSize::width(cfg.tx_size);
    let txfm_size_row = TxSize::height(cfg.tx_size);
    // Take the shift from the larger dimension in the rectangular case.
    assert!(cfg.stage_num_col <= MAX_TXFM_STAGE_NUM);
    assert!(cfg.stage_num_row <= MAX_TXFM_STAGE_NUM);
    let rect_type = get_rect_tx_log_ratio(txfm_size_col, txfm_size_row);
    let mut stage_range_col = [0; MAX_TXFM_STAGE_NUM];
    let mut stage_range_row = [0; MAX_TXFM_STAGE_NUM];
    av1_gen_fwd_stage_range(
      &mut stage_range_col,
      &mut stage_range_row,
      &cfg,
      bd as i8
    );

    let txfm_func_col = cfg.txfm_type_col.get_func();
    let txfm_func_row = cfg.txfm_type_row.get_func();

    // Columns
    for c in 0..txfm_size_col {
      if cfg.ud_flip {
        // flip upside down
        for r in 0..txfm_size_row {
          output[r] = (input[(txfm_size_row - r - 1) * stride + c]).into();
        }
      } else {
        for r in 0..txfm_size_row {
          output[r] = (input[r * stride + c]).into();
        }
      }
      av1_round_shift_array(output, txfm_size_row, -cfg.shift[0]);
      txfm_func_col(
        &output[..txfm_size_row].to_vec(),
        &mut output[txfm_size_row..],
        cfg.cos_bit_col as usize,
        &mut stage_range_col
      );
      av1_round_shift_array(
        &mut output[txfm_size_row..],
        txfm_size_row,
        -cfg.shift[1]
      );
      if cfg.lr_flip {
        for r in 0..txfm_size_row {
          // flip from left to right
          buf[r * txfm_size_col + (txfm_size_col - c - 1)] =
            output[txfm_size_row + r];
        }
      } else {
        for r in 0..txfm_size_row {
          buf[r * txfm_size_col + c] = output[txfm_size_row + r];
        }
      }
    }

    // Rows
    for r in 0..txfm_size_row {
      txfm_func_row(
        &buf[r * txfm_size_col..],
        &mut output[r * txfm_size_col..],
        cfg.cos_bit_row as usize,
        &mut stage_range_row
      );
      av1_round_shift_array(
        &mut output[r * txfm_size_col..],
        txfm_size_col,
        -cfg.shift[2]
      );
      if rect_type.abs() == 1 {
        // Multiply everything by Sqrt2 if the transform is rectangular and the
        // size difference is a factor of 2.
        for c in 0..txfm_size_col {
          output[r * txfm_size_col + c] =
            round_shift(output[r * txfm_size_col + c] * SQRT2, SQRT2_BITS);
        }
      }
    }
  }
}

macro_rules! impl_fwd_txs {
  ($(($W:expr, $H:expr)),+) => {
    $(
      paste::item! {
        impl FwdTxfm2D for [<Block $W x $H>] {}
      }
    )*
  }
}

impl_fwd_txs! { (4, 4), (8, 8), (16, 16), (32, 32), (64, 64) }
impl_fwd_txs! { (4, 8), (8, 16), (16, 32), (32, 64) }
impl_fwd_txs! { (8, 4), (16, 8), (32, 16), (64, 32) }
impl_fwd_txs! { (4, 16), (8, 32), (16, 64) }
impl_fwd_txs! { (16, 4), (32, 8), (64, 16) }

pub fn fht4x4(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  // SIMD code may assert for transform types beyond TxType::IDTX.
  if tx_type < TxType::IDTX {
    Block4x4::fwd_txfm2d(input, output, stride, tx_type, bit_depth);
  } else {
    Block4x4::fwd_txfm2d_rs(input, output, stride, tx_type, bit_depth);
  }
}

pub fn fht8x8(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  // SIMD code may assert for transform types beyond TxType::IDTX.
  if tx_type < TxType::IDTX {
    Block8x8::fwd_txfm2d(input, output, stride, tx_type, bit_depth);
  } else {
    Block8x8::fwd_txfm2d_rs(input, output, stride, tx_type, bit_depth);
  }
}

pub fn fht16x16(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  // SIMD code may assert for transform types beyond TxType::IDTX.
  if tx_type < TxType::IDTX {
    Block16x16::fwd_txfm2d(input, output, stride, tx_type, bit_depth);
  } else {
    Block16x16::fwd_txfm2d_rs(input, output, stride, tx_type, bit_depth);
  }
}

pub fn fht32x32(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  // SIMD code may assert for transform types that aren't TxType::DCT_DCT.
  if tx_type == TxType::DCT_DCT {
    Block32x32::fwd_txfm2d(input, output, stride, tx_type, bit_depth);
  } else {
    Block32x32::fwd_txfm2d_rs(input, output, stride, tx_type, bit_depth);
  }
}

pub fn fht64x64(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  assert!(tx_type == TxType::DCT_DCT);
  let mut tmp = [0 as i32; 4096];

  Block64x64::fwd_txfm2d(input, &mut tmp, stride, tx_type, bit_depth);

  for i in 0..2 {
    for (row_out, row_in) in output[2048*i..].chunks_mut(32).zip(tmp[32*i..].chunks(64)).take(64) {
      row_out.copy_from_slice(&row_in[..32]);
    }
  }
}

pub fn fht4x8(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  // SIMD code may assert for transform types beyond TxType::IDTX.
  if tx_type < TxType::IDTX {
    Block4x8::fwd_txfm2d(input, output, stride, tx_type, bit_depth);
  } else {
    Block4x8::fwd_txfm2d_rs(input, output, stride, tx_type, bit_depth);
  }
}

pub fn fht8x4(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  // SIMD code may assert for transform types beyond TxType::IDTX.
  if tx_type < TxType::IDTX {
    Block8x4::fwd_txfm2d(input, output, stride, tx_type, bit_depth);
  } else {
    Block8x4::fwd_txfm2d_rs(input, output, stride, tx_type, bit_depth);
  }
}

pub fn fht8x16(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  // SIMD code may assert for transform types beyond TxType::IDTX.
  if tx_type < TxType::IDTX {
    Block8x16::fwd_txfm2d(input, output, stride, tx_type, bit_depth);
  } else {
    Block8x16::fwd_txfm2d_rs(input, output, stride, tx_type, bit_depth);
  }
}

pub fn fht16x8(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  // SIMD code may assert for transform types beyond TxType::IDTX.
  if tx_type < TxType::IDTX {
    Block16x8::fwd_txfm2d(input, output, stride, tx_type, bit_depth);
  } else {
    Block16x8::fwd_txfm2d_rs(input, output, stride, tx_type, bit_depth);
  }
}

pub fn fht16x32(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
    assert!(tx_type == TxType::DCT_DCT);
    Block16x32::fwd_txfm2d(input, output, stride, tx_type, bit_depth);
}

pub fn fht32x16(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
    assert!(tx_type == TxType::DCT_DCT);
    Block32x16::fwd_txfm2d(input, output, stride, tx_type, bit_depth);
}

pub fn fht32x64(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  assert!(tx_type == TxType::DCT_DCT);
  let mut tmp = [0 as i32; 2048];

  Block32x64::fwd_txfm2d(input, &mut tmp, stride, tx_type, bit_depth);

  for (row_out, row_in) in output.chunks_mut(32).
    zip(tmp.chunks(32)).take(64) {
    row_out.copy_from_slice(&row_in[..32]);
  }
}

pub fn fht64x32(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  assert!(tx_type == TxType::DCT_DCT);
  let mut tmp = [0 as i32; 2048];

  Block64x32::fwd_txfm2d(input, &mut tmp, stride, tx_type, bit_depth);

  for i in 0..2 {
    for (row_out, row_in) in output[1024*i..].chunks_mut(32).
      zip(tmp[32*i..].chunks(64)).take(32) {
      row_out.copy_from_slice(&row_in[..32]);
    }
  }
}

pub fn fht4x16(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  // SIMD code may assert for transform types beyond TxType::IDTX.
  if tx_type < TxType::IDTX {
    Block4x16::fwd_txfm2d(input, output, stride, tx_type, bit_depth);
  } else {
    Block4x16::fwd_txfm2d_rs(input, output, stride, tx_type, bit_depth);
  }
}

pub fn fht16x4(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  // SIMD code may assert for transform types beyond TxType::IDTX.
  if tx_type < TxType::IDTX {
    Block16x4::fwd_txfm2d(input, output, stride, tx_type, bit_depth);
  } else {
    Block16x4::fwd_txfm2d_rs(input, output, stride, tx_type, bit_depth);
  }
}

pub fn fht8x32(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
    assert!(tx_type == TxType::DCT_DCT);
    Block8x32::fwd_txfm2d(input, output, stride, tx_type, bit_depth);
}
pub fn fht32x8(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  assert!(tx_type == TxType::DCT_DCT);
  Block32x8::fwd_txfm2d(input, output, stride, tx_type, bit_depth);
}

pub fn fht16x64(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  assert!(tx_type == TxType::DCT_DCT);
  let mut tmp = [0 as i32; 1024];

  Block16x64::fwd_txfm2d(input, &mut tmp, stride, tx_type, bit_depth);

  for (row_out, row_in) in output.chunks_mut(16).
    zip(tmp.chunks(16)).take(64) {
    row_out.copy_from_slice(&row_in[..16]);
  }
}

pub fn fht64x16(
  input: &[i16], output: &mut [i32], stride: usize, tx_type: TxType,
  bit_depth: usize
) {
  assert!(tx_type == TxType::DCT_DCT);
  let mut tmp = [0 as i32; 1024];

  Block64x16::fwd_txfm2d(input, &mut tmp, stride, tx_type, bit_depth);

  for i in 0..2 {
    for (row_out, row_in) in output[512*i..].chunks_mut(32).
      zip(tmp[32*i..].chunks(64)).take(16) {
      row_out.copy_from_slice(&row_in[..32]);
    }
  }
}
