// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use partition::TxSize;
use partition::TxType;
use partition::TX_TYPES;

use util::*;

pub use self::forward::*;
pub use self::inverse::*;

// Blocks
use predict::*;

mod forward;
mod inverse;

static SQRT2_BITS: usize = 12;
static SQRT2: i32 = 5793;

static COS_BIT_MIN: usize = 10;

// av1_cospi_arr[i][j] = (int)round(cos(M_PI*j/128) * (1<<(cos_bit_min+i)));
const AV1_COSPI_ARR_DATA: [[i32; 64]; 7] = [
  [
    1024, 1024, 1023, 1021, 1019, 1016, 1013, 1009, 1004, 999, 993, 987, 980,
    972, 964, 955, 946, 936, 926, 915, 903, 891, 878, 865, 851, 837, 822, 807,
    792, 775, 759, 742, 724, 706, 688, 669, 650, 630, 610, 590, 569, 548, 526,
    505, 483, 460, 438, 415, 392, 369, 345, 321, 297, 273, 249, 224, 200, 175,
    150, 125, 100, 75, 50, 25,
  ],
  [
    2048, 2047, 2046, 2042, 2038, 2033, 2026, 2018, 2009, 1998, 1987, 1974,
    1960, 1945, 1928, 1911, 1892, 1872, 1851, 1829, 1806, 1782, 1757, 1730,
    1703, 1674, 1645, 1615, 1583, 1551, 1517, 1483, 1448, 1412, 1375, 1338,
    1299, 1260, 1220, 1179, 1138, 1096, 1053, 1009, 965, 921, 876, 830, 784,
    737, 690, 642, 595, 546, 498, 449, 400, 350, 301, 251, 201, 151, 100, 50,
  ],
  [
    4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973, 3948,
    3920, 3889, 3857, 3822, 3784, 3745, 3703, 3659, 3612, 3564, 3513, 3461,
    3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967, 2896, 2824, 2751, 2675,
    2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019, 1931, 1842, 1751, 1660,
    1567, 1474, 1380, 1285, 1189, 1092, 995, 897, 799, 700, 601, 501, 401,
    301, 201, 101,
  ],
  [
    8192, 8190, 8182, 8170, 8153, 8130, 8103, 8071, 8035, 7993, 7946, 7895,
    7839, 7779, 7713, 7643, 7568, 7489, 7405, 7317, 7225, 7128, 7027, 6921,
    6811, 6698, 6580, 6458, 6333, 6203, 6070, 5933, 5793, 5649, 5501, 5351,
    5197, 5040, 4880, 4717, 4551, 4383, 4212, 4038, 3862, 3683, 3503, 3320,
    3135, 2948, 2760, 2570, 2378, 2185, 1990, 1795, 1598, 1401, 1202, 1003,
    803, 603, 402, 201,
  ],
  [
    16384, 16379, 16364, 16340, 16305, 16261, 16207, 16143, 16069, 15986,
    15893, 15791, 15679, 15557, 15426, 15286, 15137, 14978, 14811, 14635,
    14449, 14256, 14053, 13842, 13623, 13395, 13160, 12916, 12665, 12406,
    12140, 11866, 11585, 11297, 11003, 10702, 10394, 10080, 9760, 9434, 9102,
    8765, 8423, 8076, 7723, 7366, 7005, 6639, 6270, 5897, 5520, 5139, 4756,
    4370, 3981, 3590, 3196, 2801, 2404, 2006, 1606, 1205, 804, 402,
  ],
  [
    32768, 32758, 32729, 32679, 32610, 32522, 32413, 32286, 32138, 31972,
    31786, 31581, 31357, 31114, 30853, 30572, 30274, 29957, 29622, 29269,
    28899, 28511, 28106, 27684, 27246, 26791, 26320, 25833, 25330, 24812,
    24279, 23732, 23170, 22595, 22006, 21403, 20788, 20160, 19520, 18868,
    18205, 17531, 16846, 16151, 15447, 14733, 14010, 13279, 12540, 11793,
    11039, 10279, 9512, 8740, 7962, 7180, 6393, 5602, 4808, 4011, 3212, 2411,
    1608, 804,
  ],
  [
    65536, 65516, 65457, 65358, 65220, 65043, 64827, 64571, 64277, 63944,
    63572, 63162, 62714, 62228, 61705, 61145, 60547, 59914, 59244, 58538,
    57798, 57022, 56212, 55368, 54491, 53581, 52639, 51665, 50660, 49624,
    48559, 47464, 46341, 45190, 44011, 42806, 41576, 40320, 39040, 37736,
    36410, 35062, 33692, 32303, 30893, 29466, 28020, 26558, 25080, 23586,
    22078, 20557, 19024, 17479, 15924, 14359, 12785, 11204, 9616, 8022, 6424,
    4821, 3216, 1608,
  ]
];

// av1_sinpi_arr_data[i][j] = (int)round((sqrt(2) * sin(j*Pi/9) * 2 / 3) * (1
// << (cos_bit_min + i))) modified so that elements j=1,2 sum to element j=4.
const AV1_SINPI_ARR_DATA: [[i32; 5]; 7] = [
  [0, 330, 621, 836, 951],
  [0, 660, 1241, 1672, 1901],
  [0, 1321, 2482, 3344, 3803],
  [0, 2642, 4964, 6689, 7606],
  [0, 5283, 9929, 13377, 15212],
  [0, 10566, 19858, 26755, 30424],
  [0, 21133, 39716, 53510, 60849]
];

#[inline]
fn cospi_arr(n: usize) -> &'static [i32; 64] {
  &AV1_COSPI_ARR_DATA[n - COS_BIT_MIN]
}

#[inline]
fn sinpi_arr(n: usize) -> &'static [i32; 5] {
  &AV1_SINPI_ARR_DATA[n - COS_BIT_MIN]
}

/// Utility function that returns the log of the ratio of the col and row sizes.
#[inline]
fn get_rect_tx_log_ratio(col: usize, row: usize) -> i8 {
  if col == row {
    return 0;
  }
  if col > row {
    if col == row * 2 {
      return 1;
    }
    if col == row * 4 {
      return 2;
    }
  }
  if row == col * 2 {
    return -1;
  }
  if row == col * 4 {
    return -2;
  }

  panic!("Unsupported transform size");
}

// performs half a butterfly
#[inline]
fn half_btf(w0: i32, in0: i32, w1: i32, in1: i32, bit: usize) -> i32 {
  let result = (w0 * in0) + (w1 * in1);
  round_shift(result, bit)
}

#[inline]
fn round_shift(value: i32, bit: usize) -> i32 {
  if bit <= 0 {
    value
  } else {
    (value + (1 << (bit - 1))) >> bit
  }
}

// clamps value to a signed integer type of bit bits
#[inline]
fn clamp_value(value: i32, bit: usize) -> i32 {
  let max_value: i32 = ((1i64 << (bit - 1)) - 1) as i32;
  let min_value: i32 = (-(1i64 << (bit - 1))) as i32;
  clamp(value, min_value, max_value)
}

pub fn av1_round_shift_array(arr: &mut [i32], size: usize, bit: i8) {
  // FIXME
  //  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  //      {
  //        if is_x86_feature_detected!("sse4.1") {
  //          return unsafe {
  //            x86_asm::av1_round_shift_array_sse4_1(arr, size, bit)
  //          };
  //        }
  //      }
  av1_round_shift_array_rs(arr, size, bit)
}

fn av1_round_shift_array_rs(arr: &mut [i32], size: usize, bit: i8) {
  if bit == 0 {
    return;
  }
  if bit > 0 {
    let bit = bit as usize;
    for i in 0..size {
      arr[i] = round_shift(arr[i], bit);
    }
  } else {
    for i in 0..size {
      arr[i] =
        clamp((1 << (-bit)) * arr[i], i32::min_value(), i32::max_value());
    }
  }
}

#[derive(Debug, Clone, Copy)]
enum TxType1D {
  DCT,
  ADST,
  FLIPADST,
  IDTX
}

// Option can be removed when the table is completely filled
fn get_1d_tx_types(tx_type: TxType) -> Option<(TxType1D, TxType1D)> {
  match tx_type {
    TxType::DCT_DCT => Some((TxType1D::DCT, TxType1D::DCT)),
    TxType::ADST_DCT => Some((TxType1D::ADST, TxType1D::DCT)),
    TxType::DCT_ADST => Some((TxType1D::DCT, TxType1D::ADST)),
    TxType::ADST_ADST => Some((TxType1D::ADST, TxType1D::ADST)),
    TxType::IDTX => Some((TxType1D::IDTX, TxType1D::IDTX)),
    TxType::V_DCT => Some((TxType1D::DCT, TxType1D::IDTX)),
    TxType::H_DCT => Some((TxType1D::IDTX, TxType1D::DCT)),
    TxType::V_ADST => Some((TxType1D::ADST, TxType1D::IDTX)),
    TxType::H_ADST => Some((TxType1D::IDTX, TxType1D::ADST)),
    _ => None
  }
}

const VTX_TAB: [TxType1D; TX_TYPES] = [
  TxType1D::DCT,
  TxType1D::ADST,
  TxType1D::DCT,
  TxType1D::ADST,
  TxType1D::FLIPADST,
  TxType1D::DCT,
  TxType1D::FLIPADST,
  TxType1D::ADST,
  TxType1D::FLIPADST,
  TxType1D::IDTX,
  TxType1D::DCT,
  TxType1D::IDTX,
  TxType1D::ADST,
  TxType1D::IDTX,
  TxType1D::FLIPADST,
  TxType1D::IDTX
];

const HTX_TAB: [TxType1D; TX_TYPES] = [
  TxType1D::DCT,
  TxType1D::DCT,
  TxType1D::ADST,
  TxType1D::ADST,
  TxType1D::DCT,
  TxType1D::FLIPADST,
  TxType1D::FLIPADST,
  TxType1D::FLIPADST,
  TxType1D::ADST,
  TxType1D::IDTX,
  TxType1D::IDTX,
  TxType1D::DCT,
  TxType1D::IDTX,
  TxType1D::ADST,
  TxType1D::IDTX,
  TxType1D::FLIPADST
];

pub fn forward_transform(
  input: &[i16], output: &mut [i32], stride: usize, tx_size: TxSize,
  tx_type: TxType, bit_depth: usize
) {
  match tx_size {
    TxSize::TX_4X4 => fht4x4(input, output, stride, tx_type, bit_depth),
    TxSize::TX_8X8 => fht8x8(input, output, stride, tx_type, bit_depth),
    TxSize::TX_16X16 => fht16x16(input, output, stride, tx_type, bit_depth),
    TxSize::TX_32X32 => fht32x32(input, output, stride, tx_type, bit_depth),
    _ => panic!("unimplemented tx size")
  }
}

pub fn inverse_transform_add(
  input: &[i32], output: &mut [u16], stride: usize, tx_size: TxSize,
  tx_type: TxType, bit_depth: usize
) {
  match tx_size {
    TxSize::TX_4X4 => iht4x4_add(input, output, stride, tx_type, bit_depth),
    TxSize::TX_8X8 => iht8x8_add(input, output, stride, tx_type, bit_depth),
    TxSize::TX_16X16 =>
      iht16x16_add(input, output, stride, tx_type, bit_depth),
    TxSize::TX_32X32 =>
      iht32x32_add(input, output, stride, tx_type, bit_depth),
    _ => panic!("unimplemented tx size")
  }
}
