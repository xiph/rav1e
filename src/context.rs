#![allow(safe_extern_statics)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]
#![allow(non_camel_case_types)]

use std::*;
use ec;
use partition::*;
use partition::BlockSize::*;
use partition::TxSize::*;
use partition::TxType::*;
use partition::PredictionMode::*;
use plane::*;
use libc::*;

const PLANES: usize = 3;

const PARTITION_PLOFFSET: usize = 4;
const PARTITION_CONTEXTS: usize = 20;
pub const PARTITION_TYPES: usize = 4;

pub const MI_SIZE_LOG2: usize = 2;
const MI_SIZE: usize = (1 << MI_SIZE_LOG2);
const MAX_MIB_SIZE_LOG2: usize = (MAX_SB_SIZE_LOG2 - MI_SIZE_LOG2);
pub const MAX_MIB_SIZE: usize = (1 << MAX_MIB_SIZE_LOG2);
pub const MAX_MIB_MASK: usize = (MAX_MIB_SIZE - 1);

const MAX_SB_SIZE_LOG2: usize = 6;
const MAX_SB_SIZE: usize = (1 << MAX_SB_SIZE_LOG2);
const MAX_SB_SQUARE: usize = (MAX_SB_SIZE * MAX_SB_SIZE);

const MAX_TX_SIZE: usize = 32;
const MAX_TX_SQUARE: usize = MAX_TX_SIZE * MAX_TX_SIZE;

const INTRA_MODES: usize = 13;
const UV_INTRA_MODES: usize = 14;
const BLOCK_SIZE_GROUPS: usize = 4;
const MAX_ANGLE_DELTA: usize = 3;
const DIRECTIONAL_MODES: usize = 8;

pub static mi_size_wide: [u8; BLOCK_SIZES_ALL] =
    [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 1, 4, 2, 8, 4, 16];
pub static mi_size_high: [u8; BLOCK_SIZES_ALL] =
    [1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 4, 1, 8, 2, 16, 4];
pub static b_width_log2_lookup: [u8; BLOCK_SIZES_ALL] =
    [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 0, 2, 1, 3, 2, 4];
pub static b_height_log2_lookup: [u8; BLOCK_SIZES_ALL] =
    [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 2, 0, 3, 1, 4, 2];
pub static tx_size_wide_log2: [usize; TX_SIZES_ALL] =
    [2, 3, 4, 5, 2, 3, 3, 4, 4, 5, 2, 4, 3, 5];
pub static tx_size_high_log2: [usize; TX_SIZES_ALL] =
    [2, 3, 4, 5, 3, 2, 4, 3, 5, 4, 4, 2, 5, 3];
// Transform block width in pixels
pub static tx_size_wide: [usize; TX_SIZES_ALL] =
    [ 4, 8, 16, 32, 4, 8, 8, 16, 16, 32, 4, 16, 8, 32 ];
// Transform block height in pixels
pub static tx_size_high: [usize; TX_SIZES_ALL] =
    [ 4, 8, 16, 32, 8, 4, 16, 8, 32, 16, 16, 4, 32, 8 ];
// Transform block width in unit
pub static tx_size_wide_unit: [usize; TX_SIZES_ALL] =
    [1, 2, 4, 8, 1, 2, 2, 4, 4, 8, 1, 4, 2, 8];
// Transform block height in unit
pub static tx_size_high_unit: [usize; TX_SIZES_ALL] =
    [1, 2, 4, 8, 2, 1, 4, 2, 8, 4, 4, 1, 8, 2];
// Width/height lookup tables in units of various block sizes
pub static block_size_wide: [u8; BLOCK_SIZES_ALL] =
    [4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 4, 16, 8, 32, 16, 64 ];
pub static block_size_high: [u8; BLOCK_SIZES_ALL] =
    [4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 16,4, 32, 8, 64, 16 ];

const EXT_TX_SIZES: usize = 4;
const EXT_TX_SET_TYPES: usize = 9;
const EXT_TX_SETS_INTRA: usize = 6;
const EXT_TX_SETS_INTER: usize = 4;
// Number of transform types in each set type
static num_ext_tx_set: [usize; EXT_TX_SET_TYPES] = [ 1, 2, 5, 7, 7, 10, 12, 16, 16];
static av1_ext_tx_used: [[usize; TX_TYPES]; EXT_TX_SET_TYPES] = [
    [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ],
    [ 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ],
    [ 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0 ],
    [ 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0 ],
    [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 ],
    [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0 ],
    [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
    [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ] ];
// Maps intra set index to the set type
/*static ext_tx_set_type_intra: [TxSetType; EXT_TX_SETS_INTRA] = [
    TxSetType::EXT_TX_SET_DCTONLY,
    TxSetType::EXT_TX_SET_DTT4_IDTX_1DDCT,
    TxSetType::EXT_TX_SET_DTT4_IDTX
];*/
// Maps inter set index to the set type
#[allow(dead_code)]
static ext_tx_set_type_inter: [TxSetType; EXT_TX_SETS_INTER] = [
    TxSetType::EXT_TX_SET_DCTONLY,
    TxSetType::EXT_TX_SET_ALL16,
    TxSetType::EXT_TX_SET_DTT9_IDTX_1DDCT,
    TxSetType::EXT_TX_SET_DCT_IDTX
];
// Maps set types above to the indices used for intra
static ext_tx_set_index_intra: [i8; EXT_TX_SET_TYPES] = [0, -1, 2, -1, 1, -1, -1, -1, -1];
// Maps set types above to the indices used for inter
static ext_tx_set_index_inter: [i8; EXT_TX_SET_TYPES] = [0, 3, -1, -1, -1, -1, 2, -1, 1];
static av1_ext_tx_intra_ind: [[u32; TX_TYPES]; EXT_TX_SETS_INTRA] =
    [
      [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
      [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
      [ 1, 3, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
      [ 1, 5, 6, 4, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0 ],
      [ 3, 4, 5, 8, 6, 7, 9, 10, 11, 0, 1, 2, 0, 0, 0, 0 ],
      [ 7, 8, 9, 12, 10, 11, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6 ],
    ];
#[allow(dead_code)]
static av1_ext_tx_inter_ind: [[usize; TX_TYPES]; EXT_TX_SETS_INTER] =
    [
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [1,5,6,4,0,0,0,0,0,0,2,3,0,0,0,0,],
        [1,3,4,2,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,],
    ];
//static ext_tx_cnt_intra: [usize;EXT_TX_SETS_INTRA] = [ 1, 7, 5 ];

static av1_ext_tx_ind: [[usize; TX_TYPES]; EXT_TX_SET_TYPES] = [ 
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],  
    [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],  
    [ 1, 3, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],  
    [ 1, 5, 6, 4, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0 ],  
    [ 1, 5, 6, 4, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0 ],  
    [ 1, 2, 3, 6, 4, 5, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0 ],  
    [ 3, 4, 5, 8, 6, 7, 9, 10, 11, 0, 1, 2, 0, 0, 0, 0 ],  
    [ 7, 8, 9, 12, 10, 11, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6 ],  
    [ 7, 8, 9, 12, 10, 11, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6 ],  
];


static av1_coefband_trans_4x4: [u8; 16] = [
    0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5,
];

static av1_coefband_trans_8x8plus: [u8; 32*32] = [
  0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5,
  // beyond MAXBAND_INDEX+1 all values are filled as 5
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5];

pub static txsize_to_bsize: [BlockSize; TX_SIZES_ALL] = [
  BLOCK_4X4,    // TX_4X4
  BLOCK_8X8,    // TX_8X8
  BLOCK_16X16,  // TX_16X16
  BLOCK_32X32,  // TX_32X32
  BLOCK_4X8,    // TX_4X8
  BLOCK_8X4,    // TX_8X4
  BLOCK_8X16,   // TX_8X16
  BLOCK_16X8,   // TX_16X8
  BLOCK_16X32,  // TX_16X32
  BLOCK_32X16,  // TX_32X16
  BLOCK_4X16,   // TX_4X16
  BLOCK_16X4,   // TX_16X4
  BLOCK_8X32,   // TX_8X32
  BLOCK_32X8,   // TX_32X8
];

static TXSIZE_SQR_MAP: [TxSize; TX_SIZES_ALL] = [
    TX_4X4,
    TX_8X8,
    TX_16X16,
    TX_32X32,
    TX_4X4,
    TX_4X4,
    TX_8X8,
    TX_8X8,
    TX_16X16,
    TX_16X16,
    TX_4X4,
    TX_4X4,
    TX_8X8,
    TX_8X8,
];

static TXSIZE_SQR_UP_MAP: [TxSize; TX_SIZES_ALL] = [
    TX_4X4,
    TX_8X8,
    TX_16X16,
    TX_32X32,
    TX_8X8,
    TX_8X8,
    TX_16X16,
    TX_16X16,
    TX_32X32,
    TX_32X32,
    TX_16X16,
    TX_16X16,
    TX_32X32,
    TX_32X32,
];

static txsize_log2_minus4: [usize; TX_SIZES_ALL] = [
    0,  // TX_4X4
    2,  // TX_8X8
    4,  // TX_16X16
    6,  // TX_32X32
    6,  // TX_64X64
    1,  // TX_4X8
    1,  // TX_8X4
    3,  // TX_8X16
    3,  // TX_16X8
    5,  // TX_16X32
    5,  // TX_32X16
    6,  // TX_32X64
    6,  // TX_64X32
    2,  // TX_4X16
];

static ss_size_lookup: [[[BlockSize; 2]; 2]; BLOCK_SIZES_ALL] = [
  //  ss_x == 0    ss_x == 0        ss_x == 1      ss_x == 1
  //  ss_y == 0    ss_y == 1        ss_y == 0      ss_y == 1
  [  [ BLOCK_4X4, BLOCK_4X4 ], [BLOCK_4X4, BLOCK_4X4 ] ],
  [  [ BLOCK_4X8, BLOCK_4X4 ], [BLOCK_4X4, BLOCK_4X4 ] ],
  [  [ BLOCK_8X4, BLOCK_4X4 ], [BLOCK_4X4, BLOCK_4X4 ] ],
  [  [ BLOCK_8X8, BLOCK_8X4 ], [BLOCK_4X8, BLOCK_4X4 ] ],
  [  [ BLOCK_8X16, BLOCK_8X8 ], [BLOCK_4X16, BLOCK_4X8 ] ],
  [  [ BLOCK_16X8, BLOCK_16X4 ], [BLOCK_8X8, BLOCK_8X4 ] ],
  [  [ BLOCK_16X16, BLOCK_16X8 ], [BLOCK_8X16, BLOCK_8X8 ] ],
  [  [ BLOCK_16X32, BLOCK_16X16 ], [BLOCK_8X32, BLOCK_8X16 ] ],
  [  [ BLOCK_32X16, BLOCK_32X8 ], [BLOCK_16X16, BLOCK_16X8 ] ],
  [  [ BLOCK_32X32, BLOCK_32X16 ], [BLOCK_16X32, BLOCK_16X16 ] ],
  [  [ BLOCK_32X64, BLOCK_32X32 ], [BLOCK_16X64, BLOCK_16X32 ] ],
  [  [ BLOCK_64X32, BLOCK_64X16 ], [BLOCK_32X32, BLOCK_32X16 ] ],
  [  [ BLOCK_64X64, BLOCK_64X32 ], [BLOCK_32X64, BLOCK_32X32 ] ],
  [  [ BLOCK_4X16, BLOCK_4X8 ], [BLOCK_4X16, BLOCK_4X8 ] ],
  [  [ BLOCK_16X4, BLOCK_16X4 ], [BLOCK_8X4, BLOCK_8X4 ] ],
  [  [ BLOCK_8X32, BLOCK_8X16 ], [BLOCK_INVALID, BLOCK_4X16 ] ],
  [  [ BLOCK_32X8, BLOCK_INVALID ], [BLOCK_16X8, BLOCK_16X4 ] ],
  [  [ BLOCK_16X64, BLOCK_16X32 ], [BLOCK_INVALID, BLOCK_8X32 ] ],
  [  [ BLOCK_64X16, BLOCK_INVALID ], [BLOCK_32X16, BLOCK_32X8 ] ],
];

pub fn get_plane_block_size(bsize: BlockSize, subsampling_x: usize, subsampling_y: usize)
    -> BlockSize {
  ss_size_lookup[bsize as usize][subsampling_x][subsampling_y]
}

// Generates 4 bit field in which each bit set to 1 represents
// a blocksize partition  1111 means we split 64x64, 32x32, 16x16
// and 8x8.  1000 means we just split the 64x64 to 32x32
static partition_context_lookup: [[u8; 2]; BLOCK_SIZES_ALL] = [
  [ 15, 15 ],  // 4X4   - [0b1111, 0b1111]
  [ 15, 14 ],  // 4X8   - [0b1111, 0b1110]
  [ 14, 15 ],  // 8X4   - [0b1110, 0b1111]
  [ 14, 14 ],  // 8X8   - [0b1110, 0b1110]
  [ 14, 12 ],  // 8X16  - [0b1110, 0b1100]
  [ 12, 14 ],  // 16X8  - [0b1100, 0b1110]
  [ 12, 12 ],  // 16X16 - [0b1100, 0b1100]
  [ 12, 8 ],   // 16X32 - [0b1100, 0b1000]
  [ 8, 12 ],   // 32X16 - [0b1000, 0b1100]
  [ 8, 8 ],    // 32X32 - [0b1000, 0b1000]
  [ 8, 0 ],    // 32X64 - [0b1000, 0b0000]
  [ 0, 8 ],    // 64X32 - [0b0000, 0b1000]
  [ 0, 0 ],    // 64X64 - [0b0000, 0b0000]

  [ 15, 12 ],  // 4X16 - [0b1111, 0b1100]
  [ 12, 15 ],  // 16X4 - [0b1100, 0b1111]
  [ 8, 14 ],   // 8X32 - [0b1110, 0b1000]
  [ 14, 8 ],   // 32X8 - [0b1000, 0b1110]
  [ 12, 0 ],   // 16X64- [0b1100, 0b0000]
  [ 0, 12 ],   // 64X16- [0b0000, 0b1100]
];

static size_group_lookup: [u8; BLOCK_SIZES_ALL] = [
  0, 0,
  0, 1,
  1, 1,
  2, 2,
  2, 3,
  3, 3,
  3, 0,
  0, 1,
  1, 2,
  2,
];

static num_pels_log2_lookup: [u8; BLOCK_SIZES_ALL] = [
  4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 6, 6, 8, 8, 10, 10];

pub static subsize_lookup: [[BlockSize; BLOCK_SIZES_ALL]; PARTITION_TYPES] =
[
  [ // PARTITION_NONE
    //                            4X4
                                  BLOCK_4X4,
    // 4X8,        8X4,           8X8
    BLOCK_4X8,     BLOCK_8X4,     BLOCK_8X8,
    // 8X16,       16X8,          16X16
    BLOCK_8X16,    BLOCK_16X8,    BLOCK_16X16,
    // 16X32,      32X16,         32X32
    BLOCK_16X32,   BLOCK_32X16,   BLOCK_32X32,
    // 32X64,      64X32,         64X64
    BLOCK_32X64,   BLOCK_64X32,   BLOCK_64X64,
    // 4X16,       16X4,          8X32
    BLOCK_4X16,    BLOCK_16X4,    BLOCK_8X32,
    // 32X8,       16X64,         64X16
    BLOCK_32X8,    BLOCK_16X64,   BLOCK_64X16,
  ], [  // PARTITION_HORZ
    //                            4X4
                                  BLOCK_INVALID,
    // 4X8,        8X4,           8X8
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_8X4,
    // 8X16,       16X8,          16X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_16X8,
    // 16X32,      32X16,         32X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_32X16,
    // 32X64,      64X32,         64X64
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_64X32,
    // 4X16,       16X4,          8X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
    // 32X8,       16X64,         64X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
  ], [  // PARTITION_VERT
    //                            4X4
                                  BLOCK_INVALID,
    // 4X8,        8X4,           8X8
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_4X8,
    // 8X16,       16X8,          16X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_8X16,
    // 16X32,      32X16,         32X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_16X32,
    // 32X64,      64X32,         64X64
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_32X64,
    // 4X16,       16X4,          8X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
    // 32X8,       16X64,         64X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
  ], [  // PARTITION_SPLIT
    //                            4X4
                                  BLOCK_INVALID,
    // 4X8,        8X4,           8X8
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_4X4,
    // 8X16,       16X8,          16X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_8X8,
    // 16X32,      32X16,         32X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_16X16,
    // 32X64,      64X32,         64X64
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_32X32,
    // 4X16,       16X4,          8X32
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
    // 32X8,       16X64,         64X16
    BLOCK_INVALID, BLOCK_INVALID, BLOCK_INVALID,
  ]
];

#[derive(Copy,Clone,PartialEq)]
#[allow(dead_code)]
enum HeadToken {
    BlockZero = 0,
    Zero = 1,
    OneEOB = 2,
    OneNEOB = 3,
    TwoPlusEOB = 4,
    TwoPlusNEOB = 5,
}

#[derive(Copy,Clone,PartialEq)]
#[allow(dead_code)]
enum TailToken {
    Two = 0,
    Three = 1,
    Four = 2,
    Cat1 = 3,
    Cat2 = 4,
    Cat3 = 5,
    Cat4 = 6,
    Cat5 = 7,
    Cat6 = 8,
}

const PLANE_TYPES: usize = 2;
const HEAD_TOKENS: usize = 5;
const TAIL_TOKENS: usize = 9;
const ENTROPY_TOKENS: usize = 12;
const COEFF_CONTEXTS: usize = 6;
const COEF_BANDS: usize = 6;
const REF_TYPES: usize = 2;
const SKIP_CONTEXTS: usize = 3;
const INTRA_INTER_CONTEXTS: usize = 4;

// Level Map
const TXB_SKIP_CONTEXTS: usize =  13;

const EOB_COEF_CONTEXTS: usize =  22;

const SIG_COEF_CONTEXTS_2D: usize =  26;
const SIG_COEF_CONTEXTS_1D: usize =  16;
const SIG_COEF_CONTEXTS_EOB: usize =  4;
const SIG_COEF_CONTEXTS: usize = SIG_COEF_CONTEXTS_2D + SIG_COEF_CONTEXTS_1D;

const COEFF_BASE_CONTEXTS: usize = SIG_COEF_CONTEXTS;
const DC_SIGN_CONTEXTS: usize =  3;

const BR_TMP_OFFSET: usize =  12;
const BR_REF_CAT: usize =  4;
const LEVEL_CONTEXTS: usize =  21;

const NUM_BASE_LEVELS: usize =  2;

const BR_CDF_SIZE: usize = 4;
const COEFF_BASE_RANGE: usize = 4 * (BR_CDF_SIZE - 1);

const COEFF_CONTEXT_BITS: usize = 6;
const COEFF_CONTEXT_MASK: usize = (1 << COEFF_CONTEXT_BITS) - 1;
const MAX_BASE_BR_RANGE: usize = COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1;

const BASE_CONTEXT_POSITION_NUM: usize = 12;

// Pad 4 extra columns to remove horizontal availability check.
const TX_PAD_HOR_LOG2: usize = 2;
const TX_PAD_HOR: usize = 4;
// Pad 6 extra rows (2 on top and 4 on bottom) to remove vertical availability
// check.
const TX_PAD_TOP: usize = 2;
const TX_PAD_BOTTOM: usize = 4;
const TX_PAD_VER: usize = (TX_PAD_TOP + TX_PAD_BOTTOM);
// Pad 16 extra bytes to avoid reading overflow in SIMD optimization.
const TX_PAD_END: usize = 16;
const TX_PAD_2D: usize = ((MAX_TX_SIZE + TX_PAD_HOR) * (MAX_TX_SIZE + TX_PAD_VER) + TX_PAD_END);

const TX_CLASSES: usize = 3;

#[derive(Copy,Clone,PartialEq)]
pub enum TxClass {
  TX_CLASS_2D = 0,
  TX_CLASS_HORIZ = 1,
  TX_CLASS_VERT = 2,
}

use context::TxClass::*;

static tx_type_to_class: [TxClass; TX_TYPES] = [
    TX_CLASS_2D,     // DCT_DCT
    TX_CLASS_2D,     // ADST_DCT
    TX_CLASS_2D,     // DCT_ADST
    TX_CLASS_2D,     // ADST_ADST
    TX_CLASS_2D,     // FLIPADST_DCT
    TX_CLASS_2D,     // DCT_FLIPADST
    TX_CLASS_2D,     // FLIPADST_FLIPADST
    TX_CLASS_2D,     // ADST_FLIPADST
    TX_CLASS_2D,     // FLIPADST_ADST
    TX_CLASS_2D,     // IDTX
    TX_CLASS_VERT,   // V_DCT
    TX_CLASS_HORIZ,  // H_DCT
    TX_CLASS_VERT,   // V_ADST
    TX_CLASS_HORIZ,  // H_ADST
    TX_CLASS_VERT,   // V_FLIPADST
    TX_CLASS_HORIZ,  // H_FLIPADST
];

static eob_to_pos_small: [u8; 33] = [
    0, 1, 2,                                        // 0-2
    3, 3,                                           // 3-4
    4, 4, 4, 4,                                     // 5-8
    5, 5, 5, 5, 5, 5, 5, 5,                         // 9-16
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6  // 17-32
];

static eob_to_pos_large: [u8; 17] = [
    6,                               // place holder
    7,                               // 33-64
    8,  8,                           // 65-128
    9,  9,  9,  9,                   // 129-256
    10, 10, 10, 10, 10, 10, 10, 10,  // 257-512
    11                               // 513-
];


static k_eob_group_start: [u16; 12] = [ 0, 1, 2, 3, 5, 9,
                                        17, 33, 65, 129, 257, 513 ];
static k_eob_offset_bits: [u16; 12] = [ 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ];

static clip_max3: [u8; 256] = [
  0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 ];

// The ctx offset table when TX is TX_CLASS_2D.
// TX col and row indices are clamped to 4

static av1_nz_map_ctx_offset_4x4: [i8; 16] = [
  0, 1, 6, 6, 1, 6, 6, 21, 6, 6, 21, 21, 6, 21, 21, 21 ];

static av1_nz_map_ctx_offset_8x8: [i8; 64] = [
  0,  1,  6,  6,  21, 21, 21, 21, 1,  6,  6,  21, 21, 21, 21, 21,
  6,  6,  21, 21, 21, 21, 21, 21, 6,  21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 ];

static av1_nz_map_ctx_offset_16x16: [i8; 256] = [
  0,  1,  6,  6,  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 1,  6,  6,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 6,  6,  21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 6,  21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21 ];

static av1_nz_map_ctx_offset_32x32: [i8; 1024] = [
  0,  1,  6,  6,  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 1,  6,  6,  21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 6,  6,  21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 6,  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 ];

static av1_nz_map_ctx_offset_8x4: [i8; 32] = [
  0,  16, 6,  6,  21, 21, 21, 21, 16, 16, 6,  21, 21, 21, 21, 21,
  16, 16, 21, 21, 21, 21, 21, 21, 16, 16, 21, 21, 21, 21, 21, 21 ];

static av1_nz_map_ctx_offset_8x16: [i8; 128] = [
  0,  11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 6,  6,  21,
  21, 21, 21, 21, 21, 6,  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 ];

static av1_nz_map_ctx_offset_16x8: [i8; 128] = [
  0,  16, 6,  6,  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 16, 16, 6,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 16, 16, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 16, 16, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 16, 16, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 16, 16, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 16, 16, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 16, 16,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 ];

static av1_nz_map_ctx_offset_16x32: [i8; 512] = [
  0,  11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
  11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 6,  6,  21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 6,  21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 ];

static av1_nz_map_ctx_offset_32x16: [i8; 512] = [
  0,  16, 6,  6,  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 16, 16, 6,  21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 16, 16, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 16, 16, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 16, 16, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 16, 16, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 16, 16, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 16, 16, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 16, 16, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 16, 16, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 16, 16, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 16, 16, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 16, 16, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 16, 16,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 16, 16, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 16, 16, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 ];

static av1_nz_map_ctx_offset_4x16: [i8; 64] = [
  0,  11, 11, 11, 11, 11, 11, 11, 6,  6,  21, 21, 6,  21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 ];

static av1_nz_map_ctx_offset_16x4: [i8; 64] = [
  0,  16, 6,  6,  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  16, 16, 6,  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  16, 16, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  16, 16, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21 ];

static av1_nz_map_ctx_offset_8x32: [i8; 256] = [
  0,  11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 6,  6,  21,
  21, 21, 21, 21, 21, 6,  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21 ];

static av1_nz_map_ctx_offset_32x8: [i8; 256] = [
  0,  16, 6,  6,  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 16, 16, 6,  21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 16, 16, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 16, 16, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 16, 16, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 16, 16, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 16, 16, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 16, 16, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21 ];

static av1_nz_map_ctx_offset: [&[i8]; TX_SIZES_ALL] = [
  &av1_nz_map_ctx_offset_4x4,    // TX_4x4
  &av1_nz_map_ctx_offset_8x8,    // TX_8x8
  &av1_nz_map_ctx_offset_16x16,  // TX_16x16
  &av1_nz_map_ctx_offset_32x32,  // TX_32x32
  //&av1_nz_map_ctx_offset_32x32,  // TX_32x32
  &av1_nz_map_ctx_offset_4x16,   // TX_4x8
  &av1_nz_map_ctx_offset_8x4,    // TX_8x4
  &av1_nz_map_ctx_offset_8x32,   // TX_8x16
  &av1_nz_map_ctx_offset_16x8,   // TX_16x8
  &av1_nz_map_ctx_offset_16x32,  // TX_16x32
  &av1_nz_map_ctx_offset_32x16,  // TX_32x16
  //&av1_nz_map_ctx_offset_32x64,  // TX_32x64
  //&av1_nz_map_ctx_offset_64x32,  // TX_64x32
  &av1_nz_map_ctx_offset_4x16,   // TX_4x16
  &av1_nz_map_ctx_offset_16x4,   // TX_16x4
  &av1_nz_map_ctx_offset_8x32,   // TX_8x32
  &av1_nz_map_ctx_offset_32x8,   // TX_32x8
  //&av1_nz_map_ctx_offset_16x32,  // TX_16x64
  //&av1_nz_map_ctx_offset_64x32,  // TX_64x16
];

const NZ_MAP_CTX_0: usize = SIG_COEF_CONTEXTS_2D;
const NZ_MAP_CTX_5: usize = (NZ_MAP_CTX_0 + 5);
const NZ_MAP_CTX_10: usize = (NZ_MAP_CTX_0 + 10);

static nz_map_ctx_offset_1d: [usize; 32] = [
  NZ_MAP_CTX_0,  NZ_MAP_CTX_5,  NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10,
  NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10,
  NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10,
  NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10,
  NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10,
  NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10, NZ_MAP_CTX_10,
  NZ_MAP_CTX_10, NZ_MAP_CTX_10 ];

const CONTEXT_MAG_POSITION_NUM: usize = 3;

static mag_ref_offset_with_txclass: [[[usize; 2]; CONTEXT_MAG_POSITION_NUM]; 3] = [
  [ [ 0, 1 ], [ 1, 0 ], [ 1, 1 ] ],
  [ [ 0, 1 ], [ 1, 0 ], [ 0, 2 ] ],
  [ [ 0, 1 ], [ 1, 0 ], [ 2, 0 ] ] ];

// End of Level Map

pub fn clamp(val: i32, min: i32, max: i32) -> i32 {
    if val < min {
        min
    }
    else if val > max {
        max
    }
    else {
        val
    }
}

fn get_ext_tx_set_type(tx_size: TxSize, is_inter: bool, use_reduced_set: bool) -> TxSetType {
    let tx_size_sqr_up = TXSIZE_SQR_UP_MAP[tx_size as usize];
    let tx_size_sqr = TXSIZE_SQR_MAP[tx_size as usize];
    if tx_size_sqr > TxSize::TX_32X32 {
        TxSetType::EXT_TX_SET_DCTONLY
    } else if tx_size_sqr_up == TxSize::TX_32X32 {
        if is_inter {
            TxSetType::EXT_TX_SET_DCT_IDTX
        } else {
            TxSetType::EXT_TX_SET_DCTONLY
        }
    } else if use_reduced_set {
        if is_inter {
            TxSetType::EXT_TX_SET_DCT_IDTX
        } else {
            TxSetType::EXT_TX_SET_DTT4_IDTX
        }
    } else if is_inter {
        if tx_size_sqr == TxSize::TX_16X16 {
            TxSetType::EXT_TX_SET_DTT9_IDTX_1DDCT
        } else {
            TxSetType::EXT_TX_SET_ALL16
        }
    } else {
        if tx_size_sqr == TxSize::TX_16X16 {
            TxSetType::EXT_TX_SET_DTT4_IDTX
        } else {
            TxSetType::EXT_TX_SET_DTT4_IDTX_1DDCT
        }
    }
}

fn get_ext_tx_set(tx_size: TxSize, is_inter: bool,
                                 use_reduced_set: bool) -> i8 {
  let set_type = get_ext_tx_set_type(tx_size, is_inter, use_reduced_set);
    if is_inter {
        ext_tx_set_index_inter[set_type as usize]
    } else {
        ext_tx_set_index_intra[set_type as usize]
    }
}

static intra_mode_to_tx_type_context: [TxType; INTRA_MODES] = [
    DCT_DCT,    // DC
    ADST_DCT,   // V
    DCT_ADST,   // H
    DCT_DCT,    // D45
    ADST_ADST,  // D135
    ADST_DCT,   // D117
    DCT_ADST,   // D153
    DCT_ADST,   // D207
    ADST_DCT,   // D63
    ADST_ADST,  // SMOOTH
    ADST_DCT,   // SMOOTH_V
    DCT_ADST,   // SMOOTH_H
    ADST_ADST,  // PAETH
];

static uv2y: [PredictionMode; UV_INTRA_MODES] = [
    DC_PRED,        // UV_DC_PRED
    V_PRED,         // UV_V_PRED
    H_PRED,         // UV_H_PRED
    D45_PRED,       // UV_D45_PRED
    D135_PRED,      // UV_D135_PRED
    D117_PRED,      // UV_D117_PRED
    D153_PRED,      // UV_D153_PRED
    D207_PRED,      // UV_D207_PRED
    D63_PRED,       // UV_D63_PRED
    SMOOTH_PRED,    // UV_SMOOTH_PRED
    SMOOTH_V_PRED,  // UV_SMOOTH_V_PRED
    SMOOTH_H_PRED,  // UV_SMOOTH_H_PRED
    PAETH_PRED,     // UV_PAETH_PRED
    DC_PRED,        // CFL_PRED
];

pub fn y_intra_mode_to_tx_type_context(pred: PredictionMode) -> TxType {
    intra_mode_to_tx_type_context[pred as usize]
}

pub fn uv_intra_mode_to_tx_type_context(pred: PredictionMode)-> TxType {
    intra_mode_to_tx_type_context[uv2y[pred as usize] as usize]
}

extern {
    static default_partition_cdf: [[u16; PARTITION_TYPES + 1]; PARTITION_CONTEXTS];
    static default_kf_y_mode_cdf: [[[u16; INTRA_MODES + 1]; INTRA_MODES]; INTRA_MODES];
    static default_if_y_mode_cdf: [[u16; INTRA_MODES + 1]; BLOCK_SIZE_GROUPS];
    static default_uv_mode_cdf: [[[u16; UV_INTRA_MODES + 1]; INTRA_MODES]; 2];
    static default_intra_ext_tx_cdf: [[[[u16; TX_TYPES + 1]; INTRA_MODES]; EXT_TX_SIZES]; EXT_TX_SETS_INTRA];
    static default_skip_cdfs: [[u16; 3];SKIP_CONTEXTS];
    static default_intra_inter_cdf: [[u16; 3];INTRA_INTER_CONTEXTS];
    static default_angle_delta_cdf: [[u16; 2 * MAX_ANGLE_DELTA + 1 + 1]; DIRECTIONAL_MODES];

    static av1_intra_scan_orders: [[SCAN_ORDER; TX_TYPES]; TX_SIZES_ALL];

    fn build_tail_cdfs(cdf_tail: &mut [u16; ENTROPY_TOKENS + 1],
                    cdf_head: &mut [u16; ENTROPY_TOKENS + 1],
                    band_zero: c_int);

    // lv_map
    static av1_default_txb_skip_cdf: [[[u16; 3]; TXB_SKIP_CONTEXTS]; TX_SIZES];
    static av1_default_dc_sign_cdf: [[[u16; 3]; DC_SIGN_CONTEXTS]; TX_SIZES];
    static av1_default_eob_extra_cdf: [[[[u16; 3]; EOB_COEF_CONTEXTS]; PLANE_TYPES]; TX_SIZES];
    
    static av1_default_eob_multi16: [[[u16; 5+1]; 2]; PLANE_TYPES];
    static av1_default_eob_multi32: [[[u16; 6+1]; 2]; PLANE_TYPES];
    static av1_default_eob_multi64: [[[u16; 7+1]; 2]; PLANE_TYPES];
    static av1_default_eob_multi128: [[[u16; 8+1]; 2]; PLANE_TYPES];
    static av1_default_eob_multi256: [[[u16; 9+1]; 2]; PLANE_TYPES];
    static av1_default_eob_multi512: [[[u16; 10+1]; 2]; PLANE_TYPES];
    static av1_default_eob_multi1024: [[[u16; 11+1]; 2]; PLANE_TYPES];

    static av1_default_coeff_base_eob_multi: [[[[u16; 3+1]; SIG_COEF_CONTEXTS_EOB];
                                          PLANE_TYPES]; TX_SIZES];
    static av1_default_coeff_base_multi: [[[[u16; 4+1]; SIG_COEF_CONTEXTS];
                                      PLANE_TYPES]; TX_SIZES];
    static av1_default_coeff_lps_multi: [[[[u16; BR_CDF_SIZE+1]; LEVEL_CONTEXTS];
                                      PLANE_TYPES]; TX_SIZES];
}

#[repr(C)]
pub struct SCAN_ORDER {
  // FIXME: don't hardcode sizes

  pub scan: &'static [u16; 64*64],
  pub iscan: &'static [u16; 64*64],
  pub neighbors: &'static [u16; ((64*64)+1)*2]
}

type CoeffModel = [[[[u16; ENTROPY_TOKENS + 1];COEFF_CONTEXTS];COEF_BANDS];REF_TYPES];

#[derive(Clone)]
pub struct CDFContext {
    partition_cdf: [[u16; PARTITION_TYPES + 1]; PARTITION_CONTEXTS],
    kf_y_cdf: [[[u16; INTRA_MODES + 1]; INTRA_MODES]; INTRA_MODES],
    y_mode_cdf: [[u16; INTRA_MODES + 1]; BLOCK_SIZE_GROUPS],
    uv_mode_cdf: [[[u16; UV_INTRA_MODES + 1]; INTRA_MODES]; 2],
    intra_ext_tx_cdf: [[[[u16; TX_TYPES + 1]; INTRA_MODES]; EXT_TX_SIZES]; EXT_TX_SETS_INTRA],
    skip_cdfs: [[u16; 3];SKIP_CONTEXTS],
    intra_inter_cdfs: [[u16; 3];INTRA_INTER_CONTEXTS],
    angle_delta_cdf: [[u16; 2 * MAX_ANGLE_DELTA + 1 + 1]; DIRECTIONAL_MODES],

    // lv_map
    txb_skip_cdf: [[[u16; 3]; TXB_SKIP_CONTEXTS]; TX_SIZES],
    dc_sign_cdf: [[[u16; 3]; DC_SIGN_CONTEXTS]; TX_SIZES],
    eob_extra_cdf: [[[[u16; 3]; EOB_COEF_CONTEXTS]; PLANE_TYPES]; TX_SIZES],

    eob_flag_cdf16: [[[u16; 5+1]; 2]; PLANE_TYPES],
    eob_flag_cdf32: [[[u16; 6+1]; 2]; PLANE_TYPES],
    eob_flag_cdf64: [[[u16; 7+1]; 2]; PLANE_TYPES],
    eob_flag_cdf128: [[[u16; 8+1]; 2]; PLANE_TYPES],
    eob_flag_cdf256: [[[u16; 9+1]; 2]; PLANE_TYPES],
    eob_flag_cdf512: [[[u16; 10+1]; 2]; PLANE_TYPES],
    eob_flag_cdf1024: [[[u16; 11+1]; 2]; PLANE_TYPES],

    coeff_base_eob_cdf: [[[[u16; 3+1]; SIG_COEF_CONTEXTS_EOB]; PLANE_TYPES];
                          TX_SIZES],
    coeff_base_cdf: [[[[u16; 4+1]; SIG_COEF_CONTEXTS]; PLANE_TYPES];
                          TX_SIZES],
    coeff_br_cdf: [[[[u16; BR_CDF_SIZE+1]; LEVEL_CONTEXTS]; PLANE_TYPES];
                          TX_SIZES],
}

impl CDFContext {
    pub fn new(_qindex: u8) -> CDFContext {
        let c = CDFContext {
            partition_cdf: default_partition_cdf,
            kf_y_cdf: default_kf_y_mode_cdf,
            y_mode_cdf: default_if_y_mode_cdf,
            uv_mode_cdf: default_uv_mode_cdf,
            intra_ext_tx_cdf: default_intra_ext_tx_cdf,
            skip_cdfs: default_skip_cdfs,
            intra_inter_cdfs: default_intra_inter_cdf,
            angle_delta_cdf: default_angle_delta_cdf,

            // lv_map
            txb_skip_cdf: av1_default_txb_skip_cdf,
            dc_sign_cdf: av1_default_dc_sign_cdf,
            eob_extra_cdf: av1_default_eob_extra_cdf,

            eob_flag_cdf16: av1_default_eob_multi16,
            eob_flag_cdf32: av1_default_eob_multi32,
            eob_flag_cdf64: av1_default_eob_multi64,
            eob_flag_cdf128: av1_default_eob_multi128,
            eob_flag_cdf256: av1_default_eob_multi256,
            eob_flag_cdf512: av1_default_eob_multi512,
            eob_flag_cdf1024: av1_default_eob_multi1024,

            coeff_base_eob_cdf: av1_default_coeff_base_eob_multi,
            coeff_base_cdf: av1_default_coeff_base_multi,
            coeff_br_cdf: av1_default_coeff_lps_multi

        };

        c
    }
}

const SUPERBLOCK_TO_PLANE_SHIFT: usize = MAX_SB_SIZE_LOG2;
const SUPERBLOCK_TO_BLOCK_SHIFT: usize = MAX_MIB_SIZE_LOG2;
const BLOCK_TO_PLANE_SHIFT: usize = MI_SIZE_LOG2;
pub const LOCAL_BLOCK_MASK: usize = (1 << SUPERBLOCK_TO_BLOCK_SHIFT) - 1;

/// Absolute offset in superblocks inside a plane, where a superblock is defined
/// to be an N*N square where N = (1 << SUPERBLOCK_TO_PLANE_SHIFT).
pub struct SuperBlockOffset {
    pub x: usize,
    pub y: usize
}

impl SuperBlockOffset {
    /// Offset of a block inside the current superblock.
    pub fn block_offset(&self, block_x: usize, block_y: usize) -> BlockOffset {
        BlockOffset {
            x: (self.x << SUPERBLOCK_TO_BLOCK_SHIFT) + block_x,
            y: (self.y << SUPERBLOCK_TO_BLOCK_SHIFT) + block_y,
        }
    }

    /// Offset of the top-left pixel of this block.
    pub fn plane_offset(&self, plane: &PlaneConfig) -> PlaneOffset {
        PlaneOffset {
            x: self.x << (SUPERBLOCK_TO_PLANE_SHIFT - plane.xdec),
            y: self.y << (SUPERBLOCK_TO_PLANE_SHIFT - plane.ydec),
        }
    }
}

/// Absolute offset in blocks inside a plane, where a block is defined
/// to be an N*N square where N = (1 << BLOCK_TO_PLANE_SHIFT).
pub struct BlockOffset {
    pub x: usize,
    pub y: usize
}

impl BlockOffset {
    /// Offset of the superblock in which this block is located.
    pub fn sb_offset(&self) -> SuperBlockOffset {
        SuperBlockOffset {
            x: self.x >> SUPERBLOCK_TO_BLOCK_SHIFT,
            y: self.y >> SUPERBLOCK_TO_BLOCK_SHIFT,
        }
    }

    /// Offset of the top-left pixel of this block.
    pub fn plane_offset(&self, plane: &PlaneConfig) -> PlaneOffset {
        let po = self.sb_offset().plane_offset(plane);
        let x_offset = self.x & LOCAL_BLOCK_MASK;
        let y_offset = self.y & LOCAL_BLOCK_MASK;
        PlaneOffset {
            x: po.x + (x_offset << BLOCK_TO_PLANE_SHIFT),
            y: po.y + (y_offset << BLOCK_TO_PLANE_SHIFT),
        }
    }

    pub fn y_in_sb(&self) -> usize {
        self.y % MAX_MIB_SIZE
    }
}

#[derive(Copy,Clone)]
pub struct Block {
    pub mode: PredictionMode,
    pub bsize: BlockSize,
    pub partition: PartitionType,
    pub skip: bool,
}

impl Block {
    pub fn default() -> Block {
        Block {
            mode: PredictionMode::DC_PRED,
            bsize: BlockSize::BLOCK_64X64,
            partition: PartitionType::PARTITION_NONE,
            skip: false,
        }
    }
    pub fn is_inter(&self) -> bool {
        self.mode >= PredictionMode::NEARESTMV
    }
}

pub struct  TXB_CTX {
    pub txb_skip_ctx: usize,
    pub dc_sign_ctx: usize,
}

#[derive(Clone, Default)]
pub struct BlockContext {
    pub cols: usize,
    pub rows: usize,
    above_partition_context: Vec<u8>,
    left_partition_context: [u8; MAX_MIB_SIZE],
    above_coeff_context: [Vec<u8>; PLANES],
    left_coeff_context: [[u8; MAX_MIB_SIZE]; PLANES],
    blocks: Vec<Vec<Block>>
}

impl BlockContext {
    pub fn new(cols: usize, rows: usize) -> BlockContext {
        // Align power of two
        let aligned_cols = (cols + ((1 << MAX_MIB_SIZE_LOG2) - 1)) & 
                            !((1 << MAX_MIB_SIZE_LOG2) - 1);
        BlockContext {
            cols,
            rows,
            above_partition_context: vec![0; aligned_cols],
            left_partition_context: [0; MAX_MIB_SIZE],
            above_coeff_context: [vec![0; cols << (MI_SIZE_LOG2 - tx_size_wide_log2[0])],
                                  vec![0; cols << (MI_SIZE_LOG2 - tx_size_wide_log2[0])],
                                  vec![0; cols << (MI_SIZE_LOG2 - tx_size_wide_log2[0])],],
            left_coeff_context: [[0; MAX_MIB_SIZE]; PLANES],
            blocks: vec![vec![Block::default(); cols]; rows]
        }
    }

    pub fn at(&mut self, bo: &BlockOffset) -> &mut Block {
        &mut self.blocks[bo.y][bo.x]
    }

    pub fn above_of(&mut self, bo: &BlockOffset) -> Block {
        if bo.y > 0 {
            self.blocks[bo.y - 1][bo.x]
        } else {
            Block::default()
        }
    }

    pub fn left_of(&mut self, bo: &BlockOffset) -> Block {
        if bo.x > 0 {
            self.blocks[bo.y][bo.x - 1]
        } else {
            Block::default()
        }
    }

    fn coeff_context(&self, plane: usize, bo: &BlockOffset) -> usize {
        (self.above_coeff_context[plane][bo.x]
         + self.left_coeff_context[plane][bo.y_in_sb()]) as usize
    }

    pub fn set_dc_sign(&mut self, cul_level: &mut u32, dc_val: i32) {
      if dc_val < 0 {
        *cul_level |= 1 << COEFF_CONTEXT_BITS;
      } else if dc_val > 0 {
        *cul_level += 2 << COEFF_CONTEXT_BITS;
      }
    }

    fn set_coeff_context(&mut self, plane: usize, bo: &BlockOffset, tx_size: TxSize,
                         xdec: usize, ydec: usize, value: u8) {
        // for subsampled planes, coeff contexts are stored sparsely at the moment
        // so we need to scale our fill by xdec and ydec
        for bx in 0..tx_size.width_mi() {
            self.above_coeff_context[plane][bo.x + (bx<<xdec)] = value;
        }
        let bo_y = bo.y_in_sb();
        for by in 0..tx_size.height_mi() {
            self.left_coeff_context[plane][bo_y + (by<<ydec)] = value;
        }
    }

    fn reset_left_coeff_context(&mut self, plane: usize) {
        for c in self.left_coeff_context[plane].iter_mut() {
            *c = 0;
        }
    }

    fn reset_left_partition_context(&mut self) {
        for c in self.left_partition_context.iter_mut() {
            *c = 0;
        }
    }
    //TODO(anyone): Add reset_left_tx_context() here then call it in reset_left_contexts()

    pub fn reset_left_contexts(&mut self) {
        for p in 0..3 {
            BlockContext::reset_left_coeff_context(self, p);
        }
        BlockContext::reset_left_partition_context(self);

        //TODO(anyone): Call reset_left_tx_context() here.
    }

    pub fn set_mode(&mut self, bo: &BlockOffset, bsize: BlockSize, mode: PredictionMode) {
        let bw = mi_size_wide[bsize as usize];
        let bh = mi_size_high[bsize as usize];

        for y in 0..bh {
            for x in 0..bw {
              self.blocks[bo.y + y as usize][bo.x + x as usize].mode = mode;
            };
        }
    }

    pub fn get_mode(&mut self, bo: &BlockOffset) -> PredictionMode {
        self.blocks[bo.y][bo.x].mode
    }

    fn partition_plane_context(&self, bo: &BlockOffset,
                               bsize: BlockSize) -> usize {
        // TODO: this should be way simpler without sub8x8
        let above_ctx = self.above_partition_context[bo.x];
        let left_ctx = self.left_partition_context[bo.y_in_sb()];
        let bsl = b_width_log2_lookup[bsize as usize] - b_width_log2_lookup[BlockSize::BLOCK_8X8 as usize];
        let above = (above_ctx >> bsl) & 1;
        let left = (left_ctx >> bsl) & 1;

        assert!(b_width_log2_lookup[bsize as usize] == b_height_log2_lookup[bsize as usize]);

        (left * 2 + above) as usize + bsl as usize * PARTITION_PLOFFSET
    }

    pub fn update_partition_context(&mut self, bo: &BlockOffset,
                                subsize : BlockSize, bsize: BlockSize) {
#[allow(dead_code)]
        let bw = mi_size_wide[bsize as usize];
        let bh = mi_size_high[bsize as usize];

        let above_ctx = &mut self.above_partition_context[bo.x..bo.x + bw as usize];
        let left_ctx = &mut self.left_partition_context[bo.y_in_sb()..bo.y_in_sb() + bh as usize];

        // update the partition context at the end notes. set partition bits
        // of block sizes larger than the current one to be one, and partition
        // bits of smaller block sizes to be zero.
        for i in 0..bw {
            above_ctx[i as usize] = partition_context_lookup[subsize as usize][0];
        }

        for i in 0..bh {
            left_ctx[i as usize] = partition_context_lookup[subsize as usize][1];
        }
    }

    fn skip_context(&mut self, bo: &BlockOffset) -> usize {
        let above_skip = if bo.y > 0 { self.above_of(bo).skip as usize }
                         else { 0 };
        let left_skip = if bo.x > 0 { self.left_of(bo).skip as usize }
                         else { 0 };
        above_skip + left_skip
    }

    pub fn set_skip(&mut self, bo: &BlockOffset, bsize: BlockSize, skip: bool) {
        let bw = mi_size_wide[bsize as usize];
        let bh = mi_size_high[bsize as usize];

        for y in 0..bh {
            for x in 0..bw {
                self.blocks[bo.y + y as usize][bo.x + x as usize].skip = skip;
            };
        }
    }

    // The mode info data structure has a one element border above and to the
    // left of the entries corresponding to real macroblocks.
    // The prediction flags in these dummy entries are initialized to 0.
    // 0 - inter/inter, inter/--, --/inter, --/--
    // 1 - intra/inter, inter/intra
    // 2 - intra/--, --/intra
    // 3 - intra/intra
    pub fn intra_inter_context(&mut self, bo: &BlockOffset) -> usize {
        let has_above = bo.y > 0;
        let has_left = bo.x > 0;

        match (has_above, has_left) {
            (true, true) => {
              let above_intra = !self.above_of(bo).is_inter();
              let left_intra = !self.left_of(bo).is_inter();
              if above_intra && left_intra { 3 }
              else { (above_intra || left_intra) as usize}
            },
            (true, _) | (_, true) => {
                2 * if has_above { !self.above_of(bo).is_inter() as usize }
                    else { !self.left_of(bo).is_inter() as usize }
            },
            (_, _) => 0,
        }
    }

    pub fn get_txb_ctx(&mut self, plane_bsize: BlockSize, tx_size: TxSize,
                       plane: usize, bo: &BlockOffset) -> TXB_CTX {
        let mut txb_ctx = TXB_CTX { txb_skip_ctx: 0,
                                dc_sign_ctx: 0 };
        const MAX_TX_SIZE_UNIT: usize = 16;
        const signs: [i8; 3] = [ 0, -1, 1 ];
        const dc_sign_contexts: [usize; 4 * MAX_TX_SIZE_UNIT + 1] = [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
        let mut dc_sign: i16 = 0;
        let txb_w_unit = tx_size_wide_unit[tx_size as usize];
        let txb_h_unit = tx_size_high_unit[tx_size as usize];

        // Decide txb_ctx.dc_sign_ctx
        for k in 0..txb_w_unit {
            let sign = self.above_coeff_context[plane][bo.x + k] >> COEFF_CONTEXT_BITS;
            assert!(sign <= 2);
            dc_sign += signs[sign as usize] as i16; 
        }

        for k in 0..txb_h_unit {
            let sign = self.left_coeff_context[plane][bo.y_in_sb() + k] >> COEFF_CONTEXT_BITS;
            assert!(sign <= 2);
            dc_sign += signs[sign as usize] as i16; 
        }

        txb_ctx.dc_sign_ctx = dc_sign_contexts[(dc_sign + 2 * MAX_TX_SIZE_UNIT as i16) as usize];

        // Decide txb_ctx.txb_skip_ctx
        if plane == 0 {
            if plane_bsize == txsize_to_bsize[tx_size as usize] {
              txb_ctx.txb_skip_ctx = 0;
            } else {
                // This is the algorithm to generate table skip_contexts[min][max].
                //    if (!max)
                //      txb_skip_ctx = 1;
                //    else if (!min)
                //      txb_skip_ctx = 2 + (max > 3);
                //    else if (max <= 3)
                //      txb_skip_ctx = 4;
                //    else if (min <= 3)
                //      txb_skip_ctx = 5;
                //    else
                //      txb_skip_ctx = 6;
                const skip_contexts: [[u8; 5]; 5] = [ [ 1, 2, 2, 2, 3 ],
                                                     [ 1, 4, 4, 4, 5 ],
                                                     [ 1, 4, 4, 4, 5 ],
                                                     [ 1, 4, 4, 4, 5 ],
                                                     [ 1, 4, 4, 4, 6 ] ];
                let mut top: u8 = 0;
                let mut left: u8 = 0;

                for k in 0..txb_w_unit {
                    top |= self.above_coeff_context[plane][bo.x + k];
                }
                top &= COEFF_CONTEXT_MASK as u8;

                for k in 0..txb_h_unit {
                    left |= self.left_coeff_context[plane][bo.y_in_sb() + k];
                }
                left &= COEFF_CONTEXT_MASK as u8;

                let max = cmp::min(top | left, 4);
                let min = cmp::min(cmp::min(top, left), 4);
                txb_ctx.txb_skip_ctx = skip_contexts[min as usize][max as usize] as usize;
            }

        } else {
            let mut top: u8 = 0;
            let mut left: u8 = 0;

            for k in 0..txb_w_unit {
                top |= self.above_coeff_context[plane][bo.x + k];
            }
            for k in 0..txb_h_unit {
                left |= self.left_coeff_context[plane][bo.y_in_sb() + k];
            }
            let ctx_base = (top != 0) as usize + (left != 0) as usize;
            let ctx_offset = if num_pels_log2_lookup[plane_bsize as usize] >
                                 num_pels_log2_lookup[txsize_to_bsize[tx_size as usize] as usize]
                                { 10 }
                             else { 7 };
            txb_ctx.txb_skip_ctx = ctx_base + ctx_offset;
        }

        txb_ctx
    }
}

#[derive(Clone)]
pub struct ContextWriterCheckpoint {
    pub w: ec::WriterCheckpoint,
    pub fc: CDFContext,
    pub bc: BlockContext
}

pub struct ContextWriter {
    pub w: ec::Writer,
    pub fc: CDFContext,
    pub bc: BlockContext
}

impl ContextWriter {
    fn cdf_element_prob(cdf: &[u16], element: usize) -> u16 {
      return if element > 0 { cdf[element - 1] } else { 32768 } - cdf[element];
    }

    fn partition_gather_horz_alike(out: &mut [u16; 2], cdf_in: &[u16], _bsize: BlockSize) {
      out[0] = 32768;
      out[0] -= ContextWriter::cdf_element_prob(cdf_in, PartitionType::PARTITION_HORZ as usize);
      out[0] -= ContextWriter::cdf_element_prob(cdf_in, PartitionType::PARTITION_SPLIT as usize);
      out[0] = 32768 - out[0];
      out[1] = 0;
    }

    fn partition_gather_vert_alike(out: &mut [u16; 2], cdf_in: &[u16], _bsize: BlockSize) {
      out[0] = 32768;
      out[0] -= ContextWriter::cdf_element_prob(cdf_in, PartitionType::PARTITION_VERT as usize);
      out[0] -= ContextWriter::cdf_element_prob(cdf_in, PartitionType::PARTITION_SPLIT as usize);
      out[0] = 32768 - out[0];
      out[1] = 0;
    }

    pub fn write_partition(&mut self, bo: &BlockOffset, p: PartitionType, bsize: BlockSize) {
        let hbs = (mi_size_wide[bsize as usize] / 2) as usize;
        let has_cols = (bo.x + hbs) < self.bc.cols;
        let has_rows = (bo.y + hbs) < self.bc.rows;
        let ctx = self.bc.partition_plane_context(&bo, bsize);
        assert!(ctx < PARTITION_CONTEXTS);
        let partition_cdf = &mut self.fc.partition_cdf[ctx];

        if !has_rows && !has_cols {
          return;
        }

        if has_rows && has_cols {
            self.w.symbol(p as u32, partition_cdf, PARTITION_TYPES);
        } else if !has_rows && has_cols {
            let mut cdf = [0u16; 2];
            ContextWriter::partition_gather_vert_alike(&mut cdf, partition_cdf, bsize);
            self.w.cdf((p == PartitionType::PARTITION_SPLIT) as u32, &cdf);
        } else {
            let mut cdf = [0u16; 2];
            ContextWriter::partition_gather_horz_alike(&mut cdf, partition_cdf, bsize);
            self.w.cdf((p == PartitionType::PARTITION_SPLIT) as u32, &cdf);
        }
    }
    pub fn write_intra_mode_kf(&mut self, bo: &BlockOffset, mode: PredictionMode) {
        let above_mode = self.bc.above_of(bo).mode as usize;
        let left_mode = self.bc.left_of(bo).mode as usize;
        let cdf = &mut self.fc.kf_y_cdf[above_mode][left_mode];
        self.w.symbol(mode as u32, cdf, INTRA_MODES);
    }
    pub fn write_intra_mode(&mut self, bsize: BlockSize, mode: PredictionMode) {
        let cdf = &mut self.fc.y_mode_cdf[size_group_lookup[bsize as usize] as usize];
        self.w.symbol(mode as u32, cdf, INTRA_MODES);
    }
    pub fn write_intra_uv_mode(&mut self, uv_mode: PredictionMode, y_mode: PredictionMode,
                               bs: BlockSize) {
        let cdf = &mut self.fc.uv_mode_cdf[bs.cfl_allowed() as usize][y_mode as usize];
        if bs.cfl_allowed() {
            self.w.symbol(uv_mode as u32, cdf, UV_INTRA_MODES);
        } else {
            self.w.symbol(uv_mode as u32, cdf, UV_INTRA_MODES - 1);
        }
    }
    pub fn write_angle_delta(&mut self, angle: i8, mode: PredictionMode) {
    self.w.symbol((angle + MAX_ANGLE_DELTA as i8) as u32,
                     &mut self.fc.angle_delta_cdf[mode as usize - PredictionMode::V_PRED as usize],
                     2 * MAX_ANGLE_DELTA + 1);
    }

    pub fn write_tx_type_lv_map(&mut self, tx_size: TxSize, tx_type: TxType, 
                                y_mode: PredictionMode, is_inter: bool,
                                use_reduced_tx_set: bool) {
        let square_tx_size = TXSIZE_SQR_MAP[tx_size as usize];
        let tx_set_type = get_ext_tx_set_type(tx_size, is_inter, use_reduced_tx_set);
        let num_tx_types = num_ext_tx_set[tx_set_type as usize];

        if num_tx_types > 1 {
          let eset = get_ext_tx_set(tx_size, is_inter, use_reduced_tx_set);
          assert!(eset > 0);
          assert!(av1_ext_tx_used[tx_set_type as usize][tx_type as usize] != 0);

          if is_inter {
              // TODO: Support inter mode once inter is enabled.
              assert!(false);
          } else {
              let intra_dir = y_mode;
              // TODO: Once use_filter_intra is enabled,
              // intra_dir =
              // fimode_to_intradir[mbmi->filter_intra_mode_info.filter_intra_mode];

              self.w.symbol(
                  av1_ext_tx_intra_ind[tx_set_type as usize][tx_type as usize],
                  &mut self.fc.intra_ext_tx_cdf[eset as usize][square_tx_size as usize][intra_dir as usize],
                  num_ext_tx_set[tx_set_type as usize]);
          }
        }
    }
    pub fn write_skip(&mut self, bo: &BlockOffset, skip: bool) {
        let ctx = self.bc.skip_context(bo);
        self.w.symbol(skip as u32, &mut self.fc.skip_cdfs[ctx], 2);
    }
    pub fn write_inter_mode(&mut self, bo: &BlockOffset, is_inter: bool) {
        let ctx = self.bc.intra_inter_context(bo);
        self.w.symbol(is_inter as u32, &mut self.fc.intra_inter_cdfs[ctx], 2);
    }

    pub fn get_txsize_entropy_ctx(&mut self, tx_size: TxSize) -> usize {
      (TXSIZE_SQR_MAP[tx_size as usize] as usize + TXSIZE_SQR_MAP[tx_size as usize] as usize + 1) >> 1
    }

    pub fn txb_init_levels(&mut self, coeffs: &[i32], width: usize, height: usize, 
                                levels_buf: &mut [u8]) {
        let mut offset = TX_PAD_TOP * (width + TX_PAD_HOR);

        for y in 0..height {
            for x in 0..width {
                levels_buf[offset] = clamp(coeffs[y*width + x].abs(), 0, 127) as u8;
                offset += 1;
            }
            offset += TX_PAD_HOR;
        }
    }

    // TODO: Figure out where this function is supposed to be used!?
/*    pub fn get_tx_type(&mut self, tx_size: TxSize, is_inter: bool,
                            use_reduced_set: bool) -> TxType {
        let tx_set_type = get_ext_tx_set_type(tx_size, is_inter, use_reduced_set);

        // TODO: Implement av1_get_tx_type() here
        let tx_type = TxType::DCT_DCT;

        tx_type
    }
*/
    pub fn av1_get_adjusted_tx_size(&mut self, tx_size: TxSize) -> TxSize {
      // TODO: Enable below commented out block if TX64X64 is enabled.
/*
      if tx_size == TX_64X64 || tx_size == TX_64X32 || tx_size == TX_32X64 {
        return TX_32X32
      }
      if (tx_size == TX_16X64) {
        return TX_16X32
      }
      if (tx_size == TX_64X16) {
        return TX_32X16
      }
*/
      tx_size
    }

    pub fn get_txb_bwl(&mut self, tx_size: TxSize) -> usize {
      let adjusted_tx_size = self.av1_get_adjusted_tx_size(tx_size);

      return tx_size_wide_log2[adjusted_tx_size as usize]
    }

    pub fn get_eob_pos_token(&mut self, eob: usize, extra: &mut u32) -> u32 {
        let t: u32;

        if eob < 33 {
          t = eob_to_pos_small[eob] as u32;
        } else {
          let e = cmp::min((eob - 1) >> 5, 16);
          t = eob_to_pos_large[e as usize] as u32;
        }
        assert!(eob as i32 >= k_eob_group_start[t as usize] as i32);
        *extra = eob as u32 - k_eob_group_start[t as usize] as u32;

        t
    }

    pub fn get_nz_mag(&mut self, levels: &[u8],
                      bwl: usize, tx_class: TxClass) -> usize {
        // May version.
        // Note: AOMMIN(level, 3) is useless for decoder since level < 3.
        let mut mag = clip_max3[levels[1] as usize];                 // { 0, 1 }
        mag += clip_max3[levels[(1 << bwl) + TX_PAD_HOR] as usize];  // { 1, 0 }

        if tx_class == TX_CLASS_2D {
          mag += clip_max3[levels[(1 << bwl) + TX_PAD_HOR + 1] as usize];          // { 1, 1 }
          mag += clip_max3[levels[2] as usize];                                    // { 0, 2 }
          mag += clip_max3[levels[(2 << bwl) + (2 << TX_PAD_HOR_LOG2)] as usize];  // { 2, 0 }
        } else if tx_class == TX_CLASS_VERT {
          mag += clip_max3[levels[(2 << bwl) + (2 << TX_PAD_HOR_LOG2)] as usize];  // { 2, 0 }
          mag += clip_max3[levels[(3 << bwl) + (3 << TX_PAD_HOR_LOG2)] as usize];  // { 3, 0 }
          mag += clip_max3[levels[(4 << bwl) + (4 << TX_PAD_HOR_LOG2)] as usize];  // { 4, 0 }
        } else {
          mag += clip_max3[levels[2] as usize];  // { 0, 2 }
          mag += clip_max3[levels[3] as usize];  // { 0, 3 }
          mag += clip_max3[levels[4] as usize];  // { 0, 4 }
        }

        mag as usize
    }

    pub fn get_nz_map_ctx_from_stats(&mut self, stats: usize,
                                      coeff_idx: usize,  // raster order
                                      bwl: usize, tx_size: TxSize,
                                      tx_class: TxClass) -> usize {
        if (tx_class as u32 | coeff_idx as u32) == 0 { return 0 };
        let mut ctx = (stats + 1) >> 1;
        ctx = cmp::min(ctx, 4);

        match tx_class {
          TX_CLASS_2D => {
            // This is the algorithm to generate av1_nz_map_ctx_offset[][]
            //   const int width = tx_size_wide[tx_size];
            //   const int height = tx_size_high[tx_size];
            //   if (width < height) {
            //     if (row < 2) return 11 + ctx;
            //   } else if (width > height) {
            //     if (col < 2) return 16 + ctx;
            //   }
            //   if (row + col < 2) return ctx + 1;
            //   if (row + col < 4) return 5 + ctx + 1;
            //   return 21 + ctx;
            return ctx + av1_nz_map_ctx_offset[tx_size as usize][coeff_idx] as usize
          }
          TX_CLASS_HORIZ => {
            let row = coeff_idx >> bwl;
            let col = coeff_idx - (row << bwl);
            return ctx + nz_map_ctx_offset_1d[col as usize]
          }
          TX_CLASS_VERT => {
            let row = coeff_idx >> bwl;
            return ctx + nz_map_ctx_offset_1d[row]
          }
        }
    }

    pub fn get_nz_map_ctx(&mut self, levels: &[u8], coeff_idx: usize, bwl: usize,
                          height: usize, scan_idx: usize,
                          is_eob: bool, tx_size: TxSize,
                          tx_class: TxClass) -> usize {
        if is_eob {
            if scan_idx == 0 { return 0 }
            if scan_idx <= (height << bwl) / 8 { return 1 }
            if scan_idx <= (height << bwl) / 4 { return 2 }
            return 3
        }
        let padded_idx = coeff_idx + ((coeff_idx >> bwl) << TX_PAD_HOR_LOG2);
        let stats =
            self.get_nz_mag(&levels[padded_idx..], bwl, tx_class);

        return self.get_nz_map_ctx_from_stats(stats, coeff_idx, bwl, tx_size,
                    tx_class)
    }

    pub fn get_nz_map_contexts(&mut self, levels: &mut [u8], scan: &[u16; 4096], eob: u16,
                                 tx_size: TxSize, tx_class: TxClass,
                                 coeff_contexts: &mut [i8]) {
        // TODO: If TX_64X64 is enabled, use av1_get_adjusted_tx_size()
        let bwl = tx_size_wide_log2[tx_size as usize];
        let height = tx_size_high[tx_size as usize];
        for i in 0..eob {
            let pos = scan[i as usize];
            coeff_contexts[pos as usize] =
                self.get_nz_map_ctx(levels, pos as usize, bwl, height, i as usize,
                               i == eob - 1, tx_size, tx_class) as i8;
        }
    }

    pub fn get_br_ctx(&mut self, levels: &[u8], c: usize,  // raster order
                      bwl: usize, tx_class: TxClass) -> usize {
        let row: usize = c >> bwl;
        let col: usize = c - (row << bwl);
        let stride: usize = (1 << bwl) + TX_PAD_HOR;
        let pos: usize = row * stride + col;
        let mut mag: usize = levels[pos + 1] as usize;

        mag += levels[pos + stride] as usize;

        match tx_class {
            TX_CLASS_2D => {
              mag += levels[pos + stride + 1] as usize;
              mag = cmp::min((mag + 1) >> 1, 6);
              if c == 0 { return mag }
              if (row < 2) && (col < 2) { return mag + 7 }
            }
            TX_CLASS_HORIZ => {
              mag += levels[pos + 2] as usize;
              mag = cmp::min((mag + 1) >> 1, 6);
              if c == 0 { return mag }
              if col == 0 { return mag + 7 }
            }
            TX_CLASS_VERT => {
              mag += levels[pos + (stride << 1)] as usize;
              mag = cmp::min((mag + 1) >> 1, 6);
              if c == 0 { return mag }
              if row == 0 { return mag + 7 }
            }
        }

        return mag + 14
    }

    pub fn get_level_mag_with_txclass(&mut self, levels: &[u8], stride: usize,
                                      row: usize, col: usize, mag: &mut [usize],
                                      tx_class: TxClass) {
      for idx in 0..CONTEXT_MAG_POSITION_NUM {
          let ref_row = row + mag_ref_offset_with_txclass[tx_class as usize][idx][0];
          let ref_col = col + mag_ref_offset_with_txclass[tx_class as usize][idx][1];
          let pos = ref_row * stride + ref_col;
          mag[idx] = levels[pos] as usize;
      }
    }

    pub fn write_coeffs_lv_map(&mut self, plane: usize, bo: &BlockOffset,
                        coeffs_in: &[i32], tx_size: TxSize, tx_type: TxType,
                        plane_bsize: BlockSize, xdec: usize, ydec: usize) {
        let pred_mode = self.bc.get_mode(bo);
        let is_inter = pred_mode >= PredictionMode::NEARESTMV;
        assert!( is_inter == false );
        // TODO: If iner mode, scan_order should use inter version of them
        let scan_order = &av1_intra_scan_orders[tx_size as usize][tx_type as usize];
        let scan = scan_order.scan;
        let mut coeffs_storage = [0 as i32; 64*64];
        let coeffs = &mut coeffs_storage[..tx_size.width()*tx_size.height()];
        let mut cul_level = 0 as u32;

        for i in 0..tx_size.width()*tx_size.height() {
            coeffs[i] = coeffs_in[scan[i] as usize];
            cul_level += coeffs[i].abs() as u32;
        }

        let mut eob = 0;

        if cul_level != 0 {
          for (i, v) in coeffs.iter().enumerate() {
              if *v != 0 {
                  eob = i + 1;
              }
          }
        }

        if plane == 0 && eob == 0 {
            assert!(tx_type == TxType::DCT_DCT);
        }

        let txs_ctx = self.get_txsize_entropy_ctx(tx_size);
        let txb_ctx = self.bc.get_txb_ctx(plane_bsize, tx_size, plane, bo);

        {
          let cdf = &mut self.fc.txb_skip_cdf[txs_ctx][txb_ctx.txb_skip_ctx];
          self.w.symbol((eob == 0) as u32, cdf, 2);
        }

        if eob == 0 {
            self.bc.set_coeff_context(plane, bo, tx_size, xdec, ydec, 0);
            return;
        }

        let mut levels_buf = [0 as u8; TX_PAD_2D];

        self.txb_init_levels(coeffs_in, tx_size.width(), tx_size.height(),
                            &mut levels_buf);

        let tx_class = tx_type_to_class[tx_type as usize];
        let plane_type = if plane == 0 { 0 as usize} else { 1 as usize };

        // TODO: Enable this, if TXK_SEL is enabled back.
        // Only y plane's tx_type is transmitted
        /*if plane == 0 {
            self.write_tx_type_lv_map(tx_size, tx_type, pred_mode, is_inter);
        }*/

        // Encode EOB
        let mut eob_extra = 0 as u32;
        let eob_pt = self.get_eob_pos_token(eob, &mut eob_extra);
        let eob_multi_size: usize = txsize_log2_minus4[tx_size as usize];
        let eob_multi_ctx: usize = if tx_class == TX_CLASS_2D { 0 } else { 1 };

        match eob_multi_size {
          0 => { self.w.symbol(eob_pt - 1,
                     &mut self.fc.eob_flag_cdf16[plane_type][eob_multi_ctx], 5); }
          1 => { self.w.symbol(eob_pt - 1,
                     &mut self.fc.eob_flag_cdf32[plane_type][eob_multi_ctx], 6); }
          2 => { self.w.symbol(eob_pt - 1,
                     &mut self.fc.eob_flag_cdf64[plane_type][eob_multi_ctx], 7); }
          3 => { self.w.symbol(eob_pt - 1,
                     &mut self.fc.eob_flag_cdf128[plane_type][eob_multi_ctx], 8); }
          4 => { self.w.symbol(eob_pt - 1,
                     &mut self.fc.eob_flag_cdf256[plane_type][eob_multi_ctx], 9); }
          5 => { self.w.symbol(eob_pt - 1,
                     &mut self.fc.eob_flag_cdf512[plane_type][eob_multi_ctx], 10); }
          _ => { self.w.symbol(eob_pt - 1,
                     &mut self.fc.eob_flag_cdf1024[plane_type][eob_multi_ctx], 11); }
        };

        let eob_offset_bits = k_eob_offset_bits[eob_pt as usize];

        if eob_offset_bits > 0 {
            let mut eob_shift = eob_offset_bits - 1;
            let mut bit = if (eob_extra & (1 << eob_shift)) != 0 { 1 } else { 0 } as u32;
            self.w.symbol(bit,
              &mut self.fc.eob_extra_cdf[txs_ctx][plane_type][eob_pt as usize],
                             2);
            for i in 1..eob_offset_bits {
              eob_shift = eob_offset_bits as u16 - 1 - i as u16;
              bit = if (eob_extra & (1 << eob_shift)) != 0 { 1 } else { 0 };
              self.w.bit(bit as u16);
            }
        }

        let mut coeff_contexts = [0 as i8; MAX_TX_SQUARE];
        let levels = &mut levels_buf[TX_PAD_TOP * (tx_size.width() + TX_PAD_HOR)..];

        self.get_nz_map_contexts(levels, scan, eob as u16, tx_size,
                                 tx_class, &mut coeff_contexts);

        let bwl = self.get_txb_bwl(tx_size);

        for c in (0..eob).rev() {
            let pos = scan[c];
            let coeff_ctx = coeff_contexts[pos as usize];
            let v = coeffs_in[pos as usize];
            let level: u32 = v.abs() as u32;

            if c == eob - 1 {
                self.w.symbol((cmp::min(level, 3) - 1) as u32,
                    &mut self.fc.coeff_base_eob_cdf[
                    txs_ctx][plane_type][coeff_ctx as usize], 3);
            } else {
                self.w.symbol((cmp::min(level, 3)) as u32,
                    &mut self.fc.coeff_base_cdf[txs_ctx][plane_type][coeff_ctx as usize],
                                 4);
            }
        }

        let update_eob = (eob - 1) as i16;

        // Loop to code all signs in the transform block,
        // starting with the sign of DC (if applicable)
        for c in 0..eob {
            let v = coeffs_in[scan[c] as usize];
            let level = v.abs() as u32;
            //let sign = (v < 0) as u32;
            let sign = if v < 0 { 1 } else { 0 };

            if level == 0 { continue; }

            if c == 0 {
                self.w.symbol(sign, &mut self.fc.dc_sign_cdf[plane_type]
                              [txb_ctx.dc_sign_ctx], 2);
            } else {
                self.w.bit(sign as u16);
            }
        }

        if update_eob >= 0 {
            for c in (0..update_eob+1).rev() {
                let pos = scan[c as usize];
                let v = coeffs_in[pos as usize];
                let level = v.abs() as u16;

                if level <= NUM_BASE_LEVELS as u16 { continue; }

                let base_range = level - 1 - NUM_BASE_LEVELS as u16;
                let br_ctx = self.get_br_ctx(levels, pos as usize, bwl, tx_class);
                let mut idx = 0;

                loop {
                  if idx >= COEFF_BASE_RANGE { break; }
                  let k = cmp::min(base_range - idx as u16, BR_CDF_SIZE as u16 - 1);
                  self.w.symbol(k as u32, &mut self.fc.coeff_br_cdf[
                          cmp::min(txs_ctx, TxSize::TX_32X32 as usize)]
                          [plane_type][br_ctx], BR_CDF_SIZE);
                  if k < BR_CDF_SIZE as u16 - 1 { break; }
                  idx += BR_CDF_SIZE - 1;
                }

                if base_range < COEFF_BASE_RANGE as u16 { continue; }
                // use 0-th order Golomb code to handle the residual level.
                self.w.write_golomb(level -
                    COEFF_BASE_RANGE as u16 - 1 - NUM_BASE_LEVELS as u16);
            }
        }

        cul_level = cmp::min(COEFF_CONTEXT_MASK as u32, cul_level);
        
        self.bc.set_dc_sign(&mut cul_level, coeffs[0]);

        self.bc.set_coeff_context(plane, bo, tx_size, xdec, ydec, cul_level as u8);
    }

    pub fn checkpoint(&mut self) -> ContextWriterCheckpoint {
        ContextWriterCheckpoint {
            w: self.w.checkpoint(),
            fc: self.fc.clone(),
            bc: self.bc.clone()
        }
    }

    pub fn rollback(&mut self, checkpoint: ContextWriterCheckpoint) {
        self.w.rollback(&checkpoint.w);
        self.fc = checkpoint.fc.clone();
        self.bc = checkpoint.bc.clone();
    }
}
