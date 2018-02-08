#![allow(safe_extern_statics)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]
#![allow(non_camel_case_types)]

use ec;
use partition::*;
use partition::TxSize::*;
use plane::*;

const PLANES: usize = 3;

const PARTITION_PLOFFSET: usize = 4;
const PARTITION_CONTEXTS: usize = 16;
const PARTITION_TYPES: usize = 4;

const MI_SIZE_LOG2: usize = 2;
const MI_SIZE: usize = (1 << MI_SIZE_LOG2);
const MAX_MIB_SIZE_LOG2: usize = (MAX_SB_SIZE_LOG2 - MI_SIZE_LOG2);
const MAX_MIB_SIZE: usize = (1 << MAX_MIB_SIZE_LOG2);
const MAX_MIB_MASK: usize = (MAX_MIB_SIZE - 1);

const MAX_SB_SIZE_LOG2: usize = 6;
const MAX_SB_SIZE: usize = (1 << MAX_SB_SIZE_LOG2);
const MAX_SB_SQUARE: usize = (MAX_SB_SIZE * MAX_SB_SIZE);

const INTRA_MODES: usize = 13;
const UV_INTRA_MODES: usize = 13;

const b_width_log2_lookup: [u8; 20] = [0, 0, 0,  0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 0, 2, 1, 3];
const b_height_log2_lookup: [u8; 20] = [ 0, 0, 0, 0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 2, 0, 3, 1];
const mi_width_log2_lookup: [u8; 20] = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 0, 2, 1, 3];
const mi_height_log2_lookup: [u8; 20] = [0, 0, 0, 0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 2, 0, 3, 1];
const tx_size_wide_log2: [usize; 14] = [2, 3, 4, 5, 2, 3, 3, 4, 4, 5, 2, 4, 3, 5];
const tx_size_high_log2: [usize; 14] = [2, 3, 4, 5, 3, 2, 4, 3, 5, 4, 4, 2, 5, 3];

const EXT_TX_SIZES: usize = 4;
const EXT_TX_SET_TYPES: usize = 6;
const EXT_TX_SETS_INTRA: usize = 3;
const EXT_TX_SETS_INTER: usize = 4;
// Number of transform types in each set type
const num_ext_tx_set: [usize; EXT_TX_SET_TYPES] = [1, 2, 5, 7, 12, 16];
// Maps intra set index to the set type
const ext_tx_set_type_intra: [TxSetType; EXT_TX_SETS_INTRA] = [
    TxSetType::EXT_TX_SET_DCTONLY, TxSetType::EXT_TX_SET_DTT4_IDTX_1DDCT, TxSetType::EXT_TX_SET_DTT4_IDTX
];
// Maps inter set index to the set type
#[allow(dead_code)]
const ext_tx_set_type_inter: [TxSetType; EXT_TX_SETS_INTER] = [
    TxSetType::EXT_TX_SET_DCTONLY, TxSetType::EXT_TX_SET_ALL16, TxSetType::EXT_TX_SET_DTT9_IDTX_1DDCT,
    TxSetType::EXT_TX_SET_DCT_IDTX
];
// Maps set types above to the indices used for intra
const ext_tx_set_index_intra: [i8; EXT_TX_SET_TYPES] = [0, -1, 2, 1, -1, -1 ];
// Maps set types above to the indices used for inter
const ext_tx_set_index_inter: [i8; EXT_TX_SET_TYPES] = [0, 3, -1, -1, 2, 1];
const av1_ext_tx_intra_ind: [[u32; TX_TYPES]; EXT_TX_SETS_INTRA] = [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],[1,5,6,4,0,0,0,0,0,0,2,3,0,0,0,0,],[1,3,4,2,0,0,0,0,0,0,0,0,0,0,0,0,],];
#[allow(dead_code)]
const av1_ext_tx_inter_ind: [[usize; TX_TYPES]; EXT_TX_SETS_INTER] = [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],[1,5,6,4,0,0,0,0,0,0,2,3,0,0,0,0,],[1,3,4,2,0,0,0,0,0,0,0,0,0,0,0,0,],[0,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,],];
const ext_tx_cnt_intra: [usize;EXT_TX_SETS_INTRA] = [ 1, 7, 5 ];

const av1_coefband_trans_4x4: [u8; 16] = [
    0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5,
];

const TXSIZE_SQR_MAP: [TxSize; TX_SIZES_ALL] = [
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

const TXSIZE_SQR_UP_MAP: [TxSize; TX_SIZES_ALL] = [
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

fn get_ext_tx_set_type(tx_size: TxSize, is_inter: bool, use_reduced_set: bool) -> TxSetType {
  let tx_size_sqr_up = TXSIZE_SQR_UP_MAP[tx_size as usize];
  let tx_size_sqr = TXSIZE_SQR_MAP[tx_size as usize];
  if tx_size_sqr > TxSize::TX_32X32 { TxSetType::EXT_TX_SET_DCTONLY }
  else if use_reduced_set {
      if is_inter { TxSetType::EXT_TX_SET_DCT_IDTX } else { TxSetType::EXT_TX_SET_DTT4_IDTX }
  }
  else if tx_size_sqr_up == TxSize::TX_32X32 {
      if is_inter { TxSetType::EXT_TX_SET_DCT_IDTX } else { TxSetType::EXT_TX_SET_DCTONLY }
  }
  else if is_inter {
      if tx_size_sqr == TxSize::TX_16X16 { TxSetType::EXT_TX_SET_DTT9_IDTX_1DDCT } else { TxSetType::EXT_TX_SET_ALL16 }
  } else {
    if tx_size_sqr == TxSize::TX_16X16 { TxSetType::EXT_TX_SET_DTT4_IDTX
    } else { TxSetType::EXT_TX_SET_DTT4_IDTX_1DDCT }
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

extern {
    static default_partition_cdf: [[u16; PARTITION_TYPES + 1]; PARTITION_CONTEXTS];
    static default_kf_y_mode_cdf: [[[u16; INTRA_MODES + 1]; INTRA_MODES]; INTRA_MODES];
    static default_uv_mode_cdf: [[u16; UV_INTRA_MODES + 1]; INTRA_MODES];
    static default_intra_ext_tx_cdf: [[[[u16; TX_TYPES + 1]; INTRA_MODES]; EXT_TX_SIZES]; EXT_TX_SETS_INTRA];
    static default_skip_cdfs: [[u16; 3];SKIP_CONTEXTS];
    static default_coef_head_cdf_4x4: [CoeffModel; PLANE_TYPES];
    static default_coef_head_cdf_8x8: [CoeffModel; PLANE_TYPES];
    static default_coef_head_cdf_16x16: [CoeffModel; PLANE_TYPES];
    static default_coef_head_cdf_32x32: [CoeffModel; PLANE_TYPES];
    static default_coef_tail_cdf: [[CoeffModel; PLANE_TYPES]; TX_SIZES];

    static av1_cat1_cdf0: [u16; 2];
    static av1_cat2_cdf0: [u16; 4];
    static av1_cat3_cdf0: [u16; 8];
    static av1_cat4_cdf0: [u16; 16];
    static av1_cat5_cdf0: [u16; 16];
    static av1_cat5_cdf1: [u16; 2];
    static av1_cat6_cdf0: [u16; 16];
    static av1_cat6_cdf1: [u16; 16];
    static av1_cat6_cdf2: [u16; 16];
    static av1_cat6_cdf3: [u16; 16];
    static av1_cat6_cdf4: [u16; 4];

    static av1_intra_scan_orders: [[SCAN_ORDER; TX_TYPES]; TX_SIZES_ALL];

    pub static exported_intra_mode_to_tx_type_context: &'static [TxType; INTRA_MODES];
}

#[repr(C)]
pub struct SCAN_ORDER {
  // FIXME: don't hardcode sizes

  pub scan: &'static [u16; 16],
  pub iscan: &'static [u16; 16],
  pub neighbors: &'static [u16; 17*2]
}

type CoeffModel = [[[[u16; ENTROPY_TOKENS + 1];COEFF_CONTEXTS];COEF_BANDS];REF_TYPES];

#[derive(Clone)]
pub struct CDFContext {
    partition_cdf: [[u16; PARTITION_TYPES + 1]; PARTITION_CONTEXTS],
    kf_y_cdf: [[[u16; INTRA_MODES + 1]; INTRA_MODES]; INTRA_MODES],
    uv_mode_cdf: [[u16; INTRA_MODES + 1]; INTRA_MODES],
    intra_ext_tx_cdf: [[[[u16; TX_TYPES + 1]; INTRA_MODES]; EXT_TX_SIZES]; EXT_TX_SETS_INTRA],
    coef_head_cdfs: [[CoeffModel; PLANE_TYPES]; TX_SIZES],
    coef_tail_cdfs: [[CoeffModel; PLANE_TYPES]; TX_SIZES],
    skip_cdfs: [[u16; 3];SKIP_CONTEXTS],
}

impl CDFContext {
    pub fn new() -> CDFContext {
        CDFContext {
            partition_cdf: default_partition_cdf,
            kf_y_cdf: default_kf_y_mode_cdf,
            uv_mode_cdf: default_uv_mode_cdf,
            intra_ext_tx_cdf: default_intra_ext_tx_cdf,
            skip_cdfs: default_skip_cdfs,
            coef_head_cdfs: [default_coef_head_cdf_4x4,
                             default_coef_head_cdf_8x8,
                             default_coef_head_cdf_16x16,
                             default_coef_head_cdf_32x32],
            coef_tail_cdfs: default_coef_tail_cdf,
        }
    }
}

const SUPERBLOCK_TO_PLANE_SHIFT: usize = 6;
const SUPERBLOCK_TO_BLOCK_SHIFT: usize = 4;
const BLOCK_TO_PLANE_SHIFT: usize = SUPERBLOCK_TO_PLANE_SHIFT - SUPERBLOCK_TO_BLOCK_SHIFT;
const LOCAL_BLOCK_MASK: usize = (1 << SUPERBLOCK_TO_BLOCK_SHIFT) - 1;

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
    pub skip: bool,
}

impl Block {
    pub fn default() -> Block {
        Block {
            mode: PredictionMode::DC_PRED,
            skip: false,
        }
    }
    pub fn is_inter(&self) -> bool {
        false
    }
}

#[derive(Clone, Default)]
pub struct BlockContext {
    cols: usize,
    rows: usize,
    above_seg_context: Vec<u8>,
    left_seg_context: [u8; MAX_MIB_SIZE],
    above_coeff_context: [Vec<u8>; PLANES],
    left_coeff_context: [[u8; MAX_MIB_SIZE]; PLANES],
    blocks: Vec<Vec<Block>>
}

impl BlockContext {
    pub fn new(cols: usize, rows: usize) -> BlockContext {
        BlockContext {
            cols,
            rows,
            above_seg_context: vec![0; cols << (MI_SIZE_LOG2 - tx_size_wide_log2[0])],
            left_seg_context: [0; MAX_MIB_SIZE],
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

    fn set_coeff_context(&mut self, plane: usize, bo: &BlockOffset, value: bool) {
        let uvalue = value as u8;
        self.above_coeff_context[plane][bo.x] = uvalue;
        self.left_coeff_context[plane][bo.y_in_sb()] = uvalue;
    }

    pub fn reset_left_coeff_context(&mut self, plane: usize) {
        for c in self.left_coeff_context[plane].iter_mut() {
            *c = 0;
        }
    }

    fn partition_plane_context(&self, bo: &BlockOffset,
                               bsize: BlockSize) -> usize {
        // TODO: this should be way simpler without sub8x8
        let above_ctx = self.above_seg_context[bo.x];
        let left_ctx = self.left_seg_context[bo.y_in_sb()];
        let bsl = mi_width_log2_lookup[bsize as usize] - mi_width_log2_lookup[BlockSize::BLOCK_8X8 as usize];
        let above = (above_ctx >> bsl) & 1;
        let left = (left_ctx >> bsl) & 1;

        assert!(b_width_log2_lookup[bsize as usize] == b_height_log2_lookup[bsize as usize]);

        ((left * 2 + above) + bsl) as usize * PARTITION_PLOFFSET
    }

    fn skip_context(&mut self, bo: &BlockOffset) -> usize {
        (self.above_of(bo).skip as usize) + (self.left_of(bo).skip as usize)
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
    pub fn write_partition(&mut self, p: PartitionType) {
        let bo = BlockOffset { x: 0, y: 0 };
        let ctx = self.bc.partition_plane_context(&bo, BlockSize::BLOCK_64X64);
        self.w.symbol(p as u32, &mut self.fc.partition_cdf[ctx], PARTITION_TYPES);
    }
    pub fn write_intra_mode_kf(&mut self, bo: &BlockOffset, mode: PredictionMode) {
        let above_mode = self.bc.above_of(bo).mode as usize;
        let left_mode = self.bc.left_of(bo).mode as usize;
        let cdf = &mut self.fc.kf_y_cdf[above_mode][left_mode];
        self.w.symbol(mode as u32, cdf, INTRA_MODES);
    }
    pub fn write_intra_uv_mode(&mut self, uv_mode: PredictionMode, y_mode: PredictionMode) {
        let cdf = &mut self.fc.uv_mode_cdf[y_mode as usize];
        self.w.symbol(uv_mode as u32, cdf, INTRA_MODES);
    }
    pub fn write_tx_type(&mut self, tx_type: TxType, y_mode: PredictionMode) {
        let tx_size = TxSize::TX_4X4;
        let square_tx_size = TXSIZE_SQR_MAP[tx_size as usize];
        let eset =
            get_ext_tx_set(tx_size, false, true);
        if eset > 0 {
            self.w.symbol(
                av1_ext_tx_intra_ind[eset as usize][tx_type as usize],
                &mut self.fc.intra_ext_tx_cdf[eset as usize][square_tx_size as usize][y_mode as usize],
                ext_tx_cnt_intra[eset as usize]);
        }
    }
    pub fn write_skip(&mut self, bo: &BlockOffset, skip: bool) {
        let ctx = self.bc.skip_context(bo);
        self.w.symbol(skip as u32, &mut self.fc.skip_cdfs[ctx], 2);
    }
    pub fn write_token_block_zero(&mut self, plane: usize, bo: &BlockOffset) {
        let plane_type = if plane > 0 { 1 } else { 0 };
        let tx_size_ctx = TXSIZE_SQR_MAP[TxSize::TX_4X4 as usize] as usize;
        let ref_type = 0;
        let band = 0;
        let ctx = self.bc.coeff_context(plane, bo);
        let cdf = &mut self.fc.coef_head_cdfs[tx_size_ctx][plane_type][ref_type][band][ctx];
        //println!("encoding token band={} ctx={}", band, ctx);
        self.w.symbol(0, cdf, HEAD_TOKENS + 1);
        self.bc.set_coeff_context(plane, bo, false);
    }
    pub fn write_coeffs(&mut self, plane: usize, bo: &BlockOffset,
                        coeffs_in: &[i32], tx_size: TxSize, tx_type: TxType) {
        let scan_order = &av1_intra_scan_orders[tx_size as usize][tx_type as usize];
        let scan = scan_order.scan;
        let mut coeffs = [0 as i32; 16];
        for i in 0..16 {
            coeffs[i] = coeffs_in[scan[i] as usize];
        }
        let mut nz_coeff = 0;
        for (i, v) in coeffs.iter().enumerate() {
            if *v != 0 {
                nz_coeff = i + 1;
            }
        }
        if nz_coeff == 0 {
            self.write_token_block_zero(plane, bo);
            return;
        }
        let tx_size = TxSize::TX_4X4;
        let plane_type = if plane > 0 { 1 } else { 0 };
        let tx_size_ctx = TXSIZE_SQR_MAP[tx_size as usize] as usize;
        let ref_type = 0;
        let neighbors = scan_order.neighbors;
        let mut token_cache = [0 as u8; 64*64];
        for (i, v) in coeffs.iter().enumerate() {
            let vabs = v.abs() as u32;
            let first = i == 0;
            let last = i == (nz_coeff - 1);
            let band = av1_coefband_trans_4x4[i];
            let ctx = if first {
                self.bc.coeff_context(plane, bo)
            } else {
                ((1 + token_cache[neighbors[2 * i + 0] as usize]
                  + token_cache[neighbors[2 * i + 1] as usize]) >> 1) as usize
            };
            let cdf = &mut self.fc.coef_head_cdfs[tx_size_ctx][plane_type][ref_type][band as usize][ctx];
            match (vabs, last) {
                (0,_) => {
                    self.w.symbol(HeadToken::Zero as u32 - !first as u32, cdf, HEAD_TOKENS + (first as usize));
                    continue
                },
                (1, false) => self.w.symbol(HeadToken::OneNEOB as u32 - !first as u32, cdf, HEAD_TOKENS + (first as usize)),
                (1, true) => self.w.symbol(HeadToken::OneEOB as u32 - !first as u32, cdf, HEAD_TOKENS + (first as usize)),
                (_, false) => self.w.symbol(HeadToken::TwoPlusNEOB as u32 - !first as u32, cdf, HEAD_TOKENS + (first as usize)),
                (_, true) => self.w.symbol(HeadToken::TwoPlusEOB as u32 - !first as u32, cdf, HEAD_TOKENS + (first as usize)),
            };
            let tailcdf = &mut self.fc.coef_tail_cdfs[tx_size_ctx][plane_type][ref_type][band as usize][ctx as usize];
            match vabs {
                0|1 => {},
                2 => self.w.symbol(TailToken::Two as u32, tailcdf, TAIL_TOKENS),
                3 => self.w.symbol(TailToken::Three as u32, tailcdf, TAIL_TOKENS),
                4 => self.w.symbol(TailToken::Four as u32, tailcdf, TAIL_TOKENS),
                5...6 => {
                    self.w.symbol(TailToken::Cat1 as u32, tailcdf, TAIL_TOKENS);
                    self.w.cdf(vabs - 5, &av1_cat1_cdf0);
                }
                7...10 => {
                    self.w.symbol(TailToken::Cat2 as u32, tailcdf, TAIL_TOKENS);
                    self.w.cdf(vabs - 7, &av1_cat2_cdf0);
                }
                11...18 => {
                    self.w.symbol(TailToken::Cat3 as u32, tailcdf, TAIL_TOKENS);
                    self.w.cdf(vabs - 11, &av1_cat3_cdf0);
                }
                19...34 => {
                    self.w.symbol(TailToken::Cat4 as u32, tailcdf, TAIL_TOKENS);
                    self.w.cdf(vabs - 19, &av1_cat4_cdf0);
                }
                35...66 => {
                    self.w.symbol(TailToken::Cat5 as u32, tailcdf, TAIL_TOKENS);
                    self.w.cdf((vabs - 35) & 0xf, &av1_cat5_cdf0);
                    self.w.cdf(((vabs - 35) >> 4) & 0x1, &av1_cat5_cdf1);
                }
                _ => {
                    self.w.symbol(TailToken::Cat6 as u32, tailcdf, TAIL_TOKENS);
                    let tx_offset = tx_size as u32 - TxSize::TX_4X4 as u32;
                    let bit_depth = 8;
                    let bits = bit_depth + 3 + tx_offset;
                    self.w.cdf((vabs - 67) & 0xf, &av1_cat6_cdf0);
                    self.w.cdf(((vabs - 67) >> 4) & 0xf, &av1_cat6_cdf1);
                    self.w.cdf(((vabs - 67) >> 8) & 0xf, &av1_cat6_cdf2);
                    if bits > 12 {
                        self.w.cdf(((vabs - 67) >> 12) & 0xf, &av1_cat6_cdf3);
                    }
                    if bits > 16 {
                        self.w.cdf(((vabs - 67) >> 16) & 0x3, &av1_cat6_cdf4);
                    }
                }
            };
            self.w.bool(*v < 0, 16384);
            let energy_class = match vabs {
                0 => 0,
                1 => 1,
                2 => 2,
                3|4 => 3,
                5...10 => 4,
                _ => 5,
            };
            token_cache[scan[i] as usize] = energy_class;
            if last {
                break;
            }
        }
        self.bc.set_coeff_context(plane, bo, true);
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
