#![allow(non_camel_case_types)]
#![allow(dead_code)]

#[derive(Copy,Clone,PartialEq,PartialOrd)]
pub enum PartitionType {
    PARTITION_NONE,
    PARTITION_HORZ,
    PARTITION_VERT,
    PARTITION_SPLIT,
    PARTITION_INVALID
}

pub const BLOCK_SIZES_ALL: usize = 19;

#[derive(Copy,Clone,PartialEq,PartialOrd)]
pub enum BlockSize {
    BLOCK_4X4,
    BLOCK_4X8,
    BLOCK_8X4,
    BLOCK_8X8,
    BLOCK_8X16,
    BLOCK_16X8,
    BLOCK_16X16,
    BLOCK_16X32,
    BLOCK_32X16,
    BLOCK_32X32,
    BLOCK_32X64,
    BLOCK_64X32,
    BLOCK_64X64,
    BLOCK_4X16,
    BLOCK_16X4,
    BLOCK_8X32,
    BLOCK_32X8,
    BLOCK_16X64,
    BLOCK_64X16,
    BLOCK_INVALID
}

impl BlockSize {
    pub fn cfl_allowed(self) -> bool {
        // TODO: fix me when enabling EXT_PARTITION_TYPES
        self <= BlockSize::BLOCK_32X32
    }
}

pub const TX_SIZES: usize = 4;
pub const TX_SIZES_ALL: usize = 14;

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum TxSize {
    TX_4X4,
    TX_8X8,
    TX_16X16,
    TX_32X32,
    TX_4X8,
    TX_8X4,
    TX_8X16,
    TX_16X8,
    TX_16X32,
    TX_32X16,
    TX_4X16,
    TX_16X4,
    TX_8X32,
    TX_32X8,
}

impl TxSize {
    pub fn width(self) -> usize {
        1<<tx_size_wide_log2[self as usize]
    }
    pub fn height(self) -> usize {
        1<<tx_size_high_log2[self as usize]
    }
    pub fn width_mi(self) -> usize {
        (1<<tx_size_wide_log2[self as usize])>>2
    }
    pub fn height_mi(self) -> usize {
        (1<<tx_size_high_log2[self as usize])>>2
    }
}


pub const TX_TYPES: usize = 16;

#[derive(Copy,Clone)]
#[repr(C)]
pub enum TxType {
    DCT_DCT = 0,    // DCT  in both horizontal and vertical
    ADST_DCT = 1,   // ADST in vertical, DCT in horizontal
    DCT_ADST = 2,   // DCT  in vertical, ADST in horizontal
    ADST_ADST = 3,  // ADST in both directions
    FLIPADST_DCT = 4,
    DCT_FLIPADST = 5,
    FLIPADST_FLIPADST = 6,
    ADST_FLIPADST = 7,
    FLIPADST_ADST = 8,
    IDTX = 9,
    V_DCT = 10,
    H_DCT = 11,
    V_ADST = 12,
    H_ADST = 13,
    V_FLIPADST = 14,
    H_FLIPADST = 15,
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum PredictionMode {
    DC_PRED,    // Average of above and left pixels
    V_PRED,     // Vertical
    H_PRED,     // Horizontal
    D45_PRED,   // Directional 45  deg = round(arctan(1/1) * 180/pi)
    D135_PRED,  // Directional 135 deg = 180 - 45
    D117_PRED,  // Directional 117 deg = 180 - 63
    D153_PRED,  // Directional 153 deg = 180 - 27
    D207_PRED,  // Directional 207 deg = 180 + 27
    D63_PRED,   // Directional 63  deg = round(arctan(2/1) * 180/pi)
    SMOOTH_PRED,  // Combination of horizontal and vertical interpolation
    SMOOTH_V_PRED,
    SMOOTH_H_PRED,
    PAETH_PRED,
    NEARESTMV,
    NEARMV,
    ZEROMV,
    NEWMV,
    // Compound ref compound modes
    NEAREST_NEARESTMV,
    NEAR_NEARMV,
    NEAREST_NEWMV,
    NEW_NEARESTMV,
    NEAR_NEWMV,
    NEW_NEARMV,
    ZERO_ZEROMV,
    NEW_NEWMV,
}

use plane::*;
use predict::*;
use context::*;

impl PredictionMode {
    pub fn predict<'a>(&self, dst: &'a mut PlaneMutSlice<'a>, tx_size: TxSize) {
        match tx_size {
            TxSize::TX_4X4 => self.predict_inner::<Block4x4>(dst),
            TxSize::TX_8X8 => self.predict_inner::<Block8x8>(dst),
            TxSize::TX_16X16 => self.predict_inner::<Block16x16>(dst),
            TxSize::TX_32X32 => self.predict_inner::<Block32x32>(dst),
            _ => unimplemented!()
        }
    }

    #[inline(always)]
    fn predict_inner<'a, B: Intra>(&self, dst: &'a mut PlaneMutSlice<'a>) {
        let above = &mut [127u16; 64][..B::W];
        let left = &mut [129u16; 64][..B::H];
        let stride = dst.plane.cfg.stride;
        let x = dst.x;
        let y = dst.y;

        if (self == &PredictionMode::V_PRED ||
            self == &PredictionMode::DC_PRED) && y != 0 {
            above.copy_from_slice(&dst.go_up(1).as_slice()[..B::W]);
        }

        if (self == &PredictionMode::H_PRED ||
            self == &PredictionMode::DC_PRED) && x != 0 {
            let left_slice = dst.go_left(1);
            for i in 0..B::H {
                left[i] = left_slice.p(0, i);
            }
        }

        let slice = dst.as_mut_slice();

        match *self {
            PredictionMode::DC_PRED => {
                match (x, y) {
                    (0, 0) => B::pred_dc_128(slice, stride),
                    (_, 0) => B::pred_dc_left(slice, stride, above, left),
                    (0, _) => B::pred_dc_top(slice, stride, above, left),
                    _ => B::pred_dc(slice, stride, above, left),
                }
            },
            PredictionMode::H_PRED => B::pred_h(slice, stride, left),
            PredictionMode::V_PRED => B::pred_v(slice, stride, above),
            _ => unimplemented!(),
        }
    }

    pub fn is_directional(self) -> bool {
        self >= PredictionMode::V_PRED && self <= PredictionMode::D63_PRED
    }
}

#[derive(Copy,Clone)]
pub enum TxSetType {
    // DCT only
    EXT_TX_SET_DCTONLY = 0,
    // DCT + Identity only
    EXT_TX_SET_DCT_IDTX = 1,
    // Discrete Trig transforms w/o flip (4) + Identity (1)
    EXT_TX_SET_DTT4_IDTX = 2,
    // Discrete Trig transforms w/o flip (4) + Identity (1) + 1D Hor/vert DCT (2)
    EXT_TX_SET_DTT4_IDTX_1DDCT = 3,
    // Discrete Trig transforms w/ flip (9) + Identity (1) + 1D Hor/Ver DCT (2)
    EXT_TX_SET_DTT9_IDTX_1DDCT = 4,
    // Discrete Trig transforms w/ flip (9) + Identity (1) + 1D Hor/Ver (6)
    EXT_TX_SET_ALL16 = 5,
    EXT_TX_SET_TYPES
}

pub fn get_subsize(bsize: BlockSize , partition: PartitionType) -> BlockSize {
    subsize_lookup[partition as usize][bsize as usize]
}
