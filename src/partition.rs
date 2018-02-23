#![allow(non_camel_case_types)]
#![allow(dead_code)]

#[derive(Copy,Clone,PartialEq)]
pub enum PartitionType {
    PARTITION_NONE,
    PARTITION_HORZ,
    PARTITION_VERT,
    PARTITION_SPLIT,
    PARTITION_INVALID = 255
}

pub const BLOCK_SIZES_ALL: usize = 19;

#[derive(Copy,Clone)]
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
    BLOCK_INVALID = 255
}

pub const TX_SIZES: usize = 4;
pub const TX_SIZES_ALL: usize = 14;

#[derive(Copy,Clone,PartialEq,PartialOrd)]
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

#[derive(Copy, Clone, Debug, PartialEq)]
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
    TM_PRED,        // True-motion
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

fn setup_left(left: &mut [u16; 4], rec: &PlaneMutSlice) {
    let left_slice = rec.go_left(1);
    for i in 0..4 {
        left[i] = left_slice.p(0, i);
    }
}

impl PredictionMode {
    pub fn predict_4x4<'a>(&self, dst: &'a mut PlaneMutSlice<'a>) {
        let mut above = [127u16; 4];
        let mut left = [129u16; 4];
        let stride = dst.plane.cfg.stride;
        let x = dst.x;
        let y = dst.y;

        if (self == &PredictionMode::V_PRED ||
            self == &PredictionMode::DC_PRED) && y != 0 {
            above.copy_from_slice(&dst.go_up(1).as_slice()[..4]);
        }

        if (self == &PredictionMode::H_PRED ||
            self == &PredictionMode::DC_PRED) && x != 0 {
            setup_left(&mut left, dst);
        }

        let slice = dst.as_mut_slice();

        match *self {
            PredictionMode::DC_PRED => {
                match (x, y) {
                    (0, 0) =>
                        pred_dc_128(slice, stride),
                    (_, 0) => {
                        pred_dc_left_4x4(slice, stride, &above[..4], &left[..4]);
                    },
                    (0, _) => {
                        pred_dc_top_4x4(slice, stride, &above[..4], &left[..4])
                    },
                    _ => {
                        Block4x4::pred_dc(slice, stride, &above[..4], &left[..4]);
                    }
                }
            },
            PredictionMode::H_PRED => {
                Block4x4::pred_h(slice, stride, &left[..4]);
            },
            PredictionMode::V_PRED => {
                Block4x4::pred_v(slice, stride, &above[..4]);
            },
            _ => unimplemented!(),
        }
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
