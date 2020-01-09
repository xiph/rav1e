// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::cpu_features::CpuFeatureLevel;
use crate::tiling::PlaneRegionMut;
use crate::transform::inverse::*;
use crate::transform::*;
use crate::{Pixel, PixelType};

use crate::asm::shared::transform::inverse::*;
use crate::asm::shared::transform::*;

pub fn inverse_transform_add<T: Pixel>(
  input: &[T::Coeff], output: &mut PlaneRegionMut<'_, T>, eob: usize,
  tx_size: TxSize, tx_type: TxType, bd: usize, cpu: CpuFeatureLevel,
) {
  match T::type_enum() {
    PixelType::U8 => {
      if let Some(func) = INV_TXFM_FNS[cpu.as_index()]
        [get_tx_size_idx(tx_size)][get_tx_type_idx(tx_type)]
      {
        return call_inverse_func(
          func,
          input,
          output,
          eob,
          tx_size.width(),
          tx_size.height(),
          bd,
        );
      }
    }
    PixelType::U16 => {}
  };

  native::inverse_transform_add(input, output, eob, tx_size, tx_type, bd, cpu);
}

macro_rules! decl_itx_fns {
  // Takes a 2d list of tx types for W and H
  ([$([$(($ENUM:pat, $TYPE1:ident, $TYPE2:ident)),*]),*], $W:expr, $H:expr,
   $OPT_LOWER:ident, $OPT_UPPER:ident) => {
    paste::item! {
      // For each tx type, declare an function for the current WxH
      $(
        $(
          extern {
            // Note: type1 and type2 are flipped
            fn [<rav1e_inv_txfm_add_ $TYPE2 _$TYPE1 _$W x $H _$OPT_LOWER>](
              dst: *mut u8, dst_stride: libc::ptrdiff_t, coeff: *mut i16,
              eob: i32
            );
          }
        )*
      )*
      // Create a lookup table for the tx types declared above
      const [<INV_TXFM_FNS_$W _$H _$OPT_UPPER>]: [Option<InvTxfmFunc>; TX_TYPES] = {
        let mut out: [Option<InvTxfmFunc>; 16] = [None; 16];
        $(
          $(
            out[get_tx_type_idx($ENUM)] = Some([<rav1e_inv_txfm_add_$TYPE2 _$TYPE1 _$W x $H _$OPT_LOWER>]);
          )*
        )*
        out
      };
    }
  };
}

macro_rules! create_wxh_tables {
  // Create a lookup table for each cpu feature
  ([$([$(($W:expr, $H:expr)),*]),*], $OPT_LOWER:ident, $OPT_UPPER:ident) => {
    paste::item! {
      const [<INV_TXFM_FNS_$OPT_UPPER>]: [[Option<InvTxfmFunc>; TX_TYPES]; 32] = {
        let mut out: [[Option<InvTxfmFunc>; TX_TYPES]; 32] = [[None; TX_TYPES]; 32];
        // For each dimension, add an entry to the table
        $(
          $(
            out[get_tx_size_idx(TxSize::[<TX_ $W X $H>])] = [<INV_TXFM_FNS_$W _$H _$OPT_UPPER>];
          )*
        )*
        out
      };
    }
  };

  // Loop through cpu features
  ($DIMS:tt, [$(($OPT_LOWER:ident, $OPT_UPPER:ident)),+]) => {
    $(
      create_wxh_tables!($DIMS, $OPT_LOWER, $OPT_UPPER);
    )*
  };
}

macro_rules! impl_itx_fns {
  ($TYPES:tt, $W:expr, $H:expr, [$(($OPT_LOWER:ident, $OPT_UPPER:ident)),+]) => {
    $(
      decl_itx_fns!($TYPES, $W, $H, $OPT_LOWER, $OPT_UPPER);
    )*
  };

  // Loop over a list of dimensions
  ($TYPES_VALID:tt, [$(($W:expr, $H:expr)),*], $OPT:tt) => {
    $(
      impl_itx_fns!($TYPES_VALID, $W, $H, $OPT);
    )*
  };

  ($TYPES64:tt, $DIMS64:tt, $TYPES32:tt, $DIMS32:tt, $TYPES16:tt, $DIMS16:tt,
   $TYPES84:tt, $DIMS84:tt, $OPT:tt) => {
    // Make 2d list of tx types for each set of dimensions. Each set of
    //   dimensions uses a superset of the previous set of tx types.
    impl_itx_fns!([$TYPES64], $DIMS64, $OPT);
    impl_itx_fns!([$TYPES64, $TYPES32], $DIMS32, $OPT);
    impl_itx_fns!([$TYPES64, $TYPES32, $TYPES16], $DIMS16, $OPT);
    impl_itx_fns!(
      [$TYPES64, $TYPES32, $TYPES16, $TYPES84], $DIMS84, $OPT
    );

    // Pool all of the dimensions together to create a table for each cpu
    // feature level.
    create_wxh_tables!(
      [$DIMS64, $DIMS32, $DIMS16, $DIMS84], $OPT
    );
  };
}

impl_itx_fns!(
  // 64x
  [(TxType::DCT_DCT, dct, dct)],
  [(64, 64), (64, 32), (32, 64), (16, 64), (64, 16)],
  // 32x
  // TODO: We are excluding identity transform as tests are failing with
  // memory misalignment for 4 cases:
  // inv_txfm2d_add_identity_identity_16x32_neon,
  // inv_txfm2d_add_identity_identity_32x16_neon,
  // inv_txfm2d_add_identity_identity_32x32_neon,
  // inv_txfm2d_add_identity_identity_8x32_neon
  //[(TxType::IDTX, identity, identity)],
  [],
  [(32, 32), (32, 16), (16, 32), (32, 8), (8, 32)],
  // 16x16
  [
    (TxType::DCT_ADST, dct, adst),
    (TxType::ADST_DCT, adst, dct),
    (TxType::DCT_FLIPADST, dct, flipadst),
    (TxType::FLIPADST_DCT, flipadst, dct),
    (TxType::V_DCT, dct, identity),
    (TxType::H_DCT, identity, dct),
    (TxType::ADST_ADST, adst, adst),
    (TxType::ADST_FLIPADST, adst, flipadst),
    (TxType::FLIPADST_ADST, flipadst, adst),
    (TxType::FLIPADST_FLIPADST, flipadst, flipadst)
  ],
  [(16, 16)],
  // 8x, 4x and 16x (minus 16x16)
  [
    (TxType::V_ADST, adst, identity),
    (TxType::H_ADST, identity, adst),
    (TxType::V_FLIPADST, flipadst, identity),
    (TxType::H_FLIPADST, identity, flipadst)
  ],
  [(16, 8), (8, 16), (16, 4), (4, 16), (8, 8), (8, 4), (4, 8), (4, 4)],
  [(neon, NEON)]
);

static INV_TXFM_FNS: [[[Option<InvTxfmFunc>; TX_TYPES]; 32];
  CpuFeatureLevel::len()] = {
  let mut out: [[[Option<InvTxfmFunc>; TX_TYPES]; 32];
    CpuFeatureLevel::len()] = [[[None; TX_TYPES]; 32]; CpuFeatureLevel::len()];
  out[CpuFeatureLevel::NEON as usize] = INV_TXFM_FNS_NEON;
  out
};

#[cfg(test)]
mod test {
  use super::*;
  use crate::asm::shared::transform::inverse::test::*;
  use crate::transform::TxSize::*;

  macro_rules! test_itx_fns {
    ($(($ENUM:pat, $TYPE1:ident, $TYPE2:ident, $W:expr, $H:expr)),*, $OPT:ident, $OPTLIT:literal, $OPT_ENUM:pat) => {
      $(
        paste::item! {
          #[test]
          fn [<inv_txfm2d_add_$TYPE2 _$TYPE1 _$W x $H _$OPT>]() {
            test_transform(
              [<TX_ $W X $H>],
              $ENUM,
              $OPT_ENUM,
            );
          }
        }
      )*
    }
  }

  test_itx_fns!(
    (TxType::DCT_DCT, dct, dct, 64, 64),
    (TxType::DCT_DCT, dct, dct, 64, 32),
    (TxType::DCT_DCT, dct, dct, 32, 64),
    (TxType::DCT_DCT, dct, dct, 16, 64),
    (TxType::DCT_DCT, dct, dct, 64, 16),
    (TxType::IDTX, identity, identity, 32, 32),
    (TxType::IDTX, identity, identity, 32, 16),
    (TxType::IDTX, identity, identity, 16, 32),
    (TxType::IDTX, identity, identity, 32, 8),
    (TxType::IDTX, identity, identity, 8, 32),
    (TxType::DCT_ADST, dct, adst, 16, 16),
    (TxType::ADST_DCT, adst, dct, 16, 16),
    (TxType::DCT_FLIPADST, dct, flipadst, 16, 16),
    (TxType::FLIPADST_DCT, flipadst, dct, 16, 16),
    (TxType::V_DCT, dct, identity, 16, 16),
    (TxType::H_DCT, identity, dct, 16, 16),
    (TxType::ADST_ADST, adst, adst, 16, 16),
    (TxType::ADST_FLIPADST, adst, flipadst, 16, 16),
    (TxType::FLIPADST_ADST, flipadst, adst, 16, 16),
    (TxType::FLIPADST_FLIPADST, flipadst, flipadst, 16, 16),
    (TxType::V_ADST, adst, identity, 16, 8),
    (TxType::H_ADST, identity, adst, 16, 8),
    (TxType::V_FLIPADST, flipadst, identity, 16, 8),
    (TxType::H_FLIPADST, identity, flipadst, 16, 8),
    (TxType::V_ADST, adst, identity, 8, 16),
    (TxType::H_ADST, identity, adst, 8, 16),
    (TxType::V_FLIPADST, flipadst, identity, 8, 16),
    (TxType::H_FLIPADST, identity, flipadst, 8, 16),
    (TxType::V_ADST, adst, identity, 16, 4),
    (TxType::H_ADST, identity, adst, 16, 4),
    (TxType::V_FLIPADST, flipadst, identity, 16, 4),
    (TxType::H_FLIPADST, identity, flipadst, 16, 4),
    (TxType::V_ADST, adst, identity, 4, 16),
    (TxType::H_ADST, identity, adst, 4, 16),
    (TxType::V_FLIPADST, flipadst, identity, 4, 16),
    (TxType::H_FLIPADST, identity, flipadst, 4, 16),
    (TxType::V_ADST, adst, identity, 8, 8),
    (TxType::H_ADST, identity, adst, 8, 8),
    (TxType::V_FLIPADST, flipadst, identity, 8, 8),
    (TxType::H_FLIPADST, identity, flipadst, 8, 8),
    (TxType::V_ADST, adst, identity, 8, 4),
    (TxType::H_ADST, identity, adst, 8, 4),
    (TxType::V_FLIPADST, flipadst, identity, 8, 4),
    (TxType::H_FLIPADST, identity, flipadst, 8, 4),
    (TxType::V_ADST, adst, identity, 4, 8),
    (TxType::H_ADST, identity, adst, 4, 8),
    (TxType::V_FLIPADST, flipadst, identity, 4, 8),
    (TxType::H_FLIPADST, identity, flipadst, 4, 8),
    (TxType::V_ADST, adst, identity, 4, 4),
    (TxType::H_ADST, identity, adst, 4, 4),
    (TxType::V_FLIPADST, flipadst, identity, 4, 4),
    (TxType::H_FLIPADST, identity, flipadst, 4, 4),
    neon,
    "neon",
    CpuFeatureLevel::NEON
  );
}
