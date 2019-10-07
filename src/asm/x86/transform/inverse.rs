use crate::tiling::PlaneRegionMut;
use crate::transform::*;
use crate::util::AlignedArray;
use crate::Pixel;

type InvTxfmFunc = unsafe extern fn(*mut u8, libc::ptrdiff_t, *const i16, i32);

pub trait InvTxfm2D: native::InvTxfm2D {
  fn match_tx_type(tx_type: TxType) -> InvTxfmFunc;

  fn inv_txfm2d_add<T>(
    input: &[i32], output: &mut PlaneRegionMut<'_, T>, tx_type: TxType,
    bd: usize,
  ) where
    T: Pixel,
  {
    if std::mem::size_of::<T>() == 1 && is_x86_feature_detected!("avx2") {
      debug_assert!(bd == 8);

      // 64x only uses 32 coeffs
      let coeff_w = Self::W.min(32);
      let coeff_h = Self::H.min(32);
      let mut coeff16: AlignedArray<[i16; 32 * 32]> =
        AlignedArray::uninitialized();

      // Transpose the input.
      // TODO: should be possible to remove changing how coeffs are written
      assert!(input.len() >= coeff_w * coeff_h);
      for j in 0..coeff_h {
        for i in 0..coeff_w {
          coeff16.array[i * coeff_h + j] = input[j * coeff_w + i] as i16;
        }
      }

      let stride = output.plane_cfg.stride as isize;
      unsafe {
        // perform the inverse transform
        Self::match_tx_type(tx_type)(
          output.data_ptr_mut() as *mut _,
          stride,
          coeff16.array.as_ptr(),
          (coeff_w * coeff_h) as i32,
        );
      }
      return;
    }
    <Self as native::InvTxfm2D>::inv_txfm2d_add(input, output, tx_type, bd);
  }
}

macro_rules! impl_itx_fns {
    // Takes a 2d list of tx types for W and H
    ([$([$(($ENUM:pat, $TYPE1:ident, $TYPE2:ident)),*]),*], $W:expr, $H:expr,
     $OPT:ident) => {
      paste::item! {
        // For each tx type, declare an function for the current WxH
        $(
          $(
            extern {
              // Note: type1 and type2 are flipped
              fn [<rav1e_inv_txfm_add_ $TYPE2 _$TYPE1 _$W x $H _$OPT>](
                dst: *mut u8, dst_stride: libc::ptrdiff_t, coeff: *const i16,
                eob: i32
              );
            }
          )*
        )*

        // Implement InvTxfm2D for WxH
        impl InvTxfm2D for crate::predict::[<Block $W x $H>] {
          fn match_tx_type(tx_type: TxType) -> InvTxfmFunc {
            // Match tx types we declared earlier to its rust enum
            match tx_type {
              $(
                $(
                  // Suppress unreachable pattern warning for _
                  a if a == $ENUM => {
                    // Note: type1 and type2 are flipped
                    [<rav1e_inv_txfm_add_$TYPE2 _$TYPE1 _$W x $H _$OPT>]
                  },
                )*
              )*
              _ => unreachable!()
            }
          }
        }
      }
    };

    // Loop over a list of dimensions
    ($TYPES_VALID:tt, [$(($W:expr, $H:expr)),*], $OPT:ident) => {
      $(
        impl_itx_fns!($TYPES_VALID, $W, $H, $OPT);
      )*
    };

    ($TYPES64:tt, $DIMS64:tt, $TYPES32:tt, $DIMS32:tt, $TYPES16:tt, $DIMS16:tt,
     $TYPES84:tt, $DIMS84:tt, $OPT:ident) => {
      // Make 2d list of tx types for each set of dimensions. Each set of
      //   dimensions uses a superset of the previous set of tx types.
      impl_itx_fns!([$TYPES64], $DIMS64, $OPT);
      impl_itx_fns!([$TYPES64, $TYPES32], $DIMS32, $OPT);
      impl_itx_fns!([$TYPES64, $TYPES32, $TYPES16], $DIMS16, $OPT);
      impl_itx_fns!(
        [$TYPES64, $TYPES32, $TYPES16, $TYPES84], $DIMS84, $OPT
      );
    };
  }

impl_itx_fns!(
  // 64x
  [(TxType::DCT_DCT, dct, dct)],
  [(64, 64), (64, 32), (32, 64), (16, 64), (64, 16)],
  // 32x
  [(TxType::IDTX, identity, identity)],
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
  avx2
);
