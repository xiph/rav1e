use crate::cpu_features::CpuFeatureLevel;
use crate::tiling::PlaneRegionMut;
use crate::transform::*;
use crate::util::AlignedArray;
use crate::Pixel;

type InvTxfmFunc = unsafe extern fn(*mut u8, libc::ptrdiff_t, *const i16, i32);

pub trait InvTxfm2D: native::InvTxfm2D {
  fn match_tx_type(tx_type: TxType) -> InvTxfmFunc;

  fn inv_txfm2d_add<T>(
    input: &[i32], output: &mut PlaneRegionMut<'_, T>, tx_type: TxType,
    bd: usize, cpu: CpuFeatureLevel,
  ) where
    T: Pixel,
  {
    if std::mem::size_of::<T>() == 1 && cpu >= CpuFeatureLevel::AVX2 {
      return unsafe {
        Self::inv_txfm2d_add_avx2(input, output, tx_type, bd);
      };
    }
    if Self::W == 4
      && Self::H == 4
      && std::mem::size_of::<T>() == 1
      && cpu >= CpuFeatureLevel::SSSE3
    {
      return unsafe {
        Self::inv_txfm2d_add_ssse3(input, output, tx_type, bd);
      };
    }
    <Self as native::InvTxfm2D>::inv_txfm2d_add(
      input, output, tx_type, bd, cpu,
    );
  }

  #[inline]
  #[target_feature(enable = "avx2")]
  unsafe fn inv_txfm2d_add_avx2<T>(
    input: &[i32], output: &mut PlaneRegionMut<'_, T>, tx_type: TxType,
    bd: usize,
  ) where
    T: Pixel,
  {
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

    // perform the inverse transform
    Self::match_tx_type(tx_type)(
      output.data_ptr_mut() as *mut _,
      stride,
      coeff16.array.as_ptr(),
      (coeff_w * coeff_h) as i32,
    );
  }

  #[inline]
  #[target_feature(enable = "ssse3")]
  unsafe fn inv_txfm2d_add_ssse3<T>(
    input: &[i32], output: &mut PlaneRegionMut<'_, T>, tx_type: TxType,
    bd: usize,
  ) where
    T: Pixel,
  {
    debug_assert!(bd == 8);

    // TODO: Support more than Block_4x4 when SSSE3 and AVX2 match.
    debug_assert!(Self::W == 4 && Self::H == 4);

    let mut coeff16: AlignedArray<[i16; 4 * 4]> =
      AlignedArray::uninitialized();

    // Transpose the input.
    assert!(input.len() >= 4 * 4);
    for j in 0..4 {
      for i in 0..4 {
        coeff16.array[i * 4 + j] = input[j * 4 + i] as i16;
      }
    }

    let stride = output.plane_cfg.stride as isize;

    // perform the inverse transform
    (match tx_type {
      TxType::DCT_DCT => rav1e_inv_txfm_add_dct_dct_4x4_ssse3,
      TxType::IDTX => rav1e_inv_txfm_add_identity_identity_4x4_ssse3,
      TxType::DCT_ADST => rav1e_inv_txfm_add_adst_dct_4x4_ssse3,
      TxType::ADST_DCT => rav1e_inv_txfm_add_dct_adst_4x4_ssse3,
      TxType::DCT_FLIPADST => rav1e_inv_txfm_add_flipadst_dct_4x4_ssse3,
      TxType::FLIPADST_DCT => rav1e_inv_txfm_add_dct_flipadst_4x4_ssse3,
      TxType::V_DCT => rav1e_inv_txfm_add_identity_dct_4x4_ssse3,
      TxType::H_DCT => rav1e_inv_txfm_add_dct_identity_4x4_ssse3,
      TxType::ADST_ADST => rav1e_inv_txfm_add_adst_adst_4x4_ssse3,
      TxType::ADST_FLIPADST => rav1e_inv_txfm_add_flipadst_adst_4x4_ssse3,
      TxType::FLIPADST_ADST => rav1e_inv_txfm_add_adst_flipadst_4x4_ssse3,
      TxType::FLIPADST_FLIPADST => {
        rav1e_inv_txfm_add_flipadst_flipadst_4x4_ssse3
      }
      TxType::V_ADST => rav1e_inv_txfm_add_identity_adst_4x4_ssse3,
      TxType::H_ADST => rav1e_inv_txfm_add_adst_identity_4x4_ssse3,
      TxType::V_FLIPADST => rav1e_inv_txfm_add_identity_flipadst_4x4_ssse3,
      TxType::H_FLIPADST => rav1e_inv_txfm_add_flipadst_identity_4x4_ssse3,
    })(
      output.data_ptr_mut() as *mut _,
      stride,
      coeff16.array.as_ptr(),
      (4 * 4) as i32,
    );
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

macro_rules! decl_tx_fns {
  ($($f:ident),+) => {
    extern {
      $(
        fn $f(
          dst: *mut u8, dst_stride: libc::ptrdiff_t, coeff: *const i16,
          eob: i32
        );
      )*
    }
  };
}

decl_tx_fns! {
  rav1e_inv_txfm_add_dct_dct_4x4_ssse3,
  rav1e_inv_txfm_add_identity_identity_4x4_ssse3,
  rav1e_inv_txfm_add_adst_dct_4x4_ssse3,
  rav1e_inv_txfm_add_dct_adst_4x4_ssse3,
  rav1e_inv_txfm_add_flipadst_dct_4x4_ssse3,
  rav1e_inv_txfm_add_dct_flipadst_4x4_ssse3,
  rav1e_inv_txfm_add_identity_dct_4x4_ssse3,
  rav1e_inv_txfm_add_dct_identity_4x4_ssse3,
  rav1e_inv_txfm_add_adst_adst_4x4_ssse3,
  rav1e_inv_txfm_add_flipadst_adst_4x4_ssse3,
  rav1e_inv_txfm_add_adst_flipadst_4x4_ssse3,
  rav1e_inv_txfm_add_flipadst_flipadst_4x4_ssse3,
  rav1e_inv_txfm_add_identity_adst_4x4_ssse3,
  rav1e_inv_txfm_add_adst_identity_4x4_ssse3,
  rav1e_inv_txfm_add_identity_flipadst_4x4_ssse3,
  rav1e_inv_txfm_add_flipadst_identity_4x4_ssse3
}

#[cfg(test)]
#[rustfmt::skip]
mod test {
  use super::*;
  use crate::frame::{AsRegion, Plane};
  use crate::transform::TxSize::*;
  use rand::random;

  macro_rules! decl_tx_fn_test {
    ($ENUM:pat, $TYPE1:ident, $TYPE2:ident, $W:expr, $H:expr, $OPT:ident, $OPTLIT:literal) => {
      paste::item! {
        #[test]
        fn [<inv_txfm2d_add_$TYPE2 _$TYPE1 _$W x $H _$OPT>]() {
          if !is_x86_feature_detected!($OPTLIT) {
            eprintln!("Ignoring {} test, not supported on this machine!", $OPTLIT);
            return;
          }

          let tx_size = [<TX_ $W X $H>];
          let mut src_storage = [0u8; 64 * 64];
          let src = &mut src_storage[..tx_size.area()];
          let mut dst = Plane::wrap(vec![0u8; tx_size.area()], tx_size.width());
          let mut res_storage = [0i16; 64 * 64];
          let res = &mut res_storage[..tx_size.area()];
          let mut freq_storage = [0i32; 64 * 64];
          let freq = &mut freq_storage[..tx_size.area()];
          for ((r, s), d) in
            res.iter_mut().zip(src.iter_mut()).zip(dst.data.iter_mut())
          {
            *s = random::<u8>();
            *d = random::<u8>();
            *r = i16::from(*s) - i16::from(*d);
          }
          forward_transform(res, freq, tx_size.width(), tx_size, $ENUM, 8);
          let mut native_dst = dst.clone();

          unsafe { crate::predict::[<Block $W x $H>]::[<inv_txfm2d_add_ $OPT>](
            freq, &mut dst.as_region_mut(), $ENUM, 8
          ); }
          <crate::predict::[<Block $W x $H>] as native::InvTxfm2D>::inv_txfm2d_add(
            freq, &mut native_dst.as_region_mut(), $ENUM, 8, CpuFeatureLevel::NATIVE
          );
          assert_eq!(native_dst.data_origin(), dst.data_origin());
        }
      }
    }
  }

  // TODO: There is no Rust code for FLIPADST,
  // so those tests will currently fail.
  // They are commented out below for reference.
  decl_tx_fn_test!(TxType::DCT_DCT, dct, dct, 64, 64, avx2, "avx2");
  decl_tx_fn_test!(TxType::DCT_DCT, dct, dct, 64, 32, avx2, "avx2");
  decl_tx_fn_test!(TxType::DCT_DCT, dct, dct, 32, 64, avx2, "avx2");
  decl_tx_fn_test!(TxType::DCT_DCT, dct, dct, 16, 64, avx2, "avx2");
  decl_tx_fn_test!(TxType::DCT_DCT, dct, dct, 64, 16, avx2, "avx2");
  decl_tx_fn_test!(TxType::IDTX, identity, identity, 32, 32, avx2, "avx2");
  decl_tx_fn_test!(TxType::IDTX, identity, identity, 32, 16, avx2, "avx2");
  decl_tx_fn_test!(TxType::IDTX, identity, identity, 16, 32, avx2, "avx2");
  decl_tx_fn_test!(TxType::IDTX, identity, identity, 32, 8, avx2, "avx2");
  decl_tx_fn_test!(TxType::IDTX, identity, identity, 8, 32, avx2, "avx2");
  decl_tx_fn_test!(TxType::DCT_ADST, dct, adst, 16, 16, avx2, "avx2");
  decl_tx_fn_test!(TxType::ADST_DCT, adst, dct, 16, 16, avx2, "avx2");
//  decl_tx_fn_test!(TxType::DCT_FLIPADST, dct, flipadst, 16, 16, avx2, "avx2");
//  decl_tx_fn_test!(TxType::FLIPADST_DCT, flipadst, dct, 16, 16, avx2, "avx2");
  decl_tx_fn_test!(TxType::V_DCT, dct, identity, 16, 16, avx2, "avx2");
  decl_tx_fn_test!(TxType::H_DCT, identity, dct, 16, 16, avx2, "avx2");
  decl_tx_fn_test!(TxType::ADST_ADST, adst, adst, 16, 16, avx2, "avx2");
//  decl_tx_fn_test!(TxType::ADST_FLIPADST, adst, flipadst, 16, 16, avx2, "avx2");
//  decl_tx_fn_test!(TxType::FLIPADST_ADST, flipadst, adst, 16, 16, avx2, "avx2");
//  decl_tx_fn_test!(TxType::FLIPADST_FLIPADST, flipadst, flipadst, 16, 16, avx2, "avx2");
  decl_tx_fn_test!(TxType::V_ADST, adst, identity, 16, 8, avx2, "avx2");
  decl_tx_fn_test!(TxType::H_ADST, identity, adst, 16, 8, avx2, "avx2");
//  decl_tx_fn_test!(TxType::V_FLIPADST, flipadst, identity, 16, 8, avx2, "avx2");
//  decl_tx_fn_test!(TxType::H_FLIPADST, identity, flipadst, 16, 8, avx2, "avx2");
  decl_tx_fn_test!(TxType::V_ADST, adst, identity, 8, 16, avx2, "avx2");
  decl_tx_fn_test!(TxType::H_ADST, identity, adst, 8, 16, avx2, "avx2");
//  decl_tx_fn_test!(TxType::V_FLIPADST, flipadst, identity, 8, 16, avx2, "avx2");
//  decl_tx_fn_test!(TxType::H_FLIPADST, identity, flipadst, 8, 16, avx2, "avx2");
  decl_tx_fn_test!(TxType::V_ADST, adst, identity, 16, 4, avx2, "avx2");
  decl_tx_fn_test!(TxType::H_ADST, identity, adst, 16, 4, avx2, "avx2");
//  decl_tx_fn_test!(TxType::V_FLIPADST, flipadst, identity, 16, 4, avx2, "avx2");
//  decl_tx_fn_test!(TxType::H_FLIPADST, identity, flipadst, 16, 4, avx2, "avx2");
  decl_tx_fn_test!(TxType::V_ADST, adst, identity, 4, 16, avx2, "avx2");
  decl_tx_fn_test!(TxType::H_ADST, identity, adst, 4, 16, avx2, "avx2");
//  decl_tx_fn_test!(TxType::V_FLIPADST, flipadst, identity, 4, 16, avx2, "avx2");
//  decl_tx_fn_test!(TxType::H_FLIPADST, identity, flipadst, 4, 16, avx2, "avx2");
  decl_tx_fn_test!(TxType::V_ADST, adst, identity, 8, 8, avx2, "avx2");
  decl_tx_fn_test!(TxType::H_ADST, identity, adst, 8, 8, avx2, "avx2");
//  decl_tx_fn_test!(TxType::V_FLIPADST, flipadst, identity, 8, 8, avx2, "avx2");
//  decl_tx_fn_test!(TxType::H_FLIPADST, identity, flipadst, 8, 8, avx2, "avx2");
  decl_tx_fn_test!(TxType::V_ADST, adst, identity, 8, 4, avx2, "avx2");
  decl_tx_fn_test!(TxType::H_ADST, identity, adst, 8, 4, avx2, "avx2");
//  decl_tx_fn_test!(TxType::V_FLIPADST, flipadst, identity, 8, 4, avx2, "avx2");
//  decl_tx_fn_test!(TxType::H_FLIPADST, identity, flipadst, 8, 4, avx2, "avx2");
  decl_tx_fn_test!(TxType::V_ADST, adst, identity, 4, 8, avx2, "avx2");
  decl_tx_fn_test!(TxType::H_ADST, identity, adst, 4, 8, avx2, "avx2");
//  decl_tx_fn_test!(TxType::V_FLIPADST, flipadst, identity, 4, 8, avx2, "avx2");
//  decl_tx_fn_test!(TxType::H_FLIPADST, identity, flipadst, 4, 8, avx2, "avx2");
  decl_tx_fn_test!(TxType::V_ADST, adst, identity, 4, 4, avx2, "avx2");
  decl_tx_fn_test!(TxType::H_ADST, identity, adst, 4, 4, avx2, "avx2");
//  decl_tx_fn_test!(TxType::V_FLIPADST, flipadst, identity, 4, 4, avx2, "avx2");
//  decl_tx_fn_test!(TxType::H_FLIPADST, identity, flipadst, 4, 4, avx2, "avx2");

  decl_tx_fn_test!(TxType::DCT_DCT, dct, dct, 4, 4, ssse3, "ssse3");
  decl_tx_fn_test!(TxType::IDTX, identity, identity, 4, 4, ssse3, "ssse3");
  decl_tx_fn_test!(TxType::DCT_ADST, dct, adst, 4, 4, ssse3, "ssse3");
  decl_tx_fn_test!(TxType::ADST_DCT, adst, dct, 4, 4, ssse3, "ssse3");
//  decl_tx_fn_test!(TxType::DCT_FLIPADST, dct, flipadst, 4, 4, ssse3, "ssse3");
//  decl_tx_fn_test!(TxType::FLIPADST_DCT, flipadst, dct, 4, 4, ssse3, "ssse3");
  decl_tx_fn_test!(TxType::V_DCT, dct, identity, 4, 4, ssse3, "ssse3");
  decl_tx_fn_test!(TxType::H_DCT, identity, dct, 4, 4, ssse3, "ssse3");
  decl_tx_fn_test!(TxType::ADST_ADST, adst, adst, 4, 4, ssse3, "ssse3");
//  decl_tx_fn_test!(TxType::ADST_FLIPADST, adst, flipadst, 4, 4, ssse3, "ssse3");
//  decl_tx_fn_test!(TxType::FLIPADST_ADST, flipadst, adst, 4, 4, ssse3, "ssse3");
//  decl_tx_fn_test!(TxType::FLIPADST_FLIPADST, flipadst, flipadst, 4, 4, ssse3, "ssse3");
  decl_tx_fn_test!(TxType::V_ADST, adst, identity, 4, 4, ssse3, "ssse3");
  decl_tx_fn_test!(TxType::H_ADST, identity, adst, 4, 4, ssse3, "ssse3");
//  decl_tx_fn_test!(TxType::V_FLIPADST, flipadst, identity, 4, 4, ssse3, "ssse3");
//  decl_tx_fn_test!(TxType::H_FLIPADST, identity, flipadst, 4, 4, ssse3, "ssse3");
}
