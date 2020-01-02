use crate::partition::BlockSize;
use crate::rdo::{Distortion, RawDistortion};
use crate::tiling::{Area, PlaneRegion};
use crate::util::AlignedArray;
use crate::{CastFromPrimitive, Pixel};
use std::arch::x86_64::*;

#[target_feature(enable = "sse2")]
pub(crate) unsafe fn sse_wxh_8x8_sse2<
  T: Pixel,
  F: Fn(Area, BlockSize) -> f64,
>(
  src1: &PlaneRegion<'_, T>, src2: &PlaneRegion<'_, T>, w: usize, h: usize,
  compute_bias: F,
) -> Distortion {
  let block_h = 8;
  let block_w = 8;

  let mut sse = Distortion::zero();
  for block_y in 0..h / block_h {
    let y_offset = block_y * block_h;

    for block_x in 0..w / block_w {
      let x_offset = block_x * block_w;

      let mut value = 0;
      for j in 0..block_h {
        let src1_ptr = src1[y_offset + j].as_ptr().add(x_offset);
        let src2_ptr = src2[y_offset + j].as_ptr().add(x_offset);

        let s1 = _mm_set_epi16(
          i16::cast_from(*src1_ptr.add(7)),
          i16::cast_from(*src1_ptr.add(6)),
          i16::cast_from(*src1_ptr.add(5)),
          i16::cast_from(*src1_ptr.add(4)),
          i16::cast_from(*src1_ptr.add(3)),
          i16::cast_from(*src1_ptr.add(2)),
          i16::cast_from(*src1_ptr.add(1)),
          i16::cast_from(*src1_ptr.add(0)),
        );
        let s2 = _mm_set_epi16(
          i16::cast_from(*src2_ptr.add(7)),
          i16::cast_from(*src2_ptr.add(6)),
          i16::cast_from(*src2_ptr.add(5)),
          i16::cast_from(*src2_ptr.add(4)),
          i16::cast_from(*src2_ptr.add(3)),
          i16::cast_from(*src2_ptr.add(2)),
          i16::cast_from(*src2_ptr.add(1)),
          i16::cast_from(*src2_ptr.add(0)),
        );

        let mut results: AlignedArray<[i32; 4]> =
          AlignedArray::uninitialized();
        let diffs = _mm_sub_epi16(s1, s2);
        let squares = _mm_madd_epi16(diffs, diffs);
        _mm_store_si128(results.array.as_mut_ptr() as *mut _, squares);

        value +=
          results.array.iter().copied().map(|val| val as u64).sum::<u64>();
      }

      let bias = compute_bias(
        // StartingAt gives the correct block offset.
        Area::StartingAt { x: x_offset as isize, y: y_offset as isize },
        BlockSize::BLOCK_8X8,
      );
      sse += RawDistortion::new(value) * bias;
    }
  }
  sse
}
