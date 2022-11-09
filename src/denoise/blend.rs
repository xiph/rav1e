//! A pox on the house of whoever decided this was a good way to code anything

use std::mem::{size_of, transmute};

use arrayvec::ArrayVec;
use wide::{f32x8, f64x4, i32x8, i64x4};

use crate::util::cast;

use super::{f32x16, f64x8};

const V_DC: i32 = -256;

pub fn blend16<
  const I0: usize,
  const I1: usize,
  const I2: usize,
  const I3: usize,
  const I4: usize,
  const I5: usize,
  const I6: usize,
  const I7: usize,
  const I8: usize,
  const I9: usize,
  const I10: usize,
  const I11: usize,
  const I12: usize,
  const I13: usize,
  const I14: usize,
  const I15: usize,
>(
  a: f32x16, b: f32x16,
) -> f32x16 {
  let x0 = blend_half16::<I0, I1, I2, I3, I4, I5, I6, I7>(a, b);
  let x1 = blend_half16::<I8, I9, I10, I11, I12, I13, I14, I15>(a, b);
  [x0, x1]
}

fn blend_half16<
  const I0: usize,
  const I1: usize,
  const I2: usize,
  const I3: usize,
  const I4: usize,
  const I5: usize,
  const I6: usize,
  const I7: usize,
>(
  a: f32x16, b: f32x16,
) -> f32x8 {
  const N: usize = 8;
  let ind: [usize; N] = [I0, I1, I2, I3, I4, I5, I6, I7];

  // lambda to find which of the four possible sources are used
  fn list_sources(ind: &[usize; N]) -> ArrayVec<usize, 4> {
    let mut source_used = [false; 4];
    for i in 0..N {
      let ix = ind[i];
      let src = ix / N;
      source_used[src & 3] = true;
    }
    // return a list of sources used.
    let mut sources = ArrayVec::new();
    for i in 0..4 {
      if source_used[i] {
        sources.push(i);
      }
    }
    sources
  }

  let sources = list_sources(&ind);
  if sources.is_empty() {
    return f32x8::ZERO;
  }

  // get indexes for the first one or two sources
  let uindex = if sources.len() > 2 { 1 } else { 2 };
  let l = blend_half_indexes::<8>(
    uindex,
    sources.get(0).copied(),
    sources.get(1).copied(),
    &ind,
  );
  let src0 = select_blend16(sources.get(0).copied(), a, b);
  let src1 = select_blend16(sources.get(1).copied(), a, b);
  let mut x0 = blend8_f32(cast(&l[..8]), src0, src1);

  // get last one or two sources
  if sources.len() > 2 {
    let m = blend_half_indexes::<8>(
      1,
      sources.get(2).copied(),
      sources.get(3).copied(),
      &ind,
    );
    let src2 = select_blend16(sources.get(2).copied(), a, b);
    let src3 = select_blend16(sources.get(3).copied(), a, b);
    let x1 = blend8_f32(cast(&m[..8]), src2, src3);

    // combine result of two blends. Unused elements are zero
    x0 |= x1;
  }

  x0
}

fn select_blend16(action: Option<usize>, a: f32x16, b: f32x16) -> f32x8 {
  match action {
    Some(0) => a[0],
    Some(1) => a[1],
    Some(2) => b[0],
    _ => b[1],
  }
}

#[inline(always)]
fn blend8_f32(indexes: &[i32; 8], a: f32x8, b: f32x8) -> f32x8 {
  let mut y = a;
  let flags = blend_flags::<8, { size_of::<f32x8>() }>(&indexes);

  assert!(flags & BLEND_OUTOFRANGE == 0);

  if flags & BLEND_ALLZERO > 0 {
    return f32x8::ZERO;
  }

  if flags & BLEND_LARGEBLOCK > 0 {
    // blend and permute 32-bit blocks
    let l = largeblock_perm::<8, 4>(indexes);
    // SAFETY: Types are of same size
    y = unsafe { transmute(blend4_f64(&l, transmute(a), transmute(b))) };
    if flags & BLEND_ADDZ == 0 {
      // no remaining zeroing
      return y;
    }
  } else if flags & BLEND_B == 0 {
    // nothing from b. just permute a
    return permute8(indexes, a);
  } else if flags & BLEND_A == 0 {
    let l = blend_perm_indexes::<8, 16, 2>(indexes);
    return permute8(cast(&l[8..]), b);
  } else if flags & (BLEND_PERMA | BLEND_PERMB) == 0 {
    // no permutation, only blending
    let mb = make_bit_mask::<8, 0x303>(indexes) as u8;
    y = f32x8::new([
      if mb & (1 << 0) > 0 { -1f32 } else { 0f32 },
      if mb & (1 << 1) > 0 { -1f32 } else { 0f32 },
      if mb & (1 << 2) > 0 { -1f32 } else { 0f32 },
      if mb & (1 << 3) > 0 { -1f32 } else { 0f32 },
      if mb & (1 << 4) > 0 { -1f32 } else { 0f32 },
      if mb & (1 << 5) > 0 { -1f32 } else { 0f32 },
      if mb & (1 << 6) > 0 { -1f32 } else { 0f32 },
      if mb & (1 << 7) > 0 { -1f32 } else { 0f32 },
    ])
    .blend(b, a);
  } else if flags & BLEND_PUNPCKLAB > 0 {
    y = a.interleave_low(b);
  } else if flags & BLEND_PUNPCKLBA > 0 {
    y = b.interleave_low(a);
  } else if flags & BLEND_PUNPCKHAB > 0 {
    y = a.interleave_high(b);
  } else if flags & BLEND_PUNPCKHBA > 0 {
    y = b.interleave_high(a);
  } else if flags & BLEND_SHUFAB > 0 {
    y = a.shuffle_nonconst(b, (flags >> BLEND_SHUFPATTERN) as u8);
  } else if flags & BLEND_SHUFBA > 0 {
    y = b.shuffle_nonconst(a, (flags >> BLEND_SHUFPATTERN) as u8);
  } else {
    // No special cases
    // permute a and b separately, then blend.
    let l = blend_perm_indexes::<8, 16, 0>(indexes);
    let ya = permute8(cast(&l[..8]), a);
    let yb = permute8(cast(&l[8..]), b);
    let mb = make_bit_mask::<8, 0x303>(indexes) as u8;
    y = f32x8::new([
      if mb & 0b10000000 > 0 { -1f32 } else { 0f32 },
      if mb & 0b01000000 > 0 { -1f32 } else { 0f32 },
      if mb & 0b00100000 > 0 { -1f32 } else { 0f32 },
      if mb & 0b00010000 > 0 { -1f32 } else { 0f32 },
      if mb & 0b00001000 > 0 { -1f32 } else { 0f32 },
      if mb & 0b00000100 > 0 { -1f32 } else { 0f32 },
      if mb & 0b00000010 > 0 { -1f32 } else { 0f32 },
      if mb & 0b00000001 > 0 { -1f32 } else { 0f32 },
    ])
    .blend(yb, ya);
  }
  if flags & BLEND_ZEROING > 0 {
    // additional zeroing needed
    let bm = i32x8::new(zero_mask_broad_8x32(indexes));
    // SAFETY: Types have the same size
    let bm1: f32x8 = unsafe { transmute(bm) };
    y = bm1 & y;
  }

  y
}

pub fn blend8<
  const I0: usize,
  const I1: usize,
  const I2: usize,
  const I3: usize,
  const I4: usize,
  const I5: usize,
  const I6: usize,
  const I7: usize,
>(
  a: f64x8, b: f64x8,
) -> f64x8 {
  let x0 = blend_half8::<I0, I1, I2, I3>(a, b);
  let x1 = blend_half8::<I4, I5, I6, I7>(a, b);
  [x0, x1]
}

fn blend_half8<
  const I0: usize,
  const I1: usize,
  const I2: usize,
  const I3: usize,
>(
  a: f64x8, b: f64x8,
) -> f64x4 {
  const N: usize = 4;
  let ind: [usize; N] = [I0, I1, I2, I3];

  // lambda to find which of the four possible sources are used
  fn list_sources(ind: &[usize; N]) -> ArrayVec<usize, 4> {
    let mut source_used = [false; 4];
    for i in 0..N {
      let ix = ind[i];
      let src = ix / N;
      source_used[src & 3] = true;
    }
    // return a list of sources used.
    let mut sources = ArrayVec::new();
    for i in 0..4 {
      if source_used[i] {
        sources.push(i);
      }
    }
    sources
  }

  let sources = list_sources(&ind);
  if sources.is_empty() {
    return f64x4::ZERO;
  }

  // get indexes for the first one or two sources
  let uindex = if sources.len() > 2 { 1 } else { 2 };
  let l = blend_half_indexes::<4>(
    uindex,
    sources.get(0).copied(),
    sources.get(1).copied(),
    &ind,
  );
  let src0 = select_blend8(sources.get(0).copied(), a, b);
  let src1 = select_blend8(sources.get(1).copied(), a, b);
  let mut x0 = blend4_f64(cast(&l[..4]), src0, src1);

  // get last one or two sources
  if sources.len() > 2 {
    let m = blend_half_indexes::<4>(
      1,
      sources.get(2).copied(),
      sources.get(3).copied(),
      &ind,
    );
    let src2 = select_blend8(sources.get(2).copied(), a, b);
    let src3 = select_blend8(sources.get(3).copied(), a, b);
    let x1 = blend4_f64(cast(&m[..4]), src2, src3);

    // combine result of two blends. Unused elements are zero
    x0 |= x1;
  }

  x0
}

// blend_half_indexes: return an Indexlist for emulating a blend function as
// blends or permutations from multiple sources
// dozero = 0: let unused elements be don't care. Multiple permutation results must be blended
// dozero = 1: zero unused elements in each permuation. Multiple permutation results can be OR'ed
// dozero = 2: indexes that are -1 or V_DC are preserved
// src1, src2: sources to blend in a partial implementation
fn blend_half_indexes<const N: usize>(
  dozero: u8, src1: Option<usize>, src2: Option<usize>, ind: &[usize; N],
) -> ArrayVec<i32, N> {
  // a is a reference to a constexpr array of permutation indexes
  let mut list = ArrayVec::new();
  // value to use for unused entries
  let u = if dozero > 0 { -1 } else { V_DC };

  for j in 0..N {
    let idx = ind[j];
    let src = idx / N;
    list.push(if src1 == Some(src) {
      (idx & (N - 1)) as i32
    } else if src2 == Some(src) {
      ((idx & (N - 1)) + N) as i32
    } else {
      u
    });
  }

  list
}

fn select_blend8(action: Option<usize>, a: f64x8, b: f64x8) -> f64x4 {
  match action {
    Some(0) => a[0],
    Some(1) => a[1],
    Some(2) => b[0],
    _ => b[1],
  }
}

#[inline(always)]
fn blend4_f64(indexes: &[i32; 4], a: f64x4, b: f64x4) -> f64x4 {
  let flags = blend_flags::<4, { size_of::<f64x4>() }>(indexes);
  assert!(
    flags & BLEND_OUTOFRANGE == 0,
    "Index out of range in blend function"
  );

  if flags & BLEND_ALLZERO != 0 {
    return f64x4::ZERO;
  }

  if flags & BLEND_B == 0 {
    return permute4(indexes, a);
  }
  if flags & BLEND_A == 0 {
    return permute4(
      &[
        if i[0] < 0 { i[0] } else { i[0] & 3 },
        if i[1] < 0 { i[1] } else { i[1] & 3 },
        if i[2] < 0 { i[2] } else { i[2] & 3 },
        if i[3] < 0 { i[3] } else { i[3] & 3 },
      ],
      b,
    );
  }

  let mut y = a;
  if flags & (BLEND_PERMA | BLEND_PERMB) == 0 {
    // no permutation, only blending
    let mb = make_bit_mask::<4, 0x302>(indexes);
    y = f64x4::new([
      if mb & (1 << 0) > 0 { -1f64 } else { 0f64 },
      if mb & (1 << 1) > 0 { -1f64 } else { 0f64 },
      if mb & (1 << 2) > 0 { -1f64 } else { 0f64 },
      if mb & (1 << 3) > 0 { -1f64 } else { 0f64 },
    ])
    .blend(b, a);
  } else if flags & BLEND_LARGEBLOCK != 0 {
    // blend and permute 128-bit blocks
    let l = largeblock_perm::<4, 2>(indexes);
    let pp = (l[0] & 0xF) as u8 | ((l[1] & 0xF) as u8) << 4;
    //   y = _mm256_permute2f128_pd(a, b, pp);
    todo!();
  } else if flags & BLEND_PUNPCKLAB != 0 {
    y = a.interleave_low(b);
  } else if flags & BLEND_PUNPCKLBA != 0 {
    y = b.interleave_low(a);
  } else if flags & BLEND_PUNPCKHAB != 0 {
    y = a.interleave_high(b);
  } else if flags & BLEND_PUNPCKHBA != 0 {
    y = b.interleave_high(a);
  } else if flags & BLEND_SHUFAB != 0 {
    y = a.shuffle_nonconst(b, ((flags >> BLEND_SHUFPATTERN) as u8) & 0xF);
  } else if flags & BLEND_SHUFBA != 0 {
    y = b.shuffle_nonconst(a, ((flags >> BLEND_SHUFPATTERN) as u8) & 0xF);
  } else {
    // No special cases
    let l = blend_perm_indexes::<4, 8, 0>(indexes);
    let ya = permute4(cast(&l[..4]), a);
    let yb = permute4(cast(&l[4..]), b);
    let mb = make_bit_mask::<4, 0x302>(indexes);
    y = f64x4::new([
      if mb & (1 << 0) > 0 { -1f64 } else { 0f64 },
      if mb & (1 << 1) > 0 { -1f64 } else { 0f64 },
      if mb & (1 << 2) > 0 { -1f64 } else { 0f64 },
      if mb & (1 << 3) > 0 { -1f64 } else { 0f64 },
    ])
    .blend(yb, ya);
  }

  if flags & BLEND_ZEROING != 0 {
    // additional zeroing needed
    let bm = i64x4::new(zero_mask_broad_4x64(indexes));
    // SAFETY: Types have the same size
    let bm1: f64x4 = unsafe { transmute(bm) };
    y = bm1 & y;
  }

  y
}

// blend_flags: returns information about how a blend function can be implemented
// The return value is composed of these flag bits:

// needs zeroing
const BLEND_ZEROING: u64 = 1;
// all is zero or don't care
const BLEND_ALLZERO: u64 = 2;
// fits blend with a larger block size (e.g permute Vec2q instead of Vec4i)
const BLEND_LARGEBLOCK: u64 = 4;
// additional zeroing needed after blend with larger block size or shift
const BLEND_ADDZ: u64 = 8;
// has data from a
const BLEND_A: u64 = 0x10;
// has data from b
const BLEND_B: u64 = 0x20;
// permutation of a needed
const BLEND_PERMA: u64 = 0x40;
// permutation of b needed
const BLEND_PERMB: u64 = 0x80;
// permutation crossing 128-bit lanes
const BLEND_CROSS_LANE: u64 = 0x100;
// same permute/blend pattern in all 128-bit lanes
const BLEND_SAME_PATTERN: u64 = 0x200;
// pattern fits punpckh(a,b)
const BLEND_PUNPCKHAB: u64 = 0x1000;
// pattern fits punpckh(b,a)
const BLEND_PUNPCKHBA: u64 = 0x2000;
// pattern fits punpckl(a,b)
const BLEND_PUNPCKLAB: u64 = 0x4000;
// pattern fits punpckl(b,a)
const BLEND_PUNPCKLBA: u64 = 0x8000;
// pattern fits palignr(a,b)
const BLEND_ROTATEAB: u64 = 0x10000;
// pattern fits palignr(b,a)
const BLEND_ROTATEBA: u64 = 0x20000;
// pattern fits shufps/shufpd(a,b)
const BLEND_SHUFAB: u64 = 0x40000;
// pattern fits shufps/shufpd(b,a)
const BLEND_SHUFBA: u64 = 0x80000;
// pattern fits rotation across lanes. count returned in bits blend_rotpattern
const BLEND_ROTATE_BIG: u64 = 0x100000;
// index out of range
const BLEND_OUTOFRANGE: u64 = 0x10000000;
// pattern for shufps/shufpd is in bit blend_shufpattern to blend_shufpattern + 7
const BLEND_SHUFPATTERN: u64 = 32;
// pattern for palignr is in bit blend_rotpattern to blend_rotpattern + 7
const BLEND_ROTPATTERN: u64 = 40;

// INSTRSET = 8
fn blend_flags<const N: usize, const V: usize>(a: &[i32; N]) -> u64 {
  let mut ret = BLEND_LARGEBLOCK | BLEND_SAME_PATTERN | BLEND_ALLZERO;
  // number of 128-bit lanes
  let n_lanes = V / 16;
  // elements per lane
  let lane_size = N / n_lanes;
  // current lane
  let mut lane = 0;
  // rotate left count
  let mut rot: u32 = 999;
  // GeNERIc pARAMeTerS canNoT BE useD In consT ConTEXTs
  let mut lane_pattern = vec![0; lane_size].into_boxed_slice();
  if lane_size == 2 && N <= 8 {
    ret |= BLEND_SHUFAB | BLEND_SHUFBA;
  }

  for ii in 0..N {
    let ix = a[ii];
    if ix < 0 {
      if ix == -1 {
        ret |= BLEND_ZEROING;
      } else if ix != V_DC {
        ret = BLEND_OUTOFRANGE;
        break;
      }
    } else {
      ret &= !BLEND_ALLZERO;
      if ix < N as i32 {
        ret |= BLEND_A;
        if ix != ii as i32 {
          ret |= BLEND_PERMA;
        }
      } else if ix < 2 * N as i32 {
        ret |= BLEND_B;
        if ix != (ii + N) as i32 {
          ret |= BLEND_PERMB;
        }
      } else {
        ret = BLEND_OUTOFRANGE;
        break;
      }
    }

    // check if pattern fits a larger block size:
    // even indexes must be even, odd indexes must fit the preceding even index + 1
    if (ii & 1) == 0 {
      if ix >= 0 && (ix & 1) > 0 {
        ret &= !BLEND_LARGEBLOCK;
      }
      let iy = a[ii + 1];
      if iy >= 0 && (iy & 1) == 0 {
        ret &= !BLEND_LARGEBLOCK;
      }
      if ix >= 0 && iy >= 0 && iy != ix + 1 {
        ret &= !BLEND_LARGEBLOCK;
      }
      if ix == -1 && iy >= 0 {
        ret |= BLEND_ADDZ;
      }
      if iy == -1 && ix >= 0 {
        ret |= BLEND_ADDZ;
      }
    }

    lane = ii / lane_size;
    if lane == 0 {
      lane_pattern[ii] = ix;
    }

    // check if crossing lanes
    if ix >= 0 {
      let lane_i = (ix & !(N as i32)) as usize / lane_size;
      if lane_i != lane {
        ret |= BLEND_CROSS_LANE;
      }
      if lane_size == 2 {
        // check if it fits pshufd
        if lane_i != lane {
          ret &= !(BLEND_SHUFAB | BLEND_SHUFBA);
        }
        if (((ix & (N as i32)) != 0) as usize ^ ii) & 1 > 0 {
          ret &= !BLEND_SHUFAB;
        } else {
          ret &= !BLEND_SHUFBA;
        }
      }
    }

    // check if same pattern in all lanes
    if lane != 0 && ix >= 0 {
      let j = ii - (lane * lane_size);
      let jx = ix - (lane * lane_size) as i32;
      if jx < 0 || (jx & !(N as i32)) >= lane_size as i32 {
        ret &= !BLEND_SAME_PATTERN;
      }
      if lane_pattern[j] < 0 {
        lane_pattern[j] = jx;
      } else if lane_pattern[j] != jx {
        ret &= !BLEND_SAME_PATTERN;
      }
    }
  }

  if ret & BLEND_LARGEBLOCK == 0 {
    ret &= !BLEND_ADDZ;
  }
  if ret & BLEND_CROSS_LANE > 0 {
    ret &= !BLEND_SAME_PATTERN;
  }
  if ret & (BLEND_PERMA | BLEND_PERMB) == 0 {
    return ret;
  }

  if ret & BLEND_SAME_PATTERN > 0 {
    // same pattern in all lanes. check if it fits unpack patterns
    ret |=
      BLEND_PUNPCKHAB | BLEND_PUNPCKHBA | BLEND_PUNPCKLAB | BLEND_PUNPCKLBA;
    for iu in 0..(lane_size as u32) {
      let ix = lane_pattern[iu as usize];
      if ix >= 0 {
        let ix = ix as u32;
        if ix != iu / 2 + (iu & 1) * N as u32 {
          ret &= !BLEND_PUNPCKLAB;
        }
        if ix != iu / 2 + ((iu & 1) ^ 1) * N as u32 {
          ret &= !BLEND_PUNPCKLBA;
        }
        if ix != (iu + lane_size as u32) / 2 + (iu & 1) * N as u32 {
          ret &= !BLEND_PUNPCKHAB;
        }
        if ix != (iu + lane_size as u32) / 2 + ((iu & 1) ^ 1) * N as u32 {
          ret &= !BLEND_PUNPCKHBA;
        }
      }
    }

    for iu in 0..(lane_size as u32) {
      // check if it fits palignr
      let ix = lane_pattern[iu as usize];
      if ix >= 0 {
        let ix = ix as u32;
        let mut t = ix & !(N as u32);
        if (ix & N as u32) > 0 {
          t += lane_size as u32;
        }
        let tb = (t + 2 * lane_size as u32 - iu) % (lane_size as u32 * 2);
        if rot == 999 {
          rot = tb;
        } else if rot != tb {
          rot = 1000;
        }
      }
    }
    if rot < 999 {
      // fits palignr
      if rot < lane_size as u32 {
        ret |= BLEND_ROTATEBA;
      } else {
        ret |= BLEND_ROTATEAB;
      }
      let elem_size = (V / N) as u32;
      ret |= (((rot & (lane_size as u32 - 1)) * elem_size) as u64)
        << BLEND_ROTPATTERN;
    }

    if lane_size == 4 {
      // check if it fits shufps
      ret |= BLEND_SHUFAB | BLEND_SHUFBA;
      for ii in 0..2 {
        let ix = lane_pattern[ii];
        if ix >= 0 {
          if ix & N as i32 > 0 {
            ret &= !BLEND_SHUFAB;
          } else {
            ret &= !BLEND_SHUFBA;
          }
        }
      }
      for ii in 2..4 {
        let ix = lane_pattern[ii];
        if ix >= 0 {
          if ix & N as i32 > 0 {
            ret &= !BLEND_SHUFBA;
          } else {
            ret &= !BLEND_SHUFAB;
          }
        }
      }
      if ret & (BLEND_SHUFAB | BLEND_SHUFBA) > 0 {
        // fits shufps/shufpd
        let mut shuf_pattern = 0u8;
        for iu in 0..lane_size {
          shuf_pattern |= ((lane_pattern[iu] & 3) as u8) << (iu * 2);
        }
        ret |= (shuf_pattern as u64) << BLEND_SHUFPATTERN;
      }
    }
  } else if n_lanes > 1 {
    // not same pattern in all lanes
    let mut rot = 999;
    for ii in 0..N {
      let ix = a[ii];
      if ix >= 0 {
        let rot2: u32 =
          (ix + 2 * N as i32 - ii as i32) as u32 % (2 * N) as u32;
        if rot == 999 {
          rot = rot2;
        } else if rot != rot2 {
          rot = 1000;
          break;
        }
      }
    }
    if rot < 2 * N as u32 {
      // fits big rotate
      ret |= BLEND_ROTATE_BIG | (rot as u64) << BLEND_ROTPATTERN;
    }
  }
  if lane_size == 2 && (ret & (BLEND_SHUFAB | BLEND_SHUFBA)) > 0 {
    for ii in 0..N {
      ret |= ((a[ii] & 1) as u64) << (BLEND_SHUFPATTERN + ii as u64);
    }
  }

  ret
}

// largeblock_perm: return indexes for replacing a permute or blend with
// a certain block size by a permute or blend with the double block size.
// Note: it is presupposed that perm_flags() indicates perm_largeblock
// It is required that additional zeroing is added if perm_flags() indicates perm_addz
fn largeblock_perm<const N: usize, const N2: usize>(
  a: &[i32; N],
) -> [i32; N2] {
  // GeNERIc pARAMeTerS canNoT BE useD In consT ConTEXTs
  assert!(N2 == N / 2);

  // Parameter a is a reference to a constexpr array of permutation indexes
  let mut list = [0; N2];
  let mut fit_addz = false;

  // check if additional zeroing is needed at current block size
  for i in (0..N).step_by(2) {
    let ix = a[i];
    let iy = a[i + 1];
    if (ix == -1 && iy >= 0) || (iy == -1 && ix >= 0) {
      fit_addz = true;
      break;
    }
  }

  // loop through indexes
  for i in (0..N).step_by(2) {
    let ix = a[i];
    let iy = a[i + 1];
    let iz = if ix >= 0 {
      ix / 2
    } else if iy >= 0 {
      iy / 2
    } else if fit_addz {
      V_DC
    } else {
      ix | iy
    };
    list[i / 2] = iz;
  }

  list
}

// perm_flags: returns information about how a permute can be implemented.
// The return value is composed of these flag bits:

// needs zeroing
const PERM_ZEROING: u64 = 1;
// permutation needed
const PERM_PERM: u64 = 2;
// all is zero or don't care
const PERM_ALLZERO: u64 = 4;
// fits permute with a larger block size (e.g permute Vec2q instead of Vec4i)
const PERM_LARGEBLOCK: u64 = 8;
// additional zeroing needed after permute with larger block size or shift
const PERM_ADDZ: u64 = 0x10;
// additional zeroing needed after perm_zext, perm_compress, or perm_expand
const PERM_ADDZ2: u64 = 0x20;
// permutation crossing 128-bit lanes
const PERM_CROSS_LANE: u64 = 0x40;
// same permute pattern in all 128-bit lanes
const PERM_SAME_PATTERN: u64 = 0x80;
// permutation pattern fits punpckh instruction
const PERM_PUNPCKH: u64 = 0x100;
// permutation pattern fits punpckl instruction
const PERM_PUNPCKL: u64 = 0x200;
// permutation pattern fits rotation within lanes. 4 bit count returned in bit perm_rot_count
const PERM_ROTATE: u64 = 0x400;
// permutation pattern fits shift right within lanes. 4 bit count returned in bit perm_rot_count
const PERM_SHRIGHT: u64 = 0x1000;
// permutation pattern fits shift left within lanes. negative count returned in bit perm_rot_count
const PERM_SHLEFT: u64 = 0x2000;
// permutation pattern fits rotation across lanes. 6 bit count returned in bit perm_rot_count
const PERM_ROTATE_BIG: u64 = 0x4000;
// permutation pattern fits broadcast of a single element.
const PERM_BROADCAST: u64 = 0x8000;
// permutation pattern fits zero extension
const PERM_ZEXT: u64 = 0x10000;
// permutation pattern fits vpcompress instruction
const PERM_COMPRESS: u64 = 0x20000;
// permutation pattern fits vpexpand instruction
const PERM_EXPAND: u64 = 0x40000;
// index out of range
const PERM_OUTOFRANGE: u64 = 0x10000000;
// rotate or shift count is in bits perm_rot_count to perm_rot_count+3
const PERM_ROT_COUNT: u64 = 32;
// pattern for pshufd is in bit perm_ipattern to perm_ipattern + 7 if perm_same_pattern and elementsize >= 4
const PERM_IPATTERN: u64 = 40;

fn permute8(indexes: &[i32; 8], a: f32x8) -> f32x8 {
  let mut y = a;
  let flags = perm_flags::<32, 8>(indexes);
  assert!(
    flags & PERM_OUTOFRANGE == 0,
    "Index out of range in permute function"
  );

  if flags & PERM_ALLZERO > 0 {
    return f32x8::ZERO;
  }

  if flags & PERM_PERM > 0 {
    if flags & PERM_LARGEBLOCK > 0 {
      let l = largeblock_perm::<8, 4>(indexes);
      // SAFETY: Types are of same size
      y = unsafe { transmute(permute4(&l, transmute(a))) };
      if flags & PERM_ADDZ == 0 {
        // no remaining zeroing
        return y;
      }
    } else if flags & PERM_SAME_PATTERN > 0 {
      if flags & PERM_PUNPCKH != 0 {
        // fits punpckhi
        y = y.interleave_high(y);
      } else if flags & PERM_PUNPCKL != 0 {
        // fits punpcklo
        y = y.interleave_low(y);
      } else {
        // general permute, same pattern in both lanes
        y = a.shuffle_nonconst(a, (flags >> PERM_IPATTERN) as u8);
      }
    } else if flags & PERM_BROADCAST > 0 && flags >> PERM_ROT_COUNT == 0 {
      // broadcast first element
      y = f32x8::from(y.to_array()[0]);
    } else if flags & PERM_ZEXT > 0 {
      // zero extension
      //       y = _mm256_castsi256_ps(_mm256_cvtepu32_epi64(
      // _mm256_castsi256_si128(_mm256_castps_si256(y))));
      todo!();
      if flags & PERM_ADDZ2 == 0 {
        return y;
      }
    } else if flags & PERM_CROSS_LANE == 0 {
      // __m256 m = constant8f<i0 & 3, i1 & 3, i2 & 3, i3 & 3, i4 & 3, i5 & 3,
      // i6 & 3, i7 & 3>();
      // y = _mm256_permutevar_ps(a, _mm256_castps_si256(m));
      todo!();
    } else {
      // full permute needed
      // __m256i permmask =
      // _mm256_castps_si256(constant8f<i0 & 7, i1 & 7, i2 & 7, i3 & 7,
      //                        i4 & 7, i5 & 7, i6 & 7, i7 & 7>());
      // y = _mm256_permutevar8x32_ps(a, permmask);
      todo!();
    }
  }

  if flags & PERM_ZEROING > 0 {
    let bm = i32x8::new(zero_mask_broad_8x32(indexes));
    // SAFETY: Types have the same size
    let bm1: f32x8 = unsafe { transmute(bm) };
    y = bm1 & y;
  }

  y
}

fn perm_flags<const ELEM_SIZE: usize, const ELEMS: usize>(
  a: &[i32; ELEMS],
) -> u64 {
  // number of 128-bit lanes
  let num_lanes = ELEM_SIZE * ELEMS / 16;
  let lane_size = ELEMS / num_lanes;
  // current lane
  let mut lane = 0usize;
  // rotate left count
  let mut rot = 999u32;
  // index to broadcasted element
  let mut broadc = 999i32;
  // remember certain patterns that do not fit
  let mut patfail = 0u32;
  // remember certain patterns need extra zeroing
  let mut addz2 = 0u32;
  // last index in perm_compress fit
  let mut compresslasti = -1i32;
  // last position in perm_compress fit
  let mut compresslastp = -1i32;
  // last index in perm_expand fit
  let mut expandlasti = -1i32;
  // last position in perm_expand fit
  let mut expandlastp = -1i32;

  let mut ret = PERM_LARGEBLOCK | PERM_SAME_PATTERN | PERM_ALLZERO;
  let mut lane_pattern = vec![0i32; lane_size].into_boxed_slice();
  for i in 0..ELEMS {
    let ix = a[i];
    // meaning of ix: -1 = set to zero, V_DC = don't care, non-negative value = permute.
    if ix == -1 {
      ret |= PERM_ZEROING;
    } else if ix != V_DC && ix as usize >= ELEMS {
      ret |= PERM_OUTOFRANGE;
    }

    if ix >= 0 {
      ret &= !PERM_ALLZERO;
      if ix != i as i32 {
        ret |= PERM_PERM;
      }
      if broadc == 999 {
        // remember broadcast index
        broadc = ix;
      } else if broadc != ix {
        // does not fit broadcast
        broadc = 1000;
      }
    }

    // check if pattern fits a larger block size:
    // even indexes must be even, odd indexes must fit the preceding even index + 1
    if i & 1 == 0 {
      if ix > 0 && ix & 1 > 0 {
        ret &= !PERM_LARGEBLOCK;
      }
      let iy = a[i + 1];
      if iy >= 0 && iy & 1 == 0 {
        ret &= !PERM_LARGEBLOCK;
      }
      if ix >= 0 && iy >= 0 && iy != ix + 1 {
        ret &= !PERM_LARGEBLOCK;
      }
      if ix == -1 && iy >= 0 {
        ret |= PERM_ADDZ;
      }
      if iy == -1 && ix >= 0 {
        ret |= PERM_ADDZ;
      }
    }

    lane = i / lane_size;
    if lane == 0 {
      // first lane, or no pattern yet
      lane_pattern[i] = ix;
    }

    // check if crossing lanes
    if ix >= 0 {
      let lanei = ix as usize / lane_size;
      if lanei != lane {
        ret |= PERM_CROSS_LANE;
      }
    }

    // check if same pattern in all lanes
    if lane != 0 && ix >= 0 {
      // not first lane
      let j1 = i - lane * lane_size;
      let jx = ix - (lane * lane_size) as i32;
      if jx < 0 || jx >= lane_size as i32 {
        // source is in another lane
        ret &= !PERM_SAME_PATTERN;
      }
      if lane_pattern[j1] < 0 {
        // pattern not known from previous lane
        lane_pattern[j1] = jx;
      } else if lane_pattern[j1] != jx {
        // not same pattern
        ret &= !PERM_SAME_PATTERN;
      }
    }

    if ix >= 0 {
      // check if pattern fits zero extension (perm_zext)
      if (ix * 2) as usize != i {
        // does not fit zero extension
        patfail |= 1;
      }
      // check if pattern fits compress (perm_compress)
      if ix > compresslasti && ix - compresslasti >= i as i32 - compresslastp {
        if i as i32 - compresslastp > 1 {
          // perm_compress may need additional zeroing
          addz2 |= 2;
        }
        compresslasti = ix;
        compresslastp = i as i32;
      } else {
        // does not fit perm_compress
        patfail |= 2;
      }
      // check if pattern fits expand (perm_expand)
      if ix > expandlasti && ix - expandlasti <= i as i32 - expandlastp {
        if ix - expandlasti > 1 {
          // perm_expand may need additional zeroing
          addz2 |= 4;
        }
        expandlasti = ix;
        expandlastp = i as i32;
      } else {
        // does not fit perm_compress
        patfail |= 4;
      }
    } else if ix == -1 && i & 1 == 0 {
      // zero extension needs additional zeroing
      addz2 |= 1;
    }
  }

  if ret & PERM_PERM == 0 {
    return ret;
  }

  if ret & PERM_LARGEBLOCK == 0 {
    ret &= !PERM_ADDZ;
  }
  if ret & PERM_CROSS_LANE > 0 {
    ret &= !PERM_SAME_PATTERN;
  }
  if patfail & 1 == 0 {
    ret |= PERM_ZEXT;
    if addz2 & 1 != 0 {
      ret |= PERM_ADDZ2;
    }
  } else if patfail & 2 == 0 {
    ret |= PERM_COMPRESS;
    if addz2 & 2 != 0 && compresslastp > 0 {
      for j in 0..(compresslastp as usize) {
        if a[j] == -1 {
          ret |= PERM_ADDZ2;
        }
      }
    }
  } else if patfail & 4 == 0 {
    ret |= PERM_EXPAND;
    if addz2 & 4 != 0 && expandlastp > 0 {
      for j in 0..(expandlastp as usize) {
        if a[j] == -1 {
          ret |= PERM_ADDZ2;
        }
      }
    }
  }

  if ret & PERM_SAME_PATTERN > 0 {
    // same pattern in all lanes. check if it fits specific patterns
    let mut fit = true;
    // fit shift or rotate
    for i in 0..lane_size {
      if lane_pattern[i] >= 0 {
        let rot1 = lane_pattern[i] as u32 + lane_size as u32
          - i as u32 % lane_size as u32;
        if rot == 999 {
          rot = rot1;
        } else if rot != rot1 {
          fit = false;
        }
      }
    }
    rot &= lane_size as u32 - 1;
    if fit {
      // fits rotate, and possible shift
      let rot2 = ((rot & ELEM_SIZE as u32) & 0xF) as u64;
      ret |= rot2 << PERM_ROT_COUNT;
      ret |= PERM_ROTATE;
      // fit shift left
      fit = true;
      let mut i = 0;
      while (i + rot as usize) < lane_size {
        // check if first rot elements are zero or don't care
        if lane_pattern[i] >= 0 {
          fit = false;
        }
        i += 1;
      }
      if fit {
        ret |= PERM_SHLEFT;
        while i < lane_size {
          if lane_pattern[i] == -1 {
            ret |= PERM_ADDZ;
          }
          i += 1;
        }
      }
      // fit shift right
      fit = true;
      i = lane_size - rot as usize;
      while i < lane_size {
        // check if last (lanesize-rot) elements are zero or don't care
        if lane_pattern[i] >= 0 {
          fit = false;
        }
        i += 1;
      }
      if fit {
        ret |= PERM_SHRIGHT;
        while i < lane_size - rot as usize {
          if lane_pattern[i] == -1 {
            ret |= PERM_ADDZ;
          }
          i += 1;
        }
      }
    }

    // fit punpckhi
    fit = true;
    let mut j2 = lane_size / 2;
    for i in 0..lane_size {
      if lane_pattern[i] >= 0 && lane_pattern[i] != j2 as i32 {
        fit = false;
      }
      if (i & 1) != 0 {
        j2 += 1;
      }
    }
    if fit {
      ret |= PERM_PUNPCKH;
    }
    // fit punpcklo
    fit = true;
    j2 = 0;
    for i in 0..lane_size {
      if lane_pattern[i] >= 0 && lane_pattern[i] != j2 as i32 {
        fit = false;
      }
      if (i & 1) != 0 {
        j2 += 1;
      }
    }
    if fit {
      ret |= PERM_PUNPCKL;
    }
    // fit pshufd
    if ELEM_SIZE >= 4 {
      let mut p = 0u64;
      for i in 0..lane_size {
        if lane_size == 4 {
          p |= ((lane_pattern[i] & 3) as u64) << (2 * i as u64);
        } else {
          // lanesize = 2
          p |= ((lane_pattern[i] & 1) as u64 * 10 + 4) << (4 * i as u64);
        }
      }
      ret |= p << PERM_IPATTERN;
    }
  } else {
    // not same pattern in all lanes
    if num_lanes > 1 {
      // Try if it fits big rotate
      for i in 0..ELEMS {
        let ix = a[i];
        if ix >= 0 {
          // rotate count
          let mut rot2: u32 =
            (ix as u32 + ELEMS as u32 - i as u32) % ELEMS as u32;
          if rot == 999 {
            // save rotate count
            rot = rot2;
          } else if rot != rot2 {
            // does not fit big rotate
            rot = 1000;
            break;
          }
        }
      }
      if rot < ELEMS as u32 {
        // fits big rotate
        ret |= PERM_ROTATE_BIG | (rot as u64) << PERM_ROT_COUNT;
      }
    }
  }

  if broadc < 999
    && ret & (PERM_ROTATE | PERM_SHRIGHT | PERM_SHLEFT | PERM_ROTATE_BIG) == 0
  {
    // fits broadcast
    ret |= PERM_BROADCAST | (broadc as u64) << PERM_ROT_COUNT;
  }

  ret
}

fn make_bit_mask<const LEN: usize, const BITS: i32>(a: &[i32; LEN]) -> u64 {
  let mut ret = 0u64;
  let sel_bit_idx = (BITS & 0xFF) as u8;
  let mut ret_bit_idx = 0u64;
  let mut flip = 0u64;
  for i in 0..LEN {
    let ix = a[i];
    if ix < 0 {
      ret_bit_idx = ((BITS >> 10) & 1) as u64;
    } else {
      ret_bit_idx = (((ix as u32) >> sel_bit_idx) & 1) as u64;
      if i < LEN / 2 {
        flip = ((BITS >> 8) & 1) as u64;
      } else {
        flip = ((BITS >> 9) & 1) as u64;
      }
      ret_bit_idx ^= flip ^ 1;
    }
    ret |= ret_bit_idx << i;
  }
  ret
}

// blend_perm_indexes: return an Indexlist for implementing a blend function as
// two permutations. N = vector size.
// dozero = 0: let unused elements be don't care. The two permutation results must be blended
// dozero = 1: zero unused elements in each permuation. The two permutation results can be OR'ed
// dozero = 2: indexes that are -1 or V_DC are preserved
fn blend_perm_indexes<const N: usize, const N2: usize, const DO_ZERO: u8>(
  a: &[i32; N],
) -> [i32; N2] {
  assert!(N * 2 == N2);
  assert!(DO_ZERO <= 2);
  let mut list = [0i32; N2];
  let u = if DO_ZERO > 0 { -1 } else { V_DC };
  for j in 0..N {
    let ix = a[j];
    if ix < 0 {
      if DO_ZERO == 2 {
        list[j] = ix;
        list[j + N] = ix;
      } else {
        list[j] = u;
        list[j + N] = u;
      }
    } else if ix < N as i32 {
      list[j] = ix;
      list[j + N] = u;
    } else {
      list[j] = u;
      list[j + N] = ix - N as i32;
    }
  }
  list
}

#[inline(always)]
fn zero_mask_broad_8x32(a: &[i32; 8]) -> [i32; 8] {
  let mut u = [0i32; 8];
  for i in 0..8 {
    u[i] = if a[i] >= 0 { -1 } else { 0 };
  }
  u
}

#[inline(always)]
fn zero_mask_broad_4x64(a: &[i32; 4]) -> [i64; 4] {
  let mut u = [0i64; 4];
  for i in 0..4 {
    u[i] = if a[i] >= 0 { -1 } else { 0 };
  }
  u
}

fn permute4(indexes: &[i32; 4], a: f64x4) -> f64x4 {
  let mut y = a;
  let flags = perm_flags::<64, 4>(indexes);

  assert!(
    flags & PERM_OUTOFRANGE == 0,
    "Index out of range in permute function"
  );

  if flags & PERM_ALLZERO != 0 {
    return f64x4::ZERO;
  }

  if flags & PERM_LARGEBLOCK != 0 {
    // permute 128-bit blocks
    let l = largeblock_perm::<4, 2>(indexes);
    let j0 = l[0];
    let j1 = l[1];

    if j0 == 0 && j1 == -1 && flags & PERM_ADDZ == 0 {
      // zero extend
      //       return _mm256_zextpd128_pd256(_mm256_castpd256_pd128(y));
      todo!();
    }
    if j0 == 1 && j1 < 0 && flags & PERM_ADDZ == 0 {
      // extract upper part, zero extend
      //       return _mm256_zextpd128_pd256(_mm256_extractf128_pd(y, 1));
      todo!();
    }
    if flags & PERM_PERM != 0 && flags & PERM_ZEROING == 0 {
      //       return _mm256_permute2f128_pd(y, y, (j0 & 1) | (j1 & 1) << 4);
      todo!();
    }
  }

  if flags & PERM_PERM != 0 {
    // permutation needed
    if flags & PERM_SAME_PATTERN != 0 {
      // same pattern in both lanes
      if flags & PERM_PUNPCKH != 0 {
        y = y.interleave_high(y);
      } else if flags & PERM_PUNPCKL != 0 {
        y = y.interleave_low(y);
      } else {
        // general permute
        let mm0 = (indexes[0] & 1)
          | (indexes[1] & 1) << 1
          | (indexes[2] & 1) << 2
          | (indexes[3] & 1) << 3;
        // select within same lane
        //         y = _mm256_permute_pd(a, mm0);
        todo!();
      }
    } else if flags & PERM_BROADCAST != 0 && (flags >> PERM_ROT_COUNT) == 0 {
      // broadcast first element
      //       y = _mm256_broadcastsd_pd(
      //           _mm256_castpd256_pd128(y));
      todo!();
    } else {
      // different patterns in two lanes
      if flags & PERM_CROSS_LANE == 0 {
        // no lane crossing
        //         constexpr uint8_t mm0 =
        //             (i0 & 1) | (i1 & 1) << 1 | (i2 & 1) << 2 | (i3 & 1) << 3;
        //         y = _mm256_permute_pd(a, mm0); // select within same lane
        todo!();
      } else {
        //         // full permute
        //         constexpr uint8_t mms =
        //             (i0 & 3) | (i1 & 3) << 2 | (i2 & 3) << 4 | (i3 & 3) << 6;
        //         y = _mm256_permute4x64_pd(a, mms);
        todo!();
      }
    }
  }

  if flags & PERM_ZEROING != 0 {
    // additional zeroing needed
    // use broad mask
    //     constexpr EList<int64_t, 4> bm = zero_mask_broad<Vec4q>(indexs);
    //     // y = _mm256_and_pd(_mm256_castsi256_pd( Vec4q().load(bm.a) ), y);  // does
    //     // not work with INSTRSET = 7
    //     __m256i bm1 = _mm256_loadu_si256((const __m256i *)(bm.a));
    //     y = _mm256_and_pd(_mm256_castsi256_pd(bm1), y);
    todo!();
  }
  y
}
