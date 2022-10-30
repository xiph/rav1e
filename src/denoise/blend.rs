//! A pox on the house of whoever decided this was a good way to code anything

use std::mem::size_of;

use arrayvec::ArrayVec;
use wide::{f32x8, f64x4};

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
  let mut x0 =
    blend8_f32(l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], src0, src1);

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
    let x1 =
      blend8_f32(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], src2, src3);

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
fn blend8_f32(
  i0: i32, i1: i32, i2: i32, i3: i32, i4: i32, i5: i32, i6: i32, i7: i32,
  a: f32x8, b: f32x8,
) -> f32x8 {
  let indexes = [i0, i1, i2, i3, i4, i5, i6, i7];
  let y = a;
  let flags = blend_flags::<8, { size_of::<f32x8>() }>(&indexes);
  todo!()
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
  let mut x0 = blend4_f64(l[0], l[1], l[2], l[3], src0, src1);

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
    let x1 = blend4_f64(m[0], m[1], m[2], m[3], src2, src3);

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
fn blend4_f64(
  i0: i32, i1: i32, i2: i32, i3: i32, a: f64x4, b: f64x4,
) -> f64x4 {
  todo!()
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
  let mut r = BLEND_LARGEBLOCK | BLEND_SAME_PATTERN | BLEND_ALLZERO;
  // number of 128-bit lanes
  let n_lanes = V / 16;
  // elements per lane
  let lane_size = N / n_lanes;
  // current lane
  let mut lane = 0;
  // rotate left count
  let mut rot: u32 = 999;
  // GeNERIc pARAMeTerS canNoT BE useD In consT ConTEXTs
  let mut lane_pattern = vec![0; lane_size];
  if lane_size == 2 && N <= 8 {
    r |= BLEND_SHUFAB | BLEND_SHUFBA;
  }

  for ii in 0..N {
    let ix = a[ii];
    if ix < 0 {
      if ix == -1 {
        r |= BLEND_ZEROING;
      } else if ix != V_DC {
        r = BLEND_OUTOFRANGE;
        break;
      }
    } else {
      r &= !BLEND_ALLZERO;
      if ix < N as i32 {
        r |= BLEND_A;
        if ix != ii as i32 {
          r |= BLEND_PERMA;
        }
      } else if ix < 2 * N as i32 {
        r |= BLEND_B;
        if ix != (ii + N) as i32 {
          r |= BLEND_PERMB;
        }
      } else {
        r = BLEND_OUTOFRANGE;
        break;
      }
    }

    // check if pattern fits a larger block size:
    // even indexes must be even, odd indexes must fit the preceding even index + 1
    if (ii & 1) == 0 {
      if ix >= 0 && (ix & 1) > 0 {
        r &= !BLEND_LARGEBLOCK;
      }
      let iy = a[ii + 1];
      if iy >= 0 && (iy & 1) == 0 {
        r &= !BLEND_LARGEBLOCK;
      }
      if ix >= 0 && iy >= 0 && iy != ix + 1 {
        r &= !BLEND_LARGEBLOCK;
      }
      if ix == -1 && iy >= 0 {
        r |= BLEND_ADDZ;
      }
      if iy == -1 && ix >= 0 {
        r |= BLEND_ADDZ;
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
        r |= BLEND_CROSS_LANE;
      }
      if lane_size == 2 {
        // check if it fits pshufd
        if lane_i != lane {
          r &= !(BLEND_SHUFAB | BLEND_SHUFBA);
        }
        if (((ix & (N as i32)) != 0) as usize ^ ii) & 1 > 0 {
          r &= !BLEND_SHUFAB;
        } else {
          r &= !BLEND_SHUFBA;
        }
      }
    }

    // check if same pattern in all lanes
    if lane != 0 && ix >= 0 {
      let j = ii - (lane * lane_size);
      let jx = ix - (lane * lane_size) as i32;
      if jx < 0 || (jx & !(N as i32)) >= lane_size as i32 {
        r &= !BLEND_SAME_PATTERN;
      }
      if lane_pattern[j] < 0 {
        lane_pattern[j] = jx;
      } else if lane_pattern[j] != jx {
        r &= !BLEND_SAME_PATTERN;
      }
    }
  }

  if r & BLEND_LARGEBLOCK == 0 {
    r &= !BLEND_ADDZ;
  }
  if r & BLEND_CROSS_LANE > 0 {
    r &= !BLEND_SAME_PATTERN;
  }
  if r & (BLEND_PERMA | BLEND_PERMB) == 0 {
    return r;
  }

  if r & BLEND_SAME_PATTERN > 0 {
    // same pattern in all lanes. check if it fits unpack patterns
    r |= BLEND_PUNPCKHAB | BLEND_PUNPCKHBA | BLEND_PUNPCKLAB | BLEND_PUNPCKLBA;
    for iu in 0..(lane_size as u32) {
      let ix = lane_pattern[iu as usize];
      if ix >= 0 {
        let ix = ix as u32;
        if ix != iu / 2 + (iu & 1) * N as u32 {
          r &= !BLEND_PUNPCKLAB;
        }
        if ix != iu / 2 + ((iu & 1) ^ 1) * N as u32 {
          r &= !BLEND_PUNPCKLBA;
        }
        if ix != (iu + lane_size as u32) / 2 + (iu & 1) * N as u32 {
          r &= !BLEND_PUNPCKHAB;
        }
        if ix != (iu + lane_size as u32) / 2 + ((iu & 1) ^ 1) * N as u32 {
          r &= !BLEND_PUNPCKHBA;
        }
      }
    }

    for iu in 0..(lane_size as u32) {
      // check if it fits palignr
      let ix = lane_pattern[iu as usize];
      if ix >= 0 {
        let ix = ix as u32;
        let t = ix & !(N as u32);
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
        r |= BLEND_ROTATEBA;
      } else {
        r |= BLEND_ROTATEAB;
      }
      let elem_size = (V / N) as u32;
      r |= (((rot & (lane_size as u32 - 1)) * elem_size) as u64)
        << BLEND_ROTPATTERN;
    }

    if lane_size == 4 {
      // check if it fits shufps
      r |= BLEND_SHUFAB | BLEND_SHUFBA;
      for ii in 0..2 {
        let ix = lane_pattern[ii];
        if ix >= 0 {
          if ix & N as i32 > 0 {
            r &= !BLEND_SHUFAB;
          } else {
            r &= !BLEND_SHUFBA;
          }
        }
      }
      for ii in 2..4 {
        let ix = lane_pattern[ii];
        if ix >= 0 {
          if ix & N as i32 > 0 {
            r &= !BLEND_SHUFBA;
          } else {
            r &= !BLEND_SHUFAB;
          }
        }
      }
      if r & (BLEND_SHUFAB | BLEND_SHUFBA) > 0 {
        // fits shufps/shufpd
        let shuf_pattern = 0u8;
        for iu in 0..lane_size {
          shuf_pattern |= ((lane_pattern[iu] & 3) as u8) << (iu * 2);
        }
        r |= (shuf_pattern as u64) << BLEND_SHUFPATTERN;
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
      r |= BLEND_ROTATE_BIG | (rot as u64) << BLEND_ROTPATTERN;
    }
  }
  if lane_size == 2 && (r & (BLEND_SHUFAB | BLEND_SHUFBA)) > 0 {
    for ii in 0..N {
      r |= ((a[ii] & 1) as u64) << (BLEND_SHUFPATTERN + ii as u64);
    }
  }

  r
}
