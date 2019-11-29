use std::fmt;
use std::ops;

use crate::kernel::gen::*;
use crate::kernel::plane::Plane;
use crate::kernel::*;

// These are fully unrolled all the time. We don't have any nested loops
// here, so the biggest function will "just" have `128 / avx2_width()`
// or just 32 unrolled iterations.
// The generated functions are pretty large it seems, even for 8x8 sizes.
// It might be worthwhile to not unroll completely, especially as we start
// implementing 32 block transforms. This however poses issues for how we
// propagate the zeros in the second half of the 64 blocks.
// As I haven't even implemented 16 block transforms yet, let alone 64
// blocks, I have no idea if the const zeros thing will help.

fn tx_blocks_iter() -> impl Iterator<Item = Block> {
  Block::blocks().iter().cloned().filter(|b| {
    // no tx for > 64, also try to limit the number of kernels we generate
    // by restricting the odd block sizes.
    b.w() <= 64 && b.h() <= 64 && b.rect_log_ratio() < 2
  })
}
fn tx_inv_block_shift(b: Block) -> u32 {
  match b {
    Block(4, 4) | Block(4, 8) | Block(8, 4) => 0,

    Block(8, 8)
    | Block(8, 16)
    | Block(16, 8)
    | Block(4, 16)
    | Block(16, 4)
    | Block(16, 32)
    | Block(32, 16)
    | Block(32, 64)
    | Block(64, 32) => 1,

    Block(16, 16)
    | Block(16, 64)
    | Block(64, 16)
    | Block(64, 64)
    | Block(32, 32)
    | Block(8, 32)
    | Block(32, 8) => 2,

    _ => unreachable!(),
  }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum TxType {
  Id,
  Dct,
  Adst { flip: bool },
}
impl TxType {
  fn types() -> &'static [TxType] {
    const C: &'static [TxType] = &[
      TxType::Id,
      TxType::Dct,
      TxType::Adst { flip: false },
      TxType::Adst { flip: true },
    ];
    C
  }
  fn fn_suffix(&self) -> &'static str {
    match self {
      TxType::Id => "id",
      TxType::Dct => "dct",
      TxType::Adst { flip: false } => "adst",
      TxType::Adst { flip: true } => "flip_adst",
    }
  }
  fn flip(&self) -> bool {
    match self {
      TxType::Adst { flip } => *flip,
      _ => false,
    }
  }

  fn table_idx(&self) -> usize {
    match self {
      TxType::Id => 0,
      TxType::Dct => 1,
      TxType::Adst { flip: false } => 2,
      TxType::Adst { flip: true } => 3,
    }
  }

  fn inv_disable(&self, size: usize) -> bool {
    match (self, size) {
      (TxType::Adst { .. }, s) | (TxType::Adst { .. }, s) if s >= 32 => true,
      (TxType::Adst { flip: true }, _) => true,
      (TxType::Id, s) if s >= 64 => true,
      _ => false,
    }
  }
}
impl fmt::Display for TxType {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    f.pad(self.fn_suffix())
  }
}
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord)]
struct Tx2dType {
  col: TxType,
  row: TxType,
}
impl Tx2dType {
  fn types() -> impl Iterator<Item = Self> {
    let tys = TxType::types();
    tys
      .iter()
      .flat_map(move |&col| tys.iter().map(move |&row| Tx2dType { col, row }))
  }
  fn fn_suffix(&self) -> String {
    format!("{}_{}", self.row, self.col)
  }
  fn module_name(&self) -> Ident {
    let s = format!("x_{}_{}", self.row, self.col);
    Ident::new(&s, Span::call_site())
  }
}
impl fmt::Display for Tx2dType {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    f.pad(&self.fn_suffix())
  }
}

#[derive(Clone, Debug)]
enum IdxMap {
  /// Read/write.
  Idx(usize),
  /// The rest are read only
  Neg(usize),
  HalfBtf(NegIdx, usize, NegIdx, usize),
  AddClamp(NegIdx, NegIdx),
  Clamp(NegIdx),
  Const(SimdValue),
}
use self::IdxMap::{AddClamp, Clamp, HalfBtf};

impl From<usize> for IdxMap {
  fn from(v: usize) -> IdxMap {
    IdxMap::Idx(v)
  }
}
impl From<NegIdx> for IdxMap {
  fn from(v: NegIdx) -> IdxMap {
    match v {
      Pos(idx) => idx.into(),
      Neg(idx) => IdxMap::Neg(idx),
    }
  }
}
struct Stage<T>
where
  T: Array + ?Sized,
{
  idx_map: Vec<IdxMap>,
  prev: T,
}
impl<T> Stage<T>
where
  T: Array,
{
  fn next(prev: T, map: Vec<IdxMap>) -> Self {
    Stage { prev, idx_map: map }
  }
  fn const_fill(prev: T, to_len: usize, v: SimdValue) -> Self {
    assert_eq!(prev.ty(), v.ty());
    let len = prev.len();
    let mut map = Vec::with_capacity(to_len);
    for i in 0..len.min(to_len) {
      map.push(i.into());
    }
    if to_len > len {
      for _ in 0..(to_len - len) {
        map.push(IdxMap::Const(v.clone()));
      }
    }
    let o = Stage { prev, idx_map: map };

    o
  }
  fn into_vararr(self) -> VarArr {
    unimplemented!()
  }
}
impl<T> Array for Stage<T>
where
  T: Array + ?Sized,
{
  fn ty(&self) -> SimdType {
    self.prev.ty()
  }
  fn len(&self) -> usize {
    self.prev.len().max(self.idx_map.len())
  }
  fn get(&self, idx: usize) -> SimdValue {
    assert!(self.len() > idx);
    match self.idx_map.get(idx) {
      None => self.prev.get(idx), // passthrough
      Some(IdxMap::Idx(idx)) => self.prev.get(*idx),
      Some(IdxMap::Neg(idx)) => self.prev.get(*idx).neg(),
      Some(IdxMap::HalfBtf(w0, in0, w1, in1)) => {
        let in0 = self.prev.get(*in0);
        let in1 = self.prev.get(*in1);
        half_btf(*w0, &in0, *w1, &in1)
      }
      Some(IdxMap::AddClamp(l, r)) => match (*l, *r) {
        (Pos(l), Pos(r)) => {
          let l = self.prev.get(l);
          let r = self.prev.get(r);
          (l + r).clamp(quote!(min_value), quote!(max_value))
        }
        (Pos(l), Neg(r)) => {
          let l = self.prev.get(l);
          let r = self.prev.get(r);
          (l - r).clamp(quote!(min_value), quote!(max_value))
        }
        (Neg(l), Pos(r)) => {
          let l = self.prev.get(l);
          let r = self.prev.get(r);
          (r - l).clamp(quote!(min_value), quote!(max_value))
        }
        (Neg(l), Neg(r)) => {
          let l = self.prev.get(l);
          let r = self.prev.get(r);
          (l.neg() - r).clamp(quote!(min_value), quote!(max_value))
        }
      },
      Some(IdxMap::Clamp(idx)) => {
        let v = self.prev.get_neg(*idx);
        clamp(&v)
      }
      Some(IdxMap::Const(v)) => {
        assert_eq!(self.ty(), v.ty());
        v.clone()
      }
    }
  }
  fn set(&self, dst: &mut TokenStream, idx: usize, v: SimdValue) {
    assert!(self.len() > idx);
    assert_eq!(self.ty(), v.ty());
    let idx = match self.idx_map.get(idx) {
      None => idx,
      Some(IdxMap::Idx(idx)) => *idx,
      Some(map) => {
        panic!("{:?} is read only; can't write", map);
      }
    };

    self.prev.set(dst, idx, v);
  }
}

impl VarArr {
  fn add_clamp_merge<P, T, U>(
    dst: &mut TokenStream, prefix: P, lhs: T, rhs: U, map: &[(NegIdx, NegIdx)],
  ) -> VarArr
  where
    P: Display,
    T: Array,
    U: Array,
  {
    assert_eq!(lhs.ty(), rhs.ty());

    let mut names = Vec::with_capacity(map.len());
    for (i, &(l, r)) in map.iter().enumerate() {
      let l = lhs.get_neg(l);
      let r = rhs.get_neg(r);
      let v = clamp_pair(&l, &r)
        .let_(dst, format_args!("{}_{}", prefix, i))
        .unwrap_value();
      names.push(v);
    }

    VarArr { ty: lhs.ty(), prefix: prefix.to_string(), names }
  }
}

fn cospi_inv(ty: SimdType, idx: NegIdx) -> SimdValue {
  let v = match idx {
    Neg(idx) => quote!(-COSPI_INV[#idx]),
    Pos(idx) => quote!(COSPI_INV[#idx]),
  };

  ty.splat(v)
}

fn half_btf(
  w0: NegIdx, in0: &SimdValue, w1: NegIdx, in1: &SimdValue,
) -> SimdValue {
  assert_eq!(in0.ty(), in1.ty());
  let w0 = cospi_inv(in0.ty(), w0);
  let w1 = cospi_inv(in0.ty(), w1);
  let t = (w0 * in0) + (w1 * in1);
  SimdValue::from(
    t.ty(),
    quote! {
      { #t.round_shift(INV_COS_BIT as u32) }
    },
  )
}
fn clamp_pair(l: &SimdValue, r: &SimdValue) -> SimdValue {
  let t = (l + r).clamp(quote!(min_value), quote!(max_value));
  // wrap in a block for readability:
  SimdValue::from(t.ty(), quote!({ #t }))
}
fn clamp(v: &SimdValue) -> SimdValue {
  let t = v.clamp(quote!(min_value), quote!(max_value));
  // wrap in a block for readability:
  SimdValue::from(t.ty(), quote!({ #t }))
}

fn iidentity4<T, U>(_: &mut TokenStream, _disc: U, input: T) -> impl Array
where
  T: Array,
  U: Display,
{
  input.map(|_idx, v| {
    let v = v.ty().splat(quote!(SQRT2)) * v;
    // XXX magic number!
    v.round_shift(12u32)
  })
}
fn iidentity8<T, U>(_: &mut TokenStream, _disc: U, input: T) -> impl Array
where
  T: Array,
  U: Display,
{
  input.map(|_idx, v| v.ty().splat(quote!(2)) * v)
}
fn iidentity16<T, U>(_: &mut TokenStream, _disc: U, input: T) -> impl Array
where
  T: Array,
  U: Display,
{
  input.map(|_idx, v| {
    let v = v.ty().splat(quote!(SQRT2 * 2)) * v;
    // XXX magic number!
    v.round_shift(12u32)
  })
}
fn iidentity32<T, U>(_: &mut TokenStream, _disc: U, input: T) -> impl Array
where
  T: Array,
  U: Display,
{
  input.map(|_idx, v| v.ty().splat(quote!(4)) * v)
}

fn iadst4<T, U>(dst: &mut TokenStream, disc: U, input: T) -> impl Array
where
  T: Array,
  U: Display,
{
  let ty = input.ty();
  let sinpi_inv = |i: usize| -> SimdValue { ty.splat(quote!(SINPI_INV[#i])) };

  let x0 = input.get(0);
  let x1 = input.get(1);
  let x2 = input.get(2);
  let x3 = input.get(3);

  let s0 = sinpi_inv(1) * &x0;
  let s1 = sinpi_inv(2) * &x0;
  let s2 = sinpi_inv(3) * &x1;
  let s3 = sinpi_inv(4) * &x2;
  let s4 = sinpi_inv(1) * &x2;
  let s5 = sinpi_inv(2) * &x3;
  let s6 = sinpi_inv(4) * &x3;

  let s7 = (&x0 - &x2) + &x3;

  let s0 = s0 + s3;
  let s1 = s1 - s4;
  let s3 = s2;
  let s2 = sinpi_inv(3) * s7;

  let s0 = s0 + s5;
  let s1 = s1 - s6;

  let x0 = &s0 + &s3;
  let x1 = &s1 + &s3;
  let x2 = s2;
  let x3 = s0 + s1;

  let x3 = x3 - s3;

  let bit = 12u32;
  let out = vec![
    x0.round_shift(bit),
    x1.round_shift(bit),
    x2.round_shift(bit),
    x3.round_shift(bit),
  ];
  let out = Vector::new(ty, out);

  VarArr::let_(dst, format_args!("{}_iadst4", disc), &out)
}
fn iadst8<T, U>(dst: &mut TokenStream, disc: U, input: T) -> impl Array
where
  T: Array,
  U: Display,
{
  let stg1 = vec![
    7usize.into(),
    0usize.into(),
    5usize.into(),
    2usize.into(),
    3usize.into(),
    4usize.into(),
    1usize.into(),
    6usize.into(),
  ];
  let stg1 = Stage::next(input, stg1);

  let stg2 = vec![
    HalfBtf(Pos(4), 0, Pos(60), 1),
    HalfBtf(Pos(60), 0, Neg(4), 1),
    HalfBtf(Pos(20), 2, Pos(44), 3),
    HalfBtf(Pos(44), 2, Neg(20), 3),
    HalfBtf(Pos(36), 4, Pos(28), 5),
    HalfBtf(Pos(28), 4, Neg(36), 5),
    HalfBtf(Pos(52), 6, Pos(12), 7),
    HalfBtf(Pos(12), 6, Neg(52), 7),
  ];
  let stg2 = Stage::next(stg1, stg2);

  let stg3 = vec![
    AddClamp(Pos(0), Pos(4)),
    AddClamp(Pos(1), Pos(5)),
    AddClamp(Pos(2), Pos(6)),
    AddClamp(Pos(3), Pos(7)),
    AddClamp(Pos(0), Neg(4)),
    AddClamp(Pos(1), Neg(5)),
    AddClamp(Pos(2), Neg(6)),
    AddClamp(Pos(3), Neg(7)),
  ];
  let stg3 = Stage::next(stg2, stg3);

  let stg4 = vec![
    0usize.into(),
    1usize.into(),
    2usize.into(),
    3usize.into(),
    HalfBtf(Pos(16), 4, Pos(48), 5),
    HalfBtf(Pos(48), 4, Neg(16), 5),
    HalfBtf(Neg(48), 6, Pos(16), 7),
    HalfBtf(Pos(16), 6, Pos(48), 7),
  ];
  let stg4 = Stage::next(stg3, stg4);

  let stg5 = vec![
    AddClamp(Pos(0), Pos(2)),
    AddClamp(Pos(1), Pos(3)),
    AddClamp(Pos(0), Neg(2)),
    AddClamp(Pos(1), Neg(3)),
    AddClamp(Pos(4), Pos(6)),
    AddClamp(Pos(5), Pos(7)),
    AddClamp(Pos(4), Neg(6)),
    AddClamp(Pos(5), Neg(7)),
  ];
  let stg5 = Stage::next(stg4, stg5);

  let stg6 = vec![
    0usize.into(),
    1usize.into(),
    HalfBtf(Pos(32), 2, Pos(32), 3),
    HalfBtf(Pos(32), 2, Neg(32), 3),
    4usize.into(),
    5usize.into(),
    HalfBtf(Pos(32), 6, Pos(32), 7),
    HalfBtf(Pos(32), 6, Neg(32), 7),
  ];
  let stg6 = Stage::next(stg5, stg6);

  let stg7 = vec![
    0usize.into(),
    Neg(4).into(),
    6usize.into(),
    Neg(2).into(),
    3usize.into(),
    Neg(7).into(),
    5usize.into(),
    Neg(1).into(),
  ];
  let stg7 = Stage::next(stg6, stg7);

  VarArr::let_(dst, format_args!("{}_iadst8", disc), &stg7)
}

fn idct4<T, U>(dst: &mut TokenStream, disc: U, input: T) -> impl Array
where
  T: Array,
  U: Display,
{
  assert!(input.len() >= 4, "{} < {}", input.len(), 4);

  let stg1 = vec![0usize.into(), 2usize.into(), 1usize.into(), 3usize.into()];
  let stg1 = Stage::next(input, stg1).trunc(4);

  let stg2 = vec![
    IdxMap::HalfBtf(Pos(32), 0, Pos(32), 1),
    IdxMap::HalfBtf(Pos(32), 0, Neg(32), 1),
    IdxMap::HalfBtf(Pos(48), 2, Neg(16), 3),
    IdxMap::HalfBtf(Pos(16), 2, Pos(48), 3),
  ];
  let stg2 = Stage::next(stg1, stg2);
  let stg2 = VarArr::let_(dst, format_args!("{}_idct4_stg2", disc), &stg2);

  let stg3 = vec![
    AddClamp(Pos(0), Pos(3)),
    AddClamp(Pos(1), Pos(2)),
    AddClamp(Pos(1), Neg(2)),
    AddClamp(Pos(0), Neg(3)),
  ];
  let stg3 = Stage::next(stg2, stg3);
  VarArr::let_(dst, format_args!("{}_idct4_stg3", disc), &stg3)
}
fn idct8<T, U>(dst: &mut TokenStream, disc: U, input: T) -> impl Array
where
  T: Array,
  U: Display,
{
  assert!(input.len() >= 8, "{} < {}", input.len(), 8);

  let idct4 = vec![0usize.into(), 2usize.into(), 4usize.into(), 6usize.into()];
  let idct4 = Stage::next(&input, idct4);
  // XXX can't use format_args! here??? Just here??
  let idct4 = self::idct4(dst, format!("{}_idct8", disc), idct4);

  let stg1 = vec![1usize.into(), 5usize.into(), 3usize.into(), 7usize.into()];
  let stg1 = Stage::next(&input, stg1).trunc(4);

  let stg2 = vec![
    HalfBtf(Pos(56), 0, Neg(8), 3),
    HalfBtf(Pos(24), 1, Neg(40), 2),
    HalfBtf(Pos(40), 1, Pos(24), 2),
    HalfBtf(Pos(8), 0, Pos(56), 3),
  ];
  let stg2 = Stage::next(stg1, stg2);
  let stg2 = VarArr::let_(dst, format_args!("{}_idct8_stg2", disc), &stg2);

  let stg3 = vec![
    AddClamp(Pos(0), Pos(1)),
    AddClamp(Pos(0), Neg(1)),
    AddClamp(Neg(2), Pos(3)),
    AddClamp(Pos(2), Pos(3)),
  ];
  let stg3 = Stage::next(stg2, stg3);
  let stg3 = VarArr::let_(dst, format_args!("{}_idct8_stg3", disc), &stg3);

  let stg4 = vec![
    0usize.into(),
    HalfBtf(Neg(32), 1, Pos(32), 2),
    HalfBtf(Pos(32), 1, Pos(32), 2),
    3usize.into(),
  ];
  let stg4 = Stage::next(stg3, stg4);
  let stg4 = VarArr::let_(dst, format_args!("{}_idct8_stg4", disc), &stg4);

  let mut stg5 = Vec::new();
  for i in 0..4 {
    stg5.push((Pos(i), Pos(3 - i)));
  }
  for i in 0..4 {
    stg5.push((Pos(3 - i), Neg(i)));
  }

  VarArr::add_clamp_merge(
    dst,
    format_args!("{}_idct8", disc),
    idct4,
    stg4,
    &stg5,
  )
}
fn idct16<T, U>(dst: &mut TokenStream, disc: U, input: T) -> impl Array
where
  T: Array,
  U: Display,
{
  assert!(input.len() >= 16, "{} < {}", input.len(), 16);

  let idct8 = (0..8usize).map(|i| (i * 2).into()).collect::<Vec<_>>();
  let idct8 = Stage::next(&input, idct8);
  // XXX can't use format_args! here??? Just here??
  let idct8 = self::idct8(dst, format!("{}_idct16", disc), idct8);

  let stg1 = vec![
    1usize.into(),
    9usize.into(),
    5usize.into(),
    13usize.into(),
    3usize.into(),
    11usize.into(),
    7usize.into(),
    15usize.into(),
  ];
  let stg1 = Stage::next(&input, stg1).trunc(8);

  let stg2 = vec![
    HalfBtf(Pos(60), 0, Neg(4), 7),
    HalfBtf(Pos(28), 1, Neg(36), 6),
    HalfBtf(Pos(44), 2, Neg(20), 5),
    HalfBtf(Pos(12), 3, Neg(52), 4),
    HalfBtf(Pos(52), 3, Pos(12), 4),
    HalfBtf(Pos(20), 2, Pos(44), 5),
    HalfBtf(Pos(36), 1, Pos(28), 6),
    HalfBtf(Pos(4), 0, Pos(60), 7),
  ];
  let stg2 = Stage::next(stg1, stg2);
  let stg2 = VarArr::let_(dst, format_args!("{}_idct16_stg2", disc), &stg2);

  let stg3 = vec![
    AddClamp(Pos(0), Pos(1)),
    AddClamp(Pos(0), Neg(1)),
    AddClamp(Neg(2), Pos(3)),
    AddClamp(Pos(2), Pos(3)),
    AddClamp(Pos(4), Pos(5)),
    AddClamp(Pos(4), Neg(5)),
    AddClamp(Neg(6), Pos(7)),
    AddClamp(Pos(6), Pos(7)),
  ];
  let stg3 = Stage::next(stg2, stg3);
  let stg3 = VarArr::let_(dst, format_args!("{}_idct16_stg3", disc), &stg3);

  let stg4 = vec![
    0usize.into(),
    HalfBtf(Neg(16), 1, Pos(48), 6),
    HalfBtf(Neg(48), 2, Neg(16), 5),
    3usize.into(),
    4usize.into(),
    HalfBtf(Neg(16), 2, Pos(48), 5),
    HalfBtf(Pos(48), 1, Pos(16), 6),
    7usize.into(),
  ];
  let stg4 = Stage::next(stg3, stg4);
  let stg4 = VarArr::let_(dst, format_args!("{}_idct16_stg4", disc), &stg4);

  let stg5 = vec![
    AddClamp(Pos(0), Pos(3)),
    AddClamp(Pos(1), Pos(2)),
    AddClamp(Pos(1), Neg(2)),
    AddClamp(Pos(0), Neg(3)),
    AddClamp(Neg(4), Pos(7)),
    AddClamp(Neg(5), Pos(6)),
    AddClamp(Pos(5), Pos(6)),
    AddClamp(Pos(4), Pos(7)),
  ];
  let stg5 = Stage::next(stg4, stg5);
  let stg5 = VarArr::let_(dst, format_args!("{}_idct16_stg5", disc), &stg5);

  let stg6 = vec![
    0usize.into(),
    1usize.into(),
    HalfBtf(Neg(32), 2, Pos(32), 5),
    HalfBtf(Neg(32), 3, Pos(32), 4),
    HalfBtf(Pos(32), 3, Pos(32), 4),
    HalfBtf(Pos(32), 2, Pos(32), 5),
    6usize.into(),
    7usize.into(),
  ];
  let stg6 = Stage::next(stg5, stg6);
  let stg6 = VarArr::let_(dst, format_args!("{}_idct16_stg6", disc), &stg6);

  let mut stg7 = Vec::new();
  for i in 0..8 {
    stg7.push((Pos(i), Pos(7 - i)));
  }
  for i in 0..8 {
    stg7.push((Pos(7 - i), Neg(i)));
  }

  VarArr::add_clamp_merge(
    dst,
    format_args!("{}_idct16", disc),
    idct8,
    stg6,
    &stg7,
  )
}
fn idct32<T, U>(dst: &mut TokenStream, disc: U, input: T) -> impl Array
  where
    T: Array,
    U: Display,
{
  assert!(input.len() >= 32, "{} < {}", input.len(), 32);

  let stg1 = vec![
    0usize.into(),
    16usize.into(),
    8usize.into(),
    24usize.into(),
    4usize.into(),
    20usize.into(),
    12usize.into(),
    28usize.into(),
    2usize.into(),
    18usize.into(),
    10usize.into(),
    26usize.into(),
    6usize.into(),
    22usize.into(),
    14usize.into(),
    30usize.into(),
    1usize.into(),
    17usize.into(),
    9usize.into(),
    25usize.into(),
    5usize.into(),
    21usize.into(),
    13usize.into(),
    29usize.into(),
    3usize.into(),
    19usize.into(),
    11usize.into(),
    27usize.into(),
    7usize.into(),
    23usize.into(),
    15usize.into(),
    31usize.into(),
  ];
  let stg1 = Stage::next(&input, stg1);

  let mut stg2 = vec![];
  for i in 0..16usize {
    stg2.push(i.into());
  }
  let stg2b = vec![
    HalfBtf(Pos(62), 16, Neg(2), 31),
    HalfBtf(Pos(30), 17, Neg(34), 30),
    HalfBtf(Pos(46), 18, Neg(18), 29),
    HalfBtf(Pos(14), 19, Neg(50), 28),
    HalfBtf(Pos(54), 20, Neg(10), 27),
    HalfBtf(Pos(22), 21, Neg(42), 26),
    HalfBtf(Pos(38), 22, Neg(26), 25),
    HalfBtf(Pos(6),  23, Neg(58), 24),
    HalfBtf(Pos(58), 23, Pos(6), 24),
    HalfBtf(Pos(26), 22, Pos(38), 25),
    HalfBtf(Pos(42), 21, Pos(22), 26),
    HalfBtf(Pos(10), 20, Pos(54), 27),
    HalfBtf(Pos(50), 19, Pos(14), 28),
    HalfBtf(Pos(18), 18, Pos(46), 29),
    HalfBtf(Pos(34), 17, Pos(30), 30),
    HalfBtf(Pos(2),  16, Pos(62), 31),
  ];
  stg2.extend(stg2b.into_iter());
  let stg2 = Stage::next(stg1, stg2);
  let stg2 = VarArr::let_(dst, format_args!("{}_idct32_stg2", disc),
                          &stg2);

  let mut stg3 = vec![];
  for i in 0..8usize {
    stg3.push(i.into());
  }
  let stg3b = vec![
    HalfBtf(Pos(60), 8,  Neg(4),  15),
    HalfBtf(Pos(28), 9,  Neg(36), 14),
    HalfBtf(Pos(44), 10, Neg(20), 13),
    HalfBtf(Pos(12), 11, Neg(52), 12),
    HalfBtf(Pos(52), 11, Pos(12), 12),
    HalfBtf(Pos(20), 10, Pos(44), 13),
    HalfBtf(Pos(36), 9,  Pos(28), 14),
    HalfBtf(Pos(4),  8,  Pos(60), 15),
    AddClamp(Pos(16), Pos(17)),
    AddClamp(Pos(16), Neg(17)),
    AddClamp(Neg(18), Pos(19)),
    AddClamp(Pos(18), Pos(19)),
    AddClamp(Pos(20), Pos(21)),
    AddClamp(Pos(20), Neg(21)),
    AddClamp(Neg(22), Pos(23)),
    AddClamp(Pos(22), Pos(23)),
    AddClamp(Pos(24), Pos(25)),
    AddClamp(Pos(24), Neg(25)),
    AddClamp(Neg(26), Pos(27)),
    AddClamp(Pos(26), Pos(27)),
    AddClamp(Pos(28), Pos(29)),
    AddClamp(Pos(28), Neg(29)),
    AddClamp(Neg(30), Pos(31)),
    AddClamp(Pos(30), Pos(31)),
  ];
  stg3.extend(stg3b.into_iter());
  let stg3 = Stage::next(stg2, stg3);
  let stg3 = VarArr::let_(dst, format_args!("{}_idct32_stg3", disc),
                          &stg3);

  let stg4 = vec![
    0usize.into(),
    1usize.into(),
    2usize.into(),
    3usize.into(),
    HalfBtf(Pos(56), 4,  Neg(8),  7),
    HalfBtf(Pos(24), 5,  Neg(40), 6),
    HalfBtf(Pos(40), 5, Pos(24), 6),
    HalfBtf(Pos(8),  4, Pos(56), 7),
    AddClamp(Pos(8), Pos(9)),
    AddClamp(Pos(8), Neg(9)),
    AddClamp(Neg(10), Pos(11)),
    AddClamp(Pos(10), Pos(11)),
    AddClamp(Pos(12), Pos(13)),
    AddClamp(Pos(12), Neg(13)),
    AddClamp(Neg(14), Pos(15)),
    AddClamp(Pos(14), Pos(15)),
    16usize.into(),
    HalfBtf(Neg(8), 17, Pos(56), 30),
    HalfBtf(Neg(56), 18, Neg(8), 29),
    19usize.into(),
    20usize.into(),
    HalfBtf(Neg(40), 21, Pos(24), 26),
    HalfBtf(Neg(24), 22, Neg(40), 25),
    23usize.into(),
    24usize.into(),
    HalfBtf(Neg(40), 22,  Pos(24),  25),
    HalfBtf(Pos(24), 21,  Pos(40), 26),
    27usize.into(),
    28usize.into(),
    HalfBtf(Neg(8), 18, Pos(56), 29),
    HalfBtf(Pos(56), 17, Pos(8), 30),
    31usize.into(),
  ];
  let stg4 = Stage::next(stg3, stg4);
  let stg4 = VarArr::let_(dst, format_args!("{}_idct32_stg4", disc), &stg4);

  let stg5 = vec![
    HalfBtf(Pos(32), 0, Pos(32), 1),
    HalfBtf(Pos(32), 0, Neg(32), 1),
    HalfBtf(Pos(48), 2, Neg(16), 3),
    HalfBtf(Pos(16), 2, Pos(48), 3),
    AddClamp(Pos(4), Pos(5)),
    AddClamp(Pos(4), Neg(5)),
    AddClamp(Neg(6), Pos(7)),
    AddClamp(Pos(6), Pos(7)),
    8usize.into(),
    HalfBtf(Neg(16), 9, Pos(48), 14),
    HalfBtf(Neg(48), 10, Neg(16), 13),
    11usize.into(),
    12usize.into(),
    HalfBtf(Neg(16), 10, Pos(48), 13),
    HalfBtf(Pos(48), 9, Pos(16), 14),
    15usize.into(),
    AddClamp(Pos(16), Pos(19)),
    AddClamp(Pos(17), Pos(18)),
    AddClamp(Pos(17), Neg(18)),
    AddClamp(Pos(16), Neg(19)),
    AddClamp(Neg(20), Pos(23)),
    AddClamp(Neg(21), Pos(22)),
    AddClamp(Pos(21), Pos(22)),
    AddClamp(Pos(20), Pos(23)),
    AddClamp(Pos(24), Pos(27)),
    AddClamp(Pos(25), Pos(26)),
    AddClamp(Pos(25), Neg(26)),
    AddClamp(Pos(24), Neg(27)),
    AddClamp(Neg(28), Pos(31)),
    AddClamp(Neg(29), Pos(30)),
    AddClamp(Pos(29), Pos(30)),
    AddClamp(Pos(28), Pos(31)),
  ];
  let stg5 = Stage::next(stg4, stg5);
  let stg5 = VarArr::let_(dst, format_args!("{}_idct32_stg5", disc), &stg5);

  let stg6 = vec![
    AddClamp(Pos(0), Pos(3)),
    AddClamp(Pos(1), Pos(2)),
    AddClamp(Pos(1), Neg(2)),
    AddClamp(Pos(0), Neg(3)),
    4usize.into(),
    HalfBtf(Neg(32), 5, Pos(32), 6),
    HalfBtf(Pos(32), 5, Pos(32), 6),
    7usize.into(),
    AddClamp(Pos(8), Pos(11)),
    AddClamp(Pos(9), Pos(10)),
    AddClamp(Pos(9), Neg(10)),
    AddClamp(Pos(8), Neg(11)),
    AddClamp(Neg(12), Pos(15)),
    AddClamp(Neg(13), Pos(14)),
    AddClamp(Pos(13), Pos(14)),
    AddClamp(Pos(12), Pos(15)),
    16usize.into(),
    17usize.into(),
    HalfBtf(Neg(16), 18, Pos(48), 29),
    HalfBtf(Neg(16), 19, Pos(48), 28),
    HalfBtf(Neg(48), 20, Neg(16), 27),
    HalfBtf(Neg(48), 21, Neg(16), 26),
    22usize.into(),
    23usize.into(),
    24usize.into(),
    25usize.into(),
    HalfBtf(Neg(16), 21, Pos(48), 26),
    HalfBtf(Neg(16), 20, Pos(48), 27),
    HalfBtf(Pos(48), 19, Pos(16), 28),
    HalfBtf(Pos(48), 18, Pos(16), 29),
    30usize.into(),
    31usize.into(),
  ];
  let stg6 = Stage::next(stg5, stg6);
  let stg6 = VarArr::let_(dst, format_args!("{}_idct32_stg6", disc), &stg6);

  let stg7 = vec![
    AddClamp(Pos(0), Pos(7)),
    AddClamp(Pos(1), Pos(6)),
    AddClamp(Pos(2), Pos(5)),
    AddClamp(Pos(3), Pos(4)),
    AddClamp(Pos(3), Neg(4)),
    AddClamp(Pos(2), Neg(5)),
    AddClamp(Pos(1), Neg(6)),
    AddClamp(Pos(0), Neg(7)),
    8usize.into(),
    9usize.into(),
    HalfBtf(Neg(32), 10, Pos(32), 13),
    HalfBtf(Neg(32), 11, Pos(32), 12),
    HalfBtf(Pos(32), 11, Pos(32), 12),
    HalfBtf(Pos(32), 10, Pos(32), 13),
    14usize.into(),
    15usize.into(),
    AddClamp(Pos(16), Pos(23)),
    AddClamp(Pos(17), Pos(22)),
    AddClamp(Pos(18), Pos(21)),
    AddClamp(Pos(19), Pos(20)),
    AddClamp(Pos(19), Neg(20)),
    AddClamp(Pos(18), Neg(21)),
    AddClamp(Pos(17), Neg(22)),
    AddClamp(Pos(16), Neg(23)),
    AddClamp(Neg(24), Pos(31)),
    AddClamp(Neg(25), Pos(30)),
    AddClamp(Neg(26), Pos(29)),
    AddClamp(Neg(27), Pos(28)),
    AddClamp(Pos(27), Pos(28)),
    AddClamp(Pos(26), Pos(29)),
    AddClamp(Pos(25), Pos(30)),
    AddClamp(Pos(24), Pos(31)),
  ];
  let stg7 = Stage::next(stg6, stg7);
  let stg7 = VarArr::let_(dst, format_args!("{}_idct32_stg7", disc), &stg7);

  let mut stg8 = Vec::new();
  for i in 0..8 {
    stg8.push(AddClamp(Pos(i), Pos(16 - i)));
  }
  for i in 0..8 {
    stg8.push(AddClamp(Pos(7 - i), Neg(i + 8)));
  }
  for i in 16..20usize {
    stg8.push(i.into())
  }
  stg8.extend(vec![
    HalfBtf(Neg(32), 20, Pos(32), 27),
    HalfBtf(Neg(32), 21, Pos(32), 26),
    HalfBtf(Neg(32), 22, Pos(32), 25),
    HalfBtf(Neg(32), 23, Pos(32), 24),
    HalfBtf(Pos(32), 23, Pos(32), 24),
    HalfBtf(Pos(32), 22, Pos(32), 25),
    HalfBtf(Pos(32), 21, Pos(32), 26),
    HalfBtf(Pos(32), 20, Pos(32), 27),
  ].into_iter());
  for i in 28..32usize {
    stg8.push(i.into())
  }
  let stg8 = Stage::next(stg7, stg8);
  let stg8 = VarArr::let_(dst, format_args!("{}_idct32_stg8", disc), &stg8);

  let mut stg9 = Vec::new();
  for i in 0..16 {
    stg9.push(AddClamp(Pos(i), Pos(31 - i)));
  }
  for i in 0..16 {
    stg9.push(AddClamp(Pos(15 - i), Neg(i + 16)));
  }
  let stg9 = Stage::next(stg8, stg9);
  let stg9 = VarArr::let_(dst, format_args!("{}_idct32", disc),
                          &stg9);

  stg9
}

fn inv_tx_kernel(
  dst: &mut Module, ty: Tx2dType, b: Block, px: PixelType, feature: IsaFeature,
) -> Option<(TokenStream, TokenStream)> {
  // This produces some pretty incomprehensible code. Sowwy!

  const MAX_LANES: usize = 8;
  const MAX_PTR_LANES: usize = 8;

  if ty.col.inv_disable(b.w()) {
    return None;
  }
  if ty.row.inv_disable(b.h()) {
    return None;
  }

  feature.to_tokens(&mut *dst);

  let fn_name_str = format!(
    "inv_tx_add_{}_{}_{}_{}",
    ty.fn_suffix(),
    b.fn_suffix(),
    px.type_str(),
    feature.fn_suffix(),
  );
  let fn_name = Ident::new(&fn_name_str, Span::call_site());

  let w = b.w();
  let h = b.h();

  let input_ptr = Ident::new("input", Span::call_site());
  let input_stride = w.min(32);
  let input = Plane::new_const_stride(input_ptr.clone(), input_stride);
  let output_ptr = Ident::new("output", Span::call_site());
  let output = Plane::new(&output_ptr);

  let buffer_ptr = Ident::new("buffer_ptr", Span::call_site());
  let buffer = Plane::new_const_stride(buffer_ptr.clone(), w);

  let rect_type = b.rect_log_ratio();

  let calc_prim = PrimType::I32; // TODO use i16 if possible
  let col_lanes = b.w().min(MAX_LANES);
  let row_lanes = b.h().min(MAX_LANES);
  let area = b.area();

  let col_simd = SimdType::new(calc_prim, col_lanes);
  let row_simd = SimdType::new(calc_prim, row_lanes);

  let mut body = TokenStream::default();
  let mut nerf_table_output = false;

  // row
  let row_zero = row_simd.splat(quote!(0));
  let gather_ty = row_simd;
  let gather_ptr_ty = quote! { <Simd<[*const #calc_prim; #row_lanes]>> };
  let scatter_ptr_ty = quote! { <Simd<[*mut #calc_prim; #row_lanes]>> };
  // note: buffer *can not* be left uninitialized here.
  body.extend(quote! {
    let range = bit_depth + 8;
    let max_value = ((1i64 << (range - 1)) - 1) as #calc_prim;
    let min_value = (-(1i64 << (range - 1))) as #calc_prim;
    let mut buffer: AlignedArray<[#calc_prim; #area]> =
      AlignedArray::new([0; #area]);
    let mut buffer_ptr = buffer.as_mut_ptr();
    let all_mask = <Simd<[m8; #row_lanes]>>::splat(true);
  });
  let gather_v =
    SimdValue::default(gather_ty).let_(&mut body, "input_gather_initial");
  let gather_offsets = gather_ty.indices(|idx| {
    let idx = (idx * input_stride) as u32;
    quote!(#idx)
  });
  let gather_offsets = gather_offsets.let_(&mut body, "gather_offsets");

  let iter = quote!((0..#h.min(32)).step_by(#row_lanes));

  loop {
    let row_loop = {
      let mut body = TokenStream::default();

      // input is in row major order, but we need to read continuous columns
      // for vectorization. So we have to gather here :/

      // TODO: > 8 gathers
      assert!(col_simd.w() <= 8);
      let input = input.add_rc(quote!(c), 0usize);
      body.extend(quote! {
        let gather_ptr_base = #gather_ptr_ty::splat(#input);
      });
      let mut gathered_input = vec![];
      for r in 0..w {
        let r_u32 = r as u32;
        let ptrs = quote! {
          gather_ptr_base
            .add(Simd::<[usize; #row_lanes]>::from_cast(#gather_offsets + #r_u32))
        };
        let vname_str = format!("input_{}", r);
        let vname = Ident::new(&vname_str, Span::call_site());
        if r < 32 {
          body.extend(quote! {
            let #vname = #ptrs.read(all_mask, #gather_v);
          });
        } else {
          let zero = gather_ty.splat(quote!(0));
          body.extend(quote! {
            let #vname = #zero;
          });
        }
        let v = SimdValue::from(row_simd, quote!(#vname));
        gathered_input.push(v);
      }

      let input = Vector::new(row_simd, gathered_input);

      let input = input.map(|_idx, v| {
        let v = if rect_type == 1 {
          let inv_sqrt2 = v.ty().splat(quote!(INV_SQRT2));
          (v * inv_sqrt2).round_shift(quote!(SQRT2_BITS as u32))
        } else {
          v
        };
        clamp(&v)
      });
      let input = Stage::const_fill(input, b.w(), row_zero.clone());
      let input = VarArr::let_(&mut body, "input", &input);

      let out: Box<dyn Array> = match (ty.row, b.w()) {
        (TxType::Id, 4) => Box::new(iidentity4(&mut body, "row", input)),
        (TxType::Id, 8) => Box::new(iidentity8(&mut body, "row", input)),
        (TxType::Id, 16) => Box::new(iidentity16(&mut body, "row", input)),
        (TxType::Id, 32) => Box::new(iidentity32(&mut body, "row", input)),
        (TxType::Dct, 4) => Box::new(idct4(&mut body, "row", input)),
        (TxType::Dct, 8) => Box::new(idct8(&mut body, "row", input)),
        (TxType::Dct, 16) => Box::new(idct16(&mut body, "row", input)),
        (TxType::Dct, 32) => Box::new(idct32(&mut body, "row", input)),
        (TxType::Adst { flip: false }, 4) => {
          Box::new(iadst4(&mut body, "row", input))
        }
        (TxType::Adst { flip: false }, 8) => {
          Box::new(iadst8(&mut body, "row", input))
        }
        _ => {
          nerf_table_output = true;
          break;
        }
      };

      // now write the transposed output
      let out = VarArr::let_(&mut body, "out", &out);
      let buffer = buffer.add_rc(quote!(c), 0usize);
      body.extend(quote! {
        let scatter_ptr_base = #scatter_ptr_ty::splat(#buffer);
      });
      for r in 0..w {
        let offsets = gather_ty.indices(|idx| {
          let idx = (idx * w + r) as u32;
          quote!(#idx)
        });
        let ptrs = quote! {
          scatter_ptr_base
            .add(Simd::<[usize; #row_lanes]>::from_cast(#offsets))
        };

        let v = out.get(r);
        body.extend(quote! {
          #ptrs.write(all_mask, #v);
        });
      }

      body
    };
    body.extend(quote! {
      for c in #iter {
        #row_loop
      }
    });
    break;
  }

  // columns
  body.extend(quote! {
    let range = ::std::cmp::max(bit_depth + 6, 16);
    let max_value = ((1i64 << (range - 1)) - 1) as #calc_prim;
    let min_value = (-(1i64 << (range - 1))) as #calc_prim;
    let mut buffer_ptr = buffer.as_ptr();
  });
  let px_simd = SimdType::new(px.into(), col_lanes);
  let add_min = quote!(0);
  let add_max = quote!(((1 << bit_depth) - 1) as #px);
  let iter = quote!((0..#w).step_by(#col_lanes));
  loop {
    let col_loop = {
      let mut body = TokenStream::default();

      // can't use Slice here because we need to read columns; ie
      // all rows of columns 0..8, then all rows of 8..16, etc
      let mut input = Vec::new();
      for r in 0..h {
        let buffer = buffer.add_rc(r, 0usize);
        let v = col_simd.uload(&buffer);
        input.push(v);
      }
      let input = Vector::new(col_simd, input).map(|_idx, v| {
        let shift = tx_inv_block_shift(b);
        clamp(&v.round_shift(quote!(#shift)))
      });
      let input = VarArr::let_(&mut body, "input", &input);

      let out = match (ty.col, b.h()) {
        (TxType::Id, 4) => {
          Box::new(iidentity4(&mut body, "col", input)) as Box<dyn Array>
        }
        (TxType::Id, 8) => Box::new(iidentity8(&mut body, "col", input)),
        (TxType::Id, 16) => Box::new(iidentity16(&mut body, "col", input)),
        (TxType::Id, 32) => Box::new(iidentity32(&mut body, "col", input)),
        (TxType::Dct, 4) => Box::new(idct4(&mut body, "col", input)),
        (TxType::Dct, 8) => Box::new(idct8(&mut body, "col", input)),
        (TxType::Dct, 16) => Box::new(idct16(&mut body, "col", input)),
        (TxType::Dct, 32) => Box::new(idct32(&mut body, "col", input)),
        (TxType::Adst { flip: false }, 4) => {
          Box::new(iadst4(&mut body, "col", input))
        }
        (TxType::Adst { flip: false }, 8) => {
          Box::new(iadst8(&mut body, "col", input))
        }
        _ => {
          nerf_table_output = true;
          break;
        }
      };

      let out = out.map(|_idx, v| v.round_shift(quote!(4)));

      // now add the inv tx, clamp, and write to `output`:
      for i in 0..out.len() {
        let out_ptr = output.add_rc(i, 0usize);
        let original = px_simd.uload(&out_ptr).cast(col_simd);
        let inv = out.get(i);
        // TODO saturating adds for PixelType::U8 + PrimType::I*.
        let new = original + inv;
        let new = new.clamp(&add_min, &add_max).cast(px_simd);

        new.ustore(&mut body, out_ptr);
      }

      body.extend(quote! {
        #output_ptr = #output_ptr.add(#col_lanes);
        #buffer_ptr = #buffer_ptr.add(#col_lanes);
      });

      body
    };
    body.extend(quote! {
      for r in #iter {
        #col_loop
      }
    });
    break;
  }

  // I've used `&(mut)?` here so that we get the usual Rust function
  // parameter attributes. Specifically, noalias attributes.
  dst.extend(quote! {
    #[inline(never)]
    #[allow(unused_variables)]
    #[allow(unused_mut)]
    #[allow(unused_assignments)]
    #[allow(unused_parens)]
    pub unsafe fn #fn_name(input: &#calc_prim,
                           output: &mut #px, output_stride: u32,
                           bit_depth: u8, ) {
      let mut input = input as *const #calc_prim;
      let mut output = output as *mut #px;
      let output_stride = output_stride as usize;

      #body
    }
  });

  if nerf_table_output {
    // don't print duplicate warnings; each IsaFeature will get one,
    // same for the pixel type
    if let IsaFeature::Native = feature {
      if let PixelType::U8 = px {
        // don't panic for now
        println!(
          "cargo:warning=unimplemented inv transform: \
           ty = {}, b = {:?}",
          ty, b
        );
      }
    }
    return None;
  }

  let feature_idx = feature.index();
  let b_enum = b.table_idx();
  let row_idx = ty.row.table_idx();
  let col_idx = ty.col.table_idx();
  let idx = quote! {
    [#feature_idx][#b_enum][#row_idx][#col_idx]
  };
  let path = dst.item_path(&fn_name);

  Some((idx, path))
}

// TODO? The forward transform is vectorized, but isn't unrolled.
// Need to do some profiling to see how much of an issue it is.
/*fn fwd_tx_kernel(
  dst: &mut Module, ty: Tx2dType, b: Block, px: PixelType, feature: IsaFeature,
) {

  feature.to_tokens(&mut *dst);

  let fn_name_str = format!(
    "fwd_tx_{}_{}_{}_{}",
    ty.fn_suffix(),
    b.fn_suffix(),
    px.type_str(),
    feature.fn_suffix(),
  );
  let fn_name = Ident::new(&fn_name_str, Span::call_site());

  let calc_prim = PrimType::I32; // TODO use i16 if possible
  let shift_idx = if px == PixelType::U8 {
    // use static index so LLVM can avoid some branches.
    // TODO this may not always be valid? idk.
    quote!(0)
  } else {
    quote!(shift_idx)
  };
  let col_lanes = b.w().min(calc_prim.avx2_width());
  let row_lanes = b.h().min(calc_prim.avx2_width());

  let mut body = TokenStream::default();

  let fwd_shift = format!("FWD_SHIFT_{}X{}", b.w(), b.h());
  let fwd_shift = Ident::new(&fwd_shift, Span::call_site());

  let shift0 = Var::new("shift0", quote!(-#fwd_shift[#shift_idx][0]));
  shift0.to_tokens(&mut body);
  let shift0 = shift0.name();
  let shift1 = Var::new("shift1", quote!(-#fwd_shift[#shift_idx][1]));
  shift1.to_tokens(&mut body);
  let shift1 = shift1.name();
  let shift2 = Var::new("shift2", quote!(-#fwd_shift[#shift_idx][2]));
  shift2.to_tokens(&mut body);
  let shift2 = shift2.name();

  // columns
  for c in (0..b.w()).step_by(col_lanes) {

  }
  for r in (0..b.h()).step_by(row_lanes) {

  }

  dst.extend(quote! {
    #[inline(never)]
    #[allow(unused_variables)]
    #[allow(unused_mut)]
    #[allow(unused_assignments)]
    #[allow(unused_parens)]
    pub unsafe fn #fn_name(left: &#px, left_stride: u32,
                           right: &#px, right_stride: u32,
                           coeffs: &mut #calc_prim,
                           bit_depth: u8, ) {
      let left = left as *const #px;
      let right = right as *const #px;
      let left_stride = left_stride as usize;
      let right_stride = right_stride as usize;

      let shift_idx = (bd - 8) / 2;
      let max_sample_val = ((1 << bit_depth) - 1) as i32;
      let intermediate_bits = 4 - if bit_depth == 12 { 2 } else { 0 };

      #body
    }
  });
}*/

pub(super) fn inv_tx_add_kernels(file: &mut dyn Write) {
  write_prelude(file);

  let args = vec![
    quote!(input: &i32),
    quote!(output: &mut T),
    quote!(output_stride: u32),
    quote!(bit_depth: u8),
  ];
  let mut kernels = KernelSet::new(
    "InvTxAddF",
    &args,
    None,
    "INV_TX_ADD",
    vec![quote!(4), quote!(4)],
  );

  for isa in IsaFeature::sets() {
    let mut isa_module = Module::new_root(isa.module_name());
    for px in PixelType::types_iter() {
      let mut px_module = isa_module.new_child(px.module_name());
      for tx in Tx2dType::types() {
        let mut tx_module =
          px_module.new_child_file("inv_tx_add", tx.module_name());
        StdImports.to_tokens(&mut tx_module);
        tx_module.extend(quote! {
          use crate::transform::inverse::*;
        });
        for block in tx_blocks_iter() {
          if let Some((idx, path)) =
            inv_tx_kernel(&mut tx_module, tx, block, px, isa)
          {
            kernels.push_kernel(px, idx, path);
          }
        }
        tx_module.finish_child(&mut px_module);
      }
      px_module.finish_child(&mut isa_module);
    }
    isa_module.finish_root(file);
  }

  println!("generated {} inv tx add kernels", kernels.len());

  let tables = kernels.tables();
  writeln!(file, "{}", tables).expect("write inv tx add kernel tables");
}
