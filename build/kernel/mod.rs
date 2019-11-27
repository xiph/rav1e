use proc_macro2::{Ident, Punct, Spacing, Span, TokenStream, TokenTree};
use quote::*;

use std::env;
use std::fmt::Display;
use std::io::Write;
use std::ops::{self, *};
use std::path::{Path, PathBuf};

pub mod gen;
pub mod plane;

pub use self::plane::Plane;

pub const MAX_UNROLL: usize = 32;

pub fn macro_<T>(mac: T) -> TokenStream
where
  T: Display,
{
  let mac = Ident::new(&mac.to_string(), Span::call_site());
  let mut ts = quote!(#mac);
  ts.extend(
    Some(TokenTree::Punct(Punct::new('!', Spacing::Alone).into())).into_iter(),
  );
  ts
}
pub fn call_macro<T, U>(mac: T, args: U) -> TokenStream
where
  T: Display,
  U: ToTokens,
{
  let mac = Ident::new(&mac.to_string(), Span::call_site());
  let mut ts = quote!(#mac);
  ts.extend(
    Some(TokenTree::Punct(Punct::new('!', Spacing::Alone).into())).into_iter(),
  );
  args.to_tokens(&mut ts);
  ts
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Block(usize, usize);
impl Block {
  fn blocks() -> &'static [Block] {
    const C: &'static [Block] = &[
      Block(4, 4),
      Block(4, 8),
      Block(4, 16),
      Block(8, 4),
      Block(8, 8),
      Block(8, 16),
      Block(8, 32),
      Block(16, 4),
      Block(16, 8),
      Block(16, 16),
      Block(16, 32),
      Block(16, 64),
      Block(32, 8),
      Block(32, 16),
      Block(32, 32),
      Block(32, 64),
      Block(64, 16),
      Block(64, 32),
      Block(64, 64),
      Block(64, 128),
      Block(128, 64),
      Block(128, 128),
    ];
    C
  }
  fn blocks_iter() -> impl Iterator<Item = Block> {
    Self::blocks().iter().cloned()
  }
  fn fn_suffix(&self) -> String {
    format!("{}_{}", self.0, self.1)
  }

  fn w(&self) -> usize {
    self.0
  }
  fn h(&self) -> usize {
    self.1
  }
  fn area(&self) -> usize {
    self.w() * self.h()
  }
  fn rect_log_ratio(&self) -> u32 {
    fn ilog(this: usize) -> isize {
      use std::mem::size_of;
      (size_of::<usize>() * 8 - this.leading_zeros() as usize) as isize
    }
    (ilog(self.w()) - ilog(self.h())).abs() as _
  }

  fn table_idx(&self) -> usize {
    match self {
      Block(4, 4) => 0,
      Block(4, 8) => 1,
      Block(8, 4) => 2,
      Block(8, 8) => 3,
      Block(8, 16) => 4,
      Block(16, 8) => 5,
      Block(16, 16) => 6,
      Block(16, 32) => 7,
      Block(32, 16) => 8,
      Block(32, 32) => 9,
      Block(32, 64) => 10,
      Block(64, 32) => 11,
      Block(64, 64) => 12,
      Block(64, 128) => 13,
      Block(128, 64) => 14,
      Block(128, 128) => 15,
      Block(4, 16) => 16,
      Block(16, 4) => 17,
      Block(8, 32) => 18,
      Block(32, 8) => 19,
      Block(16, 64) => 20,
      Block(64, 16) => 21,
      _ => unreachable!(),
    }
  }

  fn as_type(&self) -> BlockType {
    BlockType(*self)
  }
  fn as_enum(&self) -> BlockEnum {
    BlockEnum(*self)
  }
}

struct BlockType(Block);
struct BlockEnum(Block);
impl ToTokens for BlockType {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    let s = format!("Block{}x{}", (self.0).0, (self.0).1);
    let s = Ident::new(&s, Span::call_site());
    tokens.extend(quote!(crate::util::#s));
  }
}
impl ToTokens for BlockEnum {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    let s = format!("BLOCK_{}X{}", (self.0).0, (self.0).1);
    let s = Ident::new(&s, Span::call_site());
    tokens.extend(quote!(crate::partition::BlockSize::#s));
  }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PixelType {
  U8,
  U16,
}
impl PixelType {
  fn types() -> &'static [PixelType] {
    const C: &'static [PixelType] = &[PixelType::U8, PixelType::U16];
    C
  }
  fn types_iter() -> impl Iterator<Item = PixelType> {
    Self::types().iter().cloned()
  }
  fn type_str(&self) -> &'static str {
    match self {
      PixelType::U8 => "u8",
      PixelType::U16 => "u16",
    }
  }
  fn module_name(&self) -> Ident {
    let s = match self {
      PixelType::U8 => "p_u8",
      PixelType::U16 => "p_u16",
    };
    Ident::new(s, Span::call_site())
  }
  fn avx2_width(&self) -> usize {
    match self {
      PixelType::U8 => 256 / 8,
      PixelType::U16 => 256 / 16,
    }
  }
}
impl ToTokens for PixelType {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    tokens.extend(match self {
      PixelType::U8 => quote!(u8),
      PixelType::U16 => quote!(u16),
    });
  }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PrimType {
  U8,
  I8,
  U16,
  I16,
  U32,
  I32,
}
impl PrimType {
  fn type_str(&self) -> &'static str {
    match self {
      PrimType::U8 => "u8",
      PrimType::I8 => "i8",
      PrimType::U16 => "u16",
      PrimType::I16 => "i16",
      PrimType::U32 => "u32",
      PrimType::I32 => "i32",
    }
  }
  fn bits(&self) -> usize {
    match self {
      PrimType::U8 | PrimType::I8 => 8,
      PrimType::U16 | PrimType::I16 => 16,
      PrimType::U32 | PrimType::I32 => 32,
    }
  }
  fn avx2_width(&self) -> usize {
    256 / self.bits()
  }
  fn is_signed(&self) -> bool {
    match self {
      PrimType::I8 | PrimType::I16 | PrimType::I32 => true,
      _ => false,
    }
  }
}
impl ToTokens for PrimType {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    tokens.extend(match self {
      PrimType::U8 => quote!(u8),
      PrimType::I8 => quote!(i8),
      PrimType::U16 => quote!(u16),
      PrimType::I16 => quote!(i16),
      PrimType::U32 => quote!(u32),
      PrimType::I32 => quote!(i32),
    });
  }
}
impl From<PixelType> for PrimType {
  fn from(px: PixelType) -> Self {
    match px {
      PixelType::U8 => PrimType::U8,
      PixelType::U16 => PrimType::U16,
    }
  }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SimdType(PrimType, usize);
impl SimdType {
  fn new(prim: PrimType, width: usize) -> Self {
    SimdType(prim, width)
  }
  fn assume_type(&self, value: TokenStream) -> SimdValue {
    SimdValue(self.clone(), value)
  }
  fn ty(&self) -> PrimType {
    self.0
  }
  fn w(&self) -> usize {
    self.1
  }
  fn is_signed(&self) -> bool {
    self.ty().is_signed()
  }

  fn bit_width(&self) -> usize {
    self.ty().bits() * self.w()
  }

  fn uload(&self, ptr: &TokenStream) -> SimdValue {
    SimdValue::uload(*self, ptr)
  }
  fn aload(&self, ptr: &TokenStream) -> SimdValue {
    SimdValue::aload(*self, ptr)
  }

  fn splat<T>(&self, v: T) -> SimdValue
  where
    T: ToTokens,
  {
    let ety = self.ty();
    SimdValue::from(*self, quote!(<#self>::splat(#v as #ety)))
  }

  fn indices_ty(&self) -> SimdType {
    SimdType(PrimType::U32, self.w())
  }
  fn indices<F>(&self, mut f: F) -> SimdValue
  where
    F: FnMut(usize) -> TokenStream,
  {
    let mut idxs = TokenStream::default();
    for i in 0..self.w() {
      let v = f(i);
      idxs.extend(quote! { #v, });
    }
    let ty = self.indices_ty();
    SimdValue::from(ty, quote! { <#ty>::new(#idxs) })
  }
}
impl ToTokens for SimdType {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    let ty = &self.0;
    let width = &self.1;
    tokens.extend(quote!(Simd<[#ty; #width]>));
  }
}
#[derive(Clone, Debug)]
struct SimdValue(SimdType, TokenStream);
impl SimdValue {
  fn from<T>(ty: SimdType, v: T) -> Self
  where
    T: ToTokens,
  {
    SimdValue(ty, v.into_token_stream())
  }
  fn default(ty: SimdType) -> Self {
    Self::from(ty, quote! { <#ty>::default() })
  }
  fn from_cast<T>(ty: SimdType, v: T) -> Self
  where
    T: ToTokens,
  {
    let v = quote!(<#ty>::from_cast(#v));
    SimdValue(ty, v)
  }
  fn uload<T>(ty: SimdType, ptr: T) -> Self
  where
    T: ToTokens,
  {
    let w = ty.w();
    let ety = ty.ty();
    let v = quote! {
      <#ty>::from(*(#ptr as *const [#ety; #w]))
    };
    SimdValue(ty, v)
  }
  fn aload<T>(ty: SimdType, ptr: T) -> Self
  where
    T: ToTokens,
  {
    let v = quote! {
      *(#ptr as *const #ty)
    };
    SimdValue(ty, v)
  }
  fn ustore<T>(&self, dst: &mut TokenStream, ptr: T)
  where
    T: ToTokens,
  {
    let w = self.ty().w();
    let ety = self.ty().ty();
    let v = quote! {
      *(#ptr as *mut [#ety; #w]) = ::std::mem::transmute::<_, [#ety; #w]>(#self);
    };
    dst.extend(v);
  }
  fn astore<T>(&self, dst: &mut TokenStream, ptr: T)
  where
    T: ToTokens,
  {
    let ty = self.ty();
    let v = quote! {
      *(#ptr as *mut #ty) = #self;
    };
    dst.extend(v);
  }
  fn ty(&self) -> SimdType {
    self.0
  }
  fn w(&self) -> usize {
    self.ty().w()
  }
  fn value(&self) -> &TokenStream {
    &self.1
  }
  fn unwrap_value(self) -> TokenStream {
    self.1
  }

  fn bitcast(&self, to: SimdType) -> Self {
    assert_eq!(self.ty().bit_width(), to.bit_width());
    let v = quote!(<#to>::from_bits(#self));
    SimdValue(to, v)
  }
  fn cast(&self, to: SimdType) -> Self {
    assert_eq!(self.ty().w(), to.w());
    let v = quote!(<#to>::from_cast(#self));
    SimdValue(to, v)
  }

  fn abs(&self) -> Self {
    SimdValue(self.ty(), quote!(#self.abs()))
  }
  fn clamp<T, U>(&self, min: T, max: U) -> Self
  where
    T: ToTokens,
    U: ToTokens,
  {
    let elem = self.ty().ty();
    let v = quote!(#self.clamp(#min as #elem, #max as #elem));
    SimdValue::from(self.ty(), v)
  }
  fn round_shift<T>(&self, bit: T) -> Self
  where
    T: ToTokens,
  {
    let v = quote!(#self.round_shift(#bit));
    SimdValue::from(self.ty(), v)
  }
  fn min(&self, rhs: &Self) -> Self {
    let v = quote!(#self.min(#rhs));
    SimdValue::from(self.ty(), v)
  }
  fn max(&self, rhs: &Self) -> Self {
    let v = quote!(#self.max(#rhs));
    SimdValue::from(self.ty(), v)
  }
  fn butterfly(&self, rhs: &Self) -> (Self, Self) {
    (self + rhs, self - rhs)
  }

  /// Take `self` and extend its width to `to_width` elements, filling
  /// with `v` (a value of the vector primitive).
  fn extend<T>(&self, to_width: usize, v: T) -> SimdValue
  where
    T: ToTokens,
  {
    assert_ne!(to_width, 0);

    let len = self.w();
    if to_width == len {
      return self.clone();
    }

    let rhs = self.ty().splat(v);
    let mut idxs = Vec::with_capacity(to_width);
    for i in 0..to_width {
      if i <= len {
        idxs.push(quote!(#i));
      } else {
        idxs.push(quote!(#len));
      }
    }

    Self::shuffle2(self, &rhs, &idxs)
  }

  /// We can't use shuffle!() in the build script. Instead we must craft
  /// a token stream which includes `shuffle!(..)`
  fn shuffle2<T>(l: &SimdValue, r: &SimdValue, idx: &[T]) -> Self
  where
    T: ToTokens,
  {
    assert_eq!(l.ty(), r.ty());

    let mut idxs = TokenStream::default();
    let idx_len = idx.len();
    for (ii, i) in idx.iter().enumerate() {
      if ii + 1 == idx_len {
        // trailing comma not allowed *facepalm*
        idxs.extend(quote! { #i });
      } else {
        idxs.extend(quote! { #i, });
      }
    }

    let mut ts = macro_("shuffle");
    ts.extend(quote! {
     { #l, #r, [#idxs] }
    });

    let ty = SimdType::new(l.ty().ty(), idx.len());
    SimdValue::from(ty, ts)
  }
  fn shuffle<T>(&self, idx: &[T]) -> Self
  where
    T: ToTokens,
  {
    let mut idxs = TokenStream::default();
    let idx_len = idx.len();
    for (ii, i) in idx.iter().enumerate() {
      if ii + 1 == idx_len {
        // trailing comma not allowed *facepalm*
        idxs.extend(quote! { #i });
      } else {
        idxs.extend(quote! { #i, });
      }
    }

    let mut ts = macro_("shuffle");
    ts.extend(quote! {
     { #self, [#idxs] }
    });

    let ty = SimdType::new(self.ty().ty(), idx.len());
    SimdValue::from(ty, ts)
  }
  fn select_range(&self, range: Range<usize>) -> Self {
    //assert!(self.w() >= range.end, "{} < {}", self.w(), range.end);
    let idxs = range.map(|i| i as u32).collect::<Vec<_>>();
    self.shuffle(&idxs)
  }
  fn concat(&self, rhs: &SimdValue) -> Self {
    let new_len = (self.w() + rhs.w()) as u32;
    let idxs = (0u32..new_len).collect::<Vec<_>>();

    Self::shuffle2(self, rhs, &idxs)
  }
  fn let_<T>(&self, dest: &mut TokenStream, name: T) -> Self
  where
    T: Display,
  {
    let name = Ident::new(&name.to_string(), Span::call_site());
    let ty = self.ty();
    dest.extend(quote! {
      let #name: #ty = #self;
    });
    SimdValue::from(ty, quote!(#name))
  }
  fn let_mut<T>(&self, dest: &mut TokenStream, name: T) -> Self
  where
    T: Display,
  {
    let name = Ident::new(&name.to_string(), Span::call_site());
    let ty = self.ty();
    dest.extend(quote! {
      let mut #name: #ty = #self;
    });
    SimdValue::from(ty, quote!(#name))
  }

  #[allow(dead_code)]
  fn debug<T>(&self, dst: &mut TokenStream, name: T)
  where
    T: Display,
  {
    let fmt = format!("{}: {{:?}}", name);
    dst.extend(call_macro(
      "println",
      quote! {{
        #fmt, #self,
      }},
    ));
  }
}
impl<'a> Add<&'a SimdValue> for &'a SimdValue {
  type Output = SimdValue;
  fn add(self, rhs: &'a SimdValue) -> SimdValue {
    assert_eq!(self.ty(), rhs.ty());
    let ty = self.ty();
    let v = quote! { (#self + #rhs) };
    SimdValue::from(ty, v)
  }
}
impl<'a> Sub<&'a SimdValue> for &'a SimdValue {
  type Output = SimdValue;
  fn sub(self, rhs: &'a SimdValue) -> SimdValue {
    assert_eq!(self.ty(), rhs.ty());
    let ty = self.ty();
    let v = quote! { (#self - #rhs) };
    SimdValue::from(ty, v)
  }
}
impl<'a> Mul<&'a SimdValue> for &'a SimdValue {
  type Output = SimdValue;
  fn mul(self, rhs: &'a SimdValue) -> SimdValue {
    assert_eq!(self.ty(), rhs.ty());
    let ty = self.ty();
    let v = quote! { (#self * #rhs) };
    SimdValue::from(ty, v)
  }
}
impl<'a> ops::Neg for &'a SimdValue {
  type Output = SimdValue;
  fn neg(self) -> SimdValue {
    assert!(self.ty().is_signed());
    let v = quote!((-#self));
    SimdValue::from(self.ty(), v)
  }
}
impl<'a> Add<&'a SimdValue> for SimdValue {
  type Output = SimdValue;
  fn add(self, rhs: &'a SimdValue) -> SimdValue {
    &self + rhs
  }
}
impl<'a> Sub<&'a SimdValue> for SimdValue {
  type Output = SimdValue;
  fn sub(self, rhs: &'a SimdValue) -> SimdValue {
    &self - rhs
  }
}
impl<'a> Mul<&'a SimdValue> for SimdValue {
  type Output = SimdValue;
  fn mul(self, rhs: &'a SimdValue) -> SimdValue {
    &self * rhs
  }
}
impl Add<SimdValue> for SimdValue {
  type Output = SimdValue;
  fn add(self, rhs: SimdValue) -> SimdValue {
    &self + &rhs
  }
}
impl Sub<SimdValue> for SimdValue {
  type Output = SimdValue;
  fn sub(self, rhs: SimdValue) -> SimdValue {
    &self - &rhs
  }
}
impl Mul<SimdValue> for SimdValue {
  type Output = SimdValue;
  fn mul(self, rhs: SimdValue) -> SimdValue {
    &self * &rhs
  }
}
impl ops::Neg for SimdValue {
  type Output = SimdValue;
  fn neg(self) -> SimdValue {
    -(&self)
  }
}
impl ToTokens for SimdValue {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    self.1.to_tokens(tokens);
  }
}

struct Var<T> {
  name: Ident,
  mutable: bool,
  value: T,
}
impl<T> Var<T>
where
  T: ToTokens,
{
  fn new<U>(name: U, value: T) -> Self
  where
    U: Display,
  {
    let name = Ident::new(&name.to_string(), Span::call_site());
    Self::new_ident(name, false, value)
  }
  fn new_mut<U>(name: U, value: T) -> Self
  where
    U: Display,
  {
    let name = Ident::new(&name.to_string(), Span::call_site());
    Self::new_ident(name, true, value)
  }
  fn new_ident(name: Ident, mutable: bool, value: T) -> Self {
    Var { name, mutable, value }
  }

  fn let_<U>(dest: &mut TokenStream, name: U, value: T) -> Ident
  where
    U: Display,
    T: ToTokens,
  {
    let name = Ident::new(&name.to_string(), Span::call_site());
    dest.extend(quote! {
      let #name = #value;
    });
    name
  }
  fn let_mut<U>(dest: &mut TokenStream, name: U, value: T) -> Ident
  where
    U: Display,
    T: ToTokens,
  {
    let name = Ident::new(&name.to_string(), Span::call_site());
    dest.extend(quote! {
      let mut #name = #value;
    });
    name
  }

  fn add_assign(&self, dst: &mut TokenStream, v: TokenStream) {
    assert!(self.mutable);
    let name = &self.name;
    dst.extend(quote! {
      #name += #v;
    });
  }
  fn assign<U>(&self, dst: &mut TokenStream, v: U)
  where
    U: ToTokens,
  {
    assert!(self.mutable);
    let name = &self.name;
    dst.extend(quote! {
      #name = #v;
    });
  }

  fn name(&self) -> &Ident {
    &self.name
  }
  fn value(&self) -> &T {
    &self.value
  }
}
impl<T> ToTokens for Var<T>
where
  T: ToTokens,
{
  fn to_tokens(&self, tokens: &mut TokenStream) {
    let name = &self.name;
    let value = &self.value;
    if !self.mutable {
      tokens.extend(quote! {
        let #name = #value;
      });
    } else {
      tokens.extend(quote! {
        let mut #name = #value;
      });
    }
  }
}
impl<T> Deref for Var<T>
where
  T: ToTokens,
{
  type Target = Ident;
  fn deref(&self) -> &Ident {
    &self.name
  }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IsaFeature {
  Native,

  // x86
  Sse2,
  Sse3,
  Ssse3,
  Sse4_1,
  Sse4_2,
  Avx,
  Avx2,
  Avx512,

  // ARM
  Neon,
}
impl IsaFeature {
  fn fn_suffix(&self) -> &'static str {
    match self {
      IsaFeature::Native => "native",
      IsaFeature::Sse2 => "sse2",
      IsaFeature::Sse3 => "sse3",
      IsaFeature::Ssse3 => "ssse3",
      IsaFeature::Sse4_1 => "sse4_1",
      IsaFeature::Sse4_2 => "sse4_2",
      IsaFeature::Avx => "avx",
      IsaFeature::Avx2 => "avx2",
      IsaFeature::Avx512 => "avx512",
      IsaFeature::Neon => "neon",
    }
  }
  fn target_feature(&self) -> &'static str {
    match self {
      IsaFeature::Native => {
        panic!("IsaFeature::Native isn't a target feature")
      }
      IsaFeature::Sse2 => "sse2",
      IsaFeature::Sse3 => "sse3",
      IsaFeature::Ssse3 => "ssse3",
      IsaFeature::Sse4_1 => "sse4.1",
      IsaFeature::Sse4_2 => "sse4.2",
      IsaFeature::Avx => "avx",
      IsaFeature::Avx2 => "avx2",
      IsaFeature::Avx512 => "avx512",
      IsaFeature::Neon => "neon",
    }
  }

  fn module_name(&self) -> Ident {
    let name = self.fn_suffix();
    let name = format!("i_{}", name);
    Ident::new(&name, Span::call_site())
  }

  fn index(&self) -> usize {
    match self {
      IsaFeature::Native => 0,

      IsaFeature::Sse2 => 1,
      IsaFeature::Ssse3 => 2,
      IsaFeature::Avx2 => 3,

      IsaFeature::Neon => panic!("TODO: fix rav1e for ARM"),
      _ => panic!("{:?} has no index", self),
    }
  }

  fn sets() -> Vec<IsaFeature> {
    let mut out = vec![IsaFeature::Native];
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    if arch == "x86_64" || arch == "x86" {
      out.push(IsaFeature::Sse2);
      out.push(IsaFeature::Ssse3);
      out.push(IsaFeature::Avx2);
    } else if arch.contains("arm") {
      // TODO need to add requisite code to rav1e for ARM
      //out.push(IsaFeature::Neon);
    }

    out
  }
}

impl ToTokens for IsaFeature {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    if let IsaFeature::Native = self {
      return;
    }

    let s = self.target_feature();
    tokens.extend(quote! {
      #[target_feature(enable = #s)]
    });
  }
}

use self::NegIdx::*;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NegIdx {
  Neg(usize),
  Pos(usize),
}
impl NegIdx {
  fn i(&self) -> usize {
    match self {
      Neg(i) | Pos(i) => *i,
    }
  }
  fn is_neg(&self) -> bool {
    match self {
      Neg(_) => true,
      _ => false,
    }
  }
}
impl From<usize> for NegIdx {
  fn from(v: usize) -> NegIdx {
    Pos(v)
  }
}
impl ops::Neg for NegIdx {
  type Output = NegIdx;
  fn neg(self) -> NegIdx {
    match self {
      Pos(i) => Neg(i),
      Neg(i) => Pos(i),
    }
  }
}

trait Array {
  fn ty(&self) -> SimdType;
  fn len(&self) -> usize;
  fn get(&self, idx: usize) -> SimdValue;
  fn get_neg(&self, idx: NegIdx) -> SimdValue {
    let v = self.get(idx.i());
    if idx.is_neg() {
      v.neg()
    } else {
      v
    }
  }
  fn set(&self, dst: &mut TokenStream, idx: usize, v: SimdValue);

  fn map<F>(self, f: F) -> ArrayMap<Self, F>
  where
    Self: Sized,
    F: Fn(usize, SimdValue) -> SimdValue,
  {
    let ty = if self.len() > 0 {
      // meh.
      f(0, self.get(0)).ty()
    } else {
      // in this case the type won't matter
      self.ty()
    };
    ArrayMap { ty, array: self, map: f }
  }
  fn trunc(self, len: usize) -> ArrayTrunc<Self>
  where
    Self: Sized,
  {
    assert!(self.len() >= len);
    ArrayTrunc { len, array: self }
  }

  fn assign(&self, dst: &mut TokenStream, from: &dyn Array) {
    assert_eq!(self.ty(), from.ty());
    assert_eq!(self.len(), from.len());

    for i in 0..self.len() {
      let v = from.get(i);
      self.set(dst, i, v);
    }
  }

  #[allow(dead_code)]
  fn debug(&self, dst: &mut TokenStream, name: &dyn Display) {
    for i in 0..self.len() {
      let item = self.get(i);
      let fmt = format!("{}[{}]: {{:?}}", name, i);
      dst.extend(call_macro(
        "println",
        quote! {{
          #fmt, #item,
        }},
      ));
    }
  }
  #[allow(dead_code)]
  fn debug_transpose(&self, dst: &mut TokenStream, name: &dyn Display) {
    for r in 0..self.ty().w() {
      let mut args = TokenStream::default();
      let mut fmt = format!("{}[{}]: [", name, r);

      for i in 0..self.len() {
        let item = self.get(i);

        args.extend(quote! {
          #item.extract_unchecked(#r),
        });
        fmt.push_str("{:?}");
        if i != self.len() - 1 {
          fmt.push_str(", ");
        }
      }
      fmt.push_str("]");
      dst.extend(call_macro(
        "println",
        quote! {{
          #fmt, #args
        }},
      ));
    }
  }
}
impl<'a, T> Array for &'a T
where
  T: Array + ?Sized,
{
  fn ty(&self) -> SimdType {
    (&**self).ty()
  }
  fn len(&self) -> usize {
    (&**self).len()
  }
  fn get(&self, idx: usize) -> SimdValue {
    (&**self).get(idx)
  }
  fn set(&self, dst: &mut TokenStream, idx: usize, v: SimdValue) {
    (&**self).set(dst, idx, v)
  }
}
impl Array for Box<dyn Array> {
  fn ty(&self) -> SimdType {
    (&**self).ty()
  }
  fn len(&self) -> usize {
    (&**self).len()
  }
  fn get(&self, idx: usize) -> SimdValue {
    (&**self).get(idx)
  }
  fn set(&self, dst: &mut TokenStream, idx: usize, v: SimdValue) {
    (&**self).set(dst, idx, v)
  }
}

struct Slice<T = Ident>(SimdType, T, usize)
where
  T: ToTokens;
impl<T> Slice<T>
where
  T: ToTokens,
{
  fn from_ptr(ty: SimdType, name: T, len: usize) -> Self {
    Slice(ty, name, len)
  }
}
impl<T> Array for Slice<T>
where
  T: ToTokens,
{
  fn ty(&self) -> SimdType {
    self.0
  }
  fn len(&self) -> usize {
    self.2
  }
  fn get(&self, idx: usize) -> SimdValue {
    assert!(self.len() > idx);
    let ty = self.ty();
    SimdValue::from(self.0, quote! { (*(#self as *const #ty).add(#idx)) })
  }
  fn set(&self, dst: &mut TokenStream, idx: usize, v: SimdValue) {
    assert!(self.len() > idx);
    assert_eq!(self.0, v.ty());
    let ty = self.ty();
    dst.extend(quote! {
      *(#self as *mut #ty).add(#idx) = #v;
    });
  }
}
impl<T> ToTokens for Slice<T>
where
  T: ToTokens,
{
  fn to_tokens(&self, to: &mut TokenStream) {
    self.1.to_tokens(to);
  }
}

struct Vector(SimdType, Vec<SimdValue>);
impl Vector {
  fn new(ty: SimdType, v: Vec<SimdValue>) -> Self {
    Vector(ty, v)
  }
}
impl Array for Vector {
  fn ty(&self) -> SimdType {
    self.0
  }
  fn len(&self) -> usize {
    self.1.len()
  }
  fn get(&self, idx: usize) -> SimdValue {
    self.1[idx].clone()
  }
  fn set(&self, _dst: &mut TokenStream, _idx: usize, _v: SimdValue) {
    panic!("Vector is read only");
  }
}

/// An "array" of variables. Really just a indexed set of variables.
#[derive(Clone)]
struct VarArr {
  ty: SimdType,
  prefix: String,
  names: Vec<TokenStream>,
}
impl VarArr {
  fn let_<T>(dst: &mut TokenStream, prefix: T, slice: &dyn Array) -> Self
  where
    T: Display,
  {
    let names = (0..slice.len())
      .map(|i| {
        slice.get(i).let_(dst, format_args!("{}_{}", prefix, i)).unwrap_value()
      })
      .collect();
    VarArr { ty: slice.ty(), prefix: prefix.to_string(), names }
  }
  fn let_mut<T>(dst: &mut TokenStream, prefix: T, slice: &dyn Array) -> Self
  where
    T: Display,
  {
    let names = (0..slice.len())
      .map(|i| {
        slice
          .get(i)
          .let_mut(dst, format_args!("{}_{}", prefix, i))
          .unwrap_value()
      })
      .collect();
    VarArr { ty: slice.ty(), prefix: prefix.to_string(), names }
  }
}
impl Array for VarArr {
  fn ty(&self) -> SimdType {
    self.ty
  }
  fn len(&self) -> usize {
    self.names.len()
  }
  fn get(&self, idx: usize) -> SimdValue {
    assert!(self.len() > idx);
    SimdValue::from(self.ty(), &self.names[idx])
  }
  fn set(&self, dst: &mut TokenStream, idx: usize, v: SimdValue) {
    assert!(self.len() > idx);
    assert_eq!(self.ty(), v.ty());
    let name = &self.names[idx];
    dst.extend(quote! {
      #name = #v;
    });
  }
}
struct ArrayTrunc<T>
where
  T: Array,
{
  len: usize,
  array: T,
}
impl<T> Array for ArrayTrunc<T>
where
  T: Array,
{
  fn ty(&self) -> SimdType {
    self.array.ty()
  }
  fn len(&self) -> usize {
    self.len
  }
  fn get(&self, idx: usize) -> SimdValue {
    assert!(self.len() > idx);
    self.array.get(idx)
  }
  fn set(&self, dst: &mut TokenStream, idx: usize, v: SimdValue) {
    assert!(self.len() > idx);
    self.array.set(dst, idx, v);
  }
}

struct ArrayMap<T, F>
where
  T: Array,
  F: Fn(usize, SimdValue) -> SimdValue,
{
  ty: SimdType,
  array: T,
  map: F,
}
impl<T, F> ArrayMap<T, F>
where
  T: Array,
  F: Fn(usize, SimdValue) -> SimdValue,
{
}
impl<T, F> Array for ArrayMap<T, F>
where
  T: Array,
  F: Fn(usize, SimdValue) -> SimdValue,
{
  fn ty(&self) -> SimdType {
    self.ty
  }
  fn len(&self) -> usize {
    self.array.len()
  }
  fn get(&self, idx: usize) -> SimdValue {
    assert!(self.len() > idx);
    (&self.map)(idx, self.array.get(idx))
  }
  fn set(&self, dst: &mut TokenStream, idx: usize, v: SimdValue) {
    // Yes, this is a bit weird, but it is useful for modifying
    // the value to set.
    assert!(self.len() > idx);
    let v = (&self.map)(idx, v);
    self.array.set(dst, idx, v);
  }
}

/// An output array which transposes the indices so that we write a transposed
/// block. Since we work on vector units, this uses scatters, which are theoretically
/// fire-and-forget-able, assuming no reads thereafter.
/// This is used only by row transform when writing the row transform outputs.
struct TransposedWriteArray {
  /// The simd type written
  ty: SimdType,
  ptr: TokenStream,
  block: Block,
}
impl TransposedWriteArray {
  fn new(ty: SimdType, ptr: TokenStream, block: Block) -> Self {
    TransposedWriteArray { ty, ptr, block }
  }
}
impl Array for TransposedWriteArray {
  fn ty(&self) -> SimdType {
    self.ty
  }
  fn len(&self) -> usize {
    self.block.h()
  }
  fn get(&self, _idx: usize) -> SimdValue {
    // we could actually allow reads. But you really shouldn't
    // be reading from this in practice.
    panic!("TransposedWriteArray is write only")
  }
  fn set(&self, dst: &mut TokenStream, idx: usize, v: SimdValue) {
    assert_eq!(self.ty(), v.ty());
    assert!(self.len() > idx);

    // ugh vector widths are so pepega.

    const MAX_PTR_LANES: usize = 8;

    let stride = self.block.h();

    let lanes = self.ty().w().min(MAX_PTR_LANES);
    let ety = self.ty().ty();
    let scatter_ty = SimdType::new(ety, lanes);

    let ptr = &self.ptr;
    let ptr = quote!(#ptr.add(#idx));
    dst.extend(quote! {
      let scatter_ptr_base = <Simd<[*mut #ety; #lanes]>>::splat(#ptr);
    });
    dst.extend(quote! {
      let scatter_all_mask = <Simd<[m8; #lanes]>>::splat(true);
    });

    // construct a scatter write:
    for l in (0..self.ty().w()).step_by(lanes) {
      let offsets = scatter_ty.indices(|idx| {
        let idx = (idx * stride + l) as u32;
        quote!(#idx)
      });
      let offsets = offsets.let_(dst, "scatter_offsets");
      let ptrs = quote! {
        scatter_ptr_base.add(#offsets.cast())
      };
      let v = if self.ty().w() > MAX_PTR_LANES {
        v.select_range(l..l + lanes).let_(dst, "scatter_value_part")
      } else {
        v.clone()
      };
      dst.extend(quote! {
        #ptrs.write(scatter_all_mask, #v);
      });
    }
  }
}

struct Module {
  parents: Vec<Ident>,
  filename: Option<PathBuf>,
  name: Ident,
  tt: TokenStream,
}
impl Module {
  fn new_root(name: Ident) -> Self {
    Module {
      parents: Vec::new(),
      filename: None,
      name,
      tt: Default::default(),
    }
  }
  fn finish_root(self, out: &mut dyn Write) {
    assert_eq!(self.parents.len(), 0, "don't use this for children");
    let this = quote!(#self);
    writeln!(out, "{}", this).expect("failed to write module partition");
  }

  fn new_child(&self, name: Ident) -> Module {
    let mut parents = self.parents.clone();
    parents.push(self.name.clone());
    Module { parents, filename: None, name, tt: Default::default() }
  }
  fn new_child_file(&self, file_prefix: &str, name: Ident) -> Module {
    let mut path = file_prefix.to_owned();
    for parent in self.parents.iter() {
      path.push_str(&format!("_{}", parent));
    }
    path.push_str(&format!("_{}_{}_kernels.rs", self.name, name));

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir);

    let mut parents = self.parents.clone();
    parents.push(self.name.clone());
    Module {
      parents,
      filename: Some(out_dir.join(path)),
      name,
      tt: Default::default(),
    }
  }

  fn item_path(&self, item: &Ident) -> TokenStream {
    let mut t = quote!(self);
    for parent in self.parents.iter() {
      t = quote!(#t::#parent);
    }

    let this = &self.name;
    quote!(#t::#this::#item)
  }
  fn finish_child(mut self, parent: &mut Module) {
    assert_ne!(self.parents.len(), 0, "don't use this for the root");
    assert_eq!(self.parents.last().unwrap(), &parent.name, "parent mismatch");

    parent.tt.extend(quote!(#self));

    // write to the target path:
    if let Some(path) = self.filename.take() {
      gen::write_kernel(path, |f| {
        writeln!(f, "{}", self.tt).expect("write kernel submodule file");
      });
    }
  }
}

impl Deref for Module {
  type Target = TokenStream;
  fn deref(&self) -> &TokenStream {
    &self.tt
  }
}
impl DerefMut for Module {
  fn deref_mut(&mut self) -> &mut TokenStream {
    &mut self.tt
  }
}
impl ToTokens for Module {
  fn to_tokens(&self, to: &mut TokenStream) {
    let name = &self.name;
    let inner = &self.tt;
    if let Some(ref fp) = self.filename {
      let path = format!("{}", fp.display());
      to.extend(quote! {
        #[path = #path]
        pub mod #name;
      });
    } else {
      to.extend(quote! {
        pub mod #name {
          #inner
        }
      });
    }
  }
}

#[derive(Clone, Debug)]
struct KernelSet {
  fn_ty: (Ident, TokenStream),
  kname: String,
  u8_ks: Vec<(TokenStream, TokenStream)>,
  u16_ks: Vec<(TokenStream, TokenStream)>,
  /// `BLOCK_SIZES_ALL` and `CpuFeatureLevel::len()` are implicit.
  table_array_sizes: Vec<TokenStream>,
}
impl KernelSet {
  fn new(
    fn_ty_name: &str, fn_args: &[TokenStream], fn_ret: Option<TokenStream>,
    kname: &str, table_array_sizes: Vec<TokenStream>,
  ) -> Self {
    let fn_ty_name = Ident::new(fn_ty_name, Span::call_site());

    let mut fn_ty_args = TokenStream::default();
    for arg in fn_args.iter() {
      fn_ty_args.extend(quote! {
        #arg,
      });
    }

    let fn_ty = quote!(unsafe fn(#fn_ty_args) #fn_ret );

    KernelSet {
      fn_ty: (fn_ty_name, fn_ty),
      kname: kname.into(),
      u8_ks: Vec::new(),
      u16_ks: Vec::new(),
      table_array_sizes,
    }
  }
  fn push_kernel(
    &mut self, px: PixelType, idx: TokenStream, path: TokenStream,
  ) {
    let ks = match px {
      PixelType::U8 => &mut self.u8_ks,
      PixelType::U16 => &mut self.u16_ks,
    };
    ks.push((idx, path));
  }

  fn table_ty(&self, px: PixelType) -> TokenStream {
    let fn_ty_name = &self.fn_ty.0;
    let mut table_ty = quote!(Option<#fn_ty_name<#px>>);
    for size in self.table_array_sizes.iter() {
      table_ty = quote!([#table_ty; #size]);
    }

    quote!([[#table_ty; BlockSize::BLOCK_SIZES_ALL]; CpuFeatureLevel::len()])
  }
  fn table_default(&self) -> TokenStream {
    let mut table = quote!(None);
    for size in self.table_array_sizes.iter() {
      table = quote!([#table; #size]);
    }

    quote!([[#table; BlockSize::BLOCK_SIZES_ALL]; CpuFeatureLevel::len()])
  }
  fn table_init(ks: &[(TokenStream, TokenStream)]) -> TokenStream {
    let mut table_init = TokenStream::default();
    for &(ref idx, ref path) in ks.iter() {
      table_init.extend(quote! {
        out #idx = Some(#path as _);
      });
    }

    table_init
  }

  fn tables(self) -> TokenStream {
    let fn_ty_name = &self.fn_ty.0;
    let fn_ty = &self.fn_ty.1;

    let u8_table_name = format!("U8_{}_KERNELS", self.kname);
    let u8_table_name = Ident::new(&u8_table_name, Span::call_site());
    let u16_table_name = format!("U16_{}_KERNELS", self.kname);
    let u16_table_name = Ident::new(&u16_table_name, Span::call_site());

    let u8_table_ty = self.table_ty(PixelType::U8);
    let u16_table_ty = self.table_ty(PixelType::U16);

    let u8_table_init = Self::table_init(&self.u8_ks);
    let u16_table_init = Self::table_init(&self.u16_ks);

    let table_default = self.table_default();

    quote! {
      type #fn_ty_name<T> = #fn_ty;

      pub(super) static #u8_table_name: #u8_table_ty = {
        let mut out: #u8_table_ty = #table_default;
        #u8_table_init
        out
      };
      pub(super) static #u16_table_name: #u16_table_ty = {
        let mut out: #u16_table_ty = #table_default;
        #u16_table_init
        out
      };
    }
  }

  fn len(&self) -> usize {
    self.u8_ks.len() + self.u16_ks.len()
  }
}
