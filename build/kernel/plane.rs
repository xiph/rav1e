use std::fmt::Display;

use proc_macro2::{Ident, Span, TokenStream};
use quote::*;

/// A (ptr, row stride) pair. Typically passed as separate arguments,
/// both mutable as values.
#[derive(Clone, Debug)]
pub struct Plane {
  ptr: Ident,
  stride: TokenStream,
}
impl Plane {
  pub fn new(ptr_name: &Ident) -> Self {
    let stride_name = format!("{}_stride", ptr_name);
    let stride_name = Ident::new(&stride_name, Span::call_site());
    Plane { ptr: ptr_name.clone(), stride: quote!(#stride_name) }
  }
  pub fn new_stride<T>(ptr: &Ident, stride: T) -> Self
  where
    T: Display,
  {
    let stride_name = Ident::new(&stride.to_string(), Span::call_site());
    Plane { ptr: ptr.clone(), stride: quote!(#stride_name) }
  }
  pub fn new_const_stride(ptr: Ident, stride: usize) -> Self {
    Plane { ptr, stride: quote!(#stride) }
  }

  pub fn stride(&self) -> &TokenStream {
    &self.stride
  }

  pub fn add_rc<T, U>(&self, row: T, col: U) -> TokenStream
  where
    T: ToTokens,
    U: ToTokens,
  {
    let ptr = &self.ptr;
    let stride = &self.stride;
    quote! {
      #ptr.add((#row) * #stride + #col)
    }
  }
  pub fn add<T>(&self, col: T) -> TokenStream
  where
    T: ToTokens,
  {
    let ptr = &self.ptr;
    quote! { #ptr.add(#col) }
  }

  pub fn next_row(&self, into: &mut TokenStream) {
    self.next_nth(1, into)
  }
  pub fn next_nth(&self, n: usize, into: &mut TokenStream) {
    let ptr = &self.ptr;
    let stride = &self.stride;
    into.extend(quote! {
      #ptr = #ptr.add(#n * (#stride as usize));
    });
  }
}
impl ToTokens for Plane {
  fn to_tokens(&self, to: &mut TokenStream) {
    self.ptr.to_tokens(to);
  }
}
