#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

#[cfg_attr(feature = "cargo-clippy", allow(const_static_lifetime))]
#[cfg_attr(feature = "cargo-clippy", allow(unreadable_literal))]

pub mod av {
  include!(concat!(env!("OUT_DIR"), "/av.rs"));
}

pub use av::*;

#[cfg(test)]
mod tests {
  use super::av::*;
  use std::ffi::CStr;
  use std::mem;
  #[test]
  fn config() {
    println!("{}", unsafe {
      CStr::from_ptr(avformat_configuration()).to_string_lossy()
    });
  }
}
