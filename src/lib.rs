// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(safe_extern_statics)]
#![deny(bare_trait_objects)]

#[macro_use]
extern crate serde_derive;
extern crate bincode;

#[cfg(all(test, feature="decode_test_dav1d"))]
extern crate dav1d_sys;

#[cfg(test)]
extern crate interpolate_name;

#[cfg(test)]
#[macro_use]
extern crate pretty_assertions;

pub mod ec;
pub mod partition;
pub mod plane;
pub mod transform;
pub mod quantize;
pub mod predict;
pub mod rdo;
pub mod rdo_tables;
#[macro_use]
pub mod util;
pub mod context;
pub mod entropymode;
pub mod token_cdfs;
pub mod deblock;
pub mod segmentation;
pub mod cdef;
pub mod lrf;
pub mod encoder;
pub mod mc;
pub mod me;
pub mod metrics;
pub mod scan_order;
pub mod scenechange;
pub mod rate;
pub mod tiling;

mod api;
mod header;
mod frame;

pub use crate::api::*;
pub use crate::encoder::*;
pub use crate::header::*;
pub use crate::util::{CastFromPrimitive, Pixel};

pub use crate::frame::Frame;

/// Version information
///
/// The information is recovered from `Cargo.toml` and `git describe`, when available.
///
/// ```
/// use rav1e::version;
/// use semver::Version;
///
/// let major = version::major();
/// let minor = version::minor();
/// let patch = version::patch();
///
/// let short = version::short();
///
/// let v1 = Version::new(major, minor, patch);
/// let v2 = Version::parse(&short).unwrap();
///
/// assert_eq!(v1, v2);
///```
///
pub mod version {
  /// Major version component
  ///
  /// It is increased every time a release presents a incompatible API change.
  pub fn major() -> u64 {
    env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap()
  }
  /// Minor version component
  ///
  /// It is increased every time a release presents new functionalities are added
  /// in a backwards-compatible manner.
  pub fn minor() -> u64 {
    env!("CARGO_PKG_VERSION_MINOR").parse().unwrap()
  }
  /// Patch version component
  ///
  /// It is increased every time a release provides only backwards-compatible bugfixes.
  pub fn patch() -> u64 {
    env!("CARGO_PKG_VERSION_PATCH").parse().unwrap()
  }

  /// Version information as presented in `[package]` `version`.
  ///
  /// e.g. `0.1.0``
  ///
  /// Can be parsed by [semver](https://crates.io/crates/semver).
  pub fn short() -> String {
    env!("CARGO_PKG_VERSION").to_string()
  }

  /// Version information as presented in `[package] version` followed by the
  /// short commit hash if present.
  ///
  /// e.g. `0.1.0 - g743d464`
  ///
  pub fn long() -> String {
    let s = short();
    let hash = hash();

    if hash.is_empty() {
      s
    } else {
      format!("{} - {}", s, hash)
    }
  }

  /// Commit hash (short)
  ///
  /// Short hash of the git commit used by this build
  ///
  /// e.g. `g743d464`
  ///
  pub fn hash() -> String {
    env!("VERGEN_SHA_SHORT").to_string()
  }

  /// Version information with the information
  /// provided by `git describe --tags`.
  ///
  /// e.g. `0.1.0 (v0.1.0-1-g743d464)`
  ///
  pub fn full() -> String {
    let semver = env!("VERGEN_SEMVER_LIGHTWEIGHT");
    format!("{} ({})", short(), semver)
  }
}
#[cfg(all(test, any(feature="decode_test", feature="decode_test_dav1d")))]
mod test_encode_decode;

#[cfg(all(test, feature="decode_test"))]
mod test_encode_decode_aom;

#[cfg(all(test, feature="decode_test_dav1d"))]
mod test_encode_decode_dav1d;

