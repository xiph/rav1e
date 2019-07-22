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

// Override assert! and assert_eq! in tests
#[cfg(test)]
#[macro_use]
extern crate pretty_assertions;

#[cfg(cargo_c)]
mod capi;

mod dist;
mod ec;
mod partition;
mod transform;
mod quantize;
mod predict;
mod rdo;
mod rdo_tables;
#[macro_use]
mod util;
mod context;
mod entropymode;
mod token_cdfs;
mod deblock;
mod segmentation;
mod cdef;
mod lrf;
mod lrf_simd;
mod encoder;
mod mc;
mod me;
mod metrics;
mod scan_order;
mod scenechange;
mod rate;
mod tiling;
mod recon_intra;

mod api;
mod header;
mod frame;

use crate::encoder::*;

pub use crate::api::{Context, Config, Packet, EncoderStatus};
pub use crate::frame::Frame;
pub use crate::util::{CastFromPrimitive, Pixel};

pub mod prelude {
  pub use crate::api::*;
  pub use crate::frame::Frame;
  pub use crate::frame::FrameParameters;
  pub use crate::frame::FrameTypeOverride;
  pub use crate::frame::Plane;
  pub use crate::frame::PlaneConfig;
  pub use crate::encoder::Tune;
  pub use crate::partition::BlockSize;
  pub use crate::util::{CastFromPrimitive, Pixel};
}

/// Basic data structures
pub mod data {
  pub use crate::frame::Frame;
  pub use crate::frame::FrameParameters;
  pub use crate::api::{
    Packet, Point, Rational, FrameType, EncoderStatus
  };
  pub use crate::util::{CastFromPrimitive, Pixel};
}

/// Color model information
pub mod color {
  pub use crate::api::color::*;
}

/// Encoder configuration and settings
pub mod config {
  pub use crate::api::{
    Config, EncoderConfig, SpeedSettings, PredictionModesSetting,
  };
}


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

#[cfg(feature="bench")]
pub mod bench {
  pub mod api { pub use crate::api::*; }
  pub mod cdef { pub use crate::cdef::*; }
  pub mod context { pub use crate::context::*; }
  pub mod dist { pub use crate::dist::*; }
  pub mod ec { pub use crate::ec::*; }
  pub mod encoder { pub use crate::encoder::*; }
  pub mod partition { pub use crate::partition::*; }
  pub mod frame { pub use crate::frame::*; }
  pub mod predict { pub use crate::predict::*; }
  pub mod rdo { pub use crate::rdo::*; }
  pub mod transform { pub use crate::transform::*; }
  pub mod util { pub use crate::util::*; }
}
