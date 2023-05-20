// Copyright (c) 2017-2022, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! rav1e is an [AV1] video encoder. It is designed to eventually cover all use
//! cases, though in its current form it is most suitable for cases where
//! libaom (the reference encoder) is too slow.
//!
//! ## Features
//!
//! * Intra and inter frames
//! * 64x64 superblocks
//! * 4x4 to 64x64 RDO-selected square and 2:1/1:2 rectangular blocks
//! * DC, H, V, Paeth, smooth, and a subset of directional prediction modes
//! * DCT, (FLIP-)ADST and identity transforms (up to 64x64, 16x16 and 32x32
//!   respectively)
//! * 8-, 10- and 12-bit depth color
//! * 4:2:0 (full support), 4:2:2 and 4:4:4 (limited) chroma sampling
//! * Variable speed settings
//! * Near real-time encoding at high speed levels
//!
//! ## Usage
//!
//! Encoding is done through the [`Context`] struct. Examples on
//! [`Context::receive_packet`] show how to create a [`Context`], send frames
//! into it and receive packets of encoded data.
//!
//! [AV1]: https://aomediacodec.github.io/av1-spec/av1-spec.pdf
//! [`Context`]: struct.Context.html
//! [`Context::receive_packet`]: struct.Context.html#method.receive_packet

// Safety lints
#![deny(bare_trait_objects)]
#![deny(clippy::as_ptr_cast_mut)]
#![deny(clippy::large_stack_arrays)]
// Performance lints
#![warn(clippy::inefficient_to_string)]
#![warn(clippy::invalid_upcast_comparisons)]
#![warn(clippy::iter_with_drain)]
#![warn(clippy::linkedlist)]
#![warn(clippy::mutex_integer)]
#![warn(clippy::naive_bytecount)]
#![warn(clippy::needless_bitwise_bool)]
#![warn(clippy::needless_collect)]
#![warn(clippy::or_fun_call)]
#![warn(clippy::stable_sort_primitive)]
#![warn(clippy::suboptimal_flops)]
#![warn(clippy::trivial_regex)]
#![warn(clippy::trivially_copy_pass_by_ref)]
#![warn(clippy::unnecessary_join)]
#![warn(clippy::unused_async)]
#![warn(clippy::zero_sized_map_values)]
// Correctness lints
#![deny(clippy::case_sensitive_file_extension_comparisons)]
#![deny(clippy::copy_iterator)]
#![deny(clippy::expl_impl_clone_on_copy)]
#![deny(clippy::float_cmp)]
#![warn(clippy::imprecise_flops)]
#![deny(clippy::manual_instant_elapsed)]
#![deny(clippy::mem_forget)]
#![deny(clippy::path_buf_push_overwrite)]
#![deny(clippy::same_functions_in_if_condition)]
#![deny(clippy::unchecked_duration_subtraction)]
#![deny(clippy::unicode_not_nfc)]
// Clarity/formatting lints
#![warn(clippy::checked_conversions)]
#![allow(clippy::comparison_chain)]
#![warn(clippy::derive_partial_eq_without_eq)]
#![allow(clippy::enum_variant_names)]
#![warn(clippy::explicit_deref_methods)]
#![warn(clippy::filter_map_next)]
#![warn(clippy::flat_map_option)]
#![warn(clippy::fn_params_excessive_bools)]
#![warn(clippy::implicit_clone)]
#![warn(clippy::iter_not_returning_iterator)]
#![warn(clippy::iter_on_empty_collections)]
#![warn(clippy::macro_use_imports)]
#![warn(clippy::manual_clamp)]
#![warn(clippy::manual_let_else)]
#![warn(clippy::manual_ok_or)]
#![warn(clippy::manual_string_new)]
#![warn(clippy::map_flatten)]
#![warn(clippy::match_bool)]
#![warn(clippy::mut_mut)]
#![warn(clippy::needless_borrow)]
#![warn(clippy::needless_continue)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![warn(clippy::range_minus_one)]
#![warn(clippy::range_plus_one)]
#![warn(clippy::ref_binding_to_reference)]
#![warn(clippy::ref_option_ref)]
#![warn(clippy::trait_duplication_in_bounds)]
#![warn(clippy::unused_peekable)]
#![warn(clippy::unused_rounding)]
#![warn(clippy::unused_self)]
#![allow(clippy::upper_case_acronyms)]
#![warn(clippy::verbose_bit_mask)]
#![warn(clippy::verbose_file_reads)]
// Documentation lints
#![warn(clippy::doc_link_with_quotes)]
#![warn(clippy::doc_markdown)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
// FIXME: We should fix instances of this lint and change it to `warn`
#![allow(clippy::missing_safety_doc)]

// Override assert! and assert_eq! in tests
#[cfg(test)]
#[macro_use]
extern crate pretty_assertions;

#[macro_use]
extern crate log;

pub(crate) mod built_info {
  // The file has been placed there by the build script.
  include!(concat!(env!("OUT_DIR"), "/built.rs"));
}

mod serialize {
  cfg_if::cfg_if! {
    if #[cfg(feature="serialize")] {
      pub use serde::*;
    } else {
      pub use noop_proc_macro::{Deserialize, Serialize};
    }
  }
}

mod wasm_bindgen {
  cfg_if::cfg_if! {
    if #[cfg(feature="wasm")] {
      pub use wasm_bindgen::prelude::*;
    } else {
      pub use noop_proc_macro::wasm_bindgen;
    }
  }
}

#[cfg(any(cargo_c, feature = "capi"))]
pub mod capi;

#[macro_use]
mod transform;
#[macro_use]
mod cpu_features;

mod activity;
pub(crate) mod asm;
mod dist;
mod ec;
mod partition;
mod predict;
mod quantize;
mod rdo;
mod rdo_tables;
#[macro_use]
mod util;
mod cdef;
#[doc(hidden)]
pub mod context;
mod deblock;
mod encoder;
mod entropymode;
mod levels;
mod lrf;
mod mc;
mod me;
mod rate;
mod recon_intra;
mod sad_plane;
mod scan_order;
#[cfg(feature = "scenechange")]
pub mod scenechange;
#[cfg(not(feature = "scenechange"))]
mod scenechange;
mod segmentation;
mod stats;
#[doc(hidden)]
pub mod tiling;
mod token_cdfs;

mod api;
mod frame;
mod header;

use crate::encoder::*;

pub use crate::api::{
  Config, Context, EncoderConfig, EncoderStatus, InvalidConfig, Packet,
};
pub use crate::frame::Frame;
pub use crate::util::{CastFromPrimitive, Pixel, PixelType};

/// Commonly used types and traits.
pub mod prelude {
  pub use crate::api::*;
  pub use crate::encoder::{Sequence, Tune};
  pub use crate::frame::{
    Frame, FrameParameters, FrameTypeOverride, Plane, PlaneConfig,
  };
  pub use crate::partition::BlockSize;
  pub use crate::predict::PredictionMode;
  pub use crate::transform::TxType;
  pub use crate::util::{CastFromPrimitive, Pixel, PixelType};
}

/// Basic data structures
pub mod data {
  pub use crate::api::{
    ChromaticityPoint, EncoderStatus, FrameType, Packet, Rational,
  };
  pub use crate::frame::{Frame, FrameParameters};
  pub use crate::stats::EncoderStats;
  pub use crate::util::{CastFromPrimitive, Pixel, PixelType};
}

pub use crate::api::color;

/// Encoder configuration and settings
pub mod config {
  pub use crate::api::config::{
    GrainTableSegment, NoiseGenArgs, TransferFunction, NUM_UV_COEFFS,
    NUM_UV_POINTS, NUM_Y_COEFFS, NUM_Y_POINTS,
  };
  pub use crate::api::{
    Config, EncoderConfig, InvalidConfig, PredictionModesSetting,
    RateControlConfig, RateControlError, RateControlSummary, SpeedSettings,
  };
  pub use crate::cpu_features::CpuFeatureLevel;
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
/// assert_eq!(v1.major, v2.major);
/// ```
pub mod version {
  /// Major version component
  ///
  /// It is increased every time a release presents a incompatible API change.
  ///
  /// # Panics
  ///
  /// Will panic if package is not built with Cargo,
  /// or if the package version is not a valid triplet of integers.
  pub fn major() -> u64 {
    env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap()
  }
  /// Minor version component
  ///
  /// It is increased every time a release presents new functionalities are added
  /// in a backwards-compatible manner.
  ///
  /// # Panics
  ///
  /// Will panic if package is not built with Cargo,
  /// or if the package version is not a valid triplet of integers.
  pub fn minor() -> u64 {
    env!("CARGO_PKG_VERSION_MINOR").parse().unwrap()
  }
  /// Patch version component
  ///
  /// It is increased every time a release provides only backwards-compatible bugfixes.
  ///
  /// # Panics
  ///
  /// Will panic if package is not built with Cargo,
  /// or if the package version is not a valid triplet of integers.
  pub fn patch() -> u64 {
    env!("CARGO_PKG_VERSION_PATCH").parse().unwrap()
  }

  /// Version information as presented in `[package]` `version`.
  ///
  /// e.g. `0.1.0`
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
      format!("{s} - {hash}")
    }
  }

  cfg_if::cfg_if! {
    if #[cfg(feature="git_version")] {
      fn git_version() -> &'static str {
        crate::built_info::GIT_VERSION.unwrap_or_default()
      }

      fn git_hash() -> &'static str {
        crate::built_info::GIT_COMMIT_HASH.unwrap_or_default()
      }
    } else {
      fn git_version() -> &'static str {
        "UNKNOWN"
      }

      fn git_hash() -> &'static str {
        "UNKNOWN"
      }
    }
  }
  /// Commit hash (short)
  ///
  /// Short hash of the git commit used by this build
  ///
  /// e.g. `g743d464`
  ///
  pub fn hash() -> String {
    git_hash().to_string()
  }

  /// Version information with the information
  /// provided by `git describe --tags`.
  ///
  /// e.g. `0.1.0 (v0.1.0-1-g743d464)`
  ///
  pub fn full() -> String {
    format!("{} ({})", short(), git_version(),)
  }
}
#[cfg(all(
  any(test, fuzzing),
  any(feature = "decode_test", feature = "decode_test_dav1d")
))]
mod test_encode_decode;

#[cfg(feature = "bench")]
pub mod bench {
  pub mod api {
    pub use crate::api::*;
  }
  pub mod cdef {
    pub use crate::cdef::*;
  }
  pub mod context {
    pub use crate::context::*;
  }
  pub mod dist {
    pub use crate::dist::*;
  }
  pub mod ec {
    pub use crate::ec::*;
  }
  pub mod encoder {
    pub use crate::encoder::*;
  }
  pub mod mc {
    pub use crate::mc::*;
  }
  pub mod partition {
    pub use crate::partition::*;
  }
  pub mod frame {
    pub use crate::frame::*;
  }
  pub mod predict {
    pub use crate::predict::*;
  }
  pub mod rdo {
    pub use crate::rdo::*;
  }
  pub mod tiling {
    pub use crate::tiling::*;
  }
  pub mod transform {
    pub use crate::transform::*;
  }
  pub mod util {
    pub use crate::util::*;
  }
  pub mod cpu_features {
    pub use crate::cpu_features::*;
  }
}

#[cfg(fuzzing)]
pub mod fuzzing;
