// Copyright (c) 2017-2020, The rav1e contributors. All rights reserved
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

#![deny(bare_trait_objects)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_ptr_alignment)]
#![allow(clippy::cognitive_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::verbose_bit_mask)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::wrong_self_convention)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::upper_case_acronyms)]
#![warn(clippy::expl_impl_clone_on_copy)]
#![warn(clippy::linkedlist)]
#![warn(clippy::map_flatten)]
#![warn(clippy::mem_forget)]
#![warn(clippy::mut_mut)]
#![warn(clippy::mutex_integer)]
#![warn(clippy::needless_borrow)]
#![warn(clippy::needless_continue)]
#![warn(clippy::path_buf_push_overwrite)]
#![warn(clippy::range_plus_one)]
#![feature(type_ascription)]

// Override assert! and assert_eq! in tests
#[cfg(test)]
#[macro_use]
extern crate pretty_assertions;

#[macro_use]
extern crate log;

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

mod rayon {
  cfg_if::cfg_if! {
    if #[cfg(all(target_arch="wasm32", not(target_feature = "atomics")))] {
      pub struct ThreadPoolBuilder ();
      impl ThreadPoolBuilder {
        pub fn new() -> ThreadPoolBuilder {
          ThreadPoolBuilder()
        }
        pub fn build(self) -> Result<ThreadPool, ()> {
          Ok(ThreadPool())
        }
        pub fn num_threads(self, _num_threads: usize) -> ThreadPoolBuilder {
          ThreadPoolBuilder()
        }
      }
      #[derive(Debug)]
      pub struct ThreadPool ();
      impl ThreadPool {
        pub fn install<OP, R>(&self, op: OP) -> R where
              OP: FnOnce() -> R + Send,
                  R: Send, {
          op()
        }
      }

      pub mod iter {
        pub trait IntoParallelIterator {
            type Iter: Iterator<Item = Self::Item>;
            type Item: Send;

            fn into_par_iter(self) -> Self::Iter;
        }

        impl<I: IntoIterator> IntoParallelIterator for I where
          I::Item : Send {
          type Item = I::Item;
          type Iter = I::IntoIter;

          fn into_par_iter(self) -> I::IntoIter {
            self.into_iter()
          }
        }

        pub trait IntoParallelRefMutIterator<'data> {
            type Iter: IntoParallelIterator<Item = Self::Item>;
            type Item: Send + 'data;

            fn par_iter_mut(&'data mut self) -> Self::Iter;
        }

        impl<'data, I: 'data + ?Sized> IntoParallelRefMutIterator<'data> for I
        where
            &'data mut I: IntoParallelIterator,
        {
            type Iter = <&'data mut I as IntoParallelIterator>::Iter;
            type Item = <&'data mut I as IntoParallelIterator>::Item;

            fn par_iter_mut(&'data mut self) -> Self::Iter {
                self.into_par_iter()
            }
        }

        pub trait ParallelIterator: Iterator {
          fn flat_map_iter<U, F>(self, f: F) -> std::iter::FlatMap<Self, U, F>
          where
            Self: Sized,
            U: IntoIterator,
            F: FnMut(<Self as Iterator>::Item) -> U,
          {
            self.flat_map(f)
          }
        }

        impl<I: Iterator> ParallelIterator for I {}
      }

      pub mod slice {
        pub trait ParallelSlice<T: Sync> {
          fn par_chunks_exact(
            &self, chunk_size: usize,
          ) -> std::slice::ChunksExact<'_, T>;
        }

        impl<T: Sync> ParallelSlice<T> for [T] {
          #[inline]
          fn par_chunks_exact(
            &self, chunk_size: usize,
          ) -> std::slice::ChunksExact<'_, T> {
            self.chunks_exact(chunk_size)
          }
        }
      }

      pub mod prelude {
        pub use super::iter::*;
        pub use super::slice::*;
      }

      pub fn join<A, B, RA, RB>(oper_a: A, oper_b: B) -> (RA, RB)
      where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send {
        (oper_a(), oper_b())
      }
    } else {
      pub use rayon::*;
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
mod context;
mod deblock;
mod encoder;
mod entropymode;
mod lrf;
mod mc;
mod me;
mod rate;
mod recon_intra;
mod scan_order;
#[cfg(feature = "scenechange")]
pub mod scenechange;
#[cfg(not(feature = "scenechange"))]
mod scenechange;
mod segmentation;
mod stats;
mod tiling;
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
