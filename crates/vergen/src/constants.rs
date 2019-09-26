// Copyright (c) 2016, 2018 vergen developers
//
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. All files in the project carrying such notice may not be copied,
// modified, or distributed except according to those terms.

//! Flags used to control the build script output.

bitflags!(
    /// Constants Flags
    ///
    /// Use these to toggle off the generation of constants you won't use.
    ///
    /// ```
    /// # extern crate vergen;
    /// #
    /// # use vergen::ConstantsFlags;
    /// #
    /// # fn main() {
    /// let mut actual_flags = ConstantsFlags::all();
    /// actual_flags.toggle(ConstantsFlags::SHA_SHORT);
    /// actual_flags.toggle(ConstantsFlags::BUILD_DATE);
    /// actual_flags.toggle(ConstantsFlags::SEMVER_LIGHTWEIGHT);
    /// actual_flags.toggle(ConstantsFlags::SEMVER_FROM_CARGO_PKG);
    ///
    /// let expected_flags = ConstantsFlags::BUILD_TIMESTAMP |
    ///     ConstantsFlags::SHA |
    ///     ConstantsFlags::COMMIT_DATE |
    ///     ConstantsFlags::TARGET_TRIPLE |
    ///     ConstantsFlags::SEMVER |
    ///     ConstantsFlags::REBUILD_ON_HEAD_CHANGE;
    ///
    /// assert_eq!(actual_flags, expected_flags)
    /// # }
    /// ```
    pub struct ConstantsFlags: u64 {
        /// Generate the build timestamp constant.
        ///
        /// `2018-08-09T15:15:57.282334589+00:00`
        const BUILD_TIMESTAMP        = 0b0000_0000_0001;
        /// Generate the build date constant.
        ///
        /// `2018-08-09`
        const BUILD_DATE             = 0b0000_0000_0010;
        /// Generate the SHA constant.
        ///
        /// `75b390dc6c05a6a4aa2791cc7b3934591803bc22`
        const SHA                    = 0b0000_0000_0100;
        /// Generate the short SHA constant.
        ///
        /// `75b390d`
        const SHA_SHORT              = 0b0000_0000_1000;
        /// Generate the commit date constant.
        ///
        /// `2018-08-08`
        const COMMIT_DATE            = 0b0000_0001_0000;
        /// Generate the target triple constant.
        ///
        /// `x86_64-unknown-linux-gnu`
        const TARGET_TRIPLE          = 0b0000_0010_0000;
        /// Generate the semver constant.
        ///
        /// This defaults to the output of `git describe`.  If that output is
        /// empty, the the `CARGO_PKG_VERSION` environment variable is used.
        ///
        /// `v0.1.0`
        const SEMVER                 = 0b0000_0100_0000;
        /// Generate the semver constant, including lightweight tags.
        ///
        /// This defaults to the output of `git describe --tags`.  If that output
        /// is empty, the the `CARGO_PKG_VERSION` environment variable is used.
        ///
        /// `v0.1.0`
        const SEMVER_LIGHTWEIGHT     = 0b0000_1000_0000;
        /// Generate the `cargo:rebuild-if-changed=.git/HEAD` and the
        /// `cargo:rebuild-if-changed=.git/<ref>` cargo build output.
        const REBUILD_ON_HEAD_CHANGE = 0b0001_0000_0000;
        /// Generate the semver constant from `CARGO_PKG_VERSION`.  This is
        /// mutually exclusive with the `SEMVER` flag.
        ///
        /// `0.1.0`
        const SEMVER_FROM_CARGO_PKG  = 0b0010_0000_0000;
    }
);

pub const BUILD_TIMESTAMP_NAME: &str = "VERGEN_BUILD_TIMESTAMP";
pub const BUILD_DATE_NAME: &str = "VERGEN_BUILD_DATE";
pub const SHA_NAME: &str = "VERGEN_SHA";
pub const SHA_SHORT_NAME: &str = "VERGEN_SHA_SHORT";
pub const COMMIT_DATE_NAME: &str = "VERGEN_COMMIT_DATE";
pub const TARGET_TRIPLE_NAME: &str = "VERGEN_TARGET_TRIPLE";
pub const SEMVER_NAME: &str = "VERGEN_SEMVER";
pub const SEMVER_TAGS_NAME: &str = "VERGEN_SEMVER_LIGHTWEIGHT";

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn bitflags_dont_change() {
    assert_eq!(ConstantsFlags::BUILD_TIMESTAMP.bits(), 0b0000_0001);
    assert_eq!(ConstantsFlags::BUILD_DATE.bits(), 0b0000_0010);
    assert_eq!(ConstantsFlags::SHA.bits(), 0b0000_0100);
    assert_eq!(ConstantsFlags::SHA_SHORT.bits(), 0b0000_1000);
    assert_eq!(ConstantsFlags::COMMIT_DATE.bits(), 0b0001_0000);
    assert_eq!(ConstantsFlags::TARGET_TRIPLE.bits(), 0b0010_0000);
    assert_eq!(ConstantsFlags::SEMVER.bits(), 0b0100_0000);
    assert_eq!(ConstantsFlags::SEMVER_LIGHTWEIGHT.bits(), 0b1000_0000);
    assert_eq!(
      ConstantsFlags::REBUILD_ON_HEAD_CHANGE.bits(),
      0b0001_0000_0000
    );
    assert_eq!(ConstantsFlags::SEMVER_FROM_CARGO_PKG.bits(), 0b0010_0000_0000);
  }

  #[test]
  fn constants_dont_change() {
    assert_eq!(BUILD_TIMESTAMP_NAME, "VERGEN_BUILD_TIMESTAMP");
    assert_eq!(BUILD_DATE_NAME, "VERGEN_BUILD_DATE");
    assert_eq!(SHA_NAME, "VERGEN_SHA");
    assert_eq!(SHA_SHORT_NAME, "VERGEN_SHA_SHORT");
    assert_eq!(COMMIT_DATE_NAME, "VERGEN_COMMIT_DATE");
    assert_eq!(TARGET_TRIPLE_NAME, "VERGEN_TARGET_TRIPLE");
    assert_eq!(SEMVER_NAME, "VERGEN_SEMVER");
    assert_eq!(SEMVER_TAGS_NAME, "VERGEN_SEMVER_LIGHTWEIGHT");
  }
}
