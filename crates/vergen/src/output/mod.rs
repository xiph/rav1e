// Copyright (c) 2016, 2018 vergen developers
//
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. All files in the project carrying such notice may not be copied,
// modified, or distributed except according to those terms.

//! Output types
use crate::constants::*;
use chrono::Utc;
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::process::Command;

//pub mod codegen;
pub mod envvar;

pub fn generate_build_info(
  flags: ConstantsFlags,
) -> Result<HashMap<VergenKey, String>, Box<dyn Error>> {
  let mut build_info = HashMap::new();
  let now = Utc::now();

  if flags.contains(ConstantsFlags::BUILD_TIMESTAMP) {
    build_info.insert(VergenKey::BuildTimestamp, now.to_rfc3339());
  }

  if flags.contains(ConstantsFlags::BUILD_DATE) {
    build_info
      .insert(VergenKey::BuildDate, now.format("%Y-%m-%d").to_string());
  }

  if flags.contains(ConstantsFlags::SHA) {
    let sha = run_command(Command::new("git").args(&["rev-parse", "HEAD"]));
    build_info.insert(VergenKey::Sha, sha);
  }

  if flags.contains(ConstantsFlags::SHA_SHORT) {
    let sha =
      run_command(Command::new("git").args(&["rev-parse", "--short", "HEAD"]));
    build_info.insert(VergenKey::ShortSha, sha);
  }

  if flags.contains(ConstantsFlags::COMMIT_DATE) {
    let commit_date = run_command(Command::new("git").args(&[
      "log",
      "--pretty=format:'%ad'",
      "-n1",
      "--date=short",
    ]));
    build_info.insert(
      VergenKey::CommitDate,
      commit_date.trim_matches('\'').to_string(),
    );
  }

  if flags.contains(ConstantsFlags::TARGET_TRIPLE) {
    let target_triple =
      env::var("TARGET").unwrap_or_else(|_| "UNKNOWN".to_string());
    build_info.insert(VergenKey::TargetTriple, target_triple);
  }

  if flags.contains(ConstantsFlags::SEMVER) {
    let describe = run_command(Command::new("git").args(&["describe"]));

    let semver = if describe.eq_ignore_ascii_case(&"UNKNOWN") {
      env::var("CARGO_PKG_VERSION")?
    } else {
      describe
    };
    build_info.insert(VergenKey::Semver, semver);
  } else if flags.contains(ConstantsFlags::SEMVER_FROM_CARGO_PKG) {
    build_info.insert(VergenKey::Semver, env::var("CARGO_PKG_VERSION")?);
  }

  if flags.contains(ConstantsFlags::SEMVER_LIGHTWEIGHT) {
    let describe =
      run_command(Command::new("git").args(&["describe", "--tags"]));

    let semver = if describe.eq_ignore_ascii_case(&"UNKNOWN") {
      env::var("CARGO_PKG_VERSION")?
    } else {
      describe
    };
    build_info.insert(VergenKey::SemverLightweight, semver);
  }

  Ok(build_info)
}

fn run_command(command: &mut Command) -> String {
  if let Ok(o) = command.output() {
    if o.status.success() {
      return String::from_utf8_lossy(&o.stdout).trim().to_owned();
    }
  }
  "UNKNOWN".to_owned()
}

/// Build information keys.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum VergenKey {
  /// The build timestamp. (VERGEN_BUILD_TIMESTAMP)
  BuildTimestamp,
  /// The build date. (VERGEN_BUILD_DATE)
  BuildDate,
  /// The latest commit SHA. (VERGEN_SHA)
  Sha,
  /// The latest commit short SHA. (VERGEN_SHA_SHORT)
  ShortSha,
  /// The commit date. (VERGEN_COMMIT_DATE).
  CommitDate,
  /// The target triple. (VERGEN_TARGET_TRIPLE)
  TargetTriple,
  /// The semver version from the last git tag. (VERGEN_SEMVER)
  Semver,
  /// The semver version from the last git tag, including lightweight.
  /// (VERGEN_SEMVER_LIGHTWEIGHT)
  SemverLightweight,
}

impl VergenKey {
  /// Get the name for the given key.
  pub fn name(self) -> &'static str {
    match self {
      VergenKey::BuildTimestamp => BUILD_TIMESTAMP_NAME,
      VergenKey::BuildDate => BUILD_DATE_NAME,
      VergenKey::Sha => SHA_NAME,
      VergenKey::ShortSha => SHA_SHORT_NAME,
      VergenKey::CommitDate => COMMIT_DATE_NAME,
      VergenKey::TargetTriple => TARGET_TRIPLE_NAME,
      VergenKey::Semver => SEMVER_NAME,
      VergenKey::SemverLightweight => SEMVER_TAGS_NAME,
    }
  }
}

#[cfg(test)]
mod test {}
