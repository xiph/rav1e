// Copyright (c) 2016, 2018 vergen developers
//
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. All files in the project carrying such notice may not be copied,
// modified, or distributed except according to those terms.

//! Build time information.
use crate::constants::ConstantsFlags;
use crate::output::generate_build_info;
use crate::VergenError;
use std::error::Error;
use std::fs::{self, File};
use std::io::Read;
use std::path::PathBuf;

/// Generate the `cargo:` key output
///
/// The keys that can be generated include:
/// * `cargo:rustc-env=<key>=<value>` where key/value pairs are controlled by the supplied `ConstantsFlags`.
/// * `cargo:rustc-rerun-if-changed=.git/HEAD`
/// * `cargo:rustc-rerun-if-changed=<file .git/HEAD points to>`
///
/// # Example `build.rs`
///
/// ```
/// use vergen::{ConstantsFlags, generate_cargo_keys};
///
/// generate_cargo_keys(ConstantsFlags::all()).expect("Unable to generate cargo keys!");
/// ```
pub fn generate_cargo_keys(
  flags: ConstantsFlags,
) -> Result<(), Box<dyn Error>> {
  // Generate the build info map.
  let build_info = generate_build_info(flags)?;

  // Generate the 'cargo:' key output
  for (k, v) in build_info {
    println!("cargo:rustc-env={}={}", k.name(), v);
  }

  let git_dir_or_file = PathBuf::from(".git");
  if let Ok(metadata) = fs::metadata(&git_dir_or_file) {
    if metadata.is_dir() {
      // Echo the HEAD path
      let git_head_path = git_dir_or_file.join("HEAD");
      println!("cargo:rerun-if-changed={}", git_head_path.display());

      // Determine where HEAD points and echo that path also.
      let mut f = File::open(&git_head_path)?;
      let mut git_head_contents = String::new();
      let _ = f.read_to_string(&mut git_head_contents)?;
      eprintln!("HEAD contents: {}", git_head_contents);
      let ref_vec: Vec<&str> = git_head_contents.split(": ").collect();

      if ref_vec.len() == 2 {
        let current_head_file = ref_vec[1];
        let git_refs_path = PathBuf::from(".git").join(current_head_file);
        println!("cargo:rerun-if-changed={}", git_refs_path.display());
      } else {
        eprintln!("You are most likely in a detached HEAD state");
      }
    } else if metadata.is_file() {
      // We are in a worktree, so find out where the actual worktrees/<name>/HEAD file is.
      let mut git_file = File::open(&git_dir_or_file)?;
      let mut git_contents = String::new();
      let _ = git_file.read_to_string(&mut git_contents)?;
      let dir_vec: Vec<&str> = git_contents.split(": ").collect();
      eprintln!(".git contents: {}", git_contents);
      let git_path = dir_vec[1].trim();

      // Echo the HEAD psth
      let git_head_path = PathBuf::from(git_path).join("HEAD");
      println!("cargo:rerun-if-changed={}", git_head_path.display());

      // Find out what the full path to the .git dir is.
      let mut actual_git_dir = PathBuf::from(git_path);
      actual_git_dir.pop();
      actual_git_dir.pop();

      // Determine where HEAD points and echo that path also.
      let mut f = File::open(&git_head_path)?;
      let mut git_head_contents = String::new();
      let _ = f.read_to_string(&mut git_head_contents)?;
      eprintln!("HEAD contents: {}", git_head_contents);
      let ref_vec: Vec<&str> = git_head_contents.split(": ").collect();

      if ref_vec.len() == 2 {
        let current_head_file = ref_vec[1];
        let git_refs_path = actual_git_dir.join(current_head_file);
        println!("cargo:rerun-if-changed={}", git_refs_path.display());
      } else {
        eprintln!("You are most likely in a detached HEAD state");
      }
    } else {
      return Err(Box::new(VergenError::new(
        "Invalid .git format (Not a directory or a file)",
      )));
    };
  } else {
    eprintln!("Unable to generate 'cargo:rerun-if-changed'");
  }

  Ok(())
}
