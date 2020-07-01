// Copyright (c) 2017-2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(clippy::print_literal)]
#![allow(clippy::unused_io_amount)]

use rustc_version::{version, Version};
#[allow(unused_imports)]
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::exit;

#[allow(dead_code)]
fn rerun_dir<P: AsRef<Path>>(dir: P) {
  for entry in fs::read_dir(dir).unwrap() {
    let entry = entry.unwrap();
    let path = entry.path();
    println!("cargo:rerun-if-changed={}", path.to_string_lossy());

    if path.is_dir() {
      rerun_dir(path);
    }
  }
}

#[allow(dead_code)]
fn hash_changed(
  files: &[&str], out_dir: &str, config: &Path,
) -> Option<([u8; 8], PathBuf)> {
  use std::collections::hash_map::DefaultHasher;
  use std::hash::Hasher;
  use std::io::Read;

  let mut hasher = DefaultHasher::new();

  let paths = files.iter().map(Path::new).chain(std::iter::once(config));

  for path in paths {
    if let Ok(mut f) = std::fs::File::open(path) {
      let mut buf = Vec::new();
      f.read_to_end(&mut buf).unwrap();

      hasher.write(&buf);
    } else {
      panic!("Cannot open {}", path.display());
    }
  }

  let hash = hasher.finish().to_be_bytes();

  let hash_path = Path::new(&out_dir).join("asm.hash");

  if let Ok(old_hash) = std::fs::read(&hash_path) {
    if old_hash == hash {
      return None;
    }
  }

  Some((hash, hash_path))
}

#[cfg(feature = "asm")]
fn build_nasm_files() {
  use std::fs::File;
  use std::io::Write;
  let out_dir = env::var("OUT_DIR").unwrap();

  let dest_path = Path::new(&out_dir).join("config.asm");
  let mut config_file = File::create(&dest_path).unwrap();
  config_file.write(b"	%define private_prefix rav1e\n").unwrap();
  config_file.write(b"	%define ARCH_X86_32 0\n").unwrap();
  config_file.write(b" %define ARCH_X86_64 1\n").unwrap();
  config_file.write(b"	%define PIC 1\n").unwrap();
  config_file.write(b" %define STACK_ALIGNMENT 16\n").unwrap();
  config_file.write(b" %define HAVE_AVX512ICL 1\n").unwrap();
  if env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos" {
    config_file.write(b" %define PREFIX 1\n").unwrap();
  }

  let asm_files = &[
    "src/x86/ipred.asm",
    "src/x86/ipred_ssse3.asm",
    "src/x86/itx.asm",
    "src/x86/itx_ssse3.asm",
    "src/x86/mc_avx2.asm",
    "src/x86/mc_avx512.asm",
    "src/x86/mc_sse.asm",
    "src/x86/me.asm",
    "src/x86/sad_sse2.asm",
    "src/x86/sad_avx.asm",
    "src/x86/satd.asm",
    "src/x86/sse.asm",
    "src/x86/cdef.asm",
    "src/x86/tables.asm",
  ];

  if let Some((hash, hash_path)) =
    hash_changed(asm_files, &out_dir, &dest_path)
  {
    let mut config_include_arg = String::from("-I");
    config_include_arg.push_str(&out_dir);
    config_include_arg.push('/');
    nasm_rs::compile_library_args(
      "rav1easm",
      asm_files,
      &[&config_include_arg, "-Isrc/"],
    )
    .expect(
      "NASM build failed. Make sure you have nasm installed. https://nasm.us",
    );
    std::fs::write(hash_path, &hash[..]).unwrap();
  } else {
    println!("cargo:rustc-link-search={}", out_dir);
  }
  println!("cargo:rustc-link-lib=static=rav1easm");
  rerun_dir("src/x86");
  rerun_dir("src/ext/x86");
}

#[cfg(feature = "asm")]
fn build_asm_files() {
  use std::fs::File;
  use std::io::Write;
  let out_dir = env::var("OUT_DIR").unwrap();

  let dest_path = Path::new(&out_dir).join("config.h");
  let mut config_file = File::create(&dest_path).unwrap();
  config_file.write(b" #define PRIVATE_PREFIX rav1e_\n").unwrap();
  config_file.write(b" #define ARCH_AARCH64 1\n").unwrap();
  config_file.write(b" #define ARCH_ARM 0\n").unwrap();
  config_file.write(b" #define CONFIG_LOG 1 \n").unwrap();
  config_file.write(b" #define HAVE_ASM 1\n").unwrap();
  config_file.sync_all().unwrap();

  let asm_files = &[
    "src/arm/64/cdef.S",
    "src/arm/64/cdef16.S",
    "src/arm/64/mc.S",
    "src/arm/64/mc16.S",
    "src/arm/64/itx.S",
    "src/arm/64/itx16.S",
    "src/arm/64/ipred.S",
    "src/arm/64/ipred16.S",
    "src/arm/64/sad.S",
    "src/arm/tables.S",
  ];

  if let Some((hash, hash_path)) =
    hash_changed(asm_files, &out_dir, &dest_path)
  {
    cc::Build::new()
      .files(asm_files)
      .include(".")
      .include(&out_dir)
      .compile("rav1e-aarch64");

    std::fs::write(hash_path, &hash[..]).unwrap();
  } else {
    println!("cargo:rustc-link-search={}", out_dir);
    println!("cargo:rustc-link-lib=static=rav1e-aarch64");
  }
  rerun_dir("src/arm");
}

fn rustc_version_check() {
  // This should match the version in the CI
  const REQUIRED_VERSION: &str = "1.43.1";
  if version().unwrap() < Version::parse(REQUIRED_VERSION).unwrap() {
    eprintln!("rav1e requires rustc >= {}.", REQUIRED_VERSION);
    exit(1);
  }
}

#[cfg(feature = "asm")]
fn is_nasm_new_enough(
  min_version: Option<(u8, u8, u8)>, nasm_path: &Path,
) -> Result<(), String> {
  match get_output(std::process::Command::new(nasm_path).arg("-v")) {
    Ok(version) => {
      if version.contains("NASM version 0.") {
        Err(version)
      } else if let Some((major, minor, micro)) = min_version {
        let ver = parse_nasm_version(&version)?;
        eprintln!("{:?} {:?} {:?}", min_version, version, ver);
        if major > ver.0
          || (major == ver.0 && minor > ver.1)
          || (major == ver.0 && minor == ver.1 && micro > ver.2)
        {
          Err(version)
        } else {
          Ok(())
        }
      } else {
        Ok(())
      }
    }
    Err(err) => Err(err),
  }
}

#[cfg(feature = "asm")]
fn parse_nasm_version(version_string: &str) -> Result<(u8, u8, u8), String> {
  // Rust regex lib doesn't support empty expressions in alterations,
  // so workaround the other way
  let regex1 =
    regex::Regex::new(r"(?:NASM version )?(\d+)\.(\d+)\.(\d+)").unwrap();
  if let Some(captures) = regex1.captures(version_string) {
    Ok((
      captures[1].parse().expect("Invalid version component"),
      captures[2].parse().expect("Invalid version component"),
      captures[3].parse().expect("Invalid version component"),
    ))
  } else {
    let regex2 = regex::Regex::new(r"(?:NASM version )?(\d+)\.(\d+)").unwrap();
    let captures = regex2
      .captures(version_string)
      .ok_or_else(|| "Unable to parse NASM version string".to_string())?;
    Ok((
      captures[1].parse().expect("Invalid version component"),
      captures[2].parse().expect("Invalid version component"),
      0,
    ))
  }
}

#[test]
fn test_parse_nasm_version() {
  let ver_str = "NASM version 2.14.02 compiled on Jan 22 2019";
  assert_eq!((2, 14, 2), parse_nasm_version(ver_str));
  let ver_str = "NASM version 2.14 compiled on Jan 22 2019";
  assert_eq!((2, 14, 0), parse_nasm_version(ver_str));
}

#[cfg(feature = "asm")]
fn nasm_version_check() -> std::path::PathBuf {
  let nasm_path = std::path::PathBuf::from("nasm");
  match is_nasm_new_enough(Some((2, 14, 0)), &nasm_path) {
    Ok(_) => nasm_path,
    Err(version) => {
      panic!("This version of NASM is too old: {}", version);
    }
  }
}

#[cfg(feature = "asm")]
fn get_output(cmd: &mut std::process::Command) -> Result<String, String> {
  let out = cmd.output().map_err(|e| e.to_string())?;
  if out.status.success() {
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
  } else {
    Err(String::from_utf8_lossy(&out.stderr).to_string())
  }
}

#[allow(unused_variables)]
fn main() {
  rustc_version_check();

  let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
  let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
  // let env = env::var("CARGO_CFG_TARGET_ENV").unwrap();

  #[cfg(feature = "asm")]
  {
    if arch == "x86_64" {
      println!("cargo:rustc-cfg={}", "nasm_x86_64");
      nasm_version_check();
      build_nasm_files()
    }
    if arch == "aarch64" {
      println!("cargo:rustc-cfg={}", "asm_neon");
      build_asm_files()
    }
  }

  if os == "windows" && cfg!(feature = "decode_test") {
    panic!("Unsupported feature on this platform!");
  }

  vergen::generate_cargo_keys(vergen::ConstantsFlags::all())
    .expect("Unable to generate the cargo keys!");

  println!("cargo:rustc-env=PROFILE={}", env::var("PROFILE").unwrap());
}
