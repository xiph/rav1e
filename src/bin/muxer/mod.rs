// Copyright (c) 2017-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

mod y4m;

pub use self::y4m::write_y4m_frame;

use crate::error::*;
pub use av_format::muxer::Muxer;
pub use av_ivf::muxer::IvfMuxer;
use std::ffi::OsStr;
use std::fs;
use std::fs::File;
use std::io;
use std::io::Write;
use std::path::Path;

pub fn open_output(
  path: &str, overwrite: bool,
) -> Result<Box<dyn Write>, CliError> {
  if !overwrite {
    check_file(path)?;
  }

  match path {
    "-" => Ok(Box::new(io::stdout())),
    _ => Ok(Box::new(
      File::create(path).map_err(|e| e.context("Cannot open output file"))?,
    )),
  }
}

pub fn create_muxer(path: &str) -> Result<Box<dyn Muxer>, CliError> {
  let ext = Path::new(path)
    .extension()
    .and_then(OsStr::to_str)
    .map(str::to_lowercase)
    // Default to "ivf" because things like `/dev/null` need to work.
    .unwrap_or_else(|| "ivf".into());

  match &ext[..] {
    "ivf" => Ok(Box::new(IvfMuxer::new())),
    _e => {
      panic!("{} is not a supported extension, please change to .ivf", ext);
    }
  }
}

fn check_file(path: &str) -> Result<(), CliError> {
  if is_file(path) {
    eprint!("File '{}' already exists. Overwrite ? [y/N] ", path);
    io::stdout().flush().unwrap();

    let mut option_input = String::new();
    io::stdin().read_line(&mut option_input).expect("Failed to read option.");

    match option_input.as_str().trim() {
      "y" | "Y" => return Ok(()),
      _ => return Err(CliError::new("Not overwriting, exiting.")),
    };
  }
  Ok(())
}

cfg_if::cfg_if! {
  if #[cfg(unix)] {
    fn is_file<P: AsRef<Path>>(path: P) -> bool {
        use std::os::unix::fs::*;
      fs::metadata(path).map(|meta| {
        !meta.file_type().is_char_device() && !meta.file_type().is_socket()
      }).unwrap_or(false)
    }
  } else {
    fn is_file<P: AsRef<Path>>(path: P) -> bool {
      fs::metadata(path).is_ok()
    }
  }
}
