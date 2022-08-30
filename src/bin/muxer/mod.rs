// Copyright (c) 2017-2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

mod ivf;
use self::ivf::IvfMuxer;

mod y4m;
pub use self::y4m::write_y4m_frame;

use rav1e::prelude::*;

use anyhow::Result;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufWriter, Stdout, Write};
use std::path::Path;
use std::{fs, io};

use crate::error::*;

pub trait Muxer: Send {
  fn write_header(
    &mut self, width: usize, height: usize, framerate_num: usize,
    framerate_den: usize, subsampling: (usize, usize), bit_depth: u8,
    primaries: ColorPrimaries, xfer: TransferCharacteristics,
    matrix: MatrixCoefficients, full_range: bool,
  ) -> Result<()>;

  fn write_frame(
    &mut self, pts: u64, data: &[u8], frame_type: FrameType,
  ) -> Result<()>;

  fn flush(&mut self) -> Result<()>;
}

pub fn create_muxer<P: AsRef<Path>>(
  path: P, overwrite: bool,
) -> Result<Box<dyn Muxer + Send>, CliError> {
  if !overwrite {
    check_file(path.as_ref())?;
  }

  if let Some(path) = path.as_ref().to_str() {
    if path == "-" {
      return IvfMuxer::<Stdout>::open(path);
    }
  }

  let ext = path
    .as_ref()
    .extension()
    .and_then(OsStr::to_str)
    .map(str::to_lowercase)
    .unwrap_or_else(|| "ivf".into());

  match &ext[..] {
    "ivf" => IvfMuxer::<BufWriter<File>>::open(path),
    _e => {
      panic!("{} is not a supported extension, please change to .ivf", ext);
    }
  }
}

fn check_file<P: AsRef<Path>>(path: P) -> Result<(), CliError> {
  if is_file(path.as_ref()) {
    eprint!(
      "File '{}' already exists. Overwrite ? [y/N] ",
      path.as_ref().display()
    );
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
    use std::os::unix::fs::*;
    fn is_file<P: AsRef<Path>>(path: P) -> bool {
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
