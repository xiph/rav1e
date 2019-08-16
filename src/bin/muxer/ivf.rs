use super::Muxer;
use ivf::*;
use rav1e::prelude::*;
use std::fs::File;
use std::io;
use std::io::Write;

use crate::error::*;

pub struct IvfMuxer {
  output: Box<dyn Write>,
}

impl Muxer for IvfMuxer {
  fn write_header(
    &mut self, width: usize, height: usize, framerate_num: usize,
    framerate_den: usize,
  ) {
    write_ivf_header(
      &mut self.output,
      width,
      height,
      framerate_num,
      framerate_den,
    );
  }

  fn write_frame(&mut self, pts: u64, data: &[u8], _frame_type: FrameType) {
    write_ivf_frame(&mut self.output, pts, data);
  }

  fn flush(&mut self) -> io::Result<()> {
    self.output.flush()
  }
}

impl IvfMuxer {
  pub fn open(path: &str) -> Result<Box<dyn Muxer>, CliError> {
    let ivf = IvfMuxer {
      output: match path {
        "-" => Box::new(std::io::stdout()),
        f => Box::new(
          File::create(&f)
            .map_err(|e| e.context("Cannot open output file"))?,
        ),
      },
    };
    Ok(Box::new(ivf))
  }
}
