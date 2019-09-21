use rav1e::prelude::*;
use std::io;

pub mod y4m;

pub trait Decoder {
  fn get_video_details(&self) -> VideoDetails;
  fn read_frame<T: Pixel>(
    &mut self, cfg: &VideoDetails,
  ) -> Result<Frame<T>, DecodeError>;
}

#[derive(Debug)]
pub enum DecodeError {
  EOF,
  BadInput,
  UnknownColorspace,
  ParseError,
  IoError(io::Error),
  MemoryLimitExceeded,
}

#[derive(Debug, Clone, Copy)]
pub struct VideoDetails {
  pub width: usize,
  pub height: usize,
  pub bit_depth: usize,
  pub chroma_sampling: ChromaSampling,
  pub chroma_sample_position: ChromaSamplePosition,
  pub time_base: Rational,
}

impl Default for VideoDetails {
  fn default() -> Self {
    VideoDetails {
      width: 640,
      height: 480,
      bit_depth: 8,
      chroma_sampling: ChromaSampling::Cs420,
      chroma_sample_position: ChromaSamplePosition::Unknown,
      time_base: Rational { num: 30, den: 1 },
    }
  }
}
