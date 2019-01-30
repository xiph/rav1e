use encoder::Frame;
use y4m::Colorspace;
use std::io;
use encoder::ChromaSampling;
use encoder::ChromaSamplePosition;
use api::Rational;

pub mod y4m;

pub trait Decoder {
  fn get_video_details(&self) -> VideoDetails;
  fn read_frame(&mut self, cfg: &VideoDetails) -> Result<Frame, DecodeError>;
}

#[derive(Debug, Clone, Copy)]
pub struct VideoDetails {
  pub width: usize,
  pub height: usize,
  pub bits: usize,
  pub bytes: usize,
  pub color_space: Colorspace,
  pub bit_depth: usize,
  pub chroma_sampling: ChromaSampling,
  pub chroma_sample_position: ChromaSamplePosition,
  pub framerate: Rational,
}

impl Default for VideoDetails {
  fn default() -> Self {
    VideoDetails {
      width: 640,
      height: 480,
      bits: 8,
      bytes: 1,
      color_space: Colorspace::Cmono,
      bit_depth: 8,
      chroma_sampling: ChromaSampling::Cs420,
      chroma_sample_position: ChromaSamplePosition::Unknown,
      framerate: Rational { num: 1, den: 1 }
    }
  }
}

#[derive(Debug)]
pub enum DecodeError {
  EOF,
  BadInput,
  UnknownColorspace,
  ParseError,
  IoError(io::Error),
}