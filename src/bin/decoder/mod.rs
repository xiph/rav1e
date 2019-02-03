use std::io;
use rav1e::*;

pub mod y4m;


pub trait Decoder {
  fn get_video_details(&self) -> VideoDetails;
  fn read_frame(&mut self, cfg: &VideoDetails) -> Result<Frame, DecodeError>;
}

#[derive(Debug)]
pub enum DecodeError {
  EOF,
  BadInput,
  UnknownColorspace,
  ParseError,
  IoError(io::Error),
}
