use std::io::Read;

use rav1e::Rational;
use decoder::DecodeError;
use decoder::Decoder;
use decoder::VideoDetails;
use encoder::ChromaSamplePosition;
use encoder::ChromaSampling;
use encoder::Frame;
use util::Fixed;

impl Decoder for y4m::Decoder<'_, Box<dyn Read>> {
  fn get_video_details(&self) -> VideoDetails {
    let width = self.get_width();
    let height = self.get_height();
    let bytes = self.get_bytes_per_sample();
    let color_space = self.get_colorspace();
    let bit_depth = color_space.get_bit_depth();
    let mono = match color_space {
        y4m::Colorspace::Cmono => true,
        _ => false
    };
    let (chroma_sampling, chroma_sample_position) = map_y4m_color_space(color_space);
    let framerate = self.get_framerate();
    let time_base =  Rational::new(framerate.den as u64, framerate.num as u64);
    VideoDetails {
      width,
      height,
      bytes,
      bit_depth,
      mono,
      chroma_sampling,
      chroma_sample_position,
      time_base,
    }
  }

  fn read_frame(&mut self, cfg: &VideoDetails) -> Result<Frame, DecodeError> {
    self.read_frame()
      .map(|frame| {
        let mut f = Frame::new(
          cfg.width.align_power_of_two(3),
          cfg.height.align_power_of_two(3),
          cfg.chroma_sampling
        );
        f.planes[0].copy_from_raw_u8(frame.get_y_plane(), cfg.width * cfg.bytes, cfg.bytes);
        f.planes[1].copy_from_raw_u8(
          frame.get_u_plane(),
          cfg.width * cfg.bytes / 2,
          cfg.bytes
        );
        f.planes[2].copy_from_raw_u8(
          frame.get_v_plane(),
          cfg.width * cfg.bytes / 2,
          cfg.bytes
        );
        f
      })
      .map_err(|e| e.into())
  }
}

impl From<y4m::Error> for DecodeError {
  fn from(e: y4m::Error) -> DecodeError {
    match e {
      y4m::Error::EOF => DecodeError::EOF,
      y4m::Error::BadInput => DecodeError::BadInput,
      y4m::Error::UnknownColorspace => DecodeError::UnknownColorspace,
      y4m::Error::ParseError => DecodeError::ParseError,
      y4m::Error::IoError(e) => DecodeError::IoError(e),
    }
  }
}

pub fn map_y4m_color_space(
  color_space: y4m::Colorspace
) -> (ChromaSampling, ChromaSamplePosition) {
  use y4m::Colorspace::*;
  use ChromaSampling::*;
  use ChromaSamplePosition::*;
  match color_space {
    C420jpeg | C420paldv => (Cs420, Unknown),
    C420mpeg2 => (Cs420, Vertical),
    C420 | C420p10 | C420p12 => (Cs420, Colocated),
    C422 | C422p10 | C422p12 => (Cs422, Colocated),
    C444 | C444p10 | C444p12 => (Cs444, Colocated),
    _ =>
      panic!("Chroma characteristics unknown for the specified color space.")
  }
}
