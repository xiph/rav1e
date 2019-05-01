use std::io::Read;

use rav1e::Rational;
use crate::decoder::DecodeError;
use crate::decoder::Decoder;
use crate::decoder::VideoDetails;
use crate::ChromaSamplePosition;
use crate::ChromaSampling;
use crate::Frame;
use rav1e::*;

impl Decoder for y4m::Decoder<'_, Box<dyn Read>> {
  fn get_video_details(&self) -> VideoDetails {
    let width = self.get_width();
    let height = self.get_height();
    let color_space = self.get_colorspace();
    let bit_depth = color_space.get_bit_depth();
    let (chroma_sampling, chroma_sample_position) = map_y4m_color_space(color_space);
    let framerate = self.get_framerate();
    let time_base =  Rational::new(framerate.den as u64, framerate.num as u64);

    VideoDetails {
      width,
      height,
      bit_depth,
      chroma_sampling,
      chroma_sample_position,
      time_base,
    }
  }

  fn read_frame<T: Pixel>(&mut self, cfg: &VideoDetails) -> Result<Frame<T>, DecodeError> {
    let bytes = self.get_bytes_per_sample();
    self.read_frame()
      .map(|frame| {
        let mut f: Frame<T> = Frame::new(cfg.width, cfg.height, cfg.chroma_sampling);

        let (chroma_period, _) = cfg.chroma_sampling.sampling_period();

        f.planes[0].copy_from_raw_u8(frame.get_y_plane(), cfg.width * bytes, bytes);
        f.planes[1].copy_from_raw_u8(
          frame.get_u_plane(),
          cfg.width * bytes / chroma_period,
          bytes
        );
        f.planes[2].copy_from_raw_u8(
          frame.get_v_plane(),
          cfg.width * bytes / chroma_period,
          bytes
        );
        f
      })
      .map_err(Into::into)
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
  use crate::ChromaSampling::*;
  use crate::ChromaSamplePosition::*;
  match color_space {
    Cmono => (Cs400, Unknown),
    C420jpeg | C420paldv => (Cs420, Unknown),
    C420mpeg2 => (Cs420, Vertical),
    C420 | C420p10 | C420p12 => (Cs420, Colocated),
    C422 | C422p10 | C422p12 => (Cs422, Colocated),
    C444 | C444p10 | C444p12 => (Cs444, Colocated),
  }
}
