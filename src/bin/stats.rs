use rav1e::prelude::*;
use rav1e::{Packet, Pixel};
use std::fmt;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub struct FrameSummary {
  // Frame size in bytes
  pub size: usize,
  pub input_frameno: u64,
  pub frame_type: FrameType,
  // PSNR for Y, U, and V planes
  pub psnr: Option<(f64, f64, f64)>,
}

impl<T: Pixel> From<Packet<T>> for FrameSummary {
  fn from(packet: Packet<T>) -> Self {
    Self {
      size: packet.data.len(),
      input_frameno: packet.input_frameno,
      frame_type: packet.frame_type,
      psnr: packet.psnr,
    }
  }
}

impl fmt::Display for FrameSummary {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "Input Frame {} - {} - {} bytes{}",
      self.input_frameno,
      self.frame_type,
      self.size,
      if let Some(psnr) = self.psnr {
        format!(
          " - PSNR: Y: {:.4}  Cb: {:.4}  Cr: {:.4}",
          psnr.0, psnr.1, psnr.2
        )
      } else {
        String::new()
      }
    )
  }
}

#[derive(Debug, Clone)]
pub struct ProgressInfo {
  // Frame rate of the video
  frame_rate: Rational,
  // The length of the whole video, in frames, if known
  total_frames: Option<usize>,
  // The time the encode was started
  time_started: Instant,
  // List of frames encoded so far
  frame_info: Vec<FrameSummary>,
  // Video size so far in bytes.
  //
  // This value will be updated in the CLI very frequently, so we cache the previous value
  // to reduce the overall complexity.
  encoded_size: usize,
  // Whether to display PSNR statistics during and at end of encode
  show_psnr: bool,
}

impl ProgressInfo {
  pub fn new(
    frame_rate: Rational, total_frames: Option<usize>, show_psnr: bool,
  ) -> Self {
    Self {
      frame_rate,
      total_frames,
      time_started: Instant::now(),
      frame_info: Vec::with_capacity(total_frames.unwrap_or_default()),
      encoded_size: 0,
      show_psnr,
    }
  }

  pub fn add_frame(&mut self, frame: FrameSummary) {
    self.encoded_size += frame.size;
    self.frame_info.push(frame);
  }

  pub fn frames_encoded(&self) -> usize {
    self.frame_info.len()
  }

  pub fn encoding_fps(&self) -> f64 {
    let duration = Instant::now().duration_since(self.time_started);
    self.frame_info.len() as f64
      / (duration.as_secs() as f64 + duration.subsec_millis() as f64 / 1000f64)
  }

  pub fn video_fps(&self) -> f64 {
    self.frame_rate.num as f64 / self.frame_rate.den as f64
  }

  // Returns the bitrate of the frames so far, in bits/second
  pub fn bitrate(&self) -> usize {
    let bits = self.encoded_size * 8;
    let seconds = self.frame_info.len() as f64 / self.video_fps();
    (bits as f64 / seconds) as usize
  }

  // Estimates the final filesize in bytes, if the number of frames is known
  pub fn estimated_size(&self) -> usize {
    self
      .total_frames
      .map(|frames| self.encoded_size * frames / self.frames_encoded())
      .unwrap_or_default()
  }

  // Estimates the remaining encoding time in seconds, if the number of frames is known
  pub fn estimated_time(&self) -> f64 {
    self
      .total_frames
      .map(|frames| {
        (frames - self.frames_encoded()) as f64 / self.encoding_fps()
      })
      .unwrap_or_default()
  }

  // Number of frames of given type which appear in the video
  fn get_frame_type_count(&self, frame_type: FrameType) -> usize {
    self
      .frame_info
      .iter()
      .filter(|frame| frame.frame_type == frame_type)
      .count()
  }

  fn get_frame_type_avg_size(&self, frame_type: FrameType) -> usize {
    let count = self.get_frame_type_count(frame_type);
    if count == 0 {
      return 0;
    }
    self
        .frame_info
        .iter()
        .filter(|frame| frame.frame_type == frame_type)
        .map(|frame| frame.size)
        .sum::<usize>()
        / count
  }

  pub fn print_summary(&self) {
    info!("{}", self);
    info!("----------");
    self.print_frame_type_summary(FrameType::KEY);
    self.print_frame_type_summary(FrameType::INTER);
    self.print_frame_type_summary(FrameType::INTRA_ONLY);
    self.print_frame_type_summary(FrameType::SWITCH);
    if self.show_psnr {
      self.print_video_psnr();
    }
  }

  fn print_frame_type_summary(&self, frame_type: FrameType) {
    let count = self.get_frame_type_count(frame_type);
    let size = self.get_frame_type_avg_size(frame_type);
    info!(
      "{:17} {:>6}    avg size: {:>7} B",
      format!("{}:", frame_type),
      count,
      size
    );
  }

  fn print_video_psnr(&self) {
    info!("----------");
    let psnr_y =
        self.frame_info.iter().map(|fi| fi.psnr.unwrap().0).sum::<f64>()
            / self.frame_info.len() as f64;
    let psnr_u =
        self.frame_info.iter().map(|fi| fi.psnr.unwrap().1).sum::<f64>()
            / self.frame_info.len() as f64;
    let psnr_v =
        self.frame_info.iter().map(|fi| fi.psnr.unwrap().2).sum::<f64>()
            / self.frame_info.len() as f64;
    info!(
      "Mean PSNR: Y: {:.4}  Cb: {:.4}  Cr: {:.4}  Avg: {:.4}",
      psnr_y,
      psnr_u,
      psnr_v,
      (psnr_y + psnr_u + psnr_v) / 3.0
    )
  }
}

impl fmt::Display for ProgressInfo {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if let Some(total_frames) = self.total_frames {
      write!(
                f,
                "encoded {}/{} frames, {:.3} fps, {:.2} Kb/s, est. size: {:.2} MB, est. time: {:.0} s",
                self.frames_encoded(),
                total_frames,
                self.encoding_fps(),
                self.bitrate() as f64 / 1000f64,
                self.estimated_size() as f64 / (1024 * 1024) as f64,
                self.estimated_time()
            )
    } else {
      write!(
        f,
        "encoded {} frames, {:.3} fps, {:.2} Kb/s",
        self.frames_encoded(),
        self.encoding_fps(),
        self.bitrate() as f64 / 1000f64
      )
    }
  }
}
