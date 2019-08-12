use rav1e::config::MetricsEnabled;
use rav1e::data::QualityMetrics;
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
  /// Contains metrics such as PSNR, SSIM, etc.
  pub metrics: QualityMetrics,
}

impl<T: Pixel> From<Packet<T>> for FrameSummary {
  fn from(packet: Packet<T>) -> Self {
    Self {
      size: packet.data.len(),
      input_frameno: packet.input_frameno,
      frame_type: packet.frame_type,
      metrics: packet.metrics,
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
      if let Some(psnr) = self.metrics.psnr {
        format!(
          " - PSNR: Y: {:.4}  Cb: {:.4}  Cr: {:.4}",
          psnr.y, psnr.u, psnr.v
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
  // Which metrics to display during and at end of encode
  metrics_enabled: MetricsEnabled,
}

impl ProgressInfo {
  pub fn new(
    frame_rate: Rational, total_frames: Option<usize>,
    metrics_enabled: MetricsEnabled,
  ) -> Self {
    Self {
      frame_rate,
      total_frames,
      time_started: Instant::now(),
      frame_info: Vec::with_capacity(total_frames.unwrap_or_default()),
      encoded_size: 0,
      metrics_enabled,
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
  pub fn get_frame_type_count(&self, frame_type: FrameType) -> usize {
    self
      .frame_info
      .iter()
      .filter(|frame| frame.frame_type == frame_type)
      .count()
  }

  // Size in bytes of all frames of given frame type
  pub fn get_frame_type_size(&self, frame_type: FrameType) -> usize {
    self
      .frame_info
      .iter()
      .filter(|frame| frame.frame_type == frame_type)
      .map(|frame| frame.size)
      .sum()
  }

  pub fn print_summary(&self) -> String {
    let (key, key_size) = (
      self.get_frame_type_count(FrameType::KEY),
      self.get_frame_type_size(FrameType::KEY),
    );
    let (inter, inter_size) = (
      self.get_frame_type_count(FrameType::INTER),
      self.get_frame_type_size(FrameType::INTER),
    );
    let (ionly, ionly_size) = (
      self.get_frame_type_count(FrameType::INTRA_ONLY),
      self.get_frame_type_size(FrameType::INTRA_ONLY),
    );
    let (switch, switch_size) = (
      self.get_frame_type_count(FrameType::SWITCH),
      self.get_frame_type_size(FrameType::SWITCH),
    );
    format!(
      "\
       Key Frames: {:>6}    avg size: {:>7} B\n\
       Inter:      {:>6}    avg size: {:>7} B\n\
       Intra Only: {:>6}    avg size: {:>7} B\n\
       Switch:     {:>6}    avg size: {:>7} B\
       {}{}",
      key,
      key_size / key,
      inter,
      inter_size.checked_div(inter).unwrap_or(0),
      ionly,
      ionly_size / key,
      switch,
      switch_size / key,
      if self.metrics_enabled != MetricsEnabled::None {
        let psnr_y =
          sum_metric(&self.frame_info, |fi| fi.metrics.psnr.unwrap().y);
        let psnr_u =
          sum_metric(&self.frame_info, |fi| fi.metrics.psnr.unwrap().u);
        let psnr_v =
          sum_metric(&self.frame_info, |fi| fi.metrics.psnr.unwrap().v);
        let psnr_avg = sum_metric(&self.frame_info, |fi| {
          fi.metrics.psnr.unwrap().weighted_avg
        });
        format!(
          "\nMean PSNR: Avg: {:.4}  Y: {:.4}  Cb: {:.4}  Cr: {:.4}",
          psnr_avg, psnr_y, psnr_u, psnr_v
        )
      } else {
        String::new()
      },
      if self.metrics_enabled == MetricsEnabled::All {
        let psnr_hvs = sum_metric(&self.frame_info, |fi| {
          fi.metrics.psnr_hvs.unwrap().weighted_avg
        });
        let ssim = sum_metric(&self.frame_info, |fi| {
          fi.metrics.ssim.unwrap().weighted_avg
        });
        let ms_ssim = sum_metric(&self.frame_info, |fi| {
          fi.metrics.ms_ssim.unwrap().weighted_avg
        });
        let ciede =
          sum_metric(&self.frame_info, |fi| fi.metrics.ciede.unwrap());
        let apsnr_y =
          sum_metric(&self.frame_info, |fi| fi.metrics.apsnr.unwrap().y);
        let apsnr_u =
          sum_metric(&self.frame_info, |fi| fi.metrics.apsnr.unwrap().u);
        let apsnr_v =
          sum_metric(&self.frame_info, |fi| fi.metrics.apsnr.unwrap().v);
        let apsnr_avg = sum_metric(&self.frame_info, |fi| {
          fi.metrics.apsnr.unwrap().weighted_avg
        });
        format!(
          "\nPSNR HVS: {:.4}\nSSIM: {:.4}\nMS SSIM: {:.4}\nCIEDE2000: {:.4}\nAPSNR: Avg: {:.4}  Y: {:.4}  Cb: {:.4}  Cr: {:.4}",
          psnr_hvs, ssim, ms_ssim, ciede, apsnr_avg, apsnr_y, apsnr_u, apsnr_v
        )
      } else {
        String::new()
      }
    )
  }
}

fn sum_metric<F: FnMut(&FrameSummary) -> f64>(
  frame_info: &[FrameSummary], map_fn: F,
) -> f64 {
  frame_info.iter().map(map_fn).sum::<f64>() / frame_info.len() as f64
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
