// Copyright (c) 2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.
#![deny(missing_docs)]

/// Color model information
pub mod color;
/// Encoder Configuration
pub mod config;

#[cfg(test)]
mod test;

pub use color::*;
pub use config::*;

use arrayvec::ArrayVec;
use bitstream_io::*;
use err_derive::Error;
use itertools::Itertools;
use serde_derive::{Deserialize, Serialize};

use crate::context::*;
use crate::dist::get_satd;
use crate::encoder::*;
use crate::frame::*;
use crate::metrics::calculate_frame_psnr;
use crate::partition::*;
use crate::predict::PredictionMode;
use crate::rate::RCState;
use crate::rate::FRAME_NSUBTYPES;
use crate::rate::FRAME_SUBTYPE_I;
use crate::rate::FRAME_SUBTYPE_P;
use crate::rate::FRAME_SUBTYPE_SEF;
use crate::scenechange::SceneChangeDetector;
use crate::stats::EncoderStats;
use crate::tiling::{Area, TileRect};
use crate::transform::TxSize;
use crate::util::Pixel;

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::{cmp, fmt, io, iter};

// TODO: use the num crate?
/// A rational number.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Rational {
  /// Numerator.
  pub num: u64,
  /// Denominator.
  pub den: u64,
}

impl Rational {
  /// Creates a rational number from the given numerator and denominator.
  pub fn new(num: u64, den: u64) -> Self {
    Rational { num, den }
  }

  /// Returns a rational number that is the reciprocal of the given one.
  pub fn from_reciprocal(reciprocal: Self) -> Self {
    Rational { num: reciprocal.den, den: reciprocal.num }
  }

  /// Returns the rational number as a floating-point number.
  pub fn as_f64(self) -> f64 {
    self.num as f64 / self.den as f64
  }
}

/// Possible types of a frame.
#[allow(dead_code, non_camel_case_types)]
#[derive(Debug, Eq, PartialEq, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub enum FrameType {
  /// Key frame.
  KEY,
  /// Inter-frame.
  INTER,
  /// Intra-only frame.
  INTRA_ONLY,
  /// Switching frame.
  SWITCH,
}

impl fmt::Display for FrameType {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    use self::FrameType::*;
    match self {
      KEY => write!(f, "Key frame"),
      INTER => write!(f, "Inter frame"),
      INTRA_ONLY => write!(f, "Intra only frame"),
      SWITCH => write!(f, "Switching frame"),
    }
  }
}

/// The set of options that controls frame re-ordering and reference picture
///  selection.
/// The options stored here are invariant over the whole encode.
#[derive(Debug, Clone, Copy)]
pub(crate) struct InterConfig {
  /// Whether frame re-ordering is enabled.
  reorder: bool,
  /// Whether P-frames can use multiple references.
  pub(crate) multiref: bool,
  /// The depth of the re-ordering pyramid.
  /// The current code cannot support values larger than 2.
  pub(crate) pyramid_depth: u64,
  /// Number of input frames in group.
  pub(crate) group_input_len: u64,
  /// Number of output frames in group.
  /// This includes both hidden frames and "show existing frame" frames.
  group_output_len: u64,
}

impl InterConfig {
  fn new(enc_config: &EncoderConfig) -> InterConfig {
    let reorder = !enc_config.low_latency;
    // A group always starts with (group_output_len - group_input_len) hidden
    //  frames, followed by group_input_len shown frames.
    // The shown frames iterate over the input frames in order, with frames
    //  already encoded as hidden frames now displayed with Show Existing
    //  Frame.
    // For example, for a pyramid depth of 2, the group is as follows:
    //                      |TU         |TU |TU |TU
    // idx_in_group_output:   0   1   2   3   4   5
    // input_frameno:         4   2   1  SEF  3  SEF
    // output_frameno:        1   2   3   4   5   6
    // level:                 0   1   2   1   2   0
    //                        ^^^^^   ^^^^^^^^^^^^^
    //                        hidden      shown
    // TODO: This only works for pyramid_depth <= 2 --- after that we need
    //  more hidden frames in the middle of the group.
    let pyramid_depth = if reorder { 2 } else { 0 };
    let group_input_len = 1 << pyramid_depth;
    let group_output_len = group_input_len + pyramid_depth;
    InterConfig {
      reorder,
      multiref: reorder || enc_config.speed_settings.multiref,
      pyramid_depth,
      group_input_len,
      group_output_len,
    }
  }

  /// Get the index of an output frame in its re-ordering group given the output
  ///  frame number of the frame in the current keyframe gop.
  /// When re-ordering is disabled, this always returns 0.
  pub(crate) fn get_idx_in_group_output(
    &self, output_frameno_in_gop: u64,
  ) -> u64 {
    // The first frame in the GOP should be a keyframe and is not re-ordered,
    //  so we should not be calling this function on it.
    debug_assert!(output_frameno_in_gop > 0);
    (output_frameno_in_gop - 1) % self.group_output_len
  }

  /// Get the order-hint of an output frame given the output frame number of the
  ///  frame in the current keyframe gop and the index of that output frame
  ///  in its re-ordering gorup.
  pub(crate) fn get_order_hint(
    &self, output_frameno_in_gop: u64, idx_in_group_output: u64,
  ) -> u32 {
    // The first frame in the GOP should be a keyframe, but currently this
    //  function only handles inter frames.
    // We could return 0 for keyframes if keyframe support is needed.
    debug_assert!(output_frameno_in_gop > 0);
    // Which P-frame group in the current gop is this output frame in?
    // Subtract 1 because the first frame in the gop is always a keyframe.
    let group_idx = (output_frameno_in_gop - 1) / self.group_output_len;
    // Get the offset to the corresponding input frame.
    // TODO: This only works with pyramid_depth <= 2.
    let offset = if idx_in_group_output < self.pyramid_depth {
      self.group_input_len >> idx_in_group_output
    } else {
      idx_in_group_output - self.pyramid_depth + 1
    };
    // Construct the final order hint relative to the start of the group.
    (self.group_input_len * group_idx + offset) as u32
  }

  /// Get the level of the current frame in the pyramid.
  pub(crate) fn get_level(&self, idx_in_group_output: u64) -> u64 {
    if !self.reorder {
      0
    } else if idx_in_group_output < self.pyramid_depth {
      // Hidden frames are output first (to be shown in the future).
      idx_in_group_output
    } else {
      // Shown frames
      // TODO: This only works with pyramid_depth <= 2.
      pos_to_lvl(
        idx_in_group_output - self.pyramid_depth + 1,
        self.pyramid_depth,
      )
    }
  }

  pub(crate) fn get_slot_idx(&self, level: u64, order_hint: u32) -> u32 {
    // Frames with level == 0 are stored in slots 0..4, and frames with higher
    //  values of level in slots 4..8
    if level == 0 {
      (order_hint >> self.pyramid_depth) & 3
    } else {
      // This only works with pyramid_depth <= 4.
      3 + level as u32
    }
  }

  pub(crate) fn get_show_frame(&self, idx_in_group_output: u64) -> bool {
    idx_in_group_output >= self.pyramid_depth
  }

  pub(crate) fn get_show_existing_frame(
    &self, idx_in_group_output: u64,
  ) -> bool {
    // The self.reorder test here is redundant, but short-circuits the rest,
    //  avoiding a bunch of work when it's false.
    self.reorder
      && self.get_show_frame(idx_in_group_output)
      && (idx_in_group_output - self.pyramid_depth + 1).count_ones() == 1
      && idx_in_group_output != self.pyramid_depth
  }

  pub(crate) fn get_input_frameno(
    &self, output_frameno_in_gop: u64, gop_input_frameno_start: u64,
  ) -> u64 {
    if output_frameno_in_gop == 0 {
      gop_input_frameno_start
    } else {
      let idx_in_group_output =
        self.get_idx_in_group_output(output_frameno_in_gop);
      let order_hint =
        self.get_order_hint(output_frameno_in_gop, idx_in_group_output);
      gop_input_frameno_start + order_hint as u64
    }
  }

  fn max_reordering_latency(&self) -> u64 {
    self.group_input_len
  }

  pub(crate) fn keyframe_lookahead_distance(&self) -> u64 {
    cmp::max(1, self.max_reordering_latency()) + 1
  }
}

pub(crate) struct ContextInner<T: Pixel> {
  frame_count: u64,
  limit: Option<u64>,
  inter_cfg: InterConfig,
  output_frameno: u64,
  frames_processed: u64,
  /// Maps *input_frameno* to frames
  frame_q: BTreeMap<u64, Option<Arc<Frame<T>>>>, //    packet_q: VecDeque<Packet>
  /// Maps *output_frameno* to frame data
  frame_invariants: BTreeMap<u64, FrameInvariants<T>>,
  /// A list of the input_frameno for keyframes in this encode.
  /// Needed so that we don't need to keep all of the frame_invariants in
  ///  memory for the whole life of the encode.
  keyframes: BTreeSet<u64>,
  /// List of keyframes force-enabled by the user
  keyframes_forced: BTreeSet<u64>,
  /// A storage space for reordered frames.
  packet_data: Vec<u8>,
  /// Maps `output_frameno` to `gop_output_frameno_start`.
  gop_output_frameno_start: BTreeMap<u64, u64>,
  /// Maps `output_frameno` to `gop_input_frameno_start`.
  pub(crate) gop_input_frameno_start: BTreeMap<u64, u64>,
  keyframe_detector: SceneChangeDetector,
  pub(crate) config: EncoderConfig,
  seq: Sequence,
  rc_state: RCState,
  maybe_prev_log_base_q: Option<i64>,
  /// The next `output_frameno` to be assigned MVs and costs by the lookahead.
  next_lookahead_output_frameno: u64,
  /// The next `input_frameno` to be processed by frame type selection.
  next_frametype_selection_frame: u64,
}

/// The encoder context.
///
/// Contains the encoding state.
pub struct Context<T: Pixel> {
  inner: ContextInner<T>,
  config: EncoderConfig,
  pool: rayon::ThreadPool,
  is_flushing: bool,
}

/// Status that can be returned by [`Context`] functions.
///
/// [`Context`]: struct.Context.html
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
pub enum EncoderStatus {
  /// The encoder needs more data to produce an output packet.
  ///
  /// May be emitted by [`Context::receive_packet()`] when frame reordering is
  /// enabled.
  ///
  /// [`Context::receive_packet()`]: struct.Context.html#method.receive_packet
  #[error(display = "need more data")]
  NeedMoreData,
  /// There are enough frames in the queue.
  ///
  /// May be emitted by [`Context::send_frame()`] when trying to send a frame
  /// after the encoder has been flushed.
  ///
  /// [`Context::send_frame()`]: struct.Context.html#method.send_frame
  #[error(display = "enough data")]
  EnoughData,
  /// The encoder has already produced the number of frames requested.
  ///
  /// May be emitted by [`Context::receive_packet()`] after a flush request had
  /// been processed or the frame limit had been reached.
  ///
  /// [`Context::receive_packet()`]: struct.Context.html#method.receive_packet
  #[error(display = "limit reached")]
  LimitReached,
  /// A frame had been encoded but not emitted yet.
  #[error(display = "encoded")]
  Encoded,
  /// Generic fatal error.
  #[error(display = "failure")]
  Failure,
  /// A frame was encoded in the first pass of a 2-pass encode, but its stats
  /// data was not retrieved with [`Context::twopass_out()`], or not enough
  /// stats data was provided in the second pass of a 2-pass encode to encode
  /// the next frame.
  ///
  /// [`Context::twopass_out()`]: struct.Context.html#method.twopass_out
  #[error(display = "not ready")]
  NotReady,
}

/// Represents a packet.
///
/// A packet contains one shown frame together with zero or more additional
/// frames.
#[derive(Debug, PartialEq)]
pub struct Packet<T: Pixel> {
  /// The packet data.
  pub data: Vec<u8>,
  /// The reconstruction of the shown frame.
  pub rec: Option<Frame<T>>,
  /// The number of the input frame corresponding to the one shown frame in the
  /// TU stored in this packet. Since AV1 does not explicitly reorder frames,
  /// these will increase sequentially.
  // TODO: When we want to add VFR support, we will need a more explicit time
  // stamp here.
  pub input_frameno: u64,
  /// Type of the shown frame.
  pub frame_type: FrameType,
  /// PSNR for Y, U, and V planes for the shown frame.
  pub psnr: Option<(f64, f64, f64)>,
  /// QP selected for the frame.
  pub qp: u8,
  /// Block-level encoding stats for the frame
  pub enc_stats: EncoderStats,
}

impl<T: Pixel> fmt::Display for Packet<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "Frame {} - {} - {} bytes",
      self.input_frameno,
      self.frame_type,
      self.data.len()
    )
  }
}

/// Types which can be converted into frames.
///
/// This trait is used in [`Context::send_frame`] to allow for passing in
/// frames with optional frame parameters and optionally frames wrapped in
/// `Arc` (to allow for zero-copy, since the encoder uses frames in `Arc`
/// internally).
///
/// [`Context::send_frame`]: struct.Context.html#method.send_frame
pub trait IntoFrame<T: Pixel> {
  /// Converts the type into a tuple of frame and parameters.
  fn into(self) -> (Option<Arc<Frame<T>>>, Option<FrameParameters>);
}

impl<T: Pixel> IntoFrame<T> for Option<Arc<Frame<T>>> {
  fn into(self) -> (Option<Arc<Frame<T>>>, Option<FrameParameters>) {
    (self, None)
  }
}

impl<T: Pixel> IntoFrame<T> for Arc<Frame<T>> {
  fn into(self) -> (Option<Arc<Frame<T>>>, Option<FrameParameters>) {
    (Some(self), None)
  }
}

impl<T: Pixel> IntoFrame<T> for (Arc<Frame<T>>, FrameParameters) {
  fn into(self) -> (Option<Arc<Frame<T>>>, Option<FrameParameters>) {
    (Some(self.0), Some(self.1))
  }
}

impl<T: Pixel> IntoFrame<T> for (Arc<Frame<T>>, Option<FrameParameters>) {
  fn into(self) -> (Option<Arc<Frame<T>>>, Option<FrameParameters>) {
    (Some(self.0), self.1)
  }
}

impl<T: Pixel> IntoFrame<T> for Frame<T> {
  fn into(self) -> (Option<Arc<Frame<T>>>, Option<FrameParameters>) {
    (Some(Arc::new(self)), None)
  }
}

impl<T: Pixel> IntoFrame<T> for (Frame<T>, FrameParameters) {
  fn into(self) -> (Option<Arc<Frame<T>>>, Option<FrameParameters>) {
    (Some(Arc::new(self.0)), Some(self.1))
  }
}

impl<T: Pixel> IntoFrame<T> for (Frame<T>, Option<FrameParameters>) {
  fn into(self) -> (Option<Arc<Frame<T>>>, Option<FrameParameters>) {
    (Some(Arc::new(self.0)), self.1)
  }
}

impl<T: Pixel> Context<T> {
  /// Allocates and returns a new frame.
  ///
  /// # Examples
  ///
  /// ```
  /// use rav1e::prelude::*;
  ///
  /// # fn main() -> Result<(), InvalidConfig> {
  /// let cfg = Config::default();
  /// let ctx: Context<u8> = cfg.new_context()?;
  /// let frame = ctx.new_frame();
  /// # Ok(())
  /// # }
  /// ```
  #[inline]
  pub fn new_frame(&self) -> Frame<T> {
    Frame::new(
      self.config.width,
      self.config.height,
      self.config.chroma_sampling,
    )
  }

  /// Sends the frame for encoding.
  ///
  /// This method adds the frame into the frame queue and runs the first passes
  /// of the look-ahead computation.
  ///
  /// Passing `None` is equivalent to calling [`flush`].
  ///
  /// # Errors
  ///
  /// If this method is called with a frame after the encoder has been flushed,
  /// the [`EncoderStatus::EnoughData`] error is returned.
  ///
  /// # Examples
  ///
  /// ```
  /// use rav1e::prelude::*;
  ///
  /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
  /// let cfg = Config::default();
  /// let mut ctx: Context<u8> = cfg.new_context().unwrap();
  /// let f1 = ctx.new_frame();
  /// let f2 = f1.clone();
  /// let info = FrameParameters {
  ///   frame_type_override: FrameTypeOverride::Key
  /// };
  ///
  /// // Send the plain frame data
  /// ctx.send_frame(f1)?;
  /// // Send the data and the per-frame parameters
  /// // In this case the frame is forced to be a keyframe.
  /// ctx.send_frame((f2, info))?;
  /// // Flush the encoder, it is equivalent to a call to `flush()`
  /// ctx.send_frame(None)?;
  /// # Ok(())
  /// # }
  /// ```
  ///
  /// [`flush`]: #method.flush
  /// [`EncoderStatus::EnoughData`]: enum.EncoderStatus.html#variant.EnoughData
  #[inline]
  pub fn send_frame<F>(&mut self, frame: F) -> Result<(), EncoderStatus>
  where
    F: IntoFrame<T>,
  {
    let (frame, params) = frame.into();

    if frame.is_none() {
      if self.is_flushing {
        return Ok(());
      }
      self.inner.limit = Some(self.inner.frame_count);
      self.is_flushing = true;
    } else if self.is_flushing {
      return Err(EncoderStatus::EnoughData);
    }

    self.inner.send_frame(frame, params)
  }

  /// Returns the first-pass data of a two-pass encode for the frame that was
  /// just encoded.
  ///
  /// This should be called BEFORE every call to [`receive_packet`] (including
  /// the very first one), even if no packet was produced by the last call to
  /// [`receive_packet`], if any (i.e., [`EncoderStatus::Encoded`] was
  /// returned).  It needs to be called once more after
  /// [`EncoderStatus::LimitReached`] is returned, to retrieve the header that
  /// should be written to the front of the stats file (overwriting the
  /// placeholder header that was emitted at the start of encoding).
  ///
  /// It is still safe to call this function when [`receive_packet`] returns
  /// any other error. It will return `None` instead of returning a duplicate
  /// copy of the previous frame's data.
  ///
  /// [`receive_packet`]: #method.receive_packet
  /// [`EncoderStatus::Encoded`]: enum.EncoderStatus.html#variant.Encoded
  /// [`EncoderStatus::LimitReached`]:
  /// enum.EncoderStatus.html#variant.LimitReached
  #[inline]
  pub fn twopass_out(&mut self) -> Option<&[u8]> {
    let params = self
      .inner
      .rc_state
      .get_twopass_out_params(&self.inner, self.inner.output_frameno);
    self.inner.rc_state.twopass_out(params)
  }

  /// Returns the number of bytes of the stats file needed before the next
  /// frame of the second pass in a two-pass encode can be encoded.
  ///
  /// This is a lower bound (more might be required), but if `0` is returned,
  /// then encoding can proceed. This is just a hint to the application, and
  /// does not need to be called for encoding the second pass to work, so long
  /// as the application continues to provide more data to [`twopass_in`] in a
  /// loop until [`twopass_in`] returns `0`.
  ///
  /// [`twopass_in`]: #method.twopass_in
  #[inline]
  pub fn twopass_bytes_needed(&mut self) -> usize {
    self.inner.rc_state.twopass_in(None).unwrap_or(0)
  }

  /// Provides the stats data produced in the first pass of a two-pass encode
  /// to the second pass.
  ///
  /// On success this returns the number of bytes of the data which were
  /// consumed. When encoding the second pass of a two-pass encode, this should
  /// be called repeatedly in a loop before every call to [`receive_packet`]
  /// (including the very first one) until no bytes are consumed, or until
  /// [`twopass_bytes_needed`] returns `0`.
  ///
  /// [`receive_packet`]: #method.receive_packet
  /// [`twopass_bytes_needed`]: #method.twopass_bytes_needed
  #[inline]
  pub fn twopass_in(&mut self, buf: &[u8]) -> Result<usize, EncoderStatus> {
    self.inner.rc_state.twopass_in(Some(buf)).or(Err(EncoderStatus::Failure))
  }

  /// Encodes the next frame and returns the encoded data.
  ///
  /// This method is where the main encoding work is done.
  ///
  /// # Examples
  ///
  /// Encoding a single frame:
  ///
  /// ```
  /// use rav1e::prelude::*;
  ///
  /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
  /// let cfg = Config::default();
  /// let mut ctx: Context<u8> = cfg.new_context()?;
  /// let frame = ctx.new_frame();
  ///
  /// ctx.send_frame(frame)?;
  /// ctx.flush();
  ///
  /// loop {
  ///     match ctx.receive_packet() {
  ///         Ok(packet) => { /* Mux the packet. */ },
  ///         Err(EncoderStatus::Encoded) => (),
  ///         Err(EncoderStatus::LimitReached) => break,
  ///         Err(err) => Err(err)?,
  ///     }
  /// }
  /// # Ok(())
  /// # }
  /// ```
  ///
  /// Encoding a sequence of frames:
  ///
  /// ```
  /// use std::sync::Arc;
  /// use rav1e::prelude::*;
  ///
  /// fn encode_frames(
  ///     ctx: &mut Context<u8>,
  ///     mut frames: impl Iterator<Item=Frame<u8>>
  /// ) -> Result<(), EncoderStatus> {
  ///     // This is a slightly contrived example, intended to showcase the
  ///     // various statuses that can be returned from receive_packet().
  ///     // Assume that, for example, there are a lot of frames in the
  ///     // iterator, which are produced lazily, so you don't want to send
  ///     // them all in at once as to not exhaust the memory.
  ///     loop {
  ///         match ctx.receive_packet() {
  ///             Ok(packet) => { /* Mux the packet. */ },
  ///             Err(EncoderStatus::Encoded) => {
  ///                 // A frame was encoded without emitting a packet. This is
  ///                 // normal, just proceed as usual.
  ///             },
  ///             Err(EncoderStatus::LimitReached) => {
  ///                 // All frames have been encoded. Time to break out of the
  ///                 // loop.
  ///                 break;
  ///             },
  ///             Err(EncoderStatus::NeedMoreData) => {
  ///                 // The encoder has requested additional frames. Push the
  ///                 // next frame in, or flush the encoder if there are no
  ///                 // frames left (on None).
  ///                 ctx.send_frame(frames.next().map(Arc::new))?;
  ///             },
  ///             Err(EncoderStatus::EnoughData) => {
  ///                 // Since we aren't trying to push frames after flushing,
  ///                 // this should never happen in this example.
  ///                 unreachable!();
  ///             },
  ///             Err(EncoderStatus::NotReady) => {
  ///                 // We're not doing two-pass encoding, so this can never
  ///                 // occur.
  ///                 unreachable!();
  ///             },
  ///             Err(EncoderStatus::Failure) => {
  ///                 return Err(EncoderStatus::Failure);
  ///             },
  ///         }
  ///     }
  ///
  ///     Ok(())
  /// }
  /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
  /// #     let mut cfg = Config::default();
  /// #     // So it runs faster.
  /// #     cfg.enc.width = 16;
  /// #     cfg.enc.height = 16;
  /// #     let mut ctx: Context<u8> = cfg.new_context()?;
  /// #
  /// #     let frames = vec![ctx.new_frame(); 4].into_iter();
  /// #     encode_frames(&mut ctx, frames);
  /// #
  /// #     Ok(())
  /// # }
  /// ```
  #[inline]
  pub fn receive_packet(&mut self) -> Result<Packet<T>, EncoderStatus> {
    let inner = &mut self.inner;
    let pool = &mut self.pool;

    pool.install(|| inner.receive_packet())
  }

  /// Flushes the encoder.
  ///
  /// Flushing signals the end of the video. After the encoder has been
  /// flushed, no additional frames are accepted.
  #[inline]
  pub fn flush(&mut self) {
    self.send_frame(None).unwrap();
  }

  /// Produces a sequence header matching the current encoding context.
  ///
  /// Its format is compatible with the AV1 Matroska and ISOBMFF specification.
  /// Note that the returned header does not include any config OBUs which are
  /// required for some uses. See [the specification].
  ///
  /// [the specification]:
  /// https://aomediacodec.github.io/av1-isobmff/#av1codecconfigurationbox-section
  #[inline]
  pub fn container_sequence_header(&self) -> Vec<u8> {
    fn sequence_header_inner(seq: &Sequence) -> io::Result<Vec<u8>> {
      let mut buf = Vec::new();

      {
        let mut bw = BitWriter::endian(&mut buf, BigEndian);
        bw.write_bit(true)?; // marker
        bw.write(7, 1)?; // version
        bw.write(3, seq.profile)?;
        bw.write(5, 31)?; // level
        bw.write_bit(false)?; // tier
        bw.write_bit(seq.bit_depth > 8)?; // high_bitdepth
        bw.write_bit(seq.bit_depth == 12)?; // twelve_bit
        bw.write_bit(seq.bit_depth == 1)?; // monochrome
        bw.write_bit(seq.chroma_sampling != ChromaSampling::Cs444)?; // chroma_subsampling_x
        bw.write_bit(seq.chroma_sampling == ChromaSampling::Cs420)?; // chroma_subsampling_y
        bw.write(2, 0)?; // sample_position
        bw.write(3, 0)?; // reserved
        bw.write_bit(false)?; // initial_presentation_delay_present

        bw.write(4, 0)?; // reserved
      }

      Ok(buf)
    }

    let seq = Sequence::new(&self.config);

    sequence_header_inner(&seq).unwrap()
  }
}

impl<T: Pixel> ContextInner<T> {
  pub fn new(enc: &EncoderConfig) -> Self {
    // initialize with temporal delimiter
    let packet_data = TEMPORAL_DELIMITER.to_vec();

    let maybe_ac_qi_max =
      if enc.quantizer < 255 { Some(enc.quantizer as u8) } else { None };

    ContextInner {
      frame_count: 0,
      limit: None,
      inter_cfg: InterConfig::new(enc),
      output_frameno: 0,
      frames_processed: 0,
      frame_q: BTreeMap::new(),
      frame_invariants: BTreeMap::new(),
      // As an optimization for lookahead,
      // Initialize this with the first frame set as a keyframe.
      keyframes: iter::once(0).collect(),
      keyframes_forced: BTreeSet::new(),
      packet_data,
      gop_output_frameno_start: BTreeMap::new(),
      gop_input_frameno_start: BTreeMap::new(),
      keyframe_detector: SceneChangeDetector::default(),
      config: enc.clone(),
      seq: Sequence::new(enc),
      rc_state: RCState::new(
        enc.width as i32,
        enc.height as i32,
        enc.time_base.den as i64,
        enc.time_base.num as i64,
        enc.bitrate,
        maybe_ac_qi_max,
        enc.min_quantizer,
        enc.max_key_frame_interval as i32,
        enc.reservoir_frame_delay,
      ),
      maybe_prev_log_base_q: None,
      next_lookahead_output_frameno: 0,
      next_frametype_selection_frame: 0,
    }
  }

  pub fn send_frame(
    &mut self, frame: Option<Arc<Frame<T>>>, params: Option<FrameParameters>,
  ) -> Result<(), EncoderStatus> {
    let input_frameno = self.frame_count;
    if frame.is_some() {
      self.frame_count += 1;
    }
    self.frame_q.insert(input_frameno, frame);

    if let Some(params) = params {
      if params.frame_type_override == FrameTypeOverride::Key {
        self.keyframes_forced.insert(input_frameno);
      }
    }

    self.compute_lookahead_data();
    Ok(())
  }

  fn get_frame(&self, input_frameno: u64) -> Arc<Frame<T>> {
    // Clones only the arc, so low cost overhead
    self
      .frame_q
      .get(&input_frameno)
      .as_ref()
      .unwrap()
      .as_ref()
      .unwrap()
      .clone()
  }

  fn get_lookahead_ready_frames_count(&self, out_frameno: u64) -> usize {
    self
      .frame_invariants
      .iter()
      .filter(|&(&out_no, fi)| {
        out_no >= out_frameno
          && !fi.show_existing_frame
          && !fi.invalid
          && !fi.lookahead_complete
      })
      .count()
  }

  fn get_unencoded_fis_count(&self) -> usize {
    self
      .frame_invariants
      .iter()
      .filter(|&(&out_no, fi)| {
        out_no >= self.output_frameno && !fi.show_existing_frame && !fi.invalid
      })
      .count()
  }

  fn get_unencoded_ready_fis_count(&self) -> usize {
    self
      .frame_invariants
      .iter()
      .filter(|&(&out_no, fi)| {
        out_no >= self.output_frameno
          && !fi.show_existing_frame
          && !fi.invalid
          && fi.lookahead_complete
      })
      .count()
  }

  fn get_last_unencoded_ready_output_frameno(&self) -> u64 {
    self
      .frame_invariants
      .iter()
      .rfind(|&(&out_no, fi)| {
        out_no >= self.output_frameno
          && !fi.show_existing_frame
          && !fi.invalid
          && fi.lookahead_complete
      })
      .map(|(&out_no, _)| out_no)
      .unwrap_or(0)
  }

  fn lookahead_fis_filled(&self) -> bool {
    let ready_frames =
      self.get_lookahead_ready_frames_count(self.output_frameno);
    ready_frames
      > cmp::max(self.config.rdo_lookahead_frames, REF_FRAMES)
        + self.inter_cfg.keyframe_lookahead_distance() as usize
  }

  fn lookahead_buffer_filled(&self) -> bool {
    let ready_frames = self
      .next_lookahead_output_frameno
      .saturating_sub(self.get_last_unencoded_ready_output_frameno());
    ready_frames as usize
      > cmp::max(self.config.rdo_lookahead_frames, REF_FRAMES)
  }

  fn can_analyze_lookahead(&self) -> bool {
    let ready_frames = self
      .get_lookahead_ready_frames_count(self.next_lookahead_output_frameno);
    ready_frames > self.inter_cfg.keyframe_lookahead_distance() as usize
  }

  /// Indicates whether more frames need to have lookahead computations performed
  /// before proceeding to frame type decision.
  fn needs_more_lookahead_computed(&self) -> bool {
    let ready_frames = self.get_unencoded_fis_count();
    let buffer_filled = ready_frames
      > cmp::max(self.config.rdo_lookahead_frames, REF_FRAMES)
        + self.inter_cfg.keyframe_lookahead_distance() as usize;
    !buffer_filled
  }

  /// Indicates whether more frame types need to be decided before proceeding
  /// to encoding.
  fn needs_more_frame_types_decided(&self) -> bool {
    self.needs_more_frames(self.next_frametype_selection_frame)
      && (self.get_unencoded_ready_fis_count()
        < self.inter_cfg.keyframe_lookahead_distance() as usize + 1
        || self.is_flushing())
  }

  pub fn needs_more_frames(&self, frame_count: u64) -> bool {
    self.limit.map(|limit| frame_count < limit).unwrap_or(true)
  }

  fn is_flushing(&self) -> bool {
    self.frame_q.values().any(|f| f.is_none())
  }

  fn next_keyframe_input_frameno(
    &self, gop_input_frameno_start: u64, ignore_limit: bool,
  ) -> u64 {
    let next_detected = self
      .keyframes
      .iter()
      .find(|&&input_frameno| input_frameno > gop_input_frameno_start)
      .cloned();
    let mut next_limit =
      gop_input_frameno_start + self.config.max_key_frame_interval;
    if !ignore_limit && self.limit.is_some() {
      next_limit = next_limit.min(self.limit.unwrap());
    }
    if next_detected.is_none() {
      return next_limit;
    }
    cmp::min(next_detected.unwrap(), next_limit)
  }

  fn set_frame_properties(
    &mut self, output_frameno: u64,
  ) -> Result<(), EncoderStatus> {
    let fi = self.build_frame_properties(output_frameno)?;
    self.frame_invariants.insert(output_frameno, fi);

    Ok(())
  }

  fn build_frame_properties(
    &mut self, output_frameno: u64,
  ) -> Result<FrameInvariants<T>, EncoderStatus> {
    let (prev_gop_output_frameno_start, prev_gop_input_frameno_start) =
      if output_frameno == 0 {
        (0, 0)
      } else {
        (
          self.gop_output_frameno_start[&(output_frameno - 1)],
          self.gop_input_frameno_start[&(output_frameno - 1)],
        )
      };

    self
      .gop_output_frameno_start
      .insert(output_frameno, prev_gop_output_frameno_start);
    self
      .gop_input_frameno_start
      .insert(output_frameno, prev_gop_input_frameno_start);

    let output_frameno_in_gop =
      output_frameno - self.gop_output_frameno_start[&output_frameno];
    let mut input_frameno = self.inter_cfg.get_input_frameno(
      output_frameno_in_gop,
      self.gop_input_frameno_start[&output_frameno],
    );

    if output_frameno_in_gop > 0 {
      let next_keyframe_input_frameno = self.next_keyframe_input_frameno(
        self.gop_input_frameno_start[&output_frameno],
        false,
      );
      let prev_input_frameno =
        self.frame_invariants[&(output_frameno - 1)].input_frameno;
      if input_frameno >= next_keyframe_input_frameno {
        if !self.inter_cfg.reorder
          || ((output_frameno_in_gop - 1) % self.inter_cfg.group_output_len
            == 0
            && prev_input_frameno == (next_keyframe_input_frameno - 1))
        {
          input_frameno = next_keyframe_input_frameno;

          // If we'll return early, do it before modifying the state.
          match self.frame_q.get(&input_frameno) {
            Some(Some(_)) => {}
            _ => {
              return Err(EncoderStatus::NeedMoreData);
            }
          }

          *self.gop_output_frameno_start.get_mut(&output_frameno).unwrap() =
            output_frameno;
          *self.gop_input_frameno_start.get_mut(&output_frameno).unwrap() =
            next_keyframe_input_frameno;
        } else {
          let fi = FrameInvariants::new_inter_frame(
            &self.frame_invariants[&(output_frameno - 1)],
            &self.inter_cfg,
            self.gop_input_frameno_start[&output_frameno],
            output_frameno_in_gop,
            next_keyframe_input_frameno,
          );
          assert!(fi.invalid);
          return Ok(fi);
        }
      }
    }

    match self.frame_q.get(&input_frameno) {
      Some(Some(_)) => {}
      _ => {
        return Err(EncoderStatus::NeedMoreData);
      }
    }

    // Now that we know the input_frameno, look up the correct frame type
    let frame_type = if self.keyframes.contains(&input_frameno) {
      FrameType::KEY
    } else {
      FrameType::INTER
    };
    if frame_type == FrameType::KEY {
      *self.gop_output_frameno_start.get_mut(&output_frameno).unwrap() =
        output_frameno;
      *self.gop_input_frameno_start.get_mut(&output_frameno).unwrap() =
        input_frameno;
    }

    let output_frameno_in_gop =
      output_frameno - self.gop_output_frameno_start[&output_frameno];
    if output_frameno_in_gop == 0 {
      let fi = FrameInvariants::new_key_frame(
        self.config.clone(),
        self.seq,
        self.gop_input_frameno_start[&output_frameno],
      );
      assert!(!fi.invalid);
      Ok(fi)
    } else {
      let next_keyframe_input_frameno = self.next_keyframe_input_frameno(
        self.gop_input_frameno_start[&output_frameno],
        false,
      );
      let fi = FrameInvariants::new_inter_frame(
        &self.frame_invariants[&(output_frameno - 1)],
        &self.inter_cfg,
        self.gop_input_frameno_start[&output_frameno],
        output_frameno_in_gop,
        next_keyframe_input_frameno,
      );
      assert!(!fi.invalid);
      Ok(fi)
    }
  }

  pub(crate) fn done_processing(&self) -> bool {
    self.limit.map(|limit| self.frames_processed == limit).unwrap_or(false)
  }

  /// Computes lookahead motion vectors and fills in `lookahead_mvs`,
  /// `rec_buffer` and `lookahead_rec_buffer` on the `FrameInvariants`. This
  /// function must be called after every new `FrameInvariants` is initially
  /// computed.
  fn compute_lookahead_motion_vectors(&mut self, output_frameno: u64) {
    let fi = self.frame_invariants.get_mut(&output_frameno).unwrap();

    // We're only interested in valid frames which are not show-existing-frame.
    // Those two don't modify the rec_buffer so there's no need to do anything
    // special about it either, it'll propagate on its own.
    if fi.invalid || fi.show_existing_frame {
      return;
    }

    let frame = self.frame_q[&fi.input_frameno].as_ref().unwrap();

    // TODO: some of this work, like downsampling, could be reused in the
    // actual encoding.
    let mut fs = FrameState::new_with_frame(fi, frame.clone());
    fs.input_hres.downsample_from(&frame.planes[0]);
    fs.input_hres.pad(fi.width, fi.height);
    fs.input_qres.downsample_from(&fs.input_hres);
    fs.input_qres.pad(fi.width, fi.height);

    #[cfg(feature = "dump_lookahead_data")]
    {
      let plane = &fs.input_qres;
      image::GrayImage::from_fn(
        plane.cfg.width as u32,
        plane.cfg.height as u32,
        |x, y| image::Luma([plane.p(x as usize, y as usize).as_()]),
      )
      .save(format!("{}-qres.png", fi.input_frameno))
      .unwrap();
      let plane = &fs.input_hres;
      image::GrayImage::from_fn(
        plane.cfg.width as u32,
        plane.cfg.height as u32,
        |x, y| image::Luma([plane.p(x as usize, y as usize).as_()]),
      )
      .save(format!("{}-hres.png", fi.input_frameno))
      .unwrap();
    }

    // Do not modify the next output frame's FrameInvariants.
    if self.output_frameno == output_frameno {
      // We do want to propagate the lookahead_rec_buffer though.
      let rfs = Arc::new(ReferenceFrame {
        order_hint: fi.order_hint,
        // Use the original frame contents.
        frame: frame.clone(),
        input_hres: fs.input_hres,
        input_qres: fs.input_qres,
        cdfs: fs.cdfs,
        // TODO: can we set MVs here? We can probably even compute these MVs
        // right now instead of in encode_tile?
        frame_mvs: fs.frame_mvs,
        output_frameno,
      });
      for i in 0..(REF_FRAMES as usize) {
        if (fi.refresh_frame_flags & (1 << i)) != 0 {
          fi.lookahead_rec_buffer.frames[i] = Some(Arc::clone(&rfs));
          fi.lookahead_rec_buffer.deblock[i] = fs.deblock;
        }
      }

      return;
    }

    // Our lookahead_rec_buffer should be filled with correct original frame
    // data from the previous frames. Copy it into rec_buffer because that's
    // what the MV search uses. During the actual encoding rec_buffer is
    // overwritten with its correct values anyway.
    fi.rec_buffer = fi.lookahead_rec_buffer.clone();

    // TODO: as in the encoding code, key frames will have no references.
    // However, for block importance purposes we want key frames to act as
    // P-frames in this instance.
    fi.compute_lookahead_motion_vectors(&mut fs);

    #[cfg(feature = "dump_lookahead_data")]
    {
      use crate::partition::RefType::*;

      let second_ref_frame = if !self.inter_cfg.multiref {
        LAST_FRAME // make second_ref_frame match first
      } else if fi.idx_in_group_output == 0 {
        LAST2_FRAME
      } else {
        ALTREF_FRAME
      };

      // Use the default index, it corresponds to the last P-frame or to the
      // backwards lower reference (so the closest previous frame).
      let index = if second_ref_frame.to_index() != 0 { 0 } else { 1 };

      let mvs = &fs.frame_mvs[index];
      use byteorder::{NativeEndian, WriteBytesExt};
      let mut buf = vec![];
      buf.write_u64::<NativeEndian>(mvs.rows as u64).unwrap();
      buf.write_u64::<NativeEndian>(mvs.cols as u64).unwrap();
      for y in 0..mvs.rows {
        for x in 0..mvs.cols {
          let mv = mvs[y][x];
          buf.write_i16::<NativeEndian>(mv.row).unwrap();
          buf.write_i16::<NativeEndian>(mv.col).unwrap();
        }
      }
      ::std::fs::write(format!("{}-mvs.bin", fi.input_frameno), buf).unwrap();
    }

    // Set lookahead_rec_buffer on this FrameInvariants for future
    // FrameInvariants to pick it up.
    let rfs = Arc::new(ReferenceFrame {
      order_hint: fi.order_hint,
      // Use the original frame contents.
      frame: frame.clone(),
      input_hres: fs.input_hres,
      input_qres: fs.input_qres,
      cdfs: fs.cdfs,
      frame_mvs: fs.frame_mvs,
      output_frameno,
    });
    for i in 0..(REF_FRAMES as usize) {
      if (fi.refresh_frame_flags & (1 << i)) != 0 {
        fi.lookahead_rec_buffer.frames[i] = Some(Arc::clone(&rfs));
        fi.lookahead_rec_buffer.deblock[i] = fs.deblock;
      }
    }
  }

  /// Computes lookahead intra cost approximations and fills in
  /// `lookahead_intra_costs` and `lookahead_inter_costs` on the `FrameInvariants`.
  fn compute_lookahead_costs(&mut self, output_frameno: u64) {
    let fi = self.frame_invariants.get(&output_frameno).unwrap();

    // We're only interested in valid frames which are not show-existing-frame.
    if fi.invalid || fi.show_existing_frame {
      return;
    }

    let frame = self.frame_q[&fi.input_frameno].as_ref().unwrap();

    let mut plane_after_prediction = frame.planes[0].clone();

    // Unfortunately we can't reuse the motion vectors calculated earlier in
    // lookahead, because those only apply to the ref frames of this frame.
    // We need to know the motion vectors of the previous `keyframe_lookahead_distance`
    // frames to have correct inter costs for scene flash detection.
    //
    // FIXME: The MV calculation code is currently very tightly coupled to ref frames.
    // It would be better to decouple it so that we can calculate MVs between any two
    // frames, but that is a major undertaking. So for now, this code checkpoints
    // the FI state then does the temporary MV calculations on that copy, only
    // modifying the real FI state to save the computed intra/inter costs.
    let mut fi_copy = fi.clone();

    let mut fs = FrameState::new_with_frame(&fi_copy, frame.clone());
    fs.input_hres.downsample_from(&frame.planes[0]);
    fs.input_hres.pad(fi_copy.width, fi_copy.height);
    fs.input_qres.downsample_from(&fs.input_hres);
    fs.input_qres.pad(fi_copy.width, fi_copy.height);
    let output_framenos = (1..=self.inter_cfg.keyframe_lookahead_distance())
      .filter_map(|distance| {
        let input_frameno = fi_copy.input_frameno.checked_sub(distance);
        input_frameno.map(|input_frameno| {
          *self
            .frame_invariants
            .iter()
            .find(|&(_, this_fi)| {
              this_fi.input_frameno == input_frameno
                && !this_fi.show_existing_frame
                && !this_fi.invalid
            })
            .unwrap()
            .0
        })
      })
      .collect::<Vec<u64>>();

    for (distance, output_frameno) in
      (1..=self.inter_cfg.keyframe_lookahead_distance())
        .zip(output_framenos.into_iter())
    {
      let idx = (distance as usize) - 1;
      fi_copy.ref_frames[idx] = idx as u8;
      update_rec_buffer(output_frameno, &mut fi_copy, fs.clone());
    }

    fi_copy.compute_lookahead_motion_vectors(&mut fs);

    // Map the MVs we just calculated to their correct input framenos.
    let mut mv_idxs_to_framenos = ArrayVec::<[_; 5]>::new();

    for (mv_index, &rec_index) in fi_copy.ref_frames.iter().enumerate() {
      let ref_input_frameno = fi_copy.rec_buffer.frames[rec_index as usize]
        .as_ref()
        .and_then(|reference| {
          self
            .frame_invariants
            .get(&reference.output_frameno)
            .map(|fi| fi.input_frameno)
        });
      if let Some(ref_input_frameno) = ref_input_frameno {
        if !mv_idxs_to_framenos
          .iter()
          .any(|&(_, fno)| fno == ref_input_frameno)
        {
          mv_idxs_to_framenos.push((mv_index, ref_input_frameno));
        }
      }
    }

    let fi = self.frame_invariants.get_mut(&output_frameno).unwrap();

    for y in 0..fi_copy.h_in_imp_b {
      for x in 0..fi_copy.w_in_imp_b {
        let plane_org = frame.planes[0].region(Area::Rect {
          x: (x * IMPORTANCE_BLOCK_SIZE) as isize,
          y: (y * IMPORTANCE_BLOCK_SIZE) as isize,
          width: IMPORTANCE_BLOCK_SIZE,
          height: IMPORTANCE_BLOCK_SIZE,
        });

        // TODO: other intra prediction modes.
        let edge_buf = get_intra_edges(
          &frame.planes[0].as_region(),
          TileBlockOffset(BlockOffset { x, y }),
          0,
          0,
          BlockSize::BLOCK_8X8,
          PlaneOffset {
            x: (x * IMPORTANCE_BLOCK_SIZE) as isize,
            y: (y * IMPORTANCE_BLOCK_SIZE) as isize,
          },
          TxSize::TX_8X8,
          fi_copy.sequence.bit_depth,
          Some(PredictionMode::DC_PRED),
        );

        let mut plane_after_prediction_region = plane_after_prediction
          .region_mut(Area::Rect {
            x: (x * IMPORTANCE_BLOCK_SIZE) as isize,
            y: (y * IMPORTANCE_BLOCK_SIZE) as isize,
            width: IMPORTANCE_BLOCK_SIZE,
            height: IMPORTANCE_BLOCK_SIZE,
          });

        PredictionMode::DC_PRED.predict_intra(
          TileRect {
            x: x * IMPORTANCE_BLOCK_SIZE,
            y: y * IMPORTANCE_BLOCK_SIZE,
            width: IMPORTANCE_BLOCK_SIZE,
            height: IMPORTANCE_BLOCK_SIZE,
          },
          &mut plane_after_prediction_region,
          TxSize::TX_8X8,
          fi_copy.sequence.bit_depth,
          &[], // Not used by DC_PRED.
          0,   // Not used by DC_PRED.
          &edge_buf,
        );

        let plane_after_prediction_region =
          plane_after_prediction.region(Area::Rect {
            x: (x * IMPORTANCE_BLOCK_SIZE) as isize,
            y: (y * IMPORTANCE_BLOCK_SIZE) as isize,
            width: IMPORTANCE_BLOCK_SIZE,
            height: IMPORTANCE_BLOCK_SIZE,
          });

        fi.lookahead_intra_costs[y * fi.w_in_imp_b + x] = get_satd(
          &plane_org,
          &plane_after_prediction_region,
          IMPORTANCE_BLOCK_SIZE,
          IMPORTANCE_BLOCK_SIZE,
          self.config.bit_depth,
        );

        // Compute inter costs needed for keyframe detection.
        for distance in 1..=self.inter_cfg.keyframe_lookahead_distance() {
          if fi.input_frameno < distance || fi.frame_type == FrameType::KEY {
            break;
          }
          let ref_input_frameno = fi.input_frameno - distance;
          let ref_frame = self.frame_q[&ref_input_frameno].as_ref().unwrap();
          let mv_idx = mv_idxs_to_framenos
            .iter()
            .find(|&(_, fno)| *fno == ref_input_frameno)
            .map(|(idx, _)| *idx);

          fi.lookahead_inter_costs[distance as usize - 1]
            [y * fi.w_in_imp_b + x] = compute_inter_costs(
            &frame,
            &ref_frame,
            x,
            y,
            mv_idx.map(|idx| fi_copy.lookahead_mvs[idx][y * 2][x * 2]),
            self.config.bit_depth,
          );
        }
      }
    }

    #[cfg(feature = "dump_lookahead_data")]
    {
      use byteorder::{NativeEndian, WriteBytesExt};

      let data = &fi.lookahead_intra_costs;
      let mut buf = vec![];
      buf.write_u64::<NativeEndian>(fi.h_in_imp_b as u64).unwrap();
      buf.write_u64::<NativeEndian>(fi.w_in_imp_b as u64).unwrap();
      for y in 0..fi.h_in_imp_b {
        for x in 0..fi.w_in_imp_b {
          let importance = data[y * fi.w_in_imp_b + x];
          buf.write_u32::<NativeEndian>(importance).unwrap();
        }
      }
      ::std::fs::write(format!("{}-intra-cost.bin", fi.input_frameno), buf)
        .unwrap();

      for i in 0..fi.lookahead_inter_costs.len() {
        let data = &fi.lookahead_inter_costs[i];
        let mut buf = vec![];
        buf.write_u64::<NativeEndian>(fi.h_in_imp_b as u64).unwrap();
        buf.write_u64::<NativeEndian>(fi.w_in_imp_b as u64).unwrap();
        for y in 0..fi.h_in_imp_b {
          for x in 0..fi.w_in_imp_b {
            let importance = data[y * fi.w_in_imp_b + x];
            buf.write_u32::<NativeEndian>(importance).unwrap();
          }
        }
        ::std::fs::write(
          format!("{}-inter-cost-{}.bin", fi.input_frameno, i),
          buf,
        )
        .unwrap();
      }
    }
  }

  fn compute_lookahead_data(&mut self) {
    'reset: loop {
      // Compute lookahead frame invariants--these are initially tried as Inter frames,
      // and will have MVs and costs calculated on them.
      while !self.lookahead_fis_filled() || self.is_flushing() {
        let next_output_frameno =
          self.frame_invariants.keys().last().map(|key| key + 1).unwrap_or(0);
        if self.set_frame_properties(next_output_frameno).is_ok() {
          self.compute_lookahead_motion_vectors(next_output_frameno);
        } else {
          // Frame not yet read into frame queue
          break;
        }
      }

      while (self.can_analyze_lookahead() && !self.lookahead_buffer_filled())
        || (self.is_flushing()
          && self
            .frame_invariants
            .keys()
            .last()
            .map(|out_no| *out_no >= self.next_lookahead_output_frameno)
            .unwrap_or(false))
      {
        self.compute_lookahead_costs(self.next_lookahead_output_frameno);
        self.next_lookahead_output_frameno += 1;
      }

      // At this point we have `rdo_lookahead_frames` lookahead computed.
      // Now we take frames until we no longer have `rdo_lookahead_frames`
      // (or we are at the end of the video) and run frame type detection on them.
      while (self.lookahead_buffer_filled() || self.is_flushing())
        && self.needs_more_frame_types_decided()
      {
        let output_frameno = self
          .frame_invariants
          .iter()
          .find(|&(_, fi)| {
            fi.input_frameno == self.next_frametype_selection_frame
              && !fi.show_existing_frame
              && !fi.invalid
          })
          .map(|(fno, _)| *fno)
          .unwrap();
        self
          .frame_invariants
          .get_mut(&output_frameno)
          .unwrap()
          .lookahead_complete = true;

        let frame_set = self
          .frame_invariants
          .values()
          .filter(|fi| {
            fi.input_frameno >= self.next_frametype_selection_frame
              && !fi.show_existing_frame
              && !fi.invalid
          })
          .sorted_by_key(|fi| fi.input_frameno)
          .collect::<Vec<_>>();
        debug_assert!(
          frame_set[0].input_frameno == self.next_frametype_selection_frame
        );

        if frame_set[0].frame_type == FrameType::KEY {
          // If this is already marked as a key frame, we've already run analysis
          // on it and we want to avoid extra work (or an infinite reset loop).
          self.keyframes.insert(self.next_frametype_selection_frame);
          self.next_frametype_selection_frame += 1;
          continue;
        }

        self.keyframe_detector.analyze_next_frame(
          &frame_set,
          &self.config,
          &self.inter_cfg,
          &mut self.keyframes,
          &self.keyframes_forced,
        );

        let is_key_frame =
          self.keyframes.contains(&self.next_frametype_selection_frame);

        // If this frame is determined to be a Key frame, we need to reset the
        // lookahead queue to an empty state and recalculate lookahead on all
        // the frames following it. This is currently necessary in order to
        // have lookahead calculations available for frame type detection.
        // The goal behind defaulting frames in the lookahead queue to Inter
        // and only resetting if we find a Key frame is that Key frames should
        // be far less common than Inter frames, so we are minimizing the number
        // of times we need to duplicate the lookahead calculations.
        if is_key_frame {
          self.reset_lookahead_queue(self.next_frametype_selection_frame);
          continue 'reset;
        }

        self.next_frametype_selection_frame += 1;
      }

      break;
    }
  }

  fn reset_lookahead_queue(&mut self, starting_input_frame: u64) {
    // If we don't match a frame, it means we're already at the end of the video.
    if let Some(cutoff_output_frame) = self
      .frame_invariants
      .iter()
      .find(|&(_, fi)| fi.input_frameno >= starting_input_frame)
      .map(|(output_frame, _)| *output_frame)
    {
      self.next_lookahead_output_frameno = cutoff_output_frame;
      self.frame_invariants.split_off(&cutoff_output_frame);
      self.gop_output_frameno_start.split_off(&cutoff_output_frame);
      self.gop_input_frameno_start.split_off(&cutoff_output_frame);
    }
  }

  /// Computes the block importances for the current output frame.
  fn compute_block_importances(&mut self, output_frameno: u64) {
    // SEF don't need block importances.
    if self.frame_invariants[&output_frameno].show_existing_frame {
      return;
    }

    // Get a list of output_framenos that we want to propagate through.
    let output_framenos = self
      .frame_invariants
      .iter()
      .skip_while(move |(&key, _)| key < output_frameno)
      .filter(|(_, fi)| !fi.invalid && !fi.show_existing_frame)
      .map(|(&output_frameno, _)| output_frameno)
      .take(self.config.rdo_lookahead_frames + 1)
      .collect::<Vec<_>>();

    // The first one should be the current output frame.
    assert_eq!(output_framenos[0], output_frameno);

    // First, initialize them all with zeros.
    for output_frameno in output_framenos.iter() {
      let fi = self.frame_invariants.get_mut(output_frameno).unwrap();
      for x in fi.block_importances.iter_mut() {
        *x = 0.;
      }
    }

    // Now compute and propagate the block importances from the end. The
    // current output frame will get its block importances from the future
    // frames.
    for &output_frameno in output_framenos.iter().skip(1).rev() {
      // Remove fi from the map temporarily and put it back in in the end of
      // the iteration. This is required because we need to mutably borrow
      // referenced fis from the map, and that wouldn't be possible if this was
      // an active borrow.
      let fi = self.frame_invariants.remove(&output_frameno).unwrap();

      // TODO: see comment above about key frames not having references.
      if fi.frame_type == FrameType::KEY {
        self.frame_invariants.insert(output_frameno, fi);
        continue;
      }

      let frame = self.frame_q[&fi.input_frameno].as_ref().unwrap();

      // There can be at most 3 of these.
      let mut unique_indices = ArrayVec::<[_; 3]>::new();

      for (mv_index, &rec_index) in fi.ref_frames.iter().enumerate() {
        if !unique_indices.iter().any(|&(_, r)| r == rec_index) {
          unique_indices.push((mv_index, rec_index));
        }
      }

      // Compute and propagate the importance, split evenly between the
      // referenced frames.
      for &(mv_index, rec_index) in unique_indices.iter() {
        // Use rec_buffer here rather than lookahead_rec_buffer because
        // rec_buffer still contains the reference frames for the current frame
        // (it's only overwritten when the frame is encoded), while
        // lookahead_rec_buffer already contains reference frames for the next
        // frame (for the reference propagation to work correctly).
        let reference =
          fi.rec_buffer.frames[rec_index as usize].as_ref().unwrap();
        let reference_frame = &reference.frame;
        let reference_output_frameno = reference.output_frameno;

        // We should never use frame as its own reference.
        assert_ne!(reference_output_frameno, output_frameno);

        for y in 0..fi.h_in_imp_b {
          for x in 0..fi.w_in_imp_b {
            let mv = fi.lookahead_mvs[mv_index][y * 2][x * 2];

            // Coordinates of the top-left corner of the reference block, in MV
            // units.
            let (reference_x, reference_y) =
              compute_reference_coordinates(x, y, Some(mv));

            // Use a cached inter cost if it's available, otherwise compute it now.
            let reference_input_frameno = self
              .frame_invariants
              .get(&reference_output_frameno)
              .map(|fi| fi.input_frameno);
            let cached_inter_cost_idx =
              reference_input_frameno.and_then(|reference_input_frameno| {
                if fi.input_frameno > reference_input_frameno {
                  Some(fi.input_frameno - reference_input_frameno - 1)
                } else {
                  None
                }
              });
            let inter_cost = cached_inter_cost_idx
              .and_then(|idx| fi.lookahead_inter_costs.get(idx as usize))
              .map(|costs| costs[y * fi.w_in_imp_b + x])
              .unwrap_or_else(|| {
                compute_inter_costs(
                  &frame,
                  &reference_frame,
                  x,
                  y,
                  Some(mv),
                  self.config.bit_depth,
                )
              }) as f32;

            let intra_cost =
              fi.lookahead_intra_costs[y * fi.w_in_imp_b + x] as f32;

            let future_importance =
              fi.block_importances[y * fi.w_in_imp_b + x];

            let propagate_fraction = (1. - inter_cost / intra_cost).max(0.);
            let propagate_amount = (intra_cost + future_importance)
              * propagate_fraction
              / unique_indices.len() as f32;

            if let Some(reference_frame_block_importances) = self
              .frame_invariants
              .get_mut(&reference_output_frameno)
              .map(|fi| &mut fi.block_importances)
            {
              let mut propagate =
                |block_x_in_mv_units, block_y_in_mv_units, fraction| {
                  let x = block_x_in_mv_units / BLOCK_SIZE_IN_MV_UNITS;
                  let y = block_y_in_mv_units / BLOCK_SIZE_IN_MV_UNITS;

                  // TODO: propagate partially if the block is partially off-frame
                  // (possible on right and bottom edges)?
                  if x >= 0
                    && y >= 0
                    && (x as usize) < fi.w_in_imp_b
                    && (y as usize) < fi.h_in_imp_b
                  {
                    reference_frame_block_importances
                      [y as usize * fi.w_in_imp_b + x as usize] +=
                      propagate_amount * fraction;
                  }
                };

              // Coordinates of the top-left corner of the block intersecting the
              // reference block from the top-left.
              let top_left_block_x = (reference_x
                - if reference_x < 0 {
                  BLOCK_SIZE_IN_MV_UNITS - 1
                } else {
                  0
                })
                / BLOCK_SIZE_IN_MV_UNITS
                * BLOCK_SIZE_IN_MV_UNITS;
              let top_left_block_y = (reference_y
                - if reference_y < 0 {
                  BLOCK_SIZE_IN_MV_UNITS - 1
                } else {
                  0
                })
                / BLOCK_SIZE_IN_MV_UNITS
                * BLOCK_SIZE_IN_MV_UNITS;

              debug_assert!(reference_x >= top_left_block_x);
              debug_assert!(reference_y >= top_left_block_y);

              let top_right_block_x =
                top_left_block_x + BLOCK_SIZE_IN_MV_UNITS;
              let top_right_block_y = top_left_block_y;
              let bottom_left_block_x = top_left_block_x;
              let bottom_left_block_y =
                top_left_block_y + BLOCK_SIZE_IN_MV_UNITS;
              let bottom_right_block_x = top_right_block_x;
              let bottom_right_block_y = bottom_left_block_y;

              let top_left_block_fraction = ((top_right_block_x - reference_x)
                * (bottom_left_block_y - reference_y))
                as f32
                / BLOCK_AREA_IN_MV_UNITS as f32;

              propagate(
                top_left_block_x,
                top_left_block_y,
                top_left_block_fraction,
              );

              let top_right_block_fraction =
                ((reference_x + BLOCK_SIZE_IN_MV_UNITS - top_right_block_x)
                  * (bottom_left_block_y - reference_y))
                  as f32
                  / BLOCK_AREA_IN_MV_UNITS as f32;

              propagate(
                top_right_block_x,
                top_right_block_y,
                top_right_block_fraction,
              );

              let bottom_left_block_fraction = ((top_right_block_x
                - reference_x)
                * (reference_y + BLOCK_SIZE_IN_MV_UNITS - bottom_left_block_y))
                as f32
                / BLOCK_AREA_IN_MV_UNITS as f32;

              propagate(
                bottom_left_block_x,
                bottom_left_block_y,
                bottom_left_block_fraction,
              );

              let bottom_right_block_fraction =
                ((reference_x + BLOCK_SIZE_IN_MV_UNITS - top_right_block_x)
                  * (reference_y + BLOCK_SIZE_IN_MV_UNITS
                    - bottom_left_block_y)) as f32
                  / BLOCK_AREA_IN_MV_UNITS as f32;

              propagate(
                bottom_right_block_x,
                bottom_right_block_y,
                bottom_right_block_fraction,
              );
            }
          }
        }
      }

      self.frame_invariants.insert(output_frameno, fi);
    }

    // Get the final block importance values for the current output frame.
    if !output_framenos.is_empty() {
      let fi = self.frame_invariants.get_mut(&output_framenos[0]).unwrap();

      for y in 0..fi.h_in_imp_b {
        for x in 0..fi.w_in_imp_b {
          let intra_cost =
            fi.lookahead_intra_costs[y * fi.w_in_imp_b + x] as f32;

          let importance = &mut fi.block_importances[y * fi.w_in_imp_b + x];
          if intra_cost > 0. {
            *importance = (1. + *importance / intra_cost).log2();
          } else {
            *importance = 0.;
          }

          assert!(*importance >= 0.);
        }
      }

      #[cfg(feature = "dump_lookahead_data")]
      {
        let data = &fi.block_importances;
        use byteorder::{NativeEndian, WriteBytesExt};
        let mut buf = vec![];
        buf.write_u64::<NativeEndian>(fi.h_in_imp_b as u64).unwrap();
        buf.write_u64::<NativeEndian>(fi.w_in_imp_b as u64).unwrap();
        for y in 0..fi.h_in_imp_b {
          for x in 0..fi.w_in_imp_b {
            let importance = data[y * fi.w_in_imp_b + x];
            buf.write_f32::<NativeEndian>(importance).unwrap();
          }
        }
        ::std::fs::write(format!("{}-imps.bin", fi.input_frameno), buf)
          .unwrap();
      }
    }
  }

  pub fn receive_packet(&mut self) -> Result<Packet<T>, EncoderStatus> {
    if self.done_processing() {
      return Err(EncoderStatus::LimitReached);
    }

    if self.needs_more_lookahead_computed()
      || self.needs_more_frame_types_decided()
    {
      if self.is_flushing() {
        self.compute_lookahead_data();
      } else {
        return Err(EncoderStatus::NeedMoreData);
      }
    }

    // Find the next output_frameno corresponding to a non-skipped frame.
    self.output_frameno = self
      .frame_invariants
      .iter()
      .skip_while(|(&output_frameno, _)| output_frameno < self.output_frameno)
      .find(|(_, fi)| !fi.invalid)
      .map(|(&output_frameno, _)| output_frameno)
      .unwrap();

    let input_frameno =
      self.frame_invariants[&self.output_frameno].input_frameno;
    if !self.needs_more_frames(input_frameno) {
      return Err(EncoderStatus::LimitReached);
    }

    let cur_output_frameno = self.output_frameno;
    self.compute_block_importances(cur_output_frameno);

    let ret = {
      let fi = self.frame_invariants.get(&cur_output_frameno).unwrap();
      if fi.show_existing_frame {
        if !self.rc_state.ready() {
          return Err(EncoderStatus::NotReady);
        }
        let mut fs = FrameState::new(fi);

        let sef_data = encode_show_existing_frame(fi, &mut fs);
        let bits = (sef_data.len() * 8) as i64;
        self.packet_data.extend(sef_data);
        self.rc_state.update_state(
          bits,
          FRAME_SUBTYPE_SEF,
          fi.show_frame,
          0,
          false,
          false,
        );
        let rec = if fi.show_frame { Some(fs.rec) } else { None };
        self.output_frameno += 1;

        let input_frameno = fi.input_frameno;
        let frame_type = fi.frame_type;
        let bit_depth = fi.sequence.bit_depth;
        let qp = fi.base_q_idx;
        self.finalize_packet(
          rec,
          input_frameno,
          frame_type,
          bit_depth,
          qp,
          fs.enc_stats,
        )
      } else if let Some(f) = self.frame_q.get(&fi.input_frameno) {
        if !self.rc_state.ready() {
          return Err(EncoderStatus::NotReady);
        }
        if let Some(frame) = f.clone() {
          let fti = fi.get_frame_subtype();
          let qps = self.rc_state.select_qi(
            self,
            self.output_frameno,
            fti,
            self.maybe_prev_log_base_q,
          );
          let fi = self.frame_invariants.get_mut(&cur_output_frameno).unwrap();
          fi.set_quantizers(&qps);

          if self.rc_state.needs_trial_encode(fti) {
            let mut fs = FrameState::new_with_frame(fi, frame.clone());
            let data = encode_frame(fi, &mut fs);
            self.rc_state.update_state(
              (data.len() * 8) as i64,
              fti,
              fi.show_frame,
              qps.log_target_q,
              true,
              false,
            );
            let qps = self.rc_state.select_qi(
              self,
              self.output_frameno,
              fti,
              self.maybe_prev_log_base_q,
            );
            let fi =
              self.frame_invariants.get_mut(&cur_output_frameno).unwrap();
            fi.set_quantizers(&qps);
          }

          let fi = self.frame_invariants.get_mut(&cur_output_frameno).unwrap();
          let mut fs = FrameState::new_with_frame(fi, frame.clone());
          let data = encode_frame(fi, &mut fs);
          let enc_stats = fs.enc_stats.clone();
          self.maybe_prev_log_base_q = Some(qps.log_base_q);
          // TODO: Add support for dropping frames.
          self.rc_state.update_state(
            (data.len() * 8) as i64,
            fti,
            fi.show_frame,
            qps.log_target_q,
            false,
            false,
          );
          self.packet_data.extend(data);

          fs.rec.pad(fi.width, fi.height);

          // TODO avoid the clone by having rec Arc.
          let rec = if fi.show_frame { Some(fs.rec.clone()) } else { None };

          update_rec_buffer(self.output_frameno, fi, fs);

          // Copy persistent fields into subsequent FrameInvariants.
          let rec_buffer = fi.rec_buffer.clone();
          for subsequent_fi in self
            .frame_invariants
            .iter_mut()
            .skip_while(|(&output_frameno, _)| {
              output_frameno <= cur_output_frameno
            })
            .map(|(_, fi)| fi)
            // Here we want the next valid non-show-existing-frame inter frame.
            //
            // Copying to show-existing-frame frames isn't actually required
            // for correct encoding, but it's needed for the reconstruction to
            // work correctly.
            .filter(|fi| !fi.invalid)
            .take_while(|fi| fi.frame_type != FrameType::KEY)
          {
            subsequent_fi.rec_buffer = rec_buffer.clone();
            subsequent_fi.set_ref_frame_sign_bias();

            // Stop after the first non-show-existing-frame.
            if !subsequent_fi.show_existing_frame {
              break;
            }
          }

          let fi = self.frame_invariants.get(&self.output_frameno).unwrap();

          self.output_frameno += 1;

          if fi.show_frame {
            let input_frameno = fi.input_frameno;
            let frame_type = fi.frame_type;
            let bit_depth = fi.sequence.bit_depth;
            let qp = fi.base_q_idx;
            self.finalize_packet(
              rec,
              input_frameno,
              frame_type,
              bit_depth,
              qp,
              enc_stats,
            )
          } else {
            Err(EncoderStatus::Encoded)
          }
        } else {
          Err(EncoderStatus::NeedMoreData)
        }
      } else {
        Err(EncoderStatus::NeedMoreData)
      }
    };

    if let Ok(ref pkt) = ret {
      self.garbage_collect(pkt.input_frameno);
    }

    ret
  }

  fn finalize_packet(
    &mut self, rec: Option<Frame<T>>, input_frameno: u64,
    frame_type: FrameType, bit_depth: usize, qp: u8, enc_stats: EncoderStats,
  ) -> Result<Packet<T>, EncoderStatus> {
    let data = self.packet_data.clone();
    self.packet_data.clear();
    if write_temporal_delimiter(&mut self.packet_data).is_err() {
      return Err(EncoderStatus::Failure);
    }

    let mut psnr = None;
    if self.config.show_psnr {
      if let Some(ref rec) = rec {
        let original_frame = self.get_frame(input_frameno);
        psnr = Some(calculate_frame_psnr(&*original_frame, rec, bit_depth));
      }
    }

    self.frames_processed += 1;
    Ok(Packet { data, rec, input_frameno, frame_type, psnr, qp, enc_stats })
  }

  fn garbage_collect(&mut self, cur_input_frameno: u64) {
    let frame_q_start = self.frame_q.keys().next().cloned().unwrap_or(0);
    let delete_to = cur_input_frameno
      .saturating_sub(self.inter_cfg.keyframe_lookahead_distance() as u64);
    for i in frame_q_start..delete_to {
      self.frame_q.remove(&i);
    }

    let output_lookback = self.inter_cfg.keyframe_lookahead_distance() as u64
      * self.inter_cfg.group_output_len
      / self.inter_cfg.group_input_len;
    if self.output_frameno < output_lookback {
      return;
    }
    let fi_start = self.frame_invariants.keys().next().cloned().unwrap_or(0);
    for i in fi_start..(self.output_frameno - output_lookback) {
      self.frame_invariants.remove(&i);
      self.gop_output_frameno_start.remove(&i);
      self.gop_input_frameno_start.remove(&i);
    }
  }

  /// Counts the number of output frames of each subtype in the next
  ///  reservoir_frame_delay temporal units (needed for rate control).
  /// Returns the number of output frames (excluding SEF frames) and output TUs
  ///  until the last keyframe in the next reservoir_frame_delay temporal units,
  ///  or the end of the interval, whichever comes first.
  /// The former is needed because it indicates the number of rate estimates we
  ///  will make.
  /// The latter is needed because it indicates the number of times new bitrate
  ///  is added to the buffer.
  pub(crate) fn guess_frame_subtypes(
    &self, nframes: &mut [i32; FRAME_NSUBTYPES + 1],
    reservoir_frame_delay: i32,
  ) -> (i32, i32) {
    for fti in 0..=FRAME_NSUBTYPES {
      nframes[fti] = 0;
    }

    // Two-pass calls this function before receive_packet(), and in particular
    // before the very first send_frame(), when the following maps are empty.
    // In this case, return 0 as the default value.
    let mut prev_keyframe_input_frameno = *self
      .gop_input_frameno_start
      .get(&self.output_frameno)
      .unwrap_or_else(|| {
        assert!(self.output_frameno == 0);
        &0
      });
    let mut prev_keyframe_output_frameno = *self
      .gop_output_frameno_start
      .get(&self.output_frameno)
      .unwrap_or_else(|| {
        assert!(self.output_frameno == 0);
        &0
      });

    let mut prev_keyframe_ntus = 0;
    // Does not include SEF frames.
    let mut prev_keyframe_nframes = 0;
    let mut acc: [i32; FRAME_NSUBTYPES + 1] = [0; FRAME_NSUBTYPES + 1];
    // Updates the frame counts with the accumulated values when we hit a
    //  keyframe.
    fn collect_counts(
      nframes: &mut [i32; FRAME_NSUBTYPES + 1],
      acc: &mut [i32; FRAME_NSUBTYPES + 1],
    ) {
      for fti in 0..=FRAME_NSUBTYPES {
        nframes[fti] += acc[fti];
        acc[fti] = 0;
      }
      acc[FRAME_SUBTYPE_I] += 1;
    }
    let mut output_frameno = self.output_frameno;
    let mut ntus = 0;
    // Does not include SEF frames.
    let mut nframes_total = 0;
    while ntus < reservoir_frame_delay {
      let output_frameno_in_gop =
        output_frameno - prev_keyframe_output_frameno;
      let is_kf = if let Some(fi) = self.frame_invariants.get(&output_frameno)
      {
        if fi.frame_type == FrameType::KEY {
          prev_keyframe_input_frameno = fi.input_frameno;
          // We do not currently use forward keyframes, so they should always
          //  end the current TU (thus we always increment ntus below).
          debug_assert!(fi.show_frame);
          true
        } else {
          false
        }
      } else {
        // It is possible to be invoked for the first time from twopass_out()
        //  before receive_packet() is called, in which case frame_invariants
        //  will not be populated.
        // Force the first frame in each GOP to be a keyframe in that case.
        output_frameno_in_gop == 0
      };
      if is_kf {
        collect_counts(nframes, &mut acc);
        prev_keyframe_output_frameno = output_frameno;
        prev_keyframe_ntus = ntus;
        prev_keyframe_nframes = nframes_total;
        output_frameno += 1;
        ntus += 1;
        nframes_total += 1;
        continue;
      }
      let idx_in_group_output =
        self.inter_cfg.get_idx_in_group_output(output_frameno_in_gop);
      let input_frameno = prev_keyframe_input_frameno
        + self
          .inter_cfg
          .get_order_hint(output_frameno_in_gop, idx_in_group_output)
          as u64;
      // For rate control purposes, ignore any limit on frame count that has
      //  been set.
      // We pretend that we will keep encoding frames forever to prevent the
      //  control loop from driving us into the rails as we come up against a
      //  hard stop (with no more chance to correct outstanding errors).
      let next_keyframe_input_frameno =
        self.next_keyframe_input_frameno(prev_keyframe_input_frameno, true);
      // If we are re-ordering, we may skip some output frames in the final
      //  re-order group of the GOP.
      if input_frameno >= next_keyframe_input_frameno {
        // If we have encoded enough whole groups to reach the next keyframe,
        //  then start the next keyframe gop.
        if 1
          + (output_frameno - prev_keyframe_output_frameno)
            / self.inter_cfg.group_output_len
            * self.inter_cfg.group_input_len
          >= next_keyframe_input_frameno - prev_keyframe_input_frameno
        {
          collect_counts(nframes, &mut acc);
          prev_keyframe_input_frameno = input_frameno;
          prev_keyframe_output_frameno = output_frameno;
          prev_keyframe_ntus = ntus;
          prev_keyframe_nframes = nframes_total;
          // We do not currently use forward keyframes, so they should always
          //  end the current TU.
          output_frameno += 1;
          ntus += 1;
        }
        output_frameno += 1;
        continue;
      }
      if self.inter_cfg.get_show_existing_frame(idx_in_group_output) {
        acc[FRAME_SUBTYPE_SEF] += 1;
      } else {
        // TODO: Implement golden P-frames.
        let fti = FRAME_SUBTYPE_P
          + (self.inter_cfg.get_level(idx_in_group_output) as usize);
        acc[fti] += 1;
        nframes_total += 1;
      }
      if self.inter_cfg.get_show_frame(idx_in_group_output) {
        ntus += 1;
      }
      output_frameno += 1;
    }
    if prev_keyframe_output_frameno <= self.output_frameno {
      // If there were no keyframes at all, or only the first frame was a
      //  keyframe, the accumulators never flushed and still contain counts for
      //  the entire buffer.
      // In both cases, we return these counts.
      collect_counts(nframes, &mut acc);
      (nframes_total, ntus)
    } else {
      // Otherwise, we discard what remains in the accumulators as they contain
      //  the counts from and past the last keyframe.
      (prev_keyframe_nframes, prev_keyframe_ntus)
    }
  }
}
