// Copyright (c) 2001-2016, Alliance for Open Media. All rights reserved
// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(non_camel_case_types)]
#![cfg_attr(feature = "cargo-clippy", allow(cast_lossless))]
#![cfg_attr(feature = "cargo-clippy", allow(identity_op))]
#![cfg_attr(feature = "cargo-clippy", allow(needless_range_loop))]

use bitstream_io::{BitWriter, BE};
use std;

pub const OD_BITRES: u8 = 3;
const EC_PROB_SHIFT: u32 = 6;
const EC_MIN_PROB: u32 = 4;
type ec_window = u32;

/// Public trait interface to a bitstream writer; can be used either
/// to record tokens for later writing (using a new::WriterRecorder()
/// as a Writer) or to write atual final bits out using a range
/// encoder (using a new::WriterEncoder() as a Writer).  A
/// WriterRecorder's contents can be replayed into a WriterEncoder.
pub trait Writer {
  /// Write a symbol s, using the passed in cdf reference; leaves cdf unchanged
  fn symbol(&mut self, s: u32, cdf: &[u16]);
  /// Write a symbol s, using the passed in cdf reference; updates the referenced cdf.
  fn symbol_with_update(&mut self, s: u32, cdf: &mut [u16]);
  /// Write a bool using passed in probability
  fn bool(&mut self, val: bool, f: u16);
  /// Write a single bit with flat proability
  fn bit(&mut self, bit: u16);
  /// Write literal bits with flat probability
  fn literal(&mut self, bits: u8, s: u32);
  /// Write passed level as a golomb code
  fn write_golomb(&mut self, level: u16);
  /// Return current length of range-coded bitsream in integer bits
  fn tell(&mut self) -> u32;
  /// Return currrent length of range-coded bitsream in fractional
  /// bits with OD_BITRES decimal precision
  fn tell_frac(&mut self) -> u32;
  /// Save current point in coding/recording to a checkpoint  
  fn checkpoint(&mut self) -> WriterCheckpoint;
  /// Restore saved position in coding/recording from a checkpoint  
  fn rollback(&mut self, &WriterCheckpoint);
}

/// StorageBackend is an internal trait used to tie a specific Writer
/// implementation's storage to the generic Writer.  It would be
/// private, but Rust is deprecating 'private trait in a public
/// interface' support.
pub trait StorageBackend {
  /// Store partially-computed range code into given storage backend
  fn store(&mut self, l: ec_window, r: u16);
  /// Return byte-length of encoded stream to date  
  fn stream_bytes(&mut self) -> usize;
  /// Backend implenetaiton of checkpoint to pass through Writer interface  
  fn checkpoint(&mut self) -> WriterCheckpoint;
  /// Backend implenetaiton of rollback to pass through Writer interface  
  fn rollback(&mut self, &WriterCheckpoint);
}

#[derive(Debug, Clone)]
pub struct WriterBase<S> {
  /// The number of values in the current range.
  rng: u16,
  /// The number of bits of data in the current value.
  cnt: i16,
  /// Debug enable flag
  debug: bool,
  /// Use-specific storage
  s: S
}

#[derive(Debug, Clone)]
pub struct WriterRecorder {
  /// Storage for tokens
  storage: Vec<(ec_window, u16)>,
  /// Bytes that would be shifted out to date
  bytes: usize  
}

#[derive(Debug, Clone)]
pub struct WriterEncoder {
  /// A buffer for output bytes with their associated carry flags.
  precarry: Vec<u16>,
  /// The low end of the current range.
  low: ec_window,
}

#[derive(Clone)]
pub struct WriterCheckpoint {
  /// Byte length coded/recorded to date  
  stream_bytes: usize,
  /// To be defined by backend  
  backend_var: usize,
  /// Saved number of values in the current range.
  rng: u16,
  /// Saved number of bits of data in the current value.
  cnt: i16,
}

/// Constructor for a recording Writer
impl WriterRecorder {
  pub fn new () -> WriterBase<WriterRecorder> {
    WriterBase::new(WriterRecorder {
      storage: Vec::new(),
      bytes: 0
    })
  }
}

/// Constructor for a encoding Writer
impl WriterEncoder {
  pub fn new () -> WriterBase<WriterEncoder> {
    WriterBase::new(WriterEncoder {
      precarry: Vec::new(),
      low: 0
    })
  }
}

/// The Recorder does not produce a range-coded bitstream, but it
/// still tracks the range coding progress like in an Encoder, as it
/// neds to be able to report bit costs for RDO decsions.  It stores a
/// pair of mostly-computed range coding values per token recorded.
impl StorageBackend for WriterBase<WriterRecorder> {
  fn store(&mut self, l: ec_window, r: u16) {
    let d = 16 - WriterBase::<Self>::ilog_nz(r);
    let mut c = self.cnt;
    let mut s = c + (d as i16);

    if s >= 0 {
      c += 16;
      if s >= 8 {
        self.s.bytes += 1;
        c -= 8;
      }
      self.s.bytes += 1;
      s = c + (d as i16) - 24;
    }
    self.rng = r << d;
    self.cnt = s;
    self.s.storage.push((l,r));
  }
  fn stream_bytes(&mut self) -> usize {
      self.s.bytes
  }
  fn checkpoint(&mut self) -> WriterCheckpoint {
    WriterCheckpoint {
      stream_bytes: self.s.bytes,
      backend_var: self.s.storage.len(),
      rng: self.rng,
      cnt: self.cnt,
    }
  }    
  fn rollback(&mut self, checkpoint: &WriterCheckpoint) {
    self.rng = checkpoint.rng;
    self.cnt = checkpoint.cnt;
    self.s.bytes = checkpoint.stream_bytes;
    self.s.storage.truncate(checkpoint.backend_var);
  }
}

/// An Encoder produces an actual range-coded bitstream from passed in
/// tokens.  It does not retain any information about the coded
/// tokens, only the resulting bitstream, and so it cannot be replayed
/// (only checkpointed and rolled back).
impl StorageBackend for WriterBase<WriterEncoder> {
  fn store(&mut self, l: ec_window, r: u16) {
    let mut low = l + self.s.low;
    let mut c = self.cnt;
    let d = 16 - WriterBase::<Self>::ilog_nz(r);
    let mut s = c + (d as i16);

    if s >= 0 {
      c += 16;
      let mut m = (1 << c) - 1;
      if s >= 8 {
        self.s.precarry.push((low >> c) as u16);
        low &= m;
        c -= 8;
        m >>= 8;
      }
      self.s.precarry.push((low >> c) as u16);
      s = c + (d as i16) - 24;
      low &= m;
    }
    self.s.low = low << d;
    self.rng = r << d;
    self.cnt = s;
  }
  fn stream_bytes(&mut self) -> usize {
      self.s.precarry.len()
  }
  fn checkpoint(&mut self) -> WriterCheckpoint {
    WriterCheckpoint {
        stream_bytes: self.s.precarry.len(),
        backend_var: self.s.low as usize,
        rng: self.rng,
        cnt: self.cnt,
    }
  }
  fn rollback(&mut self, checkpoint: &WriterCheckpoint) {
    self.rng = checkpoint.rng;
    self.cnt = checkpoint.cnt;
    self.s.low = checkpoint.backend_var as ec_window;
    self.s.precarry.truncate(checkpoint.stream_bytes);
  }
}

/// A few local helper functions needed by the Writer that are not
/// part of the public interface.
impl<S> WriterBase<S>{
  /// Internal constructor called by the subtypes that implement the
  /// actual encoder and Recorder.
  fn new(storage: S) -> Self {
    WriterBase {
      rng: 0x8000,
      cnt: -9,
      debug: false, 
      s: storage
    }
  }
  /// Given the current total integer number of bits used and the current value of
  /// rng, computes the fraction number of bits used to `OD_BITRES` precision.
  /// This is used by `od_ec_enc_tell_frac()` and `od_ec_dec_tell_frac()`.
  /// `nbits_total`: The number of whole bits currently used, i.e., the value
  ///                returned by `od_ec_enc_tell()` or `od_ec_dec_tell()`.
  /// `rng`: The current value of rng from either the encoder or decoder state.
  /// Return: The number of bits scaled by `2**OD_BITRES`.
  ///         This will always be slightly larger than the exact value (e.g., all
  ///         rounding error is in the positive direction).
  fn frac_compute(nbits_total: u32, mut rng: u32) -> u32 {
    // To handle the non-integral number of bits still left in the encoder/decoder
    //  state, we compute the worst-case number of bits of val that must be
    //  encoded to ensure that the value is inside the range for any possible
    //  subsequent bits.
    // The computation here is independent of val itself (the decoder does not
    //  even track that value), even though the real number of bits used after
    //  od_ec_enc_done() may be 1 smaller if rng is a power of two and the
    //  corresponding trailing bits of val are all zeros.
    // If we did try to track that special case, then coding a value with a
    //  probability of 1/(1 << n) might sometimes appear to use more than n bits.
    // This may help explain the surprising result that a newly initialized
    //  encoder or decoder claims to have used 1 bit.
    let nbits = nbits_total << OD_BITRES;
    let mut l = 0;
    let mut i = OD_BITRES;
    while i > 0 {
      i -= 1;
      rng = (rng * rng) >> 15;
      let b = rng >> 16;
      l = (l << 1) | b;
      rng >>= b;
    }
    nbits - l
  }
  /// Simple calculation of position of leading 1 bit.
  fn ilog_nz(x: u16) -> u16 {
    16 - (x.leading_zeros() as u16)
  }
  /// Function to update the CDF for Writer calls that do so.
  fn update_cdf(cdf: &mut [u16], val: u32) {
    let nsymbs = cdf.len() - 1;
    let nsymbs2speed: [usize; 17] =
      [0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
    debug_assert!(nsymbs < 17);
    let rate = 3
      + (cdf[nsymbs] > 15) as usize
      + (cdf[nsymbs] > 31) as usize
      + nsymbs2speed[nsymbs]; // + get_msb(nsymbs);
    let mut tmp = 32768;

    // Single loop (faster)
    for (i, v) in cdf[..nsymbs - 1].iter_mut().enumerate() {
      tmp = if i as u32 == val { 0 } else { tmp };
      if tmp < *v {
        *v -= (*v - tmp) >> rate;
      } else {
        *v += (tmp - *v) >> rate;
      }
    }
    cdf[nsymbs] += (cdf[nsymbs] < 32) as u16;
  }

  #[cfg(debug)]
  fn print_backtrace(&self, s: u32) {
    use backtrace;
    let mut depth = 3;
    backtrace::trace(|frame| {
      let ip = frame.ip();

      depth -= 1;

      if depth == 0 {
        backtrace::resolve(ip, |symbol| {
          if let Some(name) = symbol.name() {
            eprintln!("Writing symbol {} from {}", s, name);
          }
        });
        false
      } else {
        true
      }
    });
  }
}

/// Replay implemenetation specific to the Recorder
impl WriterBase<WriterRecorder> {
  /// Replays the partiall-computer range tokens out of the Recorder's
  /// storage and into the passed in Writer, which may be an Encoder
  /// or another Recorder.  Clears the Recorder after replay.
  pub fn replay(&mut self, dest: &mut StorageBackend) {
    for i in 0..self.s.storage.len() {
      let (l, r) = self.s.storage[i];
      dest.store(l, r);
    }
    self.s.storage.truncate(0);
    self.s.bytes = 0;
  }
}

/// Done implementation specific to the Encoder
impl WriterBase<WriterEncoder> {
  /// Indicates that there are no more symbols to encode.  Flushes
  /// remaining state into coding and returns a vector containing the
  /// final bitstream.
  pub fn done(&mut self) -> Vec<u8> {
    // We output the minimum number of bits that ensures that the symbols encoded
    // thus far will be decoded correctly regardless of the bits that follow.
    let l = self.s.low;
    let mut c = self.cnt;
    let mut s = 10;
    let m = 0x3FFF;
    let mut e = ((l + m) & !m) | (m + 1);

    s += c;

    if s > 0 {
      let mut n = (1 << (c + 16)) - 1;

      loop {
        self.s.precarry.push((e >> (c + 16)) as u16);
        e &= n;
        s -= 8;
        c -= 8;
        n >>= 8;

        if s <= 0 {
          break;
        }
      }
    }

    let mut c = 0;
    let mut offs = self.s.precarry.len();
    let mut out = vec![0 as u8; offs];
    while offs > 0 {
      offs -= 1;
      c += self.s.precarry[offs];
      out[offs] = c as u8;
      c >>= 8;
    }

    out
  }
}

/// Generic/shared implementation for Writers with StorageBackends (ie, Encoders and Recorders)
impl<S> Writer for WriterBase<S> where WriterBase<S>: StorageBackend {
  /// Encode a single binary value.
  /// `val`: The value to encode (0 or 1).
  /// `f`: The probability that the val is one, scaled by 32768.
  fn bool(&mut self, val: bool, f: u16) {
    debug_assert!(0 < f);
    debug_assert!(f < 32768);
    self.symbol(if val {1} else {0}, &[f, 0]);
  }
  /// Encode a single boolean value.
  /// `val`: The value to encode (false or true).
  /// `f`: The probability that the val is true, scaled by 32768.
  fn bit(&mut self, bit: u16) {
    self.bool(bit == 1, 16384);
  }
  /// Encode a literal bitstring, bit by bit in MSB order, with flat
  /// probability.
  /// 'bits': Length of bitstring
  /// 's': Bit string to encode
  fn literal(&mut self, bits: u8, s: u32) {
    for bit in (0..bits).rev() {
      self.bit((1 & (s >> bit)) as u16);
    }
  }
  /// Encodes a symbol given a cumulative distribution function (CDF) table in Q15.
  /// `s`: The index of the symbol to encode.
  /// `cdf`: The CDF, such that symbol s falls in the range
  ///        `[s > 0 ? cdf[s - 1] : 0, cdf[s])`.
  ///       The values must be monotonically non-decreasing, and the last value
  ///       must be exactly 32768. There should be at most 16 values.
  fn symbol(&mut self, s: u32, cdf: &[u16]) {
    debug_assert!(cdf[cdf.len() - 1] == 0);
    let nms =  cdf.len() - s as usize;
    let fl = if s > 0 { cdf[s as usize - 1] } else { 32768 };
    let fh = cdf[s as usize];
    debug_assert!(fh <= fl);
    debug_assert!(fl <= 32768);

    let u: u32;
    let v: u32;
    let mut l = 0;
    let mut r = self.rng as u32;
    debug_assert!(32768 <= r); 
    if fl < 32768 {
      u = (((r >> 8) * (fl as u32 >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
            + EC_MIN_PROB * nms as u32;
      v = (((r >> 8) * (fh as u32 >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
            + EC_MIN_PROB * (nms - 1) as u32;
      l += r - u;
      r = u - v;
    } else {
        r -= (((r >> 8) * (fh as u32 >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
            + EC_MIN_PROB * (nms - 1) as u32;
    }
    self.store(l, r as u16);
  }
  /// Encodes a symbol given a cumulative distribution function (CDF)
  /// table in Q15, then updates the CDF probabilities to relect we've
  /// written one more symbol 's'.  
  /// `s`: The index of the symbol to encode.
  /// `cdf`: The CDF, such that symbol s falls in the range
  ///        `[s > 0 ? cdf[s - 1] : 0, cdf[s])`.
  ///       The values must be monotonically non-decreasing, and the last value
  ///       must be exactly 32768. There should be at most 16 values.
  fn symbol_with_update(&mut self, s: u32, cdf: &mut [u16]) {
    let nsymbs = cdf.len() - 1;
    #[cfg(debug)]
    {
      if self.debug {
        self.print_backtrace(s);
      }
    }
    self.symbol(s, &cdf[..nsymbs]);
    Self::update_cdf(cdf, s);
  }
  /// Encode a golomb to the bitstream.
  /// 'level': passed in value to encode
  fn write_golomb(&mut self, level: u16) {
    let x = level + 1;
    let mut i = x;
    let mut length = 0;

    while i != 0 {
      i >>= 1;
      length += 1;
    }
    debug_assert!(length > 0);

    for _ in 0..length - 1 {
      self.bit(0);
    }

    for i in (0..length).rev() {
      self.bit((x >> i) & 0x01);
    }
  }
  /// Returns the number of bits "used" by the encoded symbols so far.
  /// This same number can be computed in either the encoder or the
  /// decoder, and is suitable for making coding decisions.  The value
  /// will be the same whether using an Encoder or Recorder.
  /// Return: The integer number of bits.
  ///         This will always be slightly larger than the exact value (e.g., all
  ///          rounding error is in the positive direction).
  fn tell(&mut self) -> u32 {
    // The 10 here counteracts the offset of -9 baked into cnt, and adds 1 extra
    // bit, which we reserve for terminating the stream.
    (((self.stream_bytes() * 8) as i32) + (self.cnt as i32) + 10) as u32
  }
  /// Returns the number of bits "used" by the encoded symbols so far.
  /// This same number can be computed in either the encoder or the
  /// decoder, and is suitable for making coding decisions. The value
  /// will be the same whether using an Encoder or Recorder.
  /// Return: The number of bits scaled by `2**OD_BITRES`.
  ///         This will always be slightly larger than the exact value (e.g., all
  ///          rounding error is in the positive direction).
  fn tell_frac(&mut self) -> u32 {
    Self::frac_compute(self.tell(), self.rng as u32)
  }
  /// Save current point in coding/recording to a checkpoint that can
  /// be restored later.  A WriterCheckpoint can be generated for an
  /// Encoder or Recorder, but can only be used to rollback the Writer
  /// instance from which it was generated.
  fn checkpoint (&mut self) -> WriterCheckpoint {
    StorageBackend::checkpoint(self)
  }
  /// Roll back a given Writer to the state saved in the WriterCheckpoint
  /// 'wc': Saved Writer state/posiiton to restore  
  fn rollback (&mut self, wc: &WriterCheckpoint) {
    StorageBackend::rollback(self,wc)
  }
}

pub trait BCodeWriter {
  fn recenter_nonneg(&mut self, r: u16, v: u16) -> u16;
  fn recenter_finite_nonneg(&mut self, n: u16, r: u16, v: u16) -> u16;
  fn write_quniform(&mut self, n: u16, v: u16) -> Result<(), std::io::Error>;
  fn write_subexpfin(
    &mut self, n: u16, k: u16, v: u16
  ) -> Result<(), std::io::Error>;
  fn write_refsubexpfin(
    &mut self, n: u16, k: u16, r: i16, v: i16
  ) -> Result<(), std::io::Error>;
  fn write_s_refsubexpfin(
    &mut self, n: u16, k: u16, r: i16, v: i16
  ) -> Result<(), std::io::Error>;
}

impl<'a> BCodeWriter for BitWriter<'a, BE> {
  fn recenter_nonneg(&mut self, r: u16, v: u16) -> u16 {
    /* Recenters a non-negative literal v around a reference r */
    if v > (r << 1) {
      v
    } else if v >= r {
      (v - r) << 1
    } else {
      ((r - v) << 1) - 1
    }
  }
  fn recenter_finite_nonneg(&mut self, n: u16, r: u16, v: u16) -> u16 {
    /* Recenters a non-negative literal v in [0, n-1] around a
           reference r also in [0, n-1] */
    if (r << 1) <= n {
      self.recenter_nonneg(r, v)
    } else {
      self.recenter_nonneg(n - 1 - r, n - 1 - v)
    }
  }
  fn write_quniform(&mut self, n: u16, v: u16) -> Result<(), std::io::Error> {
    /* Encodes a value v in [0, n-1] quasi-uniformly */
    if n <= 1 {
      return Ok(());
    };
    let l = 31 ^ ((n - 1) + 1).leading_zeros();
    let m = (1 << l) - n;
    if v < m {
      self.write(l - 1, v)
    } else {
      self.write(l - 1, m + ((v - m) >> 1))?;
      self.write_bit(((v - m) & 1) != 0)
    }
  }
  fn write_subexpfin(
    &mut self, n: u16, k: u16, v: u16
  ) -> Result<(), std::io::Error> {
    /* Finite subexponential code that codes a symbol v in [0, n-1] with parameter k */
    let mut i = 0;
    let mut mk = 0;
    loop {
      let b = if i > 0 { k + i - 1 } else { k };
      let a = 1 << b;
      if n <= mk + 3 * a {
        return self.write_quniform(n - mk, v - mk);
      } else {
        let t = v >= mk + a;
        self.write_bit(t)?;
        if t {
          i += 1;
          mk += a;
        } else {
          return self.write(b as u32, v - mk);
        }
      }
    }
  }
  fn write_refsubexpfin(
    &mut self, n: u16, k: u16, r: i16, v: i16
  ) -> Result<(), std::io::Error> {
    /* Finite subexponential code that codes a symbol v in [0, n-1] with
           parameter k based on a reference ref also in [0, n-1].
           Recenters symbol around r first and then uses a finite subexponential code. */
    let recentered_v = self.recenter_finite_nonneg(n, r as u16, v as u16);
    self.write_subexpfin(n, k, recentered_v)
  }
  fn write_s_refsubexpfin(
    &mut self, n: u16, k: u16, r: i16, v: i16
  ) -> Result<(), std::io::Error> {
    /* Signed version of the above function */
    self.write_refsubexpfin(
      (n << 1) - 1,
      k,
      r + (n - 1) as i16,
      v + (n - 1) as i16
    )
  }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct od_ec_dec {
  pub buf: *const ::std::os::raw::c_uchar,
  pub eptr: *const ::std::os::raw::c_uchar,
  pub end_window: ec_window,
  pub nend_bits: ::std::os::raw::c_int,
  pub tell_offs: i32,
  pub end: *const ::std::os::raw::c_uchar,
  pub bptr: *const ::std::os::raw::c_uchar,
  pub dif: ec_window,
  pub rng: u16,
  pub cnt: i16,
  pub error: ::std::os::raw::c_int
}

#[cfg(test)]
mod test {
  use super::*;
  use std::mem;

  struct Reader<'a> {
    dec: od_ec_dec,
    _dummy: &'a [u8]
  }

  extern {
    fn od_ec_dec_init(
      dec: *mut od_ec_dec, buf: *const ::std::os::raw::c_uchar, storage: u32
    );
    fn od_ec_decode_bool_q15(
      dec: *mut od_ec_dec, f: ::std::os::raw::c_uint
    ) -> ::std::os::raw::c_int;
    fn od_ec_decode_cdf_q15(
      dec: *mut od_ec_dec, cdf: *const u16, nsyms: ::std::os::raw::c_int
    ) -> ::std::os::raw::c_int;
  }

  impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
      let mut r = Reader { dec: unsafe { mem::uninitialized() }, _dummy: buf };

      unsafe { od_ec_dec_init(&mut r.dec, buf.as_ptr(), buf.len() as u32) };

      r
    }

    fn bool(&mut self, f: u32) -> bool {
      unsafe { od_ec_decode_bool_q15(&mut self.dec, f) != 0 }
    }

    fn symbol(&mut self, icdf: &[u16]) -> i32 {
      let nsyms = icdf.len();
      unsafe {
        od_ec_decode_cdf_q15(&mut self.dec, icdf.as_ptr(), nsyms as i32)
      }
    }
  }

  #[test]
  fn booleans() {
    let mut w = WriterEncoder::new();

    w.bool(false, 1);
    w.bool(true, 2);
    w.bool(false, 3);
    w.bool(true, 1);
    w.bool(true, 2);
    w.bool(false, 3);

    let b = w.done();

    let mut r = Reader::new(&b);

    assert_eq!(r.bool(1), false);
    assert_eq!(r.bool(2), true);
    assert_eq!(r.bool(3), false);
    assert_eq!(r.bool(1), true);
    assert_eq!(r.bool(2), true);
    assert_eq!(r.bool(3), false);
  }

  #[test]
  fn cdf() {
    let cdf = [7296, 3819, 1716, 0];

    let mut w = WriterEncoder::new();

    w.symbol(0, &cdf);
    w.symbol(0, &cdf);
    w.symbol(0, &cdf);
    w.symbol(1, &cdf);
    w.symbol(1, &cdf);
    w.symbol(1, &cdf);
    w.symbol(2, &cdf);
    w.symbol(2, &cdf);
    w.symbol(2, &cdf);

    let b = w.done();

    let mut r = Reader::new(&b);

    assert_eq!(r.symbol(&cdf), 0);
    assert_eq!(r.symbol(&cdf), 0);
    assert_eq!(r.symbol(&cdf), 0);
    assert_eq!(r.symbol(&cdf), 1);
    assert_eq!(r.symbol(&cdf), 1);
    assert_eq!(r.symbol(&cdf), 1);
    assert_eq!(r.symbol(&cdf), 2);
    assert_eq!(r.symbol(&cdf), 2);
    assert_eq!(r.symbol(&cdf), 2);
  }

  #[test]
  fn mixed() {
    let cdf = [7296, 3819, 1716, 0];

    let mut w = WriterEncoder::new();

    w.symbol(0, &cdf);
    w.bool(true, 2);
    w.symbol(0, &cdf);
    w.bool(true, 2);
    w.symbol(0, &cdf);
    w.bool(true, 2);
    w.symbol(1, &cdf);
    w.bool(true, 1);
    w.symbol(1, &cdf);
    w.bool(false, 2);
    w.symbol(1, &cdf);
    w.symbol(2, &cdf);
    w.symbol(2, &cdf);
    w.symbol(2, &cdf);

    let b = w.done();

    let mut r = Reader::new(&b);

    assert_eq!(r.symbol(&cdf), 0);
    assert_eq!(r.bool(2), true);
    assert_eq!(r.symbol(&cdf), 0);
    assert_eq!(r.bool(2), true);
    assert_eq!(r.symbol(&cdf), 0);
    assert_eq!(r.bool(2), true);
    assert_eq!(r.symbol(&cdf), 1);
    assert_eq!(r.bool(1), true);
    assert_eq!(r.symbol(&cdf), 1);
    assert_eq!(r.bool(2), false);
    assert_eq!(r.symbol(&cdf), 1);
    assert_eq!(r.symbol(&cdf), 2);
    assert_eq!(r.symbol(&cdf), 2);
    assert_eq!(r.symbol(&cdf), 2);
  }
}
