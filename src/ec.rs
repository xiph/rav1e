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

use bitstream_io::{BitWriter, BE};
use std;

pub const OD_BITRES: u8 = 3;
const EC_PROB_SHIFT: u32 = 6;
const EC_MIN_PROB: u32 = 4;

pub struct Writer {
    enc: od_ec_enc,
    debug: bool,
}

pub type od_ec_window = u32;

#[derive(Debug)]
pub struct od_ec_enc {
    /// A buffer for output bytes with their associated carry flags.
    pub precarry: Vec<u16>,
    /// The low end of the current range.
    pub low: od_ec_window,
    /// The number of values in the current range.
    pub rng: u16,
    /// The number of bits of data in the current value.
    pub cnt: i16,
}

impl od_ec_enc {
    fn new() -> od_ec_enc {
        od_ec_enc {
            precarry: Vec::new(),
            low: 0,
            rng: 0x8000,
            // This is initialized to -9 so that it crosses zero after we've
            // accumulated one byte + one carry bit
            cnt: -9,
        }
    }

    /// Encode a single binary value.
    /// `val`: The value to encode (0 or 1).
    /// `f`: The probability that the val is one, scaled by 32768.
    fn od_ec_encode_bool_q15(&mut self, val: bool, f: u16) {
        assert!(0 < f);
        assert!(f < 32768);
        let mut l = self.low;
        let mut r = self.rng as u32;
        assert!(32768 <= r);

        let mut v = ((r >> 8) * (f as u32 >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT);
        v += EC_MIN_PROB;
        if val {
            l += r - v
        };
        r = if val {
            v
        } else {
            r - v
        };

        self.od_ec_enc_normalize(l, r as u16);
    }

    /// Encodes a symbol given a cumulative distribution function (CDF) table in Q15.
    /// `s`: The index of the symbol to encode.
    /// `cdf`: The CDF, such that symbol s falls in the range
    ///        `[s > 0 ? cdf[s - 1] : 0, cdf[s])`.
    ///       The values must be monotonically non-decreasing, and the last value
    ///       must be exactly 32768. There should be at most 16 values.
    fn od_ec_encode_cdf_q15(&mut self, s: usize, cdf: &[u16]) {
        assert!(cdf[cdf.len() - 1] == 0);
        let nsyms = cdf.len();
        self.od_ec_encode_q15(
            if s > 0 {
                cdf[s - 1]
            } else {
                32768
            },
            cdf[s],
            s,
            nsyms,
        );
    }

    /// Encodes a symbol given its frequency in Q15.
    /// `fl`: 32768 minus the cumulative frequency of all symbols that come
    ///       before the one to be encoded.
    /// `fh`: 32768 moinus the cumulative frequency of all symbols up to and
    ///       including the one to be encoded.
    fn od_ec_encode_q15(&mut self, fl: u16, fh: u16, s: usize, nsyms: usize) {
        let mut l = self.low;
        let mut r = self.rng as u32;
        let u: u32;
        let v: u32;
        assert!(32768 <= r);

        assert!(fh <= fl);
        assert!(fl <= 32768);
        let n = nsyms - 1;
        if fl < 32768 {
            u = (((r >> 8) * (fl as u32 >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) +
                EC_MIN_PROB * (n - (s - 1)) as u32;
            v = (((r >> 8) * (fh as u32 >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) +
                EC_MIN_PROB * (n - (s + 0)) as u32;
            l += r - u;
            r = u - v;
        } else {
            r -= (((r >> 8) * (fh as u32 >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) +
                EC_MIN_PROB * (n - (s + 0)) as u32;
        }

        self.od_ec_enc_normalize(l, r as u16);
    }

    fn od_ilog_nz(x: u16) -> u16 {
        16 - (x.leading_zeros() as u16)
    }

    /// Takes updated low and range values, renormalizes them so that
    /// 32768 <= `rng` < 65536 (flushing bytes from low to the pre-carry buffer if
    /// necessary), and stores them back in the encoder context.
    /// `low0`: The new value of low.
    /// `rng`: The new value of the range.
    fn od_ec_enc_normalize(&mut self, low0: od_ec_window, rng: u16) {
        let mut low = low0;
        let mut c = self.cnt;
        let d = 16 - od_ec_enc::od_ilog_nz(rng);
        let mut s = c + (d as i16);

        if s >= 0 {
            c += 16;
            let mut m = (1 << c) - 1;
            if s >= 8 {
                self.precarry.push((low >> c) as u16);
                low &= m;
                c -= 8;
                m >>= 8;
            }
            self.precarry.push((low >> c) as u16);
            s = c + (d as i16) - 24;
            low &= m;
        }
        self.low = low << d;
        self.rng = rng << d;
        self.cnt = s;
    }

    /// Indicates that there are no more symbols to encode.
    /// Returns a vector containing the final bitstream.
    fn od_ec_enc_done(&mut self) -> Vec<u8> {
        // We output the minimum number of bits that ensures that the symbols encoded
        // thus far will be decoded correctly regardless of the bits that follow.
        let l = self.low;
        let r = self.rng as u32;
        let mut c = self.cnt;
        let mut s = 9;
        let mut m = 0x7FFF;
        let mut e = (l + m) & !m;

        while (e | m) >= l + r {
            s += 1;
            m >>= 1;
            e = (l + m) & !m;
        }
        s += c;

        if s > 0 {
            let mut n = (1 << (c + 16)) - 1;

            loop {
                self.precarry.push((e >> (c + 16)) as u16);
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
        let mut offs = self.precarry.len();
        let mut out = vec![0 as u8; offs];
        while offs > 0 {
            offs -= 1;
            c = self.precarry[offs] + c;
            out[offs] = c as u8;
            c >>= 8;
        }

        out
    }

    /// Returns the number of bits "used" by the encoded symbols so far.
    /// This same number can be computed in either the encoder or the decoder, and is
    ///  suitable for making coding decisions.
    /// Return: The number of bits.
    ///         This will always be slightly larger than the exact value (e.g., all
    ///          rounding error is in the positive direction).
    pub fn od_ec_enc_tell(&mut self) -> u32 {
        // The 10 here counteracts the offset of -9 baked into cnt, and adds 1 extra
        // bit, which we reserve for terminating the stream.
        (((self.precarry.len() * 8) as i32) + (self.cnt as i32) + 10) as u32
    }

    /// Returns the number of bits "used" by the encoded symbols so far.
    /// This same number can be computed in either the encoder or the decoder, and is
    /// suitable for making coding decisions.
    /// Return: The number of bits scaled by `2**OD_BITRES`.
    ///         This will always be slightly larger than the exact value (e.g., all
    ///          rounding error is in the positive direction).
    pub fn od_ec_enc_tell_frac(&mut self) -> u32 {
        od_ec_enc::od_ec_tell_frac(self.od_ec_enc_tell(), self.rng as u32)
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
    fn od_ec_tell_frac(nbits_total: u32, mut rng: u32) -> u32 {
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
}

impl Writer {
    pub fn new() -> Writer {
        use std::env;
        Writer {
            enc: od_ec_enc::new(),
            debug: env::var_os("RAV1E_DEBUG").is_some(),
        }
    }
    pub fn done(&mut self) -> Vec<u8> {
        self.enc.od_ec_enc_done()
    }
    pub fn cdf(&mut self, s: u32, cdf: &[u16]) {
        self.enc.od_ec_encode_cdf_q15(s as usize, cdf)
    }
    pub fn bool(&mut self, val: bool, f: u16) {
        self.enc.od_ec_encode_bool_q15(val, f)
    }
    fn update_cdf(cdf: &mut [u16], val: u32, nsymbs: usize) {
        let nsymbs2speed: [usize; 17] = [ 0, 0, 1, 1, 2, 2, 2, 2, 2,
                                              2, 2, 2, 2, 2, 2, 2, 2 ];
        assert!(nsymbs < 17);
        let rate = 3 + (cdf[nsymbs] > 15) as usize + (cdf[nsymbs] > 31) as usize +
                   nsymbs2speed[nsymbs];  // + get_msb(nsymbs);
        let mut tmp = 32768;

        // Single loop (faster)
        for i in 0..(nsymbs - 1) {
            tmp = if i as u32 == val {
                0
            } else {
                tmp
            };
            if tmp < cdf[i] {
                cdf[i] -= (cdf[i] - tmp) >> rate;
            } else {
                cdf[i] += (tmp - cdf[i]) >> rate;
            }
        }
        cdf[nsymbs] += (cdf[nsymbs] < 32) as u16;
    }
    pub fn symbol(&mut self, s: u32, cdf: &mut [u16], nsymbs: usize) {
        if self.debug {
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
        self.cdf(s, &cdf[..nsymbs]);
        Writer::update_cdf(cdf, s, nsymbs);
    }
    pub fn bit(&mut self, bit: u16) {
        self.enc.od_ec_encode_bool_q15(bit == 1, 16384);
    }

    pub fn write_golomb(&mut self, level: u16) {
        let x = level + 1;
        let mut i = x;
        let mut length = 0;

        while i != 0 {
            i >>= 1;
            length += 1;
        }
        assert!(length > 0);

        for _ in 0..length - 1 {
            self.bit(0);
        }

        for i in (0..length).rev() {
            self.bit((x >> i) & 0x01);
        }
    }

    #[allow(dead_code)]
    pub fn tell(&mut self) -> u32 {
        self.enc.od_ec_enc_tell_frac()
    }

    pub fn tell_frac(&mut self) -> u32 {
        self.enc.od_ec_enc_tell_frac()
    }

    pub fn checkpoint(&mut self) -> WriterCheckpoint {
        WriterCheckpoint {
            precarry_len: self.enc.precarry.len(),
            low: self.enc.low,
            rng: self.enc.rng,
            cnt: self.enc.cnt,
        }
    }

    pub fn rollback(&mut self, checkpoint: &WriterCheckpoint) {
        self.enc.precarry.truncate(checkpoint.precarry_len);
        self.enc.low = checkpoint.low;
        self.enc.rng = checkpoint.rng;
        self.enc.cnt = checkpoint.cnt;
    }
}

pub trait BCodeWriter {
    fn recenter_nonneg(&mut self, r: u16, v: u16) -> u16;
    fn recenter_finite_nonneg(&mut self, n: u16, r: u16, v: u16) -> u16;
    fn write_quniform(&mut self, n: u16, v: u16) -> Result<(), std::io::Error>;
    fn write_subexpfin(&mut self, n: u16, k: u16, v: u16) -> Result<(), std::io::Error>;
    fn write_refsubexpfin(&mut self, n: u16, k: u16, r: i16, v: i16) -> Result<(), std::io::Error>;
    fn write_s_refsubexpfin(&mut self, n: u16, k: u16, r: i16, v: i16) -> Result<(), std::io::Error>;
}

impl<'a> BCodeWriter for BitWriter<'a, BE> {
    fn recenter_nonneg(&mut self, r: u16, v: u16) -> u16 {
        /* Recenters a non-negative literal v around a reference r */
        if v > (r << 1) {
            return v;
        } else if v >= r {
            return (v - r) << 1;
        } else {
            return ((r - v) << 1) - 1;
        }
    }
    fn recenter_finite_nonneg(&mut self, n: u16, r: u16, v: u16) -> u16 {
        /* Recenters a non-negative literal v in [0, n-1] around a
           reference r also in [0, n-1] */
        if (r << 1) <= n {
            return self.recenter_nonneg(r, v);
        } else {
            return self.recenter_nonneg(n - 1 - r, n - 1 - v);
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
            return self.write(l - 1, v);
        } else {
            self.write(l - 1, m + ((v - m) >> 1))?;
            return self.write_bit(((v - m) & 1) != 0);
        }
    }
    fn write_subexpfin(&mut self, n: u16, k: u16, v: u16) -> Result<(), std::io::Error> {
        /* Finite subexponential code that codes a symbol v in [0, n-1] with parameter k */
        let mut i = 0;
        let mut mk = 0;
        loop {
            let b = if i > 0 {
                k + i - 1
            } else {
                k
            };
            let a = 1 << b;
            if n <= mk + 3 * a {
                return self.write_quniform(n - mk, v - mk);
            } else {
                let t = v >= mk + a;
                self.write_bit(t)?;
                if t {
                    i = i + 1;
                    mk += a;
                } else {
                    return self.write(b as u32, v - mk);
                }
            }
        }
    }
    fn write_refsubexpfin(&mut self, n: u16, k: u16, r: i16, v: i16) -> Result<(), std::io::Error> {
        /* Finite subexponential code that codes a symbol v in [0, n-1] with
           parameter k based on a reference ref also in [0, n-1].
           Recenters symbol around r first and then uses a finite subexponential code. */
        let recentered_v = self.recenter_finite_nonneg(n, r as u16, v as u16);
        return self.write_subexpfin(n, k, recentered_v);
    }
    fn write_s_refsubexpfin(&mut self, n: u16, k: u16, r: i16, v: i16) -> Result<(), std::io::Error> {
        /* Signed version of the above function */
        return self.write_refsubexpfin((n << 1) - 1, k, r + (n - 1) as i16, v + (n - 1) as i16);
    }
}

#[derive(Clone)]
pub struct WriterCheckpoint {
    precarry_len: usize,
    low: od_ec_window,
    rng: u16,
    cnt: i16,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct od_ec_dec {
    pub buf: *const ::std::os::raw::c_uchar,
    pub eptr: *const ::std::os::raw::c_uchar,
    pub end_window: od_ec_window,
    pub nend_bits: ::std::os::raw::c_int,
    pub tell_offs: i32,
    pub end: *const ::std::os::raw::c_uchar,
    pub bptr: *const ::std::os::raw::c_uchar,
    pub dif: od_ec_window,
    pub rng: u16,
    pub cnt: i16,
    pub error: ::std::os::raw::c_int,
}

#[cfg(test)]
mod test {
    use super::*;
    use std::mem;

    struct Reader<'a> {
        dec: od_ec_dec,
        _dummy: &'a [u8],
    }

    extern "C" {
        fn od_ec_dec_init(dec: *mut od_ec_dec, buf: *const ::std::os::raw::c_uchar, storage: u32);
        fn od_ec_decode_bool_q15(dec: *mut od_ec_dec, f: ::std::os::raw::c_uint) -> ::std::os::raw::c_int;
        fn od_ec_decode_cdf_q15(dec: *mut od_ec_dec, cdf: *const u16, nsyms: ::std::os::raw::c_int) -> ::std::os::raw::c_int;
    }

    impl<'a> Reader<'a> {
        fn new(buf: &'a [u8]) -> Self {
            let mut r = Reader {
                dec: unsafe { mem::uninitialized() },
                _dummy: buf,
            };

            unsafe { od_ec_dec_init(&mut r.dec, buf.as_ptr(), buf.len() as u32) };

            r
        }

        fn bool(&mut self, f: u32) -> bool {
            unsafe { od_ec_decode_bool_q15(&mut self.dec, f) != 0 }
        }

        fn cdf(&mut self, icdf: &[u16]) -> i32 {
            let nsyms = icdf.len();
            unsafe { od_ec_decode_cdf_q15(&mut self.dec, icdf.as_ptr(), nsyms as i32) }
        }
    }

    #[test]
    fn booleans() {
        let mut w = Writer::new();

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

        let mut w = Writer::new();

        w.cdf(0, &cdf);
        w.cdf(0, &cdf);
        w.cdf(0, &cdf);
        w.cdf(1, &cdf);
        w.cdf(1, &cdf);
        w.cdf(1, &cdf);
        w.cdf(2, &cdf);
        w.cdf(2, &cdf);
        w.cdf(2, &cdf);

        let b = w.done();

        let mut r = Reader::new(&b);

        assert_eq!(r.cdf(&cdf), 0);
        assert_eq!(r.cdf(&cdf), 0);
        assert_eq!(r.cdf(&cdf), 0);
        assert_eq!(r.cdf(&cdf), 1);
        assert_eq!(r.cdf(&cdf), 1);
        assert_eq!(r.cdf(&cdf), 1);
        assert_eq!(r.cdf(&cdf), 2);
        assert_eq!(r.cdf(&cdf), 2);
        assert_eq!(r.cdf(&cdf), 2);
    }

    #[test]
    fn mixed() {
        let cdf = [7296, 3819, 1716, 0];

        let mut w = Writer::new();

        w.cdf(0, &cdf);
        w.bool(true, 2);
        w.cdf(0, &cdf);
        w.bool(true, 2);
        w.cdf(0, &cdf);
        w.bool(true, 2);
        w.cdf(1, &cdf);
        w.bool(true, 1);
        w.cdf(1, &cdf);
        w.bool(false, 2);
        w.cdf(1, &cdf);
        w.cdf(2, &cdf);
        w.cdf(2, &cdf);
        w.cdf(2, &cdf);

        let b = w.done();

        let mut r = Reader::new(&b);

        assert_eq!(r.cdf(&cdf), 0);
        assert_eq!(r.bool(2), true);
        assert_eq!(r.cdf(&cdf), 0);
        assert_eq!(r.bool(2), true);
        assert_eq!(r.cdf(&cdf), 0);
        assert_eq!(r.bool(2), true);
        assert_eq!(r.cdf(&cdf), 1);
        assert_eq!(r.bool(1), true);
        assert_eq!(r.cdf(&cdf), 1);
        assert_eq!(r.bool(2), false);
        assert_eq!(r.cdf(&cdf), 1);
        assert_eq!(r.cdf(&cdf), 2);
        assert_eq!(r.cdf(&cdf), 2);
        assert_eq!(r.cdf(&cdf), 2);
    }
}
