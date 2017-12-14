// Copyright (c) 2001-2016, Alliance for Open Media. All rights reserved
// Copyright (c) 2017, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(non_camel_case_types)]

pub const OD_BITRES: u8 = 3;

pub struct Writer {
    enc: od_ec_enc
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

        let v = ((r >> 8) * (f as u32)) >> 7;
        if val { l += r - v };
        r = if val { v } else { r - v };

        self.od_ec_enc_normalize(l, r as u16);
    }

    /// Encodes a symbol given a cumulative distribution function (CDF) table in Q15.
    /// `s`: The index of the symbol to encode.
    /// `cdf`: The CDF, such that symbol s falls in the range
    ///        `[s > 0 ? cdf[s - 1] : 0, cdf[s])`.
    ///       The values must be monotonically non-decreasing, and the last value
    ///       must be exactly 32768. There should be at most 16 values.
    fn od_ec_encode_cdf_q15(&mut self, s: usize, cdf: &[u16]) {
        assert!(cdf[cdf.len() - 1] == od_ec_enc::od_icdf(32768));
        self.od_ec_encode_q15(if s > 0 { cdf[s - 1] } else { od_ec_enc::od_icdf(0) }, cdf[s]);
    }

    fn od_icdf(x: u16) -> u16 {
        32768 - x
    }

    /// Encodes a symbol given its frequency in Q15.
    /// `fl`: 32768 minus the cumulative frequency of all symbols that come
    ///       before the one to be encoded.
    /// `fh`: 32768 moinus the cumulative frequency of all symbols up to and
    ///       including the one to be encoded.
    fn od_ec_encode_q15(&mut self, fl: u16, fh: u16) {
        let mut l = self.low;
        let mut r = self.rng as u32;
        let u: u32;
        let v: u32;
        assert!(32768 <= r);

        assert!(fh < fl);
        assert!(fl <= 32768);
        if fl < 32768 {
            u = ((r >> 8) * (fl as u32)) >> 7;
            v = ((r >> 8) * (fh as u32)) >> 7;
            l += r - u;
            r = u - v;
        } else {
            r -= ((r >> 8) * (fh as u32)) >> 7;
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

                if s <= 0 { break; }
            };
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
        Writer { enc: od_ec_enc::new() }
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
        let rate = 4 + if cdf[nsymbs] > 31 { 1 } else { 0 } + (31 ^ (nsymbs as u32).leading_zeros());
        let rate2 = 5;
        let mut tmp: i32;
        let diff: i32;
        let tmp0 = 1 << rate2;
        tmp = 32768 - tmp0;
        diff = ((32768 - ((nsymbs as i32) << rate2)) >> rate) << rate;
        for i in 0..(nsymbs - 1) {
            if i as u32 == val {
                tmp -= diff;
            }
            let prev_cdf = cdf[i] as i32;
            cdf[i] = (prev_cdf + ((tmp - prev_cdf) >> rate)) as u16;
            tmp -= tmp0;
        }
        if cdf[nsymbs] < 32 {
            cdf[nsymbs] += 1;
        }
    }
    pub fn symbol(&mut self, s: u32, cdf: &mut [u16], nsymbs: usize) {
        self.cdf(s, &cdf[..nsymbs]);
        Writer::update_cdf(cdf, s, nsymbs);
    }

    #[allow(dead_code)]
    pub fn tell(&mut self) -> u32 {
        self.enc.od_ec_enc_tell_frac()
    }

    pub fn tell_frac(&mut self) -> u32 {
        self.enc.od_ec_enc_tell_frac()
    }
}
