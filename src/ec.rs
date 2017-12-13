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
}
