#![allow(safe_extern_statics)]

extern {
    static dc_qlookup: [i16; 256];
    static ac_qlookup: [i16; 256];
}

pub fn quantize_in_place(qindex: usize, coeffs: &mut [i32]) {
    coeffs[0] /= dc_qlookup[qindex] as i32;
    //coeffs[0] = 0;
    if coeffs[0] > 34 {
        coeffs[0] = 34;
    }
    if coeffs[0] < -34 {
        coeffs[0] = -34;
    }
    for c in coeffs[1..].iter_mut() {
        *c /= ac_qlookup[qindex] as i32;
        if *c > 34 {
            *c = 34;
        }
        if *c < -34 {
            *c = -34;
        }
    }
    coeffs[15] = 0;
}

pub fn dequantize(qindex:usize, coeffs: &[i32], rcoeffs: &mut [i32]) {
    rcoeffs[0] = coeffs[0] * dc_qlookup[qindex] as i32;
    for i in 1..16 {
        rcoeffs[i] = coeffs[i] * ac_qlookup[qindex] as i32;
    }
}
