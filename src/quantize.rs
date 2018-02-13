extern {
    static dc_qlookup_Q3: [i16; 256];
    static ac_qlookup_Q3: [i16; 256];
}

pub fn dc_q(qindex: usize) -> i16 {
    unsafe {
        dc_qlookup_Q3[qindex]
    }
}

pub fn ac_q(qindex: usize) -> i16 {
    unsafe {
        ac_qlookup_Q3[qindex]
    }
}

pub fn quantize_in_place(qindex: usize, coeffs: &mut [i32]) {
    coeffs[0] /= dc_q(qindex) as i32;
    for c in coeffs[1..].iter_mut() {
        *c /= ac_q(qindex) as i32;
    }
    coeffs[15] = 0;
}

pub fn dequantize(qindex:usize, coeffs: &[i32], rcoeffs: &mut [i32]) {
    rcoeffs[0] = coeffs[0] * dc_q(qindex) as i32;
    for i in 1..16 {
        rcoeffs[i] = coeffs[i] * ac_q(qindex) as i32;
    }
}
