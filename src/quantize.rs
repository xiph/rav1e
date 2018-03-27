use partition::TxSize;

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

pub fn quantize_in_place(qindex: usize, coeffs: &mut [i32], tx_size: TxSize) {
    let tx_scale = match tx_size {
        TxSize::TX_32X32 => 2,
        _ => 1
    };
    coeffs[0] *= tx_scale;
    coeffs[0] /= dc_q(qindex) as i32;
    let ac_quant = ac_q(qindex) as i32;

    for c in coeffs[1..].iter_mut() {
        *c *= tx_scale;
        *c /= ac_quant;
    }
    // workaround for bug in token coder
    *coeffs.last_mut().unwrap() = 0;
}

pub fn dequantize(qindex:usize, coeffs: &[i32], rcoeffs: &mut [i32], tx_size: TxSize) {
    let tx_scale = match tx_size {
        TxSize::TX_32X32 => 2,
        _ => 1
    };
    rcoeffs[0] = (coeffs[0] * dc_q(qindex) as i32) / tx_scale;
    let ac_quant = ac_q(qindex) as i32;

    for (r, &c) in rcoeffs.iter_mut().zip(coeffs.iter()).skip(1) {
        *r = c * ac_quant / tx_scale;
    }
}
