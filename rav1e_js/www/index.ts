// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

import { Encoder, EncoderConfig, Frame, Packet, ChromaSamplePosition } from "rav1e";

const conf: EncoderConfig = new EncoderConfig()
    .setDim(96, 64)
    .setSpeed(9)
    .setChromaSamplePosition(ChromaSamplePosition.Unknown);
console.log(conf.debug())

try {
    // could raise (catchable) error, if EncoderConfig is invalid
    const enc: Encoder = new Encoder(conf);
    console.log(enc.debug());

    const f: Frame = enc.newFrame();

    for (let i = 0; i < 10; i++) {
        enc.sendFrame(f);
    }
    enc.flush();
    console.log(enc.debug());

    for (let i = 0; i < 20; i++) {
        try {
            const p: Packet = enc.receivePacket();
            console.log(p.display())
        } catch (e) {
            console.warn(e);
        }
    }
} catch (e) {
    console.error(e);
}
