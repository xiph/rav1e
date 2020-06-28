// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

import React from 'react';
import "./style.css";

const HelpAndAbout: React.FC = () => (
    <details>
        <summary>About and Help</summary>

        <h3>About</h3>
        <p>Thank you for trying the <a href="https://github.com/xiph/rav1e">rav1e</a>-encode-in-browser demo ^.^</p>
        <p>You are able to send frames to the encoder, encode them and view a list of the emitted Packets. And the best thing? Everything is running locally in your browser thanks to <a href="https://webassembly.org">webassembly</a>!</p>

        <h3>Buttons</h3>
        <table style={{ tableLayout: "fixed", borderCollapse: "collapse" }}>
            <tbody>
                <tr>
                    <td><strong>Name</strong></td>
                    <td><strong>Functionality</strong></td>
                    <td><strong>Details</strong></td>
                </tr>
                <tr>
                    <td>Send Frame</td>
                    <td>Send a new (blank) frame to the encoder.</td>
                    <td></td>
                </tr>
                <tr>
                    <td>Flush</td>
                    <td>Signals the End of the Video.</td>
                    <td>Disables option to send new frames</td>
                </tr>
                <tr>
                    <td>Receive Packet</td>
                    <td>Encode the next frame.</td>
                    <td>Needs ~11 Frames in the queue</td>
                </tr>
            </tbody>
        </table>
    </details>
)

export default HelpAndAbout;