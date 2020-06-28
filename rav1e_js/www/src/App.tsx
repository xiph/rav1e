// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

import React, { useState } from 'react';

import { EncoderConfigDetails, HelpAndAbout } from "./components";
import { Encoder, EncoderConfig, Packet } from "rav1e";
import { EncoderStatus } from "./utils";

// Configuration for encoder
const conf = new EncoderConfig()
	.setDim(64, 96)
	.setRdoLookaheadFrame(1);

// JSON-string representing configuration
const configStr = conf.toJSON();

// Encoder
const enc = new Encoder(conf);

// Frame to send to encoder
const f = enc.newFrame();

export default function App() {
	const [encoderStatus, setEncoderStatus] = useState(EncoderStatus.Initialized);
	const [flushing, setFlushing] = useState(false);
	const [frameQ_length, setFrameQ_length] = useState(0);
	const [packets, setPackets] = useState(Array<Packet>())

	const sendFrame = () => {
		enc.sendFrame(f);
		setFrameQ_length(frameQ_length + 1);
		setEncoderStatus(EncoderStatus.Received)
	};

	const flush = () => {
		enc.flush();
		setFlushing(true);
		setEncoderStatus(EncoderStatus.Flushed);
	}

	const receivePacket = () => {
		try {
			const p = enc.receivePacket();
			setEncoderStatus(EncoderStatus.Encoded);

			// append new packet
			const newPackets = Array.from(packets);
			newPackets.push(p);
			setPackets(newPackets);

			// needed to receive the "limit reached" status
			if (flushing && (frameQ_length - newPackets.length) === 0) {
				receivePacket();
			}
		} catch (e) {
			if (e === "encoded") {
				receivePacket();
			} else if (e === "need more data") {
				setEncoderStatus(EncoderStatus.NeedMoreData);
			} else if (e === "limit reached") {
				setEncoderStatus(EncoderStatus.LimitReached);
			} else {
				setEncoderStatus(e);
				console.error(e);
			}
		}
	}

	return (
		<div>
			<h1>Rav1e_js demo</h1>

			<button disabled={flushing} onClick={sendFrame}>Send Frame</button>
			<button disabled={flushing || packets.length === frameQ_length} onClick={flush}>Flush</button>
			<button disabled={packets.length === frameQ_length} onClick={receivePacket}>Receive Packet</button>

			<p>Frames in queue: {frameQ_length - packets.length}</p>
			<p>Encoder Status: {encoderStatus}</p>

			<HelpAndAbout />
			<EncoderConfigDetails configStr={configStr} />

			{packets.length !== 0 ? <h3>Encoded packets</h3> : <div></div>}
			<ol start={0} >{packets.map((p) => <li key={p.display()}>{p.display()}</li>)}</ol>
		</div>
	);
}
