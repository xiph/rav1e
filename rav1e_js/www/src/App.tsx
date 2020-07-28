// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

import React, { useEffect } from 'react';

import { ChromaSampling, FrameEncoder, EncoderConfig, Frame } from "rav1e";

export default function App() {
	// useEffect, because then it runs after the browser rendered the display content
	useEffect(() => {
		// create Frame from ferris.png
		// Hint: all frames need to have the same dimensions!
		const ferris_img = document.getElementById("ferris") as HTMLImageElement;
		const ferris_f = Frame.from_img(ferris_img);
		console.log(ferris_f.debug());

		// create Frame from octocat.png
		const octocat_img = document.getElementById("octocat") as HTMLImageElement;
		const octocat_f = Frame.from_img(octocat_img);
		console.log(octocat_f.debug());

		// configure encoder
		const enc = new FrameEncoder(
			new EncoderConfig()
				.setDim(ferris_img.width, ferris_img.height)
				// .setColorDescription(...) (is not available yet)
				.setChromaSampling(ChromaSampling.Cs444)
		);

		// send ferris frames to encoder
		for (let i = 0; i < 10; i++) {
			enc.sendFrame(ferris_f);
		}
		// send octocat frames to encoder
		for (let i = 0; i < 10; i++) {
			enc.sendFrame(octocat_f);
		}

		// flush the encoder
		enc.flush();
		console.log("flushed")

		// encode frames
		const receivePacket = () => {
			try {
				const p = enc.receivePacket();
				console.log(p.display());
			} catch (e) {
				if (e === "encoded") {
					console.warn(e);
					receivePacket();
				} else {
					console.warn(e);
				}
			}
		}
		for (let i = 0; i < 25; i++) {
			receivePacket();
		}
	})

	return (<></>);
}
