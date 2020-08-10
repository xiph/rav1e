// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

import React from 'react';

import { ChromaSampling, ColorPrimaries, EncoderConfig, MatrixCoefficients, TransferCharacteristics, VideoEncoder } from "rav1e";

export default function App() {
	const video = document.getElementById("video") as HTMLVideoElement;

	let init = () => {
		console.log(video.videoWidth, video.videoHeight);

		// configure encoder
		const enc = new VideoEncoder(
			new EncoderConfig()
				.setSpeed(10)
				.setDim(video.videoWidth, video.videoHeight)
				.setColorDescription(
					ColorPrimaries.BT709,
					TransferCharacteristics.BT709,
					MatrixCoefficients.BT709
				)
				// ChromaSampling needs to fit to the ChromaSampling of the Frame
				.setChromaSampling(ChromaSampling.Cs444)
		);
		enc.sendVideo(video);

		video.addEventListener("ended", (e) => {
			enc.flush()

			// encode all frames
			while (true) {
				try {
					const p = enc.receivePacket();
					console.log(p.display());
				} catch (e) {
					if (e === "encoded") {
						console.warn(e);
					} else if (e === "limit reached") {
						console.warn(e);
						break;
					} else {
						console.error(e);
					}
				}
			}
		})
	}

	// this is needed to support more browsers
	if (video.videoWidth !== 0 && video.videoHeight !== 0) {
		init()
	} else {
		video.onloadedmetadata = init;
	}



	return (<>
		<p>
			Please open your developer console and start the video!
		</p>
	</>);
}
