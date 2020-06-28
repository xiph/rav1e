// Copyright (c) 2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

export enum EncoderStatus {
	Encoded = "Encoded frame",
	Flushed = "Flushed",
	Initialized = "Initialized successfully",
	LimitReached = "Limit reached",
	NeedMoreData = "Need more data (send more frames or flush the encoder)",
	Received = "Received frame"
}