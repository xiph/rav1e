// Copyright (c) 2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![no_main]
#[macro_use] extern crate libfuzzer_sys;
extern crate rav1e;
use rav1e::fuzzing::*;

fuzz_target!(|data| {
  let _ = pretty_env_logger::try_init();

  fuzz_encode_decode(data)
});
