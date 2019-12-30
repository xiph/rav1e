// Copyright (c) 2017-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

// TODO: Nice to have (although I wasn't able to find a way to do it yet in rust): zero-fill arrays that are
// shorter than required.  Need const fn (Rust Issue #24111) or const generics (Rust RFC #2000)
macro_rules! cdf {
    ($($x:expr),+) =>  {[$(32768 - $x),+, 0, 0]}
}

macro_rules! cdf_size {
  ($x:expr) => {
    $x + 1
  };
}
