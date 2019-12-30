// Copyright (c) 2017-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

pub trait Dim {
  const W: usize;
  const H: usize;
}

macro_rules! blocks_dimension {
  ($(($W:expr, $H:expr)),+) => {
    paste::item! {
      $(
        pub struct [<Block $W x $H>];

        impl Dim for [<Block $W x $H>] {
          const W: usize = $W;
          const H: usize = $H;
        }
      )*
    }
  };
}

blocks_dimension! {
  (4, 4), (8, 8), (16, 16), (32, 32), (64, 64),
  (4, 8), (8, 16), (16, 32), (32, 64),
  (8, 4), (16, 8), (32, 16), (64, 32),
  (4, 16), (8, 32), (16, 64),
  (16, 4), (32, 8), (64, 16)
}
