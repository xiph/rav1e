// Copyright (c) 2017-2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

mod align;
#[macro_use]
mod cdf;
mod uninit;

use std::sync::Arc;

pub use v_frame::math::*;
pub use v_frame::pixel::*;

pub use align::*;
pub use uninit::*;

// There does exist `Arc::get_mut_unchecked` to do this,
// but it is currently nightly only.
// And because the Arc fields are private, we have to do something much more annoying.
//
// Once `get_mut_unchecked` is stable in stdlib, we should use that instead.
//
// Why does it matter so much that this exists?
// `Arc::make_mut` may clone the inner data without our awareness.
// That may (although has not, somehow) cause issues with data consistency.
// But the issue we have encountered is that these clones make rav1e slower.
// If we know that we are not writing to the same part of this Arc as another thread,
// such as when tiling, we can avoid the clone.
pub(crate) unsafe fn arc_get_mut_unsafe<T>(this: &mut Arc<T>) -> &mut T {
  let count = Arc::strong_count(this);
  let raw = Arc::into_raw(Arc::clone(this));
  for _ in 0..count {
    Arc::decrement_strong_count(raw);
  }
  let inner = Arc::get_mut(this).unwrap();
  for _ in 0..count {
    Arc::increment_strong_count(raw);
  }
  Arc::from_raw(raw);
  inner
}
