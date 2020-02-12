pub mod math;
pub mod pixel;
pub mod plane;

mod serialize {
  cfg_if::cfg_if! {
     if #[cfg(feature="serialize")] {
       pub use serde::*;
      } else {
        pub use noop_proc_macro::{Deserialize, Serialize};
     }
  }
}
pub mod preamble {
  pub use crate::math::*;
  pub use crate::pixel::*;
}
