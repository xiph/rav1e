use std::env;
use std::fmt::Display;
use std::fs::File;
use std::io::{BufWriter, Cursor, Write};
use std::path::{Path, PathBuf};

use super::*;

pub mod dist;
pub mod mc;
pub mod tx;

struct StdImports;
impl ToTokens for StdImports {
  fn to_tokens(&self, to: &mut TokenStream) {
    to.extend(quote! {
      use crate::cpu_features::CpuFeatureLevel;
      use crate::partition::BlockSize;
      use crate::util::{AlignedArray, ISimd, round_shift, };

      use packed_simd::{Simd, i32x8, m8, FromCast, FromBits, shuffle, Cast, };

      use std::slice::{from_raw_parts, from_raw_parts_mut, };
    });
  }
}

/// Ensure we don't unroll too much or not enough! Either way and the resulting
/// kernel is likely to be invalid anyway, so we panic loudly here.
#[derive(Debug)]
struct UnrollCheck<'a>(usize, &'a Ident, usize);
impl<'a> UnrollCheck<'a> {
  fn new(kernel: &'a Ident) -> Self {
    UnrollCheck(0, kernel, MAX_UNROLL)
  }
  fn new_max(kernel: &'a Ident, max: usize) -> Self {
    UnrollCheck(0, kernel, max)
  }
  fn check(&mut self) {
    self.0 += 1;

    if self.0 > self.2 {
      panic!("{} tried to unroll too much: {} vs {}", self.1, self.0, self.2);
    }
  }
  fn finish(self) {
    if self.0 < self.2 {
      panic!("{} didn't unroll enough: {} vs {}", self.1, self.0, self.2);
    }
  }
}

pub fn write_kernels() {
  write_kernel("inv_tx_add_kernels.rs", tx::inv_tx_add_kernels);
  write_kernel("sad_kernels.rs", dist::sad_kernels);
  write_kernel("satd_kernels.rs", dist::satd_kernels);
  write_kernel("put_8tap_kernels.rs", mc::put_8tap_kernels);
  write_kernel("prep_8tap_kernels.rs", mc::prep_8tap_kernels);
  write_kernel("mc_avg_kernels.rs", mc::mc_avg_kernels);
}

pub(super) fn write_kernel<T, F>(out: T, f: F)
where
  T: Into<PathBuf>,
  F: FnOnce(&mut dyn Write),
{
  const BUFFER: usize = 16 * 1024 * 1024;

  let out = out.into();
  let out = if out.is_relative() {
    let out_dir = env::var_os("OUT_DIR").expect("need OUT_DIR");
    let out_dir = Path::new(&out_dir);
    out_dir.join(out)
  } else {
    out
  };

  {
    let file = File::create(&out)
      .unwrap_or_else(|e| panic!("create {}: {:?}", out.display(), e));
    let mut file = BufWriter::with_capacity(BUFFER, file);

    f(&mut file);
  }
  try_rustfmt(out);
}

fn write_prelude(out: &mut dyn Write) {
  writeln!(
    out,
    r#"
use crate::cpu_features::CpuFeatureLevel;
use crate::partition::BlockSize;
use crate::util::{{AlignedArray, ISimd, round_shift, }};

use packed_simd::{{Simd, i32x8, m8, FromCast, FromBits, }};

use std::slice::{{from_raw_parts, from_raw_parts_mut, }};
"#
  )
  .unwrap();
}

/// try to run `rustfmt` on `p`, but don't panic if rustfmt isn't
/// present. Formatting `TokenStream` puts everything on a single
/// line, so we need to run `rustfmt` so that the kernels are
/// actually legible.
fn try_rustfmt<T>(p: T)
where
  T: AsRef<Path>,
{
  use std::cell::RefCell;
  use std::collections::VecDeque;
  use std::process::{Child, Command};

  struct ChildCleanup(Child);
  impl Drop for ChildCleanup {
    fn drop(&mut self) {
      // wait for the formatting to finish
      let _ = self.0.wait();
    }
  }
  impl From<Child> for ChildCleanup {
    fn from(v: Child) -> ChildCleanup {
      ChildCleanup(v)
    }
  }

  thread_local! {
    static FORMATS: RefCell<VecDeque<ChildCleanup>> = RefCell::default();
  }
  // ensure we don't spawn too many processes
  // TODO use the jobserver or parse NUM_JOBS?
  const MAX_PROCESSES: usize = 16;
  FORMATS.with(|f| {
    let mut f = f.borrow_mut();
    if f.len() >= MAX_PROCESSES {
      f.pop_front();
    }
  });

  let p = p.as_ref();

  let mut cmd = Command::new("rustfmt");
  cmd
    .current_dir(p.parent().unwrap())
    .arg("--unstable-features")
    // I don't want to wait for you, man, and have already formatted
    // all the child files.
    .arg("--skip-children")
    .arg(p);

  // ignore all errors
  if let Ok(child) = cmd.spawn() {
    // add the process to the thread local set.
    // that thread local variable will be cleaned up when this
    // thread exits, so we wait on the child in drop to make sure it
    // finishes before we exit and otherwise don't block
    FORMATS.with(move |f| {
      f.borrow_mut().push_back(child.into());
    });
  }
}
