#[cfg(feature = "parallel")]
extern crate rayon;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::process::Stdio;

fn x86_triple(os: &str) -> &'static str {
  match os {
    "darwin" => "-fmacho32",
    "windows" => "-fwin32",
    _ => "-felf32",
  }
}

fn x86_64_triple(os: &str) -> &'static str {
  match os {
    "darwin" => "-fmacho64",
    "windows" => "-fwin64",
    _ => "-felf64",
  }
}

fn parse_triple(trip: &str) -> &'static str {
  let parts = trip.split('-').collect::<Vec<_>>();
  // ARCH-VENDOR-OS-ENVIRONMENT
  // or ARCH-VENDOR-OS
  // we don't care about environ so doesn't matter if triple doesn't have it
  if parts.len() < 3 {
    return "";
  }

  match parts[0] {
    "x86_64" => x86_64_triple(parts[2]),
    "x86" | "i386" | "i586" | "i686" => x86_triple(parts[2]),
    _ => "",
  }
}

/// # Example
///
/// ```no_run
/// nasm_rs::compile_library("libfoo.a", &["foo.s", "bar.s"]);
/// ```
pub fn compile_library(output: &str, files: &[&str]) {
  compile_library_args(output, files, &[]);
}

/// # Example
///
/// ```no_run
/// nasm_rs::compile_library_args("libfoo.a", &["foo.s", "bar.s"], &["-Fdwarf"]);
/// ```
pub fn compile_library_args<P: AsRef<Path>>(
  output: &str, files: &[P], args: &[&str],
) {
  let mut b = Build::new();
  for file in files {
    b.file(file);
  }
  for arg in args {
    b.flag(arg);
  }
  b.compile(output);
}

pub struct Build {
  files: Vec<PathBuf>,
  flags: Vec<String>,
  target: Option<String>,
  out_dir: Option<PathBuf>,
  archiver: Option<PathBuf>,
  nasm: Option<PathBuf>,
  debug: bool,
}

impl Default for Build {
  fn default() -> Self {
    Self {
      files: Vec::new(),
      flags: Vec::new(),
      archiver: None,
      out_dir: None,
      nasm: None,
      target: None,
      debug: env::var("DEBUG").ok().map_or(false, |d| d != "false"),
    }
  }
}

impl Build {
  pub fn new() -> Self {
    Self::default()
  }

  /// Add a file which will be compiled
  ///
  /// e.g. `"foo.s"`
  pub fn file<P: AsRef<Path>>(&mut self, p: P) -> &mut Self {
    self.files.push(p.as_ref().to_owned());
    self
  }

  /// Set multiple files
  pub fn files<P: AsRef<Path>, I: IntoIterator<Item = P>>(
    &mut self, files: I,
  ) -> &mut Self {
    for file in files {
      self.file(file);
    }
    self
  }

  /// Add a directory to the `-I` include path
  pub fn include<P: AsRef<Path>>(&mut self, dir: P) -> &mut Self {
    let mut flag = format!("-I{}", dir.as_ref().display());
    // nasm requires trailing slash, but `Path` may omit it.
    if !flag.ends_with('/') {
      flag += "/";
    }
    self.flags.push(flag);
    self
  }

  /// Pre-define a macro with an optional value
  pub fn define<'a, V: Into<Option<&'a str>>>(
    &mut self, var: &str, val: V,
  ) -> &mut Self {
    let val = val.into();
    let flag = if let Some(val) = val {
      format!("-D{}={}", var, val)
    } else {
      format!("-D{}", var)
    };
    self.flags.push(flag);
    self
  }

  /// Configures whether the assembler will generate debug information.
  ///
  /// This option is automatically scraped from the `DEBUG` environment
  /// variable by build scripts (only enabled when the profile is "debug"), so
  /// it's not required to call this function.
  pub fn debug(&mut self, enable: bool) -> &mut Self {
    self.debug = enable;
    self
  }

  /// Add an arbitrary flag to the invocation of the assembler
  ///
  /// e.g. `"-Fdwarf"`
  pub fn flag(&mut self, flag: &str) -> &mut Self {
    self.flags.push(flag.to_owned());
    self
  }

  /// Configures the target this configuration will be compiling for.
  ///
  /// This option is automatically scraped from the `TARGET` environment
  /// variable by build scripts, so it's not required to call this function.
  pub fn target(&mut self, target: &str) -> &mut Self {
    self.target = Some(target.to_owned());
    self
  }

  /// Configures the output directory where all object files and static libraries will be located.
  ///
  /// This option is automatically scraped from the OUT_DIR environment variable by build scripts,
  /// so it's not required to call this function.
  pub fn out_dir<P: AsRef<Path>>(&mut self, out_dir: P) -> &mut Self {
    self.out_dir = Some(out_dir.as_ref().to_owned());
    self
  }

  /// Configures the tool used to assemble archives.
  ///
  /// This option is automatically determined from the target platform or a
  /// number of environment variables, so it's not required to call this
  /// function.
  pub fn archiver<P: AsRef<Path>>(&mut self, archiver: P) -> &mut Self {
    self.archiver = Some(archiver.as_ref().to_owned());
    self
  }

  /// Configures path to `nasm` command
  pub fn nasm<P: AsRef<Path>>(&mut self, nasm: P) -> &mut Self {
    self.nasm = Some(nasm.as_ref().to_owned());
    self
  }

  /// Run the compiler, generating the file output
  ///
  /// The name output should be the base name of the library,
  /// without file extension, and without "lib" prefix.
  ///
  /// The output file will have target-specific name,
  /// such as `lib*.a` (non-MSVC) or `*.lib` (MSVC).
  pub fn compile(&mut self, lib_name: &str) {
    // Trim name for backwards comatibility
    let lib_name = if lib_name.starts_with("lib") && lib_name.ends_with(".a") {
      &lib_name[3..lib_name.len() - 2]
    } else {
      lib_name.trim_end_matches(".lib")
    };

    let target = self.get_target();
    let output = if target.ends_with("-msvc") {
      format!("{}.lib", lib_name)
    } else {
      format!("lib{}.a", lib_name)
    };

    let dst = &self.get_out_dir();
    let objects = self.compile_objects();
    self.archive(&dst, &output, &objects[..]);

    println!("cargo:rustc-link-search={}", dst.display());
  }

  /// Run the compiler, generating .o files
  ///
  /// The files can be linked in a separate step, e.g. passed to `cc`
  pub fn compile_objects(&mut self) -> Vec<PathBuf> {
    let target = self.get_target();

    let nasm = self.find_nasm();
    let args = self.get_args(&target);

    let src = &PathBuf::from(
      env::var_os("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR must be set"),
    );
    let dst = &self.get_out_dir();

    self
      .make_iter()
      .map(|file| self.compile_file(&nasm, file.as_ref(), &args, src, dst))
      .collect::<Vec<_>>()
  }

  fn get_args(&self, target: &str) -> Vec<&str> {
    let mut args = vec![parse_triple(&target)];

    if self.debug {
      args.push("-g");
    }

    for arg in &self.flags {
      args.push(arg);
    }

    args
  }

  #[cfg(feature = "parallel")]
  fn make_iter(&self) -> rayon::slice::Iter<PathBuf> {
    self.files.par_iter()
  }

  #[cfg(not(feature = "parallel"))]
  fn make_iter(&self) -> std::slice::Iter<PathBuf> {
    self.files.iter()
  }

  fn compile_file(
    &self, nasm: &Path, file: &Path, new_args: &[&str], src: &Path, dst: &Path,
  ) -> PathBuf {
    let obj = dst.join(file).with_extension("o");
    let mut cmd = Command::new(nasm);
    cmd.args(&new_args[..]);
    std::fs::create_dir_all(&obj.parent().unwrap()).unwrap();

    run(cmd.arg(src.join(file)).arg("-o").arg(&obj));
    obj
  }

  fn archive(&self, out_dir: &Path, lib: &str, objs: &[PathBuf]) {
    let ar = if cfg!(target_env = "msvc") {
      self.archiver.clone().unwrap_or_else(|| "lib".into())
    } else {
      self
        .archiver
        .clone()
        .or_else(|| env::var_os("AR").map(|a| a.into()))
        .unwrap_or_else(|| "ar".into())
    };
    if cfg!(target_env = "msvc") {
      let mut out_param = OsString::new();
      out_param.push("/OUT:");
      out_param.push(out_dir.join(lib).as_os_str());
      run(Command::new(ar).arg(out_param).args(objs));
    } else {
      run(Command::new(ar).arg("crus").arg(out_dir.join(lib)).args(objs));
    }
  }

  fn get_out_dir(&self) -> PathBuf {
    self.out_dir.clone().unwrap_or_else(|| {
      PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR must be set"))
    })
  }

  fn get_target(&self) -> String {
    self
      .target
      .clone()
      .unwrap_or_else(|| env::var("TARGET").expect("TARGET must be set"))
  }

  fn find_nasm(&mut self) -> PathBuf {
    match self.nasm.clone() {
      Some(path) => path,
      None => {
        let nasm_path = PathBuf::from("nasm");
        match is_nasm_new_enough(&nasm_path) {
          Ok(_) => nasm_path,
          Err(version) => {
            panic!("This version of NASM is too old: {}", version);
          }
        }
      }
    }
  }
}

fn get_output(cmd: &mut Command) -> Result<String, String> {
  let out = cmd.output().map_err(|e| e.to_string())?;
  if out.status.success() {
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
  } else {
    Err(String::from_utf8_lossy(&out.stderr).to_string())
  }
}

/// Returns version string if nasm is too old,
/// or error message string if it's unusable.
fn is_nasm_new_enough(nasm_path: &Path) -> Result<(), String> {
  match get_output(Command::new(nasm_path).arg("-v")) {
    Ok(version) => {
      if version.contains("NASM version 0.") {
        Err(version)
      } else {
        Ok(())
      }
    }
    Err(err) => Err(err),
  }
}

fn run(cmd: &mut Command) {
  eprintln!("running: {:?}", cmd);

  let status =
    match cmd.stdout(Stdio::inherit()).stderr(Stdio::inherit()).status() {
      Ok(status) => status,

      Err(e) => panic!("failed to spawn process: {:?} - {}", cmd, e),
    };

  if !status.success() {
    panic!("nonzero exit status: {}", status);
  }
}

#[test]
fn test_build() {
  let mut build = Build::new();
  build.file("test");
  build.archiver("ar");
  build.include("./");
  build.include("dir");
  build.define("foo", Some("1"));
  build.define("bar", None);
  build.flag("-test");
  build.target("i686-unknown-linux-musl");
  build.out_dir("/tmp");

  assert_eq!(
    build.get_args("i686-unknown-linux-musl"),
    &["-felf32", "-I./", "-Idir/", "-Dfoo=1", "-Dbar", "-test"]
  );
}
