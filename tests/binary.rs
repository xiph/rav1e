#[cfg(feature = "binaries")]
mod binary {
  use assert_cmd::Command;
  use interpolate_name::interpolate_test;
  use rand::distributions::Alphanumeric;
  use rand::{thread_rng, Rng};
  use std::env::temp_dir;
  use std::fs::File;
  use std::io::Read;
  use std::path::{Path, PathBuf};

  fn get_y4m_input() -> Vec<u8> {
    let mut input = File::open(format!(
      "{}/tests/small_input.y4m",
      env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let mut data = Vec::new();
    input.read_to_end(&mut data).unwrap();
    data
  }

  fn get_tempfile_path(extension: &str) -> PathBuf {
    let mut path = temp_dir();
    let filename = thread_rng()
      .sample_iter(Alphanumeric)
      .take(12)
      .map(char::from)
      .collect::<String>();
    path.push(format!("{}.{}", filename, extension));
    path
  }

  #[cfg(not(windows))]
  fn get_rav1e_command(high_bitdepth: bool) -> Command {
    let mut cmd = Command::cargo_bin("rav1e").unwrap();
    if high_bitdepth {
      cmd.arg("--high-bitdepth");
    }
    cmd.env_clear();
    cmd
  }

  #[cfg(windows)]
  // `env_clear` doesn't work on Windows: https://github.com/rust-lang/rust/issues/31259
  fn get_rav1e_command(high_bitdepth: bool) -> Command {
    let mut cmd = Command::cargo_bin("rav1e").unwrap();
    if high_bitdepth {
      cmd.arg("--high-bitdepth");
    }
    cmd
  }

  fn get_common_cmd(outfile: &Path, high_bitdepth: bool) -> Command {
    let mut cmd = get_rav1e_command(high_bitdepth);
    cmd.args(["--bitrate", "1000"]).arg("-o").arg(outfile).arg("-y");
    cmd
  }

  #[interpolate_test(low_bitdepth, false)]
  #[interpolate_test(high_bitdepth, true)]
  fn one_pass_qp_based(high_bitdepth: bool) {
    let outfile = get_tempfile_path("ivf");

    get_rav1e_command(high_bitdepth)
      .args(["--quantizer", "100"])
      .arg("-o")
      .arg(&outfile)
      .arg("-")
      .write_stdin(get_y4m_input())
      .assert()
      .success();
  }

  #[interpolate_test(low_bitdepth, false)]
  #[interpolate_test(high_bitdepth, true)]
  fn one_pass_bitrate_based(high_bitdepth: bool) {
    let outfile = get_tempfile_path("ivf");

    get_common_cmd(&outfile, high_bitdepth)
      .arg("-")
      .write_stdin(get_y4m_input())
      .assert()
      .success();
  }

  #[interpolate_test(low_bitdepth, false)]
  #[interpolate_test(high_bitdepth, true)]
  fn two_pass_bitrate_based(high_bitdepth: bool) {
    let outfile = get_tempfile_path("ivf");
    let passfile = get_tempfile_path("pass");

    get_common_cmd(&outfile, high_bitdepth)
      .arg("--first-pass")
      .arg(&passfile)
      .arg("-")
      .write_stdin(get_y4m_input())
      .assert()
      .success();

    get_common_cmd(&outfile, high_bitdepth)
      .arg("--second-pass")
      .arg(&passfile)
      .arg("-")
      .write_stdin(get_y4m_input())
      .assert()
      .success();
  }

  #[interpolate_test(low_bitdepth, false)]
  #[interpolate_test(high_bitdepth, true)]
  fn two_pass_bitrate_based_constrained(high_bitdepth: bool) {
    let outfile = get_tempfile_path("ivf");
    let passfile = get_tempfile_path("pass");

    get_common_cmd(&outfile, high_bitdepth)
      .args(["--reservoir-frame-delay", "14"])
      .arg("--first-pass")
      .arg(&passfile)
      .arg("-")
      .write_stdin(get_y4m_input())
      .assert()
      .success();

    get_common_cmd(&outfile, high_bitdepth)
      .args(["--reservoir-frame-delay", "14"])
      .arg("--second-pass")
      .arg(&passfile)
      .arg("-")
      .write_stdin(get_y4m_input())
      .assert()
      .success();
  }

  #[interpolate_test(low_bitdepth, false)]
  #[interpolate_test(high_bitdepth, true)]

  fn three_pass_bitrate_based(high_bitdepth: bool) {
    let outfile = get_tempfile_path("ivf");
    let pass1file = get_tempfile_path("pass1");
    let pass2file = get_tempfile_path("pass2");

    get_common_cmd(&outfile, high_bitdepth)
      .arg("--first-pass")
      .arg(&pass1file)
      .arg("-")
      .write_stdin(get_y4m_input())
      .assert()
      .success();

    get_common_cmd(&outfile, high_bitdepth)
      .arg("--second-pass")
      .arg(&pass1file)
      .arg("--first-pass")
      .arg(&pass2file)
      .arg("-")
      .write_stdin(get_y4m_input())
      .assert()
      .success();

    get_common_cmd(&outfile, high_bitdepth)
      .arg("--second-pass")
      .arg(&pass2file)
      .arg("-")
      .write_stdin(get_y4m_input())
      .assert()
      .success();
  }
}
