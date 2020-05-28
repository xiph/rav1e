#[cfg(feature = "binaries")]
mod binary {
  use assert_cmd::Command;
  use rand::distributions::Alphanumeric;
  use rand::{thread_rng, Rng};
  use std::env::temp_dir;
  use std::fs::File;
  use std::io::Read;
  use std::path::PathBuf;

  fn get_y4m_input() -> Vec<u8> {
    let mut input = File::open(&format!(
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
    let filename =
      thread_rng().sample_iter(&Alphanumeric).take(12).collect::<String>();
    path.push(format!("{}.{}", filename, extension));
    path
  }

  #[cfg(not(windows))]
  fn get_rav1e_command() -> Command {
    let mut cmd = Command::cargo_bin("rav1e").unwrap();
    cmd.env_clear();
    cmd
  }

  #[cfg(windows)]
  // `env_clear` doesn't work on Windows: https://github.com/rust-lang/rust/issues/31259
  fn get_rav1e_command() -> Command {
    Command::cargo_bin("rav1e").unwrap()
  }

  fn get_common_cmd(outfile: &PathBuf) -> Command {
    let mut cmd = get_rav1e_command();
    cmd.arg("--bitrate").arg("1000").arg("-o").arg(outfile);
    cmd
  }

  #[test]
  fn one_pass_qp_based() {
    let mut cmd = get_rav1e_command();
    let outfile = get_tempfile_path("ivf");

    cmd
      .arg("--quantizer")
      .arg("100")
      .arg("-o")
      .arg(&outfile)
      .arg("-")
      .write_stdin(get_y4m_input())
      .assert()
      .success();
  }

  #[test]
  fn one_pass_bitrate_based() {
    let outfile = get_tempfile_path("ivf");

    let mut cmd = get_common_cmd(&outfile);
    cmd.arg("-").write_stdin(get_y4m_input()).assert().success();
  }

  #[test]
  fn two_pass_bitrate_based() {
    let outfile = get_tempfile_path("ivf");
    let passfile = get_tempfile_path("pass");

    let mut cmd1 = get_common_cmd(&outfile);
    cmd1
      .arg("--first-pass")
      .arg(&passfile)
      .arg("-")
      .write_stdin(get_y4m_input())
      .assert()
      .success();

    let mut cmd2 = get_common_cmd(&outfile);
    cmd2
      .arg("--second-pass")
      .arg(&passfile)
      .arg("-")
      .write_stdin(get_y4m_input())
      .assert()
      .success();
  }
  #[test]
  fn two_pass_bitrate_based_constrained() {
    let outfile = get_tempfile_path("ivf");
    let passfile = get_tempfile_path("pass");

    let mut cmd1 = get_common_cmd(&outfile);
    cmd1
      .arg("--reservoir-frame-delay")
      .arg("14")
      .arg("--first-pass")
      .arg(&passfile)
      .arg("-")
      .write_stdin(get_y4m_input())
      .assert()
      .success();

    let mut cmd2 = get_common_cmd(&outfile);
    cmd2
      .arg("--reservoir-frame-delay")
      .arg("14")
      .arg("--second-pass")
      .arg(&passfile)
      .arg("-")
      .write_stdin(get_y4m_input())
      .assert()
      .success();
  }
}
