use std::error::Error;

#[derive(Debug, Error)]
pub enum CliError {
  #[error(display = "{}: {}", msg, io)]
  Io { msg: String, io: std::io::Error },
  #[error(display = "{}: {:?}", msg, status)]
  Enc { msg: String, status: rav1e::EncoderStatus },
  #[error(display = "Cannot parse option `{}`: {}", opt, err)]
  ParseInt { opt: String, err: std::num::ParseIntError }
}

pub trait ToError {
  fn context(self, msg: &str) -> CliError;
}

impl ToError for std::io::Error {
  fn context(self, msg: &str) -> CliError {
    CliError::Io { msg: msg.to_owned(), io: self }
  }
}

impl ToError for rav1e::EncoderStatus {
  fn context(self, msg: &str) -> CliError {
    CliError::Enc { msg: msg.to_owned(), status: self }
  }
}

impl ToError for std::num::ParseIntError {
  fn context(self, opt: &str) -> CliError {
    CliError::ParseInt { opt: opt.to_lowercase(), err: self }
  }
}

pub fn print_error(e: &dyn Error) {
    eprintln!("error: {}", e);
    let mut cause = e.source();
    while let Some(e) = cause {
        eprintln!("caused by: {}", e);
        cause = e.source();
    }
}
