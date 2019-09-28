extern crate chrono;

pub fn main() {
  let now = chrono::Utc::now();
  println!("cargo:rustc-env=VERGEN_BUILD_TIMESTAMP={}", now.to_rfc3339());
}
