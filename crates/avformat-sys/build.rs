use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn format_write(builder: bindgen::Builder) -> String {
  builder.generate().unwrap().to_string()
}

fn main() {
  let libs = metadeps::probe().unwrap();
  let headers = libs.get("libavformat").unwrap().include_paths.clone();

  let mut builder = bindgen::builder()
    .header("data/av.h")
    .default_enum_style(bindgen::EnumVariation::ModuleConsts);

  for header in headers {
    builder = builder.clang_arg("-I").clang_arg(header.to_str().unwrap());
  }

  // Manually fix the comment so rustdoc won't try to pick them
  let s = format_write(builder);

  let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

  let mut file = File::create(out_path.join("av.rs")).unwrap();

  let _ = file.write(s.as_bytes());
}
