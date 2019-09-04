#![no_main]
#[macro_use] extern crate libfuzzer_sys;
extern crate rav1e;
use rav1e::fuzzing::*;

fuzz_target!(|data| {
  let _ = pretty_env_logger::try_init();

  fuzz_encode_decode(data)
});
