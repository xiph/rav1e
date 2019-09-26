// Copyright (c) 2016, 2018 vergen developers
//
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. All files in the project carrying such notice may not be copied,
// modified, or distributed except according to those terms.

//! Geneer
//! the `include!` macro within your project.
use crate::constants::{ConstantsFlags, CONST_PREFIX, CONST_TYPE};
use crate::output::generate_build_info;
use std::error::Error;
use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn gen_const<W: Write>(f: &mut W, comment: &str, name: &str, value: &str) -> Result<(), Box<dyn Error>> {
    writeln!(
        f,
        "{}\n{}{}{}\"{}\";",
        comment, CONST_PREFIX, name, CONST_TYPE, value
    )?;
    Ok(())
}

#[cfg(test)]
mod test {
    use super::gen_const;
    use crate::constants::ConstantsFlags;
    use crate::output::generate_build_info;
    use regex::Regex;
    use std::io::Cursor;

    lazy_static! {
        static ref CONST_RE: Regex =
            Regex::new(r#"^/// .*[\r\n]+pub const [A-Z_]+: \&str = ".*";[\r\n]+$"#)
                .expect("Unable to create const regex");
    }

    #[test]
    fn gen_const_output() {
        let flags = ConstantsFlags::all();
        let build_info = generate_build_info(flags).expect("Unable to generate build_info map!");

        for (k, v) in build_info {
            let buffer = Vec::new();
            let mut cursor = Cursor::new(buffer);
            gen_const(&mut cursor, k.comment(), k.name(), &v)
                .expect("Unable to generate const string");
            let const_str = String::from_utf8_lossy(&cursor.get_ref());
            assert!(CONST_RE.is_match(&const_str));
        }
    }
}
