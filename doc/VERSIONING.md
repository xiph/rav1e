rav1e follows Cargo's versioning scheme: https://doc.rust-lang.org/cargo/reference/manifest.html#the-version-field

Because rav1e is not yet at version 1.0.0, all changes that break the API require a minor-version bump.

The API is defined as:
- public functions in src/api.rs
- command line parameters to the rav1e binary
