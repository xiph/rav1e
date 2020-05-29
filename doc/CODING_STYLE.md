# Coding Style

<details>
<summary><b>Table of Content</b></summary>

- [General](#general)
- [Assertions](#assertions)
</details>

## General
The coding style for Rust code follows the [Rust Style Guide](https://github.com/rust-dev-tools/fmt-rfcs/blob/master/guide/guide.md)

Exceptions:
- Each indentation level is 2 spaces.

## Assertions
Assertions should never be reachable by any sequence of API calls.

Assertions should never be reachable by any sequence of crav1e API calls
which do not depend on C undefined behavior, or contain pointers that cause
invalid behavior when accessed.

Prefer assert()! to debug_assert()! in the following situations:
 - Unsafe code.
 - Code called once per tile or less where debug build testing
   is unlikely to expose errors (for example, the tests are #[ignored] by default,
   or it is not covered by unit tests at all)
