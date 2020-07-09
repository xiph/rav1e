# rav1e docs

<details>
<summary><b>Table of Content</b></summary>

- [../Readme](#readme)
- [AOMAnalyzer](#aomanalyzer)
- [Coding Style](#coding-style)
- [Frame Type Selection](#frame-type-selection)
- [Glossary](#glossary)
- [Profiling](#profiling)
- [Quality- & Speed-Features](#quality---speed-features)
- [Rate-control Empirical Analysis](#rate-control-empirical-analysis)
- [File Structure](#file-structure)
- [Versioning](#versioning)
</details>

_**NOTE:** Headline may link to page._

## [../Readme](../README.md)
Main README of rav1e.

## [AOMAnalyzer](AOM_ANALYZER.md)
Analyze `.ivf`-files with `AOM Analyzer`:
* Local Analyzer
* Online Analyzer

## [Coding Style](CODING_STYLE.md)

## [Frame Type Selection](FRAME_TYPE_SELECTION.md)
- Current Features/Process
- Detection Algorithm
- Desired Improvements

## [Glossary](GLOSSARY.md)
Explanation of various special terms.

## [Profiling](PROFILING.md)
List of various profiling tools:
- Cargo integrations
- Generic profiling
- Tracing
- Codegen Inspection

## [Quality- & Speed-Features](QUALITY_&_SPEED_FEATURES.md)
Overview of quality and speed-features for rav1e and other state-of-the-art encoder.

## [Rate-control Empirical Analysis](regress_log-bitrate_wrt_log-quantizer.ipynb)
Notebook documenting how rate-control constants were derived from empirical data.
These constants determine the initial values of `RCState::log_scale`, `RCState::exp` and `RCState::scalefilter`.

## [File Structure](STRUCTURE.md)
- High-level directory structure
- Overview of `src/*`

## Versioning
rav1e follows Cargo's versioning scheme: https://doc.rust-lang.org/cargo/reference/manifest.html#the-version-field

Because rav1e is not yet at version 1.0.0, all changes that break the API require a minor-version bump.

The API is defined as:
- public functions in src/api.rs
- command line parameters to the rav1e binary
