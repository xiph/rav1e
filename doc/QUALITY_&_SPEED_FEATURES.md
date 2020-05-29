# Quality- & Speed-Features

<details>
<summary><b>Table of Content</b></summary>

- [Quality Features](#quality-features)
  - [Required](#required)
  - [Unique](#unique)
- [Speed Features](#speed-features)
</details>

## Quality Features

### Required

Quality features that are required for to reach parity with state-of-the-art encoders.

| Feature                     |       rav1e        |       libaom       |            x264             |        x265        |
| --------------------------- | :----------------: | :----------------: | :-------------------------: | :----------------: |
| Exact RDO                   | :white_check_mark: |        :x:         |     :white_check_mark:      | :white_check_mark: |
| Temporal RDO                | :white_check_mark: |        :x:         | :white_check_mark: (mbtree) | :white_check_mark: |
| Smart distortion            | :white_check_mark: |        :x:         | :white_check_mark: (psyrd)  | :white_check_mark: |
| Automatic QM                |        :x:         |        :x:         |             :x:             | :white_check_mark: |
| Rational luma/chroma weight | :white_check_mark: |        :x:         |     :white_check_mark:      | :white_check_mark: |
| B-pyramid                   | :white_check_mark: | :white_check_mark: |     :white_check_mark:      | :white_check_mark: |

### Unique

Quality features where rav1e can go beyond state-of-the-art encoders.

| Feature                     |       rav1e        | libaom | x264  |      x265       |
| --------------------------- | :----------------: | :----: | :---: | :-------------: |
| Joint loop filter search    | :white_check_mark: |  :x:   |  :x:  |       :x:       |
| Chunk-compatible first pass | :white_check_mark: |  :x:   |  :x:  | :x: (in UHDKit) |

## Speed Features

| Feature                       |       rav1e        |       libaom       |        x264        |        x265        |
| ----------------------------- | :----------------: | :----------------: | :----------------: | :----------------: |
| Pruning using approximate RDO | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Frame-parallel encoding       |        :x:         |        :x:         | :white_check_mark: | :white_check_mark: |
| Model-based RDO cost          |        :x:         |    :x: (broken)    | :white_check_mark: |         ?          |


