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
| Exact RDO                   | :heavy_check_mark: |        :x:         |     :heavy_check_mark:      | :heavy_check_mark: |
| Temporal RDO                | :heavy_check_mark: |        :x:         | :heavy_check_mark: (mbtree) | :heavy_check_mark: |
| Smart distortion            | :heavy_check_mark: |        :x:         | :heavy_check_mark: (psyrd)  | :heavy_check_mark: |
| Automatic QM                |        :x:         |        :x:         |             :x:             | :heavy_check_mark: |
| Rational luma/chroma weight | :heavy_check_mark: |        :x:         |     :heavy_check_mark:      | :heavy_check_mark: |
| B-pyramid                   | :heavy_check_mark: | :heavy_check_mark: |     :heavy_check_mark:      | :heavy_check_mark: |

### Unique

Quality features where rav1e can go beyond state-of-the-art encoders.

| Feature                     |       rav1e        | libaom | x264  |      x265       |
| --------------------------- | :----------------: | :----: | :---: | :-------------: |
| Joint loop filter search    | :heavy_check_mark: |  :x:   |  :x:  |       :x:       |
| Chunk-compatible first pass | :heavy_check_mark: |  :x:   |  :x:  | :x: (in UHDKit) |

## Speed Features

| Feature                       |       rav1e        |       libaom       |        x264        |        x265        |
| ----------------------------- | :----------------: | :----------------: | :----------------: | :----------------: |
| Pruning using approximate RDO | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Frame-parallel encoding       |        :x:         |        :x:         | :heavy_check_mark: | :heavy_check_mark: |
| Model-based RDO cost          |        :x:         |    :x: (broken)    | :heavy_check_mark: |         ?          |


