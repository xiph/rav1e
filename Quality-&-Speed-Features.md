PSA: this page is extremely outdated. Please refer to `README.md` and the Rust documentation for the `config` module.

### Required Quality Features

Quality features that are required for to reach parity with state-of-the-art encoders.

| Feature       |  rav1e        | libaom | x264 | x265 |
| ------------- |:-------------:|:-----:|:-----:|:---:|
| Exact RDO |:white_check_mark:| :x: |:white_check_mark:|:white_check_mark:|
| Motion-adaptive quantizer (CRF) |:white_check_mark:|:x:|:white_check_mark:|:white_check_mark:|
| Temporal RDO |:white_check_mark:|:x:|:white_check_mark: (mbtree) |:white_check_mark:|
| Smart distortion |:white_check_mark:| Broken (cdef-dist, daala-dist) |:white_check_mark: (psyrd) |:white_check_mark:|
| Automatic QM |:white_check_mark:|:x:|:x:|:white_check_mark:|
| Rational luma/chroma weight |:white_check_mark:|:x:|:white_check_mark:|:white_check_mark:|
| B-pyramid |:white_check_mark:|:x:|:white_check_mark:|:white_check_mark:|
|  Good rate control |:white_check_mark:|:x:|:white_check_mark:|:white_check_mark:|

### Unique Quality Features

Quality features where rav1e can go beyond state-of-the-art encoders.

| Feature       |  rav1e        | libaom | x264 | x265 |
| ------------- |:-------------:|:-----:|:-----:|:---:|
| Joint loop filter search |:white_check_mark:|:x:|:x:|:x:|
| Chunk-compatible first pass |:white_check_mark:|:x:|:x:|:x: (in UHDKit) |
| Dynamic programming mvs |:white_check_mark:|:x:|:x:|:x:|
| Auto film grain |:white_check_mark:|:x:|:x:|:x:|

### Speed Features

| Feature       |  rav1e        | libaom | x264 | x265 |
| ------------- |:-------------:|:-----:|:-----:|:---:|
| Pruning using approximate RDO |:white_check_mark:|:x:|:white_check_mark:|:white_check_mark:|
| Frame-parallel encoding |:white_check_mark:|:x:|:white_check_mark:|:white_check_mark:|
| ML trained pruning |:white_check_mark:|:x:|:x:|:x:|
| Approximate transforms |:white_check_mark:|:white_check_mark: (by accident) |:x:|:x:|
| Model-based RDO cost |:white_check_mark:|:x: (broken) |:white_check_mark:| ? |


