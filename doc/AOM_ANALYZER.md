# AOMAnalyzer

<details>
<summary><b>Table of Contents</b></summary>

- [Local Analyzer](#local-analyzer)
- [Online Analyzer](#online-analyzer)
</details>

## Local Analyzer

1. Download the [AOM Analyzer](http://aomanalyzer.org) ([source](https://github.com/xiph/aomanalyzer)).
2. Download [inspect.js](https://media.xiph.org/analyzer/inspect.js) ([mirror](https://github.com/xiph/aomanalyzer/files/4685593/inspect.wasm.gz)) and [inspect.wasm](https://media.xiph.org/analyzer/inspect.wasm) ([mirror](https://github.com/xiph/aomanalyzer/files/4685594/inspect.js.gz)) and save them in the same directory.
3. Run the analyzer:
   ```
   AOMAnalyzer path_to_inspect.js output.ivf
   ```

## Online Analyzer

If your `.ivf` file is hosted somewhere (and CORS is enabled on your web server) you can use:

```
https://arewecompressedyet.com/analyzer/?d=https://media.xiph.org/analyzer/inspect.js&f=path_to_output.ivf
```
