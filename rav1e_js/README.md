# rav1e_js
The slowest and most dangerous AV1 encoder of the web.

- [About](#about)
- [Usage](#usage)
  - [Api](#api)
  - [Example Website](#example-website)
  - [Testing](#testing)

## About
`rav1e_js` aims to bring the power of [`rav1e`](https://github.com/xiph/rav1e) to the web!

## Usage

### Api

1. Clone + enter the repository
    ```bash
    git clone https://github.com/xiph/rav1e.git
    cd rav1e/rav1e_js/
    ```
2. Install [`wasm-pack`](https://github.com/rustwasm/wasm-pack)
    ```bash
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
    ```
3. Build package
    ```bash
    # Build (emits wasm-, js/ts-files to pkg/)
    wasm-pack build
    ```
4. Add file dependency to `package.json`
    ```ts
    {
        // ...
        "dependencies": {
            "rav1e": "file:path/to/pkg",
            // ...
        },
        // ...
    }
    ```
5. Use it in your project:
    ```ts
    import { ChromaSampling, EncoderConfig, VideoEncoder } from "rav1e";

    // ...
    ```

### Example Website
Run [`rebuild.sh`](./rebuild.sh):
```bash
bash rebuild.sh
```
or run:
```bash
wasm-pack build

cd www/
npm install
npm start
# website served at localhost:3000
```

Please **first** enter the developer console and then start playing the video. You should see logging about the data collection and encoding.

**Note!:** This can take quite a while.

If it doesn't start, please try:
1. reload the webpage

### Testing
```bash
# test in browser
wasm-pack test --headless --[firefox, chrome, safari]

# test in node
wasm-pack test --node
```
