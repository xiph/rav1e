# rav1e_js

## Install `wasm-pack`

```bash
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

## Usage

```bash
# Build (emits wasm-, js/ts-files to pkg/)
wasm-pack build

# Test in Headless Browser
wasm-pack test --headless --firefox
```

## Example Website
```bash
# make sure you have the latest npm version
npm install -g npm@latest

cd rav1e_js
wasm-pack build

cd www/
npm install
npm start
# website served at localhost:8080
# check the developer-console
```
