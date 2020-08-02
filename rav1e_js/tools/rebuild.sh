# env
WASM_PACK=wasm-pack
# WASM_OPT=

rm -r ./pkg www/node_modules/

# exit on error
set -e

echo "# build"
$WASM_PACK build

# echo "# optimize"
# $WASM_OPT pkg/rav1e_js_bg.wasm -o pkg/rav1e_js_bg.wasm -O # -all 

echo "# install"
cd www
yarn install

echo "# start"
yarn start
