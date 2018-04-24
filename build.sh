#!/bin/bash

# Stop on first error
set -e

#SEQ=!!!!! ENTER YOUR FAVORITE Y4M HERE !!!!!

if [[ -z "${SEQ}" ]]; then
  SEQ=nyan.y4m
  wget -nc https://mf4.xiph.org/~ltrudeau/videos/nyan.y4m
fi


if [ ! -f $SEQ ]; then
  (>&2 echo "ERROR: Failed to find $SEQ")
  (>&2 echo "Please recheck the variables")
  exit 1 # terminate and indicate error
fi

# Hide githash to detect version changes
GITHASH=".git/rav1e.githash"

# Get previous version
EXPECTED_VERSION="42"
if [ -f $GITHASH ]; then
  EXPECTED_VERSION=$(cat $GITHASH)
fi

# Get current version
ACTUAL_VERSION=$(git submodule status | xargs)

AOM_TEST="aom_test"
if [ "$ACTUAL_VERSION" != "$EXPECTED_VERSION" ] || [ ! -f ./${AOM_TEST}/aomdec ]; then

# Store current version to file
echo $ACTUAL_VERSION > $GITHASH

# Update aombuild
git submodule update --init

# Get configure command from readme
CONFIGURE_CMD=$(fgrep cmake README.md)

# Create aom_test folder if none
mkdir -p $AOM_TEST
pushd $AOM_TEST

if [ -f Makefile ]; then
  # Clean if needed
  make clean
  make distclean
fi
echo CONFIGURE COMMAND
echo $CONFIGURE_CMD
eval $CONFIGURE_CMD

# auto detect the number of cores and parallel build
make -j$(nproc --all)
popd

fi

# File containing the encoded sequence
ENC_FILE="enc_file.ivf"
# File containing the decoded sequence
DEC_FILE="dec_file.y4m"

# Print the backtrace on error
export RUST_BACKTRACE=1

# Build and run encoder
cargo run --bin rav1e --release -- $SEQ -o $ENC_FILE -s 2

# Decode
${AOM_TEST}/aomdec $ENC_FILE -o $DEC_FILE

# Daala tools support coming soon
#DAALA_TOOLS="../daala/tools/"
# Convert to png
#${DAALA_TOOLS}/y4m2png -o out.png $DEC_FILE

# Compute and print PSNR (not working)
#${DAALA_TOOLS}/dump_psnr $DEC_OUT $SEQ

# Compute and print CIEDE2000 (not working)
#${DAALA_TOOLS}/dump_ciede2000.py $DEC_OUT $SEQ

# Show decoded sequence
# --pause
mpv --loop $DEC_FILE
