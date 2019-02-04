#!/bin/bash

# Stop on first error
set -e

#SEQ=!!!!! ENTER YOUR FAVORITE Y4M HERE !!!!!

IS_RELEASE=1

for arg in "$@"; do
  shift
  case "$arg" in
    "--debug") IS_RELEASE=0 ;;
    *)        set -- "$@" "$arg"
  esac
done

if [[ -z "${SEQ}" ]]; then
  SEQ=nyan.y4m
  SEQ10=nyan10.y4m
  SEQ12=nyan12.y4m
  
  wget -nc https://mf4.xiph.org/~ltrudeau/videos/nyan.y4m
  #wget -nc https://people.xiph.org/~tdaede/nyan10.y4m
  #wget -nc https://people.xiph.org/~tdaede/nyan12.y4m
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
if [[ "$ACTUAL_VERSION" != "$EXPECTED_VERSION" ]] || [[ ! -f ./${AOM_TEST}/aomdec ]]; then

    # Store current version to file
    echo $ACTUAL_VERSION > $GITHASH

    # Update aombuild
    git submodule update --init

    # Clean project files
    cargo clean

    # Get configure command from readme
    CONFIGURE_CMD=$(fgrep "cmake ../aom" README.md)

    # Handle cases where the cmake binary is absent or named differently
    if ! [ -x "$(command -v cmake)" ]; then
        if ! [ -x "$(command -v cmake3)" ]; then
            echo "\`cmake\` is required to build libaom. Aborting." >&2
            exit 1
        fi

        CONFIGURE_CMD=${CONFIGURE_CMD/cmake/cmake3}
    fi

    # Wipe and create aom_test folder
    rm -fR $AOM_TEST
    mkdir -p $AOM_TEST
    pushd $AOM_TEST

    echo CONFIGURE COMMAND
    echo $CONFIGURE_CMD
    eval $CONFIGURE_CMD

    # auto detect the number of cores and parallel build
    make -j$(nproc --all)
    popd

fi

# File containing the encoded sequence
ENC_FILE="enc_file.ivf"
# File containing the reconstructed sequence
REC_FILE="rec_file.y4m"
# File containing the decoded sequence
DEC_FILE="dec_file.y4m"

# Print the backtrace on error
export RUST_BACKTRACE=1

# Build and run encoder
BUILD_TYPE=""
if [ $IS_RELEASE == 1 ]; then
  BUILD_TYPE="--release"
fi

cargo run --bin rav1e $BUILD_TYPE -- $SEQ -o $ENC_FILE -s 3 -r $REC_FILE

# Decode
${AOM_TEST}/aomdec $ENC_FILE -o $DEC_FILE

# Input/Output compare
tail -n+2 $DEC_FILE > /tmp/dec_file
tail -n+2 $REC_FILE > /tmp/rec_file
cmp /tmp/dec_file /tmp/rec_file || printf '\e[1;31m%-6s\e[m\n\n' 'Desync detected!!!' && exit 1

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

# Repeat for high bit depth clips
#cargo run --bin rav1e --release -- $SEQ10 -o $ENC_FILE -s 3 -r $REC_FILE
#${AOM_TEST}/aomdec $ENC_FILE -o $DEC_FILE
#cmp <(tail -n+2 $DEC_FILE) <(tail -n+2 $REC_FILE)
#mpv --loop $DEC_FILE

#cargo run --bin rav1e --release -- $SEQ12 -o $ENC_FILE -s 3 -r $REC_FILE
#${AOM_TEST}/aomdec $ENC_FILE -o $DEC_FILE
#cmp <(tail -n+2 $DEC_FILE) <(tail -n+2 $REC_FILE)
#mpv --loop $DEC_FILE
