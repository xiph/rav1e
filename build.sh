#!/bin/sh

# Stop on first error
set -e

#SEQ=!!!!! ENTER YOUR FAVORITE Y4M HERE !!!!!

unset IS_RELEASE

for arg in "$@"; do
  case $arg in
  --debug) unset IS_RELEASE ;; # Unneeded, but explicit
  --release) IS_RELEASE=true ;;
  *) continue ;;
  esac
done

if [ -z "$SEQ" ] && SEQ=small_input.y4m; then
  SEQ=tests/small_input.y4m
fi

if [ ! -f $SEQ ]; then
  echo "ERROR: Failed to find $SEQ" >&2
  echo "Please recheck the variables" >&2
  exit 1
fi

# File containing the encoded sequence
ENC_FILE=enc_file.ivf
# File containing the reconstructed sequence
REC_FILE=rec_file.y4m
# File containing the decoded sequence
DEC_FILE=dec_file.y4m

# Print the backtrace on error
export RUST_BACKTRACE=1

# Build and run encoder

if ! type cargo > /dev/null 2>&1; then
  echo "cargo not found" >&2
  exit 1
fi

if ! cargo build ${IS_RELEASE:+--release}; then
  e=$?
  echo "Failed to build rav1e" >&2
  exit $e
fi

if ! cargo run --bin rav1e ${IS_RELEASE:+--release} -- "$SEQ" -o $ENC_FILE -s 3 -r $REC_FILE; then
  echo "rav1e failed to run" >&2
  exit 1
fi

# Decode
if type aomdec > /dev/null 2>&1 &&
  aomdec $ENC_FILE | tee $DEC_FILE | tail -n+2 > dec_file; then
  # Input/Output compare
  if ! tail -n+2 $REC_FILE | cmp dec_file -; then
    if test "$(tput colors 2> /dev/null || echo 0)" -gt 8; then
      RED=$(tput setaf 1) RESET=$(tput sgr0)
    fi
    printf "${RED}%s${RESET}"'\n\n' "Desync detected!!!"
    rm dec_file
    exit 1
  fi
  rm dec_file 2> /dev/null
else
  echo "aomdec not found or failed, not doing decode tests"
fi

cat << 'EOF' > /dev/null
# Daala tools support coming soon
DAALA_TOOLS="../daala/tools/"
# Convert to png
${DAALA_TOOLS}/y4m2png -o out.png $DEC_FILE

# Compute and print PSNR (not working)
${DAALA_TOOLS}/dump_psnr $DEC_OUT $SEQ

# Compute and print CIEDE2000 (not working)
${DAALA_TOOLS}/dump_ciede2000.py $DEC_OUT $SEQ
EOF

# Show decoded sequence
# --pause
type mpv > /dev/null 2>&1 && mpv --loop $DEC_FILE

cat << 'EOF' > /dev/null
