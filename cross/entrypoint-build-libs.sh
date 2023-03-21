#!/bin/sh

for arg in "$@"; do
  arg=$(echo $arg | perl -pne \
    's/(?:(?<=\s)|^)build(?=\s)/cbuild --library-type staticlib --library-type cdylib/')
  set -- "$@" "$arg"
  shift
done

set -ex
exec "$@"
