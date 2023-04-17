#!/bin/sh

for arg in "$@"; do
  arg=$(echo $arg | perl -pne \
    's/(?:(?<=\s)|^)build(?=\s)/cinstall --library-type staticlib --library-type cdylib --prefix dist/')
  set -- "$@" "$arg"
  shift
done

set -ex
exec "$@"
