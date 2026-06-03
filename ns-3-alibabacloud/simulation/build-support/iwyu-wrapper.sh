#!/usr/bin/env bash
echo >>"$1"/iwyu.log include-what-you-use  "${@:2}"
exec include-what-you-use "${@:2}" 2> >(tee -a "$1"/iwyu.log >&2)
