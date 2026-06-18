#! /usr/bin/env python3
# -*- Mode: python; py-indent-offset: 4; indent-tabs-mode: nil; coding: utf-8; -*-

def lp64_to_ilp32(lp64path, ilp32path):
    import re
    lp64bindings = None
    with open(lp64path, "r") as lp64file:
        lp64bindings = lp64file.read()
    with open(ilp32path, "w") as ilp32file:
        ilp32bindings = re.sub("unsigned long(?!( long))", "unsigned long long", lp64bindings)
        ilp32file.write(ilp32bindings)

if __name__ == "__main__":
    import sys
    print(sys.argv)
    exit(lp64_to_ilp32(sys.argv[1], sys.argv[2]))