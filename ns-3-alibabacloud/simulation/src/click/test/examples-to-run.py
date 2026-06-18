#! /usr/bin/env python3
## -*- Mode: python; py-indent-offset: 4; indent-tabs-mode: nil; coding: utf-8; -*-

# A list of C++ examples to run in order to ensure that they remain
# buildable and runnable over time.  Each tuple in the list contains
#
#     (example_name, do_run, do_valgrind_run).
#
# See test.py for more information.
cpp_examples = [
    ("nsclick-simple-lan --clickConfigFolder=../../src/click/examples", "NSCLICK == True", "False"),
    ("nsclick-raw-wlan --clickConfigFolder=../../src/click/examples", "NSCLICK == True", "False"),
    ("nsclick-udp-client-server-csma --clickConfigFolder=../../src/click/examples", "NSCLICK == True", "False"),
    ("nsclick-udp-client-server-wifi --clickConfigFolder=../../src/click/examples", "NSCLICK == True", "False"),
    ("nsclick-routing --clickConfigFolder=../../src/click/examples", "NSCLICK == True", "False"),
    ("nsclick-defines --clickConfigFolder=../../src/click/examples", "NSCLICK == True", "False"),
]

# A list of Python examples to run in order to ensure that they remain
# runnable over time.  Each tuple in the list contains
#
#     (example_name, do_run).
#
# See test.py for more information.
python_examples = []
