#! /usr/bin/env python3
## -*- Mode: python; py-indent-offset: 4; indent-tabs-mode: nil; coding: utf-8; -*-

# A list of C++ examples to run in order to ensure that they remain
# buildable and runnable over time.  Each tuple in the list contains
#
#     (example_name, do_run, do_valgrind_run).
#
# See test.py for more information.
cpp_examples = [
    ("wave-simple-80211p", "True", "True"),
    ("wave-simple-device", "True", "True"),
    ("vanet-routing-compare --totaltime=2 --80211Mode=1", "True", "True"),
    ("vanet-routing-compare --totaltime=2 --80211Mode=2", "True", "True"),
    ("vanet-routing-compare --totaltime=2 --80211Mode=3", "True", "True"),
]

# A list of Python examples to run in order to ensure that they remain
# runnable over time.  Each tuple in the list contains
#
#     (example_name, do_run).
#
# See test.py for more information.
python_examples = []
