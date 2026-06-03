#! /usr/bin/env python3
## -*- Mode: python; py-indent-offset: 4; indent-tabs-mode: nil; coding: utf-8; -*-

# A list of C++ examples to run in order to ensure that they remain
# buildable and runnable over time.  Each tuple in the list contains
#
#     (example_name, do_run, do_valgrind_run).
#
# See test.py for more information.
cpp_examples = [
    ("dummy-network", "True", "True"),
    ("fd-emu-ping", "False", "True"),
    ("fd-emu-onoff", "False", "True"),
    ("fd-emu-udp-echo", "False", "True"),
    ("realtime-dummy-network", "False", "True"),
    ("fd2fd-onoff", "True", "True"),
    ("fd-tap-ping", "False", "True"),
    ("realtime-fd2fd-onoff", "False", "True"),
]

# A list of Python examples to run in order to ensure that they remain
# runnable over time.  Each tuple in the list contains
#
#     (example_name, do_run).
#
# See test.py for more information.
