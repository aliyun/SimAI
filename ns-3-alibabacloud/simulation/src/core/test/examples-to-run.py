#! /usr/bin/env python3
## -*- Mode: python; py-indent-offset: 4; indent-tabs-mode: nil; coding: utf-8; -*-

# A list of C++ examples to run in order to ensure that they remain
# buildable and runnable over time.  Each tuple in the list contains
#
#     (example_name, do_run, do_valgrind_run).
#
# See test.py for more information.
cpp_examples = [
    ("main-attribute-value", "True", "True"),
    ("main-callback", "True", "True"),
    ("sample-simulator", "True", "True"),
    ("main-ptr", "True", "True"),
    ("main-random-variable", "True", "False"),
    ("sample-random-variable", "True", "True"),
    ("test-string-value-formatting", "True", "True"),
]

# A list of Python examples to run in order to ensure that they remain
# runnable over time.  Each tuple in the list contains
#
#     (example_name, do_run).
#
# See test.py for more information.
python_examples = [
    ("sample-simulator.py", "True"),
]
