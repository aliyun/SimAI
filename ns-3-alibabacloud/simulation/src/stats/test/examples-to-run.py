#! /usr/bin/env python3
## -*- Mode: python; py-indent-offset: 4; indent-tabs-mode: nil; coding: utf-8; -*-

# A list of C++ examples to run in order to ensure that they remain
# buildable and runnable over time.  Each tuple in the list contains
#
#     (example_name, do_run, do_valgrind_run).
#
# See test.py for more information.
cpp_examples = [
    ("double-probe-example", "True", "True"),
    ("file-aggregator-example", "True", "True"),
    ("file-helper-example", "True", "True"),
    ("gnuplot-aggregator-example", "True", "True"),
    ("gnuplot-example", "False", "False"),
    ("gnuplot-helper-example", "True", "True"),
]

# A list of Python examples to run in order to ensure that they remain
# runnable over time.  Each tuple in the list contains
#
#     (example_name, do_run).
#
# See test.py for more information.
python_examples = []
