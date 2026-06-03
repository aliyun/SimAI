#! /usr/bin/env python3
## -*- Mode: python; py-indent-offset: 4; indent-tabs-mode: nil; coding: utf-8; -*-

# A list of C++ examples to run in order to ensure that they remain
# buildable and runnable over time.  Each tuple in the list contains
#
#     (example_name, do_run, do_valgrind_run).
#
# See test.py for more information.
cpp_examples = [
    ("adaptive-red-tests --testNumber=1", "True", "True"),
    ("adaptive-red-tests --testNumber=2", "True", "True"),
    ("adaptive-red-tests --testNumber=3", "True", "True"),
    ("adaptive-red-tests --testNumber=4", "True", "True"),
    ("adaptive-red-tests --testNumber=6", "True", "True"),
    ("adaptive-red-tests --testNumber=7", "True", "True"),
    ("adaptive-red-tests --testNumber=8", "True", "True"),
    ("adaptive-red-tests --testNumber=9", "True", "True"),
    ("adaptive-red-tests --testNumber=10", "True", "True"),
    ("adaptive-red-tests --testNumber=11", "True", "True"),
    ("adaptive-red-tests --testNumber=12", "True", "True"),
    ("adaptive-red-tests --testNumber=13", "True", "True"),
    ("adaptive-red-tests --testNumber=14", "True", "True"),
    ("adaptive-red-tests --testNumber=15", "True", "True"),
    ("codel-vs-pfifo-asymmetric --routerWanQueueDiscType=PfifoFast --simDuration=10", "True", "True"),
    ("codel-vs-pfifo-asymmetric --routerWanQueueDiscType=CoDel --simDuration=10", "True", "False"),
    ("codel-vs-pfifo-basic-test --queueDiscType=PfifoFast --simDuration=10", "True", "False"),
    ("codel-vs-pfifo-basic-test --queueDiscType=CoDel --simDuration=10", "True", "False"),
    ("pfifo-vs-red --queueDiscType=PfifoFast", "True", "True"),
    ("pfifo-vs-red --queueDiscType=PfifoFast --modeBytes=1", "True", "False"),
    ("pfifo-vs-red --queueDiscType=RED", "True", "True"),
    ("pfifo-vs-red --queueDiscType=RED --modeBytes=1", "True", "False"),
    ("red-tests --testNumber=1", "True", "True"),
    ("red-tests --testNumber=3", "True", "False"),
    ("red-tests --testNumber=4", "True", "False"),
    ("red-tests --testNumber=5", "True", "False"),
    ("red-vs-ared --queueDiscType=RED", "True", "True"),
    ("red-vs-ared --queueDiscType=RED --modeBytes=true", "True", "False"),
    ("red-vs-ared --queueDiscType=ARED", "True", "True"),
    ("red-vs-ared --queueDiscType=ARED --modeBytes=true", "True", "False"),
]

# A list of Python examples to run in order to ensure that they remain
# runnable over time.  Each tuple in the list contains
#
#     (example_name, do_run).
#
# See test.py for more information.
python_examples = []
