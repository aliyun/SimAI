.. include:: replace.txt
.. highlight:: bash


Enabling/disabling |ns3| Tests and Examples
-------------------------------------------

The |ns3| distribution includes many examples and tests that are used to validate the |ns3| system.  Users, however, may not always want these examples and tests to be run for their installation of |ns3|.

This chapter discusses how to build |ns3| with or without its examples and tests.

How to enable/disable examples and tests in |ns3|
*************************************************

There are 3 ways to enable/disable examples and tests in |ns3|:

#. Using build.py when |ns3| is built for the first time
#. Using ns3 once |ns3| has been built
#. Using the |ns3| configuration file once |ns3| has been built

Enable/disable examples and tests using build.py
++++++++++++++++++++++++++++++++++++++++++++++++

You can use build.py to enable/disable examples and tests when |ns3| is built for the first time.

By default, examples and tests are not built in |ns3|.  

From the ns-3-allinone directory, you can build |ns3| without any
examples or tests simply by doing: ::

  $ ./build.py

Running test.py in the top level |ns3| directory now will cause no examples or tests to be run:
 
.. sourcecode:: text

  0 of 0 tests passed (0 passed, 0 skipped, 0 failed, 0 crashed, 0 valgrind errors)

If you would like build |ns3| with examples and tests, then do the following from the ns-3-allinone directory: ::

  $ ./build.py --enable-examples --enable-tests

Running test.py in the top level |ns3| directory will cause all of the examples and tests to be run:
 
.. sourcecode:: text

  170 of 170 tests passed (170 passed, 0 skipped, 0 failed, 0 crashed, 0 valgrind errors)

Enable/disable examples and tests using ns3
+++++++++++++++++++++++++++++++++++++++++++

You can use ns3 to enable/disable examples and tests once |ns3| has been built.

By default, examples and tests are not built in |ns3|.  

From the top level |ns3| directory, you can build |ns3| without any
examples or tests simply by doing: ::

  $ ./ns3 configure
  $ ./ns3 build

Running test.py now will cause no examples or tests to be run:
 
.. sourcecode:: text

  0 of 0 tests passed (0 passed, 0 skipped, 0 failed, 0 crashed, 0 valgrind errors)

If you would like build |ns3| with examples and tests, then do the following from the top level |ns3| directory: ::

  $ ./ns3 configure --enable-examples --enable-tests
  $ ./ns3 build

Running test.py will cause all of the examples and tests to be run:
 
.. sourcecode:: text

  170 of 170 tests passed (170 passed, 0 skipped, 0 failed, 0 crashed, 0 valgrind errors)

Enable/disable examples and tests using the |ns3| configuration file
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A configuration file, .ns3rc, has been added to |ns3| that allows users to specify whether examples and tests should be built or not.  You can use this file to enable/disable examples and tests once |ns3| has been built.

When enabling disabling examples and tests, the precedence rules are as follows:

#. the --enable-examples/--disable-examples configure strings override any .ns3rc file
#. the --enable-tests/--disable-tests configure strings override any .ns3rc file
#. the .ns3rc file in the top level |ns3| directory is next consulted, if present
#. the system searches for ~/.ns3rc if the .ns3rc file was not found in the previous step

If none of the above exists, then examples and tests will not be built.

The maintained version of the .ns3rc file in the |ns3| source code repository resides in the ``utils`` directory.  The reason for this is if it were in the top-level directory of the repository, it would be prone to accidental checkins from maintainers that enable the modules they want to use.  Therefore, users need to manually copy the .ns3rc from the ``utils`` directory to their preferred place (top level directory or their home directory) to enable persistent enabling of examples and tests.

Assuming that you are in the top level |ns3| directory, you can get a copy of the .ns3rc file that is in the ``utils`` directory as follows: ::

  $ cp utils/.ns3rc .

The .ns3rc file should now be in your top level |ns3| directory, and it contains the following:

.. sourcecode:: python

  #! /usr/bin/env python
  
  # A list of the modules that will be enabled when ns-3 is run.
  # Modules that depend on the listed modules will be enabled also.
  #
  # All modules can be enabled by choosing 'all_modules'.
  modules_enabled = ['all_modules']
  
  # Set this equal to true if you want examples to be run.
  examples_enabled = False
  
  # Set this equal to true if you want tests to be run.
  tests_enabled = False

From the top level |ns3| directory, you can build |ns3| without any
examples or tests simply by doing: ::

  $ ./ns3 configure
  $ ./ns3 build

Running test.py now will cause no examples or tests to be run:
 
.. sourcecode:: text

  0 of 0 tests passed (0 passed, 0 skipped, 0 failed, 0 crashed, 0 valgrind errors)

If you would like build |ns3| with examples and tests, use your
favorite editor to change the values in the .ns3rc file for
examples_enabled and tests_enabled file to be True:

.. sourcecode:: python

  #! /usr/bin/env python
  
  # A list of the modules that will be enabled when ns-3 is run.
  # Modules that depend on the listed modules will be enabled also.
  #
  # All modules can be enabled by choosing 'all_modules'.
  modules_enabled = ['all_modules']
  
  # Set this equal to true if you want examples to be run.
  examples_enabled = True
  
  # Set this equal to true if you want tests to be run.
  tests_enabled = True

From the top level |ns3| directory, you can build |ns3| with examples
and tests simply by doing: ::

  $ ./ns3 configure
  $ ./ns3 build

Running test.py will cause all of the examples and tests to be run:
 
.. sourcecode:: text

  170 of 170 tests passed (170 passed, 0 skipped, 0 failed, 0 crashed, 0 valgrind errors)
