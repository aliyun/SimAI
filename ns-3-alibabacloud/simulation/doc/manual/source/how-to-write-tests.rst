.. include:: replace.txt
.. highlight:: cpp

How to write tests
------------------

A primary goal of the ns-3 project is to help users to improve the 
validity and credibility of their results.  There are many elements
to obtaining valid models and simulations, and testing is a major
component.  If you contribute models or examples to ns-3, you may
be asked to contribute test code.  Models that you contribute will be
used for many years by other people, who probably have no idea upon
first glance whether the model is correct.  The test code that you
write for your model will help to avoid future regressions in
the output and will aid future users in understanding the verification
and bounds of applicability of your models.

There are many ways to verify the correctness of a model's implementation.
In this section,
we hope to cover some common cases that can be used as a guide to
writing new tests.

Sample TestSuite skeleton
*************************

When starting from scratch (i.e. not adding a TestCase to an existing
TestSuite), these things need to be decided up front:

* What the test suite will be called 
* What type of test it will be (Build Verification Test, Unit Test,
  System Test, or Performance Test)
* Where the test code will live (either in an existing ns-3 module or
  separately in src/test/ directory).  You will have to edit the wscript
  file in that directory to compile your new code, if it is a new file.

A program called ``utils/create-module.py`` is a good starting point.
This program can be invoked such as ``create-module.py router`` for
a hypothetical new module called ``router``.  Once you do this, you
will see a ``router`` directory, and a ``test/router-test-suite.cc`` 
test suite.  This file can be a starting point for your initial test.
This is a working test suite, although the actual tests performed are
trivial.  Copy it over to your module's test directory, and do a global
substitution of "Router" in that file for something pertaining to
the model that you want to test.  You can also edit things such as a
more descriptive test case name. 

You also need to add a block into your wscript to get this test to
compile:

.. sourcecode:: python

    module_test.source = [
        'test/router-test-suite.cc',
        ]

Before you actually start making this do useful things, it may help
to try to run the skeleton.  Make sure that ns-3 has been configured with
the "--enable-tests" option.  Let's assume that your new test suite
is called "router" such as here:

::

  RouterTestSuite::RouterTestSuite ()
    : TestSuite ("router", UNIT)

Try this command:

.. sourcecode:: bash

  $ ./test.py -s router

Output such as below should be produced:

.. sourcecode:: text

  PASS: TestSuite router
  1 of 1 tests passed (1 passed, 0 skipped, 0 failed, 0 crashed, 0 valgrind errors)

See src/lte/test/test-lte-antenna.cc for a worked example.

Test macros
***********

There are a number of macros available for checking test program
output with expected output.  These macros are defined in 
``src/core/model/test.h``.

The main set of macros that are used include the following:

::

  NS_TEST_ASSERT_MSG_EQ(actual, limit, msg)
  NS_TEST_ASSERT_MSG_NE(actual, limit, msg)
  NS_TEST_ASSERT_MSG_LT(actual, limit, msg)
  NS_TEST_ASSERT_MSG_GT(actual, limit, msg)
  NS_TEST_ASSERT_MSG_EQ_TOL(actual, limit, tol, msg)  

The first argument ``actual`` is the value under test, the second value
``limit`` is the expected value (or the value to test against), and the
last argument ``msg`` is the error message to print out if the test fails.

The first four macros above test for equality, inequality, less than, or
greater than, respectively.  The fifth macro above tests for equality,
but within a certain tolerance.  This variant is useful when testing
floating point numbers for equality against a limit, where you want to
avoid a test failure due to rounding errors.

Finally, there are variants of the above where the keyword ``ASSERT``
is replaced by ``EXPECT``.  These variants are designed specially for
use in methods (especially callbacks) returning void.  Reserve their
use for callbacks that you use in your test programs; otherwise,
use the ``ASSERT`` variants.

How to add an example program to the test suite
***********************************************

There are two methods for adding an example program to the the test
suite.  Normally an example is added using only one of these methods
to avoid running the example twice.

First, you can "smoke test" that examples compile and run successfully
to completion (without memory leaks) using the ``examples-to-run.py``
script located in your module's test directory.  Briefly, by including
an instance of this file in your test directory, you can cause the
test runner to execute the examples listed.  It is usually best to
make sure that you select examples that have reasonably short run
times so as to not bog down the tests.  See the example in
``src/lte/test/`` directory.  The exit status of the example will be
checked when run and a non-zero exit status can be used to indicate
that the example has failed.  This is the easiest way to add an example
to the test suite but has limited checks.

The second method you can use to add an example to the test suite is
more complicated but enables checking of the example output
(``std::out`` and ``std::err``).  This approach uses the test suite
framework with a specialized ``TestSuite`` or ``TestCase`` class
designed to run an example and compare the output with a specified
known "good" reference file.  To use an example program as a test you
need to create a test suite file and add it to the appropriate list in
your module wscript file. The "good" output reference file needs to be
generated for detecting regressions.

If you are thinking about using this class, strongly consider using a
standard test instead.  The TestSuite class has better checking using
the ``NS_TEST_*`` macros and in almost all cases is the better approach.
If your test can be done with a TestSuite class you will be asked by
the reviewers to rewrite the test when you do a pull request.

Let's assume your module is called ``mymodule``, and the example
program is ``mymodule/examples/mod-example.cc``.  First you should
create a test file ``mymodule/test/mymodule-examples-test-suite.cc``
which looks like this:

::

   #include "ns3/example-as-test.h"
   static ns3::ExampleAsTestSuite g_modExampleOne ("mymodule-example-mod-example-one", "mod-example", NS_TEST_SOURCEDIR, "--arg-one");
   static ns3::ExampleAsTestSuite g_modExampleTwo ("mymodule-example-mod-example-two", "mod-example", NS_TEST_SOURCEDIR, "--arg-two");

The arguments to the constructor are the name of the test suite, the
example to run, the directory that contains the "good" reference file
(the macro ``NS_TEST_SOURCEDIR`` is normally the correct directory),
and command line arguments for the example.  In the preceding code the
same example is run twice with different arguments.

You then need to add that newly created test suite file to the list of
test sources in ``mymodule/wscript``.  Building of examples
is an option so you need to guard the inclusion of the test suite:

.. sourcecode:: python

   if (bld.env['ENABLE_EXAMPLES']):
      module.source.append('model/mymodule-examples-test-suite.cc')

Since you modified a wscript file you need to reconfigure and rebuild
everything.

You just added new tests so you will need to generate the "good"
output reference files that will be used to verify the example:

.. sourcecode :: bash

   ./test.py --suite="mymodule-example-*" --update

This will run all tests starting with "mymodule-example-" and save new
"good" reference files.  Updating the reference files should be done
when you create the test and whenever output changes.  When updating
the reference output you should inspect it to ensure that it is valid.
The reference files should be committed with the new test.

This completes the process of adding a new example.

You can now run the test with the standard ``test.py`` script.  For
example to run the suites you just added:

.. sourcecode:: bash

   ./test.py --suite="mymodule-example-*"

This will run all ``mymodule-example-...`` tests and report whether they
produce output matching the reference files.

You can also add multiple examples as test cases to a ``TestSuite``
using ``ExampleAsTestCase``.  See
``src/core/test/examples-as-tests-test-suite.cc`` for examples of
setting examples as tests.

When setting up an example for use by this class you should be very
careful about what output the example generates.  For example, writing
output which includes simulation time (especially high resolution
time) makes the test sensitive to potentially minor changes in event
times.  This makes the reference output hard to verify and hard to
keep up-to-date.  Output as little as needed for the example and
include only behavioral state that is important for determining if the
example has run correctly.

Testing for boolean outcomes
****************************

Testing outcomes when randomness is involved
********************************************

Testing output data against a known distribution
************************************************

Providing non-trivial input vectors of data
*******************************************

Storing and referencing non-trivial output data
***********************************************

Presenting your output test data
********************************

