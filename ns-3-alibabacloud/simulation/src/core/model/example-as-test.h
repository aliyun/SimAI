/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Lawrence Livermore National Laboratory
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Peter D. Barnes, Jr. <pdbarnes@llnl.gov>
 */

#ifndef NS3_EXAMPLE_AS_TEST_SUITE_H
#define NS3_EXAMPLE_AS_TEST_SUITE_H

#include "ns3/test.h"

#include <string>

/**
 * \file
 * \ingroup testing
 * Enable examples to be run as meaningful tests.
 * Declaration of classes ns3::ExampleAsTestSuite and ns3::ExampleAsTestCase.
 */

namespace ns3 {


/**
 * \ingroup testing
 * Execute an example program as a test, by comparing the output
 * to a reference file.
 *
 * User can subclass and override the GetCommandTemplate and
 * GetPostProcessingCommand methods if more complex
 * example invocation patterns are required.
 *
 * \see examples-as-tests-test-suite.cc
 */
class ExampleAsTestCase : public TestCase
{
public:
  /**
   * Constructor.
   * \param [in] name The test case name, typically the program name
   *                  and summary of the arguments, such as `my-example-foo`
   * \param [in] program The actual example program names, such as `my-example`
   * \param [in] dataDir The location of the reference file.
   *                  This is normally provided by the symbol
   *                  `NS_TEST_SOURCEDIR` in the `module-examples-test-suite.cc`
   *                  file.
   *                  The reference file should be named after
   *                  the test case name,
   *                  for example `my-example-foo.log`.  If you use
   *                  the `--update` argument to `test.py` or
   *                  `test-runner` the reference file will be created
   *                  with the correct name.
   * \param [in] args Any additional arguments to the program.
   */
  ExampleAsTestCase (const std::string name,
                     const std::string program,
                     const std::string dataDir,
                     const std::string args = "");

  /** Destructor. */
  virtual ~ExampleAsTestCase (void);

  /**
   * Customization point for more complicated patterns
   * to invoke the example program.
   *
   * \returns The string to be given to the `ns3 --command-template=` argument.
   */
  virtual std::string GetCommandTemplate (void) const;

  /**
   * Customization point for tests requiring post-processing of stdout.
   *
   * For example to sort return "| sort"
   *
   * Default is "", no processing step.
   *
   * \returns The string of post-processing commands
   */
  virtual std::string GetPostProcessingCommand (void) const;

  // Inherited
  virtual void DoRun (void);

protected:
  std::string m_program;  /**< The program to run. */
  std::string m_dataDir;  /**< The source directory for the test. */
  std::string m_args;     /**< Any additional arguments to the program. */

};  // class ExampleAsTestCase

/**
 * \ingroup testing
 * Execute an example program as a test suite.
 *
 * You can use this TestSuite to add an example to the test suite with
 * checking of the example output (`std::out` and `std::err`).  This
 * is an alternative to adding an example using the
 * `examples-to-run.py` file. The key difference between the two
 * methods is what criteria is used to for success.  Examples added to
 * `examples-to-run.py` will be run and the exit status checked
 * (non-zero indicates failure).  ExampleAsTestSuite adds checking of
 * output against a specified known "good" reference file.  
 *
 * \warning If you are thinking about using this class, strongly
 * consider using a standard test instead.  The TestSuite class has
 * better checking using the NS_TEST_* macros and in almost all cases
 * is the better approach.  If your test can be done with a TestSuite
 * class you will be asked by the reviewers to rewrite the test when
 * you do a pull request.
 *
 * \par Test Addition
 *
 * To use an example program as a test you need to create a test suite
 * file and add it to the appropriate list in your module CMakeLists.txt
 * file. The "good" output reference file needs to be generated for
 * detecting regressions.
 *
 * Let's assume your module is called `mymodule`, and the example
 * program is `mymodule/examples/mod-example.cc`.  First you should
 * create a test file `mymodule/test/mymodule-examples-test-suite.cc`
 * which looks like this:
 * 
 * \code{.cpp}
 * #include "ns3/example-as-test.h"
 * static ns3::ExampleAsTestSuite g_modExampleOne ("mymodule-example-mod-example-one", "mod-example", NS_TEST_SOURCEDIR, "--arg-one");
 * static ns3::ExampleAsTestSuite g_modExampleTwo ("mymodule-example-mod-example-two", "mod-example", NS_TEST_SOURCEDIR, "--arg-two");
 * \endcode
 *
 * The arguments to the constructor is the name of the test suite, the
 * example to run, the directory that contains the "good" reference file
 * (the macro `NS_TEST_SOURCEDIR` is normally the correct directory),
 * and command line arguments for the example.  In the preceding code
 * the same example is run twice with different arguments.
 *
 * You then need to add that newly created test suite file to the list
 * of test sources in `mymodule/CMakeLists.txt`.   Building of examples
 * is an option so you need to guard the inclusion of the test suite:
 *
 * \code{.py}
 * if (bld.env['ENABLE_EXAMPLES']):
 *    module.source.append('model/mymodule-examples-test-suite.cc')
 * \endcode
 *
 * Since you modified a CMakeLists.txt file you need to reconfigure and
 * rebuild everything.
 *
 * You just added new tests so you will need to generate the "good"
 * output reference files that will be used to verify the example:
 *
 * `./test.py --suite="mymodule-example-*" --update`
 *
 * This will run all tests starting with "mymodule-example-" and save
 * new "good" reference files.  Updating the reference file should be
 * done when you create the test and whenever output changes.  When
 * updating the reference output you should inspect it to ensure that
 * it is valid.  The reference files should be committed with the new
 * test.
 *
 * \par Test Verification
 *
 * You can run the test with the standard `test.py` script.  For
 * example to run the suites you just added:
 *
 * `./test.py --suite="mymodule-example-*"`
 *
 * This will run all `mymodule-example-...` tests and report whether they
 * produce output matching the reference files.
 * 
 * \par Writing good examples for testing
 *
 * When setting up an example for use by this class you should be very
 * careful about what output the example generates.  For example,
 * writing output which includes simulation time (especially high
 * resolution time) makes the test sensitive to potentially minor
 * changes in event times.  This makes the reference output hard to
 * verify and hard to keep up-to-date.  Output as little as needed for
 * the example and include only behavioral state that is important for
 * determining if the example has run correctly.
 * 
 */
class ExampleAsTestSuite : public TestSuite
{
public:
  /**
   * \copydoc ExampleAsTestCase::ExampleAsTestCase
   * \param [in] duration Amount of time this test takes to execute
   *             (defaults to QUICK).
   */
  ExampleAsTestSuite (const std::string name,
                      const std::string program,
                      const std::string dataDir,
                      const std::string args = "",
                      const TestDuration duration = QUICK);
};  // class ExampleAsTestSuite


} // namespace ns3

#endif  /* NS3_EXAMPLE_TEST_SUITE_H */
