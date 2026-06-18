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

#include "ns3/example-as-test.h"
#include "ns3/system-path.h"

#include <vector>

using namespace ns3;

/**
 * \file
 * \ingroup examples-as-tests
 * Examples-as-tests test suite
 */

/**
 * \ingroup core-tests
 * \defgroup examples-as-tests Examples as tests test suite
 *
 * Runs several examples as tests in order to test ExampleAsTestSuite and ExampleAsTestCase.
 */
namespace ns3 {

namespace tests {

/**
 * \ingroup examples-as-tests
 * Run command line example as a test case
 */
class CommandLineExampleTestCase : public ExampleAsTestCase
{
public:
  /**
   * Default constructor
   */
  CommandLineExampleTestCase();

  /**
   * Destructor
   */
  virtual ~CommandLineExampleTestCase();

  /**
  * Override this function to filter the version string from
  * the command-line-example output.
  * Since the version changes each time a commit is made it shouldn't
  * be tested as part of the command-line-example output.
  * 
  * \returns The string of post-processing commands.
  */
  virtual std::string GetPostProcessingCommand (void) const;
};

CommandLineExampleTestCase::CommandLineExampleTestCase ()
    :   ExampleAsTestCase ("core-example-command-line", 
                           "command-line-example",
                           NS_TEST_SOURCEDIR)
{}

CommandLineExampleTestCase::~CommandLineExampleTestCase ()
{}

std::string
CommandLineExampleTestCase::GetPostProcessingCommand (void) const
{
  //Delete the line that starts with Program Version: 
  return std::string(R"__(| sed -e "/^Program Version:.*$/d")__");
}

/**
 * \ingroup examples-as-tests
 * Run examples as tests, checking stdout for regressions.
 */
class ExamplesAsTestsTestSuite : public TestSuite
{
public:
  ExamplesAsTestsTestSuite ();
};


ExamplesAsTestsTestSuite::ExamplesAsTestsTestSuite ()
  : TestSuite ("examples-as-tests-test-suite", UNIT)
{
  AddTestCase (new ExampleAsTestCase ("core-example-simulator", "sample-simulator", NS_TEST_SOURCEDIR));

  AddTestCase (new ExampleAsTestCase ("core-example-sample-random-variable", "sample-random-variable", NS_TEST_SOURCEDIR));

  AddTestCase (new CommandLineExampleTestCase ());
}

/**
 * \ingroup examples-tests
 * ExampleAsTestsTestSuite instance variable.
 * Tests multiple examples in a single TestSuite using AddTestCase to add the examples to the suite.
 */
static ExamplesAsTestsTestSuite g_examplesAsTestsTestSuite;

/**
 * \ingroup examples-tests
 * ExampleTestSuite instance variables.
 *
 * Tests ExampleTestSuite which runs a single example as test suite as specified in constructor arguments.
 */

static ExampleAsTestSuite g_exampleCommandLineTest ("core-example-simulator", "sample-simulator", NS_TEST_SOURCEDIR);

}  // namespace tests

}  // namespace ns3





