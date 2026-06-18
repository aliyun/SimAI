/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 */

// An essential include is test.h
#include "ns3/test.h"

/**
 * \file
 * \ingroup core-tests
 * \ingroup testing
 * \ingroup testing-example
 *
 * Example use of TestSuite.
 *
 * Assume this is the test suite for `class Class`
 * in module 'module'.
 *
 * Document the file as follows.  (Note: for purposes of this
 * illustration Doxygen comments are set off with triple  `\\\\\\';
 * you should use the normal Javadoc format. )
 *
 * Add it to the groups module-tests and class-tests:
 *
 * \verbatim /// \file
/// \ingroup module-tests
/// \ingroup class-tests
/// Class test suite. \endverbatim
 *
 * Define the class-tests group:
 * \verbatim /// \ingroup module-tests
/// \defgroup class-tests Class test suite \endverbatim
 *
 * Make sure test.h is included:
 * \verbatim #include "ns3/test.h" \endverbatim
 *
 * Put the test code in namespace ns3::tests.
 * \verbatim namespace ns3 {
  namespace tests { \endverbatim
 *
 * Write your test cases and final test suite, and put them in
 * your test group:
 * \verbatim /// \ingroup class-tests
class ClassTestSuite : public TestSuite {...}; \endverbatim
 *
 * Create a static variable to hold the instance of your test suite:
 * \verbatim /// \ingroup class-tests
/// ClassTestSuite instance variable.
static ClassTestSuite g_classTestSuite; \endverbatim
 *
 * Finally, close the ingroup and namespace blocks:
 * \verbatim   }  // namespace tests
}  // namespace ns3 \endverbatim
*/


/**
 * \ingroup core-tests
 * \defgroup testing-example Example use of TestSuite
 */

namespace ns3 {

namespace tests {


/**
 * \ingroup testing-example
 * This is an example TestCase.
 */
class SampleTestCase1 : public TestCase
{
public:
  /** Constructor. */
  SampleTestCase1 ();
  /** Destructor. */
  virtual ~SampleTestCase1 ();

private:
  virtual void DoRun (void);
};

/** Add some help text to this case to describe what it is intended to test. */
SampleTestCase1::SampleTestCase1 ()
  : TestCase ("Sample test case (does nothing)")
{}

/**
 * This destructor does nothing but we include it as a reminder that
 * the test case should clean up after itself
 */
SampleTestCase1::~SampleTestCase1 ()
{}

/**
 * This method is the pure virtual method from class TestCase that every
 * TestCase must implement
 */
void
SampleTestCase1::DoRun (void)
{
  // A wide variety of test macros are available in src/core/test.h
  NS_TEST_ASSERT_MSG_EQ (true, true, "true doesn't equal true for some reason");
  // Use this one for floating point comparisons
  NS_TEST_ASSERT_MSG_EQ_TOL (0.01, 0.01, 0.001, "Numbers are not equal within tolerance");
}

/**
 * \ingroup testing-example
 * The TestSuite class names the TestSuite, identifies what type of TestSuite,
 * and enables the TestCases to be run.  Typically, only the constructor for
 * this class must be defined
 */
class SampleTestSuite : public TestSuite
{
public:
  /** Constructor. */
  SampleTestSuite ();
};

SampleTestSuite::SampleTestSuite ()
  : TestSuite ("sample")
{
  AddTestCase (new SampleTestCase1);
}

// Do not forget to allocate an instance of this TestSuite
/**
 * \ingroup testing-example
 * SampleTestSuite instance variable.
 */
static SampleTestSuite g_sampleTestSuite;


}    // namespace tests

}  // namespace ns3
