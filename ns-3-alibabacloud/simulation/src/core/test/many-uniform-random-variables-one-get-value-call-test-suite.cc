/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 University of Washington
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
 * Author: Mitch Watrous (watrous@u.washington.edu)
 */

#include "ns3/test.h"
#include "ns3/config.h"
#include "ns3/double.h"
#include "ns3/random-variable-stream.h"
#include <vector>


/**
 * \file
 * \ingroup core-tests
 * \ingroup randomvariable
 * \ingroup randomvariable-tests
 * Test for many uniform random variable streams.
 */

namespace ns3 {

namespace tests {


/**
 * \ingroup randomvariable-tests
 * Test case for many uniform distribution random variable stream generators
 */
class ManyUniformRandomVariablesOneGetValueCallTestCase : public TestCase
{
public:
  /** Constructor. */
  ManyUniformRandomVariablesOneGetValueCallTestCase ();
  /** Destructor. */
  virtual ~ManyUniformRandomVariablesOneGetValueCallTestCase ();

private:
  virtual void DoRun (void);
};

ManyUniformRandomVariablesOneGetValueCallTestCase::ManyUniformRandomVariablesOneGetValueCallTestCase ()
  : TestCase ("Many Uniform Random Variables with One GetValue() Call")
{}

ManyUniformRandomVariablesOneGetValueCallTestCase::~ManyUniformRandomVariablesOneGetValueCallTestCase ()
{}

void
ManyUniformRandomVariablesOneGetValueCallTestCase::DoRun (void)
{
  const double min = 0.0;
  const double max = 10.0;

  Config::SetDefault ("ns3::UniformRandomVariable::Min", DoubleValue (min));
  Config::SetDefault ("ns3::UniformRandomVariable::Max", DoubleValue (max));

  // Get 1 value from many uniform random number generators.
  double value;
  const int count = 1000000;
  std::vector< Ptr<UniformRandomVariable> > uniformStreamVector (count);
  for (int i = 0; i < count; i++)
    {
      uniformStreamVector.push_back (CreateObject<UniformRandomVariable> ());
      value = uniformStreamVector.back ()->GetValue ();

      NS_TEST_ASSERT_MSG_GT (value, min, "Value less than minimum.");
      NS_TEST_ASSERT_MSG_LT (value, max, "Value greater than maximum.");
    }
}

/**
 * \ingroup randomvariable-tests
 * Test suite for many uniform distribution random variable stream generators
 */
class ManyUniformRandomVariablesOneGetValueCallTestSuite : public TestSuite
{
public:
  /** Constructor. */
  ManyUniformRandomVariablesOneGetValueCallTestSuite ();
};

ManyUniformRandomVariablesOneGetValueCallTestSuite::ManyUniformRandomVariablesOneGetValueCallTestSuite ()
  : TestSuite ("many-uniform-random-variables-one-get-value-call", PERFORMANCE)
{
  AddTestCase (new ManyUniformRandomVariablesOneGetValueCallTestCase);
}

/**
 * \ingroup randomvariable-tests
 * ManuUniformRandomVariablesOneGetValueCallTestSuite instance variable.
 */
static ManyUniformRandomVariablesOneGetValueCallTestSuite g_manyUniformRandomVariablesOneGetValueCallTestSuite;


}    // namespace tests

}  // namespace ns3
