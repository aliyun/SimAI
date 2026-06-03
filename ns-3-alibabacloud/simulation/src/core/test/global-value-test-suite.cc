/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
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
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#include "ns3/global-value.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"

/**
 * \file
 * \ingroup core
 * \ingroup core-tests
 * \ingroup global-value-tests
 * GlobalValue test suite
 */

/**
 * \ingroup core-tests
 * \defgroup global-value-tests GlobalValue test suite
 */

namespace ns3 {

namespace tests {


/**
 * \ingroup global-value-tests
 * Test for the ability to get at a GlobalValue.
 */
class GlobalValueTestCase : public TestCase
{
public:
  /** Constructor. */
  GlobalValueTestCase ();
  /** Destructor. */
  virtual ~GlobalValueTestCase ()
  {}

private:
  virtual void DoRun (void);
};

GlobalValueTestCase::GlobalValueTestCase ()
  : TestCase ("Check GlobalValue mechanism")
{}

void
GlobalValueTestCase::DoRun (void)
{
  //
  // Typically these are static globals but we can make one on the stack to
  // keep it hidden from the documentation.
  //
  GlobalValue uint = GlobalValue ("TestUint", "help text",
                                  UintegerValue (10),
                                  MakeUintegerChecker<uint32_t> ());

  //
  // Make sure we can get at the value and that it was initialized correctly.
  //
  UintegerValue uv;
  uint.GetValue (uv);
  NS_TEST_ASSERT_MSG_EQ (uv.Get (), 10, "GlobalValue \"TestUint\" not initialized as expected");

  //
  // Remove the global value for a valgrind clean run
  //
  GlobalValue::Vector *vector = GlobalValue::GetVector ();
  for (GlobalValue::Vector::iterator i = vector->begin (); i != vector->end (); ++i)
    {
      if ((*i) == &uint)
        {
          vector->erase (i);
          break;
        }
    }
}

/**
 * \ingroup global-value-tests
 * The Test Suite that glues all of the Test Cases together.
 */
class GlobalValueTestSuite : public TestSuite
{
public:
  /** Constructor. */
  GlobalValueTestSuite ();
};

GlobalValueTestSuite::GlobalValueTestSuite ()
  : TestSuite ("global-value")
{
  AddTestCase (new GlobalValueTestCase);
}

/**
 * \ingroup global-value-tests
 * GlobalValueTestSuite instance variable.
 */
static GlobalValueTestSuite g_globalValueTestSuite;


}    // namespace tests

}  // namespace ns3

