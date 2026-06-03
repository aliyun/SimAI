/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 IITP RAS
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
 * Based on
 *      NS-2 AODV model developed by the CMU/MONARCH group and optimized and
 *      tuned by Samir Das and Mahesh Marina, University of Cincinnati;
 *
 *      AODV-UU implementation by Erik Nordström of Uppsala University
 *      http://core.it.uu.se/core/index.php/AODV-UU
 *
 * Authors: Elena Buchatskaia <borovkovaes@iitp.ru>
 *          Pavel Boyko <boyko@iitp.ru>
 */
#include "ns3/aodv-id-cache.h"
#include "ns3/test.h"

namespace ns3 {
namespace aodv {

/**
 * \ingroup aodv
 * \defgroup aodv-test AODV module tests
 */


/**
 * \ingroup aodv-test
 * \ingroup tests
 *
 * \brief Unit test for id cache
 */
class IdCacheTest : public TestCase
{
public:
  IdCacheTest () : TestCase ("Id Cache"),
                   cache (Seconds (10))
  {
  }
  virtual void DoRun ();

private:
  /// Timeout test function #1
  void CheckTimeout1 ();
  /// Timeout test function #2
  void CheckTimeout2 ();
  /// Timeout test function #3
  void CheckTimeout3 ();

  /// ID cache
  IdCache cache;
};

void
IdCacheTest::DoRun ()
{
  NS_TEST_EXPECT_MSG_EQ (cache.GetLifeTime (), Seconds (10), "Lifetime");
  NS_TEST_EXPECT_MSG_EQ (cache.IsDuplicate (Ipv4Address ("1.2.3.4"), 3), false, "Unknown ID & address");
  NS_TEST_EXPECT_MSG_EQ (cache.IsDuplicate (Ipv4Address ("1.2.3.4"), 4), false, "Unknown ID");
  NS_TEST_EXPECT_MSG_EQ (cache.IsDuplicate (Ipv4Address ("4.3.2.1"), 3), false, "Unknown address");
  NS_TEST_EXPECT_MSG_EQ (cache.IsDuplicate (Ipv4Address ("1.2.3.4"), 3), true, "Known address & ID");
  cache.SetLifetime (Seconds (15));
  NS_TEST_EXPECT_MSG_EQ (cache.GetLifeTime (), Seconds (15), "New lifetime");
  cache.IsDuplicate (Ipv4Address ("1.1.1.1"), 4);
  cache.IsDuplicate (Ipv4Address ("1.1.1.1"), 4);
  cache.IsDuplicate (Ipv4Address ("2.2.2.2"), 5);
  cache.IsDuplicate (Ipv4Address ("3.3.3.3"), 6);
  NS_TEST_EXPECT_MSG_EQ (cache.GetSize (), 6, "trivial");

  Simulator::Schedule (Seconds (5), &IdCacheTest::CheckTimeout1, this);
  Simulator::Schedule (Seconds (11), &IdCacheTest::CheckTimeout2, this);
  Simulator::Schedule (Seconds (30), &IdCacheTest::CheckTimeout3, this);
  Simulator::Run ();
  Simulator::Destroy ();
}

void
IdCacheTest::CheckTimeout1 ()
{
  NS_TEST_EXPECT_MSG_EQ (cache.GetSize (), 6, "Nothing expire");
}

void
IdCacheTest::CheckTimeout2 ()
{
  NS_TEST_EXPECT_MSG_EQ (cache.GetSize (), 3, "3 records left");
}

void
IdCacheTest::CheckTimeout3 ()
{
  NS_TEST_EXPECT_MSG_EQ (cache.GetSize (), 0, "All records expire");
}

/**
 * \ingroup aodv-test
 * \ingroup tests
 *
 * \brief Id Cache Test Suite
 */
class IdCacheTestSuite : public TestSuite
{
public:
  IdCacheTestSuite () : TestSuite ("aodv-routing-id-cache", UNIT)
  {
    AddTestCase (new IdCacheTest, TestCase::QUICK);
  }
} g_idCacheTestSuite; ///< the test suite

}  // namespace aodv
}  // namespace ns3
