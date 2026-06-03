/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright 2010 University of Washington
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

/*
 * This test suite is intended to test mobility use cases in general,
 * as typically used by user programs (i.e. with the helper layer
 * involved).
 */

#include "ns3/test.h"
#include "ns3/boolean.h"
#include "ns3/simulator.h"
#include "ns3/scheduler.h"
#include "ns3/vector.h"
#include "ns3/mobility-model.h"
#include "ns3/waypoint-mobility-model.h"
#include "ns3/mobility-helper.h"

using namespace ns3;

/**
 * \ingroup mobility-test
 * \ingroup tests
 *
 * \brief Test whether course change notifications occur regardless of calls
 * to Update() position (which are triggered by calls to GetPosition())
 */
class WaypointLazyNotifyFalse : public TestCase
{
public:
  WaypointLazyNotifyFalse ();
  virtual ~WaypointLazyNotifyFalse ();

private:
  /**
   * Test X position function
   * \param expectedXPos the expected X position
   */
  void TestXPosition (double expectedXPos);
  /**
   * Course change callback
   * \param path the path
   * \param model the mobility model
   */
  void CourseChangeCallback (std::string path, Ptr<const MobilityModel> model);
  virtual void DoRun (void);
  Ptr<Node> m_node; ///< mode
  Ptr<WaypointMobilityModel> m_mob; ///< modility model
  int m_courseChanges; ///< course changes
};

WaypointLazyNotifyFalse::WaypointLazyNotifyFalse ()
  : TestCase ("Test behavior when LazyNotify is false"),
    m_courseChanges (0)
{
}

WaypointLazyNotifyFalse::~WaypointLazyNotifyFalse ()
{
}

void
WaypointLazyNotifyFalse::TestXPosition (double expectedXPos)
{
  Vector pos = m_mob->GetPosition ();
  NS_TEST_EXPECT_MSG_EQ_TOL (pos.x, expectedXPos, 0.001, "Position not equal");
}

void
WaypointLazyNotifyFalse::CourseChangeCallback (std::string path, Ptr<const MobilityModel> model)
{
  // All waypoints (at 10 second intervals) should trigger a course change
  NS_TEST_EXPECT_MSG_EQ_TOL (m_courseChanges * 10.0, Simulator::Now ().GetSeconds (), 0.001, "Course change not notified correctly");
  m_courseChanges++;
}

void
WaypointLazyNotifyFalse::DoRun (void)
{
  m_node = CreateObject<Node> ();
  m_mob = CreateObject<WaypointMobilityModel> ();
  // LazyNotify should by default be false
  m_node->AggregateObject (m_mob);
  Waypoint wpt (Seconds (0.0), Vector (0.0, 0.0, 0.0));
  m_mob->AddWaypoint (wpt);
  Waypoint wpt2 (Seconds (10.0), Vector (10.0, 10.0, 10.0));
  m_mob->AddWaypoint (wpt2);
  Waypoint wpt3 (Seconds (20.0), Vector (20.0, 20.0, 20.0));
  m_mob->AddWaypoint (wpt3);

  Simulator::Schedule (Seconds (5.0), &WaypointLazyNotifyFalse::TestXPosition, this, 5);
  Simulator::Run ();
  Simulator::Destroy ();
}

/**
 * \ingroup mobility-test
 * \ingroup tests
 *
 * \brief Waypoint Lazy Notify True
 */
class WaypointLazyNotifyTrue : public TestCase
{
public:
  WaypointLazyNotifyTrue ();
  virtual ~WaypointLazyNotifyTrue ();

private:
  /**
   * Text X position function
   * \param expectedXPos the expected X position
   */
  void TestXPosition (double expectedXPos);
  /**
   * Course change callback
   * \param path the path
   * \param model the mobility model
   */
  void CourseChangeCallback (std::string path, Ptr<const MobilityModel> model);
  virtual void DoRun (void);
  Ptr<Node> m_node; ///< node
  Ptr<WaypointMobilityModel> m_mob; ///< modility model
};

WaypointLazyNotifyTrue::WaypointLazyNotifyTrue ()
  : TestCase ("Test behavior when LazyNotify is true")
{
}

WaypointLazyNotifyTrue::~WaypointLazyNotifyTrue ()
{
}

void
WaypointLazyNotifyTrue::TestXPosition (double expectedXPos)
{
  Vector pos = m_mob->GetPosition ();
  NS_TEST_EXPECT_MSG_EQ_TOL (pos.x, expectedXPos, 0.001, "Position not equal");
}

void
WaypointLazyNotifyTrue::CourseChangeCallback (std::string path, Ptr<const MobilityModel> model)
{
  // This should trigger at time 15 only, since that is the first time that
  // position is updated due to LazyNotify
  NS_TEST_EXPECT_MSG_EQ_TOL (15, Simulator::Now ().GetSeconds (), 0.001, "Course change not notified correctly");
}

void
WaypointLazyNotifyTrue::DoRun (void)
{
  m_node = CreateObject<Node> ();
  m_mob = CreateObject<WaypointMobilityModel> ();
  m_mob->SetAttributeFailSafe ("LazyNotify", BooleanValue (true));
  m_node->AggregateObject (m_mob);
  Waypoint wpt (Seconds (0.0), Vector (0.0, 0.0, 0.0));
  m_mob->AddWaypoint (wpt);
  Waypoint wpt2 (Seconds (10.0), Vector (10.0, 10.0, 10.0));
  m_mob->AddWaypoint (wpt2);
  Waypoint wpt3 (Seconds (20.0), Vector (20.0, 20.0, 20.0));
  m_mob->AddWaypoint (wpt3);

  Simulator::Schedule (Seconds (15.0), &WaypointLazyNotifyTrue::TestXPosition, this, 15);
  Simulator::Run ();
  Simulator::Destroy ();
}

/**
 * \ingroup mobility-test
 * \ingroup tests
 *
 * \brief Waypoint Initial Position Is Waypoint Test
 */
class WaypointInitialPositionIsWaypoint : public TestCase
{
public:
  WaypointInitialPositionIsWaypoint ();
  virtual ~WaypointInitialPositionIsWaypoint ();

private:
  /**
   * Text X position function
   * \param model the mobility model
   * \param expectedXPos the expected X position
   */
  void TestXPosition (Ptr<const WaypointMobilityModel> model, double expectedXPos);
  /**
   * Test number of way points
   * \param model the mobility model
   * \param num the number of way points
   */
  void TestNumWaypoints (Ptr<const WaypointMobilityModel> model, uint32_t num);
  virtual void DoRun (void);
  Ptr<WaypointMobilityModel> m_mob1; ///< mobility model 1
  Ptr<WaypointMobilityModel> m_mob2; ///< mobility model 2
  Ptr<WaypointMobilityModel> m_mob3; ///< mobility model 3
  Ptr<WaypointMobilityModel> m_mob4; ///< mobility model 4
  Ptr<WaypointMobilityModel> m_mob5; ///< mobility model 5
};

WaypointInitialPositionIsWaypoint::WaypointInitialPositionIsWaypoint ()
  : TestCase ("Test behavior of Waypoint InitialPositionIsWaypoint")
{
}

WaypointInitialPositionIsWaypoint::~WaypointInitialPositionIsWaypoint ()
{
}

void
WaypointInitialPositionIsWaypoint::TestXPosition (Ptr<const WaypointMobilityModel> model, double expectedXPos)
{
  Vector pos = model->GetPosition ();
  NS_TEST_EXPECT_MSG_EQ_TOL (pos.x, expectedXPos, 0.001, "Position not equal");
}

void
WaypointInitialPositionIsWaypoint::TestNumWaypoints (Ptr<const WaypointMobilityModel> model, uint32_t num)
{
  NS_TEST_EXPECT_MSG_EQ (model->WaypointsLeft (), num, "Unexpected number of waypoints left");
}

void
WaypointInitialPositionIsWaypoint::DoRun (void)
{
  // Case 1:  InitialPositionIsWaypoint == false, and we call SetPosition
  // without any waypoints added.  There should be no waypoints after
  // time 0
  m_mob1 = CreateObject<WaypointMobilityModel> ();
  m_mob1->SetAttributeFailSafe ("InitialPositionIsWaypoint", BooleanValue (false));
  m_mob1->SetPosition (Vector (10.0, 10.0, 10.0));
  // At time 1s, there should be no waypoints
  Simulator::Schedule (Seconds (1.0), &WaypointInitialPositionIsWaypoint::TestNumWaypoints, this, m_mob1, 0);
  // At time 15s, the model should still be at x position 10.0
  Simulator::Schedule (Seconds (15.0), &WaypointInitialPositionIsWaypoint::TestXPosition, this, m_mob1, 10.0);

  // Case 2:  InitialPositionIsWaypoint == false, and we call SetPosition
  // after adding a waypoint.
  m_mob2 = CreateObject<WaypointMobilityModel> ();
  m_mob2->SetAttributeFailSafe ("InitialPositionIsWaypoint", BooleanValue (false));
  Waypoint wpt21 (Seconds (5.0), Vector (15.0, 15.0, 15.0));
  m_mob2->AddWaypoint (wpt21);
  Waypoint wpt22 (Seconds (10.0), Vector (20.0, 20.0, 20.0));
  m_mob2->AddWaypoint (wpt22);
  m_mob2->SetPosition (Vector (10.0, 10.0, 10.0));
  // At time 3, no waypoints have been hit, so position should be 10 and
  // numWaypoints should be 2, or 1 excluding the next one
  Simulator::Schedule (Seconds (3.0), &WaypointInitialPositionIsWaypoint::TestXPosition, this, m_mob2, 10.0);
  Simulator::Schedule (Seconds (3.0), &WaypointInitialPositionIsWaypoint::TestNumWaypoints, this, m_mob2, 1);
  // At time 8, check that X position is 18 (i.e. position is interpolating
  // between 15 and 20) and there is one waypoint left, but we exclude
  // the next one so we test for zero waypoints
  Simulator::Schedule (Seconds (8.0), &WaypointInitialPositionIsWaypoint::TestXPosition, this, m_mob2, 18.0);
  Simulator::Schedule (Seconds (8.0), &WaypointInitialPositionIsWaypoint::TestNumWaypoints, this, m_mob2, 0);

  // Case 3:  InitialPositionIsWaypoint == true, and we call SetPosition
  // without any waypoints added.
  m_mob3 = CreateObject<WaypointMobilityModel> ();
  m_mob3->SetAttributeFailSafe ("InitialPositionIsWaypoint", BooleanValue (true));
  m_mob3->SetPosition (Vector (10.0, 10.0, 10.0));
  // At time 1s, there should be zero waypoints not counting the next one
  Simulator::Schedule (Seconds (1.0), &WaypointInitialPositionIsWaypoint::TestNumWaypoints, this, m_mob3, 0);
  // At time 15s, the model should still be at x position 10.0
  Simulator::Schedule (Seconds (15.0), &WaypointInitialPositionIsWaypoint::TestXPosition, this, m_mob3, 10.0);

  // Case 4:  InitialPositionIsWaypoint == true, and we call SetPosition
  // after adding a waypoint.
  m_mob4 = CreateObject<WaypointMobilityModel> ();
  m_mob4->SetAttributeFailSafe ("InitialPositionIsWaypoint", BooleanValue (true));
  Waypoint wpt41 (Seconds (5.0), Vector (15.0, 15.0, 15.0));
  m_mob4->AddWaypoint (wpt41);
  Waypoint wpt42 (Seconds (10.0), Vector (20.0, 20.0, 20.0));
  m_mob4->AddWaypoint (wpt42);
  // Here, SetPosition() is called after waypoints have been added.  In
  // this case, the initial position is set until the time of the first
  // waypoint, at which time it jumps to the waypoint and begins moving
  m_mob4->SetPosition (Vector (10.0, 10.0, 10.0));
  // At time 3, position should be fixed still at 10
  Simulator::Schedule (Seconds (3.0), &WaypointInitialPositionIsWaypoint::TestXPosition, this, m_mob4, 10.0);
  Simulator::Schedule (Seconds (3.0), &WaypointInitialPositionIsWaypoint::TestNumWaypoints, this, m_mob4, 1);
  // At time 6, we should be moving between 15 and 20
  Simulator::Schedule (Seconds (6.0), &WaypointInitialPositionIsWaypoint::TestXPosition, this, m_mob4, 16.0);
  // At time 15, we should be fixed at 20
  Simulator::Schedule (Seconds (15.0), &WaypointInitialPositionIsWaypoint::TestXPosition, this, m_mob4, 20.0);

  // case 5:  If waypoint and SetPosition both called at time 0,
  // SetPosition takes precedence
  m_mob5 = CreateObject<WaypointMobilityModel> ();
  m_mob5->SetAttributeFailSafe ("InitialPositionIsWaypoint", BooleanValue (true));
  // Note:  The below statement would result in a crash, because it would
  // violate the rule that waypoints must increase in start time
  // m_mob5->SetPosition (Vector (10.0, 10.0, 10.0));
  Waypoint wpt51 (Seconds (0.0), Vector (200.0, 200.0, 200.0));
  m_mob5->AddWaypoint (wpt51);
  Waypoint wpt52 (Seconds (5.0), Vector (15.0, 15.0, 15.0));
  m_mob5->AddWaypoint (wpt52);
  Waypoint wpt53 (Seconds (10.0), Vector (20.0, 20.0, 20.0));
  m_mob5->AddWaypoint (wpt53);
  // Here, since waypoints already exist, the below SetPosition will cancel
  // out wpt51 above, and model will stay at initial position until time 5
  m_mob5->SetPosition (Vector (10.0, 10.0, 10.0));
  Simulator::Schedule (Seconds (3.0), &WaypointInitialPositionIsWaypoint::TestXPosition, this, m_mob5, 10.0);

  Simulator::Run ();
  Simulator::Destroy ();
}

/**
 * \ingroup mobility-test
 * \ingroup tests
 *
 * \brief Waypoint Mobility Model Via Helper Test
 */
class WaypointMobilityModelViaHelper : public TestCase
{
public:
  WaypointMobilityModelViaHelper ();
  virtual ~WaypointMobilityModelViaHelper ();

private:
  /**
   * Text X position function
   * \param mob the mobility model
   * \param expectedXPos the expected X position
   */
  void TestXPosition (Ptr<const WaypointMobilityModel> mob, double expectedXPos);
  virtual void DoRun (void);
};

WaypointMobilityModelViaHelper::WaypointMobilityModelViaHelper ()
  : TestCase ("Test behavior using MobilityHelper and PositionAllocator")
{
}

WaypointMobilityModelViaHelper::~WaypointMobilityModelViaHelper ()
{
}

void
WaypointMobilityModelViaHelper::TestXPosition (Ptr<const WaypointMobilityModel> mob, double expectedXPos)
{
  Vector pos = mob->GetPosition ();
  NS_TEST_EXPECT_MSG_EQ_TOL (pos.x, expectedXPos, 0.001, "Position not equal");
}

// WaypointMobilityModel tests using the helper
void
WaypointMobilityModelViaHelper::DoRun (void)
{
  NodeContainer c;
  c.Create (1);
  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);
  // When InitialPositionIsWaypoint is false (default), the position
  // set by the position allocator is ignored.  The first waypoint set will
  // set the initial position (with velocity 0 until first waypoint time)
  mobility.SetMobilityModel ("ns3::WaypointMobilityModel",
                             "InitialPositionIsWaypoint", BooleanValue (false));
  mobility.Install (c);

  // Get back a pointer to this
  Ptr<WaypointMobilityModel> mob = c.Get (0)->GetObject<WaypointMobilityModel> ();
  // Waypoint added at time 0 will override initial position
  Waypoint wpt (Seconds (5.0), Vector (20.0, 20.0, 20.0));
  Waypoint wpt2 (Seconds (10.0), Vector (10.0, 10.0, 10.0));
  mob->AddWaypoint (wpt);
  mob->AddWaypoint (wpt2);
  // At time 3 (before first waypoint, position is 20
  Simulator::Schedule (Seconds (3), &WaypointMobilityModelViaHelper::TestXPosition, this, mob, 20);
  // At time 7.5 (midway between points 1 and 2, position is 15
  Simulator::Schedule (Seconds (7.5), &WaypointMobilityModelViaHelper::TestXPosition, this, mob, 15);

  // When InitialPositionIsWaypoint is true, the position allocator creates
  // the first waypoint, and movement occurs between this origin and the
  // initial waypoint below at 5 seconds
  NodeContainer c2;
  c2.Create (1);
  MobilityHelper mobility2;
  Ptr<ListPositionAllocator> positionAlloc2 = CreateObject<ListPositionAllocator> ();
  positionAlloc2->Add (Vector (0.0, 0.0, 0.0));
  mobility2.SetPositionAllocator (positionAlloc2);
  mobility2.SetMobilityModel ("ns3::WaypointMobilityModel",
                              "InitialPositionIsWaypoint", BooleanValue (true));
  mobility2.Install (c2);
  Ptr<WaypointMobilityModel> mob2 = c2.Get (0)->GetObject<WaypointMobilityModel> ();
  Waypoint wpt3 (Seconds (5.0), Vector (20.0, 20.0, 20.0));
  mob2->AddWaypoint (wpt3);
  // Move to position 12 at 3 seconds
  Simulator::Schedule (Seconds (3), &WaypointMobilityModelViaHelper::TestXPosition, this, mob2, 12);

  Simulator::Run ();
  Simulator::Destroy ();
}

/**
 * \ingroup mobility-test
 * \ingroup tests
 *
 * \brief Mobility Test Suite
 */
class MobilityTestSuite : public TestSuite
{
public:
  MobilityTestSuite ();
};

MobilityTestSuite::MobilityTestSuite ()
  : TestSuite ("mobility", UNIT)
{
  AddTestCase (new WaypointLazyNotifyFalse, TestCase::QUICK);
  AddTestCase (new WaypointLazyNotifyTrue, TestCase::QUICK);
  AddTestCase (new WaypointInitialPositionIsWaypoint, TestCase::QUICK);
  AddTestCase (new WaypointMobilityModelViaHelper, TestCase::QUICK);
}

static MobilityTestSuite mobilityTestSuite; ///< the test suite
