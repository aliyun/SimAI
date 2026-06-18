/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 Phillip Sitbon
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
 * Author: Phillip Sitbon <phillip@sitbon.net>
 */

#include "ns3/simulator.h"
#include "ns3/boolean.h"
#include "ns3/config.h"
#include "ns3/waypoint-mobility-model.h"
#include "ns3/test.h"

using namespace ns3;

/**
 * \ingroup mobility-test
 * \ingroup tests
 *
 * \brief Waypoint Mobility Model Notify Test
 */
class WaypointMobilityModelNotifyTest : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param lazy lazy?
   */
  WaypointMobilityModelNotifyTest (bool lazy)
    : TestCase (lazy ? "Check Waypoint Mobility Model LAZY notification accuracy"
                : "Check Waypoint Mobility Model NON-LAZY notification accuracy"),
      lazyNotify (lazy)
  {
  }
  virtual ~WaypointMobilityModelNotifyTest ()
  {
  }

private:
  std::vector<Ptr<MobilityModel> > mobilityStack; ///< mobilty model
  uint32_t mobilityCount; ///< mobility count
  uint32_t waypointCount; ///< waypoint count
  std::deque<Waypoint> waypoints; ///< waypoints
  bool lazyNotify; ///< lazy notify?
private:
  virtual void DoRun (void);
  virtual void DoTeardown (void);
  /// Force updates
  void ForceUpdates (void);
  /**
   * Course change callback
   * \param model the mobility model
   */
  void CourseChangeCallback (Ptr<const MobilityModel> model);
};

void
WaypointMobilityModelNotifyTest::DoTeardown (void)
{
  mobilityStack.clear();
  waypoints.clear();
}

void
WaypointMobilityModelNotifyTest::DoRun (void)
{
  mobilityCount = 1;
  waypointCount = 100;

  ObjectFactory mobilityFactory;
  mobilityFactory.SetTypeId ("ns3::WaypointMobilityModel");
  mobilityFactory.Set ("LazyNotify", BooleanValue (lazyNotify));

  // Populate the vector of mobility models.
  for (uint32_t i = 0; i < mobilityCount; i++)
    {
      // Create a new mobility model.
      Ptr<MobilityModel> model = mobilityFactory.Create ()->GetObject<MobilityModel> ();

      // Add this mobility model to the stack.
      mobilityStack.push_back (model);
      Simulator::Schedule (Seconds (0.0), &Object::Initialize, model);
    } 

  Waypoint wpt (Seconds (0.0), Vector (0.0, 0.0, 0.0));

  // Create waypoints
  for ( uint32_t iw = 0; iw < waypointCount; ++iw )
    {
      wpt.time += Seconds (1.0);
      waypoints.push_back (wpt);
    }

  // Add the same waypoints to each node
  std::vector<Ptr<MobilityModel> >::iterator i;
  for (i = mobilityStack.begin (); i != mobilityStack.end (); ++i)
    {
      Ptr<WaypointMobilityModel> mob = (*i)->GetObject<WaypointMobilityModel> ();
      mob->TraceConnectWithoutContext ("CourseChange", MakeCallback (&WaypointMobilityModelNotifyTest::CourseChangeCallback, this));

      for ( std::deque<Waypoint>::iterator w = waypoints.begin (); w != waypoints.end (); ++w )
        {
          mob->AddWaypoint (*w);
        }
    }

  // Schedule updates at non-waypoint times to make sure lazy notifications don't happen
  for ( double updateTime = 0.5; updateTime <= ((double)waypointCount + 1.5); updateTime += 1.0 )
    {
      Simulator::Schedule (Seconds (updateTime), &WaypointMobilityModelNotifyTest::ForceUpdates, this);
    }

  Simulator::Stop (Seconds ((double)waypointCount + 2.0));
  Simulator::Run ();
  Simulator::Destroy ();
}
void
WaypointMobilityModelNotifyTest::ForceUpdates (void)
{
  std::vector<Ptr<MobilityModel> >::iterator i;
  for (i = mobilityStack.begin (); i != mobilityStack.end (); ++i)
    {
      Ptr<WaypointMobilityModel> mob = (*i)->GetObject<WaypointMobilityModel> ();
      mob->Update ();
    }
}
void
WaypointMobilityModelNotifyTest::CourseChangeCallback (Ptr<const MobilityModel> model)
{
  const Time now = Simulator::Now ();
  const double sec = now.GetSeconds ();
  Ptr<const WaypointMobilityModel> mob = model->GetObject<WaypointMobilityModel> ();

  NS_TEST_EXPECT_MSG_EQ (now, mob->m_current.time, "Waypoint time not properly updated");

  if ( !lazyNotify )
    {
      // All waypoints are on second boundaries only
      NS_TEST_EXPECT_MSG_EQ (sec - ((double)((int)sec)) + sec, sec,
                             "Course didn't change on one second time boundary with NON-LAZY notifications");
    }
  else
    {
      // Updates should happen at the times they are forced, in between waypoints.
      NS_TEST_EXPECT_MSG_EQ (sec - ((double)((int)sec)), 0.5,
                             "Course didn't change between waypoints with LAZY notifications");
    }
}

/**
 * \ingroup mobility-test
 * \ingroup tests
 *
 * \brief Waypoint Mobility Model Add Waypoint Test
 */
class WaypointMobilityModelAddWaypointTest : public TestCase
{
public:
  WaypointMobilityModelAddWaypointTest ()
    : TestCase ("Check Waypoint Mobility Model waypoint add")
  {
  }
  virtual ~WaypointMobilityModelAddWaypointTest ()
  {
  }

private:
  Ptr<MobilityModel> m_mobilityModel; ///< mobility model
  uint32_t m_waypointCount; ///< waypoint count
  uint32_t m_waypointCounter; ///< waypoint counter
  Waypoint m_nextWaypoint; ///< next waypoint
private:
  virtual void DoRun (void);
  virtual void DoTeardown (void);
  /**
   * Course change calback
   * \param model the mobility model
   */
  void CourseChangeCallback (Ptr<const MobilityModel> model);
};


void
WaypointMobilityModelAddWaypointTest::DoTeardown (void)
{
  m_mobilityModel = 0;
}

void
WaypointMobilityModelAddWaypointTest::DoRun (void)
{
  m_waypointCount = 10;
  m_waypointCounter = 1;

  ObjectFactory mobilityFactory;
  mobilityFactory.SetTypeId ("ns3::WaypointMobilityModel");
  mobilityFactory.Set ("LazyNotify", BooleanValue (false));

  // Create a new mobility model.
  m_mobilityModel = mobilityFactory.Create ()->GetObject<MobilityModel> ();
  m_mobilityModel->TraceConnectWithoutContext ("CourseChange", MakeCallback (&WaypointMobilityModelAddWaypointTest::CourseChangeCallback, this));

  // Add this mobility model to the stack.
  Simulator::Schedule (Seconds (0.0), &Object::Initialize, m_mobilityModel);

  Ptr<WaypointMobilityModel> mob = DynamicCast<WaypointMobilityModel> (m_mobilityModel);
  Waypoint m_nextWaypoint (Seconds (m_waypointCounter), Vector (0.0, 0.0, 0.0));
  mob->AddWaypoint (m_nextWaypoint);

  Simulator::Stop (Seconds ((double)m_waypointCount + 2.0));
  Simulator::Run ();
  Simulator::Destroy ();
}

void
WaypointMobilityModelAddWaypointTest::CourseChangeCallback (Ptr<const MobilityModel> model)
{
  const Time now = Simulator::Now ();
  Ptr<WaypointMobilityModel> mob = DynamicCast<WaypointMobilityModel> (m_mobilityModel);

  std::cout << now << " CourseChangeCallback" << std::endl;

  NS_TEST_EXPECT_MSG_EQ (now, Seconds (m_waypointCounter), "Waypoint time not properly set");

  if (now < Seconds ((double)m_waypointCount) )
    {
      m_waypointCounter ++;
      m_nextWaypoint = Waypoint (Seconds (m_waypointCounter), Vector (0.0, 0.0, 0.0));
      mob->AddWaypoint (m_nextWaypoint);

    }
}

/**
 * \ingroup mobility-test
 * \ingroup tests
 *
 * \brief Waypoint Mobility Model Test Suite
 */
static struct WaypointMobilityModelTestSuite : public TestSuite
{
  WaypointMobilityModelTestSuite () : TestSuite ("waypoint-mobility-model", UNIT)
  {
    AddTestCase (new WaypointMobilityModelNotifyTest (true), TestCase::QUICK);
    AddTestCase (new WaypointMobilityModelNotifyTest (false), TestCase::QUICK);
    AddTestCase (new WaypointMobilityModelAddWaypointTest (), TestCase::QUICK);
  }
} g_waypointMobilityModelTestSuite; ///< the test suite
