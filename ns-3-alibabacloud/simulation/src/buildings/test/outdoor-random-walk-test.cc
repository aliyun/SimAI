/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 SIGNET Lab, Department of Information Engineering,
 * University of Padova
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
 */

#include "ns3/abort.h"
#include "ns3/test.h"
#include "ns3/config.h"
#include "ns3/building.h"
#include "ns3/building-position-allocator.h"
#include "ns3/random-walk-2d-outdoor-mobility-model.h"
#include "ns3/mobility-helper.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/double.h"
#include "ns3/pointer.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("OutdoorRandomWalkTest");

/**
 * \ingroup building-test
 * \ingroup tests
 * 
 * Test case for the class OutdoorRandomWalkTestCase. It checks if the
 * positions visited by the user are outside buildings
 */
class OutdoorRandomWalkTestCase : public TestCase
{
public:
  OutdoorRandomWalkTestCase ();
  virtual ~OutdoorRandomWalkTestCase ();

private:
  virtual void DoRun (void);

  /**
   * Check that the position is the expected one
   * \param model Mobility model
   */
  void CheckPositionOutdoor (Ptr<RandomWalk2dOutdoorMobilityModel> model);

  std::vector<Ptr<Building> > m_buildings; //!< Buildings
};

OutdoorRandomWalkTestCase::OutdoorRandomWalkTestCase ()
  : TestCase ("Test case for the BuildingsChannelConditionModel"), m_buildings ()
{}

OutdoorRandomWalkTestCase::~OutdoorRandomWalkTestCase ()
{}

void
OutdoorRandomWalkTestCase::CheckPositionOutdoor (Ptr<RandomWalk2dOutdoorMobilityModel> model)
{
  auto position = model->GetPosition ();
  for (auto building : m_buildings)
    {
      NS_TEST_ASSERT_MSG_EQ (building->IsInside (position), false, "Position " << position << " is inside");
    }
}

void
OutdoorRandomWalkTestCase::DoRun (void)
{
  // create a grid of buildings
  double buildingSizeX = 100; // m
  double buildingSizeY = 50; // m
  double streetWidth = 25; // m
  double buildingHeight = 10; // m
  uint32_t numBuildingsX = 20;
  uint32_t numBuildingsY = 20;
  double maxAxisX = (buildingSizeX + streetWidth) * numBuildingsX;
  double maxAxisY = (buildingSizeY + streetWidth) * numBuildingsY;

  for (uint32_t buildingIdX = 0; buildingIdX < numBuildingsX; ++buildingIdX)
    {
      for (uint32_t buildingIdY = 0; buildingIdY < numBuildingsY; ++buildingIdY)
        {
          Ptr < Building > building;
          building = CreateObject<Building> ();

          building->SetBoundaries (Box (buildingIdX * (buildingSizeX + streetWidth),
                                        buildingIdX * (buildingSizeX + streetWidth) + buildingSizeX,
                                        buildingIdY * (buildingSizeY + streetWidth),
                                        buildingIdY * (buildingSizeY + streetWidth) + buildingSizeY,
                                        0.0, buildingHeight));
          building->SetNRoomsX (1);
          building->SetNRoomsY (1);
          building->SetNFloors (1);
          m_buildings.push_back (building);
        }
    }

  // create one node
  NodeContainer nodes;
  nodes.Create (1);

  // set the RandomWalk2dOutdoorMobilityModel mobility model
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::RandomWalk2dOutdoorMobilityModel",
                             "Bounds", RectangleValue (
                               Rectangle (-streetWidth, maxAxisX, -streetWidth, maxAxisY)));
  // create an OutdoorPositionAllocator and set its boundaries to match those of the mobility model
  Ptr<OutdoorPositionAllocator> position = CreateObject<OutdoorPositionAllocator> ();
  Ptr<UniformRandomVariable> xPos = CreateObject<UniformRandomVariable>();
  xPos->SetAttribute ("Min", DoubleValue (-streetWidth));
  xPos->SetAttribute ("Max", DoubleValue (maxAxisX));
  Ptr<UniformRandomVariable> yPos = CreateObject<UniformRandomVariable>();
  yPos->SetAttribute ("Min", DoubleValue (-streetWidth));
  yPos->SetAttribute ("Max", DoubleValue (maxAxisY));
  position->SetAttribute ("X", PointerValue (xPos));
  position->SetAttribute ("Y", PointerValue (yPos));
  mobility.SetPositionAllocator (position);
  // install the mobility model
  mobility.Install (nodes.Get (0));

  auto mobilityModel = nodes.Get (0)->GetObject<RandomWalk2dOutdoorMobilityModel>();

  // get maxChecks positions, check if they are outdoors
  double testStep = 10; // s
  int maxChecks = 1000;
  for (int i = 0; i < maxChecks; ++i)
    {
      Simulator::Schedule (Seconds (i * testStep),
                           &OutdoorRandomWalkTestCase::CheckPositionOutdoor, this, mobilityModel);
    }

  Simulator::Stop (Seconds (maxChecks * testStep + 1));
  Simulator::Run ();
  Simulator::Destroy ();
}

/**
 * \ingroup building-test
 * \ingroup tests
 * 
 * Test suite for the buildings channel condition model
 */
class OutdoorRandomWalkTestSuite : public TestSuite
{
public:
  OutdoorRandomWalkTestSuite ();
};

OutdoorRandomWalkTestSuite::OutdoorRandomWalkTestSuite ()
  : TestSuite ("outdoor-random-walk-model", UNIT)
{
  AddTestCase (new OutdoorRandomWalkTestCase, TestCase::QUICK);
}

/// Static variable for test initialization
static OutdoorRandomWalkTestSuite OutdoorRandomWalkTestSuite;
