/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011, 2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */



#include "ns3/log.h"
#include "ns3/test.h"
#include <ns3/mobility-building-info.h>
#include <ns3/constant-position-mobility-model.h>
#include <ns3/building.h>
#include <ns3/buildings-helper.h>
#include <ns3/mobility-helper.h>
#include <ns3/simulator.h>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("BuildingsHelperTest");

/**
 * \ingroup building-test
 * \ingroup tests
 * 
 * \brief Struct representing a position in a building
 */
struct PositionInBuilding
{
  PositionInBuilding ();
  Vector   pos;   //!< coordinates of the mobility model instance
  bool     indoor; //!< true if indoor, false otherwise
  uint32_t bid;   //!< building id
  uint16_t rx;    //!< room x
  uint16_t ry;    //!< room y
  uint16_t fn;    //!< floor number
};

PositionInBuilding::PositionInBuilding ()
  : pos (0,0,0),
    indoor (false),
    bid (0xffffffff),
    rx (0),
    ry (0),
    fn (0)
{
}

/**
 * \ingroup building-test
 * \ingroup tests

 * Data to construct a Building object. We don't want to pass Building
 * objects to the TestCase constructor because otherwise BuildingList
 * would contain all of them (even if only one is meant to be in the
 * test case).
 * 
 */
struct BuildingData
{
  BuildingData ();
  double xmin;   //!< X min coordinate
  double xmax;   //!< X max coordinate
  double ymin;   //!< Y min coordinate
  double ymax;   //!< Y max coordinate
  double zmin;   //!< Z min coordinate
  double zmax;   //!< Z max coordinate
  uint16_t nrx;  //!< Number of rooms (X coord)
  uint16_t nry;  //!< Number of rooms (Y coord)
  uint16_t nf;   //!< Number of floors
};

BuildingData::BuildingData ()
  : xmin (0),
    xmax (0),
    ymin (0),
    ymax (0),
    zmin (0),
    zmax (0),
    nrx (0),
    nry (0),
    nf (0)
{
}

/**
 * \ingroup building-test
 * \ingroup tests
 * 
 * \brief BuildingsHelper test 
 */
class BuildingsHelperOneTestCase : public TestCase
{
public:
  /**
   * Build the testcase name
   * \param pib Position in building
   * \param bd Building data
   * \return the TestCase name
   */
  static std::string BuildNameString (PositionInBuilding pib, BuildingData bd);

  /**
   * Constructor
   * \param pib Position in building
   * \param bd Building data
   */
  BuildingsHelperOneTestCase (PositionInBuilding pib, BuildingData bd);

private:
  virtual void DoRun (void);

  PositionInBuilding m_pib; //!< Position in the building
  BuildingData m_bd; //!< Building data
  
};

std::string BuildingsHelperOneTestCase::BuildNameString (PositionInBuilding pib, BuildingData bd)
{
  std::ostringstream oss;
  oss <<  "pos=" << pib.pos;
  if (pib.indoor)
    {     
      oss << ", bid=" << pib.bid
	  << ", rx=" << pib.rx 
	  << ", ry=" << pib.ry
	  << ", fn=" << pib.fn;    
    }
  else
    {
      oss << ", outdoor";
    }
  return oss.str ();
}


BuildingsHelperOneTestCase::BuildingsHelperOneTestCase (PositionInBuilding pib, BuildingData bd)
  : TestCase (BuildNameString (pib, bd)),
    m_pib (pib),
    m_bd (bd)
{
}

void
BuildingsHelperOneTestCase::DoRun ()
{
  NS_LOG_FUNCTION (this << BuildNameString (m_pib, m_bd));
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");

  NodeContainer nodes;
  nodes.Create (1);
  mobility.Install (nodes);
  
  Ptr<ConstantPositionMobilityModel> bmm = nodes.Get (0)->GetObject<ConstantPositionMobilityModel> ();
  bmm->SetPosition (m_pib.pos);

  NS_LOG_LOGIC ("create building");
  Ptr<Building> b = CreateObject<Building> ();
  b->SetBoundaries (Box (m_bd.xmin, m_bd.xmax, m_bd.ymin, m_bd.ymax, m_bd.zmin, m_bd.zmax));
  b->SetNFloors (m_bd.nf);
  b->SetNRoomsX (m_bd.nrx);
  b->SetNRoomsY (m_bd.nry);
  Ptr<MobilityBuildingInfo> buildingInfo = CreateObject<MobilityBuildingInfo> (b);
  bmm->AggregateObject (buildingInfo); // operation usually done by BuildingsHelper::Install

  
  NS_TEST_ASSERT_MSG_EQ (buildingInfo->IsIndoor (), m_pib.indoor, "indoor/outdoor mismatch");
  if (m_pib.indoor)
    {
      NS_LOG_LOGIC (" got bid=" << buildingInfo->GetBuilding ()->GetId () << ", f=" << (uint32_t) buildingInfo->GetFloorNumber () << ", rx=" << (uint32_t) buildingInfo->GetRoomNumberX () << ", roomY=" << (uint32_t) buildingInfo->GetRoomNumberY ());
      // only one building in this test, so Id will be 0
      NS_TEST_ASSERT_MSG_EQ (buildingInfo->GetBuilding ()->GetId (), 0, "Building ID mismatch");
      NS_TEST_ASSERT_MSG_EQ ((uint32_t) buildingInfo->GetFloorNumber (), m_pib.fn, "floor number mismatch");
      NS_TEST_ASSERT_MSG_EQ ((uint32_t) buildingInfo->GetRoomNumberX (), m_pib.rx, "x room number mismatch");
      NS_TEST_ASSERT_MSG_EQ ((uint32_t) buildingInfo->GetRoomNumberY (), m_pib.ry, "y room number mismatch");
    }

  Simulator::Destroy ();  
}


/**
 * \ingroup building-test
 * \ingroup tests
 * 
 * \brief BuildingsHelper TestSuite
 */
class BuildingsHelperTestSuite : public TestSuite
{
public:
  BuildingsHelperTestSuite ();
};


BuildingsHelperTestSuite::BuildingsHelperTestSuite ()
  : TestSuite ("buildings-helper", UNIT)
{
  NS_LOG_FUNCTION (this);

  BuildingData b1;
  b1.xmin = 1;
  b1.xmax = 3;
  b1.ymin = 1;
  b1.ymax = 2;
  b1.zmin = 0;
  b1.zmax = 4;
  b1.nrx = 1;
  b1.nry = 1;
  b1.nf = 1;  

  Vector vp1 (1.5, 1.5, 0.5);
  PositionInBuilding p1;
  p1.pos = vp1;
  p1.indoor = true;
  p1.bid = 0;
  p1.rx = 1;
  p1.ry = 1;
  p1.fn = 1;
  AddTestCase (new BuildingsHelperOneTestCase (p1, b1), TestCase::QUICK);

  Vector vp2 (1.5, 0.5, 0.5);
  PositionInBuilding p2;
  p2.pos = vp2;
  p2.indoor = false;
  AddTestCase (new BuildingsHelperOneTestCase (p2, b1), TestCase::QUICK);

  Vector vp3 (1.5, 2.5, 0.5);
  PositionInBuilding p3;
  p3.pos = vp3;
  p3.indoor = false;
  AddTestCase (new BuildingsHelperOneTestCase (p3, b1), TestCase::QUICK);

  Vector vp4 (1.5, 1.5, 5);
  PositionInBuilding p4;
  p4.pos = vp4;
  p4.indoor = false;
  AddTestCase (new BuildingsHelperOneTestCase (p4, b1), TestCase::QUICK);

  Vector vp5 (2.5, 1.6, 3.5);
  PositionInBuilding p5;
  p5.pos = vp5;
  p5.indoor = true;
  p5.bid = 0;
  p5.rx = 1;
  p5.ry = 1;
  p5.fn = 1;
  AddTestCase (new BuildingsHelperOneTestCase (p5, b1), TestCase::QUICK);

  Vector vp6 (0.9999, 1.5, 1.5);
  PositionInBuilding p6;
  p6.pos = vp6;
  p6.indoor = false;
  AddTestCase (new BuildingsHelperOneTestCase (p6, b1), TestCase::QUICK);

  Vector vp7 (3.0001, 1.5, 2.5);
  PositionInBuilding p7;
  p7.pos = vp7;
  p7.indoor = false;
  AddTestCase (new BuildingsHelperOneTestCase (p7, b1), TestCase::QUICK);

  Vector vp8 (1.001, 1.001, -0.01);
  PositionInBuilding p8;
  p8.pos = vp8;
  p8.indoor = false;
  AddTestCase (new BuildingsHelperOneTestCase (p8, b1), TestCase::QUICK);

  Vector vp9 (1.5, 1.5, 4.001);
  PositionInBuilding p9;
  p9.pos = vp9;
  p9.indoor = false;
  AddTestCase (new BuildingsHelperOneTestCase (p9, b1), TestCase::QUICK);




  BuildingData b2;
  b2.xmin = -1;
  b2.xmax = 0.5;
  b2.ymin = -2;
  b2.ymax = 0.5;
  b2.zmin = 0;
  b2.zmax = 2;
  b2.nrx = 3;
  b2.nry = 5;
  b2.nf = 4;

  Vector vq1 (-0.7, -1.1, 1.2);
  PositionInBuilding q1;
  q1.pos = vq1;
  q1.indoor = true;
  q1.bid = 1;
  q1.rx = 1;
  q1.ry = 2;
  q1.fn = 3;
  AddTestCase (new BuildingsHelperOneTestCase (q1, b2), TestCase::QUICK);

  Vector vq2 (0.2, 0.3, 0.2);
  PositionInBuilding q2;
  q2.pos = vq2;
  q2.indoor = true;
  q2.bid = 1;
  q2.rx = 3;
  q2.ry = 5;
  q2.fn = 1;
  AddTestCase (new BuildingsHelperOneTestCase (q2, b2), TestCase::QUICK);

  Vector vq3 (0.6, -1.75, 1.5);
  PositionInBuilding q3;
  q3.pos = vq3;
  q3.indoor = false;
  AddTestCase (new BuildingsHelperOneTestCase (q3, b2), TestCase::QUICK);

  Vector vq4 (-1.01, 0.3, 1.99);
  PositionInBuilding q4;
  q4.pos = vq4;
  q4.indoor = false;
  AddTestCase (new BuildingsHelperOneTestCase (q4, b2), TestCase::QUICK);

  Vector vq5 (-0.8, 0.7, 0.01);
  PositionInBuilding q5;
  q5.pos = vq5;
  q5.indoor = false;
  AddTestCase (new BuildingsHelperOneTestCase (q5, b2), TestCase::QUICK);

  Vector vq6 (0.2, 0.3, -0.2);
  PositionInBuilding q6;
  q6.pos = vq6;
  q6.indoor = false;
  AddTestCase (new BuildingsHelperOneTestCase (q6, b2), TestCase::QUICK);

  Vector vq7 (0.2, 0.3, 2.001);
  PositionInBuilding q7;
  q7.pos = vq7;
  q7.indoor = false;
  AddTestCase (new BuildingsHelperOneTestCase (q7, b2), TestCase::QUICK);     
}

/// Static variable for test initialization
static BuildingsHelperTestSuite buildingsHelperAntennaTestSuiteInstance;
