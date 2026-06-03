/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Marco Miozzo <marco.miozzo@cttc.es>
 *         Nicola Baldo <nbaldo@cttc.es>
 */

#include <ns3/simulator.h>

#include <ns3/log.h>

#include <ns3/buildings-helper.h>
#include <ns3/string.h>
#include <ns3/double.h>
#include <ns3/building.h>
#include <ns3/enum.h>
#include <ns3/constant-position-mobility-model.h>
#include <ns3/mobility-building-info.h>
#include "buildings-pathloss-test.h"



using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("BuildingsPathlossTest");

/*
 * Test 1.1 BuildingsPathlossModel Pathloss compound test
 *
 * This TestSuite tests the BuildingPathlossModel by reproducing
 * several communication scenarios 
 */
BuildingsPathlossTestSuite::BuildingsPathlossTestSuite ()
  : TestSuite ("buildings-pathloss-test", SYSTEM)
{

  LogComponentEnable ("BuildingsPathlossTest", LOG_LEVEL_ALL);

  double freq = 869e6;  // E_UTRA BAND #5 see table 5.5-1 of 36.101

  AddTestCase (new BuildingsPathlossTestCase (freq, 1, 2, UrbanEnvironment, LargeCity, 137.93, "OH Urban Large city"), TestCase::QUICK);

  AddTestCase (new BuildingsPathlossTestCase (freq, 1, 2, UrbanEnvironment, SmallCity, 137.88, "OH Urban small city"), TestCase::QUICK);

  AddTestCase (new BuildingsPathlossTestCase (freq, 1, 2, SubUrbanEnvironment, LargeCity, 128.03, "loss OH SubUrban"), TestCase::QUICK);

  AddTestCase (new BuildingsPathlossTestCase (freq, 1, 2, OpenAreasEnvironment, LargeCity, 110.21, "loss OH OpenAreas"), TestCase::QUICK);

  // Test #2 COST231 Model (1500 < freq < 2000~2170 MHz) (Macro<->UE)

  freq = 2.1140e9; // E_UTRA BAND #1 see table 5.5-1 of 36.101

  AddTestCase (new BuildingsPathlossTestCase (freq, 1, 2, UrbanEnvironment, LargeCity, 148.55, "COST231 Urban Large city"), TestCase::QUICK);

  AddTestCase (new BuildingsPathlossTestCase (freq, 1, 2, UrbanEnvironment, SmallCity, 150.64, "COST231 Urban small city and suburban"), TestCase::QUICK);

  // Test #3 2.6 GHz model (Macro<->UE)

  freq = 2.620e9; // E_UTRA BAND #7 see table 5.5-1 of 36.101

  AddTestCase (new BuildingsPathlossTestCase (freq, 1, 2, UrbanEnvironment, SmallCity, 121.83, "2.6GHz model"), TestCase::QUICK);

  // Test #4 ITU1411 LOS model (Macro<->UE)

  freq = 2.1140e9; // E_UTRA BAND #1 see table 5.5-1 of 36.101
  AddTestCase (new BuildingsPathlossTestCase (freq, 1, 3, UrbanEnvironment, LargeCity, 81.00, "ITU1411 LOS"), TestCase::QUICK);

  // Test #5 ITU1411 NLOS model (Macro<->UE)

  freq = 2.1140e9; // E_UTRA BAND #1 see table 5.5-1 of 36.101

  AddTestCase (new BuildingsPathlossTestCase (freq, 1, 4, UrbanEnvironment, LargeCity, 143.69, "ITU1411 NLOS"), TestCase::QUICK);

  // Test #6 ITUP1238 (HeNB <-> UE)

  freq = 2.1140e9; // E_UTRA BAND #1 see table 5.5-1 of 36.101
  AddTestCase (new BuildingsPathlossTestCase (freq, 5, 6, UrbanEnvironment, LargeCity, 88.3855, "ITUP1238"), TestCase::QUICK);

  // Test #7 Outdoor -> Indoor OkumuraHata (Macro<->UE)

  freq = 2.1140e9; // E_UTRA BAND #1 see table 5.5-1 of 36.101
  // The loss is as in test #2 (large city) plus the building penetration loss
  // which for ConcreteWithWindows is equal to 7 dB -> 148.55 + 7 = 155.55
  AddTestCase (new BuildingsPathlossTestCase (freq, 1, 7, UrbanEnvironment, LargeCity, 155.55, "Okumura Hata Outdoor -> Indoor"), TestCase::QUICK);

  // Test #8 Outdoor -> Indoor ITU1411 (Macro<->UE)
  freq = 2.1140e9; // E_UTRA BAND #1 see table 5.5-1 of 36.101
  // The loss is as in test #4 plus the building penetration loss
  // which for ConcreteWithWindows is equal to 7 dB -> 81.000 + 7 = 88.000
  AddTestCase (new BuildingsPathlossTestCase (freq, 1, 8, UrbanEnvironment, LargeCity, 88.000, "ITU1411 LOS Outdoor -> Indoor"), TestCase::QUICK);

  // Test #9 Indoor -> Outdoor LOS (HeNB <-> UE)

  freq = 2.1140e9; // E_UTRA BAND #1 see table 5.5-1 of 36.101
  // The loss is similar of test #4 plus the building penetration loss
  // which for ConcreteWithWindows is equal to 7 dB and the height gain
  // (2 floors x 2 dB/floor = 4) -> 81.838 + 7 - 4 = 84.838
  AddTestCase (new BuildingsPathlossTestCase (freq, 9, 10, UrbanEnvironment, LargeCity, 84.838, "ITU1411 LOS Indoor -> Outdoor"), TestCase::QUICK);

  // Test #10 Indoor -> Outdoor NLOS (HeNB <-> UE)

  freq = 2.1140e9; // E_UTRA BAND #1 see table 5.5-1 of 36.101
  // The loss is similar as in test #4 plus the building penetration loss
  // which for ConcreteWithWindows is equal to 7 dB and the height gain
  // (2 floors x 2 dB/floor = 4) -> 180.90 + 7 - 4 = 183.90
  AddTestCase (new BuildingsPathlossTestCase (freq, 9, 11, UrbanEnvironment, LargeCity, 183.90, "ITU1411 NLOS Indoor -> Outdoor"), TestCase::QUICK);


}


/// Static variable for test initialization
static BuildingsPathlossTestSuite buildingsPathlossTestSuite;


/*
 * TestCase
 */

BuildingsPathlossTestCase::BuildingsPathlossTestCase (double freq, uint16_t m1, uint16_t m2, EnvironmentType env, CitySize city, double refValue, std::string name)
  : TestCase ("LOSS calculation: " + name),
    m_freq (freq),
    m_mobilityModelIndex1 (m1),
    m_mobilityModelIndex2 (m2),
    m_env (env),
    m_city (city),
    m_lossRef (refValue)
{
}

BuildingsPathlossTestCase::~BuildingsPathlossTestCase ()
{
}

void
BuildingsPathlossTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);

  // the building basically occupies the negative x plane, so any node
  // in this area will fall in the building 
  Ptr<Building> building1 = CreateObject<Building> ();
  building1->SetBoundaries (Box (-3000, -1, -4000, 4000.0, 0.0, 12));
  building1->SetBuildingType (Building::Residential);
  building1->SetExtWallsType (Building::ConcreteWithWindows);
  building1->SetNFloors (3);
  
  Ptr<MobilityModel> mma = CreateMobilityModel (m_mobilityModelIndex1);
  Ptr<MobilityModel> mmb = CreateMobilityModel (m_mobilityModelIndex2);

  Ptr<HybridBuildingsPropagationLossModel> propagationLossModel = CreateObject<HybridBuildingsPropagationLossModel> ();
  propagationLossModel->SetAttribute ("Frequency", DoubleValue (m_freq));
  propagationLossModel->SetAttribute ("Environment", EnumValue (m_env));
  propagationLossModel->SetAttribute ("CitySize", EnumValue (m_city));
  // cancel shadowing effect
  propagationLossModel->SetAttribute ("ShadowSigmaOutdoor", DoubleValue (0.0));
  propagationLossModel->SetAttribute ("ShadowSigmaIndoor", DoubleValue (0.0));
  propagationLossModel->SetAttribute ("ShadowSigmaExtWalls", DoubleValue (0.0));

  double loss = propagationLossModel->GetLoss (mma, mmb);

  NS_LOG_INFO ("Calculated loss: " << loss);
  NS_LOG_INFO ("Theoretical loss: " << m_lossRef);
 
  NS_TEST_ASSERT_MSG_EQ_TOL (loss, m_lossRef, 0.1, "Wrong loss !");
  Simulator::Destroy ();
}

Ptr<MobilityModel>
BuildingsPathlossTestCase::CreateMobilityModel (uint16_t index)
{

  /*
   * The purpose of this method is to defer the creation of the
   * MobilityModel instances to when DoRun() is called. In a previous
   * version, MobilityModel instances where created directly in the
   * constructor of the test suite, which caused subtle bugs due to
   * "static initialization order fiasco". An example of such a subtle
   * bug is that logging via NS_LOG failed for some modules.
   * 
   */

  double hm = 1;
  double hb = 30;
  double henbHeight = 10.0;

  Ptr<MobilityModel> mm;
  
  switch (index)
    {
    case 1:
      mm = CreateObject<ConstantPositionMobilityModel> ();
      mm->SetPosition (Vector (0.0, 0.0, hb));
      break;

    case 2:
      mm = CreateObject<ConstantPositionMobilityModel> ();
      mm->SetPosition (Vector (2000, 0.0, hm));
      break;

    case 3:
      mm = CreateObject<ConstantPositionMobilityModel> ();
      mm->SetPosition (Vector (100, 0.0, hm));
      break;

    case 4:
      mm = CreateObject<ConstantPositionMobilityModel> ();
      mm->SetPosition (Vector (900, 0.0, hm));
      break;

    case 5:
      mm = CreateObject<ConstantPositionMobilityModel> ();
      mm->SetPosition (Vector (-5, 0.0, hm));
      break;

    case 6:
      mm = CreateObject<ConstantPositionMobilityModel> ();
      mm->SetPosition (Vector (-5, 30, henbHeight));
      break;

    case 7:
      mm = CreateObject<ConstantPositionMobilityModel> ();
      mm->SetPosition (Vector (-2000, 0.0, hm));
      break;

    case 8:
      mm = CreateObject<ConstantPositionMobilityModel> ();
      mm->SetPosition (Vector (-100, 0.0, hm));
      break;

    case 9:
      mm = CreateObject<ConstantPositionMobilityModel> ();
      mm->SetPosition (Vector (0, 0.0, hm));
      break;

    case 10:
      mm = CreateObject<ConstantPositionMobilityModel> ();
      mm->SetPosition (Vector (-100, 0.0, henbHeight));
      break;

    case 11:
      mm = CreateObject<ConstantPositionMobilityModel> ();
      mm->SetPosition (Vector (-500, 0.0, henbHeight));
      break;

    default:
      mm = 0;
      break;
    }
  Ptr<MobilityBuildingInfo> buildingInfo = CreateObject<MobilityBuildingInfo> ();
  mm->AggregateObject (buildingInfo); // operation usually done by BuildingsHelper::Install
  buildingInfo->MakeConsistent (mm);
  return mm;
}
