/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Piotr Gawlowicz
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
 * Author: Piotr Gawlowicz <gawlowicz.p@gmail.com>
 *
 */

#include <ns3/simulator.h>
#include <ns3/log.h>
#include <ns3/callback.h>
#include <ns3/config.h>
#include <ns3/string.h>
#include <ns3/double.h>
#include <ns3/enum.h>
#include <ns3/boolean.h>
#include <ns3/pointer.h>
#include <ns3/integer.h>

#include "ns3/mobility-helper.h"
#include "ns3/lte-helper.h"

#include <ns3/ff-mac-scheduler.h>
#include <ns3/lte-enb-net-device.h>
#include <ns3/lte-enb-phy.h>
#include <ns3/lte-enb-rrc.h>
#include <ns3/lte-ue-net-device.h>
#include <ns3/lte-ue-phy.h>
#include <ns3/lte-ue-rrc.h>

#include "lte-ffr-simple.h"
#include <ns3/lte-common.h>

#include "lte-test-uplink-power-control.h"
#include <ns3/lte-rrc-sap.h>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LteUplinkPowerControlTest");

/**
 * TestSuite
 */

LteUplinkPowerControlTestSuite::LteUplinkPowerControlTestSuite ()
  : TestSuite ("lte-uplink-power-control", SYSTEM)
{
//  LogLevel logLevel = (LogLevel)(LOG_PREFIX_FUNC | LOG_PREFIX_TIME | LOG_LEVEL_DEBUG);
//  LogComponentEnable ("LteUplinkPowerControlTest", logLevel);
  NS_LOG_INFO ("Creating LteUplinkPowerControlTestSuite");

  AddTestCase (new LteUplinkOpenLoopPowerControlTestCase ("OpenLoopTest1"), TestCase::QUICK);
  AddTestCase (new LteUplinkClosedLoopPowerControlAbsoluteModeTestCase ("ClosedLoopAbsoluteModeTest1"), TestCase::QUICK);
  AddTestCase (new LteUplinkClosedLoopPowerControlAccumulatedModeTestCase ("ClosedLoopAccumulatedModeTest1"), TestCase::QUICK);
}

static LteUplinkPowerControlTestSuite lteUplinkPowerControlTestSuite;

/**
 * TestCase Data
 */
void
PuschTxPowerNofitication (LteUplinkPowerControlTestCase *testcase,
                          uint16_t cellId, uint16_t rnti, double txPower)
{
  testcase->PuschTxPowerTrace (cellId, rnti, txPower);
}

void
PucchTxPowerNofitication (LteUplinkPowerControlTestCase *testcase,
                          uint16_t cellId, uint16_t rnti, double txPower)
{
  testcase->PucchTxPowerTrace (cellId, rnti, txPower);
}

void
SrsTxPowerNofitication (LteUplinkPowerControlTestCase *testcase,
                        uint16_t cellId, uint16_t rnti, double txPower)
{
  testcase->SrsTxPowerTrace (cellId, rnti, txPower);
}

LteUplinkPowerControlTestCase::LteUplinkPowerControlTestCase (std::string name)
  : TestCase (name)
{
  NS_LOG_INFO ("Creating LteUplinkPowerControlTestCase");
}

LteUplinkPowerControlTestCase::~LteUplinkPowerControlTestCase ()
{
}

void
LteUplinkPowerControlTestCase::TeleportUe (uint32_t x, uint32_t y,
                                           double expectedPuschTxPower, double expectedPucchTxPower, double expectedSrsTxPower)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG ("Teleport UE to : (" << x << ", " << y << ", 0)");

  m_ueMobility->SetPosition (Vector (x, y, 0.0));
  m_teleportTime = Simulator::Now ();

  m_expectedPuschTxPower = expectedPuschTxPower;
  m_expectedPucchTxPower = expectedPucchTxPower;
  m_expectedSrsTxPower = expectedSrsTxPower;
}

void
LteUplinkPowerControlTestCase::SetTpcConfiguration (uint32_t tpc, uint32_t tpcNum,
                                                    double expectedPuschTxPower, double expectedPucchTxPower, double expectedSrsTxPower)
{
  NS_LOG_FUNCTION (this);

  m_teleportTime = Simulator::Now ();

  m_expectedPuschTxPower = expectedPuschTxPower;
  m_expectedPucchTxPower = expectedPucchTxPower;
  m_expectedSrsTxPower = expectedSrsTxPower;

  m_ffrSimple->SetTpc (tpc, tpcNum, m_accumulatedMode);
}

void
LteUplinkPowerControlTestCase::PuschTxPowerTrace (uint16_t cellId, uint16_t rnti, double txPower)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG ("PuschTxPower : CellId: " << cellId << " RNTI: " << rnti << " PuschTxPower: " << txPower);
  //wait because of RSRP filtering
  if ( (Simulator::Now () - m_teleportTime ) < MilliSeconds (50))
    {
      return;
    }
  NS_TEST_ASSERT_MSG_EQ_TOL (txPower, m_expectedPuschTxPower, 0.01, "Wrong Pusch Tx Power");
}

void
LteUplinkPowerControlTestCase::PucchTxPowerTrace (uint16_t cellId, uint16_t rnti, double txPower)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG ("PucchTxPower : CellId: " << cellId << " RNTI: " << rnti << " PuschTxPower: " << txPower);
  //wait because of RSRP filtering
  if ( (Simulator::Now () - m_teleportTime ) < MilliSeconds (50))
    {
      return;
    }

  NS_TEST_ASSERT_MSG_EQ_TOL (txPower, m_expectedPucchTxPower, 0.01, "Wrong Pucch Tx Power");
}

void
LteUplinkPowerControlTestCase::SrsTxPowerTrace (uint16_t cellId, uint16_t rnti, double txPower)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG ("SrsTxPower : CellId: " << cellId << " RNTI: " << rnti << " PuschTxPower: " << txPower);
  //wait because of RSRP filtering
  if ( (Simulator::Now () - m_teleportTime ) < MilliSeconds (50))
    {
      return;
    }
  NS_TEST_ASSERT_MSG_EQ_TOL (txPower, m_expectedSrsTxPower, 0.01, "Wrong Srs Tx Power");
}

void
LteUplinkPowerControlTestCase::DoRun (void)
{
}


LteUplinkOpenLoopPowerControlTestCase::LteUplinkOpenLoopPowerControlTestCase (std::string name)
  : LteUplinkPowerControlTestCase ("Uplink Open Loop Power Control: " + name)
{
  NS_LOG_INFO ("Creating LteUplinkPowerControlTestCase");
}

LteUplinkOpenLoopPowerControlTestCase::~LteUplinkOpenLoopPowerControlTestCase ()
{
}

void
LteUplinkOpenLoopPowerControlTestCase::DoRun (void)
{
  Config::Reset ();
  Config::SetDefault ("ns3::LteHelper::UseIdealRrc", BooleanValue (false));

  double eNbTxPower = 30;
  Config::SetDefault ("ns3::LteEnbPhy::TxPower", DoubleValue (eNbTxPower));
  Config::SetDefault ("ns3::LteUePhy::TxPower", DoubleValue (10.0));
  Config::SetDefault ("ns3::LteUePhy::EnableUplinkPowerControl", BooleanValue (true));

  Config::SetDefault ("ns3::LteUePowerControl::ClosedLoop", BooleanValue (false));
  Config::SetDefault ("ns3::LteUePowerControl::AccumulationEnabled", BooleanValue (false));
  Config::SetDefault ("ns3::LteUePowerControl::PoNominalPusch", IntegerValue (-90));
  Config::SetDefault ("ns3::LteUePowerControl::PsrsOffset", IntegerValue (9));

  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();

  uint16_t bandwidth = 25;
  double d1 = 0;

  // Create Nodes: eNodeB and UE
  NodeContainer enbNodes;
  NodeContainer ueNodes;
  enbNodes.Create (1);
  ueNodes.Create (1);
  NodeContainer allNodes = NodeContainer ( enbNodes, ueNodes);

/*   the topology is the following:
 *
 *   eNB1-------------------------UE
 *                  d1
 */

  // Install Mobility Model
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));                  // eNB1
  positionAlloc->Add (Vector (d1, 0.0, 0.0));           // UE1

  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.SetPositionAllocator (positionAlloc);
  mobility.Install (allNodes);
  m_ueMobility = ueNodes.Get (0)->GetObject<MobilityModel> ();

  // Create Devices and install them in the Nodes (eNB and UE)
  NetDeviceContainer enbDevs;
  NetDeviceContainer ueDevs;
  lteHelper->SetSchedulerType ("ns3::PfFfMacScheduler");

  lteHelper->SetEnbDeviceAttribute ("DlBandwidth", UintegerValue (bandwidth));
  lteHelper->SetEnbDeviceAttribute ("UlBandwidth", UintegerValue (bandwidth));

  enbDevs = lteHelper->InstallEnbDevice (enbNodes);
  ueDevs = lteHelper->InstallUeDevice (ueNodes);

  Ptr<LteUePhy> uePhy = DynamicCast<LteUePhy>( ueDevs.Get (0)->GetObject<LteUeNetDevice> ()->GetPhy () );
  m_ueUpc = uePhy->GetUplinkPowerControl ();

  m_ueUpc->TraceConnectWithoutContext ("ReportPuschTxPower",
                                       MakeBoundCallback (&PuschTxPowerNofitication, this));
  m_ueUpc->TraceConnectWithoutContext ("ReportPucchTxPower",
                                       MakeBoundCallback (&PucchTxPowerNofitication, this));
  m_ueUpc->TraceConnectWithoutContext ("ReportSrsTxPower",
                                       MakeBoundCallback (&SrsTxPowerNofitication, this));

  // Attach a UE to a eNB
  lteHelper->Attach (ueDevs, enbDevs.Get (0));

  // Activate a data radio bearer
  enum EpsBearer::Qci q = EpsBearer::GBR_CONV_VOICE;
  EpsBearer bearer (q);
  lteHelper->ActivateDataRadioBearer (ueDevs, bearer);

  //Changing UE position
  Simulator::Schedule (MilliSeconds (0),
                       &LteUplinkPowerControlTestCase::TeleportUe, this, 0, 0, -40, -40, -40);
  Simulator::Schedule (MilliSeconds (200),
                       &LteUplinkPowerControlTestCase::TeleportUe, this, 200, 0, 8.9745, 8.9745, 11.9745);
  Simulator::Schedule (MilliSeconds (300),
                       &LteUplinkPowerControlTestCase::TeleportUe, this, 400, 0, 14.9951, 14.9951, 17.9951 );
  Simulator::Schedule (MilliSeconds (400),
                       &LteUplinkPowerControlTestCase::TeleportUe, this, 600, 0, 18.5169, 18.5169, 21.5169 );
  Simulator::Schedule (MilliSeconds (500),
                       &LteUplinkPowerControlTestCase::TeleportUe, this, 800, 0, 21.0157, 21.0157, 23 );
  Simulator::Schedule (MilliSeconds (600),
                       &LteUplinkPowerControlTestCase::TeleportUe, this, 1000, 0, 22.9539, 22.9539, 23 );
  Simulator::Schedule (MilliSeconds (700),
                       &LteUplinkPowerControlTestCase::TeleportUe, this, 1200, 0, 23, 10, 23 );
  Simulator::Schedule (MilliSeconds (800),
                       &LteUplinkPowerControlTestCase::TeleportUe, this, 400, 0, 14.9951, 14.9951, 17.9951 );
  Simulator::Schedule (MilliSeconds (900),
                       &LteUplinkPowerControlTestCase::TeleportUe, this, 800, 0, 21.0157, 21.0157, 23 );
  Simulator::Schedule (MilliSeconds (1000),
                       &LteUplinkPowerControlTestCase::TeleportUe, this, 0, 0, -40, -40, -40 );
  Simulator::Schedule (MilliSeconds (1100),
                       &LteUplinkPowerControlTestCase::TeleportUe, this, 100, 0, 2.9539, 2.9539, 5.9539 );
  Simulator::Stop (Seconds (1.200));
  Simulator::Run ();

  Simulator::Destroy ();
}

LteUplinkClosedLoopPowerControlAbsoluteModeTestCase::LteUplinkClosedLoopPowerControlAbsoluteModeTestCase (std::string name)
  : LteUplinkPowerControlTestCase ("Uplink Closed Loop Power Control: " + name)
{
  NS_LOG_INFO ("Creating LteUplinkClosedLoopPowerControlAbsoluteModeTestCase");
}

LteUplinkClosedLoopPowerControlAbsoluteModeTestCase::~LteUplinkClosedLoopPowerControlAbsoluteModeTestCase ()
{
}

void
LteUplinkClosedLoopPowerControlAbsoluteModeTestCase::DoRun (void)
{
  Config::Reset ();
  Config::SetDefault ("ns3::LteHelper::UseIdealRrc", BooleanValue (false));

  double eNbTxPower = 30;
  Config::SetDefault ("ns3::LteEnbPhy::TxPower", DoubleValue (eNbTxPower));
  Config::SetDefault ("ns3::LteUePhy::TxPower", DoubleValue (10.0));
  Config::SetDefault ("ns3::LteUePhy::EnableUplinkPowerControl", BooleanValue (true));

  Config::SetDefault ("ns3::LteUePowerControl::ClosedLoop", BooleanValue (true));
  Config::SetDefault ("ns3::LteUePowerControl::AccumulationEnabled", BooleanValue (false));
  Config::SetDefault ("ns3::LteUePowerControl::PoNominalPusch", IntegerValue (-90));
  Config::SetDefault ("ns3::LteUePowerControl::PsrsOffset", IntegerValue (9));

  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  lteHelper->SetFfrAlgorithmType ("ns3::LteFfrSimple");

  uint16_t bandwidth = 25;
  double d1 = 100;

  // Create Nodes: eNodeB and UE
  NodeContainer enbNodes;
  NodeContainer ueNodes;
  enbNodes.Create (1);
  ueNodes.Create (1);
  NodeContainer allNodes = NodeContainer ( enbNodes, ueNodes);

/*   the topology is the following:
 *
 *   eNB1-------------------------UE
 *                  d1
 */

  // Install Mobility Model
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));                  // eNB1
  positionAlloc->Add (Vector (d1, 0.0, 0.0));           // UE1

  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.SetPositionAllocator (positionAlloc);
  mobility.Install (allNodes);
  m_ueMobility = ueNodes.Get (0)->GetObject<MobilityModel> ();

  // Create Devices and install them in the Nodes (eNB and UE)
  NetDeviceContainer enbDevs;
  NetDeviceContainer ueDevs;
  lteHelper->SetSchedulerType ("ns3::PfFfMacScheduler");

  lteHelper->SetEnbDeviceAttribute ("DlBandwidth", UintegerValue (bandwidth));
  lteHelper->SetEnbDeviceAttribute ("UlBandwidth", UintegerValue (bandwidth));

  enbDevs = lteHelper->InstallEnbDevice (enbNodes);
  ueDevs = lteHelper->InstallUeDevice (ueNodes);

  Ptr<LteUePhy> uePhy = DynamicCast<LteUePhy>( ueDevs.Get (0)->GetObject<LteUeNetDevice> ()->GetPhy () );
  m_ueUpc = uePhy->GetUplinkPowerControl ();

  m_ueUpc->TraceConnectWithoutContext ("ReportPuschTxPower",
                                       MakeBoundCallback (&PuschTxPowerNofitication, this));
  m_ueUpc->TraceConnectWithoutContext ("ReportPucchTxPower",
                                       MakeBoundCallback (&PucchTxPowerNofitication, this));
  m_ueUpc->TraceConnectWithoutContext ("ReportSrsTxPower",
                                       MakeBoundCallback (&SrsTxPowerNofitication, this));

  // Attach a UE to a eNB
  lteHelper->Attach (ueDevs, enbDevs.Get (0));

  // Activate a data radio bearer
  enum EpsBearer::Qci q = EpsBearer::GBR_CONV_VOICE;
  EpsBearer bearer (q);
  lteHelper->ActivateDataRadioBearer (ueDevs, bearer);

  PointerValue tmp;
  enbDevs.Get (0)->GetAttribute ("LteFfrAlgorithm", tmp);
  m_ffrSimple = DynamicCast<LteFfrSimple>(tmp.GetObject ());
  m_accumulatedMode = false;

  //Changing TPC value
  Simulator::Schedule (MilliSeconds (0),
                       &LteUplinkPowerControlTestCase::SetTpcConfiguration, this, 1, 0, 1.9539, 1.9539, 4.9539);
  Simulator::Schedule (MilliSeconds (100),
                       &LteUplinkPowerControlTestCase::SetTpcConfiguration, this, 0, 0, -1.0461, -1.0461, 1.9539);
  Simulator::Schedule (MilliSeconds (200),
                       &LteUplinkPowerControlTestCase::SetTpcConfiguration, this, 2, 0, 3.9539, 3.9539, 6.9539);
  Simulator::Schedule (MilliSeconds (300),
                       &LteUplinkPowerControlTestCase::SetTpcConfiguration, this, 3, 0, 6.9539, 6.9539, 9.9539);
  Simulator::Schedule (MilliSeconds (400),
                       &LteUplinkPowerControlTestCase::SetTpcConfiguration, this, 0, 0, -1.0461, -1.0461, 1.9539);
  Simulator::Schedule (MilliSeconds (500),
                       &LteUplinkPowerControlTestCase::SetTpcConfiguration, this, 1, 0, 1.9539, 1.9539, 4.9539);
  Simulator::Schedule (MilliSeconds (600),
                       &LteUplinkPowerControlTestCase::SetTpcConfiguration, this, 3, 0, 6.9539, 6.9539, 9.9539);
  Simulator::Schedule (MilliSeconds (800),
                       &LteUplinkPowerControlTestCase::SetTpcConfiguration, this, 2, 0, 3.9539, 3.9539, 6.9539);
  Simulator::Stop (Seconds (1.000));
  Simulator::Run ();

  Simulator::Destroy ();
}

LteUplinkClosedLoopPowerControlAccumulatedModeTestCase::LteUplinkClosedLoopPowerControlAccumulatedModeTestCase (std::string name)
  : LteUplinkPowerControlTestCase ("Uplink Closed Loop Power Control: " + name)
{
  NS_LOG_INFO ("Creating LteUplinkClosedLoopPowerControlAccumulatedModeTestCase");
}

LteUplinkClosedLoopPowerControlAccumulatedModeTestCase::~LteUplinkClosedLoopPowerControlAccumulatedModeTestCase ()
{
}

void
LteUplinkClosedLoopPowerControlAccumulatedModeTestCase::DoRun (void)
{
  Config::Reset ();
  Config::SetDefault ("ns3::LteHelper::UseIdealRrc", BooleanValue (false));

  double eNbTxPower = 30;
  Config::SetDefault ("ns3::LteEnbPhy::TxPower", DoubleValue (eNbTxPower));
  Config::SetDefault ("ns3::LteUePhy::TxPower", DoubleValue (10.0));
  Config::SetDefault ("ns3::LteUePhy::EnableUplinkPowerControl", BooleanValue (true));

  Config::SetDefault ("ns3::LteUePowerControl::ClosedLoop", BooleanValue (true));
  Config::SetDefault ("ns3::LteUePowerControl::AccumulationEnabled", BooleanValue (true));
  Config::SetDefault ("ns3::LteUePowerControl::PoNominalPusch", IntegerValue (-90));
  Config::SetDefault ("ns3::LteUePowerControl::PsrsOffset", IntegerValue (9));

  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  lteHelper->SetFfrAlgorithmType ("ns3::LteFfrSimple");

  uint16_t bandwidth = 25;
  double d1 = 10;

  // Create Nodes: eNodeB and UE
  NodeContainer enbNodes;
  NodeContainer ueNodes;
  enbNodes.Create (1);
  ueNodes.Create (1);
  NodeContainer allNodes = NodeContainer ( enbNodes, ueNodes);

/*   the topology is the following:
 *
 *   eNB1-------------------------UE
 *                  d1
 */

  // Install Mobility Model
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));                  // eNB1
  positionAlloc->Add (Vector (d1, 0.0, 0.0));           // UE1

  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.SetPositionAllocator (positionAlloc);
  mobility.Install (allNodes);
  m_ueMobility = ueNodes.Get (0)->GetObject<MobilityModel> ();

  // Create Devices and install them in the Nodes (eNB and UE)
  NetDeviceContainer enbDevs;
  NetDeviceContainer ueDevs;
  lteHelper->SetSchedulerType ("ns3::PfFfMacScheduler");

  lteHelper->SetEnbDeviceAttribute ("DlBandwidth", UintegerValue (bandwidth));
  lteHelper->SetEnbDeviceAttribute ("UlBandwidth", UintegerValue (bandwidth));

  enbDevs = lteHelper->InstallEnbDevice (enbNodes);
  ueDevs = lteHelper->InstallUeDevice (ueNodes);

  Ptr<LteUePhy> uePhy = DynamicCast<LteUePhy>( ueDevs.Get (0)->GetObject<LteUeNetDevice> ()->GetPhy () );
  m_ueUpc = uePhy->GetUplinkPowerControl ();

  m_ueUpc->TraceConnectWithoutContext ("ReportPuschTxPower",
                                       MakeBoundCallback (&PuschTxPowerNofitication, this));
  m_ueUpc->TraceConnectWithoutContext ("ReportPucchTxPower",
                                       MakeBoundCallback (&PucchTxPowerNofitication, this));
  m_ueUpc->TraceConnectWithoutContext ("ReportSrsTxPower",
                                       MakeBoundCallback (&SrsTxPowerNofitication, this));

  // Attach a UE to a eNB
  lteHelper->Attach (ueDevs, enbDevs.Get (0));

  // Activate a data radio bearer
  enum EpsBearer::Qci q = EpsBearer::GBR_CONV_VOICE;
  EpsBearer bearer (q);
  lteHelper->ActivateDataRadioBearer (ueDevs, bearer);

  PointerValue tmp;
  enbDevs.Get (0)->GetAttribute ("LteFfrAlgorithm", tmp);
  m_ffrSimple = DynamicCast<LteFfrSimple>(tmp.GetObject ());
  m_accumulatedMode = true;

  //Changing TPC value
  Simulator::Schedule (MilliSeconds (0),
                       &LteUplinkPowerControlTestCase::SetTpcConfiguration, this, 1, 0, -17.0461, -17.0461, -14.0461);
  Simulator::Schedule (MilliSeconds (100),
                       &LteUplinkPowerControlTestCase::SetTpcConfiguration, this, 0, 20, -37.0461, -37.0461, -34.0461);
  Simulator::Schedule (MilliSeconds (200),
                       &LteUplinkPowerControlTestCase::SetTpcConfiguration, this, 0, 20, -40, 10, -37.0461);
  Simulator::Schedule (MilliSeconds (300),
                       &LteUplinkPowerControlTestCase::SetTpcConfiguration, this, 2, 1, -39.0461, -39.0461, -36.0461);
  Simulator::Schedule (MilliSeconds (400),
                       &LteUplinkPowerControlTestCase::SetTpcConfiguration, this, 3, 10, -9.0461, -9.0461, -6.0461);
  Simulator::Schedule (MilliSeconds (500),
                       &LteUplinkPowerControlTestCase::SetTpcConfiguration, this, 2, 15, 5.9539, 5.9539, 8.9539);
  Simulator::Schedule (MilliSeconds (600),
                       &LteUplinkPowerControlTestCase::SetTpcConfiguration, this, 3, 1, 8.9539, 8.9539, 11.9539);
  Simulator::Schedule (MilliSeconds (700),
                       &LteUplinkPowerControlTestCase::SetTpcConfiguration, this, 2, 10, 18.9539, 18.9539, 21.9539);
  Simulator::Schedule (MilliSeconds (800),
                       &LteUplinkPowerControlTestCase::SetTpcConfiguration, this, 2, 20, 23, 23, 23);
  Simulator::Schedule (MilliSeconds (900),
                       &LteUplinkPowerControlTestCase::SetTpcConfiguration, this, 0, 1, 22.9539, 22.9539, 23);
  Simulator::Schedule (MilliSeconds (1000),
                       &LteUplinkPowerControlTestCase::SetTpcConfiguration, this, 0, 20, 2.9539, 2.9539, 5.9539);
  Simulator::Schedule (MilliSeconds (1100),
                       &LteUplinkPowerControlTestCase::SetTpcConfiguration, this, 2, 5, 7.9539, 7.9539, 10.9539);
  Simulator::Stop (Seconds (1.200));
  Simulator::Run ();

  Simulator::Destroy ();
}
