/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Alexander Krotov
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
 * Author: Alexander Krotov <krotov@iitp.ru>
 *
 */

#include "lte-test-primary-cell-change.h"

#include <ns3/boolean.h>
#include <ns3/double.h>
#include <ns3/integer.h>
#include <ns3/log.h>
#include <ns3/simulator.h>

#include <ns3/internet-stack-helper.h>
#include <ns3/ipv4-address-helper.h>
#include <ns3/ipv4-interface-container.h>
#include <ns3/friis-spectrum-propagation-loss.h>
#include <ns3/ipv4-static-routing-helper.h>
#include <ns3/lte-enb-net-device.h>
#include <ns3/lte-helper.h>
#include <ns3/lte-ue-net-device.h>
#include <ns3/lte-ue-rrc.h>
#include <ns3/mobility-helper.h>
#include <ns3/net-device-container.h>
#include <ns3/node-container.h>
#include <ns3/point-to-point-epc-helper.h>
#include <ns3/point-to-point-helper.h>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LtePrimaryCellChangeTest");

/*
 * Test Suite
 */

LtePrimaryCellChangeTestSuite::LtePrimaryCellChangeTestSuite ()
  : TestSuite ("lte-primary-cell-change", SYSTEM)
{
  // Test that handover from eNB to eNB with one carrier works within this framework.
  AddTestCase (new LtePrimaryCellChangeTestCase ("ideal RRC, RngRun=1", true, 1, 1, 0, 1), TestCase::QUICK);
  AddTestCase (new LtePrimaryCellChangeTestCase ("real RRC, RngRun=1", false, 1, 1, 0, 1), TestCase::QUICK);

  // Test that handover between first carriers of eNBs works.
  AddTestCase (new LtePrimaryCellChangeTestCase ("ideal RRC, RngRun=1", true, 1, 2, 0, 2), TestCase::QUICK);
  AddTestCase (new LtePrimaryCellChangeTestCase ("real RRC, RngRun=1", false, 1, 2, 0, 2), TestCase::QUICK);

  // Test that handover from second carrier of first eNB to first carrier of second eNB works.
  AddTestCase (new LtePrimaryCellChangeTestCase ("ideal RRC, RngRun=1", true, 1, 2, 1, 2), TestCase::QUICK);
  AddTestCase (new LtePrimaryCellChangeTestCase ("real RRC, RngRun=1", false, 1, 2, 1, 2), TestCase::QUICK);

  // Test that handover from first carrier of first eNB to first carrier of second eNB works.
  AddTestCase (new LtePrimaryCellChangeTestCase ("ideal RRC, RngRun=1", true, 1, 2, 0, 3), TestCase::QUICK);
  AddTestCase (new LtePrimaryCellChangeTestCase ("real RRC, RngRun=1", false, 1, 2, 0, 3), TestCase::QUICK);

  // Test that handover from second carrier of first eNB to second carrier of second eNB works.
  AddTestCase (new LtePrimaryCellChangeTestCase ("ideal RRC, RngRun=1", true, 1, 2, 1, 3), TestCase::QUICK);
  AddTestCase (new LtePrimaryCellChangeTestCase ("real RRC, RngRun=1", false, 1, 2, 1, 3), TestCase::QUICK);

  // Test intra-eNB inter-frequency handover.
  AddTestCase (new LtePrimaryCellChangeTestCase ("ideal RRC, RngRun=1", true, 1, 2, 0, 1), TestCase::QUICK);
  AddTestCase (new LtePrimaryCellChangeTestCase ("real RRC, RngRun=1", false, 1, 2, 0, 1), TestCase::QUICK);
  AddTestCase (new LtePrimaryCellChangeTestCase ("ideal RRC, RngRun=1", true, 1, 2, 1, 0), TestCase::QUICK);
  AddTestCase (new LtePrimaryCellChangeTestCase ("real RRC, RngRun=1", false, 1, 2, 1, 0), TestCase::QUICK);
  AddTestCase (new LtePrimaryCellChangeTestCase ("ideal RRC, RngRun=1", true, 1, 4, 3, 1), TestCase::QUICK);
  AddTestCase (new LtePrimaryCellChangeTestCase ("real RRC, RngRun=1", false, 1, 4, 3, 1), TestCase::QUICK);
} // end of LtePrimaryCellChangeTestSuite::LtePrimaryCellChangeTestSuite ()

static LtePrimaryCellChangeTestSuite g_ltePrimaryCellChangeTestSuite;

/*
 * Test Case
 */

LtePrimaryCellChangeTestCase::LtePrimaryCellChangeTestCase (
  std::string name, bool isIdealRrc, int64_t rngRun,
  uint8_t numberOfComponentCarriers, uint8_t sourceComponentCarrier, uint8_t targetComponentCarrier)
  : TestCase (name),
    m_isIdealRrc {isIdealRrc},
    m_rngRun {rngRun},
    m_numberOfComponentCarriers {numberOfComponentCarriers},
    m_sourceComponentCarrier {sourceComponentCarrier},
    m_targetComponentCarrier {targetComponentCarrier}
{
  NS_LOG_FUNCTION (this << GetName ());
}


LtePrimaryCellChangeTestCase::~LtePrimaryCellChangeTestCase ()
{
  NS_LOG_FUNCTION (this << GetName ());
}

void
LtePrimaryCellChangeTestCase::DoRun ()
{
  NS_LOG_FUNCTION (this << GetName ());

  Config::SetGlobal ("RngRun", UintegerValue (m_rngRun));

  Config::SetDefault ("ns3::LteEnbNetDevice::DlEarfcn", UintegerValue (100));
  Config::SetDefault ("ns3::LteEnbNetDevice::UlEarfcn", UintegerValue (100 + 18000));
  Config::SetDefault ("ns3::LteEnbNetDevice::DlBandwidth", UintegerValue (25));
  Config::SetDefault ("ns3::LteEnbNetDevice::UlBandwidth", UintegerValue (25));
  Config::SetDefault ("ns3::LteUeNetDevice::DlEarfcn", UintegerValue (100));

  // Create helpers.
  auto lteHelper = CreateObject<LteHelper> ();
  lteHelper->SetAttribute ("PathlossModel", TypeIdValue (ns3::FriisSpectrumPropagationLossModel::GetTypeId ()));
  lteHelper->SetAttribute ("UseIdealRrc", BooleanValue (m_isIdealRrc));
  lteHelper->SetAttribute ("NumberOfComponentCarriers", UintegerValue (m_numberOfComponentCarriers));

  auto epcHelper = CreateObject<PointToPointEpcHelper> ();
  lteHelper->SetEpcHelper (epcHelper);

  // Create nodes.
  NodeContainer enbNodes;
  enbNodes.Create (2);
  auto ueNode = CreateObject<Node> ();

  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (enbNodes);
  mobility.Install (ueNode);

  // Physical layer.
  auto enbDevs = lteHelper->InstallEnbDevice (enbNodes);
  auto ueDev = DynamicCast<LteUeNetDevice> (lteHelper->InstallUeDevice (ueNode).Get (0));

  auto sourceEnbDev = DynamicCast<LteEnbNetDevice> (enbDevs.Get (m_sourceComponentCarrier / m_numberOfComponentCarriers));
  auto targetEnbDev = DynamicCast<LteEnbNetDevice> (enbDevs.Get (m_targetComponentCarrier / m_numberOfComponentCarriers));

  // Network layer.
  InternetStackHelper internet;
  internet.Install (ueNode);
  epcHelper->AssignUeIpv4Address (NetDeviceContainer (ueDev));

  std::map<uint8_t, Ptr<ComponentCarrierUe>> ueCcMap = ueDev->GetCcMap ();
  ueDev->SetDlEarfcn (ueCcMap.at (m_sourceComponentCarrier)->GetDlEarfcn ());

  // Attach UE to specified component carrier
  lteHelper->Attach (ueDev, sourceEnbDev, m_sourceComponentCarrier % m_numberOfComponentCarriers);

  // Connect to trace sources in UEs
  Config::Connect ("/NodeList/*/DeviceList/*/LteUeRrc/StateTransition",
                   MakeCallback (&LtePrimaryCellChangeTestCase::StateTransitionCallback,
                                 this));
  Config::Connect ("/NodeList/*/DeviceList/*/LteUeRrc/ConnectionEstablished",
                   MakeCallback (&LtePrimaryCellChangeTestCase::ConnectionEstablishedCallback,
                                 this));

  uint16_t targetCellId = targetEnbDev->GetCcMap ().at (m_targetComponentCarrier % m_numberOfComponentCarriers)->GetCellId ();

  lteHelper->AddX2Interface (enbNodes);
  lteHelper->HandoverRequest (Seconds (1.0), ueDev, sourceEnbDev, targetCellId);

  // Run simulation.
  Simulator::Stop (Seconds (2.0));
  Simulator::Run ();

  uint16_t expectedCellId = targetCellId;
  uint16_t actualCellId = ueDev->GetRrc ()->GetCellId ();
  NS_TEST_ASSERT_MSG_EQ (expectedCellId, actualCellId, "IMSI " << ueDev->GetImsi () << " has attached to an unexpected cell");

  NS_TEST_ASSERT_MSG_EQ (m_lastState.at (ueDev->GetImsi ()),
                         LteUeRrc::CONNECTED_NORMALLY,
                         "UE " << ueDev->GetImsi ()
                               << " is not at CONNECTED_NORMALLY state");

  // Destroy simulator.
  Simulator::Destroy ();
} // end of void LtePrimaryCellChangeTestCase::DoRun ()


void
LtePrimaryCellChangeTestCase::StateTransitionCallback (
  std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti,
  LteUeRrc::State oldState, LteUeRrc::State newState)
{
  NS_LOG_FUNCTION (this << imsi << cellId << rnti << oldState << newState);
  m_lastState[imsi] = newState;
}

void
LtePrimaryCellChangeTestCase::ConnectionEstablishedCallback (
  std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
  NS_LOG_FUNCTION (this << imsi << cellId << rnti);
}
