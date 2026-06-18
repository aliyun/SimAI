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

#include "lte-test-secondary-cell-selection.h"

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

NS_LOG_COMPONENT_DEFINE ("LteSecondaryCellSelectionTest");

/*
 * Test Suite
 */

LteSecondaryCellSelectionTestSuite::LteSecondaryCellSelectionTestSuite ()
  : TestSuite ("lte-secondary-cell-selection", SYSTEM)
{
  // REAL RRC PROTOCOL, either 2 or 4 UEs connecting to 2 or 4 component carriers

  AddTestCase (new LteSecondaryCellSelectionTestCase ("EPC, real RRC, RngRun=1", false, 1U, 2), TestCase::QUICK);
  AddTestCase (new LteSecondaryCellSelectionTestCase ("EPC, real RRC, RngRun=1", false, 1U, 4), TestCase::QUICK);

  // IDEAL RRC PROTOCOL, either 2 or 4 UEs connecting to 2 or 4 component carriers

  AddTestCase (new LteSecondaryCellSelectionTestCase ("EPC, ideal RRC, RngRun=1", true, 1U, 2), TestCase::QUICK);
  AddTestCase (new LteSecondaryCellSelectionTestCase ("EPC, ideal RRC, RngRun=1", true, 1U, 4), TestCase::QUICK);

} // end of LteSecondaryCellSelectionTestSuite::LteSecondaryCellSelectionTestSuite ()

static LteSecondaryCellSelectionTestSuite g_lteSecondaryCellSelectionTestSuite;

/*
 * Test Case
 */

LteSecondaryCellSelectionTestCase::LteSecondaryCellSelectionTestCase (
  std::string name, bool isIdealRrc, uint64_t rngRun, uint8_t numberOfComponentCarriers)
  : TestCase (name),
    m_isIdealRrc (isIdealRrc),
    m_rngRun (rngRun),
    m_numberOfComponentCarriers (numberOfComponentCarriers)
{
  NS_LOG_FUNCTION (this << GetName ());
}


LteSecondaryCellSelectionTestCase::~LteSecondaryCellSelectionTestCase ()
{
  NS_LOG_FUNCTION (this << GetName ());
}

void
LteSecondaryCellSelectionTestCase::DoRun ()
{
  NS_LOG_FUNCTION (this << GetName ());

  Config::SetGlobal ("RngRun", UintegerValue (m_rngRun));

  // Create helpers.
  auto lteHelper = CreateObject<LteHelper> ();
  lteHelper->SetAttribute ("PathlossModel", TypeIdValue (ns3::FriisSpectrumPropagationLossModel::GetTypeId ()));
  lteHelper->SetAttribute ("UseIdealRrc", BooleanValue (m_isIdealRrc));
  lteHelper->SetAttribute ("NumberOfComponentCarriers", UintegerValue (m_numberOfComponentCarriers));

  auto epcHelper = CreateObject<PointToPointEpcHelper> ();
  lteHelper->SetEpcHelper (epcHelper);

  // Create nodes.
  auto enbNode = CreateObject<Node> ();
  NodeContainer ueNodes;
  ueNodes.Create (m_numberOfComponentCarriers);

  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (enbNode);
  mobility.Install (ueNodes);

  // Physical layer.
  auto enbDev = DynamicCast<LteEnbNetDevice> (lteHelper->InstallEnbDevice (enbNode).Get (0));
  auto ueDevs = lteHelper->InstallUeDevice (ueNodes);

  // Network layer.
  InternetStackHelper internet;
  internet.Install (ueNodes);
  epcHelper->AssignUeIpv4Address (ueDevs);

  auto ueDev = DynamicCast<LteUeNetDevice> (ueDevs.Get (0));
  std::map< uint8_t, Ptr<ComponentCarrierUe> > ueCcMap = ueDev->GetCcMap ();
  for (auto it: ueCcMap)
    {
      NS_LOG_DEBUG ("Assign DL EARFCN " << it.second->GetDlEarfcn() << " to UE " << ueDevs.Get (it.first)->GetNode ()->GetId ());
      DynamicCast<LteUeNetDevice> (ueDevs.Get (it.first))->SetDlEarfcn (it.second->GetDlEarfcn());
    }

  // Enable Idle mode cell selection.
  lteHelper->Attach (ueDevs);

  // Connect to trace sources in UEs
  Config::Connect ("/NodeList/*/DeviceList/*/LteUeRrc/StateTransition",
                   MakeCallback (&LteSecondaryCellSelectionTestCase::StateTransitionCallback,
                                 this));
  Config::Connect ("/NodeList/*/DeviceList/*/LteUeRrc/ConnectionEstablished",
                   MakeCallback (&LteSecondaryCellSelectionTestCase::ConnectionEstablishedCallback,
                                 this));

  // Run simulation.
  Simulator::Stop (Seconds (2.0));
  Simulator::Run ();

  for (auto &it: enbDev->GetCcMap ())
    {
      ueDev = DynamicCast<LteUeNetDevice> (ueDevs.Get (it.first));
      uint16_t expectedCellId = it.second->GetCellId ();
      uint16_t actualCellId = ueDev->GetRrc ()->GetCellId ();
      NS_LOG_DEBUG ("RNTI " << ueDev->GetRrc ()->GetRnti () << " attached to cell ID: " << actualCellId);
      NS_TEST_ASSERT_MSG_EQ (expectedCellId, actualCellId, "IMSI " << ueDev->GetImsi () << " has attached to an unexpected cell");

      NS_TEST_ASSERT_MSG_EQ (m_lastState.at (ueDev->GetImsi ()),
                             LteUeRrc::CONNECTED_NORMALLY,
                             "UE " << ueDev->GetImsi ()
                                   << " is not at CONNECTED_NORMALLY state");
    }

  // Destroy simulator.
  Simulator::Destroy ();
} // end of void LteSecondaryCellSelectionTestCase::DoRun ()


void
LteSecondaryCellSelectionTestCase::StateTransitionCallback (
  std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti,
  LteUeRrc::State oldState, LteUeRrc::State newState)
{
  NS_LOG_FUNCTION (this << imsi << cellId << rnti << LteUeRrc::ToString (oldState) << LteUeRrc::ToString (newState));
  m_lastState[imsi] = newState;
}


void
LteSecondaryCellSelectionTestCase::ConnectionEstablishedCallback (
  std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
  NS_LOG_FUNCTION (this << imsi << cellId << rnti);
}
