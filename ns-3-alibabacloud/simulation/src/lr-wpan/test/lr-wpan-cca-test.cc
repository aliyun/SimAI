/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Fraunhofer FKIE
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
 * Author:
 *  Sascha Alexander Jopen <jopen@cs.uni-bonn.de>
 */

#include <ns3/log.h>
#include <ns3/core-module.h>
#include <ns3/lr-wpan-module.h>
#include <ns3/propagation-loss-model.h>
#include <ns3/propagation-delay-model.h>
#include <ns3/simulator.h>
#include <ns3/single-model-spectrum-channel.h>
#include <ns3/constant-position-mobility-model.h>
#include <ns3/packet.h>
#include "ns3/rng-seed-manager.h"

#include <iostream>
#include <iomanip>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("lr-wpan-clear-channel-assessment-test");

/**
 * \ingroup lr-wpan-test
 * \ingroup tests
 *
 * \brief LrWpan CCA Test
 */
class LrWpanCcaTestCase : public TestCase
{
public:
  LrWpanCcaTestCase ();

private:
  /**
   * \brief Function called when PlmeCcaConfirm is hit.
   * \param testcase The TestCase.
   * \param device The LrWpanNetDevice.
   * \param status The device status.
   */
  static void PlmeCcaConfirm (LrWpanCcaTestCase *testcase, Ptr<LrWpanNetDevice> device, LrWpanPhyEnumeration status);
  /**
   * \brief Function called when PhyTxBegin is hit.
   * \param testcase The TestCase.
   * \param device The LrWpanNetDevice.
   * \param packet The packet.
   */
  static void PhyTxBegin (LrWpanCcaTestCase *testcase, Ptr<LrWpanNetDevice> device, Ptr<const Packet> packet);
  /**
   * \brief Function called when PhyTxEnd is hit.
   * \param testcase The TestCase.
   * \param device The LrWpanNetDevice.
   * \param packet The packet.
   */
  static void PhyTxEnd (LrWpanCcaTestCase *testcase, Ptr<LrWpanNetDevice> device, Ptr<const Packet> packet);
  /**
   * \brief Function called when PhyRxBegin is hit.
   * \param testcase The TestCase.
   * \param device The LrWpanNetDevice.
   * \param packet The packet.
   */
  static void PhyRxBegin (LrWpanCcaTestCase *testcase, Ptr<LrWpanNetDevice> device, Ptr<const Packet> packet);
  /**
   * \brief Function called when PhyRxEnd is hit.
   * \param testcase The TestCase.
   * \param device The LrWpanNetDevice.
   * \param packet The packet.
   * \param sinr The received SINR.
   */
  static void PhyRxEnd (LrWpanCcaTestCase *testcase, Ptr<LrWpanNetDevice> device, Ptr<const Packet> packet, double sinr);
  /**
   * \brief Function called when PhyRxDrop is hit.
   * \param testcase The TestCase.
   * \param device The LrWpanNetDevice.
   * \param packet The packet.
   */
  static void PhyRxDrop (LrWpanCcaTestCase *testcase, Ptr<LrWpanNetDevice> device, Ptr<const Packet> packet);

  virtual void DoRun (void);

  LrWpanPhyEnumeration m_status; //!< PHY status.

};

LrWpanCcaTestCase::LrWpanCcaTestCase ()
  : TestCase ("Test the 802.15.4 clear channel assessment")
{
  m_status = IEEE_802_15_4_PHY_UNSPECIFIED;
}

void
LrWpanCcaTestCase::PlmeCcaConfirm (LrWpanCcaTestCase *testcase, Ptr<LrWpanNetDevice> device, LrWpanPhyEnumeration status)
{
  std::cout << std::setiosflags (std::ios::fixed) << std::setprecision (9) << "[" << Simulator::Now ().As (Time::S) << "] " << device->GetMac ()->GetShortAddress () << " PlmeCcaConfirm: " << LrWpanHelper::LrWpanPhyEnumerationPrinter (status) << std::endl;

  testcase->m_status = status;
}

void
LrWpanCcaTestCase::PhyTxBegin (LrWpanCcaTestCase *testcase, Ptr<LrWpanNetDevice> device, Ptr<const Packet> packet)
{
  std::ostringstream os;
  packet->Print (os);
  std::cout << std::setiosflags (std::ios::fixed) << std::setprecision (9) << "[" << Simulator::Now ().As (Time::S) << "] " << device->GetMac ()->GetShortAddress () << " PhyTxBegin: " << os.str () << std::endl;
}

void
LrWpanCcaTestCase::PhyTxEnd (LrWpanCcaTestCase *testcase, Ptr<LrWpanNetDevice> device, Ptr<const Packet> packet)
{
  std::ostringstream os;
  packet->Print (os);
  std::cout << std::setiosflags (std::ios::fixed) << std::setprecision (9) << "[" << Simulator::Now ().As (Time::S) << "] " << device->GetMac ()->GetShortAddress () << " PhyTxEnd: " << os.str () << std::endl;
}

void
LrWpanCcaTestCase::PhyRxBegin (LrWpanCcaTestCase *testcase, Ptr<LrWpanNetDevice> device, Ptr<const Packet> packet)
{
  std::ostringstream os;
  packet->Print (os);
  std::cout << std::setiosflags (std::ios::fixed) << std::setprecision (9) << "[" << Simulator::Now ().As (Time::S) << "] " << device->GetMac ()->GetShortAddress () << " PhyRxBegin: " << os.str () << std::endl;
}

void
LrWpanCcaTestCase::PhyRxEnd (LrWpanCcaTestCase *testcase, Ptr<LrWpanNetDevice> device, Ptr<const Packet> packet, double sinr)
{
  std::ostringstream os;
  packet->Print (os);
  std::cout << std::setiosflags (std::ios::fixed) << std::setprecision (9) << "[" << Simulator::Now ().As (Time::S) << "] " << device->GetMac ()->GetShortAddress () << " PhyRxEnd (" << sinr << "): " << os.str () << std::endl;

  // The first packet was received. Now start a CCA, to try to detect the second packet which is still being transmitted.
  device->GetPhy ()->PlmeCcaRequest ();
}

void LrWpanCcaTestCase::PhyRxDrop (LrWpanCcaTestCase *testcase, Ptr<LrWpanNetDevice> device, Ptr<const Packet> packet)
{
  std::ostringstream os;
  packet->Print (os);
  std::cout << std::setiosflags (std::ios::fixed) << std::setprecision (9) << "[" << Simulator::Now ().As (Time::S) << "] " << device->GetMac ()->GetShortAddress () << " PhyRxDrop: " << os.str () << std::endl;
}

void
LrWpanCcaTestCase::DoRun (void)
{
  // Tx Power: 0 dBm
  // Receiver Sensitivity: -106.58 dBm
  // CCA channel busy condition: Rx power > -96.58 dBm
  // Log distance reference loss at 1 m distance for channel 11 (2405 MHz): 40.0641 dB
  // Log distance free space path loss exponent: 2

  // Test setup:
  // Start transmission of a short packet from node 0 to node 1 and at the same
  // time a transmission of a large packet from node 2 to node 1.
  // Both transmissions should start without backoff (per configuration) because
  // the CCA on both nodes should detect a free medium.
  // The shorter packet will be received first. After reception of the short
  // packet, which might be destroyed due to interference of the large
  // packet, node 1 will start a CCA. Depending on the distance between node 1
  // and node 2, node 1 should detect a busy medium, because the transmission of
  // the large packet is still in progress. For the above mentioned scenario
  // parameters, the distance for the CCA detecting a busy medium is about
  // 669.5685 m.

  // Enable calculation of FCS in the trailers. Only necessary when interacting with real devices or wireshark.
  // GlobalValue::Bind ("ChecksumEnabled", BooleanValue (true));

  // Set the random seed and run number for this test
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (6);

  // Create 3 nodes, and a NetDevice for each one
  Ptr<Node> n0 = CreateObject <Node> ();
  Ptr<Node> n1 = CreateObject <Node> ();
  Ptr<Node> n2 = CreateObject <Node> ();

  Ptr<LrWpanNetDevice> dev0 = CreateObject<LrWpanNetDevice> ();
  Ptr<LrWpanNetDevice> dev1 = CreateObject<LrWpanNetDevice> ();
  Ptr<LrWpanNetDevice> dev2 = CreateObject<LrWpanNetDevice> ();

  // Make random variable stream assignment deterministic
  dev0->AssignStreams (0);
  dev1->AssignStreams (10);
  dev2->AssignStreams (20);

  dev0->SetAddress (Mac16Address ("00:01"));
  dev1->SetAddress (Mac16Address ("00:02"));
  dev2->SetAddress (Mac16Address ("00:03"));

  // Each device must be attached to the same channel
  Ptr<SingleModelSpectrumChannel> channel = CreateObject<SingleModelSpectrumChannel> ();
  Ptr<LogDistancePropagationLossModel> propModel = CreateObject<LogDistancePropagationLossModel> ();
  propModel->SetReference (1.0, 40.0641); // Reference loss at 1m distance for 2405 MHz (channel 11)
  propModel->SetPathLossExponent (2); // Free space path loss exponent
  Ptr<ConstantSpeedPropagationDelayModel> delayModel = CreateObject<ConstantSpeedPropagationDelayModel> ();
  channel->AddPropagationLossModel (propModel);
  channel->SetPropagationDelayModel (delayModel);

  dev0->SetChannel (channel);
  dev1->SetChannel (channel);
  dev2->SetChannel (channel);

  // To complete configuration, a LrWpanNetDevice must be added to a node
  n0->AddDevice (dev0);
  n1->AddDevice (dev1);
  n2->AddDevice (dev2);

  Ptr<ConstantPositionMobilityModel> sender0Mobility = CreateObject<ConstantPositionMobilityModel> ();
  sender0Mobility->SetPosition (Vector (0, 0, 0));
  dev0->GetPhy ()->SetMobility (sender0Mobility);
  Ptr<ConstantPositionMobilityModel> sender1Mobility = CreateObject<ConstantPositionMobilityModel> ();
  sender1Mobility->SetPosition (Vector (0, 669, 0));
  dev1->GetPhy ()->SetMobility (sender1Mobility);
  Ptr<ConstantPositionMobilityModel> sender2Mobility = CreateObject<ConstantPositionMobilityModel> ();
  sender2Mobility->SetPosition (Vector (0, 1338, 0));
  dev2->GetPhy ()->SetMobility (sender2Mobility);

  // Disable the NetDevices queue management.
  dev0->GetMac ()->SetMcpsDataConfirmCallback (MakeNullCallback<void, McpsDataConfirmParams> ());
  dev1->GetMac ()->SetMcpsDataConfirmCallback (MakeNullCallback<void, McpsDataConfirmParams> ());
  dev2->GetMac ()->SetMcpsDataConfirmCallback (MakeNullCallback<void, McpsDataConfirmParams> ());

  // Set the CCA confirm callback.
  dev1->GetPhy ()->SetPlmeCcaConfirmCallback (MakeBoundCallback (&LrWpanCcaTestCase::PlmeCcaConfirm, this, dev1));

  // Start sending without backoff, if the channel is free.
  dev0->GetCsmaCa ()->SetMacMinBE (0);
  dev2->GetCsmaCa ()->SetMacMinBE (0);

  // Connect trace sources.
  dev0->GetPhy ()->TraceConnectWithoutContext ("PhyTxBegin", MakeBoundCallback (&LrWpanCcaTestCase::PhyTxBegin, this, dev0));
  dev0->GetPhy ()->TraceConnectWithoutContext ("PhyTxEnd", MakeBoundCallback (&LrWpanCcaTestCase::PhyTxEnd, this, dev0));
  dev2->GetPhy ()->TraceConnectWithoutContext ("PhyTxBegin", MakeBoundCallback (&LrWpanCcaTestCase::PhyTxBegin, this, dev2));
  dev2->GetPhy ()->TraceConnectWithoutContext ("PhyTxEnd", MakeBoundCallback (&LrWpanCcaTestCase::PhyTxEnd, this, dev2));
  dev1->GetPhy ()->TraceConnectWithoutContext ("PhyRxBegin", MakeBoundCallback (&LrWpanCcaTestCase::PhyRxBegin, this, dev1));
  dev1->GetPhy ()->TraceConnectWithoutContext ("PhyRxEnd", MakeBoundCallback (&LrWpanCcaTestCase::PhyRxEnd, this, dev1));
  dev1->GetPhy ()->TraceConnectWithoutContext ("PhyRxDrop", MakeBoundCallback (&LrWpanCcaTestCase::PhyRxDrop, this, dev1));

  m_status = IEEE_802_15_4_PHY_UNSPECIFIED;

  Ptr<Packet> p0 = Create<Packet> (1);  // 1 byte of dummy data
  McpsDataRequestParams params0;
  params0.m_srcAddrMode = SHORT_ADDR;
  params0.m_dstAddrMode = SHORT_ADDR;
  params0.m_dstPanId = 0;
  params0.m_dstAddr = Mac16Address ("00:02");
  params0.m_msduHandle = 0;
  params0.m_txOptions = TX_OPTION_NONE;
  Simulator::ScheduleNow (&LrWpanMac::McpsDataRequest, dev0->GetMac (), params0, p0);

  Ptr<Packet> p1 = Create<Packet> (100);  // 100 bytes of dummy data
  McpsDataRequestParams params1;
  params1.m_srcAddrMode = SHORT_ADDR;
  params1.m_dstAddrMode = SHORT_ADDR;
  params1.m_dstPanId = 0;
  params1.m_dstAddr = Mac16Address ("00:02");
  params1.m_msduHandle = 0;
  params1.m_txOptions = TX_OPTION_NONE;
  Simulator::ScheduleNow (&LrWpanMac::McpsDataRequest, dev2->GetMac (), params1, p1);

  Simulator::Run ();

  NS_TEST_EXPECT_MSG_EQ (m_status, IEEE_802_15_4_PHY_BUSY, "CCA status BUSY (as expected)");

  m_status = IEEE_802_15_4_PHY_UNSPECIFIED;

  sender2Mobility->SetPosition (Vector (0, 1340, 0));

  Simulator::ScheduleNow (&LrWpanMac::McpsDataRequest, dev0->GetMac (), params0, p0);
  Simulator::ScheduleNow (&LrWpanMac::McpsDataRequest, dev2->GetMac (), params1, p1);

  Simulator::Run ();

  NS_TEST_EXPECT_MSG_EQ (m_status, IEEE_802_15_4_PHY_IDLE, "CCA status IDLE (as expected)");

  Simulator::Destroy ();
}

/**
 * \ingroup lr-wpan-test
 * \ingroup tests
 *
 * \brief LrWpan ACK TestSuite
 */
class LrWpanCcaTestSuite : public TestSuite
{
public:
  LrWpanCcaTestSuite ();
};

LrWpanCcaTestSuite::LrWpanCcaTestSuite ()
  : TestSuite ("lr-wpan-clear-channel-assessment", UNIT)
{
  AddTestCase (new LrWpanCcaTestCase, TestCase::QUICK);
}

static LrWpanCcaTestSuite g_lrWpanCcaTestSuite; //!< Static variable for test initialization
