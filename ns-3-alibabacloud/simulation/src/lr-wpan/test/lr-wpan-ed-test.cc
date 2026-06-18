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

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("lr-wpan-energy-detection-test");

/**
 * \ingroup lr-wpan-test
 * \ingroup tests
 *
 * \brief LrWpan Energy Detection Test
 */
class LrWpanEdTestCase : public TestCase
{
public:
  LrWpanEdTestCase ();

private:
  virtual void DoRun (void);

  /**
   * \brief Function called when PlmeEdConfirm is hit.
   * \param status The PHY status.
   * \param level The ED level.
   */
  void PlmeEdConfirm (LrWpanPhyEnumeration status, uint8_t level);

  LrWpanPhyEnumeration m_status; //!< PHY status.
  uint8_t m_level; //!< ED level.
};

LrWpanEdTestCase::LrWpanEdTestCase ()
  : TestCase ("Test the 802.15.4 energie detection")
{
  m_status = IEEE_802_15_4_PHY_UNSPECIFIED;
  m_level = 0;
}

void
LrWpanEdTestCase::PlmeEdConfirm (LrWpanPhyEnumeration status, uint8_t level)
{
  NS_LOG_UNCOND ("Energy Detection completed with status " << LrWpanHelper::LrWpanPhyEnumerationPrinter (status) << " and energy level " << static_cast<uint32_t> (level));
  m_status = status;
  m_level = level;
}

void
LrWpanEdTestCase::DoRun (void)
{
  // Tx Power: 0 dBm
  // Receiver Sensitivity: -106.58 dBm
  // Do energy detection for a single packet, arriving with 5 dB, 10 dB, 25 dB,
  // 40 dB, relative to RX Power / Sensitivity. This should yield 0, 0, 127,
  // and 255 as the reported energy levels.
  // TODO: Maybe there should be a test for several interfering packets.
  // TODO: There should be tests for signals not originating from 802.15.4
  //       devices.

  // Test setup:
  // Two nodes in communication range. The propagation model is adjusted to
  // give us the above mentioned RX powers.
  // Node 1 sends a packet to node 2. Node 2 starts energy detection, while the
  // packet reception is in progress. The detected energy level is compared to
  // the expected values.

  // Enable calculation of FCS in the trailers. Only necessary when interacting with real devices or wireshark.
  // GlobalValue::Bind ("ChecksumEnabled", BooleanValue (true));

  // Set the random seed and run number for this test
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (6);

  // Create 2 nodes, and a NetDevice for each one
  Ptr<Node> n0 = CreateObject <Node> ();
  Ptr<Node> n1 = CreateObject <Node> ();

  Ptr<LrWpanNetDevice> dev0 = CreateObject<LrWpanNetDevice> ();
  Ptr<LrWpanNetDevice> dev1 = CreateObject<LrWpanNetDevice> ();

  // Make random variable stream assignment deterministic
  dev0->AssignStreams (0);
  dev1->AssignStreams (10);

  dev0->SetAddress (Mac16Address ("00:01"));
  dev1->SetAddress (Mac16Address ("00:02"));

  // Each device must be attached to the same channel
  Ptr<SingleModelSpectrumChannel> channel = CreateObject<SingleModelSpectrumChannel> ();
  Ptr<FixedRssLossModel> propModel = CreateObject<FixedRssLossModel> ();
  Ptr<ConstantSpeedPropagationDelayModel> delayModel = CreateObject<ConstantSpeedPropagationDelayModel> ();
  channel->AddPropagationLossModel (propModel);
  channel->SetPropagationDelayModel (delayModel);

  dev0->SetChannel (channel);
  dev1->SetChannel (channel);

  // To complete configuration, a LrWpanNetDevice must be added to a node
  n0->AddDevice (dev0);
  n1->AddDevice (dev1);

  Ptr<ConstantPositionMobilityModel> sender0Mobility = CreateObject<ConstantPositionMobilityModel> ();
  sender0Mobility->SetPosition (Vector (0, 0, 0));
  dev0->GetPhy ()->SetMobility (sender0Mobility);
  Ptr<ConstantPositionMobilityModel> sender1Mobility = CreateObject<ConstantPositionMobilityModel> ();
  // Configure position 10 m distance
  sender1Mobility->SetPosition (Vector (0, 10, 0));
  dev1->GetPhy ()->SetMobility (sender1Mobility);

  // Set the ED confirm callback.
  dev0->GetPhy ()->SetPlmeEdConfirmCallback (MakeCallback (&LrWpanEdTestCase::PlmeEdConfirm, this));
  dev1->GetPhy ()->SetPlmeEdConfirmCallback (MakeCallback (&LrWpanEdTestCase::PlmeEdConfirm, this));

  // Configure the RX Power to be -107.58 dBm, i.e. 1 dB below receiver sensitivity.
  propModel->SetRss (-107.58);

  m_status = IEEE_802_15_4_PHY_UNSPECIFIED;
  m_level = 0;
  Ptr<Packet> p0 = Create<Packet> (100);  // 100 bytes of dummy data
  McpsDataRequestParams params;
  params.m_srcAddrMode = SHORT_ADDR;
  params.m_dstAddrMode = SHORT_ADDR;
  params.m_dstPanId = 0;
  params.m_dstAddr = Mac16Address ("00:02");
  params.m_msduHandle = 0;
  params.m_txOptions = TX_OPTION_NONE;
  Simulator::ScheduleNow (&LrWpanMac::McpsDataRequest, dev0->GetMac (), params, p0);

  Simulator::Schedule (Seconds (0.0025), &LrWpanPhy::PlmeEdRequest, dev1->GetPhy ());

  Simulator::Run ();

  NS_TEST_EXPECT_MSG_EQ (m_status, IEEE_802_15_4_PHY_SUCCESS, "ED status SUCCESS (as expected)");
  NS_TEST_EXPECT_MSG_EQ (m_level, 0, "ED reported signal level 0 (as expected)");


  // Configure the RX Power to be -106.58 dBm, i.e. exectly to receiver sensitivity.
  propModel->SetRss (-106.58);

  m_status = IEEE_802_15_4_PHY_UNSPECIFIED;
  m_level = 0;
  Ptr<Packet> p1 = Create<Packet> (100);  // 100 bytes of dummy data
  params.m_srcAddrMode = SHORT_ADDR;
  params.m_dstAddrMode = SHORT_ADDR;
  params.m_dstPanId = 0;
  params.m_dstAddr = Mac16Address ("00:02");
  params.m_msduHandle = 0;
  params.m_txOptions = TX_OPTION_NONE;
  Simulator::ScheduleNow (&LrWpanMac::McpsDataRequest, dev0->GetMac (), params, p1);

  Simulator::Schedule (Seconds (0.0025), &LrWpanPhy::PlmeEdRequest, dev1->GetPhy ());

  Simulator::Run ();

  NS_TEST_EXPECT_MSG_EQ (m_status, IEEE_802_15_4_PHY_SUCCESS, "ED status SUCCESS (as expected)");
  NS_TEST_EXPECT_MSG_EQ (m_level, 0, "ED reported signal level 0 (as expected)");


  // Configure the RX Power to be -81.58 dBm, i.e. 25 dB above receiver sensitivity.
  propModel->SetRss (-81.58);

  m_status = IEEE_802_15_4_PHY_UNSPECIFIED;
  m_level = 0;
  Ptr<Packet> p2 = Create<Packet> (100);  // 100 bytes of dummy data
  params.m_srcAddrMode = SHORT_ADDR;
  params.m_dstAddrMode = SHORT_ADDR;
  params.m_dstPanId = 0;
  params.m_dstAddr = Mac16Address ("00:02");
  params.m_msduHandle = 0;
  params.m_txOptions = TX_OPTION_NONE;
  Simulator::ScheduleNow (&LrWpanMac::McpsDataRequest, dev0->GetMac (), params, p2);

  Simulator::Schedule (Seconds (0.0025), &LrWpanPhy::PlmeEdRequest, dev1->GetPhy ());

  Simulator::Run ();

  NS_TEST_EXPECT_MSG_EQ (m_status, IEEE_802_15_4_PHY_SUCCESS, "ED status SUCCESS (as expected)");
  NS_TEST_EXPECT_MSG_EQ (m_level, 127, "ED reported signal level 127 (as expected)");


  // Configure the RX Power to be -66.58 dBm, i.e. 40 dB above receiver sensitivity.
  propModel->SetRss (-66.58);

  m_status = IEEE_802_15_4_PHY_UNSPECIFIED;
  m_level = 0;
  Ptr<Packet> p3 = Create<Packet> (100);  // 100 bytes of dummy data
  params.m_srcAddrMode = SHORT_ADDR;
  params.m_dstAddrMode = SHORT_ADDR;
  params.m_dstPanId = 0;
  params.m_dstAddr = Mac16Address ("00:02");
  params.m_msduHandle = 0;
  params.m_txOptions = TX_OPTION_NONE;
  Simulator::ScheduleNow (&LrWpanMac::McpsDataRequest, dev0->GetMac (), params, p3);

  Simulator::Schedule (Seconds (0.0025), &LrWpanPhy::PlmeEdRequest, dev1->GetPhy ());

  Simulator::Run ();

  NS_TEST_EXPECT_MSG_EQ (m_status, IEEE_802_15_4_PHY_SUCCESS, "ED status SUCCESS (as expected)");
  NS_TEST_EXPECT_MSG_EQ (m_level, 255, "ED reported signal level 255 (as expected)");

  // Test ED at sender.
  m_status = IEEE_802_15_4_PHY_UNSPECIFIED;
  m_level = 0;
  Ptr<Packet> p4 = Create<Packet> (100);  // 100 bytes of dummy data
  params.m_srcAddrMode = SHORT_ADDR;
  params.m_dstAddrMode = SHORT_ADDR;
  params.m_dstPanId = 0;
  params.m_dstAddr = Mac16Address ("00:02");
  params.m_msduHandle = 0;
  params.m_txOptions = TX_OPTION_NONE;
  Simulator::ScheduleNow (&LrWpanMac::McpsDataRequest, dev0->GetMac (), params, p4);

  Simulator::Schedule (Seconds (0.0025), &LrWpanPhy::PlmeEdRequest, dev0->GetPhy ());

  Simulator::Run ();

  NS_TEST_EXPECT_MSG_EQ (m_status, IEEE_802_15_4_PHY_TX_ON, "ED status TX_ON (as expected)");
  NS_TEST_EXPECT_MSG_EQ (m_level, 0, "ED reported signal level 0 (as expected)");

  Simulator::Destroy ();
}

/**
 * \ingroup lr-wpan-test
 * \ingroup tests
 *
 * \brief LrWpan Energy Detection TestSuite
 */
class LrWpanEdTestSuite : public TestSuite
{
public:
  LrWpanEdTestSuite ();
};

LrWpanEdTestSuite::LrWpanEdTestSuite ()
  : TestSuite ("lr-wpan-energy-detection", UNIT)
{
  AddTestCase (new LrWpanEdTestCase, TestCase::QUICK);
}

static LrWpanEdTestSuite g_lrWpanEdTestSuite; //!< Static variable for test initialization
