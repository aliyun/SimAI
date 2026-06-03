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
 * Modifications:
 *  Tommaso Pecorella <tommaso.pecorella@unifi.it>
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
#include <string>
#include <fstream>
#include <streambuf>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("lr-wpan-ack-test");

/**
 * \ingroup lr-wpan
 * \defgroup lr-wpan-test LrWpan module tests
 */

/**
 * \ingroup lr-wpan-test
 * \ingroup tests
 *
 * \brief LrWpan ACK Test
 */
class LrWpanAckTestCase : public TestCase
{
public:

  /**
   * Test modes
   */
  typedef enum
  {
    EXTENDED_ADDRESS_UNICAST,   //!< extended addresses
    SHORT_ADDRESS_UNICAST,      //!< short addresses, unicast
    SHORT_ADDRESS_MULTICAST,    //!< short addresses, multicast
    SHORT_ADDRESS_BROADCAST,    //!< short addresses, broadcast
  } TestMode_e;

  /**
   * Create test case
   *
   * \param prefix Unique file names prefix
   * \param mode   Test mode
   */
  LrWpanAckTestCase (const char * const prefix, TestMode_e mode);

  /**
   * \brief Function called when DataIndication is hit on dev0.
   * \param params The MCPS params.
   * \param p the packet.
   */
  void DataIndicationDev0 (McpsDataIndicationParams params, Ptr<Packet> p);
  /**
   * \brief Function called when DataIndication is hit on dev1.
   * \param params The MCPS params.
   * \param p the packet.
   */
  void DataIndicationDev1 (McpsDataIndicationParams params, Ptr<Packet> p);
  /**
   * \brief Function called when DataConfirm is hit on dev0.
   * \param params The MCPS params.
   */
  void DataConfirmDev0 (McpsDataConfirmParams params);
  /**
   * \brief Function called when DataConfirm is hit on dev1.
   * \param params The MCPS params.
   */
  void DataConfirmDev1 (McpsDataConfirmParams params);

private:
  virtual void DoRun (void);

  std::string m_prefix; //!< Filename prefix
  Time m_requestTime; //!< Request time.
  Time m_requestSentTime; //!< Request successfully sent time.
  Time m_replyTime; //!< Reply time.
  Time m_replySentTime; //!< Reply successfully sent time.
  Time m_replyArrivalTime; //!< Reply arrival time.
  TestMode_e m_mode; //!< Test mode.
  Ptr<LrWpanNetDevice> m_dev0;  //!< 1st LrWpanNetDevice.
  Ptr<LrWpanNetDevice> m_dev1;  //!< 2nd LrWpanNetDevice.
};

LrWpanAckTestCase::LrWpanAckTestCase (const char * const prefix, TestMode_e mode)
  : TestCase ("Test the 802.15.4 ACK handling")
{
  m_prefix = prefix;
  m_requestTime = Seconds (0);
  m_requestSentTime = Seconds (0);
  m_replyTime = Seconds (0);
  m_replySentTime = Seconds (0);
  m_replyArrivalTime = Seconds (0);
  m_mode = mode;
}

void
LrWpanAckTestCase::DataIndicationDev0 (McpsDataIndicationParams params, Ptr<Packet> p)
{
  m_replyArrivalTime = Simulator::Now ();
}

void
LrWpanAckTestCase::DataIndicationDev1 (McpsDataIndicationParams params, Ptr<Packet> p)
{
  Ptr<Packet> pkt = Create<Packet> (10);  // 10 bytes of dummy data
  McpsDataRequestParams replyParams;
  replyParams.m_dstPanId = 0;
  replyParams.m_msduHandle = 0;
  replyParams.m_txOptions = TX_OPTION_NONE;

  if (m_mode == EXTENDED_ADDRESS_UNICAST)
    {
      replyParams.m_srcAddrMode = EXT_ADDR;
      replyParams.m_dstAddrMode = EXT_ADDR;
      replyParams.m_dstExtAddr = Mac64Address ("00:00:00:00:00:00:00:01");
   }
  else
    {
      replyParams.m_srcAddrMode = SHORT_ADDR;
      replyParams.m_dstAddrMode = SHORT_ADDR;
      replyParams.m_dstAddr = Mac16Address ("00:01");
    }
  m_replyTime = Simulator::Now ();
  m_dev1->GetMac ()->McpsDataRequest (replyParams, pkt);
}

void
LrWpanAckTestCase::DataConfirmDev0 (McpsDataConfirmParams params)
{
  m_requestSentTime = Simulator::Now ();
}

void
LrWpanAckTestCase::DataConfirmDev1 (McpsDataConfirmParams params)
{
  m_replySentTime = Simulator::Now ();
}

void
LrWpanAckTestCase::DoRun (void)
{
  // Test setup:
  // Two nodes well in communication range.
  // Node 1 sends a request packet to node 2 with ACK request bit set. Node 2
  // immediately answers with a reply packet on reception of the request.
  // We expect the ACK of the request packet to always arrive at node 1 before
  // the reply packet sent by node 2.
  // This, of course, unelss the packet is sent to a broadcast or multicast address
  // in this case we don't expect any ACK.

  // Enable calculation of FCS in the trailers. Only necessary when interacting with real devices or wireshark.
  // GlobalValue::Bind ("ChecksumEnabled", BooleanValue (true));

  // Set the random seed and run number for this test
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (6);

  Packet::EnablePrinting ();

  // Helper - used to create traces
  LrWpanHelper helper;
  std::string asciiPrefix;

  // Create 2 nodes, and a NetDevice for each one
  Ptr<Node> n0 = CreateObject <Node> ();
  Ptr<Node> n1 = CreateObject <Node> ();

  m_dev0 = CreateObject<LrWpanNetDevice> ();
  m_dev1 = CreateObject<LrWpanNetDevice> ();

  // Make random variable stream assignment deterministic
  m_dev0->AssignStreams (0);
  m_dev1->AssignStreams (10);

  // Add short addresses.
  m_dev0->SetAddress (Mac16Address ("00:01"));
  m_dev1->SetAddress (Mac16Address ("00:02"));
  // Add extended addresses.
  m_dev0->GetMac()->SetExtendedAddress (Mac64Address ("00:00:00:00:00:00:00:01"));
  m_dev1->GetMac()->SetExtendedAddress (Mac64Address ("00:00:00:00:00:00:00:02"));

  // Each device must be attached to the same channel
  Ptr<SingleModelSpectrumChannel> channel = CreateObject<SingleModelSpectrumChannel> ();
  Ptr<LogDistancePropagationLossModel> propModel = CreateObject<LogDistancePropagationLossModel> ();
  Ptr<ConstantSpeedPropagationDelayModel> delayModel = CreateObject<ConstantSpeedPropagationDelayModel> ();
  channel->AddPropagationLossModel (propModel);
  channel->SetPropagationDelayModel (delayModel);

  m_dev0->SetChannel (channel);
  m_dev1->SetChannel (channel);

  // To complete configuration, a LrWpanNetDevice must be added to a node
  n0->AddDevice (m_dev0);
  n1->AddDevice (m_dev1);

  Ptr<ConstantPositionMobilityModel> sender0Mobility = CreateObject<ConstantPositionMobilityModel> ();
  sender0Mobility->SetPosition (Vector (0,0,0));
  m_dev0->GetPhy ()->SetMobility (sender0Mobility);
  Ptr<ConstantPositionMobilityModel> sender1Mobility = CreateObject<ConstantPositionMobilityModel> ();
  // Configure position 10 m distance
  sender1Mobility->SetPosition (Vector (0,10,0));
  m_dev1->GetPhy ()->SetMobility (sender1Mobility);

  McpsDataConfirmCallback cb0;
  cb0 = MakeCallback (&LrWpanAckTestCase::DataConfirmDev0, this);
  m_dev0->GetMac ()->SetMcpsDataConfirmCallback (cb0);

  McpsDataIndicationCallback cb1;
  cb1 = MakeCallback (&LrWpanAckTestCase::DataIndicationDev0, this);
  m_dev0->GetMac ()->SetMcpsDataIndicationCallback (cb1);

  McpsDataConfirmCallback cb2;
  cb2 = MakeCallback (&LrWpanAckTestCase::DataConfirmDev1, this);
  m_dev1->GetMac ()->SetMcpsDataConfirmCallback (cb2);

  McpsDataIndicationCallback cb3;
  cb3 = MakeCallback (&LrWpanAckTestCase::DataIndicationDev1, this);
  m_dev1->GetMac ()->SetMcpsDataIndicationCallback (cb3);

  Ptr<Packet> p0 = Create<Packet> (50);  // 50 bytes of dummy data
  McpsDataRequestParams params;
  uint8_t expectedAckCount = 0;
  switch (m_mode)
  {
    case SHORT_ADDRESS_UNICAST:
      params.m_srcAddrMode = SHORT_ADDR;
      params.m_dstAddrMode = SHORT_ADDR;
      params.m_dstAddr = Mac16Address ("00:02");
      expectedAckCount = 1;
      break;
    case SHORT_ADDRESS_MULTICAST:
      params.m_srcAddrMode = SHORT_ADDR;
      params.m_dstAddrMode = SHORT_ADDR;
      params.m_dstAddr = Mac16Address::GetMulticast (Ipv6Address::GetAllNodesMulticast ());
      expectedAckCount = 0;
      break;
    case SHORT_ADDRESS_BROADCAST:
      params.m_srcAddrMode = SHORT_ADDR;
      params.m_dstAddrMode = SHORT_ADDR;
      params.m_dstAddr = Mac16Address::GetBroadcast ();
      expectedAckCount = 0;
      break;
    case EXTENDED_ADDRESS_UNICAST:
      params.m_srcAddrMode = EXT_ADDR;
      params.m_dstAddrMode = EXT_ADDR;
      params.m_dstExtAddr = Mac64Address ("00:00:00:00:00:00:00:02");
      expectedAckCount = 1;
      break;
  }
  params.m_dstPanId = 0;
  params.m_msduHandle = 0;
  params.m_txOptions = TX_OPTION_ACK;
  m_requestTime = Simulator::Now ();
  Simulator::ScheduleNow (&LrWpanMac::McpsDataRequest, m_dev0->GetMac (), params, p0);

  helper.EnableAscii (CreateTempDirFilename (m_prefix), m_dev0);
  Simulator::Run ();

  std::ifstream traceFile (CreateTempDirFilename (m_prefix)+"-0-0.tr");
  uint8_t ackCounter = 0;
  std::string sub ("Frame Type = 2");
  for( std::string line; getline ( traceFile, line ); )
    {
      if (line.find(sub, 0) != std::string::npos)
        {
          ackCounter++;
        }
    }
  traceFile.close ();

  // Note: the packet being correctly sent includes receiving an ACK in case of for unicact packets.
  NS_TEST_EXPECT_MSG_LT (m_requestTime, m_replyTime, "Sent the request before the reply (as expected)");
  NS_TEST_EXPECT_MSG_GT (m_requestSentTime, Time (0), "The request was sent (as expected)");
  NS_TEST_EXPECT_MSG_LT (m_requestSentTime, m_replyArrivalTime, "The request was sent before the reply arrived (as expected)");
  NS_TEST_EXPECT_MSG_LT (m_replySentTime, m_replyArrivalTime, "The reply was sent before the reply arrived (as expected)");
  NS_TEST_EXPECT_MSG_EQ (ackCounter, expectedAckCount, "The right amount of ACKs have been seen on the channel (as expected)");

  m_dev0 = 0;
  m_dev1 = 0;

  Simulator::Destroy ();
}


/**
 * \ingroup lr-wpan-test
 * \ingroup tests
 *
 * \brief LrWpan ACK TestSuite
 */
class LrWpanAckTestSuite : public TestSuite
{
public:
  LrWpanAckTestSuite ();
};

LrWpanAckTestSuite::LrWpanAckTestSuite ()
  : TestSuite ("lr-wpan-ack", UNIT)
{
  AddTestCase (new LrWpanAckTestCase ("short-unicast", LrWpanAckTestCase::SHORT_ADDRESS_UNICAST), TestCase::QUICK);
  AddTestCase (new LrWpanAckTestCase ("short-multicast", LrWpanAckTestCase::SHORT_ADDRESS_MULTICAST), TestCase::QUICK);
  AddTestCase (new LrWpanAckTestCase ("short-broadcast", LrWpanAckTestCase::SHORT_ADDRESS_BROADCAST), TestCase::QUICK);
  AddTestCase (new LrWpanAckTestCase ("extended-unicast", LrWpanAckTestCase::EXTENDED_ADDRESS_UNICAST), TestCase::QUICK);
}

static LrWpanAckTestSuite g_lrWpanAckTestSuite; //!< Static variable for test initialization
