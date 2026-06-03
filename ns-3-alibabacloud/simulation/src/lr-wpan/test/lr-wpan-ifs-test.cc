/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 Ritsumeikan University, Shiga, Japan
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
 *  Alberto Gallegos Ramonet <ramonet@fc.ritsumei.ac.jp>
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

NS_LOG_COMPONENT_DEFINE ("lr-wpan-ifs-test");


/**
 * \ingroup lr-wpan-test
 * \ingroup tests
 *
 * \brief LrWpan Dataframe transmission with Interframe Space
 */
class LrWpanDataIfsTestCase : public TestCase
{
public:
  LrWpanDataIfsTestCase ();
  virtual ~LrWpanDataIfsTestCase ();

private:
  /**
   * \brief Function called when DataConfirm is hit.
   * \param testcase pointer to the testcase
   * \param dev originating NetDevice
   * \param params the MCPS params
   */
  static void DataConfirm (LrWpanDataIfsTestCase *testcase,
                           Ptr<LrWpanNetDevice> dev,
                           McpsDataConfirmParams params);

  /**
   * \brief Function called when DataReceived is hit.
   * \param testcase pointer to the testcase
   * \param dev originating NetDevice
   * \param p packet
   */
  static void DataReceivedDev0 (LrWpanDataIfsTestCase *testcase,
                                Ptr<LrWpanNetDevice> dev,
                                Ptr<const Packet> p);

  /**
   * \brief Function called when PhyDataRxStart is hit.
   * \param testcase pointer to the testcase
   * \param dev originating NetDevice
   * \param p packet
   */
  static void PhyDataRxStart (LrWpanDataIfsTestCase *testcase,
                              Ptr<LrWpanNetDevice> dev,
                              Ptr<const Packet> p);

  /**
   * \brief Function called when DataConfirm is hit.
   * \param testcase pointer to the testcase
   * \param dev originating NetDevice
   * \param p packet
   */
  static void DataReceivedDev1 (LrWpanDataIfsTestCase *testcase,
                                Ptr<LrWpanNetDevice> dev,
                                Ptr<const Packet>);

  /**
   * \brief Function called when the IFS ends.
   * \param testcase pointer to the testcase
   * \param dev originating NetDevice
   * \param IfsTime the IFS time
   */
  static void IfsEnd (LrWpanDataIfsTestCase *testcase,
                      Ptr<LrWpanNetDevice> dev,
                      Time IfsTime);



  virtual void DoRun (void);
  Time m_lastTxTime; //!< The time of the last transmitted packet
  Time m_ackRxTime; //!< The time of the received acknowledgment.
  Time m_endIfs; //!< The time where the Interframe Space ended.
  Time m_phyStartRx; //!< The time the phy start receiving a packet.


};


LrWpanDataIfsTestCase::LrWpanDataIfsTestCase ()
  : TestCase ("Lrwpan: IFS tests")
{}

LrWpanDataIfsTestCase::~LrWpanDataIfsTestCase ()
{}

void
LrWpanDataIfsTestCase::DataConfirm (LrWpanDataIfsTestCase *testcase, Ptr<LrWpanNetDevice> dev, McpsDataConfirmParams params)
{
  // get the end time of transmission
  testcase->m_lastTxTime = Simulator::Now ();
}

void
LrWpanDataIfsTestCase::DataReceivedDev0 (LrWpanDataIfsTestCase *testcase, Ptr<LrWpanNetDevice> dev, Ptr<const Packet> p)
{
  // Callback for Data received in the Dev0
  Ptr<Packet> RxPacket = p->Copy ();
  LrWpanMacHeader receivedMacHdr;
  RxPacket->RemoveHeader (receivedMacHdr);

  if (receivedMacHdr.IsAcknowledgment ())
    {
      testcase->m_ackRxTime = Simulator::Now ();
      std::cout << Simulator::Now ().GetSeconds () << " | Dev0 (Node 0) received Acknowledgment.\n";
    }
  else if (receivedMacHdr.GetShortDstAddr ().IsBroadcast ())
    {
      std::cout << Simulator::Now ().GetSeconds () << " | Dev0 (Node 0) received Broadcast. \n";
    }

}

void
LrWpanDataIfsTestCase::PhyDataRxStart (LrWpanDataIfsTestCase *testcase, Ptr<LrWpanNetDevice> dev, Ptr<const Packet>)
{
  //get the start time the phy in dev 0 ( Node 0) start receiving a frame
  testcase->m_phyStartRx = Simulator::Now ();
}

void
LrWpanDataIfsTestCase::DataReceivedDev1 (LrWpanDataIfsTestCase *testcase, Ptr<LrWpanNetDevice> dev, Ptr<const Packet> p)
{
  // Callback for Data received in the Dev1
  Ptr<Packet> RxPacket = p->Copy ();
  LrWpanMacHeader receivedMacHdr;
  RxPacket->RemoveHeader (receivedMacHdr);

  if (receivedMacHdr.GetShortDstAddr ().IsBroadcast ())
    {
      std::cout << Simulator::Now ().GetSeconds () << " | Dev1 (Node 1) received Broadcast. \n";

      // Bcst received, respond with another bcst

      Ptr<Packet> p0 = Create<Packet> (50);      // 50 bytes of dummy data
      McpsDataRequestParams params1;
      params1.m_dstPanId = 0;
      params1.m_srcAddrMode = SHORT_ADDR;
      params1.m_dstAddrMode = SHORT_ADDR;
      params1.m_dstAddr = Mac16Address ("ff:ff");
      params1.m_msduHandle = 0;

      Simulator::ScheduleNow (&LrWpanMac::McpsDataRequest,
                              dev->GetMac (), params1,p0);
    }

}


void
LrWpanDataIfsTestCase::IfsEnd (LrWpanDataIfsTestCase *testcase, Ptr<LrWpanNetDevice> dev, Time IfsTime)
{
  // take the time of the end of the IFS
  testcase->m_endIfs = Simulator::Now ();
}


void
LrWpanDataIfsTestCase::DoRun ()
{
  // Test of Interframe Spaces (IFS)

  // The MAC layer needs a finite amount of time to process the data received from the PHY.
  // To allow this, to successive transmitted frames must be separated for at least one IFS.
  // The IFS size depends on the transmitted frame. This test verifies that the IFS is correctly
  // implemented and its size correspond to the situations described by the standard.
  // For more info see IEEE 802.15.4-2011 Section 5.1.1.3

  LogComponentEnableAll (LOG_PREFIX_TIME);
  LogComponentEnableAll (LOG_PREFIX_FUNC);
  LogComponentEnable ("LrWpanPhy", LOG_LEVEL_DEBUG);
  LogComponentEnable ("LrWpanMac", LOG_LEVEL_DEBUG);
  LogComponentEnable ("LrWpanCsmaCa", LOG_LEVEL_DEBUG);

  // Create 2 nodes, and a NetDevice for each one
  Ptr<Node> n0 = CreateObject <Node> ();
  Ptr<Node> n1 = CreateObject <Node> ();

  Ptr<LrWpanNetDevice> dev0 = CreateObject<LrWpanNetDevice> ();
  Ptr<LrWpanNetDevice> dev1 = CreateObject<LrWpanNetDevice> ();

  dev0->SetAddress (Mac16Address ("00:01"));
  dev1->SetAddress (Mac16Address ("00:02"));

  // Each device must be attached to the same channel
  Ptr<SingleModelSpectrumChannel> channel = CreateObject<SingleModelSpectrumChannel> ();
  Ptr<LogDistancePropagationLossModel> propModel = CreateObject<LogDistancePropagationLossModel> ();
  Ptr<ConstantSpeedPropagationDelayModel> delayModel = CreateObject<ConstantSpeedPropagationDelayModel> ();
  channel->AddPropagationLossModel (propModel);
  channel->SetPropagationDelayModel (delayModel);

  dev0->SetChannel (channel);
  dev1->SetChannel (channel);

  // To complete configuration, a LrWpanNetDevice must be added to a node
  n0->AddDevice (dev0);
  n1->AddDevice (dev1);

  // Connect to trace files in the MAC layer
  dev0->GetMac ()->TraceConnectWithoutContext ("IfsEnd", MakeBoundCallback (&LrWpanDataIfsTestCase::IfsEnd, this, dev0));
  dev0->GetMac ()->TraceConnectWithoutContext ("MacRx", MakeBoundCallback (&LrWpanDataIfsTestCase::DataReceivedDev0, this, dev0));
  dev0->GetPhy ()->TraceConnectWithoutContext ("PhyRxBegin", MakeBoundCallback (&LrWpanDataIfsTestCase::PhyDataRxStart, this, dev0));
  dev1->GetMac ()->TraceConnectWithoutContext ("MacRx", MakeBoundCallback (&LrWpanDataIfsTestCase::DataReceivedDev1, this, dev1));

  Ptr<ConstantPositionMobilityModel> sender0Mobility = CreateObject<ConstantPositionMobilityModel> ();
  sender0Mobility->SetPosition (Vector (0,0,0));
  dev0->GetPhy ()->SetMobility (sender0Mobility);
  Ptr<ConstantPositionMobilityModel> sender1Mobility = CreateObject<ConstantPositionMobilityModel> ();
  // Configure position 10 m distance
  sender1Mobility->SetPosition (Vector (0,10,0));
  dev1->GetPhy ()->SetMobility (sender1Mobility);

  McpsDataConfirmCallback cb0;
  cb0 = MakeBoundCallback (&LrWpanDataIfsTestCase::DataConfirm, this, dev0);
  dev0->GetMac ()->SetMcpsDataConfirmCallback (cb0);

  McpsDataConfirmCallback cb1;
  cb1 = MakeBoundCallback (&LrWpanDataIfsTestCase::DataConfirm, this, dev1);
  dev1->GetMac ()->SetMcpsDataConfirmCallback (cb1);

  Ptr<Packet> p0 = Create<Packet> (2);
  McpsDataRequestParams params;
  params.m_dstPanId = 0;

  params.m_srcAddrMode = SHORT_ADDR;
  params.m_dstAddrMode = SHORT_ADDR;
  params.m_dstAddr = Mac16Address ("00:02");
  params.m_msduHandle = 0;

  Time ifsSize;

  ////////////////////////  SIFS ///////////////////////////

  Simulator::ScheduleWithContext (1, Seconds (0.0),
                                  &LrWpanMac::McpsDataRequest,
                                  dev0->GetMac (), params, p0);


  Simulator::Run ();

  // MPDU = MAC header (11 bytes) + MSDU (2 bytes)+ MAC trailer (2 bytes)  = 15)
  // MPDU (15 bytes) < 18 bytes therefore IFS = SIFS
  // SIFS = 12 symbols (192 Microseconds on a 2.4Ghz O-QPSK PHY)
  ifsSize = m_endIfs - m_lastTxTime;
  NS_TEST_EXPECT_MSG_EQ (ifsSize, Time (MicroSeconds (192)), "Wrong Short InterFrame Space (SIFS) Size after dataframe Tx");
  std::cout << "----------------------------------\n";

  ////////////////////////  LIFS ///////////////////////////

  p0 = Create<Packet> (6);

  Simulator::ScheduleWithContext (1, Seconds (0.0),
                                  &LrWpanMac::McpsDataRequest,
                                  dev0->GetMac (), params, p0);


  Simulator::Run ();

  // MPDU = MAC header (11 bytes) + MSDU (6 bytes)+ MAC trailer (2 bytes)  = 19)
  // MPDU (19 bytes) > 18 bytes therefore IFS = LIFS
  // LIFS = 40 symbols (640 Microseconds on a 2.4Ghz O-QPSK PHY)
  ifsSize = m_endIfs - m_lastTxTime;
  NS_TEST_EXPECT_MSG_EQ (ifsSize, Time (MicroSeconds (640)), "Wrong Long InterFrame Space (LIFS) Size after dataframe Tx");
  std::cout << "----------------------------------\n";

  //////////////////////// SIFS after ACK //////////////////

  params.m_txOptions = TX_OPTION_ACK;
  p0 = Create<Packet> (2);

  Simulator::ScheduleWithContext (1, Seconds (0.0),
                                  &LrWpanMac::McpsDataRequest,
                                  dev0->GetMac (), params, p0);

  Simulator::Run ();

  // MPDU = MAC header (11 bytes) + MSDU (2 bytes)+ MAC trailer (2 bytes)  = 15)
  // MPDU (15 bytes) < 18 bytes therefore IFS = SIFS
  // SIFS = 12 symbols (192 Microseconds on a 2.4Ghz O-QPSK PHY)
  ifsSize = m_endIfs - m_ackRxTime;
  NS_TEST_EXPECT_MSG_EQ (ifsSize, Time (MicroSeconds (192)), "Wrong Short InterFrame Space (SIFS) Size after ACK Rx");
  std::cout << "----------------------------------\n";

  ////////////////////////  LIFS after ACK //////////////////

  params.m_txOptions = TX_OPTION_ACK;
  p0 = Create<Packet> (6);

  Simulator::ScheduleWithContext (1, Seconds (0.0),
                                  &LrWpanMac::McpsDataRequest,
                                  dev0->GetMac (), params, p0);


  Simulator::Run ();

  // MPDU = MAC header (11 bytes) + MSDU (6 bytes)+ MAC trailer (2 bytes)  = 19)
  // MPDU (19 bytes) > 18 bytes therefore IFS = LIFS
  // LIFS = 40 symbols (640 Microseconds on a 2.4Ghz O-QPSK PHY)
  ifsSize = m_endIfs - m_ackRxTime;
  NS_TEST_EXPECT_MSG_EQ (ifsSize, Time (MicroSeconds (640)), "Wrong Long InterFrame Space (LIFS) Size after ACK Rx");
  std::cout << "----------------------------------\n";

  /////////////////////// BCST frame with immediate BCST response //////////////////

  // A packet is broadcasted and the receiving device respond with another broadcast.
  // The devices are configured to not have any backoff delays in their CSMA/CA.
  // In most cases, a device receive a packet after its IFS, however in this test,
  // the receiving device of the reply broadcast will still be in its IFS when the
  // broadcast is received (i.e. a PHY StartRX () occur before the end of IFS).
  // This demonstrates that a device can start receiving a frame even during an IFS.

  // Makes the backoff delay period = 0 in the CSMA/CA
  dev0->GetCsmaCa ()->SetMacMinBE (0);
  dev1->GetCsmaCa ()->SetMacMinBE (0);

  p0 = Create<Packet> (50);  // 50 bytes of dummy data
  params.m_dstPanId = 0;
  params.m_srcAddrMode = SHORT_ADDR;
  params.m_dstAddrMode = SHORT_ADDR;
  params.m_dstAddr = Mac16Address ("ff:ff");
  params.m_msduHandle = 0;

  Simulator::ScheduleWithContext (1, Seconds (0.0),
                                  &LrWpanMac::McpsDataRequest,
                                  dev0->GetMac (), params, p0);

  Simulator::Run ();

  NS_TEST_ASSERT_MSG_GT (m_endIfs, m_phyStartRx, "Error, IFS end time should be greater than PHY start Rx time");

  //////////////////////////////////////////////////////////////////////////////////

  Simulator::Destroy ();

}


/**
 * \ingroup lr-wpan-test
 * \ingroup tests
 *
 * \brief LrWpan IFS TestSuite
 */

class LrWpanIfsTestSuite : public TestSuite
{
public:
  LrWpanIfsTestSuite ();
};

LrWpanIfsTestSuite::LrWpanIfsTestSuite ()
  : TestSuite ("lr-wpan-ifs-test", UNIT)
{
  AddTestCase (new LrWpanDataIfsTestCase, TestCase::QUICK);
}

static LrWpanIfsTestSuite lrWpanIfsTestSuite; //!< Static variable for test initialization




