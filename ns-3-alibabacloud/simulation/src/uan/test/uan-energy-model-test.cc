/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 Andrea Sacco
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
 * Author: Andrea Sacco <andrea.sacco85@gmail.com>
 */

#include "ns3/log.h"
#include "ns3/test.h"
#include "ns3/simple-device-energy-model.h"
#include "ns3/uan-net-device.h"
#include "ns3/simulator.h"
#include "ns3/packet.h"
#include "ns3/node.h"
#include "ns3/uan-helper.h"
#include "ns3/basic-energy-source-helper.h"
#include "ns3/acoustic-modem-energy-model-helper.h"
#include "ns3/acoustic-modem-energy-model.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/uan-channel.h"
#include "ns3/uan-noise-model-default.h"
#include "ns3/uan-prop-model-ideal.h"
#include "ns3/uan-header-common.h"
#include "ns3/uan-phy.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("UanEnergyModelTestSuite");

/**
 * \ingroup uan
 * \defgroup uan-test uan module tests
 */


/**
 * \ingroup uan-test
 * \ingroup tests
 *
 * \brief Acoustic Modem Energy Test Case
 */
class AcousticModemEnergyTestCase : public TestCase
{
public:
  AcousticModemEnergyTestCase ();
  ~AcousticModemEnergyTestCase ();

  /**
   * Receive packet function
   * \param dev the device
   * \param pkt the packet
   * \param mode the mode
   * \param sender the address of the sender
   * \returns true if successful
   */
  bool RxPacket (Ptr<NetDevice> dev, Ptr<const Packet> pkt, uint16_t mode, const Address &sender);
  /**
   * Send one packet function
   * \param node the node to send to
   */
  void SendOnePacket (Ptr<Node> node);

  void DoRun (void);

  double m_simTime; ///< simulation time
  uint32_t m_bytesRx; ///< bytes received
  uint32_t m_sentPackets; ///< number of sent packets
  uint32_t m_packetSize; ///< packet size
  Ptr<Node> m_node; ///< node
  Ptr<Node> m_gateway; ///< the gateway
};

AcousticModemEnergyTestCase::AcousticModemEnergyTestCase ()
  : TestCase ("Acoustic Modem energy model test case"),
    m_simTime (25),
    m_bytesRx (0),
    m_sentPackets (0),
    m_packetSize (17)
{
}

AcousticModemEnergyTestCase::~AcousticModemEnergyTestCase ()
{
  m_node = 0;
  m_gateway = 0;
}

void
AcousticModemEnergyTestCase::SendOnePacket (Ptr<Node> node)
{
  // create an empty 17 bytes packet
  Ptr<Packet> pkt = Create<Packet> (m_packetSize);
  // send the packet in broadcast
  Ptr<UanNetDevice> dev = node->GetDevice (0)->GetObject<UanNetDevice> ();
  dev->Send (pkt, dev->GetBroadcast (), 0);
  // increase the sent packets number
  ++m_sentPackets;

  Simulator::Schedule (Seconds (10),
                       &AcousticModemEnergyTestCase::SendOnePacket,
                       this,
                       node);
}

bool
AcousticModemEnergyTestCase::RxPacket (Ptr<NetDevice> dev, Ptr<const Packet> pkt, 
                                       [[maybe_unused]] uint16_t mode, 
                                       [[maybe_unused]] const Address &sender)
{
  // increase the total bytes received
  m_bytesRx += pkt->GetSize ();

  return true;
}

void
AcousticModemEnergyTestCase::DoRun ()
{
  // create a generic node
  m_node = CreateObject<Node> ();

  // create a default underwater channel
  Ptr<UanChannel> channel = CreateObject<UanChannel> ();
  Ptr<UanNoiseModelDefault> noise = CreateObject<UanNoiseModelDefault> ();
  channel->SetPropagationModel (CreateObject<UanPropModelIdeal> ());
  channel->SetNoiseModel (noise);

  // install the underwater communication stack
  UanHelper uan;
  Ptr<UanNetDevice> devNode = uan.Install (m_node, channel);

  // compute a packet (header + payload) duration
  uint32_t datarate = devNode->GetPhy ()->GetMode (0).GetDataRateBps ();
  UanHeaderCommon hd;
  double packetDuration = (m_packetSize + hd.GetSerializedSize ()) * 8.0 / (double) datarate;

  // energy source
  BasicEnergySourceHelper eh;
  eh.Set ("BasicEnergySourceInitialEnergyJ", DoubleValue (10000000.0));
  eh.Install (m_node);

  // mobility model
  Ptr<ConstantPositionMobilityModel> mobility = CreateObject<ConstantPositionMobilityModel> ();
  mobility->SetPosition (Vector (0,0,-500));
  m_node->AggregateObject (mobility);

  // micro modem energy model
  AcousticModemEnergyModelHelper modemHelper;
  Ptr<EnergySource> source = m_node->GetObject<EnergySourceContainer> ()->Get (0);
  DeviceEnergyModelContainer cont = modemHelper.Install (devNode,source);

  // Schedule a packet every 10 seconds
  Simulator::ScheduleNow (&AcousticModemEnergyTestCase::SendOnePacket,
                          this,
                          m_node);

  // create a gateway node
  m_gateway = CreateObject<Node> ();

  // install the underwater communication stack
  Ptr<UanNetDevice> devGateway = uan.Install (m_gateway, channel);

  // energy source
  eh.Set ("BasicEnergySourceInitialEnergyJ", DoubleValue (10000000.0));
  eh.Install (m_gateway);

  // mobility model
  Ptr<ConstantPositionMobilityModel> mobility2 = CreateObject<ConstantPositionMobilityModel> ();
  mobility2->SetPosition (Vector (0,0,0));
  m_gateway->AggregateObject (mobility2);

  // micro modem energy model
  Ptr<EnergySource> source2 = m_gateway->GetObject<EnergySourceContainer> ()->Get (0);
  DeviceEnergyModelContainer cont2 = modemHelper.Install (devGateway, source2);

  // set the receive callback
  Ptr<NetDevice> dev = m_gateway->GetDevice (0);
  dev->SetReceiveCallback (MakeCallback (&AcousticModemEnergyTestCase::RxPacket,
                                         this));

  // run the simulation
  Simulator::Stop (Seconds (m_simTime));
  Simulator::Run ();

  uint32_t receivedPackets = m_bytesRx / m_packetSize;
  Ptr<EnergySource> src1 = m_gateway->GetObject<EnergySourceContainer> ()->Get (0);
  double consumed1 = src1->GetInitialEnergy () - src1->GetRemainingEnergy ();
  double computed1 = cont2.Get (0)->GetObject<AcousticModemEnergyModel> ()->GetRxPowerW () * packetDuration * receivedPackets +
    cont2.Get (0)->GetObject<AcousticModemEnergyModel> ()->GetIdlePowerW () * (m_simTime - packetDuration * receivedPackets);

  NS_TEST_ASSERT_MSG_EQ_TOL (consumed1, computed1, 1.0e-5,
                             "Incorrect gateway consumed energy!");

  Ptr<EnergySource> src2 = m_node->GetObject<EnergySourceContainer> ()->Get (0);
  double consumed2 = src2->GetInitialEnergy () - src2->GetRemainingEnergy ();
  double computed2 = cont.Get (0)->GetObject<AcousticModemEnergyModel> ()->GetTxPowerW () * packetDuration * m_sentPackets +
    cont.Get (0)->GetObject<AcousticModemEnergyModel> ()->GetIdlePowerW () * (m_simTime - packetDuration * m_sentPackets);

  NS_TEST_ASSERT_MSG_EQ_TOL (consumed2, computed2, 1.0e-5,
                             "Incorrect node consumed energy!");

  Simulator::Destroy ();
}

/**
 * \ingroup uan-test
 * \ingroup tests
 *
 * \brief Acoustic Modem Energy Depletion Test Case
 */
class AcousticModemEnergyDepletionTestCase : public TestCase
{
public:
  AcousticModemEnergyDepletionTestCase ();
  ~AcousticModemEnergyDepletionTestCase ();

  /// Depletion handler function
  void DepletionHandler (void);
  /**
   * Send one packet function
   * \param node the node to send to
   */
  void SendOnePacket (Ptr<Node> node);

  void DoRun (void);

  double m_simTime; ///< Simulation time
  uint32_t m_callbackCount; ///< callback count
  uint32_t m_packetSize; ///< packet size
  Ptr<Node> m_node; ///< the node
};

AcousticModemEnergyDepletionTestCase::AcousticModemEnergyDepletionTestCase ()
  : TestCase ("Acoustic Modem energy depletion test case"),
    m_simTime (25),
    m_callbackCount (0),
    m_packetSize (17)
{
}

AcousticModemEnergyDepletionTestCase::~AcousticModemEnergyDepletionTestCase ()
{
  m_node = 0;
}

void
AcousticModemEnergyDepletionTestCase::DepletionHandler (void)
{
  // increase callback count
  m_callbackCount++;
}

void
AcousticModemEnergyDepletionTestCase::SendOnePacket (Ptr<Node> node)
{
  // create an empty packet
  Ptr<Packet> pkt = Create<Packet> (m_packetSize);
  // send the packet in broadcast
  Ptr<UanNetDevice> dev = node->GetDevice (0)->GetObject<UanNetDevice> ();
  dev->Send (pkt, dev->GetBroadcast (), 0);

  Simulator::Schedule (Seconds (10),
                       &AcousticModemEnergyDepletionTestCase::SendOnePacket,
                       this,
                       node);
}

void
AcousticModemEnergyDepletionTestCase::DoRun (void)
{
  // create a generic node
  m_node = CreateObject<Node> ();

  // create a default underwater channel
  Ptr<UanChannel> channel = CreateObject<UanChannel> ();
  Ptr<UanNoiseModelDefault> noise = CreateObject<UanNoiseModelDefault> ();
  channel->SetPropagationModel (CreateObject<UanPropModelIdeal> ());
  channel->SetNoiseModel (noise);

  // install the underwater communication stack
  UanHelper uan;
  Ptr<UanNetDevice> devNode = uan.Install (m_node, channel);

  // set an empty energy source
  BasicEnergySourceHelper eh;
  eh.Set ("BasicEnergySourceInitialEnergyJ", DoubleValue (0.0));
  eh.Install (m_node);

  // mobility model
  Ptr<ConstantPositionMobilityModel> mobility = CreateObject<ConstantPositionMobilityModel> ();
  mobility->SetPosition (Vector (0,0,0));
  m_node->AggregateObject (mobility);

  // micro modem energy model
  AcousticModemEnergyModelHelper modemHelper;
  Ptr<EnergySource> source = m_node->GetObject<EnergySourceContainer> ()->Get (0);
  // set the depletion callback
  AcousticModemEnergyModel::AcousticModemEnergyDepletionCallback callback =
    MakeCallback (&AcousticModemEnergyDepletionTestCase::DepletionHandler, this);
  modemHelper.SetDepletionCallback (callback);
  DeviceEnergyModelContainer cont = modemHelper.Install (devNode,source);

  // try to send a packet
  Simulator::ScheduleNow (&AcousticModemEnergyDepletionTestCase::SendOnePacket,
                          this,
                          m_node);

  Simulator::Stop (Seconds (m_simTime));
  Simulator::Run ();
  Simulator::Destroy ();

  NS_TEST_ASSERT_MSG_EQ (m_callbackCount, 1, "Callback not invoked");
}

// -------------------------------------------------------------------------- //

/**
 * \ingroup uan-test
 * \ingroup tests
 *
 * \brief Unit test suite for underwater energy model. Include test on acoustic modem,
 * acoustic modem energy depletion.
 */
class UanEnergyModelTestSuite : public TestSuite
{
public:
  UanEnergyModelTestSuite ();
};

UanEnergyModelTestSuite::UanEnergyModelTestSuite ()
  : TestSuite ("uan-energy-model", UNIT)
{
  AddTestCase (new AcousticModemEnergyTestCase, TestCase::QUICK);
  AddTestCase (new AcousticModemEnergyDepletionTestCase, TestCase::QUICK);
}

// create an instance of the test suite
static UanEnergyModelTestSuite g_uanEnergyModelTestSuite;
