/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 * Author: Leonard Tracy <lentracy@gmail.com>
 */

#include "ns3/uan-net-device.h"
#include "ns3/uan-channel.h"
#include "ns3/uan-mac-aloha.h"
#include "ns3/uan-phy-gen.h"
#include "ns3/uan-transducer-hd.h"
#include "ns3/uan-prop-model-ideal.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/simulator.h"
#include "ns3/test.h"
#include "ns3/node.h"
#include "ns3/object-factory.h"
#include "ns3/pointer.h"
#include "ns3/callback.h"

using namespace ns3;

/**
 * \ingroup uan-test
 * \ingroup tests
 *
 * \brief Uan Test
 */
class UanTest : public TestCase
{
public:
  UanTest ();

  virtual void DoRun (void);
private:
  /**
   * Create node function
   * \param pos the position of the device
   * \param chan the communication channel
   * \returns the UAN device
   */
  Ptr<UanNetDevice> CreateNode (Vector pos, Ptr<UanChannel> chan);
  /**
   * Phy test function 
   * \returns true if successful
   */
  bool DoPhyTests ();
  /**
   * Do one Phy test function
   * \param t1 the time to send first packet
   * \param t2 the time to send the second packet
   * \param r1 first distance constant
   * \param r2 second distance constant
   * \param prop the propagation model
   * \param mode1 the send mode for device 1
   * \param mode2 the send mode for device 2
   * \returns number of bytes received
   */
  uint32_t DoOnePhyTest (Time t1, Time t2, uint32_t r1, uint32_t r2, Ptr<UanPropModel> prop, uint16_t mode1 = 0, uint16_t mode2 = 0);
  /**
   * Receive packet function
   * \param dev the device
   * \param pkt the packet
   * \param mode the receive mode
   * \param sender the address of the sender
   * \returns true if successful
   */
  bool RxPacket (Ptr<NetDevice> dev, Ptr<const Packet> pkt, uint16_t mode, const Address &sender);
  /**
   * Send one packet function
   * \param dev the device
   * \param mode the transmit mode
   */
  void SendOnePacket (Ptr<UanNetDevice> dev, uint16_t mode);
  ObjectFactory m_phyFac; ///< Phy 
  uint32_t m_bytesRx; ///< byes received

};


UanTest::UanTest () : TestCase ("UAN")
{

}

bool
UanTest::RxPacket (Ptr<NetDevice> dev, Ptr<const Packet> pkt, 
                   [[maybe_unused]] uint16_t mode, 
                   [[maybe_unused]] const Address &sender)
{
  m_bytesRx += pkt->GetSize ();
  return true;
}
void
UanTest::SendOnePacket (Ptr<UanNetDevice> dev, uint16_t mode)
{
  Ptr<Packet> pkt = Create<Packet> (17);
  dev->SetTxModeIndex (mode);
  dev->Send (pkt, dev->GetBroadcast (), 0);

}
Ptr<UanNetDevice>
UanTest::CreateNode (Vector pos, Ptr<UanChannel> chan)
{


  Ptr<UanPhy> phy = m_phyFac.Create<UanPhy> ();
  Ptr<Node> node = CreateObject<Node> ();
  Ptr<UanNetDevice> dev = CreateObject<UanNetDevice> ();
  Ptr<UanMacAloha> mac = CreateObject<UanMacAloha> ();
  Ptr<ConstantPositionMobilityModel> mobility = CreateObject<ConstantPositionMobilityModel> ();

  Ptr<UanTransducerHd> trans = CreateObject<UanTransducerHd> ();

  mobility->SetPosition (pos);
  node->AggregateObject (mobility);
  mac->SetAddress (Mac8Address::Allocate ());

  dev->SetPhy (phy);
  dev->SetMac (mac);
  dev->SetChannel (chan);
  dev->SetTransducer (trans);
  node->AddDevice (dev);

  return dev;
}


uint32_t
UanTest::DoOnePhyTest (Time txTime1,
                       Time txTime2,
                       uint32_t r1,
                       uint32_t r2,
                       Ptr<UanPropModel> prop,
                       uint16_t mode1,
                       uint16_t mode2)
{

  Ptr<UanChannel> channel = CreateObject<UanChannel> ();
  channel->SetAttribute ("PropagationModel", PointerValue (prop));

  Ptr<UanNetDevice> dev0 = CreateNode (Vector (r1,50,50), channel);
  Ptr<UanNetDevice> dev1 = CreateNode (Vector (0,50,50), channel);
  Ptr<UanNetDevice> dev2 = CreateNode (Vector (r1 + r2, 50, 50), channel);

  dev0->SetReceiveCallback (MakeCallback (&UanTest::RxPacket, this));

  Simulator::Schedule (txTime1, &UanTest::SendOnePacket, this, dev1, mode1);
  Simulator::Schedule (txTime2, &UanTest::SendOnePacket, this, dev2, mode2);

  m_bytesRx = 0;
  Simulator::Stop (Seconds (20.0));
  Simulator::Run ();
  Simulator::Destroy ();

  return m_bytesRx;
}


bool
UanTest::DoPhyTests ()
{
  // Phy Gen / Default PER / Default SINR
  UanModesList mList;
  UanTxMode mode = UanTxModeFactory::CreateMode (UanTxMode::FSK, 80, 80, 10000, 4000, 2, "TestMode");
  mList.AppendMode (UanTxMode (mode));
  Ptr<UanPhyPerGenDefault> perDef = CreateObject<UanPhyPerGenDefault> ();
  Ptr<UanPhyCalcSinrDefault> sinrDef = CreateObject<UanPhyCalcSinrDefault> ();
  m_phyFac.SetTypeId ("ns3::UanPhyGen");
  m_phyFac.Set ("PerModel", PointerValue (perDef));
  m_phyFac.Set ("SinrModel", PointerValue (sinrDef));
  m_phyFac.Set ("SupportedModes", UanModesListValue (mList));

  Ptr<UanPropModelIdeal> prop = CreateObject<UanPropModelIdeal> ();

  // No collision (Get 2 packets)
  NS_TEST_ASSERT_MSG_EQ_RETURNS_BOOL (DoOnePhyTest (Seconds (1.0), Seconds (3.001), 50, 50, prop),
                                      34, "Should have received 34 bytes from 2 disjoint packets");

  // Collision (Lose both packets)
  NS_TEST_ASSERT_MSG_EQ_RETURNS_BOOL (DoOnePhyTest (Seconds (1.0), Seconds (2.99), 50, 50, prop),
                                      0, "Expected collision resulting in loss of both packets");


  // Phy Gen / FH-FSK SINR check

  Ptr<UanPhyCalcSinrFhFsk> sinrFhfsk = CreateObject <UanPhyCalcSinrFhFsk> ();
  m_phyFac.Set ("PerModel", PointerValue (perDef));
  m_phyFac.Set ("SinrModel", PointerValue (sinrFhfsk));
  m_phyFac.Set ("SupportedModes", UanModesListValue (mList));

#ifdef UAN_PROP_BH_INSTALLED
  Ptr<UanPropModelBh> prop = CreateObject<UanPropModelBh> ();
#endif // UAN_PROP_BH_INSTALLED

  //  No collision (Get 2 packets)
  NS_TEST_ASSERT_MSG_EQ_RETURNS_BOOL (DoOnePhyTest (Seconds (1.0), Seconds (3.001), 50, 50, prop),
                                      34, "Should have received 34 bytes from 2 disjoint packets");

  // Should correctly receive first arriving packet
  NS_TEST_ASSERT_MSG_EQ_RETURNS_BOOL (DoOnePhyTest (Seconds (1.0), Seconds (1.0126), 50, 50, prop),
                                      17, "Should have received 17 bytes from first arriving packet");

  // Packets should collide and both be lost
  NS_TEST_ASSERT_MSG_EQ_RETURNS_BOOL (DoOnePhyTest (Seconds (1.0), Seconds (1.0 + 7.01 * (13.0 / 80.0)), 50, 50, prop),
                                      0, "Packets should collide, but received data");

  // Phy Dual
  UanTxMode mode00 = UanTxModeFactory::CreateMode (UanTxMode::FSK, 80, 80, 10000, 4000, 2, "TestMode00");
  UanTxMode mode10 = UanTxModeFactory::CreateMode (UanTxMode::FSK, 80, 80, 11000, 4000, 2, "TestMode10");
  UanTxMode mode20 = UanTxModeFactory::CreateMode (UanTxMode::FSK, 80, 80, 15000, 4000, 2, "TestMode20");
  UanTxMode mode01 = UanTxModeFactory::CreateMode (UanTxMode::FSK, 80, 80, 10000, 4000, 2, "TestMode01");
  UanTxMode mode11 = UanTxModeFactory::CreateMode (UanTxMode::FSK, 80, 80, 11000, 4000, 2, "TestMode11");
  UanTxMode mode21 = UanTxModeFactory::CreateMode (UanTxMode::FSK, 80, 80, 15000, 4000, 2, "TestMode21");

  UanModesList m0;
  m0.AppendMode (mode00);
  m0.AppendMode (mode10);
  m0.AppendMode (mode20);
  UanModesList m1;
  m1.AppendMode (mode01);
  m1.AppendMode (mode11);
  m1.AppendMode (mode21);

  m_phyFac = ObjectFactory ();
  m_phyFac.SetTypeId ("ns3::UanPhyDual");
  m_phyFac.Set ("SupportedModesPhy1", UanModesListValue (m0));
  m_phyFac.Set ("SupportedModesPhy2", UanModesListValue (m1));

  // No collision (Get 2 packets)
  NS_TEST_ASSERT_MSG_EQ_RETURNS_BOOL (DoOnePhyTest (Seconds (1.0), Seconds (3.01), 50, 50, prop),
                                      34, "Expected no collision");

  NS_TEST_ASSERT_MSG_EQ_RETURNS_BOOL (DoOnePhyTest (Seconds (1.0), Seconds (2.99), 50, 50, prop, 0, 0),
                                      0, "Expected collision with both packets lost");

  NS_TEST_ASSERT_MSG_EQ_RETURNS_BOOL (DoOnePhyTest (Seconds (1.0), Seconds (2.99), 50, 50, prop, 0, 2),
                                      17, "Expected collision with only one packets lost");

  NS_TEST_ASSERT_MSG_EQ_RETURNS_BOOL (DoOnePhyTest (Seconds (1.0), Seconds (2.99), 50, 50, prop, 0, 5),
                                      34, "Expected no collision");

  NS_TEST_ASSERT_MSG_EQ_RETURNS_BOOL (DoOnePhyTest (Seconds (1.0), Seconds (2.99), 50, 50, prop, 2, 3),
                                      34, "Expected no collision");

  return false;
}

void
UanTest::DoRun (void)
{

  Ptr<UanPhyPerUmodem> per = CreateObject<UanPhyPerUmodem> ();
  Ptr<Packet> pkt = Create<Packet> (1000);
  double error = per->CalcPer (pkt, 9, UanPhyGen::GetDefaultModes ()[0]);
  NS_TEST_ASSERT_MSG_EQ_TOL (error, 0.539, 0.001, "Got PER outside of tolerance");

#ifdef UAN_PROP_BH_INSTALLED
  //  Correct DB lookup for BH prop.
  Ptr<UanPropModelBh> propBh = CreateObject<UanPropModelBh> ();
  BellhopResp resp = propBh->GetResp (10000, 50, 50, 1000);

  NS_TEST_ASSERT_MSG_EQ_TOL (resp.GetPathLossDb (), -44.1753, 0.001, "Got BH Pathloss outside of tolerance");

  NS_TEST_ASSERT_MSG_EQ_TOL (resp.GetPdp ().GetTap (4).GetAmp (), 0.14159, 0.001, "Got BH arrival outside of tolerance");

#endif // UAN_PROP_BH_INSTALLED

  DoPhyTests ();
}

/**
 * \ingroup uan-test
 * \ingroup tests
 *
 * \brief Uan Test Suite
 */
class UanTestSuite : public TestSuite
{
public:
  UanTestSuite ();
};

UanTestSuite::UanTestSuite ()
  :  TestSuite ("devices-uan", UNIT)
{
  AddTestCase (new UanTest, TestCase::QUICK);
}

static UanTestSuite g_uanTestSuite; ///< the test suite

