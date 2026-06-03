/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Dalian University of Technology
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
 * Author: Junling Bu <linlinjavaer@gmail.com>
 */

#include <iostream>
#include "ns3/test.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/config.h"
#include "ns3/data-rate.h"
#include "ns3/vector.h"
#include "ns3/string.h"
#include "ns3/packet-socket-address.h"
#include "ns3/mobility-model.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/sta-wifi-mac.h"
#include "ns3/qos-txop.h"
#include "ns3/position-allocator.h"
#include "ns3/packet-socket-helper.h"
#include "ns3/mobility-helper.h"
#include "ns3/wifi-net-device.h"
#include "ns3/packet-socket-server.h"
#include "ns3/packet-socket-client.h"
#include "ns3/ocb-wifi-mac.h"
#include "ns3/wifi-80211p-helper.h"
#include "ns3/wave-mac-helper.h"

using namespace ns3;
// helper function to assign streams to random variables, to control
// randomness in the tests
static void
AssignWifiRandomStreams (Ptr<WifiMac> mac, int64_t stream)
{
  int64_t currentStream = stream;
  PointerValue ptr;
  if (!mac->GetQosSupported ())
    {
      mac->GetAttribute ("Txop", ptr);
      Ptr<Txop> txop = ptr.Get<Txop> ();
      currentStream += txop->AssignStreams (currentStream);
    }
  else
    {
      mac->GetAttribute ("VO_Txop", ptr);
      Ptr<QosTxop> vo_txop = ptr.Get<QosTxop> ();
      currentStream += vo_txop->AssignStreams (currentStream);

      mac->GetAttribute ("VI_Txop", ptr);
      Ptr<QosTxop> vi_txop = ptr.Get<QosTxop> ();
      currentStream += vi_txop->AssignStreams (currentStream);

      mac->GetAttribute ("BE_Txop", ptr);
      Ptr<QosTxop> be_txop = ptr.Get<QosTxop> ();
      currentStream += be_txop->AssignStreams (currentStream);

      mac->GetAttribute ("BK_Txop", ptr);
      Ptr<QosTxop> bk_txop = ptr.Get<QosTxop> ();
      currentStream += bk_txop->AssignStreams (currentStream);
    }
}

/**
 * \ingroup wave-test
 * \ingroup tests
 *
 * \brief Ocb Wifi Mac Test Case
 */
class OcbWifiMacTestCase : public TestCase
{
public:
  OcbWifiMacTestCase (void);
  virtual ~OcbWifiMacTestCase (void);
private:
  virtual void DoRun (void);

  /**
   * MAC associate function
   * \param context the context
   * \param bssid the BSSID
   */
  void MacAssoc (std::string context, Mac48Address bssid);
  /**
   * Phy receive ok trace function
   * \param context the context
   * \param packet the packet
   * \param snr the SNR
   * \param mode the mode
   * \param preamble the preamble
   */
  void PhyRxOkTrace (std::string context, Ptr<const Packet> packet, double snr, WifiMode mode, enum WifiPreamble preamble);
  /**
   * Phy transmit trace function
   * \param context the context
   * \param packet the packet
   * \param mode the mode
   * \param preamble the preamble
   * \param txPower the transmit power
   */
  void PhyTxTrace (std::string context, Ptr<const Packet> packet, WifiMode mode, WifiPreamble preamble, uint8_t txPower);
  /**
   * Get current position function
   * \param i the current position index
   * \returns the current position vector
   */
  Vector GetCurrentPosition (uint32_t i);
  /**
   * Advance position function
   * \param node the node
   */
  void AdvancePosition (Ptr<Node> node);

  /// Pre random configuration function
  void PreRandomConfiguration (void);
  /**
   * Configure AP STA mode function
   * \param static_node the static node
   * \param mobile_node the mobile node
   */
  void ConfigureApStaMode (Ptr<Node> static_node, Ptr<Node> mobile_node);
  /**
   * Configure adhoc mode function
   * \param static_node the static node
   * \param mobile_node the mobile node
   */
  void ConfigureAdhocMode (Ptr<Node> static_node, Ptr<Node> mobile_node);
  /**
   * Configure OCB mode function
   * \param static_node the static node
   * \param mobile_node the mobile node
   */
  void ConfigureOcbMode (Ptr<Node> static_node, Ptr<Node> mobile_node);
  /**
   * Post device configuration function
   * \param static_node the static node
   * \param mobile_node the mobile node
   */
  void PostDeviceConfiguration (Ptr<Node> static_node, Ptr<Node> mobile_node);

  Time phytx_time; ///< Phy transmit time
  Vector phytx_pos; ///< Phy transmit position

  Time macassoc_time; ///< MAC associate time
  Vector macassoc_pos; ///< MAC associate position

  Time phyrx_time; ///< Phy receive time
  Vector phyrx_pos; ///< Phy receive position

  // nodes.Get (0) is static node
  // nodes.Get (1) is mobile node
  NodeContainer nodes; ///< the nodes
};

OcbWifiMacTestCase::OcbWifiMacTestCase (void)
  : TestCase ("Association time: Ap+Sta mode vs Adhoc mode vs Ocb mode")
{
}

OcbWifiMacTestCase::~OcbWifiMacTestCase (void)
{
}

// mobility is like walk on line with velocity 5 m/s
// We prefer to update 0.5m every 0.1s rather than 5m every 1s
void
OcbWifiMacTestCase::AdvancePosition (Ptr<Node> node)
{
  Ptr<MobilityModel> mobility = node->GetObject<MobilityModel> ();
  Vector pos = mobility->GetPosition ();
  pos.x -= 0.5;
  if (pos.x < 1.0 )
    {
      pos.x = 1.0;
      return;
    }
  mobility->SetPosition (pos);

  Simulator::Schedule (Seconds (0.1), &OcbWifiMacTestCase::AdvancePosition, this, node);
}

// here are only two nodes, a stationary and a mobile one
// the i value of the first = 0; the i value of second = 1.
Vector
OcbWifiMacTestCase::GetCurrentPosition (uint32_t i)
{
  NS_ASSERT (i < 2);
  Ptr<Node> node = nodes.Get (i);
  Ptr<MobilityModel> mobility = node->GetObject<MobilityModel> ();
  Vector pos = mobility->GetPosition ();
  return pos;
}

void
OcbWifiMacTestCase::MacAssoc (std::string context,Mac48Address bssid)
{
  if (macassoc_time == Time (0))
    {
      macassoc_time = Now ();
      macassoc_pos = GetCurrentPosition (1);
      std::cout << "MacAssoc time = " << macassoc_time.As (Time::NS)
                << " position = " << macassoc_pos
                << std::endl;
    }
}

// We want to get the time that sta receives the first beacon frame from AP
// it means that in this time this sta has ability to receive frame
void
OcbWifiMacTestCase::PhyRxOkTrace (std::string context, Ptr<const Packet> packet, double snr, WifiMode mode, enum WifiPreamble preamble)
{
  if (phyrx_time == Time (0))
    {
      phyrx_time = Now ();
      phyrx_pos = GetCurrentPosition (1);
      std::cout << "PhyRxOk time = " << phyrx_time.As (Time::NS)
                << " position = " << phyrx_pos
                << std::endl;
    }
}

// We want to get the time that STA sends the first data packet successfully
void
OcbWifiMacTestCase::PhyTxTrace (std::string context, Ptr<const Packet> packet, WifiMode mode, WifiPreamble preamble, uint8_t txPower)
{
  WifiMacHeader h;
  packet->PeekHeader (h);
  if ((phytx_time == Time (0)) && h.IsData ())
    {
      phytx_time = Now ();
      phytx_pos = GetCurrentPosition (1);
      std::cout << "PhyTx data time = " << phytx_time.As (Time::NS)
                << " position = " << phytx_pos
                << std::endl;
    }
}

void
OcbWifiMacTestCase::ConfigureApStaMode (Ptr<Node> static_node, Ptr<Node> mobile_node)
{
  YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default ();
  YansWifiPhyHelper wifiPhy;
  wifiPhy.SetChannel (wifiChannel.Create ());

  Ssid ssid = Ssid ("wifi-default");
  WifiMacHelper wifiStaMac;
  wifiStaMac.SetType ("ns3::StaWifiMac", "Ssid", SsidValue (ssid));
  WifiMacHelper wifiApMac;
  wifiApMac.SetType ("ns3::ApWifiMac","Ssid", SsidValue (ssid));

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211p);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode", StringValue ("OfdmRate6MbpsBW10MHz"),
                                "ControlMode",StringValue ("OfdmRate6MbpsBW10MHz"));
  wifi.Install (wifiPhy, wifiStaMac, mobile_node);
  wifi.Install (wifiPhy, wifiApMac, static_node);
}

void
OcbWifiMacTestCase::ConfigureAdhocMode (Ptr<Node> static_node, Ptr<Node> mobile_node)
{
  YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default ();
  YansWifiPhyHelper wifiPhy;
  wifiPhy.SetChannel (wifiChannel.Create ());

  WifiMacHelper wifiMac;
  wifiMac.SetType ("ns3::AdhocWifiMac");

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211p);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode", StringValue ("OfdmRate6MbpsBW10MHz"),
                                "ControlMode",StringValue ("OfdmRate6MbpsBW10MHz"));
  wifi.Install (wifiPhy, wifiMac, mobile_node);
  wifi.Install (wifiPhy, wifiMac, static_node);
}

void
OcbWifiMacTestCase::ConfigureOcbMode (Ptr<Node> static_node, Ptr<Node> mobile_node)
{
  YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default ();
  YansWifiPhyHelper wifiPhy;
  wifiPhy.SetChannel (wifiChannel.Create ());

  NqosWaveMacHelper wifi80211pMac = NqosWaveMacHelper::Default ();

  Wifi80211pHelper wifi80211p = Wifi80211pHelper::Default ();
  wifi80211p.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                      "DataMode", StringValue ("OfdmRate6MbpsBW10MHz"),
                                      "ControlMode",StringValue ("OfdmRate6MbpsBW10MHz"));
  wifi80211p.Install (wifiPhy, wifi80211pMac, mobile_node);
  wifi80211p.Install (wifiPhy, wifi80211pMac, static_node);
}

void
OcbWifiMacTestCase::PostDeviceConfiguration (Ptr<Node> static_node, Ptr<Node> mobile_node)
{
  Ptr<WifiNetDevice> static_device = DynamicCast<WifiNetDevice> (static_node->GetDevice (0));
  Ptr<WifiNetDevice> mobile_device = DynamicCast<WifiNetDevice> (mobile_node->GetDevice (0));

  // Fix the stream assignment to the Dcf Txop objects (backoffs)
  // The below stream assignment will result in the Txop object
  // using a backoff value of zero for this test when the
  // Txop::EndTxNoAck() calls to StartBackoffNow()
  AssignWifiRandomStreams (static_device->GetMac (), 21);
  AssignWifiRandomStreams (mobile_device->GetMac (), 22);

  // setup mobility
  // the initial position of static node is at 0,
  // and the initial position of mobile node is 350.
  MobilityHelper mobility;
  mobility.Install (mobile_node);
  mobility.Install (static_node);
  Ptr<MobilityModel> mm = mobile_node->GetObject<MobilityModel> ();
  Vector possta = mm->GetPosition ();
  possta.x = 350;
  mm->SetPosition (possta);
  Simulator::Schedule (Seconds (1.0), &OcbWifiMacTestCase::AdvancePosition, this, mobile_node);

  PacketSocketAddress socket;
  socket.SetSingleDevice (mobile_device->GetIfIndex ());
  socket.SetPhysicalAddress (static_device->GetAddress ());
  socket.SetProtocol (1);

  // give packet socket powers to nodes.
  PacketSocketHelper packetSocket;
  packetSocket.Install (static_node);
  packetSocket.Install (mobile_node);

  Ptr<PacketSocketClient> client = CreateObject<PacketSocketClient> ();
  client->SetRemote (socket);
  mobile_node->AddApplication (client);
  client->SetStartTime (Seconds (0.5));
  client->SetStopTime (Seconds (70.0));

  Ptr<PacketSocketServer> server = CreateObject<PacketSocketServer> ();
  server->SetLocal (socket);
  static_node->AddApplication (server);
  server->SetStartTime (Seconds (0.0));
  server->SetStopTime (Seconds (70.5));

  phytx_time = macassoc_time = phyrx_time = Time ();
  phytx_pos = macassoc_pos = phyrx_pos = Vector ();

  if ( DynamicCast<StaWifiMac> (mobile_device->GetMac () ) )
    {
      // This trace is available only in a StaWifiMac
      Config::Connect ("/NodeList/1/DeviceList/*/Mac/Assoc", MakeCallback (&OcbWifiMacTestCase::MacAssoc, this));
    }
  Config::Connect ("/NodeList/1/DeviceList/*/Phy/State/RxOk", MakeCallback (&OcbWifiMacTestCase::PhyRxOkTrace, this));
  Config::Connect ("/NodeList/1/DeviceList/*/Phy/State/Tx", MakeCallback (&OcbWifiMacTestCase::PhyTxTrace, this));
}

/**
 *
 *   static-node:0    <----       mobile-node:1
 *        *   ------ 350m -------    *
 *
 * the node transmit range is less than 150m
 *
 * Ap+Sta mode vs Adhoc mode vs Ocb mode
 * first test the time point when the stationary node is
 * an AP and the mobile node is a Sta
 * then test when one Ad-hoc node and another Ad-hoc node
 * last test when one OCB node and another OCB node
 */
void
OcbWifiMacTestCase::DoRun ()
{
  std::cout << "test time point for Ap-Sta mode" << std::endl;
  PreRandomConfiguration ();
  nodes = NodeContainer ();
  nodes.Create (2);
  Ptr<Node> static_node = nodes.Get (0);
  Ptr<Node> mobile_node = nodes.Get (1);
  ConfigureApStaMode (static_node, mobile_node);
  PostDeviceConfiguration (static_node, mobile_node);
  Simulator::Stop (Seconds (71.0));
  Simulator::Run ();
  Simulator::Destroy ();
  NS_TEST_ASSERT_MSG_LT (phyrx_time, macassoc_time, "In Sta mode with AP, you cannot associate until receive beacon or AssocResponse frame" );
  NS_TEST_ASSERT_MSG_LT (macassoc_time, phytx_time, "In Sta mode with AP,  you cannot send data packet until associate" );
  //Are these position tests redundant with time check tests?
  //NS_TEST_ASSERT_MSG_GT ((phyrx_pos.x - macassoc_pos.x), 0.0, "");
  //actually macassoc_pos.x - phytx_pos.x is greater than 0
  //however associate switch to send is so fast with less than 100ms
  //and in our mobility model that every 0.1s update position,
  //so turn out to be that macassoc_pos.x - phytx_pos.x is equal to 0
  //NS_TEST_ASSERT_MSG_GT ((macassoc_pos.x - phytx_pos.x), 0.0, "");

  std::cout << "test time point for Adhoc mode" << std::endl;
  PreRandomConfiguration ();
  nodes = NodeContainer ();
  nodes.Create (2);
  static_node = nodes.Get (0);
  mobile_node = nodes.Get (1);
  ConfigureAdhocMode (static_node, mobile_node);
  PostDeviceConfiguration (static_node, mobile_node);
  Simulator::Stop (Seconds (71.0));
  Simulator::Run ();
  Simulator::Destroy ();
  // below test assert will fail, because AdhocWifiMac has not implement state machine.
  // if someone takes a look at the output in adhoc mode and in Ocb mode
  // he will find these two outputs are almost same.
  //NS_TEST_ASSERT_MSG_LT (phyrx_time, macassoc_time, "In Adhoc mode, you cannot associate until receive beacon or AssocResponse frame" );
  //NS_TEST_ASSERT_MSG_LT (macassoc_time, phytx_time, "In Adhoc mode,  you cannot send data packet until associate" );
  //NS_TEST_ASSERT_MSG_GT ((phyrx_pos.x - macassoc_pos.x), 0.0, "");
  // below test assert result refer to Ap-Sta mode
  //NS_TEST_ASSERT_MSG_GT ((macassoc_pos.x - phytx_pos.x), 0.0, "");

  std::cout << "test time point for Ocb mode" << std::endl;
  PreRandomConfiguration ();
  nodes = NodeContainer ();
  nodes.Create (2);
  static_node = nodes.Get (0);
  mobile_node = nodes.Get (1);
  ConfigureOcbMode (static_node, mobile_node);
  PostDeviceConfiguration (static_node, mobile_node);
  Simulator::Stop (Seconds (71.0));
  Simulator::Run ();
  Simulator::Destroy ();
  NS_TEST_ASSERT_MSG_EQ (macassoc_time.GetNanoSeconds(), 0, "In Ocb mode, there is no associate state machine" );
  NS_TEST_ASSERT_MSG_LT (phytx_time, phyrx_time, "before mobile node receives frames from far static node, it can send data packet directly" );
  NS_TEST_ASSERT_MSG_EQ (macassoc_pos.x, 0.0, "");
  NS_TEST_ASSERT_MSG_GT ((phytx_pos.x - phyrx_pos.x), 0.0, "");
}
void
OcbWifiMacTestCase::PreRandomConfiguration ()
{
  // Assign a seed and run number, and later fix the assignment of streams to
  // WiFi random variables, so that the first backoff used is zero slots
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (17);
  // the WiFi random variables is set in PostDeviceConfiguration method.
}

/**
 * \ingroup wave-test
 * \ingroup tests
 *
 * \brief Ocb Test Suite
 */
class OcbTestSuite : public TestSuite
{
public:
  OcbTestSuite ();
};

OcbTestSuite::OcbTestSuite ()
  : TestSuite ("wave-80211p-ocb", UNIT)
{
  // TestDuration for TestCase can be QUICK, EXTENSIVE or TAKES_FOREVER
  AddTestCase (new OcbWifiMacTestCase, TestCase::QUICK);
}

// Do not forget to allocate an instance of this TestSuite
static OcbTestSuite ocbTestSuite; ///< the test suite

