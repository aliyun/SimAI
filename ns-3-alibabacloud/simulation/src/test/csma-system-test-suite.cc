/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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
 */

// This is not a test of CsmaNetDevice model behavior per-se, but
// instead is a roll up of several end-to-end examples in examples/csma
// directory, converted into system tests.  Writing a test suite
// to test Csma itself is for further study.

#include <string>

#include "ns3/address.h"
#include "ns3/application-container.h"
#include "ns3/bridge-helper.h"
#include "ns3/callback.h"
#include "ns3/config.h"
#include "ns3/csma-helper.h"
#include "ns3/csma-star-helper.h"
#include "ns3/inet-socket-address.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/node.h"
#include "ns3/data-rate.h"
#include "ns3/node-container.h"
#include "ns3/on-off-helper.h"
#include "ns3/packet.h"
#include "ns3/packet-sink-helper.h"
#include "ns3/packet-socket-helper.h"
#include "ns3/packet-socket-address.h"
#include "ns3/pointer.h"
#include "ns3/simple-channel.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"
#include "ns3/v4ping-helper.h"

using namespace ns3;

/**
 * \ingroup system-tests-csma
 * 
 * \brief CSMA Bridge mode test.
 */
class CsmaBridgeTestCase : public TestCase
{
public:
  CsmaBridgeTestCase ();
  virtual ~CsmaBridgeTestCase ();

private:
  virtual void DoRun (void);

  /**
   * Sink called when a packet is received.
   * \param p Received packet (unused).
   * \param ad Sender's address (uused).
   */
  void SinkRx (Ptr<const Packet> p, const Address &ad);
  uint32_t m_count; //!< Counter of received packets.
};

// Add some help text to this case to describe what it is intended to test
CsmaBridgeTestCase::CsmaBridgeTestCase ()
  : TestCase ("Bridge example for Carrier Sense Multiple Access (CSMA) networks"), m_count (0)
{
}

CsmaBridgeTestCase::~CsmaBridgeTestCase ()
{
}

void 
CsmaBridgeTestCase::SinkRx (Ptr<const Packet> p, const Address &ad)
{
  m_count++;
}

// Network topology
//
//        n0     n1
//        |      | 
//       ----------
//       | Switch |
//       ----------
//        |      | 
//        n2     n3
//
// - CBR/UDP test flow from n0 to n1; test that packets received on n1 
//
void
CsmaBridgeTestCase::DoRun (void)
{
  NodeContainer terminals;
  terminals.Create (4);

  NodeContainer csmaSwitch;
  csmaSwitch.Create (1);

  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", DataRateValue (5000000));
  csma.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));

  NetDeviceContainer terminalDevices;
  NetDeviceContainer switchDevices;

  for (int i = 0; i < 4; i++)
    {
      NetDeviceContainer link = csma.Install (NodeContainer (terminals.Get (i), csmaSwitch));
      terminalDevices.Add (link.Get (0));
      switchDevices.Add (link.Get (1));
    }

  // Create the bridge netdevice, which will do the packet switching
  Ptr<Node> switchNode = csmaSwitch.Get (0);
  BridgeHelper bridge;
  bridge.Install (switchNode, switchDevices);

  InternetStackHelper internet;
  internet.Install (terminals);

  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  ipv4.Assign (terminalDevices);

  uint16_t port = 9;   // Discard port (RFC 863)

  // Create the OnOff application to send UDP datagrams from n0 to n1.
  //
  // Make packets be sent about every DefaultPacketSize / DataRate = 
  // 4096 bits / (5000 bits/second) = 0.82 second.
  OnOffHelper onoff ("ns3::UdpSocketFactory", 
                     Address (InetSocketAddress (Ipv4Address ("10.1.1.2"), port)));
  onoff.SetConstantRate (DataRate (5000));

  ApplicationContainer app = onoff.Install (terminals.Get (0));
  app.Start (Seconds (1.0));
  app.Stop (Seconds (10.0));

  PacketSinkHelper sink ("ns3::UdpSocketFactory",
                         Address (InetSocketAddress (Ipv4Address::GetAny (), port)));
  app = sink.Install (terminals.Get (1));
  app.Start (Seconds (0.0));

  // Trace receptions
  Config::ConnectWithoutContext ("/NodeList/1/ApplicationList/0/$ns3::PacketSink/Rx", MakeCallback (&CsmaBridgeTestCase::SinkRx, this));

  Simulator::Run ();
  Simulator::Destroy ();

  // We should have sent and received 10 packets
  NS_TEST_ASSERT_MSG_EQ (m_count, 10, "Bridge should have passed 10 packets");
}

/**
 * \ingroup system-tests-csma
 * 
 * \brief CSMA Broadcast mode test.
 */
class CsmaBroadcastTestCase : public TestCase
{
public:
  CsmaBroadcastTestCase ();
  virtual ~CsmaBroadcastTestCase ();

private:
  virtual void DoRun (void);

  /**
   * Sink called when a packet is received by a node.
   * \param p Received packet (unused).
   * \param ad Sender's address (uused).
   * @{
   */
  void SinkRxNode1 (Ptr<const Packet> p, const Address &ad);
  void SinkRxNode2 (Ptr<const Packet> p, const Address &ad);
  /** @} */

  /**
   * Sink called when a packet is dropped.
   * \param p Received packet (unused).
   */
  void DropEvent (Ptr<const Packet> p);

  uint32_t m_countNode1; //!< Counter of received packets on node 1.
  uint32_t m_countNode2; //!< Counter of received packets on node 2.
  uint32_t m_drops;      //!< Counter of dropped packets.
};

// Add some help text to this case to describe what it is intended to test
CsmaBroadcastTestCase::CsmaBroadcastTestCase ()
  : TestCase ("Broadcast example for Carrier Sense Multiple Access (CSMA) networks"), m_countNode1 (0), m_countNode2 (0), m_drops (0)
{
}

CsmaBroadcastTestCase::~CsmaBroadcastTestCase ()
{
}

void 
CsmaBroadcastTestCase::SinkRxNode1 (Ptr<const Packet> p, const Address &ad)
{
  m_countNode1++;
}

void 
CsmaBroadcastTestCase::SinkRxNode2 (Ptr<const Packet> p, const Address &ad)
{
  m_countNode2++;
}

void 
CsmaBroadcastTestCase::DropEvent (Ptr<const Packet> p)
{
  m_drops++;
}

//
// Example of the sending of a datagram to a broadcast address
//
// Network topology
//     ==============
//       |          |
//       n0    n1   n2
//       |     |
//     ==========
//
//   n0 originates UDP broadcast to 255.255.255.255/discard port, which 
//   is replicated and received on both n1 and n2
//
void
CsmaBroadcastTestCase::DoRun (void)
{
  NodeContainer c;
  c.Create (3);
  NodeContainer c0 = NodeContainer (c.Get (0), c.Get (1));
  NodeContainer c1 = NodeContainer (c.Get (0), c.Get (2));

  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", DataRateValue (DataRate (5000000)));
  csma.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));

  NetDeviceContainer n0 = csma.Install (c0);
  NetDeviceContainer n1 = csma.Install (c1);

  InternetStackHelper internet;
  internet.Install (c);

  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.0.0", "255.255.255.0");
  ipv4.Assign (n0);
  ipv4.SetBase ("192.168.1.0", "255.255.255.0");
  ipv4.Assign (n1);


  // RFC 863 discard port ("9") indicates packet should be thrown away
  // by the system.  We allow this silent discard to be overridden
  // by the PacketSink application.
  uint16_t port = 9;

  // Create the OnOff application to send UDP datagrams from n0.
  //
  // Make packets be sent about every DefaultPacketSize / DataRate = 
  // 4096 bits / (5000 bits/second) = 0.82 second.
  OnOffHelper onoff ("ns3::UdpSocketFactory", 
                     Address (InetSocketAddress (Ipv4Address ("255.255.255.255"), port)));
  onoff.SetConstantRate (DataRate (5000));

  ApplicationContainer app = onoff.Install (c0.Get (0));
  // Start the application
  app.Start (Seconds (1.0));
  app.Stop (Seconds (10.0));

  // Create an optional packet sink to receive these packets
  PacketSinkHelper sink ("ns3::UdpSocketFactory",
                         Address (InetSocketAddress (Ipv4Address::GetAny (), port)));
  app = sink.Install (c0.Get (1));
  app.Add (sink.Install (c1.Get (1)));
  app.Start (Seconds (1.0));
  app.Stop (Seconds (10.0));

  // Trace receptions
  Config::ConnectWithoutContext ("/NodeList/1/ApplicationList/0/$ns3::PacketSink/Rx", MakeCallback (&CsmaBroadcastTestCase::SinkRxNode1, this));
  Config::ConnectWithoutContext ("/NodeList/2/ApplicationList/0/$ns3::PacketSink/Rx", MakeCallback (&CsmaBroadcastTestCase::SinkRxNode2, this));

  Simulator::Run ();
  Simulator::Destroy ();

  // We should have sent and received 10 packets
  NS_TEST_ASSERT_MSG_EQ (m_countNode1, 10, "Node 1 should have received 10 packets");
  NS_TEST_ASSERT_MSG_EQ (m_countNode2, 10, "Node 2 should have received 10 packets");
}

/**
 * \ingroup system-tests-csma
 * 
 * \brief CSMA Multicast mode test.
 */
class CsmaMulticastTestCase : public TestCase
{
public:
  CsmaMulticastTestCase ();
  virtual ~CsmaMulticastTestCase ();

private:
  virtual void DoRun (void);

  /**
   * Sink called when a packet is received by a node.
   * \param p Received packet (unused).
   * \param ad Sender's address (uused).
   */
  void SinkRx (Ptr<const Packet> p, const Address &ad);

  /**
   * Sink called when a packet is dropped.
   * \param p Received packet (unused).
   */
  void DropEvent (Ptr<const Packet> p);

  uint32_t m_count; //!< Counter of received packets.
  uint32_t m_drops; //!< Counter of dropped packets.
};

// Add some help text to this case to describe what it is intended to test
CsmaMulticastTestCase::CsmaMulticastTestCase ()
  : TestCase ("Multicast example for Carrier Sense Multiple Access (CSMA) networks"), m_count (0), m_drops (0)
{
}

CsmaMulticastTestCase::~CsmaMulticastTestCase ()
{
}

void 
CsmaMulticastTestCase::SinkRx (Ptr<const Packet> p, const Address& ad)
{
  m_count++;
}

void 
CsmaMulticastTestCase::DropEvent (Ptr<const Packet> p)
{
  m_drops++;
}

// Network topology
//
//                     Lan1
//                 ===========
//                 |    |    | 
//       n0   n1   n2   n3   n4
//       |    |    |
//       ===========
//           Lan0
//
// - Multicast source is at node n0;
// - Multicast forwarded by node n2 onto LAN1;
// - Nodes n0, n1, n2, n3, and n4 receive the multicast frame.
// - Node n4 listens for the data 
//
void
CsmaMulticastTestCase::DoRun (void)
{
  //
  // Set up default values for the simulation.
  //
  // Select DIX/Ethernet II-style encapsulation (no LLC/Snap header)
  Config::SetDefault ("ns3::CsmaNetDevice::EncapsulationMode", StringValue ("Dix"));

  NodeContainer c;
  c.Create (5);
  // We will later want two subcontainers of these nodes, for the two LANs
  NodeContainer c0 = NodeContainer (c.Get (0), c.Get (1), c.Get (2));
  NodeContainer c1 = NodeContainer (c.Get (2), c.Get (3), c.Get (4));

  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", DataRateValue (DataRate (5000000)));
  csma.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));
 
  // We will use these NetDevice containers later, for IP addressing
  NetDeviceContainer nd0 = csma.Install (c0);  // First LAN
  NetDeviceContainer nd1 = csma.Install (c1);  // Second LAN

  InternetStackHelper internet;
  internet.Install (c);

  Ipv4AddressHelper ipv4Addr;
  ipv4Addr.SetBase ("10.1.1.0", "255.255.255.0");
  ipv4Addr.Assign (nd0);
  ipv4Addr.SetBase ("10.1.2.0", "255.255.255.0");
  ipv4Addr.Assign (nd1);

  //
  // Now we can configure multicasting.  As described above, the multicast 
  // source is at node zero, which we assigned the IP address of 10.1.1.1 
  // earlier.  We need to define a multicast group to send packets to.  This
  // can be any multicast address from 224.0.0.0 through 239.255.255.255
  // (avoiding the reserved routing protocol addresses).
  //

  Ipv4Address multicastSource ("10.1.1.1");
  Ipv4Address multicastGroup ("225.1.2.4");

  // Now, we will set up multicast routing.  We need to do three things:
  // 1) Configure a (static) multicast route on node n2
  // 2) Set up a default multicast route on the sender n0 
  // 3) Have node n4 join the multicast group
  // We have a helper that can help us with static multicast
  Ipv4StaticRoutingHelper multicast;

  // 1) Configure a (static) multicast route on node n2 (multicastRouter)
  Ptr<Node> multicastRouter = c.Get (2);  // The node in question
  Ptr<NetDevice> inputIf = nd0.Get (2);  // The input NetDevice
  NetDeviceContainer outputDevices;  // A container of output NetDevices
  outputDevices.Add (nd1.Get (0));  // (we only need one NetDevice here)

  multicast.AddMulticastRoute (multicastRouter, multicastSource, 
                               multicastGroup, inputIf, outputDevices);

  // 2) Set up a default multicast route on the sender n0 
  Ptr<Node> sender = c.Get (0);
  Ptr<NetDevice> senderIf = nd0.Get (0);
  multicast.SetDefaultMulticastRoute (sender, senderIf);

  //
  // Create an OnOff application to send UDP datagrams from node zero to the
  // multicast group (node four will be listening).
  //

  uint16_t multicastPort = 9;   // Discard port (RFC 863)

  // Configure a multicast packet generator.
  //
  // Make packets be sent about every defaultPacketSize / dataRate = 
  // 4096 bits / (5000 bits/second) = 0.82 second.
  OnOffHelper onoff ("ns3::UdpSocketFactory", 
                     Address (InetSocketAddress (multicastGroup, multicastPort)));
  onoff.SetConstantRate (DataRate (5000));

  ApplicationContainer srcC = onoff.Install (c0.Get (0));

  //
  // Tell the application when to start and stop.
  //
  srcC.Start (Seconds (1.));
  srcC.Stop (Seconds (10.));

  // Create an optional packet sink to receive these packets
  PacketSinkHelper sink ("ns3::UdpSocketFactory",
                         InetSocketAddress (Ipv4Address::GetAny (), multicastPort));

  ApplicationContainer sinkC = sink.Install (c1.Get (2)); // Node n4 
  // Start the sink
  sinkC.Start (Seconds (1.0));
  sinkC.Stop (Seconds (10.0));

  // Trace receptions
  Config::ConnectWithoutContext ("/NodeList/4/ApplicationList/0/$ns3::PacketSink/Rx", MakeCallback (&CsmaMulticastTestCase::SinkRx, this));

  //
  // Now, do the actual simulation.
  //
  Simulator::Run ();
  Simulator::Destroy ();

  // We should have sent and received 10 packets
  NS_TEST_ASSERT_MSG_EQ (m_count, 10, "Node 4 should have received 10 packets");
}

/**
 * \ingroup system-tests-csma
 * 
 * \brief CSMA One Subnet mode test.
 */
class CsmaOneSubnetTestCase : public TestCase
{
public:
  CsmaOneSubnetTestCase ();
  virtual ~CsmaOneSubnetTestCase ();

private:
  virtual void DoRun (void);

  /**
   * Sink called when a packet is received by a node.
   * \param p Received packet (unused).
   * \param ad Sender's address (uused).
   * @{
   */
  void SinkRxNode0 (Ptr<const Packet> p, const Address &ad);
  void SinkRxNode1 (Ptr<const Packet> p, const Address &ad);
  /** @} */

  /**
   * Sink called when a packet is dropped.
   * \param p Received packet (unused).
   */
  void DropEvent (Ptr<const Packet> p);
  uint32_t m_countNode0; //!< Counter of received packets on node 0.
  uint32_t m_countNode1; //!< Counter of received packets on node 1.
  uint32_t m_drops;      //!< Counter of dropped packets.
};

// Add some help text to this case to describe what it is intended to test
CsmaOneSubnetTestCase::CsmaOneSubnetTestCase ()
  : TestCase ("One subnet example for Carrier Sense Multiple Access (CSMA) networks"), m_countNode0 (0), m_countNode1 (0), m_drops (0)
{
}

CsmaOneSubnetTestCase::~CsmaOneSubnetTestCase ()
{
}

void 
CsmaOneSubnetTestCase::SinkRxNode0 (Ptr<const Packet> p, const Address &ad)
{
  m_countNode0++;
}

void 
CsmaOneSubnetTestCase::SinkRxNode1 (Ptr<const Packet> p, const Address &ad)
{
  m_countNode1++;
}

void 
CsmaOneSubnetTestCase::DropEvent (Ptr<const Packet> p)
{
  m_drops++;
}

// Network topology
//
//       n0    n1   n2   n3
//       |     |    |    |
//       =================
//              LAN
//
// - CBR/UDP flows from n0 to n1 and from n3 to n0
// - DropTail queues 
//
void
CsmaOneSubnetTestCase::DoRun (void)
{
  NodeContainer nodes;
  nodes.Create (4);

  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", DataRateValue (5000000));
  csma.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));
  //
  // Now fill out the topology by creating the net devices required to connect
  // the nodes to the channels and hooking them up.
  //
  NetDeviceContainer devices = csma.Install (nodes);

  InternetStackHelper internet;
  internet.Install (nodes);

  // We've got the "hardware" in place.  Now we need to add IP addresses.
  //
  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces = ipv4.Assign (devices);

  uint16_t port = 9;   // Discard port (RFC 863)

  //
  // Create an OnOff application to send UDP datagrams from node zero
  // to node 1.
  //
  // Make packets be sent about every defaultPacketSize / dataRate = 
  // 4096 bits / (5000 bits/second) = 0.82 second.
  OnOffHelper onoff ("ns3::UdpSocketFactory", 
                     Address (InetSocketAddress (interfaces.GetAddress (1), port)));
  onoff.SetConstantRate (DataRate (5000));

  ApplicationContainer app = onoff.Install (nodes.Get (0));
  // Start the application
  app.Start (Seconds (1.0));
  app.Stop (Seconds (10.0));

  // Create an optional packet sink to receive these packets
  PacketSinkHelper sink ("ns3::UdpSocketFactory",
                         Address (InetSocketAddress (Ipv4Address::GetAny (), port)));
  app = sink.Install (nodes.Get (1));
  app.Start (Seconds (0.0));

  // 
  // Create a similar flow from n3 to n0, starting at time 1.1 seconds
  //
  onoff.SetAttribute ("Remote", 
                      AddressValue (InetSocketAddress (interfaces.GetAddress (0), port)));
  app = onoff.Install (nodes.Get (3));
  app.Start (Seconds (1.1));
  app.Stop (Seconds (10.0));

  app = sink.Install (nodes.Get (0));
  app.Start (Seconds (0.0));

  // Trace receptions
  Config::ConnectWithoutContext ("/NodeList/0/ApplicationList/1/$ns3::PacketSink/Rx", MakeCallback (&CsmaOneSubnetTestCase::SinkRxNode0, this));
  Config::ConnectWithoutContext ("/NodeList/1/ApplicationList/0/$ns3::PacketSink/Rx", MakeCallback (&CsmaOneSubnetTestCase::SinkRxNode1, this));

  //
  // Now, do the actual simulation.
  //
  Simulator::Run ();
  Simulator::Destroy ();

  // We should have sent and received 10 packets
  NS_TEST_ASSERT_MSG_EQ (m_countNode0, 10, "Node 0 should have received 10 packets");
  NS_TEST_ASSERT_MSG_EQ (m_countNode1, 10, "Node 1 should have received 10 packets");
}

/**
 * \ingroup system-tests-csma
 * 
 * \brief CSMA PacketSocket test.
 */
class CsmaPacketSocketTestCase : public TestCase
{
public:
  CsmaPacketSocketTestCase ();
  virtual ~CsmaPacketSocketTestCase ();

private:
  virtual void DoRun (void);
  /**
   * Sink called when a packet is received by a node.
   * \param path Sink path.
   * \param p Received packet (unused).
   * \param ad Sender's address (uused).
   */
  void SinkRx (std::string path, Ptr<const Packet> p, const Address &ad);

  /**
   * Sink called when a packet is dropped.
   * \param p Received packet (unused).
   */
  void DropEvent (Ptr<const Packet> p);

  uint32_t m_count; //!< Counter of received packets.
  uint32_t m_drops; //!< Counter of dropped packets.
};

// Add some help text to this case to describe what it is intended to test
CsmaPacketSocketTestCase::CsmaPacketSocketTestCase ()
  : TestCase ("Packet socket example for Carrier Sense Multiple Access (CSMA) networks"), m_count (0), m_drops (0)
{
}

CsmaPacketSocketTestCase::~CsmaPacketSocketTestCase ()
{
}

void 
CsmaPacketSocketTestCase::SinkRx (std::string path, Ptr<const Packet> p, const Address& address)
{
  m_count++;
}

void 
CsmaPacketSocketTestCase::DropEvent (Ptr<const Packet> p)
{
  m_drops++;
}

//
// Network topology
//
//       n0    n1   n2   n3
//       |     |    |    |
//     =====================
//
// - Packet socket flow from n0 to n1 and from node n3 to n0
// -- We will test reception at node n0
// - Default 512 byte packets generated by traffic generator
//
void
CsmaPacketSocketTestCase::DoRun (void)
{
  // Here, we will explicitly create four nodes.
  NodeContainer nodes;
  nodes.Create (4);

  PacketSocketHelper packetSocket;
  packetSocket.Install (nodes);

  // create the shared medium used by all csma devices.
  Ptr<CsmaChannel> channel = CreateObjectWithAttributes<CsmaChannel> (
      "DataRate", DataRateValue (DataRate (5000000)),
      "Delay", TimeValue (MilliSeconds (2)));

  // use a helper function to connect our nodes to the shared channel.
  CsmaHelper csma;
  csma.SetDeviceAttribute ("EncapsulationMode", StringValue ("Llc"));
  NetDeviceContainer devs = csma.Install (nodes, channel);

  // Create the OnOff application to send raw datagrams
  //
  // Make packets be sent about every DefaultPacketSize / DataRate = 
  // 4096 bits / (5000 bits/second) = 0.82 second.
  PacketSocketAddress socket;
  socket.SetSingleDevice (devs.Get (0)->GetIfIndex ());
  socket.SetPhysicalAddress (devs.Get (1)->GetAddress ());
  socket.SetProtocol (2);
  OnOffHelper onoff ("ns3::PacketSocketFactory", Address (socket));
  onoff.SetConstantRate (DataRate (5000));
  ApplicationContainer apps = onoff.Install (nodes.Get (0));
  apps.Start (Seconds (1.0));
  apps.Stop (Seconds (10.0));

  socket.SetSingleDevice (devs.Get (3)->GetIfIndex ());
  socket.SetPhysicalAddress (devs.Get (0)->GetAddress ());
  socket.SetProtocol (3);
  onoff.SetAttribute ("Remote", AddressValue (socket));
  apps = onoff.Install (nodes.Get (3));
  apps.Start (Seconds (1.0));
  apps.Stop (Seconds (10.0));

  PacketSinkHelper sink = PacketSinkHelper ("ns3::PacketSocketFactory",
                                            socket);
  apps = sink.Install (nodes.Get (0));
  apps.Start (Seconds (0.0));
  apps.Stop (Seconds (20.0));

  // Trace receptions
  Config::Connect ("/NodeList/0/ApplicationList/*/$ns3::PacketSink/Rx",
                   MakeCallback (&CsmaPacketSocketTestCase::SinkRx, this));
 
  Simulator::Run ();
  Simulator::Destroy ();

  // We should have received 10 packets on node 0
  NS_TEST_ASSERT_MSG_EQ (m_count, 10, "Node 0 should have received 10 packets");
}

/**
 * \ingroup system-tests-csma
 * 
 * \brief CSMA PING test.
 */
class CsmaPingTestCase : public TestCase
{
public:
  CsmaPingTestCase ();
  virtual ~CsmaPingTestCase ();

private:
  virtual void DoRun (void);
  /**
   * Sink called when a packet is received by a node.
   * \param p Received packet (unused).
   * \param ad Sender's address (uused).
   */
  void SinkRx (Ptr<const Packet> p, const Address &ad);

  /**
   * Sink called when a PING ois received.
   * \param context Context path (unused).
   * \param rtt Round Trip Time (unused).
   */
  void PingRtt (std::string context, Time rtt);

  /**
   * Sink called when a packet is dropped.
   * \param p Received packet (unused).
   */
  void DropEvent (Ptr<const Packet> p);

  uint32_t m_countSinkRx;   //!< Counter of received packets.
  uint32_t m_countPingRtt;  //!< Counter of PING received.
  uint32_t m_drops;         //!< Counter of dropped packets.
};

// Add some help text to this case to describe what it is intended to test
CsmaPingTestCase::CsmaPingTestCase ()
  : TestCase ("Ping example for Carrier Sense Multiple Access (CSMA) networks"), m_countSinkRx (0), m_countPingRtt (0), m_drops (0)
{
}

CsmaPingTestCase::~CsmaPingTestCase ()
{
}

void 
CsmaPingTestCase::SinkRx (Ptr<const Packet> p, const Address &ad)
{
  m_countSinkRx++;
}

void 
CsmaPingTestCase::PingRtt (std::string context, Time rtt)
{
  m_countPingRtt++;
}

void 
CsmaPingTestCase::DropEvent (Ptr<const Packet> p)
{
  m_drops++;
}

// Network topology
//
//       n0    n1   n2   n3
//       |     |    |    |
//     =====================
//
//  node n0,n1,n3 pings to node n2
//  node n0 generates protocol 2 (IGMP) to node n3
//
void
CsmaPingTestCase::DoRun (void)
{
  // Here, we will explicitly create four nodes.
  NodeContainer c;
  c.Create (4);

  // connect all our nodes to a shared channel.
  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", DataRateValue (DataRate (5000000)));
  csma.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));
  csma.SetDeviceAttribute ("EncapsulationMode", StringValue ("Llc"));
  NetDeviceContainer devs = csma.Install (c);

  // add an ip stack to all nodes.
  InternetStackHelper ipStack;
  ipStack.Install (c);

  // assign ip addresses
  Ipv4AddressHelper ip;
  ip.SetBase ("192.168.1.0", "255.255.255.0");
  Ipv4InterfaceContainer addresses = ip.Assign (devs);

  // Create the OnOff application to send UDP datagrams from n0 to n1.
  //
  // Make packets be sent about every DefaultPacketSize / DataRate = 
  // 4096 bits / (5000 bits/second) = 0.82 second.
  Config::SetDefault ("ns3::Ipv4RawSocketImpl::Protocol", StringValue ("2"));
  InetSocketAddress dst = InetSocketAddress (addresses.GetAddress (3));
  OnOffHelper onoff = OnOffHelper ("ns3::Ipv4RawSocketFactory", dst);
  onoff.SetConstantRate (DataRate (5000));

  ApplicationContainer apps = onoff.Install (c.Get (0));
  apps.Start (Seconds (1.0));
  apps.Stop (Seconds (10.0));

  PacketSinkHelper sink = PacketSinkHelper ("ns3::Ipv4RawSocketFactory", dst);
  apps = sink.Install (c.Get (3));
  apps.Start (Seconds (0.0));
  apps.Stop (Seconds (11.0));

  V4PingHelper ping = V4PingHelper (addresses.GetAddress (2));
  NodeContainer pingers;
  pingers.Add (c.Get (0));
  pingers.Add (c.Get (1));
  pingers.Add (c.Get (3));
  apps = ping.Install (pingers);
  apps.Start (Seconds (2.0));
  apps.Stop (Seconds (5.0));

  // Trace receptions
  Config::ConnectWithoutContext ("/NodeList/3/ApplicationList/0/$ns3::PacketSink/Rx", 
                                 MakeCallback (&CsmaPingTestCase::SinkRx, this));

  // Trace pings
  Config::Connect ("/NodeList/*/ApplicationList/*/$ns3::V4Ping/Rtt",
                   MakeCallback (&CsmaPingTestCase::PingRtt, this));

  Simulator::Run ();
  Simulator::Destroy ();

  // We should have sent and received 10 packets
  NS_TEST_ASSERT_MSG_EQ (m_countSinkRx, 10, "Node 3 should have received 10 packets");

  // We should have 3 pingers that ping every second for 3 seconds.
  NS_TEST_ASSERT_MSG_EQ (m_countPingRtt, 9, "Node 2 should have been pinged 9 times");
}

/**
 * \ingroup system-tests-csma
 * 
 * \brief CSMA Raw IP test.
 */
class CsmaRawIpSocketTestCase : public TestCase
{
public:
  CsmaRawIpSocketTestCase ();
  virtual ~CsmaRawIpSocketTestCase ();

private:
  virtual void DoRun (void);

  /**
   * Sink called when a packet is received by a node.
   * \param p Received packet (unused).
   * \param ad Sender's address (uused).
   */
  void SinkRx (Ptr<const Packet> p, const Address &ad);

  /**
   * Sink called when a packet is dropped.
   * \param p Received packet (unused).
   */
  void DropEvent (Ptr<const Packet> p);

  uint32_t m_count; //!< Counter of received packets.
  uint32_t m_drops; //!< Counter of dropped packets.
};

// Add some help text to this case to describe what it is intended to test
CsmaRawIpSocketTestCase::CsmaRawIpSocketTestCase ()
  : TestCase ("Raw internet protocol socket example for Carrier Sense Multiple Access (CSMA) networks"), m_count (0), m_drops (0)
{
}

CsmaRawIpSocketTestCase::~CsmaRawIpSocketTestCase ()
{
}

void 
CsmaRawIpSocketTestCase::SinkRx (Ptr<const Packet> p, const Address &ad)
{
  m_count++;
}

void 
CsmaRawIpSocketTestCase::DropEvent (Ptr<const Packet> p)
{
  m_drops++;
}

//
// Network topology
//    (sender)         (receiver)
//       n0    n1   n2   n3
//       |     |    |    |
//     =====================
//
// Node n0 sends data to node n3 over a raw IP socket.  The protocol
// number used is 2.
//
void
CsmaRawIpSocketTestCase::DoRun (void)
{
  // Here, we will explicitly create four nodes.
  NodeContainer c;
  c.Create (4);

  // connect all our nodes to a shared channel.
  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", DataRateValue (DataRate (5000000)));
  csma.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));
  csma.SetDeviceAttribute ("EncapsulationMode", StringValue ("Llc"));
  NetDeviceContainer devs = csma.Install (c);

  // add an ip stack to all nodes.
  InternetStackHelper ipStack;
  ipStack.Install (c);

  // assign ip addresses
  Ipv4AddressHelper ip;
  ip.SetBase ("192.168.1.0", "255.255.255.0");
  Ipv4InterfaceContainer addresses = ip.Assign (devs);

  // IP protocol configuration
  //
  // Make packets be sent about every DefaultPacketSize / DataRate = 
  // 4096 bits / (5000 bits/second) = 0.82 second.
  Config::SetDefault ("ns3::Ipv4RawSocketImpl::Protocol", StringValue ("2"));
  InetSocketAddress dst = InetSocketAddress (addresses.GetAddress (3));
  OnOffHelper onoff = OnOffHelper ("ns3::Ipv4RawSocketFactory", dst);
  onoff.SetConstantRate (DataRate (5000));

  ApplicationContainer apps = onoff.Install (c.Get (0));
  apps.Start (Seconds (1.0));
  apps.Stop (Seconds (10.0));

  PacketSinkHelper sink = PacketSinkHelper ("ns3::Ipv4RawSocketFactory", dst);
  apps = sink.Install (c.Get (3));
  apps.Start (Seconds (0.0));
  apps.Stop (Seconds (12.0));

  // Trace receptions
  Config::ConnectWithoutContext ("/NodeList/3/ApplicationList/0/$ns3::PacketSink/Rx", 
                                 MakeCallback (&CsmaRawIpSocketTestCase::SinkRx, this));

  Simulator::Run ();
  Simulator::Destroy ();

  // We should have sent and received 10 packets
  NS_TEST_ASSERT_MSG_EQ (m_count, 10, "Node 3 should have received 10 packets");
}

/**
 * \ingroup system-tests-csma
 * 
 * \brief CSMA star mode test.
 */
class CsmaStarTestCase : public TestCase
{
public:
  CsmaStarTestCase ();
  virtual ~CsmaStarTestCase ();

private:
  virtual void DoRun (void);

  /**
   * Sink called when a packet is received by a node.
   * \param p Received packet (unused).
   * \param ad Sender's address (uused).
   */
  void SinkRx (Ptr<const Packet> p, const Address &ad);

  /**
   * Sink called when a packet is dropped.
   * \param p Received packet (unused).
   */
  void DropEvent (Ptr<const Packet> p);

  uint32_t m_count; //!< Counter of received packets.
  uint32_t m_drops; //!< Counter of dropped packets.
};

// Add some help text to this case to describe what it is intended to test
CsmaStarTestCase::CsmaStarTestCase ()
  : TestCase ("Star example for Carrier Sense Multiple Access (CSMA) networks"), m_count (0), m_drops (0)
{
}

CsmaStarTestCase::~CsmaStarTestCase ()
{
}

void 
CsmaStarTestCase::SinkRx (Ptr<const Packet> p, const Address& ad)
{
  m_count++;
}

void 
CsmaStarTestCase::DropEvent (Ptr<const Packet> p)
{
  m_drops++;
}

// Network topology (default)
//
//            n2     +          +     n3          .
//             | ... |\        /| ... |           .
//             ======= \      / =======           .
//              CSMA    \    /   CSMA             .
//                       \  /                     .
//            n1     +--- n0 ---+     n4          .
//             | ... |   /  \   | ... |           .
//             =======  /    \  =======           .
//              CSMA   /      \  CSMA             .
//                    /        \                  .
//            n6     +          +     n5          .
//             | ... |          | ... |           .
//             =======          =======           .
//              CSMA             CSMA             .
//
void
CsmaStarTestCase::DoRun (void)
{
  //
  // Default number of nodes in the star.
  //
  uint32_t nSpokes = 7;

  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", StringValue ("100Mbps"));
  csma.SetChannelAttribute ("Delay", StringValue ("1ms"));
  CsmaStarHelper star (nSpokes, csma);

  NodeContainer fillNodes;

  //
  // Just to be nasy, hang some more nodes off of the CSMA channel for each
  // spoke, so that there are a total of 16 nodes on each channel.  Stash
  // all of these new devices into a container.
  //
  NetDeviceContainer fillDevices;

  uint32_t nFill = 14;
  for (uint32_t i = 0; i < star.GetSpokeDevices ().GetN (); ++i)
    {
      Ptr<Channel> channel = star.GetSpokeDevices ().Get (i)->GetChannel ();
      Ptr<CsmaChannel> csmaChannel = channel->GetObject<CsmaChannel> ();
      NodeContainer newNodes;
      newNodes.Create (nFill);
      fillNodes.Add (newNodes);
      fillDevices.Add (csma.Install (newNodes, csmaChannel));
    }

  InternetStackHelper internet;
  star.InstallStack (internet);
  internet.Install (fillNodes);

  star.AssignIpv4Addresses (Ipv4AddressHelper ("10.1.0.0", "255.255.255.0"));

  //
  // We assigned addresses to the logical hub and the first "drop" of the 
  // CSMA network that acts as the spoke, but we also have a number of fill
  // devices (nFill) also hanging off the CSMA network.  We have got to 
  // assign addresses to them as well.  We put all of the fill devices into
  // a single device container, so the first nFill devices are associated
  // with the channel connected to spokeDevices.Get (0), the second nFill
  // devices afe associated with the channel connected to spokeDevices.Get (1)
  // etc.
  //
  Ipv4AddressHelper address;
  for(uint32_t i = 0; i < star.SpokeCount (); ++i)
    {
      std::ostringstream subnet;
      subnet << "10.1." << i << ".0";
      address.SetBase (subnet.str ().c_str (), "255.255.255.0", "0.0.0.3");

      for (uint32_t j = 0; j < nFill; ++j)
        {
          address.Assign (fillDevices.Get (i * nFill + j));
        }
    }

  //
  // Create a packet sink on the star "hub" to receive packets.
  // 
  uint16_t port = 50000;
  Address hubLocalAddress (InetSocketAddress (Ipv4Address::GetAny (), port));
  PacketSinkHelper packetSinkHelper ("ns3::TcpSocketFactory", hubLocalAddress);
  ApplicationContainer hubApp = packetSinkHelper.Install (star.GetHub ());
  hubApp.Start (Seconds (1.0));
  hubApp.Stop (Seconds (10.0));

  //
  // Create OnOff applications to send TCP to the hub, one on each spoke node.
  //
  // Make packets be sent about every DefaultPacketSize / DataRate = 
  // 4096 bits / (5000 bits/second) = 0.82 second.
  OnOffHelper onOffHelper ("ns3::TcpSocketFactory", Address ());
  onOffHelper.SetConstantRate (DataRate (5000));

  ApplicationContainer spokeApps;

  for (uint32_t i = 0; i < star.SpokeCount (); ++i)
    {
      AddressValue remoteAddress (InetSocketAddress (star.GetHubIpv4Address (i), port));
      onOffHelper.SetAttribute ("Remote", remoteAddress);
      spokeApps.Add (onOffHelper.Install (star.GetSpokeNode (i)));
    }

  spokeApps.Start (Seconds (1.0));
  spokeApps.Stop (Seconds (10.0));

  //
  // Because we are evil, we also add OnOff applications to send TCP to the hub 
  // from the fill devices on each CSMA link.  The first nFill nodes in the 
  // fillNodes container are on the CSMA network talking to the zeroth device
  // on the hub node.  The next nFill nodes are on the CSMA network talking to
  // the first device on the hub node, etc.  So the ith fillNode is associated
  // with the hub address found on the (i / nFill)th device on the hub node.
  //
  ApplicationContainer fillApps;

  for (uint32_t i = 0; i < fillNodes.GetN (); ++i)
    {
      AddressValue remoteAddress (InetSocketAddress (star.GetHubIpv4Address (i / nFill), port));
      onOffHelper.SetAttribute ("Remote", remoteAddress);
      fillApps.Add (onOffHelper.Install (fillNodes.Get (i)));
    }

  fillApps.Start (Seconds (1.0));
  fillApps.Stop (Seconds (10.0));

  //
  // Turn on global static routing so we can actually be routed across the star.
  //
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  // Trace receptions
  Config::ConnectWithoutContext ("/NodeList/0/ApplicationList/*/$ns3::PacketSink/Rx", 
                                 MakeCallback (&CsmaStarTestCase::SinkRx, this));

  Simulator::Run ();
  Simulator::Destroy ();

  // The hub node should have received 10 packets from the nFill + 1
  // nodes on each spoke.
  NS_TEST_ASSERT_MSG_EQ (m_count, 10 * ( nSpokes * (nFill + 1)), "Hub node did not receive the proper number of packets");
}

/**
 * \ingroup system-tests-csma
 * 
 * \brief CSMA TestSuite.
 */
class CsmaSystemTestSuite : public TestSuite
{
public:
  CsmaSystemTestSuite ();
};

CsmaSystemTestSuite::CsmaSystemTestSuite ()
  : TestSuite ("csma-system", UNIT)
{
  AddTestCase (new CsmaBridgeTestCase, TestCase::QUICK);
  AddTestCase (new CsmaBroadcastTestCase, TestCase::QUICK);
  AddTestCase (new CsmaMulticastTestCase, TestCase::QUICK);
  AddTestCase (new CsmaOneSubnetTestCase, TestCase::QUICK);
  AddTestCase (new CsmaPacketSocketTestCase, TestCase::QUICK);
  AddTestCase (new CsmaPingTestCase, TestCase::QUICK);
  AddTestCase (new CsmaRawIpSocketTestCase, TestCase::QUICK);
  AddTestCase (new CsmaStarTestCase, TestCase::QUICK);
}

/// Do not forget to allocate an instance of this TestSuite
static CsmaSystemTestSuite csmaSystemTestSuite;
