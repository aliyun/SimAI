/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 University of Washington
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
 */

#include "ns3/log.h"
#include "ns3/abort.h"
#include "ns3/test.h"
#include "ns3/pcap-file.h"
#include "ns3/config.h"
#include "ns3/string.h"
#include "ns3/uinteger.h"
#include "ns3/data-rate.h"
#include "ns3/inet-socket-address.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/csma-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/packet-sink-helper.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/node-container.h"
#include "ns3/simulator.h"
#include "ns3tcp-socket-writer.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("Ns3SocketTest");

/**
 * \ingroup system-tests-tcp
 * 
 * \brief Tests of TCP implementations from the application/socket perspective 
 * using point-to-point links.
 */
class Ns3TcpSocketTestCaseP2P : public TestCase
{
public:
  Ns3TcpSocketTestCaseP2P ();
  virtual ~Ns3TcpSocketTestCaseP2P () {}

private:
  virtual void DoRun (void);
  bool m_writeResults;  //!< True if write PCAP files.

  /**
   * Receive a TCP packet.
   * \param path The callback context (unused).
   * \param p The received packet.
   * \param address The sender's address (unused).
   */
  void SinkRx (std::string path, Ptr<const Packet> p, const Address &address);

  TestVectors<uint32_t> m_inputs;     //!< Sent packets test vector.
  TestVectors<uint32_t> m_responses;  //!< Received packets test vector.
};

Ns3TcpSocketTestCaseP2P::Ns3TcpSocketTestCaseP2P ()
  : TestCase ("Check that ns-3 TCP successfully transfers an application data write of various sizes (point-to-point)"),
    m_writeResults (false)
{
}

void 
Ns3TcpSocketTestCaseP2P::SinkRx (std::string path, Ptr<const Packet> p, const Address &address)
{
  m_responses.Add (p->GetSize ());
}

void
Ns3TcpSocketTestCaseP2P::DoRun (void)
{
  uint16_t sinkPort = 50000;
  double sinkStopTime = 40;  // sec; will trigger Socket::Close
  double writerStopTime = 30;  // sec; will trigger Socket::Close
  double simStopTime = 60;  // sec
  Time sinkStopTimeObj = Seconds (sinkStopTime);
  Time writerStopTimeObj = Seconds (writerStopTime);
  Time simStopTimeObj= Seconds (simStopTime);

  Ptr<Node> n0 = CreateObject<Node> ();
  Ptr<Node> n1 = CreateObject<Node> ();

  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

  NetDeviceContainer devices;
  devices = pointToPoint.Install (n0, n1);

  InternetStackHelper internet;
  internet.InstallAll ();

  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.252");
  Ipv4InterfaceContainer ifContainer = address.Assign (devices);

  Ptr<SocketWriter> socketWriter = CreateObject<SocketWriter> ();
  Address sinkAddress (InetSocketAddress (ifContainer.GetAddress (1), sinkPort));
  socketWriter->Setup (n0, sinkAddress);
  n0->AddApplication (socketWriter);
  socketWriter->SetStartTime (Seconds (0.));
  socketWriter->SetStopTime (writerStopTimeObj);

  PacketSinkHelper sink ("ns3::TcpSocketFactory",
                         InetSocketAddress (Ipv4Address::GetAny (), sinkPort));
  ApplicationContainer apps = sink.Install (n1);
  // Start the sink application at time zero, and stop it at sinkStopTime
  apps.Start (Seconds (0.0));
  apps.Stop (sinkStopTimeObj);

  Config::Connect ("/NodeList/*/ApplicationList/*/$ns3::PacketSink/Rx",
                   MakeCallback (&Ns3TcpSocketTestCaseP2P::SinkRx, this));

  Simulator::Schedule (Seconds (2), &SocketWriter::Connect, socketWriter);
  // Send 1, 10, 100, 1000 bytes
  Simulator::Schedule (Seconds (10), &SocketWriter::Write, socketWriter, 1);
  m_inputs.Add (1);
  Simulator::Schedule (Seconds (12), &SocketWriter::Write, socketWriter, 10);
  m_inputs.Add (10);
  Simulator::Schedule (Seconds (14), &SocketWriter::Write, socketWriter, 100);
  m_inputs.Add (100);
  Simulator::Schedule (Seconds (16), &SocketWriter::Write, socketWriter, 1000);
  m_inputs.Add (536);
  m_inputs.Add (464);  // ns-3 TCP default segment size of 536
  Simulator::Schedule (writerStopTimeObj, &SocketWriter::Close, socketWriter);

  if (m_writeResults)
    {
      pointToPoint.EnablePcapAll ("tcp-socket-test-case-1");
    }

  Simulator::Stop (simStopTimeObj);
  Simulator::Run ();
  Simulator::Destroy ();

  // Compare inputs and outputs
  NS_TEST_ASSERT_MSG_EQ (m_inputs.GetN (), m_responses.GetN (), "Incorrect number of expected receive events");
  for (uint32_t i = 0; i < m_responses.GetN (); i++)
    {
      uint32_t in = m_inputs.Get (i);
      uint32_t out = m_responses.Get (i);
      NS_TEST_ASSERT_MSG_EQ (in, out, "Mismatch:  expected " << in << " bytes, got " << out << " bytes");
    }
}

/**
 * \ingroup system-tests-tcp
 * 
 * \brief Tests of TCP implementations from the application/socket perspective 
 * using CSMA links.
 */
class Ns3TcpSocketTestCaseCsma : public TestCase
{
public:
  Ns3TcpSocketTestCaseCsma ();
  virtual ~Ns3TcpSocketTestCaseCsma () {}

private:
  virtual void DoRun (void);
  bool m_writeResults;  //!< True if write PCAP files.

  /**
   * Receive a TCP packet.
   * \param path The callback context (unused).
   * \param p The received packet.
   * \param address The sender's address (unused).
   */
  void SinkRx (std::string path, Ptr<const Packet> p, const Address &address);

  TestVectors<uint32_t> m_inputs;     //!< Sent packets test vector.
  TestVectors<uint32_t> m_responses;  //!< Received packets test vector.
};

Ns3TcpSocketTestCaseCsma::Ns3TcpSocketTestCaseCsma ()
  : TestCase ("Check to see that ns-3 TCP successfully transfers an application data write of various sizes (CSMA)"),
    m_writeResults (false)
{
}

void 
Ns3TcpSocketTestCaseCsma::SinkRx (std::string path, Ptr<const Packet> p, const Address &address)
{
  m_responses.Add (p->GetSize ());
}

void
Ns3TcpSocketTestCaseCsma::DoRun (void)
{
  uint16_t sinkPort = 50000;
  double sinkStopTime = 40;  // sec; will trigger Socket::Close
  double writerStopTime = 30;  // sec; will trigger Socket::Close
  double simStopTime = 60;  // sec
  Time sinkStopTimeObj = Seconds (sinkStopTime);
  Time writerStopTimeObj = Seconds (writerStopTime);
  Time simStopTimeObj= Seconds (simStopTime);

  Config::SetDefault ("ns3::TcpSocket::SegmentSize", UintegerValue (1000));

  NodeContainer nodes;
  nodes.Create (2);
  Ptr<Node> n0 = nodes.Get (0);
  Ptr<Node> n1 = nodes.Get (1);

  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", StringValue ("5Mbps"));
  csma.SetChannelAttribute ("Delay", StringValue ("2ms"));

  NetDeviceContainer devices;
  devices = csma.Install (nodes);

  InternetStackHelper internet;
  internet.InstallAll ();

  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.252");
  Ipv4InterfaceContainer ifContainer = address.Assign (devices);

  Ptr<SocketWriter> socketWriter = CreateObject<SocketWriter> ();
  Address sinkAddress (InetSocketAddress (ifContainer.GetAddress (1), sinkPort));
  socketWriter->Setup (n0, sinkAddress);
  n0->AddApplication (socketWriter);
  socketWriter->SetStartTime (Seconds (0.));
  socketWriter->SetStopTime (writerStopTimeObj);

  PacketSinkHelper sink ("ns3::TcpSocketFactory",
                         InetSocketAddress (Ipv4Address::GetAny (), sinkPort));
  ApplicationContainer apps = sink.Install (n1);
  // Start the sink application at time zero, and stop it at sinkStopTime
  apps.Start (Seconds (0.0));
  apps.Stop (sinkStopTimeObj);

  Config::Connect ("/NodeList/*/ApplicationList/*/$ns3::PacketSink/Rx",
                   MakeCallback (&Ns3TcpSocketTestCaseCsma::SinkRx, this));

  Simulator::Schedule (Seconds (2), &SocketWriter::Connect, socketWriter);
  // Send 1, 10, 100, 1000 bytes
  // PointToPoint default MTU is 576 bytes, which leaves 536 bytes for TCP
  Simulator::Schedule (Seconds (10), &SocketWriter::Write, socketWriter, 1);
  m_inputs.Add (1);
  Simulator::Schedule (Seconds (12), &SocketWriter::Write, socketWriter, 10);
  m_inputs.Add (10);
  Simulator::Schedule (Seconds (14), &SocketWriter::Write, socketWriter, 100);
  m_inputs.Add (100);
  Simulator::Schedule (Seconds (16), &SocketWriter::Write, socketWriter, 1000);
  m_inputs.Add (1000);
  // Next packet will fragment
  Simulator::Schedule (Seconds (16), &SocketWriter::Write, socketWriter, 1001);
  m_inputs.Add (1000);
  m_inputs.Add (1);
  Simulator::Schedule (writerStopTimeObj, &SocketWriter::Close, socketWriter);

  if (m_writeResults)
    {
      csma.EnablePcapAll ("tcp-socket-test-case-2", false);
    }
  Simulator::Stop (simStopTimeObj);
  Simulator::Run ();
  Simulator::Destroy ();

  // Compare inputs and outputs
  NS_TEST_ASSERT_MSG_EQ (m_inputs.GetN (), m_responses.GetN (), "Incorrect number of expected receive events");
  for (uint32_t i = 0; i < m_responses.GetN (); i++)
    {
      uint32_t in = m_inputs.Get (i);
      uint32_t out = m_responses.Get (i);
      NS_TEST_ASSERT_MSG_EQ (in, out, "Mismatch:  expected " << in << " bytes, got " << out << " bytes");
    }
}

/**
 * \ingroup system-tests-tcp
 * 
 * TCP implementations from the application/socket perspective TestSuite.
 */
class Ns3TcpSocketTestSuite : public TestSuite
{
public:
  Ns3TcpSocketTestSuite ();
};

Ns3TcpSocketTestSuite::Ns3TcpSocketTestSuite ()
  : TestSuite ("ns3-tcp-socket", SYSTEM)
{
  AddTestCase (new Ns3TcpSocketTestCaseP2P, TestCase::QUICK);
  AddTestCase (new Ns3TcpSocketTestCaseCsma, TestCase::QUICK);
}

/// Do not forget to allocate an instance of this TestSuite.
static Ns3TcpSocketTestSuite g_ns3TcpSocketTestSuite;
