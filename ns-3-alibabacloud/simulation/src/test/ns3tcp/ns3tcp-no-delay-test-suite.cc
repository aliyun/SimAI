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

#include "ns3/log.h"
#include "ns3/abort.h"
#include "ns3/test.h"
#include "ns3/pcap-file.h"
#include "ns3/config.h"
#include "ns3/string.h"
#include "ns3/uinteger.h"
#include "ns3/boolean.h"
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

NS_LOG_COMPONENT_DEFINE ("Ns3TcpNoDelayTest");

/**
 * \ingroup system-tests-tcp
 * 
 * \brief Tests of Nagle's algorithm and the TCP no delay option.
 */
class Ns3TcpNoDelayTestCase : public TestCase
{
public:
  /**
   * Constructor.
   * 
   * \param noDelay Enable or disable TCP no delay option.
   */
  Ns3TcpNoDelayTestCase (bool noDelay);
  virtual ~Ns3TcpNoDelayTestCase () {}

private:
  virtual void DoRun (void);
  bool m_noDelay;       //!< Enable or disable TCP no delay option.
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

Ns3TcpNoDelayTestCase::Ns3TcpNoDelayTestCase (bool noDelay)
  : TestCase ("Check that ns-3 TCP Nagle's algorithm works correctly and that we can turn it off."),
    m_noDelay (noDelay),
    m_writeResults (false)
{
}

void 
Ns3TcpNoDelayTestCase::SinkRx (std::string, Ptr<const Packet> p, const Address &)
{
  m_responses.Add (p->GetSize ());
}

void
Ns3TcpNoDelayTestCase::DoRun (void)
{
  uint16_t sinkPort = 50000;
  double sinkStopTime = 8;  // sec; will trigger Socket::Close
  double writerStopTime = 5;  // sec; will trigger Socket::Close
  double simStopTime = 10;  // sec
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
                   MakeCallback (&Ns3TcpNoDelayTestCase::SinkRx, this));

  // Enable or disable TCP no delay option
  Config::SetDefault ("ns3::TcpSocket::TcpNoDelay", BooleanValue (m_noDelay));
  // This test was written with initial window of 1 segment
  Config::SetDefault ("ns3::TcpSocket::InitialCwnd", UintegerValue (1));

  // Connect the socket writer
  Simulator::Schedule (Seconds (1), &SocketWriter::Connect, socketWriter);

  // Write 5 packets to get some bytes in flight and some acks going
  Simulator::Schedule (Seconds (2), &SocketWriter::Write, socketWriter, 2680);
  m_inputs.Add (536);
  m_inputs.Add (536);
  m_inputs.Add (536);
  m_inputs.Add (536);
  m_inputs.Add (536);

  // Write one byte after 10 ms to ensure that some data is outstanding 
  // and the window is big enough
  Simulator::Schedule (Seconds (2.010), &SocketWriter::Write, socketWriter, 1);

  // If Nagle is not enabled, i.e. no delay is on, add an input for a 1-byte 
  // packet to be received
  if (m_noDelay)
    {
      m_inputs.Add (1);
    }

  // One ms later, write 535 bytes, i.e. one segment size - 1
  Simulator::Schedule (Seconds (2.012), &SocketWriter::Write, socketWriter, 535);

  // If Nagle is not enabled, add an input for a 535 byte packet,
  // otherwise, we should get a single "full" packet of 536 bytes
  if (m_noDelay)
    {
      m_inputs.Add (535);
    }
  else
    {
      m_inputs.Add (536);
    }

  // Close down the socket
  Simulator::Schedule (writerStopTimeObj, &SocketWriter::Close, socketWriter);

  if (m_writeResults)
    {
      std::ostringstream oss;
      if (m_noDelay)
        {
          oss << "tcp-no-delay-on-test-case";
          pointToPoint.EnablePcapAll (oss.str ());
        }
      else
        {
          oss << "tcp-no-delay-off-test-case";
          pointToPoint.EnablePcapAll (oss.str ());
        }
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
 * TCP Nagle's algorithm and the TCP no delay option TestSuite.
 */
class Ns3TcpNoDelayTestSuite : public TestSuite
{
public:
  Ns3TcpNoDelayTestSuite ();
};

Ns3TcpNoDelayTestSuite::Ns3TcpNoDelayTestSuite ()
  : TestSuite ("ns3-tcp-no-delay", SYSTEM)
{
  AddTestCase (new Ns3TcpNoDelayTestCase (true), TestCase::QUICK);
  AddTestCase (new Ns3TcpNoDelayTestCase (false), TestCase::QUICK);
}

/// Do not forget to allocate an instance of this TestSuite.
static Ns3TcpNoDelayTestSuite g_ns3TcpNoDelayTestSuite;
