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

#include <iomanip>

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
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/packet-sink-helper.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/node-container.h"
#include "ns3/simulator.h"
#include "ns3/error-model.h"
#include "ns3/pointer.h"
#include "ns3tcp-socket-writer.h"
#include "ns3/tcp-westwood.h"
#include "ns3/tcp-header.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("Ns3TcpLossTest");

// The below boolean constants should only be changed to 'true'
// during test debugging (i.e. do not commit the value 'true')

// set to 'true' to have the test suite overwrite the response vectors
// stored in the test directory.  This should only be done if you are
// convinced through other means (e.g. pcap tracing or logging) that the
// revised vectors are the correct ones.  In other words, don't simply
// enable this to true to clear a failing test without looking at the
// results closely.
const bool WRITE_VECTORS = false;           //!< Set to true to write response vectors.
const bool WRITE_PCAP = false;              //!< Set to true to write out pcap.
const bool WRITE_LOGGING = false;           //!< Set to true to write logging.
const uint32_t PCAP_LINK_TYPE = 1187373557; //!< Some large random number -- we use to verify data was written by this program.
const uint32_t PCAP_SNAPLEN   = 64;         //!< Don't bother to save much data.

/**
 * \ingroup system-tests-tcp
 * 
 * \brief Tests of TCP implementation loss behavior.
 */
class Ns3TcpLossTestCase : public TestCase
{
public:
  Ns3TcpLossTestCase ();

  /**
   * Constructor.
   * 
   * \param tcpModel The TCP model name.
   * \param testCase Testcase number.
   */
  Ns3TcpLossTestCase (std::string tcpModel, uint32_t testCase);
  virtual ~Ns3TcpLossTestCase ()
  {
  }

private:
  virtual void DoSetup (void);
  virtual void DoRun (void);
  virtual void DoTeardown (void);

  Ptr<OutputStreamWrapper> m_osw; //!< The output stream.
  std::string m_pcapFilename;     //!< The PCAP filename.
  PcapFile m_pcapFile;            //!< The PCAP ffile.
  uint32_t m_testCase;            //!< Testcase number.
  uint32_t m_totalTxBytes;        //!< Total number of bytes to send.
  uint32_t m_currentTxBytes;      //!< Current number of bytes sent.
  bool m_writeVectors;            //!< True if response vectors have to be written (and not read).
  bool m_writeResults;            //!< True if write PCAP files.
  bool m_writeLogging;            //!< True if write logging.
  bool m_needToClose;             //!< Check if the sending socket need to be closed.
  std::string m_tcpModel;         //!< The TCP model name.

  /**
   * Check that the transmitted packets are consitent with the trace.
   * This callback is hooked to ns3::Ipv4L3Protocol/Tx.
   * 
   * \param context The callback context (unused).
   * \param packet The transmitted packet.
   * \param ipv4 The IPv4 object that did send the packet (unused).
   * \param interface The IPv4 interface that did send the packet (unused).
   */
  void Ipv4L3Tx (std::string context, Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface);
  /**
   * CWND trace.
   * 
   * \param oldval The old value.
   * \param newval The new value.
   */
  void CwndTracer (uint32_t oldval, uint32_t newval);
  /**
   * Write to the socket until the buffer is full.
   * 
   * \param localSocket The output socket.
   * \param txSpace The space left on the socket (unused).
   */
  void WriteUntilBufferFull (Ptr<Socket> localSocket, uint32_t txSpace);
  /**
   * Start transmitting a TCP flow.
   * 
   * \param localSocket The sending socket.
   * \param servAddress The IPv4 address of the server (i.e., the destination address).
   * \param servPort The TCP port of the server (i.e., the destination port).
   */
  void StartFlow (Ptr<Socket> localSocket, Ipv4Address servAddress, uint16_t servPort);

};

Ns3TcpLossTestCase::Ns3TcpLossTestCase ()
  : TestCase ("Check the operation of the TCP state machine for several cases"),
    m_testCase (0),
    m_totalTxBytes (200000),
    m_currentTxBytes (0),
    m_writeVectors (WRITE_VECTORS),
    m_writeResults (WRITE_PCAP),
    m_writeLogging (WRITE_LOGGING),
    m_needToClose (true),
    m_tcpModel ("ns3::TcpWestwood")
{
}

Ns3TcpLossTestCase::Ns3TcpLossTestCase (std::string tcpModel, uint32_t testCase)
  : TestCase ("Check the behaviour of TCP upon packet losses"),
    m_testCase (testCase),
    m_totalTxBytes (200000),
    m_currentTxBytes (0),
    m_writeVectors (WRITE_VECTORS),
    m_writeResults (WRITE_PCAP),
    m_writeLogging (WRITE_LOGGING),
    m_needToClose (true),
    m_tcpModel (tcpModel)
{
}

void
Ns3TcpLossTestCase::DoSetup (void)
{
  // This test was written before SACK was added to ns-3
  Config::SetDefault ("ns3::TcpSocketBase::Sack", BooleanValue (false));
  // This test was written with initial window of 1 segment
  Config::SetDefault ("ns3::TcpSocket::InitialCwnd", UintegerValue (1));
  // This test was written with the TCP Classic Recovery algorithm
  Config::SetDefault ("ns3::TcpL4Protocol::RecoveryType", TypeIdValue (TypeId (ns3::TcpClassicRecovery::GetTypeId ())));

  //
  // We expect there to be a file called ns3tcp-state-response-vectors.pcap in
  // the data directory
  //
  std::ostringstream oss;
  oss << "ns3tcp-loss-" << m_tcpModel << m_testCase << "-response-vectors.pcap";
  m_pcapFilename = CreateDataDirFilename (oss.str ());

  if (m_writeVectors)
    {
      m_pcapFile.Open (m_pcapFilename, std::ios::out | std::ios::binary);
      m_pcapFile.Init (PCAP_LINK_TYPE, PCAP_SNAPLEN);
    }
  else
    {
      m_pcapFile.Open (m_pcapFilename, std::ios::in | std::ios::binary);
      NS_ABORT_MSG_UNLESS (m_pcapFile.GetDataLinkType () == PCAP_LINK_TYPE,
                           "Wrong response vectors in directory: opening " <<
                           m_pcapFilename);
    }
}

void
Ns3TcpLossTestCase::DoTeardown (void)
{
  m_pcapFile.Close ();
}

void
Ns3TcpLossTestCase::Ipv4L3Tx (std::string, Ptr<const Packet> packet, Ptr<Ipv4>, uint32_t)
{
  //
  // We're not testing IP so remove and toss the header.  In order to do this,
  // though, we need to copy the packet since we have a const version.
  //
  Ptr<Packet> received = packet->Copy ();
  Ipv4Header ipHeader;
  received->RemoveHeader (ipHeader);

  //
  // What is left is the TCP header and any data that may be sent.  We aren't
  // sending any TCP data, so we expect what remains is only TCP header, which
  // is a small thing to save.
  //
  if (m_writeVectors)
    {
      //
      // Save the TCP under test response for later testing.
      //
      Time tNow = Simulator::Now ();
      int64_t tMicroSeconds = tNow.GetMicroSeconds ();


      m_pcapFile.Write (uint32_t (tMicroSeconds / 1000000),
                        uint32_t (tMicroSeconds % 1000000),
                        received);
    }
  else
    {
      //
      // Read the TCP under test expected response from the expected vector
      // file and see if it still does the right thing.
      //
      uint8_t expectedBuffer[PCAP_SNAPLEN];
      uint32_t tsSec, tsUsec, inclLen, origLen, readLen;
      m_pcapFile.Read (expectedBuffer, sizeof(expectedBuffer), tsSec, tsUsec, inclLen, origLen, readLen);

      NS_LOG_INFO ("read " << readLen << " bytes");

      uint8_t *actual = new uint8_t[readLen];
      received->CopyData (actual, readLen);

      int result = memcmp (actual, expectedBuffer, readLen);

      TcpHeader expectedHeader, receivedHeader;
      Ptr<Packet> expected = Create<Packet> (expectedBuffer, readLen);

      expected->RemoveHeader (expectedHeader);
      received->RemoveHeader (receivedHeader);

      NS_LOG_DEBUG ("Expected " << expectedHeader << " received: " << receivedHeader);

      delete [] actual;

      //
      // Avoid streams of errors -- only report the first.
      //
      if (IsStatusSuccess ())
        {
          NS_TEST_EXPECT_MSG_EQ (result, 0, "Expected data comparison error: " << m_tcpModel << "-" << m_testCase);
        }
    }
}

void
Ns3TcpLossTestCase::CwndTracer (uint32_t oldval, uint32_t newval)
{
  if (m_writeLogging)
    {
      *(m_osw->GetStream ()) << "Moving cwnd from " << oldval << " to " << newval
                             << " at time " << Simulator::Now ().GetSeconds ()
                             << " seconds" << std::endl;
    }
}

////////////////////////////////////////////////////////////////////
// Implementing an "application" to send bytes over a TCP connection
void
Ns3TcpLossTestCase::WriteUntilBufferFull (Ptr<Socket> localSocket, uint32_t)
{
  while (m_currentTxBytes < m_totalTxBytes)
    {
      uint32_t left = m_totalTxBytes - m_currentTxBytes;
      uint32_t dataOffset = m_currentTxBytes % 1040;
      uint32_t toWrite = 1040 - dataOffset;
      uint32_t txAvail = localSocket->GetTxAvailable ();
      toWrite = std::min (toWrite, left);
      toWrite = std::min (toWrite, txAvail);
      if (txAvail == 0)
        {
          return;
        }
      if (m_writeLogging)
        {
          std::clog << "Submitting " << toWrite
                    << " bytes to TCP socket" << std::endl;
        }
      int amountSent = localSocket->Send (0, toWrite, 0);
      NS_ASSERT (amountSent > 0);  // Given GetTxAvailable() non-zero, amountSent should not be zero
      m_currentTxBytes += amountSent;
    }
  if (m_needToClose)
    {
      if (m_writeLogging)
        {
          std::clog << "Close socket at "
                    <<  Simulator::Now ().GetSeconds () << std::endl;
        }
      localSocket->Close ();
      m_needToClose = false;
    }
}

void
Ns3TcpLossTestCase::StartFlow (Ptr<Socket> localSocket,
                               Ipv4Address servAddress,
                               uint16_t servPort)
{
  if (m_writeLogging)
    {
      std::clog << "Starting flow at time "
                <<  Simulator::Now ().GetSeconds () << std::endl;
    }
  localSocket->Connect (InetSocketAddress (servAddress, servPort)); // connect

  // tell the tcp implementation to call WriteUntilBufferFull again
  // if we blocked and new tx buffer space becomes available
  localSocket->SetSendCallback (MakeCallback
                                  (&Ns3TcpLossTestCase::WriteUntilBufferFull,
                                  this));
  WriteUntilBufferFull (localSocket, localSocket->GetTxAvailable ());
}

void
Ns3TcpLossTestCase::DoRun (void)
{
  // Network topology
  //
  //           8Mb/s, 0.1ms       0.8Mb/s, 100ms
  //       s1-----------------r1-----------------k1
  //
  // Example corresponding to simulations in the paper "Simulation-based
  // Comparisons of Tahoe, Reno, and SACK TCP

  Config::SetDefault ("ns3::TcpSocketBase::Sack", BooleanValue (false));

  std::ostringstream tcpModel;
  tcpModel << "ns3::Tcp" << m_tcpModel;
  if (m_tcpModel.compare ("WestwoodPlus") == 0)
    {
      Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                          TypeIdValue (TcpWestwood::GetTypeId ()));
      Config::SetDefault ("ns3::TcpWestwood::ProtocolType",
                          EnumValue (TcpWestwood::WESTWOODPLUS));
    }
  else
    {
      Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                          StringValue (tcpModel.str ()));
    }

  Config::SetDefault ("ns3::TcpSocket::SegmentSize", UintegerValue (1000));
  Config::SetDefault ("ns3::TcpSocket::DelAckCount", UintegerValue (1));
  Config::SetDefault ("ns3::TcpSocketBase::Timestamp", BooleanValue (false));

  if (m_writeLogging)
    {
      LogComponentEnableAll (LOG_PREFIX_FUNC);
      LogComponentEnable ("Ns3TcpLossTest", LOG_LEVEL_ALL);
      LogComponentEnable ("ErrorModel", LOG_LEVEL_DEBUG);
      LogComponentEnable ("TcpWestwood", LOG_LEVEL_ALL);
      LogComponentEnable ("TcpCongestionOps", LOG_LEVEL_INFO);
      LogComponentEnable ("TcpSocketBase", LOG_LEVEL_INFO);
    }

  ////////////////////////////////////////////////////////
  // Topology construction
  //

  // Create three nodes: s1, r1, and k1
  NodeContainer s1r1;
  s1r1.Create (2);

  NodeContainer r1k1;
  r1k1.Add (s1r1.Get (1));
  r1k1.Create (1);

  // Set up TCP/IP stack to all nodes (and create loopback device at device 0)
  InternetStackHelper internet;
  internet.InstallAll ();

  // Connect the nodes
  PointToPointHelper p2p;
  p2p.SetDeviceAttribute ("DataRate", DataRateValue (DataRate (8000000)));
  p2p.SetChannelAttribute ("Delay", TimeValue (Seconds (0.0001)));
  NetDeviceContainer dev0 = p2p.Install (s1r1);
  p2p.SetDeviceAttribute ("DataRate", DataRateValue (DataRate (800000)));
  p2p.SetChannelAttribute ("Delay", TimeValue (Seconds (0.1)));
  NetDeviceContainer dev1 = p2p.Install (r1k1);

  // Add IP addresses to each network interfaces
  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.3.0", "255.255.255.0");
  ipv4.Assign (dev0);
  ipv4.SetBase ("10.1.2.0", "255.255.255.0");
  Ipv4InterfaceContainer ipInterfs = ipv4.Assign (dev1);

  // Set up routes to all nodes
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  ////////////////////////////////////////////////////////
  // Send 20000 (totalTxBytes) bytes from node s1 to node k1
  //

  // Create a packet sink to receive packets on node k1
  uint16_t servPort = 50000; // Destination port number
  PacketSinkHelper sink ("ns3::TcpSocketFactory",
                         InetSocketAddress (Ipv4Address::GetAny (), servPort));
  ApplicationContainer apps = sink.Install (r1k1.Get (1));
  apps.Start (Seconds (0.0));
  apps.Stop (Seconds (100.0));

  // Create a data source to send packets on node s0.
  // Instead of full application, here use the socket directly by
  // registering callbacks in function StarFlow().
  Ptr<Socket> localSocket = Socket::CreateSocket (s1r1.Get (0), TcpSocketFactory::GetTypeId ());
  localSocket->Bind ();
  Simulator::ScheduleNow (&Ns3TcpLossTestCase::StartFlow,
                          this,
                          localSocket,
                          ipInterfs.GetAddress (1),
                          servPort);

  Config::Connect ("/NodeList/0/$ns3::Ipv4L3Protocol/Tx",
                   MakeCallback (&Ns3TcpLossTestCase::Ipv4L3Tx, this));

  Config::ConnectWithoutContext
    ("/NodeList/0/$ns3::TcpL4Protocol/SocketList/0/CongestionWindow",
    MakeCallback (&Ns3TcpLossTestCase::CwndTracer, this));

  ////////////////////////////////////////////////////////
  // Set up loss model at node k1
  //
  std::list<uint32_t> sampleList;
  switch (m_testCase)
    {
    case 0:
      break;
    case 1:
      // Force a loss for 15th data packet. TCP cwnd will be at 14 segments
      // (14000 bytes) when duplicate acknowledgments start to come.
      sampleList.push_back (16);
      break;
    case 2:
      sampleList.push_back (16);
      sampleList.push_back (17);
      break;
    case 3:
      sampleList.push_back (16);
      sampleList.push_back (17);
      sampleList.push_back (18);
      break;
    case 4:
      sampleList.push_back (16);
      sampleList.push_back (17);
      sampleList.push_back (18);
      sampleList.push_back (19);
      break;
    default:
      NS_FATAL_ERROR ("Program fatal error: loss value " << m_testCase << " not supported.");
      break;
    }

  Ptr<ReceiveListErrorModel> pem = CreateObject<ReceiveListErrorModel> ();
  pem->SetList (sampleList);
  dev1.Get (1)->SetAttribute ("ReceiveErrorModel", PointerValue (pem));

  // One can toggle the comment for the following line on or off to see the
  // effects of finite send buffer modelling.  One can also change the size of
  // that buffer.
  // localSocket->SetAttribute("SndBufSize", UintegerValue(4096));

  std::ostringstream oss;
  oss << "tcp-loss-" << m_tcpModel << m_testCase << "-test-case";
  if (m_writeResults)
    {
      p2p.EnablePcapAll (oss.str ());
      p2p.EnableAsciiAll (oss.str ());
    }

  std::ostringstream oss2;
  oss2 << "src/test/ns3tcp/Tcp" << m_tcpModel << "." << m_testCase << ".log";
  AsciiTraceHelper ascii;
  if (m_writeLogging)
    {
      m_osw = ascii.CreateFileStream (oss2.str ());
      *(m_osw->GetStream ()) << std::setprecision (9) << std::fixed;
      p2p.EnableAsciiAll (m_osw);
    }

  // Finally, set up the simulator to run.  The 1000 second hard limit is a
  // failsafe in case some change above causes the simulation to never end
  Simulator::Stop (Seconds (1000));
  Simulator::Run ();
  Simulator::Destroy ();
}

/**
 * \ingroup system-tests-tcp
 * 
 * TCP implementation loss behavior TestSuite.
 */
class Ns3TcpLossTestSuite : public TestSuite
{
public:
  Ns3TcpLossTestSuite ();
};

Ns3TcpLossTestSuite::Ns3TcpLossTestSuite ()
  : TestSuite ("ns3-tcp-loss", SYSTEM)
{
  // We can't use NS_TEST_SOURCEDIR variable here because we use subdirectories
  SetDataDir ("src/test/ns3tcp/response-vectors");
  Packet::EnablePrinting ();  // Enable packet metadata for all test cases

  AddTestCase (new Ns3TcpLossTestCase ("NewReno", 0), TestCase::QUICK);
  AddTestCase (new Ns3TcpLossTestCase ("NewReno", 1), TestCase::QUICK);
  AddTestCase (new Ns3TcpLossTestCase ("NewReno", 2), TestCase::QUICK);
  AddTestCase (new Ns3TcpLossTestCase ("NewReno", 3), TestCase::QUICK);
  AddTestCase (new Ns3TcpLossTestCase ("NewReno", 4), TestCase::QUICK);

  AddTestCase (new Ns3TcpLossTestCase ("Westwood", 0), TestCase::QUICK);
  AddTestCase (new Ns3TcpLossTestCase ("Westwood", 1), TestCase::QUICK);
  AddTestCase (new Ns3TcpLossTestCase ("Westwood", 2), TestCase::QUICK);
  AddTestCase (new Ns3TcpLossTestCase ("Westwood", 3), TestCase::QUICK);
  AddTestCase (new Ns3TcpLossTestCase ("Westwood", 4), TestCase::QUICK);

  AddTestCase (new Ns3TcpLossTestCase ("WestwoodPlus", 0), TestCase::QUICK);
  AddTestCase (new Ns3TcpLossTestCase ("WestwoodPlus", 1), TestCase::QUICK);
  AddTestCase (new Ns3TcpLossTestCase ("WestwoodPlus", 2), TestCase::QUICK);
  AddTestCase (new Ns3TcpLossTestCase ("WestwoodPlus", 3), TestCase::QUICK);
  AddTestCase (new Ns3TcpLossTestCase ("WestwoodPlus", 4), TestCase::QUICK);

}

/// Do not forget to allocate an instance of this TestSuite.
static Ns3TcpLossTestSuite g_ns3TcpLossTestSuite;

