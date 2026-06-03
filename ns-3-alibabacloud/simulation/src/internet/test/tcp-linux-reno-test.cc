/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 Apoorva Bhargava <apoorvabhargava13@gmail.com>
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
 */
#include "ns3/log.h"
#include "ns3/test.h"
#include "ns3/simple-channel.h"
#include "ns3/node.h"
#include "ns3/config.h"
#include "ns3/tcp-header.h"
#include "tcp-general-test.h"
#include "ns3/tcp-linux-reno.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpLinuxRenoTest");

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * This unit test checks that the slow start and congestion avoidance
 * behavior matches Linux behavior as follows:
 * 1) in both slow start and congestion avoidance phases, presence or
 *    absence of delayed acks does not alter the window growth
 * 2) in congestion avoidance phase, the arithmetic for counting the number
 *    of segments acked and deciding when to increment the congestion window
 *    (i.e. following the Linux function tcp_cong_avoid_ai()) is followed.
 * Different segment sizes (524 bytes and 1500 bytes) are tested.
 *
 * This is the Slow Start test.
 */
class
TcpLinuxRenoSSTest : public TcpGeneralTest
{
public:
  /**
   * \brief Constructor.
   * \param segmentSize Segment size.
   * \param packetSize Size of the packets.
   * \param packets Number of packets.
   * \param initialCwnd Initial congestion window
   * \param delayedAck Delayed Acknowledgement
   * \param expectedCwnd Expected value of m_cWnd
   * \param congControl Type of congestion control.
   * \param desc The test description.
   */
  TcpLinuxRenoSSTest (uint32_t segmentSize, uint32_t packetSize, uint32_t packets,
                      uint32_t initialCwnd, uint32_t delayedAck,
                      uint32_t expectedCwnd,
                      TypeId& congControl, const std::string &desc);

protected:
  virtual void CWndTrace (uint32_t oldValue, uint32_t newValue);
  void QueueDrop (SocketWho who);
  void PhyDrop (SocketWho who);

  virtual void ConfigureEnvironment ();
  virtual void ConfigureProperties ();
  virtual void DoTeardown ();

  bool   m_initial;             //!< First cycle flag.

private:
  virtual void Tx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who);
  virtual void Rx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who);
  uint32_t m_segmentSize;       //!< Segment size.
  uint32_t m_packetSize;        //!< Packet size.
  uint32_t m_packets;           //!< Packet counter.
  uint32_t m_initialCwnd;       //!< Initial congestion window
  uint32_t m_delayedAck;        //!< Delayed Acknowledgement
  uint32_t m_lastCwnd;          //!< Last cWnd value reported
  uint32_t m_expectedCwnd;      //!< Expected final cWnd value
};

TcpLinuxRenoSSTest::TcpLinuxRenoSSTest (uint32_t segmentSize,
                                        uint32_t packetSize,
                                        uint32_t packets,
                                        uint32_t initialCwnd,
                                        uint32_t delayedAck,
                                        uint32_t expectedCwnd,
                                        TypeId &typeId,
                                        const std::string &desc)
  : TcpGeneralTest (desc),
    m_initial (true),
    m_segmentSize (segmentSize),
    m_packetSize (packetSize),
    m_packets (packets),
    m_initialCwnd (initialCwnd),
    m_delayedAck (delayedAck),
    m_lastCwnd (0),
    m_expectedCwnd (expectedCwnd)
{
  m_congControlTypeId = typeId;
}

void
TcpLinuxRenoSSTest::ConfigureEnvironment ()
{
  TcpGeneralTest::ConfigureEnvironment ();
  SetPropagationDelay (MilliSeconds (5));
  SetAppPktCount (m_packets);
  SetAppPktSize (m_packetSize);
}

void
TcpLinuxRenoSSTest::ConfigureProperties ()
{
  TcpGeneralTest::ConfigureProperties ();
  SetInitialCwnd (SENDER, m_initialCwnd);
  SetDelAckMaxCount (RECEIVER, m_delayedAck);
  SetSegmentSize (SENDER, m_segmentSize);
  SetSegmentSize (RECEIVER, m_segmentSize);
}

void
TcpLinuxRenoSSTest::QueueDrop (SocketWho who)
{
  NS_FATAL_ERROR ("Drop on the queue; cannot validate slow start");
}

void
TcpLinuxRenoSSTest::PhyDrop (SocketWho who)
{
  NS_FATAL_ERROR ("Drop on the phy: cannot validate slow start");
}

void
TcpLinuxRenoSSTest::CWndTrace (uint32_t oldValue, uint32_t newValue)
{
  NS_LOG_FUNCTION (this << oldValue << newValue);
  uint32_t segSize = GetSegSize (TcpGeneralTest::SENDER);
  uint32_t increase = newValue - oldValue;
  m_lastCwnd = newValue;

  if (m_initial)
    {
      m_initial = false;
      NS_TEST_ASSERT_MSG_EQ (newValue, m_initialCwnd * m_segmentSize, "The first update is for ACK of SYN and should initialize cwnd");
      return;
    }

  // ACK for first data packet is received to speed up the connection
  if (oldValue == m_initialCwnd * m_segmentSize)
    {
      return;
    }

  NS_TEST_ASSERT_MSG_EQ (increase, m_delayedAck * segSize, "Increase different than segsize");
  NS_TEST_ASSERT_MSG_LT_OR_EQ (newValue, GetInitialSsThresh (SENDER), "cWnd increased over ssth");

  NS_LOG_INFO ("Incremented cWnd by " << m_delayedAck * segSize << " bytes in Slow Start " <<
               "achieving a value of " << newValue);
}

void
TcpLinuxRenoSSTest::Tx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who)
{
  NS_LOG_FUNCTION (this << p << h << who);
}

void
TcpLinuxRenoSSTest::Rx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who)
{
  NS_LOG_FUNCTION (this << p << h << who);
}


void
TcpLinuxRenoSSTest::DoTeardown (void)
{
  NS_TEST_ASSERT_MSG_EQ (m_lastCwnd, m_expectedCwnd, "Congestion window did not evolve as expected");
  TcpGeneralTest::DoTeardown (); // call up to base class method to finish
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * This unit test checks that the slow start and congestion avoidance
 * behavior matches Linux behavior as follows:
 * 1) in both slow start and congestion avoidance phases, presence or
 *    absence of delayed acks does not alter the window growth
 * 2) in congestion avoidance phase, the arithmetic for counting the number
 *    of segments acked and deciding when to increment the congestion window
 *    (i.e. following the Linux function tcp_cong_avoid_ai()) is followed.
 * Different segment sizes (524 bytes and 1500 bytes) are tested.
 *
 * This is the Congestion avoidance test.
 */
class
TcpLinuxRenoCongAvoidTest : public TcpGeneralTest
{
public:
  /**
   * \brief Constructor.
   * \param segmentSize Segment size.
   * \param packetSize Size of the packets.
   * \param packets Number of packets.
   * \param initialCwnd Initial congestion window (segments)
   * \param initialSSThresh Initial slow start threshold (bytes)
   * \param delayedAck Delayed Acknowledgement
   * \param expectedCwnd Expected final m_cWnd value
   * \param congControl Type of congestion control.
   * \param desc The test description.
   */
  TcpLinuxRenoCongAvoidTest (uint32_t segmentSize, uint32_t packetSize,
                             uint32_t packets, uint32_t initialCwnd,
                             uint32_t initialSSThresh, uint32_t delayedAck,
                             uint32_t expectedCwnd,
                             TypeId& congControl, const std::string &desc);

protected:
  virtual void CWndTrace (uint32_t oldValue, uint32_t newValue);
  virtual void QueueDrop (SocketWho who);
  virtual void PhyDrop (SocketWho who);

  virtual void ConfigureEnvironment ();
  virtual void ConfigureProperties ();
  virtual void DoTeardown ();

private:
  virtual void Tx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who);
  virtual void Rx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who);
  uint32_t m_segmentSize;       //!< Segment size.
  uint32_t m_packetSize;        //!< Size of the packets used in socket writes.
  uint32_t m_packets;           //!< Number of packets to send to the socket.
  uint32_t m_initialCwnd;       //!< Initial congestion window (segments)
  uint32_t m_initialSSThresh;   //!< Initial slow start threshold (bytes)
  uint32_t m_delayedAck;        //!< Delayed Acknowledgement
  uint32_t m_lastCwnd;          //!< Last cWnd value reported
  uint32_t m_expectedCwnd;      //!< Expected final cWnd value
  uint32_t m_increment;         //!< Congestion window increment.
  bool m_initial;               //!< True on first run.
  bool m_inCongAvoidance;       //!< True if in congestion avoidance
  bool m_inSlowStartPhase;      //!< True if in slow start
};


TcpLinuxRenoCongAvoidTest::TcpLinuxRenoCongAvoidTest (uint32_t segmentSize,
                                                      uint32_t packetSize,
                                                      uint32_t packets,
                                                      uint32_t initialCwnd,
                                                      uint32_t initialSSThresh,
                                                      uint32_t delayedAck,
                                                      uint32_t expectedCwnd,
                                                      TypeId &typeId,
                                                      const std::string &desc)
  : TcpGeneralTest (desc),
    m_segmentSize (segmentSize),
    m_packetSize (packetSize),
    m_packets (packets),
    m_initialCwnd (initialCwnd),
    m_initialSSThresh (initialSSThresh),
    m_delayedAck (delayedAck),
    m_lastCwnd (0),
    m_expectedCwnd (expectedCwnd),
    m_increment (0),
    m_initial (true),
    m_inCongAvoidance (false),
    m_inSlowStartPhase (true)
{
  m_congControlTypeId = typeId;
}

void
TcpLinuxRenoCongAvoidTest::ConfigureEnvironment ()
{
  TcpGeneralTest::ConfigureEnvironment ();
  SetAppPktSize (m_packetSize);
  SetAppPktCount (m_packets);
  SetMTU (1500);
}

void TcpLinuxRenoCongAvoidTest::ConfigureProperties ()
{
  TcpGeneralTest::ConfigureProperties ();
  SetSegmentSize (SENDER, m_segmentSize);
  SetSegmentSize (RECEIVER, m_segmentSize);
  SetInitialCwnd (SENDER, m_initialCwnd);
  SetDelAckMaxCount (RECEIVER, m_delayedAck);
  SetInitialSsThresh (SENDER, m_initialSSThresh);
}

void
TcpLinuxRenoCongAvoidTest::CWndTrace (uint32_t oldValue, uint32_t newValue)
{
  NS_LOG_FUNCTION (this << oldValue << newValue);
  m_lastCwnd = newValue;
  if (m_initial)
    {
      m_initial = false;
      NS_TEST_ASSERT_MSG_EQ (newValue, m_initialCwnd * m_segmentSize, "The first update is for ACK of SYN and should initialize cwnd");
      return;
    }

  if ((newValue >= m_initialSSThresh * m_segmentSize) && !m_inCongAvoidance && (oldValue != m_initialSSThresh))
    {
      m_inCongAvoidance = true;
      m_inSlowStartPhase = false;
      return;
    }

  if (m_inSlowStartPhase)
    {
      return;
    }

  m_increment = newValue - oldValue;

  NS_TEST_ASSERT_MSG_EQ (m_increment, m_segmentSize, "Increase different than segsize");
}

void
TcpLinuxRenoCongAvoidTest::QueueDrop (SocketWho who)
{
  NS_FATAL_ERROR ("Drop on the queue; cannot validate congestion avoidance");
}

void
TcpLinuxRenoCongAvoidTest::PhyDrop (SocketWho who)
{
  NS_FATAL_ERROR ("Drop on the phy: cannot validate congestion avoidance");
}

void
TcpLinuxRenoCongAvoidTest::Tx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who)
{
  NS_LOG_FUNCTION (this << p << h << who);
}

void
TcpLinuxRenoCongAvoidTest::Rx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who)
{
  NS_LOG_FUNCTION (this << p << h << who);
}

void
TcpLinuxRenoCongAvoidTest::DoTeardown (void)
{
  NS_TEST_ASSERT_MSG_EQ (m_lastCwnd, m_expectedCwnd, "Congestion window did not evolve as expected");
  TcpGeneralTest::DoTeardown (); // call up to base class method to finish
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief TestSuite for the behavior of Linux Reno
 */
class TcpLinuxRenoTestSuite : public TestSuite
{
public:
  TcpLinuxRenoTestSuite () : TestSuite ("tcp-linux-reno-test", UNIT)
  {
    TypeId cong_control_type = TcpLinuxReno::GetTypeId ();
    // Test the behavior of Slow Start phase with small segment size
    // (524 bytes) and delayed acknowledgement of 1 and 2 segments
    //
    // Expected data pattern starting at simulation time 10:
    // (cwnd = 2 segments)     1  ->
    // (cwnd = 2 segments)     2  ->
    // (time 10.01s)              <-  ACK of 1
    // cwnd increased to 3 segments; send two more
    // (cwnd = 3 segments)     3  ->
    // (cwnd = 3 segments)     4  ->
    // (time 10.011s)             <-  ACK of 2
    // cwnd increased to 4 segments; send two more
    // (cwnd = 4 segments)     5  ->
    // (cwnd = 4 segments)     6  ->
    // (time 10.02s)              <-  ACK of 3
    // cwnd increased to 5 segments; send two more but only one more to send
    // (cwnd = 5 segments) 7+FIN  ->
    //                            <-  ACK of 4
    //                            <-  ACK of 5
    //                            <-  ACK of 6
    //                            <-  ACK of 7
    // cwnd should be at 9 segments
    //                            <-  FIN/ACK
    AddTestCase (new TcpLinuxRenoSSTest (524,   // segment size
                                         524,   // socket send size
                                         7,     // socket sends (i.e. packets)
                                         2,     // initial cwnd
                                         1,     // delayed ack count
                                         9 * 524, // expected final cWnd
                                         cong_control_type,
                                         "Slow Start MSS = 524, socket send size = 524, delack = 1 " + cong_control_type.GetName ()),
                 TestCase::QUICK);

    // Next, enabling delayed acks should not have an effect on the final
    // cWnd achieved
    AddTestCase (new TcpLinuxRenoSSTest (524,   // segment size
                                         524,   // socket send size
                                         7,     // socket sends
                                         2,     // initial cwnd
                                         2,     // delayed ack count
                                         9 * 524, // expected final cWnd
                                         cong_control_type,
                                         "Slow Start MSS = 524, socket send size = 524, delack = 2 " + cong_control_type.GetName ()),
                 TestCase::QUICK);

    // Test the behavior of Slow Start phase with standard segment size
    // (1500 bytes) and delayed acknowledgement of 1 and 2 segments
    //
    // We still expect m_cWnd to end up at 9 segments
    AddTestCase (new TcpLinuxRenoSSTest (1500,  // segment size
                                         1500,  // socket send size
                                         7,     // socket sends
                                         2,     // initial cwnd
                                         1,     // delayed ack count
                                         9 * 1500, // expected final cWnd
                                         cong_control_type,
                                         "Slow Start MSS = 1500, socket send size = 524, delack = 1 " + cong_control_type.GetName ()),
                 TestCase::QUICK);

    // Enable delayed acks; we still expect m_cWnd to end up at 9 segments
    AddTestCase (new TcpLinuxRenoSSTest (1500,  // segment size
                                         1500,  // socket send size
                                         7,     // socket sends
                                         2,     // initial cwnd
                                         2,     // delayed ack count
                                         9 * 1500, // expected final cWnd
                                         cong_control_type,
                                         "Slow Start MSS = 1500, socket send size = 524, delack = 2 " + cong_control_type.GetName ()),
                 TestCase::QUICK);

    // Test the behavior of Congestion Avoidance phase with small segment size
    // (524 bytes) and delayed acknowledgement of 1 and 2.  One important thing
    // to confirm is that delayed ACK behavior does not affect the congestion
    // window growth and final value because LinuxReno TCP counts segments acked
    //
    // Expected data pattern starting at simulation time 10:
    // (cwnd = 1 segment)      1  ->
    // (time 11s)                 <-  ACK of 1
    // (cwnd = 2 slow start)   2  ->
    // (can send one more  )   3  ->
    // (time 12s           )      <- ACK of 2
    // at this ACK, snd_cwnd >= ssthresh of 2, so go into CongestionAvoidance
    // snd_cwnd_count will be increased to 1, but less than current window 2
    // send one new segment to replace the one that was acked
    // (cwnd = 2 CA        )   4  ->
    // (again, time 12s    )      <- ACK of 3
    // at this ACK, snd_cwnd >= ssthresh of 2, so stay in CongestionAvoidance
    // snd_cwnd_count (m_cWndCnt) will be increased to 2, equal to w
    // We can increase cWnd to three segments and reset snd_cwnd_count
    //                         5  ->
    //                      6+FIN ->
    // (time 13s           )      <- ACK of 4
    // increase m_cWndCnt to 1
    // (time 13s           )      <- ACK of 5
    // increase m_cWndCnt to 2
    // (time 13s           )      <- ACK of 6
    // increase m_cWndCnt to 3, equal to window, so increase m_cWnd by one seg.
    // Final value of m_cWnd should be 4 * 524 = 2096
    AddTestCase (new TcpLinuxRenoCongAvoidTest (524,    // segment size
                                                524,    // socket send size
                                                6,      // socket sends
                                                1,      // initial cwnd
                                                2 * 524, // initial ssthresh
                                                1,      // delayed ack count
                                                4 * 524, // expected final cWnd
                                                cong_control_type,
                                                "Congestion Avoidance MSS = 524, socket send size = 524, delack = 1 " + cong_control_type.GetName ()),
                 TestCase::QUICK);

    // Repeat with delayed acks enabled:  should result in same final cWnd
    // Expected data pattern starting at simulation time 10:
    // (cwnd = 1 segment)      1  ->
    // (time 11s)                 <-  ACK of 1, ns-3 will always ack 1st seg
    // (cwnd = 2 slow start)   2  ->
    // (can send one more  )   3  ->
    // (time 12s           )      <- ACK of 3 (combined ack of 2 segments)
    // at this ACK, snd_cwnd >= ssthresh of 2, so go into CongestionAvoidance
    // snd_cwnd_count will be increased to 1+1 = current window 2
    // send one new segment to replace the one that was acked
    // (cwnd = 3 CA        )   4  ->
    // (cwnd = 3 CA        )   5  ->
    // (cwnd = 3 CA        )   6  ->
    // (time 13s           )      <- ACK of 5 (combined ack of 2 segments)
    // (time 13s           )      <- ACK of 6 (ack of 1 segment due to FIN)
    // increase m_cWndCnt to 3, equal to window, so increase m_cWnd by one seg.
    // Final value of m_cWnd should be 4 * 524 = 2096
    AddTestCase (new TcpLinuxRenoCongAvoidTest (524,    // segment size
                                                524,    // socket send size
                                                6,      // socket sends
                                                1,      // initial cwnd
                                                2,      // initial ssthresh
                                                2,      // delayed ack count
                                                4 * 524, // expected final cWnd
                                                cong_control_type,
                                                "Congestion Avoidance MSS = 524, socket send size = 524, delack = 2 " + cong_control_type.GetName ()),
                 TestCase::QUICK);

    // Test the behavior of Congestion Avoidance phase with standard segment size (i.e 1500 bytes)
    // and delayed acknowledgement of 1 and 2
    // Test the behavior of Congestion Avoidance phase with standard segment
    // size (1500 bytes) and delayed acknowledgement of 1 and 2.
    // This should result in the same pattern of segment exchanges as
    // above.
    AddTestCase (new TcpLinuxRenoCongAvoidTest (1500,   // segment size
                                                1500,   // socket send size
                                                6,      // socket sends
                                                1,      // initial cwnd
                                                2,      // initial ssthresh
                                                1,      // delayed ack count
                                                4 * 1500, // expected final cWnd
                                                cong_control_type,
                                                "Congestion Avoidance MSS = 1500, socket send size = 1500, delack = 1 " + cong_control_type.GetName ()),
                 TestCase::QUICK);

    AddTestCase (new TcpLinuxRenoCongAvoidTest (1500,   // segment size
                                                1500,   // socket send size
                                                6,      // socket sends
                                                1,      // initial cwnd
                                                2,      // initial ssthresh
                                                2,      // delayed ack count
                                                4 * 1500, // expected final cWnd
                                                cong_control_type,
                                                "Congestion Avoidance MSS = 1500, socket send size = 1500, delack = 2 " + cong_control_type.GetName ()),
                 TestCase::QUICK);
  }
};

static TcpLinuxRenoTestSuite g_tcpLinuxRenoTestSuite; //!< Static variable for test initialization
