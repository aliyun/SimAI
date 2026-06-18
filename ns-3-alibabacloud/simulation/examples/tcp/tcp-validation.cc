/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 Cable Television Laboratories, Inc.
 * Copyright (c) 2020 Tom Henderson (adapted for DCTCP testing)
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions, and the following disclaimer,
 *    without modification.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. The names of the authors may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * Alternatively, provided that this notice is retained in full, this
 * software may be distributed under the terms of the GNU General
 * Public License ("GPL") version 2, in which case the provisions of the
 * GPL apply INSTEAD OF those given above.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// This program is designed to observe long-running TCP congestion control
// behavior over a configurable bottleneck link.  The program is also
// instrumented to check progam data against validated results, when
// the validation option is enabled.
//
// ---> downstream (primary data transfer from servers to clients)
// <--- upstream (return acks and ICMP echo response)
//
//              ----   bottleneck link    ----
//  servers ---| WR |--------------------| LR |--- clients
//              ----                      ----
//  ns-3 node IDs:
//  nodes 0-2    3                         4        5-7
//
// - The box WR is notionally a WAN router, aggregating all server links
// - The box LR is notionally a LAN router, aggregating all client links
// - Three servers are connected to WR, three clients are connected to LR
//
// clients and servers are configured for ICMP measurements and TCP throughput
// and latency measurements in the downstream direction
//
// All link rates are enforced by a point-to-point (P2P) ns-3 model with full
// duplex operation.  Dynamic queue limits
// (BQL) are enabled to allow for queueing to occur at the priority queue layer;
// the notional P2P hardware device queue is limited to three packets.
//
// One-way link delays and link rates
// -----------------------------------
// (1) server to WR links, 1000 Mbps, 1us delay
// (2) bottleneck link:  configurable rate, configurable delay
// (3) client to LR links, 1000 Mbps, 1us delay
//
// By default, ns-3 FQ-CoDel model is installed on all interfaces, but
// the bottleneck queue uses CoDel by default and is configurable.
//
// The ns-3 FQ-CoDel model uses ns-3 defaults:
// - 100ms interval
// - 5ms target
// - drop batch size of 64 packets
// - minbytes of 1500
//
// Default simulation time is 70 sec.  For single flow experiments, the flow is
// started at simulation time 5 sec; if a second flow is used, it starts
// at 15 sec.
//
// ping frequency is set at 100ms.
//
// A command-line option to enable a step-threshold CE threshold
// from the CoDel queue model is provided.
//
// Measure:
//  - ping RTT
//  - TCP RTT estimate
//  - TCP throughput
//
// IPv4 addressing
// ----------------------------
// pingServer       10.1.1.2 (ping source)
// firstServer      10.1.2.2 (data sender)
// secondServer     10.1.3.2 (data sender)
// pingClient       192.168.1.2
// firstClient      192.168.2.2
// secondClient     192.168.3.2
//
// Program Options:
// ---------------
//    --firstTcpType:   first TCP type (cubic, dctcp, or reno) [cubic]
//    --secondTcpType:  second TCP type (cubic, dctcp, or reno) []
//    --queueType:      bottleneck queue type (fq, codel, pie, or red) [codel]
//    --baseRtt:        base RTT [+80ms]
//    --ceThreshold:    CoDel CE threshold (for DCTCP) [+1ms]
//    --linkRate:       data rate of bottleneck link [50000000bps]
//    --stopTime:       simulation stop time [+1.16667min]
//    --queueUseEcn:    use ECN on queue [false]
//    --enablePcap:     enable Pcap [false]
//    --validate:       validation case to run []
//
// validation cases (and syntax of how to run):
// ------------
// Case 'dctcp-10ms':  DCTCP single flow, 10ms base RTT, 50 Mbps link, ECN enabled, CoDel:
//     ./ns3 run 'tcp-validation --firstTcpType=dctcp --linkRate=50Mbps --baseRtt=10ms --queueUseEcn=1 --stopTime=15s --validate=1 --validation=dctcp-10ms'
//    - Throughput between 48 Mbps and 49 Mbps for time greater than 5.6s
//    - DCTCP alpha below 0.1 for time greater than 5.4s
//    - DCTCP alpha between 0.06 and 0.085 for time greater than 7s
//
// Case 'dctcp-80ms': DCTCP single flow, 80ms base RTT, 50 Mbps link, ECN enabled, CoDel:
//     ./ns3 run 'tcp-validation --firstTcpType=dctcp --linkRate=50Mbps --baseRtt=80ms --queueUseEcn=1 --stopTime=40s --validate=1 --validation=dctcp-80ms'
//    - Throughput less than 20 Mbps for time less than 14s
//    - Throughput less than 48 Mbps for time less than 30s
//    - Throughput between 47.5 Mbps and 48.5 for time greater than 32s
//    - DCTCP alpha above 0.1 for time less than 7.5
//    - DCTCP alpha below 0.01 for time greater than 11 and less than 30
//    - DCTCP alpha between 0.015 and 0.025 for time greater than 34
//
// Case 'cubic-50ms-no-ecn': CUBIC single flow, 50ms base RTT, 50 Mbps link, ECN disabled, CoDel:
//     ./ns3 run 'tcp-validation --firstTcpType=cubic --linkRate=50Mbps --baseRtt=50ms --queueUseEcn=0 --stopTime=20s --validate=1 --validation=cubic-50ms-no-ecn'
//    - Maximum value of cwnd is 511 segments at 5.4593 seconds
//    - cwnd decreases to 173 segments at 5.80304 seconds
//    - cwnd reaches another local maxima around 14.2815 seconds of 236 segments
//    - cwnd reaches a second maximum around 18.048 seconds of 234 segments
//
// Case 'cubic-50ms-ecn': CUBIC single flow, 50ms base RTT, 50 Mbps link, ECN enabled, CoDel:
//     ./ns3 run 'tcp-validation --firstTcpType=cubic --linkRate=50Mbps --baseRtt=50ms --queueUseEcn=0 --stopTime=20s --validate=1 --validation=cubic-50ms-no-ecn'
//    - Maximum value of cwnd is 511 segments at 5.4593 seconds
//    - cwnd decreases to 173 segments at 5.7939 seconds
//    - cwnd reaches another local maxima around 14.3477 seconds of 236 segments
//    - cwnd reaches a second maximum around 18.064 seconds of 234 segments

#include <iostream>
#include <fstream>
#include <string>
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/applications-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/internet-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/point-to-point-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpValidation");

// These variables are declared outside of main() so that they can
// be used in trace sinks.
uint32_t g_firstBytesReceived = 0;
uint32_t g_secondBytesReceived = 0;
uint32_t g_marksObserved = 0;
uint32_t g_dropsObserved = 0;
std::string g_validate = "";  // Empty string disables this mode
bool g_validationFailed = false;

void
TraceFirstCwnd (std::ofstream* ofStream, uint32_t oldCwnd, uint32_t newCwnd)
{
  // TCP segment size is configured below to be 1448 bytes
  // so that we can report cwnd in units of segments
  if (g_validate == "")
    {
      *ofStream << Simulator::Now ().GetSeconds () << " " << static_cast<double> (newCwnd) / 1448 << std::endl;
    }
  // Validation checks; both the ECN enabled and disabled cases are similar
  if (g_validate == "cubic-50ms-no-ecn" || g_validate == "cubic-50ms-ecn")
    {
      double now = Simulator::Now ().GetSeconds ();
      double cwnd = static_cast<double> (newCwnd) / 1448;
      if ((now > 5.43) && (now < 5.465) && (cwnd < 500))
        {
          NS_LOG_WARN ("now " << Now ().As (Time::S) << " cwnd " << cwnd << " (expected >= 500)");
          g_validationFailed = true;
        }
      else if ((now > 5.795) && (now < 6) && (cwnd > 190))
        {
          NS_LOG_WARN ("now " << Now ().As (Time::S) << " cwnd " << cwnd << " (expected <= 190)");
          g_validationFailed = true;
        }
      else if ((now > 14) && (now < 14.197) && (cwnd < 224))
        {
          NS_LOG_WARN ("now " << Now ().As (Time::S) << " cwnd " << cwnd << " (expected >= 224)");
          g_validationFailed = true;
        }
      else if ((now > 17) && (now < 18.026) && (cwnd < 212))
        {
          NS_LOG_WARN ("now " << Now ().As (Time::S) << " cwnd " << cwnd << " (expected >= 212)");
          g_validationFailed = true;
        }
    }
}

void
TraceFirstDctcp (std::ofstream* ofStream, uint32_t bytesMarked, uint32_t bytesAcked, double alpha)
{
  if (g_validate == "")
    {
      *ofStream << Simulator::Now ().GetSeconds () << " " << alpha << std::endl;
    }
  // Validation checks
  if (g_validate == "dctcp-80ms")
    {
      double now = Simulator::Now ().GetSeconds ();
      if ((now < 7.5) && (alpha < 0.1))
        {
          NS_LOG_WARN ("now " << Now ().As (Time::S) << " alpha " << alpha << " (expected >= 0.1)");
          g_validationFailed = true;
        }
      else if ((now > 11) && (now < 30) && (alpha > 0.01))
        {
          NS_LOG_WARN ("now " << Now ().As (Time::S) << " alpha " << alpha << " (expected <= 0.01)");
          g_validationFailed = true;
        }
      else if ((now > 34) && (alpha < 0.015) && (alpha > 0.025))
        {
          NS_LOG_WARN ("now " << Now ().As (Time::S) << " alpha " << alpha << " (expected 0.015 <= alpha <= 0.025)");
          g_validationFailed = true;
        }
    }
  else if (g_validate == "dctcp-10ms")
    {
      double now = Simulator::Now ().GetSeconds ();
      if ((now > 5.6) && (alpha > 0.1))
        {
          NS_LOG_WARN ("now " << Now ().As (Time::S) << " alpha " << alpha << " (expected <= 0.1)");
          g_validationFailed = true;
        }
      if ((now > 7) && ((alpha > 0.09) || (alpha < 0.055)))
        {
          NS_LOG_WARN ("now " << Now ().As (Time::S) << " alpha " << alpha << " (expected 0.09 <= alpha <= 0.055)");
          g_validationFailed = true;
        }
    }
}

void
TraceFirstRtt (std::ofstream* ofStream, Time oldRtt, Time newRtt)
{
  if (g_validate == "")
    {
      *ofStream << Simulator::Now ().GetSeconds () << " " << newRtt.GetSeconds () * 1000 << std::endl;
    }
}

void
TraceSecondCwnd (std::ofstream* ofStream, uint32_t oldCwnd, uint32_t newCwnd)
{
  // TCP segment size is configured below to be 1448 bytes
  // so that we can report cwnd in units of segments
  if (g_validate == "")
    {
      *ofStream << Simulator::Now ().GetSeconds () << " " << static_cast<double> (newCwnd) / 1448 << std::endl;
    }
}

void
TraceSecondRtt (std::ofstream* ofStream, Time oldRtt, Time newRtt)
{
  if (g_validate == "")
    {
      *ofStream << Simulator::Now ().GetSeconds () << " " << newRtt.GetSeconds () * 1000 << std::endl;
    }
}

void
TraceSecondDctcp (std::ofstream* ofStream, uint32_t bytesMarked, uint32_t bytesAcked, double alpha)
{
  if (g_validate == "")
    {
      *ofStream << Simulator::Now ().GetSeconds () << " " << alpha << std::endl;
    }
}

void
TracePingRtt (std::ofstream* ofStream, Time rtt)
{
  if (g_validate == "")
    {
      *ofStream << Simulator::Now ().GetSeconds () << " " << rtt.GetSeconds () * 1000 << std::endl;
    }
}

void
TraceFirstRx (Ptr<const Packet> packet, const Address &address)
{
  g_firstBytesReceived += packet->GetSize ();
}

void
TraceSecondRx (Ptr<const Packet> packet, const Address &address)
{
  g_secondBytesReceived += packet->GetSize ();
}

void
TraceQueueDrop (std::ofstream* ofStream, Ptr<const QueueDiscItem> item)
{
  if (g_validate == "")
    {
      *ofStream << Simulator::Now ().GetSeconds () << " " << std::hex << item->Hash () << std::endl;
    }
  g_dropsObserved++;
}

void
TraceQueueMark (std::ofstream* ofStream, Ptr<const QueueDiscItem> item, const char* reason)
{
  if (g_validate == "")
    {
      *ofStream << Simulator::Now ().GetSeconds () << " " << std::hex << item->Hash () << std::endl;
    }
  g_marksObserved++;
}

void
TraceQueueLength (std::ofstream* ofStream, DataRate queueLinkRate, uint32_t oldVal, uint32_t newVal)
{
  // output in units of ms
  if (g_validate == "")
    {
      *ofStream << Simulator::Now ().GetSeconds () << " " << std::fixed << static_cast<double> (newVal * 8) / (queueLinkRate.GetBitRate () / 1000) << std::endl;
    }
}

void
TraceMarksFrequency (std::ofstream* ofStream, Time marksSamplingInterval)
{
  if (g_validate == "")
    {
      *ofStream << Simulator::Now ().GetSeconds () << " " << g_marksObserved << std::endl;
    }
  g_marksObserved = 0;
  Simulator::Schedule (marksSamplingInterval, &TraceMarksFrequency, ofStream, marksSamplingInterval);
}

void
TraceFirstThroughput (std::ofstream* ofStream, Time throughputInterval)
{
  double throughput = g_firstBytesReceived * 8 / throughputInterval.GetSeconds () / 1e6;
  if (g_validate == "")
    {
      *ofStream << Simulator::Now ().GetSeconds () << " " << throughput << std::endl;
    }
  g_firstBytesReceived = 0;
  Simulator::Schedule (throughputInterval, &TraceFirstThroughput, ofStream, throughputInterval);
  if (g_validate == "dctcp-80ms")
    {
      double now = Simulator::Now ().GetSeconds ();
      if ((now < 14) && (throughput > 20))
        {
          NS_LOG_WARN ("now " << Now ().As (Time::S) << " throughput " << throughput << " (expected <= 20)");
          g_validationFailed = true;
        }
      if ((now < 30) && (throughput > 48))
        {
          NS_LOG_WARN ("now " << Now ().As (Time::S) << " throughput " << throughput << " (expected <= 48)");
          g_validationFailed = true;
        }
      if ((now > 32) && ((throughput < 47.5) || (throughput > 48.5)))
        {
          NS_LOG_WARN ("now " << Now ().As (Time::S) << " throughput " << throughput << " (expected 47.5 <= throughput <= 48.5)");
          g_validationFailed = true;
        }
    }
  else if (g_validate == "dctcp-10ms")
    {
      double now = Simulator::Now ().GetSeconds ();
      if ((now > 5.6) && ((throughput < 48) || (throughput > 49)))
        {
          NS_LOG_WARN ("now " << Now ().As (Time::S) << " throughput " << throughput << " (expected 48 <= throughput <= 49)");
          g_validationFailed = true;
        }
    }
}

void
TraceSecondThroughput (std::ofstream* ofStream, Time throughputInterval)
{
  if (g_validate == "")
    {
      *ofStream << Simulator::Now ().GetSeconds () << " " << g_secondBytesReceived * 8 / throughputInterval.GetSeconds () / 1e6 << std::endl;
    }
  g_secondBytesReceived = 0;
  Simulator::Schedule (throughputInterval, &TraceSecondThroughput, ofStream, throughputInterval);
}

void
ScheduleFirstTcpCwndTraceConnection (std::ofstream* ofStream)
{
  Config::ConnectWithoutContextFailSafe ("/NodeList/1/$ns3::TcpL4Protocol/SocketList/0/CongestionWindow", MakeBoundCallback (&TraceFirstCwnd, ofStream));
}

void
ScheduleFirstTcpRttTraceConnection (std::ofstream* ofStream)
{
  Config::ConnectWithoutContextFailSafe ("/NodeList/1/$ns3::TcpL4Protocol/SocketList/0/RTT", MakeBoundCallback (&TraceFirstRtt, ofStream));
}

void
ScheduleFirstDctcpTraceConnection (std::ofstream* ofStream)
{
  Config::ConnectWithoutContextFailSafe ("/NodeList/1/$ns3::TcpL4Protocol/SocketList/0/CongestionOps/$ns3::TcpDctcp/CongestionEstimate", MakeBoundCallback (&TraceFirstDctcp, ofStream));
}

void
ScheduleSecondDctcpTraceConnection (std::ofstream* ofStream)
{
  Config::ConnectWithoutContextFailSafe ("/NodeList/2/$ns3::TcpL4Protocol/SocketList/0/CongestionOps/$ns3::TcpDctcp/CongestionEstimate", MakeBoundCallback (&TraceSecondDctcp, ofStream));
}

void
ScheduleFirstPacketSinkConnection (void)
{
  Config::ConnectWithoutContextFailSafe ("/NodeList/6/ApplicationList/*/$ns3::PacketSink/Rx", MakeCallback (&TraceFirstRx));
}

void
ScheduleSecondTcpCwndTraceConnection (std::ofstream* ofStream)
{
  Config::ConnectWithoutContext ("/NodeList/2/$ns3::TcpL4Protocol/SocketList/0/CongestionWindow", MakeBoundCallback (&TraceSecondCwnd, ofStream));
}

void
ScheduleSecondTcpRttTraceConnection (std::ofstream* ofStream)
{
  Config::ConnectWithoutContext ("/NodeList/2/$ns3::TcpL4Protocol/SocketList/0/RTT", MakeBoundCallback (&TraceSecondRtt, ofStream));
}

void
ScheduleSecondPacketSinkConnection (void)
{
  Config::ConnectWithoutContext ("/NodeList/7/ApplicationList/*/$ns3::PacketSink/Rx", MakeCallback (&TraceSecondRx));
}

int
main (int argc, char *argv[])
{
  ////////////////////////////////////////////////////////////
  // variables not configured at command line               //
  ////////////////////////////////////////////////////////////
  uint32_t pingSize = 100; // bytes
  bool enableSecondTcp = false;
  bool enableLogging = false;
  Time pingInterval = MilliSeconds (100);
  Time marksSamplingInterval = MilliSeconds (100);
  Time throughputSamplingInterval = MilliSeconds (200);
  std::string pingTraceFile = "tcp-validation-ping.dat";
  std::string firstTcpRttTraceFile = "tcp-validation-first-tcp-rtt.dat";
  std::string firstTcpCwndTraceFile = "tcp-validation-first-tcp-cwnd.dat";
  std::string firstDctcpTraceFile = "tcp-validation-first-dctcp-alpha.dat";
  std::string firstTcpThroughputTraceFile = "tcp-validation-first-tcp-throughput.dat";
  std::string secondTcpRttTraceFile = "tcp-validation-second-tcp-rtt.dat";
  std::string secondTcpCwndTraceFile = "tcp-validation-second-tcp-cwnd.dat";
  std::string secondTcpThroughputTraceFile = "tcp-validation-second-tcp-throughput.dat";
  std::string secondDctcpTraceFile = "tcp-validation-second-dctcp-alpha.dat";
  std::string queueMarkTraceFile = "tcp-validation-queue-mark.dat";
  std::string queueDropTraceFile = "tcp-validation-queue-drop.dat";
  std::string queueMarksFrequencyTraceFile = "tcp-validation-queue-marks-frequency.dat";
  std::string queueLengthTraceFile = "tcp-validation-queue-length.dat";

  ////////////////////////////////////////////////////////////
  // variables configured at command line                   //
  ////////////////////////////////////////////////////////////
  std::string firstTcpType = "cubic";
  std::string secondTcpType = "";
  std::string queueType = "codel";
  Time stopTime = Seconds (70);
  Time baseRtt = MilliSeconds (80);
  DataRate linkRate ("50Mbps");
  bool queueUseEcn = false;
  Time ceThreshold = MilliSeconds (1);
  bool enablePcap = false;

  ////////////////////////////////////////////////////////////
  // Override ns-3 defaults                                 //
  ////////////////////////////////////////////////////////////
  Config::SetDefault ("ns3::TcpSocket::SegmentSize", UintegerValue (1448));
  // Increase default buffer sizes to improve throughput over long delay paths
  //Config::SetDefault ("ns3::TcpSocket::SndBufSize",UintegerValue (8192000));
  //Config::SetDefault ("ns3::TcpSocket::RcvBufSize",UintegerValue (8192000));
  Config::SetDefault ("ns3::TcpSocket::SndBufSize",UintegerValue (32768000));
  Config::SetDefault ("ns3::TcpSocket::RcvBufSize",UintegerValue (32768000));
  Config::SetDefault ("ns3::TcpSocket::InitialCwnd", UintegerValue (10));
  Config::SetDefault ("ns3::TcpL4Protocol::RecoveryType", TypeIdValue (TcpPrrRecovery::GetTypeId ()));

  ////////////////////////////////////////////////////////////
  // command-line argument parsing                          //
  ////////////////////////////////////////////////////////////
  CommandLine cmd (__FILE__);
  cmd.AddValue ("firstTcpType", "first TCP type (cubic, dctcp, or reno)", firstTcpType);
  cmd.AddValue ("secondTcpType", "second TCP type (cubic, dctcp, or reno)", secondTcpType);
  cmd.AddValue ("queueType", "bottleneck queue type (fq, codel, pie, or red)", queueType);
  cmd.AddValue ("baseRtt", "base RTT", baseRtt);
  cmd.AddValue ("ceThreshold", "CoDel CE threshold (for DCTCP)", ceThreshold);
  cmd.AddValue ("linkRate", "data rate of bottleneck link", linkRate);
  cmd.AddValue ("stopTime", "simulation stop time", stopTime);
  cmd.AddValue ("queueUseEcn", "use ECN on queue", queueUseEcn);
  cmd.AddValue ("enablePcap", "enable Pcap", enablePcap);
  cmd.AddValue ("validate", "validation case to run", g_validate);
  cmd.Parse (argc, argv);

  // If validation is selected, perform some configuration checks
  if (g_validate != "")
    {
      NS_ABORT_MSG_UNLESS (g_validate == "dctcp-10ms"
                           || g_validate == "dctcp-80ms"
                           || g_validate == "cubic-50ms-no-ecn"
                           || g_validate == "cubic-50ms-ecn", "Unknown test");
      if (g_validate == "dctcp-10ms" || g_validate == "dctcp-80ms")
        {
          NS_ABORT_MSG_UNLESS (firstTcpType == "dctcp", "Incorrect TCP");
          NS_ABORT_MSG_UNLESS (secondTcpType == "", "Incorrect TCP");
          NS_ABORT_MSG_UNLESS (linkRate == DataRate ("50Mbps"), "Incorrect data rate");
          NS_ABORT_MSG_UNLESS (queueUseEcn == true, "Incorrect ECN configuration");
          NS_ABORT_MSG_UNLESS (stopTime >= Seconds (15), "Incorrect stopTime");
          if (g_validate == "dctcp-10ms")
            {
              NS_ABORT_MSG_UNLESS (baseRtt == MilliSeconds (10), "Incorrect RTT");
            }
          else if (g_validate == "dctcp-80ms")
            {
              NS_ABORT_MSG_UNLESS (baseRtt == MilliSeconds (80), "Incorrect RTT");
            }
        }
      else if (g_validate == "cubic-50ms-no-ecn" || g_validate == "cubic-50ms-ecn")
        {
          NS_ABORT_MSG_UNLESS (firstTcpType == "cubic", "Incorrect TCP");
          NS_ABORT_MSG_UNLESS (secondTcpType == "", "Incorrect TCP");
          NS_ABORT_MSG_UNLESS (baseRtt == MilliSeconds (50), "Incorrect RTT");
          NS_ABORT_MSG_UNLESS (linkRate == DataRate ("50Mbps"), "Incorrect data rate");
          NS_ABORT_MSG_UNLESS (stopTime >= Seconds (20), "Incorrect stopTime");
          if (g_validate == "cubic-50ms-no-ecn")
            {
              NS_ABORT_MSG_UNLESS (queueUseEcn == false, "Incorrect ECN configuration");
            }
          else if (g_validate == "cubic-50ms-ecn")
            {
              NS_ABORT_MSG_UNLESS (queueUseEcn == true, "Incorrect ECN configuration");
            }
        }
    }

  if (enableLogging)
    {
      LogComponentEnable ("TcpSocketBase", (LogLevel)(LOG_PREFIX_FUNC | LOG_PREFIX_NODE | LOG_PREFIX_TIME | LOG_LEVEL_ALL));
      LogComponentEnable ("TcpDctcp", (LogLevel)(LOG_PREFIX_FUNC | LOG_PREFIX_NODE | LOG_PREFIX_TIME | LOG_LEVEL_ALL));
    }

  Time oneWayDelay = baseRtt / 2;

  TypeId firstTcpTypeId;
  if (firstTcpType == "reno")
    {
      firstTcpTypeId = TcpLinuxReno::GetTypeId ();
    }
  else if (firstTcpType == "cubic")
    {
      firstTcpTypeId = TcpCubic::GetTypeId ();
    }
  else if (firstTcpType == "dctcp")
    {
      firstTcpTypeId = TcpDctcp::GetTypeId ();
      Config::SetDefault ("ns3::CoDelQueueDisc::CeThreshold", TimeValue (ceThreshold));
      Config::SetDefault ("ns3::FqCoDelQueueDisc::CeThreshold", TimeValue (ceThreshold));
      if (queueUseEcn == false)
        {
          std::cout << "Warning: using DCTCP with queue ECN disabled" << std::endl;
        }
    }
  else
    {
      NS_FATAL_ERROR ("Fatal error:  tcp unsupported");
    }
  TypeId secondTcpTypeId;
  if (secondTcpType == "reno")
    {
      enableSecondTcp = true;
      secondTcpTypeId = TcpLinuxReno::GetTypeId ();
    }
  else if (secondTcpType == "cubic")
    {
      enableSecondTcp = true;
      secondTcpTypeId = TcpCubic::GetTypeId ();
    }
  else if (secondTcpType == "dctcp")
    {
      enableSecondTcp = true;
      secondTcpTypeId = TcpDctcp::GetTypeId ();
    }
  else if (secondTcpType == "")
    {
      enableSecondTcp = false;
      NS_LOG_DEBUG ("No second TCP selected");
    }
  else
    {
      NS_FATAL_ERROR ("Fatal error:  tcp unsupported");
    }
  TypeId queueTypeId;
  if (queueType == "fq")
    {
      queueTypeId = FqCoDelQueueDisc::GetTypeId ();
    }
  else if (queueType == "codel")
    {
      queueTypeId = CoDelQueueDisc::GetTypeId ();
    }
  else if (queueType == "pie")
    {
      queueTypeId = PieQueueDisc::GetTypeId ();
    }
  else if (queueType == "red")
    {
      queueTypeId = RedQueueDisc::GetTypeId ();
    }
  else
    {
      NS_FATAL_ERROR ("Fatal error:  queueType unsupported");
    }

  if (queueUseEcn)
    {
      Config::SetDefault ("ns3::CoDelQueueDisc::UseEcn", BooleanValue (true));
      Config::SetDefault ("ns3::FqCoDelQueueDisc::UseEcn", BooleanValue (true));
      Config::SetDefault ("ns3::PieQueueDisc::UseEcn", BooleanValue (true));
      Config::SetDefault ("ns3::RedQueueDisc::UseEcn", BooleanValue (true));
    }
  // Enable TCP to use ECN regardless
  Config::SetDefault ("ns3::TcpSocketBase::UseEcn", StringValue ("On"));

  // Report on configuration
  if (enableSecondTcp)
    {
      NS_LOG_DEBUG ("first TCP: " << firstTcpTypeId.GetName () << "; second TCP: " << secondTcpTypeId.GetName () << "; queue: " << queueTypeId.GetName () << "; ceThreshold: " << ceThreshold.GetSeconds () * 1000 << "ms");
    }
  else 
    {
      NS_LOG_DEBUG ("first TCP: " << firstTcpTypeId.GetName () << "; queue: " << queueTypeId.GetName () << "; ceThreshold: " << ceThreshold.GetSeconds () * 1000 << "ms");
    }
  
  // Write traces only if we are not in validation mode (g_validate == "")
  std::ofstream pingOfStream;
  std::ofstream firstTcpRttOfStream;
  std::ofstream firstTcpCwndOfStream;
  std::ofstream firstTcpThroughputOfStream;
  std::ofstream firstTcpDctcpOfStream;
  std::ofstream secondTcpRttOfStream;
  std::ofstream secondTcpCwndOfStream;
  std::ofstream secondTcpThroughputOfStream;
  std::ofstream secondTcpDctcpOfStream;
  std::ofstream queueDropOfStream;
  std::ofstream queueMarkOfStream;
  std::ofstream queueMarksFrequencyOfStream;
  std::ofstream queueLengthOfStream;
  if (g_validate == "")
    {
      pingOfStream.open (pingTraceFile.c_str (), std::ofstream::out);
      firstTcpRttOfStream.open (firstTcpRttTraceFile.c_str (), std::ofstream::out);
      firstTcpCwndOfStream.open (firstTcpCwndTraceFile.c_str (), std::ofstream::out);
      firstTcpThroughputOfStream.open (firstTcpThroughputTraceFile.c_str (), std::ofstream::out);
      if (firstTcpType == "dctcp")
        {
          firstTcpDctcpOfStream.open (firstDctcpTraceFile.c_str (), std::ofstream::out);
        }
      if (enableSecondTcp)
        {
          secondTcpRttOfStream.open (secondTcpRttTraceFile.c_str (), std::ofstream::out);
          secondTcpCwndOfStream.open (secondTcpCwndTraceFile.c_str (), std::ofstream::out);
          secondTcpThroughputOfStream.open (secondTcpThroughputTraceFile.c_str (), std::ofstream::out);
          if (secondTcpType == "dctcp")
            {
              secondTcpDctcpOfStream.open (secondDctcpTraceFile.c_str (), std::ofstream::out);
            }
        }
      queueDropOfStream.open (queueDropTraceFile.c_str (), std::ofstream::out);
      queueMarkOfStream.open (queueMarkTraceFile.c_str (), std::ofstream::out);
      queueMarksFrequencyOfStream.open (queueMarksFrequencyTraceFile.c_str (), std::ofstream::out);
      queueLengthOfStream.open (queueLengthTraceFile.c_str (), std::ofstream::out);
    }

  ////////////////////////////////////////////////////////////
  // scenario setup                                         //
  ////////////////////////////////////////////////////////////
  Ptr<Node> pingServer = CreateObject<Node> ();
  Ptr<Node> firstServer = CreateObject<Node> ();
  Ptr<Node> secondServer = CreateObject<Node> ();
  Ptr<Node> wanRouter = CreateObject<Node> ();
  Ptr<Node> lanRouter = CreateObject<Node> ();
  Ptr<Node> pingClient = CreateObject<Node> ();
  Ptr<Node> firstClient = CreateObject<Node> ();
  Ptr<Node> secondClient = CreateObject<Node> ();

  // Device containers
  NetDeviceContainer pingServerDevices;
  NetDeviceContainer firstServerDevices;
  NetDeviceContainer secondServerDevices;
  NetDeviceContainer wanLanDevices;
  NetDeviceContainer pingClientDevices;
  NetDeviceContainer firstClientDevices;
  NetDeviceContainer secondClientDevices;

  PointToPointHelper p2p;
  p2p.SetQueue ("ns3::DropTailQueue", "MaxSize", QueueSizeValue (QueueSize ("3p")));
  p2p.SetDeviceAttribute ("DataRate", DataRateValue (DataRate ("1000Mbps")));
  // Add delay only on the WAN links
  p2p.SetChannelAttribute ("Delay", TimeValue (MicroSeconds (1)));
  pingServerDevices = p2p.Install (wanRouter, pingServer);
  firstServerDevices = p2p.Install (wanRouter, firstServer);
  secondServerDevices = p2p.Install (wanRouter, secondServer);
  p2p.SetChannelAttribute ("Delay", TimeValue (oneWayDelay));
  wanLanDevices = p2p.Install (wanRouter, lanRouter);
  p2p.SetQueue ("ns3::DropTailQueue", "MaxSize", QueueSizeValue (QueueSize ("3p")));
  p2p.SetChannelAttribute ("Delay", TimeValue (MicroSeconds (1)));
  pingClientDevices = p2p.Install (lanRouter, pingClient);
  firstClientDevices = p2p.Install (lanRouter, firstClient);
  secondClientDevices = p2p.Install (lanRouter, secondClient);

  // Limit the bandwidth on the wanRouter->lanRouter interface
  Ptr<PointToPointNetDevice> p = wanLanDevices.Get (0)->GetObject<PointToPointNetDevice> ();
  p->SetAttribute ("DataRate", DataRateValue (linkRate));

  InternetStackHelper stackHelper;
  stackHelper.Install (pingServer);
  Ptr<TcpL4Protocol> proto;
  stackHelper.Install (firstServer);
  proto = firstServer->GetObject<TcpL4Protocol> ();
  proto->SetAttribute ("SocketType", TypeIdValue (firstTcpTypeId));
  stackHelper.Install (secondServer);
  stackHelper.Install (wanRouter);
  stackHelper.Install (lanRouter);
  stackHelper.Install (pingClient);

  stackHelper.Install (firstClient);
  // Set the per-node TCP type here
  proto = firstClient->GetObject<TcpL4Protocol> ();
  proto->SetAttribute ("SocketType", TypeIdValue (firstTcpTypeId));
  stackHelper.Install (secondClient);

  if (enableSecondTcp)
    {
      proto = secondClient->GetObject<TcpL4Protocol> ();
      proto->SetAttribute ("SocketType", TypeIdValue (secondTcpTypeId));
      proto = secondServer->GetObject<TcpL4Protocol> ();
      proto->SetAttribute ("SocketType", TypeIdValue (secondTcpTypeId));
    }

  // InternetStackHelper will install a base TrafficControLayer on the node,
  // but the Ipv4AddressHelper below will install the default FqCoDelQueueDisc
  // on all single device nodes.  The below code overrides the configuration
  // that is normally done by the Ipv4AddressHelper::Install() method by
  // instead explicitly configuring the queue discs we want on each device.
  TrafficControlHelper tchFq;
  tchFq.SetRootQueueDisc ("ns3::FqCoDelQueueDisc");
  tchFq.SetQueueLimits ("ns3::DynamicQueueLimits", "HoldTime", StringValue ("1ms"));
  tchFq.Install (pingServerDevices);
  tchFq.Install (firstServerDevices);
  tchFq.Install (secondServerDevices);
  tchFq.Install (wanLanDevices.Get (1));
  tchFq.Install (pingClientDevices);
  tchFq.Install (firstClientDevices);
  tchFq.Install (secondClientDevices);
  // Install queue for bottleneck link
  TrafficControlHelper tchBottleneck;
  tchBottleneck.SetRootQueueDisc (queueTypeId.GetName ());
  tchBottleneck.SetQueueLimits ("ns3::DynamicQueueLimits", "HoldTime", StringValue ("1ms"));
  tchBottleneck.Install (wanLanDevices.Get (0));

  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer pingServerIfaces = ipv4.Assign (pingServerDevices);
  ipv4.SetBase ("10.1.2.0", "255.255.255.0");
  Ipv4InterfaceContainer firstServerIfaces = ipv4.Assign (firstServerDevices);
  ipv4.SetBase ("10.1.3.0", "255.255.255.0");
  Ipv4InterfaceContainer secondServerIfaces = ipv4.Assign (secondServerDevices);
  ipv4.SetBase ("172.16.1.0", "255.255.255.0");
  Ipv4InterfaceContainer wanLanIfaces = ipv4.Assign (wanLanDevices);
  ipv4.SetBase ("192.168.1.0", "255.255.255.0");
  Ipv4InterfaceContainer pingClientIfaces = ipv4.Assign (pingClientDevices);
  ipv4.SetBase ("192.168.2.0", "255.255.255.0");
  Ipv4InterfaceContainer firstClientIfaces = ipv4.Assign (firstClientDevices);
  ipv4.SetBase ("192.168.3.0", "255.255.255.0");
  Ipv4InterfaceContainer secondClientIfaces = ipv4.Assign (secondClientDevices);

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  ////////////////////////////////////////////////////////////
  // application setup                                      //
  ////////////////////////////////////////////////////////////
  V4PingHelper pingHelper ("192.168.1.2");
  pingHelper.SetAttribute ("Interval", TimeValue (pingInterval));
  pingHelper.SetAttribute ("Size", UintegerValue (pingSize));
  ApplicationContainer pingContainer = pingHelper.Install (pingServer);
  Ptr<V4Ping> v4Ping = pingContainer.Get (0)->GetObject<V4Ping> ();
  v4Ping->TraceConnectWithoutContext ("Rtt", MakeBoundCallback (&TracePingRtt, &pingOfStream));
  pingContainer.Start (Seconds (1));
  pingContainer.Stop (stopTime - Seconds (1));

  ApplicationContainer firstApp;
  uint16_t firstPort = 5000;
  BulkSendHelper tcp ("ns3::TcpSocketFactory", Address ());
  // set to large value:  e.g. 1000 Mb/s for 60 seconds = 7500000000 bytes
  tcp.SetAttribute ("MaxBytes", UintegerValue (7500000000));
  // Configure first TCP client/server pair
  InetSocketAddress firstDestAddress (firstClientIfaces.GetAddress (1), firstPort);
  tcp.SetAttribute ("Remote", AddressValue (firstDestAddress));
  firstApp = tcp.Install (firstServer);
  firstApp.Start (Seconds (5));
  firstApp.Stop (stopTime - Seconds (1));

  Address firstSinkAddress (InetSocketAddress (Ipv4Address::GetAny (), firstPort));
  ApplicationContainer firstSinkApp;
  PacketSinkHelper firstSinkHelper ("ns3::TcpSocketFactory", firstSinkAddress);
  firstSinkApp = firstSinkHelper.Install (firstClient);
  firstSinkApp.Start (Seconds (5));
  firstSinkApp.Stop (stopTime - MilliSeconds (500));

  // Configure second TCP client/server pair
  if (enableSecondTcp)
    {
      BulkSendHelper tcp ("ns3::TcpSocketFactory", Address ());
      uint16_t secondPort = 5000;
      ApplicationContainer secondApp;
      InetSocketAddress secondDestAddress (secondClientIfaces.GetAddress (1), secondPort);
      tcp.SetAttribute ("Remote", AddressValue (secondDestAddress));
      secondApp = tcp.Install (secondServer);
      secondApp.Start (Seconds (15));
      secondApp.Stop (stopTime - Seconds (1));

      Address secondSinkAddress (InetSocketAddress (Ipv4Address::GetAny (), secondPort));
      PacketSinkHelper secondSinkHelper ("ns3::TcpSocketFactory", secondSinkAddress);
      ApplicationContainer secondSinkApp;
      secondSinkApp = secondSinkHelper.Install (secondClient);
      secondSinkApp.Start (Seconds (15));
      secondSinkApp.Stop (stopTime - MilliSeconds (500));
    }

  // Setup traces that can be hooked now
  Ptr<TrafficControlLayer> tc;
  Ptr<QueueDisc> qd;
  // Trace drops and marks for bottleneck
  tc = wanLanDevices.Get (0)->GetNode ()->GetObject<TrafficControlLayer> ();
  qd = tc->GetRootQueueDiscOnDevice (wanLanDevices.Get (0));
  qd->TraceConnectWithoutContext ("Drop", MakeBoundCallback (&TraceQueueDrop, &queueDropOfStream));
  qd->TraceConnectWithoutContext ("Mark", MakeBoundCallback (&TraceQueueMark, &queueMarkOfStream));
  qd->TraceConnectWithoutContext ("BytesInQueue", MakeBoundCallback (&TraceQueueLength, &queueLengthOfStream, linkRate));

  // Setup scheduled traces; TCP traces must be hooked after socket creation
  Simulator::Schedule (Seconds (5) + MilliSeconds (100), &ScheduleFirstTcpRttTraceConnection, &firstTcpRttOfStream);
  Simulator::Schedule (Seconds (5) + MilliSeconds (100), &ScheduleFirstTcpCwndTraceConnection, &firstTcpCwndOfStream);
  Simulator::Schedule (Seconds (5) + MilliSeconds (100), &ScheduleFirstPacketSinkConnection);
  if (firstTcpType == "dctcp")
    {
      Simulator::Schedule (Seconds (5) + MilliSeconds (100), &ScheduleFirstDctcpTraceConnection, &firstTcpDctcpOfStream);
    }
  Simulator::Schedule (throughputSamplingInterval, &TraceFirstThroughput, &firstTcpThroughputOfStream, throughputSamplingInterval);
  if (enableSecondTcp)
    {
      // Setup scheduled traces; TCP traces must be hooked after socket creation
      Simulator::Schedule (Seconds (15) + MilliSeconds (100), &ScheduleSecondTcpRttTraceConnection, &secondTcpRttOfStream);
      Simulator::Schedule (Seconds (15) + MilliSeconds (100), &ScheduleSecondTcpCwndTraceConnection, &secondTcpCwndOfStream);
      Simulator::Schedule (Seconds (15) + MilliSeconds (100), &ScheduleSecondPacketSinkConnection);
      Simulator::Schedule (throughputSamplingInterval, &TraceSecondThroughput, &secondTcpThroughputOfStream, throughputSamplingInterval);
      if (secondTcpType == "dctcp")
        {
          Simulator::Schedule (Seconds (15) + MilliSeconds (100), &ScheduleSecondDctcpTraceConnection, &secondTcpDctcpOfStream);
        }
    }
  Simulator::Schedule (marksSamplingInterval, &TraceMarksFrequency, &queueMarksFrequencyOfStream, marksSamplingInterval);

  if (enablePcap)
    {
      p2p.EnablePcapAll ("tcp-validation", false);
    }

  Simulator::Stop (stopTime);
  Simulator::Run ();
  Simulator::Destroy ();

  if (g_validate == "")
    {
      pingOfStream.close ();
      firstTcpCwndOfStream.close ();
      firstTcpRttOfStream.close ();
      if (firstTcpType == "dctcp")
        {
          firstTcpDctcpOfStream.close ();
        }
      firstTcpThroughputOfStream.close ();
      if (enableSecondTcp)
        {
          secondTcpCwndOfStream.close ();
          secondTcpRttOfStream.close ();
          secondTcpThroughputOfStream.close ();
          if (secondTcpType == "dctcp")
            {
              secondTcpDctcpOfStream.close ();
            }
        }
      queueDropOfStream.close ();
      queueMarkOfStream.close ();
      queueMarksFrequencyOfStream.close ();
      queueLengthOfStream.close ();
    }

  if (g_validationFailed)
    {
      NS_FATAL_ERROR ("Validation failed");
    }
}

