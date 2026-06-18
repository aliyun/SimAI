/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 NITK Surathkal
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
 * Authors: Bhaskar Kataria <bhaskar.k7920@gmail.com>
 *          Tom Henderson <tomhend@u.washington.edu>
 *          Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 *          Vivek Jain <jain.vivek.anand@gmail.com>
 *          Ankit Deepak <adadeepak8@gmail.com>
 * This script is written using Tom Henderson's L4S evaluation available at https://gitlab.com/tomhend/modules/l4s-evaluation
 */

// The 9 configurations below test BIC and DCTCP under various scenarios.
// Scenarios are numbered 1-9.  By default, scenario number 0 (i.e., no
// scenario) is configured, which means that the user is free to set
// any of the parameters freely; if scenarios 1 through 9 are selected,
// the scenario parameters are fixed.
//
// The configuration of the scenarios starts from basic TCP BIC without ECN
// with the base RTT of 80ms and then in the next scenario, ECN is enabled,
// and gradually, the complexity of the scenario increases and the last scenario
// consists of 2 flows one with BIC and other with DCTCP and finally tests
// The performance of the L4S mode of FqCoDel queue disc.

/** Network topology - For Configuration 1, 2
 *
 *      1Gbps, 40ms      100Mbps, 1us       1Gbps, 1us
 *   n0 ---------- n2 ---------------- n3 --------------n4
 *
 * Configuration 1: single flow TCP BIC without ECN, base RTT of 80 ms
 *
 * Expected Result:
 * Throughput:        Oscillating around 100ms.
 * Ping RTT:          Oscillating between 80ms and 80.5ms
 * Drop frequency:    Should be around 1 drop per 100ms with around 9 spikes in 10 seconds
 * Length:            below 5ms but with frequent spikes
 * Mark Frequency:    There should not be any marks
 * Congestion Window: Oscillating between around 600 segments and 800 segments
 * TCP RTT:           Oscillating below 5ms frequent spikes up to 90ms
 * Queue Delay-N0:    Below 5ms with frequent spikes
 *
 * Configuration 2: single flow TCP BIC with ECN, base RTT of 80 ms
 *
 * Expected Result:
 * Throughput:        Reaching 100Mbps and oscillating more than the previous configuration.
 * Ping RTT:          Oscillating between 80ms and 80.5ms
 * Drop frequency:    There should not be any drops
 * Length:            Below 5ms with frequent spikes, average around 3ms
 * Mark Frequency:    Should be around 1 mark per 100ms with around 9 spikes in 10 seconds
 * Congestion Window: Oscillating between around 400 segments and 800 segments
 * TCP RTT:           Below 85ms with frequent spikes up to 95ms (more than the previous configuration)
 * Queue Delay-n0:    Below 5ms with frequent spikes
 */

/** Network topology - For Configuration 3, 4
 *
 *     1Gbps, 1ms      100Mbps, 1us        1Gbps, 1us
 *   n0 ---------- n2 ---------------- n3 --------------n4
 *
 * Configuration 3: single flow DCTCP with ECN ECT(0), CE_threshold = 1ms, base RTT of 1ms
 *
 * Expected Result:
 * Throughput:        Fixed just below 100Mbps.
 * Ping RTT:          Oscillating below 1.5ms
 * Drop frequency:    There should not be any drops
 * Length:            Very frequently oscillating between 0.5ms and 1ms with frequent spikes up to 1.2ms
 * Mark Frequency:    Around 160 marks per 100ms
 * Congestion Window: Very frequently oscillating at a low value (below 25)
 * TCP RTT:           Very frequently oscillating, below 5ms
 * Queue Delay-n0:    Very frequently oscillating between 0.5ms and 1ms with spikes up to 1.2ms
 *
 * Configuration 4: single flow DCTCP with ECN ECT(1), L4S mode enabled on FqCoDel, base RTT of 1ms
 *
 * Expected Result:
 * Throughput:        Fixed just below 100Mbps.
 * Ping RTT:          Oscillating below 1.5ms
 * Drop frequency:    There should not be any drops
 * Length:            Very frequently oscillating between 0.5ms and 1ms with frequent spikes up to 1.2ms
 * Mark Frequency:    Around 160 marks per 100ms
 * Congestion Window: Very frequently oscillating at a low value (below 25)
 * TCP RTT:           Very frequently oscillating, below 5ms
 * Queue Delay-n0:    Very frequently oscillating between 0.5ms and 1ms with spikes up to 1.2ms
 */

/** Network topology - For Configuration 5, 6
 *
 *    1Gbps, 40ms                            1Gbps, 1us
 * n0--------------|                    |---------------n4
 *                 |    100Mbps, 1us    |
 *                 n2------------------n3
 *    1Gbps, 40ms  |                    |    1Gbps, 1us
 * n1--------------|                    |---------------n5
 *
 * Configuration 5: Two flows TCP BIC without ECN, base RTT of 80 ms
 *
 * Expected Result:
 * Throughput-n0:        Oscillating around 100 before second flow starts, and after that, oscillating around 50.
 * Throughput-n1:        Oscillating around 50.
 * Ping RTT:             Oscillating and reaching around 80.5ms
 * Drop frequency:       Around 1 drop per 100ms, more spikes than the single flow
 * Length:               Around 5ms with spikes reaching 10ms
 * Mark Frequency:       There should not be any marks
 * Congestion Window-n0: Oscillating between 800 segments and 600 segments before second flow starts,
 *                       after that oscillating around 300 segments
 * Congestion Window-n0: Oscillating around 300 segments
 * TCP RTT-n0:           Below 85ms with spikes reaching 90ms before second flow starts, and after that
 *                       below 85ms with spikes reaching 90ms
 * TCP RTT-n0:           Below 85ms with spikes reaching 90ms
 * Queue Delay-n0:       Below 5ms with spikes reaching 10ms before second flow starts and after that
 *                       below 5ms with spikes up to 15ms.
 * Queue Delay-n0:       Below 5ms with spikes up to 15ms.
 *
 * Configuration 6: Two flows TCP BIC with ECN, base RTT of 80 ms
 *
 * Expected Result:
 *
 * Throughput-n0:        Oscillating around 100 before second flow starts, and after that, oscillating around 50.
 * Throughput-n1:        Oscillating around 50.
 * Ping RTT:             Oscillating and reaching around 80.5ms
 * Drop frequency:       There should not be any drops
 * Mark Frequency:       Around 1 drop per 100ms, more spikes than the single flow
 * Length:               Around 5ms with spikes reaching 15ms
 * Congestion Window-n0: Oscillating between 800 segments and 600 segments before second flow starts,
 *                       after that oscillating around 300 segments
 * Congestion Window-n0: Oscillating around 300 segments
 * TCP RTT-n0:           Below 85ms with spikes reaching 95ms before second flow starts, and after that
 *                       below 85ms with spikes reaching 100ms
 * TCP RTT-n0:           Below 85ms with spikes reaching 100ms
 * Queue Delay-n0:       Below 5ms with spikes reaching 15ms before second flow starts and after that
 *                       below 5ms with spikes up to 20ms.
 * Queue Delay-n0:       Below 5ms with spikes up to 20ms.
 */

/**Network topology - For Configuration 7, 8
 *
 *    1Gbps, 1ms                            1Gbps, 1us
 * n0--------------|                    |---------------n4
 *                 |    100Mbps, 1us    |
 *                 n2------------------n3
 *     1Gbps, 1ms  |                    |    1Gbps, 1us
 * n1--------------|                    |---------------n5
 *
 * Configuration 7: Two flows DCTCP with ECN ECT(0), CE_threshold = 1ms, base RTT of 1ms
 *
 * Expected Result:
 * Throughput-n0:        Fixed around 100 before second flow starts, and after that, fixed around 50.
 * Throughput-n1:        Fixed around 50.
 * Ping RTT:             Oscillating and reaching around 1.5ms
 * Drop frequency:       There should not be any drops
 * Length:               Around 1ms.
 * Mark Frequency:       Oscillating little above 150 marks per 100ms before second flow start,
 *                       and after that oscillating little below 250 marks per 100ms
 * Congestion Window-n0: Oscillating around 20 segments before second flow starts
 *                       then oscillating around 11 segments
 * Congestion Window-n1: Oscillating around 11 segments
 * TCP RTT-n0:           Oscillating between 2ms and 5ms before seconds flow starts and after that
 *                       oscillating between 3ms and 5ms
 * TCP RTT-n1:           Oscillating between 3ms and 5ms
 * Queue Delay-n0:       Oscillating frequently below 1ms before second flow starts and after that just below 1.4ms
 * Queue Delay-n1:       Oscillating frequently between 0.4ms and 1.4ms
 *
 * Configuration 8: Two flows DCTCP with ECN ECT(1), L4S mode enabled on FqCoDel, base RTT of 1ms
 *
 * Throughput-n0:        Fixed around 100 before second flow starts, and after that, fixed around 50.
 * Throughput-n1:        Fixed around 50.
 * Ping RTT:             Oscillating and reaching around 1.5ms
 * Drop frequency:       There should not be any drops
 * Length:               Around 1ms.
 * Mark Frequency:       Oscillating little above 150 marks per 100ms before second flow start,
 *                       and after that oscillating little below 250 marks per 100ms
 * Congestion Window-n0: Oscillating around 20 segments before second flow starts
 *                       then oscillating around 11 segments
 * Congestion Window-n1: Oscillating around 11 segments
 * TCP RTT-n0:           Oscillating between 2ms and 5ms before seconds flow starts and after that
 *                       oscillating between 3ms and 5ms
 * TCP RTT-n1:           Oscillating between 3ms and 5ms
 * Queue Delay-n0:       Oscillating frequently below 1ms before second flow starts and after that just below 1.4ms
 * Queue Delay-n1:       Oscillating frequently between 0.4ms and 1.4ms
 */

/**Network topology - For Configuration 9
 *
 *    1Gbps, 1ms                            1Gbps, 1us
 * n0--------------|                    |---------------n4
 *                 |    100Mbps, 1us    |
 *                 n2------------------n3
 *     1Gbps, 80ms |                    |    1Gbps, 1us
 * n1--------------|                    |---------------n5
 *
 * Configuration 9: One TCP BIC (n0) with base RTT of 80 ms, one DCTCP (n1) with ECN ECT(1), L4S mode enabled, base RTT 1ms
 *
 * Throughput-n0:        Oscillating below 100Mbps before second flow starts, and after that, oscillating around 50Mbps.
 * Throughput-n1:        Oscillating around 50Mbps.
 * Ping RTT:             Oscillating and reaching around 80.5ms
 * Drop frequency:       There should not be any drops
 * Length:               Around 5ms with frequent spikes above 10ms before second flow starts and after that spikes below 10ms.
 * Mark Frequency:       Oscillating 1 marks per 100ms before second flow start,
 *                       and after that oscillating below 200 marks per 100ms
 * Congestion Window-n0: Oscillating around 600 segments before second flow starts
 *                       then oscillating around 300 segments
 * Congestion Window-n1: Oscillating around 300 segments
 * TCP RTT-n0:           Around 85ms with spikes reaching around 95ms before second flow starts and after that around 85ms
 *                       with spikes reaching 100ms
 * TCP RTT-n1:           Around 85ms with spikes reaching 100ms
 * Queue Delay-n0:       Around 5ms with spikes above 10ms before second flow starts and nearly the same after that too.
 * Queue Delay-n1:       Oscillating frequently between 0.4ms and 1.4ms with spikes reaching 3ms.
 *
 */

/**
* clients and servers are configured for ICMP measurements and TCP throughput
* and latency measurements in the downstream direction
*
* Depending on the scenario, the middlebox and endpoints will be
* configured differently. Scenarios can be configured explicitly by
* scenario Numbers as well as by combinations of input arguments.
*
* All link rates are enforced by a point-to-point (P2P) ns-3 model with full
* duplex operation.  The link rate and delays are enforced by this model
* (in contrast to netem and shaping in the testbed).  Dynamic queue limits
* (BQL) are enabled to allow for queueing to occur at the priority queue layer;
*
* By default, the ns-3 FQ-CoDel model is installed on all interfaces.
*
* The ns-3 FQ-CoDel model uses ns-3 defaults:
* - 100ms interval
* - 5ms target
* - drop batch size of 64 packets
* - minbytes of 1500
*
* Default simulation time is 70 sec.  For single flow experiments, the flow is
* started at simulation time 5 sec; if a second flow is used, it starts
* at 15 sec.
*
* ping frequency is set at 100ms
* Note that pings may miss the peak of queue buildups for short-lived flows;
* hence, we trace also the queue length expressed in units of time at
* the bottleneck link rate.
*
* A command-line option to enable CE Threshold is provided
*
* Results will be available in ns-3-dev/results/FqCoDel-L4S consisting of 14 .dat files
*
* Measure:
*  - ping RTT
*  - TCP RTT estimate
*  - TCP throughput
*  - Queue delay
*
* IPv4 addressing
* ----------------------------
* pingServer       10.1.1.2 (ping source)
* n0Server         10.1.2.2 (data sender)
* n1Server         10.1.3.2 (data sender)
* pingClient       192.168.1.2
* n4Client         192.168.2.2
* n5Client         192.168.3.2
*
* Program options
* ---------------
*    --n0TcpType:           First TCP type (bic, dctcp, or reno) [bic]
*    --n1TcpType:           Second TCP type (cubic, dctcp, or reno) []
*    --scenarioNum:         Scenario number from the scenarios avalaible in the file (1-9) [0]
*    --bottleneckQueueType: n2 queue type (fq or codel) [fq]
*    --baseRtt:             base RTT [80ms]
*    --useCeThreshold:      use CE threshold [false]
*    --useEcn:              use ECN [true]
*    --ceThreshold:         CE threshold [1ms]
*    --bottleneckRate       data rate of bottleneck [100Mbps]
*    --linkrate:            data rate of edge link to bottleneck link [1Gbps]
*    --stopTime:            simulation stop time [70s]
*    --enablePcap:          enable Pcap [false]
*    (additional arguments to control trace names)
**/

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/traffic-control-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("FqCoDelL4SExample");

uint32_t checkTimes;
double avgQueueDiscSize;

uint32_t g_n0BytesReceived = 0;
uint32_t g_n1BytesReceived = 0;
uint32_t g_marksObserved = 0;
uint32_t g_dropsObserved = 0;

void
TraceN0Cwnd (std::ofstream* ofStream, uint32_t oldCwnd, uint32_t newCwnd)
{
  // TCP segment size is configured below to be 1448 bytes
  // so that we can report cwnd in units of segments
  *ofStream << Simulator::Now ().GetSeconds () << " " << static_cast<double> (newCwnd) / 1448 << std::endl;
}

void
TraceN1Cwnd (std::ofstream* ofStream, uint32_t oldCwnd, uint32_t newCwnd)
{
  // TCP segment size is configured below to be 1448 bytes
  // so that we can report cwnd in units of segments
  *ofStream << Simulator::Now ().GetSeconds () << " " << static_cast<double> (newCwnd) / 1448 << std::endl;
}

void
TraceN0Rtt (std::ofstream* ofStream, Time oldRtt, Time newRtt)
{
  *ofStream << Simulator::Now ().GetSeconds () << " " << newRtt.GetSeconds () * 1000 << std::endl;
}

void
TraceN1Rtt (std::ofstream* ofStream, Time oldRtt, Time newRtt)
{
  *ofStream << Simulator::Now ().GetSeconds () << " " << newRtt.GetSeconds () * 1000 << std::endl;
}

void
TracePingRtt (std::ofstream* ofStream, Time rtt)
{
  *ofStream << Simulator::Now ().GetSeconds () << " " << rtt.GetSeconds () * 1000 << std::endl;
}

void
TraceN0Rx (Ptr<const Packet> packet, const Address &address)
{
  g_n0BytesReceived += packet->GetSize ();
}

void
TraceN1Rx (Ptr<const Packet> packet, const Address &address)
{
  g_n1BytesReceived += packet->GetSize ();
}

void
TraceDrop (std::ofstream* ofStream, Ptr<const QueueDiscItem> item)
{
  *ofStream << Simulator::Now ().GetSeconds () << " " << std::hex << item->Hash () << std::endl;
  g_dropsObserved++;
}

void
TraceMark (std::ofstream* ofStream, Ptr<const QueueDiscItem> item, const char* reason)
{
  *ofStream << Simulator::Now ().GetSeconds () << " " << std::hex << item->Hash () << std::endl;
  g_marksObserved++;
}

void
TraceQueueLength  (std::ofstream* ofStream, DataRate linkRate, uint32_t oldVal, uint32_t newVal)
{
  // output in units of ms
  *ofStream << Simulator::Now ().GetSeconds () << " " << std::fixed << static_cast<double> (newVal * 8) / (linkRate.GetBitRate () / 1000) << std::endl;
}

void
TraceDropsFrequency (std::ofstream* ofStream, Time dropsSamplingInterval)
{
  *ofStream << Simulator::Now ().GetSeconds () << " " << g_dropsObserved << std::endl;
  g_dropsObserved = 0;
  Simulator::Schedule (dropsSamplingInterval, &TraceDropsFrequency, ofStream, dropsSamplingInterval);
}

void
TraceMarksFrequency (std::ofstream* ofStream, Time marksSamplingInterval)
{
  *ofStream << Simulator::Now ().GetSeconds () << " " << g_marksObserved << std::endl;
  g_marksObserved = 0;
  Simulator::Schedule (marksSamplingInterval, &TraceMarksFrequency, ofStream, marksSamplingInterval);
}

void
TraceN0Throughput (std::ofstream* ofStream, Time throughputInterval)
{
  *ofStream << Simulator::Now ().GetSeconds () << " " << g_n0BytesReceived * 8 / throughputInterval.GetSeconds () / 1e6 << std::endl;
  g_n0BytesReceived = 0;
  Simulator::Schedule (throughputInterval, &TraceN0Throughput, ofStream, throughputInterval);
}

void
TraceN1Throughput (std::ofstream* ofStream, Time throughputInterval)
{
  *ofStream << Simulator::Now ().GetSeconds () << " " << g_n1BytesReceived * 8 / throughputInterval.GetSeconds () / 1e6 << std::endl;
  g_n1BytesReceived = 0;
  Simulator::Schedule (throughputInterval, &TraceN1Throughput, ofStream, throughputInterval);
}

void
ScheduleN0TcpCwndTraceConnection (std::ofstream* ofStream)
{
  Config::ConnectWithoutContext ("/NodeList/1/$ns3::TcpL4Protocol/SocketList/0/CongestionWindow", MakeBoundCallback (&TraceN0Cwnd, ofStream));
}

void
ScheduleN0TcpRttTraceConnection (std::ofstream* ofStream)
{
  Config::ConnectWithoutContext ("/NodeList/1/$ns3::TcpL4Protocol/SocketList/0/RTT", MakeBoundCallback (&TraceN0Rtt, ofStream));
}

void
ScheduleN0PacketSinkConnection (void)
{
  Config::ConnectWithoutContext ("/NodeList/6/ApplicationList/*/$ns3::PacketSink/Rx", MakeCallback (&TraceN0Rx));
}

void
ScheduleN1TcpCwndTraceConnection (std::ofstream* ofStream)
{
  Config::ConnectWithoutContext ("/NodeList/2/$ns3::TcpL4Protocol/SocketList/0/CongestionWindow", MakeBoundCallback (&TraceN1Cwnd, ofStream));
}

void
ScheduleN1TcpRttTraceConnection (std::ofstream* ofStream)
{
  Config::ConnectWithoutContext ("/NodeList/2/$ns3::TcpL4Protocol/SocketList/0/RTT", MakeBoundCallback (&TraceN1Rtt, ofStream));
}

void
ScheduleN1PacketSinkConnection (void)
{
  Config::ConnectWithoutContext ("/NodeList/7/ApplicationList/*/$ns3::PacketSink/Rx", MakeCallback (&TraceN1Rx));
}

static void
PacketDequeue (std::ofstream* n0OfStream, std::ofstream* n1OfStream, Ptr<QueueDiscItem const> item)
{
  Ptr<Packet> p = item->GetPacket ();
  Ptr<const Ipv4QueueDiscItem> iqdi = Ptr<const Ipv4QueueDiscItem> (dynamic_cast<const Ipv4QueueDiscItem *> (PeekPointer (item)));
  Ipv4Address address = iqdi->GetHeader ().GetDestination ();
  Time qDelay = Simulator::Now () - item->GetTimeStamp ();
  if (address == "192.168.2.2")
    {
      *n0OfStream << Simulator::Now ().GetSeconds () << " " << qDelay.GetMicroSeconds () / 1000.0 << std::endl;
    }
  else if (address == "192.168.3.2")
    {
      *n1OfStream << Simulator::Now ().GetSeconds () << " " << qDelay.GetMicroSeconds () / 1000.0 << std::endl;
    }
}

int
main (int argc, char *argv[])
{
  ////////////////////////////////////////////////////////////
  // variables not configured at command line               //
  ////////////////////////////////////////////////////////////
  Time stopTime = Seconds (70);
  Time baseRtt = MilliSeconds (80);
  uint32_t pingSize = 100; // bytes
  Time pingInterval = MilliSeconds (100);
  Time marksSamplingInterval = MilliSeconds (100);
  Time throughputSamplingInterval = MilliSeconds (200);
  DataRate bottleneckRate ("100Mbps");

  std::string dir = "results/FqCoDel-L4S/";
  std::string dirToSave = "mkdir -p " + dir;
  if (system (dirToSave.c_str ()) == -1)
    {
      exit (1);
    }

  std::string pingTraceFile = dir + "ping.dat";
  std::string n0TcpRttTraceFile = dir + "n0-tcp-rtt.dat";
  std::string n0TcpCwndTraceFile = dir + "n0-tcp-cwnd.dat";
  std::string n0TcpThroughputTraceFile = dir + "n0-tcp-throughput.dat";
  std::string n1TcpRttTraceFile = dir + "n1-tcp-rtt.dat";
  std::string n1TcpCwndTraceFile = dir + "n1-tcp-cwnd.dat";
  std::string n1TcpThroughputTraceFile = dir + "n1-tcp-throughput.dat";
  std::string dropTraceFile = dir + "drops.dat";
  std::string dropsFrequencyTraceFile = dir + "drops-frequency.dat";
  std::string lengthTraceFile = dir + "length.dat";
  std::string markTraceFile = dir + "mark.dat";
  std::string marksFrequencyTraceFile = dir + "marks-frequency.dat";
  std::string queueDelayN0TraceFile = dir + "queue-delay-n0.dat";
  std::string queueDelayN1TraceFile = dir + "queue-delay-n1.dat";

  ////////////////////////////////////////////////////////////
  // variables configured at command line                   //
  ////////////////////////////////////////////////////////////
  bool enablePcap = false;
  bool useCeThreshold = false;
  Time ceThreshold = MilliSeconds (1);
  std::string n0TcpType = "bic";
  std::string n1TcpType = "";
  bool enableN1Tcp = false;
  bool useEcn = true;
  std::string queueType = "fq";
  std::string linkDataRate = "1Gbps";
  uint32_t scenarioNum = 0;

  ////////////////////////////////////////////////////////////
  // Override ns-3 defaults                                 //
  ////////////////////////////////////////////////////////////
  Config::SetDefault ("ns3::TcpSocket::SegmentSize", UintegerValue (1448));
  // Increase default buffer sizes to improve throughput over long delay paths
  Config::SetDefault ("ns3::TcpSocket::SndBufSize",UintegerValue (8192000));
  Config::SetDefault ("ns3::TcpSocket::RcvBufSize",UintegerValue (8192000));
  Config::SetDefault ("ns3::TcpSocket::InitialCwnd", UintegerValue (10));
  Config::SetDefault ("ns3::TcpL4Protocol::RecoveryType", TypeIdValue (TcpPrrRecovery::GetTypeId ()));

  ////////////////////////////////////////////////////////////
  // command-line argument parsing                          //
  ////////////////////////////////////////////////////////////
  CommandLine cmd;
  cmd.AddValue ("n0TcpType", "n0 TCP type (bic, dctcp, or reno)", n0TcpType);
  cmd.AddValue ("n1TcpType", "n1 TCP type (bic, dctcp, or reno)", n1TcpType);
  cmd.AddValue ("scenarioNum", "Scenario number from the scenarios avalaible in the file (1-9)", scenarioNum);
  cmd.AddValue ("bottleneckQueueType", "n2 queue type (fq or codel)", queueType);
  cmd.AddValue ("baseRtt", "base RTT", baseRtt);
  cmd.AddValue ("useCeThreshold", "use CE Threshold", useCeThreshold);
  cmd.AddValue ("useEcn", "use ECN", useEcn);
  cmd.AddValue ("ceThreshold", "CoDel CE threshold", ceThreshold);
  cmd.AddValue ("bottleneckRate", "data rate of bottleneck", bottleneckRate);
  cmd.AddValue ("linkRate", "data rate of edge link", linkDataRate);
  cmd.AddValue ("stopTime", "simulation stop time", stopTime);
  cmd.AddValue ("enablePcap", "enable Pcap", enablePcap);
  cmd.AddValue ("pingTraceFile", "filename for ping tracing", pingTraceFile);
  cmd.AddValue ("n0TcpRttTraceFile", "filename for n0 rtt tracing", n0TcpRttTraceFile);
  cmd.AddValue ("n0TcpCwndTraceFile", "filename for n0 cwnd tracing", n0TcpCwndTraceFile);
  cmd.AddValue ("n0TcpThroughputTraceFile", "filename for n0 throughput tracing", n0TcpThroughputTraceFile);
  cmd.AddValue ("n1TcpRttTraceFile", "filename for n1 rtt tracing", n1TcpRttTraceFile);
  cmd.AddValue ("n1TcpCwndTraceFile", "filename for n1 cwnd tracing", n1TcpCwndTraceFile);
  cmd.AddValue ("n1TcpThroughputTraceFile", "filename for n1 throughput tracing", n1TcpThroughputTraceFile);
  cmd.AddValue ("dropTraceFile", "filename for n2 drops tracing", dropTraceFile);
  cmd.AddValue ("dropsFrequencyTraceFile", "filename for n2 drop frequency tracing", dropsFrequencyTraceFile);
  cmd.AddValue ("lengthTraceFile", "filename for n2 queue length tracing", lengthTraceFile);
  cmd.AddValue ("markTraceFile", "filename for n2 mark tracing", markTraceFile);
  cmd.AddValue ("marksFrequencyTraceFile", "filename for n2 mark frequency tracing", marksFrequencyTraceFile);
  cmd.AddValue ("queueDelayN0TraceFile", "filename for n0 queue delay tracing", queueDelayN0TraceFile);
  cmd.AddValue ("queueDelayN1TraceFile", "filename for n1 queue delay tracing", queueDelayN1TraceFile);
  cmd.Parse (argc, argv);
  Time oneWayDelay = baseRtt / 2;
  TypeId n0TcpTypeId;
  TypeId n1TcpTypeId;
  TypeId queueTypeId;
  if (!scenarioNum)
    {
      if (useEcn)
        {
          Config::SetDefault ("ns3::TcpSocketBase::UseEcn", StringValue ("On"));
        }

      if (n0TcpType == "reno")
        {
          n0TcpTypeId = TcpNewReno::GetTypeId ();
        }
      else if (n0TcpType == "bic")
        {
          n0TcpTypeId = TcpBic::GetTypeId ();
        }
      else if (n0TcpType == "dctcp")
        {
          n0TcpTypeId = TcpDctcp::GetTypeId ();
        }
      else
        {
          NS_FATAL_ERROR ("Fatal error:  tcp unsupported");
        }

      if (n1TcpType == "reno")
        {
          enableN1Tcp = true;
          n1TcpTypeId = TcpNewReno::GetTypeId ();
        }
      else if (n1TcpType == "bic")
        {
          enableN1Tcp = true;
          n1TcpTypeId = TcpBic::GetTypeId ();
        }
      else if (n1TcpType == "dctcp")
        {
          enableN1Tcp = true;
          n1TcpTypeId = TypeId::LookupByName ("ns3::TcpDctcp");
        }
      else if (n1TcpType == "")
        {
          NS_LOG_DEBUG ("No N1 TCP selected");
        }
      else
        {
          NS_FATAL_ERROR ("Fatal error:  tcp unsupported");
        }

      if (queueType == "fq")
        {
          queueTypeId = FqCoDelQueueDisc::GetTypeId ();
        }
      else if (queueType == "codel")
        {
          queueTypeId = CoDelQueueDisc::GetTypeId ();
        }
      else
        {
          NS_FATAL_ERROR ("Fatal error:  queueType unsupported");
        }
      if (useCeThreshold)
        {
          Config::SetDefault ("ns3::FqCoDelQueueDisc::CeThreshold", TimeValue (ceThreshold));
        }
    }
  else if (scenarioNum == 1 || scenarioNum == 2 || scenarioNum == 5 || scenarioNum == 6)
    {
      if (scenarioNum == 2 || scenarioNum == 6)
        {
          Config::SetDefault ("ns3::TcpSocketBase::UseEcn", StringValue ("On"));
        }
      n0TcpTypeId = TcpBic::GetTypeId ();
      if (scenarioNum == 5 || scenarioNum == 6)
        {
          enableN1Tcp = true;
          n1TcpTypeId = TcpBic::GetTypeId ();
        }
      queueTypeId = FqCoDelQueueDisc::GetTypeId ();
    }
  else if (scenarioNum == 3 || scenarioNum == 4 || scenarioNum == 7 || scenarioNum == 8 || scenarioNum == 9)
    {
      Config::SetDefault ("ns3::TcpSocketBase::UseEcn", StringValue ("On"));
      n0TcpTypeId = TcpDctcp::GetTypeId ();
      queueTypeId = FqCoDelQueueDisc::GetTypeId ();
      oneWayDelay = MicroSeconds (500);
      Config::SetDefault ("ns3::FqCoDelQueueDisc::CeThreshold", TimeValue (MilliSeconds (1)));
      if (scenarioNum == 9)
        {
          n0TcpTypeId = TcpBic::GetTypeId ();
          // For TCP Bic base RTT is 80 and base RTT for dctcp is set to 1 while setting delay for p2p devices
          oneWayDelay = MilliSeconds (40);
        }
      if (scenarioNum == 4 || scenarioNum == 8 || scenarioNum == 9)
        {
          Config::SetDefault ("ns3::FqCoDelQueueDisc::UseL4s", BooleanValue (true));
          Config::SetDefault ("ns3::TcpDctcp::UseEct0", BooleanValue (false));
        }
      if (scenarioNum == 7 || scenarioNum == 8 || scenarioNum == 9)
        {
          enableN1Tcp = true;
          n1TcpTypeId = TcpDctcp::GetTypeId ();
        }
    }
  else
    {
      NS_FATAL_ERROR ("Fatal error:  scenario unavailble");
    }

  std::ofstream pingOfStream;
  pingOfStream.open (pingTraceFile.c_str (), std::ofstream::out);
  std::ofstream n0TcpRttOfStream;
  n0TcpRttOfStream.open (n0TcpRttTraceFile.c_str (), std::ofstream::out);
  std::ofstream n0TcpCwndOfStream;
  n0TcpCwndOfStream.open (n0TcpCwndTraceFile.c_str (), std::ofstream::out);
  std::ofstream n0TcpThroughputOfStream;
  n0TcpThroughputOfStream.open (n0TcpThroughputTraceFile.c_str (), std::ofstream::out);
  std::ofstream n1TcpRttOfStream;
  n1TcpRttOfStream.open (n1TcpRttTraceFile.c_str (), std::ofstream::out);
  std::ofstream n1TcpCwndOfStream;
  n1TcpCwndOfStream.open (n1TcpCwndTraceFile.c_str (), std::ofstream::out);
  std::ofstream n1TcpThroughputOfStream;
  n1TcpThroughputOfStream.open (n1TcpThroughputTraceFile.c_str (), std::ofstream::out);

  // Queue disc files
  std::ofstream dropOfStream;
  dropOfStream.open (dropTraceFile.c_str (), std::ofstream::out);
  std::ofstream markOfStream;
  markOfStream.open (markTraceFile.c_str (), std::ofstream::out);
  std::ofstream dropsFrequencyOfStream;
  dropsFrequencyOfStream.open (dropsFrequencyTraceFile.c_str (), std::ofstream::out);
  std::ofstream marksFrequencyOfStream;
  marksFrequencyOfStream.open (marksFrequencyTraceFile.c_str (), std::ofstream::out);
  std::ofstream lengthOfStream;
  lengthOfStream.open (lengthTraceFile.c_str (), std::ofstream::out);
  std::ofstream queueDelayN0OfStream;
  queueDelayN0OfStream.open (queueDelayN0TraceFile.c_str (), std::ofstream::out);
  std::ofstream queueDelayN1OfStream;
  queueDelayN1OfStream.open (queueDelayN1TraceFile.c_str (), std::ofstream::out);

  ////////////////////////////////////////////////////////////
  // scenario setup                                         //
  ////////////////////////////////////////////////////////////
  Ptr<Node> pingServer = CreateObject<Node> ();
  Ptr<Node> n0Server = CreateObject<Node> ();
  Ptr<Node> n1Server = CreateObject<Node> ();
  Ptr<Node> n2 = CreateObject<Node> ();
  Ptr<Node> n3 = CreateObject<Node> ();
  Ptr<Node> pingClient = CreateObject<Node> ();
  Ptr<Node> n4Client = CreateObject<Node> ();
  Ptr<Node> n5Client = CreateObject<Node> ();

  // Device containers
  NetDeviceContainer pingServerDevices;
  NetDeviceContainer n0ServerDevices;
  NetDeviceContainer n1ServerDevices;
  NetDeviceContainer n2n3Devices;
  NetDeviceContainer pingClientDevices;
  NetDeviceContainer n4ClientDevices;
  NetDeviceContainer n5ClientDevices;

  PointToPointHelper p2p;
  p2p.SetQueue ("ns3::DropTailQueue", "MaxSize", QueueSizeValue (QueueSize ("3p")));
  p2p.SetDeviceAttribute ("DataRate", DataRateValue (DataRate (linkDataRate)));
  // Add delay only on the server links
  p2p.SetChannelAttribute ("Delay", TimeValue (oneWayDelay));
  pingServerDevices = p2p.Install (n2, pingServer);
  n0ServerDevices = p2p.Install (n2, n0Server);

  // In scenario 9, base RTT of n1server (dctcp) is 1ms
  if (scenarioNum == 9)
    {
      p2p.SetChannelAttribute ("Delay", TimeValue (MicroSeconds (500)));
    }
  n1ServerDevices = p2p.Install (n2, n1Server);
  p2p.SetChannelAttribute ("Delay", TimeValue (MicroSeconds (1)));
  n2n3Devices = p2p.Install (n2, n3);
  pingClientDevices = p2p.Install (n3, pingClient);
  n4ClientDevices = p2p.Install (n3, n4Client);
  n5ClientDevices = p2p.Install (n3, n5Client);
  Ptr<PointToPointNetDevice> p = n2n3Devices.Get (0)->GetObject<PointToPointNetDevice> ();
  p->SetAttribute ("DataRate", DataRateValue (bottleneckRate));

  InternetStackHelper stackHelper;
  stackHelper.InstallAll ();

  // Set the per-node TCP type here
  Ptr<TcpL4Protocol> proto;
  proto = n4Client->GetObject<TcpL4Protocol> ();
  proto->SetAttribute ("SocketType", TypeIdValue (n0TcpTypeId));
  proto = n0Server->GetObject<TcpL4Protocol> ();
  proto->SetAttribute ("SocketType", TypeIdValue (n0TcpTypeId));
  if (enableN1Tcp)
    {
      proto = n5Client->GetObject<TcpL4Protocol> ();
      proto->SetAttribute ("SocketType", TypeIdValue (n1TcpTypeId));
      proto = n1Server->GetObject<TcpL4Protocol> ();
      proto->SetAttribute ("SocketType", TypeIdValue (n1TcpTypeId));
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
  tchFq.Install (n0ServerDevices);
  tchFq.Install (n1ServerDevices);
  tchFq.Install (n2n3Devices.Get (1));  // n2 queue for bottleneck link
  tchFq.Install (pingClientDevices);
  tchFq.Install (n4ClientDevices);
  tchFq.Install (n5ClientDevices);
  TrafficControlHelper tchN2;
  tchN2.SetRootQueueDisc (queueTypeId.GetName ());
  tchN2.SetQueueLimits ("ns3::DynamicQueueLimits", "HoldTime", StringValue ("1000ms"));
  tchN2.Install (n2n3Devices.Get (0));

  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer pingServerIfaces = ipv4.Assign (pingServerDevices);
  ipv4.SetBase ("10.1.2.0", "255.255.255.0");
  Ipv4InterfaceContainer n0ServerIfaces = ipv4.Assign (n0ServerDevices);
  ipv4.SetBase ("10.1.3.0", "255.255.255.0");
  Ipv4InterfaceContainer secondServerIfaces = ipv4.Assign (n1ServerDevices);
  ipv4.SetBase ("172.16.1.0", "255.255.255.0");
  Ipv4InterfaceContainer n2n3Ifaces = ipv4.Assign (n2n3Devices);
  ipv4.SetBase ("192.168.1.0", "255.255.255.0");
  Ipv4InterfaceContainer pingClientIfaces = ipv4.Assign (pingClientDevices);
  ipv4.SetBase ("192.168.2.0", "255.255.255.0");
  Ipv4InterfaceContainer n4ClientIfaces = ipv4.Assign (n4ClientDevices);
  ipv4.SetBase ("192.168.3.0", "255.255.255.0");
  Ipv4InterfaceContainer n5ClientIfaces = ipv4.Assign (n5ClientDevices);

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

  BulkSendHelper tcp ("ns3::TcpSocketFactory", Address ());
  // set to large value:  e.g. 1000 Mb/s for 60 seconds = 7500000000 bytes
  tcp.SetAttribute ("MaxBytes", UintegerValue (7500000000));
  // Configure n4/n0 TCP client/server pair
  uint16_t n4Port = 5000;
  ApplicationContainer n0App;
  InetSocketAddress n0DestAddress (n4ClientIfaces.GetAddress (1), n4Port);
  tcp.SetAttribute ("Remote", AddressValue (n0DestAddress));
  n0App = tcp.Install (n0Server);
  n0App.Start (Seconds (5));
  n0App.Stop (stopTime - Seconds (1));

  Address n4SinkAddress (InetSocketAddress (Ipv4Address::GetAny (), n4Port));
  PacketSinkHelper n4SinkHelper ("ns3::TcpSocketFactory", n4SinkAddress);
  ApplicationContainer n4SinkApp;
  n4SinkApp = n4SinkHelper.Install (n4Client);
  n4SinkApp.Start (Seconds (5));
  n4SinkApp.Stop (stopTime - MilliSeconds (500));

  // Configure second TCP client/server pair
  if (enableN1Tcp)
    {
      uint16_t n5Port = 5000;
      ApplicationContainer secondApp;
      InetSocketAddress n1DestAddress (n5ClientIfaces.GetAddress (1), n5Port);
      tcp.SetAttribute ("Remote", AddressValue (n1DestAddress));
      secondApp = tcp.Install (n1Server);
      secondApp.Start (Seconds (15));
      secondApp.Stop (stopTime - Seconds (1));

      Address n5SinkAddress (InetSocketAddress (Ipv4Address::GetAny (), n5Port));
      PacketSinkHelper n5SinkHelper ("ns3::TcpSocketFactory", n5SinkAddress);
      ApplicationContainer n5SinkApp;
      n5SinkApp = n5SinkHelper.Install (n5Client);
      n5SinkApp.Start (Seconds (15));
      n5SinkApp.Stop (stopTime - MilliSeconds (500));
    }

  // Setup traces that can be hooked now
  Ptr<TrafficControlLayer> tc;
  Ptr<QueueDisc> qd;
  tc = n2n3Devices.Get (0)->GetNode ()->GetObject<TrafficControlLayer> ();
  qd = tc->GetRootQueueDiscOnDevice (n2n3Devices.Get (0));
  qd->TraceConnectWithoutContext ("Drop", MakeBoundCallback (&TraceDrop, &dropOfStream));
  qd->TraceConnectWithoutContext ("Mark", MakeBoundCallback (&TraceMark, &markOfStream));
  qd->TraceConnectWithoutContext ("BytesInQueue", MakeBoundCallback (&TraceQueueLength, &lengthOfStream, bottleneckRate));
  qd->TraceConnectWithoutContext ("Dequeue", MakeBoundCallback (&PacketDequeue, &queueDelayN0OfStream, &queueDelayN1OfStream));

  // Setup scheduled traces; TCP traces must be hooked after socket creation
  Simulator::Schedule (Seconds (5) + MilliSeconds (100), &ScheduleN0TcpRttTraceConnection, &n0TcpRttOfStream);
  Simulator::Schedule (Seconds (5) + MilliSeconds (100), &ScheduleN0TcpCwndTraceConnection, &n0TcpCwndOfStream);
  Simulator::Schedule (Seconds (5) + MilliSeconds (100), &ScheduleN0PacketSinkConnection);
  Simulator::Schedule (throughputSamplingInterval, &TraceN0Throughput, &n0TcpThroughputOfStream, throughputSamplingInterval);
  // Setup scheduled traces; TCP traces must be hooked after socket creation
  if (enableN1Tcp)
    {
      Simulator::Schedule (Seconds (15) + MilliSeconds (100), &ScheduleN1TcpRttTraceConnection, &n1TcpRttOfStream);
      Simulator::Schedule (Seconds (15) + MilliSeconds (100), &ScheduleN1TcpCwndTraceConnection, &n1TcpCwndOfStream);
      Simulator::Schedule (Seconds (15) + MilliSeconds (100), &ScheduleN1PacketSinkConnection);
    }
  Simulator::Schedule (throughputSamplingInterval, &TraceN1Throughput, &n1TcpThroughputOfStream, throughputSamplingInterval);
  Simulator::Schedule (marksSamplingInterval, &TraceMarksFrequency, &marksFrequencyOfStream, marksSamplingInterval);
  Simulator::Schedule (marksSamplingInterval, &TraceDropsFrequency, &dropsFrequencyOfStream, marksSamplingInterval);

  if (enablePcap)
    {
      p2p.EnablePcapAll ("FqCoDel-L4S-example", false);
    }

  Simulator::Stop (stopTime);
  Simulator::Run ();

  pingOfStream.close ();
  n0TcpCwndOfStream.close ();
  n0TcpRttOfStream.close ();
  n0TcpThroughputOfStream.close ();
  n1TcpCwndOfStream.close ();
  n1TcpRttOfStream.close ();
  n1TcpThroughputOfStream.close ();
  dropOfStream.close ();
  markOfStream.close ();
  dropsFrequencyOfStream.close ();
  marksFrequencyOfStream.close ();
  lengthOfStream.close ();
  queueDelayN0OfStream.close ();
  queueDelayN1OfStream.close ();
}
