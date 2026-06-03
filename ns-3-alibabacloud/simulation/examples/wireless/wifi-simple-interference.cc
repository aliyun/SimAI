/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 The Boeing Company
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

// This script configures three nodes on an 802.11b physical layer, with
// 802.11b NICs in adhoc mode.  There is a transmitter, receiver, and
// interferer.  The transmitter sends one packet to the receiver and
// the receiver receives it with a certain configurable RSS (by default,
// -80 dBm).  The interferer does not do carrier sense and also sends
// the packet to interfere with the primary packet.  The channel model
// is clear channel.
//
// Therefore, at the receiver, the reception looks like this:
//
//     ------------------time---------------->
//     t0
//
//     |------------------------------------|
//     |                                    |
//     | primary received frame (time t0)   |
//     |                                    |
//     |------------------------------------|
//
//
//         t1
//         |-----------------------------------|
//         |                                   |
//         |  interfering frame (time t1)      |
//         |                                   |
//         |-----------------------------------|
//
// The orientation is:
//     n2  ---------> n0 <---------- n1
//  interferer      receiver       transmitter
//
// The configurable parameters are:
//   - Prss (primary rss) (-80 dBm default)
//   - Irss (interfering rss) (-95 dBm default)
//   - delta (microseconds, (t1-t0), may be negative, default 0)
//   - PpacketSize (primary packet size) (bytes, default 1000)
//   - IpacketSize (interferer packet size) (bytes, default 1000)
//
// For instance, for this configuration, the interfering frame arrives
// at -90 dBm with a time offset of 3.2 microseconds:
//
// ./ns3 run "wifi-simple-interference --Irss=-90 --delta=3.2"
//
// Note that all ns-3 attributes (not just the ones exposed in the below
// script) can be changed at command line; see the documentation.
//
// This script can also be helpful to put the Wifi layer into verbose
// logging mode; this command will turn on all wifi logging:
//
// ./ns3 run "wifi-simple-interference --verbose=1"
//
// When you are done, you will notice a pcap trace file in your directory.
// If you have tcpdump installed, you can try this:
//
// tcpdump -r wifi-simple-interference-0-0.pcap -nn -tt
// reading from file wifi-simple-interference-0-0.pcap, link-type IEEE802_11_RADIO (802.11 plus BSD radio information header)
// 10.008704 10008704us tsft 1.0 Mb/s 2437 MHz (0x00c0) -80dB signal -98dB noise IP 10.1.1.2.49153 > 10.1.1.255.80: UDP, length 1000
//
// Next, try this command and look at the tcpdump-- you should see two packets
// that are no longer interfering:
// ./ns3 run "wifi-simple-interference --delta=30000"

#include "ns3/command-line.h"
#include "ns3/config.h"
#include "ns3/double.h"
#include "ns3/string.h"
#include "ns3/log.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/ssid.h"
#include "ns3/mobility-helper.h"
#include "ns3/yans-wifi-channel.h"
#include "ns3/mobility-model.h"
#include "ns3/internet-stack-helper.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("WifiSimpleInterference");

static inline std::string PrintReceivedPacket (Ptr<Socket> socket)
{
  Address addr;

  std::ostringstream oss;

  while (socket->Recv ())
    {
      socket->GetSockName (addr);
      InetSocketAddress iaddr = InetSocketAddress::ConvertFrom (addr);

      oss << "Received one packet!  Socket: " << iaddr.GetIpv4 () << " port: " << iaddr.GetPort ();
    }

  return oss.str ();
}

static void ReceivePacket (Ptr<Socket> socket)
{
  NS_LOG_UNCOND (PrintReceivedPacket (socket));
}

static void GenerateTraffic (Ptr<Socket> socket, uint32_t pktSize,
                             uint32_t pktCount, Time pktInterval )
{
  if (pktCount > 0)
    {
      socket->Send (Create<Packet> (pktSize));
      Simulator::Schedule (pktInterval, &GenerateTraffic,
                           socket, pktSize,pktCount - 1, pktInterval);
    }
  else
    {
      socket->Close ();
    }
}

int main (int argc, char *argv[])
{
  std::string phyMode ("DsssRate1Mbps");
  double Prss = -80;  // -dBm
  double Irss = -95;  // -dBm
  double delta = 0;  // microseconds
  uint32_t PpacketSize = 1000; // bytes
  uint32_t IpacketSize = 1000; // bytes
  bool verbose = false;

  // these are not command line arguments for this version
  uint32_t numPackets = 1;
  double interval = 1.0; // seconds
  double startTime = 10.0; // seconds
  double distanceToRx = 100.0; // meters

  double offset = 91;  // This is a magic number used to set the
                       // transmit power, based on other configuration
  CommandLine cmd (__FILE__);
  cmd.AddValue ("phyMode", "Wifi Phy mode", phyMode);
  cmd.AddValue ("Prss", "Intended primary received signal strength (dBm)", Prss);
  cmd.AddValue ("Irss", "Intended interfering received signal strength (dBm)", Irss);
  cmd.AddValue ("delta", "time offset (microseconds) for interfering signal", delta);
  cmd.AddValue ("PpacketSize", "size of application packet sent", PpacketSize);
  cmd.AddValue ("IpacketSize", "size of interfering packet sent", IpacketSize);
  cmd.AddValue ("verbose", "turn on all WifiNetDevice log components", verbose);
  cmd.Parse (argc, argv);
  // Convert to time object
  Time interPacketInterval = Seconds (interval);

  // Fix non-unicast data rate to be the same as that of unicast
  Config::SetDefault ("ns3::WifiRemoteStationManager::NonUnicastMode",
                      StringValue (phyMode));

  NodeContainer c;
  c.Create (3);

  // The below set of helpers will help us to put together the wifi NICs we want
  WifiHelper wifi;
  if (verbose)
    {
      wifi.EnableLogComponents ();  // Turn on all Wifi logging
    }
  wifi.SetStandard (WIFI_STANDARD_80211b);

  YansWifiPhyHelper wifiPhy;

  // ns-3 supports RadioTap and Prism tracing extensions for 802.11b
  wifiPhy.SetPcapDataLinkType (WifiPhyHelper::DLT_IEEE802_11_RADIO);

  YansWifiChannelHelper wifiChannel;
  wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
  wifiChannel.AddPropagationLoss ("ns3::LogDistancePropagationLossModel");
  wifiPhy.SetChannel (wifiChannel.Create ());

  // Add a mac and disable rate control
  WifiMacHelper wifiMac;
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode",StringValue (phyMode),
                                "ControlMode",StringValue (phyMode));
  // Set it to adhoc mode
  wifiMac.SetType ("ns3::AdhocWifiMac");
  NetDeviceContainer devices = wifi.Install (wifiPhy, wifiMac, c.Get (0));
  // This will disable these sending devices from detecting a signal
  // so that they do not backoff
  wifiPhy.Set ("TxGain", DoubleValue (offset + Prss) );
  devices.Add (wifi.Install (wifiPhy, wifiMac, c.Get (1)));
  wifiPhy.Set ("TxGain", DoubleValue (offset + Irss) );
  devices.Add (wifi.Install (wifiPhy, wifiMac, c.Get (2)));

  // Note that with FixedRssLossModel, the positions below are not
  // used for received signal strength.
  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (distanceToRx, 0.0, 0.0));
  positionAlloc->Add (Vector (-1 * distanceToRx, 0.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (c);

  InternetStackHelper internet;
  internet.Install (c);

  TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
  Ptr<Socket> recvSink = Socket::CreateSocket (c.Get (0), tid);
  InetSocketAddress local = InetSocketAddress (Ipv4Address ("10.1.1.1"), 80);
  recvSink->Bind (local);
  recvSink->SetRecvCallback (MakeCallback (&ReceivePacket));

  Ptr<Socket> source = Socket::CreateSocket (c.Get (1), tid);
  InetSocketAddress remote = InetSocketAddress (Ipv4Address ("255.255.255.255"), 80);
  source->SetAllowBroadcast (true);
  source->Connect (remote);

  // Interferer will send to a different port; we will not see a
  // "Received packet" message
  Ptr<Socket> interferer = Socket::CreateSocket (c.Get (2), tid);
  InetSocketAddress interferingAddr = InetSocketAddress (Ipv4Address ("255.255.255.255"), 49000);
  interferer->SetAllowBroadcast (true);
  interferer->Connect (interferingAddr);

  // Tracing
  wifiPhy.EnablePcap ("wifi-simple-interference", devices.Get (0));

  // Output what we are doing
  NS_LOG_UNCOND ("Primary packet RSS=" << Prss << " dBm and interferer RSS=" << Irss << " dBm at time offset=" << delta << " ms");

  Simulator::ScheduleWithContext (source->GetNode ()->GetId (),
                                  Seconds (startTime), &GenerateTraffic,
                                  source, PpacketSize, numPackets, interPacketInterval);

  Simulator::ScheduleWithContext (interferer->GetNode ()->GetId (),
                                  Seconds (startTime + delta / 1000000.0), &GenerateTraffic,
                                  interferer, IpacketSize, numPackets, interPacketInterval);

  Simulator::Run ();
  Simulator::Destroy ();

  return 0;
}
