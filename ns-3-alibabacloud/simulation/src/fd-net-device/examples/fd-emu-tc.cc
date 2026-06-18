/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Universita' degli Studi di Napoli Federico II
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
 * Author: Pasquale Imputato <p.imputato@gmail.com>
 */

/*
 *                                          node
 *                     --------------------------------------------------
 *                     |                                                |
 *                     |        pfifo_fast    queueDiscType [pfifo_fast]|
 *                     |                      bql [false]               |
 *                     |        interface 0   interface 1               |
 *                     |             |             |                    |
 *                     --------------------------------------------------
 *                                   |             |
 *   1 Gbps access incoming link     |             |           100 Mbps bottleneck outgoing link
 * -----------------------------------             -----------------------------------
 *
 * This example builds a node with two interfaces in emulation mode in 
 * either {raw, netmap}. The aim is to explore different qdiscs behaviours 
 * on the backlog of a device emulated bottleneck side.
 *
 * If you run emulation in netmap mode, you need before to load the 
 * netmap.ko module.  The user is responsible for configuring and building
 * netmap separately.
 */

#include "ns3/abort.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/fd-net-device-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/ipv4-list-routing-helper.h"
#include "ns3/traffic-control-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TrafficControlEmu");

void
TcPacketsInQueue (Ptr<QueueDisc> q, Ptr<OutputStreamWrapper> stream)
{
  Simulator::Schedule (Seconds (0.001), &TcPacketsInQueue, q, stream);
  *stream->GetStream () << Simulator::Now ().GetSeconds () << " backlog " << q->GetNPackets () << "p " << q->GetNBytes () << "b " << " dropped "
                        << q->GetStats ().nTotalDroppedPackets << "p " << q->GetStats ().nTotalDroppedBytes << "b " << std::endl;
}

#ifdef HAVE_NETMAP_USER_H
void
Inflight (Ptr<NetmapNetDevice> dev, Ptr<OutputStreamWrapper> stream)
{
  Simulator::Schedule (Seconds (0.001), &Inflight, dev, stream);
  *stream->GetStream () << dev->GetBytesInNetmapTxRing () << std::endl;
}
#endif

int
main (int argc, char *argv[])
{
  NS_LOG_INFO ("Traffic Control Emulation Example");

  std::string deviceName0 ("enx503f56005a2a");
  std::string deviceName1 ("eno1");
  std::string ip0 ("10.0.1.2");
  std::string ip1 ("10.0.2.1");
  std::string mask0 ("255.255.255.0");
  std::string mask1 ("255.255.255.0");
  std::string m0 ("00:00:00:aa:00:01");
  std::string m1 ("00:00:00:aa:00:02");
  std::string queueDiscType = "PfifoFast";
  bool bql = false;

  uint32_t index = 0;
  bool writer = true;
#ifdef HAVE_PACKET_H
  std::string emuMode ("raw");
#else        // HAVE_NETMAP_USER_H is true (otherwise this example is not compiled)
  std::string emuMode ("netmap");
#endif

  CommandLine cmd;
  cmd.AddValue ("deviceName0", "Device name", deviceName0);
  cmd.AddValue ("deviceName1", "Device name", deviceName1);
  cmd.AddValue ("ip0", "Local IP address", ip0);
  cmd.AddValue ("ip1", "Local IP address", ip1);
  cmd.AddValue ("mask0", "Local mask", mask0);
  cmd.AddValue ("mask1", "Local mask", mask1);
  cmd.AddValue ("m0", "Mac address", m0);
  cmd.AddValue ("m1", "Mac address", m1);
  cmd.AddValue ("writer", "Enable write stats", writer);
  cmd.AddValue ("queueDiscType", "Bottleneck queue disc type in {PfifoFast, ARED, CoDel, FqCoDel, PIE}", queueDiscType);
  cmd.AddValue ("bql", "Enable byte queue limits on bottleneck netdevice", bql);
  cmd.AddValue ("index", "Experiment index", index);
  cmd.AddValue ("emuMode", "Emulation mode in {raw, netmap}", emuMode);
  cmd.Parse (argc, argv);

  Ipv4Address localIp0 (ip0.c_str ());
  Ipv4Address localIp1 (ip1.c_str ());
  Ipv4Mask netmask0 (mask0.c_str ());
  Ipv4Mask netmask1 (mask1.c_str ());
  Mac48AddressValue mac0 (m0.c_str ());
  Mac48AddressValue mac1 (m1.c_str ());

  //
  // Since we are using a real piece of hardware we need to use the realtime
  // simulator.
  //
  GlobalValue::Bind ("SimulatorImplementationType", StringValue ("ns3::RealtimeSimulatorImpl"));

  //
  // Since we are going to be talking to real-world machines, we need to enable
  // calculation of checksums in our protocols.
  //
  GlobalValue::Bind ("ChecksumEnabled", BooleanValue (true));

  //
  // In such a simple topology, the use of the helper API can be a hindrance
  // so we drop down into the low level API and do it manually.
  //
  // First we need a single node.
  //
  NS_LOG_INFO ("Create Node");
  Ptr<Node> node = CreateObject<Node> ();

  //
  // Create an emu device, allocate a MAC address and point the device to the
  // Linux device name.  The device needs a transmit queueing discipline so
  // create a droptail queue and give it to the device.  Finally, "install"
  // the device into the node.
  //
  // Do understand that the ns-3 allocated MAC address will be sent out over
  // your network since the emu net device will spoof it.  By default, this
  // address will have an Organizationally Unique Identifier (OUI) of zero.
  // The Internet Assigned Number Authority IANA
  //
  //  http://www.iana.org/assignments/ethernet-numberslocalIp
  //
  // reports that this OUI is unassigned, and so should not conflict with
  // real hardware on your net.  It may raise all kinds of red flags in a
  // real environment to have packets from a device with an obviously bogus
  // OUI flying around.  Be aware.
  //

  NS_LOG_INFO ("Create Devices");

  FdNetDeviceHelper *helper0 = nullptr;
  FdNetDeviceHelper *helper1 = nullptr;

#ifdef HAVE_PACKET_H
  if (emuMode == "raw")
    {
      EmuFdNetDeviceHelper *raw0 = new EmuFdNetDeviceHelper;
      raw0->SetDeviceName (deviceName0);
      helper0 = raw0;

      EmuFdNetDeviceHelper *raw1 = new EmuFdNetDeviceHelper;
      raw1->SetDeviceName (deviceName1);
      helper1 = raw1;
    }
#endif
#ifdef HAVE_NETMAP_USER_H
  if (emuMode == "netmap")
    {
      NetmapNetDeviceHelper *netmap0 = new NetmapNetDeviceHelper;
      netmap0->SetDeviceName (deviceName0);
      helper0 = netmap0;

      NetmapNetDeviceHelper *netmap1 = new NetmapNetDeviceHelper;
      netmap1->SetDeviceName (deviceName1);
      helper1 = netmap1;
    }
#endif

  if ((helper0 == nullptr) || (helper1 == nullptr))
    {
      NS_ABORT_MSG (emuMode << " not supported.");
    }

  NetDeviceContainer devices0 = helper0->Install (node);
  NetDeviceContainer devices1 = helper1->Install (node);

  Ptr<NetDevice> device0 = devices0.Get (0);
  device0->SetAttribute ("Address", mac0);

  Ptr<NetDevice> device1 = devices1.Get (0);
  device1->SetAttribute ("Address", mac1);

  //
  // Add a default internet stack to the node.  This gets us the ns-3 versions
  // of ARP, IPv4, ICMP, UDP and TCP.
  //
  NS_LOG_INFO ("Add Internet Stack");
  InternetStackHelper internetStackHelper;
  internetStackHelper.Install (node);

  Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();

  NS_LOG_INFO ("Create IPv4 Interface 0");
  uint32_t interface0 = ipv4->AddInterface (device0);
  Ipv4InterfaceAddress address0 = Ipv4InterfaceAddress (localIp0, netmask0);
  ipv4->AddAddress (interface0, address0);
  ipv4->SetMetric (interface0, 0);
  ipv4->SetForwarding (interface0, true);
  ipv4->SetUp (interface0);

  NS_LOG_INFO ("Create IPv4 Interface 1");
  uint32_t interface1 = ipv4->AddInterface (device1);
  Ipv4InterfaceAddress address1 = Ipv4InterfaceAddress (localIp1, netmask1);
  ipv4->AddAddress (interface1, address1);
  ipv4->SetMetric (interface1, 0);
  ipv4->SetForwarding (interface1, true);
  ipv4->SetUp (interface1);

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  // Traffic control configurations
  // Access link side
  TrafficControlHelper tch0;
  tch0.SetRootQueueDisc ("ns3::PfifoFastQueueDisc", "MaxSize", StringValue ("1000p"));
  tch0.Install (devices0);

  // Bottleneck link side
  TrafficControlHelper tch1;

  if (queueDiscType.compare ("PfifoFast") == 0)
    {
      tch1.SetRootQueueDisc ("ns3::PfifoFastQueueDisc", "MaxSize", StringValue ("1000p"));
    }
  else if (queueDiscType.compare ("ARED") == 0)
    {
      tch1.SetRootQueueDisc ("ns3::RedQueueDisc");
      Config::SetDefault ("ns3::RedQueueDisc::ARED", BooleanValue (true));
      Config::SetDefault ("ns3::RedQueueDisc::MaxSize",
                          QueueSizeValue (QueueSize (QueueSizeUnit::BYTES, 1000000)));
      Config::SetDefault ("ns3::RedQueueDisc::MeanPktSize", UintegerValue (1000));
      Config::SetDefault ("ns3::RedQueueDisc::LinkBandwidth", StringValue ("100Mbps"));
      Config::SetDefault ("ns3::RedQueueDisc::MinTh", DoubleValue (83333));
      Config::SetDefault ("ns3::RedQueueDisc::MinTh", DoubleValue (250000));
    }
  else if (queueDiscType.compare ("CoDel") == 0)
    {
      tch1.SetRootQueueDisc ("ns3::CoDelQueueDisc");
    }
  else if (queueDiscType.compare ("FqCoDel") == 0)
    {
      tch1.SetRootQueueDisc ("ns3::FqCoDelQueueDisc");
    }
  else if (queueDiscType.compare ("PIE") == 0)
    {
      tch1.SetRootQueueDisc ("ns3::PieQueueDisc");
      Config::SetDefault ("ns3::PieQueueDisc::MaxSize",
                          QueueSizeValue (QueueSize (QueueSizeUnit::PACKETS, 1000)));
      Config::SetDefault ("ns3::PieQueueDisc::QueueDelayReference", TimeValue (Seconds (0.02)));
      Config::SetDefault ("ns3::PieQueueDisc::Tupdate", TimeValue (Seconds (0.032)));
      Config::SetDefault ("ns3::PieQueueDisc::A", DoubleValue (2));
      Config::SetDefault ("ns3::PieQueueDisc::B", DoubleValue (20));
    }
  else
    {
      NS_ABORT_MSG ("--queueDiscType not valid");
    }

  if (bql)
    {
      tch1.SetQueueLimits ("ns3::DynamicQueueLimits");
    }

  QueueDiscContainer qdiscs = tch1.Install (devices1);
  Ptr<QueueDisc> q = qdiscs.Get (0);

  if (writer)
    {
      AsciiTraceHelper ascii;
      Ptr<OutputStreamWrapper> stream = ascii.CreateFileStream ("exp-" + std::to_string (index) + "-router-tc-stats.txt");
      Simulator::Schedule (Seconds (0.001), &TcPacketsInQueue, q, stream);

#ifdef HAVE_NETMAP_USER_H
      if (emuMode.compare ("netmap") == 0)
        {
          Ptr<NetmapNetDevice> dev = StaticCast <NetmapNetDevice> (device1);
          Ptr<OutputStreamWrapper> stream2 = ascii.CreateFileStream ("exp-" + std::to_string (index) + "-router-inflight.txt");
          Simulator::Schedule (Seconds (0.001), &Inflight, dev, stream2);
        }
#endif

      Ipv4GlobalRoutingHelper g;
      Ptr<OutputStreamWrapper> routingStream = Create<OutputStreamWrapper> ("router-routes.txt", std::ios::out);
      g.PrintRoutingTableAllAt (Seconds (3), routingStream);
    }

  NS_LOG_INFO ("Run Emulation.");
  Simulator::Stop (Seconds (50.0));
  Simulator::Run ();
  Simulator::Destroy ();
  delete helper0;
  delete helper1;
  NS_LOG_INFO ("Done.");
}
