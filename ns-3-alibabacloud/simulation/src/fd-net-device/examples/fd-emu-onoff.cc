/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 University of Washington, 2012 INRIA
 *               2017 Università' degli Studi di Napoli Federico II
 *               2019 NITK Surathkal
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
 * Author: Alina Quereilhac <alina.quereilhac@inria.fr>
 * Extended by: Pasquale Imputato <p.imputato@gmail.com>
 *              Harsh Patel <thadodaharsh10@gmail.com>
 *              Hrishikesh Hiraskar <hrishihiraskar@gmail.com>
 *              Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 *
 */

// +----------------------+     +-----------------------+
// |      client host     |     |      server host      |
// +----------------------+     +-----------------------+
// |     ns-3 Node 0      |     |      ns-3 Node 1      |
// |  +----------------+  |     |   +----------------+  |
// |  |    ns-3 TCP    |  |     |   |    ns-3 TCP    |  |
// |  +----------------+  |     |   +----------------+  |
// |  |    ns-3 IPv4   |  |     |   |    ns-3 IPv4   |  |
// |  +----------------+  |     |   +----------------+  |
// |  |   FdNetDevice  |  |     |   |   FdNetDevice  |  |
// |  |       or       |  |     |   |       or       |  |
// |  | NetmapNetDevice|  |     |   | NetmapNetDevice|  |
// |  |       or       |  |     |   |       or       |  |
// |  |  DpdkNetDevice |  |     |   |  DpdkNetDevice |  |
// |  |    10.1.1.1    |  |     |   |    10.1.1.2    |  |
// |  +----------------+  |     |   +----------------+  |
// |  |       fd       |  |     |   |       fd       |  |
// |  |       or       |  |     |   |       or       |  |
// |  |       EAL      |  |     |   |       EAL      |  |
// |  +----------------+  |     |   +----------------+  |
// |    |    eth0    |    |     |     |    eth0    |    |
// |    |     or     |    |     |     |     or     |    |
// |    | 0000:00.1f |    |     |     | 0000:00.1f |    |
// +----+------------+----+     +-----+------------+----+
//
//         10.1.1.11                     10.1.1.12
//
//             |                            |
//             +----------------------------+
//
// This example is aimed at measuring the throughput of the FdNetDevice
// when using the EmuFdNetDeviceHelper. This is achieved by saturating
// the channel with TCP traffic. Then the throughput can be obtained from
// the generated .pcap files.
//
// To run this example you will need two hosts (client & server).
// Steps to run the experiment:
//
// 1 - Connect the 2 computers with an Ethernet cable.
// 2 - Set the IP addresses on both Ethernet devices.
//
// client machine: $ sudo ip addr add dev eth0 10.1.1.11/24
// server machine: $ sudo ip addr add dev eth0 10.1.1.12/24
//
// 3 - Set both Ethernet devices to promiscuous mode.
//
// both machines: $ sudo ip link set eth0 promisc on
//
// 3' - If you run emulation in netmap or dpdk mode, you need before to load
//      the netmap.ko or dpdk modules. The user is in charge to configure and
//      build netmap/dpdk separately.
//
// 4 - Give root suid to the raw or netmap socket creator binary.
//     If the --enable-sudo option was used to configure ns-3 with ns3, then the following
//     step will not be necessary.
//
// both hosts: $ sudo chown root.root build/src/fd-net-device/ns3-dev-raw-sock-creator
// both hosts: $ sudo chmod 4755 build/src/fd-net-device/ns3-dev-raw-sock-creator
//
// or (if you run emulation in netmap mode):
// both hosts: $ sudo chown root.root build/src/fd-net-device/ns3-dev-netmap-device-creator
// both hosts: $ sudo chmod 4755 build/src/fd-net-device/ns3-dev-netmap-device-creator
//
// 4' - If you run emulation in dpdk mode, you will need to run example as root.
//
// 5 - In case of DpdkNetDevice, use device address instead of device name
//
// For example: 0000:00:1f.6 (can be obtained by lspci)
//
// 6 - Run the server side:
//
// server host: $ ./ns3 run="fd-emu-onoff --serverMode=1"
//
// 7 - Run the client side:
//
// client host: $ ./ns3 run="fd-emu-onoff"
//

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/config-store-module.h"
#include "ns3/fd-net-device-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("EmuFdNetDeviceSaturationExample");

int
main (int argc, char *argv[])
{
  uint16_t sinkPort = 8000;
  uint32_t packetSize = 1400; // bytes
  std::string dataRate ("1000Mb/s");
  bool serverMode = false;

  std::string deviceName ("eth0");
  std::string client ("10.1.1.1");
  std::string server ("10.1.1.2");
  std::string netmask ("255.255.255.0");
  std::string macClient ("00:00:00:00:00:01");
  std::string macServer ("00:00:00:00:00:02");
  std::string transportProt = "Tcp";
  std::string socketType;
#ifdef HAVE_PACKET_H
  std::string emuMode ("raw");
#elif HAVE_NETMAP_USER_H
  std::string emuMode ("netmap");
#else        // HAVE_DPDK_USER_H is true (otherwise this example is not compiled)
  std::string emuMode ("dpdk");
#endif

  CommandLine cmd (__FILE__);
  cmd.AddValue ("deviceName", "Device name (in raw, netmap mode) or Device address (in dpdk mode, eg: 0000:00:1f.6). Use `lspci` to find device address.", deviceName);
  cmd.AddValue ("client", "Local IP address (dotted decimal only please)", client);
  cmd.AddValue ("server", "Remote IP address (dotted decimal only please)", server);
  cmd.AddValue ("localmask", "Local mask address (dotted decimal only please)", netmask);
  cmd.AddValue ("serverMode", "1:true, 0:false, default client", serverMode);
  cmd.AddValue ("mac-client", "Mac Address for Server Client : 00:00:00:00:00:01", macClient);
  cmd.AddValue ("mac-server", "Mac Address for Server Default : 00:00:00:00:00:02", macServer);
  cmd.AddValue ("data-rate", "Data rate defaults to 1000Mb/s", dataRate);
  cmd.AddValue ("transportProt", "Transport protocol to use: Tcp, Udp", transportProt);
  cmd.AddValue ("emuMode", "Emulation mode in {raw, netmap}", emuMode);
  cmd.Parse (argc, argv);

  if (transportProt.compare ("Tcp") == 0)
    {
      socketType = "ns3::TcpSocketFactory";
    }
  else
    {
      socketType = "ns3::UdpSocketFactory";
    }

  Ipv4Address remoteIp;
  Ipv4Address localIp;
  Mac48AddressValue localMac;

  Config::SetDefault ("ns3::TcpSocket::SegmentSize", UintegerValue (packetSize));

  if (serverMode)
    {
      remoteIp = Ipv4Address (client.c_str ());
      localIp = Ipv4Address (server.c_str ());
      localMac = Mac48AddressValue (macServer.c_str ());
    }
  else
    {
      remoteIp = Ipv4Address (server.c_str ());
      localIp = Ipv4Address (client.c_str ());
      localMac =  Mac48AddressValue (macClient.c_str ());
    }

  Ipv4Mask localMask (netmask.c_str ());

  GlobalValue::Bind ("SimulatorImplementationType", StringValue ("ns3::RealtimeSimulatorImpl"));

  GlobalValue::Bind ("ChecksumEnabled", BooleanValue (true));

  NS_LOG_INFO ("Create Node");
  Ptr<Node> node = CreateObject<Node> ();

  NS_LOG_INFO ("Create Device");
  FdNetDeviceHelper* helper = nullptr;

#ifdef HAVE_PACKET_H
  if (emuMode == "raw")
    {
      EmuFdNetDeviceHelper* raw = new EmuFdNetDeviceHelper;
      raw->SetDeviceName (deviceName);
      helper = raw;
    }
#endif
#ifdef HAVE_NETMAP_USER_H
  if (emuMode == "netmap")
    {
      NetmapNetDeviceHelper* netmap = new NetmapNetDeviceHelper;
      netmap->SetDeviceName (deviceName);
      helper = netmap;
    }
#endif
#ifdef HAVE_DPDK_USER_H
  if (emuMode == "dpdk")
    {
      DpdkNetDeviceHelper* dpdk = new DpdkNetDeviceHelper ();
      // Use e1000 driver library (this is for IGb PMD supproting Intel 1GbE NIC)
      // NOTE: DPDK supports multiple Poll Mode Drivers (PMDs) and you can use it
      // based on your NIC. You just need to set pmd library as follows:
      dpdk->SetPmdLibrary ("librte_pmd_e1000.so");
      // Set dpdk driver to use for the NIC. `uio_pci_generic` supports most NICs.
      dpdk->SetDpdkDriver ("uio_pci_generic");
      // Set device name
      dpdk->SetDeviceName (deviceName);
      helper = dpdk;
    }
#endif

  if (helper == nullptr)
    {
      NS_ABORT_MSG (emuMode << " not supported.");
    }

  NetDeviceContainer devices = helper->Install (node);
  Ptr<NetDevice> device = devices.Get (0);
  device->SetAttribute ("Address", localMac);

  NS_LOG_INFO ("Add Internet Stack");
  InternetStackHelper internetStackHelper;
  internetStackHelper.SetIpv4StackInstall (true);
  internetStackHelper.Install (node);

  NS_LOG_INFO ("Create IPv4 Interface");
  Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
  uint32_t interface = ipv4->AddInterface (device);
  Ipv4InterfaceAddress address = Ipv4InterfaceAddress (localIp, localMask);
  ipv4->AddAddress (interface, address);
  ipv4->SetMetric (interface, 1);
  ipv4->SetUp (interface);

  if (serverMode)
    {
      Address sinkLocalAddress (InetSocketAddress (localIp, sinkPort));
      PacketSinkHelper sinkHelper (socketType, sinkLocalAddress);
      ApplicationContainer sinkApp = sinkHelper.Install (node);
      sinkApp.Start (Seconds (1.0));
      sinkApp.Stop (Seconds (60.0));

      helper->EnablePcap ("fd-server", device);
    }
  else
    {
      AddressValue remoteAddress (InetSocketAddress (remoteIp, sinkPort));
      OnOffHelper onoff (socketType, Address ());
      onoff.SetAttribute ("Remote", remoteAddress);
      onoff.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
      onoff.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));
      onoff.SetAttribute ("DataRate", DataRateValue (dataRate));
      onoff.SetAttribute ("PacketSize", UintegerValue (packetSize));

      ApplicationContainer clientApps = onoff.Install (node);
      clientApps.Start (Seconds (4.0));
      clientApps.Stop (Seconds (58.0));

      helper->EnablePcap ("fd-client", device);
    }

  Simulator::Stop (Seconds (61.0));
  Simulator::Run ();
  Simulator::Destroy ();
  delete helper;

  return 0;
}

