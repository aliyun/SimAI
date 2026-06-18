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
 * This example builds a node with a device in emulation mode in {raw, netmap}.
 * The aim is to measure the maximum tx rate in pps achievable with 
 * NetmapNetDevice and FdNetDevice on a specific machine.
 * The emulated device must be connected and in promiscuous mode.
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
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/ipv4-list-routing-helper.h"
#include "ns3/applications-module.h"
#include "ns3/traffic-control-module.h"

#include <chrono>
#include <unistd.h>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("NetmapEmulationSendExample");

// This function sends a number of packets by means of the SendFrom method or
// the Write method (depending on the level value) of a FdNetDevice or
// of a NetmapNetDevice (depending on the emulation mode value).

static void
Send (Ptr<NetDevice> dev, int level, std::string emuMode)
{
  Ptr<FdNetDevice> device = DynamicCast<FdNetDevice> (dev);

  int packets = 10000000;

  Mac48Address sender = Mac48Address ("00:00:00:aa:00:01");
  Mac48Address receiver = Mac48Address ("ff:ff:ff:ff:ff:ff");

  int packetsSize = 64;
  Ptr<Packet> packet = Create<Packet> (packetsSize);
  EthernetHeader header;

  ssize_t len =  (size_t) packet->GetSize ();
  uint8_t *buffer = (uint8_t*)malloc (len);
  packet->CopyData (buffer, len);

  int sent = 0;
  int failed = 0;

  Ptr<NetDeviceQueue> ndq = nullptr;
  if (emuMode == "netmap")
    {
      Ptr<NetDeviceQueueInterface> ndqi = dev->GetObject<NetDeviceQueueInterface> ();
      ndq = ndqi->GetTxQueue (0);
    }

  std::cout << ((level == 0) ? "Writing" : "Sending") << std::endl;

  // period to print the stats
  std::chrono::milliseconds period (1000);

  auto t1 = std::chrono::high_resolution_clock::now ();

  while (packets > 0)
    {
      // in case of netmap emulated device we check for
      // available slot in the netmap transmission ring
      if (ndq)
        {
          while (ndq->IsStopped ())
            {
              usleep (10);
            }
        }

      if (level == 1)
        {
          if (device->SendFrom (packet, sender, receiver, 0) == false)
            {
              failed++;
            }
          sent++;
          packet->RemoveHeader (header);
        }

      if (level == 0)
        {
          if (device->Write (buffer, len) != len)
            {
              failed++;
            }
          sent++;
        }

      auto t2 = std::chrono::high_resolution_clock::now ();

      if (t2 - t1 >= period)
        {
          // print stats
          std::chrono::duration<double, std::milli> dur = (t2 - t1); // in ms
          double estimatedThr = ((sent - failed) * packetsSize * 8) / 1000000; // in Mbps
          std::cout << sent << " packets sent in " << dur.count () << " ms, failed " << failed << " (" << estimatedThr << " Mbps estimated throughput)" << std::endl;
          sent = 0;
          failed = 0;
          t1 = std::chrono::high_resolution_clock::now ();
        }
      packets--;
    }
}

int
main (int argc, char *argv[])
{
  std::string deviceName ("eno1");
  int level = 0;

#ifdef HAVE_PACKET_H
  std::string emuMode ("raw");
#else        // HAVE_NETMAP_USER_H is true (otherwise this example is not compiled)
  std::string emuMode ("netmap");
#endif

  CommandLine cmd;
  cmd.AddValue ("deviceName", "Device name", deviceName);
  cmd.AddValue ("level", "Enable send (1) or write (0) level test", level);
  cmd.AddValue ("emuMode", "Emulation mode in {raw, netmap}", emuMode);

  cmd.Parse (argc, argv);

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

  if (helper == nullptr)
    {
      NS_ABORT_MSG (emuMode << " not supported.");
    }

  NetDeviceContainer devices = helper->Install (node);
  Ptr<NetDevice> device = devices.Get (0);
  device->SetAttribute ("Address", Mac48AddressValue (Mac48Address::Allocate ()));

  Simulator::Schedule (Seconds (3), &Send, device, level, emuMode);

  NS_LOG_INFO ("Run Emulation.");
  Simulator::Stop (Seconds (6.0));
  Simulator::Run ();
  Simulator::Destroy ();
  delete helper;
  NS_LOG_INFO ("Done.");
}
