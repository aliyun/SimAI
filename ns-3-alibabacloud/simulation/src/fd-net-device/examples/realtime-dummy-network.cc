/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 University of Washington, 2012 INRIA
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

// Network topology
//
#include <sys/socket.h>
#include <errno.h>

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/fd-net-device-module.h"
#include "ns3/internet-apps-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("RealtimeDummyNetworkExample");

int
main (int argc, char *argv[])
{
  CommandLine cmd (__FILE__);
  cmd.Parse (argc, argv);

  GlobalValue::Bind ("SimulatorImplementationType", StringValue ("ns3::RealtimeSimulatorImpl"));
  GlobalValue::Bind ("ChecksumEnabled", BooleanValue (true));

  NodeContainer nodes;
  nodes.Create (2);

  InternetStackHelper stack;
  stack.Install (nodes);

  FdNetDeviceHelper fd;
  NetDeviceContainer devices = fd.Install (nodes);

  int sv[2];
  if (socketpair (AF_UNIX, SOCK_DGRAM, 0, sv) < 0)
    {
      NS_FATAL_ERROR ("Error creating pipe=" << strerror (errno));
    }

  Ptr<NetDevice> d1 = devices.Get (0);
  Ptr<FdNetDevice> device1 = d1->GetObject<FdNetDevice> ();
  device1->SetFileDescriptor (sv[0]);

  Ptr<NetDevice> d2 = devices.Get (1);
  Ptr<FdNetDevice> device2 = d2->GetObject<FdNetDevice> ();
  device2->SetFileDescriptor (sv[1]);

  Ipv4AddressHelper addresses;
  addresses.SetBase ("10.0.0.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces = addresses.Assign (devices);

  Ptr<V4Ping> app = CreateObject<V4Ping> ();
  app->SetAttribute ("Remote", Ipv4AddressValue (interfaces.GetAddress (0)));
  app->SetAttribute ("Verbose", BooleanValue (true));
  nodes.Get (1)->AddApplication (app);
  app->SetStartTime (Seconds (0.0));
  app->SetStopTime (Seconds (4.0));

  fd.EnablePcapAll ("realtime-dummy-network", false);

  Simulator::Stop (Seconds (5.));
  Simulator::Run ();
  Simulator::Destroy ();
}
