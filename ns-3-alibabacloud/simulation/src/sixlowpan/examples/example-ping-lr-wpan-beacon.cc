/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Ritsumeikan University, Shiga, Japan
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
 * Author: Alberto Gallegos Ramonet <ramonet@fc.ritsumei.ac.jp>
 */


#include <fstream>
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/mobility-module.h"
#include "ns3/spectrum-module.h"
#include "ns3/propagation-module.h"
#include "ns3/sixlowpan-module.h"
#include "ns3/lr-wpan-module.h"

using namespace ns3;


static void dataSentMacConfirm (McpsDataConfirmParams params)
{
  // In the case of transmissions with the Ack flag activated, the transaction is only
  // successful if the Ack was received.
  if (params.m_status == LrWpanMcpsDataConfirmStatus::IEEE_802_15_4_SUCCESS)
    {
      NS_LOG_UNCOND ("**********" << Simulator::Now ().As (Time::S) << " | Transmission successfully sent");
    }
}


int main (int argc, char** argv)
{

  bool verbose = false;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("verbose", "turn on log components", verbose);
  cmd.Parse (argc, argv);

  if (verbose)
    {
      LogComponentEnableAll (LOG_PREFIX_TIME);
      LogComponentEnableAll (LOG_PREFIX_FUNC);
      LogComponentEnable ("LrWpanMac", LOG_LEVEL_INFO);
      LogComponentEnable ("LrWpanCsmaCa", LOG_LEVEL_INFO);
      LogComponentEnable ("LrWpanHelper", LOG_LEVEL_ALL);
      LogComponentEnable ("Ping6Application", LOG_LEVEL_INFO);
    }


  NodeContainer nodes;
  nodes.Create (2);

  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.SetPositionAllocator ("ns3::GridPositionAllocator",
                                 "MinX", DoubleValue (0.0),
                                 "MinY", DoubleValue (0.0),
                                 "DeltaX", DoubleValue (20),
                                 "DeltaY", DoubleValue (20),
                                 "GridWidth", UintegerValue (3),
                                 "LayoutType", StringValue ("RowFirst"));
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (nodes);

  LrWpanHelper lrWpanHelper;
  // Add and install the LrWpanNetDevice for each node
  NetDeviceContainer lrwpanDevices = lrWpanHelper.Install (nodes);

  Ptr<LrWpanNetDevice> dev1 = lrwpanDevices.Get (0)->GetObject<LrWpanNetDevice> ();
  Ptr<LrWpanNetDevice> dev2 = lrwpanDevices.Get (1)->GetObject<LrWpanNetDevice> ();

  McpsDataConfirmCallback cb1;
  cb1 = MakeCallback (&dataSentMacConfirm);
  dev1->GetMac ()->SetMcpsDataConfirmCallback (cb1);
  dev2->GetMac ()->SetMcpsDataConfirmCallback (cb1);


  // Fake PAN association, coordinator assignment, short address assignment and initialization
  // of beacon-enabled mode in 802.15.4-2011.
  // This is needed because the lr-wpan module does not provide (yet)
  // a full PAN association procedure.

  // AssociateToBeaconPan (devices, PAN ID, Coordinator Address, Beacon Order, Superframe Order)

  // Must be careful not setting the beacon order (BO) and the superframe order (SO) too far apart
  // or the ping reply (ICMPV6 echo reply) can time out during the inactive period of the superframe.
  // A full time table of the BO/SO time equivalence can be found at the end of this document.
  // The current configuration is BO = 14, SO = 13 :

  //           Contention Access Period (CAP)                           Inactive
  //              (125.82912 secs)                                     (125.82088)
  //   |---------------------------------------------|-------------------------------------------|
  // Beacon                                                                                   Beacon
  //                            Beacon Interval = 251.65 secs
  //   |-----------------------------------------------------------------------------------------|

  lrWpanHelper.AssociateToBeaconPan (lrwpanDevices, 0, Mac16Address ("00:01"), 14,13);


  InternetStackHelper internetv6;
  internetv6.Install (nodes);

  SixLowPanHelper sixlowpan;
  NetDeviceContainer devices = sixlowpan.Install (lrwpanDevices);

  Ipv6AddressHelper ipv6;
  ipv6.SetBase (Ipv6Address ("2001:2::"), Ipv6Prefix (64));
  Ipv6InterfaceContainer deviceInterfaces;
  deviceInterfaces = ipv6.Assign (devices);


  // Send ping packets after the 2nd second of the simulation during the
  // first 8 seconds of the CAP in the incoming superframe

  uint32_t packetSize = 10;
  uint32_t maxPacketCount = 5;
  Time interPacketInterval = Seconds (1);
  Ping6Helper ping6;

  ping6.SetLocal (deviceInterfaces.GetAddress (0, 1));
  ping6.SetRemote (deviceInterfaces.GetAddress (1, 1));

  ping6.SetAttribute ("MaxPackets", UintegerValue (maxPacketCount));
  ping6.SetAttribute ("Interval", TimeValue (interPacketInterval));
  ping6.SetAttribute ("PacketSize", UintegerValue (packetSize));
  ApplicationContainer apps = ping6.Install (nodes.Get (0));

  apps.Start (Seconds (2.0));
  apps.Stop (Seconds (10.0));

  AsciiTraceHelper ascii;
  lrWpanHelper.EnableAsciiAll (ascii.CreateFileStream ("Ping-6LoW-lr-wpan-beacon.tr"));
  lrWpanHelper.EnablePcapAll (std::string ("Ping-6LoW-lr-wpan-beacon"), true);


  Simulator::Stop (Seconds (600));

  Simulator::Run ();
  Simulator::Destroy ();

}

// BO/SO values to time equivalence
// These times are only valid for a 250kbps O-QPSK modulation,
// times differ with other modulation configurations.

// +------------------------+
// | BO/SO |  Time (secs)   |
// +------------------------+
// |   0   | 0.01536 secs   |
// |   1   | 0.03072 secs   |
// |   2   | 0.06144 secs   |
// |   3   | 0.12288 secs   |
// |   4   | 0.24576 secs   |
// |   5   | 0.49152 secs   |
// |   6   | 0.98304 secs   |
// |   7   | 1.96608 secs   |
// |   8   | 3.93216 secs   |
// |   9   | 7.86432 secs   |
// |   10  | 15.72864 secs  |
// |   11  | 31.45728 secs  |
// |   12  | 62.91456 secs  |
// |   13  | 125.82912 secs |
// |   14  | 251.65 secs    |
// +------------------------+
