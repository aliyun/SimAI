/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 *  Copyright (c) 2007,2008, 2009 INRIA, UDcast
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
 * Author: Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 *                              <amine.ismail@udcast.com>
 */

// Default network topology includes:
// - A base station (BS)
// - Some number of SSs specified by the variable nbSS (defaults to 10)
// - A multicast router (ASNGW)
// - A multicast streamer

// Two Lans are setup: The first one between the multicast streamer and the
// ASNGW, the second one between the ASNGW (router) and the base station

//      +-----+    +-----+    +-----+   +-----+    +-----+
//      | SS0 |    | SS1 |    | SS2 |   | SS3 |    | SS4 |
//      +-----+    +-----+    +-----+   +-----+    +-----+
//     10.1.0.1   10.1.0.2   10.1.0.3  10.1.0.4   10.1.0.5
//      --------  --------    -------   -------   --------
//        ((*))    ((*))       ((*))     ((*))       ((*))
//
//                              LAN2 (11.1.1.0)
//                          ===============
//        10.1.0.11          |          |
//              +------------+        ASNGW           multicast Streamer
//       ((*))==|Base Station|          |  (12.1.1.0)   |
//              +------------+         ==================
//                                            LAN1
//
//        ((*))    ((*))       ((*))        ((*))    ((*))
//       -------   --------   --------    -------   --------
//      10.1.0.6   10.1.0.7   10.1.0.8    10.1.0.9   10.1.0.10
//       +-----+    +-----+    +-----+    +-----+    +-----+
//       | SS5 |    | SS6 |    | SS7 |    | SS8 |    | SS9 |
//       +-----+    +-----+    +-----+    +-----+    +-----+

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"
#include "ns3/config-store-module.h"
#include "ns3/wimax-module.h"
#include "ns3/csma-module.h"
#include <iostream>
#include "ns3/global-route-manager.h"
#include "ns3/internet-module.h"
#include "ns3/vector.h"
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("WimaxMulticastSimulation");

int main (int argc, char *argv[])
{
  bool verbose = false;

  NodeContainer ssNodes;
  std::vector<Ptr<SubscriberStationNetDevice> > ss;
  NetDeviceContainer ssDevs;
  Ipv4InterfaceContainer SSinterfaces;

  NodeContainer bsNodes;
  Ptr<BaseStationNetDevice> bs;
  NetDeviceContainer bsDevs, bsDevsOne;
  Ipv4InterfaceContainer BSinterfaces;

  UdpTraceClientHelper udpClient;
  ApplicationContainer clientApps;

  Ptr<SimpleOfdmWimaxChannel> channel;
  NodeContainer Streamer_Node;
  NodeContainer ASNGW_Node;

  Ptr<ConstantPositionMobilityModel> BSPosition;
  std::vector<Ptr<RandomWaypointMobilityModel> > SSPosition;
  std::vector<Ptr<RandomRectanglePositionAllocator> > SSPosAllocator;

  // default values
  int nbSS = 10, duration = 7, schedType = 0;
  WimaxHelper::SchedulerType scheduler = WimaxHelper::SCHED_TYPE_SIMPLE;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("nbSS", "number of subscriber station to create", nbSS);
  cmd.AddValue ("scheduler", "type of scheduler to use with the netdevices", schedType);
  cmd.AddValue ("duration", "duration of the simulation in seconds", duration);
  cmd.AddValue ("verbose", "turn on all WimaxNetDevice log components", verbose);
  cmd.Parse (argc, argv);

  LogComponentEnable ("UdpTraceClient", LOG_LEVEL_INFO);
  LogComponentEnable ("UdpServer", LOG_LEVEL_INFO);

  switch (schedType)
    {
    case 0:
      scheduler = WimaxHelper::SCHED_TYPE_SIMPLE;
      break;
    case 1:
      scheduler = WimaxHelper::SCHED_TYPE_MBQOS;
      break;
    case 2:
      scheduler = WimaxHelper::SCHED_TYPE_RTPS;
      break;
    default:
      scheduler = WimaxHelper::SCHED_TYPE_SIMPLE;
    }

  ss.resize (nbSS);
  SSPosition.resize (nbSS);
  SSPosAllocator.resize (nbSS);

  ssNodes.Create (nbSS);
  bsNodes.Create (1);

  Streamer_Node.Create (1);
  ASNGW_Node.Create (1);

  WimaxHelper wimax;

  channel = CreateObject<SimpleOfdmWimaxChannel> ();
  channel->SetPropagationModel (SimpleOfdmWimaxChannel::COST231_PROPAGATION);
  ssDevs = wimax.Install (ssNodes,
                          WimaxHelper::DEVICE_TYPE_SUBSCRIBER_STATION,
                          WimaxHelper::SIMPLE_PHY_TYPE_OFDM,
                          channel,
                          scheduler);
  Ptr<WimaxNetDevice> dev = wimax.Install (bsNodes.Get (0),
                                           WimaxHelper::DEVICE_TYPE_BASE_STATION,
                                           WimaxHelper::SIMPLE_PHY_TYPE_OFDM,
                                           channel,
                                           scheduler);

  BSPosition = CreateObject<ConstantPositionMobilityModel> ();

  BSPosition->SetPosition (Vector (1000, 0, 0));
  bsNodes.Get (0)->AggregateObject (BSPosition);
  bsDevs.Add (dev);
  if (verbose)
    {
      wimax.EnableLogComponents ();  // Turn on all wimax logging
    }

  for (int i = 0; i < nbSS; i++)
    {
      SSPosition[i] = CreateObject<RandomWaypointMobilityModel> ();
      SSPosAllocator[i] = CreateObject<RandomRectanglePositionAllocator> ();
      Ptr<UniformRandomVariable> xVar = CreateObject<UniformRandomVariable> ();
      xVar->SetAttribute ("Min", DoubleValue ((i / 40.0) * 2000));
      xVar->SetAttribute ("Max", DoubleValue ((i / 40.0 + 1) * 2000));
      SSPosAllocator[i]->SetX (xVar);
      Ptr<UniformRandomVariable> yVar = CreateObject<UniformRandomVariable> ();
      yVar->SetAttribute ("Min", DoubleValue ((i / 40.0) * 2000));
      yVar->SetAttribute ("Max", DoubleValue ((i / 40.0 + 1) * 2000));
      SSPosAllocator[i]->SetY (yVar);
      SSPosition[i]->SetAttribute ("PositionAllocator", PointerValue (SSPosAllocator[i]));
      SSPosition[i]->SetAttribute ("Speed", StringValue ("ns3::UniformRandomVariable[Min=10.3|Max=40.7]"));
      SSPosition[i]->SetAttribute ("Pause", StringValue ("ns3::ConstantRandomVariable[Constant=0.01]"));

      ss[i] = ssDevs.Get (i)->GetObject<SubscriberStationNetDevice> ();
      ss[i]->SetModulationType (WimaxPhy::MODULATION_TYPE_QAM16_12);
      ssNodes.Get (i)->AggregateObject (SSPosition[i]);

    }

  bs = bsDevs.Get (0)->GetObject<BaseStationNetDevice> ();

  CsmaHelper csmaASN_BS;
  CsmaHelper csmaStreamer_ASN;

  // First LAN BS and ASN
  NodeContainer LAN_ASN_BS;

  LAN_ASN_BS.Add (bsNodes.Get (0));

  LAN_ASN_BS.Add (ASNGW_Node.Get (0));

  csmaASN_BS.SetChannelAttribute ("DataRate", DataRateValue (DataRate (10000000)));
  csmaASN_BS.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));
  csmaASN_BS.SetDeviceAttribute ("Mtu", UintegerValue (1500));
  NetDeviceContainer LAN_ASN_BS_Devs = csmaASN_BS.Install (LAN_ASN_BS);

  NetDeviceContainer BS_CSMADevs;

  BS_CSMADevs.Add (LAN_ASN_BS_Devs.Get (0));

  NetDeviceContainer ASN_Devs1;
  ASN_Devs1.Add (LAN_ASN_BS_Devs.Get (1));

  // Second LAN ASN-GW and Streamer

  NodeContainer LAN_ASN_STREAMER;
  LAN_ASN_STREAMER.Add (ASNGW_Node.Get (0));
  LAN_ASN_STREAMER.Add (Streamer_Node.Get (0));

  csmaStreamer_ASN.SetChannelAttribute ("DataRate", DataRateValue (DataRate (10000000)));
  csmaStreamer_ASN.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));
  csmaStreamer_ASN.SetDeviceAttribute ("Mtu", UintegerValue (1500));

  NetDeviceContainer LAN_ASN_STREAMER_Devs = csmaStreamer_ASN.Install (LAN_ASN_STREAMER);

  NetDeviceContainer STREAMER_Devs;
  NetDeviceContainer ASN_Devs2;
  ASN_Devs2.Add (LAN_ASN_STREAMER_Devs.Get (0));
  STREAMER_Devs.Add (LAN_ASN_STREAMER_Devs.Get (1));

  MobilityHelper mobility;
  InternetStackHelper stack;
  mobility.Install (bsNodes);
  stack.Install (bsNodes);
  mobility.Install (ssNodes);
  stack.Install (ssNodes);
  stack.Install (Streamer_Node);
  stack.Install (ASNGW_Node);

  Ipv4AddressHelper address;

  address.SetBase ("10.1.0.0", "255.255.255.0");
  bsDevsOne.Add (bs);
  BSinterfaces = address.Assign (bsDevsOne);
  SSinterfaces = address.Assign (ssDevs);

  address.SetBase ("11.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer BSCSMAInterfaces = address.Assign (BS_CSMADevs);
  Ipv4InterfaceContainer ASNCSMAInterfaces1 = address.Assign (ASN_Devs1);

  address.SetBase ("12.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer ASNCSMAInterfaces2 = address.Assign (ASN_Devs2);
  Ipv4InterfaceContainer StreamerCSMAInterfaces = address.Assign (STREAMER_Devs);

  Ipv4Address multicastSource ("12.1.1.2");
  Ipv4Address multicastGroup ("224.30.10.81");

  Ipv4StaticRoutingHelper multicast;
  // 1) Configure a (static) multicast route on ASNGW (multicastRouter)
  Ptr<Node> multicastRouter = ASNGW_Node.Get (0); // The node in question
  Ptr<NetDevice> inputIf = ASN_Devs2.Get (0); // The input NetDevice


  multicast.AddMulticastRoute (multicastRouter, multicastSource, multicastGroup, inputIf, ASN_Devs1);

  // 2) Set up a default multicast route on the sender n0
  Ptr<Node> sender = Streamer_Node.Get (0);
  Ptr<NetDevice> senderIf = STREAMER_Devs.Get (0);
  multicast.SetDefaultMulticastRoute (sender, senderIf);

  // 1) Configure a (static) multicast route on ASNGW (multicastRouter)
  multicastRouter = bsNodes.Get (0); // The node in question
  inputIf = BS_CSMADevs.Get (0); // The input NetDevice

  multicast.AddMulticastRoute (multicastRouter, multicastSource, multicastGroup, inputIf, bsDevsOne);

  uint16_t multicast_port = 100;

  UdpServerHelper udpServerHelper = UdpServerHelper (multicast_port);
  ApplicationContainer serverApps;
  serverApps = udpServerHelper.Install (ssNodes);
  serverApps.Start (Seconds (6));
  serverApps.Stop (Seconds (duration));

  udpClient = UdpTraceClientHelper (multicastGroup, multicast_port, "");

  clientApps = udpClient.Install (Streamer_Node.Get (0));
  clientApps.Start (Seconds (6));
  clientApps.Stop (Seconds (duration));

  IpcsClassifierRecord MulticastClassifier (Ipv4Address ("0.0.0.0"),
                                            Ipv4Mask ("0.0.0.0"),
                                            multicastGroup,
                                            Ipv4Mask ("255.255.255.255"),
                                            0,
                                            65000,
                                            multicast_port,
                                            multicast_port,
                                            17,
                                            1);

  ServiceFlow MulticastServiceFlow = wimax.CreateServiceFlow (ServiceFlow::SF_DIRECTION_DOWN,
                                                              ServiceFlow::SF_TYPE_UGS,
                                                              MulticastClassifier);

  bs->GetServiceFlowManager ()->AddMulticastServiceFlow (MulticastServiceFlow, WimaxPhy::MODULATION_TYPE_QPSK_12);

  Simulator::Stop (Seconds (duration + 0.1));
  NS_LOG_INFO ("Starting simulation.....");
  Simulator::Run ();

  for (int i = 0; i < nbSS; i++)
    {
      ss[i] = 0;
      SSPosition[i] = 0;
      SSPosAllocator[i] = 0;
    }

  bs = 0;

  Simulator::Destroy ();
  NS_LOG_INFO ("Done.");

  return 0;
}
