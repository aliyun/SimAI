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

//
// Default network topology includes a base station (BS) and some number of
// subscriber station (SS) specified by the variable nbSS (defaults to six).
// The SSs are grouped into two groups: senders and receivers. SSs from 0 to
// nbSS/2 are designed as senders and SSs from (nbSS/2 + 1) to nbSS-1 will
// designed as receivers.
// Each SS creates 3 transport connection with the BS. the fist one has as QoS
// scheduling type UGS, the second one rtPS and the third one BE.
// Senders SSs send some stamped IP packets with variable bitrate to receiver SSs
// through the BS station. receiver SSs receive the IP packets analyze them and
// based on a sequnece number decide if all the packets are coorectly received

//      +-----+    +-----+    +-----+
//      | SS0 |    | SS1 |    | SS2 |
//      +-----+    +-----+    +-----+
//     10.1.1.1   10.1.1.2   10.1.1.3
//      --------  --------    -------
//        ((*))    ((*))       ((*))
//
//                  10.1.1.7
//               +------------+
//               |Base Station| ==((*))
//               +------------+
//
//        ((*))    ((*))       ((*))
//       -------   --------   --------
//      10.1.1.4   10.1.1.5   10.1.1.6
//       +-----+    +-----+    +-----+
//       | SS3 |    | SS4 |    | SS5 |
//       +-----+    +-----+    +-----+

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"
#include "ns3/config-store-module.h"
#include "ns3/wimax-module.h"
#include "ns3/internet-module.h"
#include "ns3/global-route-manager.h"
#include <iostream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("wimaxIpV4Simulation");

int main (int argc, char *argv[])
{
  // default values
  int nbSS = 4, duration = 7, schedType = 0;
  bool verbose = false;
  WimaxHelper::SchedulerType scheduler = WimaxHelper::SCHED_TYPE_SIMPLE;
  LogComponentEnable ("UdpClient", LOG_LEVEL_INFO);
  LogComponentEnable ("UdpServer", LOG_LEVEL_INFO);


  CommandLine cmd (__FILE__);
  cmd.AddValue ("nbSS", "number of subscriber station to create", nbSS);
  cmd.AddValue ("scheduler", "type of scheduler to use with the network devices", schedType);
  cmd.AddValue ("duration", "duration of the simulation in seconds", duration);
  cmd.AddValue ("verbose", "turn on all WimaxNetDevice log components", verbose);
  cmd.Parse (argc, argv);

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

  NodeContainer ssNodes;
  NodeContainer bsNodes;

  ssNodes.Create (nbSS);
  bsNodes.Create (1);

  WimaxHelper wimax;

  NetDeviceContainer ssDevs, bsDevs;

  ssDevs = wimax.Install (ssNodes,
                          WimaxHelper::DEVICE_TYPE_SUBSCRIBER_STATION,
                          WimaxHelper::SIMPLE_PHY_TYPE_OFDM,
                          scheduler);
  bsDevs = wimax.Install (bsNodes, WimaxHelper::DEVICE_TYPE_BASE_STATION, WimaxHelper::SIMPLE_PHY_TYPE_OFDM, scheduler);

  Ptr<SubscriberStationNetDevice>* ss = new Ptr<SubscriberStationNetDevice>[nbSS];

  for (int i = 0; i < nbSS; i++)
    {
      ss[i] = ssDevs.Get (i)->GetObject<SubscriberStationNetDevice> ();
      ss[i]->SetModulationType (WimaxPhy::MODULATION_TYPE_QAM16_12);
    }

  Ptr<BaseStationNetDevice> bs;
  bs = bsDevs.Get (0)->GetObject<BaseStationNetDevice> ();

  MobilityHelper mobility;
  mobility.Install (bsNodes);
  mobility.Install (ssNodes);

  InternetStackHelper stack;
  stack.Install (bsNodes);
  stack.Install (ssNodes);

  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");

  Ipv4InterfaceContainer SSinterfaces = address.Assign (ssDevs);
  Ipv4InterfaceContainer BSinterface = address.Assign (bsDevs);
  if (verbose)
    {
      wimax.EnableLogComponents ();  // Turn on all wimax logging
    }
  /*------------------------------*/
  UdpServerHelper* udpServer = new UdpServerHelper[nbSS / 2];
  ApplicationContainer* serverApps = new ApplicationContainer[nbSS / 2];
  UdpClientHelper* udpClient = new UdpClientHelper[nbSS / 2];
  ApplicationContainer* clientApps = new ApplicationContainer[nbSS / 2];

  for (int i = 0; i < nbSS / 2; i++)
    {
      // set server port to 100+(i*10)
      udpServer[i] = UdpServerHelper (100 + (i * 10));
      serverApps[i] = udpServer[i].Install (ssNodes.Get (i));
      serverApps[i].Start (Seconds (6));
      serverApps[i].Stop (Seconds (duration));

      udpClient[i] = UdpClientHelper (SSinterfaces.GetAddress (i), 100 + (i * 10));
      udpClient[i].SetAttribute ("MaxPackets", UintegerValue (1200));
      udpClient[i].SetAttribute ("Interval", TimeValue (Seconds (0.12)));
      udpClient[i].SetAttribute ("PacketSize", UintegerValue (800));

      clientApps[i] = udpClient[i].Install (ssNodes.Get (i + (nbSS / 2)));
      clientApps[i].Start (Seconds (6));
      clientApps[i].Stop (Seconds (duration));
    }

  Simulator::Stop (Seconds (duration + 0.1));
  /*
   * Setup 1 transport connections between each SS and the BS
   */
  for (int i = 0; i < nbSS / 2; i++)
    {
      IpcsClassifierRecord DlClassifierBe (Ipv4Address ("0.0.0.0"),
                                           Ipv4Mask ("0.0.0.0"),
                                           SSinterfaces.GetAddress (i),
                                           Ipv4Mask ("255.255.255.255"),
                                           0,
                                           65000,
                                           100 + (i * 10),
                                           100 + (i * 10),
                                           17,
                                           1);
      ServiceFlow DlServiceFlowBe = wimax.CreateServiceFlow (ServiceFlow::SF_DIRECTION_DOWN,
                                                             ServiceFlow::SF_TYPE_BE,
                                                             DlClassifierBe);
      ss[i]->AddServiceFlow (DlServiceFlowBe);
      IpcsClassifierRecord ulClassifierBe (SSinterfaces.GetAddress (i + (nbSS / 2)),
                                           Ipv4Mask ("255.255.255.255"),
                                           Ipv4Address ("0.0.0.0"),
                                           Ipv4Mask ("0.0.0.0"),
                                           0,
                                           65000,
                                           100 + (i * 10),
                                           100 + (i * 10),
                                           17,
                                           1);
      ServiceFlow ulServiceFlowBe = wimax.CreateServiceFlow (ServiceFlow::SF_DIRECTION_UP,
                                                             ServiceFlow::SF_TYPE_BE,
                                                             ulClassifierBe);
      ss[i + (nbSS / 2)]->AddServiceFlow (ulServiceFlowBe);

    }

  NS_LOG_INFO ("Starting simulation.....");
  Simulator::Run ();

  delete[] clientApps;
  delete[] udpClient;
  delete[] serverApps;
  delete[] udpServer;
  for (int i = 0; i < nbSS; i++)
    {
      ss[i] = 0;
    }
  delete[] ss;

  bs = 0;

  Simulator::Destroy ();
  NS_LOG_INFO ("Done.");

  return 0;
}
