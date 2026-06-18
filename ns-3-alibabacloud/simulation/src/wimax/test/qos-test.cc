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
#include "ns3/log.h"
#include "ns3/test.h"
#include "ns3/simulator.h"
#include "ns3/node-container.h"
#include "ns3/wimax-helper.h"
#include "ns3/net-device-container.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv4-interface-container.h"
#include "ns3/ipv4-address.h"
#include "ns3/service-flow.h"
#include "ns3/ipcs-classifier-record.h"

using namespace ns3;

/**
 * \ingroup wimax-test
 * \ingroup tests
 *
 * \brief Ns3 Wimax Scheduling Test Case
 */
class Ns3WimaxSchedulingTestCase : public TestCase
{
public:
  Ns3WimaxSchedulingTestCase ();
  virtual ~Ns3WimaxSchedulingTestCase ();

private:
  virtual void DoRun (void);
  /**
   * Run once function
   * \param scheduler the wimax scheduler type
   * \returns true if successful
   */
  bool DoRunOnce (WimaxHelper::SchedulerType scheduler);

};

Ns3WimaxSchedulingTestCase::Ns3WimaxSchedulingTestCase ()
  : TestCase ("Test the 2 different schedulers")
{
}

Ns3WimaxSchedulingTestCase::~Ns3WimaxSchedulingTestCase ()
{
}

bool
Ns3WimaxSchedulingTestCase::DoRunOnce (WimaxHelper::SchedulerType scheduler)
{
  NodeContainer ssNodes;
  NodeContainer bsNodes;
  ssNodes.Create (2);
  bsNodes.Create (1);
  WimaxHelper wimax;
  NetDeviceContainer ssDevs, bsDevs;
  ssDevs = wimax.Install (ssNodes,
                          WimaxHelper::DEVICE_TYPE_SUBSCRIBER_STATION,
                          WimaxHelper::SIMPLE_PHY_TYPE_OFDM,
                          scheduler);
  bsDevs = wimax.Install (bsNodes, WimaxHelper::DEVICE_TYPE_BASE_STATION, WimaxHelper::SIMPLE_PHY_TYPE_OFDM, scheduler);
  ssDevs.Get (0)->GetObject<SubscriberStationNetDevice> ()->SetModulationType (WimaxPhy::MODULATION_TYPE_QAM16_12);
  ssDevs.Get (1)->GetObject<SubscriberStationNetDevice> ()->SetModulationType (WimaxPhy::MODULATION_TYPE_QAM16_12);

  InternetStackHelper stack;
  stack.Install (bsNodes);
  stack.Install (ssNodes);

  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");

  Ipv4InterfaceContainer SSinterfaces = address.Assign (ssDevs);
  Ipv4InterfaceContainer BSinterface = address.Assign (bsDevs);

  /*------------------------------*/
  //  UdpServerHelper udpServer;
  //  ApplicationContainer serverApps;
  //  UdpClientHelper udpClient;
  //  ApplicationContainer clientApps;
  //
  //  udpServer = UdpServerHelper (100);
  //  serverApps = udpServer.Install (ssNodes.Get (0));
  //  serverApps.Start (Seconds (1));
  //  serverApps.Stop (Seconds (2));
  //
  //  udpClient = UdpClientHelper (SSinterfaces.GetAddress (0), 100);
  //  udpClient.SetAttribute ("MaxPackets", UintegerValue (1200));
  //  udpClient.SetAttribute ("Interval", TimeValue (Seconds (0.12)));
  //  udpClient.SetAttribute ("PacketSize", UintegerValue (1024));
  //  clientApps = udpClient.Install (ssNodes.Get (1));
  //  clientApps.Start (Seconds (1));
  //  clientApps.Stop (Seconds (2));

  Simulator::Stop (Seconds (2 + 0.1));

  IpcsClassifierRecord DlClassifierUgs (Ipv4Address ("0.0.0.0"),
                                        Ipv4Mask ("0.0.0.0"),
                                        SSinterfaces.GetAddress (0),
                                        Ipv4Mask ("255.255.255.255"),
                                        0,
                                        65000,
                                        100,
                                        100,
                                        17,
                                        1);
  ServiceFlow DlServiceFlowUgs = wimax.CreateServiceFlow (ServiceFlow::SF_DIRECTION_DOWN,
                                                          ServiceFlow::SF_TYPE_RTPS,
                                                          DlClassifierUgs);
  IpcsClassifierRecord UlClassifierUgs (SSinterfaces.GetAddress (1),
                                        Ipv4Mask ("255.255.255.255"),
                                        Ipv4Address ("0.0.0.0"),
                                        Ipv4Mask ("0.0.0.0"),
                                        0,
                                        65000,
                                        100,
                                        100,
                                        17,
                                        1);
  ServiceFlow UlServiceFlowUgs = wimax.CreateServiceFlow (ServiceFlow::SF_DIRECTION_UP,
                                                          ServiceFlow::SF_TYPE_RTPS,
                                                          UlClassifierUgs);
  ssDevs.Get (0)->GetObject<SubscriberStationNetDevice> ()->AddServiceFlow (DlServiceFlowUgs);
  ssDevs.Get (1)->GetObject<SubscriberStationNetDevice> ()->AddServiceFlow (UlServiceFlowUgs);

  Simulator::Run ();
  Simulator::Destroy ();
  return false;

}

void
Ns3WimaxSchedulingTestCase::DoRun ()
{
  if (DoRunOnce (WimaxHelper::SCHED_TYPE_SIMPLE) == true)
    {
      return;
    }
  if (DoRunOnce (WimaxHelper::SCHED_TYPE_RTPS) == true)
    {
      return;
    }
}


/**
 * \ingroup wimax-test
 * \ingroup tests
 *
 * \brief Ns3 Wimax SF Type Test Case
 */
class Ns3WimaxSFTypeTestCase : public TestCase
{
public:
  Ns3WimaxSFTypeTestCase ();
  virtual ~Ns3WimaxSFTypeTestCase ();

private:
  virtual void DoRun (void);
  /**
   * Run once function
   * \param schedulingType the scheduling type
   * \returns true if successful
   */
  bool DoRunOnce (ServiceFlow::SchedulingType schedulingType);

};

Ns3WimaxSFTypeTestCase::Ns3WimaxSFTypeTestCase ()
  : TestCase ("Test the service flow scheduling types")
{
}

Ns3WimaxSFTypeTestCase::~Ns3WimaxSFTypeTestCase ()
{
}

bool
Ns3WimaxSFTypeTestCase::DoRunOnce (ServiceFlow::SchedulingType schedulingType)
{
  NodeContainer ssNodes;
  NodeContainer bsNodes;
  ssNodes.Create (2);
  bsNodes.Create (1);
  WimaxHelper wimax;
  NetDeviceContainer ssDevs, bsDevs;
  ssDevs = wimax.Install (ssNodes,
                          WimaxHelper::DEVICE_TYPE_SUBSCRIBER_STATION,
                          WimaxHelper::SIMPLE_PHY_TYPE_OFDM,
                          WimaxHelper::SCHED_TYPE_SIMPLE);
  bsDevs = wimax.Install (bsNodes,
                          WimaxHelper::DEVICE_TYPE_BASE_STATION,
                          WimaxHelper::SIMPLE_PHY_TYPE_OFDM,
                          WimaxHelper::SCHED_TYPE_SIMPLE);
  ssDevs.Get (0)->GetObject<SubscriberStationNetDevice> ()->SetModulationType (WimaxPhy::MODULATION_TYPE_QAM16_12);
  ssDevs.Get (1)->GetObject<SubscriberStationNetDevice> ()->SetModulationType (WimaxPhy::MODULATION_TYPE_QAM16_12);

  InternetStackHelper stack;
  stack.Install (bsNodes);
  stack.Install (ssNodes);

  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");

  Ipv4InterfaceContainer SSinterfaces = address.Assign (ssDevs);
  Ipv4InterfaceContainer BSinterface = address.Assign (bsDevs);

  /*------------------------------*/
  //  UdpServerHelper udpServer;
  //  ApplicationContainer serverApps;
  //  UdpClientHelper udpClient;
  //  ApplicationContainer clientApps;
  //
  //  udpServer = UdpServerHelper (100);
  //  serverApps = udpServer.Install (ssNodes.Get (0));
  //  serverApps.Start (Seconds (1));
  //  serverApps.Stop (Seconds (2));
  //
  //  udpClient = UdpClientHelper (SSinterfaces.GetAddress (0), 100);
  //  udpClient.SetAttribute ("MaxPackets", UintegerValue (1200));
  //  udpClient.SetAttribute ("Interval", TimeValue (Seconds (0.12)));
  //  udpClient.SetAttribute ("PacketSize", UintegerValue (1024));
  //  clientApps = udpClient.Install (ssNodes.Get (1));
  //  clientApps.Start (Seconds (1));
  //  clientApps.Stop (Seconds (2));

  Simulator::Stop (Seconds (2 + 0.1));

  IpcsClassifierRecord DlClassifier (Ipv4Address ("0.0.0.0"),
                                     Ipv4Mask ("0.0.0.0"),
                                     SSinterfaces.GetAddress (0),
                                     Ipv4Mask ("255.255.255.255"),
                                     0,
                                     65000,
                                     100,
                                     100,
                                     17,
                                     1);
  ServiceFlow DlServiceFlow = wimax.CreateServiceFlow (ServiceFlow::SF_DIRECTION_DOWN, schedulingType, DlClassifier);
  IpcsClassifierRecord UlClassifier (SSinterfaces.GetAddress (1),
                                     Ipv4Mask ("255.255.255.255"),
                                     Ipv4Address ("0.0.0.0"),
                                     Ipv4Mask ("0.0.0.0"),
                                     0,
                                     65000,
                                     100,
                                     100,
                                     17,
                                     1);
  ServiceFlow UlServiceFlow = wimax.CreateServiceFlow (ServiceFlow::SF_DIRECTION_UP, schedulingType, UlClassifier);
  ssDevs.Get (0)->GetObject<SubscriberStationNetDevice> ()->AddServiceFlow (DlServiceFlow);
  ssDevs.Get (1)->GetObject<SubscriberStationNetDevice> ()->AddServiceFlow (UlServiceFlow);

  Simulator::Run ();
  Simulator::Destroy ();
  return false;

}

void
Ns3WimaxSFTypeTestCase::DoRun ()
{
  if (DoRunOnce (ServiceFlow::SF_TYPE_UGS) == true)
    {
      return;
    }
  if (DoRunOnce (ServiceFlow::SF_TYPE_RTPS) == true)
    {
      return;
    }
  if (DoRunOnce (ServiceFlow::SF_TYPE_BE) == true)
    {
      return;
    }
}

/**
 * \ingroup wimax-test
 * \ingroup tests
 *
 * \brief Ns3 Wimax QoS Test Suite
 */
class Ns3WimaxQoSTestSuite : public TestSuite
{
public:
  Ns3WimaxQoSTestSuite ();
};

Ns3WimaxQoSTestSuite::Ns3WimaxQoSTestSuite ()
  : TestSuite ("wimax-qos", SYSTEM)
{
  AddTestCase (new Ns3WimaxSFTypeTestCase, TestCase::QUICK);
  AddTestCase (new Ns3WimaxSchedulingTestCase, TestCase::QUICK);
}

static Ns3WimaxQoSTestSuite ns3WimaxQoSTestSuite; ///< the test suite
