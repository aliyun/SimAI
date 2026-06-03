/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Alexander Krotov
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
 * Author: Alexander Krotov <krotov@iitp.ru>
 *
 */

#include "lte-test-aggregation-throughput-scale.h"

#include <algorithm>
#include <numeric>

#include <ns3/application-container.h>
#include <ns3/friis-spectrum-propagation-loss.h>
#include <ns3/internet-stack-helper.h>
#include <ns3/ipv4-address-helper.h>
#include <ns3/ipv4-interface-container.h>
#include <ns3/ipv4-static-routing-helper.h>
#include <ns3/log.h>
#include <ns3/lte-enb-net-device.h>
#include <ns3/lte-helper.h>
#include <ns3/lte-ue-net-device.h>
#include <ns3/lte-ue-rrc.h>
#include <ns3/mobility-helper.h>
#include <ns3/net-device-container.h>
#include <ns3/node-container.h>
#include <ns3/packet-sink.h>
#include <ns3/point-to-point-epc-helper.h>
#include <ns3/point-to-point-helper.h>
#include <ns3/simulator.h>
#include <ns3/udp-client.h>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LteAggregationThroughputScaleTest");

LteAggregationThroughputScaleTestSuite::LteAggregationThroughputScaleTestSuite ()
  : TestSuite ("lte-aggregation-throughput-scale", SYSTEM)
{
  AddTestCase (new LteAggregationThroughputScaleTestCase ("Carrier aggregation throughput scale"), TestCase::QUICK);
}

static LteAggregationThroughputScaleTestSuite g_lteAggregationThroughputScaleTestSuite;

LteAggregationThroughputScaleTestCase::LteAggregationThroughputScaleTestCase (std::string name)
  : TestCase (name)
{
  NS_LOG_FUNCTION (this << GetName ());
}


LteAggregationThroughputScaleTestCase::~LteAggregationThroughputScaleTestCase ()
{
  NS_LOG_FUNCTION (this << GetName ());
}

double
LteAggregationThroughputScaleTestCase::GetThroughput (uint8_t numberOfComponentCarriers)
{
  NS_LOG_FUNCTION (this << GetName ());

  Config::SetDefault ("ns3::LteEnbNetDevice::DlEarfcn", UintegerValue (100));
  Config::SetDefault ("ns3::LteEnbNetDevice::UlEarfcn", UintegerValue (100 + 18000));
  Config::SetDefault ("ns3::LteEnbNetDevice::DlBandwidth", UintegerValue (25));
  Config::SetDefault ("ns3::LteEnbNetDevice::UlBandwidth", UintegerValue (25));
  Config::SetDefault ("ns3::LteUeNetDevice::DlEarfcn", UintegerValue (100));

  auto lteHelper = CreateObject<LteHelper> ();
  lteHelper->SetAttribute ("PathlossModel", TypeIdValue (ns3::FriisSpectrumPropagationLossModel::GetTypeId ()));
  lteHelper->SetAttribute ("NumberOfComponentCarriers", UintegerValue (numberOfComponentCarriers));
  lteHelper->SetAttribute ("EnbComponentCarrierManager", StringValue ("ns3::RrComponentCarrierManager"));

  auto epcHelper = CreateObject<PointToPointEpcHelper> ();
  lteHelper->SetEpcHelper (epcHelper);

  auto enbNode = CreateObject<Node> ();
  auto ueNode = CreateObject<Node> ();
  auto pgwNode = epcHelper->GetPgwNode ();

  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (enbNode);
  mobility.Install (ueNode);

  InternetStackHelper internet;
  internet.Install (ueNode);

  Ipv4AddressHelper ipv4h;
  ipv4h.SetBase ("1.0.0.0", "255.0.0.0");

  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  auto ueStaticRouting = ipv4RoutingHelper.GetStaticRouting (ueNode->GetObject<Ipv4> ());
  ueStaticRouting->SetDefaultRoute (epcHelper->GetUeDefaultGatewayAddress (), 1);

  auto enbDev = DynamicCast<LteEnbNetDevice> (lteHelper->InstallEnbDevice (enbNode).Get (0));
  auto ueDevs = lteHelper->InstallUeDevice (ueNode);
  auto ueDev = DynamicCast<LteUeNetDevice> (ueDevs.Get (0));

  Ipv4InterfaceContainer ueIpIface = epcHelper->AssignUeIpv4Address (ueDevs);

  // Attach to last CC as primary
  std::map< uint8_t, Ptr<ComponentCarrierUe> > ueCcMap = ueDev->GetCcMap ();
  ueDev->SetDlEarfcn (ueCcMap.at (numberOfComponentCarriers - 1)->GetDlEarfcn ());
  lteHelper->Attach (ueDevs);
  m_expectedCellId = enbDev->GetCcMap ().at (numberOfComponentCarriers - 1)->GetCellId ();

  // Applications
  const uint16_t port = 21;

  ApplicationContainer apps;

  auto sink = CreateObject<PacketSink> ();
  sink->SetAttribute ("Protocol", StringValue ("ns3::UdpSocketFactory"));
  sink->SetAttribute ("Local", AddressValue (InetSocketAddress (ueIpIface.GetAddress (0), port)));
  ueNode->AddApplication (sink);
  apps.Add (sink);

  auto client = CreateObject<UdpClient> ();
  client->SetAttribute ("RemotePort", UintegerValue (port));
  client->SetAttribute ("MaxPackets", UintegerValue (1000000));
  client->SetAttribute ("Interval", TimeValue (Seconds (0.0001)));
  client->SetAttribute ("PacketSize", UintegerValue (1000));
  client->SetAttribute ("RemoteAddress", AddressValue (ueIpIface.GetAddress (0)));
  pgwNode->AddApplication (client);

  apps.Add (client);
  apps.Start (Seconds (1.0));

  Simulator::Stop (Seconds (2.0));
  Simulator::Run ();

  m_actualCellId = ueDev->GetRrc ()->GetCellId ();

  Simulator::Destroy ();
  return 8e-6 * sink->GetTotalRx ();
}

void
LteAggregationThroughputScaleTestCase::DoRun ()
{
  std::vector<double> throughputs;
  for (uint8_t i = 1; i <= 4; i++)
    {
      throughputs.push_back (GetThroughput (i) / i);
      NS_TEST_ASSERT_MSG_EQ (m_expectedCellId, m_actualCellId, "UE has attached to an unexpected cell");
    }
  double average = std::accumulate(begin (throughputs), end (throughputs), 0.0) / throughputs.size ();
  for (double throughput: throughputs)
    {
      NS_TEST_ASSERT_MSG_EQ_TOL (throughput, average, average * 0.01, "Throughput does not scale with number of component carriers");
    }
}
