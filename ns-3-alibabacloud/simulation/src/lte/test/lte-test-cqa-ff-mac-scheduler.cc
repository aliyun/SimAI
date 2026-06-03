/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author:  Biljana Bojovic<bbojovic@cttc.es>
 *          Dizhi Zhou <dizhi.zhou@gmail.com>
 *          Marco Miozzo <marco.miozzo@cttc.es>,
 *          Nicola Baldo <nbaldo@cttc.es>
 *
 */

#include <iostream>
#include <sstream>
#include <string>

#include <ns3/object.h>
#include <ns3/spectrum-interference.h>
#include <ns3/spectrum-error-model.h>
#include <ns3/log.h>
#include <ns3/test.h>
#include <ns3/simulator.h>
#include <ns3/packet.h>
#include <ns3/ptr.h>
#include "ns3/radio-bearer-stats-calculator.h"
#include <ns3/constant-position-mobility-model.h>
#include <ns3/ff-mac-scheduler.h>
#include <ns3/eps-bearer.h>
#include <ns3/node-container.h>
#include <ns3/mobility-helper.h>
#include <ns3/net-device-container.h>
#include <ns3/lte-ue-net-device.h>
#include <ns3/lte-enb-net-device.h>
#include <ns3/lte-ue-rrc.h>
#include <ns3/lte-helper.h>
#include "ns3/string.h"
#include "ns3/double.h"
#include <ns3/lte-enb-phy.h>
#include <ns3/lte-ue-phy.h>
#include <ns3/boolean.h>
#include <ns3/enum.h>

#include "ns3/point-to-point-epc-helper.h"
#include "ns3/network-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-helper.h"

#include "lte-test-cqa-ff-mac-scheduler.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LenaTestCqaFfMacScheduler");

LenaTestCqaFfMacSchedulerSuite::LenaTestCqaFfMacSchedulerSuite ()
  : TestSuite ("lte-cqa-ff-mac-scheduler", SYSTEM)
{
  NS_LOG_INFO ("creating LenaTestCqaFfMacSchedulerSuite");

  bool errorModel = false;

  // General config
  // Traffic: UDP traffic with fixed rate
  // Token generation rate = traffic rate
  // RLC header length = 2 bytes, PDCP header = 2 bytes
  // Simulation time = 1.0 sec
  // Throughput in this file is calculated in RLC layer

  //Test Case 1: homogeneous flow test in CQA (same distance)
  // DOWNLINK -> DISTANCE 0 -> MCS 28 -> Itbs 26 (from table 7.1.7.2.1-1 of 36.2    13)
  // Traffic info
  //   UDP traffic: payload size = 200 bytes, interval = 1 ms
  //   UDP rate in scheduler: (payload + RLC header + PDCP header + IP header + UDP header) * 1000 byte/sec -> 232000 byte/rate 
  // Totol bandwidth: 24 PRB at Itbs 26 -> 2196 -> 2196000 byte/sec
  // 1 user -> 232000 * 1 = 232000 < 2196000 -> throughput = 232000 byte/sec
  // 3 user -> 232000 * 3 = 696000 < 2196000 -> througphut = 232000 byte/sec
  // 6 user -> 232000 * 6 = 139200 < 2196000 -> throughput = 232000 byte/sec
  // 12 user -> 232000 * 12 = 2784000 > 2196000 -> throughput = 2196000 / 12 = 183000 byte/sec
  // UPLINK -> DISTANCE 0 -> MCS 28 -> Itbs 26 (from table 7.1.7.2.1-1 of 36.2    13)
  // 1 user -> 25 PRB at Itbs 26 -> 2292 -> 2292000 > 232000 -> throughput = 232000 bytes/sec
  // 3 users -> 8 PRB at Itbs 26 -> 749 -> 749000 > 232000 -> throughput = 232000 bytes/sec 
  // 6 users -> 4 PRB at Itbs 26 -> 373 -> 373000 > 232000 -> throughput = 232000 bytes/sec
  // 12 users -> 2 PRB at Itbs 26 -> 185 -> 185000 < 232000 -> throughput = 185000 bytes/sec
  AddTestCase (new LenaCqaFfMacSchedulerTestCase1 (1,0,232000,232000,200,1,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaCqaFfMacSchedulerTestCase1 (3,0,232000,232000,200,1,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaCqaFfMacSchedulerTestCase1 (6,0,232000,232000,200,1,errorModel), TestCase::EXTENSIVE);

  //AddTestCase (new LenaCqaFfMacSchedulerTestCase1 (12,0,183000,185000,200,1,errorModel));// simulation time = 1.5, otherwise, ul test will fail

  // DOWNLINK - DISTANCE 4800 -> MCS 22 -> Itbs 20 (from table 7.1.7.2.1-1 of 36.213)
  // Traffic info
  //   UDP traffic: payload size = 200 bytes, interval = 1 ms
  //   UDP rate in scheduler: (payload + RLC header + PDCP header + IP header + UDP header) * 1000 byte/sec -> 232000 byte/rate 
  // Totol bandwidth: 24 PRB at Itbs 20 -> 1383 -> 1383000 byte/sec
  // 1 user -> 903000 * 1 = 232000 < 1383000 -> throughput = 232000 byte/sec
  // 3 user -> 232000 * 3 = 696000 < 1383000 -> througphut = 232000 byte/sec
  // 6 user -> 232000 * 6 = 139200 > 1383000 -> throughput = 1383000 / 6 = 230500 byte/sec
  // 12 user -> 232000 * 12 = 2784000 > 1383000 -> throughput =  1383000 / 12 = 115250 byte/sec
  // UPLINK - DISTANCE 4800 -> MCS 14 -> Itbs 13 (from table 7.1.7.2.1-1 of 36.213)
  // 1 user -> 25 PRB at Itbs 13 -> 807 -> 807000 > 232000 -> throughput = 232000 bytes/sec
  // 3 users -> 8 PRB at Itbs 13 -> 253 -> 253000 > 232000 -> throughput = 232000 bytes/sec
  // 6 users -> 4 PRB at Itbs 13 -> 125 -> 125000 < 232000 -> throughput = 125000 bytes/sec
  // after the patch enforcing min 3 PRBs per UE:
  // 12 users -> 3 PRB at Itbs 13 -> 93  bytes * 8/12 UE/TTI  -> 62000 < 232000 -> throughput = 62000  bytes/sec
  AddTestCase (new LenaCqaFfMacSchedulerTestCase1 (1,4800,232000,232000,200,1,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaCqaFfMacSchedulerTestCase1 (3,4800,232000,232000,200,1,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaCqaFfMacSchedulerTestCase1 (6,4800,230500,125000,200,1,errorModel), TestCase::EXTENSIVE);
  //AddTestCase (new LenaCqaFfMacSchedulerTestCase1 (12,4800,115250,62000,200,1,errorModel)); // simulation time = 1.5, otherwise, ul test will fail

  // DOWNLINK - DISTANCE 6000 -> MCS 20 -> Itbs 18 (from table 7.1.7.2.1-1 of 36.213)
  // Traffic info
  //   UDP traffic: payload size = 200 bytes, interval = 1 ms
  //   UDP rate in scheduler: (payload + RLC header + PDCP header + IP header + UDP header) * 1000 byte/sec -> 232000 byte/rate 
  // Totol bandwidth: 24 PRB at Itbs 18 -> 1191 -> 1191000 byte/sec
  // 1 user -> 903000 * 1 = 232000 < 1191000 -> throughput = 232000 byte/sec
  // 3 user -> 232000 * 3 = 696000 < 1191000 -> througphut = 232000 byte/sec
  // 6 user -> 232000 * 6 = 1392000 > 1191000 -> throughput = 1191000 / 6 = 198500 byte/sec
  // 12 user -> 232000 * 12 = 2784000 > 1191000 -> throughput =  1191000 / 12 = 99250 byte/sec

  // UPLINK - DISTANCE 6000 -> MCS 12 -> Itbs 11 (from table 7.1.7.2.1-1 of 36.213)
  // 1 user -> 25 PRB at Itbs 11 -> 621 -> 621000 > 232000 -> throughput = 232000 bytes/sec
  // 3 users -> 8 PRB at Itbs 11 -> 201 -> 201000 < 232000 -> throughput = 201000  bytes/sec
  // 6 users -> 4 PRB at Itbs 11 -> 97 -> 97000 < 232000 -> throughput = 97000 bytes/sec
  // after the patch enforcing min 3 PRBs per UE:
  // 12 users -> 3 PRB at Itbs 11 -> 73 bytes * 8/12 UE/TTI -> 48667 < 232000 -> throughput = 48667 bytes/sec
  AddTestCase (new LenaCqaFfMacSchedulerTestCase1 (1,6000,232000,232000,200,1,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaCqaFfMacSchedulerTestCase1 (3,6000,232000,201000,200,1,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaCqaFfMacSchedulerTestCase1 (6,6000,198500,97000,200,1,errorModel), TestCase::EXTENSIVE);
  //AddTestCase (new LenaCqaFfMacSchedulerTestCase1 (12,6000,99250,48667,200,1, errorModel)); // simulation time = 1.5, otherwise, ul test will fail

  // DOWNLINK - DISTANCE 10000 -> MCS 14 -> Itbs 13 (from table 7.1.7.2.1-1 of 36.213)
  // Traffic info
  //   UDP traffic: payload size = 200 bytes, interval = 1 ms
  //   UDP rate in scheduler: (payload + RLC header + PDCP header + IP header + UDP header) * 1000 byte/sec -> 232000 byte/rate 
  // Totol bandwidth: 24 PRB at Itbs 13 -> 775 -> 775000 byte/sec
  // 1 user -> 903000 * 1 = 232000 < 775000 -> throughput = 232000 byte/sec
  // 3 user -> 232000 * 3 = 696000 > 775000 -> througphut = 232000 byte/sec
  // 6 user -> 232000 * 6 = 139200 > 775000 -> throughput = 775000 / 6 = 129166 byte/sec
  // 12 user -> 232000 * 12 = 2784000 > 775000 -> throughput =  775000 / 12 = 64583 byte/sec
  // UPLINK - DISTANCE 10000 -> MCS 8 -> Itbs 8 (from table 7.1.7.2.1-1 of 36.213)
  // 1 user -> 24 PRB at Itbs 8 -> 437 -> 437000 > 232000 -> throughput = 232000 bytes/sec
  // 3 users -> 8 PRB at Itbs 8 -> 137 -> 137000 < 232000 -> throughput = 137000 bytes/sec
  // 6 users -> 4 PRB at Itbs 8 -> 67 -> 67000 < 232000 -> throughput = 67000 bytes/sec
  // after the patch enforcing min 3 PRBs per UE:
  // 12 users -> 3 PRB at Itbs 8 -> 49 bytes * 8/12 UE/TTI -> 32667 < 232000 -> throughput = 32667  bytes/sec
  AddTestCase (new LenaCqaFfMacSchedulerTestCase1 (1,10000,232000,232000,200,1,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaCqaFfMacSchedulerTestCase1 (3,10000,232000,137000,200,1,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaCqaFfMacSchedulerTestCase1 (6,10000,129166,67000,200,1,errorModel), TestCase::EXTENSIVE);
  //AddTestCase (new LenaCqaFfMacSchedulerTestCase1 (12,10000,64583,32667,200,1,errorModel));// simulation time = 1.5, otherwise, ul test will fail


  // DOWNLINK - DISTANCE 100000 -> CQI == 0 -> out of range -> 0 bytes/sec
  // UPLINK - DISTANCE 100000 -> CQI == 0 -> out of range -> 0 bytes/sec
  AddTestCase (new LenaCqaFfMacSchedulerTestCase1 (1,100000,0,0,200,1,errorModel), TestCase::QUICK);

  // Test Case 2: homogeneous flow test in CQA (different distance)
  // Traffic1 info
  //   UDP traffic: payload size = 100 bytes, interval = 1 ms
  //   UDP rate in scheduler: (payload + RLC header + PDCP header + IP header + UDP header) * 1000 byte/sec -> 132000 byte/rate 
  // Maximum throughput = 4 / ( 1/2196000 + 1/1191000 + 1/1383000 + 1/775000 ) = 1209046 byte/s
  // 132000 * 4 = 528000 < 1209046 -> estimated throughput in downlink = 132000 byte/sec
  std::vector<double> dist1;
  dist1.push_back (0);       // User 0 distance --> MCS 28
  dist1.push_back (4800);    // User 1 distance --> MCS 22
  dist1.push_back (6000);    // User 2 distance --> MCS 20
  dist1.push_back (10000);   // User 3 distance --> MCS 14
  std::vector<uint16_t> packetSize1;
  packetSize1.push_back (100);
  packetSize1.push_back (100);
  packetSize1.push_back (100);
  packetSize1.push_back (100);
  std::vector<uint32_t> estThrCqaDl1;
  estThrCqaDl1.push_back (132000); // User 0 estimated TTI throughput from CQA
  estThrCqaDl1.push_back (132000); // User 1 estimated TTI throughput from CQA
  estThrCqaDl1.push_back (132000); // User 2 estimated TTI throughput from CQA
  estThrCqaDl1.push_back (132000); // User 3 estimated TTI throughput from CQA
  AddTestCase (new LenaCqaFfMacSchedulerTestCase2 (dist1,estThrCqaDl1,packetSize1,1,errorModel), TestCase::QUICK);

  // Traffic2 info
  //   UDP traffic: payload size = 200 bytes, interval = 1 ms
  //   UDP rate in scheduler: (payload + RLC header + PDCP header + IP header + UDP header) * 1000 byte/sec -> 232000 byte/rate 
  // Maximum throughput = 4 / ( 1/2196000 + 1/1191000 + 1/1383000 + 1/775000 ) = 1209046 byte/s
  // 232000 * 4 = 928000 < 1209046 -> estimated throughput in downlink = 928000 / 4 = 230000 byte/sec
  std::vector<double> dist2;
  dist2.push_back (0);       // User 0 distance --> MCS 28
  dist2.push_back (4800);    // User 1 distance --> MCS 22
  dist2.push_back (6000);    // User 2 distance --> MCS 20
  dist2.push_back (10000);   // User 3 distance --> MCS 14
  std::vector<uint16_t> packetSize2;
  packetSize2.push_back (200);
  packetSize2.push_back (200);
  packetSize2.push_back (200);
  packetSize2.push_back (200);
  std::vector<uint32_t> estThrCqaDl2;
  estThrCqaDl2.push_back (230000); // User 0 estimated TTI throughput from CQA
  estThrCqaDl2.push_back (230000); // User 1 estimated TTI throughput from CQA
  estThrCqaDl2.push_back (230000); // User 2 estimated TTI throughput from CQA
  estThrCqaDl2.push_back (230000); // User 3 estimated TTI throughput from CQA
  AddTestCase (new LenaCqaFfMacSchedulerTestCase2 (dist2,estThrCqaDl2,packetSize2,1,errorModel), TestCase::QUICK);

  // Test Case 3: heterogeneous flow test in CQA
  //   UDP traffic: payload size = [100,200,300] bytes, interval = 1 ms
  //   UDP rate in scheduler: (payload + RLC header + PDCP header + IP header + UDP header) * 1000 byte/sec -> [132000, 232000, 332000] byte/rate
  // Maximum throughput = 3 / ( 1/2196000 + 1/1191000 + 1/1383000) = 1486569 byte/s
  // 132000 + 232000 + 332000 = 696000 < 1486569 -> estimated throughput in downlink = [132000, 232000, 332000] byte/sec
  std::vector<double> dist3;
  dist3.push_back (0);    // User 0 distance --> MCS 28
  dist3.push_back (4800);    // User 1 distance --> MCS 22
  dist3.push_back (6000);    // User 2 distance --> MCS 20
  std::vector<uint16_t> packetSize3;
  packetSize3.push_back (100);
  packetSize3.push_back (200);
  packetSize3.push_back (300);
  std::vector<uint32_t> estThrCqaDl3;
  estThrCqaDl3.push_back (132000); // User 0 estimated TTI throughput from CQA
  estThrCqaDl3.push_back (232000); // User 1 estimated TTI throughput from CQA
  estThrCqaDl3.push_back (332000); // User 2 estimated TTI throughput from CQA
  AddTestCase (new LenaCqaFfMacSchedulerTestCase2 (dist3,estThrCqaDl3,packetSize3,1,errorModel), TestCase::QUICK);

}

static LenaTestCqaFfMacSchedulerSuite lenaTestCqaFfMacSchedulerSuite;

// --------------- T E S T - C A S E   # 1 ------------------------------


std::string 
LenaCqaFfMacSchedulerTestCase1::BuildNameString (uint16_t nUser, double dist)
{
  std::ostringstream oss;
  oss << nUser << " UEs, distance " << dist << " m";
  return oss.str ();
}


LenaCqaFfMacSchedulerTestCase1::LenaCqaFfMacSchedulerTestCase1 (uint16_t nUser, double dist, double thrRefDl, double thrRefUl, uint16_t packetSize, uint16_t interval,bool errorModelEnabled)
  : TestCase (BuildNameString (nUser, dist)),
    m_nUser (nUser),
    m_dist (dist),
    m_packetSize (packetSize),
    m_interval (interval),
    m_thrRefDl (thrRefDl),
    m_thrRefUl (thrRefUl),
    m_errorModelEnabled (errorModelEnabled)
{
}

LenaCqaFfMacSchedulerTestCase1::~LenaCqaFfMacSchedulerTestCase1 ()
{
}

void
LenaCqaFfMacSchedulerTestCase1::DoRun (void)
{
  NS_LOG_FUNCTION (this << GetName ());

  if (!m_errorModelEnabled)
    {
      Config::SetDefault ("ns3::LteSpectrumPhy::CtrlErrorModelEnabled", BooleanValue (false));
      Config::SetDefault ("ns3::LteSpectrumPhy::DataErrorModelEnabled", BooleanValue (false));
    }

  Config::SetDefault ("ns3::LteHelper::UseIdealRrc", BooleanValue (true));
  Config::SetDefault ("ns3::MacStatsCalculator::DlOutputFilename", StringValue (CreateTempDirFilename ("DlMacStats.txt")));
  Config::SetDefault ("ns3::MacStatsCalculator::UlOutputFilename", StringValue (CreateTempDirFilename ("UlMacStats.txt")));
  Config::SetDefault ("ns3::RadioBearerStatsCalculator::DlRlcOutputFilename", StringValue (CreateTempDirFilename ("DlRlcStats.txt")));
  Config::SetDefault ("ns3::RadioBearerStatsCalculator::UlRlcOutputFilename", StringValue (CreateTempDirFilename ("UlRlcStats.txt")));

  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  Ptr<PointToPointEpcHelper>  epcHelper = CreateObject<PointToPointEpcHelper> ();
  lteHelper->SetEpcHelper (epcHelper);

  //LogComponentEnable ("CqaFfMacScheduler", LOG_DEBUG);

  Ptr<Node> pgw = epcHelper->GetPgwNode ();

  // Create a single RemoteHost
  NodeContainer remoteHostContainer;
  remoteHostContainer.Create (1);
  Ptr<Node> remoteHost = remoteHostContainer.Get (0);
  InternetStackHelper internet;
  internet.Install (remoteHostContainer);

  // Create the Internet
  PointToPointHelper p2ph;
  p2ph.SetDeviceAttribute ("DataRate", DataRateValue (DataRate ("100Gb/s")));
  p2ph.SetDeviceAttribute ("Mtu", UintegerValue (1500));
  p2ph.SetChannelAttribute ("Delay", TimeValue (Seconds (0.001)));
  NetDeviceContainer internetDevices = p2ph.Install (pgw, remoteHost);
  Ipv4AddressHelper ipv4h;
  ipv4h.SetBase ("1.0.0.0", "255.0.0.0");
  Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign (internetDevices);
  // interface 0 is localhost, 1 is the p2p device
  Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress (1);

  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  Ptr<Ipv4StaticRouting> remoteHostStaticRouting = ipv4RoutingHelper.GetStaticRouting (remoteHost->GetObject<Ipv4> ());
  remoteHostStaticRouting->AddNetworkRouteTo (Ipv4Address ("7.0.0.0"), Ipv4Mask ("255.0.0.0"), 1);

  //Config::SetDefault ("ns3::LteAmc::AmcModel", EnumValue (LteAmc::PiroEW2010));
  //Config::SetDefault ("ns3::LteAmc::Ber", DoubleValue (0.00005));
  //Config::SetDefault ("ns3::LteSpectrumPhy::CtrlErrorModelEnabled", BooleanValue (false));
  //Config::SetDefault ("ns3::LteSpectrumPhy::DataErrorModelEnabled", BooleanValue (false));

  //Config::SetDefault ("ns3::LteEnbRrc::EpsBearerToRlcMapping", EnumValue (LteHelper::RLC_UM_ALWAYS));

//   LogComponentDisableAll (LOG_LEVEL_ALL);
  //LogComponentEnable ("LenaTestCqaFfMacScheduler", LOG_LEVEL_ALL);
   
  lteHelper->SetAttribute ("PathlossModel", StringValue ("ns3::FriisSpectrumPropagationLossModel"));

  // Create Nodes: eNodeB and UE
  NodeContainer enbNodes;
  NodeContainer ueNodes;
  enbNodes.Create (1);
  ueNodes.Create (m_nUser);

  // Install Mobility Model
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (enbNodes);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (ueNodes);

  // Create Devices and install them in the Nodes (eNB and UE)
  NetDeviceContainer enbDevs;
  NetDeviceContainer ueDevs;
  lteHelper->SetSchedulerType ("ns3::CqaFfMacScheduler");
  lteHelper->SetSchedulerAttribute ("UlCqiFilter", EnumValue (FfMacScheduler::SRS_UL_CQI));
  enbDevs = lteHelper->InstallEnbDevice (enbNodes);
  ueDevs = lteHelper->InstallUeDevice (ueNodes);


  Ptr<LteEnbNetDevice> lteEnbDev = enbDevs.Get (0)->GetObject<LteEnbNetDevice> ();
  Ptr<LteEnbPhy> enbPhy = lteEnbDev->GetPhy ();
  enbPhy->SetAttribute ("TxPower", DoubleValue (30.0));
  enbPhy->SetAttribute ("NoiseFigure", DoubleValue (5.0));

  // Set UEs' position and power
  for (int i = 0; i < m_nUser; i++)
    {
      Ptr<ConstantPositionMobilityModel> mm = ueNodes.Get (i)->GetObject<ConstantPositionMobilityModel> ();
      mm->SetPosition (Vector (m_dist, 0.0, 0.0));
      Ptr<LteUeNetDevice> lteUeDev = ueDevs.Get (i)->GetObject<LteUeNetDevice> ();
      Ptr<LteUePhy> uePhy = lteUeDev->GetPhy ();
      uePhy->SetAttribute ("TxPower", DoubleValue (23.0));
      uePhy->SetAttribute ("NoiseFigure", DoubleValue (9.0));
    }

  // Install the IP stack on the UEs
  internet.Install (ueNodes);
  Ipv4InterfaceContainer ueIpIface;
  ueIpIface = epcHelper->AssignUeIpv4Address (NetDeviceContainer (ueDevs));

  // Assign IP address to UEs
  for (uint32_t u = 0; u < ueNodes.GetN (); ++u)
    {
      Ptr<Node> ueNode = ueNodes.Get (u);
      // Set the default gateway for the UE
      Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting (ueNode->GetObject<Ipv4> ());
      ueStaticRouting->SetDefaultRoute (epcHelper->GetUeDefaultGatewayAddress (), 1);
    }

  // Attach a UE to a eNB
  lteHelper->Attach (ueDevs, enbDevs.Get (0));

  // Activate an EPS bearer on all UEs
  for (uint32_t u = 0; u < ueNodes.GetN (); ++u)
    {
      Ptr<NetDevice> ueDevice = ueDevs.Get (u);
      GbrQosInformation qos;
      qos.gbrDl = (m_packetSize + 32) * (1000 / m_interval) * 8;  // bit/s, considering IP, UDP, RLC, PDCP header size
      qos.gbrUl = (m_packetSize + 32) * (1000 / m_interval) * 8;
      qos.mbrDl = 0;
      qos.mbrUl = 0;
      
      enum EpsBearer::Qci q = EpsBearer::GBR_CONV_VOICE;
      EpsBearer bearer (q, qos);
      lteHelper->ActivateDedicatedEpsBearer (ueDevice, bearer, EpcTft::Default ());  
    }

  // Install downlink and uplink applications
  uint16_t dlPort = 1234;
  uint16_t ulPort = 2000;
  ApplicationContainer clientApps;
  ApplicationContainer serverApps;
  PacketSinkHelper dlPacketSinkHelper ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), dlPort));

  for (uint32_t u = 0; u < ueNodes.GetN (); ++u)
    {
      ++ulPort;
      PacketSinkHelper ulPacketSinkHelper ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), ulPort));
      serverApps.Add (ulPacketSinkHelper.Install (remoteHost));  // receive packets from UEs
      serverApps.Add (dlPacketSinkHelper.Install (ueNodes.Get (u))); // receive packets from remotehost

      UdpClientHelper dlClient (ueIpIface.GetAddress (u), dlPort); // downlink packets generator
      dlClient.SetAttribute ("Interval", TimeValue (MilliSeconds (m_interval)));
      dlClient.SetAttribute ("MaxPackets", UintegerValue (1000000));
      dlClient.SetAttribute ("PacketSize", UintegerValue (m_packetSize));

      UdpClientHelper ulClient (remoteHostAddr, ulPort);           // uplink packets generator
      ulClient.SetAttribute ("Interval", TimeValue (MilliSeconds (m_interval)));
      ulClient.SetAttribute ("MaxPackets", UintegerValue (1000000));
      ulClient.SetAttribute ("PacketSize", UintegerValue (m_packetSize));

      clientApps.Add (dlClient.Install (remoteHost));
      clientApps.Add (ulClient.Install (ueNodes.Get (u)));
    }

  serverApps.Start (Seconds (0.030));
  clientApps.Start (Seconds (0.030));

  double statsStartTime = 0.300; // need to allow for RRC connection establishment + SRS
  double statsDuration = 0.6;
  double tolerance = 0.1;
  Simulator::Stop (Seconds (statsStartTime + statsDuration - 0.0001));

  lteHelper->EnableRlcTraces ();
  lteHelper->EnableMacTraces ();
  Ptr<RadioBearerStatsCalculator> rlcStats = lteHelper->GetRlcStats ();
  rlcStats->SetAttribute ("StartTime", TimeValue (Seconds (statsStartTime)));
  rlcStats->SetAttribute ("EpochDuration", TimeValue (Seconds (statsDuration)));

  Simulator::Run ();


  NS_LOG_INFO ("DL - Test with " << m_nUser << " user(s) at distance " << m_dist);
  std::vector <uint64_t> dlDataRxed;
  for (int i = 0; i < m_nUser; i++)
    {
      // get the imsi
      uint64_t imsi = ueDevs.Get (i)->GetObject<LteUeNetDevice> ()->GetImsi ();
      // get the lcId
      uint8_t lcId = 4;
      uint64_t data = rlcStats->GetDlRxData (imsi, lcId);
      dlDataRxed.push_back (data);
      NS_LOG_INFO ("\tUser " << i << " imsi " << imsi << " bytes rxed " << (double)dlDataRxed.at (i) << "  thr " << (double)dlDataRxed.at (i) / statsDuration << " ref " << m_thrRefDl);
    }

  for (int i = 0; i < m_nUser; i++)
    {
      NS_TEST_ASSERT_MSG_EQ_TOL ((double)dlDataRxed.at (i) / statsDuration, m_thrRefDl, m_thrRefDl * tolerance, " Unfair Throughput!");
    }

  /**
  * Check that the uplink assignment is done in a "round robin" manner
  */

  NS_LOG_INFO ("UL - Test with " << m_nUser << " user(s) at distance " << m_dist);
  std::vector <uint64_t> ulDataRxed;
  for (int i = 0; i < m_nUser; i++)
    {
      // get the imsi
      uint64_t imsi = ueDevs.Get (i)->GetObject<LteUeNetDevice> ()->GetImsi ();
      // get the lcId
      uint8_t lcId = 4;
      ulDataRxed.push_back (rlcStats->GetUlRxData (imsi, lcId));
      NS_LOG_INFO ("\tUser " << i << " imsi " << imsi << " bytes rxed " << (double)ulDataRxed.at (i) << "  thr " << (double)ulDataRxed.at (i) / statsDuration << " ref " << m_thrRefUl);
    }

  for (int i = 0; i < m_nUser; i++)
    {
      NS_TEST_ASSERT_MSG_EQ_TOL ((double)ulDataRxed.at (i) / statsDuration, m_thrRefUl, m_thrRefUl * tolerance, " Unfair Throughput!");
    }
  Simulator::Destroy ();

}



// --------------- T E S T - C A S E   # 2 ------------------------------


std::string 
LenaCqaFfMacSchedulerTestCase2::BuildNameString (uint16_t nUser, std::vector<double> dist)
{
  std::ostringstream oss;
  oss << "distances (m) = [ " ;
  for (std::vector<double>::iterator it = dist.begin (); it != dist.end (); ++it)
    {
      oss << *it << " ";
    }
  oss << "]";
  return oss.str ();
}


LenaCqaFfMacSchedulerTestCase2::LenaCqaFfMacSchedulerTestCase2 (std::vector<double> dist, std::vector<uint32_t> estThrCqaDl, std::vector<uint16_t> packetSize, uint16_t interval,bool errorModelEnabled)
  : TestCase (BuildNameString (dist.size (), dist)),
    m_nUser (dist.size ()),
    m_dist (dist),
    m_packetSize (packetSize),
    m_interval (interval),
    m_estThrCqaDl (estThrCqaDl),
    m_errorModelEnabled (errorModelEnabled)
{
}

LenaCqaFfMacSchedulerTestCase2::~LenaCqaFfMacSchedulerTestCase2 ()
{
}

void
LenaCqaFfMacSchedulerTestCase2::DoRun (void)
{

  if (!m_errorModelEnabled)
    {
      Config::SetDefault ("ns3::LteSpectrumPhy::CtrlErrorModelEnabled", BooleanValue (false));
      Config::SetDefault ("ns3::LteSpectrumPhy::DataErrorModelEnabled", BooleanValue (false));
    }

  Config::SetDefault ("ns3::LteHelper::UseIdealRrc", BooleanValue (true));
  Config::SetDefault ("ns3::MacStatsCalculator::DlOutputFilename", StringValue (CreateTempDirFilename ("DlMacStats.txt")));
  Config::SetDefault ("ns3::MacStatsCalculator::UlOutputFilename", StringValue (CreateTempDirFilename ("UlMacStats.txt")));
  Config::SetDefault ("ns3::RadioBearerStatsCalculator::DlRlcOutputFilename", StringValue (CreateTempDirFilename ("DlRlcStats.txt")));
  Config::SetDefault ("ns3::RadioBearerStatsCalculator::UlRlcOutputFilename", StringValue (CreateTempDirFilename ("UlRlcStats.txt")));

  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  Ptr<PointToPointEpcHelper>  epcHelper = CreateObject<PointToPointEpcHelper> ();
  lteHelper->SetEpcHelper (epcHelper);

  Ptr<Node> pgw = epcHelper->GetPgwNode ();

  // Create a single RemoteHost
  NodeContainer remoteHostContainer;
  remoteHostContainer.Create (1);
  Ptr<Node> remoteHost = remoteHostContainer.Get (0);
  InternetStackHelper internet;
  internet.Install (remoteHostContainer);

  // Create the Internet
  PointToPointHelper p2ph;
  p2ph.SetDeviceAttribute ("DataRate", DataRateValue (DataRate ("100Gb/s")));
  p2ph.SetDeviceAttribute ("Mtu", UintegerValue (1500));
  p2ph.SetChannelAttribute ("Delay", TimeValue (Seconds (0.001)));
  NetDeviceContainer internetDevices = p2ph.Install (pgw, remoteHost);
  Ipv4AddressHelper ipv4h;
  ipv4h.SetBase ("1.0.0.0", "255.0.0.0");
  Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign (internetDevices);
  // interface 0 is localhost, 1 is the p2p device
  Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress (1);

  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  Ptr<Ipv4StaticRouting> remoteHostStaticRouting = ipv4RoutingHelper.GetStaticRouting (remoteHost->GetObject<Ipv4> ());
  remoteHostStaticRouting->AddNetworkRouteTo (Ipv4Address ("7.0.0.0"), Ipv4Mask ("255.0.0.0"), 1);


//   LogComponentDisableAll (LOG_LEVEL_ALL);
  //LogComponentEnable ("LenaTestCqaFfMacScheduler", LOG_LEVEL_ALL);
   
  lteHelper->SetAttribute ("PathlossModel", StringValue ("ns3::FriisSpectrumPropagationLossModel"));

  // Create Nodes: eNodeB and UE
  NodeContainer enbNodes;
  NodeContainer ueNodes;
  enbNodes.Create (1);
  ueNodes.Create (m_nUser);

  // Install Mobility Model
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (enbNodes);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (ueNodes);

  // Create Devices and install them in the Nodes (eNB and UE)
  NetDeviceContainer enbDevs;
  NetDeviceContainer ueDevs;
  lteHelper->SetSchedulerType ("ns3::CqaFfMacScheduler");
  lteHelper->SetSchedulerAttribute ("UlCqiFilter", EnumValue (FfMacScheduler::SRS_UL_CQI));
  enbDevs = lteHelper->InstallEnbDevice (enbNodes);
  ueDevs = lteHelper->InstallUeDevice (ueNodes);

  Ptr<LteEnbNetDevice> lteEnbDev = enbDevs.Get (0)->GetObject<LteEnbNetDevice> ();
  Ptr<LteEnbPhy> enbPhy = lteEnbDev->GetPhy ();
  enbPhy->SetAttribute ("TxPower", DoubleValue (30.0));
  enbPhy->SetAttribute ("NoiseFigure", DoubleValue (5.0));

  // Set UEs' position and power
  for (int i = 0; i < m_nUser; i++)
    {
      Ptr<ConstantPositionMobilityModel> mm = ueNodes.Get (i)->GetObject<ConstantPositionMobilityModel> ();
      mm->SetPosition (Vector (m_dist.at (i), 0.0, 0.0));
      Ptr<LteUeNetDevice> lteUeDev = ueDevs.Get (i)->GetObject<LteUeNetDevice> ();
      Ptr<LteUePhy> uePhy = lteUeDev->GetPhy ();
      uePhy->SetAttribute ("TxPower", DoubleValue (23.0));
      uePhy->SetAttribute ("NoiseFigure", DoubleValue (9.0));
    }

  // Install the IP stack on the UEs
  internet.Install (ueNodes);
  Ipv4InterfaceContainer ueIpIface;
  ueIpIface = epcHelper->AssignUeIpv4Address (NetDeviceContainer (ueDevs));

  // Assign IP address to UEs
  for (uint32_t u = 0; u < ueNodes.GetN (); ++u)
    {
      Ptr<Node> ueNode = ueNodes.Get (u);
      // Set the default gateway for the UE
      Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting (ueNode->GetObject<Ipv4> ());
      ueStaticRouting->SetDefaultRoute (epcHelper->GetUeDefaultGatewayAddress (), 1);
    }

  // Attach a UE to a eNB
  lteHelper->Attach (ueDevs, enbDevs.Get (0));

  // Activate an EPS bearer on all UEs

  for (uint32_t u = 0; u < ueNodes.GetN (); ++u)
    {
      Ptr<NetDevice> ueDevice = ueDevs.Get (u);
      GbrQosInformation qos;
      qos.gbrDl = (m_packetSize.at (u) + 32) * (1000 / m_interval) * 8;  // bit/s, considering IP, UDP, RLC, PDCP header size
      qos.gbrUl = (m_packetSize.at (u) + 32) * (1000 / m_interval) * 8;
      qos.mbrDl = qos.gbrDl;
      qos.mbrUl = qos.gbrUl;
  
      enum EpsBearer::Qci q = EpsBearer::GBR_CONV_VOICE;
      EpsBearer bearer (q, qos);
      lteHelper->ActivateDedicatedEpsBearer (ueDevice, bearer, EpcTft::Default ());  
    }


  // Install downlink and uplink applications
  uint16_t dlPort = 1234;
  uint16_t ulPort = 2000;
  ApplicationContainer clientApps;
  ApplicationContainer serverApps;
  PacketSinkHelper dlPacketSinkHelper ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), dlPort));

  for (uint32_t u = 0; u < ueNodes.GetN (); ++u)
    {
      ++ulPort;
      PacketSinkHelper ulPacketSinkHelper ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), ulPort));
      serverApps.Add (ulPacketSinkHelper.Install (remoteHost));  // receive packets from UEs
      serverApps.Add (dlPacketSinkHelper.Install (ueNodes.Get (u))); // receive packets from remotehost

      UdpClientHelper dlClient (ueIpIface.GetAddress (u), dlPort); // downlink packets generator
      dlClient.SetAttribute ("Interval", TimeValue (MilliSeconds (m_interval)));
      dlClient.SetAttribute ("MaxPackets", UintegerValue (1000000));
      dlClient.SetAttribute ("PacketSize", UintegerValue (m_packetSize.at (u)));

      UdpClientHelper ulClient (remoteHostAddr, ulPort);           // uplink packets generator
      ulClient.SetAttribute ("Interval", TimeValue (MilliSeconds (m_interval)));
      ulClient.SetAttribute ("MaxPackets", UintegerValue (1000000));
      ulClient.SetAttribute ("PacketSize", UintegerValue (m_packetSize.at (u)));

      clientApps.Add (dlClient.Install (remoteHost));
      clientApps.Add (ulClient.Install (ueNodes.Get (u)));
   }

  serverApps.Start (Seconds (0.030));
  clientApps.Start (Seconds (0.030));

  double statsStartTime = 0.04; // need to allow for RRC connection establishment + SRS
  double statsDuration = 0.5;
  double tolerance = 0.1;
  Simulator::Stop (Seconds (statsStartTime + statsDuration - 0.0001));

  lteHelper->EnableRlcTraces ();
  Ptr<RadioBearerStatsCalculator> rlcStats = lteHelper->GetRlcStats ();
  rlcStats->SetAttribute ("StartTime", TimeValue (Seconds (statsStartTime)));
  rlcStats->SetAttribute ("EpochDuration", TimeValue (Seconds (statsDuration)));


  Simulator::Run ();

  /**
   * Check that the downlink resource assignment is done according to CQA algorithm among users with different SINRs
   */

  NS_LOG_INFO ("DL - Test with " << m_nUser << " user(s)");
  std::vector <uint64_t> dlDataRxed;
  for (int i = 0; i < m_nUser; i++)
    {
      // get the imsi
      uint64_t imsi = ueDevs.Get (i)->GetObject<LteUeNetDevice> ()->GetImsi ();
      // get the lcId
      uint8_t lcId = 4;
      dlDataRxed.push_back (rlcStats->GetDlRxData (imsi, lcId));
      NS_LOG_INFO ("\tUser " << i << " dist " << m_dist.at (i) << " imsi " << imsi << " bytes rxed " << (double)dlDataRxed.at (i) << "  thr " << (double)dlDataRxed.at (i) / statsDuration << " ref " << m_estThrCqaDl.at (i));
    }

  for (int i = 0; i < m_nUser; i++)
    {
      NS_TEST_ASSERT_MSG_EQ_TOL ((double)dlDataRxed.at (i) / statsDuration, m_estThrCqaDl.at (i), m_estThrCqaDl.at (i) * tolerance, " Unfair Throughput!");
    }

  Simulator::Destroy ();

}
