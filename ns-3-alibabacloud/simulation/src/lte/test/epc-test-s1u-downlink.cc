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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */



#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/test.h"
#include "ns3/point-to-point-epc-helper.h"
#include "ns3/epc-enb-application.h"
#include "ns3/packet-sink-helper.h"
#include "ns3/udp-echo-helper.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/csma-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/inet-socket-address.h"
#include "ns3/packet-sink.h"
#include <ns3/ipv4-static-routing-helper.h>
#include <ns3/ipv4-static-routing.h>
#include "ns3/boolean.h"
#include "ns3/uinteger.h"
#include "ns3/config.h"
#include "ns3/eps-bearer.h"
#include "lte-test-entities.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("EpcTestS1uDownlink");


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Custom structure for testing UE downlink data
 */
struct UeDlTestData
{
  /**
   * Constructor
   *
   * \param n number of packets
   * \param s packet size
   */
  UeDlTestData (uint32_t n, uint32_t s);

  uint32_t numPkts; ///< number of packets
  uint32_t pktSize; ///< packet size
 
  Ptr<PacketSink> serverApp; ///< Server application
  Ptr<Application> clientApp; ///< Client application
};

UeDlTestData::UeDlTestData (uint32_t n, uint32_t s)
  : numPkts (n),
    pktSize (s)
{
}

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Custom structure for testing eNodeB downlink data, contains 
 * the list of data structures for UEs
 */
struct EnbDlTestData
{
  std::vector<UeDlTestData> ues; ///< list of data structure for different UEs
};



/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief EpcS1uDlTestCase class
 */
class EpcS1uDlTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param name the name of the test case instance
   * \param v list of eNodeB downlink test data information
   */
  EpcS1uDlTestCase (std::string name, std::vector<EnbDlTestData> v);
  virtual ~EpcS1uDlTestCase ();

private:
  virtual void DoRun (void);
  std::vector<EnbDlTestData> m_enbDlTestData; ///< ENB DL test data
};


EpcS1uDlTestCase::EpcS1uDlTestCase (std::string name, std::vector<EnbDlTestData> v)
  : TestCase (name),
    m_enbDlTestData (v)
{
}

EpcS1uDlTestCase::~EpcS1uDlTestCase ()
{
}
void 
EpcS1uDlTestCase::DoRun ()
{
  Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper> ();
  Ptr<Node> pgw = epcHelper->GetPgwNode ();

  // allow jumbo packets
  Config::SetDefault ("ns3::CsmaNetDevice::Mtu", UintegerValue (30000));
  Config::SetDefault ("ns3::PointToPointNetDevice::Mtu", UintegerValue (30000));
  epcHelper->SetAttribute ("S1uLinkMtu", UintegerValue (30000));
  
  // Create a single RemoteHost
  NodeContainer remoteHostContainer;
  remoteHostContainer.Create (1);
  Ptr<Node> remoteHost = remoteHostContainer.Get (0);
  InternetStackHelper internet;
  internet.Install (remoteHostContainer);

  // Create the internet
  PointToPointHelper p2ph;
  p2ph.SetDeviceAttribute ("DataRate",  DataRateValue (DataRate ("100Gb/s")));
  NetDeviceContainer internetDevices = p2ph.Install (pgw, remoteHost);  
  Ipv4AddressHelper ipv4h;
  ipv4h.SetBase ("1.0.0.0", "255.0.0.0");
  ipv4h.Assign (internetDevices);
  
  // setup default gateway for the remote hosts
  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  Ptr<Ipv4StaticRouting> remoteHostStaticRouting = ipv4RoutingHelper.GetStaticRouting (remoteHost->GetObject<Ipv4> ());

  // hardcoded UE addresses for now
  remoteHostStaticRouting->AddNetworkRouteTo (Ipv4Address ("7.0.0.0"), Ipv4Mask ("255.255.255.0"), 1);
  



  NodeContainer enbs;
  uint16_t cellIdCounter = 0;
  uint64_t imsiCounter = 0;

  for (std::vector<EnbDlTestData>::iterator enbit = m_enbDlTestData.begin ();
       enbit < m_enbDlTestData.end ();
       ++enbit)
    {
      Ptr<Node> enb = CreateObject<Node> ();
      enbs.Add (enb);

      // we test EPC without LTE, hence we use:
      // 1) a CSMA network to simulate the cell
      // 2) a raw socket opened on the CSMA device to simulate the LTE socket

      uint16_t cellId = ++cellIdCounter;

      NodeContainer ues;
      ues.Create (enbit->ues.size ());

      NodeContainer cell;
      cell.Add (ues);
      cell.Add (enb);

      CsmaHelper csmaCell;      
      NetDeviceContainer cellDevices = csmaCell.Install (cell);

      // the eNB's CSMA NetDevice acting as an LTE NetDevice. 
      Ptr<NetDevice> enbDevice = cellDevices.Get (cellDevices.GetN () - 1);

      // Note that the EpcEnbApplication won't care of the actual NetDevice type
      std::vector<uint16_t> cellIds;
      cellIds.push_back (cellId);
      epcHelper->AddEnb (enb, enbDevice, cellIds);

      // Plug test RRC entity
      Ptr<EpcEnbApplication> enbApp = enb->GetApplication (0)->GetObject<EpcEnbApplication> ();
      NS_ASSERT_MSG (enbApp != 0, "cannot retrieve EpcEnbApplication");
      Ptr<EpcTestRrc> rrc = CreateObject<EpcTestRrc> ();
      enb->AggregateObject (rrc);
      rrc->SetS1SapProvider (enbApp->GetS1SapProvider ());
      enbApp->SetS1SapUser (rrc->GetS1SapUser ());
      
      // we install the IP stack on UEs only
      InternetStackHelper internet;
      internet.Install (ues);

      // assign IP address to UEs, and install applications
      for (uint32_t u = 0; u < ues.GetN (); ++u)
        {
          Ptr<NetDevice> ueLteDevice = cellDevices.Get (u);
          Ipv4InterfaceContainer ueIpIface = epcHelper->AssignUeIpv4Address (NetDeviceContainer (ueLteDevice));

          Ptr<Node> ue = ues.Get (u);

          // disable IP Forwarding on the UE. This is because we use
          // CSMA broadcast MAC addresses for this test. The problem
          // won't happen with a LteUeNetDevice. 
          ue->GetObject<Ipv4> ()->SetAttribute ("IpForward", BooleanValue (false));
          
          uint16_t port = 1234;
          PacketSinkHelper packetSinkHelper ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), port));
          ApplicationContainer apps = packetSinkHelper.Install (ue);
          apps.Start (Seconds (1.0));
          apps.Stop (Seconds (10.0));
          enbit->ues[u].serverApp = apps.Get (0)->GetObject<PacketSink> ();
          
          Time interPacketInterval = Seconds (0.01);
          UdpEchoClientHelper client (ueIpIface.GetAddress (0), port);
          client.SetAttribute ("MaxPackets", UintegerValue (enbit->ues[u].numPkts));
          client.SetAttribute ("Interval", TimeValue (interPacketInterval));
          client.SetAttribute ("PacketSize", UintegerValue (enbit->ues[u].pktSize));
          apps = client.Install (remoteHost);
          apps.Start (Seconds (2.0));
          apps.Stop (Seconds (10.0));   
          enbit->ues[u].clientApp = apps.Get (0);

          uint64_t imsi = ++imsiCounter;
          epcHelper->AddUe (ueLteDevice, imsi);
          epcHelper->ActivateEpsBearer (ueLteDevice, imsi, EpcTft::Default (), EpsBearer (EpsBearer::NGBR_VIDEO_TCP_DEFAULT));
          Simulator::Schedule (MilliSeconds (10),
                               &EpcEnbS1SapProvider::InitialUeMessage,
                               enbApp->GetS1SapProvider (), imsi, (uint16_t) imsi);
        } 
            
    } 
  
  Simulator::Run ();

  for (std::vector<EnbDlTestData>::iterator enbit = m_enbDlTestData.begin ();
       enbit < m_enbDlTestData.end ();
       ++enbit)
    {
      for (std::vector<UeDlTestData>::iterator ueit = enbit->ues.begin ();
           ueit < enbit->ues.end ();
           ++ueit)
        {
          NS_TEST_ASSERT_MSG_EQ (ueit->serverApp->GetTotalRx (), (ueit->numPkts) * (ueit->pktSize), "wrong total received bytes");
        }      
    }
  
  Simulator::Destroy ();
}





/**
 * Test that the S1-U interface implementation works correctly
 */
class EpcS1uDlTestSuite : public TestSuite
{
public:
  EpcS1uDlTestSuite ();
  
} g_epcS1uDlTestSuiteInstance;

EpcS1uDlTestSuite::EpcS1uDlTestSuite ()
  : TestSuite ("epc-s1u-downlink", SYSTEM)
{  
  std::vector<EnbDlTestData> v1;  
  EnbDlTestData e1;
  UeDlTestData f1 (1, 100);
  e1.ues.push_back (f1);
  v1.push_back (e1);
  AddTestCase (new EpcS1uDlTestCase ("1 eNB, 1UE", v1), TestCase::QUICK);


  std::vector<EnbDlTestData> v2;  
  EnbDlTestData e2;
  UeDlTestData f2_1 (1, 100);
  e2.ues.push_back (f2_1);
  UeDlTestData f2_2 (2, 200);
  e2.ues.push_back (f2_2);
  v2.push_back (e2);
  AddTestCase (new EpcS1uDlTestCase ("1 eNB, 2UEs", v2), TestCase::QUICK);


  std::vector<EnbDlTestData> v3;  
  v3.push_back (e1);
  v3.push_back (e2);
  AddTestCase (new EpcS1uDlTestCase ("2 eNBs", v3), TestCase::QUICK);


  EnbDlTestData e3;
  UeDlTestData f3_1 (3, 50);
  e3.ues.push_back (f3_1);
  UeDlTestData f3_2 (5, 1472);
  e3.ues.push_back (f3_2);
  UeDlTestData f3_3 (1, 1);
  e3.ues.push_back (f3_2);
  std::vector<EnbDlTestData> v4;  
  v4.push_back (e3);
  v4.push_back (e1);
  v4.push_back (e2);
  AddTestCase (new EpcS1uDlTestCase ("3 eNBs", v4), TestCase::QUICK);

  std::vector<EnbDlTestData> v5;  
  EnbDlTestData e5;
  UeDlTestData f5 (10, 3000);
  e5.ues.push_back (f5);
  v5.push_back (e5);
  AddTestCase (new EpcS1uDlTestCase ("1 eNB, 10 pkts 3000 bytes each", v5), TestCase::QUICK);

  std::vector<EnbDlTestData> v6;  
  EnbDlTestData e6;
  UeDlTestData f6 (50, 3000);
  e6.ues.push_back (f6);
  v6.push_back (e6);
  AddTestCase (new EpcS1uDlTestCase ("1 eNB, 50 pkts 3000 bytes each", v6), TestCase::QUICK);
  
  std::vector<EnbDlTestData> v7;  
  EnbDlTestData e7;
  UeDlTestData f7 (10, 15000);
  e7.ues.push_back (f7);
  v7.push_back (e7);
  AddTestCase (new EpcS1uDlTestCase ("1 eNB, 10 pkts 15000 bytes each", v7), TestCase::QUICK);

  std::vector<EnbDlTestData> v8;  
  EnbDlTestData e8;
  UeDlTestData f8 (100, 15000);
  e8.ues.push_back (f8);
  v8.push_back (e8);
  AddTestCase (new EpcS1uDlTestCase ("1 eNB, 100 pkts 15000 bytes each", v8), TestCase::QUICK);
}

