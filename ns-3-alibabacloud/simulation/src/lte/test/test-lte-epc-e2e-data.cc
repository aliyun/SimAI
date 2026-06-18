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
#include "ns3/lte-helper.h"
#include "ns3/point-to-point-epc-helper.h"
#include "ns3/packet-sink-helper.h"
#include "ns3/udp-client-server-helper.h"
#include "ns3/udp-echo-helper.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/inet-socket-address.h"
#include "ns3/packet-sink.h"
#include <ns3/ipv4-static-routing-helper.h>
#include <ns3/ipv4-static-routing.h>
#include "ns3/boolean.h"
#include "ns3/uinteger.h"
#include "ns3/double.h"
#include "ns3/abort.h"
#include "ns3/mobility-helper.h"




using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LteEpcE2eData");


/**
 * \ingroup lte-test
 * \ingroup tests
 */

/// BearerTestData structure
struct BearerTestData
{
  /**
   * Constructor
   *
   * \param n the number of packets
   * \param s the packet size
   * \param i the inter packet interval in seconds
   */
  BearerTestData (uint32_t n, uint32_t s, double i);

  uint32_t numPkts; ///< the number of packets
  uint32_t pktSize; ///< the packet size
  Time interPacketInterval; ///< the inter packet interval time
 
  Ptr<PacketSink> dlServerApp; ///< the DL server app
  Ptr<Application> dlClientApp; ///< the DL client app

  Ptr<PacketSink> ulServerApp; ///< the UL server app
  Ptr<Application> ulClientApp; ///< the UL client app
};

  BearerTestData::BearerTestData (uint32_t n, uint32_t s, double i)
  : numPkts (n),
    pktSize (s),
    interPacketInterval (Seconds (i))
{
}

/// UeTestData structure
struct UeTestData
{
  std::vector<BearerTestData> bearers; ///< the bearer test data
};

/// EnbTestData structure
struct EnbTestData
{
  std::vector<UeTestData> ues; ///< the list of UEs
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test that e2e packet flow is correct. Compares the data send and the 
 * data received. Test uses mostly the PDCP stats to check the performance.
 */

class LteEpcE2eDataTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param name the reference name
   * \param v the ENB test data
   */
  LteEpcE2eDataTestCase (std::string name, std::vector<EnbTestData> v);
  virtual ~LteEpcE2eDataTestCase ();

private:
  virtual void DoRun (void);
  std::vector<EnbTestData> m_enbTestData; ///< the ENB test data
};


LteEpcE2eDataTestCase::LteEpcE2eDataTestCase (std::string name, std::vector<EnbTestData> v)
  : TestCase (name),
    m_enbTestData (v)
{
  NS_LOG_FUNCTION (this << name);
}

LteEpcE2eDataTestCase::~LteEpcE2eDataTestCase ()
{
}

void 
LteEpcE2eDataTestCase::DoRun ()
{
  NS_LOG_FUNCTION (this << GetName ());
  Config::Reset ();
  Config::SetDefault ("ns3::LteSpectrumPhy::CtrlErrorModelEnabled", BooleanValue (false));
  Config::SetDefault ("ns3::LteSpectrumPhy::DataErrorModelEnabled", BooleanValue (false));  
  Config::SetDefault ("ns3::LteHelper::UseIdealRrc", BooleanValue (true));

  Config::SetDefault ("ns3::RadioBearerStatsCalculator::DlPdcpOutputFilename", StringValue (CreateTempDirFilename ("DlPdcpStats.txt")));
  Config::SetDefault ("ns3::RadioBearerStatsCalculator::UlPdcpOutputFilename", StringValue (CreateTempDirFilename ("UlPdcpStats.txt")));

  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper> ();
  lteHelper->SetEpcHelper (epcHelper);

  lteHelper->SetAttribute("PathlossModel",
                          StringValue("ns3::FriisPropagationLossModel"));

  // allow jumbo frames on the S1-U link
  epcHelper->SetAttribute ("S1uLinkMtu", UintegerValue (30000));

  Ptr<Node> pgw = epcHelper->GetPgwNode ();
  
  // Create a single RemoteHost
  NodeContainer remoteHostContainer;
  remoteHostContainer.Create (1);
  Ptr<Node> remoteHost = remoteHostContainer.Get (0);
  InternetStackHelper internet;
  internet.Install (remoteHostContainer);

  // Create the internet
  PointToPointHelper p2ph;
  p2ph.SetDeviceAttribute ("DataRate", DataRateValue (DataRate ("100Gb/s")));
  p2ph.SetDeviceAttribute ("Mtu", UintegerValue (30000)); // jumbo frames here as well
  p2ph.SetChannelAttribute ("Delay", TimeValue (Seconds (0.010)));  
  NetDeviceContainer internetDevices = p2ph.Install (pgw, remoteHost);  
  Ipv4AddressHelper ipv4h;
  ipv4h.SetBase ("1.0.0.0", "255.0.0.0");
  Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign (internetDevices);
  Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress (1);
  
  // setup default gateway for the remote hosts
  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  Ptr<Ipv4StaticRouting> remoteHostStaticRouting = ipv4RoutingHelper.GetStaticRouting (remoteHost->GetObject<Ipv4> ());

  // hardcoded UE addresses for now
  remoteHostStaticRouting->AddNetworkRouteTo (Ipv4Address ("7.0.0.0"), Ipv4Mask ("255.255.255.0"), 1);
  

  NodeContainer enbs;
  enbs.Create (m_enbTestData.size ());
  MobilityHelper enbMobility;
  enbMobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  enbMobility.SetPositionAllocator ("ns3::GridPositionAllocator",
                                 "MinX", DoubleValue (0.0),
                                 "MinY", DoubleValue (0.0),
                                 "DeltaX", DoubleValue (10000.0),
                                 "DeltaY", DoubleValue (10000.0),
                                 "GridWidth", UintegerValue (3),
                                 "LayoutType", StringValue ("RowFirst"));
  enbMobility.Install (enbs);
  NetDeviceContainer enbLteDevs = lteHelper->InstallEnbDevice (enbs);
  NetDeviceContainer::Iterator enbLteDevIt = enbLteDevs.Begin ();
  
  uint16_t ulPort = 1000;

  for (std::vector<EnbTestData>::iterator enbit = m_enbTestData.begin ();
       enbit < m_enbTestData.end ();
       ++enbit, ++enbLteDevIt)
    {
      NS_ABORT_IF (enbLteDevIt ==  enbLteDevs.End ());
      


      NodeContainer ues;
      ues.Create (enbit->ues.size ());
      Vector enbPosition = (*enbLteDevIt)->GetNode ()->GetObject<MobilityModel> ()->GetPosition ();
      MobilityHelper ueMobility;
      ueMobility.SetPositionAllocator ("ns3::UniformDiscPositionAllocator",
                                        "X", DoubleValue (enbPosition.x),
                                        "Y", DoubleValue (enbPosition.y),
                                        "rho", DoubleValue (100.0));
      ueMobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
      ueMobility.Install (ues);
      NetDeviceContainer ueLteDevs = lteHelper->InstallUeDevice (ues);
       
      // we install the IP stack on the UEs 
      InternetStackHelper internet;
      internet.Install (ues);

      // assign IP address to UEs, and install applications
      for (uint32_t u = 0; u < ues.GetN (); ++u)
        {

          Ptr<Node> ue = ues.Get (u);          
          Ptr<NetDevice> ueLteDevice = ueLteDevs.Get (u);
          Ipv4InterfaceContainer ueIpIface = epcHelper->AssignUeIpv4Address (NetDeviceContainer (ueLteDevice));
          // set the default gateway for the UE
          Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting (ue->GetObject<Ipv4> ());          
          ueStaticRouting->SetDefaultRoute (epcHelper->GetUeDefaultGatewayAddress (), 1);


          // we can now attach the UE, which will also activate the default EPS bearer
          lteHelper->Attach (ueLteDevice, *enbLteDevIt);        
      
  
          uint16_t dlPort = 2000;          
          for (uint32_t b = 0; b < enbit->ues.at (u).bearers.size (); ++b)
            {              
              BearerTestData& bearerTestData = enbit->ues.at (u).bearers.at (b);
              
              { // Downlink
                ++dlPort;
                PacketSinkHelper packetSinkHelper ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), dlPort));
                ApplicationContainer apps = packetSinkHelper.Install (ue);
                apps.Start (Seconds (0.04));
                bearerTestData.dlServerApp = apps.Get (0)->GetObject<PacketSink> ();
          
                UdpEchoClientHelper client (ueIpIface.GetAddress (0), dlPort);
                client.SetAttribute ("MaxPackets", UintegerValue (bearerTestData.numPkts));
                client.SetAttribute ("Interval", TimeValue (bearerTestData.interPacketInterval));
                client.SetAttribute ("PacketSize", UintegerValue (bearerTestData.pktSize));
                apps = client.Install (remoteHost);
                apps.Start (Seconds (0.04));
                bearerTestData.dlClientApp = apps.Get (0);
              }

              { // Uplink
                ++ulPort;
                PacketSinkHelper packetSinkHelper ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), ulPort));
                ApplicationContainer apps = packetSinkHelper.Install (remoteHost);
                apps.Start (Seconds (0.8));
                bearerTestData.ulServerApp = apps.Get (0)->GetObject<PacketSink> ();
          
                UdpEchoClientHelper client (remoteHostAddr, ulPort);
                client.SetAttribute ("MaxPackets", UintegerValue (bearerTestData.numPkts));
                client.SetAttribute ("Interval", TimeValue (bearerTestData.interPacketInterval));
                client.SetAttribute ("PacketSize", UintegerValue (bearerTestData.pktSize));
                apps = client.Install (ue);
                apps.Start (Seconds (0.8));
                bearerTestData.ulClientApp = apps.Get (0);
              }

              EpsBearer epsBearer (EpsBearer::NGBR_VOICE_VIDEO_GAMING);

              Ptr<EpcTft> tft = Create<EpcTft> ();
              EpcTft::PacketFilter dlpf;
              dlpf.localPortStart = dlPort;
              dlpf.localPortEnd = dlPort;
              tft->Add (dlpf);
              EpcTft::PacketFilter ulpf;
              ulpf.remotePortStart = ulPort;
              ulpf.remotePortEnd = ulPort;
              tft->Add (ulpf);                            
 
              // all data will go over the dedicated bearer instead of the default EPS bearer
              lteHelper->ActivateDedicatedEpsBearer (ueLteDevice, epsBearer, tft);
            }
        } 
            
    } 

  Config::Set ("/NodeList/*/DeviceList/*/LteEnbRrc/UeMap/*/RadioBearerMap/*/LteRlc/MaxTxBufferSize",
               UintegerValue (2 * 1024 * 1024));
  Config::Set ("/NodeList/*/DeviceList/*/LteUeRrc/RadioBearerMap/*/LteRlc/MaxTxBufferSize",
               UintegerValue (2 * 1024 * 1024));


  double statsStartTime = 0.040; // need to allow for RRC connection establishment + SRS
  double statsDuration = 2.0;
    
  lteHelper->EnablePdcpTraces ();

  lteHelper->GetPdcpStats ()->SetAttribute ("StartTime", TimeValue (Seconds (statsStartTime)));
  lteHelper->GetPdcpStats ()->SetAttribute ("EpochDuration", TimeValue (Seconds (statsDuration)));
  
  
  Simulator::Stop (Seconds (statsStartTime + statsDuration - 0.0001));  
  Simulator::Run ();

  uint64_t imsiCounter = 0;

  for (std::vector<EnbTestData>::iterator enbit = m_enbTestData.begin ();
       enbit < m_enbTestData.end ();
       ++enbit)
    {
      for (std::vector<UeTestData>::iterator ueit = enbit->ues.begin ();
           ueit < enbit->ues.end ();
           ++ueit)
        {                    
          uint64_t imsi = ++imsiCounter;
          for (uint32_t b = 0; b < ueit->bearers.size (); ++b)
            {
              // LCID 0, 1, 2 are for SRBs
              // LCID 3 is (at the moment) the Default EPS bearer, and is unused in this test program
              uint8_t lcid = b+4;
              uint32_t expectedPkts = ueit->bearers.at (b).numPkts;
              uint32_t expectedBytes = (ueit->bearers.at (b).numPkts) * (ueit->bearers.at (b).pktSize);
              uint32_t txPktsPdcpDl = lteHelper->GetPdcpStats ()->GetDlTxPackets (imsi, lcid);
              uint32_t rxPktsPdcpDl = lteHelper->GetPdcpStats ()->GetDlRxPackets (imsi, lcid);
              uint32_t txPktsPdcpUl = lteHelper->GetPdcpStats ()->GetUlTxPackets (imsi, lcid);
              uint32_t rxPktsPdcpUl = lteHelper->GetPdcpStats ()->GetUlRxPackets (imsi, lcid);
              uint32_t rxBytesDl = ueit->bearers.at (b).dlServerApp->GetTotalRx ();
              uint32_t rxBytesUl = ueit->bearers.at (b).ulServerApp->GetTotalRx ();
              
              
              NS_TEST_ASSERT_MSG_EQ (txPktsPdcpDl, 
                                     expectedPkts, 
                                     "wrong TX PDCP packets in downlink for IMSI=" << imsi << " LCID=" << (uint16_t) lcid);
              
              NS_TEST_ASSERT_MSG_EQ (rxPktsPdcpDl, 
                                     expectedPkts, 
                                     "wrong RX PDCP packets in downlink for IMSI=" << imsi << " LCID=" << (uint16_t) lcid);
              NS_TEST_ASSERT_MSG_EQ (txPktsPdcpUl, 
                                     expectedPkts, 
                                     "wrong TX PDCP packets in uplink for IMSI=" << imsi << " LCID=" << (uint16_t) lcid);
              NS_TEST_ASSERT_MSG_EQ (rxPktsPdcpUl, 
                                     expectedPkts, 
                                     "wrong RX PDCP packets in uplink for IMSI=" << imsi << " LCID=" << (uint16_t) lcid);        

              NS_TEST_ASSERT_MSG_EQ (rxBytesDl, 
                                     expectedBytes, 
                                     "wrong total received bytes in downlink");
              NS_TEST_ASSERT_MSG_EQ (rxBytesUl, 
                                     expectedBytes, 
                                     "wrong total received bytes in uplink");
            }
        }      
    }
  
  Simulator::Destroy ();
}





/**
 * \ingroup tests lte-test
 *
 * \brief Test that the S1-U interface implementation works correctly
 */
class LteEpcE2eDataTestSuite : public TestSuite
{
public:
  LteEpcE2eDataTestSuite ();
  
} g_lteEpcE2eDataTestSuite; ///< the test suite

LteEpcE2eDataTestSuite::LteEpcE2eDataTestSuite ()
  : TestSuite ("lte-epc-e2e-data", SYSTEM)
{  
  std::vector<EnbTestData> v1;  
  EnbTestData e1;
  UeTestData u1;
  BearerTestData f1 (1, 100, 0.01);
  u1.bearers.push_back (f1);
  e1.ues.push_back (u1);
  v1.push_back (e1);
  AddTestCase (new LteEpcE2eDataTestCase ("1 eNB, 1UE", v1), TestCase::QUICK);

  std::vector<EnbTestData> v2;  
  EnbTestData e2;
  UeTestData u2_1;
  BearerTestData f2_1 (1, 100, 0.01);
  u2_1.bearers.push_back (f2_1);
  e2.ues.push_back (u2_1);
  UeTestData u2_2;
  BearerTestData f2_2 (2, 200, 0.01);
  u2_2.bearers.push_back (f2_2);
  e2.ues.push_back (u2_2);
  v2.push_back (e2);
  AddTestCase (new LteEpcE2eDataTestCase ("1 eNB, 2UEs", v2), TestCase::EXTENSIVE);

  std::vector<EnbTestData> v3;  
  v3.push_back (e1);
  v3.push_back (e2);
  AddTestCase (new LteEpcE2eDataTestCase ("2 eNBs", v3), TestCase::EXTENSIVE);

  EnbTestData e4;
  UeTestData u4_1;
  BearerTestData f4_1 (3, 50, 0.01);
  u4_1.bearers.push_back (f4_1);
  e4.ues.push_back (u4_1);
  UeTestData u4_2;
  BearerTestData f4_2 (5, 1400, 0.01);
  u4_2.bearers.push_back (f4_2);
  e4.ues.push_back (u4_2);
  UeTestData u4_3;
  BearerTestData f4_3 (1, 12, 0.01);
  u4_3.bearers.push_back (f4_3);
  e4.ues.push_back (u4_3);
  std::vector<EnbTestData> v4;  
  v4.push_back (e4);
  v4.push_back (e1);
  v4.push_back (e2);
  AddTestCase (new LteEpcE2eDataTestCase ("3 eNBs", v4), TestCase::EXTENSIVE);

  EnbTestData e5;
  UeTestData u5;
  BearerTestData f5 (5, 1000, 0.01);
  u5.bearers.push_back (f5);
  e5.ues.push_back (u5);
  std::vector<EnbTestData> v5;
  v5.push_back (e5);
  AddTestCase (new LteEpcE2eDataTestCase ("1 eNB, 1UE with 1000 byte packets", v5), TestCase::EXTENSIVE);


  EnbTestData e6;
  UeTestData u6;
  BearerTestData f6 (5, 1400, 0.01);
  u6.bearers.push_back (f6);
  e6.ues.push_back (u6);
  std::vector<EnbTestData> v6;
  v6.push_back (e6);
  AddTestCase (new LteEpcE2eDataTestCase ("1 eNB, 1UE with 1400 byte packets", v6), TestCase::EXTENSIVE);

  EnbTestData e7;
  UeTestData u7;
  BearerTestData f7_1 (1, 1400, 0.01);
  u7.bearers.push_back (f7_1);
  BearerTestData f7_2 (1, 100, 0.01);
  u7.bearers.push_back (f7_2);
  e7.ues.push_back (u7);
  std::vector<EnbTestData> v7;
  v7.push_back (e7);
  AddTestCase (new LteEpcE2eDataTestCase ("1 eNB, 1UE with 2 bearers", v7), TestCase::EXTENSIVE);

  EnbTestData e8;
  UeTestData u8;
  BearerTestData f8 (50, 8000, 0.02); // watch out for ns3::LteRlcUm::MaxTxBufferSize
  u8.bearers.push_back (f8);
  e8.ues.push_back (u8);
  std::vector<EnbTestData> v8;
  v8.push_back (e8);
  AddTestCase (new LteEpcE2eDataTestCase ("1 eNB, 1UE with fragmentation", v8), TestCase::EXTENSIVE);


  EnbTestData e9;
  UeTestData u9;
  BearerTestData f9 (1000, 20, 0.0001);
  u9.bearers.push_back (f9);
  e9.ues.push_back (u9);
  std::vector<EnbTestData> v9;
  v9.push_back (e9);
  AddTestCase (new LteEpcE2eDataTestCase ("1 eNB, 1UE with aggregation", v9), TestCase::EXTENSIVE);


}
