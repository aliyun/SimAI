/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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


#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/mobility-module.h>
#include <ns3/lte-module.h>
#include <ns3/internet-module.h>
#include <ns3/applications-module.h>
#include <ns3/point-to-point-module.h>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LteX2HandoverTest");

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief HandoverEvent structure
 */
struct HandoverEvent
{
  Time startTime; ///< start time
  uint32_t ueDeviceIndex; ///< UE device index
  uint32_t sourceEnbDeviceIndex; ///< source ENB device index
  uint32_t targetEnbDeviceIndex; ///< target ENB device index
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test X2 Handover. In this test is used NoOpHandoverAlgorithm and
 * the request for handover is generated manually, and it is not based on measurements.
 */
class LteX2HandoverTestCase : public TestCase
{
public:
  /**
   *
   *
   * \param nUes number of UEs in the test
   * \param nDedicatedBearers number of bearers to be activated per UE
   * \param handoverEventList
   * \param handoverEventListName
   * \param schedulerType the scheduler type
   * \param admitHo
   * \param useIdealRrc true if the ideal RRC should be used
   */
  LteX2HandoverTestCase (uint32_t nUes, uint32_t nDedicatedBearers, std::list<HandoverEvent> handoverEventList, std::string handoverEventListName, std::string schedulerType, bool admitHo, bool useIdealRrc);

private:
  /**
   * Build name string
   * \param nUes number of UEs in the test
   * \param nDedicatedBearers number of bearers to be activated per UE
   * \param handoverEventListName
   * \param schedulerType the scheduler type
   * \param admitHo
   * \param useIdealRrc true if the ideal RRC should be used
   * \returns the name string
   */
  static std::string BuildNameString (uint32_t nUes, uint32_t nDedicatedBearers, std::string handoverEventListName, std::string schedulerType, bool admitHo, bool useIdealRrc);
  virtual void DoRun (void);
  /**
   * Check connected function
   * \param ueDevice the UE device
   * \param enbDevice the ENB device
   */
  void CheckConnected (Ptr<NetDevice> ueDevice, Ptr<NetDevice> enbDevice);

  /**
   * Teleport UE between both eNBs of the test
   * \param ueNode the UE node
   */
  void TeleportUeToMiddle (Ptr<Node> ueNode);

  /**
   * Teleport UE near the target eNB of the handover
   * \param ueNode the UE node
   * \param enbNode the target eNB node
   */
  void TeleportUeNearTargetEnb (Ptr<Node> ueNode, Ptr<Node> enbNode);

  uint32_t m_nUes; ///< number of UEs in the test
  uint32_t m_nDedicatedBearers; ///< number of UEs in the test
  std::list<HandoverEvent> m_handoverEventList; ///< handover event list
  std::string m_handoverEventListName; ///< handover event list name
  bool m_epc; ///< whether to use EPC
  std::string m_schedulerType; ///< scheduler type
  bool m_admitHo; ///< whether to admit the handover request
  bool     m_useIdealRrc; ///< whether to use the ideal RRC
  Ptr<LteHelper> m_lteHelper; ///< LTE helper
  Ptr<PointToPointEpcHelper> m_epcHelper; ///< EPC helper

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief BearerData structure
 */
  struct BearerData
  {
    uint32_t bid; ///< BID
    Ptr<PacketSink> dlSink; ///< DL sink
    Ptr<PacketSink> ulSink; ///< UL sink
    uint32_t dlOldTotalRx; ///< DL old total receive
    uint32_t ulOldTotalRx; ///< UL old total receive
  };

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief UeData structure
 */
  struct UeData
  {
    uint32_t id; ///< ID
    std::list<BearerData> bearerDataList; ///< bearer ID list
  };

/**
 * \brief Save stats after handover function
 * \param ueIndex the index of the UE
 */
  void SaveStatsAfterHandover (uint32_t ueIndex);
/**
 * \brief Check stats a while after handover function
 * \param ueIndex the index of the UE
 */
  void CheckStatsAWhileAfterHandover (uint32_t ueIndex);

  std::vector<UeData> m_ueDataVector; ///< UE data vector

  const Time m_maxHoDuration; ///< maximum HO duration
  const Time m_statsDuration; ///< stats duration
  const Time m_udpClientInterval; ///< UDP client interval
  const uint32_t m_udpClientPktSize; ///< UDP client packet size

};


std::string LteX2HandoverTestCase::BuildNameString (uint32_t nUes, uint32_t nDedicatedBearers, std::string handoverEventListName, std::string schedulerType, bool admitHo, bool useIdealRrc)
{
  std::ostringstream oss;
  oss << " nUes=" << nUes
      << " nDedicatedBearers=" << nDedicatedBearers
      << " " << schedulerType
      << " admitHo=" << admitHo
      << " hoList: " << handoverEventListName;
  if (useIdealRrc)
    {
      oss << ", ideal RRC";
    }
  else
    {
      oss << ", real RRC";
    }
  return oss.str ();
}

LteX2HandoverTestCase::LteX2HandoverTestCase (uint32_t nUes, uint32_t nDedicatedBearers, std::list<HandoverEvent> handoverEventList, std::string handoverEventListName, std::string schedulerType, bool admitHo, bool useIdealRrc)
  : TestCase (BuildNameString (nUes, nDedicatedBearers, handoverEventListName, schedulerType, admitHo, useIdealRrc)),
    m_nUes (nUes),
    m_nDedicatedBearers (nDedicatedBearers),
    m_handoverEventList (handoverEventList),
    m_handoverEventListName (handoverEventListName),
    m_epc (true),
    m_schedulerType (schedulerType),
    m_admitHo (admitHo),
    m_useIdealRrc (useIdealRrc),
    m_maxHoDuration (Seconds (0.1)),
    m_statsDuration (Seconds (0.1)),
    m_udpClientInterval (Seconds (0.01)),
    m_udpClientPktSize (100)

{}

void
LteX2HandoverTestCase::DoRun ()
{
  NS_LOG_FUNCTION (this << BuildNameString (m_nUes, m_nDedicatedBearers, m_handoverEventListName, m_schedulerType, m_admitHo, m_useIdealRrc));

  Config::Reset ();
  Config::SetDefault ("ns3::UdpClient::Interval",  TimeValue (m_udpClientInterval));
  Config::SetDefault ("ns3::UdpClient::MaxPackets", UintegerValue (1000000));
  Config::SetDefault ("ns3::UdpClient::PacketSize", UintegerValue (m_udpClientPktSize));

  //Disable Uplink Power Control
  Config::SetDefault ("ns3::LteUePhy::EnableUplinkPowerControl", BooleanValue (false));

  int64_t stream = 1;

  m_lteHelper = CreateObject<LteHelper> ();
  m_lteHelper->SetAttribute ("PathlossModel", StringValue ("ns3::FriisSpectrumPropagationLossModel"));
  m_lteHelper->SetSchedulerType (m_schedulerType);
  m_lteHelper->SetHandoverAlgorithmType ("ns3::NoOpHandoverAlgorithm"); // disable automatic handover
  m_lteHelper->SetAttribute ("UseIdealRrc", BooleanValue (m_useIdealRrc));


  NodeContainer enbNodes;
  enbNodes.Create (2);
  NodeContainer ueNodes;
  ueNodes.Create (m_nUes);

  if (m_epc)
    {
      m_epcHelper = CreateObject<PointToPointEpcHelper> ();
      m_lteHelper->SetEpcHelper (m_epcHelper);
    }

  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (-3000, 0, 0)); // enb0
  positionAlloc->Add (Vector ( 3000, 0, 0)); // enb1
  for (uint16_t i = 0; i < m_nUes; i++)
    {
      positionAlloc->Add (Vector (-3000, 100, 0));
    }
  MobilityHelper mobility;
  mobility.SetPositionAllocator (positionAlloc);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (enbNodes);
  mobility.Install (ueNodes);

  NetDeviceContainer enbDevices;
  enbDevices = m_lteHelper->InstallEnbDevice (enbNodes);
  stream += m_lteHelper->AssignStreams (enbDevices, stream);
  for (NetDeviceContainer::Iterator it = enbDevices.Begin ();
       it != enbDevices.End ();
       ++it)
    {
      Ptr<LteEnbRrc> enbRrc = (*it)->GetObject<LteEnbNetDevice> ()->GetRrc ();
      enbRrc->SetAttribute ("AdmitHandoverRequest", BooleanValue (m_admitHo));
    }

  NetDeviceContainer ueDevices;
  ueDevices = m_lteHelper->InstallUeDevice (ueNodes);
  stream += m_lteHelper->AssignStreams (ueDevices, stream);

  Ipv4Address remoteHostAddr;
  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  Ipv4InterfaceContainer ueIpIfaces;
  Ptr<Node> remoteHost;
  if (m_epc)
    {
      // Create a single RemoteHost
      NodeContainer remoteHostContainer;
      remoteHostContainer.Create (1);
      remoteHost = remoteHostContainer.Get (0);
      InternetStackHelper internet;
      internet.Install (remoteHostContainer);

      // Create the Internet
      PointToPointHelper p2ph;
      p2ph.SetDeviceAttribute ("DataRate", DataRateValue (DataRate ("100Gb/s")));
      p2ph.SetDeviceAttribute ("Mtu", UintegerValue (1500));
      p2ph.SetChannelAttribute ("Delay", TimeValue (Seconds (0.010)));
      Ptr<Node> pgw = m_epcHelper->GetPgwNode ();
      NetDeviceContainer internetDevices = p2ph.Install (pgw, remoteHost);
      Ipv4AddressHelper ipv4h;
      ipv4h.SetBase ("1.0.0.0", "255.0.0.0");
      Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign (internetDevices);
      // in this container, interface 0 is the pgw, 1 is the remoteHost
      remoteHostAddr = internetIpIfaces.GetAddress (1);

      Ipv4StaticRoutingHelper ipv4RoutingHelper;
      Ptr<Ipv4StaticRouting> remoteHostStaticRouting = ipv4RoutingHelper.GetStaticRouting (remoteHost->GetObject<Ipv4> ());
      remoteHostStaticRouting->AddNetworkRouteTo (Ipv4Address ("7.0.0.0"), Ipv4Mask ("255.0.0.0"), 1);

      // Install the IP stack on the UEs
      internet.Install (ueNodes);
      ueIpIfaces = m_epcHelper->AssignUeIpv4Address (NetDeviceContainer (ueDevices));
    }

  // attachment (needs to be done after IP stack configuration)
  // all UEs attached to eNB 0 at the beginning
  m_lteHelper->Attach (ueDevices, enbDevices.Get (0));

  if (m_epc)
    {
      // always true: bool epcDl = true;
      // always true: bool epcUl = true;
      // the rest of this block is copied from lena-dual-stripe


      // Install and start applications on UEs and remote host
      uint16_t dlPort = 10000;
      uint16_t ulPort = 20000;

      // randomize a bit start times to avoid simulation artifacts
      // (e.g., buffer overflows due to packet transmissions happening
      // exactly at the same time)
      Ptr<UniformRandomVariable> startTimeSeconds = CreateObject<UniformRandomVariable> ();
      startTimeSeconds->SetAttribute ("Min", DoubleValue (0));
      startTimeSeconds->SetAttribute ("Max", DoubleValue (0.010));
      startTimeSeconds->SetStream (stream++);

      for (uint32_t u = 0; u < ueNodes.GetN (); ++u)
        {
          Ptr<Node> ue = ueNodes.Get (u);
          // Set the default gateway for the UE
          Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting (ue->GetObject<Ipv4> ());
          ueStaticRouting->SetDefaultRoute (m_epcHelper->GetUeDefaultGatewayAddress (), 1);

          UeData ueData;

          for (uint32_t b = 0; b < m_nDedicatedBearers; ++b)
            {
              ++dlPort;
              ++ulPort;

              ApplicationContainer clientApps;
              ApplicationContainer serverApps;
              BearerData bearerData = BearerData ();

              // always true: if (epcDl)
              {
                UdpClientHelper dlClientHelper (ueIpIfaces.GetAddress (u), dlPort);
                clientApps.Add (dlClientHelper.Install (remoteHost));
                PacketSinkHelper dlPacketSinkHelper ("ns3::UdpSocketFactory",
                                                     InetSocketAddress (Ipv4Address::GetAny (), dlPort));
                ApplicationContainer sinkContainer = dlPacketSinkHelper.Install (ue);
                bearerData.dlSink = sinkContainer.Get (0)->GetObject<PacketSink> ();
                serverApps.Add (sinkContainer);

              }
              // always true: if (epcUl)
              {
                UdpClientHelper ulClientHelper (remoteHostAddr, ulPort);
                clientApps.Add (ulClientHelper.Install (ue));
                PacketSinkHelper ulPacketSinkHelper ("ns3::UdpSocketFactory",
                                                     InetSocketAddress (Ipv4Address::GetAny (), ulPort));
                ApplicationContainer sinkContainer = ulPacketSinkHelper.Install (remoteHost);
                bearerData.ulSink = sinkContainer.Get (0)->GetObject<PacketSink> ();
                serverApps.Add (sinkContainer);
              }

              Ptr<EpcTft> tft = Create<EpcTft> ();
              // always true: if (epcDl)
              {
                EpcTft::PacketFilter dlpf;
                dlpf.localPortStart = dlPort;
                dlpf.localPortEnd = dlPort;
                tft->Add (dlpf);
              }
              // always true: if (epcUl)
              {
                EpcTft::PacketFilter ulpf;
                ulpf.remotePortStart = ulPort;
                ulpf.remotePortEnd = ulPort;
                tft->Add (ulpf);
              }

              // always true: if (epcDl || epcUl)
              {
                EpsBearer bearer (EpsBearer::NGBR_VIDEO_TCP_DEFAULT);
                m_lteHelper->ActivateDedicatedEpsBearer (ueDevices.Get (u), bearer, tft);
              }
              Time startTime = Seconds (startTimeSeconds->GetValue ());
              serverApps.Start (startTime);
              clientApps.Start (startTime);

              ueData.bearerDataList.push_back (bearerData);

            } // end for b

          m_ueDataVector.push_back (ueData);
        }

    }
  else // (epc == false)
    {
      // for radio bearer activation purposes, consider together home UEs and macro UEs
      for (uint32_t u = 0; u < ueDevices.GetN (); ++u)
        {
          Ptr<NetDevice> ueDev = ueDevices.Get (u);
          for (uint32_t b = 0; b < m_nDedicatedBearers; ++b)
            {
              enum EpsBearer::Qci q = EpsBearer::NGBR_VIDEO_TCP_DEFAULT;
              EpsBearer bearer (q);
              m_lteHelper->ActivateDataRadioBearer (ueDev, bearer);
            }
        }
    }


  m_lteHelper->AddX2Interface (enbNodes);

  // check initial RRC connection
  const Time maxRrcConnectionEstablishmentDuration = Seconds (0.080);
  for (NetDeviceContainer::Iterator it = ueDevices.Begin (); it != ueDevices.End (); ++it)
    {
      Simulator::Schedule (maxRrcConnectionEstablishmentDuration,
                           &LteX2HandoverTestCase::CheckConnected,
                           this, *it, enbDevices.Get (0));
    }

  // schedule handover events and corresponding checks

  Time stopTime = Seconds (0);
  for (std::list<HandoverEvent>::iterator hoEventIt = m_handoverEventList.begin ();
       hoEventIt != m_handoverEventList.end ();
       ++hoEventIt)
    {
      // Teleport the UE between both eNBs just before the handover starts
      Simulator::Schedule (hoEventIt->startTime - MilliSeconds (10),
                           &LteX2HandoverTestCase::TeleportUeToMiddle,
                           this,
                           ueNodes.Get (hoEventIt->ueDeviceIndex));

      Simulator::Schedule (hoEventIt->startTime,
                           &LteX2HandoverTestCase::CheckConnected,
                           this,
                           ueDevices.Get (hoEventIt->ueDeviceIndex),
                           enbDevices.Get (hoEventIt->sourceEnbDeviceIndex));

      m_lteHelper->HandoverRequest (hoEventIt->startTime,
                                    ueDevices.Get (hoEventIt->ueDeviceIndex),
                                    enbDevices.Get (hoEventIt->sourceEnbDeviceIndex),
                                    enbDevices.Get (hoEventIt->targetEnbDeviceIndex));

      // Once the handover is finished, teleport the UE near the target eNB
      Simulator::Schedule (hoEventIt->startTime + MilliSeconds (40),
                           &LteX2HandoverTestCase::TeleportUeNearTargetEnb,
                           this,
                           ueNodes.Get (hoEventIt->ueDeviceIndex),
                           enbNodes.Get (m_admitHo ? hoEventIt->targetEnbDeviceIndex
                                         : hoEventIt->sourceEnbDeviceIndex));

      Time hoEndTime = hoEventIt->startTime + m_maxHoDuration;
      Simulator::Schedule (hoEndTime,
                           &LteX2HandoverTestCase::CheckConnected,
                           this,
                           ueDevices.Get (hoEventIt->ueDeviceIndex),
                           enbDevices.Get (m_admitHo ? hoEventIt->targetEnbDeviceIndex : hoEventIt->sourceEnbDeviceIndex));
      Simulator::Schedule (hoEndTime, &LteX2HandoverTestCase::SaveStatsAfterHandover,
                           this, hoEventIt->ueDeviceIndex);

      Time checkStatsAfterHoTime = hoEndTime + m_statsDuration;
      Simulator::Schedule (checkStatsAfterHoTime, &LteX2HandoverTestCase::CheckStatsAWhileAfterHandover,
                           this, hoEventIt->ueDeviceIndex);
      if (stopTime <= checkStatsAfterHoTime)
        {
          stopTime = checkStatsAfterHoTime + MilliSeconds (1);
        }
    }

  // m_lteHelper->EnableRlcTraces ();
  // m_lteHelper->EnablePdcpTraces();


  Simulator::Stop (stopTime);

  Simulator::Run ();

  Simulator::Destroy ();

}

void
LteX2HandoverTestCase::CheckConnected (Ptr<NetDevice> ueDevice, Ptr<NetDevice> enbDevice)
{
  Ptr<LteUeNetDevice> ueLteDevice = ueDevice->GetObject<LteUeNetDevice> ();
  Ptr<LteUeRrc> ueRrc = ueLteDevice->GetRrc ();
  NS_TEST_ASSERT_MSG_EQ (ueRrc->GetState (), LteUeRrc::CONNECTED_NORMALLY, "Wrong LteUeRrc state!");


  Ptr<LteEnbNetDevice> enbLteDevice = enbDevice->GetObject<LteEnbNetDevice> ();
  Ptr<LteEnbRrc> enbRrc = enbLteDevice->GetRrc ();
  uint16_t rnti = ueRrc->GetRnti ();
  Ptr<UeManager> ueManager = enbRrc->GetUeManager (rnti);
  NS_TEST_ASSERT_MSG_NE (ueManager, 0, "RNTI " << rnti << " not found in eNB");

  UeManager::State ueManagerState = ueManager->GetState ();
  NS_TEST_ASSERT_MSG_EQ (ueManagerState, UeManager::CONNECTED_NORMALLY, "Wrong UeManager state!");
  NS_ASSERT_MSG (ueManagerState == UeManager::CONNECTED_NORMALLY, "Wrong UeManager state!");

  uint16_t ueCellId = ueRrc->GetCellId ();
  uint16_t enbCellId = enbLteDevice->GetCellId ();
  uint8_t ueDlBandwidth = ueRrc->GetDlBandwidth ();
  uint8_t enbDlBandwidth = enbLteDevice->GetDlBandwidth ();
  uint8_t ueUlBandwidth = ueRrc->GetUlBandwidth ();
  uint8_t enbUlBandwidth = enbLteDevice->GetUlBandwidth ();
  uint8_t ueDlEarfcn = ueRrc->GetDlEarfcn ();
  uint8_t enbDlEarfcn = enbLteDevice->GetDlEarfcn ();
  uint8_t ueUlEarfcn = ueRrc->GetUlEarfcn ();
  uint8_t enbUlEarfcn = enbLteDevice->GetUlEarfcn ();
  uint64_t ueImsi = ueLteDevice->GetImsi ();
  uint64_t enbImsi = ueManager->GetImsi ();

  NS_TEST_ASSERT_MSG_EQ (ueImsi, enbImsi, "inconsistent IMSI");
  NS_TEST_ASSERT_MSG_EQ (ueCellId, enbCellId, "inconsistent CellId");
  NS_TEST_ASSERT_MSG_EQ (ueDlBandwidth, enbDlBandwidth, "inconsistent DlBandwidth");
  NS_TEST_ASSERT_MSG_EQ (ueUlBandwidth, enbUlBandwidth, "inconsistent UlBandwidth");
  NS_TEST_ASSERT_MSG_EQ (ueDlEarfcn, enbDlEarfcn, "inconsistent DlEarfcn");
  NS_TEST_ASSERT_MSG_EQ (ueUlEarfcn, enbUlEarfcn, "inconsistent UlEarfcn");

  ObjectMapValue enbDataRadioBearerMapValue;
  ueManager->GetAttribute ("DataRadioBearerMap", enbDataRadioBearerMapValue);
  NS_TEST_ASSERT_MSG_EQ (enbDataRadioBearerMapValue.GetN (), m_nDedicatedBearers + 1, "wrong num bearers at eNB");

  ObjectMapValue ueDataRadioBearerMapValue;
  ueRrc->GetAttribute ("DataRadioBearerMap", ueDataRadioBearerMapValue);
  NS_TEST_ASSERT_MSG_EQ (ueDataRadioBearerMapValue.GetN (), m_nDedicatedBearers + 1, "wrong num bearers at UE");

  ObjectMapValue::Iterator enbBearerIt = enbDataRadioBearerMapValue.Begin ();
  ObjectMapValue::Iterator ueBearerIt = ueDataRadioBearerMapValue.Begin ();
  while (enbBearerIt != enbDataRadioBearerMapValue.End ()
         && ueBearerIt != ueDataRadioBearerMapValue.End ())
    {
      Ptr<LteDataRadioBearerInfo> enbDrbInfo = enbBearerIt->second->GetObject<LteDataRadioBearerInfo> ();
      Ptr<LteDataRadioBearerInfo> ueDrbInfo = ueBearerIt->second->GetObject<LteDataRadioBearerInfo> ();
      //NS_TEST_ASSERT_MSG_EQ (enbDrbInfo->m_epsBearer, ueDrbInfo->m_epsBearer, "epsBearer differs");
      NS_TEST_ASSERT_MSG_EQ ((uint32_t) enbDrbInfo->m_epsBearerIdentity, (uint32_t) ueDrbInfo->m_epsBearerIdentity, "epsBearerIdentity differs");
      NS_TEST_ASSERT_MSG_EQ ((uint32_t) enbDrbInfo->m_drbIdentity, (uint32_t) ueDrbInfo->m_drbIdentity, "drbIdentity differs");
      //NS_TEST_ASSERT_MSG_EQ (enbDrbInfo->m_rlcConfig, ueDrbInfo->m_rlcConfig, "rlcConfig differs");
      NS_TEST_ASSERT_MSG_EQ ((uint32_t) enbDrbInfo->m_logicalChannelIdentity, (uint32_t) ueDrbInfo->m_logicalChannelIdentity, "logicalChannelIdentity differs");
      //NS_TEST_ASSERT_MSG_EQ (enbDrbInfo->m_logicalChannelConfig, ueDrbInfo->m_logicalChannelConfig, "logicalChannelConfig differs");

      ++enbBearerIt;
      ++ueBearerIt;
    }
  NS_ASSERT_MSG (enbBearerIt == enbDataRadioBearerMapValue.End (), "too many bearers at eNB");
  NS_ASSERT_MSG (ueBearerIt == ueDataRadioBearerMapValue.End (), "too many bearers at UE");
}

void
LteX2HandoverTestCase::TeleportUeToMiddle (Ptr<Node> ueNode)
{
  Ptr<MobilityModel> ueMobility = ueNode->GetObject<MobilityModel> ();
  ueMobility->SetPosition (Vector (0.0, 0.0, 0.0));
}

void
LteX2HandoverTestCase::TeleportUeNearTargetEnb (Ptr<Node> ueNode, Ptr<Node> enbNode)
{
  Ptr<MobilityModel> enbMobility = enbNode->GetObject<MobilityModel> ();
  Vector pos = enbMobility->GetPosition ();

  Ptr<MobilityModel> ueMobility = ueNode->GetObject<MobilityModel> ();
  ueMobility->SetPosition (pos + Vector (0.0, 100.0, 0.0));
}

void
LteX2HandoverTestCase::SaveStatsAfterHandover (uint32_t ueIndex)
{
  for (std::list<BearerData>::iterator it = m_ueDataVector.at (ueIndex).bearerDataList.begin ();
       it != m_ueDataVector.at (ueIndex).bearerDataList.end ();
       ++it)
    {
      it->dlOldTotalRx = it->dlSink->GetTotalRx ();
      it->ulOldTotalRx = it->ulSink->GetTotalRx ();
    }
}

void
LteX2HandoverTestCase::CheckStatsAWhileAfterHandover (uint32_t ueIndex)
{
  uint32_t b = 1;
  for (std::list<BearerData>::iterator it = m_ueDataVector.at (ueIndex).bearerDataList.begin ();
       it != m_ueDataVector.at (ueIndex).bearerDataList.end ();
       ++it)
    {
      uint32_t dlRx = it->dlSink->GetTotalRx () - it->dlOldTotalRx;
      uint32_t ulRx = it->ulSink->GetTotalRx () - it->ulOldTotalRx;
      uint32_t expectedBytes = m_udpClientPktSize * (m_statsDuration / m_udpClientInterval).GetDouble ();
      
      NS_TEST_ASSERT_MSG_EQ (dlRx, expectedBytes, "too few RX bytes in DL, ue=" << ueIndex << ", b=" << b);
      NS_TEST_ASSERT_MSG_EQ (ulRx, expectedBytes, "too few RX bytes in UL, ue=" << ueIndex << ", b=" << b);
      ++b;
    }
}


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief LTE X2 Handover Test Suite.
 *
 * In this test suite, we use NoOpHandoverAlgorithm, i.e. "handover algorithm which does nothing"
 * is used and handover is triggered manually. The automatic handover algorithms (A2A4, A3Rsrp)
 * are not tested.
 *
 * The tests are designed to check that eNB-buffered data received while a handover is in progress
 * is not lost but successfully forwarded. But the test suite doesn't test for possible loss of
 * RLC-buffered data because "lossless" handover is not implemented, and there are other application
 * send patterns (outside of the range tested here) that may incur losses.
 */
class LteX2HandoverTestSuite : public TestSuite
{
public:
  LteX2HandoverTestSuite ();
};


LteX2HandoverTestSuite::LteX2HandoverTestSuite ()
  : TestSuite ("lte-x2-handover", SYSTEM)
{
  // in the following:
  // fwd means handover from enb 0 to enb 1
  // bwd means handover from enb 1 to enb 0

  HandoverEvent ue1fwd;
  ue1fwd.startTime = MilliSeconds (100);
  ue1fwd.ueDeviceIndex = 0;
  ue1fwd.sourceEnbDeviceIndex = 0;
  ue1fwd.targetEnbDeviceIndex = 1;

  HandoverEvent ue1bwd;
  ue1bwd.startTime = MilliSeconds (400);
  ue1bwd.ueDeviceIndex = 0;
  ue1bwd.sourceEnbDeviceIndex = 1;
  ue1bwd.targetEnbDeviceIndex = 0;

  HandoverEvent ue1fwdagain;
  ue1fwdagain.startTime = MilliSeconds (700);
  ue1fwdagain.ueDeviceIndex = 0;
  ue1fwdagain.sourceEnbDeviceIndex = 0;
  ue1fwdagain.targetEnbDeviceIndex = 1;

  HandoverEvent ue2fwd;
  ue2fwd.startTime = MilliSeconds (110);
  ue2fwd.ueDeviceIndex = 1;
  ue2fwd.sourceEnbDeviceIndex = 0;
  ue2fwd.targetEnbDeviceIndex = 1;

  HandoverEvent ue2bwd;
  ue2bwd.startTime = MilliSeconds (350);
  ue2bwd.ueDeviceIndex = 1;
  ue2bwd.sourceEnbDeviceIndex = 1;
  ue2bwd.targetEnbDeviceIndex = 0;

  std::string hel0name ("none");
  std::list<HandoverEvent> hel0;

  std::string hel1name ("1 fwd");
  std::list<HandoverEvent> hel1;
  hel1.push_back (ue1fwd);

  std::string hel2name ("1 fwd & bwd");
  std::list<HandoverEvent> hel2;
  hel2.push_back (ue1fwd);
  hel2.push_back (ue1bwd);

  std::string hel3name ("1 fwd & bwd & fwd");
  std::list<HandoverEvent> hel3;
  hel3.push_back (ue1fwd);
  hel3.push_back (ue1bwd);
  hel3.push_back (ue1fwdagain);

  std::string hel4name ("1+2 fwd");
  std::list<HandoverEvent> hel4;
  hel4.push_back (ue1fwd);
  hel4.push_back (ue2fwd);

  std::string hel5name ("1+2 fwd & bwd");
  std::list<HandoverEvent> hel5;
  hel5.push_back (ue1fwd);
  hel5.push_back (ue1bwd);
  hel5.push_back (ue2fwd);
  hel5.push_back (ue2bwd);

  std::string hel6name ("2 fwd");
  std::list<HandoverEvent> hel6;
  hel6.push_back (ue2fwd);

  std::string hel7name ("2 fwd & bwd");
  std::list<HandoverEvent> hel7;
  hel7.push_back (ue2fwd);
  hel7.push_back (ue2bwd);

  std::vector<std::string> schedulers;
  schedulers.push_back ("ns3::RrFfMacScheduler");
  schedulers.push_back ("ns3::PfFfMacScheduler");
  for (std::vector<std::string>::iterator schedIt = schedulers.begin (); schedIt != schedulers.end (); ++schedIt)
    {
      for (int32_t useIdealRrc = 1; useIdealRrc >= 0; --useIdealRrc)
        {
          //                                     nUes, nDBearers, helist, name, sched, admitHo, idealRrc
          AddTestCase (new LteX2HandoverTestCase (  1,    0,    hel0, hel0name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  2,    0,    hel0, hel0name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  1,    5,    hel0, hel0name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  2,    5,    hel0, hel0name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  1,    0,    hel1, hel1name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  1,    1,    hel1, hel1name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  1,    2,    hel1, hel1name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  1,    0,    hel1, hel1name, *schedIt, false, useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  1,    1,    hel1, hel1name, *schedIt, false, useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  1,    2,    hel1, hel1name, *schedIt, false, useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  2,    0,    hel1, hel1name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  2,    1,    hel1, hel1name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  2,    2,    hel1, hel1name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  2,    0,    hel1, hel1name, *schedIt, false, useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  2,    1,    hel1, hel1name, *schedIt, false, useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  2,    2,    hel1, hel1name, *schedIt, false, useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  1,    0,    hel2, hel2name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  1,    1,    hel2, hel2name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  1,    2,    hel2, hel2name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  1,    0,    hel3, hel3name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  1,    1,    hel3, hel3name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  1,    2,    hel3, hel3name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  2,    0,    hel3, hel3name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  2,    1,    hel3, hel3name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  2,    2,    hel3, hel3name, *schedIt, true,  useIdealRrc), TestCase::QUICK);
          AddTestCase (new LteX2HandoverTestCase (  2,    0,    hel4, hel4name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  2,    1,    hel4, hel4name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  2,    2,    hel4, hel4name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  2,    0,    hel5, hel5name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  2,    1,    hel5, hel5name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  2,    2,    hel5, hel5name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  3,    0,    hel3, hel3name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  3,    1,    hel3, hel3name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  3,    2,    hel3, hel3name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  3,    0,    hel4, hel4name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  3,    1,    hel4, hel4name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  3,    2,    hel4, hel4name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  3,    0,    hel5, hel5name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  3,    1,    hel5, hel5name, *schedIt, true,  useIdealRrc), TestCase::EXTENSIVE);
          AddTestCase (new LteX2HandoverTestCase (  3,    2,    hel5, hel5name, *schedIt, true,  useIdealRrc), TestCase::QUICK);

        }
    }
}

static LteX2HandoverTestSuite g_lteX2HandoverTestSuiteInstance;
