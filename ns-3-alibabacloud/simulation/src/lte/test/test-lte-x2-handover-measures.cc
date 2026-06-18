/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Authors: Nicola Baldo <nbaldo@cttc.es>
 *          Manuel Requena <manuel.requena@cttc.es>
 */

#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/mobility-module.h>
#include <ns3/lte-module.h>
#include <ns3/internet-module.h>
#include <ns3/applications-module.h>
#include <ns3/point-to-point-module.h>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LteX2HandoverMeasuresTest");

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief CheckPointEvent structure
 */
struct CheckPointEvent
{
  Time checkStartTime; ///< check start time
  Time checkStopTime; ///< check stop time
  Time checkInterval; ///< check interval
  uint32_t ueDeviceIndex; ///< UE device index
  uint32_t enbDeviceIndex; ///< ENB device index

  /**
   *  Constructor
   *
   * \param start the start time
   * \param stop the stop time
   * \param interval the interval time
   * \param ueIndex the UE index
   * \param enbIndex the ENB index
   */
  CheckPointEvent (Time start, Time stop, Time interval, uint32_t ueIndex, uint32_t enbIndex)
    : checkStartTime (start),
      checkStopTime (stop),
      checkInterval (interval),
      ueDeviceIndex (ueIndex),
      enbDeviceIndex (enbIndex)
  {}
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test different X2 handover measures and algorithms, e.g. A2A4RsrqHandoverAlgorithm and
 * A3RsrpHandoverAlgorithm. Test defines different handover parameters and scenario configurations.
 */
class LteX2HandoverMeasuresTestCase : public TestCase
{
public:
  /**
   *
   *
   * \param nEnbs number of eNBs in the test
   * \param nUes number of UEs in the test
   * \param nDedicatedBearers number of bearers to be activated per UE
   * \param checkPointEventList
   * \param checkPointEventListName
   * \param useUdp true if UDP is to be used, false if TCP is to be used
   * \param schedulerType type of scheduler to be used (e.g. "ns3::PfFfMacScheduler")
   * \param handoverAlgorithmType type of handover algorithm to be used (e.g. "ns3::A3RsrpHandoverAlgorithm")
   * \param admitHo
   * \param useIdealRrc true if ideal RRC is to be used, false if real RRC is to be used
   *
   * \return
   */
  LteX2HandoverMeasuresTestCase (uint32_t nEnbs, uint32_t nUes, uint32_t nDedicatedBearers,
                                 std::list<CheckPointEvent> checkPointEventList,
                                 std::string checkPointEventListName,
                                 bool useUdp, std::string schedulerType,
                                 std::string handoverAlgorithmType, bool admitHo,
                                 bool useIdealRrc);

private:
  /**
   * Build name string
   * \param nEnbs number of eNBs in the test
   * \param nUes number of UEs in the test
   * \param nDedicatedBearers number of bearers to be activated per UE
   * \param checkPointEventListName
   * \param useUdp true if UDP is to be used, false if TCP is to be used
   * \param schedulerType the scheduler type
   * \param handoverAlgorithmType type of handover algorithm to be used (e.g. "ns3::A3RsrpHandoverAlgorithm")
   * \param admitHo
   * \param useIdealRrc true if the ideal RRC should be used
   * \returns the name string
   */
  static std::string BuildNameString (uint32_t nEnbs, uint32_t nUes, uint32_t nDedicatedBearers,
                                      std::string checkPointEventListName,
                                      bool useUdp, std::string schedulerType,
                                      std::string handoverAlgorithmType, bool admitHo,
                                      bool useIdealRrc);
  virtual void DoRun (void);
  /**
   * Check connected function
   * \param ueDevice the UE device
   * \param enbDevice the ENB device
   */
  void CheckConnected (Ptr<NetDevice> ueDevice, Ptr<NetDevice> enbDevice);

  uint32_t m_nEnbs; ///< number of eNBs in the test
  uint32_t m_nUes; ///< number of UEs in the test
  uint32_t m_nDedicatedBearers; ///< number of UEs in the test
  std::list<CheckPointEvent> m_checkPointEventList; ///< check point event list
  std::string m_checkPointEventListName; ///< check point event list name
  bool m_epc; ///< whether to use EPC
  bool m_useUdp; ///<  whether to use UDP traffic
  std::string m_schedulerType; ///< scheduler type
  std::string m_handoverAlgorithmType; ///< handover algorithm type
  bool m_admitHo; ///< whether to configure to admit handover
  bool m_useIdealRrc; ///< whether to use ideal RRC
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
   * \brief Save stats  function
   * \param ueIndex the index of the UE
   */
  void SaveStats (uint32_t ueIndex);
  /**
   * \brief Check stats  function
   * \param ueIndex the index of the UE
   */
  void CheckStats (uint32_t ueIndex);

  std::vector<UeData> m_ueDataVector; ///< UE data vector

  const Time m_maxHoDuration; ///< maximum HO duration
  const Time m_statsDuration; ///< stats duration
  const Time m_udpClientInterval; ///< UDP client interval
  const uint32_t m_udpClientPktSize; ///< UDP client packet size
};


std::string
LteX2HandoverMeasuresTestCase::BuildNameString (uint32_t nEnbs, uint32_t nUes, uint32_t nDedicatedBearers,
                                                std::string checkPointEventListName,
                                                bool useUdp, std::string schedulerType,
                                                std::string handoverAlgorithmType, bool admitHo,
                                                bool useIdealRrc)
{
  std::ostringstream oss;
  oss << "nEnbs=" << nEnbs
      << " nUes=" << nUes
      << " nDedicatedBearers=" << nDedicatedBearers
      << " udp=" << useUdp
      << " " << schedulerType
      << " " << handoverAlgorithmType
      << " admitHo=" << admitHo
      << " hoList: " << checkPointEventListName;
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

LteX2HandoverMeasuresTestCase::LteX2HandoverMeasuresTestCase (uint32_t nEnbs, uint32_t nUes, uint32_t nDedicatedBearers,
                                                              std::list<CheckPointEvent> checkPointEventList,
                                                              std::string checkPointEventListName,
                                                              bool useUdp, std::string schedulerType,
                                                              std::string handoverAlgorithmType, bool admitHo,
                                                              bool useIdealRrc)
  : TestCase (BuildNameString (nEnbs, nUes, nDedicatedBearers,
                               checkPointEventListName, useUdp, schedulerType,
                               handoverAlgorithmType, admitHo, useIdealRrc)),
    m_nEnbs (nEnbs),
    m_nUes (nUes),
    m_nDedicatedBearers (nDedicatedBearers),
    m_checkPointEventList (checkPointEventList),
    m_checkPointEventListName (checkPointEventListName),
    m_epc (true),
    m_useUdp (useUdp),
    m_schedulerType (schedulerType),
    m_handoverAlgorithmType (handoverAlgorithmType),
    m_admitHo (admitHo),
    m_useIdealRrc (useIdealRrc),
    m_maxHoDuration (Seconds (0.1)),
    m_statsDuration (Seconds (0.5)),
    m_udpClientInterval (Seconds (0.01)),
    m_udpClientPktSize (100)
{}

void
LteX2HandoverMeasuresTestCase::DoRun ()
{
  NS_LOG_FUNCTION (this << BuildNameString (m_nEnbs, m_nUes, m_nDedicatedBearers,
                                            m_checkPointEventListName,
                                            m_useUdp, m_schedulerType,
                                            m_handoverAlgorithmType, m_admitHo,
                                            m_useIdealRrc));

  Config::Reset ();
  Config::SetDefault ("ns3::UdpClient::Interval", TimeValue (m_udpClientInterval));
  Config::SetDefault ("ns3::UdpClient::MaxPackets", UintegerValue (1000000));
  Config::SetDefault ("ns3::UdpClient::PacketSize", UintegerValue (m_udpClientPktSize));
  Config::SetDefault ("ns3::LteEnbRrc::HandoverJoiningTimeoutDuration", TimeValue (MilliSeconds (200)));
  Config::SetDefault ("ns3::LteEnbPhy::TxPower", DoubleValue (20));

  //Disable Uplink Power Control
  Config::SetDefault ("ns3::LteUePhy::EnableUplinkPowerControl", BooleanValue (false));

  int64_t stream = 1;

  m_lteHelper = CreateObject<LteHelper> ();
  m_lteHelper->SetAttribute ("PathlossModel", StringValue ("ns3::FriisSpectrumPropagationLossModel"));
  m_lteHelper->SetAttribute ("UseIdealRrc", BooleanValue (m_useIdealRrc));
  m_lteHelper->SetSchedulerType (m_schedulerType);

  if (m_handoverAlgorithmType == "ns3::A2A4RsrqHandoverAlgorithm")
    {
      m_lteHelper->SetHandoverAlgorithmType ("ns3::A2A4RsrqHandoverAlgorithm");
      m_lteHelper->SetHandoverAlgorithmAttribute ("ServingCellThreshold",
                                                  UintegerValue (30));
      m_lteHelper->SetHandoverAlgorithmAttribute ("NeighbourCellOffset",
                                                  UintegerValue (1));
    }
  else if (m_handoverAlgorithmType == "ns3::A3RsrpHandoverAlgorithm")
    {
      m_lteHelper->SetHandoverAlgorithmType ("ns3::A3RsrpHandoverAlgorithm");
      m_lteHelper->SetHandoverAlgorithmAttribute ("Hysteresis",
                                                  DoubleValue (1.5));
      m_lteHelper->SetHandoverAlgorithmAttribute ("TimeToTrigger",
                                                  TimeValue (MilliSeconds (128)));
    }
  else
    {
      NS_FATAL_ERROR ("Unknown handover algorithm " << m_handoverAlgorithmType);
    }

  double distance = 1000.0; // m
  double speed = 150;       // m/s

  NodeContainer enbNodes;
  enbNodes.Create (m_nEnbs);
  NodeContainer ueNodes;
  ueNodes.Create (m_nUes);

  if (m_epc)
    {
      m_epcHelper = CreateObject<PointToPointEpcHelper> ();
      m_lteHelper->SetEpcHelper (m_epcHelper);
    }

  // Install Mobility Model in eNBs
  // eNBs are located along a line in the X axis
  Ptr<ListPositionAllocator> enbPositionAlloc = CreateObject<ListPositionAllocator> ();
  for (uint16_t i = 0; i < m_nEnbs; i++)
    {
      Vector enbPosition (distance * (i + 1), 0, 0);
      enbPositionAlloc->Add (enbPosition);
    }
  MobilityHelper enbMobility;
  enbMobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  enbMobility.SetPositionAllocator (enbPositionAlloc);
  enbMobility.Install (enbNodes);

  // Install Mobility Model in UE
  // UE moves with a constant speed along the X axis
  MobilityHelper ueMobility;
  ueMobility.SetMobilityModel ("ns3::ConstantVelocityMobilityModel");
  ueMobility.Install (ueNodes);
  for (uint16_t i = 0; i < m_nUes; i++)
    {
      ueNodes.Get (i)->GetObject<MobilityModel> ()->SetPosition (Vector (0, 0, 0));
      ueNodes.Get (i)->GetObject<ConstantVelocityMobilityModel> ()->SetVelocity (Vector (speed, 0, 0));
    }

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
      bool epcDl = true;
      bool epcUl = false;
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

              if (m_useUdp)
                {
                  if (epcDl)
                    {
                      UdpClientHelper dlClientHelper (ueIpIfaces.GetAddress (u), dlPort);
                      clientApps.Add (dlClientHelper.Install (remoteHost));
                      PacketSinkHelper dlPacketSinkHelper ("ns3::UdpSocketFactory",
                                                           InetSocketAddress (Ipv4Address::GetAny (), dlPort));
                      ApplicationContainer sinkContainer = dlPacketSinkHelper.Install (ue);
                      bearerData.dlSink = sinkContainer.Get (0)->GetObject<PacketSink> ();
                      serverApps.Add (sinkContainer);

                    }
                  if (epcUl)
                    {
                      UdpClientHelper ulClientHelper (remoteHostAddr, ulPort);
                      clientApps.Add (ulClientHelper.Install (ue));
                      PacketSinkHelper ulPacketSinkHelper ("ns3::UdpSocketFactory",
                                                           InetSocketAddress (Ipv4Address::GetAny (), ulPort));
                      ApplicationContainer sinkContainer = ulPacketSinkHelper.Install (remoteHost);
                      bearerData.ulSink = sinkContainer.Get (0)->GetObject<PacketSink> ();
                      serverApps.Add (sinkContainer);
                    }
                }
              else // use TCP
                {
                  if (epcDl)
                    {
                      BulkSendHelper dlClientHelper ("ns3::TcpSocketFactory",
                                                     InetSocketAddress (ueIpIfaces.GetAddress (u), dlPort));
                      dlClientHelper.SetAttribute ("MaxBytes", UintegerValue (0));
                      clientApps.Add (dlClientHelper.Install (remoteHost));
                      PacketSinkHelper dlPacketSinkHelper ("ns3::TcpSocketFactory",
                                                           InetSocketAddress (Ipv4Address::GetAny (), dlPort));
                      ApplicationContainer sinkContainer = dlPacketSinkHelper.Install (ue);
                      bearerData.dlSink = sinkContainer.Get (0)->GetObject<PacketSink> ();
                      serverApps.Add (sinkContainer);
                    }
                  if (epcUl)
                    {
                      BulkSendHelper ulClientHelper ("ns3::TcpSocketFactory",
                                                     InetSocketAddress (remoteHostAddr, ulPort));
                      ulClientHelper.SetAttribute ("MaxBytes", UintegerValue (0));
                      clientApps.Add (ulClientHelper.Install (ue));
                      PacketSinkHelper ulPacketSinkHelper ("ns3::TcpSocketFactory",
                                                           InetSocketAddress (Ipv4Address::GetAny (), ulPort));
                      ApplicationContainer sinkContainer = ulPacketSinkHelper.Install (remoteHost);
                      bearerData.ulSink = sinkContainer.Get (0)->GetObject<PacketSink> ();
                      serverApps.Add (sinkContainer);
                    }
                } // end if (useUdp)

              Ptr<EpcTft> tft = Create<EpcTft> ();
              if (epcDl)
                {
                  EpcTft::PacketFilter dlpf;
                  dlpf.localPortStart = dlPort;
                  dlpf.localPortEnd = dlPort;
                  tft->Add (dlpf);
                }
              if (epcUl)
                {
                  EpcTft::PacketFilter ulpf;
                  ulpf.remotePortStart = ulPort;
                  ulpf.remotePortEnd = ulPort;
                  tft->Add (ulpf);
                }

              if (epcDl || epcUl)
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
      NS_LOG_FUNCTION (maxRrcConnectionEstablishmentDuration);
      Simulator::Schedule (maxRrcConnectionEstablishmentDuration,
                           &LteX2HandoverMeasuresTestCase::CheckConnected,
                           this, *it, enbDevices.Get (0));
    }

  // schedule the checkpoint events

  Time stopTime = Seconds (0);
  for (std::list<CheckPointEvent>::iterator checkPointEventIt = m_checkPointEventList.begin ();
       checkPointEventIt != m_checkPointEventList.end ();
       ++checkPointEventIt)
    {
      for (Time checkPointTime = checkPointEventIt->checkStartTime;
           checkPointTime < checkPointEventIt->checkStopTime;
           checkPointTime += checkPointEventIt->checkInterval)
        {
          Simulator::Schedule (checkPointTime, &LteX2HandoverMeasuresTestCase::CheckConnected,
                               this, ueDevices.Get (checkPointEventIt->ueDeviceIndex),
                               enbDevices.Get (checkPointEventIt->enbDeviceIndex));

          Time saveStatsTime = checkPointTime;
          Simulator::Schedule (saveStatsTime, &LteX2HandoverMeasuresTestCase::SaveStats,
                               this, checkPointEventIt->ueDeviceIndex);

          Time checkStats = saveStatsTime + m_statsDuration;
          Simulator::Schedule (checkStats, &LteX2HandoverMeasuresTestCase::CheckStats,
                               this, checkPointEventIt->ueDeviceIndex);

          if (stopTime <= checkStats)
            {
              stopTime = checkStats + Seconds (1);
            }
        }
    }

  Simulator::Stop (stopTime);
  Simulator::Run ();
  Simulator::Destroy ();
}

void
LteX2HandoverMeasuresTestCase::CheckConnected (Ptr<NetDevice> ueDevice, Ptr<NetDevice> enbDevice)
{
  NS_LOG_FUNCTION (ueDevice << enbDevice);

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
LteX2HandoverMeasuresTestCase::SaveStats (uint32_t ueIndex)
{
  NS_LOG_FUNCTION (ueIndex);
  for (std::list<BearerData>::iterator it = m_ueDataVector.at (ueIndex).bearerDataList.begin ();
       it != m_ueDataVector.at (ueIndex).bearerDataList.end ();
       ++it)
    {
      if (it->dlSink)
        {
          it->dlOldTotalRx = it->dlSink->GetTotalRx ();
        }
      if (it->ulSink)
        {
          it->ulOldTotalRx = it->ulSink->GetTotalRx ();
        }
    }
}

void
LteX2HandoverMeasuresTestCase::CheckStats (uint32_t ueIndex)
{
  NS_LOG_FUNCTION (ueIndex);
  uint32_t b = 1;
  for (std::list<BearerData>::iterator it = m_ueDataVector.at (ueIndex).bearerDataList.begin ();
       it != m_ueDataVector.at (ueIndex).bearerDataList.end ();
       ++it)
    {
      uint32_t dlRx = 0;
      uint32_t ulRx = 0;

      if (it->dlSink)
        {
          dlRx = it->dlSink->GetTotalRx () - it->dlOldTotalRx;
        }

      if (it->ulSink)
        {
          ulRx = it->ulSink->GetTotalRx () - it->ulOldTotalRx;
        }
      double expectedBytes = m_udpClientPktSize * (m_statsDuration / m_udpClientInterval).GetDouble ();

      NS_LOG_LOGIC ("expBytes " << expectedBytes << " dlRx " << dlRx << " ulRx " << ulRx);

      //                                tolerance
      if (it->dlSink)
        {
          NS_TEST_ASSERT_MSG_GT (dlRx,   0.500 * expectedBytes, "too few RX bytes in DL, ue=" << ueIndex << ", b=" << b);
        }
      if (it->ulSink)
        {
          NS_TEST_ASSERT_MSG_GT (ulRx,   0.500 * expectedBytes, "too few RX bytes in UL, ue=" << ueIndex << ", b=" << b);
        }
      ++b;
    }
}


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Lte X2 Handover Measures Test Suite
 */
class LteX2HandoverMeasuresTestSuite : public TestSuite
{
public:
  LteX2HandoverMeasuresTestSuite ();
};


LteX2HandoverMeasuresTestSuite::LteX2HandoverMeasuresTestSuite ()
  : TestSuite ("lte-x2-handover-measures", SYSTEM)
{
  Time checkInterval = Seconds (1);

  std::string cel1name ("ho: 0 -> 1");
  std::list<CheckPointEvent> cel1;
  cel1.push_back (CheckPointEvent (Seconds (1), Seconds (10.1), checkInterval, 0, 0));
  cel1.push_back (CheckPointEvent (Seconds (11), Seconds (17), checkInterval, 0, 1));

  std::string cel2name ("ho: 0 -> 1 -> 2");
  std::list<CheckPointEvent> cel2;
  cel2.push_back (CheckPointEvent (Seconds (1), Seconds (10.1), checkInterval, 0, 0));
  cel2.push_back (CheckPointEvent (Seconds (11), Seconds (17.1), checkInterval, 0, 1));
  cel2.push_back (CheckPointEvent (Seconds (18), Seconds (24), checkInterval, 0, 2));

  std::string cel3name ("ho: 0 -> 1 -> 2 -> 3");
  std::list<CheckPointEvent> cel3;
  cel3.push_back (CheckPointEvent (Seconds (1), Seconds (10.1), checkInterval,  0, 0));
  cel3.push_back (CheckPointEvent (Seconds (11), Seconds (17.1), checkInterval, 0, 1));
  cel3.push_back (CheckPointEvent (Seconds (18), Seconds (24.1), checkInterval, 0, 2));
  cel3.push_back (CheckPointEvent (Seconds (25), Seconds (37), checkInterval, 0, 3));


  int32_t useIdealRrc;
  std::string sched = "ns3::PfFfMacScheduler";
  std::string ho = "ns3::A2A4RsrqHandoverAlgorithm";
  for (useIdealRrc = 1; useIdealRrc >= 0; --useIdealRrc)
    {
      //                                          nEnbs, nUes, nDBearers, celist, name, useUdp, sched, ho, admitHo, idealRrc
      AddTestCase (new LteX2HandoverMeasuresTestCase (2,   1,    0,    cel1, cel1name, true, sched, ho, true, useIdealRrc), TestCase::TAKES_FOREVER);
      AddTestCase (new LteX2HandoverMeasuresTestCase (2,   1,    1,    cel1, cel1name, true, sched, ho, true, useIdealRrc), TestCase::QUICK);
      AddTestCase (new LteX2HandoverMeasuresTestCase (2,   1,    2,    cel1, cel1name, true, sched, ho, true, useIdealRrc), TestCase::TAKES_FOREVER);
      AddTestCase (new LteX2HandoverMeasuresTestCase (3,   1,    0,    cel2, cel2name, true, sched, ho, true, useIdealRrc), TestCase::TAKES_FOREVER);
      AddTestCase (new LteX2HandoverMeasuresTestCase (3,   1,    1,    cel2, cel2name, true, sched, ho, true, useIdealRrc), TestCase::TAKES_FOREVER);
      AddTestCase (new LteX2HandoverMeasuresTestCase (3,   1,    2,    cel2, cel2name, true, sched, ho, true, useIdealRrc), TestCase::EXTENSIVE);
      AddTestCase (new LteX2HandoverMeasuresTestCase (4,   1,    0,    cel3, cel3name, true, sched, ho, true, useIdealRrc), TestCase::EXTENSIVE);
      AddTestCase (new LteX2HandoverMeasuresTestCase (4,   1,    1,    cel3, cel3name, true, sched, ho, true, useIdealRrc), TestCase::TAKES_FOREVER);
      AddTestCase (new LteX2HandoverMeasuresTestCase (4,   1,    2,    cel3, cel3name, true, sched, ho, true, useIdealRrc), TestCase::TAKES_FOREVER);
    }

  sched = "ns3::RrFfMacScheduler";
  for (useIdealRrc = 1; useIdealRrc >= 0; --useIdealRrc)
    {
      //                                          nEnbs, nUes, nDBearers, celist, name, useUdp, sched, admitHo, idealRrc
      AddTestCase (new LteX2HandoverMeasuresTestCase (2,   1,    0,    cel1, cel1name, true, sched, ho, true, useIdealRrc), TestCase::EXTENSIVE);
      AddTestCase (new LteX2HandoverMeasuresTestCase (3,   1,    0,    cel2, cel2name, true, sched, ho, true, useIdealRrc), TestCase::TAKES_FOREVER);
      AddTestCase (new LteX2HandoverMeasuresTestCase (4,   1,    0,    cel3, cel3name, true, sched, ho, true, useIdealRrc), TestCase::TAKES_FOREVER);
    }

  ho = "ns3::A3RsrpHandoverAlgorithm";
  sched = "ns3::PfFfMacScheduler";
  for (useIdealRrc = 1; useIdealRrc >= 0; --useIdealRrc)
    {
      //                                          nEnbs, nUes, nDBearers, celist, name, useUdp, sched, admitHo, idealRrc
      AddTestCase (new LteX2HandoverMeasuresTestCase (2,   1,    0,    cel1, cel1name, true, sched, ho, true, useIdealRrc), TestCase::EXTENSIVE);
      AddTestCase (new LteX2HandoverMeasuresTestCase (3,   1,    0,    cel2, cel2name, true, sched, ho, true, useIdealRrc), TestCase::TAKES_FOREVER);
      AddTestCase (new LteX2HandoverMeasuresTestCase (4,   1,    0,    cel3, cel3name, true, sched, ho, true, useIdealRrc), TestCase::TAKES_FOREVER);
    }

  sched = "ns3::RrFfMacScheduler";
  for (useIdealRrc = 1; useIdealRrc >= 0; --useIdealRrc)
    {
      //                                          nEnbs, nUes, nDBearers, celist, name, useUdp, sched, admitHo, idealRrc
      AddTestCase (new LteX2HandoverMeasuresTestCase (2,   1,    0,    cel1, cel1name, true, sched, ho, true, useIdealRrc), TestCase::QUICK);
      AddTestCase (new LteX2HandoverMeasuresTestCase (3,   1,    0,    cel2, cel2name, true, sched, ho, true, useIdealRrc), TestCase::TAKES_FOREVER);
      AddTestCase (new LteX2HandoverMeasuresTestCase (4,   1,    0,    cel3, cel3name, true, sched, ho, true, useIdealRrc), TestCase::EXTENSIVE);
    }

} // end of LteX2HandoverMeasuresTestSuite ()

static LteX2HandoverMeasuresTestSuite g_lteX2HandoverMeasuresTestSuiteInstance;
