/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Piotr Gawlowicz
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
 * Author: Piotr Gawlowicz <gawlowicz.p@gmail.com>
 * Based on lte-test-interference.{h,cc} by Manuel Requena <manuel.requena@cttc.es>
 *                                                                              Nicola Baldo <nbaldo@cttc.es>
 */

#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/string.h"
#include "ns3/double.h"
#include <ns3/enum.h>
#include "ns3/boolean.h"
#include <ns3/pointer.h>
#include "ns3/mobility-helper.h"
#include "ns3/lte-helper.h"
#include "ns3/ff-mac-scheduler.h"

#include "ns3/lte-enb-phy.h"
#include "ns3/lte-enb-net-device.h"

#include "ns3/lte-ue-phy.h"
#include "ns3/lte-ue-net-device.h"

#include "lte-test-interference-fr.h"

#include <ns3/lte-chunk-processor.h>

#include "lte-simple-spectrum-phy.h"
#include "ns3/spectrum-value.h"
#include "ns3/lte-spectrum-value-helper.h"


using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LteInterferenceFrTest");



/**
 * TestSuite
 */

LteInterferenceFrTestSuite::LteInterferenceFrTestSuite ()
  : TestSuite ("lte-interference-fr", SYSTEM)
{
//  LogLevel logLevel = (LogLevel)(LOG_PREFIX_FUNC | LOG_PREFIX_TIME | LOG_LEVEL_DEBUG);
//  LogComponentEnable ("LteInterferenceFrTest", logLevel);

  AddTestCase (new LteInterferenceHardFrTestCase ("d1=50, d2=20",  50.000000, 20.000000,  356449.932732, 10803.280215), TestCase::QUICK);
  AddTestCase (new LteInterferenceHardFrTestCase ("d1=50, d2=50",  50.000000, 50.000000,  356449.932732, 10803.280215), TestCase::QUICK);
  AddTestCase (new LteInterferenceHardFrTestCase ("d1=50, d2=200",  50.000000, 200.000000,  356449.932732, 10803.280215), TestCase::QUICK);
  AddTestCase (new LteInterferenceHardFrTestCase ("d1=50, d2=500",  50.000000, 500.000000,  356449.932732, 10803.280215), TestCase::QUICK);

  AddTestCase (new LteInterferenceStrictFrTestCase ("d1=50, d2=20",  50.000000, 20.000000,  0.160000, 0.159998, 356449.932732, 10803.280215, 18), TestCase::QUICK);
  AddTestCase (new LteInterferenceStrictFrTestCase ("d1=50, d2=50",  50.000000, 50.000000, 0.999997, 0.999907, 356449.932732, 10803.280215, 28), TestCase::QUICK);
  AddTestCase (new LteInterferenceStrictFrTestCase ("d1=50, d2=200",  50.000000, 200.000000,  15.999282, 15.976339, 356449.932732, 10803.280215, 30), TestCase::QUICK);
  AddTestCase (new LteInterferenceStrictFrTestCase ("d1=50, d2=500",  50.000000, 500.000000,  99.971953, 99.082845, 356449.932732, 10803.280215, 30), TestCase::QUICK);

}

static LteInterferenceFrTestSuite LteInterferenceFrTestSuite;


/**
 * TestCase Data
 */
LteInterferenceHardFrTestCase::LteInterferenceHardFrTestCase (std::string name, double d1, double d2, double dlSinr, double ulSinr)
  : TestCase ("Test: " + name),
    m_d1 (d1),
    m_d2 (d2),
    m_expectedDlSinrDb (10 * std::log10 (dlSinr))
{
  NS_LOG_INFO ("Creating LteInterferenceFrTestCase");
}

LteInterferenceHardFrTestCase::~LteInterferenceHardFrTestCase ()
{
}

void
LteInterferenceHardFrTestCase::DoRun (void)
{
  NS_LOG_INFO (this << GetName ());
  NS_LOG_DEBUG ("LteInterferenceHardFrTestCase");

  Config::Reset ();
  Config::SetDefault ("ns3::LteHelper::UseIdealRrc", BooleanValue (false));

  Config::SetDefault ("ns3::LteSpectrumPhy::CtrlErrorModelEnabled", BooleanValue (false));
  Config::SetDefault ("ns3::LteSpectrumPhy::DataErrorModelEnabled", BooleanValue (false));
  Config::SetDefault ("ns3::LteAmc::AmcModel", EnumValue (LteAmc::PiroEW2010));
  Config::SetDefault ("ns3::LteAmc::Ber", DoubleValue (0.00005));

  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  lteHelper->SetFfrAlgorithmType ("ns3::LteFrHardAlgorithm");

  lteHelper->SetAttribute ("PathlossModel", StringValue ("ns3::FriisSpectrumPropagationLossModel"));

  // Create Nodes: eNodeB and UE
  NodeContainer enbNodes;
  NodeContainer ueNodes1;
  NodeContainer ueNodes2;
  enbNodes.Create (2);
  ueNodes1.Create (1);
  ueNodes2.Create (1);
  NodeContainer allNodes = NodeContainer ( enbNodes, ueNodes1, ueNodes2);

  // the topology is the following:
  //         d2
  //  UE1-----------eNB2
  //   |             |
  // d1|             |d1
  //   |     d2      |
  //  eNB1----------UE2
  //
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));   // eNB1
  positionAlloc->Add (Vector (m_d2, m_d1, 0.0)); // eNB2
  positionAlloc->Add (Vector (0.0, m_d1, 0.0));  // UE1
  positionAlloc->Add (Vector (m_d2, 0.0, 0.0));  // UE2
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.SetPositionAllocator (positionAlloc);
  mobility.Install (allNodes);

  // Create Devices and install them in the Nodes (eNB and UE)
  NetDeviceContainer enbDevs;
  NetDeviceContainer ueDevs1;
  NetDeviceContainer ueDevs2;
  lteHelper->SetSchedulerType ("ns3::PfFfMacScheduler");
  lteHelper->SetSchedulerAttribute ("UlCqiFilter", EnumValue (FfMacScheduler::PUSCH_UL_CQI));

  lteHelper->SetFfrAlgorithmAttribute ("DlSubBandOffset", UintegerValue (0));
  lteHelper->SetFfrAlgorithmAttribute ("DlSubBandwidth", UintegerValue (12));
  lteHelper->SetFfrAlgorithmAttribute ("UlSubBandOffset", UintegerValue (0));
  lteHelper->SetFfrAlgorithmAttribute ("UlSubBandwidth", UintegerValue (25));
  enbDevs.Add (lteHelper->InstallEnbDevice (enbNodes.Get (0)));

  lteHelper->SetFfrAlgorithmAttribute ("DlSubBandOffset", UintegerValue (12));
  lteHelper->SetFfrAlgorithmAttribute ("DlSubBandwidth", UintegerValue (12));
  lteHelper->SetFfrAlgorithmAttribute ("UlSubBandOffset", UintegerValue (0));
  lteHelper->SetFfrAlgorithmAttribute ("UlSubBandwidth", UintegerValue (25));
  enbDevs.Add (lteHelper->InstallEnbDevice (enbNodes.Get (1)));

  ueDevs1 = lteHelper->InstallUeDevice (ueNodes1);
  ueDevs2 = lteHelper->InstallUeDevice (ueNodes2);

  lteHelper->Attach (ueDevs1, enbDevs.Get (0));
  lteHelper->Attach (ueDevs2, enbDevs.Get (1));

  // Activate an EPS bearer
  enum EpsBearer::Qci q = EpsBearer::GBR_CONV_VOICE;
  EpsBearer bearer (q);
  lteHelper->ActivateDataRadioBearer (ueDevs1, bearer);
  lteHelper->ActivateDataRadioBearer (ueDevs2, bearer);

  // Use testing chunk processor in the PHY layer
  // It will be used to test that the SNR is as intended
  // we plug in two instances, one for DL and one for UL

  Ptr<LtePhy> ue1Phy = ueDevs1.Get (0)->GetObject<LteUeNetDevice> ()->GetPhy ()->GetObject<LtePhy> ();
  Ptr<LteChunkProcessor> testDlSinr1 = Create<LteChunkProcessor> ();
  LteSpectrumValueCatcher dlSinr1Catcher;
  testDlSinr1->AddCallback (MakeCallback (&LteSpectrumValueCatcher::ReportValue, &dlSinr1Catcher));
  ue1Phy->GetDownlinkSpectrumPhy ()->AddDataSinrChunkProcessor (testDlSinr1);

  Ptr<LtePhy> enb1phy = enbDevs.Get (0)->GetObject<LteEnbNetDevice> ()->GetPhy ()->GetObject<LtePhy> ();
  Ptr<LteChunkProcessor> testUlSinr1 = Create<LteChunkProcessor> ();
  LteSpectrumValueCatcher ulSinr1Catcher;
  testUlSinr1->AddCallback (MakeCallback (&LteSpectrumValueCatcher::ReportValue, &ulSinr1Catcher));
  enb1phy->GetUplinkSpectrumPhy ()->AddDataSinrChunkProcessor (testUlSinr1);

  // same as above for eNB2 and UE2

  Ptr<LtePhy> ue2Phy = ueDevs2.Get (0)->GetObject<LteUeNetDevice> ()->GetPhy ()->GetObject<LtePhy> ();
  Ptr<LteChunkProcessor> testDlSinr2 = Create<LteChunkProcessor> ();
  LteSpectrumValueCatcher dlSinr2Catcher;
  testDlSinr2->AddCallback (MakeCallback (&LteSpectrumValueCatcher::ReportValue, &dlSinr2Catcher));
  ue2Phy->GetDownlinkSpectrumPhy ()->AddDataSinrChunkProcessor (testDlSinr2);

  Ptr<LtePhy> enb2phy = enbDevs.Get (1)->GetObject<LteEnbNetDevice> ()->GetPhy ()->GetObject<LtePhy> ();
  Ptr<LteChunkProcessor> testUlSinr2 = Create<LteChunkProcessor> ();
  LteSpectrumValueCatcher ulSinr2Catcher;
  testUlSinr2->AddCallback (MakeCallback (&LteSpectrumValueCatcher::ReportValue, &ulSinr2Catcher));
  enb1phy->GetUplinkSpectrumPhy ()->AddDataSinrChunkProcessor (testUlSinr2);

// need to allow for RRC connection establishment + SRS
  Simulator::Stop (Seconds (0.200));
  Simulator::Run ();


  for (uint32_t i = 0; i < 12; i++)
    {
      double dlSinr1 = dlSinr1Catcher.GetValue ()->operator[] (i);
      double dlSinr1Db = 10.0 * std::log10 (dlSinr1);
      NS_TEST_ASSERT_MSG_EQ_TOL (dlSinr1Db, m_expectedDlSinrDb, 0.01, "Wrong SINR in DL! (eNB1 --> UE1)");


      double dlSinr2 = dlSinr2Catcher.GetValue ()->operator[] (i);
      NS_TEST_ASSERT_MSG_EQ_TOL (dlSinr2, 0, 0.01, "Wrong SINR in DL! (eNB2 --> UE2)");
    }

  for (uint32_t i = 12; i < 24; i++)
    {
      double dlSinr1 = dlSinr1Catcher.GetValue ()->operator[] (i);
      NS_TEST_ASSERT_MSG_EQ_TOL (dlSinr1, 0, 0.01, "Wrong SINR in DL! (eNB1 --> UE1)");

      double dlSinr2 = dlSinr2Catcher.GetValue ()->operator[] (i);
      double dlSinr2Db = 10.0 * std::log10 (dlSinr2);
      NS_TEST_ASSERT_MSG_EQ_TOL (dlSinr2Db, m_expectedDlSinrDb, 0.01, "Wrong SINR in DL! (eNB2 --> UE2)");
    }

  //FR algorithms do not operate in uplink now, so we do not test it
//  double ulSinr1Db = 10.0 * std::log10 (testUlSinr1->GetValue ()->operator[] (0));
//  NS_LOG_DEBUG("ulSinr1Db: "<< ulSinr1Db);
//  NS_TEST_ASSERT_MSG_EQ_TOL (ulSinr1Db, m_expectedUlSinrDb, 0.01, "Wrong SINR in UL!  (UE1 --> eNB1)");
//
//  double ulSinr2Db = 10.0 * std::log10 (testUlSinr2->GetValue ()->operator[] (0));
//  NS_LOG_DEBUG("ulSinr2Db: "<< ulSinr2Db);
//  NS_TEST_ASSERT_MSG_EQ_TOL (ulSinr2Db, m_expectedUlSinrDb, 0.01, "Wrong SINR in UL!  (UE2 --> eNB2)");

  Simulator::Destroy ();
}

LteInterferenceStrictFrTestCase::LteInterferenceStrictFrTestCase (std::string name, double d1, double d2,
                                                                  double commonDlSinr, double commonUlSinr, double edgeDlSinr, double edgeUlSinr,
                                                                  uint32_t rspqThreshold)
  : TestCase ("Test: " + name),
    m_d1 (d1),
    m_d2 (d2),
    m_commonDlSinrDb (10 * std::log10 (commonDlSinr)),
    m_edgeDlSinrDb (10 * std::log10 (edgeDlSinr)),
    m_rspqThreshold (rspqThreshold)
{
  NS_LOG_INFO ("Creating LteInterferenceFrTestCase");
}

LteInterferenceStrictFrTestCase::~LteInterferenceStrictFrTestCase ()
{
}

void
LteInterferenceStrictFrTestCase::DoRun (void)
{
  NS_LOG_INFO (this << GetName ());
  NS_LOG_DEBUG ("LteInterferenceStrictFrTestCase");

  Config::Reset ();
  Config::SetDefault ("ns3::LteHelper::UseIdealRrc", BooleanValue (true));

  Config::SetDefault ("ns3::LteSpectrumPhy::CtrlErrorModelEnabled", BooleanValue (false));
  Config::SetDefault ("ns3::LteSpectrumPhy::DataErrorModelEnabled", BooleanValue (false));
  Config::SetDefault ("ns3::LteAmc::AmcModel", EnumValue (LteAmc::PiroEW2010));
  Config::SetDefault ("ns3::LteAmc::Ber", DoubleValue (0.00005));

  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  lteHelper->SetFfrAlgorithmType ("ns3::LteFrStrictAlgorithm");
  lteHelper->SetFfrAlgorithmAttribute ("RsrqThreshold", UintegerValue (m_rspqThreshold));
  lteHelper->SetFfrAlgorithmAttribute ("CenterPowerOffset",
                                       UintegerValue (LteRrcSap::PdschConfigDedicated::dB0));
  lteHelper->SetFfrAlgorithmAttribute ("EdgePowerOffset",
                                       UintegerValue (LteRrcSap::PdschConfigDedicated::dB0));


  lteHelper->SetAttribute ("PathlossModel", StringValue ("ns3::FriisSpectrumPropagationLossModel"));

  // Create Nodes: eNodeB and UE
  NodeContainer enbNodes;
  NodeContainer ueNodes1;
  NodeContainer ueNodes2;
  enbNodes.Create (2);
  ueNodes1.Create (2);
  ueNodes2.Create (2);
  NodeContainer allNodes = NodeContainer ( enbNodes, ueNodes1, ueNodes2);

  // the topology is the following:
  //         d2
  //  UE1-----------eNB2
  //   |             |
  // d1|             |d1
  //   |     d2      |
  //  eNB1----------UE2
  //
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0,  0.0,  0.0));  // eNB1
  positionAlloc->Add (Vector (m_d2, m_d1, 0.0));  // eNB2

  positionAlloc->Add (Vector (0.0, m_d1, 0.0));  // UE1-eNB1
  positionAlloc->Add (Vector (0.5 * m_d2, 0.0, 0.0));  // UE2-eNB1

  positionAlloc->Add (Vector (m_d2, 0.0, 0.0));   // UE1-eNB2
  positionAlloc->Add (Vector (0.5 * m_d2, m_d1, 0.0));    // UE2-eNB2

  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.SetPositionAllocator (positionAlloc);
  mobility.Install (allNodes);

  // Create Devices and install them in the Nodes (eNB and UE)
  NetDeviceContainer enbDevs;
  NetDeviceContainer ueDevs1;
  NetDeviceContainer ueDevs2;
  lteHelper->SetSchedulerType ("ns3::PfFfMacScheduler");
  lteHelper->SetSchedulerAttribute ("UlCqiFilter", EnumValue (FfMacScheduler::PUSCH_UL_CQI));


  lteHelper->SetFfrAlgorithmAttribute ("DlCommonSubBandwidth", UintegerValue (12));
  lteHelper->SetFfrAlgorithmAttribute ("DlEdgeSubBandOffset", UintegerValue (0));
  lteHelper->SetFfrAlgorithmAttribute ("DlEdgeSubBandwidth", UintegerValue (6));
  lteHelper->SetFfrAlgorithmAttribute ("UlCommonSubBandwidth", UintegerValue (25));
  lteHelper->SetFfrAlgorithmAttribute ("UlEdgeSubBandOffset", UintegerValue (0));
  lteHelper->SetFfrAlgorithmAttribute ("UlEdgeSubBandwidth", UintegerValue (0));

  enbDevs.Add (lteHelper->InstallEnbDevice (enbNodes.Get (0)));

  lteHelper->SetFfrAlgorithmAttribute ("DlCommonSubBandwidth", UintegerValue (12));
  lteHelper->SetFfrAlgorithmAttribute ("DlEdgeSubBandOffset", UintegerValue (6));
  lteHelper->SetFfrAlgorithmAttribute ("DlEdgeSubBandwidth", UintegerValue (6));
  enbDevs.Add (lteHelper->InstallEnbDevice (enbNodes.Get (1)));

  ueDevs1 = lteHelper->InstallUeDevice (ueNodes1);
  ueDevs2 = lteHelper->InstallUeDevice (ueNodes2);

  lteHelper->Attach (ueDevs1, enbDevs.Get (0));
  lteHelper->Attach (ueDevs2, enbDevs.Get (1));

  // Activate an EPS bearer
  enum EpsBearer::Qci q = EpsBearer::GBR_CONV_VOICE;
  EpsBearer bearer (q);
  lteHelper->ActivateDataRadioBearer (ueDevs1, bearer);
  lteHelper->ActivateDataRadioBearer (ueDevs2, bearer);

  // Use testing chunk processor in the PHY layer
  // It will be used to test that the SNR is as intended
  // we plug in two instances, one for DL and one for UL

  Ptr<LtePhy> ue1Phy = ueDevs1.Get (0)->GetObject<LteUeNetDevice> ()->GetPhy ()->GetObject<LtePhy> ();
  Ptr<LteChunkProcessor> testDlSinr1 = Create<LteChunkProcessor> ();
  LteSpectrumValueCatcher dlSinr1Catcher;
  testDlSinr1->AddCallback (MakeCallback (&LteSpectrumValueCatcher::ReportValue, &dlSinr1Catcher));
  ue1Phy->GetDownlinkSpectrumPhy ()->AddDataSinrChunkProcessor (testDlSinr1);

  Ptr<LtePhy> enb1phy = enbDevs.Get (0)->GetObject<LteEnbNetDevice> ()->GetPhy ()->GetObject<LtePhy> ();
  Ptr<LteChunkProcessor> testUlSinr1 = Create<LteChunkProcessor> ();
  LteSpectrumValueCatcher ulSinr1Catcher;
  testUlSinr1->AddCallback (MakeCallback (&LteSpectrumValueCatcher::ReportValue, &ulSinr1Catcher));
  enb1phy->GetUplinkSpectrumPhy ()->AddDataSinrChunkProcessor (testUlSinr1);

  // same as above for eNB2 and UE2

  Ptr<LtePhy> ue2Phy = ueDevs2.Get (0)->GetObject<LteUeNetDevice> ()->GetPhy ()->GetObject<LtePhy> ();
  Ptr<LteChunkProcessor> testDlSinr2 = Create<LteChunkProcessor> ();
  LteSpectrumValueCatcher dlSinr2Catcher;
  testDlSinr2->AddCallback (MakeCallback (&LteSpectrumValueCatcher::ReportValue, &dlSinr2Catcher));
  ue2Phy->GetDownlinkSpectrumPhy ()->AddDataSinrChunkProcessor (testDlSinr2);

  Ptr<LtePhy> enb2phy = enbDevs.Get (1)->GetObject<LteEnbNetDevice> ()->GetPhy ()->GetObject<LtePhy> ();
  Ptr<LteChunkProcessor> testUlSinr2 = Create<LteChunkProcessor> ();
  LteSpectrumValueCatcher ulSinr2Catcher;
  testUlSinr2->AddCallback (MakeCallback (&LteSpectrumValueCatcher::ReportValue, &ulSinr2Catcher));
  enb1phy->GetUplinkSpectrumPhy ()->AddDataSinrChunkProcessor (testUlSinr2);

// need to allow for UE Measurement report
  Simulator::Stop (Seconds (2.000));
  Simulator::Run ();


  for (uint32_t i = 0; i < 12; i++)
    {
      double dlSinr1 = dlSinr1Catcher.GetValue ()->operator[] (i);
      double dlSinr1Db = 10.0 * std::log10 (dlSinr1);
      NS_TEST_ASSERT_MSG_EQ_TOL (dlSinr1Db, m_commonDlSinrDb, 0.01, "Wrong SINR in DL! (eNB1 --> UE1)");


      double dlSinr2 = dlSinr2Catcher.GetValue ()->operator[] (i);
      double dlSinr2Db = 10.0 * std::log10 (dlSinr2);
      NS_TEST_ASSERT_MSG_EQ_TOL (dlSinr2Db, m_commonDlSinrDb, 0.01, "Wrong SINR in DL! (eNB2 --> UE2)");
    }

  for (uint32_t i = 12; i < 18; i++)
    {
      double dlSinr1 = dlSinr1Catcher.GetValue ()->operator[] (i);
      double dlSinr1Db = 10.0 * std::log10 (dlSinr1);
      NS_TEST_ASSERT_MSG_EQ_TOL (dlSinr1Db, m_edgeDlSinrDb, 0.01, "Wrong SINR in DL! (eNB1 --> UE1)");


      double dlSinr2 = dlSinr2Catcher.GetValue ()->operator[] (i);
      NS_TEST_ASSERT_MSG_EQ_TOL (dlSinr2, 0, 0.01, "Wrong SINR in DL! (eNB2 --> UE2)");
    }

  for (uint32_t i = 18; i < 24; i++)
    {
      double dlSinr1 = dlSinr1Catcher.GetValue ()->operator[] (i);
      NS_TEST_ASSERT_MSG_EQ_TOL (dlSinr1, 0, 0.01, "Wrong SINR in DL! (eNB1 --> UE1)");

      double dlSinr2 = dlSinr2Catcher.GetValue ()->operator[] (i);
      double dlSinr2Db = 10.0 * std::log10 (dlSinr2);
      NS_TEST_ASSERT_MSG_EQ_TOL (dlSinr2Db, m_edgeDlSinrDb, 0.01, "Wrong SINR in DL! (eNB2 --> UE2)");
    }


  //FR algorithms do not operate in uplink now, so we do not test it
//  double ulSinr1Db = 10.0 * std::log10 (testUlSinr1->GetValue ()->operator[] (0));
//  NS_LOG_DEBUG("ulSinr1Db: "<< ulSinr1Db);
//  NS_TEST_ASSERT_MSG_EQ_TOL (ulSinr1Db, m_expectedUlSinrDb, 0.01, "Wrong SINR in UL!  (UE1 --> eNB1)");
//
//  double ulSinr2Db = 10.0 * std::log10 (testUlSinr2->GetValue ()->operator[] (0));
//  NS_LOG_DEBUG("ulSinr2Db: "<< ulSinr2Db);
//  NS_TEST_ASSERT_MSG_EQ_TOL (ulSinr2Db, m_expectedUlSinrDb, 0.01, "Wrong SINR in UL!  (UE2 --> eNB2)");

  Simulator::Destroy ();
}
