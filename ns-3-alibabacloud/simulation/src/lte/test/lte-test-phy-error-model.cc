/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011-2013 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Marco Miozzo <marco.miozzo@cttc.es>
 *         Nicola Baldo <nbaldo@cttc.es>
 */

#include <ns3/object.h>
#include <ns3/spectrum-interference.h>
#include <ns3/spectrum-error-model.h>
#include <ns3/log.h>
#include <ns3/test.h>
#include <ns3/simulator.h>
#include <ns3/packet.h>
#include <ns3/ptr.h>
#include <iostream>
#include <ns3/radio-bearer-stats-calculator.h>
#include <ns3/mobility-building-info.h>
#include <ns3/hybrid-buildings-propagation-loss-model.h>
#include <ns3/eps-bearer.h>
#include <ns3/node-container.h>
#include <ns3/mobility-helper.h>
#include <ns3/net-device-container.h>
#include <ns3/lte-ue-net-device.h>
#include <ns3/lte-enb-net-device.h>
#include <ns3/lte-ue-rrc.h>
#include <ns3/lte-helper.h>
#include <ns3/string.h>
#include <ns3/double.h>
#include <ns3/lte-enb-phy.h>
#include <ns3/lte-ue-phy.h>
#include <ns3/config.h>
#include <ns3/boolean.h>
#include <ns3/enum.h>
#include <ns3/integer.h>
#include <ns3/ff-mac-scheduler.h>
#include <ns3/buildings-helper.h>

#include "lte-test-phy-error-model.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LteTestPhyErrorModel");

LenaTestPhyErrorModelSuite::LenaTestPhyErrorModelSuite ()
  : TestSuite ("lte-phy-error-model", SYSTEM)
{
  NS_LOG_INFO ("creating LenaTestPhyErrorModelTestCase");
  
  
  for (uint32_t rngRun = 1; rngRun <= 3; ++rngRun)
    {

      // Tests on DL Control Channels (PCFICH+PDCCH)
      // the tolerance is calculated with the following octave code:
      //
      // n =  1000; # TX packets
      // for p=1-[0.007 0.045 0.206 0.343]
      //    tol = n*p - binoinv(0.001, n, p)
      // endfor

      // 1 interfering eNB SINR -2.0 BLER 0.007 TB size 217
      AddTestCase (new LenaDlCtrlPhyErrorModelTestCase (2, 1078, 0.007, 9,
                                                        Seconds (0.04), rngRun),
                   (rngRun == 1) ? TestCase::QUICK : TestCase::TAKES_FOREVER);
      // 2 interfering eNBs SINR -4.0 BLER 0.037 TB size 217
      AddTestCase (new LenaDlCtrlPhyErrorModelTestCase (3, 1040, 0.045, 21,
                                                        Seconds (0.04), rngRun),
                   (rngRun == 1) ? TestCase::EXTENSIVE : TestCase::TAKES_FOREVER);
      // 3 interfering eNBs SINR -6.0 BLER 0.21 TB size 133
      AddTestCase (new LenaDlCtrlPhyErrorModelTestCase (4, 1250, 0.206, 40,
                                                        Seconds (0.12), rngRun),
                   (rngRun == 1) ? TestCase::EXTENSIVE : TestCase::TAKES_FOREVER);
      // 4 interfering eNBs SINR -7.0 BLER 0.34 TB size 133
      AddTestCase (new LenaDlCtrlPhyErrorModelTestCase (5, 1260, 0.343, 47,
                                                        Seconds (0.12), rngRun),
                   (rngRun == 1) ? TestCase::EXTENSIVE : TestCase::TAKES_FOREVER);

      // Tests on DL Data channels (PDSCH)
      // the tolerance is calculated with the following octave code:
      //
      // n =  1000; # TX packets
      // for p=1-[0.33 0.11 0.2 0.3 0.55 0.14]
      //    tol = n*p - binoinv(0.005, n, p)
      // endfor

      // MCS 2 TB size of 256 bits BLER 0.33 SINR -5.51
      AddTestCase (new LenaDataPhyErrorModelTestCase (4, 1800, 0.33, 39,
                                                      Seconds (0.04), rngRun),
                   (rngRun == 1) ? TestCase::QUICK : TestCase::TAKES_FOREVER);
      // MCS 2 TB size of 528 bits BLER 0.11 SINR -5.51
      AddTestCase (new LenaDataPhyErrorModelTestCase (2, 1800, 0.11, 26,
                                                      Seconds (0.04), rngRun),
                   (rngRun == 1) ? TestCase::EXTENSIVE : TestCase::TAKES_FOREVER);
      // MCS 2 TB size of 1088 bits BLER 0.02 SINR -5.51
      AddTestCase (new LenaDataPhyErrorModelTestCase (1, 1800, 0.02, 33,
                                                      Seconds (0.04), rngRun),
                   (rngRun == 1) ? TestCase::EXTENSIVE : TestCase::TAKES_FOREVER);
      // MCS 12 TB size of 4800 bits  BLER 0.3  SINR 4.43
      AddTestCase (new LenaDataPhyErrorModelTestCase (1, 600, 0.3, 38,
                                                      Seconds (0.04), rngRun),
                   (rngRun == 1) ? TestCase::EXTENSIVE : TestCase::TAKES_FOREVER);
      // MCS 12 TB size of 1632 bits  BLER 0.55  SINR 4.43
      AddTestCase (new LenaDataPhyErrorModelTestCase (3, 600, 0.55, 40,
                                                      Seconds (0.04), rngRun),
                   (rngRun == 1) ? TestCase::EXTENSIVE : TestCase::TAKES_FOREVER);
      // MCS 16 TB size of 7272 bits (3648 x 3584) BLER 0.14 SINR 8.48
      // BLER 0.14 = 1 - ((1-0.075)*(1-0.075))
      AddTestCase (new LenaDataPhyErrorModelTestCase (1, 470, 0.14, 29,
                                                      Seconds (0.04), rngRun),
                   (rngRun == 1) ? TestCase::EXTENSIVE : TestCase::TAKES_FOREVER);

    }

}

static LenaTestPhyErrorModelSuite lenaTestPhyErrorModelSuite;

std::string 
LenaDataPhyErrorModelTestCase::BuildNameString (uint16_t nUser, uint16_t dist, uint32_t rngRun)
{
  std::ostringstream oss;
  oss << "DataPhyErrorModel " <<  nUser << " UEs, distance " << dist << " m, RngRun " << rngRun;
  return oss.str ();
}

LenaDataPhyErrorModelTestCase::LenaDataPhyErrorModelTestCase (
  uint16_t nUser, uint16_t dist, double blerRef,
  uint16_t toleranceRxPackets, Time statsStartTime, uint32_t rngRun)
  : TestCase (BuildNameString (nUser, dist, rngRun)),
    m_nUser (nUser),
    m_dist (dist),
    m_blerRef (blerRef),
    m_toleranceRxPackets (toleranceRxPackets),
    m_statsStartTime (statsStartTime),
    m_rngRun (rngRun)
{
}

LenaDataPhyErrorModelTestCase::~LenaDataPhyErrorModelTestCase ()
{
}

void
LenaDataPhyErrorModelTestCase::DoRun (void)
{
  
  double ber = 0.03;
  Config::SetDefault ("ns3::LteAmc::Ber", DoubleValue (ber));
  Config::SetDefault ("ns3::LteAmc::AmcModel", EnumValue (LteAmc::PiroEW2010));
  Config::SetDefault ("ns3::LteSpectrumPhy::CtrlErrorModelEnabled", BooleanValue (false));
  Config::SetDefault ("ns3::LteSpectrumPhy::DataErrorModelEnabled", BooleanValue (true));
  Config::SetDefault ("ns3::RrFfMacScheduler::HarqEnabled", BooleanValue (false));
  Config::SetGlobal ("RngRun", UintegerValue (m_rngRun));

  //Disable Uplink Power Control
  Config::SetDefault ("ns3::LteUePhy::EnableUplinkPowerControl", BooleanValue (false));

  /*
   * Initialize Simulation Scenario: 1 eNB and m_nUser UEs
   */

  int64_t stream = 1;
  Ptr<LteHelper> lena = CreateObject<LteHelper> ();
  
  // Create Nodes: eNodeB and UE
  NodeContainer enbNodes;
  NodeContainer ueNodes;
  enbNodes.Create (1);
  ueNodes.Create (m_nUser);

  // Install Mobility Model
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (enbNodes);
  BuildingsHelper::Install (enbNodes);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (ueNodes);
  BuildingsHelper::Install (ueNodes);
  
  // remove random shadowing component
  lena->SetAttribute ("PathlossModel", StringValue ("ns3::HybridBuildingsPropagationLossModel"));
  lena->SetPathlossModelAttribute ("ShadowSigmaOutdoor", DoubleValue (0.0));
  lena->SetPathlossModelAttribute ("ShadowSigmaIndoor", DoubleValue (0.0));
  lena->SetPathlossModelAttribute ("ShadowSigmaExtWalls", DoubleValue (0.0));

  // Create Devices and install them in the Nodes (eNB and UE)
  NetDeviceContainer enbDevs;
  NetDeviceContainer ueDevs;
  lena->SetSchedulerType ("ns3::RrFfMacScheduler");
  lena->SetSchedulerAttribute ("UlCqiFilter", EnumValue (FfMacScheduler::PUSCH_UL_CQI));
  
  enbDevs = lena->InstallEnbDevice (enbNodes);
  stream += lena->AssignStreams (enbDevs, stream);
  ueDevs = lena->InstallUeDevice (ueNodes);
  stream += lena->AssignStreams (ueDevs, stream);

  // Attach a UE to a eNB
  lena->Attach (ueDevs, enbDevs.Get (0));

  // Activate an EPS bearer
  enum EpsBearer::Qci q = EpsBearer::GBR_CONV_VOICE;
  EpsBearer bearer (q);
  lena->ActivateDataRadioBearer (ueDevs, bearer);
  

  Ptr<LteEnbNetDevice> lteEnbDev = enbDevs.Get (0)->GetObject<LteEnbNetDevice> ();
  Ptr<LteEnbPhy> enbPhy = lteEnbDev->GetPhy ();
  enbPhy->SetAttribute ("TxPower", DoubleValue (43.0));
  enbPhy->SetAttribute ("NoiseFigure", DoubleValue (5.0));
  // place the HeNB over the default rooftop level (20 mt.)
  Ptr<MobilityModel> mm = enbNodes.Get (0)->GetObject<MobilityModel> ();
  mm->SetPosition (Vector (0.0, 0.0, 30.0));

  // Set UEs' position and power
  for (int i = 0; i < m_nUser; i++)
    {
      Ptr<MobilityModel> mm1 = ueNodes.Get (i)->GetObject<MobilityModel> ();
      mm1->SetPosition (Vector (m_dist, 0.0, 1.0));
      Ptr<LteUeNetDevice> lteUeDev = ueDevs.Get (i)->GetObject<LteUeNetDevice> ();
      Ptr<LteUePhy> uePhy = lteUeDev->GetPhy ();
      uePhy->SetAttribute ("TxPower", DoubleValue (23.0));
      uePhy->SetAttribute ("NoiseFigure", DoubleValue (9.0));
    }
    
  Time statsDuration = Seconds (1.0);
  Simulator::Stop (m_statsStartTime + statsDuration - Seconds (0.0001));

  lena->EnableRlcTraces ();
  Ptr<RadioBearerStatsCalculator> rlcStats = lena->GetRlcStats ();
  rlcStats->SetAttribute ("StartTime", TimeValue (m_statsStartTime));
  rlcStats->SetAttribute ("EpochDuration", TimeValue (statsDuration));

  Simulator::Run ();

  NS_LOG_INFO ("\tTest downlink data shared channels (PDSCH)");
  NS_LOG_INFO ("Test with " << m_nUser << " user(s) at distance " << m_dist << " expected BLER " << m_blerRef);
  for (int i = 0; i < m_nUser; i++)
    {
      // get the imsi
      uint64_t imsi = ueDevs.Get (i)->GetObject<LteUeNetDevice> ()->GetImsi ();
      uint8_t lcId = 3;

      double dlRxPackets = rlcStats->GetDlRxPackets (imsi, lcId);
      double dlTxPackets = rlcStats->GetDlTxPackets (imsi, lcId);
      [[maybe_unused]] double dlBler =  1.0 - (dlRxPackets/dlTxPackets);
      double expectedDlRxPackets = dlTxPackets -dlTxPackets*m_blerRef;
      NS_LOG_INFO ("\tUser " << i << " imsi " << imsi << " DOWNLINK"
                   << " pkts rx " << dlRxPackets << " tx " << dlTxPackets
                   << " BLER " << dlBler << " Err " << std::fabs (m_blerRef - dlBler)
                   << " expected rx " << expectedDlRxPackets
                   << " difference " << std::abs (expectedDlRxPackets - dlRxPackets)
                   << " tolerance " << m_toleranceRxPackets);

      // sanity check for whether the tx packets reported by the stats are correct
      // we expect one packet per TTI
      double expectedDlTxPackets = static_cast<double> (statsDuration.GetMilliSeconds ());
      NS_TEST_ASSERT_MSG_EQ_TOL (dlTxPackets, expectedDlTxPackets, expectedDlTxPackets * 0.005, 
                                 " too different DL TX packets reported");

      // this is the main test condition: check that the RX packets are within the expected range
      NS_TEST_ASSERT_MSG_EQ_TOL (dlRxPackets, expectedDlRxPackets, m_toleranceRxPackets, 
                                 " too different DL RX packets reported");
    }


  Simulator::Destroy ();
}




std::string 
LenaDlCtrlPhyErrorModelTestCase::BuildNameString (uint16_t nEnb, uint16_t dist, uint32_t rngRun)
{
  std::ostringstream oss;
  oss << "DlCtrlPhyErrorModel " << nEnb << " eNBs, distance " << dist << " m, RngRun " << rngRun;
  return oss.str ();
}


LenaDlCtrlPhyErrorModelTestCase::LenaDlCtrlPhyErrorModelTestCase (
  uint16_t nEnb, uint16_t dist, double blerRef,
  uint16_t toleranceRxPackets, Time statsStartTime, uint32_t rngRun)
  : TestCase (BuildNameString (nEnb, dist, rngRun)),
    m_nEnb (nEnb),
    m_dist (dist),
    m_blerRef (blerRef),
    m_toleranceRxPackets (toleranceRxPackets),
    m_statsStartTime (statsStartTime),
    m_rngRun (rngRun)
{
}

LenaDlCtrlPhyErrorModelTestCase::~LenaDlCtrlPhyErrorModelTestCase ()
{
}

void
LenaDlCtrlPhyErrorModelTestCase::DoRun (void)
{
  
  double ber = 0.03;
  Config::SetDefault ("ns3::LteAmc::Ber", DoubleValue (ber));
  Config::SetDefault ("ns3::LteAmc::AmcModel", EnumValue (LteAmc::PiroEW2010));
  Config::SetDefault ("ns3::LteSpectrumPhy::CtrlErrorModelEnabled", BooleanValue (true));
  Config::SetDefault ("ns3::LteSpectrumPhy::DataErrorModelEnabled", BooleanValue (false));
  Config::SetDefault ("ns3::RrFfMacScheduler::HarqEnabled", BooleanValue (false));
  Config::SetGlobal ("RngRun", UintegerValue (m_rngRun));

  Config::SetDefault ("ns3::RadioBearerStatsCalculator::DlRlcOutputFilename", StringValue (CreateTempDirFilename ("DlRlcStats.txt")));
  Config::SetDefault ("ns3::RadioBearerStatsCalculator::UlRlcOutputFilename", StringValue (CreateTempDirFilename ("UlRlcStats.txt")));

  //Disable Uplink Power Control
  Config::SetDefault ("ns3::LteUePhy::EnableUplinkPowerControl", BooleanValue (false));

  /*
   * Initialize Simulation Scenario: 1 eNB and m_nUser UEs
   */

  int64_t stream = 1;
  Ptr<LteHelper> lena = CreateObject<LteHelper> ();
  
  // Create Nodes: eNodeB and UE
  NodeContainer enbNodes;
  NodeContainer ueNodes;
  enbNodes.Create (m_nEnb);
  ueNodes.Create (1);
  
  // Install Mobility Model
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (enbNodes);
  BuildingsHelper::Install (enbNodes);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (ueNodes);
  BuildingsHelper::Install (ueNodes);
  
  // remove random shadowing component
  lena->SetAttribute ("PathlossModel", StringValue ("ns3::HybridBuildingsPropagationLossModel"));
  lena->SetPathlossModelAttribute ("ShadowSigmaOutdoor", DoubleValue (0.0));
  lena->SetPathlossModelAttribute ("ShadowSigmaIndoor", DoubleValue (0.0));
  lena->SetPathlossModelAttribute ("ShadowSigmaExtWalls", DoubleValue (0.0));
  
  // Create Devices and install them in the Nodes (eNB and UE)
  NetDeviceContainer enbDevs;
  NetDeviceContainer ueDevs;
  lena->SetSchedulerType ("ns3::RrFfMacScheduler");
  lena->SetSchedulerAttribute ("UlCqiFilter", EnumValue (FfMacScheduler::PUSCH_UL_CQI));
  
  enbDevs = lena->InstallEnbDevice (enbNodes);
  stream += lena->AssignStreams (enbDevs, stream);
  ueDevs = lena->InstallUeDevice (ueNodes);
  stream += lena->AssignStreams (ueDevs, stream);
  
  // Attach a UE to one eNB (the others are interfering ones)
  lena->Attach (ueDevs, enbDevs.Get (0));
  
  // Activate an EPS bearer
  enum EpsBearer::Qci q = EpsBearer::GBR_CONV_VOICE;
  EpsBearer bearer (q);
  lena->ActivateDataRadioBearer (ueDevs, bearer);
  
  // Set UEs' position and power
  for (int i = 0; i < m_nEnb; i++)
    {
      // place the HeNB over the default rooftop level (20 mt.)
      Ptr<MobilityModel> mm = enbNodes.Get (i)->GetObject<MobilityModel> ();
      mm->SetPosition (Vector (0.0, 0.0, 30.0));
      Ptr<LteEnbNetDevice> lteEnbDev = enbDevs.Get (i)->GetObject<LteEnbNetDevice> ();
      Ptr<LteEnbPhy> enbPhy = lteEnbDev->GetPhy ();
      enbPhy->SetAttribute ("TxPower", DoubleValue (43.0));
      enbPhy->SetAttribute ("NoiseFigure", DoubleValue (5.0));
    }
  
  // Set UEs' position and power
  Ptr<MobilityModel> mm = ueNodes.Get (0)->GetObject<MobilityModel> ();
  mm->SetPosition (Vector (m_dist, 0.0, 1.0));
  Ptr<LteUeNetDevice> lteUeDev = ueDevs.Get (0)->GetObject<LteUeNetDevice> ();
  Ptr<LteUePhy> uePhy = lteUeDev->GetPhy ();
  uePhy->SetAttribute ("TxPower", DoubleValue (23.0));
  uePhy->SetAttribute ("NoiseFigure", DoubleValue (9.0));
  
  Time statsDuration = Seconds (1.0);
  Simulator::Stop (m_statsStartTime + statsDuration - Seconds (0.0001));

  lena->EnableRlcTraces ();
  Ptr<RadioBearerStatsCalculator> rlcStats = lena->GetRlcStats ();
  rlcStats->SetAttribute ("StartTime", TimeValue (m_statsStartTime));
  rlcStats->SetAttribute ("EpochDuration", TimeValue (statsDuration));

  Simulator::Run ();
  
  NS_LOG_INFO ("\tTest downlink control channels (PCFICH+PDCCH)");
  NS_LOG_INFO ("Test with " << m_nEnb << " eNB(s) at distance " << m_dist << " expected BLER " << m_blerRef);
  int nUser = 1;
  for (int i = 0; i < nUser; i++)
    {
      // get the imsi
      uint64_t imsi = ueDevs.Get (i)->GetObject<LteUeNetDevice> ()->GetImsi ();
      uint8_t lcId = 3;
      double dlRxPackets = rlcStats->GetDlRxPackets (imsi, lcId);
      double dlTxPackets = rlcStats->GetDlTxPackets (imsi, lcId);
      [[maybe_unused]] double dlBler = 1.0 - (dlRxPackets/dlTxPackets);
      double expectedDlRxPackets = dlTxPackets -dlTxPackets*m_blerRef;
      NS_LOG_INFO ("\tUser " << i << " imsi " << imsi << " DOWNLINK"
                   << " pkts rx " << dlRxPackets << " tx " << dlTxPackets
                   << " BLER " << dlBler << " Err " << std::fabs (m_blerRef - dlBler)
                   << " expected rx " << expectedDlRxPackets
                   << " difference " << std::abs (expectedDlRxPackets - dlRxPackets)
                   << " tolerance " << m_toleranceRxPackets);

      // sanity check for whether the tx packets reported by the stats are correct
      // we expect one packet per TTI
      double expectedDlTxPackets = static_cast<double> (statsDuration.GetMilliSeconds ());
      NS_TEST_ASSERT_MSG_EQ_TOL (dlTxPackets, expectedDlTxPackets, expectedDlTxPackets * 0.005, 
                                 " too different DL TX packets reported");

      // this is the main test condition: check that the RX packets are within the expected range
      NS_TEST_ASSERT_MSG_EQ_TOL (dlRxPackets, expectedDlRxPackets, m_toleranceRxPackets,
                                 "too different DL RX packets reported");

    }
  
  Simulator::Destroy ();
}
