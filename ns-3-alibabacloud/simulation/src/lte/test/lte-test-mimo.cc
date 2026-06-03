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
 * Author: Marco Miozzo <marco.miozzo@cttc.es>
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
#include <ns3/mobility-building-info.h>
#include <ns3/buildings-propagation-loss-model.h>
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
#include <ns3/rr-ff-mac-scheduler.h>
#include <ns3/pf-ff-mac-scheduler.h>
#include <ns3/pointer.h>
#include <ns3/enum.h>
#include <ns3/buildings-helper.h>
#include "lte-test-mimo.h"


using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LteTestMimo");

LenaTestMimoSuite::LenaTestMimoSuite ()
  : TestSuite ("lte-mimo", SYSTEM)
{
  NS_LOG_INFO ("creating LenaMimoTestCase");

  // RR DOWNLINK- DISTANCE 300
  // interval 1 : [0.1, 0.2) sec TxMode 0: MCS 20 -> TB size 1191 bytes
  // interval 2 : [0.3, 0.4) sec TxMode 1: MCS 26 -> TB size 1836 bytes
  // interval 3 : [0.5, 0.6) sec TxMode 2: MCS 18 -> TB size  967 bytes (x2 layers)
  // --> 
  std::vector<uint32_t> estThrDl;
  estThrDl.push_back (119100); // interval 1 : estimated throughput for TxMode 1
  estThrDl.push_back (183600); // interval 2 : estimated throughput for TxMode 2
  estThrDl.push_back (193400); // interval 3 : estimated throughput for TxMode 3
  AddTestCase (new LenaMimoTestCase(300, estThrDl, "ns3::RrFfMacScheduler", true), TestCase::QUICK);
  AddTestCase (new LenaMimoTestCase(300, estThrDl, "ns3::PfFfMacScheduler", true), TestCase::QUICK);
  AddTestCase (new LenaMimoTestCase(300, estThrDl, "ns3::RrFfMacScheduler", false), TestCase::QUICK);
  AddTestCase (new LenaMimoTestCase(300, estThrDl, "ns3::PfFfMacScheduler", false), TestCase::QUICK);

}

static LenaTestMimoSuite lenaTestMimoSuite;

std::string 
LenaMimoTestCase::BuildNameString (uint16_t dist, std::string schedulerType, bool useIdealRrc)
{
  std::ostringstream oss;
  oss << " UE distance " << dist << " m" << " Scheduler " << schedulerType;
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

LenaMimoTestCase::LenaMimoTestCase (uint16_t dist, std::vector<uint32_t> estThrDl, std::string schedulerType, bool useIdealRrc)
  : TestCase (BuildNameString (dist, schedulerType, useIdealRrc)),              
    m_dist (dist),
    m_estThrDl (estThrDl),
    m_schedulerType (schedulerType),
    m_useIdealRrc (useIdealRrc)
{
}

LenaMimoTestCase::~LenaMimoTestCase ()
{
}

void
LenaMimoTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this << GetName ());
  Config::SetDefault ("ns3::LteSpectrumPhy::DataErrorModelEnabled", BooleanValue (false));
  Config::SetDefault ("ns3::LteAmc::AmcModel", EnumValue (LteAmc::PiroEW2010));
  Config::SetDefault ("ns3::LteHelper::UseIdealRrc", BooleanValue (m_useIdealRrc));
  Config::SetDefault ("ns3::MacStatsCalculator::DlOutputFilename", StringValue (CreateTempDirFilename ("DlMacStats.txt")));
  Config::SetDefault ("ns3::MacStatsCalculator::UlOutputFilename", StringValue (CreateTempDirFilename ("UlMacStats.txt")));
  Config::SetDefault ("ns3::RadioBearerStatsCalculator::DlRlcOutputFilename", StringValue (CreateTempDirFilename ("DlRlcStats.txt")));
  Config::SetDefault ("ns3::RadioBearerStatsCalculator::UlRlcOutputFilename", StringValue (CreateTempDirFilename ("UlRlcStats.txt")));

  //Disable Uplink Power Control
  Config::SetDefault ("ns3::LteUePhy::EnableUplinkPowerControl", BooleanValue (false));

  /**
   * Initialize Simulation Scenario: 1 eNB and m_nUser UEs
   */


  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  Config::SetDefault ("ns3::RrFfMacScheduler::HarqEnabled", BooleanValue (false));
  Config::SetDefault ("ns3::PfFfMacScheduler::HarqEnabled", BooleanValue (false));
  
//   lteHelper->SetSchedulerAttribute ("HarqEnabled", BooleanValue (false));
  
  
  lteHelper->SetAttribute ("PathlossModel", StringValue ("ns3::HybridBuildingsPropagationLossModel"));
  lteHelper->SetPathlossModelAttribute ("ShadowSigmaOutdoor", DoubleValue (0.0));
  lteHelper->SetPathlossModelAttribute ("ShadowSigmaIndoor", DoubleValue (0.0));
  lteHelper->SetPathlossModelAttribute ("ShadowSigmaExtWalls", DoubleValue (0.0));
  
//   lteHelper->EnableLogComponents ();

  // Create Nodes: eNodeB and UE
  NodeContainer enbNodes;
  NodeContainer ueNodes;
  enbNodes.Create (1);
  ueNodes.Create (1);

  // Install Mobility Model
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (enbNodes);
  BuildingsHelper::Install (enbNodes);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (ueNodes);
  BuildingsHelper::Install (ueNodes);

  // Create Devices and install them in the Nodes (eNB and UE)
  NetDeviceContainer enbDevs;
  NetDeviceContainer ueDevs;
  lteHelper->SetSchedulerType (m_schedulerType);
  enbDevs = lteHelper->InstallEnbDevice (enbNodes);
  ueDevs = lteHelper->InstallUeDevice (ueNodes);
  
  // Attach a UE to a eNB
  lteHelper->Attach (ueDevs, enbDevs.Get (0));

  // Activate an EPS bearer
  enum EpsBearer::Qci q = EpsBearer::GBR_CONV_VOICE;
  EpsBearer bearer (q);
  lteHelper->ActivateDataRadioBearer (ueDevs, bearer);
  

  Ptr<LteEnbNetDevice> lteEnbDev = enbDevs.Get (0)->GetObject<LteEnbNetDevice> ();
  Ptr<LteEnbPhy> enbPhy = lteEnbDev->GetPhy ();
  enbPhy->SetAttribute ("TxPower", DoubleValue (46.0));
  enbPhy->SetAttribute ("NoiseFigure", DoubleValue (5.0));
  Ptr<MobilityModel> mmenb = enbNodes.Get (0)->GetObject<MobilityModel> ();
  mmenb->SetPosition (Vector (0.0, 0.0, 30.0));

  // Set UE's position and power
  Ptr<MobilityModel> mmue = ueNodes.Get (0)->GetObject<MobilityModel> ();
  mmue->SetPosition (Vector (m_dist, 0.0, 1.0));
  Ptr<LteUeNetDevice> lteUeDev = ueDevs.Get (0)->GetObject<LteUeNetDevice> ();
  Ptr<LteUePhy> uePhy = lteUeDev->GetPhy ();
  uePhy->SetAttribute ("TxPower", DoubleValue (23.0));
  uePhy->SetAttribute ("NoiseFigure", DoubleValue (9.0));
  
  // need to allow for RRC connection establishment + SRS before enabling traces
  lteHelper->EnableRlcTraces ();
  lteHelper->EnableMacTraces ();
  double simulationTime = 0.6; 
  double tolerance = 0.1;
  
  uint8_t rnti = 1;
  Ptr<LteEnbNetDevice> enbNetDev = enbDevs.Get (0)->GetObject<LteEnbNetDevice> ();
  
  PointerValue ptrval;
  enbNetDev->GetCcMap()[0]->GetAttribute ("FfMacScheduler", ptrval);
  Ptr<PfFfMacScheduler> pfsched;
  Ptr<RrFfMacScheduler> rrsched;
  if (m_schedulerType.compare ("ns3::RrFfMacScheduler") == 0)
    {
      rrsched = ptrval.Get<RrFfMacScheduler> ();
      if (rrsched == 0)
        {
          NS_FATAL_ERROR ("No RR Scheduler available");
        }
      Simulator::Schedule (Seconds (0.2), &RrFfMacScheduler::TransmissionModeConfigurationUpdate, rrsched, rnti, 1);
      Simulator::Schedule (Seconds (0.4), &RrFfMacScheduler::TransmissionModeConfigurationUpdate, rrsched, rnti, 2);
    }
  else if (m_schedulerType.compare ("ns3::PfFfMacScheduler") == 0)
    {
      pfsched = ptrval.Get<PfFfMacScheduler> ();
      if (pfsched == 0)
        {
          NS_FATAL_ERROR ("No Pf Scheduler available");
        }
      
      Simulator::Schedule (Seconds (0.2), &PfFfMacScheduler::TransmissionModeConfigurationUpdate, pfsched, rnti, 1);
      Simulator::Schedule (Seconds (0.4), &PfFfMacScheduler::TransmissionModeConfigurationUpdate, pfsched, rnti, 2);
    }
  else
    {
      NS_FATAL_ERROR ("Scheduler not supported by this test");
    }
    
  
  Ptr<RadioBearerStatsCalculator> rlcStats = lteHelper->GetRlcStats ();
  rlcStats->SetAttribute ("EpochDuration", TimeValue (Seconds (0.1)));

  NS_LOG_INFO (m_schedulerType << " MIMO test:");
  double sampleTime = 0.199999; // at 0.2 RlcStats are reset
  for (uint8_t j = 0; j < m_estThrDl.size (); j ++)
    {
      NS_LOG_INFO ("\t test with user at distance " << m_dist << " time " << sampleTime);
      // get the imsi
      uint64_t imsi = ueDevs.Get (0)->GetObject<LteUeNetDevice> ()->GetImsi ();
      uint8_t lcId = 3;
      Time t = Seconds (sampleTime);
      Simulator::Schedule(t, &LenaMimoTestCase::GetRlcBufferSample, this, rlcStats, imsi, lcId);
      sampleTime += 0.2;
    }
  Simulator::Stop (Seconds (simulationTime));
  Simulator::Run ();
  Simulator::Destroy ();

  NS_LOG_INFO ("Check consistency");
    for (uint8_t i = 0; i < m_estThrDl.size (); i++)
      {
        NS_LOG_INFO ("interval " << i + 1 << ": bytes rxed " << (double)m_dlDataRxed.at (i) << " ref " << m_estThrDl.at (i));
        NS_TEST_ASSERT_MSG_EQ_TOL ((double)m_dlDataRxed.at (i) , m_estThrDl.at (i), m_estThrDl.at (i) * tolerance, " Unfair Throughput!");
      }

}


void
LenaMimoTestCase::GetRlcBufferSample (Ptr<RadioBearerStatsCalculator> rlcStats, uint64_t imsi, uint8_t lcId)
{
  m_dlDataRxed.push_back (rlcStats->GetDlRxData (imsi, lcId));
  NS_LOG_INFO (Simulator::Now () << "\t get bytes " << m_dlDataRxed.at (m_dlDataRxed.size () - 1));
}
