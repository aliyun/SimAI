/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Biljana Bojovic <bbojovic@cttc.es>
 *
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
#include "ns3/radio-bearer-stats-calculator.h"
#include <ns3/constant-position-mobility-model.h>
#include "ns3/ff-mac-scheduler.h"
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
#include <ns3/config-store-module.h>
#include <errno.h>

#include "lte-test-carrier-aggregation.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TestCarrierAggregation");

bool CarrierAggregationTestCase::s_writeResults = false; // set to true to write response vectors
const std::string dlResultsFileName = "carrier_aggregation_results_dl.txt";
const std::string ulResultsFileName = "carrier_aggregation_results_ul.txt";


void
LteTestDlSchedulingCallback (CarrierAggregationTestCase *testcase, std::string path, DlSchedulingCallbackInfo dlInfo)
{
  testcase->DlScheduling (dlInfo);
}

void
LteTestUlSchedulingCallback (CarrierAggregationTestCase *testcase, std::string path,
                             uint32_t frameNo, uint32_t subframeNo, uint16_t rnti,
                             uint8_t mcs, uint16_t sizeTb, uint8_t ccId)
{
  testcase->UlScheduling (frameNo, subframeNo, rnti, mcs, sizeTb, ccId);
}


TestCarrierAggregationSuite::TestCarrierAggregationSuite ()
  : TestSuite ("lte-carrier-aggregation", SYSTEM)
{
  NS_LOG_INFO ("creating CarrierAggregationTestCase");

  if (CarrierAggregationTestCase::s_writeResults) // write result vectors to file
    {
      std::cout<<"\n Running TestCarrierAggregationSuite with activated option to write results to files."
          "Dl results will be written to "<<dlResultsFileName<<" and uplink results to "<<ulResultsFileName<<std::endl;

      bool abort = false;

      if (fopen(dlResultsFileName.c_str(), "r"))
        {
          std::cout<<"\nResults file "<<dlResultsFileName<<" already exists. Move it out of the way or specify another downlink results file name."<<std::endl;
          abort = true;
        }
      if (fopen(ulResultsFileName.c_str(), "r"))
        {
          std::cout<<"\nResults file "<<ulResultsFileName<<" already exists. Move it out of the way or specify another uplink results file name."<<std::endl;
          abort = true;
        }

      if (abort)
        return;

      AddTestCase (new CarrierAggregationTestCase (1,0, 100, 100, 1), TestCase::QUICK);
      AddTestCase (new CarrierAggregationTestCase (3,0, 100, 100, 1), TestCase::QUICK);
      AddTestCase (new CarrierAggregationTestCase (6,0, 100, 100, 1), TestCase::QUICK);
      AddTestCase (new CarrierAggregationTestCase (9,0, 100, 100, 1), TestCase::QUICK);
      AddTestCase (new CarrierAggregationTestCase (12,0, 100, 100, 1), TestCase::QUICK);
      AddTestCase (new CarrierAggregationTestCase (15,0, 100, 100, 1), TestCase::QUICK);

      AddTestCase (new CarrierAggregationTestCase (1,0, 100, 100, 2), TestCase::QUICK);
      AddTestCase (new CarrierAggregationTestCase (3,0, 100, 100, 2), TestCase::QUICK);
      AddTestCase (new CarrierAggregationTestCase (6,0, 100, 100, 2), TestCase::QUICK);
      AddTestCase (new CarrierAggregationTestCase (9,0, 100, 100, 2), TestCase::QUICK);
      AddTestCase (new CarrierAggregationTestCase (12,0, 100, 100, 2), TestCase::QUICK);
      AddTestCase (new CarrierAggregationTestCase (15,0, 100, 100, 2), TestCase::QUICK);

      AddTestCase (new CarrierAggregationTestCase (1,0, 100, 100, 3), TestCase::QUICK);
      AddTestCase (new CarrierAggregationTestCase (3,0, 100, 100, 3), TestCase::QUICK);
      AddTestCase (new CarrierAggregationTestCase (6,0, 100, 100, 3), TestCase::QUICK);
      AddTestCase (new CarrierAggregationTestCase (9,0, 100, 100, 3), TestCase::QUICK);
      AddTestCase (new CarrierAggregationTestCase (12,0, 100, 100, 3), TestCase::QUICK);
      AddTestCase (new CarrierAggregationTestCase (15,0, 100, 100, 3), TestCase::QUICK);
    }
  else
    {
      // bandwidth is 25 and there are 2 carriers
      AddTestCase (new CarrierAggregationTestCase (1,0, 25, 25, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (3,0, 25, 25, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (6,0, 25, 25, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (9,0, 25, 25, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (12,0, 25, 25, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (15,0, 25, 25, 2), TestCase::EXTENSIVE);

      AddTestCase (new CarrierAggregationTestCase (1,4800, 25, 25, 2), TestCase::QUICK);
      AddTestCase (new CarrierAggregationTestCase (3,4800, 25, 25, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (6,4800, 25, 25, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (9,4800, 25, 25, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (12,4800, 25, 25, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (15,4800, 25, 25, 2), TestCase::EXTENSIVE);

      AddTestCase (new CarrierAggregationTestCase (1,6000, 25, 25, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (3,6000, 25, 25, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (6,6000, 25, 25, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (9,6000, 25, 25, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (12,6000, 25, 25, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (15,6000, 25, 25, 2), TestCase::EXTENSIVE);

      AddTestCase (new CarrierAggregationTestCase (1,20000, 25, 25, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (3,20000, 25, 25, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (6,20000, 25, 25, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (9,20000, 25, 25, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (12,20000, 25, 25, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (15,20000, 25, 25, 2), TestCase::QUICK);


      // bandwidth is 25 and there are 3 carriers
      AddTestCase (new CarrierAggregationTestCase (1,0, 25, 25, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (3,0, 25, 25, 3), TestCase::QUICK);
      AddTestCase (new CarrierAggregationTestCase (6,0, 25, 25, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (9,0, 25, 25, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (12,0, 25, 25, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (15,0, 25, 25, 3), TestCase::EXTENSIVE);

      AddTestCase (new CarrierAggregationTestCase (1,4800, 25, 25, 3), TestCase::QUICK);
      AddTestCase (new CarrierAggregationTestCase (3,4800, 25, 25, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (6,4800, 25, 25, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (9,4800, 25, 25, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (12,4800, 25, 25, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (15,4800, 25, 25, 3), TestCase::EXTENSIVE);

      AddTestCase (new CarrierAggregationTestCase (1,6000, 25, 25, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (3,6000, 25, 25, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (6,6000, 25, 25, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (9,6000, 25, 25, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (12,6000, 25, 25, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (15,6000, 25, 25, 3), TestCase::EXTENSIVE);

      AddTestCase (new CarrierAggregationTestCase (1,20000, 25, 25, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (3,20000, 25, 25, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (6,20000, 25, 25, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (9,20000, 25, 25, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (12,20000, 25, 25, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (15,20000, 25, 25, 3), TestCase::EXTENSIVE);


      // bandwidth = 6 RB and there are 3 carriers
      AddTestCase (new CarrierAggregationTestCase (1,0, 6, 6, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (3,0, 6, 6, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (6,0, 6, 6, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (9,0, 6, 6, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (12,0, 6, 6, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (15,0, 6, 6, 3), TestCase::EXTENSIVE);

      AddTestCase (new CarrierAggregationTestCase (1,4800, 6, 6, 3), TestCase::QUICK);
      AddTestCase (new CarrierAggregationTestCase (3,4800, 6, 6, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (6,4800, 6, 6, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (9,4800, 6, 6, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (12,4800, 6, 6, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (15,4800, 6, 6, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (1,6000, 6, 6, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (3,6000, 6, 6, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (6,6000, 6, 6, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (9,6000, 6, 6, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (12,6000, 6, 6, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (15,6000, 6, 6, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (1,20000, 6, 6, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (3,20000, 6, 6, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (6,20000, 6, 6, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (9,20000, 6, 6, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (12,20000, 6, 6, 3), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (15,20000, 6, 6, 3), TestCase::EXTENSIVE);

      // bandwidth = 6 RB and there are 2 carriers
      AddTestCase (new CarrierAggregationTestCase (1,0, 6, 6, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (3,0, 6, 6, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (6,0, 6, 6, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (9,0, 6, 6, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (12,0, 6, 6, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (15,0, 6, 6, 2), TestCase::EXTENSIVE);

      AddTestCase (new CarrierAggregationTestCase (1,4800, 6, 6, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (3,4800, 6, 6, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (6,4800, 6, 6, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (9,4800, 6, 6, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (12,4800, 6, 6, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (15,4800, 6, 6, 2), TestCase::EXTENSIVE);

      AddTestCase (new CarrierAggregationTestCase (1,6000, 6, 6, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (3,6000, 6, 6, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (6,6000, 6, 6, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (9,6000, 6, 6, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (12,6000, 6, 6, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (15,6000, 6, 6, 2), TestCase::QUICK);

      AddTestCase (new CarrierAggregationTestCase (1,20000, 6, 6, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (3,20000, 6, 6, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (6,20000, 6, 6, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (9,20000, 6, 6, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (12,20000, 6, 6, 2), TestCase::EXTENSIVE);
      AddTestCase (new CarrierAggregationTestCase (15,20000, 6, 6, 2), TestCase::EXTENSIVE);
    }
}


static TestCarrierAggregationSuite lenaTestRrFfMacSchedulerSuite;

std::string 
CarrierAggregationTestCase::BuildNameString (uint16_t nUser, uint16_t dist, uint32_t dlBandwidth, uint32_t ulBandwidth, uint32_t numberOfComponentCarriers)
{
  std::ostringstream oss;
  oss << nUser << " UEs, distance " << dist << " m"<< " dlBandwidth "<< dlBandwidth <<" ulBandwidth "<< ulBandwidth <<" number of carriers "<<numberOfComponentCarriers;
  return oss.str ();
}

CarrierAggregationTestCase::CarrierAggregationTestCase (uint16_t nUser, uint16_t dist, uint32_t dlbandwidth, uint32_t ulBandwidth, uint32_t numberOfComponentCarriers)
  : TestCase (BuildNameString (nUser, dist, dlbandwidth, ulBandwidth, numberOfComponentCarriers)),
    m_nUser (nUser),
    m_dist (dist),
    m_dlBandwidth (dlbandwidth),
    m_ulBandwidth (ulBandwidth),
    m_numberOfComponentCarriers(numberOfComponentCarriers)
{
   m_dlThroughput = 0;
   m_statsDuration = 0.6;
}

CarrierAggregationTestCase::~CarrierAggregationTestCase ()
{
}

void
CarrierAggregationTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this << m_nUser << m_dist << m_dlBandwidth << m_ulBandwidth << m_numberOfComponentCarriers);

  Config::SetDefault ("ns3::LteEnbNetDevice::DlEarfcn", UintegerValue (100));
  Config::SetDefault ("ns3::LteEnbNetDevice::UlEarfcn", UintegerValue (100 + 18000));
  Config::SetDefault ("ns3::LteEnbNetDevice::DlBandwidth", UintegerValue (m_dlBandwidth));
  Config::SetDefault ("ns3::LteEnbNetDevice::UlBandwidth", UintegerValue (m_ulBandwidth));
  Config::SetDefault ("ns3::LteUeNetDevice::DlEarfcn", UintegerValue (100));

  Config::SetDefault ("ns3::LteHelper::UseCa", BooleanValue (true));
  Config::SetDefault ("ns3::LteHelper::NumberOfComponentCarriers", UintegerValue (m_numberOfComponentCarriers));
  Config::SetDefault ("ns3::LteHelper::EnbComponentCarrierManager", StringValue ("ns3::RrComponentCarrierManager"));
  Config::SetDefault ("ns3::LteSpectrumPhy::CtrlErrorModelEnabled", BooleanValue (false));
  Config::SetDefault ("ns3::LteSpectrumPhy::DataErrorModelEnabled", BooleanValue (false));
  Config::SetDefault ("ns3::LteHelper::UseIdealRrc", BooleanValue (true));

  Config::SetDefault ("ns3::MacStatsCalculator::DlOutputFilename", StringValue (CreateTempDirFilename ("DlMacStats.txt")));
  Config::SetDefault ("ns3::MacStatsCalculator::UlOutputFilename", StringValue (CreateTempDirFilename ("UlMacStats.txt")));
  Config::SetDefault ("ns3::RadioBearerStatsCalculator::DlRlcOutputFilename", StringValue (CreateTempDirFilename ("DlRlcStats.txt")));
  Config::SetDefault ("ns3::RadioBearerStatsCalculator::UlRlcOutputFilename", StringValue (CreateTempDirFilename ("UlRlcStats.txt")));
  Config::SetDefault ("ns3::RadioBearerStatsCalculator::DlPdcpOutputFilename", StringValue (CreateTempDirFilename ("DlPdcpStats.txt")));
  Config::SetDefault ("ns3::RadioBearerStatsCalculator::UlPdcpOutputFilename", StringValue (CreateTempDirFilename ("UlPdcpStats.txt")));
  Config::SetDefault ("ns3::PhyStatsCalculator::DlRsrpSinrFilename", StringValue (CreateTempDirFilename ("DlRsrpSinrStats.txt")));
  Config::SetDefault ("ns3::PhyStatsCalculator::UlSinrFilename", StringValue (CreateTempDirFilename ("UlSinrStats.txt")));
  Config::SetDefault ("ns3::PhyStatsCalculator::UlInterferenceFilename", StringValue (CreateTempDirFilename ("UlInterferenceStats.txt")));
  Config::SetDefault ("ns3::PhyRxStatsCalculator::DlRxOutputFilename", StringValue (CreateTempDirFilename ("DlRxPhyStats.txt")));
  Config::SetDefault ("ns3::PhyRxStatsCalculator::UlRxOutputFilename", StringValue (CreateTempDirFilename ("UlRxPhyStats.txt")));
  Config::SetDefault ("ns3::PhyTxStatsCalculator::DlTxOutputFilename", StringValue (CreateTempDirFilename ("DlTxPhyStats.txt")));
  Config::SetDefault ("ns3::PhyTxStatsCalculator::UlTxOutputFilename", StringValue (CreateTempDirFilename ("UlTxPhyStats.txt")));

  /**
   * Initialize Simulation Scenario: 1 eNB and m_nUser UEs
   */

  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  
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
  lteHelper->SetSchedulerType ("ns3::PfFfMacScheduler");
  lteHelper->SetSchedulerAttribute ("UlCqiFilter", EnumValue (FfMacScheduler::SRS_UL_CQI));
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


  double statsStartTime = 0.300; // need to allow for RRC connection establishment + SRS

  Simulator::Stop (Seconds (statsStartTime + m_statsDuration - 0.0001));

  Config::Connect ("/NodeList/*/DeviceList/*/ComponentCarrierMap/*/LteEnbMac/DlScheduling",
                    MakeBoundCallback (&LteTestDlSchedulingCallback, this));

  Config::Connect ("/NodeList/*/DeviceList/*/ComponentCarrierMap/*/LteEnbMac/UlScheduling",
                    MakeBoundCallback (&LteTestUlSchedulingCallback, this));

  lteHelper->EnableTraces();

  Simulator::Run ();

  /**
   * Check that the assignment is done in a RR fashion
   */
  NS_LOG_INFO ("DL - Test with " << m_nUser << " user(s) at distance " << m_dist);
  std::vector <uint64_t> dlDataRxed;

  // tolerance increases with the number of users because the lc 0 and lc 1 will go always over primary carrier, so as the number of users increases the difference between primary and secondary
  //carrier will increase

  bool testDownlinkShare = true;

  for (std::map <uint8_t, uint32_t>::iterator itDownlink = m_ccDownlinkTraffic.begin(); itDownlink!=m_ccDownlinkTraffic.end(); itDownlink++)
   {

      if (itDownlink == m_ccDownlinkTraffic.begin())
        {
           NS_LOG_INFO ("Downlink traffic per carrier:"<<itDownlink->second);
        }
      else
        {
          if (itDownlink->second != m_ccDownlinkTraffic.begin()->second)
           {
             testDownlinkShare = false;
             break;
           }
        }
   }

  bool testUplinkShare = true;

  for (std::map <uint8_t, uint32_t>::iterator itUplink = m_ccUplinkTraffic.begin(); itUplink!=m_ccUplinkTraffic.end(); itUplink++)
   {
      if (itUplink == m_ccUplinkTraffic.begin())
       {
          NS_LOG_INFO ("Uplink traffic per carrier:"<<itUplink->second);
       }
      else
        {
          if (itUplink->second != m_ccUplinkTraffic.begin()->second)
            {
              testUplinkShare = false;
              break;
            }
        }
   }

  NS_TEST_ASSERT_MSG_EQ (m_ccUplinkTraffic.size(), m_numberOfComponentCarriers, "Number of carriers in uplink does not correspond to number of carriers being configured in test.");
  NS_TEST_ASSERT_MSG_EQ (m_ccDownlinkTraffic.size(), m_numberOfComponentCarriers, "Number of carriers in downlink does not correspond to number of carriers being configured in test.");

  NS_TEST_ASSERT_MSG_EQ (testDownlinkShare, true, " Downlink traffic not split equally between carriers!");
  NS_TEST_ASSERT_MSG_EQ (testUplinkShare, true , " Uplink traffic not split equally between carriers");

  if (s_writeResults)
     WriteResultToFile ();

  Simulator::Destroy ();
}

void
CarrierAggregationTestCase::DlScheduling (DlSchedulingCallbackInfo dlInfo)
{
  //NS_LOG_FUNCTION (dlInfo.frameNo << dlInfo.subframeNo << dlInfo.rnti << (uint32_t) dlInfo.mcsTb1 << dlInfo.sizeTb1 << (uint32_t) dlInfo.mcsTb2 << dlInfo.sizeTb2<<(uint16_t)dlInfo.componentCarrierId);
  // need to allow for RRC connection establishment + CQI feedback reception + persistent data transmission
  if (Simulator::Now () > MilliSeconds (300))
    {
      if (m_ccDownlinkTraffic.find(dlInfo.componentCarrierId) == m_ccDownlinkTraffic.end())
        {
          m_ccDownlinkTraffic.insert (std::pair<uint8_t, uint32_t> (dlInfo.componentCarrierId, dlInfo.sizeTb1 + dlInfo.sizeTb2));
        }
      else
        {
          m_ccDownlinkTraffic[dlInfo.componentCarrierId]+= (dlInfo.sizeTb1 + dlInfo.sizeTb2);
        }

      m_dlThroughput += dlInfo.sizeTb1 + dlInfo.sizeTb2;
    }
}

void
CarrierAggregationTestCase::UlScheduling (uint32_t frameNo, uint32_t subframeNo, uint16_t rnti,
                                       uint8_t mcs, uint16_t sizeTb, uint8_t componentCarrierId)
{
  //NS_LOG_FUNCTION (frameNo << subframeNo << rnti << (uint32_t) mcs << sizeTb);
  // need to allow for RRC connection establishment + SRS transmission
  if (Simulator::Now () > MilliSeconds (300))
    {
      if (m_ccUplinkTraffic.find(componentCarrierId) == m_ccUplinkTraffic.end())
        {
          m_ccUplinkTraffic.insert (std::pair<uint8_t, uint32_t> (componentCarrierId, sizeTb));
        }
      else
        {
          m_ccUplinkTraffic[componentCarrierId] += sizeTb;
        }

      m_ulThroughput += sizeTb;
    }
}

void
CarrierAggregationTestCase::WriteResultToFile ()
{
  std::ofstream dlOutFile;
  dlOutFile.open (dlResultsFileName.c_str(), std::ofstream::out | std::ofstream::app);
  dlOutFile.setf (std::ios_base::fixed);

  if (!dlOutFile.is_open ())
    {
      NS_LOG_ERROR ("Can't open file "<<dlResultsFileName);
      return;
    }
  dlOutFile << m_nUser <<" "<<m_numberOfComponentCarriers <<" "<< ((m_dlThroughput*8)/m_statsDuration)/m_nUser<<std::endl;
  dlOutFile.close ();

  std::ofstream ulOutFile;
  ulOutFile.open (ulResultsFileName.c_str(), std::ofstream::out | std::ofstream::app);
  ulOutFile.setf (std::ios_base::fixed);

  if (!ulOutFile.is_open ())
    {
      NS_LOG_ERROR ("Can't open file "<<ulResultsFileName);
      return;
    }
  ulOutFile << m_nUser <<" "<<m_numberOfComponentCarriers <<" "<< ((m_ulThroughput*8)/m_statsDuration)/m_nUser<<std::endl;
  ulOutFile.close ();
}

