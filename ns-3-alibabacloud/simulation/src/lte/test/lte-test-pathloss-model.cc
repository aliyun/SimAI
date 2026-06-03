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

#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/spectrum-test.h"
#include "ns3/lte-phy-tag.h"
#include "ns3/lte-chunk-processor.h"
#include <ns3/hybrid-buildings-propagation-loss-model.h>
#include <ns3/node-container.h>
#include <ns3/mobility-helper.h>
#include <ns3/buildings-helper.h>
#include <ns3/lte-helper.h>
#include <ns3/single-model-spectrum-channel.h>
#include "ns3/string.h"
#include "ns3/double.h"
#include <ns3/boolean.h>
#include <ns3/building.h>
#include <ns3/enum.h>
#include <ns3/net-device-container.h>
#include <ns3/lte-ue-net-device.h>
#include <ns3/lte-enb-net-device.h>
#include <ns3/lte-ue-rrc.h>
#include <ns3/lte-enb-phy.h>
#include <ns3/lte-ue-phy.h>
#include "lte-test-ue-phy.h"
#include "lte-test-pathloss-model.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LtePathlossModelTest");

/**
 * Test 1.1 Pathloss compound test
 */

/**
 * This TestSuite tests the BuildingPathlossModel by reproducing
 * several communication scenarios 
 */


void
LteTestPathlossDlSchedCallback (LtePathlossModelSystemTestCase *testcase, std::string path,
		                        DlSchedulingCallbackInfo dlInfo)
{
  testcase->DlScheduling (dlInfo);
}



LtePathlossModelTestSuite::LtePathlossModelTestSuite ()
  : TestSuite ("lte-pathloss-model", SYSTEM)
{
  // LogLevel logLevel = (LogLevel)(LOG_PREFIX_FUNC | LOG_PREFIX_TIME | LOG_LEVEL_ALL);
  // LogComponentEnable ("LteHelper", logLevel);
  // LogComponentEnable ("LtePathlossModelTest", logLevel);
  // LogComponentEnable ("BuildingsPropagationLossModel", logLevel);
  // LogComponentEnable ("LteInterference", logLevel);
  // LogComponentEnable ("LteSpectrumValueHelper", logLevel);
  // LogComponentEnable ("LteEnbNetDevice", logLevel);

  struct SnrEfficiencyMcs
  {
    double  snrDb;
    double  efficiency;
    int  mcsIndex;
  };

  /**
  * Test vectors: SNRDB, Spectral Efficiency, MCS index
  * From XXX
  */
  SnrEfficiencyMcs snrEfficiencyMcs[] = {
    { -5.00000,  0.08024,        -1},
    { -4.00000,  0.10030,        -1},
    { -3.00000,  0.12518,        -1},
    { -2.00000,  0.15589,        0},
    { -1.00000,  0.19365,        0},
    { 0.00000,   0.23983,        2},
    { 1.00000,   0.29593,        2},
    { 2.00000,   0.36360,        2},
    { 3.00000,   0.44451,        4},
    { 4.00000,   0.54031,        4},
    { 5.00000,   0.65251,        6},
    { 6.00000,   0.78240,        6},
    { 7.00000,   0.93086,        8},
    { 8.00000,   1.09835,        8},
    { 9.00000,   1.28485,        10},
    { 10.00000,  1.48981,        12},
    { 11.00000,  1.71229,        12},
    { 12.00000,  1.95096,        14},
    { 13.00000,  2.20429,        14},
    { 14.00000,  2.47062,        16},
    { 15.00000,  2.74826,        18},
    { 16.00000,  3.03560,        18},
    { 17.00000,  3.33115,        20},
    { 18.00000,  3.63355,        20},
    { 19.00000,  3.94163,        22},
    { 20.00000,  4.25439,        22},
    { 21.00000,  4.57095,        24},
    { 22.00000,  4.89060,        24},
    { 23.00000,  5.21276,        26},
    { 24.00000,  5.53693,        26},
    { 25.00000,  5.86271,        28},
    { 26.00000,  6.18980,        28},
    { 27.00000,  6.51792,        28},
    { 28.00000,  6.84687,        28},
    { 29.00000,  7.17649,        28},
    { 30.00000,  7.50663,        28},
  };


  double txPowerDbm = 30; // default eNB TX power over whole bandwidth
  double txPowerLin = std::pow (10, (txPowerDbm - 30)/10);
  double ktDbm = -174;    // reference LTE noise PSD
  double noisePowerDbm = ktDbm + 10 * std::log10 (25 * 180000); // corresponds to kT*bandwidth in linear units
  double receiverNoiseFigureDb = 9.0; // default UE noise figure
  double noiseLin = std::pow (10, (noisePowerDbm-30+receiverNoiseFigureDb)/10);

  // reference values obtained with the octave script src/lte/test/reference/lte_pathloss.m

  double loss[] = {81.062444, 134.078605, 144.259958};
  double dist[] = {100.0, 500.0, 1500};

  int numOfTests = sizeof (loss) / sizeof (double);
  for ( int i = 0 ; i < numOfTests; i++ )
  {
    //     double lossDb = txPowerDbm - snrEfficiencyMcs[i].snrDb - noisePowerDbm - receiverNoiseFigureDb;
    double sinrLin = (txPowerLin/(pow(10, loss[i]/10))) / noiseLin;
    //     double sinrDb = txPowerDbm- noisePowerDbm - receiverNoiseFigureDb - loss[i];
    double sinrDb = 10 * std::log10 (sinrLin);
    NS_LOG_INFO (" Ptx " << txPowerDbm << " Pn " << noisePowerDbm << " Fn " << receiverNoiseFigureDb << " Pl " << loss[i] << " dist " << dist[i]);

    int mcs = -1;
    int numSnrEfficiencyMcsEntries = sizeof (snrEfficiencyMcs) / sizeof (SnrEfficiencyMcs);
    for (int j = 0; j < numSnrEfficiencyMcsEntries && snrEfficiencyMcs[j].snrDb < sinrDb; ++j)
      {
        mcs = snrEfficiencyMcs[j].mcsIndex;
      }

    std::ostringstream name;
    name << " snr= " << sinrDb << " dB, "
         << " mcs= " << snrEfficiencyMcs[i].mcsIndex;
    AddTestCase (new LtePathlossModelSystemTestCase (name.str (),  sinrDb, dist[i], mcs), TestCase::QUICK);
  }




  
}

static LtePathlossModelTestSuite ltePathlossModelTestSuite;




LtePathlossModelSystemTestCase::LtePathlossModelSystemTestCase (std::string name, double snrDb, double dist, uint16_t mcsIndex)
: TestCase (name),
m_snrDb (snrDb),
m_distance (dist),
m_mcsIndex (mcsIndex)
{
  std::ostringstream sstream1, sstream2;
  sstream1 << " snr=" << snrDb 
  << " mcs=" << mcsIndex << " distance=" << dist;
  
  NS_LOG_INFO ("Creating LtePathlossModelSystemTestCase: " + sstream1.str ());
}

LtePathlossModelSystemTestCase::~LtePathlossModelSystemTestCase ()
{
}

void
LtePathlossModelSystemTestCase::DoRun (void)
{
  Config::SetDefault ("ns3::MacStatsCalculator::DlOutputFilename", StringValue (CreateTempDirFilename ("DlMacStats.txt")));
  Config::SetDefault ("ns3::MacStatsCalculator::UlOutputFilename", StringValue (CreateTempDirFilename ("UlMacStats.txt")));
  Config::SetDefault ("ns3::RadioBearerStatsCalculator::DlRlcOutputFilename", StringValue (CreateTempDirFilename ("DlRlcStats.txt")));
  Config::SetDefault ("ns3::RadioBearerStatsCalculator::UlRlcOutputFilename", StringValue (CreateTempDirFilename ("UlRlcStats.txt")));
  /**
  * Simulation Topology
  */
  //Disable Uplink Power Control
  Config::SetDefault ("ns3::LteUePhy::EnableUplinkPowerControl", BooleanValue (false));

  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  //   lteHelper->EnableLogComponents ();
  lteHelper->SetAttribute ("PathlossModel", StringValue ("ns3::HybridBuildingsPropagationLossModel"));

  // set frequency. This is important because it changes the behavior of the path loss model
  lteHelper->SetEnbDeviceAttribute ("DlEarfcn", UintegerValue (200));
  lteHelper->SetEnbDeviceAttribute ("UlEarfcn", UintegerValue (18200));
  lteHelper->SetUeDeviceAttribute ("DlEarfcn", UintegerValue (200));

  // remove shadowing component
  lteHelper->SetPathlossModelAttribute ("ShadowSigmaOutdoor", DoubleValue (0.0));
  lteHelper->SetPathlossModelAttribute ("ShadowSigmaIndoor", DoubleValue (0.0));
  lteHelper->SetPathlossModelAttribute ("ShadowSigmaExtWalls", DoubleValue (0.0));
  
  // Create Nodes: eNodeB and UE
  NodeContainer enbNodes;
  NodeContainer ueNodes;
  enbNodes.Create (1);
  ueNodes.Create (1);
  NodeContainer allNodes = NodeContainer ( enbNodes, ueNodes );
  
  // Install Mobility Model
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (allNodes);
  BuildingsHelper::Install (allNodes);

  
  // Create Devices and install them in the Nodes (eNB and UE)
  NetDeviceContainer enbDevs;
  NetDeviceContainer ueDevs;
  lteHelper->SetSchedulerType ("ns3::RrFfMacScheduler");
  enbDevs = lteHelper->InstallEnbDevice (enbNodes);
  ueDevs = lteHelper->InstallUeDevice (ueNodes);
  
  Ptr<MobilityModel> mm_enb = enbNodes.Get (0)->GetObject<MobilityModel> ();
  mm_enb->SetPosition (Vector (0.0, 0.0, 30.0));
  Ptr<MobilityModel> mm_ue = ueNodes.Get (0)->GetObject<MobilityModel> ();
  mm_ue->SetPosition (Vector (m_distance, 0.0, 1.0));
  
  Ptr<LteEnbNetDevice> lteEnbDev = enbDevs.Get (0)->GetObject<LteEnbNetDevice> ();
  Ptr<LteEnbPhy> enbPhy = lteEnbDev->GetPhy ();
  enbPhy->SetAttribute ("TxPower", DoubleValue (30.0));
  enbPhy->SetAttribute ("NoiseFigure", DoubleValue (5.0));
  
  Ptr<LteUeNetDevice> lteUeDev = ueDevs.Get (0)->GetObject<LteUeNetDevice> ();
  Ptr<LteUePhy> uePhy = lteUeDev->GetPhy ();
  uePhy->SetAttribute ("TxPower", DoubleValue (23.0));
  uePhy->SetAttribute ("NoiseFigure", DoubleValue (9.0));
  
  
  // Attach a UE to a eNB
  lteHelper->Attach (ueDevs, enbDevs.Get (0));

  // Activate an EPS bearer
  enum EpsBearer::Qci q = EpsBearer::GBR_CONV_VOICE;
  EpsBearer bearer (q);
  lteHelper->ActivateDataRadioBearer (ueDevs, bearer);
  
  // Use testing chunk processor in the PHY layer
  // It will be used to test that the SNR is as intended
  //Ptr<LtePhy> uePhy = ueDevs.Get (0)->GetObject<LteUeNetDevice> ()->GetPhy ()->GetObject<LtePhy> ();
  Ptr<LteChunkProcessor> testSinr = Create<LteChunkProcessor> ();
  LteSpectrumValueCatcher sinrCatcher;
  testSinr->AddCallback (MakeCallback (&LteSpectrumValueCatcher::ReportValue, &sinrCatcher));
  uePhy->GetDownlinkSpectrumPhy ()->AddCtrlSinrChunkProcessor (testSinr);
   
//   Config::Connect ("/NodeList/0/DeviceList/0/LteEnbMac/DlScheduling",
//                    MakeBoundCallback (&LteTestPathlossDlSchedCallback, this));

  lteHelper->EnableMacTraces ();
  lteHelper->EnableRlcTraces ();
                   
  Simulator::Stop (Seconds (0.035));
  Simulator::Run ();
  
  double calculatedSinrDb = 10.0 * std::log10 (sinrCatcher.GetValue ()->operator[] (0));
  NS_LOG_INFO ("Distance " << m_distance << " Calculated SINR " << calculatedSinrDb << " ref " << m_snrDb);
  Simulator::Destroy ();
  NS_TEST_ASSERT_MSG_EQ_TOL (calculatedSinrDb, m_snrDb, 0.001, "Wrong SINR !");
}


void
LtePathlossModelSystemTestCase::DlScheduling (DlSchedulingCallbackInfo dlInfo)
{
  static bool firstTime = true;
  
  if ( firstTime )
  {
    firstTime = false;
    NS_LOG_INFO ("SNR\tRef_MCS\tCalc_MCS");
  }
  
  
  // need to allow for RRC connection establishment + SRS transmission
  if (Simulator::Now () > MilliSeconds (21))
  {
    NS_LOG_INFO (m_snrDb << "\t" << m_mcsIndex << "\t" << (uint16_t)dlInfo.mcsTb1);
    
    NS_TEST_ASSERT_MSG_EQ ((uint16_t)dlInfo.mcsTb1, m_mcsIndex, "Wrong MCS index");
  }
}
