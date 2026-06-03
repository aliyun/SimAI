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
 * Author: Marco Miozzo <marco.miozzo@cttc.es>,
 *         Nicola Baldo <nbaldo@cttc.es>
 *         Dizhi Zhou <dizhi.zhou@gmail.com>
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

#include "lte-test-tdbet-ff-mac-scheduler.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LenaTestTdBetFfMacScheduler");

LenaTestTdBetFfMacSchedulerSuite::LenaTestTdBetFfMacSchedulerSuite ()
  : TestSuite ("lte-tdbet-ff-mac-scheduler", SYSTEM)
{
  NS_LOG_INFO ("creating LenaTestTdBetFfMacSchedulerSuite");

  bool errorModel = false;

  //Test Case 1: AMC works in TDBET

  //Note: here the MCS is calculated by the wideband CQI

  // DOWNLINK- DISTANCE 0 -> MCS 28 -> Itbs 26 (from table 7.1.7.2.1-1 of 36.213)
  // 1 user -> 24 PRB at Itbs 26 -> 2196 -> 2196000 bytes/sec
  // 3 users -> 2196000 among 3 users -> 732000  bytes/sec
  // 6 users -> 2196000 among 6 users -> 366000  bytes/sec
  // 12 users -> 2196000 among 12 users -> 183000  bytes/sec
  // UPLINK- DISTANCE 0 -> MCS 28 -> Itbs 26 (from table 7.1.7.2.1-1 of 36.213)
  // 1 user -> 25 PRB at Itbs 26 -> 2292 -> 2292000 bytes/sec
  // 3 users -> 8 PRB at Itbs 26 -> 749 -> 749000 bytes/sec
  // 6 users -> 4 PRB at Itbs 26 -> 373 -> 373000 bytes/sec
  // after the patch enforcing min 3 PRBs per UE:
  // 12 users -> 3 PRB at Itbs 26 -> 277 bytes * 8/12 UE/TTI -> 184670 bytes/sec
  AddTestCase (new LenaTdBetFfMacSchedulerTestCase1 (1,0,2196000,2292000, errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaTdBetFfMacSchedulerTestCase1 (3,0,732000,749000, errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaTdBetFfMacSchedulerTestCase1 (6,0,366000,373000, errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaTdBetFfMacSchedulerTestCase1 (12,0,183000,184670, errorModel), TestCase::EXTENSIVE);

  // DOWNLINK - DISTANCE 4800 -> MCS 22 -> Itbs 20 (from table 7.1.7.2.1-1 of 36.213)
  // 1 user -> 24 PRB at Itbs 20 -> 1383 -> 1383000 bytes/sec
  // 3 users -> 1383000 among 3 users ->461000  bytes/sec
  // 6 users -> 1383000 among 6 users ->230500  bytes/sec
  // 12 users -> 1383000 among 12 users ->115250  bytes/sec 
  // UPLINK - DISTANCE 4800 -> MCS 14 -> Itbs 13 (from table 7.1.7.2.1-1 of 36.213)
  // 1 user -> 25 PRB at Itbs 13 -> 807 -> 807000 bytes/sec
  // 3 users -> 8 PRB at Itbs 13 -> 253 -> 253000 bytes/sec
  // 6 users -> 4 PRB at Itbs 13 -> 125 -> 125000 bytes/sec
  // after the patch enforcing min 3 PRBs per UE:
  // 12 users -> 3 PRB at Itbs 13 -> 93  bytes * 8/12 UE/TTI  -> 62000 bytes/sec
  AddTestCase (new LenaTdBetFfMacSchedulerTestCase1 (1,4800,1383000,807000, errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaTdBetFfMacSchedulerTestCase1 (3,4800,461000,253000, errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaTdBetFfMacSchedulerTestCase1 (6,4800,230500,125000, errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaTdBetFfMacSchedulerTestCase1 (12,4800,115250,62000, errorModel), TestCase::EXTENSIVE);

  // DOWNLINK - DISTANCE 6000 -> MCS 20 -> Itbs 18 (from table 7.1.7.2.1-1 of 36.213)
  // 1 user -> 24 PRB at Itbs 18 -> 1191 -> 1191000 byte/sec
  // 3 users -> 1191000 among 3 users ->397000  bytes/sec
  // 6 users -> 1191000 among 6 users ->198500  bytes/sec
  // 12 users -> 1191000 among 12 users ->99250  bytes/sec
  // UPLINK - DISTANCE 6000 -> MCS 12 -> Itbs 11 (from table 7.1.7.2.1-1 of 36.213)
  // 1 user -> 25 PRB at Itbs 11 -> 621 -> 621000 bytes/sec
  // 3 users -> 8 PRB at Itbs 11 -> 201 -> 201000 bytes/sec
  // 6 users -> 4 PRB at Itbs 11 -> 97 -> 97000 bytes/sec
  // 12 users -> 3 PRB at Itbs 11 -> 73 bytes * 8/12 UE/TTI -> 48667 bytes/sec
  AddTestCase (new LenaTdBetFfMacSchedulerTestCase1 (1,6000,1191000,621000,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaTdBetFfMacSchedulerTestCase1 (3,6000,397000,201000,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaTdBetFfMacSchedulerTestCase1 (6,6000,198500,97000,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaTdBetFfMacSchedulerTestCase1 (12,6000,99250,48667,errorModel), TestCase::EXTENSIVE);
  
  // DOWNLINK - DISTANCE 10000 -> MCS 14 -> Itbs 13 (from table 7.1.7.2.1-1 of 36.213)
  // 1 user -> 24 PRB at Itbs 13 -> 775 -> 775000 byte/sec
  // 3 users -> 775000 among 3 users -> 258333  bytes/sec
  // 6 users -> 775000 among 6 users -> 129166  bytes/sec
  // 12 users -> 775000 among 12 users ->64583  bytes/sec
  // UPLINK - DISTANCE 9000 -> MCS 8 -> Itbs 8 (from table 7.1.7.2.1-1 of 36.213)
  // 1 user -> 24 PRB at Itbs 8 -> 421 -> 421000 bytes/sec
  // 3 users -> 8 PRB at Itbs 8 -> 137 -> 137000 bytes/sec
  // 6 users -> 4 PRB at Itbs 8 -> 67 -> 67000 bytes/sec
  // after the patch enforcing min 3 PRBs per UE:
  // 12 users -> 3 PRB at Itbs 8 -> 49 bytes * 8/12 UE/TTI -> 32667 bytes/sec
  AddTestCase (new LenaTdBetFfMacSchedulerTestCase1 (1,10000,775000,421000,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaTdBetFfMacSchedulerTestCase1 (3,10000,258333,137000,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaTdBetFfMacSchedulerTestCase1 (6,10000,129166,67000,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaTdBetFfMacSchedulerTestCase1 (12,10000,64583,32667,errorModel), TestCase::EXTENSIVE);

  // DOWNLINK - DISTANCE 20000 -> MCS 8 -> Itbs 8 (from table 7.1.7.2.1-1 of 36.213)
  // 1 user -> 24 PRB at Itbs 8 -> 421 -> 421000 bytes/sec
  // 3 users -> 421000 among 3 users ->140333  bytes/sec
  // 6 users -> 421000 among 6 users ->70166  bytes/sec
  // 12 users -> 421000 among 12 users ->35083  bytes/sec 
  // UPLINK - DISTANCE 20000 -> MCS 2 -> Itbs 2 (from table 7.1.7.2.1-1 of 36.213)
  // 1 user -> 25 PRB at Itbs 2 -> 137 -> 137000 bytes/sec
  // 3 users -> 8 PRB at Itbs 2 -> 41 -> 41000 bytes/sec
  // 6 users -> 4 PRB at Itbs 2 -> 22 -> 22000 bytes/sec
  // after the patch enforcing min 3 PRBs per UE:
  // 12 users -> 3 PRB at Itbs 2 -> 18 bytes * 8/12 UE/TTI -> 12000 bytes/sec
  AddTestCase (new LenaTdBetFfMacSchedulerTestCase1 (1,20000,421000,137000,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaTdBetFfMacSchedulerTestCase1 (3,20000,140333,41000,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaTdBetFfMacSchedulerTestCase1 (6,20000,70166,22000,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaTdBetFfMacSchedulerTestCase1 (12,20000,35083,12000,errorModel), TestCase::EXTENSIVE);

  // DOWNLINK - DISTANCE 100000 -> CQI == 0 -> out of range -> 0 bytes/sec
  // UPLINK - DISTANCE 100000 -> CQI == 0 -> out of range -> 0 bytes/sec
  AddTestCase (new LenaTdBetFfMacSchedulerTestCase1 (1,100000,0,0,errorModel), TestCase::QUICK);

  // Test Case 2: fairness check

  std::vector<double> dist;
  dist.push_back (0);       // User 0 distance --> MCS 28
  dist.push_back (4800);    // User 1 distance --> MCS 22
  dist.push_back (6000);    // User 2 distance --> MCS 14
  dist.push_back (10000);    // User 3 distance --> MCS 8
  dist.push_back (20000);    // User 4 distance --> MCS 8
  std::vector<uint32_t> estAchievableRateDl;
  estAchievableRateDl.push_back (2196000);
  estAchievableRateDl.push_back (1383000);
  estAchievableRateDl.push_back (775000);
  estAchievableRateDl.push_back (421000);
  estAchievableRateDl.push_back (421000);
  std::vector<uint32_t> estThrTdBetUl;
  estThrTdBetUl.push_back (469000); // User 0 estimated TTI throughput from TDBET
  estThrTdBetUl.push_back (157000); // User 1 estimated TTI throughput from TDBET
  estThrTdBetUl.push_back (125000); // User 2 estimated TTI throughput from TDBET
  estThrTdBetUl.push_back (85000);  // User 3 estimated TTI throughput from TDBET
  estThrTdBetUl.push_back (26000);  // User 4 estimated TTI throughput from TDBET
  AddTestCase (new LenaTdBetFfMacSchedulerTestCase2 (dist, estAchievableRateDl, estThrTdBetUl, errorModel), TestCase::QUICK);
}

static LenaTestTdBetFfMacSchedulerSuite lenaTestTdBetFfMacSchedulerSuite;


// --------------- T E S T - C A S E   # 1 ------------------------------


std::string 
LenaTdBetFfMacSchedulerTestCase1::BuildNameString (uint16_t nUser, double dist)
{
  std::ostringstream oss;
  oss << nUser << " UEs, distance " << dist << " m";
  return oss.str ();
}

LenaTdBetFfMacSchedulerTestCase1::LenaTdBetFfMacSchedulerTestCase1 (uint16_t nUser, double dist, double thrRefDl, double thrRefUl, bool errorModelEnabled)
  : TestCase (BuildNameString (nUser, dist)),
    m_nUser (nUser),
    m_dist (dist),
    m_thrRefDl (thrRefDl),
    m_thrRefUl (thrRefUl),
    m_errorModelEnabled (errorModelEnabled)
{
}

LenaTdBetFfMacSchedulerTestCase1::~LenaTdBetFfMacSchedulerTestCase1 ()
{
}

void
LenaTdBetFfMacSchedulerTestCase1::DoRun (void)
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

  //Disable Uplink Power Control
  Config::SetDefault ("ns3::LteUePhy::EnableUplinkPowerControl", BooleanValue (false));

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
  lteHelper->SetSchedulerType ("ns3::TdBetFfMacScheduler");
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
  double statsDuration = 0.6;
  double tolerance = 0.1;
  Simulator::Stop (Seconds (statsStartTime + statsDuration - 0.000001));

  lteHelper->EnableMacTraces ();
  lteHelper->EnableRlcTraces ();
  Ptr<RadioBearerStatsCalculator> rlcStats = lteHelper->GetRlcStats ();
  rlcStats->SetAttribute ("StartTime", TimeValue (Seconds (statsStartTime)));
  rlcStats->SetAttribute ("EpochDuration", TimeValue (Seconds (statsDuration)));

  NS_LOG_DEBUG ("Start ");

  Simulator::Run ();

  /**
   * Check that the downlink assignment is done in a "TD blind equal throughput" manner
   */
  NS_LOG_INFO ("DL - Test with " << m_nUser << " user(s) at distance " << m_dist);
  std::vector <uint64_t> dlDataRxed;
  for (int i = 0; i < m_nUser; i++)
    {
      // get the imsi
      uint64_t imsi = ueDevs.Get (i)->GetObject<LteUeNetDevice> ()->GetImsi ();
      // get the lcId
      uint8_t lcId = 3;
      dlDataRxed.push_back (rlcStats->GetDlRxData (imsi, lcId));
      NS_LOG_INFO ("\tUser " << i << " imsi " << imsi << " bytes rxed " << (double)dlDataRxed.at (i) << "  thr " << (double)dlDataRxed.at (i) / statsDuration << " ref " << m_thrRefDl);
    }
  /**
  * Check that the assignment is done in a "TD blind equal throughput" manner among users
  * with equal SINRs: the bandwidth should be distributed according to the 
  * ratio of the estimated throughput per TTI of each user; therefore equally 
  * partitioning the whole bandwidth achievable from a single users in a TTI
  */
  for (int i = 0; i < m_nUser; i++)
    {
      NS_TEST_ASSERT_MSG_EQ_TOL ((double)dlDataRxed.at (i) / statsDuration, m_thrRefDl, m_thrRefDl * tolerance, " Unfair Throughput!");
    }

  /**
  * Check that the uplink assignment is done in a "TD blind equal throughput" manner
  */
  NS_LOG_INFO ("UL - Test with " << m_nUser << " user(s) at distance " << m_dist);
  std::vector <uint64_t> ulDataRxed;
  for (int i = 0; i < m_nUser; i++)
    {
      // get the imsi
      uint64_t imsi = ueDevs.Get (i)->GetObject<LteUeNetDevice> ()->GetImsi ();
      // get the lcId
      uint8_t lcId = 3;
      ulDataRxed.push_back (rlcStats->GetUlRxData (imsi, lcId));
      NS_LOG_INFO ("\tUser " << i << " imsi " << imsi << " bytes rxed " << (double)ulDataRxed.at (i) << "  thr " << (double)ulDataRxed.at (i) / statsDuration << " ref " << m_thrRefUl);
    }
  /**
  * Check that the assignment is done in a "TD blind equal throughput" manner among users
  * with equal SINRs: the bandwidth should be distributed according to the 
  * ratio of the estimated throughput per TTI of each user; therefore equally 
  * partitioning the whole bandwidth achievable from a single users in a TTI
  */
  for (int i = 0; i < m_nUser; i++)
    {
      NS_TEST_ASSERT_MSG_EQ_TOL ((double)ulDataRxed.at (i) / statsDuration, m_thrRefUl, m_thrRefUl * tolerance, " Unfair Throughput!");
    }
  Simulator::Destroy ();

}



// --------------- T E S T - C A S E   # 2 ------------------------------


std::string 
LenaTdBetFfMacSchedulerTestCase2::BuildNameString (uint16_t nUser, std::vector<double> dist)
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


LenaTdBetFfMacSchedulerTestCase2::LenaTdBetFfMacSchedulerTestCase2 (std::vector<double> dist, std::vector<uint32_t> estAchievableRateDl, std::vector<uint32_t> estThrTdBetUl, bool errorModelEnabled)
  : TestCase (BuildNameString (dist.size (), dist)),
    m_nUser (dist.size ()),
    m_dist (dist),
    m_achievableRateDl (estAchievableRateDl),
    m_estThrTdBetUl (estThrTdBetUl),
    m_errorModelEnabled (errorModelEnabled)
{
}

LenaTdBetFfMacSchedulerTestCase2::~LenaTdBetFfMacSchedulerTestCase2 ()
{
}

void
LenaTdBetFfMacSchedulerTestCase2::DoRun (void)
{

  NS_LOG_FUNCTION (this);

  if (!m_errorModelEnabled)
    {
      Config::SetDefault ("ns3::LteSpectrumPhy::CtrlErrorModelEnabled", BooleanValue (false));
      Config::SetDefault ("ns3::LteSpectrumPhy::DataErrorModelEnabled", BooleanValue (false));
    }
  Config::SetDefault ("ns3::LteHelper::UseIdealRrc", BooleanValue (false));
  Config::SetDefault ("ns3::MacStatsCalculator::DlOutputFilename", StringValue (CreateTempDirFilename ("DlMacStats.txt")));
  Config::SetDefault ("ns3::MacStatsCalculator::UlOutputFilename", StringValue (CreateTempDirFilename ("UlMacStats.txt")));
  Config::SetDefault ("ns3::RadioBearerStatsCalculator::DlRlcOutputFilename", StringValue (CreateTempDirFilename ("DlRlcStats.txt")));
  Config::SetDefault ("ns3::RadioBearerStatsCalculator::UlRlcOutputFilename", StringValue (CreateTempDirFilename ("UlRlcStats.txt")));

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
  lteHelper->SetSchedulerType ("ns3::TdBetFfMacScheduler");
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
      mm->SetPosition (Vector (m_dist.at (i), 0.0, 0.0));
      Ptr<LteUeNetDevice> lteUeDev = ueDevs.Get (i)->GetObject<LteUeNetDevice> ();
      Ptr<LteUePhy> uePhy = lteUeDev->GetPhy ();
      uePhy->SetAttribute ("TxPower", DoubleValue (23.0));
      uePhy->SetAttribute ("NoiseFigure", DoubleValue (9.0));
    }


  double statsStartTime = 0.300; // need to allow for RRC connection establishment + SRS
  double statsDuration = 0.4;
  double tolerance = 0.1;
  Simulator::Stop (Seconds (statsStartTime + statsDuration - 0.000001));

  lteHelper->EnableMacTraces ();
  lteHelper->EnableRlcTraces ();
  Ptr<RadioBearerStatsCalculator> rlcStats = lteHelper->GetRlcStats ();
  rlcStats->SetAttribute ("StartTime", TimeValue (Seconds (statsStartTime)));
  rlcStats->SetAttribute ("EpochDuration", TimeValue (Seconds (statsDuration)));

  Simulator::Run ();

  NS_LOG_INFO ("DL - Test with " << m_nUser << " user(s)");
  std::vector <uint64_t> dlDataRxed;
  double totalData = 0;
  double estTotalThr = 0;
  double estUeThr = 0;
  for (int i = 0; i < m_nUser; i++)
    {
      // get the imsi
      uint64_t imsi = ueDevs.Get (i)->GetObject<LteUeNetDevice> ()->GetImsi ();
      // get the lcId
      uint8_t lcId = 3;
      dlDataRxed.push_back (rlcStats->GetDlRxData (imsi, lcId));
      totalData += (double)dlDataRxed.at (i);
      estTotalThr += 1 / m_achievableRateDl.at (i);
      NS_LOG_INFO ("\tUser " << i << " dist " << m_dist.at (i) << " imsi " << imsi << " bytes rxed " << (double)dlDataRxed.at (i) << "  thr " << (double)dlDataRxed.at (i) / statsDuration << " ref " << m_nUser);
    }
    
  estTotalThr = m_nUser * (1 / estTotalThr);
  estUeThr = estTotalThr / m_nUser;
  /**
  * Check that the assignment is done in a "TD blind equal throughput" manner among users
  * with different SINRs: the bandwidth should be distributed equally in long term
  */
  for (int i = 0; i < m_nUser; i++)
    {
      double thrRatio = (double) 1 / m_nUser;
      double estThrRatio = (double)dlDataRxed.at (i) / totalData;
      NS_LOG_INFO ("\tUser " << i << " thrRatio " << thrRatio << " estThrRatio " << estThrRatio);
      NS_TEST_ASSERT_MSG_EQ_TOL (estThrRatio, thrRatio, tolerance, " Unfair Throughput!");
      NS_TEST_ASSERT_MSG_EQ_TOL ((double)dlDataRxed.at (i) / statsDuration, estUeThr, estUeThr * tolerance, " Unfair Throughput!");

    }

  /**
  * Check that the assignment in uplink is done in a round robin manner.
  */

  NS_LOG_INFO ("UL - Test with " << m_nUser);
  std::vector <uint64_t> ulDataRxed;
  for (int i = 0; i < m_nUser; i++)
    {
      // get the imsi
      uint64_t imsi = ueDevs.Get (i)->GetObject<LteUeNetDevice> ()->GetImsi ();
      // get the lcId
      uint8_t lcId = 3;
      ulDataRxed.push_back (rlcStats->GetUlRxData (imsi, lcId));
      NS_LOG_INFO ("\tUser " << i << " dist " << m_dist.at (i) << " bytes rxed " << (double)ulDataRxed.at (i) << "  thr " << (double)ulDataRxed.at (i) / statsDuration << " ref " << (double)m_estThrTdBetUl.at (i));
      NS_TEST_ASSERT_MSG_EQ_TOL ((double)ulDataRxed.at (i) / statsDuration, (double)m_estThrTdBetUl.at (i), (double)m_estThrTdBetUl.at (i) * tolerance, " Unfair Throughput!");
    }
  Simulator::Destroy ();

}
