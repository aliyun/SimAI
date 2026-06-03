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
#include "lte-test-fdmt-ff-mac-scheduler.h"
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

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LenaTestFdMtFfMacScheduler");

LenaTestFdMtFfMacSchedulerSuite::LenaTestFdMtFfMacSchedulerSuite ()
  : TestSuite ("lte-fdmt-ff-mac-scheduler", SYSTEM)
{
  NS_LOG_INFO ("creating LenaTestFdMtFfMacSchedulerSuite");

  bool errorModel = false;

  //Test Case : AMC works in FDMT

  //Note: here the MCS is calculated by the narrowband CQI

  // DOWNLINK - DISTANCE 0 -> MCS 28 -> Itbs 26 (from table 7.1.7.2.1-1 of 36.213)
  // 1 user -> 24 PRB at Itbs 26 -> 2196 -> 2196000 bytes/sec for one UE; 0 bytes/sec for other UEs
  // 3 users -> 2196000 among 3 users -> 2196000 bytes/sec for one UE; 0 bytes/sec for other UEs
  // 6 users -> 2196000 among 6 users -> 2196000 bytes/sec for one UE; 0 bytes/sec for other UEs
  // 12 users -> 2196000 among 12 users -> 2196000 bytes/sec for one UE; 0 bytes/sec for other UEs
  // UPLINK- DISTANCE 0 -> MCS 28 -> Itbs 26 (from table 7.1.7.2.1-1 of 36.213)
  // 1 user -> 25 PRB at Itbs 26 -> 2292 -> 2292000 bytes/sec
  // 3 users -> 8 PRB at Itbs 26 -> 749 -> 749000 bytes/sec
  // 6 users -> 4 PRB at Itbs 26 -> 373 -> 373000 bytes/sec
  // after the patch enforcing min 3 PRBs per UE:
  // 12 users -> 3 PRB at Itbs 26 -> 277 bytes * 8/12 UE/TTI -> 184670 bytes/sec
  AddTestCase (new LenaFdMtFfMacSchedulerTestCase (1,0,2196000,2292000, errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaFdMtFfMacSchedulerTestCase (3,0,2196000,749000,errorModel), TestCase::QUICK);
  AddTestCase (new LenaFdMtFfMacSchedulerTestCase (6,0,2196000,373000, errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaFdMtFfMacSchedulerTestCase (12,0,2196000,184670, errorModel), TestCase::EXTENSIVE);

  // DOWNLINK - DISTANCE 4800 -> MCS 22 -> Itbs 20 (from table 7.1.7.2.1-1 of 36.213)
  // 1 user -> 24 PRB at Itbs 20 -> 1383 -> 1383000 byte/secfor one UE; 0 bytes/sec for other UEs
  // 3 users -> 1383000 among 3 users -> 1383000 bytes/sec for one UE; 0 bytes/sec for other UEs
  // 6 users -> 1383000 among 6 users -> 1383000 bytes/sec for one UE; 0 bytes/sec for other UEs
  // 12 users -> 1383000 among 12 users -> 1383000 bytes/sec for one UE; 0 bytes/sec for other UEs
  // UPLINK - DISTANCE 4800 -> MCS 14 -> Itbs 13 (from table 7.1.7.2.1-1 of 36.213)
  // 1 user -> 25 PRB at Itbs 13 -> 807 -> 807000 bytes/sec
  // 3 users -> 8 PRB at Itbs 13 -> 253 -> 253000 bytes/sec
  // 6 users -> 4 PRB at Itbs 13 -> 125 -> 125000 bytes/sec
  // after the patch enforcing min 3 PRBs per UE:
  // 12 users -> 3 PRB at Itbs 13 -> 93  bytes * 8/12 UE/TTI  -> 62000 bytes/sec
  AddTestCase (new LenaFdMtFfMacSchedulerTestCase (1,4800,1383000,807000,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaFdMtFfMacSchedulerTestCase (3,4800,1383000,253000,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaFdMtFfMacSchedulerTestCase (6,4800,1383000,125000,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaFdMtFfMacSchedulerTestCase (12,4800,1383000,62000,errorModel), TestCase::EXTENSIVE);


  // DOWNLINK - DISTANCE 6000 -> MCS 20 -> Itbs 18 (from table 7.1.7.2.1-1 of 36.213)
  // 1 user -> 24 PRB at Itbs 18 -> 1191 -> 1191000 byte/sec for one UE; 0 bytes/sec for other UEs
  // 3 users -> 1191000 among 3 users -> 1191000 bytes/sec for one UE; 0 bytes/sec for other UEs
  // 6 users -> 1191000 among 6 users -> 1191000 bytes/sec for one UE; 0 bytes/sec for other UEs
  // 12 users ->1191000 among 12 users -> 1191000 bytes/sec for one UE; 0 bytes/sec for other UEs
  // UPLINK - DISTANCE 6000 -> MCS 12 -> Itbs 11 (from table 7.1.7.2.1-1 of 36.213)
  // 1 user -> 25 PRB at Itbs 11 -> 621 -> 621000 bytes/sec
  // 3 users -> 8 PRB at Itbs 11 -> 201 -> 201000 bytes/sec
  // 6 users -> 4 PRB at Itbs 11 -> 97 -> 97000 bytes/sec
  // after the patch enforcing min 3 PRBs per UE:
  // 12 users -> 3 PRB at Itbs 11 -> 73 bytes * 8/12 UE/TTI -> 48667 bytes/sec
  AddTestCase (new LenaFdMtFfMacSchedulerTestCase (1,6000,1191000,621000, errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaFdMtFfMacSchedulerTestCase (3,6000,1191000,201000, errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaFdMtFfMacSchedulerTestCase (6,6000,1191000,97000, errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaFdMtFfMacSchedulerTestCase (12,6000,1191000,48667, errorModel), TestCase::EXTENSIVE);

  // DOWNLINK - DISTANCE 10000 -> MCS 14 -> Itbs 13 (from table 7.1.7.2.1-1 of 36.213)
  // 1 user -> 24 PRB at Itbs 13 -> 775 -> 775000 byte/sec for one UE; 0 bytes/sec for other UEs
  // 3 users -> 775000 among 3 users -> 775000 bytes/sec for one UE; 0 bytes/sec for other UEs
  // 6 users -> 775000 among 6 users -> 775000 bytes/sec for one UE; 0 bytes/sec for other UEs
  // 12 users -> 775000 among 12 users -> 775000 bytes/sec for one UE; 0 bytes/sec for other UEs
  // UPLINK - DISTANCE 10000 -> MCS 8 -> Itbs 8 (from table 7.1.7.2.1-1 of 36.213)
  // 1 user -> 24 PRB at Itbs 8 -> 437 -> 437000 bytes/sec
  // 3 users -> 8 PRB at Itbs 8 -> 137 -> 137000 bytes/sec
  // 6 users -> 4 PRB at Itbs 8 -> 67 -> 67000 bytes/sec
  // after the patch enforcing min 3 PRBs per UE:
  // 12 users -> 3 PRB at Itbs 8 -> 49 bytes * 8/12 UE/TTI -> 32667 bytes/sec
  AddTestCase (new LenaFdMtFfMacSchedulerTestCase (1,10000,775000,437000,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaFdMtFfMacSchedulerTestCase (3,10000,775000,137000,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaFdMtFfMacSchedulerTestCase (6,10000,775000,67000,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaFdMtFfMacSchedulerTestCase (12,10000,775000,32667,errorModel), TestCase::EXTENSIVE);
 
  // DONWLINK - DISTANCE 20000 -> MCS 8 -> Itbs 8 (from table 7.1.7.2.1-1 of 36.213)
  // 1 user -> 24 PRB at Itbs 8 -> 421 -> 421000 bytes/sec for one UE; 0 bytes/sec for other UEs
  // 3 users -> 421000 among 3 users -> 421000 bytes/sec for one UE; 0 bytes/sec for other UEs
  // 6 users -> 421000 among 6 users -> 421000 bytes/sec for one UE; 0 bytes/sec for other UEs
  // 12 users -> 421000 among 12 users -> 421000 bytes/sec for one UE; 0 bytes/sec for other UEs
  // UPLINK - DISTANCE 20000 -> MCS 2 -> Itbs 2 (from table 7.1.7.2.1-1 of 36.213)
  // 1 user -> 25 PRB at Itbs 2 -> 233 -> 137000 bytes/sec
  // 3 users -> 8 PRB at Itbs 2 -> 69 -> 41000 bytes/sec
  // 6 users -> 4 PRB at Itbs 2 -> 32 -> 22000 bytes/sec
  // after the patch enforcing min 3 PRBs per UE:
  // 12 users -> 3 PRB at Itbs 2 -> 26 bytes * 8/12 UE/TTI -> 12000 bytes/sec
  AddTestCase (new LenaFdMtFfMacSchedulerTestCase (1,20000,421000,137000,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaFdMtFfMacSchedulerTestCase (3,20000,421000,41000,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaFdMtFfMacSchedulerTestCase (6,20000,421000,22000,errorModel), TestCase::EXTENSIVE);
  AddTestCase (new LenaFdMtFfMacSchedulerTestCase (12,20000,421000,12000,errorModel), TestCase::EXTENSIVE);

  // DOWNLINK - DISTANCE 100000 -> CQI == 0 -> out of range -> 0 bytes/sec
  // UPLINK - DISTANCE 100000 -> CQI == 0 -> out of range -> 0 bytes/sec
  AddTestCase (new LenaFdMtFfMacSchedulerTestCase (1,100000,0,0,errorModel), TestCase::QUICK);
}

static LenaTestFdMtFfMacSchedulerSuite lenaTestFdMtFfMacSchedulerSuite;


// --------------- T E S T - C A S E ------------------------------

std::string 
LenaFdMtFfMacSchedulerTestCase::BuildNameString (uint16_t nUser, double dist)
{
  std::ostringstream oss;
  oss << nUser << " UEs, distance " << dist << " m";
  return oss.str ();
}

LenaFdMtFfMacSchedulerTestCase::LenaFdMtFfMacSchedulerTestCase (uint16_t nUser, double dist, double thrRefDl, double thrRefUl, bool errorModelEnabled)
  : TestCase (BuildNameString (nUser, dist)),
    m_nUser (nUser),
    m_dist (dist),
    m_thrRefDl (thrRefDl),
    m_thrRefUl (thrRefUl),
    m_errorModelEnabled (errorModelEnabled)
{
}

LenaFdMtFfMacSchedulerTestCase::~LenaFdMtFfMacSchedulerTestCase ()
{
}


void
LenaFdMtFfMacSchedulerTestCase::DoRun (void)
{

  NS_LOG_FUNCTION (this << m_nUser << m_dist);

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
  lteHelper->SetSchedulerType ("ns3::FdMtFfMacScheduler");
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


  Simulator::Run ();

  /**
   * Check that the downlink assignment is done in a "frequency domain maximum throughput" manner
   */
  NS_LOG_INFO ("DL - Test with " << m_nUser << " user(s) at distance " << m_dist);
  std::vector <uint64_t> dlDataRxed;
  for (int i = 0; i < m_nUser; i++)
    {
      // get the imsi
      uint64_t imsi = ueDevs.Get (i)->GetObject<LteUeNetDevice> ()->GetImsi ();
      uint8_t lcId = 3;
      dlDataRxed.push_back (rlcStats->GetDlRxData (imsi, lcId));
      NS_LOG_INFO ("\tUser " << i << " imsi " << imsi << " bytes rxed " << (double)dlDataRxed.at (i) << "  thr " << (double)dlDataRxed.at (i) / statsDuration << " ref " << m_thrRefDl);
    }

  /**
  * Check that the assignment is done in a "frequency domain maximum throughput" manner among users
  * with maximum SINRs: without fading, current FD MT always assign all resources to one UE
  */

  uint8_t found = 0;
  for (int i = 0; i < m_nUser; i++)
    {
      double throughput = (double)dlDataRxed.at (i) / statsDuration;
      if (throughput != 0 && found == 0)
        {
          NS_TEST_ASSERT_MSG_EQ_TOL (throughput, m_thrRefDl, m_thrRefDl * tolerance, " Unfair Throughput!");
          found = 1;
        }
      else if (throughput != 0 && found == 1)
        {
           NS_TEST_ASSERT_MSG_EQ_TOL (0, m_thrRefDl, m_thrRefDl * tolerance, " Unfair Throughput!");
        }
      else
        NS_TEST_ASSERT_MSG_EQ_TOL (throughput, 0, 0, " Unfair Throughput!");
    }

  /**
  * Check that the uplink assignment is done in a "proportional fair" manner
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
  * Check that the assignment is done in a "proportional fair" manner among users
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

