/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011, 2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Manuel Requena <manuel.requena@cttc.es>
 *         Nicola Baldo <nbaldo@cttc.es>
 */

#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/string.h"
#include "ns3/double.h"
#include "ns3/enum.h"
#include "ns3/boolean.h"
#include "ns3/test.h"
#include "ns3/mobility-helper.h"
#include "ns3/lte-helper.h"

#include "ns3/lte-ue-phy.h"
#include "ns3/lte-ue-net-device.h"
#include "ns3/lte-enb-phy.h"
#include "ns3/lte-enb-net-device.h"
#include "ns3/ff-mac-scheduler.h"

#include "ns3/lte-global-pathloss-database.h"

#include <ns3/lte-chunk-processor.h>


using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LteAntennaTest");


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Tests that the propagation model and the antenna parameters are 
 * generate the correct values. Different test cases are created by specifying different 
 * antenna configurations and it is tested if for the given information the pathloss 
 * value is as expected.
 */
class LteEnbAntennaTestCase : public TestCase
{
public:
  /**
   * Build name string
   * \param orientationDegrees the orientation in degrees
   * \param beamwidthDegrees the beam width in degrees
   * \param x position of UE
   * \param y position of UE
   * \returns the name string
   */
  static std::string BuildNameString (double orientationDegrees, double beamwidthDegrees, double x, double y);
  /**
   * Constructor
   *
   * \param orientationDegrees the orientation in degrees
   * \param beamwidthDegrees the beam width in degrees
   * \param x position of UE
   * \param y position of UE
   * \param antennaGainDb the antenna gain in dB 
   */
  LteEnbAntennaTestCase (double orientationDegrees, double beamwidthDegrees, double x, double y, double antennaGainDb);
  LteEnbAntennaTestCase ();
  virtual ~LteEnbAntennaTestCase ();

private:
  virtual void DoRun (void);

  double m_orientationDegrees; ///< antenna orientation in degrees
  double m_beamwidthDegrees; ///< antenna beamwidth in degrees
  double m_x; ///< x position of the UE
  double m_y; ///< y position of the UE
  double m_antennaGainDb; ///< antenna gain in dB
};




std::string LteEnbAntennaTestCase::BuildNameString (double orientationDegrees, double beamwidthDegrees, double x, double y)
{
  std::ostringstream oss;
  oss <<  "o=" << orientationDegrees
      << ", bw=" << beamwidthDegrees  
      << ", x=" << x 
      << ", y=" << y;
  return oss.str ();
}


LteEnbAntennaTestCase::LteEnbAntennaTestCase (double orientationDegrees, double beamwidthDegrees, double x, double y, double antennaGainDb)
  : TestCase (BuildNameString (orientationDegrees, beamwidthDegrees, x, y)),
    m_orientationDegrees (orientationDegrees),
    m_beamwidthDegrees (beamwidthDegrees),
    m_x (x),
    m_y (y),
    m_antennaGainDb (antennaGainDb)
{
  NS_LOG_FUNCTION (this);
}

LteEnbAntennaTestCase::~LteEnbAntennaTestCase ()
{
}

void
LteEnbAntennaTestCase::DoRun (void)
{
  Config::Reset ();
  Config::SetDefault ("ns3::LteSpectrumPhy::CtrlErrorModelEnabled", BooleanValue (false));
  Config::SetDefault ("ns3::LteSpectrumPhy::DataErrorModelEnabled", BooleanValue (false));
  Config::SetDefault ("ns3::LteHelper::UseIdealRrc", BooleanValue (true));

  //Disable Uplink Power Control
  Config::SetDefault ("ns3::LteUePhy::EnableUplinkPowerControl", BooleanValue (false));

  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();

  // use 0dB Pathloss, since we are testing only the antenna gain
  lteHelper->SetAttribute ("PathlossModel", StringValue ("ns3::ConstantSpectrumPropagationLossModel"));
  lteHelper->SetPathlossModelAttribute ("Loss", DoubleValue (0.0));

  // Create Nodes: eNodeB and UE
  NodeContainer enbNodes;
  NodeContainer ueNodes;
  enbNodes.Create (1);
  ueNodes.Create (1);
  NodeContainer allNodes = NodeContainer ( enbNodes, ueNodes );

  // Install Mobility Model
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));  // eNB
  positionAlloc->Add (Vector (m_x, m_y, 0.0));  // UE
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.SetPositionAllocator (positionAlloc);
  mobility.Install (allNodes);

  // Create Devices and install them in the Nodes (eNB and UE)
  NetDeviceContainer enbDevs;
  NetDeviceContainer ueDevs;
  lteHelper->SetSchedulerType ("ns3::RrFfMacScheduler");
  lteHelper->SetSchedulerAttribute ("UlCqiFilter", EnumValue (FfMacScheduler::PUSCH_UL_CQI));
  lteHelper->SetEnbAntennaModelType ("ns3::CosineAntennaModel");
  lteHelper->SetEnbAntennaModelAttribute ("Orientation", DoubleValue (m_orientationDegrees));
  lteHelper->SetEnbAntennaModelAttribute ("HorizontalBeamwidth", DoubleValue (m_beamwidthDegrees));
  lteHelper->SetEnbAntennaModelAttribute ("MaxGain", DoubleValue (0.0));

  // set DL and UL bandwidth. 
  lteHelper->SetEnbDeviceAttribute ("DlBandwidth", UintegerValue (25));
  lteHelper->SetEnbDeviceAttribute ("UlBandwidth", UintegerValue (25));

  enbDevs = lteHelper->InstallEnbDevice (enbNodes);
  ueDevs = lteHelper->InstallUeDevice (ueNodes);

  // Attach a UE to a eNB
  lteHelper->Attach (ueDevs, enbDevs.Get (0));

  // Activate the default EPS bearer
  enum EpsBearer::Qci q = EpsBearer::NGBR_VIDEO_TCP_DEFAULT;
  EpsBearer bearer (q);
  lteHelper->ActivateDataRadioBearer (ueDevs, bearer);

  // Use testing chunk processor in the PHY layer
  // It will be used to test that the SNR is as intended
  Ptr<LtePhy> uePhy = ueDevs.Get (0)->GetObject<LteUeNetDevice> ()->GetPhy ()->GetObject<LtePhy> ();
  Ptr<LteChunkProcessor> testDlSinr = Create<LteChunkProcessor> ();
  LteSpectrumValueCatcher dlSinrCatcher;
  testDlSinr->AddCallback (MakeCallback (&LteSpectrumValueCatcher::ReportValue, &dlSinrCatcher));
  uePhy->GetDownlinkSpectrumPhy ()->AddDataSinrChunkProcessor (testDlSinr);

  Ptr<LtePhy> enbphy = enbDevs.Get (0)->GetObject<LteEnbNetDevice> ()->GetPhy ()->GetObject<LtePhy> ();
  Ptr<LteChunkProcessor> testUlSinr = Create<LteChunkProcessor> ();
  LteSpectrumValueCatcher ulSinrCatcher;
  testUlSinr->AddCallback (MakeCallback (&LteSpectrumValueCatcher::ReportValue, &ulSinrCatcher));
  enbphy->GetUplinkSpectrumPhy ()->AddDataSinrChunkProcessor (testUlSinr);


  // keep track of all path loss values in two centralized objects
  DownlinkLteGlobalPathlossDatabase dlPathlossDb;
  UplinkLteGlobalPathlossDatabase ulPathlossDb;
  // we rely on the fact that LteHelper creates the DL channel object first, then the UL channel object,
  // hence the former will have index 0 and the latter 1
  Config::Connect ("/ChannelList/0/PathLoss",
                   MakeCallback (&DownlinkLteGlobalPathlossDatabase::UpdatePathloss, &dlPathlossDb));
  Config::Connect ("/ChannelList/1/PathLoss",
                    MakeCallback (&UplinkLteGlobalPathlossDatabase::UpdatePathloss, &ulPathlossDb)); 

  Simulator::Stop (Seconds (0.035));
  Simulator::Run ();

  const double enbTxPowerDbm = 30; // default eNB TX power over whole bandwidth
  const double ueTxPowerDbm  = 10; // default UE TX power over whole bandwidth
  const double ktDbm = -174;    // reference LTE noise PSD
  const double noisePowerDbm = ktDbm + 10 * std::log10 (25 * 180000); // corresponds to kT*bandwidth in linear units
  const double ueNoiseFigureDb = 9.0; // default UE noise figure
  const double enbNoiseFigureDb = 5.0; // default eNB noise figure
  double tolerance = (m_antennaGainDb != 0) ? std::abs (m_antennaGainDb) * 0.001 : 0.001;

  // first test with SINR from LteChunkProcessor
  // this can only be done for not-too-bad SINR otherwise the measurement won't be available
  double expectedSinrDl = enbTxPowerDbm + m_antennaGainDb - noisePowerDbm + ueNoiseFigureDb;
  if (expectedSinrDl > 0)
    {
      double calculatedSinrDbDl = -INFINITY;
      if (dlSinrCatcher.GetValue () != 0)
        {
          calculatedSinrDbDl = 10.0 * std::log10 (dlSinrCatcher.GetValue ()->operator[] (0));
        }      
      // remember that propagation loss is 0dB
      double calculatedAntennaGainDbDl = - (enbTxPowerDbm - calculatedSinrDbDl - noisePowerDbm - ueNoiseFigureDb);      
      NS_LOG_INFO ("expected " << m_antennaGainDb << " actual " << calculatedAntennaGainDbDl << " tol " << tolerance);
      NS_TEST_ASSERT_MSG_EQ_TOL (calculatedAntennaGainDbDl, m_antennaGainDb, tolerance, "Wrong DL antenna gain!");
    }
  double expectedSinrUl = ueTxPowerDbm + m_antennaGainDb - noisePowerDbm + enbNoiseFigureDb;
  if (expectedSinrUl > 0)
    {      
      double calculatedSinrDbUl = -INFINITY;
      if (ulSinrCatcher.GetValue () != 0)
        {
          calculatedSinrDbUl = 10.0 * std::log10 (ulSinrCatcher.GetValue ()->operator[] (0));
        }  
      double calculatedAntennaGainDbUl = - (ueTxPowerDbm - calculatedSinrDbUl - noisePowerDbm - enbNoiseFigureDb);
      NS_TEST_ASSERT_MSG_EQ_TOL (calculatedAntennaGainDbUl, m_antennaGainDb, tolerance, "Wrong UL antenna gain!");
    }


  // repeat the same tests with the LteGlobalPathlossDatabases
  double measuredLossDl = dlPathlossDb.GetPathloss (1, 1);
  NS_TEST_ASSERT_MSG_EQ_TOL (measuredLossDl, -m_antennaGainDb, tolerance, "Wrong DL loss!");
  double measuredLossUl = ulPathlossDb.GetPathloss (1, 1);
  NS_TEST_ASSERT_MSG_EQ_TOL (measuredLossUl, -m_antennaGainDb, tolerance, "Wrong UL loss!");

  
  Simulator::Destroy ();
}


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Lte Enb Antenna Test Suite
 */
class LteAntennaTestSuite : public TestSuite
{
public:
  LteAntennaTestSuite ();
};


LteAntennaTestSuite::LteAntennaTestSuite ()
  : TestSuite ("lte-antenna", SYSTEM)
{
  NS_LOG_FUNCTION (this);

  //                                      orientation beamwidth     x            y         gain 
  AddTestCase (new LteEnbAntennaTestCase (       0.0,     90.0,    1.0,          0.0,       0.0), TestCase::QUICK);
  AddTestCase (new LteEnbAntennaTestCase (       0.0,     90.0,    1.0,          1.0,      -3.0), TestCase::QUICK);
  AddTestCase (new LteEnbAntennaTestCase (       0.0,     90.0,    1.0,         -1.0,      -3.0), TestCase::QUICK);
  AddTestCase (new LteEnbAntennaTestCase (       0.0,     90.0,   -1.0,         -1.0,   -36.396), TestCase::QUICK);
  AddTestCase (new LteEnbAntennaTestCase (       0.0,     90.0,   -1.0,         -0.0,   -1414.6), TestCase::QUICK);
  AddTestCase (new LteEnbAntennaTestCase (       0.0,     90.0,   -1.0,          1.0,   -36.396), TestCase::QUICK);
  AddTestCase (new LteEnbAntennaTestCase (      45.0,     90.0,    1.0,          1.0,       0.0), TestCase::QUICK);
  AddTestCase (new LteEnbAntennaTestCase (     -45.0,     90.0,    1.0,         -1.0,       0.0), TestCase::QUICK);
  AddTestCase (new LteEnbAntennaTestCase (      90.0,     90.0,    1.0,          1.0,      -3.0), TestCase::QUICK);
  AddTestCase (new LteEnbAntennaTestCase (     -90.0,     90.0,    1.0,         -1.0,      -3.0), TestCase::QUICK); 

  AddTestCase (new LteEnbAntennaTestCase (       0.0,    120.0,    1.0,          0.0,       0.0), TestCase::QUICK);
  AddTestCase (new LteEnbAntennaTestCase (       0.0,    120.0,    0.5,  sin(M_PI/3),      -3.0), TestCase::QUICK);
  AddTestCase (new LteEnbAntennaTestCase (       0.0,    120.0,    0.5, -sin(M_PI/3),      -3.0), TestCase::QUICK);
  AddTestCase (new LteEnbAntennaTestCase (       0.0,    120.0,   -1.0,         -2.0,   -13.410), TestCase::QUICK);
  AddTestCase (new LteEnbAntennaTestCase (       0.0,    120.0,   -1.0,          1.0,   -20.034), TestCase::QUICK);
  AddTestCase (new LteEnbAntennaTestCase (      60.0,    120.0,    0.5,  sin(M_PI/3),       0.0), TestCase::QUICK);
  AddTestCase (new LteEnbAntennaTestCase (     -60.0,    120.0,    0.5, -sin(M_PI/3),       0.0), TestCase::QUICK);
  AddTestCase (new LteEnbAntennaTestCase (     -60.0,    120.0,    0.5, -sin(M_PI/3),       0.0), TestCase::QUICK);
  AddTestCase (new LteEnbAntennaTestCase (    -120.0,    120.0,   -0.5, -sin(M_PI/3),       0.0), TestCase::QUICK);
  AddTestCase (new LteEnbAntennaTestCase (    -120.0,    120.0,    0.5, -sin(M_PI/3),      -3.0), TestCase::QUICK);
  AddTestCase (new LteEnbAntennaTestCase (    -120.0,    120.0,     -1,            0,      -3.0), TestCase::QUICK);
  AddTestCase (new LteEnbAntennaTestCase (    -120.0,    120.0,     -1,            2,   -15.578), TestCase::QUICK);
  AddTestCase (new LteEnbAntennaTestCase (    -120.0,    120.0,      1,            0,   -14.457), TestCase::QUICK);
  AddTestCase (new LteEnbAntennaTestCase (    -120.0,    120.0,      1,            2,   -73.154), TestCase::QUICK);
  AddTestCase (new LteEnbAntennaTestCase (    -120.0,    120.0,      1,         -0.1,   -12.754), TestCase::QUICK);


}

static LteAntennaTestSuite lteAntennaTestSuite;
