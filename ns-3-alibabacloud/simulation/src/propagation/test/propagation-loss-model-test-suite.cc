/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 The Boeing Company
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
 */

#include "ns3/log.h"
#include "ns3/abort.h"
#include "ns3/test.h"
#include "ns3/config.h"
#include "ns3/double.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/simulator.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("PropagationLossModelsTest");

// ===========================================================================
// This is a simple test to validate propagation loss models of ns-3 wifi.
// See the chapter in the ns-3 testing and validation guide for more detail
// ===========================================================================


/**
 * \ingroup propagation
 * \defgroup propagation-test Propagation module tests
 */


/**
 * \ingroup propagation-tests
 *
 * \brief FriisPropagationLossModel Test
 */
class FriisPropagationLossModelTestCase : public TestCase
{
public:
  FriisPropagationLossModelTestCase ();
  virtual ~FriisPropagationLossModelTestCase ();

private:
  virtual void DoRun (void);

  /// Test vector
  typedef struct {
    Vector m_position;  //!< Test node position
    double m_pt;        //!< Tx power [dBm]
    double m_pr;        //!< Rx power [W]
    double m_tolerance; //!< Tolerance
  } TestVector;

  /// Test vectors
  TestVectors<TestVector> m_testVectors;
};

FriisPropagationLossModelTestCase::FriisPropagationLossModelTestCase ()
  : TestCase ("Check to see that the ns-3 Friis propagation loss model provides correct received power"), m_testVectors ()
{
}

FriisPropagationLossModelTestCase::~FriisPropagationLossModelTestCase ()
{
}

void
FriisPropagationLossModelTestCase::DoRun (void)
{
  // The ns-3 testing manual gives more background on the values selected
  // for this test.  First, set a few defaults. 

  // the test vectors have been determined for a wavelength of 0.125 m 
  // which corresponds to a frequency of 2398339664.0 Hz in the vacuum
  Config::SetDefault ("ns3::FriisPropagationLossModel::Frequency", DoubleValue (2398339664.0));
  Config::SetDefault ("ns3::FriisPropagationLossModel::SystemLoss", DoubleValue (1.0));

  // Select a reference transmit power
  // Pt = 10^(17.0206/10)/10^3 = .05035702 W
  double txPowerW = 0.05035702;
  double txPowerdBm = 10 * std::log10 (txPowerW) + 30;

  //
  // We want to test the propagation loss model calculations at a few chosen 
  // distances and compare the results to those we have manually calculated
  // according to the model documentation.  The model reference specifies, 
  // for instance, that the received power at 100m according to the provided
  // input power will be 4.98265e-10 W.  Since this value specifies the power
  // to 1e-15 significance, we test the ns-3 calculated value for agreement 
  // within 5e-16.
  //
  TestVector testVector;

  testVector.m_position = Vector (100, 0, 0);
  testVector.m_pt = txPowerdBm;
  testVector.m_pr = 4.98265e-10;
  testVector.m_tolerance = 5e-16;
  m_testVectors.Add (testVector);

  testVector.m_position = Vector (500, 0, 0);
  testVector.m_pt = txPowerdBm;
  testVector.m_pr = 1.99306e-11;
  testVector.m_tolerance = 5e-17;
  m_testVectors.Add (testVector);

  testVector.m_position = Vector (1000, 0, 0);
  testVector.m_pt = txPowerdBm;
  testVector.m_pr = 4.98265e-12;
  testVector.m_tolerance = 5e-18;
  m_testVectors.Add (testVector);

  testVector.m_position = Vector (2000, 0, 0);
  testVector.m_pt = txPowerdBm;
  testVector.m_pr = 1.24566e-12;
  testVector.m_tolerance = 5e-18;
  m_testVectors.Add (testVector);

  // Now, check that the received power values are expected

  Ptr<MobilityModel> a = CreateObject<ConstantPositionMobilityModel> (); 
  a->SetPosition (Vector (0,0,0));
  Ptr<MobilityModel> b = CreateObject<ConstantPositionMobilityModel> (); 

  Ptr<FriisPropagationLossModel> lossModel = CreateObject<FriisPropagationLossModel> (); 
  for (uint32_t i = 0; i < m_testVectors.GetN (); ++i)
    {
      testVector = m_testVectors.Get (i);
      b->SetPosition (testVector.m_position);
      double resultdBm = lossModel->CalcRxPower (testVector.m_pt, a, b);
      double resultW = std::pow (10.0, resultdBm/10.0)/1000;
      NS_TEST_EXPECT_MSG_EQ_TOL (resultW, testVector.m_pr, testVector.m_tolerance, "Got unexpected rcv power");
    }
}

// Added for Two-Ray Ground Model - tomhewer@mac.com

/**
 * \ingroup propagation-tests
 *
 * \brief TwoRayGroundPropagationLossModel Test
 */
class TwoRayGroundPropagationLossModelTestCase : public TestCase
{
public:
  TwoRayGroundPropagationLossModelTestCase ();
  virtual ~TwoRayGroundPropagationLossModelTestCase ();

private:
  virtual void DoRun (void);

  /// Test vector
  typedef struct {
    Vector m_position;  //!< Test node position
    double m_pt;        //!< Tx power [dBm]
    double m_pr;        //!< Rx power [W]
    double m_tolerance; //!< Tolerance
  } TestVector;

  /// Test vectors
  TestVectors<TestVector> m_testVectors;
};

TwoRayGroundPropagationLossModelTestCase::TwoRayGroundPropagationLossModelTestCase ()
  : TestCase ("Check to see that the ns-3 TwoRayGround propagation loss model provides correct received power"),
    m_testVectors ()
{
}

TwoRayGroundPropagationLossModelTestCase::~TwoRayGroundPropagationLossModelTestCase ()
{
}

void
TwoRayGroundPropagationLossModelTestCase::DoRun (void)
{
  // the test vectors have been determined for a wavelength of 0.125 m 
  // which corresponds to a frequency of 2398339664.0 Hz in the vacuum
  Config::SetDefault ("ns3::TwoRayGroundPropagationLossModel::Frequency", DoubleValue (2398339664.0));
  Config::SetDefault ("ns3::TwoRayGroundPropagationLossModel::SystemLoss", DoubleValue (1.0));

  // set antenna height to 1.5m above z coordinate
  Config::SetDefault ("ns3::TwoRayGroundPropagationLossModel::HeightAboveZ", DoubleValue (1.5));

  // Select a reference transmit power of 17.0206 dBm
  // Pt = 10^(17.0206/10)/10^3 = .05035702 W
  double txPowerW = 0.05035702;
  double txPowerdBm = 10 * std::log10 (txPowerW) + 30;

  //
  // As with the Friis tests above, we want to test the propagation loss 
  // model calculations at a few chosen distances and compare the results 
  // to those we can manually calculate. Let us test the ns-3 calculated 
  // value for agreement to be within 5e-16, as above.
  //
  TestVector testVector;

  // Below the Crossover distance use Friis so this test should be the same as that above
  // Crossover = (4 * PI * TxAntennaHeight * RxAntennaHeight) / Lamdba
  // Crossover = (4 * PI * 1.5 * 1.5) / 0.125 = 226.1946m

  testVector.m_position = Vector (100, 0, 0);
  testVector.m_pt = txPowerdBm;
  testVector.m_pr = 4.98265e-10;
  testVector.m_tolerance = 5e-16;
  m_testVectors.Add (testVector);

  // These values are above the crossover distance and therefore use the Two Ray calculation

  testVector.m_position = Vector (500, 0, 0);
  testVector.m_pt = txPowerdBm;
  testVector.m_pr = 4.07891862e-12;
  testVector.m_tolerance = 5e-16;
  m_testVectors.Add (testVector);

  testVector.m_position = Vector (1000, 0, 0);
  testVector.m_pt = txPowerdBm;
  testVector.m_pr = 2.5493241375e-13;
  testVector.m_tolerance = 5e-16;
  m_testVectors.Add (testVector);

  testVector.m_position = Vector (2000, 0, 0);
  testVector.m_pt = txPowerdBm;
  testVector.m_pr = 1.593327585938e-14;
  testVector.m_tolerance = 5e-16;
  m_testVectors.Add (testVector);

  // Repeat the tests for non-zero z coordinates

  // Pr = (0.05035702 * (1.5*1.5) * (2.5*2.5)) / (500*500*500*500) = 1.13303295e-11
  // dCross = (4 * pi * 1.5 * 2.5) / 0.125 = 376.99m
  testVector.m_position = Vector (500, 0, 1);
  testVector.m_pt = txPowerdBm;
  testVector.m_pr = 1.13303295e-11;
  testVector.m_tolerance = 5e-16;
  m_testVectors.Add (testVector);

  // Pr = (0.05035702 * (1.5*1.5) * (5.5*5.5)) / (1000*1000*1000*1000) = 3.42742467375e-12
  // dCross = (4 * pi * 1.5 * 5.5) / 0.125 = 829.38m
  testVector.m_position = Vector (1000, 0, 4);
  testVector.m_pt = txPowerdBm;
  testVector.m_pr = 3.42742467375e-12;
  testVector.m_tolerance = 5e-16;
  m_testVectors.Add (testVector);

  // Pr = (0.05035702 * (1.5*1.5) * (11.5*11.5)) / (2000*2000*2000*2000) = 9.36522547734e-13
  // dCross = (4 * pi * 1.5 * 11.5) / 0.125 = 1734.15m
  testVector.m_position = Vector (2000, 0, 10);
  testVector.m_pt = txPowerdBm;
  testVector.m_pr = 9.36522547734e-13;
  testVector.m_tolerance = 5e-16;
  m_testVectors.Add (testVector);


  // Now, check that the received power values are expected

  Ptr<MobilityModel> a = CreateObject<ConstantPositionMobilityModel> (); 
  a->SetPosition (Vector (0,0,0));
  Ptr<MobilityModel> b = CreateObject<ConstantPositionMobilityModel> (); 

  Ptr<TwoRayGroundPropagationLossModel> lossModel = CreateObject<TwoRayGroundPropagationLossModel> (); 
  for (uint32_t i = 0; i < m_testVectors.GetN (); ++i)
    {
      testVector = m_testVectors.Get (i);
      b->SetPosition (testVector.m_position);
      double resultdBm = lossModel->CalcRxPower (testVector.m_pt, a, b);
      double resultW = std::pow (10.0, resultdBm / 10.0) / 1000;
      NS_TEST_EXPECT_MSG_EQ_TOL (resultW, testVector.m_pr, testVector.m_tolerance, "Got unexpected rcv power");
    }
}


/**
 * \ingroup propagation-tests
 *
 * \brief LogDistancePropagationLossModel Test
 */
class LogDistancePropagationLossModelTestCase : public TestCase
{
public:
  LogDistancePropagationLossModelTestCase ();
  virtual ~LogDistancePropagationLossModelTestCase ();

private:
  virtual void DoRun (void);

  /// Test vector
  typedef struct {
    Vector m_position;  //!< Test node position
    double m_pt;        //!< Tx power [dBm]
    double m_pr;        //!< Rx power [W]
    double m_tolerance; //!< Tolerance
  } TestVector;

  /// Test vectors
  TestVectors<TestVector> m_testVectors;
};

LogDistancePropagationLossModelTestCase::LogDistancePropagationLossModelTestCase ()
  : TestCase ("Check to see that the ns-3 Log Distance propagation loss model provides correct received power"), m_testVectors ()
{
}

LogDistancePropagationLossModelTestCase::~LogDistancePropagationLossModelTestCase ()
{
}

void
LogDistancePropagationLossModelTestCase::DoRun (void)
{
  // reference loss at 2.4 GHz is 40.045997
  Config::SetDefault ("ns3::LogDistancePropagationLossModel::ReferenceLoss", DoubleValue (40.045997));
  Config::SetDefault ("ns3::LogDistancePropagationLossModel::Exponent", DoubleValue (3));

  // Select a reference transmit power
  // Pt = 10^(17.0206/10)/10^3 = .05035702 W
  double txPowerW = 0.05035702;
  double txPowerdBm = 10 * std::log10 (txPowerW) + 30;

  //
  // We want to test the propagation loss model calculations at a few chosen 
  // distances and compare the results to those we have manually calculated
  // according to the model documentation.  The following "TestVector" objects
  // will drive the test.
  //
  TestVector testVector;

  testVector.m_position = Vector (10, 0, 0);
  testVector.m_pt = txPowerdBm;
  testVector.m_pr = 4.98265e-9;
  testVector.m_tolerance = 5e-15; 
  m_testVectors.Add (testVector);

  testVector.m_position = Vector (20, 0, 0);
  testVector.m_pt = txPowerdBm;
  testVector.m_pr = 6.22831e-10;
  testVector.m_tolerance = 5e-16;
  m_testVectors.Add (testVector);

  testVector.m_position = Vector (40, 0, 0);
  testVector.m_pt = txPowerdBm;
  testVector.m_pr = 7.78539e-11;
  testVector.m_tolerance = 5e-17;
  m_testVectors.Add (testVector);

  testVector.m_position = Vector (80, 0, 0);
  testVector.m_pt = txPowerdBm;
  testVector.m_pr = 9.73173e-12;
  testVector.m_tolerance = 5e-17;
  m_testVectors.Add (testVector);

  Ptr<MobilityModel> a = CreateObject<ConstantPositionMobilityModel> (); 
  a->SetPosition (Vector (0,0,0));
  Ptr<MobilityModel> b = CreateObject<ConstantPositionMobilityModel> (); 

  Ptr<LogDistancePropagationLossModel> lossModel = CreateObject<LogDistancePropagationLossModel> (); 
  for (uint32_t i = 0; i < m_testVectors.GetN (); ++i)
    {
      testVector = m_testVectors.Get (i);
      b->SetPosition (testVector.m_position);
      double resultdBm = lossModel->CalcRxPower (testVector.m_pt, a, b);
      double resultW = std::pow (10.0, resultdBm/10.0)/1000;
      NS_TEST_EXPECT_MSG_EQ_TOL (resultW, testVector.m_pr, testVector.m_tolerance, "Got unexpected rcv power");
    }
}

/**
 * \ingroup propagation-tests
 *
 * \brief MatrixPropagationLossModel Test
 */
class MatrixPropagationLossModelTestCase : public TestCase
{
public:
  MatrixPropagationLossModelTestCase ();
  virtual ~MatrixPropagationLossModelTestCase ();

private:
  virtual void DoRun (void);
};

MatrixPropagationLossModelTestCase::MatrixPropagationLossModelTestCase ()
  : TestCase ("Test MatrixPropagationLossModel")
{
}

MatrixPropagationLossModelTestCase::~MatrixPropagationLossModelTestCase ()
{
}

void
MatrixPropagationLossModelTestCase::DoRun (void)
{
  Ptr<MobilityModel> m[3];
  for (int i = 0; i < 3; ++i)
    {
      m[i] = CreateObject<ConstantPositionMobilityModel> ();
    }

  MatrixPropagationLossModel loss;
  // no loss by default
  loss.SetDefaultLoss (0);
  // -10 dB for 0 -> 1 and 1 -> 0
  loss.SetLoss (m[0], m[1], 10);
  // -30 dB from 0 to 2 and -100 dB from 2 to 0
  loss.SetLoss (m[0], m[2], 30, /*symmetric = */ false);
  loss.SetLoss (m[2], m[0], 100, /*symmetric = */ false);
  // default from 1 to 2

  NS_TEST_ASSERT_MSG_EQ (loss.CalcRxPower (0, m[0], m[1]), -10, "Loss 0 -> 1 incorrect");
  NS_TEST_ASSERT_MSG_EQ (loss.CalcRxPower (0, m[1], m[0]), -10, "Loss 1 -> 0 incorrect");
  NS_TEST_ASSERT_MSG_EQ (loss.CalcRxPower (0, m[0], m[2]), -30, "Loss 0 -> 2 incorrect");
  NS_TEST_ASSERT_MSG_EQ (loss.CalcRxPower (0, m[2], m[0]), -100, "Loss 2 -> 0 incorrect");
  NS_TEST_ASSERT_MSG_EQ (loss.CalcRxPower (0, m[1], m[2]), 0, "Loss 1 -> 2 incorrect");
  NS_TEST_ASSERT_MSG_EQ (loss.CalcRxPower (0, m[2], m[1]), 0, "Loss 2 -> 1 incorrect");

  Simulator::Destroy ();
}

/**
 * \ingroup propagation-tests
 *
 * \brief RangePropagationLossModel Test
 */
class RangePropagationLossModelTestCase : public TestCase
{
public:
  RangePropagationLossModelTestCase ();
  virtual ~RangePropagationLossModelTestCase ();

private:
  virtual void DoRun (void);
};

RangePropagationLossModelTestCase::RangePropagationLossModelTestCase ()
  : TestCase ("Test RangePropagationLossModel")
{
}

RangePropagationLossModelTestCase::~RangePropagationLossModelTestCase ()
{
}

void
RangePropagationLossModelTestCase::DoRun (void)
{
  Config::SetDefault ("ns3::RangePropagationLossModel::MaxRange", DoubleValue (127.2));
  Ptr<MobilityModel> a = CreateObject<ConstantPositionMobilityModel> (); 
  a->SetPosition (Vector (0,0,0));
  Ptr<MobilityModel> b = CreateObject<ConstantPositionMobilityModel> (); 
  b->SetPosition (Vector (127.1,0,0));  // within range

  Ptr<RangePropagationLossModel> lossModel = CreateObject<RangePropagationLossModel> (); 

  double txPwrdBm = -80.0;
  double tolerance = 1e-6;
  double resultdBm = lossModel->CalcRxPower (txPwrdBm, a, b);
  NS_TEST_EXPECT_MSG_EQ_TOL (resultdBm, txPwrdBm, tolerance, "Got unexpected rcv power");
  b->SetPosition (Vector (127.25,0,0));  // beyond range
  resultdBm = lossModel->CalcRxPower (txPwrdBm, a, b);
  NS_TEST_EXPECT_MSG_EQ_TOL (resultdBm, -1000.0, tolerance, "Got unexpected rcv power");
  Simulator::Destroy ();
}

/**
 * \ingroup propagation-tests
 *
 * \brief Propagation models TestSuite
 * 
 * This TestSuite tests the following models:
 *   - FriisPropagationLossModel
 *   - TwoRayGroundPropagationLossModel
 *   - LogDistancePropagationLossModel
 *   - MatrixPropagationLossModel
 *   - RangePropagationLossModel
 */
class PropagationLossModelsTestSuite : public TestSuite
{
public:
  PropagationLossModelsTestSuite ();
};

PropagationLossModelsTestSuite::PropagationLossModelsTestSuite ()
  : TestSuite ("propagation-loss-model", UNIT)
{
  AddTestCase (new FriisPropagationLossModelTestCase, TestCase::QUICK);
  AddTestCase (new TwoRayGroundPropagationLossModelTestCase, TestCase::QUICK);
  AddTestCase (new LogDistancePropagationLossModelTestCase, TestCase::QUICK);
  AddTestCase (new MatrixPropagationLossModelTestCase, TestCase::QUICK);
  AddTestCase (new RangePropagationLossModelTestCase, TestCase::QUICK);
}

/// Static variable for test initialization
static PropagationLossModelsTestSuite g_propagationLossModelsTestSuite;
