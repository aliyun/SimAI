/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 SIGNET Lab, Department of Information Engineering,
 * University of Padova
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
#include "ns3/channel-condition-model.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/simulator.h"
#include "ns3/node-container.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("ChannelConditionModelsTest");

/**
 * \ingroup propagation-tests
 *
 * Test case for the 3GPP channel condition models. It determines the
 * channel condition multiple times, estimates the LOS probability and
 * compares it with the value given by the formulas in 3GPP TR 38.901,
 * Table Table 7.4.2-1.
 */
class ThreeGppChannelConditionModelTestCase : public TestCase
{
public:
  /**
   * Constructor
   */
  ThreeGppChannelConditionModelTestCase ();

  /**
   * Destructor
   */
  virtual ~ThreeGppChannelConditionModelTestCase ();

private:
  /**
   * Builds the simulation scenario and perform the tests
   */
  virtual void DoRun (void);

  /**
   * Evaluates the channel condition between two nodes by calling the method
   * GetChannelCondition on m_condModel. If the channel condition is LOS it
   * increments m_numLos
   * \param a the mobility model of the first node
   * \param b the mobility model of the second node
   */
  void EvaluateChannelCondition (Ptr<MobilityModel> a, Ptr<MobilityModel> b);

  /**
   * Struct containing the parameters for each test
   */
  typedef struct
  {
    Vector m_positionA; //!< the position of the first node
    Vector m_positionB; //!< the position of the second node
    double m_pLos;  //!< LOS probability
    TypeId m_typeId; //!< the type ID of the channel condition model to be used
  } TestVector;

  TestVectors<TestVector> m_testVectors; //!< array containing all the test vectors
  Ptr<ThreeGppChannelConditionModel> m_condModel; //!< the channel condition model
  uint64_t m_numLos; //!< the number of LOS occurrences
  double m_tolerance; //!< tolerance
};

ThreeGppChannelConditionModelTestCase::ThreeGppChannelConditionModelTestCase ()
  : TestCase ("Test case for the child classes of ThreeGppChannelConditionModel"),
    m_testVectors (),
    m_tolerance (2e-3)
{
}

ThreeGppChannelConditionModelTestCase::~ThreeGppChannelConditionModelTestCase ()
{
}

void
ThreeGppChannelConditionModelTestCase::EvaluateChannelCondition (Ptr<MobilityModel> a, Ptr<MobilityModel> b)
{
  Ptr<ChannelCondition> cond = m_condModel->GetChannelCondition (a, b);
  if (cond->GetLosCondition () == ChannelCondition::LosConditionValue::LOS)
    {
      m_numLos++;
    }
}

void
ThreeGppChannelConditionModelTestCase::DoRun (void)
{
  // create the test vector
  TestVector testVector;

  // tests for the RMa scenario
  testVector.m_positionA = Vector (0, 0, 35.0);
  testVector.m_positionB = Vector (10, 0, 1.5);
  testVector.m_pLos = 1;
  testVector.m_typeId = ThreeGppRmaChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 35.0);
  testVector.m_positionB = Vector (100, 0, 1.5);
  testVector.m_pLos = exp (-(100.0 - 10.0) / 1000.0);
  testVector.m_typeId = ThreeGppRmaChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 35.0);
  testVector.m_positionB = Vector (1000, 0, 1.5);
  testVector.m_pLos = exp (-(1000.0 - 10.0) / 1000.0);
  testVector.m_typeId = ThreeGppRmaChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  // tests for the UMa scenario
  testVector.m_positionA = Vector (0, 0, 25.0);
  testVector.m_positionB = Vector (18, 0, 1.5);
  testVector.m_pLos = 1;
  testVector.m_typeId = ThreeGppUmaChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 25.0);
  testVector.m_positionB = Vector (50, 0, 1.5);
  testVector.m_pLos = (18.0 / 50.0 + exp (-50.0 / 63.0) * (1.0 - 18.0 / 50.0)) * (1.0 + 0);
  testVector.m_typeId = ThreeGppUmaChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 25.0);
  testVector.m_positionB = Vector (50, 0, 15);
  testVector.m_pLos = (18.0 / 50.0 + exp (-50.0 / 63.0) * (1.0 - 18.0 / 50.0)) * (1.0 + pow (2.0 / 10.0, 1.5) * 5.0 / 4.0 * pow (50.0 / 100.0, 3) * exp (-50.0 / 150.0));
  testVector.m_typeId = ThreeGppUmaChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 25.0);
  testVector.m_positionB = Vector (100, 0, 1.5);
  testVector.m_pLos = (18.0 / 100.0 + exp (-100.0 / 63.0) * (1.0 - 18.0 / 100.0)) * (1.0 + 0);
  testVector.m_typeId = ThreeGppUmaChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 25.0);
  testVector.m_positionB = Vector (100, 0, 15);
  testVector.m_pLos = (18.0 / 100.0 + exp (-100.0 / 63.0) * (1.0 - 18.0 / 100.0)) * (1.0 + pow (2.0 / 10.0, 1.5) * 5.0 / 4.0 * 1.0 * exp (-100.0 / 150.0));
  testVector.m_typeId = ThreeGppUmaChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  // tests for the UMi-Street Canyon scenario
  testVector.m_positionA = Vector (0, 0, 10.0);
  testVector.m_positionB = Vector (18, 0, 1.5);
  testVector.m_pLos = 1;
  testVector.m_typeId = ThreeGppUmiStreetCanyonChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 10.0);
  testVector.m_positionB = Vector (50, 0, 1.5);
  testVector.m_pLos = (18.0 / 50.0 + exp (-50.0 / 36.0) * (1.0 - 18.0 / 50.0));
  testVector.m_typeId = ThreeGppUmiStreetCanyonChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  m_testVectors.Add (testVector);
  testVector.m_positionA = Vector (0, 0, 10.0);
  testVector.m_positionB = Vector (100, 0, 15);
  testVector.m_pLos = (18.0 / 100.0 + exp (-100.0 / 36.0) * (1.0 - 18.0 / 100.0));
  testVector.m_typeId = ThreeGppUmiStreetCanyonChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  // tests for the Indoor Mixed Office scenario
  testVector.m_positionA = Vector (0, 0, 2.0);
  testVector.m_positionB = Vector (1.2, 0, 1.5);
  testVector.m_pLos = 1;
  testVector.m_typeId = ThreeGppIndoorMixedOfficeChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 2.0);
  testVector.m_positionB = Vector (5, 0, 1.5);
  testVector.m_pLos = exp (-(5.0 - 1.2) / 4.7);
  testVector.m_typeId = ThreeGppIndoorMixedOfficeChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 2.0);
  testVector.m_positionB = Vector (10, 0, 1.5);
  testVector.m_pLos = exp (-(10.0 - 6.5) / 32.6) * 0.32;
  testVector.m_typeId = ThreeGppIndoorMixedOfficeChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  // tests for the Indoor Open Office scenario
  testVector.m_positionA = Vector (0, 0, 3.0);
  testVector.m_positionB = Vector (5, 0, 1.5);
  testVector.m_pLos = 1;
  testVector.m_typeId = ThreeGppIndoorOpenOfficeChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 3.0);
  testVector.m_positionB = Vector (30, 0, 1.5);
  testVector.m_pLos = exp (-(30.0 - 5.0) / 70.8);
  testVector.m_typeId = ThreeGppIndoorOpenOfficeChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 3.0);
  testVector.m_positionB = Vector (100, 0, 1.5);
  testVector.m_pLos = exp (-(100.0 - 49.0) / 211.7) * 0.54;
  testVector.m_typeId = ThreeGppIndoorOpenOfficeChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  // create the factory for the channel condition models
  ObjectFactory condModelFactory;

  // create the two nodes
  NodeContainer nodes;
  nodes.Create (2);

  // create the mobility models
  Ptr<MobilityModel> a = CreateObject<ConstantPositionMobilityModel> ();
  Ptr<MobilityModel> b = CreateObject<ConstantPositionMobilityModel> ();

  // aggregate the nodes and the mobility models
  nodes.Get (0)->AggregateObject (a);
  nodes.Get (1)->AggregateObject (b);

  // Get the channel condition multiple times and compute the LOS probability
  uint32_t numberOfReps = 500000;
  for (uint32_t i = 0; i < m_testVectors.GetN (); ++i)
    {
      testVector = m_testVectors.Get (i);

      // set the distance between the two nodes
      a->SetPosition (testVector.m_positionA);
      b->SetPosition (testVector.m_positionB);

      // create the channel condition model
      condModelFactory.SetTypeId (testVector.m_typeId);
      m_condModel = condModelFactory.Create<ThreeGppChannelConditionModel> ();
      m_condModel->SetAttribute ("UpdatePeriod", TimeValue (MilliSeconds (9)));

      m_numLos = 0;
      for (uint32_t j = 0; j < numberOfReps; j++)
        {
          Simulator::Schedule (MilliSeconds (10 * j), &ThreeGppChannelConditionModelTestCase::EvaluateChannelCondition, this, a, b);
        }

      Simulator::Run ();
      Simulator::Destroy ();

      double resultPlos = double (m_numLos) / double (numberOfReps);
      NS_LOG_DEBUG (testVector.m_typeId << "  a pos " << testVector.m_positionA << " b pos " << testVector.m_positionB << " numLos " << m_numLos << " numberOfReps " << numberOfReps << " resultPlos " << resultPlos << " ref " << testVector.m_pLos);
      NS_TEST_EXPECT_MSG_EQ_TOL (resultPlos, testVector.m_pLos, m_tolerance, "Got unexpected LOS probability");
    }
}

/**
 * \ingroup propagation-tests
 *
 * Test suite for the channel condition models
 */
class ChannelConditionModelsTestSuite : public TestSuite
{
public:
  ChannelConditionModelsTestSuite ();
};

ChannelConditionModelsTestSuite::ChannelConditionModelsTestSuite ()
  : TestSuite ("propagation-channel-condition-model", UNIT)
{
  AddTestCase (new ThreeGppChannelConditionModelTestCase, TestCase::QUICK);
}

/// Static variable for test initialization
static ChannelConditionModelsTestSuite g_channelConditionModelsTestSuite;
