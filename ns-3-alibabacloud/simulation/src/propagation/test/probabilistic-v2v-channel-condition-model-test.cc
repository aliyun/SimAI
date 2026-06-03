/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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

#include "ns3/abort.h"
#include "ns3/test.h"
#include "ns3/config.h"
#include "ns3/channel-condition-model.h"
#include "ns3/probabilistic-v2v-channel-condition-model.h"
#include "ns3/three-gpp-v2v-propagation-loss-model.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/double.h"
#include "ns3/uinteger.h"
#include "ns3/boolean.h"
#include "ns3/core-module.h"
#include "ns3/node-container.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("ProbabilisticV2vChannelConditionModelsTest");

/**
 * \ingroup propagation-tests
 * 
 * Test case for the V2V Urban channel condition models using a fully
 * probabilistic model to determine LOS, NLOS and NLOSv states. The test
 * determines the channel condition multiple times,
 * estimates the LOS and NLOSv probabilities and compares them with the values
 * given by M. Boban,  X.Gong, and  W. Xu, “Modeling the evolution of line-of-sight
 * blockage for V2V channels,” in IEEE 84th Vehicular Technology
 * Conference (VTC-Fall), 2016. Methodology from channel-condition-model-
 * test-suite.cc is used, extended to a system with three states.
 */
class V2vUrbanProbChCondModelTestCase : public TestCase
{
public:
  /**
   * Constructor
   */
  V2vUrbanProbChCondModelTestCase ();

  /**
   * Destructor
   */
  virtual ~V2vUrbanProbChCondModelTestCase ();

private:
  /**
   * Builds the simulation scenario and perform the tests
   */
  virtual void DoRun (void);

  /**
   * Evaluates the channel condition between two nodes by calling the method
   * GetChannelCondition on m_condModel. If the channel condition is LOS it
   * increments m_numLos, if NLOSv it increments m_numNlosv
   * \param a the mobility model of the first node
   * \param b the mobility model of the second node
   */
  void EvaluateChannelCondition (Ptr<MobilityModel> a, Ptr<MobilityModel> b);

  /**
   * Struct containing the parameters for each test
   */
  struct TestVector
  {
    Vector m_positionA; //!< the position of the first node
    Vector m_positionB; //!< the position of the second node
    double m_pLos {0.0};  //!< LOS probability
    double m_pNlosv {0.0};  //!< NLOSv probability
    std::string m_density;  //!< the vehicles density
    TypeId m_typeId; //!< the type ID of the channel condition model to be used
  };

  TestVectors<TestVector> m_testVectors; //!< array containing all the test vectors
  Ptr<ProbabilisticV2vUrbanChannelConditionModel> m_condModel; //!< the channel condition model
  uint64_t m_numLos {0}; //!< the number of LOS occurrences
  uint64_t m_numNlosv {0}; //!< the number of NLOSv occurrences
  double m_tolerance; //!< tolerance
};

V2vUrbanProbChCondModelTestCase::V2vUrbanProbChCondModelTestCase ()
  : TestCase ("Test case for the class ProbabilisticV2vUrbanChannelConditionModel"),
    m_testVectors (),
    m_tolerance (5e-3)
{}

V2vUrbanProbChCondModelTestCase::~V2vUrbanProbChCondModelTestCase ()
{}

void
V2vUrbanProbChCondModelTestCase::EvaluateChannelCondition (Ptr<MobilityModel> a, Ptr<MobilityModel> b)
{
  Ptr<ChannelCondition> cond = m_condModel->GetChannelCondition (a, b);
  if (cond->GetLosCondition () == ChannelCondition::LosConditionValue::LOS)
    {
      m_numLos++;
    }
  else if (cond->GetLosCondition () == ChannelCondition::LosConditionValue::NLOSv)
    {
      m_numNlosv++;
    }
}

void
V2vUrbanProbChCondModelTestCase::DoRun (void)
{
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (1);

  // create the test vector
  TestVector testVector;

  // tests for the V2v Urban scenario
  testVector.m_positionA = Vector (0, 0, 1.6);
  testVector.m_positionB = Vector (10, 0, 1.6);
  testVector.m_pLos = std::min (1.0, std::max (0.0, 0.8548 * exp (-0.0064 * 10.0)));
  testVector.m_typeId = ProbabilisticV2vUrbanChannelConditionModel::GetTypeId ();
  testVector.m_pNlosv = std::min (1.0, std::max (0.0, 1 / (0.0396 * 10.0) * exp (-(log (10.0) - 5.2718) * (log (10.0) - 5.2718) / 3.4827)));
  testVector.m_density = "Low";
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 1.6);
  testVector.m_positionB = Vector (100, 0, 1.6);
  testVector.m_pLos = std::min (1.0, std::max (0.0, 0.8548 * exp (-0.0064 * 100.0)));
  testVector.m_typeId = ProbabilisticV2vUrbanChannelConditionModel::GetTypeId ();
  testVector.m_pNlosv = std::min (1.0, std::max (0.0, 1 / (0.0396 * 100.0) * exp (-(log (100.0) - 5.2718) * (log (100.0) - 5.2718) / 3.4827)));
  testVector.m_density = "Low";
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 1.6);
  testVector.m_positionB = Vector (10, 0, 1.6);
  testVector.m_pLos = std::min (1.0, std::max (0.0, 0.8372 * exp (-0.0114 * 10.0)));
  testVector.m_typeId = ProbabilisticV2vUrbanChannelConditionModel::GetTypeId ();
  testVector.m_pNlosv = std::min (1.0, std::max (0.0, 1 / (0.0312 * 10.0) * exp (-(log (10.0) - 5.0063) * (log (10.0) - 5.0063) / 2.4544)));
  testVector.m_density = "Medium";
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 1.6);
  testVector.m_positionB = Vector (100, 0, 1.6);
  testVector.m_pLos = std::min (1.0, std::max (0.0, 0.8372 * exp (-0.0114 * 100.0)));
  testVector.m_typeId = ProbabilisticV2vUrbanChannelConditionModel::GetTypeId ();
  testVector.m_pNlosv = std::min (1.0, std::max (0.0, 1 / (0.0312 * 100.0) * exp (-(log (100.0) - 5.0063) * (log (100.0) - 5.0063) / 2.4544)));
  testVector.m_density = "Medium";
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 1.6);
  testVector.m_positionB = Vector (10, 0, 1.6);
  testVector.m_pLos = std::min (1.0, std::max (0.0, 0.8962 * exp (-0.017 * 10.0)));
  testVector.m_typeId = ProbabilisticV2vUrbanChannelConditionModel::GetTypeId ();
  testVector.m_pNlosv = std::min (1.0, std::max (0.0, 1 / (0.0242 * 10.0) * exp (-(log (10.0) - 5.0115) * (log (10.0) - 5.0115) / 2.2092)));
  testVector.m_density = "High";
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 1.6);
  testVector.m_positionB = Vector (100, 0, 1.6);
  testVector.m_pLos = std::min (1.0, std::max (0.0, 0.8962 * exp (-0.017 * 100.0)));
  testVector.m_typeId = ProbabilisticV2vUrbanChannelConditionModel::GetTypeId ();
  testVector.m_pNlosv = std::min (1.0, std::max (0.0, 1 / (0.0242 * 100.0) * exp (-(log (100.0) - 5.0115) * (log (100.0) - 5.0115) / 2.2092)));
  testVector.m_density = "High";
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
      m_condModel = condModelFactory.Create<ProbabilisticV2vUrbanChannelConditionModel> ();
      m_condModel->SetAttribute ("UpdatePeriod", TimeValue (MilliSeconds (9)));
      m_condModel->AssignStreams (1);
      m_condModel->SetAttribute ("Density", StringValue (testVector.m_density));

      m_numLos = 0;
      m_numNlosv = 0;
      for (uint32_t j = 0; j < numberOfReps; j++)
        {
          Simulator::Schedule (MilliSeconds (10 * j), &V2vUrbanProbChCondModelTestCase::EvaluateChannelCondition, this, a, b);
        }

      Simulator::Run ();
      Simulator::Destroy ();

      double resultPlos = double (m_numLos) / double (numberOfReps);
      double resultPnlosv = double (m_numNlosv) / double (numberOfReps);
      NS_LOG_DEBUG (testVector.m_typeId << "  a pos " << testVector.m_positionA << " b pos " << testVector.m_positionB << " numLos " << m_numLos << " numberOfReps " << numberOfReps << " resultPlos " << resultPlos << " ref " << testVector.m_pLos);
      NS_TEST_EXPECT_MSG_EQ_TOL (resultPlos, testVector.m_pLos, m_tolerance, "Got unexpected LOS probability");
      NS_LOG_DEBUG (testVector.m_typeId << "  a pos " << testVector.m_positionA << " b pos " << testVector.m_positionB << " numNlosv " << m_numNlosv << " numberOfReps " << numberOfReps << " resultPnlosv " << resultPnlosv << " ref " << testVector.m_pNlosv);
      NS_TEST_EXPECT_MSG_EQ_TOL (resultPnlosv, testVector.m_pNlosv, m_tolerance, "Got unexpected NLOSv probability");
    }
}

/**
 * \ingroup propagation-tests
 * 
 * Test case for the V2V Highway channel condition models using a fully
 * probabilistic model to determine LOS, NLOS and NLOSv states. The test
 * determines the channel condition multiple times,
 * estimates the LOS and NLOS probabilities and compares them with the values
 * given by M. Boban,  X.Gong, and  W. Xu, “Modeling the evolution of line-of-sight
 * blockage for V2V channels,” in IEEE 84th Vehicular Technology
 * Conference (VTC-Fall), 2016. Methodology from channel-condition-model-
 * test-suite.cc is used, extended to a system with three states.
 */
class V2vHighwayProbChCondModelTestCase : public TestCase
{
public:
  /**
   * Constructor
   */
  V2vHighwayProbChCondModelTestCase ();

  /**
   * Destructor
   */
  virtual ~V2vHighwayProbChCondModelTestCase ();

private:
  /**
   * Builds the simulation scenario and perform the tests
   */
  virtual void DoRun (void);

  /**
   * Evaluates the channel condition between two nodes by calling the method
   * GetChannelCondition on m_condModel. If the channel condition is LOS it
   * increments m_numLos, if NLOS it increments m_numNlos
   * \param a the mobility model of the first node
   * \param b the mobility model of the second node
   */
  void EvaluateChannelCondition (Ptr<MobilityModel> a, Ptr<MobilityModel> b);

  /**
   * Struct containing the parameters for each test
   */
  struct TestVector
  {
    Vector m_positionA; //!< the position of the first node
    Vector m_positionB; //!< the position of the second node
    double m_pLos {0.0};  //!< LOS probability
    double m_pNlos {0.0};  //!< NLOS probability
    std::string m_density;  //!< the vehicles density
    TypeId m_typeId; //!< the type ID of the channel condition model to be used
  };

  TestVectors<TestVector> m_testVectors; //!< array containing all the test vectors
  Ptr<ProbabilisticV2vHighwayChannelConditionModel> m_condModel; //!< the channel condition model
  uint64_t m_numLos {0}; //!< the number of LOS occurrences
  uint64_t m_numNlos {0}; //!< the number of NLOS occurrences
  double m_tolerance; //!< tolerance
};

V2vHighwayProbChCondModelTestCase::V2vHighwayProbChCondModelTestCase ()
  : TestCase ("Test case for the class ProbabilisticV2vHighwayChannelConditionModel"),
    m_testVectors (),
    m_tolerance (5e-3)
{}

V2vHighwayProbChCondModelTestCase::~V2vHighwayProbChCondModelTestCase ()
{}

void
V2vHighwayProbChCondModelTestCase::EvaluateChannelCondition (Ptr<MobilityModel> a, Ptr<MobilityModel> b)
{
  Ptr<ChannelCondition> cond = m_condModel->GetChannelCondition (a, b);
  if (cond->GetLosCondition () == ChannelCondition::LosConditionValue::LOS)
    {
      m_numLos++;
    }
  else if (cond->GetLosCondition () == ChannelCondition::LosConditionValue::NLOS)
    {
      m_numNlos++;
    }
}

void
V2vHighwayProbChCondModelTestCase::DoRun (void)
{
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (1);

  // create the test vector
  TestVector testVector;

  // tests for the V2v Highway scenario
  testVector.m_positionA = Vector (0, 0, 1.6);
  testVector.m_positionB = Vector (10, 0, 1.6);
  double aLos = 1.5e-6;
  double bLos = -0.0015;
  double cLos = 1.0;
  testVector.m_pLos = std::min (1.0, std::max (0.0, aLos * 10.0 * 10.0 + bLos * 10.0 + cLos));
  testVector.m_typeId = ProbabilisticV2vHighwayChannelConditionModel::GetTypeId ();
  double aNlos = -2.9e-7;
  double bNlos = 0.00059;
  double cNlos = 0.0017;
  testVector.m_pNlos = std::min (1.0, std::max (0.0, aNlos * 10.0 * 10.0 + bNlos * 10.0 + cNlos));
  testVector.m_density = "Low";
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 1.6);
  testVector.m_positionB = Vector (100, 0, 1.6);
  testVector.m_pLos = std::min (1.0, std::max (0.0, aLos * 100.0 * 100.0 + bLos * 100.0 + cLos));
  testVector.m_typeId = ProbabilisticV2vHighwayChannelConditionModel::GetTypeId ();
  testVector.m_pNlos = std::min (1.0, std::max (0.0, aNlos * 100.0 * 100.0 + bNlos * 100.0 + cNlos));
  testVector.m_density = "Low";
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 1.6);
  testVector.m_positionB = Vector (10, 0, 1.6);
  aLos = 2.7e-6;
  bLos = -0.0025;
  cLos = 1.0;
  testVector.m_pLos = std::min (1.0, std::max (0.0, aLos * 10.0 * 10.0 + bLos * 10.0 + cLos));
  testVector.m_typeId = ProbabilisticV2vHighwayChannelConditionModel::GetTypeId ();
  aNlos = -3.7e-7;
  bNlos = 0.00061;
  cNlos = 0.015;
  testVector.m_pNlos = std::min (1.0, std::max (0.0, aNlos * 10.0 * 10.0 + bNlos * 10.0 + cNlos));
  testVector.m_density = "Medium";
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 1.6);
  testVector.m_positionB = Vector (100, 0, 1.6);
  testVector.m_pLos = std::min (1.0, std::max (0.0, aLos * 100.0 * 100.0 + bLos * 100.0 + cLos));
  testVector.m_typeId = ProbabilisticV2vHighwayChannelConditionModel::GetTypeId ();
  testVector.m_pNlos = std::min (1.0, std::max (0.0, aNlos * 100.0 * 100.0 + bNlos * 100.0 + cNlos));
  testVector.m_density = "Medium";
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 1.6);
  testVector.m_positionB = Vector (10, 0, 1.6);
  aLos = 3.2e-6;
  bLos = -0.003;
  cLos = 1.0;
  testVector.m_pLos = std::min (1.0, std::max (0.0, aLos * 10.0 * 10.0 + bLos * 10.0 + cLos));
  testVector.m_typeId = ProbabilisticV2vHighwayChannelConditionModel::GetTypeId ();
  aNlos = -4.1e-7;
  bNlos = 0.00067;
  cNlos = 0.0;
  testVector.m_pNlos = std::min (1.0, std::max (0.0, aNlos * 10.0 * 10.0 + bNlos * 10.0 + cNlos));
  testVector.m_density = "High";
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 1.6);
  testVector.m_positionB = Vector (100, 0, 1.6);
  testVector.m_pLos = std::min (1.0, std::max (0.0, aLos * 100.0 * 100.0 + bLos * 100.0 + cLos));
  testVector.m_typeId = ProbabilisticV2vHighwayChannelConditionModel::GetTypeId ();
  testVector.m_pNlos = std::min (1.0, std::max (0.0, aNlos * 100.0 * 100.0 + bNlos * 100.0 + cNlos));
  testVector.m_density = "High";
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
      m_condModel = condModelFactory.Create<ProbabilisticV2vHighwayChannelConditionModel> ();
      m_condModel->SetAttribute ("UpdatePeriod", TimeValue (MilliSeconds (9)));
      m_condModel->AssignStreams (1);
      m_condModel->SetAttribute ("Density", StringValue (testVector.m_density));

      m_numLos = 0;
      m_numNlos = 0;
      for (uint32_t j = 0; j < numberOfReps; j++)
        {
          Simulator::Schedule (MilliSeconds (10 * j), &V2vHighwayProbChCondModelTestCase::EvaluateChannelCondition, this, a, b);
        }

      Simulator::Run ();
      Simulator::Destroy ();

      double resultPlos = double (m_numLos) / double (numberOfReps);
      double resultPnlos = double (m_numNlos) / double (numberOfReps);
      NS_LOG_DEBUG (testVector.m_typeId << "  a pos " << testVector.m_positionA << " b pos " << testVector.m_positionB << " numLos " << m_numLos << " numberOfReps " << numberOfReps << " resultPlos " << resultPlos << " ref " << testVector.m_pLos);
      NS_TEST_EXPECT_MSG_EQ_TOL (resultPlos, testVector.m_pLos, m_tolerance, "Got unexpected LOS probability");
      NS_LOG_DEBUG (testVector.m_typeId << "  a pos " << testVector.m_positionA << " b pos " << testVector.m_positionB << " numNlos " << m_numNlos << " numberOfReps " << numberOfReps << " resultPnlos " << resultPnlos << " ref " << testVector.m_pNlos);
      NS_TEST_EXPECT_MSG_EQ_TOL (resultPnlos, testVector.m_pNlos, m_tolerance, "Got unexpected NLOS probability");
    }
}

/**
 * \ingroup propagation-tests
 * 
 * Test suite for the probabilistic V2V channel condition model
 *
 * The tests V2vUrbanProbChCondModelTestCase and
 * V2vHighwayProbChCondModelTestCase test a fully probabilistic
 * model for V2V channel condition, for Urban and Highway V2V scenarios,
 * respectively. Basically, the model determines NLOS/LOS/NLOSv state based
 * on probability formulas, derived from: M. Boban,  X.Gong, and  W. Xu,
 * “Modeling the evolution of line-of-sight blockage for V2V channels,”
 * in IEEE 84th Vehicular Technology Conference (VTC-Fall), 2016.
 *
 */
class ProbabilisticV2vChCondModelsTestSuite : public TestSuite
{
public:
  ProbabilisticV2vChCondModelsTestSuite ();
};

ProbabilisticV2vChCondModelsTestSuite::ProbabilisticV2vChCondModelsTestSuite ()
  : TestSuite ("probabilistic-v2v-channel-condition-model", SYSTEM)
{
  AddTestCase (new V2vUrbanProbChCondModelTestCase, TestCase::QUICK);   // test for a fully probabilistic model (NLOS vs LOS vs NLOSv), in V2V urban scenario
  AddTestCase (new V2vHighwayProbChCondModelTestCase, TestCase::QUICK); // test for a fully probabilistic model (NLOS vs LOS vs NLOSv), in V2V highway scenario*/
}

/// Static variable for test initialization
static ProbabilisticV2vChCondModelsTestSuite g_probabilisticV2vChCondModelsTestSuite;
