/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
#include "ns3/buildings-channel-condition-model.h"
#include "ns3/channel-condition-model.h"
#include "ns3/three-gpp-v2v-channel-condition-model.h"
#include "ns3/three-gpp-v2v-propagation-loss-model.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/buildings-module.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/double.h"
#include "ns3/uinteger.h"
#include "ns3/boolean.h"
#include "ns3/core-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("ThreeGppV2vChannelConditionModelsTest");

/**
 * \ingroup building-test
 * \ingroup tests
 * 
 * Test case for the classes ThreeGppV2vUrbanChannelConditionModel,
 * and ThreeGppV2vHighwayChannelConditionModel to test their code to
 * deterministically determine NLOS state. The test checks if the
 * channel condition is correctly determined when a building is deployed in the
 * scenario. Methodology from buildings-channel-condition-model-test.cc is used.
 */
class ThreeGppV2vBuildingsChCondModelTestCase : public TestCase
{
public:
  /**
   * Constructor
   */
  ThreeGppV2vBuildingsChCondModelTestCase ();

  /**
   * Destructor
   */
  virtual ~ThreeGppV2vBuildingsChCondModelTestCase ();

private:
  /**
   * Builds the simulation scenario and perform the tests
   */
  virtual void DoRun (void);

  /**
   * Struct containing the parameters for each test
   */
  typedef struct
  {
    Vector m_positionA; //!< the position of the first node
    Vector m_positionB; //!< the position of the second node
    ChannelCondition::LosConditionValue m_losCond; //!< the correct channel condition
    TypeId m_typeId; //!< the type ID of the channel condition model to be used
  } TestVector;

  TestVectors<TestVector> m_testVectors; //!< array containg all the test vectors
};

ThreeGppV2vBuildingsChCondModelTestCase::ThreeGppV2vBuildingsChCondModelTestCase ()
  : TestCase ("Test case for the ThreeGppV2vUrban and ThreeGppV2vHighway ChannelConditionModel with building"), m_testVectors ()
{}

ThreeGppV2vBuildingsChCondModelTestCase::~ThreeGppV2vBuildingsChCondModelTestCase ()
{}

void
ThreeGppV2vBuildingsChCondModelTestCase::DoRun (void)
{
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (1);

  TestVector testVector;
  //Add vectors for ThreeGppV2vUrbanChannelConditionModel
  testVector.m_positionA = Vector (-5.0, 5.0, 1.5);
  testVector.m_positionB = Vector (20.0, 5.0, 1.5);
  testVector.m_losCond = ChannelCondition::LosConditionValue::NLOS;
  testVector.m_typeId = ThreeGppV2vUrbanChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0.0, 11.0, 1.5);
  testVector.m_positionB = Vector (4.0, 11.0, 1.5);
  testVector.m_losCond = ChannelCondition::LosConditionValue::LOS;
  testVector.m_typeId = ThreeGppV2vUrbanChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0.0, 11.0, 1.5);
  testVector.m_positionB = Vector (1000.0, 11.0, 1.5);
  testVector.m_losCond = ChannelCondition::LosConditionValue::NLOSv;
  testVector.m_typeId = ThreeGppV2vUrbanChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  //Now add same vectors for ThreeGppV2vHighwayChannelConditionModel
  testVector.m_positionA = Vector (-5.0, 5.0, 1.5);
  testVector.m_positionB = Vector (20.0, 5.0, 1.5);
  testVector.m_losCond = ChannelCondition::LosConditionValue::NLOS;
  testVector.m_typeId = ThreeGppV2vHighwayChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0.0, 11.0, 1.5);
  testVector.m_positionB = Vector (4.0, 11.0, 1.5);
  testVector.m_losCond = ChannelCondition::LosConditionValue::LOS;
  testVector.m_typeId = ThreeGppV2vHighwayChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0.0, 11.0, 1.5);
  testVector.m_positionB = Vector (1000.0, 11.0, 1.5);
  testVector.m_losCond = ChannelCondition::LosConditionValue::NLOSv;
  testVector.m_typeId = ThreeGppV2vHighwayChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  // create the factory for the channel condition models
  ObjectFactory condModelFactory;

  // Deploy nodes and building and get the channel condition
  NodeContainer nodes;
  nodes.Create (2);

  Ptr<MobilityModel> a = CreateObject<ConstantPositionMobilityModel> ();
  nodes.Get (0)->AggregateObject (a);

  Ptr<MobilityModel> b = CreateObject<ConstantPositionMobilityModel> ();
  nodes.Get (1)->AggregateObject (b);

  Ptr<Building> building = Create<Building> ();
  building->SetNRoomsX (1);
  building->SetNRoomsY (1);
  building->SetNFloors (1);
  building->SetBoundaries (Box (0.0, 10.0, 0.0, 10.0, 0.0, 5.0));

  BuildingsHelper::Install (nodes);

  for (uint32_t i = 0; i < m_testVectors.GetN (); ++i)
    {
      testVector = m_testVectors.Get (i);
      condModelFactory.SetTypeId (testVector.m_typeId);
      Ptr<ChannelConditionModel> condModel = DynamicCast<ChannelConditionModel> (condModelFactory.Create ());
      condModel->AssignStreams (1);

      a->SetPosition (testVector.m_positionA);
      b->SetPosition (testVector.m_positionB);
      Ptr<MobilityBuildingInfo> buildingInfoA = a->GetObject<MobilityBuildingInfo> ();
      buildingInfoA->MakeConsistent (a);
      Ptr<MobilityBuildingInfo> buildingInfoB = b->GetObject<MobilityBuildingInfo> ();
      buildingInfoB->MakeConsistent (b);
      Ptr<ChannelCondition> cond;
      cond = condModel->GetChannelCondition (a, b);

      NS_LOG_DEBUG ("Got " << cond->GetLosCondition () << " expected condition " << testVector.m_losCond);
      NS_TEST_ASSERT_MSG_EQ (cond->GetLosCondition (), testVector.m_losCond, "Got unexpected channel condition");
    }

  Simulator::Destroy ();
}

/**
 * Test case for the 3GPP V2V Urban channel condition models (probabilistic
 * model for LOS/NLOSv states). It determines the channel condition multiple times,
 * estimates the LOS probability and compares it with the value given by the
 * formulas in 3GPP TR 37.885, Table 6.2-1. Methodology from channel-condition-model-
 * test-suite.cc is used.
 */
class ThreeGppV2vUrbanLosNlosvChCondModelTestCase : public TestCase
{
public:
  /**
   * Constructor
   */
  ThreeGppV2vUrbanLosNlosvChCondModelTestCase ();

  /**
   * Destructor
   */
  virtual ~ThreeGppV2vUrbanLosNlosvChCondModelTestCase ();

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
  Ptr<ThreeGppV2vUrbanChannelConditionModel> m_condModel; //!< the channel condition model
  uint64_t m_numLos {0}; //!< the number of LOS occurrences
  double m_tolerance; //!< tolerance
};

ThreeGppV2vUrbanLosNlosvChCondModelTestCase::ThreeGppV2vUrbanLosNlosvChCondModelTestCase ()
  : TestCase ("Test case for the class ThreeGppV2vUrbanChannelConditionModel"),
    m_testVectors (),
    m_tolerance (2e-3)
{}

ThreeGppV2vUrbanLosNlosvChCondModelTestCase::~ThreeGppV2vUrbanLosNlosvChCondModelTestCase ()
{}

void
ThreeGppV2vUrbanLosNlosvChCondModelTestCase::EvaluateChannelCondition (Ptr<MobilityModel> a, Ptr<MobilityModel> b)
{
  Ptr<ChannelCondition> cond = m_condModel->GetChannelCondition (a, b);
  if (cond->GetLosCondition () == ChannelCondition::LosConditionValue::LOS)
    {
      m_numLos++;
    }
}

void
ThreeGppV2vUrbanLosNlosvChCondModelTestCase::DoRun (void)
{
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (1);

  // create the test vector
  TestVector testVector;

  // tests for the V2v Urban scenario
  testVector.m_positionA = Vector (0, 0, 1.6);
  testVector.m_positionB = Vector (10, 0, 1.6);
  testVector.m_pLos = std::min (1.0, 1.05 * exp (-0.0114 * 10.0));
  testVector.m_typeId = ThreeGppV2vUrbanChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 1.6);
  testVector.m_positionB = Vector (100, 0, 1.6);
  testVector.m_pLos = std::min (1.0, 1.05 * exp (-0.0114 * 100.0));
  testVector.m_typeId = ThreeGppV2vUrbanChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 1.6);
  testVector.m_positionB = Vector (1000, 0, 1.6);
  testVector.m_pLos = std::min (1.0, 1.05 * exp (-0.0114 * 1000.0));
  testVector.m_typeId = ThreeGppV2vUrbanChannelConditionModel::GetTypeId ();
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

  BuildingsHelper::Install (nodes);

  // Get the channel condition multiple times and compute the LOS probability
  uint32_t numberOfReps = 500000;
  for (uint32_t i = 0; i < m_testVectors.GetN (); ++i)
    {
      testVector = m_testVectors.Get (i);

      // set the distance between the two nodes
      a->SetPosition (testVector.m_positionA);
      b->SetPosition (testVector.m_positionB);
      Ptr<MobilityBuildingInfo> buildingInfoA = a->GetObject<MobilityBuildingInfo> ();
      buildingInfoA->MakeConsistent (a);
      Ptr<MobilityBuildingInfo> buildingInfoB = b->GetObject<MobilityBuildingInfo> ();
      buildingInfoB->MakeConsistent (b);

      // create the channel condition model
      condModelFactory.SetTypeId (testVector.m_typeId);
      m_condModel = condModelFactory.Create<ThreeGppV2vUrbanChannelConditionModel> ();
      m_condModel->SetAttribute ("UpdatePeriod", TimeValue (MilliSeconds (9)));
      m_condModel->AssignStreams (1);

      m_numLos = 0;
      for (uint32_t j = 0; j < numberOfReps; j++)
        {
          Simulator::Schedule (MilliSeconds (10 * j), &ThreeGppV2vUrbanLosNlosvChCondModelTestCase::EvaluateChannelCondition, this, a, b);
        }

      Simulator::Run ();
      Simulator::Destroy ();

      double resultPlos = double (m_numLos) / double (numberOfReps);
      NS_LOG_DEBUG (testVector.m_typeId << "  a pos " << testVector.m_positionA << " b pos " << testVector.m_positionB << " numLos " << m_numLos << " numberOfReps " << numberOfReps << " resultPlos " << resultPlos << " ref " << testVector.m_pLos);
      NS_TEST_EXPECT_MSG_EQ_TOL (resultPlos, testVector.m_pLos, m_tolerance, "Got unexpected LOS probability");
    }
}

/**
 * Test case for the 3GPP V2V Highway channel condition models (probabilistic
 * model for LOS/NLOSv states). It determines the channel condition multiple times,
 * estimates the LOS probability and compares it with the value given by the
 * formulas in 3GPP TR 37.885, Table 6.2-1. Methodology from channel-condition-model-
 * test-suite.cc is used.
 */
class ThreeGppV2vHighwayLosNlosvChCondModelTestCase : public TestCase
{
public:
  /**
   * Constructor
   */
  ThreeGppV2vHighwayLosNlosvChCondModelTestCase ();

  /**
   * Destructor
   */
  virtual ~ThreeGppV2vHighwayLosNlosvChCondModelTestCase ();

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
  Ptr<ThreeGppV2vHighwayChannelConditionModel> m_condModel; //!< the channel condition model
  uint64_t m_numLos {0}; //!< the number of LOS occurrences
  double m_tolerance; //!< tolerance
};

ThreeGppV2vHighwayLosNlosvChCondModelTestCase::ThreeGppV2vHighwayLosNlosvChCondModelTestCase ()
  : TestCase ("Test case for the class ThreeGppV2vHighwayChannelConditionModel"),
    m_testVectors (),
    m_tolerance (2e-3)
{}

ThreeGppV2vHighwayLosNlosvChCondModelTestCase::~ThreeGppV2vHighwayLosNlosvChCondModelTestCase ()
{}

void
ThreeGppV2vHighwayLosNlosvChCondModelTestCase::EvaluateChannelCondition (Ptr<MobilityModel> a, Ptr<MobilityModel> b)
{
  Ptr<ChannelCondition> cond = m_condModel->GetChannelCondition (a, b);
  if (cond->GetLosCondition () == ChannelCondition::LosConditionValue::LOS)
    {
      m_numLos++;
    }
}

void
ThreeGppV2vHighwayLosNlosvChCondModelTestCase::DoRun (void)
{
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (1);

  // create the test vector
  TestVector testVector;

  // tests for the V2v Highway scenario
  testVector.m_positionA = Vector (0, 0, 1.6);
  testVector.m_positionB = Vector (10, 0, 1.6);
  testVector.m_pLos = std::min (1.0, 0.0000021013 * 10.0 * 10.0 - 0.002 * 10.0 + 1.0193);
  testVector.m_typeId = ThreeGppV2vHighwayChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 1.6);
  testVector.m_positionB = Vector (100, 0, 1.6);
  testVector.m_pLos = std::min (1.0, 0.0000021013 * 100.0 * 100.0 - 0.002 * 100.0 + 1.0193);
  testVector.m_typeId = ThreeGppV2vHighwayChannelConditionModel::GetTypeId ();
  m_testVectors.Add (testVector);

  testVector.m_positionA = Vector (0, 0, 1.6);
  testVector.m_positionB = Vector (1000, 0, 1.6);
  testVector.m_pLos = std::max (0.0, 0.54 - 0.001 * (1000.0 - 475));
  testVector.m_typeId = ThreeGppV2vHighwayChannelConditionModel::GetTypeId ();
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

  BuildingsHelper::Install (nodes);

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
      m_condModel = condModelFactory.Create<ThreeGppV2vHighwayChannelConditionModel> ();
      m_condModel->SetAttribute ("UpdatePeriod", TimeValue (MilliSeconds (9)));
      m_condModel->AssignStreams (1);

      m_numLos = 0;
      for (uint32_t j = 0; j < numberOfReps; j++)
        {
          Simulator::Schedule (MilliSeconds (10 * j), &ThreeGppV2vHighwayLosNlosvChCondModelTestCase::EvaluateChannelCondition, this, a, b);
        }

      Simulator::Run ();
      Simulator::Destroy ();

      double resultPlos = static_cast<double> (m_numLos) / static_cast<double> (numberOfReps);
      NS_LOG_DEBUG (testVector.m_typeId << "  a pos " << testVector.m_positionA << " b pos " << testVector.m_positionB << " numLos " << m_numLos << " numberOfReps " << numberOfReps << " resultPlos " << resultPlos << " ref " << testVector.m_pLos);
      NS_TEST_EXPECT_MSG_EQ_TOL (resultPlos, testVector.m_pLos, m_tolerance, "Got unexpected LOS probability");
    }
}


/**
 * \ingroup building-test
 * \ingroup tests
 * 
 * Test suite for the 3GPP V2V channel condition model
 *
 * Note that, in 3GPP V2V scenarios, the channel condition model is
 * determined based on a two step procedure: 1st) NLOS state is determined
 * based on a deterministic model (using buildings), and 2nd) the LOS or NLOSv
 * state is determined based on a probabilistic model (using 3GPP formulas), in
 * case that the vehicles are not in NLOS condition.
 *
 * The test ThreeGppV2vBuildingsChCondModelTestCase checks the
 * 1st step of the procedure, the deterministic one, using buildings for
 * both \link ns3::ThreeGppV2vUrbanChannelConditionModel \endlink and
 * \link ns3::ThreeGppV2vHighwayChannelConditionModel \endlink .
 *
 * The tests ThreeGppV2vUrbanLosNlosvChCondModelTestCase and
 * ThreeGppV2vHighwayLosNlosvChCondModelTestCase check the
 * 2nd step of the procedure, the probabilistic one, without buildings, for
 * the V2V Urban and V2V Highway scenarios, respectively.
 *
 */
class ThreeGppV2vChCondModelsTestSuite : public TestSuite
{
public:
  ThreeGppV2vChCondModelsTestSuite ();
};

ThreeGppV2vChCondModelsTestSuite::ThreeGppV2vChCondModelsTestSuite ()
  : TestSuite ("three-gpp-v2v-channel-condition-model", SYSTEM)
{
  AddTestCase (new ThreeGppV2vBuildingsChCondModelTestCase, TestCase::QUICK);  // test for the deterministic procedure (NLOS vs LOS/NLOSv), based on buildings
  AddTestCase (new ThreeGppV2vUrbanLosNlosvChCondModelTestCase, TestCase::QUICK);   // test for the probabilistic procedure (LOS vs NLOSv), in V2V urban scenario
  AddTestCase (new ThreeGppV2vHighwayLosNlosvChCondModelTestCase, TestCase::QUICK); // test for the probabilistic procedure (LOS vs NLOSv), in V2V highway scenario
}

/// Static variable for test initialization
static ThreeGppV2vChCondModelsTestSuite ThreeGppV2vChCondModelsTestSuite;
