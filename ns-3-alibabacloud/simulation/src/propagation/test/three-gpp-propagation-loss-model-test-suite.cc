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
#include "ns3/boolean.h"
#include "ns3/channel-condition-model.h"
#include "ns3/three-gpp-propagation-loss-model.h"
#include "ns3/three-gpp-v2v-propagation-loss-model.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/constant-velocity-mobility-model.h"
#include "ns3/mobility-helper.h"
#include "ns3/simulator.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("ThreeGppPropagationLossModelsTest");

/**
 * \ingroup propagation-tests
 * 
 * Test case for the class ThreeGppRmaPropagationLossModel.
 * It computes the pathloss between two nodes and compares it with the value
 * obtained using the formula in 3GPP TR 38.901, Table 7.4.1-1.
 */
class ThreeGppRmaPropagationLossModelTestCase : public TestCase
{
public:
  /**
   * Constructor
   */
  ThreeGppRmaPropagationLossModelTestCase ();

  /**
   * Destructor
   */
  virtual ~ThreeGppRmaPropagationLossModelTestCase ();

private:
  /**
   * Build the simulation scenario and run the tests
   */
  virtual void DoRun (void);

  /**
   * Struct containing the parameters for each test
   */
  typedef struct
  {
    double m_distance; //!< 2D distance between UT and BS in meters
    bool m_isLos; //!< if true LOS, if false NLOS
    double m_frequency; //!< carrier frequency in Hz
    double m_pt;  //!< transmitted power in dBm
    double m_pr;  //!< received power in dBm
  } TestVector;

  TestVectors<TestVector> m_testVectors; //!< array containing all the test vectors
  double m_tolerance; //!< tolerance
};

ThreeGppRmaPropagationLossModelTestCase::ThreeGppRmaPropagationLossModelTestCase ()
  : TestCase ("Test for the ThreeGppRmaPropagationLossModel class"),
  m_testVectors (),
  m_tolerance (5e-2)
{
}

ThreeGppRmaPropagationLossModelTestCase::~ThreeGppRmaPropagationLossModelTestCase ()
{
}

void
ThreeGppRmaPropagationLossModelTestCase::DoRun (void)
{
  TestVector testVector;

  testVector.m_distance = 10.0;
  testVector.m_isLos = true;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -77.3784;
  m_testVectors.Add (testVector);

  testVector.m_distance = 100.0;
  testVector.m_isLos = true;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -87.2965;
  m_testVectors.Add (testVector);

  testVector.m_distance = 1000.0;
  testVector.m_isLos = true;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -108.5577;
  m_testVectors.Add (testVector);

  testVector.m_distance = 10000.0;
  testVector.m_isLos = true;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -140.3896;
  m_testVectors.Add (testVector);

  testVector.m_distance = 10.0;
  testVector.m_isLos = false;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -77.3784;
  m_testVectors.Add (testVector);

  testVector.m_distance = 100.0;
  testVector.m_isLos = false;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -95.7718;
  m_testVectors.Add (testVector);

  testVector.m_distance = 1000.0;
  testVector.m_isLos = false;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -133.5223;
  m_testVectors.Add (testVector);

  testVector.m_distance = 5000.0;
  testVector.m_isLos = false;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -160.5169;
  m_testVectors.Add (testVector);

  // Create the nodes for BS and UT
  NodeContainer nodes;
  nodes.Create (2);

  // Create the mobility models
  Ptr<MobilityModel> a = CreateObject<ConstantPositionMobilityModel> ();
  nodes.Get (0)->AggregateObject (a);
  Ptr<MobilityModel> b = CreateObject<ConstantPositionMobilityModel> ();
  nodes.Get (1)->AggregateObject (b);

  // Use a deterministic channel condition model
  Ptr<ChannelConditionModel> losCondModel = CreateObject<AlwaysLosChannelConditionModel> ();
  Ptr<ChannelConditionModel> nlosCondModel = CreateObject<NeverLosChannelConditionModel> ();

  // Create the propagation loss model
  Ptr<ThreeGppRmaPropagationLossModel> lossModel = CreateObject<ThreeGppRmaPropagationLossModel> ();
  lossModel->SetAttribute ("ShadowingEnabled", BooleanValue (false)); // disable the shadow fading

  for (uint32_t i = 0; i < m_testVectors.GetN (); i++)
    {
      TestVector testVector = m_testVectors.Get (i);

      Vector posBs = Vector (0.0, 0.0, 35.0);
      Vector posUt = Vector (testVector.m_distance, 0.0, 1.5);

      // set the LOS or NLOS condition
      if (testVector.m_isLos)
        {
          lossModel->SetChannelConditionModel (losCondModel);
        }
      else
        {
          lossModel->SetChannelConditionModel (nlosCondModel);
        }

      a->SetPosition (posBs);
      b->SetPosition (posUt);

      lossModel->SetAttribute ("Frequency", DoubleValue (testVector.m_frequency));
      NS_TEST_EXPECT_MSG_EQ_TOL (lossModel->CalcRxPower (testVector.m_pt, a, b), testVector.m_pr, m_tolerance, "Got unexpected rcv power");
    }

  Simulator::Destroy ();
}

/**
 * \ingroup propagation-tests
 * 
 * Test case for the class ThreeGppUmaPropagationLossModel.
 * It computes the pathloss between two nodes and compares it with the value
 * obtained using the formula in 3GPP TR 38.901, Table 7.4.1-1.
 */
class ThreeGppUmaPropagationLossModelTestCase : public TestCase
{
public:
  /**
   * Constructor
   */
  ThreeGppUmaPropagationLossModelTestCase ();

  /**
   * Destructor
   */
  virtual ~ThreeGppUmaPropagationLossModelTestCase ();

private:
  /**
   * Build the simulation scenario and run the tests
   */
  virtual void DoRun (void);

  /**
   * Struct containing the parameters for each test
   */
  typedef struct
  {
    double m_distance; //!< 2D distance between UT and BS in meters
    bool m_isLos; //!< if true LOS, if false NLOS
    double m_frequency; //!< carrier frequency in Hz
    double m_pt; //!< transmitted power in dBm
    double m_pr; //!< received power in dBm
  } TestVector;

  TestVectors<TestVector> m_testVectors; //!< array containing all the test vectors
  double m_tolerance; //!< tolerance
};

ThreeGppUmaPropagationLossModelTestCase::ThreeGppUmaPropagationLossModelTestCase ()
  : TestCase ("Test for the ThreeGppUmaPropagationLossModel class"),
  m_testVectors (),
  m_tolerance (5e-2)
{
}

ThreeGppUmaPropagationLossModelTestCase::~ThreeGppUmaPropagationLossModelTestCase ()
{
}

void
ThreeGppUmaPropagationLossModelTestCase::DoRun (void)
{
  TestVector testVector;

  testVector.m_distance = 10.0;
  testVector.m_isLos = true;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -72.9380;
  m_testVectors.Add (testVector);

  testVector.m_distance = 100.0;
  testVector.m_isLos = true;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -86.2362;
  m_testVectors.Add (testVector);

  testVector.m_distance = 1000.0;
  testVector.m_isLos = true;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -109.7252;
  m_testVectors.Add (testVector);

  testVector.m_distance = 5000.0;
  testVector.m_isLos = true;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -137.6794;
  m_testVectors.Add (testVector);

  testVector.m_distance = 10.0;
  testVector.m_isLos = false;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -82.5131;
  m_testVectors.Add (testVector);

  testVector.m_distance = 100.0;
  testVector.m_isLos = false;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -106.1356;
  m_testVectors.Add (testVector);

  testVector.m_distance = 1000.0;
  testVector.m_isLos = false;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -144.7641;
  m_testVectors.Add (testVector);

  testVector.m_distance = 5000.0;
  testVector.m_isLos = false;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -172.0753;
  m_testVectors.Add (testVector);

  // Create the nodes for BS and UT
  NodeContainer nodes;
  nodes.Create (2);

  // Create the mobility models
  Ptr<MobilityModel> a = CreateObject<ConstantPositionMobilityModel> ();
  nodes.Get (0)->AggregateObject (a);
  Ptr<MobilityModel> b = CreateObject<ConstantPositionMobilityModel> ();
  nodes.Get (1)->AggregateObject (b);

  // Use a deterministic channel condition model
  Ptr<ChannelConditionModel> losCondModel = CreateObject<AlwaysLosChannelConditionModel> ();
  Ptr<ChannelConditionModel> nlosCondModel = CreateObject<NeverLosChannelConditionModel> ();

  // Create the propagation loss model
  Ptr<ThreeGppUmaPropagationLossModel> lossModel = CreateObject<ThreeGppUmaPropagationLossModel> ();
  lossModel->SetAttribute ("ShadowingEnabled", BooleanValue (false)); // disable the shadow fading

  for (uint32_t i = 0; i < m_testVectors.GetN (); i++)
    {
      TestVector testVector = m_testVectors.Get (i);

      Vector posBs = Vector (0.0, 0.0, 25.0);
      Vector posUt = Vector (testVector.m_distance, 0.0, 1.5);

      // set the LOS or NLOS condition
      if (testVector.m_isLos)
        {
          lossModel->SetChannelConditionModel (losCondModel);
        }
      else
        {
          lossModel->SetChannelConditionModel (nlosCondModel);
        }

      a->SetPosition (posBs);
      b->SetPosition (posUt);

      lossModel->SetAttribute ("Frequency", DoubleValue (testVector.m_frequency));
      NS_TEST_EXPECT_MSG_EQ_TOL (lossModel->CalcRxPower (testVector.m_pt, a, b), testVector.m_pr, m_tolerance, "Got unexpected rcv power");
    }

  Simulator::Destroy ();
}

/**
 * \ingroup propagation-tests
 * 
 * Test case for the class ThreeGppUmiStreetCanyonPropagationLossModel.
 * It computes the pathloss between two nodes and compares it with the value
 * obtained using the formula in 3GPP TR 38.901, Table 7.4.1-1.
 */
class ThreeGppUmiPropagationLossModelTestCase : public TestCase
{
public:
  /**
   * Constructor
   */
  ThreeGppUmiPropagationLossModelTestCase ();

  /**
   * Destructor
   */
  virtual ~ThreeGppUmiPropagationLossModelTestCase ();

private:
  /**
   * Build the simulation scenario and run the tests
   */
  virtual void DoRun (void);

  /**
   * Struct containing the parameters for each test
   */
  typedef struct
  {
    double m_distance; //!< 2D distance between UT and BS in meters
    bool m_isLos; //!< if true LOS, if false NLOS
    double m_frequency; //!< carrier frequency in Hz
    double m_pt; //!< transmitted power in dBm
    double m_pr; //!< received power in dBm
  } TestVector;

  TestVectors<TestVector> m_testVectors; //!< array containing all the test vectors
  double m_tolerance; //!< tolerance
};

ThreeGppUmiPropagationLossModelTestCase::ThreeGppUmiPropagationLossModelTestCase ()
  : TestCase ("Test for the ThreeGppUmiPropagationLossModel class"),
  m_testVectors (),
  m_tolerance (5e-2)
{
}

ThreeGppUmiPropagationLossModelTestCase::~ThreeGppUmiPropagationLossModelTestCase ()
{
}

void
ThreeGppUmiPropagationLossModelTestCase::DoRun (void)
{
  TestVector testVector;

  testVector.m_distance = 10.0;
  testVector.m_isLos = true;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -69.8591;
  m_testVectors.Add (testVector);

  testVector.m_distance = 100.0;
  testVector.m_isLos = true;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -88.4122;
  m_testVectors.Add (testVector);

  testVector.m_distance = 1000.0;
  testVector.m_isLos = true;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -119.3114;

  testVector.m_distance = 5000.0;
  testVector.m_isLos = true;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -147.2696;

  testVector.m_distance = 10.0;
  testVector.m_isLos = false;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -76.7563;

  testVector.m_distance = 100.0;
  testVector.m_isLos = false;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -107.9432;

  testVector.m_distance = 1000.0;
  testVector.m_isLos = false;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -143.1886;

  testVector.m_distance = 5000.0;
  testVector.m_isLos = false;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -167.8617;

  // Create the nodes for BS and UT
  NodeContainer nodes;
  nodes.Create (2);

  // Create the mobility models
  Ptr<MobilityModel> a = CreateObject<ConstantPositionMobilityModel> ();
  nodes.Get (0)->AggregateObject (a);
  Ptr<MobilityModel> b = CreateObject<ConstantPositionMobilityModel> ();
  nodes.Get (1)->AggregateObject (b);

  // Use a deterministic channel condition model
  Ptr<ChannelConditionModel> losCondModel = CreateObject<AlwaysLosChannelConditionModel> ();
  Ptr<ChannelConditionModel> nlosCondModel = CreateObject<NeverLosChannelConditionModel> ();

  // Create the propagation loss model
  Ptr<ThreeGppUmiStreetCanyonPropagationLossModel> lossModel = CreateObject<ThreeGppUmiStreetCanyonPropagationLossModel> ();
  lossModel->SetAttribute ("ShadowingEnabled", BooleanValue (false)); // disable the shadow fading

  for (uint32_t i = 0; i < m_testVectors.GetN (); i++)
    {
      TestVector testVector = m_testVectors.Get (i);

      Vector posBs = Vector (0.0, 0.0, 10.0);
      Vector posUt = Vector (testVector.m_distance, 0.0, 1.5);

      // set the LOS or NLOS condition
      if (testVector.m_isLos)
        {
          lossModel->SetChannelConditionModel (losCondModel);
        }
      else
        {
          lossModel->SetChannelConditionModel (nlosCondModel);
        }

      a->SetPosition (posBs);
      b->SetPosition (posUt);

      lossModel->SetAttribute ("Frequency", DoubleValue (testVector.m_frequency));
      NS_TEST_EXPECT_MSG_EQ_TOL (lossModel->CalcRxPower (testVector.m_pt, a, b), testVector.m_pr, m_tolerance, "Got unexpected rcv power");
    }

  Simulator::Destroy ();
}

/**
 * \ingroup propagation-tests
 * 
 * Test case for the class ThreeGppIndoorOfficePropagationLossModel.
 * It computes the pathloss between two nodes and compares it with the value
 * obtained using the formula in 3GPP TR 38.901, Table 7.4.1-1.
 */
class ThreeGppIndoorOfficePropagationLossModelTestCase : public TestCase
{
public:
  /**
   * Constructor
   */
  ThreeGppIndoorOfficePropagationLossModelTestCase ();

  /**
   * Destructor
   */
  virtual ~ThreeGppIndoorOfficePropagationLossModelTestCase ();

private:
  /**
   * Build the simulation scenario and run the tests
   */
  virtual void DoRun (void);

  /**
   * Struct containing the parameters for each test
   */
  typedef struct
  {
    double m_distance; //!< 2D distance between UT and BS in meters
    bool m_isLos; //!< if true LOS, if false NLOS
    double m_frequency; //!< carrier frequency in Hz
    double m_pt; //!< transmitted power in dBm
    double m_pr; //!< received power in dBm
  } TestVector;

  TestVectors<TestVector> m_testVectors; //!< array containing all the test vectors
  double m_tolerance; //!< tolerance
};

ThreeGppIndoorOfficePropagationLossModelTestCase::ThreeGppIndoorOfficePropagationLossModelTestCase ()
  : TestCase ("Test for the ThreeGppIndoorOfficePropagationLossModel class"),
  m_testVectors (),
  m_tolerance (5e-2)
{
}

ThreeGppIndoorOfficePropagationLossModelTestCase::~ThreeGppIndoorOfficePropagationLossModelTestCase ()
{
}

void
ThreeGppIndoorOfficePropagationLossModelTestCase::DoRun (void)
{
  TestVector testVector;

  testVector.m_distance = 1.0;
  testVector.m_isLos = true;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -50.8072;
  m_testVectors.Add (testVector);

  testVector.m_distance = 10.0;
  testVector.m_isLos = true;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -63.7630;
  m_testVectors.Add (testVector);

  testVector.m_distance = 50.0;
  testVector.m_isLos = true;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -75.7750;
  m_testVectors.Add (testVector);

  testVector.m_distance = 100.0;
  testVector.m_isLos = true;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -80.9802;
  m_testVectors.Add (testVector);

  testVector.m_distance = 1.0;
  testVector.m_isLos = false;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -50.8072;
  m_testVectors.Add (testVector);

  testVector.m_distance = 10.0;
  testVector.m_isLos = false;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -73.1894;
  m_testVectors.Add (testVector);

  testVector.m_distance = 50.0;
  testVector.m_isLos = false;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -99.7824;
  m_testVectors.Add (testVector);

  testVector.m_distance = 100.0;
  testVector.m_isLos = false;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -111.3062;
  m_testVectors.Add (testVector);

  // Create the nodes for BS and UT
  NodeContainer nodes;
  nodes.Create (2);

  // Create the mobility models
  Ptr<MobilityModel> a = CreateObject<ConstantPositionMobilityModel> ();
  nodes.Get (0)->AggregateObject (a);
  Ptr<MobilityModel> b = CreateObject<ConstantPositionMobilityModel> ();
  nodes.Get (1)->AggregateObject (b);

  // Use a deterministic channel condition model
  Ptr<ChannelConditionModel> losCondModel = CreateObject<AlwaysLosChannelConditionModel> ();
  Ptr<ChannelConditionModel> nlosCondModel = CreateObject<NeverLosChannelConditionModel> ();

  // Create the propagation loss model
  Ptr<ThreeGppIndoorOfficePropagationLossModel> lossModel = CreateObject<ThreeGppIndoorOfficePropagationLossModel> ();
  lossModel->SetAttribute ("ShadowingEnabled", BooleanValue (false)); // disable the shadow fading

  for (uint32_t i = 0; i < m_testVectors.GetN (); i++)
    {
      TestVector testVector = m_testVectors.Get (i);

      Vector posBs = Vector (0.0, 0.0, 3.0);
      Vector posUt = Vector (testVector.m_distance, 0.0, 1.5);

      // set the LOS or NLOS condition
      if (testVector.m_isLos)
        {
          lossModel->SetChannelConditionModel (losCondModel);
        }
      else
        {
          lossModel->SetChannelConditionModel (nlosCondModel);
        }

      a->SetPosition (posBs);
      b->SetPosition (posUt);

      lossModel->SetAttribute ("Frequency", DoubleValue (testVector.m_frequency));
      NS_TEST_EXPECT_MSG_EQ_TOL (lossModel->CalcRxPower (testVector.m_pt, a, b), testVector.m_pr, m_tolerance, "Got unexpected rcv power");
    }

  Simulator::Destroy ();
}

/**
 * \ingroup propagation-tests
 * 
 * Test case for the class ThreeGppV2vUrbanPropagationLossModel.
 * It computes the pathloss between two nodes and compares it with the value
 * obtained using the formula in 3GPP TR 37.885 Table 6.2.1-1 for v2v
 * communications (sidelink).
 *
 * Note that 3GPP TR 37.885 defines 3 different channel states for vehicular
 * environments: LOS, NLOS, and NLOSv, the latter representing the case in which
 * the LOS path is blocked by other vehicles in the scenario. However, for
 * computing the pathloss, only the two states are considered: LOS/NLOSv or NLOS
 * (see TR 37.885 Section 6.2.1). In case of NLOSv, an additional vehicle
 * blockage loss may be added, according to a log-normal random variable.
 * Here, we test both conditions: LOS/NLOSv (without vehicle blockage
 * loss) and NLOS.
 */
class ThreeGppV2vUrbanPropagationLossModelTestCase : public TestCase
{
public:
  /**
   * Constructor
   */
  ThreeGppV2vUrbanPropagationLossModelTestCase ();

  /**
   * Destructor
   */
  virtual ~ThreeGppV2vUrbanPropagationLossModelTestCase ();

private:
  /**
   * Build the simulation scenario and run the tests
   */
  virtual void DoRun (void);

  /**
   * Struct containing the parameters for each test
   */
  typedef struct
  {
    double m_distance; //!< 2D distance between UT and BS in meters
    bool m_isLos; //!< if true LOS/NLOSv, if false NLOS
    double m_frequency; //!< carrier frequency in Hz
    double m_pt;  //!< transmitted power in dBm
    double m_pr;  //!< received power in dBm
  } TestVector;

  TestVectors<TestVector> m_testVectors; //!< array containing all the test vectors
  double m_tolerance; //!< tolerance
};

ThreeGppV2vUrbanPropagationLossModelTestCase::ThreeGppV2vUrbanPropagationLossModelTestCase ()
  : TestCase ("Test for the ThreeGppV2vUrbanPropagationLossModel class."),
  m_testVectors (),
  m_tolerance (5e-2)
{
}

ThreeGppV2vUrbanPropagationLossModelTestCase::~ThreeGppV2vUrbanPropagationLossModelTestCase ()
{
}

void
ThreeGppV2vUrbanPropagationLossModelTestCase::DoRun (void)
{
  TestVector testVector;

  testVector.m_distance = 10.0;
  testVector.m_isLos = true;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -68.1913;
  m_testVectors.Add (testVector);

  testVector.m_distance = 100.0;
  testVector.m_isLos = true;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -84.8913;
  m_testVectors.Add (testVector);

  testVector.m_distance = 1000.0;
  testVector.m_isLos = true;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -101.5913;
  m_testVectors.Add (testVector);

  testVector.m_distance = 10.0;
  testVector.m_isLos = false;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -80.0605;
  m_testVectors.Add (testVector);

  testVector.m_distance = 100.0;
  testVector.m_isLos = false;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -110.0605;
  m_testVectors.Add (testVector);

  testVector.m_distance = 1000.0;
  testVector.m_isLos = false;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -140.0605;
  m_testVectors.Add (testVector);

  // Create the nodes for BS and UT
  NodeContainer nodes;
  nodes.Create (2);

  // Create the mobility models
  Ptr<MobilityModel> a = CreateObject<ConstantPositionMobilityModel> ();
  nodes.Get (0)->AggregateObject (a);
  Ptr<MobilityModel> b = CreateObject<ConstantPositionMobilityModel> ();
  nodes.Get (1)->AggregateObject (b);

  // Use a deterministic channel condition model
  Ptr<ChannelConditionModel> losCondModel = CreateObject<AlwaysLosChannelConditionModel> ();
  Ptr<ChannelConditionModel> nlosCondModel = CreateObject<NeverLosChannelConditionModel> ();

  // Create the propagation loss model
  Ptr<ThreeGppPropagationLossModel> lossModel = CreateObject<ThreeGppV2vUrbanPropagationLossModel> ();
  lossModel->SetAttribute ("ShadowingEnabled", BooleanValue (false)); // disable the shadow fading

  for (uint32_t i = 0; i < m_testVectors.GetN (); i++)
    {
      TestVector testVector = m_testVectors.Get (i);

      Vector posUe1 = Vector (0.0, 0.0, 1.6);
      Vector posUe2 = Vector (testVector.m_distance, 0.0, 1.6);

      // set the LOS or NLOS condition
      if (testVector.m_isLos)
        {
          lossModel->SetChannelConditionModel (losCondModel);
        }
      else
        {
          lossModel->SetChannelConditionModel (nlosCondModel);
        }

      a->SetPosition (posUe1);
      b->SetPosition (posUe2);

      lossModel->SetAttribute ("Frequency", DoubleValue (testVector.m_frequency));
      NS_TEST_EXPECT_MSG_EQ_TOL (lossModel->CalcRxPower (testVector.m_pt, a, b), testVector.m_pr, m_tolerance, "Got unexpected rcv power");
    }

  Simulator::Destroy ();
}

/**
 * \ingroup propagation-tests
 * 
 * Test case for the class ThreeGppV2vHighwayPropagationLossModel.
 * It computes the pathloss between two nodes and compares it with the value
 * obtained using the formula in 3GPP TR 37.885 Table 6.2.1-1 for v2v
 * communications (sidelink).
 *
 * Note that 3GPP TR 37.885 defines 3 different channel states for vehicular
 * environments: LOS, NLOS and NLOSv, the latter representing the case in which
 * the LOS path is blocked by other vehicles in the scenario. However, for
 * computing the pathloss, only two states are considered: LOS/NLOSv or NLOS
 * (see TR 37.885 Section 6.2.1). In case of NLOSv, an additional vehicle
 * blockage loss may be added, according to a log-normal random variable.
 * Here, we test both conditions: LOS/NLOSv (without vehicle blockage
 * loss) and NLOS.
 */
class ThreeGppV2vHighwayPropagationLossModelTestCase : public TestCase
{
public:
  /**
   * Constructor
   */
  ThreeGppV2vHighwayPropagationLossModelTestCase ();

  /**
   * Destructor
   */
  virtual ~ThreeGppV2vHighwayPropagationLossModelTestCase ();

private:
  /**
   * Build the simulation scenario and run the tests
   */
  virtual void DoRun (void);

  /**
   * Struct containing the parameters for each test
   */
  typedef struct
  {
    double m_distance; //!< 2D distance between UT and BS in meters
    bool m_isLos; //!< if true LOS/NLOSv, if false NLOS
    double m_frequency; //!< carrier frequency in Hz
    double m_pt;  //!< transmitted power in dBm
    double m_pr;  //!< received power in dBm
  } TestVector;

  TestVectors<TestVector> m_testVectors; //!< array containing all the test vectors
  double m_tolerance; //!< tolerance
};

ThreeGppV2vHighwayPropagationLossModelTestCase::ThreeGppV2vHighwayPropagationLossModelTestCase ()
  : TestCase ("Test for the ThreeGppV2vHighwayPropagationLossModel"),
  m_testVectors (),
  m_tolerance (5e-2)
{
}

ThreeGppV2vHighwayPropagationLossModelTestCase::~ThreeGppV2vHighwayPropagationLossModelTestCase ()
{
}

void
ThreeGppV2vHighwayPropagationLossModelTestCase::DoRun (void)
{
  TestVector testVector;

  testVector.m_distance = 10.0;
  testVector.m_isLos = true;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -66.3794;
  m_testVectors.Add (testVector);

  testVector.m_distance = 100.0;
  testVector.m_isLos = true;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -86.3794;
  m_testVectors.Add (testVector);

  testVector.m_distance = 1000.0;
  testVector.m_isLos = true;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -106.3794;
  m_testVectors.Add (testVector);

  testVector.m_distance = 10.0;
  testVector.m_isLos = false;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -80.0605;
  m_testVectors.Add (testVector);

  testVector.m_distance = 100.0;
  testVector.m_isLos = false;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -110.0605;
  m_testVectors.Add (testVector);

  testVector.m_distance = 1000.0;
  testVector.m_isLos = false;
  testVector.m_frequency = 5.0e9;
  testVector.m_pt = 0.0;
  testVector.m_pr = -140.0605;
  m_testVectors.Add (testVector);

  // Create the nodes for BS and UT
  NodeContainer nodes;
  nodes.Create (2);

  // Create the mobility models
  Ptr<MobilityModel> a = CreateObject<ConstantPositionMobilityModel> ();
  nodes.Get (0)->AggregateObject (a);
  Ptr<MobilityModel> b = CreateObject<ConstantPositionMobilityModel> ();
  nodes.Get (1)->AggregateObject (b);

  // Use a deterministic channel condition model
  Ptr<ChannelConditionModel> losCondModel = CreateObject<AlwaysLosChannelConditionModel> ();
  Ptr<ChannelConditionModel> nlosCondModel = CreateObject<NeverLosChannelConditionModel> ();

  // Create the propagation loss model
  Ptr<ThreeGppPropagationLossModel> lossModel = CreateObject<ThreeGppV2vHighwayPropagationLossModel> ();
  lossModel->SetAttribute ("ShadowingEnabled", BooleanValue (false)); // disable the shadow fading

  for (uint32_t i = 0; i < m_testVectors.GetN (); i++)
    {
      TestVector testVector = m_testVectors.Get (i);

      Vector posUe1 = Vector (0.0, 0.0, 1.6);
      Vector posUe2 = Vector (testVector.m_distance, 0.0, 1.6);

      // set the LOS or NLOS condition
      if (testVector.m_isLos)
        {
          lossModel->SetChannelConditionModel (losCondModel);
        }
      else
        {
          lossModel->SetChannelConditionModel (nlosCondModel);
        }

      a->SetPosition (posUe1);
      b->SetPosition (posUe2);

      lossModel->SetAttribute ("Frequency", DoubleValue (testVector.m_frequency));
      NS_TEST_EXPECT_MSG_EQ_TOL (lossModel->CalcRxPower (testVector.m_pt, a, b), testVector.m_pr, m_tolerance, "Got unexpected rcv power");
    }

  Simulator::Destroy ();
}

/**
 * \ingroup propagation-tests
 * 
 * Test to check if the shadowing fading is correctly computed
 */
class ThreeGppShadowingTestCase : public TestCase
{
public:
  ThreeGppShadowingTestCase ();
  virtual ~ThreeGppShadowingTestCase ();

private:
  virtual void DoRun (void);

  /**
   * Run the experiment
   * \param testNum the index of the experiment
   * \param propagationLossModelType the type id of the propagation loss model
   *        to be used
   * \param hBs the BS height in meters
   * \param hUt the UT height in meters
   * \param distance the intial distance between the BS and the UT
   * \param shadowingEnabled true if shadowging must be enabled
   */
  void RunTest (uint16_t testNum, std::string propagationLossModelType, double hBs, double hUt, double distance, bool shadowingEnabled);


  /**
   * Compute the propagation loss
   * \param a the first mobility model
   * \param b the second mobility model
   * \param testNum the index of the experiment
   */
  void EvaluateLoss (Ptr<MobilityModel> a, Ptr<MobilityModel> b, uint8_t testNum);

  /**
   * Change the channel condition model
   * \param ccm the new ChannelConditionModel
   */
  void ChangeChannelCondition (Ptr<ChannelConditionModel> ccm);

  /**
   * Struct containing the parameters for each test
   */
  typedef struct
  {
    std::string m_propagationLossModelType; //!< the propagation loss model type id
    double m_hBs; //!< the BS height in meters
    double m_hUt; //!< the UT height in meters
    double m_distance; //!< the initial 2D distance in meters between BS and UT in meters
    double m_shadowingStdLos; //!< the standard deviation of the shadowing component in the LOS case in dB
    double m_shadowingStdNlos; //!< the standard deviation of the shadowing component in the NLOS case in dB
  } TestVector;

  TestVectors<TestVector> m_testVectors; //!< array containing all the test vectors
  Ptr<ThreeGppPropagationLossModel> m_lossModel; //!< the propagation loss model
  std::map<uint16_t /* index of experiment */, std::vector<double> /* loss in dB for each run */> m_results; //!< used to store the test results
};

ThreeGppShadowingTestCase::ThreeGppShadowingTestCase ()
  : TestCase ("Test to check if the shadow fading is correctly computed")
{
}

ThreeGppShadowingTestCase::~ThreeGppShadowingTestCase ()
{
}

void
ThreeGppShadowingTestCase::EvaluateLoss (Ptr<MobilityModel> a, Ptr<MobilityModel> b, uint8_t testNum)
{
  double loss = m_lossModel->CalcRxPower (0, a, b);
  m_results.at (testNum).push_back (loss);
}

void
ThreeGppShadowingTestCase::ChangeChannelCondition (Ptr<ChannelConditionModel> ccm)
{
  m_lossModel->SetChannelConditionModel (ccm);
}

void
ThreeGppShadowingTestCase::RunTest (uint16_t testNum, std::string propagationLossModelType, double hBs, double hUt, double distance, bool shadowingEnabled)
{
  // Add a new entry for this test in the results map
  m_results [testNum] = std::vector<double> ();

  // Create the nodes for BS and UT
  NodeContainer nodes;
  nodes.Create (2);

  // Create the mobility models
  Ptr<MobilityModel> a = CreateObject<ConstantPositionMobilityModel> ();
  a->SetPosition (Vector (0.0, 0.0, hBs));
  nodes.Get (0)->AggregateObject (a);

  Ptr<ConstantVelocityMobilityModel> b = CreateObject<ConstantVelocityMobilityModel> ();
  nodes.Get (1)->AggregateObject (b);
  b->SetPosition (Vector (0.0, distance, hUt));
  b->SetVelocity (Vector (1.0, 0.0, 0.0));

  // Create the propagation loss model
  ObjectFactory propagationLossModelFactory = ObjectFactory (propagationLossModelType);
  m_lossModel = propagationLossModelFactory.Create<ThreeGppPropagationLossModel> ();
  m_lossModel->SetAttribute ("Frequency", DoubleValue (3.5e9));
  m_lossModel->SetAttribute ("ShadowingEnabled", BooleanValue (shadowingEnabled)); // enable the shadow fading

  // Set the channel condition to LOS
  Ptr<ChannelConditionModel> losCondModel = CreateObject<AlwaysLosChannelConditionModel> ();
  m_lossModel->SetChannelConditionModel (losCondModel);
  // Schedule a transition to NLOS
  Ptr<ChannelConditionModel> nlosCondModel = CreateObject<NeverLosChannelConditionModel> ();
  Simulator::Schedule (Seconds (99.5), &ThreeGppShadowingTestCase::ChangeChannelCondition, this, nlosCondModel);

  // Schedule multiple calls to EvaluateLoss. Use both EvaluateLoss (a,b) and
  // EvaluateLoss (b,a) to check if the reciprocity holds.
  for (int i = 0; i < 200; i++)
    {
      if (i % 2 == 0)
        Simulator::Schedule (MilliSeconds (1000 * i), &ThreeGppShadowingTestCase::EvaluateLoss, this, a, b, testNum);
      else
        Simulator::Schedule (MilliSeconds (1000 * i), &ThreeGppShadowingTestCase::EvaluateLoss, this, b, a, testNum);
    }

  Simulator::Run ();
  Simulator::Destroy ();
}

void
ThreeGppShadowingTestCase::DoRun (void)
{
  // The test scenario is composed of two nodes, one fixed
  // at position (0,0) and the other moving with constant velocity from
  // position (0,50) to position (200,50).
  // The channel condition changes from LOS to NLOS when the second node
  // reaches position (100,50).
  // Each experiment computes the propagation loss between the two nodes
  // every second, until the final position is reached, and saves the
  // results in an entry of the map m_results.
  // We run numSamples experiments and estimate the mean propagation loss in
  // each position by averaging among the samples.
  // Then, we perform the null hypothesis test with a significance level of
  // 0.05.
  // This procedure is repeated for all the 3GPP propagation scenarios, i.e.,
  // RMa, UMa, UMi and Indoor-Office.

  TestVector testVector;
  testVector.m_propagationLossModelType = "ns3::ThreeGppRmaPropagationLossModel";
  testVector.m_hBs = 25;
  testVector.m_hUt = 1.6;
  testVector.m_distance = 100;
  testVector.m_shadowingStdLos = 4;
  testVector.m_shadowingStdNlos = 8;
  m_testVectors.Add (testVector);

  testVector.m_propagationLossModelType = "ns3::ThreeGppRmaPropagationLossModel";
  testVector.m_hBs = 25;
  testVector.m_hUt = 1.6;
  testVector.m_distance = 4000; // beyond the breakpoint distance
  testVector.m_shadowingStdLos = 6;
  testVector.m_shadowingStdNlos = 8;
  m_testVectors.Add (testVector);

  testVector.m_propagationLossModelType = "ns3::ThreeGppUmaPropagationLossModel";
  testVector.m_hBs = 25;
  testVector.m_hUt = 1.6;
  testVector.m_distance = 100;
  testVector.m_shadowingStdLos = 4;
  testVector.m_shadowingStdNlos = 6;
  m_testVectors.Add (testVector);

  testVector.m_propagationLossModelType = "ns3::ThreeGppUmiStreetCanyonPropagationLossModel";
  testVector.m_hBs = 10;
  testVector.m_hUt = 1.6;
  testVector.m_distance = 100;
  testVector.m_shadowingStdLos = 4;
  testVector.m_shadowingStdNlos = 7.82;
  m_testVectors.Add (testVector);

  testVector.m_propagationLossModelType = "ns3::ThreeGppIndoorOfficePropagationLossModel";
  testVector.m_hBs = 3;
  testVector.m_hUt = 1;
  testVector.m_distance = 50;
  testVector.m_shadowingStdLos = 3;
  testVector.m_shadowingStdNlos = 8.03;
  m_testVectors.Add (testVector);
  
  testVector.m_propagationLossModelType = "ns3::ThreeGppV2vUrbanPropagationLossModel";
  testVector.m_hBs = 1.6;
  testVector.m_hUt = 1.6;
  testVector.m_distance = 50;
  testVector.m_shadowingStdLos = 3;
  testVector.m_shadowingStdNlos = 4;
  m_testVectors.Add (testVector);
  
  testVector.m_propagationLossModelType = "ns3::ThreeGppV2vHighwayPropagationLossModel";
  testVector.m_hBs = 1.6;
  testVector.m_hUt = 1.6;
  testVector.m_distance = 50;
  testVector.m_shadowingStdLos = 3;
  testVector.m_shadowingStdNlos = 4;
  m_testVectors.Add (testVector);

  uint16_t numSamples = 250;

  for (uint16_t tvIndex = 0; tvIndex < m_testVectors.GetN (); tvIndex++)
    {
      TestVector tv = m_testVectors.Get (tvIndex);

      // run the experiments.
      for (uint16_t sampleIndex = 0; sampleIndex < numSamples; sampleIndex++)
        {
          RunTest (sampleIndex, tv.m_propagationLossModelType, tv.m_hBs, tv.m_hUt, tv.m_distance, true);
        }

      // analyze the results
      std::vector<double> mean_vector; // the vector containing the mean propagation loss for each position (sample mean)

      uint16_t numPositions = m_results.at (0).size ();
      for (uint16_t k = 0; k < numPositions; k++)
        {
          // compute the mean propagation loss in position k
          double mean = 0.0;
          for (auto resIt : m_results)
            {
              mean += resIt.second.at (k);
            }
          mean /= m_results.size ();
          mean_vector.push_back (mean);
        }

      // compute the true mean - just the pathloss, without the shadowing component
      RunTest (numSamples, tv.m_propagationLossModelType, tv.m_hBs, tv.m_hUt, tv.m_distance, false);
      std::vector<double> true_mean = m_results.at (numSamples); // the result of the last test is the true mean

      // perform the null hypothesis test for the LOS case
      // positions from (0, 50) to (99, 50) are LOS
      for (uint16_t i = 0; i < mean_vector.size () / 2; i++)
        {
          double z = (mean_vector.at (i) - true_mean.at (i)) / (tv.m_shadowingStdLos / std::sqrt (mean_vector.size () / 2));
          NS_TEST_EXPECT_MSG_EQ_TOL (z, 0.0, 1.96, "Null hypothesis test (LOS case) for the shadowing component rejected");
        }

      // perform the null hypothesis test for the NLOS case
      // positions from (100, 50) to (199, 50) are NLOS
      for (uint16_t i = mean_vector.size () / 2; i < mean_vector.size (); i++)
        {
          double z = (mean_vector.at (i) - true_mean.at (i)) / (tv.m_shadowingStdNlos / std::sqrt (mean_vector.size () / 2));
          NS_TEST_EXPECT_MSG_EQ_TOL (z, 0.0, 1.96, "Null hypothesis test (NLOS case) for the shadowing component rejected");
        }
    }
}

/**
 * \ingroup propagation-tests
 *
 * \brief 3GPP Propagation models TestSuite
 * 
 * This TestSuite tests the following models:
 *   - ThreeGppRmaPropagationLossModel
 *   - ThreeGppUmaPropagationLossModel
 *   - ThreeGppUmiPropagationLossModel
 *   - ThreeGppIndoorOfficePropagationLossModel
 *   - ThreeGppV2vUrbanPropagationLossModel
 *   - ThreeGppV2vHighwayPropagationLossModel
 *   - ThreeGppShadowing
 */
class ThreeGppPropagationLossModelsTestSuite : public TestSuite
{
public:
  ThreeGppPropagationLossModelsTestSuite ();
};

ThreeGppPropagationLossModelsTestSuite::ThreeGppPropagationLossModelsTestSuite ()
  : TestSuite ("three-gpp-propagation-loss-model", UNIT)
{
  AddTestCase (new ThreeGppRmaPropagationLossModelTestCase, TestCase::QUICK);
  AddTestCase (new ThreeGppUmaPropagationLossModelTestCase, TestCase::QUICK);
  AddTestCase (new ThreeGppUmiPropagationLossModelTestCase, TestCase::QUICK);
  AddTestCase (new ThreeGppIndoorOfficePropagationLossModelTestCase, TestCase::QUICK);
  AddTestCase (new ThreeGppV2vUrbanPropagationLossModelTestCase, TestCase::QUICK);
  AddTestCase (new ThreeGppV2vHighwayPropagationLossModelTestCase, TestCase::QUICK);
  AddTestCase (new ThreeGppShadowingTestCase, TestCase::QUICK);
}

/// Static variable for test initialization
static ThreeGppPropagationLossModelsTestSuite g_propagationLossModelsTestSuite;
