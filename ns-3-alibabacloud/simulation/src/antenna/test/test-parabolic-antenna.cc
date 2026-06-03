/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011,12 CTTC
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */

#include <ns3/log.h>
#include <ns3/test.h>
#include <ns3/double.h>
#include <ns3/parabolic-antenna-model.h>
#include <ns3/simulator.h>
#include <cmath>
#include <string>
#include <iostream>
#include <sstream>


using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TestParabolicAntennaModel");

/**
 * \ingroup antenna-tests
 *
 * \brief Test condition (equal to or less than)
 */
enum ParabolicAntennaModelGainTestCondition  {
  EQUAL = 0,
  LESSTHAN = 1
};

/**
 * \ingroup antenna-tests
 *
 * \brief ParabolicAntennaModel Test
 */
class ParabolicAntennaModelTestCase : public TestCase
{
public:
  /**
   * Build the test name
   * \param a Antenna angle
   * \param b Beamwidth
   * \param o Orientation
   * \param g MaxGain
   * \return the test name
   */
  static std::string BuildNameString (Angles a, double b, double o, double g);
  /**
   * Constructor
   * \param a Antenna angle
   * \param b Beamwidth
   * \param o Orientation
   * \param g MaxGain
   * \param expectedGainDb Expeted antenna gain
   * \param cond Test condition
   */
  ParabolicAntennaModelTestCase (Angles a, double b, double o, double g, double expectedGainDb, ParabolicAntennaModelGainTestCondition cond);


private:
  virtual void DoRun (void);

  Angles m_a; //!< Antenna angle
  double m_b; //!< Beamwidth
  double m_o; //!< Orientation
  double m_g; //!< MaxGain
  double m_expectedGain;  //!< Expected gain
  ParabolicAntennaModelGainTestCondition m_cond; //!< Test condition
};

std::string ParabolicAntennaModelTestCase::BuildNameString (Angles a, double b, double o, double g)
{
  std::ostringstream oss;
  oss <<  "theta=" << a.GetInclination () << " , phi=" << a.GetAzimuth () 
      << ", beamdwidth=" << b << "deg"
      << ", orientation=" << o 
      << ", maxAttenuation=" << g << " dB";
  return oss.str ();
}


ParabolicAntennaModelTestCase::ParabolicAntennaModelTestCase (Angles a, double b, double o, double g, double expectedGainDb, ParabolicAntennaModelGainTestCondition cond)
  : TestCase (BuildNameString (a, b, o, g)),
    m_a (a),
    m_b (b),
    m_o (o),
    m_g (g),
    m_expectedGain (expectedGainDb),
    m_cond (cond)
{
}

void
ParabolicAntennaModelTestCase::DoRun ()
{
  NS_LOG_FUNCTION (this << BuildNameString (m_a, m_b, m_o, m_g));

  Ptr<ParabolicAntennaModel> a = CreateObject<ParabolicAntennaModel> ();
  a->SetAttribute ("Beamwidth", DoubleValue (m_b));
  a->SetAttribute ("Orientation", DoubleValue (m_o));
  a->SetAttribute ("MaxAttenuation", DoubleValue (m_g));
  double actualGain = a->GetGainDb (m_a);
  switch (m_cond) 
    {
    case EQUAL:
      NS_TEST_EXPECT_MSG_EQ_TOL (actualGain, m_expectedGain, 0.001, "wrong value of the radiation pattern");
      break;
    case LESSTHAN:
      NS_TEST_EXPECT_MSG_LT (actualGain, m_expectedGain, "gain higher than expected");
      break;
    default:
      break;
    }
}


/**
 * \ingroup antenna-tests
 *
 * \brief ParabolicAntennaModel TestSuite
 */
class ParabolicAntennaModelTestSuite : public TestSuite
{
public:
  ParabolicAntennaModelTestSuite ();
};

ParabolicAntennaModelTestSuite::ParabolicAntennaModelTestSuite ()
  : TestSuite ("parabolic-antenna-model", UNIT)
{ 

  // with a 60 deg beamwidth, gain is -20dB at +-77.460 degrees from boresight
  //                                                                         phi,                     theta, beamwidth, orientation,  maxAttn,  expectedGain,   condition
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians    (0),   DegreesToRadians (90)),        60,           0,       20,             0,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians   (30),   DegreesToRadians (90)),        60,           0,       20,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians  (-30),   DegreesToRadians (90)),        60,           0,       20,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians  (-90),   DegreesToRadians (90)),        60,           0,       20,           -20,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians   (90),   DegreesToRadians (90)),        60,           0,       20,           -20,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians  (100),   DegreesToRadians (90)),        60,           0,       20,           -20,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians  (150),   DegreesToRadians (90)),        60,           0,       20,           -20,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians  (180),   DegreesToRadians (90)),        60,           0,       20,           -20,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians (-100),   DegreesToRadians (90)),        60,           0,       20,           -20,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians (-150),   DegreesToRadians (90)),        60,           0,       20,           -20,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians (-180),   DegreesToRadians (90)),        60,           0,       20,           -20,     EQUAL), TestCase::QUICK);

  // with a 60 deg beamwidth, gain is -10dB at +-54.772 degrees from boresight
  // test positive orientation
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians   (60),   DegreesToRadians (90)),        60,          60,       10,             0,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians   (90),   DegreesToRadians (90)),        60,          60,       10,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians   (30),   DegreesToRadians (90)),        60,          60,       10,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians  (-30),   DegreesToRadians (90)),        60,          60,       10,           -10,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians  (150),   DegreesToRadians (90)),        60,          60,       10,           -10,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians  (160),   DegreesToRadians (90)),        60,          60,       10,           -10,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians  (210),   DegreesToRadians (90)),        60,          60,       10,           -10,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians  (240),   DegreesToRadians (90)),        60,          60,       10,           -10,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians  (-40),   DegreesToRadians (90)),        60,          60,       10,           -10,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians  (-90),   DegreesToRadians (90)),        60,          60,       10,           -10,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians (-120),   DegreesToRadians (90)),        60,          60,       10,           -10,     EQUAL), TestCase::QUICK);

  // test negative orientation and different beamwidths
  // with a 80 deg beamwidth, gain is -20dB at +- 73.030 degrees from boresight
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians (-150),   DegreesToRadians (90)),       80,        -150,       10,             0,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians (-110),   DegreesToRadians (90)),       80,        -150,       10,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians (-190),   DegreesToRadians (90)),       80,        -150,       10,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians  (-70),   DegreesToRadians (90)),       80,        -150,       10,           -10,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians   (92),   DegreesToRadians (90)),       80,        -150,       10,           -10,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians  (-30),   DegreesToRadians (90)),       80,        -150,       10,           -10,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians    (0),   DegreesToRadians (90)),       80,        -150,       10,           -10,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians   (60),   DegreesToRadians (90)),       80,        -150,       10,           -10,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians   (90),   DegreesToRadians (90)),       80,        -150,       10,           -10,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians   (30),   DegreesToRadians (90)),       80,        -150,       10,           -10,     EQUAL), TestCase::QUICK);



  // test elevation angle
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians    (0),   DegreesToRadians (88)),        60,           0,       20,             0,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians   (30),   DegreesToRadians (88)),        60,           0,       20,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians  (-30),   DegreesToRadians (88)),        60,           0,       20,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians  (-90),   DegreesToRadians (88)),        60,           0,       20,           -20,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians (-180),   DegreesToRadians (88)),        60,           0,       20,           -20,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians   (60),   DegreesToRadians (93)),        60,          60,       20,             0,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians   (90),   DegreesToRadians (93)),        60,          60,       20,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians   (30),   DegreesToRadians (93)),        60,          60,       20,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians (-120),   DegreesToRadians (93)),        60,          60,       20,           -20,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians (-150),   DegreesToRadians (93)),       100,        -150,       10,             0,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians (-100),   DegreesToRadians (93)),       100,        -150,       10,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians (-200),   DegreesToRadians (93)),       100,        -150,       10,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians  (-30),   DegreesToRadians (93)),       100,        -150,       10,           -10,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians   (90), DegreesToRadians (80.5)),       100,        -150,       10,           -10,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians    (0), DegreesToRadians (80.5)),        60,           0,       20,             0,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians   (30), DegreesToRadians (80.5)),        60,           0,       20,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians  (-30), DegreesToRadians (80.5)),        60,           0,       20,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians  (100), DegreesToRadians (80.5)),        60,           0,       20,           -20,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians (-150), DegreesToRadians (80.5)),       100,        -150,       30,             0,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians (-100), DegreesToRadians (80.5)),       100,        -150,       30,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new ParabolicAntennaModelTestCase (Angles (DegreesToRadians (-200), DegreesToRadians (80.5)),       100,        -150,       30,            -3,     EQUAL), TestCase::QUICK);

};

/// Static variable for test initialization
static ParabolicAntennaModelTestSuite g_staticParabolicAntennaModelTestSuiteInstance;
