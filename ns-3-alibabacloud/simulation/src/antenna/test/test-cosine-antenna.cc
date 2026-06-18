/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 CTTC
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
#include <ns3/cosine-antenna-model.h>
#include <ns3/simulator.h>
#include <cmath>
#include <string>
#include <iostream>
#include <sstream>


using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TestCosineAntennaModel");

/**
 * \ingroup antenna-tests
 *
 * \brief Test condition (equal to or less than)
 */
enum CosineAntennaModelGainTestCondition  {
  EQUAL = 0,
  LESSTHAN = 1
};

/**
 * \ingroup antenna-tests
 *
 * \brief CosineAntennaModel Test
 */
class CosineAntennaModelTestCase : public TestCase
{
public:
  /**
   * Build the test name
   * \param a Antenna angle
   * \param b Horizontal and Vertical Beamwidth
   * \param o Orientation
   * \param g MaxGain
   * \return the test name
   */
  static std::string BuildNameString (Angles a, double b, double o, double g);
  /**
   * Constructor
   * \param a Antenna angle
   * \param b Horizontal and Vertical Beamwidth
   * \param o Orientation
   * \param g MaxGain
   * \param expectedGainDb Expeted antenna gain
   * \param cond Test condition
   */
  CosineAntennaModelTestCase (Angles a, double b, double o, double g, double expectedGainDb, CosineAntennaModelGainTestCondition cond);


private:
  virtual void DoRun (void);
  Angles m_a; //!< Antenna angle
  double m_b; //!< Horizontal and Vertical Beamwidth
  double m_o; //!< Orientation
  double m_g; //!< MaxGain
  double m_expectedGain;  //!< Expected gain
  CosineAntennaModelGainTestCondition m_cond; //!< Test condition
};

std::string CosineAntennaModelTestCase::BuildNameString (Angles a, double b, double o, double g)
{
  std::ostringstream oss;
  oss <<  "theta=" << a.GetInclination () << " , phi=" << a.GetAzimuth () 
      << ", beamdwidth=" << b << "deg"
      << ", orientation=" << o 
      << ", maxGain=" << g << " dB";
  return oss.str ();
}


CosineAntennaModelTestCase::CosineAntennaModelTestCase (Angles a, double b, double o, double g, double expectedGainDb, CosineAntennaModelGainTestCondition cond)
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
CosineAntennaModelTestCase::DoRun ()
{
  NS_LOG_FUNCTION (this << BuildNameString (m_a, m_b, m_o, m_g));

  Ptr<CosineAntennaModel> a = CreateObject<CosineAntennaModel> ();
  a->SetAttribute ("HorizontalBeamwidth", DoubleValue (m_b));
  a->SetAttribute ("VerticalBeamwidth", DoubleValue (m_b));
  a->SetAttribute ("Orientation", DoubleValue (m_o));
  a->SetAttribute ("MaxGain", DoubleValue (m_g));
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
 * \brief CosineAntennaModel TestSuite
 */
class CosineAntennaModelTestSuite : public TestSuite
{
public:
  CosineAntennaModelTestSuite ();
};

CosineAntennaModelTestSuite::CosineAntennaModelTestSuite ()
  : TestSuite ("cosine-antenna-model", UNIT)
{ 
  // to calculate the azimut angle offset for a given gain in db:
  // phideg = (2*acos(10^(targetgaindb/(20*n))))*180/pi
  // e.g., with a 60 deg beamwidth, gain is -20dB at +- 74.945 degrees from boresight

  //                                                                      phi,                     theta, beamwidth, orientation,  maxGain,  expectedGain,   condition
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians    (0),   DegreesToRadians (90)),        60,           0,        0,             0,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians   (30),   DegreesToRadians (90)),        60,           0,        0,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (-30),   DegreesToRadians (90)),        60,           0,        0,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (-90),   DegreesToRadians (90)),        60,           0,        0,           -20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians   (90),   DegreesToRadians (90)),        60,           0,        0,           -20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (100),   DegreesToRadians (90)),        60,           0,        0,           -20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (150),   DegreesToRadians (90)),        60,           0,        0,           -20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (180),   DegreesToRadians (90)),        60,           0,        0,           -20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (-100),   DegreesToRadians (90)),        60,           0,        0,           -20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (-150),   DegreesToRadians (90)),        60,           0,        0,           -20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (-180),   DegreesToRadians (90)),        60,           0,        0,           -20,  LESSTHAN), TestCase::QUICK);

  // test positive orientation
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians   (60),   DegreesToRadians (90)),        60,          60,        0,             0,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians   (90),   DegreesToRadians (90)),        60,          60,        0,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians   (30),   DegreesToRadians (90)),        60,          60,        0,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (-30),   DegreesToRadians (90)),        60,          60,        0,           -20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (150),   DegreesToRadians (90)),        60,          60,        0,           -20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (160),   DegreesToRadians (90)),        60,          60,        0,           -20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (210),   DegreesToRadians (90)),        60,          60,        0,           -20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (240),   DegreesToRadians (90)),        60,          60,        0,           -20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (-40),   DegreesToRadians (90)),        60,          60,        0,           -20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (-90),   DegreesToRadians (90)),        60,          60,        0,           -20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (-120),   DegreesToRadians (90)),        60,          60,        0,           -20,  LESSTHAN), TestCase::QUICK);

  // test negative orientation and different beamwidths
  // with a 100 deg beamwidth, gain is -20dB at +- 117.47 degrees from boresight
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (-150),   DegreesToRadians (90)),       100,        -150,        0,             0,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (-100),   DegreesToRadians (90)),       100,        -150,        0,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (-200),   DegreesToRadians (90)),       100,        -150,        0,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (-32.531),DegreesToRadians (90)),       100,        -150,        0,           -20,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (92.531), DegreesToRadians (90)),       100,        -150,        0,           -20,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (-30),   DegreesToRadians (90)),       100,        -150,        0,           -20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians    (0),   DegreesToRadians (90)),       100,        -150,        0,           -20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians   (60),   DegreesToRadians (90)),       100,        -150,        0,           -20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians   (90),   DegreesToRadians (90)),       100,        -150,        0,           -20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians   (30),   DegreesToRadians (90)),       100,        -150,        0,           -20,  LESSTHAN), TestCase::QUICK);
  // with a 150 deg beamwidth, gain is -10dB at +- 124.93 degrees from boresight, and -20dB at +- 155.32 degrees from boresight
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (-150),   DegreesToRadians (90)),       150,        -150,        0,             0,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (135),   DegreesToRadians (90)),       150,        -150,        0,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (-75),   DegreesToRadians (90)),       150,        -150,        0,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (85.070), DegreesToRadians (90)),       150,        -150,        0,           -10,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (-25.070),DegreesToRadians (90)),       150,        -150,        0,           -10,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (5.3230), DegreesToRadians (90)),       150,        -150,        0,           -20,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (54.677), DegreesToRadians (90)),       150,        -150,        0,           -20,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians   (30),   DegreesToRadians (90)),       150,        -150,        0,           -20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians   (20),   DegreesToRadians (90)),       150,        -150,        0,           -20,  LESSTHAN), TestCase::QUICK);
  // test flat beam, with beamwidth=360 deg
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians    (0),   DegreesToRadians (90)),       360,           0,        0,             0,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (180),   DegreesToRadians (90)),       360,           0,        0,             0,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (-180),   DegreesToRadians (90)),       360,           0,        0,             0,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians    (0),   DegreesToRadians  (0)),       360,           0,        0,             0,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians    (0),   DegreesToRadians(180)),       360,           0,        0,             0,     EQUAL), TestCase::QUICK);
  
  // test maxGain
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians    (0),   DegreesToRadians (90)),        60,           0,       10,            10,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians   (30),   DegreesToRadians (90)),        60,           0,       22,            19,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (-30),   DegreesToRadians (90)),        60,           0,       -4,            -7,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (-90),   DegreesToRadians (90)),        60,           0,       10,           -10,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians   (90),   DegreesToRadians (90)),        60,           0,      -20,           -40,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (100),   DegreesToRadians (90)),        60,           0,       40,            20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (-150),   DegreesToRadians (90)),       100,        -150,        2,             2,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (-100),   DegreesToRadians (90)),       100,        -150,        4,             1,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (-200),   DegreesToRadians (90)),       100,        -150,       -1,            -4,     EQUAL), TestCase::QUICK);


  // test elevation angle
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians    (0),   DegreesToRadians (60)),        60,           0,        0,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians   (30),   DegreesToRadians (60)),        60,           0,        0,            -6,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (-30),   DegreesToRadians (60)),        60,           0,        0,            -6,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (-90),   DegreesToRadians (60)),        60,           0,        0,           -20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (-180),   DegreesToRadians (60)),        60,           0,        0,           -20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians   (60),  DegreesToRadians (120)),        60,          60,        0,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians   (90),  DegreesToRadians (120)),        60,          60,        0,            -6,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians   (30),  DegreesToRadians (120)),        60,          60,        0,            -6,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (-120),  DegreesToRadians (120)),        60,          60,        0,           -20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (-150),  DegreesToRadians (140)),       100,        -150,        0,            -3,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (-100),  DegreesToRadians (140)),       100,        -150,        0,            -6,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (-200),  DegreesToRadians (140)),       100,        -150,        0,            -6,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (-30),  DegreesToRadians (140)),       100,        -150,        0,           -20,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians    (0),   DegreesToRadians (60)),        60,           0,       10,             7,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians   (30),   DegreesToRadians (60)),        60,           0,       22,            16,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (-30),   DegreesToRadians (60)),        60,           0,       -4,           -10,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (-90),   DegreesToRadians (60)),        60,           0,       10,           -13,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians   (90),   DegreesToRadians (60)),        60,           0,      -20,           -43,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians  (100),   DegreesToRadians (60)),        60,           0,       40,            17,  LESSTHAN), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (-150),   DegreesToRadians (40)),       100,        -150,        2,            -1,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (-100),   DegreesToRadians (40)),       100,        -150,        4,            -2,     EQUAL), TestCase::QUICK);
  AddTestCase (new CosineAntennaModelTestCase (Angles (DegreesToRadians (-200),   DegreesToRadians (40)),       100,        -150,       -1,            -7,     EQUAL), TestCase::QUICK);

};

/// Static variable for test initialization
static CosineAntennaModelTestSuite g_staticCosineAntennaModelTestSuiteInstance;
