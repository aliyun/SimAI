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
#include <ns3/antenna-model.h>
#include <cmath>
#include <string>
#include <iostream>
#include <sstream>

using namespace ns3;

/**
 * \ingroup tests
 *
 * \brief Angles Test using one vector for initialization
 */
class OneVectorConstructorTestCase : public TestCase
{
public:
  /**
   * Build the test name
   * \param v test parameter
   * \return the test name
   */
  static std::string BuildNameString (Vector v);
  /**
   * Constructor
   * \param v vector
   * \param a expected angle
   */
  OneVectorConstructorTestCase (Vector v, Angles a);

private:
  virtual void DoRun (void);

  Vector m_v; //!< vector
  Angles m_a; //!< expected angle
};

std::string OneVectorConstructorTestCase::BuildNameString (Vector v)
{
  std::ostringstream oss;
  oss << " v = " << v;
  return oss.str ();
}


OneVectorConstructorTestCase::OneVectorConstructorTestCase (Vector v, Angles a)
  : TestCase (BuildNameString (v)),
    m_v (v),
    m_a (a)
{}

void
OneVectorConstructorTestCase::DoRun ()
{
  Angles a (m_v);
  NS_TEST_EXPECT_MSG_EQ_TOL ( a.GetAzimuth (), m_a.GetAzimuth (), 1e-10, "incorrect phi");
  NS_TEST_EXPECT_MSG_EQ_TOL ( a.GetInclination (), m_a.GetInclination (), 1e-10, "incorrect theta");
}



/**
 * \ingroup tests
 *
 * \brief Angles Test using two vectors for initialization
 */
class TwoVectorsConstructorTestCase : public TestCase
{
public:
  /**
   * Build the test name
   * \param v test parameter
   * \param o test parameter
   * \return the test name
   */
  static std::string BuildNameString (Vector v, Vector o);
  /**
   * Constructor
   * \param v point
   * \param o origin
   * \param a expected angle
   */
  TwoVectorsConstructorTestCase (Vector v, Vector o, Angles a);

private:
  virtual void DoRun (void);

  Vector m_v; //!< point
  Vector m_o; //!< origin
  Angles m_a; //!< expected angle
};

std::string TwoVectorsConstructorTestCase::BuildNameString (Vector v, Vector o)
{
  std::ostringstream oss;
  oss << " v = " << v << ", o = " << o;
  return oss.str ();
}


TwoVectorsConstructorTestCase::TwoVectorsConstructorTestCase (Vector v, Vector o, Angles a)
  : TestCase (BuildNameString (v, o)),
    m_v (v),
    m_o (o),
    m_a (a)
{}

void
TwoVectorsConstructorTestCase::DoRun ()
{
  Angles a (m_v, m_o);
  NS_TEST_EXPECT_MSG_EQ_TOL ( a.GetAzimuth (), m_a.GetAzimuth (), 1e-10, "incorrect phi");
  NS_TEST_EXPECT_MSG_EQ_TOL ( a.GetInclination (), m_a.GetInclination (), 1e-10, "incorrect theta");
}



/**
 * \ingroup tests
 *
 * \brief Angles TestSuite
 */
class AnglesTestSuite : public TestSuite
{
public:
  AnglesTestSuite ();
};

AnglesTestSuite::AnglesTestSuite ()
  : TestSuite ("angles", UNIT)
{
  AddTestCase (new OneVectorConstructorTestCase (Vector (1, 0, 0), Angles (0, M_PI_2)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (-1, 0, 0), Angles (M_PI, M_PI_2)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (0, 1, 0), Angles (M_PI_2, M_PI_2)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (0, -1, 0), Angles (-M_PI_2, M_PI_2)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (0, 0, 1), Angles (0, 0)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (0, 0, -1), Angles (0, M_PI)), TestCase::QUICK);

  AddTestCase (new OneVectorConstructorTestCase (Vector (2, 0, 0), Angles (0, M_PI_2)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (-2, 0, 0), Angles (M_PI, M_PI_2)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (0, 2, 0), Angles (M_PI_2, M_PI_2)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (0, -2, 0), Angles (-M_PI_2, M_PI_2)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (0, 0, 2), Angles (0, 0)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (0, 0, -2), Angles (0, M_PI)), TestCase::QUICK);

  AddTestCase (new OneVectorConstructorTestCase (Vector (1, 0, 1), Angles (0, M_PI_4)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (1, 0, -1), Angles (0, 3 * M_PI_4)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (1, 1, 0), Angles (M_PI_4, M_PI_2)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (1, -1, 0), Angles (-M_PI_4, M_PI_2)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (-1, 0, 1), Angles (M_PI, M_PI_4)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (-1, 0, -1), Angles (M_PI, 3 * M_PI_4)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (-1, 1, 0), Angles (3 * M_PI_4, M_PI_2)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (-1, -1, 0), Angles (-3 * M_PI_4, M_PI_2)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (0, 1, 1), Angles (M_PI_2, M_PI_4)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (0, 1, -1), Angles (M_PI_2, 3 * M_PI_4)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (0, -1, 1), Angles (-M_PI_2, M_PI_4)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (0, -1, -1), Angles (-M_PI_2, 3 * M_PI_4)), TestCase::QUICK);

  AddTestCase (new OneVectorConstructorTestCase (Vector (1, 1, std::sqrt (2)), Angles (M_PI_4, M_PI_4)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (1, 1, -std::sqrt (2)), Angles (M_PI_4, 3 * M_PI_4)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (1, -1, std::sqrt (2)), Angles (-M_PI_4, M_PI_4)), TestCase::QUICK);
  AddTestCase (new OneVectorConstructorTestCase (Vector (-1, 1, std::sqrt (2)), Angles (3 * M_PI_4, M_PI_4)), TestCase::QUICK);




  AddTestCase (new TwoVectorsConstructorTestCase (Vector (1, 0, 0), Vector (0, 0, 0), Angles (0, M_PI_2)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (-1, 0, 0), Vector (0, 0, 0), Angles (M_PI, M_PI_2)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (0, 1, 0), Vector (0, 0, 0), Angles (M_PI_2, M_PI_2)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (0, -1, 0), Vector (0, 0, 0), Angles (-M_PI_2, M_PI_2)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (0, 0, 1), Vector (0, 0, 0), Angles (0, 0)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (0, 0, -1), Vector (0, 0, 0), Angles (0, M_PI)), TestCase::QUICK);

  AddTestCase (new TwoVectorsConstructorTestCase (Vector (2, 0, 0), Vector (0, 0, 0), Angles (0, M_PI_2)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (-2, 0, 0), Vector (0, 0, 0), Angles (M_PI, M_PI_2)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (0, 2, 0), Vector (0, 0, 0), Angles (M_PI_2, M_PI_2)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (0, -2, 0), Vector (0, 0, 0), Angles (-M_PI_2, M_PI_2)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (0, 0, 2), Vector (0, 0, 0), Angles (0, 0)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (0, 0, -2), Vector (0, 0, 0), Angles (0, M_PI)), TestCase::QUICK);

  AddTestCase (new TwoVectorsConstructorTestCase (Vector (1, 0, 1), Vector (0, 0, 0), Angles (0, M_PI_4)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (1, 0, -1), Vector (0, 0, 0), Angles (0, 3 * M_PI_4)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (1, 1, 0), Vector (0, 0, 0), Angles (M_PI_4, M_PI_2)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (1, -1, 0), Vector (0, 0, 0), Angles (-M_PI_4, M_PI_2)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (-1, 0, 1), Vector (0, 0, 0), Angles (M_PI, M_PI_4)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (-1, 0, -1), Vector (0, 0, 0), Angles (M_PI, 3 * M_PI_4)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (-1, 1, 0), Vector (0, 0, 0), Angles (3 * M_PI_4, M_PI_2)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (-1, -1, 0), Vector (0, 0, 0), Angles (-3 * M_PI_4, M_PI_2)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (0, 1, 1), Vector (0, 0, 0), Angles (M_PI_2, M_PI_4)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (0, 1, -1), Vector (0, 0, 0), Angles (M_PI_2, 3 * M_PI_4)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (0, -1, 1), Vector (0, 0, 0), Angles (-M_PI_2, M_PI_4)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (0, -1, -1), Vector (0, 0, 0), Angles (-M_PI_2, 3 * M_PI_4)), TestCase::QUICK);

  AddTestCase (new TwoVectorsConstructorTestCase (Vector (1, 1, std::sqrt (2)), Vector (0, 0, 0), Angles (M_PI_4, M_PI_4)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (1, 1, -std::sqrt (2)), Vector (0, 0, 0), Angles (M_PI_4, 3 * M_PI_4)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (1, -1, std::sqrt (2)), Vector (0, 0, 0), Angles (-M_PI_4, M_PI_4)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (-1, 1, std::sqrt (2)), Vector (0, 0, 0), Angles (3 * M_PI_4, M_PI_4)), TestCase::QUICK);


  AddTestCase (new TwoVectorsConstructorTestCase (Vector (3, 2, 2), Vector (2, 2, 2), Angles (0, M_PI_2)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (1, 2, 2), Vector (2, 2, 2), Angles (M_PI, M_PI_2)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (2, 3, 2), Vector (2, 2, 2), Angles (M_PI_2, M_PI_2)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (-1, 2, 2), Vector (-1, 3, 2), Angles (-M_PI_2, M_PI_2)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (4, -2, 7), Vector (4, -2, 6), Angles (0, 0)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (0, -5, -1), Vector (0, -5, 0), Angles (0, M_PI)), TestCase::QUICK);

  AddTestCase (new TwoVectorsConstructorTestCase (Vector (-2, 2, -1), Vector (-4, 2, -1), Angles (0, M_PI_2)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (2, 2, 0), Vector (4, 2, 0), Angles (M_PI, M_PI_2)), TestCase::QUICK);

  AddTestCase (new TwoVectorsConstructorTestCase (Vector (-1, 4, 4), Vector (-2, 4, 3), Angles (0, M_PI_4)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (0, -2, -6), Vector (-1, -2, -5), Angles (0, 3 * M_PI_4)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (77, 3, 43), Vector (78, 2, 43), Angles (3 * M_PI_4, M_PI_2)), TestCase::QUICK);

  AddTestCase (new TwoVectorsConstructorTestCase (Vector (24, -2, -6 - std::sqrt (2)), Vector (23, -3, -6), Angles (M_PI_4, 3 * M_PI_4)), TestCase::QUICK);
  AddTestCase (new TwoVectorsConstructorTestCase (Vector (0.5, 11.45, std::sqrt (2) - 1), Vector (-0.5, 12.45, -1), Angles (-M_PI_4, M_PI_4)), TestCase::QUICK);


}

/// Static variable for test initialization
static AnglesTestSuite g_staticAnglesTestSuiteInstance;
