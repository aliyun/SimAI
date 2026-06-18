/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 University of Washington
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
 * Author: Benjamin Cizdziel <ben.cizdziel@gmail.com>
 */

#include <ns3/test.h>
#include <ns3/log.h>
#include <cmath>
#include <ns3/geographic-positions.h>

/**
 * This test verifies the accuracy of the RandCartesianPointsAroundGeographicPoint()
 * method in the GeographicPositions class, which generates uniformly 
 * distributed random points (in ECEF Cartesian coordinates) within a given 
 * altitude above earth's surface centered around a given origin point (on 
 * earth's surface, in geographic/geodetic coordinates) within a given distance 
 * radius (using arc length of earth's surface, not pythagorean distance).
 * Distance radius is measured as if all generated points are on earth's 
 * surface (with altitude = 0). Assumes earth is a perfect sphere. To verify the 
 * method, this test checks that the generated points are within the given 
 * maximum distance radius from the origin. Since this is testing the distance 
 * radius from the origin, all points generated in this test are on earth's 
 * surface (altitude = 0), since the distance radius has been defined as if all 
 * points are on earth's surface. The pythagorean (straight-line) distance 
 * between each generated point and the origin is first calculated, and then 
 * using this distance and the radius of the earth, the distance radius to the
 * origin (arc length) is calculated. This distance radius is compared to the 
 * max distance radius to ensure that it is less than the maximum.
 */
NS_LOG_COMPONENT_DEFINE ("RandCartAroundGeoTest");

using namespace ns3;

/**
 * 0.1 meter tolerance for testing, which is very small compared to the maximum 
 * distances from origin being tested
 */
const double TOLERANCE = 0.1;

/// earth's radius in meters if modeled as a perfect sphere
static const double EARTH_RADIUS = 6371e3; 

/**
 * \ingroup mobility-test
 * \ingroup tests
 *
 * \brief Rand Cart Around Geo Test Case
 */
class RandCartAroundGeoTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param originLatitude origin latitude
   * \param originLongitude origin longitude
   * \param maxAltitude maximum altitude
   * \param numPoints number of points
   * \param maxDistFromOrigin maximum distance from origin
   * \param uniRand random variable
   */
  RandCartAroundGeoTestCase (double originLatitude, 
                             double originLongitude,
                             double maxAltitude, 
                             int numPoints, 
                             double maxDistFromOrigin,
                             Ptr<UniformRandomVariable> uniRand);
  virtual ~RandCartAroundGeoTestCase ();

private:
  virtual void DoRun (void);
  /**
   * name function
   * \param originLatitude the origin latitude
   * \param originLongitude the origin longitude
   * \param maxDistFromOrigin the maximum distance from the origin
   * \returns the name string
   */
  static std::string Name (double originLatitude, 
                           double originLongitude,
                           double maxDistFromOrigin);
  double m_originLatitude; ///< origin latitude
  double m_originLongitude; ///< origin longitude
  double m_maxAltitude; ///< maximum altitude
  int m_numPoints; ///< number of points
  double m_maxDistFromOrigin; ///< maximum distance from origin
  Ptr<UniformRandomVariable> m_uniRand; ///< random number
};

std::string 
RandCartAroundGeoTestCase::Name (double originLatitude, 
                                 double originLongitude,
                                 double maxDistFromOrigin)
{
  std::ostringstream oss;
  oss << "origin latitude = " << originLatitude << " degrees, "
      << "origin longitude = " << originLongitude << " degrees, "
      << "max distance from origin = " << maxDistFromOrigin;
  return oss.str();
}

RandCartAroundGeoTestCase::RandCartAroundGeoTestCase (double originLatitude, 
                                                      double originLongitude,
                                                      double maxAltitude, 
                                                      int numPoints, 
                                                      double maxDistFromOrigin,
                                                      Ptr<UniformRandomVariable> uniRand)
  : TestCase (Name (originLatitude, originLongitude, maxDistFromOrigin)),
    m_originLatitude (originLatitude),
    m_originLongitude (originLongitude),
    m_maxAltitude (maxAltitude),
    m_numPoints (numPoints),
    m_maxDistFromOrigin (maxDistFromOrigin),
    m_uniRand (uniRand)
{
}

RandCartAroundGeoTestCase::~RandCartAroundGeoTestCase ()
{
}

void
RandCartAroundGeoTestCase::DoRun (void)
{  
  std::list<Vector> points =  GeographicPositions::RandCartesianPointsAroundGeographicPoint (m_originLatitude, 
                                                                                             m_originLongitude,
                                                                                             m_maxAltitude, 
                                                                                             m_numPoints, 
                                                                                             m_maxDistFromOrigin,
                                                                                             m_uniRand);
  Vector origin = GeographicPositions::GeographicToCartesianCoordinates (m_originLatitude, 
                                                                         m_originLongitude, 
                                                                         m_maxAltitude,
                                                                         GeographicPositions::SPHERE);
  Vector randPoint;
  while (!points.empty ())
    {
      randPoint = points.front ();
      points.pop_front ();

      // pythagorean distance between random point and origin, not distance on surface of earth
      double straightDistFromOrigin = sqrt (pow (randPoint.x - origin.x, 2) + 
                                            pow (randPoint.y - origin.y, 2) + 
                                            pow (randPoint.z - origin.z, 2));

      // arc length distance between random point and origin, on surface of earth
      double arcDistFromOrigin = 2 * EARTH_RADIUS * asin (straightDistFromOrigin / (2 * EARTH_RADIUS));

      NS_TEST_ASSERT_MSG_LT (arcDistFromOrigin, 
                             m_maxDistFromOrigin + TOLERANCE,  
                             "random point (" << randPoint.x << ", " << randPoint.y 
                             << ", " << randPoint.z << ") is outside of max radius from origin");
    }
}

/**
 * \ingroup mobility-test
 * \ingroup tests
 *
 * \brief Rand Cart Around Geo Test Suite
 */
class RandCartAroundGeoTestSuite : public TestSuite
{
public:
  RandCartAroundGeoTestSuite ();
};

RandCartAroundGeoTestSuite::RandCartAroundGeoTestSuite ()
  : TestSuite ("rand-cart-around-geo", UNIT)
{
  NS_LOG_INFO ("creating RandCartAroundGeoTestSuite");
  Ptr<UniformRandomVariable> uniRand = CreateObject<UniformRandomVariable> ();
  uniRand->SetStream (5);
  for (double originLatitude = -89.9; originLatitude <= 89.9; originLatitude += 35.96)
    {
      for (double originLongitude = 0; originLongitude <= 360; originLongitude += 72)
        {
          for (double maxDistFromOrigin = 1000; maxDistFromOrigin <= 1000000; maxDistFromOrigin *= 10)
            {
              AddTestCase (new RandCartAroundGeoTestCase (originLatitude, 
                                                          originLongitude, 
                                                          0, // on earth's surface
                                                          50, // 50 points generated
                                                          maxDistFromOrigin,
                                                          uniRand), 
                           TestCase::QUICK);
            }
        }
    }
}

static RandCartAroundGeoTestSuite g_RandCartAroundGeoTestSuite; ///< the test suite
