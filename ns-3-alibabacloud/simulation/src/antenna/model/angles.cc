/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011, 2012 CTTC
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
#include <cmath>
#include "angles.h"


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Angles");

bool Angles::m_printDeg = false;

/// Degrees to Radians conversion constant
const double DEG_TO_RAD = M_PI / 180.0;
/// Radians to Degrees conversion constant
const double RAD_TO_DEG = 180.0 / M_PI;


double
DegreesToRadians (double degrees)
{
  return degrees * DEG_TO_RAD;
}


double
RadiansToDegrees (double radians)
{
  return radians * RAD_TO_DEG;
}


std::vector<double>
DegreesToRadians (const std::vector<double> &degrees)
{
  std::vector<double> radians;
  radians.reserve (degrees.size ());
  for (size_t i = 0; i < degrees.size (); i++)
    {
      radians.push_back (DegreesToRadians (degrees[i]));
    }
  return radians;

}


std::vector<double>
RadiansToDegrees (const std::vector<double> &radians)
{
  std::vector<double> degrees;
  degrees.reserve (radians.size ());
  for (size_t i = 0; i < radians.size (); i++)
    {
      degrees.push_back (RadiansToDegrees (radians[i]));
    }
  return degrees;
}


double
WrapTo360 (double a)
{
  a = fmod (a, 360);
  if (a < 0)
    {
      a += 360;
    }

  NS_ASSERT_MSG (0 <= a && a < 360, "Invalid wrap, a=" << a);
  return a;
}


double
WrapTo180 (double a)
{
  a = fmod (a + 180, 360);
  if (a < 0)
    {
      a += 360;
    }
  a -= 180;

  NS_ASSERT_MSG (-180 <= a && a < 180, "Invalid wrap, a=" << a);
  return a;
}


double
WrapTo2Pi (double a)
{
  a = fmod (a, 2 * M_PI);
  if (a < 0)
    {
      a += 2 * M_PI;
    }

  NS_ASSERT_MSG (0 <= a && a < 2 * M_PI, "Invalid wrap, a=" << a);
  return a;
}


double
WrapToPi (double a)
{
  a = fmod (a + M_PI, 2 * M_PI);
  if (a < 0)
    {
      a += 2 * M_PI;
    }
  a -= M_PI;

  NS_ASSERT_MSG (-M_PI <= a && a < M_PI, "Invalid wrap, a=" << a);
  return a;
}


std::ostream&
operator<< (std::ostream& os, const Angles& a)
{
  double azim, incl;
  std::string unit;

  if (a.m_printDeg)
    {
      azim = RadiansToDegrees (a.m_azimuth);
      incl = RadiansToDegrees (a.m_inclination);
      unit = "deg";
    }
  else
    {
      azim = a.m_azimuth;
      incl = a.m_inclination;
      unit = "rad";
    }

  os << "(" << azim << ", " << incl << ") " << unit;
  return os;
}

std::istream&
operator>> (std::istream& is, Angles& a)
{
  char c;
  is >> a.m_azimuth >> c >> a.m_inclination;
  if (c != ':')
    {
      is.setstate (std::ios_base::failbit);
    }
  return is;
}


Angles::Angles ()
  : Angles (NAN, NAN)
{}


Angles::Angles (double azimuth, double inclination)
  : m_azimuth (azimuth),
    m_inclination (inclination)
{
  NormalizeAngles ();
}


Angles::Angles (Vector v)
  : m_azimuth (std::atan2 (v.y, v.x)),
    m_inclination (std::acos (v.z / v.GetLength ()))
{
  // azimuth and inclination angles for zero-length vectors are not defined
  if (v.x == 0.0 && v.y == 0.0 && v.z == 0.0)
    {
      m_azimuth = NAN;
      m_inclination = NAN;
    }

  NormalizeAngles ();
}

Angles::Angles (Vector v, Vector o)
  : Angles (v - o)
{}



void
Angles::SetAzimuth (double azimuth)
{
  m_azimuth = azimuth;
  NormalizeAngles ();
}


void
Angles::SetInclination (double inclination)
{
  m_inclination = inclination;
  NormalizeAngles ();
}


double
Angles::GetAzimuth (void) const
{
  return m_azimuth;
}


double
Angles::GetInclination (void) const
{
  return m_inclination;
}


void
Angles::NormalizeAngles (void)
{
  CheckIfValid ();

  // Normalize azimuth angle
  if (std::isnan (m_azimuth))
    {
      return;
    }

  m_azimuth = WrapToPi (m_azimuth);
}


void
Angles::CheckIfValid (void) const
{
  if (std::isfinite (m_inclination) || std::isfinite (m_azimuth))
    {
      NS_ASSERT_MSG (0.0 <= m_inclination && m_inclination <= M_PI,
                     "m_inclination=" << m_inclination << " not valid, should be in [0, pi] rad");
    }
  else
    {
      // infinite or nan inclination or azimuth angle
      NS_LOG_WARN ("Undefined angle: " << *this);
    }

}

}

