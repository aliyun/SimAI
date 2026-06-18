/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#include "vector.h"
#include "fatal-error.h"
#include "log.h"
#include <cmath>
#include <sstream>
#include <tuple>

/**
 * \file
 * \ingroup attribute_Vector
 * ns3::Vector, ns3::Vector2D and ns3::Vector3D attribute value implementations.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Vector");

ATTRIBUTE_HELPER_CPP (Vector3D);
ATTRIBUTE_HELPER_CPP (Vector2D);

// compatibility for mobility code
Ptr<const AttributeChecker> MakeVectorChecker (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  return MakeVector3DChecker ();
}


Vector3D::Vector3D (double _x, double _y, double _z)
  : x (_x),
    y (_y),
    z (_z)
{
  NS_LOG_FUNCTION (this << _x << _y << _z);
}

Vector3D::Vector3D ()
  : x (0.0),
    y (0.0),
    z (0.0)
{
  NS_LOG_FUNCTION (this);
}

Vector2D::Vector2D (double _x, double _y)
  : x (_x),
    y (_y)
{
  NS_LOG_FUNCTION (this << _x << _y);
}

Vector2D::Vector2D ()
  : x (0.0),
    y (0.0)
{
  NS_LOG_FUNCTION (this);
}

double
Vector3D::GetLength () const
{
  NS_LOG_FUNCTION (this);
  return std::sqrt (x * x + y * y + z * z);
}
double
Vector2D::GetLength () const
{
  NS_LOG_FUNCTION (this);
  return std::sqrt (x * x + y * y);
}

double
Vector3D::GetLengthSquared () const
{
  NS_LOG_FUNCTION (this);
  return x * x + y * y + z * z;
}
double
Vector2D::GetLengthSquared () const
{
  NS_LOG_FUNCTION (this);
  return x * x + y * y;
}

double
CalculateDistance (const Vector3D &a, const Vector3D &b)
{
  NS_LOG_FUNCTION (a << b);
  return (b - a).GetLength ();
}
double
CalculateDistance (const Vector2D &a, const Vector2D &b)
{
  NS_LOG_FUNCTION (a << b);
  return (b - a).GetLength ();
}

double
CalculateDistanceSquared (const Vector3D &a, const Vector3D &b)
{
  NS_LOG_FUNCTION (a << b);
  return (b - a).GetLengthSquared ();
}
double
CalculateDistanceSquared (const Vector2D &a, const Vector2D &b)
{
  NS_LOG_FUNCTION (a << b);
  return (b - a).GetLengthSquared ();
}

std::ostream &operator << (std::ostream &os, const Vector3D &vector)
{
  os << vector.x << ":" << vector.y << ":" << vector.z;
  return os;
}
std::istream &operator >> (std::istream &is, Vector3D &vector)
{
  char c1, c2;
  is >> vector.x >> c1 >> vector.y >> c2 >> vector.z;
  if (c1 != ':'
      || c2 != ':')
    {
      is.setstate (std::ios_base::failbit);
    }
  return is;
}
bool operator < (const Vector3D &a, const Vector3D &b)
{
  return std::tie (a.x, a.y, a.z) <
         std::tie (b.x, b.y, b.z);
}
bool operator <= (const Vector3D &a, const Vector3D &b)
{
  return std::tie (a.x, a.y, a.z) <=
         std::tie (b.x, b.y, b.z);
}
bool operator > (const Vector3D &a, const Vector3D &b)
{
  return std::tie (a.x, a.y, a.z) >
         std::tie (b.x, b.y, b.z);
}
bool operator >= (const Vector3D &a, const Vector3D &b)
{
  return std::tie (a.x, a.y, a.z) >=
         std::tie (b.x, b.y, b.z);
}
bool operator == (const Vector3D &a, const Vector3D &b)
{
  return std::tie (a.x, a.y, a.z) ==
         std::tie (b.x, b.y, b.z);
}
bool operator != (const Vector3D &a, const Vector3D &b)
{
  return !(a == b);
}
Vector3D
operator + (const Vector3D &a, const Vector3D &b)
{
  return Vector3D (a.x + b.x, a.y + b.y, a.z + b.z);
}
Vector3D
operator - (const Vector3D &a, const Vector3D &b)
{
  return Vector3D (a.x - b.x, a.y - b.y, a.z - b.z);
}
std::ostream &operator << (std::ostream &os, const Vector2D &vector)
{
  os << vector.x << ":" << vector.y;
  return os;
}
std::istream &operator >> (std::istream &is, Vector2D &vector)
{
  char c1;
  is >> vector.x >> c1 >> vector.y;
  if (c1 != ':')
    {
      is.setstate (std::ios_base::failbit);
    }
  return is;
}
bool operator < (const Vector2D &a, const Vector2D &b)
{
  return std::tie (a.x, a.y) <
         std::tie (b.x, b.y);
}
bool operator <= (const Vector2D &a, const Vector2D &b)
{
  return std::tie (a.x, a.y) <=
         std::tie (b.x, b.y);
}
bool operator > (const Vector2D &a, const Vector2D &b)
{
  return std::tie (a.x, a.y) >
         std::tie (b.x, b.y);
}
bool operator >= (const Vector2D &a, const Vector2D &b)
{
  return std::tie (a.x, a.y) >=
         std::tie (b.x, b.y);
}
bool operator == (const Vector2D &a, const Vector2D &b)
{
  return std::tie (a.x, a.y) ==
         std::tie (b.x, b.y);
}
bool operator != (const Vector2D &a, const Vector2D &b)
{
  return !(a == b);
}
Vector2D
operator + (const Vector2D &a, const Vector2D &b)
{
  return Vector2D (a.x + b.x, a.y + b.y);
}
Vector2D
operator - (const Vector2D &a, const Vector2D &b)
{
  return Vector2D (a.x - b.x, a.y - b.y);
}

} // namespace ns3
