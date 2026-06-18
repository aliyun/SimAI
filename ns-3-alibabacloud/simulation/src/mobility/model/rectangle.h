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
#ifndef RECTANGLE_H
#define RECTANGLE_H

#include "ns3/attribute.h"
#include "ns3/attribute-helper.h"
#include "ns3/vector.h"

namespace ns3 {

/**
 * \ingroup mobility
 * \brief a 2d rectangle
 * \see attribute_Rectangle
 */
class Rectangle
{
public:
  /**
   * enum for naming sides
   */
  enum Side {
    RIGHT,
    LEFT,
    TOP,
    BOTTOM
  };
  /**
   * \param _xMin x coordinates of left boundary.
   * \param _xMax x coordinates of right boundary.
   * \param _yMin y coordinates of bottom boundary.
   * \param _yMax y coordinates of top boundary.
   *
   * Create a rectangle.
   */
  Rectangle (double _xMin, double _xMax,
             double _yMin, double _yMax);
  /**
   * Create a zero-sized rectangle located at coordinates (0.0,0.0)
   */
  Rectangle ();
  /**
   * \param position the position to test.
   * \return true if the input position is located within the rectangle, false otherwise.
   *
   * This method compares only the x and y coordinates of the input position.
   * It ignores the z coordinate.
   */
  bool IsInside (const Vector &position) const;
  /**
   * \param position the position to test.
   * \return the side of the rectangle the input position is closest to.
   *
   * This method compares only the x and y coordinates of the input position.
   * It ignores the z coordinate.
   */
  Side GetClosestSide (const Vector &position) const;
  /**
   * \param current the current position
   * \param speed the current speed
   * \return the intersection point between the rectangle and the current+speed vector.
   *
   * This method assumes that the current position is located _inside_
   * the rectangle and checks for this with an assert.
   * This method compares only the x and y coordinates of the input position
   * and speed. It ignores the z coordinate.
   */
  Vector CalculateIntersection (const Vector &current, const Vector &speed) const;

  double xMin; //!< The x coordinate of the left bound of the rectangle
  double xMax; //!< The x coordinate of the right bound of the rectangle
  double yMin; //!< The y coordinate of the bottom bound of the rectangle
  double yMax; //!< The y coordinate of the top bound of the rectangle
};

std::ostream &operator << (std::ostream &os, const Rectangle &rectangle);
std::istream &operator >> (std::istream &is, Rectangle &rectangle);

ATTRIBUTE_HELPER_HEADER (Rectangle);

} // namespace ns3

#endif /* RECTANGLE_H */
