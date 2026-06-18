/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 Dan Broyles
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
 * Author: Dan Broyles <dbroyl01@ku.edu>
 */
#ifndef BOX_H
#define BOX_H

#include "ns3/attribute.h"
#include "ns3/attribute-helper.h"
#include "ns3/vector.h"

namespace ns3 {

/**
 * \ingroup mobility
 * \brief a 3d box
 * \see attribute_Box
 */
class Box
{
public:
  /**
   * Enum class to specify sides of a box
   */
  enum Side {
    RIGHT,
    LEFT,
    TOP,
    BOTTOM,
    UP,
    DOWN
  };
  /**
   * \param _xMin x coordinates of left boundary.
   * \param _xMax x coordinates of right boundary.
   * \param _yMin y coordinates of bottom boundary.
   * \param _yMax y coordinates of top boundary.
   * \param _zMin z coordinates of down boundary.
   * \param _zMax z coordinates of up boundary.
   *
   * Create a box.
   */
  Box (double _xMin, double _xMax,
       double _yMin, double _yMax,
       double _zMin, double _zMax);
  /**
   * Create a zero-sized box located at coordinates (0.0,0.0,0.0)
   */
  Box ();
  /**
   * \param position the position to test.
   * \returns true if the input position is located within the box, 
   *          false otherwise.
   *
   * This method compares the x, y, and z coordinates of the input position.
   */
  bool IsInside (const Vector &position) const;
  /**
   * \param position the position to test.
   * \returns the side of the cube the input position is closest to.
   *
   * This method compares the x, y, and z coordinates of the input position.
   */
  Side GetClosestSide (const Vector &position) const;
  /**
   * \param current the current position
   * \param speed the current speed
   * \returns the intersection point between the rectangle and the current+speed vector.
   *
   * This method assumes that the current position is located _inside_
   * the cube and checks for this with an assert.
   * This method compares only the x and y coordinates of the input position
   * and speed. It ignores the z coordinate.
   */
  Vector CalculateIntersection (const Vector &current, const Vector &speed) const;
  /**
   * \brief Checks if a line-segment between position l1 and position l2
   *        intersects a box.
   *
   * This method considers all the three coordinates, i.e., x, y, and z.
   * This function was developed by NYU Wireless and is based on the algorithm
   * described here http://www.3dkingdoms.com/weekly/weekly.php?a=21.
   * Reference: Menglei Zhang, Michele Polese, Marco Mezzavilla, Sundeep Rangan,
   * Michele Zorzi. "ns-3 Implementation of the 3GPP MIMO Channel Model for
   * Frequency Spectrum above 6 GHz". In Proceedings of the Workshop on ns-3
   * (WNS3 '17). 2017.
   *
   * \param l1 position
   * \param l2 position
   * \return true if there is a intersection, false otherwise
   */
  bool IsIntersect (const Vector &l1, const Vector &l2) const;

  /** The x coordinate of the left bound of the box */
  double xMin;
  /** The x coordinate of the right bound of the box */
  double xMax;
  /** The y coordinate of the bottom bound of the box */
  double yMin;
  /** The y coordinate of the top bound of the box */
  double yMax;
  /** The z coordinate of the down bound of the box */
  double zMin;
  /** The z coordinate of the up bound of the box */
  double zMax;
};

std::ostream &operator << (std::ostream &os, const Box &box);
std::istream &operator >> (std::istream &is, Box &box);

ATTRIBUTE_HELPER_HEADER (Box);

} // namespace ns3

#endif /* BOX_H */
