/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 Phillip Sitbon
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
 * Author: Phillip Sitbon <phillip@sitbon.net>
 */
#ifndef WAYPOINT_H
#define WAYPOINT_H

#include "ns3/attribute.h"
#include "ns3/attribute-helper.h"
#include "ns3/nstime.h"
#include "ns3/vector.h"

namespace ns3 {

/**
 * \ingroup mobility
 * \brief a (time, location) pair.
 * \see attribute_Waypoint
 */
class Waypoint
{
public:
  /**
   * \param waypointTime time of waypoint.
   * \param waypointPosition position of waypoint corresponding to the given time.
   *
   * Create a waypoint.
   */
  Waypoint (const Time &waypointTime, const Vector &waypointPosition);

  /**
   * Create a waypoint at time 0 and position (0,0,0).
   */
  Waypoint ();
  /**
   * \brief The waypoint time
   */
  Time time;
  /**
   * \brief The position of the waypoint 
   */
  Vector position;
};

ATTRIBUTE_HELPER_HEADER ( Waypoint);

std::ostream &
operator << (std::ostream &os, const Waypoint &waypoint);
std::istream &
operator >> (std::istream &is, Waypoint &waypoint);

} // namespace ns3

#endif /* WAYPOINT_H */

