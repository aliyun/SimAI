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
#include "waypoint.h"

namespace ns3 {

ATTRIBUTE_HELPER_CPP (Waypoint);

Waypoint::Waypoint (const Time &waypointTime, const Vector &waypointPosition)
  : time (waypointTime),
    position (waypointPosition)
{
}
Waypoint::Waypoint ()
  : time (Seconds (0.0)),
    position (0,0,0)
{
}

/**
 * \brief Stream insertion operator.
 *
 * \param os the stream
 * \param waypoint the waypoint
 * \returns a reference to the stream
 */
std::ostream &operator << (std::ostream &os, const Waypoint &waypoint)
{
  os << waypoint.time.GetSeconds () << "$" << waypoint.position;
  return os;
}

/**
 * \brief Stream extraction operator.
 *
 * \param is the stream
 * \param waypoint the waypoint
 * \returns a reference to the stream
 */
std::istream &operator >> (std::istream &is, Waypoint &waypoint)
{
  char separator;
  is >> waypoint.time >> separator >> waypoint.position;
  if (separator != '$')
    {
      is.setstate (std::ios_base::failbit);
    }
  return is;
}

} // namespace ns3

