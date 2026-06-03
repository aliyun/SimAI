/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 University of Washington
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
 */
#ifndef RANDOM_VARIABLE_STREAM_HELPER_H
#define RANDOM_VARIABLE_STREAM_HELPER_H

#include <string>

/**
 * \file
 * \ingroup core-helpers
 * \ingroup randomvariable
 * ns3::RandomVariableStreamHelper declaration.
 */

namespace ns3 {

/**
 * \brief Helper class for static methods involving RandomVariableStream
 */
class RandomVariableStreamHelper
{
public:
  /**
   * This helper method can be used to configure the stream number for
   * one or more RNG streams.  The path passed in should correspond to
   * an attribute path in the configuration system.  It may contain wildcards
   * and may ultimately match zero, one, or more than one object.
   *
   * The objects to match are of type PointerValue where the underlying
   * type is of type Ptr<RandomVariableStream>.  
   *
   * \code
   *    .AddAttribute ("Speed",
   *                "A random variable used to pick the speed of a random waypoint model.",
   *                StringValue("ns3::UniformRandomVariable[Min=0.3|Max=0.7]"),
   *                MakePointerAccessor (&RandomWaypointMobilityModel::m_speed),
   *                MakePointerChecker<RandomVariableStream> ())
   *
   * \endcode
   *
   * Usage would look (for node 3) like this:
   * \code
   * std::string path ("/$ns3::NodeListPriv/NodeList/3/$ns3::Node/$ns3::RandomWaypointMobilityModel/Speed/$ns3::UniformRandomVariable");
   * int64_t retVal = RandomVariableStreamHelper::AssignStreams (path, 0);
   * \endcode
   *
   * \return The number of streams set
   * \param path (possibly wildcarded) path to a RandomVariableStream PointerValue
   * \param stream stream number to set; must be non-negative
   */
  static int64_t AssignStreams (std::string path, int64_t stream);
};

} // namespace ns3

#endif /* RANDOM_VARIABLE_STREAM_HELPER_H */
