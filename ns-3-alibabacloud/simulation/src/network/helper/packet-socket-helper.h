/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
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

#ifndef PACKET_SOCKET_HELPER_H
#define PACKET_SOCKET_HELPER_H

#include "ns3/node-container.h"

namespace ns3 {

/**
 * \brief Give ns3::PacketSocket powers to ns3::Node.
 */
class PacketSocketHelper
{
public:
  /**
   * Aggregate an instance of a ns3::PacketSocketFactory onto the provided
   * node.
   *
   * \param node Node on which to aggregate the ns3::PacketSocketFactory.
   */
  void Install (Ptr<Node> node) const;

  /**
   * Aggregate an instance of a ns3::PacketSocketFactory onto the provided
   * node.
   *
   * \param nodeName The name of the node on which to aggregate the ns3::PacketSocketFactory.
   */
  void Install (std::string nodeName) const;

  /**
   * For each node in the provided container, aggregate an instance of a
   * ns3::PacketSocketFactory.
   *
   * \param c NodeContainer of the set of nodes to aggregate the 
   * ns3::PacketSocketFactory on.
   */
  void Install (NodeContainer c) const;
};

} // namespace ns3

#endif /* PACKET_SOCKET_HELPER_H */
