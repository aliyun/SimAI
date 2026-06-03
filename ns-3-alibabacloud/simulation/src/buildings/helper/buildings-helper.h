/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 CTTC
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

#ifndef BUILDINGS_HELPER_H
#define BUILDINGS_HELPER_H

#include <string>
#include <ns3/attribute.h>
#include <ns3/object-factory.h>
#include <ns3/node-container.h>
#include <ns3/ptr.h>


namespace ns3 {

class MobilityModel;
class Building;

/**
 * Helper used to install a MobilityBuildingInfo into a set of nodes.
 */
class BuildingsHelper
{
public:
  /**
  * Install the MobilityBuildingInfo to a node
  *
  * \param node the mobility model of the node to be updated
  */
  static void Install (Ptr<Node> node);     // for any nodes
  /**
  * Install the MobilityBuildingInfo to the set of nodes in a NodeContainer
  *
  * \param c the NodeContainer including the nodes to be updated
  */
  static void Install (NodeContainer c);     // for any nodes
};

}

#endif /* BUILDINGS_HELPER_H */
