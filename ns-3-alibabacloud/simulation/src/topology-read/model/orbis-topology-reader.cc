/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 Universita' di Firenze, Italy
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
 * Author: Tommaso Pecorella (tommaso.pecorella@unifi.it)
 * Author: Valerio Sartini (valesar@gmail.com)
 */

#include <fstream>
#include <cstdlib>
#include <sstream>
#include "ns3/node-container.h"
#include "ns3/log.h"
#include "orbis-topology-reader.h"


/**
 * \file
 * \ingroup topology
 * ns3::OrbisTopologyReader implementation.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("OrbisTopologyReader");

NS_OBJECT_ENSURE_REGISTERED (OrbisTopologyReader);

TypeId OrbisTopologyReader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::OrbisTopologyReader")
    .SetParent<TopologyReader> ()
    .SetGroupName ("TopologyReader")
    .AddConstructor<OrbisTopologyReader> ()
  ;
  return tid;
}

OrbisTopologyReader::OrbisTopologyReader ()
{
  NS_LOG_FUNCTION (this);
}

OrbisTopologyReader::~OrbisTopologyReader ()
{
  NS_LOG_FUNCTION (this);
}

NodeContainer
OrbisTopologyReader::Read (void)
{
  std::ifstream topgen;
  topgen.open (GetFileName ().c_str ());
  std::map<std::string, Ptr<Node> > nodeMap;
  NodeContainer nodes;

  if ( !topgen.is_open () )
    {
      return nodes;
    }

  std::string from;
  std::string to;
  std::istringstream lineBuffer;
  std::string line;

  int linksNumber = 0;
  int nodesNumber = 0;

  while (!topgen.eof ())
    {
      line.clear ();
      lineBuffer.clear ();
      from.clear ();
      to.clear ();

      getline (topgen,line);
      lineBuffer.str (line);
      lineBuffer >> from;
      lineBuffer >> to;

      if ( (!from.empty ()) && (!to.empty ()) )
        {
          NS_LOG_INFO ( linksNumber << " From: " << from << " to: " << to );
          if ( nodeMap[from] == 0 )
            {
              Ptr<Node> tmpNode = CreateObject<Node> ();
              nodeMap[from] = tmpNode;
              nodes.Add (tmpNode);
              nodesNumber++;
            }

          if (nodeMap[to] == 0)
            {
              Ptr<Node> tmpNode = CreateObject<Node> ();
              nodeMap[to] = tmpNode;
              nodes.Add (tmpNode);
              nodesNumber++;
            }

          Link link ( nodeMap[from], from, nodeMap[to], to );
          AddLink (link);

          linksNumber++;
        }
    }
  NS_LOG_INFO ("Orbis topology created with " << nodesNumber << " nodes and " << linksNumber << " links");
  topgen.close ();

  return nodes;
}

} /* namespace ns3 */

