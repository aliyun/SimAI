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

#ifndef ORBIS_TOPOLOGY_READER_H
#define ORBIS_TOPOLOGY_READER_H

#include "topology-reader.h"

/**
 * \file
 * \ingroup topology
 * ns3::OrbisTopologyReader declaration.
 */

namespace ns3 {


// ------------------------------------------------------------
// --------------------------------------------
/**
 * \ingroup topology
 *
 * \brief Topology file reader (Orbis-format type).
 *
 * This class takes an input file in Orbis format and extracts all
 * the information needed to build the topology
 * (i.e.number of nodes, links and links structure).
 * It have been tested with Orbis 0.70
 * http://www.sysnet.ucsd.edu/~pmahadevan/topo_research/topo.html
 */
class OrbisTopologyReader : public TopologyReader
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId.
   */
  static TypeId GetTypeId (void);

  OrbisTopologyReader ();
  virtual ~OrbisTopologyReader ();

  // Delete copy constructor and assignment operator to avoid misuse
  OrbisTopologyReader (const OrbisTopologyReader &) = delete;
  OrbisTopologyReader & operator = (const OrbisTopologyReader &) = delete;

  /**
   * \brief Main topology reading function.
   *
   * This method opens an input stream and reads the Orbis-format file.
   * Every row represents a topology link (the ids of a couple of nodes),
   * so the input file is read line by line to figure out how many links
   * and nodes are in the topology.
   *
   * \return The container of the nodes created (or empty container if there was an error)
   */
  virtual NodeContainer Read (void);

  // end class OrbisTopologyReader
};

// end namespace ns3
};


#endif /* ORBIS_TOPOLOGY_READER_H */

