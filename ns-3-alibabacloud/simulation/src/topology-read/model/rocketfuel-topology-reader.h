/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 Hajime Tazaki
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
 * Author: Hajime Tazaki (tazaki@sfc.wide.ad.jp)
 */

#ifndef ROCKETFUEL_TOPOLOGY_READER_H
#define ROCKETFUEL_TOPOLOGY_READER_H

#include "topology-reader.h"

/**
 * \file
 * \ingroup topology
 * ns3::RocketfuelTopologyReader declaration.
 */

namespace ns3 {


// ------------------------------------------------------------
// --------------------------------------------
/**
 * \ingroup topology
 *
 * \brief Topology file reader (Rocketfuel-format type).
 *
 * http://www.cs.washington.edu/research/networking/rocketfuel/
 *
 * May 2nd, 2010: Currently only support "weights" file and "cch" file.
 * http://www.cs.washington.edu/research/networking/rocketfuel/maps/weights-dist.tar.gz
 * http://www.cs.washington.edu/research/networking/rocketfuel/maps/rocketfuel_maps_cch.tar.gz
 */
class RocketfuelTopologyReader : public TopologyReader
{
public:
  /**
   * \brief Get the type ID.
   * \return The object TypeId
   */
  static TypeId GetTypeId (void);

  RocketfuelTopologyReader ();
  virtual ~RocketfuelTopologyReader ();

  // Delete copy constructor and assignment operator to avoid misuse
  RocketfuelTopologyReader (const RocketfuelTopologyReader &) = delete;
  RocketfuelTopologyReader & operator = (const RocketfuelTopologyReader &) = delete;

  /**
   * \brief Main topology reading function.
   *
   * This method opens an input stream and reads the Rocketfuel-format file.
   * Every row represents a topology link (the ids of a couple of nodes),
   * so the input file is read line by line to figure out how many links
   * and nodes are in the topology.
   *
   * \return The container of the nodes created (or empty container if there was an error)
   */
  virtual NodeContainer Read (void);

private:
  /**
   * \brief Topology read function from a file containing the nodes map.
   *
   * Parser for the *.cch file available at:
   * http://www.cs.washington.edu/research/networking/rocketfuel/maps/rocketfuel_maps_cch.tar.gz
   *
   * \param [in] argc Argument counter.
   * \param [in] argv Argument vector.
   * \return The container of the nodes created (or empty container if there was an error).
   */
  NodeContainer GenerateFromMapsFile (int argc, char *argv[]);

  /**
   * \brief Topology read function from a file containing the nodes weights.
   *
   * Parser for the weights.* file available at:
   * http://www.cs.washington.edu/research/networking/rocketfuel/maps/weights-dist.tar.gz
   *
   * \param [in] argc Argument counter.
   * \param [in] argv Argument vector.
   * \return The container of the nodes created (or empty container if there was an error).
   */
  NodeContainer GenerateFromWeightsFile (int argc, char *argv[]);

  /**
   * \brief Enum of the possible file types.
   */
  enum RF_FileType
  {
    RF_MAPS,
    RF_WEIGHTS,
    RF_UNKNOWN
  };


  /**
   * \brief Classifies the file type according to its content.
   *
   * \param buf the first line of the file being read
   * \return The file type (RF_MAPS, RF_WEIGHTS, or RF_UNKNOWN)
   */
  enum RF_FileType GetFileType (const char *buf);

  int m_linksNumber; //!< Number of links.
  int m_nodesNumber; //!< Number of nodes.
  std::map<std::string, Ptr<Node> > m_nodeMap; //!< Map of the nodes (name, node).

  // end class RocketfuelTopologyReader
};

// end namespace ns3
};


#endif /* ROCKETFUEL_TOPOLOGY_READER_H */


