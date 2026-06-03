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

#ifndef TOPOLOGY_READER_HELPER_H
#define TOPOLOGY_READER_HELPER_H

#include "ns3/topology-reader.h"
#include <string>

/**
 * \file
 * \ingroup topology
 * ns3::TopologyHelper declaration.
 */

namespace ns3 {

/**
 * \ingroup topology
 *
 * \brief Helper class which makes it easier to configure and use a generic TopologyReader.
 */
class TopologyReaderHelper
{
public:
  TopologyReaderHelper ();

  /**
   * \brief Sets the input file name.
   * \param [in] fileName The input file name.
   */
  void SetFileName (const std::string fileName);

  /**
   * \brief Sets the input file type. Supported file types are "Orbis", "Inet", "Rocketfuel".
   * \param [in] fileType The input file type.
   */
  void SetFileType (const std::string fileType);

  /**
   * \brief Gets a Ptr<TopologyReader> to the actual TopologyReader.
   * \return The created Topology Reader (or null if there was an error).
   */
  Ptr<TopologyReader> GetTopologyReader ();

private:
  Ptr<TopologyReader> m_inputModel;  //!< Smart pointer to the actual topology model.
  std::string m_fileName;  //!< Name of the input file.
  std::string m_fileType;  //!< Type of the input file (e.g., "Inet", "Orbis", etc.).
};

} // namespace ns3


#endif /* TOPOLOGY_READER_HELPER_H */
