/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Jaume Nin <jaume.nin@cttc,cat>
 * Based on NodeList implementation by Mathieu Lacage  <mathieu.lacage@sophia.inria.fr>
 *
 */

#ifndef BUILDING_LIST_H_
#define BUILDING_LIST_H_

#include <vector>
#include "ns3/ptr.h"

namespace ns3 {

class Building;

/**
 * Container for Building class
 */
class BuildingList
{
public:
  /// Const Iterator
  typedef std::vector< Ptr<Building> >::const_iterator Iterator;

  /**
   * \param building building to add
   * \returns index of building in list.
   *
   * This method is called automatically from Building::Building so
   * the user has little reason to call it himself.
   */
  static uint32_t Add (Ptr<Building> building);
  /**
   * \returns a C++ iterator located at the beginning of this
   *          list.
   */
  static Iterator Begin (void);
  /**
   * \returns a C++ iterator located at the end of this
   *          list.
   */
  static Iterator End (void);
  /**
   * \param n index of requested building.
   * \returns the Building associated to index n.
   */
  static Ptr<Building> GetBuilding (uint32_t n);
  /**
   * \returns the number of buildings currently in the list.
   */
  static uint32_t GetNBuildings (void);
};

} // namespace ns3

#endif /* BUILDING_LIST_H_ */


