/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr> (original node-container.h)
 *         Nicola Baldo (wrote building-container.h based on node-container.h)
 */
#ifndef BUILDING_CONTAINER_H
#define BUILDING_CONTAINER_H

#include <stdint.h>
#include <vector>
#include <ns3/building.h>

namespace ns3 {

/**
 * \brief keep track of a set of building pointers.
 *
 * Some ns-3 helpers operate on more than one building at a time.  For example
 * a PositionAllocator may want to position nodes on a set of buildings.
 * The helper methods will then usually take a BuildingContainer as a
 * parameter.  BuildingContainers hold the multiple Ptr<Building> which are used
 * to refer to the buildings.
 */
class BuildingContainer
{
public:
  /// Const iterator
  typedef std::vector<Ptr<Building> >::const_iterator Iterator;

  /**
   * Create an empty BuildingContainer.
   */
  BuildingContainer ();

  /**
   * Create a BuildingContainer with exactly one building which has been previously
   * instantiated.  The single Building is specified by a smart pointer.
   *
   * \param building The Ptr<Building> to add to the container.
   */
  BuildingContainer (Ptr<Building> building);

  /**
   * Create a BuildingContainer with exactly one building which has been previously 
   * instantiated and assigned a name using the Object Name Service.  This 
   * Building is then specified by its assigned name. 
   *
   * \param buildingName The name of the Building Object to add to the container.
   */
  BuildingContainer (std::string buildingName);

  /**
   * \brief Get an iterator which refers to the first Building in the 
   * container.
   *
   * Buildings can be retrieved from the container in two ways.  First,
   * directly by an index into the container, and second, using an iterator.
   * This method is used in the iterator method and is typically used in a 
   * for-loop to run through the Buildings
   *
   * \code
   *   BuildingContainer::Iterator i;
   *   for (i = container.Begin (); i != container.End (); ++i)
   *     {
   *       (*i)->method ();  // some Building method
   *     }
   * \endcode
   *
   * \returns an iterator which refers to the first Building in the container.
   */
  Iterator Begin (void) const;

  /**
   * \brief Get an iterator which indicates past-the-last Building in the 
   * container.
   *
   * Buildings can be retrieved from the container in two ways.  First,
   * directly by an index into the container, and second, using an iterator.
   * This method is used in the iterator method and is typically used in a 
   * for-loop to run through the Buildings
   *
   * \code
   *   BuildingContainer::Iterator i;
   *   for (i = container.Begin (); i != container.End (); ++i)
   *     {
   *       (*i)->method ();  // some Building method
   *     }
   * \endcode
   *
   * \returns an iterator which indicates an ending condition for a loop.
   */
  Iterator End (void) const;

  /**
   * \brief Get the number of Ptr<Building> stored in this container.
   *
   * Buildings can be retrieved from the container in two ways.  First,
   * directly by an index into the container, and second, using an iterator.
   * This method is used in the direct method and is typically used to
   * define an ending condition in a for-loop that runs through the stored
   * Buildings
   *
   * \code
   *   uint32_t nBuildings = container.GetN ();
   *   for (uint32_t i = 0 i < nBuildings; ++i)
   *     {
   *       Ptr<Building> p = container.Get (i)
   *       i->method ();  // some Building method
   *     }
   * \endcode
   *
   * \returns the number of Ptr<Building> stored in this container.
   */
  uint32_t GetN (void) const;

  /**
   * \brief Get the Ptr<Building> stored in this container at a given
   * index.
   *
   * Buildings can be retrieved from the container in two ways.  First,
   * directly by an index into the container, and second, using an iterator.
   * This method is used in the direct method and is used to retrieve the
   * indexed Ptr<Appliation>.
   *
   * \code
   *   uint32_t nBuildings = container.GetN ();
   *   for (uint32_t i = 0 i < nBuildings; ++i)
   *     {
   *       Ptr<Building> p = container.Get (i)
   *       i->method ();  // some Building method
   *     }
   * \endcode
   *
   * \param i the index of the requested building pointer.
   * \returns the requested building pointer.
   */
  Ptr<Building> Get (uint32_t i) const;

  /**
   * \brief Create n buildings and append pointers to them to the end of this 
   * BuildingContainer.
   *
   * Buildings are at the heart of any ns-3 simulation.  One of the first tasks that
   * any simulation needs to do is to create a number of buildings.  This method
   * automates that task.
   *
   * \param n The number of Buildings to create
   */
  void Create (uint32_t n);

  /**
   * \brief Append the contents of another BuildingContainer to the end of
   * this container.
   *
   * \param other The BuildingContainer to append.
   */
  void Add (BuildingContainer other);

  /**
   * \brief Append a single Ptr<Building> to this container.
   *
   * \param building The Ptr<Building> to append.
   */
  void Add (Ptr<Building> building);

  /**
   * \brief Append to this container the single Ptr<Building> referred to
   * via its object name service registered name.
   *
   * \param buildingName The name of the Building Object to add to the container.
   */
  void Add (std::string buildingName);

  /**
   * \brief Create a BuildingContainer that contains a list of _all_ buildings
   * stored in the ns3::BuildingList.
   *
   * Whenever a Building is created, a Ptr<Building> is added to a global list of all
   * buildings in the system.  It is sometimes useful to be able to get to all
   * buildings in one place.  This method creates a BuildingContainer that is 
   * initialized to contain all of the simulation buildings,
   *
   * \returns a BuildingContainer which contains a list of all Buildings.
   */
  static BuildingContainer GetGlobal (void);

private:
  std::vector<Ptr<Building> > m_buildings; //!< Building container
};

} // namespace ns3

#endif /* BUILDING_CONTAINER_H */
