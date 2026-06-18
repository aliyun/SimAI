/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
 * Copyright (c) 2010 Network Security Lab, University of Washington, Seattle.
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
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *          Sidharth Nabar <snabar@uw.edu>, He Wu <mdzz@u.washington.edu>
 */

#ifndef DEVICE_ENERGY_MODEL_CONTAINER_H
#define DEVICE_ENERGY_MODEL_CONTAINER_H

#include "ns3/device-energy-model.h"
#include <vector>
#include <stdint.h>

namespace ns3 {

/**
 * \ingroup energy
 * \brief Holds a vector of ns3::DeviceEnergyModel pointers.
 *
 * DeviceEnergyModelContainer returns a list of DeviceEnergyModel pointers
 * installed on a node. Users can use this list to access DeviceEnergyModel
 * objects to obtain total device energy consumption on a node easily.
 *
 * \see NetDeviceContainer
 *
 */
class DeviceEnergyModelContainer
{
public:
  /// Const iterator of DeviceEnergyModel container
  typedef std::vector< Ptr<DeviceEnergyModel> >::const_iterator Iterator;

public:
  /**
   * Creates an empty DeviceEnergyModelContainer.
   */
  DeviceEnergyModelContainer ();

  /**
   * \param model Pointer to a DeviceEnergyModel.
   *
   * Creates a DeviceEnergyModelContainer with exactly one DeviceEnergyModel
   * previously instantiated.
   */
  DeviceEnergyModelContainer (Ptr<DeviceEnergyModel> model);

  /**
   * \param modelName Name of DeviceEnergyModel.
   *
   * Creates an DeviceEnergyModelContainer with exactly one DeviceEnergyModel
   * previously instantiated and assigned a name using the Object name service.
   * This DeviceEnergyModel is specified by its assigned name.
   */
  DeviceEnergyModelContainer (std::string modelName);

  /**
   * \param a A DeviceEnergyModelContainer.
   * \param b Another DeviceEnergyModelContainer.
   *
   * Creates a DeviceEnergyModelContainer by concatenating DeviceEnergyModelContainer b
   * to DeviceEnergyModelContainer a.
   *
   * \note Can be used to concatenate 2 Ptr<DeviceEnergyModel> directly. C++
   * will be calling DeviceEnergyModelContainer constructor with Ptr<DeviceEnergyModel>
   * first.
   */
  DeviceEnergyModelContainer (const DeviceEnergyModelContainer &a,
                              const DeviceEnergyModelContainer &b);

  /**
   * \brief Get an iterator which refers to the first DeviceEnergyModel pointer
   * in the container.
   *
   * \returns An iterator which refers to the first DeviceEnergyModel in container.
   *
   * DeviceEnergyModels can be retrieved from the container in two ways. First,
   * directly by an index into the container, and second, using an iterator.
   * This method is used in the iterator method and is typically used in a
   * for-loop to run through the DeviceEnergyModels.
   *
   * \code
   *   DeviceEnergyModelContainer::Iterator i;
   *   for (i = container.Begin (); i != container.End (); ++i)
   *     {
   *       (*i)->method ();  // some DeviceEnergyModel method
   *     }
   * \endcode
   */
  Iterator Begin (void) const;

  /**
   * \brief Get an iterator which refers to the last DeviceEnergyModel pointer
   * in the container.
   *
   * \returns An iterator which refers to the last DeviceEnergyModel in container.
   *
   * DeviceEnergyModels can be retrieved from the container in two ways. First,
   * directly by an index into the container, and second, using an iterator.
   * This method is used in the iterator method and is typically used in a
   * for-loop to run through the DeviceEnergyModels.
   *
   * \code
   *   DeviceEnergyModelContainer::Iterator i;
   *   for (i = container.Begin (); i != container.End (); ++i)
   *     {
   *       (*i)->method ();  // some DeviceEnergyModel method
   *     }
   * \endcode
   */
  Iterator End (void) const;

  /**
   * \brief Get the number of Ptr<DeviceEnergyModel> stored in this container.
   *
   * \returns The number of Ptr<DeviceEnergyModel> stored in this container.
   */
  uint32_t GetN (void) const;

  /**
   * \brief Get the i-th Ptr<DeviceEnergyModel> stored in this container.
   *
   * \param i Index of the requested Ptr<DeviceEnergyModel>.
   * \returns The requested Ptr<DeviceEnergyModel>.
   */
  Ptr<DeviceEnergyModel> Get (uint32_t i) const;

  /**
   * \param container Another DeviceEnergyModelContainer.
   *
   * Appends the contents of another DeviceEnergyModelContainer to the end of
   * this DeviceEnergyModelContainer.
   */
  void Add (DeviceEnergyModelContainer container);

  /**
   * \brief Append a single Ptr<DeviceEnergyModel> to the end of this container.
   *
   * \param model Pointer to an DeviceEnergyModel.
   */
  void Add (Ptr<DeviceEnergyModel> model);

  /**
   * \brief Append a single Ptr<DeviceEnergyModel> referred to by its object
   * name to the end of this container.
   *
   * \param modelName Name of DeviceEnergyModel object.
   */
  void Add (std::string modelName);

  /**
   * \brief Removes all elements in the container.
   */
  void Clear (void);

private:
  std::vector< Ptr<DeviceEnergyModel> > m_models; //!< Container of Energy models

};

} // namespace ns3

#endif /* DEVICE_ENERGY_MODEL_CONTAINER_H */
