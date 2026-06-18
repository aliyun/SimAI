/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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
 * Authors: Sidharth Nabar <snabar@uw.edu>, He Wu <mdzz@u.washington.edu>
 */

#ifndef ENERGY_MODEL_HELPER_H
#define ENERGY_MODEL_HELPER_H

#include "ns3/attribute.h"
#include "ns3/object-factory.h"
#include "ns3/node-container.h"
#include "ns3/ptr.h"
#include "ns3/net-device.h"
#include "ns3/net-device-container.h"
#include "ns3/energy-source.h"
#include "ns3/device-energy-model.h"
#include "ns3/device-energy-model-container.h"
#include "energy-source-container.h"

namespace ns3 {

/**
 * \ingroup energy
 * \brief Creates EnergySource objects.
 *
 * This class creates and installs an energy source onto network nodes.
 * Multiple sources can exist on a network node.
 *
 */
class EnergySourceHelper
{
public:
  virtual ~EnergySourceHelper ();

  /**
   * \param name Name of attribute to set.
   * \param v Value of the attribute.
   *
   * Sets one of the attributes of underlying EnergySource.
   */
  virtual void Set (std::string name, const AttributeValue &v) = 0;

  /**
   * \param node Pointer to the node where EnergySource will be installed.
   * \returns An EnergySourceContainer which contains all the EnergySources.
   *
   * This function installs an EnergySource onto a node.
   */
  EnergySourceContainer Install (Ptr<Node> node) const;

  /**
   * \param c List of nodes where EnergySource will be installed.
   * \returns An EnergySourceContainer which contains all the EnergySources.
   *
   * This function installs an EnergySource onto a list of nodes.
   */
  EnergySourceContainer Install (NodeContainer c) const;

  /**
   * \param nodeName Name of node where EnergySource will be installed.
   * \returns An EnergySourceContainer which contains all the EnergySources.
   *
   * This function installs an EnergySource onto a node.
   */
  EnergySourceContainer Install (std::string nodeName) const;

  /**
   * \brief This function installs an EnergySource on all nodes in simulation.
   *
   * \returns An EnergySourceContainer which contains all the EnergySources.
   */
  EnergySourceContainer InstallAll (void) const;


private:
  /**
   * \param node Pointer to node where the energy source is to be installed.
   * \returns Pointer to the created EnergySource.
   *
   * Child classes of EnergySourceHelper only have to implement this function,
   * to create and aggregate an EnergySource object onto a single node. Rest of
   * the installation process (eg. installing EnergySource on set of nodes) is
   * implemented in the EnergySourceHelper base class.
   */
  virtual Ptr<EnergySource> DoInstall (Ptr<Node> node) const = 0;

};


/**
 * \ingroup energy
 * \brief Creates DeviceEnergyModel objects.
 *
 * This class helps to create and install DeviceEnergyModel onto NetDevice. A
 * DeviceEnergyModel is connected to a NetDevice (or PHY object) by callbacks.
 * Note that DeviceEnergyModel objects are *not* aggregated onto the node. They
 * can be accessed through the EnergySource object, which *is* aggregated onto
 * the node.
 *
 */
class DeviceEnergyModelHelper
{
public:
  virtual ~DeviceEnergyModelHelper ();

  /**
   * \param name Name of attribute to set.
   * \param v Value of the attribute.
   *
   * Sets one of the attributes of underlying DeviceEnergyModel.
   */
  virtual void Set (std::string name, const AttributeValue &v) = 0;

  /**
   * \param device Pointer to the NetDevice to install DeviceEnergyModel.
   * \param source The EnergySource the DeviceEnergyModel will be using.
   * \returns An DeviceEnergyModelContainer contains all the DeviceEnergyModels.
   *
   * Installs an DeviceEnergyModel with a specified EnergySource onto a
   * xNetDevice.
   */
  DeviceEnergyModelContainer Install (Ptr<NetDevice> device,
                                      Ptr<EnergySource> source) const;

  /**
   * \param deviceContainer List of NetDevices to be install DeviceEnergyModel
   * objects.
   * \param sourceContainer List of EnergySource the DeviceEnergyModel will be
   * using.
   * \returns An DeviceEnergyModelContainer contains all the DeviceEnergyModels.
   *
   * Installs DeviceEnergyModels with specified EnergySources onto a list of
   * NetDevices.
   */
  DeviceEnergyModelContainer Install (NetDeviceContainer deviceContainer,
                                      EnergySourceContainer sourceContainer) const;


private:
  /**
   * \param device The net device corresponding to DeviceEnergyModel object.
   * \param source The EnergySource the DeviceEnergyModel will be using.
   * \returns Pointer to the created DeviceEnergyModel.
   *
   * Child classes of DeviceEnergyModelHelper only have to implement this
   * function, to create and aggregate an DeviceEnergyModel object onto a single
   * node. The rest of the installation process (eg. installing EnergySource on
   * set of nodes) is implemented in the DeviceEnergyModelHelper base class.
   */
  virtual Ptr<DeviceEnergyModel> DoInstall (Ptr<NetDevice> device,
                                            Ptr<EnergySource> source) const = 0;

};

} // namespace ns3

#endif  /* ENERGY_MODEL_HELPER_H */
