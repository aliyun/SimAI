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
 *
 * Copyright (c) 2014 Wireless Communications and Networking Group (WCNG),
 * University of Rochester, Rochester, NY, USA.
 *
 * Modifications made by: Cristiano Tapparello <cristiano.tapparello@rochester.edu>
 */


#ifndef ENERGY_SOURCE_H
#define ENERGY_SOURCE_H

#include "ns3/object.h"
#include "ns3/ptr.h"
#include "ns3/type-id.h"
#include "ns3/node.h"
#include "device-energy-model-container.h"  // #include "device-energy-model.h"
#include "ns3/energy-harvester.h"

namespace ns3 {
  
/**
 * \defgroup energy Energy Models
 */

/**
 * \ingroup energy
 * \ingroup tests
 * \defgroup energy-tests Energy module tests
 */

/**
 * \ingroup energy
 *
 * \brief Energy source base class.
 *
 * This is the base class for energy sources. Energy sources keep track of
 * remaining energy. Device energy models will be updating the remaining energy
 * in the energy source. The energy source itself does not update the remaining
 * energy. Energy source also keeps a list of device energy models installed on
 * the same node. When the remaining energy level reaches 0, the energy source
 * will notify all device energy models stored in the list.
 *
 * EnergySource provides 2 types of interfaces for DeviceEnergyModels to update
 * the remaining energy stored in EnergySource:
 *  -Direct energy update interface (Joules):
 *    DecreaseRemainingEnergy
 *    IncreaseRemainingEnergy
 *  -Indirect energy update interface (Current):
 *    UpdateEnergySource
 * Direct energy update interface will decrease/increase energy from the source
 * directly (in Joules). Direct energy update interface is typically used by
 * simple DeviceEnergyModel which knows only average power consumption for each
 * of its state.
 * Indirect energy update interface uses the total current cumulated from all
 * DeviceEnergyModel to calculate energy to decrease from the source. Indirect
 * energy update interface is typically used by DeviceEnergyModel who knows its
 * current draw for each of its states. Nonlinear EnergySource also uses this
 * interface.
 *
 * Unit of energy is chosen as Joules since energy models typically calculate
 * energy as (time in seconds * power in Watts). If the energy source stores
 * energy in different units (eg. kWh), a simple converter function should
 * suffice.
 */
  
class EnergyHarvester;
  
class EnergySource : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);
  EnergySource ();
  virtual ~EnergySource ();

  /**
   * \returns Supply voltage of the energy source.
   *
   * Set method is to be defined in child class only if necessary. For sources
   * with a fixed supply voltage, set method is not needed.
   */
  virtual double GetSupplyVoltage (void) const = 0;

  /**
   * \returns Initial energy (capacity) of the energy source.
   *
   * Set method is to be defined in child class only if necessary. For sources
   * with a fixed initial energy (energy capacity), set method is not needed.
   */
  virtual double GetInitialEnergy (void) const = 0;

  /**
   * \returns Remaining energy at the energy source.
   */
  virtual double GetRemainingEnergy (void) = 0;

  /**
   * \return Energy fraction = remaining energy / initial energy [0, 1]
   *
   * This function returns the percentage of energy left in the energy source.
   */
  virtual double GetEnergyFraction (void) = 0;

  /**
   * This function goes through the list of DeviceEnergyModels to obtain total
   * current draw at the energy source and updates remaining energy. Called by
   * DeviceEnergyModels to inform EnergySource of a state change.
   */
  virtual void UpdateEnergySource (void) = 0;

  /**
   * \brief Sets pointer to node containing this EnergySource.
   *
   * \param node Pointer to node containing this EnergySource.
   */
  void SetNode (Ptr<Node> node);

  /**
   * \returns Pointer to node containing this EnergySource.
   *
   * When a subclass needs to get access to the underlying node base class to
   * print the nodeId for example, it can invoke this method.
   */
  Ptr<Node> GetNode (void) const;

  /**
   * \param deviceEnergyModelPtr Pointer to device energy model.
   *
   * This function appends a device energy model to the end of a list of
   * DeviceEnergyModelInfo structs.
   */
  void AppendDeviceEnergyModel (Ptr<DeviceEnergyModel> deviceEnergyModelPtr);

  /**
   * \param tid TypeId of the DeviceEnergyModel we are searching for.
   * \returns List of pointers to DeviceEnergyModel objects installed on node.
   */
  DeviceEnergyModelContainer FindDeviceEnergyModels (TypeId tid);

  /**
   * \param name name of the DeviceEnergyModel we are searching for.
   * \returns List of pointers to DeviceEnergyModel objects installed on node.
   */
  DeviceEnergyModelContainer FindDeviceEnergyModels (std::string name);

  /**
   * Calls Start () method of the device energy models. Device energy models are
   * not aggregated to the node, therefore we need to manually start them here.
   * Called by EnergySourceContainer, which is aggregated to the node.
   */
  void InitializeDeviceModels (void);

  /**
   * Calls Dispose () method of the device energy models. Device energy models
   * are not aggregated to the node, therefore we need to manually start them
   * here. Called by EnergySourceContainer, which is aggregated to the node.
   */
  void DisposeDeviceModels (void);
  
  /**
   * \param energyHarvesterPtr Pointer to energy harvester.
   *
   * This function connect an energy harvester to the energy source. After the
   * execution of this method, the pointer to the energy harvester is appended
   * to the end of a vector of EnergyHarvester pointer.
   * Note that the order in which different energy harvester are added to the
   * energy source does not impact the simulation results.
   */
  void ConnectEnergyHarvester (Ptr<EnergyHarvester> energyHarvesterPtr);


private:
  /**
   * All child's implementation must call BreakDeviceEnergyModelRefCycle to
   * ensure reference cycles to DeviceEnergyModel objects are broken.
   *
   * Defined in ns3::Object
   */
  virtual void DoDispose (void);

private:
  /**
   * List of device energy models installed on the same node.
   */
  DeviceEnergyModelContainer m_models;

  /**
   * Pointer to node containing this EnergySource. Used by helper class to make
   * sure device models are installed onto the corresponding node.
   */
  Ptr<Node> m_node;
    
  /**
  * Vector of EnergyHarvester pointer connected to the same energy source.
  * This vector is used by the CalculateTotalCurrent method to determine the
  * total power provided by the energy harvesters connected to the energy source.
  */
  std::vector< Ptr<EnergyHarvester> > m_harvesters;


protected:
  /**
   * \returns Total current draw from all DeviceEnergyModels.
   */
  double CalculateTotalCurrent (void);

  /**
   * This function notifies all DeviceEnergyModel of energy depletion event. It
   * is called by the child EnergySource class when energy depletion happens.
   */
  void NotifyEnergyDrained (void);

  /**
   * This function notifies all DeviceEnergyModel of energy recharged event. It
   * is called by the child EnergySource class when energy source is recharged.
   */
  void NotifyEnergyRecharged (void);

  /**
   * This function notifies all DeviceEnergyModel of energy changed event. It
   * is called by the child EnergySource class when energy source is changed.
   */
  void NotifyEnergyChanged (void);

  /**
   * This function is called to break reference cycle between EnergySource and
   * DeviceEnergyModel. Child of the EnergySource base class must call this
   * function in their implementation of DoDispose to make sure the reference
   * cycle is broken.
   *
   * Normally this work will be completed by the DoDispose function. However it
   * will be overridden in the child class. Hence we introduced this function.
   */
  void BreakDeviceEnergyModelRefCycle (void);

};

} // namespace ns3

#endif /* ENERGY_SOURCE_H */
