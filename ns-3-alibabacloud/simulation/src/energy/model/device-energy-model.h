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

#ifndef DEVICE_ENERGY_MODEL_H
#define DEVICE_ENERGY_MODEL_H

#include "ns3/object.h"
#include "ns3/ptr.h"
#include "ns3/type-id.h"
#include "ns3/node.h"

namespace ns3 {

class EnergySource;

/**
 * \ingroup energy
 * \brief Base class for device energy models.
 *
 * A device energy model should represent the energy consumption behavior of a
 * specific device. It will update remaining energy stored in the EnergySource
 * object installed on node. When energy is depleted, each DeviceEnergyModel
 * object installed on the same node will be informed by the EnergySource.
 *
 */
class DeviceEnergyModel : public Object
{
public:
  /**
   * Callback type for ChangeState function. Devices uses this state to notify
   * DeviceEnergyModel of a state change.
   */
  typedef Callback<void, int> ChangeStateCallback;

public:
  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);
  DeviceEnergyModel ();
  virtual ~DeviceEnergyModel ();

  /**
   * \param source Pointer to energy source installed on node.
   *
   * This function sets the pointer to energy source installed on node. Should
   * be called only by DeviceEnergyModel helper classes.
   */
  virtual void SetEnergySource (Ptr<EnergySource> source) = 0;

  /**
   * \returns Total energy consumption of the device.
   *
   * DeviceEnergyModel records its own energy consumption during simulation.
   */
  virtual double GetTotalEnergyConsumption (void) const = 0;

  /**
   * \param newState New state the device is in.
   *
   * DeviceEnergyModel is a state based model. This function is implemented by
   * all child of DeviceEnergyModel to change the model's state. States are to
   * be defined by each child using an enum (int).
   */
  virtual void ChangeState (int newState) = 0;

  /**
   * \returns Current draw of the device, in Ampere.
   *
   * This function returns the current draw at the device in its current state.
   * This function is called from the EnergySource to obtain the total current
   * draw at any given time during simulation.
   */
  double GetCurrentA (void) const;

  /**
   * This function is called by the EnergySource object when energy stored in
   * the energy source is depleted. Should be implemented by child classes.
   */
  virtual void HandleEnergyDepletion (void) = 0;

  /**
   * This function is called by the EnergySource object when energy stored in
   * the energy source is recharged. Should be implemented by child classes.
   */
  virtual void HandleEnergyRecharged (void) = 0;

  /**
   * This function is called by the EnergySource object when energy stored in
   * the energy source is changed. Should be implemented by child classes.
   */
  virtual void HandleEnergyChanged (void) = 0;


private:
  /**
   * \returns 0.0 as the current value, in Ampere.
   *
   * Child class does not have to implement this method if current draw for its
   * states are not know. This default method will always return 0.0A. For the
   * devices who does know the current draw of its states, this method must be
   * overwritten.
   */
  virtual double DoGetCurrentA (void) const;

};

}

#endif /* DEVICE_ENERGY_MODEL_H */
