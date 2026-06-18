/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 Andrea Sacco
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
 * Author: Andrea Sacco <andrea.sacco85@gmail.com>
 */

#ifndef SIMPLE_DEVICE_ENERGY_MODEL_H
#define SIMPLE_DEVICE_ENERGY_MODEL_H

#include "ns3/device-energy-model.h"
#include "ns3/nstime.h"
#include "ns3/traced-value.h"

namespace ns3 {

/**
 * \ingroup energy
 *
 * A simple device energy model where current drain can be set by the user.
 *
 * It is supposed to be used as a testing model for energy sources.
 *
 */
class SimpleDeviceEnergyModel : public DeviceEnergyModel
{
public:
  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);
  SimpleDeviceEnergyModel ();
  virtual ~SimpleDeviceEnergyModel ();

  /**
   * \brief Sets pointer to node.
   *
   * \param node Pointer to node.
   *
   * Implements DeviceEnergyModel::SetNode.
   */
  virtual void SetNode (Ptr<Node> node);

  /**
   * \brief Gets pointer to node.
   *
   * \returns Pointer to node.
   *
   * Implements DeviceEnergyModel::GetNode.
   */
  virtual Ptr<Node> GetNode (void) const;

  /**
   * \brief Sets pointer to EnergySouce installed on node.
   *
   * \param source Pointer to EnergySource installed on node.
   *
   * Implements DeviceEnergyModel::SetEnergySource.
   */
  virtual void SetEnergySource (Ptr<EnergySource> source);

  /**
   * \returns Total energy consumption of the vehicle.
   *
   * Implements DeviceEnergyModel::GetTotalEnergyConsumption.
   */
  virtual double GetTotalEnergyConsumption (void) const;

  /**
   * \param newState New state the device is in.
   *
   * Not implemented
   */
  virtual void ChangeState (int newState)
  {
  }

  /**
   * \brief Handles energy depletion.
   *
   * Not implemented
   */
  virtual void HandleEnergyDepletion (void)
  {
  }

  /**
   * \brief Handles energy recharged.
   *
   * Not implemented
   */
  virtual void HandleEnergyRecharged (void)
  {
  }

  /**
   * \brief Handles energy changed.
   *
   * Not implemented
   */
  virtual void HandleEnergyChanged (void)
  {
  }

  /**
   * \param current the current draw of device.
   *
   * Set the actual current draw of the device.
   */
  void SetCurrentA (double current);
private:
  void DoDispose (void);

  /**
   * \returns Current draw of device, at current state.
   *
   * Implements DeviceEnergyModel::GetCurrentA.
   */
  virtual double DoGetCurrentA (void) const;

  Time m_lastUpdateTime;        //!< Last update time
  double m_actualCurrentA;      //!< actual curred (in Ampere)
  Ptr<EnergySource> m_source;   //!< Energy source
  Ptr<Node> m_node;             //!< Node
  TracedValue<double> m_totalEnergyConsumption; //!< Total energy consumption trace
};

} // namespace ns3

#endif /* SIMPLE_DEVICE_ENERGY_MODEL_H */
