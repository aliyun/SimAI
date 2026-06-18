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
* Author: Author: Andrea Sacco <andrea.sacco85@gmail.com>
*/

#ifndef ACOUSTIC_MODEM_ENERGY_MODEL_HELPER_H
#define ACOUSTIC_MODEM_ENERGY_MODEL_HELPER_H

#include "ns3/energy-model-helper.h"
#include "ns3/acoustic-modem-energy-model.h"

namespace ns3 {

/**
 * \ingroup uan
 *
 * Assign AcousticModemEnergyModel to uan devices.
 *
 * This installer installs AcousticModemEnergyModel for only
 * UanNetDevice objects.
 */
class AcousticModemEnergyModelHelper : public DeviceEnergyModelHelper
{
public:
  /** 
   * Construct a helper which is used to add a radio energy model to a node.
   */
  AcousticModemEnergyModelHelper ();

  /**
   * Destroy an AcousticModemEnergy Helper.
   */
  ~AcousticModemEnergyModelHelper ();

  /**
   * Sets an attribute of the underlying energy model object.
   *
   * \param name The name of the attribute to set.
   * \param v The value of the attribute.
   */
  void Set (std::string name, const AttributeValue &v);

  /**
   * Sets the callback to be invoked when energy is depleted.
   *
   * \param callback Callback function for energy depletion handling.
   */
  void SetDepletionCallback (
    AcousticModemEnergyModel::AcousticModemEnergyDepletionCallback callback);


private:
  /**
   * Implements DeviceEnergyModel::Install.
   *
   * \param device Pointer to the NetDevice to install DeviceEnergyModel.
   * \param source Pointer to EnergySource installed on node.
   * \return The energy model.
   */
  virtual Ptr<DeviceEnergyModel> DoInstall (Ptr<NetDevice> device,
                                            Ptr<EnergySource> source) const;

private:
  /** Energy model factory. */
  ObjectFactory m_modemEnergy;

  /** Callback for energy depletion. */
  AcousticModemEnergyModel::AcousticModemEnergyDepletionCallback m_depletionCallback;

};

} // namespace ns3

#endif /* ACOUSTIC_MODEM_ENERGY_MODEL_HELPER_H */
