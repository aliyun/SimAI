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

#include "acoustic-modem-energy-model-helper.h"
#include "ns3/basic-energy-source-helper.h"
#include "ns3/uan-phy.h"
#include "ns3/uan-net-device.h"
#include "ns3/config.h"
#include "ns3/names.h"

namespace ns3 {

AcousticModemEnergyModelHelper::AcousticModemEnergyModelHelper ()
{
  m_modemEnergy.SetTypeId ("ns3::AcousticModemEnergyModel");
  m_depletionCallback.Nullify ();
}

AcousticModemEnergyModelHelper::~AcousticModemEnergyModelHelper ()
{
}

void
AcousticModemEnergyModelHelper::Set (std::string name, const AttributeValue &v)
{
  m_modemEnergy.Set (name, v);
}

void
AcousticModemEnergyModelHelper::SetDepletionCallback (
  AcousticModemEnergyModel::AcousticModemEnergyDepletionCallback callback)
{
  m_depletionCallback = callback;
}

/*
 * Private function starts here.
 */

Ptr<DeviceEnergyModel>
AcousticModemEnergyModelHelper::DoInstall (Ptr<NetDevice> device,
                                           Ptr<EnergySource> source) const
{
  NS_ASSERT (device != NULL);
  NS_ASSERT (source != NULL);
  // check if device is UanNetDevice
  std::string deviceName = device->GetInstanceTypeId ().GetName ();
  if (deviceName.compare ("ns3::UanNetDevice") != 0)
    {
      NS_FATAL_ERROR ("NetDevice type is not UanNetDevice!");
    }
  Ptr<Node> node = device->GetNode ();
  Ptr<AcousticModemEnergyModel> model = m_modemEnergy.Create<AcousticModemEnergyModel> ();
  NS_ASSERT (model != NULL);
  // set node pointer
  model->SetNode (node);
  // set energy source pointer
  model->SetEnergySource (source);
  // get phy layer
  Ptr<UanNetDevice> uanDevice = DynamicCast<UanNetDevice> (device);
  Ptr<UanPhy> uanPhy = uanDevice->GetPhy ();
  // set energy depletion callback
  model->SetEnergyDepletionCallback (m_depletionCallback);
  // add model to device model list in energy source
  source->AppendDeviceEnergyModel (model);
  // set node pointer
  source->SetNode (node);
  // create and install energy model callback
  DeviceEnergyModel::ChangeStateCallback cb;
  cb = MakeCallback (&DeviceEnergyModel::ChangeState, model);
  uanPhy->SetEnergyModelCallback (cb);

  return model;
}

} // namespace ns3
