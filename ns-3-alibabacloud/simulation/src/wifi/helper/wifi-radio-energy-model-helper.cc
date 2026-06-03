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

#include "wifi-radio-energy-model-helper.h"
#include "ns3/wifi-net-device.h"
#include "ns3/wifi-tx-current-model.h"
#include "ns3/wifi-phy.h"

namespace ns3 {

WifiRadioEnergyModelHelper::WifiRadioEnergyModelHelper ()
{
  m_radioEnergy.SetTypeId ("ns3::WifiRadioEnergyModel");
  m_depletionCallback.Nullify ();
  m_rechargedCallback.Nullify ();
}

WifiRadioEnergyModelHelper::~WifiRadioEnergyModelHelper ()
{
}

void
WifiRadioEnergyModelHelper::Set (std::string name, const AttributeValue &v)
{
  m_radioEnergy.Set (name, v);
}

void
WifiRadioEnergyModelHelper::SetDepletionCallback (
  WifiRadioEnergyModel::WifiRadioEnergyDepletionCallback callback)
{
  m_depletionCallback = callback;
}

void
WifiRadioEnergyModelHelper::SetRechargedCallback (
  WifiRadioEnergyModel::WifiRadioEnergyRechargedCallback callback)
{
  m_rechargedCallback = callback;
}

void
WifiRadioEnergyModelHelper::SetTxCurrentModel (std::string name,
                                               std::string n0, const AttributeValue& v0,
                                               std::string n1, const AttributeValue& v1,
                                               std::string n2, const AttributeValue& v2,
                                               std::string n3, const AttributeValue& v3,
                                               std::string n4, const AttributeValue& v4,
                                               std::string n5, const AttributeValue& v5,
                                               std::string n6, const AttributeValue& v6,
                                               std::string n7, const AttributeValue& v7)
{
  ObjectFactory factory;
  factory.SetTypeId (name);
  factory.Set (n0, v0);
  factory.Set (n1, v1);
  factory.Set (n2, v2);
  factory.Set (n3, v3);
  factory.Set (n4, v4);
  factory.Set (n5, v5);
  factory.Set (n6, v6);
  factory.Set (n7, v7);
  m_txCurrentModel = factory;
}


/*
 * Private function starts here.
 */

Ptr<DeviceEnergyModel>
WifiRadioEnergyModelHelper::DoInstall (Ptr<NetDevice> device,
                                       Ptr<EnergySource> source) const
{
  NS_ASSERT (device != NULL);
  NS_ASSERT (source != NULL);
  // check if device is WifiNetDevice
  std::string deviceName = device->GetInstanceTypeId ().GetName ();
  if (deviceName.compare ("ns3::WifiNetDevice") != 0)
    {
      NS_FATAL_ERROR ("NetDevice type is not WifiNetDevice!");
    }
  Ptr<Node> node = device->GetNode ();
  Ptr<WifiRadioEnergyModel> model = m_radioEnergy.Create ()->GetObject<WifiRadioEnergyModel> ();
  NS_ASSERT (model != NULL);

  // set energy depletion callback
  // if none is specified, make a callback to WifiPhy::SetOffMode
  Ptr<WifiNetDevice> wifiDevice = DynamicCast<WifiNetDevice> (device);
  Ptr<WifiPhy> wifiPhy = wifiDevice->GetPhy ();
  wifiPhy->SetWifiRadioEnergyModel (model);
  if (m_depletionCallback.IsNull ())
    {
      model->SetEnergyDepletionCallback (MakeCallback (&WifiPhy::SetOffMode, wifiPhy));
    }
  else
    {
      model->SetEnergyDepletionCallback (m_depletionCallback);
    }
  // set energy recharged callback
  // if none is specified, make a callback to WifiPhy::ResumeFromOff
  if (m_rechargedCallback.IsNull ())
    {
      model->SetEnergyRechargedCallback (MakeCallback (&WifiPhy::ResumeFromOff, wifiPhy));
    }
  else
    {
      model->SetEnergyRechargedCallback (m_rechargedCallback);
    }
  // add model to device model list in energy source
  source->AppendDeviceEnergyModel (model);
  // set energy source pointer
  model->SetEnergySource (source);
  // create and register energy model PHY listener
  wifiPhy->RegisterListener (model->GetPhyListener ());
  //
  if (m_txCurrentModel.GetTypeId ().GetUid ())
    {
      Ptr<WifiTxCurrentModel> txcurrent = m_txCurrentModel.Create<WifiTxCurrentModel> ();
      model->SetTxCurrentModel (txcurrent);
    }
  return model;
}

} // namespace ns3
