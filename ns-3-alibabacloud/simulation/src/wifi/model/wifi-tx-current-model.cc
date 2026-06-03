/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Universita' degli Studi di Napoli "Federico II"
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
 * Author: Stefano Avallone <stefano.avallone@unina.it>
 */

#include "ns3/log.h"
#include "ns3/double.h"
#include "wifi-tx-current-model.h"
#include "wifi-utils.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("WifiTxCurrentModel");

NS_OBJECT_ENSURE_REGISTERED (WifiTxCurrentModel);

TypeId 
WifiTxCurrentModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::WifiTxCurrentModel")
    .SetParent<Object> ()
    .SetGroupName ("Wifi")
  ;
  return tid;
}

WifiTxCurrentModel::WifiTxCurrentModel()
{
}

WifiTxCurrentModel::~WifiTxCurrentModel()
{
}

NS_OBJECT_ENSURE_REGISTERED (LinearWifiTxCurrentModel);

TypeId 
LinearWifiTxCurrentModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::LinearWifiTxCurrentModel")
    .SetParent<WifiTxCurrentModel> ()
    .SetGroupName ("Wifi")
    .AddConstructor<LinearWifiTxCurrentModel> ()
    .AddAttribute ("Eta", "The efficiency of the power amplifier.",
                   DoubleValue (0.10),
                   MakeDoubleAccessor (&LinearWifiTxCurrentModel::m_eta),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("Voltage", "The supply voltage (in Volts).",
                   DoubleValue (3.0),
                   MakeDoubleAccessor (&LinearWifiTxCurrentModel::m_voltage),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("IdleCurrent", "The current in the IDLE state (in Ampere).",
                   DoubleValue (0.273333),
                   MakeDoubleAccessor (&LinearWifiTxCurrentModel::m_idleCurrent),
                   MakeDoubleChecker<double> ())
  ;
  return tid;
}

LinearWifiTxCurrentModel::LinearWifiTxCurrentModel ()
{
  NS_LOG_FUNCTION (this);
}

LinearWifiTxCurrentModel::~LinearWifiTxCurrentModel()
{
  NS_LOG_FUNCTION (this);
}

double
LinearWifiTxCurrentModel::CalcTxCurrent (double txPowerDbm) const
{
  NS_LOG_FUNCTION (this << txPowerDbm);
  return DbmToW (txPowerDbm) / (m_voltage * m_eta) + m_idleCurrent;
}

} // namespace ns3
