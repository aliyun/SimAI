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

#include "ns3/simulator.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/energy-source.h"
#include "simple-device-energy-model.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("SimpleDeviceEnergyModel");

NS_OBJECT_ENSURE_REGISTERED (SimpleDeviceEnergyModel);

TypeId
SimpleDeviceEnergyModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SimpleDeviceEnergyModel")
    .SetParent<DeviceEnergyModel> ()
    .SetGroupName ("Energy")
    .AddConstructor<SimpleDeviceEnergyModel> ()
    .AddTraceSource ("TotalEnergyConsumption",
                     "Total energy consumption of the radio device.",
                     MakeTraceSourceAccessor (&SimpleDeviceEnergyModel::m_totalEnergyConsumption),
                     "ns3::TracedValueCallback::Double")
  ;
  return tid;
}

SimpleDeviceEnergyModel::SimpleDeviceEnergyModel ()
{
  NS_LOG_FUNCTION (this);
  m_lastUpdateTime = Seconds (0.0);
  m_actualCurrentA = 0.0;
  m_source = 0;
}

SimpleDeviceEnergyModel::~SimpleDeviceEnergyModel ()
{
  NS_LOG_FUNCTION (this);
}

void
SimpleDeviceEnergyModel::SetEnergySource (Ptr<EnergySource> source)
{
  NS_LOG_FUNCTION (this << source);
  NS_ASSERT (source != NULL);
  m_source = source;
}

void
SimpleDeviceEnergyModel::SetNode (Ptr<Node> node)
{
  NS_LOG_FUNCTION (this << node);
  NS_ASSERT (node != NULL);
  m_node = node;
}

Ptr<Node>
SimpleDeviceEnergyModel::GetNode () const
{
  NS_LOG_FUNCTION (this);
  return m_node;
}

double
SimpleDeviceEnergyModel::GetTotalEnergyConsumption (void) const
{
  NS_LOG_FUNCTION (this);
  Time duration = Simulator::Now () - m_lastUpdateTime;

  double energyToDecrease = 0.0;
  double supplyVoltage = m_source->GetSupplyVoltage ();
  energyToDecrease = duration.GetSeconds () * m_actualCurrentA * supplyVoltage;

  m_source->UpdateEnergySource ();

  return m_totalEnergyConsumption + energyToDecrease;
}

void
SimpleDeviceEnergyModel::SetCurrentA (double current)
{
  NS_LOG_FUNCTION (this << current);
  Time duration = Simulator::Now () - m_lastUpdateTime;

  double energyToDecrease = 0.0;
  double supplyVoltage = m_source->GetSupplyVoltage ();
  energyToDecrease = duration.GetSeconds () * m_actualCurrentA * supplyVoltage;

  // update total energy consumption
  m_totalEnergyConsumption += energyToDecrease;
  // update last update time stamp
  m_lastUpdateTime = Simulator::Now ();
  // update the current drain
  m_actualCurrentA = current;
  // notify energy source
  m_source->UpdateEnergySource ();
}

void
SimpleDeviceEnergyModel::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  m_source = 0;
}

double
SimpleDeviceEnergyModel::DoGetCurrentA (void) const
{
  NS_LOG_FUNCTION (this);
  return m_actualCurrentA;
}

} // namespace ns3
