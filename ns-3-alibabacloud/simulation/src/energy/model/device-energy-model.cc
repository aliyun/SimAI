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

#include "device-energy-model.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("DeviceEnergyModel");

NS_OBJECT_ENSURE_REGISTERED (DeviceEnergyModel);

TypeId
DeviceEnergyModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::DeviceEnergyModel")
    .SetParent<Object> ()
    .SetGroupName ("Energy")
  ;
  return tid;
}

DeviceEnergyModel::DeviceEnergyModel ()
{
  NS_LOG_FUNCTION (this);
}

DeviceEnergyModel::~DeviceEnergyModel ()
{
  NS_LOG_FUNCTION (this);
}

double
DeviceEnergyModel::GetCurrentA (void) const
{
  NS_LOG_FUNCTION (this);
  return DoGetCurrentA ();
}

/*
 * Private function starts here.
 */

double
DeviceEnergyModel::DoGetCurrentA (void) const
{
  NS_LOG_FUNCTION (this);
  return 0.0;
}

} // namespace ns3
