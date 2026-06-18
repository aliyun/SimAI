/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Wireless Communications and Networking Group (WCNG),
 * University of Rochester, Rochester, NY, USA.
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
 * Author: Cristiano Tapparello <cristiano.tapparello@rochester.edu>
 */

#include "energy-harvester.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("EnergyHarvester");
    
NS_OBJECT_ENSURE_REGISTERED (EnergyHarvester);

TypeId
EnergyHarvester::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::EnergyHarvester")
    .SetParent<Object> ()
    .SetGroupName ("Energy")
  ;
  return tid;
}

EnergyHarvester::EnergyHarvester ()
{
  NS_LOG_FUNCTION (this);
}

EnergyHarvester::~EnergyHarvester ()
{
  NS_LOG_FUNCTION (this);
}

void
EnergyHarvester::SetNode (Ptr<Node> node)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (node != 0);
  m_node = node;
}

Ptr<Node>
EnergyHarvester::GetNode (void) const
{
  NS_LOG_FUNCTION (this);
  return m_node;
}

void
EnergyHarvester::SetEnergySource (Ptr<EnergySource> source)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (source != 0);
  m_energySource = source;
}

Ptr<EnergySource>
EnergyHarvester::GetEnergySource (void) const
{
  NS_LOG_FUNCTION (this);
  return m_energySource;
}


double
EnergyHarvester::GetPower (void) const
{
  NS_LOG_FUNCTION (this);
  return DoGetPower ();
}

/*
 * Private function starts here.
 */

void
EnergyHarvester::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
}

double
EnergyHarvester::DoGetPower (void) const
{
  NS_LOG_FUNCTION (this);
  return 0.0;
}

}
