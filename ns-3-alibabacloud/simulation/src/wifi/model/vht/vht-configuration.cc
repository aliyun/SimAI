/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018  Sébastien Deronne
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
 * Author: Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#include "ns3/log.h"
#include "vht-configuration.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("VhtConfiguration");

NS_OBJECT_ENSURE_REGISTERED (VhtConfiguration);

VhtConfiguration::VhtConfiguration ()
{
  NS_LOG_FUNCTION (this);
}

VhtConfiguration::~VhtConfiguration ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
VhtConfiguration::GetTypeId (void)
{
  static ns3::TypeId tid = ns3::TypeId ("ns3::VhtConfiguration")
    .SetParent<Object> ()
    .SetGroupName ("Wifi")
    .AddConstructor<VhtConfiguration> ()
    ;
    return tid;
}

} //namespace ns3
