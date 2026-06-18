/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
 * Copyright (c) 2009 MIRKO BANCHI
 * Copyright (c) 2013 Dalian University of Technology
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 * Author: Mirko Banchi <mk.banchi@gmail.com>
 * Author: Junling Bu <linlinjavaer@gmail.com>
 */
#include "ns3/string.h"
#include "ns3/log.h"
#include <typeinfo>
#include "wave-mac-helper.h"
#include "wifi-80211p-helper.h"

namespace ns3 {

Wifi80211pHelper::Wifi80211pHelper ()
{
}

Wifi80211pHelper::~Wifi80211pHelper ()
{
}

Wifi80211pHelper
Wifi80211pHelper::Default (void)
{
  Wifi80211pHelper helper;
  helper.SetStandard (WIFI_STANDARD_80211p);
  helper.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                  "DataMode", StringValue ("OfdmRate6MbpsBW10MHz"),
                                  "ControlMode",StringValue ("OfdmRate6MbpsBW10MHz"),
                                  "NonUnicastMode", StringValue ("OfdmRate6MbpsBW10MHz"));
  return helper;
}

void
Wifi80211pHelper::SetStandard (enum WifiStandard standard)
{
  if (standard == WIFI_STANDARD_80211p)
    {
      WifiHelper::SetStandard (standard);
    }
  else
    {
      NS_FATAL_ERROR ("wrong standard selected!");
    }
}

void
Wifi80211pHelper::EnableLogComponents (void)
{
  WifiHelper::EnableLogComponents ();

  LogComponentEnable ("OcbWifiMac", LOG_LEVEL_ALL);
  LogComponentEnable ("VendorSpecificAction", LOG_LEVEL_ALL);
}

NetDeviceContainer
Wifi80211pHelper::Install (const WifiPhyHelper &phyHelper, const WifiMacHelper &macHelper, NodeContainer c) const
{
  [[maybe_unused]] QosWaveMacHelper const * qosMac = dynamic_cast <QosWaveMacHelper const *> (&macHelper);
  if (qosMac == 0)
    {
      [[maybe_unused]] NqosWaveMacHelper const * nqosMac = dynamic_cast <NqosWaveMacHelper const *> (&macHelper);
      if (nqosMac == 0)
        {
          NS_FATAL_ERROR ("the macHelper should be either QosWaveMacHelper or NqosWaveMacHelper"
                          ", or should be the subclass of QosWaveMacHelper or NqosWaveMacHelper");
        }
    }

  return WifiHelper::Install (phyHelper, macHelper, c);
}

} // namespace ns3
