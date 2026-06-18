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
#include "ns3/boolean.h"
#include "wave-mac-helper.h"

namespace ns3 {

NqosWaveMacHelper::NqosWaveMacHelper (void)
{

}
NqosWaveMacHelper::~NqosWaveMacHelper (void)
{

}
NqosWaveMacHelper
NqosWaveMacHelper::Default (void)
{
  NqosWaveMacHelper helper;
  // We're making non QoS-enabled Wi-Fi MACs here, so we set the
  // necessary attribute. I've carefully positioned this here so that
  // someone who knows what they're doing can override with explicit
  // attributes.
  helper.SetType ("ns3::OcbWifiMac", "QosSupported", BooleanValue (false));
  return helper;
}
void
NqosWaveMacHelper::SetType (std::string type,
                            std::string n0, const AttributeValue &v0,
                            std::string n1, const AttributeValue &v1,
                            std::string n2, const AttributeValue &v2,
                            std::string n3, const AttributeValue &v3,
                            std::string n4, const AttributeValue &v4,
                            std::string n5, const AttributeValue &v5,
                            std::string n6, const AttributeValue &v6,
                            std::string n7, const AttributeValue &v7,
                            std::string n8, const AttributeValue &v8,
                            std::string n9, const AttributeValue &v9,
                            std::string n10, const AttributeValue &v10)
{
  if (type.compare ("ns3::OcbWifiMac") != 0)
    {
      NS_FATAL_ERROR ("QosWaveMacHelper shall set OcbWifiMac");
    }
  WifiMacHelper::SetType ("ns3::OcbWifiMac",
                              n0, v0,
                              n1, v1,
                              n2, v2,
                              n3, v3,
                              n4, v4,
                              n5, v5,
                              n6, v6,
                              n7, v7,
                              n8, v8,
                              n9, v9,
                              n10, v10);
}

/**********  QosWifi80211pMacHelper *********/
QosWaveMacHelper::QosWaveMacHelper ()
{
}
QosWaveMacHelper::~QosWaveMacHelper ()
{
}
QosWaveMacHelper
QosWaveMacHelper::Default (void)
{
  QosWaveMacHelper helper;

  // We're making QoS-enabled Wi-Fi MACs here, so we set the necessary
  // attribute. I've carefully positioned this here so that someone
  // who knows what they're doing can override with explicit
  // attributes.
  helper.SetType ("ns3::OcbWifiMac", "QosSupported", BooleanValue (true));

  return helper;
}
void
QosWaveMacHelper::SetType (std::string type,
                           std::string n0, const AttributeValue &v0,
                           std::string n1, const AttributeValue &v1,
                           std::string n2, const AttributeValue &v2,
                           std::string n3, const AttributeValue &v3,
                           std::string n4, const AttributeValue &v4,
                           std::string n5, const AttributeValue &v5,
                           std::string n6, const AttributeValue &v6,
                           std::string n7, const AttributeValue &v7,
                           std::string n8, const AttributeValue &v8,
                           std::string n9, const AttributeValue &v9,
                           std::string n10, const AttributeValue &v10)
{
  if (type.compare ("ns3::OcbWifiMac") != 0)
    {
      NS_FATAL_ERROR ("QosWaveMacHelper shall set OcbWifiMac");
    }
  WifiMacHelper::SetType ("ns3::OcbWifiMac",
                             n0, v0,
                             n1, v1,
                             n2, v2,
                             n3, v3,
                             n4, v4,
                             n5, v5,
                             n6, v6,
                             n7, v7,
                             n8, v8,
                             n9, v9,
                             n10, v10);
}

} // namespace ns3
