/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Universita' degli Studi di Napoli Federico II
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
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#include "ns3/log.h"
#include "wifi-protection-manager.h"
#include "wifi-mac.h"


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("WifiProtectionManager");

NS_OBJECT_ENSURE_REGISTERED (WifiProtectionManager);

TypeId
WifiProtectionManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::WifiProtectionManager")
    .SetParent<Object> ()
    .SetGroupName ("Wifi")
  ;
  return tid;
}

WifiProtectionManager::~WifiProtectionManager ()
{
  NS_LOG_FUNCTION_NOARGS ();
}

void
WifiProtectionManager::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  m_mac = 0;
  Object::DoDispose ();
}

void
WifiProtectionManager::SetWifiMac (Ptr<WifiMac> mac)
{
  NS_LOG_FUNCTION (this << mac);
  m_mac = mac;
}

} //namespace ns3
