/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016
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

#include "ns3/wifi-net-device.h"
#include "wifi-mac-helper.h"
#include "ns3/frame-exchange-manager.h"
#include "ns3/wifi-protection-manager.h"
#include "ns3/wifi-ack-manager.h"
#include "ns3/multi-user-scheduler.h"
#include "ns3/boolean.h"

namespace ns3 {

WifiMacHelper::WifiMacHelper ()
{
  //By default, we create an AdHoc MAC layer (without QoS).
  SetType ("ns3::AdhocWifiMac");

  m_protectionManager.SetTypeId ("ns3::WifiDefaultProtectionManager");
  m_ackManager.SetTypeId ("ns3::WifiDefaultAckManager");
}

WifiMacHelper::~WifiMacHelper ()
{
}

Ptr<WifiMac>
WifiMacHelper::Create (Ptr<WifiNetDevice> device, WifiStandard standard) const
{
  NS_ABORT_MSG_IF (standard == WIFI_STANDARD_UNSPECIFIED, "No standard specified!");

  // this is a const method, but we need to force the correct QoS setting
  ObjectFactory macObjectFactory = m_mac;
  if (standard >= WIFI_STANDARD_80211n)
    {
      macObjectFactory.Set ("QosSupported", BooleanValue (true));
    }

  Ptr<WifiMac> mac = macObjectFactory.Create<WifiMac> ();
  mac->SetDevice (device);
  mac->SetAddress (Mac48Address::Allocate ());
  mac->ConfigureStandard (standard);

  Ptr<FrameExchangeManager> fem = mac->GetFrameExchangeManager ();

  if (fem != nullptr)
    {
      Ptr<WifiProtectionManager> protectionManager = m_protectionManager.Create<WifiProtectionManager> ();
      protectionManager->SetWifiMac (mac);
      fem->SetProtectionManager (protectionManager);

      Ptr<WifiAckManager> ackManager = m_ackManager.Create<WifiAckManager> ();
      ackManager->SetWifiMac (mac);
      fem->SetAckManager (ackManager);

      // create and install the Multi User Scheduler if this is an HE AP
      Ptr<ApWifiMac> apMac = DynamicCast<ApWifiMac> (mac);
      if (apMac != nullptr && standard >= WIFI_STANDARD_80211ax
          && m_muScheduler.IsTypeIdSet ())
        {
          Ptr<MultiUserScheduler> muScheduler = m_muScheduler.Create<MultiUserScheduler> ();
          apMac->AggregateObject (muScheduler);
        }
    }
  return mac;
}

} //namespace ns3
