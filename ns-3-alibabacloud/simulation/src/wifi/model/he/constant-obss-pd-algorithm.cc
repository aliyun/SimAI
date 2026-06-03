/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 University of Washington
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
#include "ns3/node.h"
#include "ns3/config.h"
#include "ns3/double.h"
#include "constant-obss-pd-algorithm.h"
#include "ns3/sta-wifi-mac.h"
#include "ns3/wifi-utils.h"
#include "ns3/wifi-phy.h"
#include "he-phy.h"
#include "ns3/wifi-net-device.h"
#include "he-configuration.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ConstantObssPdAlgorithm");
NS_OBJECT_ENSURE_REGISTERED (ConstantObssPdAlgorithm);

ConstantObssPdAlgorithm::ConstantObssPdAlgorithm ()
  : ObssPdAlgorithm ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
ConstantObssPdAlgorithm::GetTypeId (void)
{
  static ns3::TypeId tid = ns3::TypeId ("ns3::ConstantObssPdAlgorithm")
    .SetParent<ObssPdAlgorithm> ()
    .SetGroupName ("Wifi")
    .AddConstructor<ConstantObssPdAlgorithm> ()
  ;
  return tid;
}

void
ConstantObssPdAlgorithm::ConnectWifiNetDevice (const Ptr<WifiNetDevice> device)
{
  Ptr<WifiPhy> phy = device->GetPhy ();
  auto hePhy = DynamicCast<HePhy> (phy->GetPhyEntity (WIFI_MOD_CLASS_HE));
  NS_ASSERT (hePhy);
  hePhy->SetEndOfHeSigACallback (MakeCallback (&ConstantObssPdAlgorithm::ReceiveHeSigA, this));
  ObssPdAlgorithm::ConnectWifiNetDevice (device);
}

void
ConstantObssPdAlgorithm::ReceiveHeSigA (HeSigAParameters params)
{
  NS_LOG_FUNCTION (this << +params.bssColor << WToDbm (params.rssiW));

  Ptr<StaWifiMac> mac = m_device->GetMac ()->GetObject<StaWifiMac>();
  if (mac && !mac->IsAssociated ())
    {
      NS_LOG_DEBUG ("This is not an associated STA: skip OBSS PD algorithm");
      return;
    }

  Ptr<HeConfiguration> heConfiguration = m_device->GetHeConfiguration ();
  NS_ASSERT (heConfiguration);
  uint8_t bssColor = heConfiguration->GetBssColor ();

  if (bssColor == 0)
    {
      NS_LOG_DEBUG ("BSS color is 0");
      return;
    }
  if (params.bssColor == 0)
    {
      NS_LOG_DEBUG ("Received BSS color is 0");
      return;
    }
  //TODO: SRP_AND_NON-SRG_OBSS-PD_PROHIBITED=1 => OBSS_PD SR is not allowed

  bool isObss = (bssColor != params.bssColor);
  if (isObss)
    {
      if (WToDbm (params.rssiW) < m_obssPdLevel)
        {
          NS_LOG_DEBUG ("Frame is OBSS and RSSI " << WToDbm(params.rssiW) << " is below OBSS-PD level of " << m_obssPdLevel << "; reset PHY to IDLE");
          ResetPhy (params);
        }
      else
        {
          NS_LOG_DEBUG ("Frame is OBSS and RSSI is above OBSS-PD level");
        }
    }
}

} //namespace ns3
