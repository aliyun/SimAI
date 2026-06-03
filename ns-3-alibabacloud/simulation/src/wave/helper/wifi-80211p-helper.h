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
#ifndef WIFI_802_11P_HELPER_H
#define WIFI_802_11P_HELPER_H
#include "ns3/wifi-helper.h"
#include "ns3/deprecated.h"

namespace ns3 {

/**
 * \ingroup wave
 * \brief helps to create wifi 802.11p objects of
 * WifiNetDevice class
 *
 * This class can help to create a large set of similar
 * wifi 802.11p objects and to configure a large set of
 * their attributes during creation.
 */
class Wifi80211pHelper : public WifiHelper
{
public:
  Wifi80211pHelper ();
  virtual ~Wifi80211pHelper ();

  /**
   * \returns a new Wifi80211pHelper in a default state
   *
   * The default state is defined as being an OcbWifiMac MAC
   * layer with constant rate OfdmRate6MbpsBW10MHz
   * and both objects using their default attribute values.
   */
  static Wifi80211pHelper Default (void);

  /**
   * \param standard the phy standard to configure during installation
   *
   * Users can only configure 802.11p with 10MHz or 5 MHz channel bandwidth.
   * The default 802.11p standard uses 10MHz.
   */
  virtual void SetStandard (enum WifiStandard standard);

  /**
   * \param phy the PHY helper to create PHY objects
   * \param macHelper the MAC helper to create MAC objects
   * \param c the set of nodes on which a wifi device must be created
   * \returns a device container which contains all the devices created by this method.
   */
  virtual NetDeviceContainer Install (const WifiPhyHelper &phy, const WifiMacHelper &macHelper,NodeContainer c) const;

  /**
   * Helper to enable all WifiNetDevice log components with one statement
   */
  static void EnableLogComponents (void);
};

}

#endif /* WIFI_802_11P_HELPER_H */
