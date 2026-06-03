/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006, 2009 INRIA
 * Copyright (c) 2009 MIRKO BANCHI
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
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *          Mirko Banchi <mk.banchi@gmail.com>
 */

#ifndef ADHOC_WIFI_MAC_H
#define ADHOC_WIFI_MAC_H

#include "wifi-mac.h"

namespace ns3 {

/**
 * \ingroup wifi
 *
 * \brief Wifi MAC high model for an ad-hoc Wifi MAC
 */
class AdhocWifiMac : public WifiMac
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  AdhocWifiMac ();
  virtual ~AdhocWifiMac ();

  void SetAddress (Mac48Address address) override;
  void SetLinkUpCallback (Callback<void> linkUp) override;
  void Enqueue (Ptr<Packet> packet, Mac48Address to) override;
  bool CanForwardPacketsTo (Mac48Address to) const override;

private:
  void Receive (Ptr<WifiMacQueueItem> mpdu) override;
};

} //namespace ns3

#endif /* ADHOC_WIFI_MAC_H */
