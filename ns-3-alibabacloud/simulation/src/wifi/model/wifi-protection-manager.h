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

#ifndef WIFI_PROTECTION_MANAGER_H
#define WIFI_PROTECTION_MANAGER_H

#include "wifi-protection.h"
#include <memory>
#include "ns3/object.h"


namespace ns3 {

class WifiTxParameters;
class WifiMacQueueItem;
class WifiMac;

/**
 * \ingroup wifi
 *
 * WifiProtectionManager is an abstract base class. Each subclass defines a logic
 * to select the protection method for a given frame.
 */
class WifiProtectionManager : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual ~WifiProtectionManager ();

  /**
   * Set the MAC which is using this Protection Manager
   *
   * \param mac a pointer to the MAC
   */
  void SetWifiMac (Ptr<WifiMac> mac);

  /**
   * Determine the protection method to use if the given MPDU is added to the current
   * frame. Return a null pointer if the protection method is unchanged or the new
   * protection method otherwise.
   *
   * \param mpdu the MPDU to be added to the current frame
   * \param txParams the current TX parameters for the current frame
   * \return a null pointer if the protection method is unchanged or the new
   *         protection method otherwise
   */
  virtual std::unique_ptr<WifiProtection> TryAddMpdu (Ptr<const WifiMacQueueItem> mpdu,
                                                      const WifiTxParameters& txParams) = 0;

  /**
   * Determine the protection method to use if the given MSDU is aggregated to the
   * current frame. Return a null pointer if the protection method is unchanged or
   * the new protection method otherwise.
   *
   * \param msdu the MSDU to be aggregated to the current frame
   * \param txParams the current TX parameters for the current frame
   * \return a null pointer if the protection method is unchanged or the new
   *         protection method otherwise
   */
  virtual std::unique_ptr<WifiProtection> TryAggregateMsdu (Ptr<const WifiMacQueueItem> msdu,
                                                            const WifiTxParameters& txParams) = 0;

protected:
  virtual void DoDispose (void);

  Ptr<WifiMac> m_mac; //!< MAC which is using this Protection Manager
};



} //namespace ns3

#endif /* WIFI_PROTECTION_MANAGER_H */
