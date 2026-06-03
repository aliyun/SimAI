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

#ifndef WIFI_DEFAULT_PROTECTION_MANAGER_H
#define WIFI_DEFAULT_PROTECTION_MANAGER_H

#include "wifi-protection-manager.h"


namespace ns3 {

class WifiTxParameters;
class WifiMacQueueItem;
class WifiMacHeader;

/**
 * \ingroup wifi
 *
 * WifiDefaultProtectionManager is the default protection manager, which selects
 * the protection method for a frame based on its size.
 */
class WifiDefaultProtectionManager : public WifiProtectionManager
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  WifiDefaultProtectionManager ();
  virtual ~WifiDefaultProtectionManager ();

  virtual std::unique_ptr<WifiProtection> TryAddMpdu (Ptr<const WifiMacQueueItem> mpdu,
                                                      const WifiTxParameters& txParams) override;
  virtual std::unique_ptr<WifiProtection> TryAggregateMsdu (Ptr<const WifiMacQueueItem> msdu,
                                                            const WifiTxParameters& txParams) override;

protected:
  /**
   * Select the protection method for a single PSDU.
   *
   * \param hdr the MAC header of the PSDU
   * \param size the size in bytes of the PSDU
   * \param txVector the TxVector used to transmit the PSDU
   * \return the selected protection method
   */
  virtual std::unique_ptr<WifiProtection> GetPsduProtection (const WifiMacHeader& hdr, uint32_t size,
                                                             const WifiTxVector& txVector) const;
};

} //namespace ns3

#endif /* WIFI_DEFAULT_PROTECTION_MANAGER_H */
