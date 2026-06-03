/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008 INRIA
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
 * Author: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 */

#ifndef BURST_PROFILE_MANAGER_H
#define BURST_PROFILE_MANAGER_H

#include <stdint.h>
#include "cid.h"
#include "wimax-phy.h"
#include "wimax-net-device.h"

namespace ns3 {

class SSRecord;
class RngReq;

/**
 * \ingroup wimax
 * 
 * Profile manager for burst communications
 */
class BurstProfileManager : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  /**
   * Constructor
   *
   * \param device WIMAX device
   */
  BurstProfileManager (Ptr<WimaxNetDevice> device);
  ~BurstProfileManager (void);
  void DoDispose (void);
  /**
   * \returns the number of available burst profile
   */
  uint16_t GetNrBurstProfilesToDefine (void);

  /**
   * \brief returns the modulation type of a given iuc
   * \param direction should be uplink or downlink
   * \param iuc the iuc
   * \returns the modulation type of the selected iuc
   */
  WimaxPhy::ModulationType GetModulationType (uint8_t iuc,
                                              WimaxNetDevice::Direction direction) const;

  /**
   * \brief returns the burst profile
   * \param modulationType
   * \param direction should be uplink or downlink
   * \returns the modulation type of the selected iuc
   */
  uint8_t GetBurstProfile (WimaxPhy::ModulationType modulationType,
                           WimaxNetDevice::Direction direction) const;

  /**
   * \brief Get burst profile for SS
   * \param modulationType
   * \param ssRecord
   * \param rngreq
   * \returns the burst profile for SS
   */
  uint8_t GetBurstProfileForSS (const SSRecord *ssRecord, const RngReq *rngreq,
                                WimaxPhy::ModulationType &modulationType);
  /**
   * \brief Get module ation type for SS
   * \param ssRecord
   * \param rngreq
   * \returns the burst profile for SS
   */
  WimaxPhy::ModulationType GetModulationTypeForSS (const SSRecord *ssRecord,
                                                   const RngReq *rngreq);
  /**
   * \brief Get burst profile to request
   * \returns the burst profile for SS
   */
  uint8_t GetBurstProfileToRequest (void);
private:
  /// Type conversion operator
  BurstProfileManager (const BurstProfileManager &);
  /**
   * Assignment operator
   * \returns burst profile manager
   */
  BurstProfileManager& operator= (const BurstProfileManager &);

  Ptr<WimaxNetDevice> m_device; ///< the device
};

} // namespace ns3

#endif /* BURST_PROFILE_MANAGER_H */
