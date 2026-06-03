/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008,2009 INRIA, UDcast
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
 * Authors: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 *          Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 *                               <amine.ismail@UDcast.com>
 */

#ifndef LINK_MANAGER_H
#define LINK_MANAGER_H

#include <stdint.h>
#include "wimax-net-device.h"
#include "bs-net-device.h"
#include "ns3/event-id.h"
#include "cid.h"
#include "mac-messages.h"

namespace ns3 {

/**
 * \ingroup wimax
 * 
 * BaseStation Link Manager
 */
class BSLinkManager : public Object
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
   * \param bs base station device
   */
  BSLinkManager (Ptr<BaseStationNetDevice> bs);
  ~BSLinkManager (void);
  /**
   * \return the ranging opportunities to allocate
   */
  uint8_t CalculateRangingOppsToAllocate (void);
  /**
   * \return an available downlink channel
   */
  uint64_t SelectDlChannel (void);
  /**
   * \brief process a ranging request message
   * \param cid the connection identifier in which the ranging message was received
   * \param rngreq the ranging request message
   */

  void ProcessRangingRequest (Cid cid, RngReq rngreq);
  /**
   * \brief Verifies at the end of an invited ranging interval if SS sent ranging message in it or not
   * \param cid the connection identifier in which the ranging message was received
   * \param uiuc the ranging
   */
  void VerifyInvitedRanging (Cid cid, uint8_t uiuc);
private:
  /// copy constructor (disabled)
  BSLinkManager (const BSLinkManager &);
  /**
   * assignment operator (disabled)
   * \returns the BS link manager
   */
  BSLinkManager & operator= (const BSLinkManager &);

  /**
   * Perform ranging function
   * \param cid the CID
   * \param rngreq the ranging request
   */
  void PerformRanging (Cid cid, RngReq rngreq);
  /**
   * Perform initial ranging function
   * \param cid the CID
   * \param rngreq the ranging request
   * \param rngrsp the ranging response
   */
  void PerformInitialRanging (Cid cid, RngReq *rngreq, RngRsp *rngrsp);
  /**
   * Perform inivted ranging function
   * \param cid the CID
   * \param rngrsp the ranging response
   */
  void PerformInvitedRanging (Cid cid, RngRsp *rngrsp);

  /**
   * Set parameters to adjust function
   * \param rngrsp the ranging response
   */
  void SetParametersToAdjust (RngRsp *rngrsp);
  /**
   * Abort ranging function
   * \param cid the CID
   * \param rngrsp the ranging response
   * \param ssRecord the SS record
   * \param isNewSS true if a new SS
   */
  void AbortRanging (Cid cid, RngRsp *rngrsp, SSRecord *ssRecord, bool isNewSS);
  /**
   * Accept ranging function
   * \param cid the CID
   * \param rngrsp the ranging response
   * \param ssRecord the SS record
   */
  void AcceptRanging (Cid cid, RngRsp *rngrsp, SSRecord *ssRecord);
  /**
   * Continue ranging function
   * \param cid the CID
   * \param rngrsp the ranging response
   * \param ssRecord the SS record
   */
  void ContinueRanging (Cid cid, RngRsp *rngrsp, SSRecord *ssRecord);
  /**
   * Schedule range response message function
   * \param cid the CID
   * \param rngrsp the ranging response
   */
  void ScheduleRngRspMessage (Cid cid, RngRsp *rngrsp);
  /**
   * Deallocate CIDs function
   * \param cid the CID
   */
  void DeallocateCids (Cid cid);

  /**
   * Change DL channel function
   * \returns true if the DL channel changed
   */
  bool ChangeDlChannel (void);
  /**
   * Get new DL channel function
   * \returns the new DL channel
   */
  uint32_t GetNewDlChannel (void);
  /**
   * Get signal quality function
   * \returns the signal quality
   */
  uint8_t GetSignalQuality (void);
  /**
   * Is ranging acceptable function
   * \returns true if ranging is acceptable
   */
  bool IsRangingAcceptable (void);

  Ptr<BaseStationNetDevice> m_bs; ///< base station

  // ranging parameters
  uint32_t m_signalQuality; ///< signal quality
  uint8_t m_signalQualityThreshold; ///< signal quality threshold
  int tries; ///< variable to test multiple RNG-REQ by the SS for example undecodable at BS or lost RNG-RSP
};

} // namespace ns3

#endif /* LINK_MANAGER_H */
