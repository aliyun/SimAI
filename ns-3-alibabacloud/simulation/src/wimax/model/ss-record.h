/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008, 2009 INRIA, UDcast
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
 *         Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 */

#ifndef SS_RECORD_H
#define SS_RECORD_H

#include <stdint.h>
#include <ostream>
#include "ns3/mac48-address.h"
#include "wimax-connection.h"
#include "wimax-phy.h"
#include "wimax-net-device.h"
#include "ns3/ipv4-address.h"
#include <stdint.h>
#include "service-flow.h"

namespace ns3 {

class ServiceFlow;

/**
 * \ingroup wimax
 * \brief This class is used by the base station to store some information related to subscriber station in the cell.
 */
class SSRecord
{
public:
  SSRecord (void);
  /**
   * Constructor
   *
   * \param macAddress MAC address
   */
  SSRecord (Mac48Address macAddress);
  /**
   * Constructor
   *
   * \param macAddress MAC address
   * \param IPaddress IP address
   */
  SSRecord (Mac48Address macAddress, Ipv4Address IPaddress);
  ~SSRecord (void);

  /**
   * Set basic CID
   * \param basicCid the basic CID
   */
  void SetBasicCid (Cid basicCid);
  /**
   * Get basic CID
   * \returns the basic CID
   */
  Cid GetBasicCid (void) const;

  /**
   * Set primary CID
   * \param primaryCid priamry CID
   */
  void SetPrimaryCid (Cid primaryCid);
  /**
   * Get primary CID
   * \returns the CID
   */
  Cid GetPrimaryCid (void) const;

  /**
   * Set MAC address
   * \param macAddress the MAC address
   */
  void SetMacAddress (Mac48Address macAddress);
  /**
   * Get MAC address
   * \returns the MAC address
   */
  Mac48Address GetMacAddress (void) const;

  /**
   * Get ranging correction retries
   * \returns the ranging correction retries
   */
  uint8_t GetRangingCorrectionRetries (void) const;
  /// Reset ranging correction retries
  void ResetRangingCorrectionRetries (void);
  /// Increment ranging correction retries
  void IncrementRangingCorrectionRetries (void);
  /**
   * Get invited range retries
   * \returns the invited range retries
   */
  uint8_t GetInvitedRangRetries (void) const;
  /// Reset invited ranging retries
  void ResetInvitedRangingRetries (void);
  /// Increment invited ranging retries
  void IncrementInvitedRangingRetries (void);
  /**
   * Set modulation type
   * \param modulationType the modulation type
   */
  void SetModulationType (WimaxPhy::ModulationType modulationType);
  /**
   * Get modulation type
   * \returns the modulation type
   */
  WimaxPhy::ModulationType GetModulationType (void) const;

  /**
   * Set ranging status
   * \param rangingStatus the ranging status
   */
  void SetRangingStatus (WimaxNetDevice::RangingStatus rangingStatus);
  /**
   * Get ranging status
   * \returns the ranging status
   */
  WimaxNetDevice::RangingStatus GetRangingStatus (void) const;

  /// Enable poll for ranging function
  void EnablePollForRanging (void);
  /// Disable poll for ranging
  void DisablePollForRanging (void);
  /**
   * Get poll for ranging
   * \returns the poll for ranging flag
   */
  bool GetPollForRanging (void) const;

  /**
   * Check if service flows are allocated
   * \returns true if service flows are allocated
   */
  bool GetAreServiceFlowsAllocated (void) const;

  /**
   * Set poll ME bit
   * \param pollMeBit the poll me bit
   */
  void SetPollMeBit (bool pollMeBit);
  /**
   * Get poll ME bit
   * \returns the poll me bit
   */
  bool GetPollMeBit (void) const;

  /**
   * Add service flow
   * \param serviceFlow the service flow
   */
  void AddServiceFlow (ServiceFlow *serviceFlow);
  /**
   * Get service flows
   * \param schedulingType the scheduling type
   * \returns the service flow
   */
  std::vector<ServiceFlow*> GetServiceFlows (enum ServiceFlow::SchedulingType schedulingType) const;
  /**
   * Check if at least one flow has scheduling type SF_TYPE_UGS
   * \return true if at least one flow has scheduling type SF_TYPE_UGS
   */
  bool GetHasServiceFlowUgs (void) const;
  /**
   * Check if at least one flow has scheduling type SF_TYPE_RTPS
   * \return true if at least one flow has scheduling type SF_TYPE_RTPS
   */
  bool GetHasServiceFlowRtps (void) const;
  /**
   * Check if at least one flow has scheduling type SF_TYPE_NRTPS
   * \return true if at least one flow has scheduling type SF_TYPE_NRTPS
   */
  bool GetHasServiceFlowNrtps (void) const;
  /**
   * Check if at least one flow has scheduling type SF_TYPE_BE
   * \return true if at least one flow has scheduling type SF_TYPE_BE
   */
  bool GetHasServiceFlowBe (void) const;

  /**
   * Set SF transaction ID
   * \param sfTransactionId the SF transaction ID
   */
  void SetSfTransactionId (uint16_t sfTransactionId);
  /**
   * Get SF transaction ID
   * \returns the SF transaction ID
   */
  uint16_t GetSfTransactionId (void) const;

  /**
   * Set DSA response retries
   * \param dsaRspRetries the DSA response retries
   */
  void SetDsaRspRetries (uint8_t dsaRspRetries);
  /// Increment DAS response retries
  void IncrementDsaRspRetries (void);
  /**
   * Get DSA response retries
   * \returns the DSA response retries
   */
  uint8_t GetDsaRspRetries (void) const;

  /**
   * Set DSA response
   * \param dsaRsp the DSA response
   */ 
  void SetDsaRsp (DsaRsp dsaRsp);
  /**
   * Get DSA response
   * \returns the DSA response
   */
  DsaRsp GetDsaRsp (void) const;
  /**
   * Set is broadcast SS
   * \param broadcast_enable the broadcast enable flag
   */
  void SetIsBroadcastSS (bool broadcast_enable);
  /**
   * Get is broadcast SS
   * \returns the is broadcast SS flag
   */
  bool GetIsBroadcastSS (void);

  /**
   * Get IP address
   * \returns the IP address
   */
  Ipv4Address GetIPAddress (void);
  /**
   * Set IP address
   * \param IPaddress the IP address
   */
  void SetIPAddress (Ipv4Address IPaddress);
  /**
   * Set are service flows allocated
   * \param val the service flows allocated flag
   */
  void SetAreServiceFlowsAllocated (bool val);
private:
  /// Initialize
  void Initialize (void);

  Mac48Address m_macAddress; ///< MAC address
  Ipv4Address m_IPAddress; ///< IP address

  Cid m_basicCid; ///< basic CID
  Cid m_primaryCid; ///< primary CID

  uint8_t m_rangingCorrectionRetries; ///< ranging correction retries
  uint8_t m_invitedRangingRetries; ///< invited ranging retries

  WimaxPhy::ModulationType m_modulationType; ///< least robust burst profile (modulation type) for this SS
  WimaxNetDevice::RangingStatus m_rangingStatus; ///< ranging status
  bool m_pollForRanging; ///< poll for ranging
  bool m_areServiceFlowsAllocated; ///< are service floes allowed
  bool m_pollMeBit; ///< if PM (poll me) bit set for this SS
  bool m_broadcast; ///< broadcast?

  std::vector<ServiceFlow*> *m_serviceFlows; ///< service flows

  // fields for service flow creation
  uint16_t m_sfTransactionId; ///< SF transaction ID
  uint8_t m_dsaRspRetries; ///< DAS response retries
  DsaRsp m_dsaRsp; ///< DSA response
};

} // namespace ns3

#endif /* SS_RECORD_H */
