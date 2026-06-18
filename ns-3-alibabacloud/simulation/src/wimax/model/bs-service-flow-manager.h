/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 INRIA, UDcast
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
 *         Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 */

#ifndef BS_SERVICE_FLOW_MANAGER_H
#define BS_SERVICE_FLOW_MANAGER_H

#include <stdint.h>
#include "ns3/event-id.h"
#include "mac-messages.h"
#include "ns3/buffer.h"
#include "service-flow-manager.h"
#include "bs-net-device.h"

namespace ns3 {

class Packet;
class ServiceFlow;
class WimaxNetDevice;
class SSRecord;
class WimaxConnection;
class BaseStationNetDevice;

/**
 * \ingroup wimax
 * \brief BsServiceFlowManager
 */
class BsServiceFlowManager : public ServiceFlowManager
{
public:
  /// Confirmation code enumeration
  enum ConfirmationCode // as per Table 384 (not all codes implemented)
  {
    CONFIRMATION_CODE_SUCCESS, CONFIRMATION_CODE_REJECT
  };

  /**
   * Constructor
   *
   * \param device base station device
   */
  BsServiceFlowManager (Ptr<BaseStationNetDevice> device);
  ~BsServiceFlowManager (void);
  void DoDispose (void);
  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);
  
  /**
   * \brief Add a new service flow
   * \param serviceFlow the service flow to add
   */
  void AddServiceFlow (ServiceFlow *serviceFlow);
  /**
   * \param sfid the service flow identifier
   * \return the service flow which has as identifier sfid
   */
  ServiceFlow* GetServiceFlow (uint32_t sfid) const;
  /**
   * \param cid the connection identifier
   * \return the service flow which has as connection identifier cid
   */
  ServiceFlow* GetServiceFlow (Cid cid) const;
  /**
   * \param schedulingType the scheduling type
   * \return the list of service flows configured with schedulingType as a QoS class
   */
  std::vector<ServiceFlow*> GetServiceFlows (ServiceFlow::SchedulingType schedulingType) const;
  /**
   * \brief set the maximum Dynamic ServiceFlow Add (DSA) retries
   * \param maxDsaRspRetries the maximum DSA response retries
   */
  void SetMaxDsaRspRetries (uint8_t maxDsaRspRetries);

  /**
   * \return the DSA ack timeout event
   */
  EventId GetDsaAckTimeoutEvent (void) const;
  /**
   * \brief allocate service flows
   * \param dsaReq the DSA request
   * \param cid the connection identifier
   */
  void AllocateServiceFlows (const DsaReq &dsaReq, Cid cid);
  /**
   * \brief add a multicast service flow
   * \param sf the service flow
   * \param modulation the wimax phy modulation type
   */
  void AddMulticastServiceFlow (ServiceFlow sf, enum WimaxPhy::ModulationType modulation);
  /**
   * \brief process a DSA-ACK message
   * \param dsaAck the message to process
   * \param cid the identifier of the connection on which the message was received
   */
  void ProcessDsaAck (const DsaAck &dsaAck, Cid cid);

  /**
   * \brief process a DSA-Req message
   * \param dsaReq the message to process
   * \param cid the identifier of the connection on which the message was received
   * \return a pointer to the service flow 
   */
  ServiceFlow* ProcessDsaReq (const DsaReq &dsaReq, Cid cid);

private:
  /**
   * Create DSA response function
   * \param serviceFlow service flow
   * \param transactionId transaction ID
   * \return the DSA response
   */
  DsaRsp CreateDsaRsp (const ServiceFlow *serviceFlow, uint16_t transactionId);
  /**
   * \return the maximum DSA response retries
   */
  uint8_t GetMaxDsaRspRetries (void) const;
  /**
   * Create DSA response function
   * \param serviceFlow service flow
   * \param cid the identifier of the connection on which the message was received
   */
  void ScheduleDsaRsp (ServiceFlow *serviceFlow, Cid cid);
  Ptr<WimaxNetDevice> m_device; ///< the device
  uint32_t m_sfidIndex; ///< SFID index
  uint8_t m_maxDsaRspRetries; ///< maximum number of DSA response retries
  EventId m_dsaAckTimeoutEvent; ///< DSA ack timeout event
  Cid m_inuseScheduleDsaRspCid; ///< in use schedule DSA response CID
};

} // namespace ns3

#endif /* BS_SERVICE_FLOW_MANAGER_H */
