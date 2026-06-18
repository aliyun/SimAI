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

#ifndef WIMAX_CONNECTION_H
#define WIMAX_CONNECTION_H

#include <stdint.h>
#include <ostream>
#include "cid.h"
#include "wimax-mac-header.h"
#include "wimax-mac-queue.h"
#include "ns3/object.h"
#include "service-flow.h"

namespace ns3 {

class ServiceFlow;
class Cid;

/**
 * \ingroup wimax
 * Class to represent WiMAX connections
 */
class WimaxConnection : public Object
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
   * \param cid connection ID
   * \param type CID type
   */
  WimaxConnection (Cid cid, enum Cid::Type type);
  ~WimaxConnection (void);

  /**
   * Get CID function
   * \returns the CID
   */
  Cid GetCid (void) const;

  /**
   * Get type function
   * \returns the type
   */
  enum Cid::Type GetType (void) const;
  /**
   * \return the queue of the connection
   */
  Ptr<WimaxMacQueue> GetQueue (void) const;
  /**
   * \brief set the service flow associated to this connection
   * \param serviceFlow The service flow to be associated to this connection
   */
  void SetServiceFlow (ServiceFlow *serviceFlow);
  /**
   * \return the service flow associated to this connection
   */
  ServiceFlow* GetServiceFlow (void) const;

  // wrapper functions
  /**
   * \return the scheduling type of this connection
   */
  uint8_t GetSchedulingType (void) const;
  /**
   * \brief enqueue a packet in the queue of the connection
   * \param packet the packet to be enqueued
   * \param hdrType the header type of the packet
   * \param hdr the header of the packet
   * \return true if successful
   */
  bool Enqueue (Ptr<Packet> packet, const MacHeaderType &hdrType, const GenericMacHeader &hdr);
  /**
   * \brief dequeue a packet from the queue of the connection
   * \param packetType the type of the packet to dequeue
   * \return packet dequeued
   */
  Ptr<Packet> Dequeue (MacHeaderType::HeaderType packetType = MacHeaderType::HEADER_TYPE_GENERIC);
  /**
   * \brief dequeue a packet from the queue of the connection
   * Dequeue the first packet in the queue if its size is lower than 
   * availableByte, the first availableByte of the first packet otherwise
   *
   * \param packetType the type of the packet to dequeue
   * \param availableByte the number of available bytes
   * \return packet dequeued
   */
  Ptr<Packet> Dequeue (MacHeaderType::HeaderType packetType, uint32_t availableByte);
  /**
   * \return true if the connection has at least one packet in its queue, false otherwise
   */
  bool HasPackets (void) const;
  /**
   * \return true if the connection has at least one packet of type packetType in its queue, false otherwise
   * \param packetType type of packet to check in the queue
   * \return true if packets available
   */
  bool HasPackets (MacHeaderType::HeaderType packetType) const;

  /**
   * Get type string
   * \returns the type string
   */
  std::string GetTypeStr (void) const;

  /// Definition of Fragments Queue data type
  typedef std::list<Ptr<const Packet> > FragmentsQueue;
  /**
   * \brief get a queue of received fragments
   * \returns the fragments queue
   */
  const FragmentsQueue GetFragmentsQueue (void) const;
  /**
   * \brief enqueue a received packet (that is a fragment) into the fragments queue
   * \param fragment received fragment
   */
  void FragmentEnqueue (Ptr<const Packet> fragment);
  /**
   * \brief delete all enqueued fragments
   */
  void ClearFragmentsQueue (void);

private:
  virtual void DoDispose (void);

  Cid m_cid; ///< CID
  enum Cid::Type m_cidType; ///< CID type
  Ptr<WimaxMacQueue> m_queue; ///< queue
  ServiceFlow *m_serviceFlow; ///< service flow

  // FragmentsQueue stores all received fragments
  FragmentsQueue m_fragmentsQueue; ///< fragments queue
};

} // namespace ns3

#endif /* WIMAX_CONNECTION_H */
