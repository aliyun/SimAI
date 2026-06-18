/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008 INRIA
 *               2009 TELEMATICS LAB, Politecnico di Bari
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
 * Author: Giuseppe Piro <g.piro@poliba.it>
 */

/* BS outbound scheduler as per in Section 6.3.5.1 */

#ifndef BS_SCHEDULER_RTPS_H
#define BS_SCHEDULER_RTPS_H

#include <list>
#include "ns3/packet.h"
#include "wimax-phy.h"
#include "ns3/packet-burst.h"
#include "dl-mac-messages.h"
#include "bs-scheduler.h"

namespace ns3 {

class BaseStationNetDevice;
class GenericMacHeader;
class WimaxConnection;
class Cid;

/**
 * \ingroup wimax
 * \brief This class implements a simple downlink scheduler for rtPS flows.
 *
 * The DL-rtPS-Scheduler assigns the available bandwidth the service flows
 * in the following order: IR Connections, Broadcast Connections, Basic and
 * Primary Connections, UGS Connections, rtPS Connections, nrtPS Connections,
 * BE Connections.
 * The IR, Broadcast and Basic/Primary Connections must transmit
 * all packet in the queue. The UGS flows can transmit the quota of date equal to
 * the "Grant Size" (that is one of service flows QoS parameters) according to their
 * requested MaxLatency (another QoS parameter).
 * All rtPS flows are able to transmit all packet in the queue according to the available
 * bandwidth. The bandwidth saturation control has been implemented to redistribute
 *the effective available bandwidth to all rtPS flows that have at least one packet
 * to transmit. The remaining bandwidth is allocated to rntPS and BE Connections.
 */
class BSSchedulerRtps : public BSScheduler
{
public:
  BSSchedulerRtps ();
  /**
   * Constructor
   *
   * \param bs base station device
   */
  BSSchedulerRtps (Ptr<BaseStationNetDevice> bs);
  ~BSSchedulerRtps (void);

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * \brief This function returns all the downlink bursts scheduled for the next
   * downlink sub-frame
   * \returns  all the downlink bursts scheduled for the next downlink sub-frame
   */
  std::list<std::pair<OfdmDlMapIe*, Ptr<PacketBurst> > >*
  GetDownlinkBursts (void) const;

  /**
   * \brief This function adds a downlink burst to the list of downlink bursts
   * scheduled for the next downlink sub-frame
   * \param connection a pointer to connection in which the burst will be sent
   * \param diuc downlink iuc
   * \param modulationType the modulation type of the burst
   * \param burst the downlink burst to add to the downlink sub frame
   */
  void AddDownlinkBurst (Ptr<const WimaxConnection> connection, uint8_t diuc,
                         WimaxPhy::ModulationType modulationType, Ptr<PacketBurst> burst);
  /**
   * \brief Schedule function.
   */
  void Schedule (void);
  /**
   * \brief Check for IR and Broadcast connections that have packets to transmit.
   * \param connection will point to a initial ranging or broadcast connection
   * which have packets to transmit
   * \returns false if no initial ranging or broadcast connection has packets
   * to transmit, true otherwise
   */
  bool SelectIRandBCConnection (Ptr<WimaxConnection> &connection);
  /**
   * \brief Check for Basic and Primary connections that have packets to transmit.
   * \param connection will point to a basic or primary connection
   * which have packets to transmit
   * \returns false if no basic or primary connection has packets
   * to transmit, true otherwise
   */
  bool SelectMenagementConnection (Ptr<WimaxConnection> &connection);
  /**
   * \brief Check for UGS connections that have packets to transmit.
   *
   * The method return the UGS connection that have packets to transmits,
   * according to the MaxLatency specifications.
   * \param connection will point to a connection that has packets to be sent
   * \return true if successful
   */
  bool SelectUGSConnection (Ptr<WimaxConnection> &connection);
  /**
   * \brief Check for rtPS connections that have packets to transmit.
   * \param connection will point to a connection that has packets to be sent
   * \returns false if no connection has packets to be sent, true otherwise
   */
  bool SelectRTPSConnection (Ptr<WimaxConnection> &connection);
  /**
   * \brief Check for nrtPS connections that have packets to transmit.
   * \param connection will point to a connection that has packets to be sent
   * \returns false if no connection has packets to be sent, true otherwise
   */
  bool SelectNRTPSConnection (Ptr<WimaxConnection> &connection);
  /**
   * \brief Check for BE connections that have packets to transmit.
   * \param connection will point to a connection that has packets to be sent
   * \returns false if no connection has packets to be sent, true otherwise
   */
  bool SelectBEConnection (Ptr<WimaxConnection> &connection);

  /**
   * \brief Selects a connection from the list of connections having packets to be sent .
   * \param connection will point to a connection that has packets to be sent
   * \returns false if no connection has packets to be sent, true otherwise
   */
  bool SelectConnection (Ptr<WimaxConnection> &connection);

  /**
   * \brief schedules the broadcast connections
   * \param availableSymbols the remaining free OFDM symbols in the current subframe
   */
  void BSSchedulerBroadcastConnection (uint32_t &availableSymbols);

  /**
   * \brief schedules the IR connections
   * \param availableSymbols the remaining free OFDM symbols in the current subframe
   */
  void BSSchedulerInitialRangingConnection (uint32_t &availableSymbols);

  /**
   * \brief schedules the basic connections
   * \param availableSymbols the remaining free OFDM symbols in the current subframe
   */
  void BSSchedulerBasicConnection (uint32_t &availableSymbols);

  /**
   * \brief schedules the primary connection
   * \param availableSymbols the remaining free OFDM symbols in the current subframe
   */
  void BSSchedulerPrimaryConnection (uint32_t &availableSymbols);
  /**
   * \brief schedules the UGS connection
   * \param availableSymbols the remaining free OFDM symbols in the current subframe
   */
  void BSSchedulerUGSConnection (uint32_t &availableSymbols);
  /**
   * \brief Downlink Scheduler for rtPS connections.
   *
   * \param availableSymbols available symbols for rtPS flows
   *
   * This method represent the DL Scheduler for rtPS  connections.
   * The scheduler is designed to serve all rtPS connections that
   * have at least one packet to transmit, every DL Subframe.
   * To each rtPS connections are allocated,  at the beginning,
   * the bandwidth to transmit a quota of data equal to the
   * queue length.
   * If the sum of all pre-allocated bandwidth exceed the available
   * symbols for rtPS connections, the Channel Saturation Control function
   * is called to redistribute the available bandwidth.
   */
  void BSSchedulerRTPSConnection (uint32_t &availableSymbols);

  /**
   * \brief schedules the NRTPS connections
   * \param availableSymbols the remaining free OFDM symbols in the current subframe
   */
  void BSSchedulerNRTPSConnection (uint32_t &availableSymbols);

  /**
   * \brief schedules the BE connection
   * \param availableSymbols the remaining free OFDM symbols in the current subframe
   */
  void BSSchedulerBEConnection (uint32_t &availableSymbols);

  /**
   * \brief Creates a downlink UGS burst
   * \param serviceFlow the service flow of the burst
   * \param modulationType the modulation type to be used for the burst
   * \param availableSymbols maximum number of OFDM symbols to be used by the burst
   * \returns a Burst (list of packets)
   */
  Ptr<PacketBurst> CreateUgsBurst (ServiceFlow *serviceFlow,
                                   WimaxPhy::ModulationType modulationType, uint32_t availableSymbols);

private:
  std::list<std::pair<OfdmDlMapIe*, Ptr<PacketBurst> > > *m_downlinkBursts; ///< down link bursts
};

} // namespace ns3

#endif /* BS_SCHEDULER_RTPS_H */
