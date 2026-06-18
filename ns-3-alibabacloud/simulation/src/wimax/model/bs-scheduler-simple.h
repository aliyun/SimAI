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

/* BS outbound scheduler as per in Section 6.3.5.1 */

#ifndef BS_SCHEDULER_SIMPLE_H
#define BS_SCHEDULER_SIMPLE_H

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
 * 
 * BaseStation Scheduler - simplified
 */
class BSSchedulerSimple : public BSScheduler
{
public:
  BSSchedulerSimple ();
  /**
   * Constructor
   *
   * \param bs base station device
   */
  BSSchedulerSimple (Ptr<BaseStationNetDevice> bs);
  ~BSSchedulerSimple (void);

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
   * \brief the scheduling function for the downlink subframe.
   */
  void Schedule (void);
  /**
   * \brief Selects a connection from the list of connections having packets to be sent .
   * \param connection will point to a connection that have packets to be sent
   * \returns false if no connection has packets to be sent, true otherwise
   */
  bool SelectConnection (Ptr<WimaxConnection> &connection);
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

#endif /* BS_SCHEDULER_SIMPLE_H */
