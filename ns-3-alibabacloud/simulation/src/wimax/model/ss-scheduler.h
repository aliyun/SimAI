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

/* SS outbound scheduler as per in Section 6.3.5.1 */

#ifndef SS_SCHEDULER_H
#define SS_SCHEDULER_H

#include <stdint.h>
#include "ns3/packet.h"
#include "ns3/packet-burst.h"
#include "wimax-phy.h"
#include "wimax-mac-header.h"

namespace ns3 {

class SubscriberStationNetDevice;
class WimaxConnection;

/**
 * \ingroup wimax
 * \param SSScheduler class
 */
class SSScheduler : public Object
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
   * \param ss subscriber station device
   */
  SSScheduler (Ptr<SubscriberStationNetDevice> ss);
  ~SSScheduler (void);

  /**
   * Set poll me value
   * \param pollMe the poll me flag
   */
  void SetPollMe (bool pollMe);
  /**
   * Get the poll me value
   * \returns the poll me flag
   */
  bool GetPollMe (void) const;
  /**
   * \return a list of packet to be sent in the next opportunity
   * \param availableSymbols the available resources in symbols
   * \param modulationType the used modulation
   * \param packetType the type of packets to select from
   * \param connection the connection from which packets will be selected
   */
  Ptr<PacketBurst> Schedule (uint16_t availableSymbols,
                             WimaxPhy::ModulationType modulationType,
                             MacHeaderType::HeaderType packetType, Ptr<WimaxConnection> &connection);


  void DoDispose (void);
protected:
private:
  /// type conversion operator
  SSScheduler (const SSScheduler &);
  /**
   * assignment operator
   * \returns the SS scheduler
   */
  SSScheduler & operator= (const SSScheduler &);

  /**
   * Select connection
   * \returns pointer to the wimax connection
   */
  Ptr<WimaxConnection> SelectConnection (void);
  Ptr<SubscriberStationNetDevice> m_ss; ///< the subscriber station
  bool m_pollMe; ///< poll me flag

};

} // namespace ns3

#endif /* SS_SCHEDULER_H */
