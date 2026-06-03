/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Universita' di Firenze
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
 * Author: Tommaso Pecorella <tommaso.pecorella@unifi.it>
 */

#ifndef IPV6_PMTU_CACHE_H
#define IPV6_PMTU_CACHE_H

#include <map>

#include "ns3/object.h"
#include "ns3/nstime.h"
#include "ns3/type-id.h"
#include "ns3/event-id.h"
#include "ns3/ipv6-address.h"

namespace ns3 {

/**
 * \ingroup ipv6
 *
 * \brief This class implements the Path MTU cache, as defined by \RFC{1981}.
 *
 * The Path MTU is stored according to the destination address, and it is
 * cleared upon expiration (default validity time is 10 minutes).
 *
 * The "infinite lifetime" PMTU entry type is not implemented, since it is
 * useful only in an very limited number of cases. See the RFC for further
 * details.
 */

class Ipv6PmtuCache : public Object
{
public:
  class Entry;

  /**
   * \brief Get the type ID
   * \return type ID
   */
  static TypeId GetTypeId ();

  /**
   * \brief Constructor.
   */
  Ipv6PmtuCache ();

  /**
   * \brief Destructor.
   */
  ~Ipv6PmtuCache ();

  /**
   * \brief Dispose object.
   */
  virtual void DoDispose ();

  /**
   * \brief Gets the known Path MTU for the specific destination
   * \param dst the destination
   * \return the Path MTU (zero if unknown)
   */
  uint32_t GetPmtu (Ipv6Address dst);

  /**
   * \brief Sets the Path MTU for the specific destination
   * \param dst the destination
   * \param pmtu the Path MTU
   */
  void SetPmtu (Ipv6Address dst, uint32_t pmtu);

  /**
   * \brief Gets the Path MTU validity time
   * \return the Path MTU validity time
   */
  Time GetPmtuValidityTime () const;

  /**
   * \brief Sets the Path MTU validity time (minimum is 5 minutes)
   * \param validity the Path MTU validity time
   * \return true if the change was successful
   */
  bool SetPmtuValidityTime (Time validity);

private:
  /**
   * \brief Clears the Path MTU for the specific destination
   * \param dst the destination
   */
  void ClearPmtu (Ipv6Address dst);

  /**
   * \brief Path MTU table
   */
  std::map<Ipv6Address, uint32_t> m_pathMtu;

  /**
   * \brief Container of the IPv6 PMTU data (Ipv6 destination address and expiration event).
   */
  typedef std::map<Ipv6Address, EventId>::iterator pathMtuTimerIter;

  /**
   * \brief Path MTU Expiration table
   */
  std::map<Ipv6Address, EventId> m_pathMtuTimer;

  /**
   * \brief Path MTU entry validity time
   */
  Time m_validityTime;
};

}

#endif /* IPV6_PMTU_CACHE_H */
