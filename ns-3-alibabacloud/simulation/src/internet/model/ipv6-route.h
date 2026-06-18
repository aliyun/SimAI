/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007-2009 Strasbourg University
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
 * Author: Sebastien Vincent <vincent@clarinet.u-strasbg.fr>
 */

#ifndef IPV6_ROUTE_H
#define IPV6_ROUTE_H

#include <list>
#include <map>
#include <ostream>

#include "ns3/simple-ref-count.h"

#include "ns3/ipv6-address.h"

namespace ns3
{

class NetDevice;

/**
 * \ingroup ipv6Routing
 *
 * \brief IPv6 route cache entry.
 */
class Ipv6Route : public SimpleRefCount<Ipv6Route>
{
public:
  /**
   * \brief Constructor.
   */
  Ipv6Route ();

  /**
   * \brief Destructor.
   */
  virtual ~Ipv6Route ();

  /**
   * \brief Set destination address.
   * \param dest IPv6 destination address
   */
  void SetDestination (Ipv6Address dest);

  /**
   * \brief Get destination address.
   * \return destination address
   */
  Ipv6Address GetDestination () const;

  /**
   * \brief Set source address.
   * \param src IPv6 source address
   */
  void SetSource (Ipv6Address src);

  /**
   * \brief Get source address.
   * \return source address
   */
  Ipv6Address GetSource () const;

  /**
   * \brief Set gateway address.
   * \param gw IPv6 gateway address
   */
  void SetGateway (Ipv6Address gw);

  /**
   * \brief Get gateway address.
   * \return gateway address
   */
  Ipv6Address GetGateway () const;

  /**
   * \brief Set output device for outgoing packets.
   * \param outputDevice output device
   */
  void SetOutputDevice (Ptr<NetDevice> outputDevice);

  /**
   * \brief Get output device.
   * \return output device
   */
  Ptr<NetDevice> GetOutputDevice () const;

private:
  /**
   * \brief Destination address.
   */
  Ipv6Address m_dest;

  /**
   * \brief source address.
   */
  Ipv6Address m_source;

  /**
   * \brief Gateway address.
   */
  Ipv6Address m_gateway;

  /**
   * \brief Output device.
   */
  Ptr<NetDevice> m_outputDevice;
};

/**
 * \brief Stream insertion operator.
 *
 * \param os the reference to the output stream
 * \param route the Ipv6 route
 * \returns the reference to the output stream
 */
std::ostream& operator<< (std::ostream& os, Ipv6Route const& route);

/**
 * \ingroup ipv6Routing
 *
 * \brief IPv6 multicast route entry.
 */
class Ipv6MulticastRoute : public SimpleRefCount<Ipv6MulticastRoute>
{
public:
  /**
   * \brief Maximum number of multicast interfaces on a router.
   */
  static const uint32_t MAX_INTERFACES = 16;

  /**
   * \brief Maximum Time-To-Live (TTL).
   */
  static const uint32_t MAX_TTL = 255;

  /**
   * \brief Constructor.
   */
  Ipv6MulticastRoute ();

  /**
   * \brief Destructor.
   */
  virtual ~Ipv6MulticastRoute ();

  /**
   * \brief Set IPv6 group.
   * \param group Ipv6Address of the multicast group
   */
  void SetGroup (const Ipv6Address group);

  /**
   * \brief Get IPv6 group.
   * \return Ipv6Address of the multicast group
   */
  Ipv6Address GetGroup (void) const;

  /**
   * \brief Set origin address.
   * \param origin Ipv6Address of the origin address
   */
  void SetOrigin (const Ipv6Address origin);

  /**
   * \brief Get source address.
   * \return Ipv6Address of the origin address
   */
  Ipv6Address GetOrigin (void) const;

  /**
   * \brief Set parent for this route.
   * \param iif Parent (input interface) for this route
   */
  void SetParent (uint32_t iif);

  /**
   * \brief Get parent for this route.
   * \return Parent (input interface) for this route
   */
  uint32_t GetParent (void) const;

  /**
   * \brief set output TTL for this route.
   * \param oif Outgoing interface index
   * \param ttl time-to-live for this route
   */
  void SetOutputTtl (uint32_t oif, uint32_t ttl);

  /**
   * \return map of output interface Ids and TTLs for this route
   */
  std::map<uint32_t, uint32_t> GetOutputTtlMap () const;

private:
  /**
   * \brief IPv6 group.
   */
  Ipv6Address m_group;

  /**
   * \brief IPv6 origin (source).
   */
  Ipv6Address m_origin;

  /**
   * \brief Source interface.
   */
  uint32_t m_parent;

  /**
   * \brief TTLs.
   */
  std::map<uint32_t, uint32_t> m_ttls;
};

/**
 * \brief Stream insertion operator.
 *
 * \param os the reference to the output stream
 * \param route the Ipv6 multicast route
 * \returns the reference to the output stream
 */
std::ostream& operator<< (std::ostream& os, Ipv6MulticastRoute const& route);

} /* namespace ns3 */

#endif /* IPV6_ROUTE_H */

