/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#ifndef IPV4_END_POINT_DEMUX_H
#define IPV4_END_POINT_DEMUX_H

#include <stdint.h>
#include <list>
#include "ns3/ipv4-address.h"
#include "ipv4-interface.h"

namespace ns3 {

class Ipv4EndPoint;

/**
 * \ingroup ipv4
 *
 * \brief Demultiplexes packets to various transport layer endpoints
 *
 * This class serves as a lookup table to match partial or full information
 * about a four-tuple to an ns3::Ipv4EndPoint.  It internally contains a list
 * of endpoints, and has APIs to add and find endpoints in this demux.  This
 * code is shared in common to TCP and UDP protocols in ns3.  This demux
 * sits between ns3's layer four and the socket layer
 */

class Ipv4EndPointDemux {
public:
  /**
   * \brief Container of the IPv4 endpoints.
   */
  typedef std::list<Ipv4EndPoint *> EndPoints;

  /**
   * \brief Iterator to the container of the IPv4 endpoints.
   */
  typedef std::list<Ipv4EndPoint *>::iterator EndPointsI;

  Ipv4EndPointDemux ();
  ~Ipv4EndPointDemux ();

  /**
   * \brief Get the entire list of end points registered.
   * \return list of Ipv4EndPoint
   */
  EndPoints GetAllEndPoints (void);

  /**
   * \brief Lookup for port local.
   * \param port port to test
   * \return true if a port local is in EndPoints, false otherwise
   */
  bool LookupPortLocal (uint16_t port);

  /**
   * \brief Lookup for address and port.
   * \param boundNetDevice Bound NetDevice (if any)
   * \param addr address to test
   * \param port port to test
   * \return true if there is a match in EndPoints, false otherwise
   */
  bool LookupLocal (Ptr<NetDevice> boundNetDevice, Ipv4Address addr, uint16_t port);

  /**
   * \brief lookup for a match with all the parameters.
   *
   * The function will return a list of most-matching EndPoints, in this order:
   *   -# Full match
   *   -# All but local address
   *   -# Only local port and local address match
   *   -# Only local port match
   *
   * EndPoint with disabled Rx are skipped.
   *
   * \param daddr destination address to test
   * \param dport destination port to test
   * \param saddr source address to test
   * \param sport source port to test
   * \param incomingInterface the incoming interface
   * \return list of IPv4EndPoints (could be 0 element)
   */
  EndPoints Lookup (Ipv4Address daddr, 
                    uint16_t dport, 
                    Ipv4Address saddr, 
                    uint16_t sport,
                    Ptr<Ipv4Interface> incomingInterface);

  /**
   * \brief simple lookup for a match with all the parameters.
   * \param daddr destination address to test
   * \param dport destination port to test
   * \param saddr source address to test
   * \param sport source port to test
   * \return IPv4EndPoint (0 if not found)
   */
  Ipv4EndPoint *SimpleLookup (Ipv4Address daddr, 
                              uint16_t dport, 
                              Ipv4Address saddr, 
                              uint16_t sport);

  /**
   * \brief Allocate a Ipv4EndPoint.
   * \return an empty Ipv4EndPoint instance
   */
  Ipv4EndPoint *Allocate (void);

  /**
   * \brief Allocate a Ipv4EndPoint.
   * \param address IPv4 address
   * \return an Ipv4EndPoint instance
   */
  Ipv4EndPoint *Allocate (Ipv4Address address);

  /**
   * \brief Allocate a Ipv4EndPoint.
   * \param boundNetDevice Bound NetDevice (if any)
   * \param port local port
   * \return an Ipv4EndPoint instance
   */
  Ipv4EndPoint *Allocate (Ptr<NetDevice> boundNetDevice, uint16_t port);

  /**
   * \brief Allocate a Ipv4EndPoint.
   * \param boundNetDevice Bound NetDevice (if any)
   * \param address local address
   * \param port local port
   * \return an Ipv4EndPoint instance
   */
  Ipv4EndPoint *Allocate (Ptr<NetDevice> boundNetDevice, Ipv4Address address, uint16_t port);

  /**
   * \brief Allocate a Ipv4EndPoint.
   * \param boundNetDevice Bound NetDevice (if any)
   * \param localAddress local address
   * \param localPort local port
   * \param peerAddress peer address
   * \param peerPort peer port
   * \return an Ipv4EndPoint instance
   */
  Ipv4EndPoint *Allocate (Ptr<NetDevice> boundNetDevice,
                          Ipv4Address localAddress, uint16_t localPort,
                          Ipv4Address peerAddress, uint16_t peerPort);

  /**
   * \brief Remove a end point.
   * \param endPoint the end point to remove
   */
  void DeAllocate (Ipv4EndPoint *endPoint);

private:

  /**
   * \brief Allocate an ephemeral port.
   * \returns the ephemeral port
   */
  uint16_t AllocateEphemeralPort (void);

  /**
   * \brief The ephemeral port.
   */
  uint16_t m_ephemeral;

  /**
   * \brief The last ephemeral port.
   */
  uint16_t m_portLast;

  /**
   * \brief The first ephemeral port.
   */
  uint16_t m_portFirst;

  /**
   * \brief A list of IPv4 end points.
   */
  EndPoints m_endPoints;
};

} // namespace ns3

#endif /* IPV4_END_POINTS_H */
