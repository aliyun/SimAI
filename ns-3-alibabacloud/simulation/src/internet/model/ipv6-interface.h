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

#ifndef IPV6_INTERFACE_H
#define IPV6_INTERFACE_H

#include <list>
#include "ns3/ptr.h"
#include "ns3/object.h"
#include "ipv6-interface-address.h"

namespace ns3
{

class NetDevice;
class Packet;
class Node;
class NdiscCache;
class Ipv6InterfaceAddress;
class Ipv6Address;
class Ipv6Header;
class TrafficControlLayer;

/**
 * \ingroup ipv6
 *
 * \brief The IPv6 representation of a network interface.
 *
 * By default IPv6 interfaces are created in the "down" state
 * with IP "fe80::1" and a /64 prefix. Before becoming usable,
 * the user must invoke SetUp on them once the final IPv6 address
 * and mask has been set.
 */
class Ipv6Interface : public Object
{
public:
  /**
   * \brief Get the type ID
   * \return type ID
   */
  static TypeId GetTypeId ();

  /**
   * \brief Constructs an Ipv6Interface.
   */
  Ipv6Interface ();

  /**
   * \brief Destructor.
   */
  virtual ~Ipv6Interface ();

  // Delete copy constructor and assignment operator to avoid misuse
  Ipv6Interface (const Ipv6Interface &) = delete;
  Ipv6Interface & operator = (const Ipv6Interface &) = delete;

  /**
   * \brief Set node associated with interface.
   * \param node node
   */
  void SetNode (Ptr<Node> node);

  /**
   * \brief Set the NetDevice.
   * \param device NetDevice
   */
  void SetDevice (Ptr<NetDevice> device);

  /**
   * \brief Set the TrafficControlLayer.
   * \param tc TrafficControlLayer object
   */
  void SetTrafficControl (Ptr<TrafficControlLayer> tc);

  /**
   * \brief Get the NetDevice.
   * \return the NetDevice associated with this interface
   */
  virtual Ptr<NetDevice> GetDevice () const;

  /**
   * \brief Set the metric.
   * \param metric configured routing metric (cost) of this interface
   */
  void SetMetric (uint16_t metric);

  /**
   * \brief Get the metric
   * \return the metric
   */
  uint16_t GetMetric () const;

  /**
   * \brief Is the interface UP ?
   * \return true if interface is enabled, false otherwise.
   */
  bool IsUp () const;

  /**
   * \brief Is the interface DOWN ?
   * \return true if interface is disabled, false otherwise.
   */
  bool IsDown () const;

  /**
   * \brief Enable this interface.
   */
  void SetUp ();

  /**
   * \brief Disable this interface.
   */
  void SetDown ();

  /**
   * \brief If the interface allows forwarding packets.
   * \return true if forwarding is enabled, false otherwise
   */
  bool IsForwarding () const;

  /**
   * \brief Set forwarding enabled or not.
   * \param forward forwarding state
   */
  void SetForwarding (bool forward);

  /**
   * \brief Set the current hop limit.
   * \param curHopLimit the value to set
   */
  void SetCurHopLimit (uint8_t curHopLimit);

  /**
   * \brief Get the current hop limit value.
   * \return current hop limit
   */
  uint8_t GetCurHopLimit () const;

  /**
   * \brief Set the base reachable time.
   * \param baseReachableTime the value to set
   */
  void SetBaseReachableTime (uint16_t baseReachableTime);

  /**
   * \brief Get the base reachable time.
   * \return base reachable time
   */
  uint16_t GetBaseReachableTime () const;

  /**
   * \brief Set the reachable time.
   * \param reachableTime value to set
   */
  void SetReachableTime (uint16_t reachableTime);

  /**
   * \brief Get the reachable time.
   * \return reachable time
   */
  uint16_t GetReachableTime () const;

  /**
   * \brief Set the retransmission timer.
   * \param retransTimer value to set
   */
  void SetRetransTimer (uint16_t retransTimer);

  /**
   * \brief Get the retransmission timer.
   * \return retransmission timer
   */
  uint16_t GetRetransTimer () const;

  /**
   * \brief Send a packet through this interface.
   * \param p packet to send
   * \param hdr IPv6 header
   * \param dest next hop address of packet.
   *
   * \note This method will eventually call the private SendTo
   * method which must be implemented by subclasses.
   */
  void Send (Ptr<Packet> p, const Ipv6Header & hdr, Ipv6Address dest);

  /**
   * \brief Add an IPv6 address.
   * \param iface address to add
   * \return true if address was added, false otherwise
   */
  bool AddAddress (Ipv6InterfaceAddress iface);

  /**
   * \brief Get link-local address from IPv6 interface.
   * \return link-local Ipv6InterfaceAddress, assert if not found
   */
  Ipv6InterfaceAddress GetLinkLocalAddress () const;

  /**
   * \brief Checks if the address is a Solicited Multicast address for this interface.
   * \param address the address to check.
   * \return true if it is a solicited multicast address.
   */
  bool IsSolicitedMulticastAddress (Ipv6Address address) const;

  /**
   * \brief Get an address from IPv6 interface.
   * \param index index
   * \return Ipv6InterfaceAddress address whose index is i
   */
  Ipv6InterfaceAddress GetAddress (uint32_t index) const;

  /**
   * \brief Get an address which is in the same network prefix as destination.
   * \param dst destination address
   * \return Corresponding Ipv6InterfaceAddress or assert if not found
   */
  Ipv6InterfaceAddress GetAddressMatchingDestination (Ipv6Address dst);

  /**
   * \brief Get number of addresses on this IPv6 interface.
   * \return number of address
   */
  uint32_t GetNAddresses (void) const;

  /**
   * \brief Remove an address from interface.
   * \param index index to remove
   * \return Ipv6InterfaceAddress address whose index is index
   */
  Ipv6InterfaceAddress RemoveAddress (uint32_t index);

  /**
   * \brief Remove the given Ipv6 address from the interface.
   * \param address The Ipv6 address to remove
   * \returns The removed Ipv6 interface address 
   * \returns The null interface address if the interface did not contain the 
   * address or if loopback address was passed as argument
   */
  Ipv6InterfaceAddress RemoveAddress (Ipv6Address address);

  /**
   * \brief Update state of an interface address.
   * \param address IPv6 address
   * \param state new state
   */
  void SetState (Ipv6Address address, Ipv6InterfaceAddress::State_e state);

  /**
   * \brief Update NS DAD packet UID of an interface address.
   * \param address IPv6 address
   * \param uid packet UID 
   */
  void SetNsDadUid (Ipv6Address address, uint32_t uid);

  /**
   * \return NDISC cache used by this interface
   */
  Ptr<NdiscCache> GetNdiscCache () const;


protected:
  /**
   * \brief Dispose this object.
   */
  virtual void DoDispose ();

private:
  /**
   * \brief Container for the Ipv6InterfaceAddresses.
   */
  typedef std::list<std::pair<Ipv6InterfaceAddress, Ipv6Address> > Ipv6InterfaceAddressList;

  /**
   * \brief Container Iterator for the Ipv6InterfaceAddresses.
   */
  typedef std::list<std::pair<Ipv6InterfaceAddress, Ipv6Address> >::iterator Ipv6InterfaceAddressListI;

  /**
   * \brief Const Container Iterator for the Ipv6InterfaceAddresses.
   */
  typedef std::list<std::pair<Ipv6InterfaceAddress, Ipv6Address> >::const_iterator Ipv6InterfaceAddressListCI;

  /**
   * \brief Initialize interface.
   */
  void DoSetup ();

  /**
   * \brief The addresses assigned to this interface.
   */
  Ipv6InterfaceAddressList m_addresses;

  /**
   * \brief The link-local addresses assigned to this interface.
   */
  Ipv6InterfaceAddress m_linkLocalAddress;

  /**
   * \brief The state of this interface.
   */
  bool m_ifup;

  /**
   * \brief Forwarding state.
   */
  bool m_forwarding;

  /**
   * \brief The metric.
   */
  uint16_t m_metric;

  /**
   * \brief Node associated with this interface.
   */
  Ptr<Node> m_node;

  /**
   * \brief NetDevice associated with this interface.
   */
  Ptr<NetDevice> m_device;

  /**
   * \brief TrafficControlLayer associated with this interface.
   */
  Ptr<TrafficControlLayer> m_tc;

  /**
   * \brief Neighbor cache.
   */
  Ptr<NdiscCache> m_ndCache;

  /**
   * \brief Current hop limit.
   */
  uint8_t m_curHopLimit;

  /**
   * \brief Base value used for computing the random reachable time value (in millisecond).
   */
  uint16_t m_baseReachableTime;

  /**
   * \brief Reachable time (in millisecond).
   * The time a neighbor is considered reachable after receiving a reachability confirmation.
   */
  uint16_t m_reachableTime;

  /**
   * \brief Retransmission timer (in millisecond).
   * Time between retransmission of NS.
   */
  uint16_t m_retransTimer;
};

} /* namespace ns3 */

#endif /* IPV6_INTERFACE_H */

