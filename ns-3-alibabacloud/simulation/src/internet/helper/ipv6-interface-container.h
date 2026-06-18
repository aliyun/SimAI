/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008-2009 Strasbourg University
 *               2013 Universita' di Firenze
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
 *         Tommaso Pecorella <tommaso.pecorella@unifi.it>
 */

#ifndef IPV6_INTERFACE_CONTAINER_H
#define IPV6_INTERFACE_CONTAINER_H

#include <stdint.h>

#include <vector>

#include "ns3/ipv6.h"
#include "ns3/ipv6-address.h"

namespace ns3
{

/**
 * \ingroup ipv6
 *
 * \brief Keep track of a set of IPv6 interfaces.
 */
class Ipv6InterfaceContainer
{
public:

  /**
   * \brief Container Const Iterator for pairs of Ipv6 smart pointer / Interface Index.
   */
  typedef std::vector<std::pair<Ptr<Ipv6>, uint32_t> >::const_iterator Iterator;

  /**
   * \brief Constructor.
   */
  Ipv6InterfaceContainer ();

  /**
   * \returns the number of Ptr<Ipv6> and interface pairs stored in this 
   * Ipv4InterfaceContainer.
   *
   * Pairs can be retrieved from the container in two ways.  First,
   * directly by an index into the container, and second, using an iterator.
   * This method is used in the direct method and is typically used to
   * define an ending condition in a for-loop that runs through the stored
   * Nodes
   *
   * \code
   *   uint32_t nNodes = container.GetN ();
   *   for (uint32_t i = 0 i < nNodes; ++i)
   *     {
   *       std::pair<Ptr<Ipv6>, uint32_t> pair = container.Get (i);
   *       method (pair.first, pair.second);  // use the pair
   *     }
   * \endcode
   */
  uint32_t GetN (void) const;

  /**
   * \brief Get the interface index for the specified node index.
   * \param i index of the node
   * \return interface index
   */
  uint32_t GetInterfaceIndex (uint32_t i) const;

  /**
   * \brief Get the address for the specified index.
   * \param i interface index
   * \param j address index, generally index 0 is the link-local address
   * \return IPv6 address
   */
  Ipv6Address GetAddress (uint32_t i, uint32_t j) const;

  /**
   * \brief Get the link-local address for the specified index.
   * \param i index
   * \return the link-local address, or "::" if the interface has no link local address.
   */
  Ipv6Address GetLinkLocalAddress (uint32_t i);

  /**
   * \brief Get the link-local address for the node with the specified global address.
   * \param address the address to find.
   * \return the link-local address, or "::" if the interface has no link local address.
   */
  Ipv6Address GetLinkLocalAddress (Ipv6Address address);

  /**
   * \brief Add a couple IPv6/interface.
   * \param ipv6 IPv6 address
   * \param interface interface index
   */
  void Add (Ptr<Ipv6> ipv6, uint32_t interface);

  /**
   * \brief Get an iterator which refers to the first pair in the 
   * container.
   *
   * Pairs can be retrieved from the container in two ways.  First,
   * directly by an index into the container, and second, using an iterator.
   * This method is used in the iterator method and is typically used in a 
   * for-loop to run through the pairs
   *
   * \code
   *   Ipv4InterfaceContainer::Iterator i;
   *   for (i = container.Begin (); i != container.End (); ++i)
   *     {
   *       std::pair<Ptr<Ipv6>, uint32_t> pair = *i;
   *       method (pair.first, pair.second);  // use the pair
   *     }
   * \endcode
   *
   * \returns an iterator which refers to the first pair in the container.
   */
  Iterator Begin (void) const;

  /**
   * \brief Get an iterator which indicates past-the-last Node in the 
   * container.
   *
   * Nodes can be retrieved from the container in two ways.  First,
   * directly by an index into the container, and second, using an iterator.
   * This method is used in the iterator method and is typically used in a 
   * for-loop to run through the Nodes
   *
   * \code
   *   NodeContainer::Iterator i;
   *   for (i = container.Begin (); i != container.End (); ++i)
   *     {
   *       std::pair<Ptr<Ipv6>, uint32_t> pair = *i;
   *       method (pair.first, pair.second);  // use the pair
   *     }
   * \endcode
   *
   * \returns an iterator which indicates an ending condition for a loop.
   */
  Iterator End (void) const;

  /**
   * \brief Fusion with another Ipv6InterfaceContainer.
   * \param c container
   */
  void Add (const Ipv6InterfaceContainer& c);

  /**
   * \brief Add a couple of name/interface.
   * \param ipv6Name name of a node
   * \param interface interface index to add
   */
  void Add (std::string ipv6Name, uint32_t interface);

  /**
   * \brief Set the state of the stack (act as a router or as an host) for the specified index.
   * This automatically sets all the node's interfaces to the same forwarding state.
   * \param i index
   * \param state true : is a router, false : is an host
   */
  void SetForwarding (uint32_t i, bool state);

  /**
   * \brief Set the default route for all the devices (except the router itself).
   * \param router the default router index
   */
  void SetDefaultRouteInAllNodes (uint32_t router);

  /**
   * \brief Set the default route for all the devices (except the router itself).
   * Note that the route will be set to the link-local address of the node with the specified address.
   * \param routerAddr the default router address
   */
  void SetDefaultRouteInAllNodes (Ipv6Address routerAddr);

  /**
   * \brief Set the default route for the specified index.
   * \param i index
   * \param router the default router
   */
  void SetDefaultRoute (uint32_t i, uint32_t router);

  /**
   * \brief Set the default route for the specified index.
   * Note that the route will be set to the link-local address of the node with the specified address.
   * \param i index
   * \param routerAddr the default router address
   */
  void SetDefaultRoute (uint32_t i, Ipv6Address routerAddr);

private:
  /**
   * \brief Container for pairs of Ipv6 smart pointer / Interface Index.
   */
  typedef std::vector<std::pair<Ptr<Ipv6>, uint32_t> > InterfaceVector;

  /**
   * \brief List of IPv6 stack and interfaces index.
   */
  InterfaceVector m_interfaces;
};

} /* namespace ns3 */

#endif /* IPV6_INTERFACE_CONTAINER_H */

