/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@cutebugs.net>
 */

#ifndef IPV4_INTERFACE_CONTAINER_H
#define IPV4_INTERFACE_CONTAINER_H

#include <stdint.h>
#include <vector>
#include "ns3/ipv4.h"
#include "ns3/ipv4-address.h"

namespace ns3 {

/**
 * \ingroup ipv4
 *
 * \brief holds a vector of std::pair of Ptr<Ipv4> and interface index.
 *
 * Typically ns-3 Ipv4Interfaces are installed on devices using an Ipv4 address
 * helper.  The helper's Assign() method takes a NetDeviceContainer which holds 
 * some number of Ptr<NetDevice>.  For each of the NetDevices in the 
 * NetDeviceContainer the helper will find the associated Ptr<Node> and
 * Ptr<Ipv4>.  It makes sure that an interface exists on the node for the 
 * device and then adds an Ipv4Address according to the address helper settings
 * (incrementing the Ipv4Address somehow as it goes).  The helper then converts
 * the Ptr<Ipv4> and the interface index to a std::pair and adds them to a 
 * container -- a container of this type.
 *
 * The point is then to be able to implicitly associate an index into the 
 * original NetDeviceContainer (that identifies a particular net device) with
 * an identical index into the Ipv4InterfaceContainer that has a std::pair with
 * the Ptr<Ipv4> and interface index you need to play with the interface.
 *
 * @see Ipv4AddressHelper
 * @see Ipv4
 */
class Ipv4InterfaceContainer
{
public:
  /**
   * \brief Container Const Iterator for pairs of Ipv4 smart pointer / Interface Index.
   */
  typedef std::vector<std::pair<Ptr<Ipv4>, uint32_t> >::const_iterator Iterator;

  /**
   * Create an empty Ipv4InterfaceContainer.
   */
  Ipv4InterfaceContainer ();

  /**
   * Concatenate the entries in the other container with ours.
   * \param other container
   */
  void Add (const Ipv4InterfaceContainer& other);

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
   *       std::pair<Ptr<Ipv4>, uint32_t> pair = *i;
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
   *       std::pair<Ptr<Ipv4>, uint32_t> pair = *i;
   *       method (pair.first, pair.second);  // use the pair
   *     }
   * \endcode
   *
   * \returns an iterator which indicates an ending condition for a loop.
   */
  Iterator End (void) const;

  /**
   * \returns the number of Ptr<Ipv4> and interface pairs stored in this 
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
   *       std::pair<Ptr<Ipv4>, uint32_t> pair = container.Get (i);
   *       method (pair.first, pair.second);  // use the pair
   *     }
   * \endcode
   *
   * \returns the number of Ptr<Node> stored in this container.
   */
  uint32_t GetN (void) const;

  /**
   * \param i index of ipInterfacePair in container
   * \param j interface address index (if interface has multiple addresses)
   * \returns the IPv4 address of the j'th address of the interface
   *  corresponding to index i.
   * 
   * If the second parameter is omitted, the zeroth indexed address of 
   * the interface is returned.  Unless IP aliasing is being used on
   * the interface, the second parameter may typically be omitted.
   */
  Ipv4Address GetAddress (uint32_t i, uint32_t j = 0) const;

  /**
   * \brief Set a metric for the given interface
   * \param i Interface index
   * \param metric the interface metric
   */
  void SetMetric (uint32_t i, uint16_t metric);

  /**
   * Manually add an entry to the container consisting of the individual parts
   * of an entry std::pair.
   *
   * \param ipv4 pointer to Ipv4 object
   * \param interface interface index of the Ipv4Interface to add to the container
   */
  void Add (Ptr<Ipv4> ipv4, uint32_t interface);

  /**
   * Manually add an entry to the container consisting of a previously composed 
   * entry std::pair.
   *
   * \param ipInterfacePair the pair of a pointer to Ipv4 object and interface index of the Ipv4Interface to add to the container
   */
  void Add (std::pair<Ptr<Ipv4>, uint32_t> ipInterfacePair);

  /**
   * Manually add an entry to the container consisting of the individual parts
   * of an entry std::pair.
   *
   * \param ipv4Name std:string referring to the saved name of an Ipv4 Object that
   *        has been previously named using the Object Name Service.
   * \param interface interface index of the Ipv4Interface to add to the container
   */
  void Add (std::string ipv4Name, uint32_t interface);

  /**
   * Get the std::pair of an Ptr<Ipv4> and interface stored at the location 
   * specified by the index.
   *
   * \param i the index of the container entry to retrieve.
   * \return the std::pair of a Ptr<Ipv4> and an interface index
   *
   * \note The returned Ptr<Ipv4> cannot be used directly to fetch the
   *       Ipv4Interface using the returned index (the GetInterface () method
   *       is provided in class Ipv4L3Protocol, and not class Ipv4). An
   *       example usage is provided below.
   *
   * \code
   *   Ipv4InterfaceContainer c;
   *   ...
   *   std::pair<Ptr<Ipv4>, uint32_t> returnValue = c.Get (0);
   *   Ptr<Ipv4> ipv4 = returnValue.first;
   *   uint32_t index = returnValue.second;
   *   Ptr<Ipv4Interface> iface =  ipv4->GetObject<Ipv4L3Protocol> ()->GetInterface (index);
   * \endcode
   */
  std::pair<Ptr<Ipv4>, uint32_t> Get (uint32_t i) const;

private:
  /**
   * \brief Container for pairs of Ipv4 smart pointer / Interface Index.
   */
  typedef std::vector<std::pair<Ptr<Ipv4>,uint32_t> > InterfaceVector;

  /**
   * \brief List of IPv4 stack and interfaces index.
   */
  InterfaceVector m_interfaces;
};

} // namespace ns3

#endif /* IPV4_INTERFACE_CONTAINER_H */
