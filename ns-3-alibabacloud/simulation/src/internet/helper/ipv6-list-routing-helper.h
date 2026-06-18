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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#ifndef IPV6_LIST_ROUTING_HELPER_H
#define IPV6_LIST_ROUTING_HELPER_H

#include <stdint.h>

#include <list>

#include "ns3/ipv6-routing-helper.h"

namespace ns3 {

/**
 * \ingroup ipv6Helpers
 *
 * \brief Helper class that adds ns3::Ipv6ListRouting objects
 *
 * This class is expected to be used in conjunction with 
 * ns3::InternetStackHelper::SetRoutingHelper
 */
class Ipv6ListRoutingHelper : public Ipv6RoutingHelper
{
public:
  /**
   * Construct an Ipv6 Ipv6ListRoutingHelper which is used to make life easier
   * for people wanting to configure routing using Ipv6.
   */
  Ipv6ListRoutingHelper ();

  /**
   * \brief Destroy an Ipv6 Ipv6ListRoutingHelper.
   */
  virtual ~Ipv6ListRoutingHelper ();

  /**
   * \brief Construct an Ipv6ListRoutingHelper from another previously 
   * initialized instance (Copy Constructor).
   * \param o object to be copied
   */
  Ipv6ListRoutingHelper (const Ipv6ListRoutingHelper &o);

  /**
   * \returns pointer to clone of this Ipv6ListRoutingHelper 
   * 
   * This method is mainly for internal use by the other helpers;
   * clients are expected to free the dynamic memory allocated by this method
   */
  Ipv6ListRoutingHelper* Copy (void) const;

  /**
   * \param routing a routing helper
   * \param priority the priority of the associated helper
   *
   * Store in the internal list a reference to the input routing helper
   * and associated priority. These helpers will be used later by
   * the ns3::Ipv6ListRoutingHelper::Create method to create
   * an ns3::Ipv6ListRouting object and add in it routing protocols
   * created with the helpers.
   */
  void Add (const Ipv6RoutingHelper &routing, int16_t priority);
  /**
   * \param node the node on which the routing protocol will run
   * \returns a newly-created routing protocol
   *
   * This method will be called by ns3::InternetStackHelper::Install
   */
  virtual Ptr<Ipv6RoutingProtocol> Create (Ptr<Node> node) const;
private:
  /**
   * \brief Assignment operator declared private and not implemented to disallow
   * assignment and prevent the compiler from happily inserting its own.
   * \param o object to copy from
   * \returns a reference to the new object
   */
  Ipv6ListRoutingHelper &operator = (const Ipv6ListRoutingHelper &o);

  /**
   * \brief Container for pairs of Ipv6RoutingHelper pointer / priority.
   */
  std::list<std::pair<const Ipv6RoutingHelper *,int16_t> > m_list;
};

} // namespace ns3

#endif /* IPV6_LIST_ROUTING_HELPER_H */

