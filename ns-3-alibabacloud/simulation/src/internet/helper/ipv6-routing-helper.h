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

#ifndef IPV6_ROUTING_HELPER_H
#define IPV6_ROUTING_HELPER_H

#include "ns3/ptr.h"
#include "ns3/nstime.h"
#include "ns3/output-stream-wrapper.h"
#include "ns3/ipv6-list-routing.h"

namespace ns3 {

class Ipv6RoutingProtocol;
class Node;

/**
 * \ingroup ipv6Helpers
 *
 * \brief A factory to create ns3::Ipv6RoutingProtocol objects
 *
 * For each new routing protocol created as a subclass of 
 * ns3::Ipv6RoutingProtocol, you need to create a subclass of 
 * ns3::Ipv6RoutingHelper which can be used by 
 * ns3::InternetStackHelper::SetRoutingHelper and 
 * ns3::InternetStackHelper::Install.
 */
class Ipv6RoutingHelper
{
public:

  /**
   * \brief Destroy an Ipv6 Ipv6RoutingHelper.
   */
  virtual ~Ipv6RoutingHelper ();

  /**
   * \brief virtual constructor
   * \returns pointer to clone of this Ipv6RoutingHelper 
   * 
   * This method is mainly for internal use by the other helpers;
   * clients are expected to free the dynamic memory allocated by this method
   */
  virtual Ipv6RoutingHelper* Copy (void) const = 0;

  /**
   * \param node the node within which the new routing protocol will run
   * \returns a newly-created routing protocol
   */
  virtual Ptr<Ipv6RoutingProtocol> Create (Ptr<Node> node) const = 0;

  /**
   * \brief prints the routing tables of all nodes at a particular time.
   * \param printTime the time at which the routing table is supposed to be printed.
   * \param stream The output stream object to use
   * \param unit The time unit to be used in the report
   *
   * This method calls the PrintRoutingTable() method of the
   * Ipv6RoutingProtocol stored in the Ipv6 object, for all nodes at the
   * specified time; the output format is routing protocol-specific.
   */
  static void PrintRoutingTableAllAt (Time printTime, Ptr<OutputStreamWrapper> stream, Time::Unit unit = Time::S);

  /**
   * \brief prints the routing tables of all nodes at regular intervals specified by user.
   * \param printInterval the time interval for which the routing table is supposed to be printed.
   * \param stream The output stream object to use
   * \param unit The time unit to be used in the report
   *
   * This method calls the PrintRoutingTable() method of the
   * Ipv6RoutingProtocol stored in the Ipv6 object, for all nodes at the
   * specified time interval; the output format is routing protocol-specific.
   */
  static void PrintRoutingTableAllEvery (Time printInterval, Ptr<OutputStreamWrapper> stream, Time::Unit unit = Time::S);

  /**
   * \brief prints the routing tables of a node at a particular time.
   * \param printTime the time at which the routing table is supposed to be printed.
   * \param node The node ptr for which we need the routing table to be printed
   * \param stream The output stream object to use
   * \param unit The time unit to be used in the report
   *
   * This method calls the PrintRoutingTable() method of the
   * Ipv6RoutingProtocol stored in the Ipv6 object, for the selected node
   * at the specified time; the output format is routing protocol-specific.
   */
  static void PrintRoutingTableAt (Time printTime, Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit = Time::S);

  /**
   * \brief prints the routing tables of a node at regular intervals specified by user.
   * \param printInterval the time interval for which the routing table is supposed to be printed.
   * \param node The node ptr for which we need the routing table to be printed
   * \param stream The output stream object to use
   * \param unit The time unit to be used in the report
   *
   * This method calls the PrintRoutingTable() method of the
   * Ipv6RoutingProtocol stored in the Ipv6 object, for the selected node
   * at the specified interval; the output format is routing protocol-specific.
   */
  static void PrintRoutingTableEvery (Time printInterval, Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit = Time::S);

  /**
   * \brief prints the neighbor cache of all nodes at a particular time.
   * \param printTime the time at which the neighbor cache is supposed to be printed.
   * \param stream The output stream object to use
   * \param unit The time unit to be used in the report
   *
   * This method calls the PrintNdiscCache() method of the
   * NdiscCache associated with each Ipv6Interface stored in the Ipv6 object, for all nodes at the
   * specified time. The output format is similar to:
   * \verbatim
     2001:db8::f00d:beef:cafe dev 1 lladdr 00-06-00:00:00:00:00:02 REACHABLE
     \endverbatim
   * Note that the MAC address is printed as "type"-"size"-"actual address"
   */
  static void PrintNeighborCacheAllAt (Time printTime, Ptr<OutputStreamWrapper> stream, Time::Unit unit = Time::S);

  /**
   * \brief prints the neighbor cache of all nodes at regular intervals specified by user.
   * \param printInterval the time interval for which the neighbor cache is supposed to be printed.
   * \param stream The output stream object to use
   * \param unit The time unit to be used in the report
   *
   * This method calls the PrintNdiscCache() method of the
   * NdiscCache associated with each Ipv6Interface stored in the Ipv6 object, for all nodes at the
   * specified time. The output format is similar to:
   * \verbatim
     2001:db8::f00d:beef:cafe dev 1 lladdr 00-06-00:00:00:00:00:02 REACHABLE
     \endverbatim
   * Note that the MAC address is printed as "type"-"size"-"actual address"
   */
  static void PrintNeighborCacheAllEvery (Time printInterval, Ptr<OutputStreamWrapper> stream, Time::Unit unit = Time::S);

  /**
   * \brief prints the neighbor cache of a node at a particular time.
   * \param printTime the time at which the neighbor cache is supposed to be printed.
   * \param node The node ptr for which we need the neighbor cache to be printed
   * \param stream The output stream object to use
   * \param unit The time unit to be used in the report
   *
   * This method calls the PrintNdiscCache() method of the
   * NdiscCache associated with each Ipv6Interface stored in the Ipv6 object, for all nodes at the
   * specified time. The output format is similar to:
   * \verbatim
     2001:db8::f00d:beef:cafe dev 1 lladdr 00-06-00:00:00:00:00:02 REACHABLE
     \endverbatim
   * Note that the MAC address is printed as "type"-"size"-"actual address"
   */
  static void PrintNeighborCacheAt (Time printTime, Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit = Time::S);

  /**
   * \brief prints the neighbor cache of a node at regular intervals specified by user.
   * \param printInterval the time interval for which the neighbor cache is supposed to be printed.
   * \param node The node ptr for which we need the neighbor cache to be printed
   * \param stream The output stream object to use
   * \param unit The time unit to be used in the report
   *
   * This method calls the PrintNdiscCache() method of the
   * NdiscCache associated with each Ipv6Interface stored in the Ipv6 object, for all nodes at the
   * specified time. The output format is similar to:
   * \verbatim
     2001:db8::f00d:beef:cafe dev 1 lladdr 00-06-00:00:00:00:00:02 REACHABLE
     \endverbatim
   * Note that the MAC address is printed as "type"-"size"-"actual address"
   */
  static void PrintNeighborCacheEvery (Time printInterval, Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit = Time::S);

  /**
   * \brief Request a specified routing protocol &lt;T&gt; from Ipv6RoutingProtocol protocol
   *
   * If protocol is Ipv6ListRouting, then protocol will be searched in the list,
   * otherwise a simple DynamicCast will be performed
   *
   * \param protocol Smart pointer to Ipv6RoutingProtocol object
   * \return a Smart Pointer to the requested protocol (zero if the protocol can't be found)
   */
  template<class T>
  static Ptr<T> GetRouting (Ptr<Ipv6RoutingProtocol> protocol);
  
private:
  /**
   * \brief prints the routing tables of a node.
   * \param node The node ptr for which we need the routing table to be printed
   * \param stream The output stream object to use
   * \param unit The time unit to be used in the report
   *
   * This method calls the PrintRoutingTable() method of the
   * Ipv6RoutingProtocol stored in the Ipv6 object;
   * the output format is routing protocol-specific.
   */
  static void Print (Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit);

  /**
   * \brief prints the routing tables of a node at regular intervals specified by user.
   * \param printInterval the time interval for which the routing table is supposed to be printed.
   * \param node The node ptr for which we need the routing table to be printed
   * \param stream The output stream object to use
   * \param unit The time unit to be used in the report
   *
   * This method calls the PrintRoutingTable() method of the
   * Ipv6RoutingProtocol stored in the Ipv6 object, for the selected node
   * at the specified interval; the output format is routing protocol-specific.
   */
  static void PrintEvery (Time printInterval, Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit);

  /**
   * \brief prints the neighbor cache of a node.
   * \param node The node ptr for which we need the neighbor cache to be printed
   * \param stream The output stream object to use
   * \param unit The time unit to be used in the report
   *
   * This method calls the PrintNdiscCache() method of the
   * NdiscCache associated with each Ipv6Interface stored in the Ipv6 object, for all nodes at the
   * specified time. The output format is similar to:
   * \verbatim
     2001:db8::f00d:beef:cafe dev 1 lladdr 00-06-00:00:00:00:00:02 REACHABLE
     \endverbatim
   * Note that the MAC address is printed as "type"-"size"-"actual address"
   */
  static void PrintNdiscCache (Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit = Time::S);

  /**
   * \brief prints the neighbor cache of a node at regular intervals specified by user.
   * \param printInterval the time interval for which the neighbor cache is supposed to be printed.
   * \param node The node ptr for which we need the neighbor cache to be printed
   * \param stream The output stream object to use
   * \param unit The time unit to be used in the report
   *
   * This method calls the PrintNdiscCache() method of the
   * NdiscCache associated with each Ipv6Interface stored in the Ipv6 object, for all nodes at the
   * specified time. The output format is similar to:
   * \verbatim
     2001:db8::f00d:beef:cafe dev 1 lladdr 00-06-00:00:00:00:00:02 REACHABLE
     \endverbatim
   * Note that the MAC address is printed as "type"-"size"-"actual address"
   */
  static void PrintNdiscCacheEvery (Time printInterval, Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit = Time::S);
};

/**
 * \brief Request a specified routing protocol &lt;T&gt; from Ipv6RoutingProtocol protocol
 *
 * If protocol is Ipv6ListRouting, then protocol will be searched in the list,
 * otherwise a simple DynamicCast will be performed
 *
 * \param protocol Smart pointer to Ipv6RoutingProtocol object
 * \return a Smart Pointer to the requested protocol (zero if the protocol can't be found)
 */
template<class T>
Ptr<T> Ipv6RoutingHelper::GetRouting (Ptr<Ipv6RoutingProtocol> protocol)
{
  Ptr<T> ret = DynamicCast<T> (protocol);
  if (ret == 0)
    {
      // trying to check if protocol is a list routing
      Ptr<Ipv6ListRouting> lrp = DynamicCast<Ipv6ListRouting> (protocol);
      if (lrp != 0)
        {
          for (uint32_t i = 0; i < lrp->GetNRoutingProtocols ();  i++)
            {
              int16_t priority;
              ret = GetRouting<T> (lrp->GetRoutingProtocol (i, priority)); // potential recursion, if inside ListRouting is ListRouting
              if (ret != 0)
                break;
            }
        }
    }

  return ret;
}

} // namespace ns3


#endif /* IPV6_ROUTING_HELPER_H */

