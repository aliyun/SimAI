/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
//
// Copyright (c) 2009 INESC Porto
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation;
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
// Author: Gustavo J. A. M. Carneiro  <gjc@inescporto.pt> <gjcarneiro@gmail.com>
// Modifications: Tommaso Pecorella <tommaso.pecorella@unifi.it>
//

#ifndef IPV6_FLOW_PROBE_H
#define IPV6_FLOW_PROBE_H

#include "ns3/flow-probe.h"
#include "ns3/ipv6-flow-classifier.h"
#include "ns3/ipv6-l3-protocol.h"
#include "ns3/queue-item.h"

namespace ns3 {

class FlowMonitor;
class Node;

/// \ingroup flow-monitor
/// \brief Class that monitors flows at the IPv6 layer of a Node
///
/// For each node in the simulation, one instance of the class
/// Ipv4FlowProbe is created to monitor that node.  Ipv4FlowProbe
/// accomplishes this by connecting callbacks to trace sources in the
/// Ipv6L3Protocol interface of the node.
class Ipv6FlowProbe : public FlowProbe
{

public:
  /// \brief Constructor
  /// \param monitor the FlowMonitor this probe is associated with
  /// \param classifier the Ipv4FlowClassifier this probe is associated with
  /// \param node the Node this probe is associated with
  Ipv6FlowProbe (Ptr<FlowMonitor> monitor, Ptr<Ipv6FlowClassifier> classifier, Ptr<Node> node);
  virtual ~Ipv6FlowProbe ();

  /// Register this type.
  /// \return The TypeId.
  static TypeId GetTypeId (void);

  /// \brief enumeration of possible reasons why a packet may be dropped
  enum DropReason 
  {
    /// Packet dropped due to missing route to the destination
    DROP_NO_ROUTE = 0,

    /// Packet dropped due to TTL decremented to zero during IPv4 forwarding
    DROP_TTL_EXPIRE,

    /// Packet dropped due to invalid checksum in the IPv4 header
    DROP_BAD_CHECKSUM,

    /// Packet dropped due to queue overflow.  Note: only works for
    /// NetDevices that provide a TxQueue attribute of type Queue
    /// with a Drop trace source.  It currently works with Csma and
    /// PointToPoint devices, but not with WiFi or WiMax.
    DROP_QUEUE,

    /// Packet dropped by the queue disc
    DROP_QUEUE_DISC,

    DROP_INTERFACE_DOWN,   /**< Interface is down so can not send packet */
    DROP_ROUTE_ERROR,   /**< Route error */

    DROP_UNKNOWN_PROTOCOL, /**< Unknown L4 protocol */
    DROP_UNKNOWN_OPTION, /**< Unknown option */
    DROP_MALFORMED_HEADER, /**< Malformed header */

    DROP_FRAGMENT_TIMEOUT, /**< Fragment timeout exceeded */

    DROP_INVALID_REASON, /**< Fallback reason (no known reason) */
  };

protected:

  virtual void DoDispose (void);

private:
  /// Log a packet being sent
  /// \param ipHeader IP header
  /// \param ipPayload IP payload
  /// \param interface outgoing interface
  void SendOutgoingLogger (const Ipv6Header &ipHeader, Ptr<const Packet> ipPayload, uint32_t interface);
  /// Log a packet being forwarded
  /// \param ipHeader IP header
  /// \param ipPayload IP payload
  /// \param interface incoming interface
  void ForwardLogger (const Ipv6Header &ipHeader, Ptr<const Packet> ipPayload, uint32_t interface);
  /// Log a packet being received by the destination
  /// \param ipHeader IP header
  /// \param ipPayload IP payload
  /// \param interface incoming interface
  void ForwardUpLogger (const Ipv6Header &ipHeader, Ptr<const Packet> ipPayload, uint32_t interface);
  /// Log a packet being dropped
  /// \param ipHeader IP header
  /// \param ipPayload IP payload
  /// \param reason drop reason
  /// \param ipv6 pointer to the IP object dropping the packet
  /// \param ifIndex interface index
  void DropLogger (const Ipv6Header &ipHeader, Ptr<const Packet> ipPayload,
                   Ipv6L3Protocol::DropReason reason, Ptr<Ipv6> ipv6, uint32_t ifIndex);
  /// Log a packet being dropped by a queue
  /// \param ipPayload IP payload
  void QueueDropLogger (Ptr<const Packet> ipPayload);
  /// Log a packet being dropped by a queue disc
  /// \param item queue disc item
  void QueueDiscDropLogger (Ptr<const QueueDiscItem> item);

  Ptr<Ipv6FlowClassifier> m_classifier; //!< the Ipv6FlowClassifier this probe is associated with
};


} // namespace ns3

#endif /* IPV6_FLOW_PROBE_H */
