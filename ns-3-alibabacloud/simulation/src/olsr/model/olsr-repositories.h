/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2004 Francisco J. Ros
 * Copyright (c) 2007 INESC Porto
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
 * Authors: Francisco J. Ros  <fjrm@dif.um.es>
 *          Gustavo J. A. M. Carneiro <gjc@inescporto.pt>
 */


#ifndef OLSR_REPOSITORIES_H
#define OLSR_REPOSITORIES_H

#include <set>
#include <vector>

#include "ns3/ipv4-address.h"
#include "ns3/nstime.h"

namespace ns3 {
namespace olsr {


/// \ingroup olsr
/// An Interface Association Tuple.
struct IfaceAssocTuple
{
  /// Interface address of a node.
  Ipv4Address ifaceAddr;
  /// Main address of the node.
  Ipv4Address mainAddr;
  /// Time at which this tuple expires and must be removed.
  Time time;
};

static inline bool
operator == (const IfaceAssocTuple &a, const IfaceAssocTuple &b)
{
  return (a.ifaceAddr == b.ifaceAddr
          && a.mainAddr == b.mainAddr);
}

static inline std::ostream&
operator << (std::ostream &os, const IfaceAssocTuple &tuple)
{
  os << "IfaceAssocTuple(ifaceAddr=" << tuple.ifaceAddr
     << ", mainAddr=" << tuple.mainAddr
     << ", time=" << tuple.time << ")";
  return os;
}

/// \ingroup olsr
/// A Link Tuple.
struct LinkTuple
{
  /// Interface address of the local node.
  Ipv4Address localIfaceAddr;
  /// Interface address of the neighbor node.
  Ipv4Address neighborIfaceAddr;
  /// The link is considered bidirectional until this time.
  Time symTime;
  /// The link is considered unidirectional until this time.
  Time asymTime;
  /// Time at which this tuple expires and must be removed.
  Time time;
};

static inline bool
operator == (const LinkTuple &a, const LinkTuple &b)
{
  return (a.localIfaceAddr == b.localIfaceAddr
          && a.neighborIfaceAddr == b.neighborIfaceAddr);
}

static inline std::ostream&
operator << (std::ostream &os, const LinkTuple &tuple)
{
  os << "LinkTuple(localIfaceAddr=" << tuple.localIfaceAddr
     << ", neighborIfaceAddr=" << tuple.neighborIfaceAddr
     << ", symTime=" << tuple.symTime
     << ", asymTime=" << tuple.asymTime
     << ", expTime=" << tuple.time
     << ")";
  return os;
}

/// \ingroup olsr
/// A Neighbor Tuple.
struct NeighborTuple
{
  /// Main address of a neighbor node.
  Ipv4Address neighborMainAddr;
  /// Status of the link (Symmetric or not Symmetric).
  enum Status
  {
    STATUS_NOT_SYM = 0, // "not symmetric"
    STATUS_SYM = 1, // "symmetric"
  } status; //!< Status of the link.
  /// A value between 0 and 7 specifying the node's willingness to carry traffic on behalf of other nodes.
  uint8_t willingness;
};

static inline bool
operator == (const NeighborTuple &a, const NeighborTuple &b)
{
  return (a.neighborMainAddr == b.neighborMainAddr
          && a.status == b.status
          && a.willingness == b.willingness);
}

static inline std::ostream&
operator << (std::ostream &os, const NeighborTuple &tuple)
{
  os << "NeighborTuple(neighborMainAddr=" << tuple.neighborMainAddr
     << ", status=" << (tuple.status == NeighborTuple::STATUS_SYM ? "SYM" : "NOT_SYM")
     << ", willingness=" << (int) tuple.willingness << ")";
  return os;
}

/// \ingroup olsr
/// A 2-hop Tuple.
struct TwoHopNeighborTuple
{
  /// Main address of a neighbor.
  Ipv4Address neighborMainAddr;
  /// Main address of a 2-hop neighbor with a symmetric link to nb_main_addr.
  Ipv4Address twoHopNeighborAddr;
  /// Time at which this tuple expires and must be removed.
  Time expirationTime; // previously called 'time_'
};

static inline std::ostream&
operator << (std::ostream &os, const TwoHopNeighborTuple &tuple)
{
  os << "TwoHopNeighborTuple(neighborMainAddr=" << tuple.neighborMainAddr
     << ", twoHopNeighborAddr=" << tuple.twoHopNeighborAddr
     << ", expirationTime=" << tuple.expirationTime
     << ")";
  return os;
}

static inline bool
operator == (const TwoHopNeighborTuple &a, const TwoHopNeighborTuple &b)
{
  return (a.neighborMainAddr == b.neighborMainAddr
          && a.twoHopNeighborAddr == b.twoHopNeighborAddr);
}

/// \ingroup olsr
/// An MPR-Selector Tuple.
struct MprSelectorTuple
{
  /// Main address of a node which have selected this node as a MPR.
  Ipv4Address mainAddr;
  /// Time at which this tuple expires and must be removed.
  Time expirationTime; // previously called 'time_'
};

static inline bool
operator == (const MprSelectorTuple &a, const MprSelectorTuple &b)
{
  return (a.mainAddr == b.mainAddr);
}


// The type "list of interface addresses"
//typedef std::vector<nsaddr_t> addr_list_t;

/// \ingroup olsr
/// A Duplicate Tuple
struct DuplicateTuple
{
  /// Originator address of the message.
  Ipv4Address address;
  /// Message sequence number.
  uint16_t sequenceNumber;
  /// Indicates whether the message has been retransmitted or not.
  bool retransmitted;
  /// List of interfaces which the message has been received on.
  std::vector<Ipv4Address> ifaceList;
  /// Time at which this tuple expires and must be removed.
  Time expirationTime;
};

static inline bool
operator == (const DuplicateTuple &a, const DuplicateTuple &b)
{
  return (a.address == b.address
          && a.sequenceNumber == b.sequenceNumber);
}

/// \ingroup olsr
/// A Topology Tuple
struct TopologyTuple
{
  /// Main address of the destination.
  Ipv4Address destAddr;
  /// Main address of a node which is a neighbor of the destination.
  Ipv4Address lastAddr;
  /// Sequence number.
  uint16_t sequenceNumber;
  /// Time at which this tuple expires and must be removed.
  Time expirationTime;
};

static inline bool
operator == (const TopologyTuple &a, const TopologyTuple &b)
{
  return (a.destAddr == b.destAddr
          && a.lastAddr == b.lastAddr
          && a.sequenceNumber == b.sequenceNumber);
}

static inline std::ostream&
operator << (std::ostream &os, const TopologyTuple &tuple)
{
  os << "TopologyTuple(destAddr=" << tuple.destAddr
     << ", lastAddr=" << tuple.lastAddr
     << ", sequenceNumber=" << (int) tuple.sequenceNumber
     << ", expirationTime=" << tuple.expirationTime
     << ")";
  return os;
}

/// \ingroup olsr
/// Association
struct Association
{
  Ipv4Address networkAddr; //!< IPv4 Network address.
  Ipv4Mask netmask;        //!< IPv4 Network mask.
};

static inline bool
operator == (const Association &a, const Association &b)
{
  return (a.networkAddr == b.networkAddr
          && a.netmask == b.netmask);
}

static inline std::ostream&
operator << (std::ostream &os, const Association &tuple)
{
  os << "Association(networkAddr=" << tuple.networkAddr
     << ", netmask=" << tuple.netmask
     << ")";
  return os;
}

/// \ingroup olsr
/// An Association Tuple
struct AssociationTuple
{
  /// Main address of the gateway.
  Ipv4Address gatewayAddr;
  /// Network Address of network reachable through gatewayAddr
  Ipv4Address networkAddr;
  /// Netmask of network reachable through gatewayAddr
  Ipv4Mask netmask;
  /// Time at which this tuple expires and must be removed
  Time expirationTime;
};

static inline bool
operator == (const AssociationTuple &a, const AssociationTuple &b)
{
  return (a.gatewayAddr == b.gatewayAddr
          && a.networkAddr == b.networkAddr
          && a.netmask == b.netmask);
}

static inline std::ostream&
operator << (std::ostream &os, const AssociationTuple &tuple)
{
  os << "AssociationTuple(gatewayAddr=" << tuple.gatewayAddr
     << ", networkAddr=" << tuple.networkAddr
     << ", netmask=" << tuple.netmask
     << ", expirationTime=" << tuple.expirationTime
     << ")";
  return os;
}


typedef std::set<Ipv4Address>                   MprSet; //!< MPR Set type.
typedef std::vector<MprSelectorTuple>           MprSelectorSet; //!< MPR Selector Set type.
typedef std::vector<LinkTuple>                  LinkSet; //!< Link Set type.
typedef std::vector<NeighborTuple>              NeighborSet; //!< Neighbor Set type.
typedef std::vector<TwoHopNeighborTuple>        TwoHopNeighborSet; //!< 2-hop Neighbor Set type.
typedef std::vector<TopologyTuple>              TopologySet; //!< Topology Set type.
typedef std::vector<DuplicateTuple>             DuplicateSet; //!< Duplicate Set type.
typedef std::vector<IfaceAssocTuple>            IfaceAssocSet; //!< Interface Association Set type.
typedef std::vector<AssociationTuple>           AssociationSet; //!< Association Set type.
typedef std::vector<Association>                Associations; //!< Association Set type.


}
}  // namespace ns3, olsr

#endif /* OLSR_REPOSITORIES_H */
