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

#ifndef IPV6_FLOW_CLASSIFIER_H
#define IPV6_FLOW_CLASSIFIER_H

#include <stdint.h>
#include <map>
#include <atomic>

#include "ns3/ipv6-header.h"
#include "ns3/flow-classifier.h"

namespace ns3 {

class Packet;

/// Classifies packets by looking at their IP and TCP/UDP headers.
/// From these packet headers, a tuple (source-ip, destination-ip,
/// protocol, source-port, destination-port) is created, and a unique
/// flow identifier is assigned for each different tuple combination
class Ipv6FlowClassifier : public FlowClassifier
{
public:

  /// Structure to classify a packet
  struct FiveTuple
  {
    Ipv6Address sourceAddress;      //!< Source address
    Ipv6Address destinationAddress; //!< Destination address
    uint8_t protocol;               //!< Protocol
    uint16_t sourcePort;            //!< Source port
    uint16_t destinationPort;       //!< Destination port
  };

  Ipv6FlowClassifier ();

  /// \brief try to classify the packet into flow-id and packet-id
  ///
  /// \warning: it must be called only once per packet, from SendOutgoingLogger.
  ///
  /// \return true if the packet was classified, false if not (i.e. it
  /// does not appear to be part of a flow).
  /// \param ipHeader packet's IP header
  /// \param ipPayload packet's IP payload
  /// \param out_flowId packet's FlowId
  /// \param out_packetId packet's identifier
  bool Classify (const Ipv6Header &ipHeader, Ptr<const Packet> ipPayload,
                 uint32_t *out_flowId, uint32_t *out_packetId);

  /// Searches for the FiveTuple corresponding to the given flowId
  /// \param flowId the FlowId to search for
  /// \returns the FiveTuple corresponding to flowId
  FiveTuple FindFlow (FlowId flowId) const;

  /// Comparator used to sort the vector of DSCP values
  class SortByCount
  {
  public:
    /// Comparator function
    /// \param left left operand
    /// \param right right operand
    /// \return true if left DSCP is greater than right DSCP
    bool operator() (std::pair<Ipv6Header::DscpType, uint32_t> left,
                     std::pair<Ipv6Header::DscpType, uint32_t> right);
  };

  /// \brief get the DSCP values of the packets belonging to the flow with the
  /// given FlowId, sorted in decreasing order of number of packets seen with
  /// that DSCP value
  /// \param flowId the identifier of the flow of interest
  /// \returns the vector of DSCP values
  std::vector<std::pair<Ipv6Header::DscpType, uint32_t> > GetDscpCounts (FlowId flowId) const;

  virtual void SerializeToXmlStream (std::ostream &os, uint16_t indent) const;

private:

  /// Map to Flows Identifiers to FlowIds
  std::map<FiveTuple, FlowId> m_flowMap;
  /// Map to FlowIds to FlowPacketId
  std::map<FlowId, FlowPacketId> m_flowPktIdMap;
  /// Map FlowIds to (DSCP value, packet count) pairs
  std::map<FlowId, std::map<Ipv6Header::DscpType, uint32_t> > m_flowDscpMap;

#ifdef NS3_MTP
  std::atomic<bool> m_lock;
#endif
};

/**
 * \brief Less than operator.
 *
 * \param t1 the first operand
 * \param t2 the first operand
 * \returns true if the operands are equal
 */
bool operator < (const Ipv6FlowClassifier::FiveTuple &t1, const Ipv6FlowClassifier::FiveTuple &t2);

/**
 * \brief Equal to operator.
 *
 * \param t1 the first operand
 * \param t2 the first operand
 * \returns true if the operands are equal
 */
bool operator == (const Ipv6FlowClassifier::FiveTuple &t1, const Ipv6FlowClassifier::FiveTuple &t2);


} // namespace ns3

#endif /* IPV6_FLOW_CLASSIFIER_H */
