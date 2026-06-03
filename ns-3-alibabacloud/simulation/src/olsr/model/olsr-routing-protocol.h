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

#ifndef OLSR_AGENT_IMPL_H
#define OLSR_AGENT_IMPL_H

#include "olsr-header.h"
#include "ns3/test.h"
#include "olsr-state.h"
#include "olsr-repositories.h"

#include "ns3/object.h"
#include "ns3/packet.h"
#include "ns3/node.h"
#include "ns3/socket.h"
#include "ns3/event-garbage-collector.h"
#include "ns3/random-variable-stream.h"
#include "ns3/timer.h"
#include "ns3/traced-callback.h"
#include "ns3/ipv4.h"
#include "ns3/ipv4-routing-protocol.h"
#include "ns3/ipv4-static-routing.h"

#include <vector>
#include <map>

/// Testcase for MPR computation mechanism
class OlsrMprTestCase;

namespace ns3 {
namespace olsr {

///
/// \defgroup olsr OLSR Routing
/// This section documents the API of the ns-3 OLSR module. For a generic
/// functional description, please refer to the ns-3 manual.

/// \ingroup olsr
/// An %OLSR's routing table entry.
struct RoutingTableEntry
{
  Ipv4Address destAddr; //!< Address of the destination node.
  Ipv4Address nextAddr; //!< Address of the next hop.
  uint32_t interface; //!< Interface index
  uint32_t distance; //!< Distance in hops to the destination.

  RoutingTableEntry (void) : // default values
    destAddr (), nextAddr (),
    interface (0), distance (0)
  {
  }
};

class RoutingProtocol;

///
/// \ingroup olsr
///
/// \brief OLSR routing protocol for IPv4
///
class RoutingProtocol : public Ipv4RoutingProtocol
{
public:
  /**
   * Declared friend to enable unit tests.
   */
  friend class ::OlsrMprTestCase;

  static const uint16_t OLSR_PORT_NUMBER; //!< port number (698)

  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  RoutingProtocol (void);
  virtual ~RoutingProtocol (void);

  /**
   * \brief Set the OLSR main address to the first address on the indicated interface.
   *
   * \param interface IPv4 interface index
   */
  void SetMainInterface (uint32_t interface);

  /**
   * Dump the neighbor table, two-hop neighbor table, and routing table
   * to logging output (NS_LOG_DEBUG log level).  If logging is disabled,
   * this function does nothing.
   */
  void Dump (void);

  /**
   * Get the routing table entries.
   * \return the list of routing table entries discovered by OLSR
   */
  std::vector<RoutingTableEntry> GetRoutingTableEntries (void) const;

  /**
   * Gets the MPR set.
   * \return The MPR set.
   */
  MprSet GetMprSet (void) const;

  /**
   * Gets the MPR selectors.
   * \returns The MPR selectors.
   */
  const MprSelectorSet & GetMprSelectors (void) const;

  /**
   * Get the one hop neighbors.
   * \return the set of neighbors discovered by OLSR
   */
  const NeighborSet & GetNeighbors (void) const;

  /**
   * Get the two hop neighbors.
   * \return the set of two hop neighbors discovered by OLSR
   */
  const TwoHopNeighborSet & GetTwoHopNeighbors (void) const;

  /**
   * Gets the topology set.
   * \returns The topology set discovery by OLSR
   */
  const TopologySet & GetTopologySet (void) const;

  /**
   * Gets the underlying OLSR state object
   * \returns The OLSR state object
   */
  const OlsrState & GetOlsrState (void) const;

  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model.  Return the number of streams (possibly zero) that
   * have been assigned.
   *
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this model
   */
  int64_t AssignStreams (int64_t stream);

  /**
   * TracedCallback signature for Packet transmit and receive events.
   *
   * \param [in] header
   * \param [in] messages
   */
  typedef void (* PacketTxRxTracedCallback)(const PacketHeader & header, const MessageList & messages);

  /**
   * TracedCallback signature for routing table computation.
   *
   * \param [in] size Final routing table size.
   */
  typedef void (* TableChangeTracedCallback) (uint32_t size);

private:
  std::set<uint32_t> m_interfaceExclusions; //!< Set of interfaces excluded by OSLR.
  Ptr<Ipv4StaticRouting> m_routingTableAssociation; //!< Associations from an Ipv4StaticRouting instance

public:
  /**
   * Get the excluded interfaces.
   * \returns Container of excluded interfaces.
   */
  std::set<uint32_t> GetInterfaceExclusions (void) const
  {
    return m_interfaceExclusions;
  }

  /**
   * Set the interfaces to be excluded.
   * \param exceptions Container of excluded interfaces.
   */
  void SetInterfaceExclusions (std::set<uint32_t> exceptions);

  /**
   *  \brief Injects the specified (networkAddr, netmask) tuple in the list of
   *  local HNA associations to be sent by the node via HNA messages.
   *  If this tuple already exists, nothing is done.
   *
   * \param networkAddr The network address.
   * \param netmask The network mask.
   */
  void AddHostNetworkAssociation (Ipv4Address networkAddr, Ipv4Mask netmask);

  /**
   * \brief Removes the specified (networkAddr, netmask) tuple from the list of
   * local HNA associations to be sent by the node via HNA messages.
   * If this tuple does not exist, nothing is done (see "OlsrState::EraseAssociation()").
   *
   * \param networkAddr The network address.
   * \param netmask The network mask.
   */
  void RemoveHostNetworkAssociation (Ipv4Address networkAddr, Ipv4Mask netmask);

  /**
   * \brief Associates the specified Ipv4StaticRouting routing table
   *         to the OLSR routing protocol. Entries from this associated
   *         routing table that use non-olsr outgoing interfaces are added
   *         to the list of local HNA associations so that they are included
   *         in HNA messages sent by the node.
   *         If this method is called more than once, entries from the old
   *         association are deleted before entries from the new one are added.
   *  \param routingTable the Ipv4StaticRouting routing table to be associated.
   */
  void SetRoutingTableAssociation (Ptr<Ipv4StaticRouting> routingTable);

  /**
   * \brief Returns the internal HNA table
   * \returns the internal HNA table
   */
  Ptr<const Ipv4StaticRouting> GetRoutingTableAssociation (void) const;

protected:
  virtual void DoInitialize (void);
  virtual void DoDispose (void);

private:
  std::map<Ipv4Address, RoutingTableEntry> m_table; //!< Data structure for the routing table.

  Ptr<Ipv4StaticRouting> m_hnaRoutingTable; //!< Routing table for HNA routes

  EventGarbageCollector m_events; //!< Running events.

  uint16_t m_packetSequenceNumber;    //!< Packets sequence number counter.
  uint16_t m_messageSequenceNumber;   //!< Messages sequence number counter.
  uint16_t m_ansn;  //!< Advertised Neighbor Set sequence number.

  Time m_helloInterval;   //!< HELLO messages' emission interval.
  Time m_tcInterval;      //!< TC messages' emission interval.
  Time m_midInterval;     //!< MID messages' emission interval.
  Time m_hnaInterval;     //!< HNA messages' emission interval.
  uint8_t m_willingness;  //!<  Willingness for forwarding packets on behalf of other nodes.

  OlsrState m_state;  //!< Internal state with all needed data structs.
  Ptr<Ipv4> m_ipv4;   //!< IPv4 object the routing is linked to.

  /**
   * \brief Clears the routing table and frees the memory assigned to each one of its entries.
   */
  void Clear (void);

  /**
   * Returns the routing table size.
   * \return The routing table size.
   */
  uint32_t GetSize (void) const
  {
    return m_table.size ();
  }

  /**
   * \brief Deletes the entry whose destination address is given.
   * \param dest address of the destination node.
   */
  void RemoveEntry (const Ipv4Address &dest);
  /**
   * \brief Adds a new entry into the routing table.
   *
   * If an entry for the given destination existed, it is deleted and freed.
   *
   * \param dest address of the destination node.
   * \param next address of the next hop node.
   * \param interface address of the local interface.
   * \param distance distance to the destination node.
   */
  void AddEntry (const Ipv4Address &dest,
                 const Ipv4Address &next,
                 uint32_t interface,
                 uint32_t distance);
  /**
   * \brief Adds a new entry into the routing table.
   *
   * If an entry for the given destination existed, an error is thrown.
   *
   * \param dest address of the destination node.
   * \param next address of the next hop node.
   * \param interfaceAddress address of the local interface.
   * \param distance distance to the destination node.
   */
  void AddEntry (const Ipv4Address &dest,
                 const Ipv4Address &next,
                 const Ipv4Address &interfaceAddress,
                 uint32_t distance);

  /**
   * \brief Looks up an entry for the specified destination address.
   * \param [in] dest Destination address.
   * \param [out] outEntry Holds the routing entry result, if found.
   * \return true if found, false if not found.
   */
  bool Lookup (const Ipv4Address &dest,
               RoutingTableEntry &outEntry) const;

  /**
   * \brief Finds the appropriate entry which must be used in order to forward
   * a data packet to a next hop (given a destination).
   *
   * Imagine a routing table like this: [A,B] [B,C] [C,C]; being each pair of the
   * form [dest addr, next-hop addr]. In this case, if this function is invoked
   * with [A,B] then pair [C,C] is returned because C is the next hop that must be used
   * to forward a data packet destined to A. That is, C is a neighbor of this node,
   * but B isn't. This function finds the appropriate neighbor for forwarding a packet.
   *
   * \param[in] entry The routing table entry which indicates the destination node
   * we are interested in.
   *
   * \param[out] outEntry The appropriate routing table entry which indicates the next
   * hop which must be used for forwarding a data packet, or NULL if there is no such entry.
   *
   * \return True if an entry was found, false otherwise.
   */
  bool FindSendEntry (const RoutingTableEntry &entry,
                      RoutingTableEntry &outEntry) const;

public:
  // From Ipv4RoutingProtocol
  virtual Ptr<Ipv4Route> RouteOutput (Ptr<Packet> p,
                                      const Ipv4Header &header,
                                      Ptr<NetDevice> oif,
                                      Socket::SocketErrno &sockerr);
  virtual bool RouteInput (Ptr<const Packet> p,
                           const Ipv4Header &header,
                           Ptr<const NetDevice> idev,
                           UnicastForwardCallback ucb,
                           MulticastForwardCallback mcb,
                           LocalDeliverCallback lcb,
                           ErrorCallback ecb);
  virtual void SetIpv4 (Ptr<Ipv4> ipv4);

  /**
   * \returns the ipv4 object this routing protocol is associated with
   */
  NS_DEPRECATED_3_34
  virtual Ptr<Ipv4> GetIpv4 (void) const;
  virtual void PrintRoutingTable (Ptr<OutputStreamWrapper> stream, Time::Unit unit = Time::S) const;


private:
  virtual void NotifyInterfaceUp (uint32_t interface);
  virtual void NotifyInterfaceDown (uint32_t interface);
  virtual void NotifyAddAddress (uint32_t interface, Ipv4InterfaceAddress address);
  virtual void NotifyRemoveAddress (uint32_t interface, Ipv4InterfaceAddress address);

  /**
   * Send an OLSR message.
   * \param packet The packet to be sent.
   * \param containedMessages The messages contained in the packet.
   */
  void SendPacket (Ptr<Packet> packet, const MessageList &containedMessages);

  /**
   * Increments packet sequence number and returns the new value.
   * \return The packet sequence number.
   */
  inline uint16_t GetPacketSequenceNumber (void);

  /**
   * Increments message sequence number and returns the new value.
   * \return The message sequence number.
   */
  inline uint16_t GetMessageSequenceNumber (void);

  /**
   * Receive an OLSR message.
   * \param socket The receiving socket.
   */
  void RecvOlsr (Ptr<Socket> socket);

  /**
   * \brief Computates MPR set of a node following \RFC{3626} hints.
   */
  void MprComputation (void);

  /**
   * \brief Creates the routing table of the node following \RFC{3626} hints.
   */
  void RoutingTableComputation (void);

public:
  /**
   * \brief Gets the main address associated with a given interface address.
   * \param iface_addr the interface address.
   * \return the corresponding main address.
   */
  Ipv4Address GetMainAddress (Ipv4Address iface_addr) const;

private:
  /**
   *  \brief Tests whether or not the specified route uses a non-OLSR outgoing interface.
   *  \param route The route to be tested.
   *  \returns True if the outgoing interface of the specified route is a non-OLSR interface, false otherwise.
   */
  bool UsesNonOlsrOutgoingInterface (const Ipv4RoutingTableEntry &route);

  // Timer handlers
  Timer m_helloTimer; //!< Timer for the HELLO message.
  /**
   * \brief Sends a HELLO message and reschedules the HELLO timer.
   */
  void HelloTimerExpire (void);

  Timer m_tcTimer; //!< Timer for the TC message.
  /**
   * \brief Sends a TC message (if there exists any MPR selector) and reschedules the TC timer.
   */
  void TcTimerExpire (void);

  Timer m_midTimer; //!< Timer for the MID message.
  /**
   * \brief \brief Sends a MID message (if the node has more than one interface) and resets the MID timer.
   */
  void MidTimerExpire (void);

  Timer m_hnaTimer; //!< Timer for the HNA message.
  /**
   * \brief Sends an HNA message (if the node has associated hosts/networks) and reschedules the HNA timer.
   */
  void HnaTimerExpire (void);

  /**
   * \brief Removes tuple if expired. Else timer is rescheduled to expire at tuple.expirationTime.
   *
   * The task of actually removing the tuple is left to the OLSR agent.
   *
   * \param address The address of the tuple.
   * \param sequenceNumber The sequence number of the tuple.
   */
  void DupTupleTimerExpire (Ipv4Address address, uint16_t sequenceNumber);

  bool m_linkTupleTimerFirstTime; //!< Flag to indicate if it is the first time the LinkTupleTimer fires.
  /**
   * \brief Removes tuple_ if expired. Else if symmetric time
   * has expired then it is assumed a neighbor loss and agent_->nb_loss()
   * is called. In this case the timer is rescheduled to expire at
   * tuple_->time(). Otherwise the timer is rescheduled to expire at
   * the minimum between tuple_->time() and tuple_->sym_time().
   *
   * The task of actually removing the tuple is left to the OLSR agent.
   *
   * \param neighborIfaceAddr The tuple neighbor interface address.
   */
  void LinkTupleTimerExpire (Ipv4Address neighborIfaceAddr);

  /**
   * \brief Removes 2_hop neighbor tuple_ if expired. Else the timer is rescheduled to expire at tuple_->time().
   *
   * The task of actually removing the tuple is left to the OLSR agent.
   *
   * \param neighborMainAddr The neighbor main address.
   * \param twoHopNeighborAddr The 2-hop neighbor address.
   */
  void Nb2hopTupleTimerExpire (Ipv4Address neighborMainAddr, Ipv4Address twoHopNeighborAddr);

  /**
   * \brief Removes MPR selector tuple_ if expired. Else the timer is rescheduled to expire at tuple_->time().
   *
   * The task of actually removing the tuple is left to the OLSR agent.
   *
   * \param mainAddr The tuple IPv4 address.
   */
  void MprSelTupleTimerExpire (Ipv4Address mainAddr);

  /**
   * \brief Removes topology tuple_ if expired. Else the timer is rescheduled to expire at tuple_->time().
   *
   * The task of actually removing the tuple is left to the OLSR agent.
   *
   * \param destAddr The destination address.
   * \param lastAddr The last address.
   */
  void TopologyTupleTimerExpire (Ipv4Address destAddr, Ipv4Address lastAddr);

  /**
   * \brief Removes interface association tuple_ if expired. Else the timer is rescheduled to expire at tuple_->time().
   *
   * \param ifaceAddr The interface address.
   */
  void IfaceAssocTupleTimerExpire (Ipv4Address ifaceAddr);

  /**
   * \brief Removes association tuple_ if expired. Else timer is rescheduled to expire at tuple_->time().
   *
   * \param gatewayAddr The gateway address.
   * \param networkAddr The network address.
   * \param netmask  The network mask.
   */
  void AssociationTupleTimerExpire (Ipv4Address gatewayAddr, Ipv4Address networkAddr, Ipv4Mask netmask);

  /**
   * Increments the ANSN counter.
   */
  void IncrementAnsn (void);

  /// A list of pending messages which are buffered awaiting for being sent.
  olsr::MessageList m_queuedMessages;
  Timer m_queuedMessagesTimer; //!< timer for throttling outgoing messages

  /**
   * \brief OLSR's default forwarding algorithm.
   *
   * See \RFC{3626} for details.
   *
   * \param olsrMessage The %OLSR message which must be forwarded.
   * \param duplicated NULL if the message has never been considered for forwarding, or a duplicate tuple in other case.
   * \param localIface The address of the interface where the message was received from.
   * \param senderAddress The sender IPv4 address.
   */
  void ForwardDefault (olsr::MessageHeader olsrMessage,
                       DuplicateTuple *duplicated,
                       const Ipv4Address &localIface,
                       const Ipv4Address &senderAddress);

  /**
   * \brief Enques an %OLSR message which will be sent with a delay of (0, delay].
   *
   * This buffering system is used in order to piggyback several %OLSR messages in
   * a same %OLSR packet.
   *
   * \param message the %OLSR message which must be sent.
   * \param delay maximum delay the %OLSR message is going to be buffered.
   */
  void QueueMessage (const olsr::MessageHeader &message, Time delay);

  /**
   * \brief Creates as many %OLSR packets as needed in order to send all buffered
   * %OLSR messages.
   *
   * Maximum number of messages which can be contained in an %OLSR packet is
   * dictated by OLSR_MAX_MSGS constant.
   */
  void SendQueuedMessages (void);

  /**
   * \brief Creates a new %OLSR HELLO message which is buffered for being sent later on.
   */
  void SendHello (void);

  /**
   * \brief Creates a new %OLSR TC message which is buffered for being sent later on.
   */
  void SendTc (void);

  /**
   * \brief Creates a new %OLSR MID message which is buffered for being sent later on.
   */
  void SendMid (void);

  /**
   * \brief Creates a new %OLSR HNA message which is buffered for being sent later on.
   */
  void SendHna (void);

  /**
   * \brief Performs all actions needed when a neighbor loss occurs.
   *
   * Neighbor Set, 2-hop Neighbor Set, MPR Set and MPR Selector Set are updated.
   *
   * \param tuple link tuple with the information of the link to the neighbor which has been lost.
   */
  void NeighborLoss (const LinkTuple &tuple);

  /**
   * \brief Adds a duplicate tuple to the Duplicate Set.
   *
   * \param tuple The duplicate tuple to be added.
   */
  void AddDuplicateTuple (const DuplicateTuple &tuple);

  /**
   * \brief Removes a duplicate tuple from the Duplicate Set.
   *
   * \param tuple The duplicate tuple to be removed.
   */
  void RemoveDuplicateTuple (const DuplicateTuple &tuple);

  /**
   * Adds a link tuple.
   * \param tuple Thetuple to be added.
   * \param willingness The tuple willingness.
   */
  void LinkTupleAdded (const LinkTuple &tuple, uint8_t willingness);

  /**
   * \brief Removes a link tuple from the Link Set.
   *
   * \param tuple The link tuple to be removed.
   */
  void RemoveLinkTuple (const LinkTuple &tuple);

  /**
   * \brief This function is invoked when a link tuple is updated. Its aim is to
   * also update the corresponding neighbor tuple if it is needed.
   *
   * \param tuple The link tuple which has been updated.
   * \param willingness The tuple willingness.
   */
  void LinkTupleUpdated (const LinkTuple &tuple, uint8_t willingness);

  /**
   * \brief Adds a neighbor tuple to the Neighbor Set.
   *
   * \param tuple The neighbor tuple to be added.
   */
  void AddNeighborTuple (const NeighborTuple &tuple);

  /**
   * \brief Removes a neighbor tuple from the Neighbor Set.
   *
   * \param tuple The neighbor tuple to be removed.
   */
  void RemoveNeighborTuple (const NeighborTuple &tuple);

  /**
   * \brief Adds a 2-hop neighbor tuple to the 2-hop Neighbor Set.
   *
   * \param tuple The 2-hop neighbor tuple to be added.
   */
  void AddTwoHopNeighborTuple (const TwoHopNeighborTuple &tuple);

  /**
   * \brief Removes a 2-hop neighbor tuple from the 2-hop Neighbor Set.
   *
   * \param tuple The 2-hop neighbor tuple to be removed.
   */
  void RemoveTwoHopNeighborTuple (const TwoHopNeighborTuple &tuple);

  /**
   * \brief Adds an MPR selector tuple to the MPR Selector Set.
   * Advertised Neighbor Sequence Number (ANSN) is also updated.
   *
   * \param tuple The MPR selector tuple to be added.
   */
  void AddMprSelectorTuple (const MprSelectorTuple  &tuple);

  /**
   * \brief Removes an MPR selector tuple from the MPR Selector Set.
   * Advertised Neighbor Sequence Number (ANSN) is also updated.
   *
   * \param tuple The MPR selector tuple to be removed.
   */
  void RemoveMprSelectorTuple (const MprSelectorTuple &tuple);

  /**
   * \brief Adds a topology tuple to the Topology Set.
   *
   * \param tuple The topology tuple to be added.
   */
  void AddTopologyTuple (const TopologyTuple &tuple);

  /**
   * \brief Removes a topology tuple to the Topology Set.
   *
   * \param tuple The topology tuple to be removed.
   */
  void RemoveTopologyTuple (const TopologyTuple &tuple);

  /**
   * \brief Adds an interface association tuple to the Interface Association Set.
   *
   * \param tuple The interface association tuple to be added.
   */
  void AddIfaceAssocTuple (const IfaceAssocTuple &tuple);

  /**
   * \brief Removed an interface association tuple to the Interface Association Set.
   *
   * \param tuple The interface association tuple to be removed.
   */
  void RemoveIfaceAssocTuple (const IfaceAssocTuple &tuple);

  /**
   * \brief Adds a host network association tuple to the Association Set.
   *
   * \param tuple The host network association tuple to be added.
   */
  void AddAssociationTuple (const AssociationTuple &tuple);

  /**
   * \brief Removes a host network association tuple to the Association Set.
   *
   * \param tuple The host network association tuple to be removed.
   */
  void RemoveAssociationTuple (const AssociationTuple &tuple);

  /**
   * \brief Processes a HELLO message following \RFC{3626} specification.
   *
   * Link sensing and population of the Neighbor Set, 2-hop Neighbor Set and MPR
   * Selector Set are performed.
   *
   * \param msg the %OLSR message which contains the HELLO message.
   * \param receiverIface the address of the interface where the message was received from.
   * \param senderIface the address of the interface where the message was sent from.
   */
  void ProcessHello (const olsr::MessageHeader &msg,
                     const Ipv4Address &receiverIface,
                     const Ipv4Address &senderIface);

  /**
   * \brief Processes a TC message following \RFC{3626} specification.
   *
   * The Topology Set is updated (if needed) with the information of
   * the received TC message.
   *
   * \param msg The %OLSR message which contains the TC message.
   * \param senderIface The address of the interface where the message was sent from.
   *
   */
  void ProcessTc (const olsr::MessageHeader &msg,
                  const Ipv4Address &senderIface);

  /**
   * \brief Processes a MID message following \RFC{3626} specification.
   *
   * The Interface Association Set is updated (if needed) with the information
   * of the received MID message.
   *
   * \param msg the %OLSR message which contains the MID message.
   * \param senderIface the address of the interface where the message was sent from.
   */
  void ProcessMid (const olsr::MessageHeader &msg,
                   const Ipv4Address &senderIface);

  /**
   *
   * \brief Processes a HNA message following \RFC{3626} specification.
   *
   * The Host Network Association Set is updated (if needed) with the information
   * of the received HNA message.
   *
   * \param msg the %OLSR message which contains the HNA message.
   * \param senderIface the address of the interface where the message was sent from.
   *
   */
  void ProcessHna (const olsr::MessageHeader &msg,
                   const Ipv4Address &senderIface);

  /**
   * \brief Updates Link Set according to a new received HELLO message
   * (following \RFC{3626} specification). Neighbor Set is also updated if needed.
   * \param msg The received message.
   * \param hello The received HELLO sub-message.
   * \param receiverIface The interface that received the message.
   * \param senderIface The sender interface.
   */
  void LinkSensing (const olsr::MessageHeader &msg,
                    const olsr::MessageHeader::Hello &hello,
                    const Ipv4Address &receiverIface,
                    const Ipv4Address &senderIface);

  /**
   * \brief Updates the Neighbor Set according to the information contained in
   * a new received HELLO message (following \RFC{3626}).
   * \param msg The received message.
   * \param hello The received HELLO sub-message.
   */
  void PopulateNeighborSet (const olsr::MessageHeader &msg,
                            const olsr::MessageHeader::Hello &hello);

  /**
   * \brief Updates the 2-hop Neighbor Set according to the information contained
   * in a new received HELLO message (following \RFC{3626}).
   * \param msg The received message.
   * \param hello The received HELLO sub-message.
   */
  void PopulateTwoHopNeighborSet (const olsr::MessageHeader &msg,
                                  const olsr::MessageHeader::Hello &hello);

  /**
   * \brief Updates the MPR Selector Set according to the information contained in
   * a new received HELLO message (following \RFC{3626}).
   * \param msg The received message.
   * \param hello The received HELLO sub-message.
   */
  void PopulateMprSelectorSet (const olsr::MessageHeader &msg,
                               const olsr::MessageHeader::Hello &hello);

  int Degree (NeighborTuple const &tuple);

  /**
   *  Check that address is one of my interfaces.
   *  \param a the address to check.
   *  \return true if the address is own by the node.
   */
  bool IsMyOwnAddress (const Ipv4Address & a) const;

  Ipv4Address m_mainAddress; //!< the node main address.

  // One socket per interface, each bound to that interface's address
  // (reason: for OLSR Link Sensing we need to know on which interface
  // HELLO messages arrive)
  std::map< Ptr<Socket>, Ipv4InterfaceAddress > m_sendSockets; //!< Container of sockets and the interfaces they are opened onto.
  Ptr<Socket> m_recvSocket; //!< Receiving socket.


  /// Rx packet trace.
  TracedCallback <const PacketHeader &, const MessageList &> m_rxPacketTrace;

  /// Tx packet trace.
  TracedCallback <const PacketHeader &, const MessageList &> m_txPacketTrace;

  /// Routing table chanes challback
  TracedCallback <uint32_t> m_routingTableChanged;

  /// Provides uniform random variables.
  Ptr<UniformRandomVariable> m_uniformRandomVariable;

};

}
}  // namespace ns3

#endif /* OLSR_AGENT_IMPL_H */
