/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Yufei Cheng
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
 * Author: Yufei Cheng   <yfcheng@ittc.ku.edu>
 *
 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  http://wiki.ittc.ku.edu/resilinets
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 *
 * Work supported in part by NSF FIND (Future Internet Design) Program
 * under grant CNS-0626918 (Postmodern Internet Architecture),
 * NSF grant CNS-1050226 (Multilayer Network Resilience Analysis and Experimentation on GENI),
 * US Department of Defense (DoD), and ITTC at The University of Kansas.
 */

#ifndef DSR_ROUTING_H
#define DSR_ROUTING_H

#include <map>
#include <list>
#include <vector>
#include <stdint.h>
#include <cassert>
#include <sys/types.h>

#include "ns3/callback.h"
#include "ns3/object.h"
#include "ns3/node.h"
#include "ns3/ptr.h"
#include "ns3/buffer.h"
#include "ns3/packet.h"
#include "ns3/ipv4.h"
#include "ns3/ip-l4-protocol.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/icmpv4-l4-protocol.h"
#include "ns3/ipv4-interface.h"
#include "ns3/ipv4-header.h"
#include "ns3/ipv4-address.h"
#include "ns3/traced-callback.h"
#include "ns3/random-variable-stream.h"
#include "ns3/ipv4-route.h"
#include "ns3/timer.h"
#include "ns3/net-device.h"
#include "ns3/output-stream-wrapper.h"
#include "ns3/wifi-mac.h"
#include "ns3/socket.h"
#include "ns3/event-garbage-collector.h"
#include "ns3/test.h"

#include "dsr-network-queue.h"
#include "dsr-rcache.h"
#include "dsr-rreq-table.h"
#include "dsr-maintain-buff.h"
#include "dsr-passive-buff.h"
#include "dsr-option-header.h"
#include "dsr-fs-header.h"
#include "dsr-rsendbuff.h"
#include "dsr-errorbuff.h"
#include "dsr-gratuitous-reply-table.h"

namespace ns3 {

class Packet;
class Node;
class Ipv4;
class Ipv4Address;
class Ipv4Header;
class Ipv4Interface;
class Ipv4L3Protocol;
class Time;

namespace dsr {

class DsrOptions;
/**
 * \class DsrRouting
 * \brief Dsr Routing base
 */
class DsrRouting : public IpL4Protocol
{
public:
  /**
   * \brief Get the type identificator.
   * \return type identificator
   */
  static TypeId GetTypeId ();
  /**
    * \brief Define the dsr protocol number.
    */
  static const uint8_t PROT_NUMBER;
  /**
   * \brief Constructor.
   */
  DsrRouting ();
  /**
   * \brief Destructor.
   */
  virtual ~DsrRouting ();
  /**
   * \brief Get the node.
   * \return the node
   */
  Ptr<Node> GetNode () const;
  /**
   * \brief Set the node.
   * \param node the node to set
   */
  void SetNode (Ptr<Node> node);
  /**
   * \brief Set the route cache.
   * \param r the route cache to set
   */
  void SetRouteCache (Ptr<dsr::DsrRouteCache> r);
  /**
   * \brief Get the route cache.
   * \return the route cache
   */
  Ptr<dsr::DsrRouteCache> GetRouteCache () const;
  /**
   * \brief Set the node.
   * \param r the request table to set
   */
  void SetRequestTable (Ptr<dsr::DsrRreqTable> r);
  /**
    * \brief Get the request table.
    * \return the request table
    */
  Ptr<dsr::DsrRreqTable> GetRequestTable () const;
  /**
   * \brief Set the node.
   * \param r the passive buffer to set
   */
  void SetPassiveBuffer (Ptr<dsr::DsrPassiveBuffer> r);
  /**
    * \brief Get the passive buffer
    * \return the passive buffer
    */
  Ptr<dsr::DsrPassiveBuffer> GetPassiveBuffer () const;

  /**
   * \brief Checks if the link is cached in the route cache
   * See also DsrRouteCache::IsLinkCache
   *
   * \return true if the link is cached
   */
  bool IsLinkCache ();

  /**
   * \brief Extends the lifetime of a route cache entry.
   * See also DsrRouteCache::UseExtends
   *
   * \param rt the route to extend
   */
  void UseExtends (DsrRouteCacheEntry::IP_VECTOR rt);

  /**
   * \brief Lookup route cache entry with destination address dst
   * See also DsrRouteCache::LookupRoute
   *
   * \param id destination address
   * \param rt entry with destination address id, if exists
   * \return true on success
   */
  bool LookupRoute (Ipv4Address id, DsrRouteCacheEntry & rt);

  /**
   * \brief dd route link to cache
   * See also DsrRouteCache::AddRoute_Link
   *
   * \param nodelist vector of nodes
   * \param source ip address of node to add
   * \return true if the link is cached
   */
  bool AddRoute_Link (DsrRouteCacheEntry::IP_VECTOR nodelist, Ipv4Address source);

  /**
   * \brief Add route cache entry if it doesn't yet exist in route cache
   * See also DsrRouteCache::AddRoute
   *
   * \param rt route cache entry
   * \return true on success
   */
  bool AddRoute (DsrRouteCacheEntry & rt);

  /**
   * \brief Delete all the routes which includes the link from next hop address that has just been notified
   * as unreachable.
   * See also DsrRouteCache::DeleteAllRoutesIncludeLink
   *
   * \param errorSrc The error source address
   * \param unreachNode The unreachable node
   * \param node This node's ip address
   */
  void DeleteAllRoutesIncludeLink (Ipv4Address errorSrc, Ipv4Address unreachNode, Ipv4Address node);

  /**
   * \brief Update route cache entry if it has been recently used and successfully delivered the data packet.
   * See also DsrRouteCache::UpdateRouteEntry
   *
   * \param dst destination address of the route
   * \return true in success
   */
  bool UpdateRouteEntry (Ipv4Address dst);

  /**
   * Find the source request entry in the route request queue, return false if not found.
   * See also DsrRreqTable::FindSourceEntry
   *
   * \param src the source address we just received the source request
   * \param dst the destination address the request is targeted at
   * \param id the identification number for this request
   * \return true if found, false otherwise
   */
  bool FindSourceEntry (Ipv4Address src, Ipv4Address dst, uint16_t id);

  /**
   * \brief Get the netdevice from the context.
   * \param context context
   * \return the netdevice we are looking for
   */
  Ptr<NetDevice> GetNetDeviceFromContext (std::string context);
  /**
   * \brief Get the elements from the tracing context.
   * \param context context
   * \return the elements we are looking for
   */
  std::vector<std::string> GetElementsFromContext (std::string context);
  /**
   * \brief Get the node id from ip address.
   * \param address IPv4 address
   * \return the node id
   */
  uint16_t GetIDfromIP (Ipv4Address address);
  /**
   * \brief Get the ip address from id.
   * \param id unique ID
   * \return the ip address for the id
   */
  Ipv4Address GetIPfromID (uint16_t id);
  /**
   * \brief Get the Ip address from mac address.
   * \param address Mac48Address
   * \return the ip address
   */
  Ipv4Address GetIPfromMAC (Mac48Address address);
  /**
   * \brief Get the node with give ip address.
   * \param ipv4Address IPv4 address
   * \return the node associated with the ip address
   */
  Ptr<Node> GetNodeWithAddress (Ipv4Address ipv4Address);
  /**
   * \brief Print the route vector.
   * \param vec the vector to print.
   */
  void PrintVector (std::vector<Ipv4Address>& vec);
  /**
   * \brief Get the next hop of the route.
   * \param ipv4Address
   * \param vec Route
   * \return the next hop address of the route
   */
  Ipv4Address SearchNextHop (Ipv4Address ipv4Address, std::vector<Ipv4Address>& vec);
  /**
   * \brief Get the dsr protocol number.
   * \return protocol number
   */
  int GetProtocolNumber (void) const;
  /**
   * \brief The send buffer timer expire.
   */
  void SendBuffTimerExpire ();
  /**
   * \brief Check the send buffer of packets with route when send buffer timer expire.
   */
  void CheckSendBuffer ();
  /**
   * \brief When route vector corrupted, originate a new packet, normally not happening.
   * \param packet to route
   * \param source address
   * \param destination address
   * \param protocol number
   */
  void PacketNewRoute (Ptr<Packet> packet,
                       Ipv4Address source,
                       Ipv4Address destination,
                       uint8_t protocol);
  /**
   * \brief Set the route to use for data packets,
   *        used by the option headers when sending data/control packets
   * \param nextHop next hop IPv4 address
   * \param srcAddress IPv4 address of the source
   * \return the route
   */
  Ptr<Ipv4Route> SetRoute (Ipv4Address nextHop, Ipv4Address srcAddress);
  /**
   * \brief Set the priority of the packet in network queue
   * \param messageType Message type
   * \return the priority value
   */
  uint32_t GetPriority (DsrMessageType messageType);
  /**
   * \brief This function is responsible for sending error packets in case of break link to next hop
   * \param unreachNode unreachable node
   * \param destination address
   * \param originalDst address
   * \param salvage packet flag
   * \param protocol number
   */
  void SendUnreachError (Ipv4Address unreachNode, Ipv4Address destination, Ipv4Address originalDst, uint8_t salvage, uint8_t protocol);  /**
   * \brief This function is responsible for forwarding error packets along the route
   * \param rerr unreachable header
   * \param sourceRoute source routing header
   * \param nextHop IP address of next hop
   * \param protocol number
   * \param route IP route
   */
  void ForwardErrPacket (DsrOptionRerrUnreachHeader &rerr,
                         DsrOptionSRHeader &sourceRoute,
                         Ipv4Address nextHop,
                         uint8_t protocol,
                         Ptr<Ipv4Route> route);
  /**
   * \brief This function is called by higher layer protocol when sending packets
   * \param packet to send
   * \param source IP address
   * \param destination IP address
   * \param protocol number
   * \param route IP route
   */
  void Send (Ptr<Packet> packet, Ipv4Address source,
             Ipv4Address destination, uint8_t protocol, Ptr<Ipv4Route> route);
  /**
   * \brief This function is called to add ack request header for network acknowledgement
   * \param packet for ack req
   * \param nextHop IP address of the next hop
   * \return ack ID
   */
  uint16_t AddAckReqHeader (Ptr<Packet> &packet, Ipv4Address nextHop);
  /**
   * \brief This function is called by when really sending out the packet
   * \param packet to send
   * \param source IP address
   * \param nextHop IP address
   * \param protocol number
   */
  void SendPacket (Ptr<Packet> packet, Ipv4Address source, Ipv4Address nextHop, uint8_t protocol);
  /**
   * \brief This function is called to schedule sending packets from the network queue
   * \param priority for sending
   */
  void Scheduler (uint32_t priority);
  /**
   * \brief This function is called to schedule sending packets from the network queue by priority
   * \param priority schedule
   * \param continueWithFirst use all priorities
   */
  void PriorityScheduler (uint32_t priority, bool continueWithFirst);
  /**
   * \brief This function is called to increase the retransmission timer for data packet in the network queue
   */
  void IncreaseRetransTimer ();
  /**
   * \brief This function is called to send packets down stack
   * \param newEntry queue entry
   * \return true if success
   */
  bool SendRealDown (DsrNetworkQueueEntry & newEntry);
  /**
   * \brief This function is responsible for sending out data packets when have route, if no route found, it will
   * cache the packet and send out route requests
   * \param sourceRoute source route
   * \param nextHop next hop IP address
   * \param protocol number
   */
  void SendPacketFromBuffer (DsrOptionSRHeader const &sourceRoute,
                             Ipv4Address nextHop,
                             uint8_t protocol);
  /**
   * \brief Find the same passive entry
   * \param packet to process
   * \param source IP address
   * \param destination IP address
   * \param segsLeft segments left
   * \param fragmentOffset
   * \param identification
   * \param saveEntry
   * \return true if passive buffer entry
   */
  bool PassiveEntryCheck (Ptr<Packet> packet, Ipv4Address source, Ipv4Address destination, uint8_t segsLeft,
                          uint16_t fragmentOffset, uint16_t identification, bool saveEntry);

  /**
  * \brief Cancel all the packet timers
  * \param mb maintain buffer entry
  */
  void CancelPacketAllTimer (DsrMaintainBuffEntry & mb);
  /**
   * \brief Cancel the passive timer
   * \param packet to process
   * \param source IP address
   * \param destination IP address
   * \param segsLeft segments left
   * \return
   */
  bool CancelPassiveTimer (Ptr<Packet> packet, Ipv4Address source, Ipv4Address destination, uint8_t segsLeft);
  /**
   * \brief Call the cancel packet retransmission timer function
   * \param ackId acknowledge ID
   * \param ipv4Header header
   * \param realSrc source IP address
   * \param realDst destination IP address
   */
  void CallCancelPacketTimer (uint16_t ackId, Ipv4Header const& ipv4Header, Ipv4Address realSrc, Ipv4Address realDst);
  /**
   * \brief Cancel the network packet retransmission timer for a specific maintenance entry
   * \param mb maintian byffer entry
   */
  void CancelNetworkPacketTimer (DsrMaintainBuffEntry & mb);
  /**
   * \brief Cancel the passive packet retransmission timer for a specific maintenance entry
   * \param mb maintian byffer entry
   */
  void CancelPassivePacketTimer (DsrMaintainBuffEntry & mb);
  /**
   * \brief Cancel the link packet retransmission timer for a specific maintenance entry
   * \param mb maintian byffer entry
   */
  void CancelLinkPacketTimer (DsrMaintainBuffEntry & mb);
  /**
   * \brief Cancel the packet retransmission timer for a all maintenance entries with nextHop address
   * \param nextHop next hop IP address
   * \param protocol number
   */
  void CancelPacketTimerNextHop (Ipv4Address nextHop, uint8_t protocol);
  /**
   * \brief Salvage the packet which has been transmitted for 3 times
   * \param packet to process
   * \param source IP address
   * \param dst destination IP address
   * \param protocol number
   */
  void SalvagePacket (Ptr<const Packet> packet, Ipv4Address source, Ipv4Address dst, uint8_t protocol);
  /**
   * \brief Schedule the packet retransmission based on link-layer acknowledgment
   * \param mb maintenance buffer entry
   * \param protocol the protocol number
   */
  void ScheduleLinkPacketRetry   (DsrMaintainBuffEntry & mb,
                                  uint8_t protocol);
  /**
   * \brief Schedule the packet retransmission based on passive acknowledgment
   * \param mb maintenance buffer entry
   * \param protocol the protocol number
   */
  void SchedulePassivePacketRetry   (DsrMaintainBuffEntry & mb,
                                     uint8_t protocol);
  /**
   * \brief Schedule the packet retransmission based on network layer acknowledgment
   * \param mb maintenance buffer entry
   * \param isFirst see if this is the first packet retry or not
   * \param protocol the protocol number
   */
  void ScheduleNetworkPacketRetry   (DsrMaintainBuffEntry & mb,
                                     bool isFirst,
                                     uint8_t protocol);
  /**
   * \brief This function deals with packet retransmission timer expire using link acknowledgment
   * \param mb maintenance buffer entry
   * \param protocol the protocol number
   */
  void LinkScheduleTimerExpire  (DsrMaintainBuffEntry & mb,
                                 uint8_t protocol);
  /**
   * \brief This function deals with packet retransmission timer expire using network acknowledgment
   * \param mb maintenance buffer entry
   * \param protocol the protocol number
   */
  void NetworkScheduleTimerExpire  (DsrMaintainBuffEntry & mb,
                                    uint8_t protocol);
  /**
   * \brief This function deals with packet retransmission timer expire using passive acknowledgment
   * \param mb maintenance buffer entry
   * \param protocol the protocol number
   */
  void PassiveScheduleTimerExpire  (DsrMaintainBuffEntry & mb,
                                    uint8_t protocol);
  /**
   * \brief Forward the packet using the route saved in the source route option header
   * \param packet The packet
   * \param sourceRoute Source route saved in option header
   * \param ipv4Header IPv4 Header
   * \param source source address
   * \param destination destination address
   * \param targetAddress target address
   * \param protocol protocol number
   * \param route route
   */
  void ForwardPacket (Ptr<const Packet> packet,
                      DsrOptionSRHeader &sourceRoute,
                      Ipv4Header const& ipv4Header,
                      Ipv4Address source,
                      Ipv4Address destination,
                      Ipv4Address targetAddress,
                      uint8_t protocol,
                      Ptr<Ipv4Route> route);
  /**
   * \brief Broadcast the route request packet in subnet
   * \param source source address
   * \param destination destination address
   * \param protocol protocol number
   */
  void SendInitialRequest (Ipv4Address source,
                           Ipv4Address destination,
                           uint8_t protocol);
  /**
   * \brief Send the error request packet
   * \param rerr the route error header
   * \param protocol the protocol number
   */
  void SendErrorRequest (DsrOptionRerrUnreachHeader &rerr, uint8_t protocol);
  /**
   * \brief Forward the route request if the node is not the destination
   * \param packet the original packet
   * \param source address
   */
  void SendRequest (Ptr<Packet> packet,
                    Ipv4Address source);
  /**
   * \brief Schedule the intermediate route request
   * \param packet the original packet
   */
  void ScheduleInterRequest (Ptr<Packet> packet);
  /**
   * \brief Send the gratuitous reply
   * \param replyTo The destination address to send the reply to
   * \param replyFrom The source address sending the reply
   * \param nodeList Route
   * \param protocol the protocol number
   */
  void SendGratuitousReply (Ipv4Address replyTo,
                            Ipv4Address replyFrom,
                            std::vector<Ipv4Address> &nodeList,
                            uint8_t protocol);
  /**
   * Send the route reply back to the request originator with the cumulated route
   *
   * \param packet the original packet
   * \param source IPv4 address of the source (i.e. request originator)
   * \param nextHop IPv4 address of the next hop
   * \param route Route
   */
  void SendReply (Ptr<Packet> packet,
                  Ipv4Address source,
                  Ipv4Address nextHop,
                  Ptr<Ipv4Route> route);
  /**
   * this is a generating the initial route reply from the destination address, a random delay time
   * [0, m_broadcastJitter] is used before unicasting back the route reply packet
   *
   * \param packet the original packet
   * \param source IPv4 address of the source (i.e. request originator)
   * \param nextHop IPv4 address of the next hop
   * \param route Route
   */
  void ScheduleInitialReply (Ptr<Packet> packet,
                             Ipv4Address source,
                             Ipv4Address nextHop,
                             Ptr<Ipv4Route> route);
  /**
   * Schedule the cached reply to a random start time to avoid possible route reply storm
   *
   * \param packet the original packet
   * \param source IPv4 address of the source (i.e. request originator)
   * \param destination IPv4 address of the destination
   * \param route Route
   * \param hops number of hops
   */
  void ScheduleCachedReply (Ptr<Packet> packet,
                            Ipv4Address source,
                            Ipv4Address destination,
                            Ptr<Ipv4Route> route,
                            double hops);
  /**
   * Send network layer acknowledgment back to the earlier hop to notify the receipt of data packet
   *
   * \param ackId ACK ID
   * \param destination IPv4 address of the immediate ACK receiver
   * \param realSrc IPv4 address of the real source
   * \param realDst IPv4 address of the real destination
   * \param protocol the protocol number
   * \param route Route
   */
  void SendAck   (uint16_t ackId,
                  Ipv4Address destination,
                  Ipv4Address realSrc,
                  Ipv4Address realDst,
                  uint8_t protocol,
                  Ptr<Ipv4Route> route);
  /**
   * \param p packet to forward up
   * \param header IPv4 Header information
   * \param incomingInterface the Ipv4Interface on which the packet arrived
   * \return receive status
   *
   * Called from lower-level layers to send the packet up
   * in the stack.
   */
  virtual enum IpL4Protocol::RxStatus Receive (Ptr<Packet> p,
                                               Ipv4Header const &header,
                                               Ptr<Ipv4Interface> incomingInterface);

  /**
   * \param p packet to forward up
   * \param header IPv6 Header information
   * \param incomingInterface the Ipv6Interface on which the packet arrived
   * \return receive status
   *
   * Called from lower-level layers to send the packet up
   * in the stack.  Not implemented (IPv6).
   */
  virtual enum IpL4Protocol::RxStatus Receive (Ptr<Packet> p,
                                               Ipv6Header const &header,
                                               Ptr<Ipv6Interface> incomingInterface);

  void SetDownTarget (IpL4Protocol::DownTargetCallback callback);
  void SetDownTarget6 (IpL4Protocol::DownTargetCallback6 callback);
  IpL4Protocol::DownTargetCallback GetDownTarget (void) const;
  IpL4Protocol::DownTargetCallback6 GetDownTarget6 (void) const;
  /**
   * \brief Process method
   * Called from Ipv4L3Protocol::Receive.
   *
   * \param packet the packet
   * \param ipv4Header IPv4 header of the packet
   * \param dst destination address of the packet received (i.e. us)
   * \param nextHeader the next header
   * \param protocol the protocol number
   * \param isDropped if the packet must be dropped
   * \return the size processed
   */
  uint8_t Process (Ptr<Packet>& packet, Ipv4Header const& ipv4Header, Ipv4Address dst, uint8_t *nextHeader, uint8_t protocol, bool& isDropped);
  /**
   * \brief Insert a new Dsr Option.
   * \param option the option to insert
   */
  void Insert (Ptr<dsr::DsrOptions> option);
  /**
   * \brief Get the option corresponding to optionNumber.
   * \param optionNumber the option number of the option to retrieve
   * \return a matching Dsr option
   */
  Ptr<dsr::DsrOptions> GetOption (int optionNumber);
  /**
   * \brief Cancel the route request timer.
   * \param dst The dst address of the route request timer
   * \param isRemove whether to remove the route request entry or not
   */
  void CancelRreqTimer (Ipv4Address dst, bool isRemove);
  /**
   * \brief Schedule the route request retry.
   * \param packet the original packet
   * \param address List of IPv4 addresses
   * \param nonProp flag if RREQ is non-propagating
   * \param requestId Unique request ID
   * \param protocol the protocol number
   */
  void ScheduleRreqRetry (Ptr<Packet> packet, std::vector<Ipv4Address> address, bool nonProp, uint32_t requestId, uint8_t protocol);
  /**
   * Handle route discovery timer
   *
   * \param packet the original packet
   * \param address List of IPv4 addresses
   * \param requestId Unique request ID
   * \param protocol the protocol number
   */
  void RouteRequestTimerExpire (Ptr<Packet> packet, std::vector<Ipv4Address> address, uint32_t requestId, uint8_t protocol);

  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model.  Return the number of streams (possibly zero) that
   * have been assigned.
   *
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this model
   */
  int64_t AssignStreams (int64_t stream);

protected:
  /*
 *    * This function will notify other components connected to the node that a new stack member is now connected
 *       * This will be used to notify Layer 3 protocol of layer 4 protocol stack to connect them together.
 *          */
  virtual void NotifyNewAggregate ();
  /**
   * \brief Drop trace callback.
   */
  virtual void DoDispose (void);
  /**
   * The trace for drop, receive and send data packets
   */
  TracedCallback<Ptr<const Packet> > m_dropTrace; ///< packet drop trace callback
  TracedCallback <const DsrOptionSRHeader &> m_txPacketTrace; ///< packet trace callback

private:
  void Start ();
  /**
   * \brief Send the route error message when the link breaks to the next hop.
   * \param nextHop next hop address
   * \param protocol protocol number
   */
  void SendRerrWhenBreaksLinkToNextHop (Ipv4Address nextHop, uint8_t protocol);
  /**
   * \brief Promiscuous receive data packets destined to some other node.
   * \param device The network device
   * \param packet Data packet we just received
   * \param protocol The protocol we receive, need to verify it is dsr protocol
   * \param from The from address we received the packet
   * \param to The address this packet is destined for
   * \param packetType The dsr packet type, 0 is for control packet, 1 for data packet
   * \return true if the packet was processed, false otherwise
   */
  bool PromiscReceive (Ptr<NetDevice> device, Ptr<const Packet> packet, uint16_t protocol, const Address &from,
                       const Address &to, NetDevice::PacketType packetType);
  /**
   * \brief Define the list to hold DSR options.
   */
  typedef std::list<Ptr<DsrOptions> > DsrOptionList_t;
  /**
   * \brief List of DSR Options supported.
   */
  DsrOptionList_t m_options;

  Ptr<Ipv4L3Protocol> m_ipv4;                           ///< Ipv4l3Protocol

  Ptr<Ipv4Route> m_ipv4Route;                           ///< Ipv4 Route

  Ptr<Ipv4> m_ip;                                       ///< The ip ptr

  Ptr<Node> m_node;                                     ///< The node ptr

  Ipv4Address m_mainAddress;                            ///< Our own Ip address

  uint8_t segsLeft;                                     ///< The segment left value from SR header

  IpL4Protocol::DownTargetCallback m_downTarget;        ///< The callback for down layer

  uint32_t m_maxNetworkSize;                            ///< Maximum network queue size

  Time m_maxNetworkDelay;                               ///< Maximum network delay

  uint32_t m_discoveryHopLimit;                         ///< Maximum hops to go for route request

  uint8_t m_maxSalvageCount;                            ///< Maximum # times to salvage a packet

  Time  m_requestPeriod;                                ///< The base time interval between route requests

  Time m_nonpropRequestTimeout;                         ///< The non-propagation request timeout

  uint32_t m_sendRetries;                               ///< # of retries have been sent for network acknowledgment

  uint32_t m_passiveRetries;                            ///< # of retries have been sent for passive acknowledgment

  uint32_t m_linkRetries;                               ///< # of retries have been sent for link acknowledgment

  uint32_t m_rreqRetries;                               ///< Maximum number of retransmissions of RREQ with TTL = NetDiameter to discover a route

  uint32_t m_maxMaintRexmt;                             ///< Maximum number of retransmissions of data packets

  Time m_nodeTraversalTime;                             ///< Time estimated for packet to travel between two nodes

  uint32_t m_maxSendBuffLen;                            ///< The maximum number of packets that we allow a routing protocol to buffer.

  Time  m_sendBufferTimeout;                            ///< The maximum period of time that a routing protocol is allowed to buffer a packet for.

  DsrSendBuffer m_sendBuffer;                           ///< The send buffer

  DsrErrorBuffer m_errorBuffer;                         ///< The error buffer to save the error messages

  uint32_t  m_maxMaintainLen;                           ///< Max # of entries for maintenance buffer

  Time     m_maxMaintainTime;                           ///< Time out for maintenance buffer

  uint32_t m_maxCacheLen;                               ///< Max # of cache entries for route cache

  Time   m_maxCacheTime;                                ///< Max time for caching the route cache entry

  Time  m_maxRreqTime;                                  ///< Max time for caching the route request entry

  uint32_t  m_maxEntriesEachDst;                        ///< Max number of route entries to save for each destination

  DsrMaintainBuffer m_maintainBuffer;                   ///< The declaration of maintain buffer

  uint32_t m_requestId;                                 ///< The id assigned to each route request

  uint16_t m_ackId;                                     ///< The ack id assigned to each acknowledge

  uint32_t m_requestTableSize;                          ///< The max size of the request table size

  uint32_t m_requestTableIds;                           ///< The request table identifiers

  uint32_t m_maxRreqId;                                 ///< The max number of request ids for a single destination

  Time  m_blacklistTimeout;                             ///< The black list time out

  Ipv4Address m_broadcast;                              ///< The broadcast IP address

  uint32_t m_broadcastJitter;                           ///< The max time to delay route request broadcast.

  Time  m_passiveAckTimeout;                            ///< The timeout value for passive acknowledge

  uint32_t m_tryPassiveAcks;                            ///< Maximum number of packet transmission using passive acknowledgment

  Time  m_linkAckTimeout;                               ///< The timeout value for link acknowledge

  uint32_t m_tryLinkAcks;                               ///< Maximum number of packet transmission using link acknowledgment

  Timer m_sendBuffTimer;                                ///< The send buffer timer

  Time m_sendBuffInterval;                              ///< how often to check send buffer

  Time  m_gratReplyHoldoff;                             ///< The max gratuitous reply hold off time

  Time m_maxRequestPeriod;                              ///< The max request period

  uint32_t m_graReplyTableSize;                         ///< Set the gratuitous reply table size

  std::string m_cacheType;                              ///< The type of route cache

  std::string m_routeSortType;                          ///< The type of route sort methods

  uint32_t m_stabilityDecrFactor;                       ///< The initial decrease factor for link cache

  uint32_t m_stabilityIncrFactor;                       ///< The initial increase factor for link cache

  Time m_initStability;                                 ///< The initial stability value for link cache

  Time m_minLifeTime;                                   ///< The min life time

  Time m_useExtends;                                    ///< The use extension of the life time for link cache

  bool m_subRoute;                                      ///< Whether to save sub route or not

  Time m_retransIncr;                                   ///< the increase time for retransmission timer when face network congestion

  std::vector<Ipv4Address> m_finalRoute;                ///< The route cache

  std::map<Ipv4Address, Timer> m_addressReqTimer;       ///< Map IP address + RREQ timer.

  std::map<Ipv4Address, Timer> m_nonPropReqTimer;       ///< Map IP address + RREQ timer.

  std::map<NetworkKey, Timer>  m_addressForwardTimer;   ///< Map network key + forward timer.

  std::map<NetworkKey, uint32_t> m_addressForwardCnt;   ///< Map network key + forward counts.

  std::map<PassiveKey, uint32_t> m_passiveCnt;          ///< Map packet key + passive forward counts.

  std::map<PassiveKey, Timer> m_passiveAckTimer;        ///< The timer for passive acknowledgment

  std::map<LinkKey, uint32_t> m_linkCnt;                ///< Map packet key + link forward counts.

  std::map<LinkKey, Timer> m_linkAckTimer;              ///< The timer for link acknowledgment

  Ptr<dsr::DsrRouteCache> m_routeCache;                 ///< A "drop-front" queue used by the routing layer to cache routes found.

  Ptr<dsr::DsrRreqTable> m_rreqTable;                   ///< A "drop-front" queue used by the routing layer to cache route request sent.

  Ptr<dsr::DsrPassiveBuffer> m_passiveBuffer;           ///< A "drop-front" queue used by the routing layer to cache route request sent.

  uint32_t m_numPriorityQueues;                         ///< The number of priority queues used

  bool m_linkAck;                                       ///< define if we use link acknowledgement or not

  std::map<uint32_t, Ptr<dsr::DsrNetworkQueue> > m_priorityQueue;   ///< priority queues

  DsrGraReply m_graReply;                               ///< The gratuitous route reply.

  DsrNetworkQueue m_networkQueue;                       ///< The network queue.

  std::vector<Ipv4Address> m_clearList;                 ///< The node that is clear to send packet to

  std::vector<Ipv4Address> m_addresses;                 ///< The bind ipv4 addresses with next hop, src, destination address in sequence

  std::map <std::string, uint32_t> m_macToNodeIdMap;    ///< The map of mac address to node id

  Ptr<UniformRandomVariable> m_uniformRandomVariable;    ///< Provides uniform random variables.
};
}  /* namespace dsr */
}  /* namespace ns3 */

#endif /* DSR_ROUTING_H */
