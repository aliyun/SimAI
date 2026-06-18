// -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*-
//
// Copyright (c) 2006 Georgia Tech Research Corporation
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
// Author: George F. Riley<riley@ece.gatech.edu>
//

#ifndef IPV4_L3_PROTOCOL_H
#define IPV4_L3_PROTOCOL_H

#include <list>
#include <map>
#include <vector>
#include <stdint.h>
#include "ns3/ipv4-address.h"
#include "ns3/ptr.h"
#include "ns3/net-device.h"
#include "ns3/ipv4.h"
#include "ns3/traced-callback.h"
#include "ns3/ipv4-header.h"
#include "ns3/ipv4-routing-protocol.h"
#include "ns3/nstime.h"
#include "ns3/simulator.h"

class Ipv4L3ProtocolTestCase;

namespace ns3 {

class Packet;
class NetDevice;
class Ipv4Interface;
class Ipv4Address;
class Ipv4Header;
class Ipv4RoutingTableEntry;
class Ipv4Route;
class Node;
class Socket;
class Ipv4RawSocketImpl;
class IpL4Protocol;
class Icmpv4L4Protocol;

/**
 * \ingroup ipv4
 *
 * \brief Implement the IPv4 layer.
 * 
 * This is the actual implementation of IP.  It contains APIs to send and
 * receive packets at the IP layer, as well as APIs for IP routing.
 *
 * This class contains two distinct groups of trace sources.  The
 * trace sources 'Rx' and 'Tx' are called, respectively, immediately
 * after receiving from the NetDevice and immediately before sending
 * to a NetDevice for transmitting a packet.  These are low level
 * trace sources that include the Ipv4Header already serialized into
 * the packet.  In contrast, the Drop, SendOutgoing, UnicastForward,
 * and LocalDeliver trace sources are slightly higher-level and pass
 * around the Ipv4Header as an explicit parameter and not as part of
 * the packet.
 *
 * IP fragmentation and reassembly is handled at this level.
 * At the moment the fragmentation does not handle IP option headers,
 * and in particular the ones that shall not be fragmented.
 * Moreover, the actual implementation does not mimic exactly the Linux
 * kernel. Hence it is not possible, for instance, to test a fragmentation
 * attack.
 */
class Ipv4L3Protocol : public Ipv4
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  static const uint16_t PROT_NUMBER; //!< Protocol number (0x0800)

  Ipv4L3Protocol();
  virtual ~Ipv4L3Protocol ();

  // Delete copy constructor and assignment operator to avoid misuse
  Ipv4L3Protocol (const Ipv4L3Protocol &) = delete;
  Ipv4L3Protocol & operator = (const Ipv4L3Protocol &) = delete;

  /**
   * \enum DropReason
   * \brief Reason why a packet has been dropped.
   */
  enum DropReason 
  {
    DROP_TTL_EXPIRED = 1,   /**< Packet TTL has expired */
    DROP_NO_ROUTE,   /**< No route to host */
    DROP_BAD_CHECKSUM,   /**< Bad checksum */
    DROP_INTERFACE_DOWN,   /**< Interface is down so can not send packet */
    DROP_ROUTE_ERROR,   /**< Route error */
    DROP_FRAGMENT_TIMEOUT, /**< Fragment timeout exceeded */
    DROP_DUPLICATE  /**< Duplicate packet received */
  };

  /**
   * \brief Set node associated with this stack.
   * \param node node to set
   */
  void SetNode (Ptr<Node> node);

  // functions defined in base class Ipv4

  void SetRoutingProtocol (Ptr<Ipv4RoutingProtocol> routingProtocol);
  Ptr<Ipv4RoutingProtocol> GetRoutingProtocol (void) const;

  Ptr<Socket> CreateRawSocket (void);
  void DeleteRawSocket (Ptr<Socket> socket);

  virtual void Insert (Ptr<IpL4Protocol> protocol);
  virtual void Insert (Ptr<IpL4Protocol> protocol, uint32_t interfaceIndex);

  virtual void Remove (Ptr<IpL4Protocol> protocol);
  virtual void Remove (Ptr<IpL4Protocol> protocol, uint32_t interfaceIndex);

  virtual Ptr<IpL4Protocol> GetProtocol (int protocolNumber) const;
  virtual Ptr<IpL4Protocol> GetProtocol (int protocolNumber, int32_t interfaceIndex) const;

  virtual Ipv4Address SourceAddressSelection (uint32_t interface, Ipv4Address dest);

  /**
   * \param ttl default ttl to use
   *
   * When we need to send an ipv4 packet, we use this default
   * ttl value.
   */
  void SetDefaultTtl (uint8_t ttl);

  /**
   * Lower layer calls this method after calling L3Demux::Lookup
   * The ARP subclass needs to know from which NetDevice this
   * packet is coming to:
   *    - implement a per-NetDevice ARP cache
   *    - send back arp replies on the right device
   * \param device network device
   * \param p the packet
   * \param protocol protocol value
   * \param from address of the correspondent
   * \param to address of the destination
   * \param packetType type of the packet
   */
  void Receive ( Ptr<NetDevice> device, Ptr<const Packet> p, uint16_t protocol, const Address &from,
                 const Address &to, NetDevice::PacketType packetType);

  /**
   * \param packet packet to send
   * \param source source address of packet
   * \param destination address of packet
   * \param protocol number of packet
   * \param route route entry
   *
   * Higher-level layers call this method to send a packet
   * down the stack to the MAC and PHY layers.
   */
  void Send (Ptr<Packet> packet, Ipv4Address source, 
             Ipv4Address destination, uint8_t protocol, Ptr<Ipv4Route> route);
  /**
   * \param packet packet to send
   * \param ipHeader IP Header
   * \param route route entry
   *
   * Higher-level layers call this method to send a packet with IPv4 Header
   * (Intend to be used with IpHeaderInclude attribute.)
   */
  void SendWithHeader (Ptr<Packet> packet, Ipv4Header ipHeader, Ptr<Ipv4Route> route);

  uint32_t AddInterface (Ptr<NetDevice> device);
  /**
   * \brief Get an interface.
   * \param i interface index
   * \return IPv4 interface pointer
   */
  Ptr<Ipv4Interface> GetInterface (uint32_t i) const;
  uint32_t GetNInterfaces (void) const;

  int32_t GetInterfaceForAddress (Ipv4Address addr) const;
  int32_t GetInterfaceForPrefix (Ipv4Address addr, Ipv4Mask mask) const;
  int32_t GetInterfaceForDevice (Ptr<const NetDevice> device) const;
  bool IsDestinationAddress (Ipv4Address address, uint32_t iif) const;

  bool AddAddress (uint32_t i, Ipv4InterfaceAddress address);
  Ipv4InterfaceAddress GetAddress (uint32_t interfaceIndex, uint32_t addressIndex) const;
  uint32_t GetNAddresses (uint32_t interface) const;
  bool RemoveAddress (uint32_t interfaceIndex, uint32_t addressIndex);
  bool RemoveAddress (uint32_t interface, Ipv4Address address);
  Ipv4Address SelectSourceAddress (Ptr<const NetDevice> device,
                                   Ipv4Address dst, Ipv4InterfaceAddress::InterfaceAddressScope_e scope);


  void SetMetric (uint32_t i, uint16_t metric);
  uint16_t GetMetric (uint32_t i) const;
  uint16_t GetMtu (uint32_t i) const;
  bool IsUp (uint32_t i) const;
  void SetUp (uint32_t i);
  void SetDown (uint32_t i);
  bool IsForwarding (uint32_t i) const;
  void SetForwarding (uint32_t i, bool val);

  Ptr<NetDevice> GetNetDevice (uint32_t i);

  /**
   * \brief Check if an IPv4 address is unicast according to the node.
   *
   * This function checks all the node's interfaces and the respective subnet masks.
   * An address is considered unicast if it's not broadcast, subnet-broadcast or multicast.
   *
   * \param ad address
   *
   * \return true if the address is unicast
   */
  bool IsUnicast (Ipv4Address ad) const;

  /**
   * TracedCallback signature for packet send, forward, or local deliver events.
   *
   * \param [in] header the Ipv4Header
   * \param [in] packet the packet
   * \param [in] interface IP-level interface number
   */
  typedef void (* SentTracedCallback)
      (const Ipv4Header & header, Ptr<const Packet> packet, uint32_t interface);

  /**
   * TracedCallback signature for packet transmission or reception events.
   *
   * \param [in] packet the packet.
   * \param [in] ipv4 the Ipv4 protocol
   * \param [in] interface IP-level interface number
   * \deprecated The non-const \c Ptr<Ipv4> argument is deprecated
   * and will be changed to \c Ptr<const Ipv4> in a future release.
   */
  typedef void (* TxRxTracedCallback)
      (Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface);

  /**
   * TracedCallback signature for packet drop events.
   *
   * \param [in] header the Ipv4Header.
   * \param [in] packet the packet.
   * \param [in] reason the reason the packet was dropped.
   * \param [in] ipv4 the Ipv4 protocol
   * \param [in] interface IP-level interface number
   * \deprecated The non-const \c Ptr<Ipv4> argument is deprecated
   * and will be changed to \c Ptr<const Ipv4> in a future release.
   */
  typedef void (* DropTracedCallback)
      (const Ipv4Header & header, Ptr<const Packet> packet,
          DropReason reason, Ptr<Ipv4> ipv4,
          uint32_t interface);

protected:

  virtual void DoDispose (void);
  /**
   * This function will notify other components connected to the node that a new stack member is now connected
   * This will be used to notify Layer 3 protocol of layer 4 protocol stack to connect them together.
   */
  virtual void NotifyNewAggregate ();
private:
  /**
   * \brief Ipv4L3ProtocolTestCase test case.
   * \relates Ipv4L3ProtocolTestCase
   */
  friend class ::Ipv4L3ProtocolTestCase;

  // class Ipv4 attributes
  virtual void SetIpForward (bool forward);
  virtual bool GetIpForward (void) const;
  virtual void SetWeakEsModel (bool model);
  virtual bool GetWeakEsModel (void) const;

  /**
   * \brief Decrease the identification value for a dropped or recursed packet
   * \param source source IPv4 address
   * \param destination destination IPv4 address
   * \param protocol L4 protocol
   */
  void DecreaseIdentification (Ipv4Address source,
                               Ipv4Address destination,
                               uint8_t protocol);

  /**
   * \brief Construct an IPv4 header.
   * \param source source IPv4 address
   * \param destination destination IPv4 address
   * \param protocol L4 protocol
   * \param payloadSize payload size
   * \param ttl Time to Live
   * \param tos Type of Service
   * \param mayFragment true if the packet can be fragmented
   * \return newly created IPv4 header
   */
  Ipv4Header BuildHeader (
      Ipv4Address source,
      Ipv4Address destination,
      uint8_t protocol,
      uint16_t payloadSize,
      uint8_t ttl,
      uint8_t tos,
      bool mayFragment);

  /**
   * \brief Send packet with route.
   * \param route route
   * \param packet packet to send
   * \param ipHeader IPv4 header to add to the packet
   */
  void
  SendRealOut (Ptr<Ipv4Route> route,
               Ptr<Packet> packet,
               Ipv4Header const &ipHeader);

  /**
   * \brief Forward a packet.
   * \param rtentry route
   * \param p packet to forward
   * \param header IPv4 header to add to the packet
   */
  void 
  IpForward (Ptr<Ipv4Route> rtentry, 
             Ptr<const Packet> p, 
             const Ipv4Header &header);

  /**
   * \brief Forward a multicast packet.
   * \param mrtentry route
   * \param p packet to forward
   * \param header IPv4 header to add to the packet
   */
  void
  IpMulticastForward (Ptr<Ipv4MulticastRoute> mrtentry, 
                      Ptr<const Packet> p, 
                      const Ipv4Header &header);

  /**
   * \brief Deliver a packet.
   * \param p packet delivered
   * \param ip IPv4 header
   * \param iif input interface packet was received
   */
  void LocalDeliver (Ptr<const Packet> p, Ipv4Header const&ip, uint32_t iif);

  /**
   * \brief Fallback when no route is found.
   * \param p packet
   * \param ipHeader IPv4 header
   * \param sockErrno error number
   */
  void RouteInputError (Ptr<const Packet> p, const Ipv4Header & ipHeader, Socket::SocketErrno sockErrno);

  /**
   * \brief Add an IPv4 interface to the stack.
   * \param interface interface to add
   * \return index of newly added interface
   */
  uint32_t AddIpv4Interface (Ptr<Ipv4Interface> interface);

  /**
   * \brief Setup loopback interface.
   */
  void SetupLoopback (void);

  /**
   * \brief Get ICMPv4 protocol.
   * \return Icmpv4L4Protocol pointer
   */
  Ptr<Icmpv4L4Protocol> GetIcmp (void) const;

  /**
   * \brief Check if an IPv4 address is unicast.
   * \param ad address
   * \param interfaceMask the network mask
   * \return true if the address is unicast
   */
  bool IsUnicast (Ipv4Address ad, Ipv4Mask interfaceMask) const;

  /**
   * \brief Pair of a packet and an Ipv4 header.
   */
  typedef std::pair<Ptr<Packet>, Ipv4Header> Ipv4PayloadHeaderPair;

  /**
   * \brief Fragment a packet
   * \param packet the packet
   * \param ipv4Header the IPv4 header
   * \param outIfaceMtu the MTU of the interface
   * \param listFragments the list of fragments
   */
  void DoFragmentation (Ptr<Packet> packet, const Ipv4Header& ipv4Header, uint32_t outIfaceMtu, std::list<Ipv4PayloadHeaderPair>& listFragments);

  /**
   * \brief Process a packet fragment
   * \param packet the packet
   * \param ipHeader the IP header
   * \param iif Input Interface
   * \return true is the fragment completed the packet
   */
  bool ProcessFragment (Ptr<Packet>& packet, Ipv4Header & ipHeader, uint32_t iif);

  /**
   * \brief Make a copy of the packet, add the header and invoke the TX trace callback
   * \param ipHeader the IP header that will be added to the packet
   * \param packet the packet
   * \param ipv4 the Ipv4 protocol
   * \param interface the IP-level interface index
   *
   * Note: If the TracedCallback API ever is extended, we could consider
   * to check for connected functions before adding the header
   */
  void CallTxTrace (const Ipv4Header & ipHeader, Ptr<Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface);

  /**
   * \brief Container of the IPv4 Interfaces.
   */
  typedef std::vector<Ptr<Ipv4Interface> > Ipv4InterfaceList;
  /**
   * \brief Container of NetDevices registered to IPv4 and their interface indexes.
   */
  typedef std::map<Ptr<const NetDevice>, uint32_t > Ipv4InterfaceReverseContainer;
  /**
   * \brief Container of the IPv4 Raw Sockets.
   */
  typedef std::list<Ptr<Ipv4RawSocketImpl> > SocketList;

  /**
   * \brief Container of the IPv4 L4 keys: protocol number, interface index
   */
  typedef std::pair<int, int32_t> L4ListKey_t;

  /**
   * \brief Container of the IPv4 L4 instances.
   */
  typedef std::map<L4ListKey_t, Ptr<IpL4Protocol> > L4List_t;

  bool m_ipForward;      //!< Forwarding packets (i.e. router mode) state.
  bool m_weakEsModel;    //!< Weak ES model state
  L4List_t m_protocols;  //!< List of transport protocol.
  Ipv4InterfaceList m_interfaces; //!< List of IPv4 interfaces.
  Ipv4InterfaceReverseContainer m_reverseInterfacesContainer; //!< Container of NetDevice / Interface index associations.
  uint8_t m_defaultTtl;  //!< Default TTL
  std::map<std::pair<uint64_t, uint8_t>, uint16_t> m_identification; //!< Identification (for each {src, dst, proto} tuple)
  Ptr<Node> m_node; //!< Node attached to stack.

  /// Trace of sent packets
  TracedCallback<const Ipv4Header &, Ptr<const Packet>, uint32_t> m_sendOutgoingTrace;
  /// Trace of unicast forwarded packets
  TracedCallback<const Ipv4Header &, Ptr<const Packet>, uint32_t> m_unicastForwardTrace;
  /// Trace of multicast forwarded packets
  TracedCallback<const Ipv4Header &, Ptr<const Packet>, uint32_t> m_multicastForwardTrace;
  /// Trace of locally delivered packets
  TracedCallback<const Ipv4Header &, Ptr<const Packet>, uint32_t> m_localDeliverTrace;

  // The following two traces pass a packet with an IP header
  /// Trace of transmitted packets
  /// \deprecated The non-const \c Ptr<Ipv4> argument is deprecated
  /// and will be changed to \c Ptr<const Ipv4> in a future release.
  TracedCallback<Ptr<const Packet>, Ptr<Ipv4>,  uint32_t> m_txTrace;
  /// Trace of received packets
  /// \deprecated The non-const \c Ptr<Ipv4> argument is deprecated
  /// and will be changed to \c Ptr<const Ipv4> in a future release.
  TracedCallback<Ptr<const Packet>, Ptr<Ipv4>, uint32_t> m_rxTrace;
  // <ip-header, payload, reason, ifindex> (ifindex not valid if reason is DROP_NO_ROUTE)
  /// Trace of dropped packets
  /// \deprecated The non-const \c Ptr<Ipv4> argument is deprecated
  /// and will be changed to \c Ptr<const Ipv4> in a future release.
  TracedCallback<const Ipv4Header &, Ptr<const Packet>, DropReason, Ptr<Ipv4>, uint32_t> m_dropTrace;

  Ptr<Ipv4RoutingProtocol> m_routingProtocol; //!< Routing protocol associated with the stack

  SocketList m_sockets; //!< List of IPv4 raw sockets.

  /// Key identifying a fragmented packet
  typedef std::pair<uint64_t, uint32_t> FragmentKey_t;

  /// Container for fragment timeouts.
  typedef std::list< std::tuple <Time, FragmentKey_t, Ipv4Header, uint32_t > > FragmentsTimeoutsList_t;
  /// Container Iterator for fragment timeouts..
  typedef std::list< std::tuple <Time, FragmentKey_t, Ipv4Header, uint32_t > >::iterator FragmentsTimeoutsListI_t;

  /**
   * \brief Process the timeout for packet fragments
   * \param key representing the packet fragments
   * \param ipHeader the IP header of the original packet
   * \param iif Input Interface
   */
  void HandleFragmentsTimeout (FragmentKey_t key, Ipv4Header & ipHeader, uint32_t iif);

  /**
   * \brief Set a new timeout "event" for a fragmented packet
   * \param key the fragment identification
   * \param ipHeader the IPv4 header of the fragmented packet
   * \param iif input interface of the packet
   * \return an iterator to the inserted "event"
   */
  FragmentsTimeoutsListI_t SetTimeout (FragmentKey_t key, Ipv4Header ipHeader, uint32_t iif);

  /**
   * \brief Handles a fragmented packet timeout
   */
  void HandleTimeout (void);

  FragmentsTimeoutsList_t m_timeoutEventList;  //!< Timeout "events" container

  EventId m_timeoutEvent;  //!< Event for the next scheduled timeout

  /**
   * \brief A Set of Fragment belonging to the same packet (src, dst, identification and proto)
   */
  class Fragments : public SimpleRefCount<Fragments>
  {
  public:
    /**
     * \brief Constructor.
     */
    Fragments ();

    /**
     * \brief Destructor.
     */
    ~Fragments ();

    /**
     * \brief Add a fragment.
     * \param fragment the fragment
     * \param fragmentOffset the offset of the fragment
     * \param moreFragment the bit "More Fragment"
     */
    void AddFragment (Ptr<Packet> fragment, uint16_t fragmentOffset, bool moreFragment);

    /**
     * \brief If all fragments have been added.
     * \returns true if the packet is entire
     */
    bool IsEntire () const;

    /**
     * \brief Get the entire packet.
     * \return the entire packet
     */
    Ptr<Packet> GetPacket () const;

    /**
     * \brief Get the complete part of the packet.
     * \return the part we have comeplete
     */
    Ptr<Packet> GetPartialPacket () const;

    /**
     * \brief Set the Timeout iterator.
     * \param iter The iterator.
     */
    void SetTimeoutIter (FragmentsTimeoutsListI_t iter);

    /**
     * \brief Get the Timeout iterator.
     * \returns The iterator.
     */
    FragmentsTimeoutsListI_t GetTimeoutIter ();

  private:
    /**
     * \brief True if other fragments will be sent.
     */
    bool m_moreFragment;

    /**
     * \brief The current fragments.
     */
    std::list<std::pair<Ptr<Packet>, uint16_t> > m_fragments;

    /**
     * \brief Timeout iterator to "event" handler
     */
    FragmentsTimeoutsListI_t m_timeoutIter;
  };

  /// Container of fragments, stored as pairs(src+dst addr, src+dst port) / fragment
  typedef std::map< FragmentKey_t, Ptr<Fragments> > MapFragments_t;

  MapFragments_t       m_fragments; //!< Fragmented packets.
  Time                 m_fragmentExpirationTimeout; //!< Expiration timeout

  /// IETF RFC 6621, Section 6.2 de-duplication w/o IPSec
  /// RFC 6621 recommended duplicate packet tuple: {IPV hash, IP protocol, IP source address, IP destination address}
  typedef std::tuple <uint64_t, uint8_t, Ipv4Address, Ipv4Address> DupTuple_t;
  /// Maps packet duplicate tuple to expiration time
  typedef std::map<DupTuple_t, Time> DupMap_t;

  /**
   * Registers duplicate entry, return false if new
   * \param [in] p Possibly duplicate packet.
   * \param [in] header Packet \pname{p} header.
   * \return True if this packet is a duplicate
   */
  bool UpdateDuplicate (Ptr<const Packet> p, const Ipv4Header &header);
  /**
   * Remove expired duplicates packet entry
   */
  void RemoveDuplicates (void);

  bool                m_enableDpd;    //!< Enable multicast duplicate packet detection
  DupMap_t            m_dups;         //!< map of packet duplicate tuples to expiry event
  Time                m_expire;       //!< duplicate entry expiration delay
  Time                m_purge;        //!< time between purging expired duplicate entries
  EventId             m_cleanDpd;     //!< event to cleanup expired duplicate entries
};

} // Namespace ns3

#endif /* IPV4_L3_PROTOCOL_H */
