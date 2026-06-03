/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007-2009 Strasbourg University
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
 */

#ifndef NDISC_CACHE_H
#define NDISC_CACHE_H

#include <stdint.h>
#include <list>
#include <unordered_map>

#include "ns3/packet.h"
#include "ns3/nstime.h"
#include "ns3/net-device.h"
#include "ns3/ipv6-address.h"
#include "ns3/ptr.h"
#include "ns3/timer.h"
#include "ns3/output-stream-wrapper.h"

namespace ns3
{

class NetDevice;
class Ipv6Interface;
class Ipv6Header;
class Icmpv6L4Protocol;

/**
 * \ingroup ipv6
 *
 * \brief IPv6 Neighbor Discovery cache.
 */
class NdiscCache : public Object
{
public:
  class Entry;

  /**
   * \brief Get the type ID
   * \return type ID
   */
  static TypeId GetTypeId ();

  /**
   * \brief Default value for unres qlen.
   */
  static const uint32_t DEFAULT_UNRES_QLEN = 3;

  /**
   * \brief Constructor.
   */
  NdiscCache ();

  /**
   * \brief Destructor.
   */
  ~NdiscCache ();

  // Delete default and copy constructor, and assignment operator to avoid misuse
  NdiscCache (NdiscCache const &) = delete;
  NdiscCache & operator = (NdiscCache const &) = delete;

  /**
   * \brief Get the NetDevice associated with this cache.
   * \return NetDevice
   */
  Ptr<NetDevice> GetDevice () const;

  /**
   * \brief Get the Ipv6Interface associated with this cache.
   * \returns The Ipv6Interface.
   */
  Ptr<Ipv6Interface> GetInterface () const;

  /**
   * \brief Lookup in the cache.
   * \param dst destination address.
   * \return the entry if found, 0 otherwise.
   */
  virtual NdiscCache::Entry* Lookup (Ipv6Address dst);

  /**
   * \brief Lookup in the cache for a MAC address.
   * \param dst destination MAC address.
   * \return a list of matching entries.
   */
  std::list<NdiscCache::Entry*> LookupInverse (Address dst);

  /**
   * \brief Add an entry.
   * \param to address to add
   * \return an new Entry
   */
  virtual NdiscCache::Entry* Add (Ipv6Address to);

  /**
   * \brief Delete an entry.
   * \param entry pointer to delete from the list.
   */
  void Remove (NdiscCache::Entry* entry);

  /**
   * \brief Flush the cache.
   */
  void Flush ();

  /**
   * \brief Set the max number of waiting packet.
   * \param unresQlen value to set
   */
  void SetUnresQlen (uint32_t unresQlen);

  /**
   * \brief Get the max number of waiting packet.
   * \return max number
   */
  uint32_t GetUnresQlen ();

  /**
   * \brief Set the device and interface.
   * \param device the device
   * \param interface the IPv6 interface
   * \param icmpv6 the ICMPv6 protocol
   */
  void SetDevice (Ptr<NetDevice> device, Ptr<Ipv6Interface> interface, Ptr<Icmpv6L4Protocol> icmpv6);

  /**
   * \brief Print the NDISC cache entries
   *
   * \param stream the ostream the NDISC cache entries is printed to
   */
  void PrintNdiscCache (Ptr<OutputStreamWrapper> stream);

  /**
   * \brief Pair of a packet and an Ipv4 header.
   */
  typedef std::pair<Ptr<Packet>, Ipv6Header> Ipv6PayloadHeaderPair;

  /**
   * \ingroup ipv6
   *
   * \brief A record that holds information about a NdiscCache entry.
   */
  class Entry
  {
public:
    /**
     * \brief Constructor.
     * \param nd The NdiscCache this entry belongs to.
     */
    Entry (NdiscCache* nd);

    virtual ~Entry() = default;

    /**
     * \brief Changes the state to this entry to INCOMPLETE.
     * \param p packet that wait to be sent
     */
    void MarkIncomplete (Ipv6PayloadHeaderPair p);

    /**
     * \brief Changes the state to this entry to REACHABLE.
     * \param mac MAC address
     * \return the list of packet waiting
     */
    std::list<Ipv6PayloadHeaderPair> MarkReachable (Address mac);

    /**
     * \brief Changes the state to this entry to PROBE.
     */
    void MarkProbe ();

    /**
     * \brief Changes the state to this entry to STALE.
     * \param mac L2 address
     * \return the list of packet waiting
     */
    std::list<Ipv6PayloadHeaderPair> MarkStale (Address mac);

    /**
     * \brief Changes the state to this entry to STALE.
     */
    void MarkStale ();

    /**
     * \brief Changes the state to this entry to REACHABLE.
     */
    void MarkReachable ();

    /**
     * \brief Change the state to this entry to DELAY.
     */
    void MarkDelay ();

    /**
     * \brief Change the state to this entry to PERMANENT.
     */
    void MarkPermanent ();

    /**
     * \brief Add a packet (or replace old value) in the queue.
     * \param p packet to add
     */
    void AddWaitingPacket (Ipv6PayloadHeaderPair p);

    /**
     * \brief Clear the waiting packet list.
     */
    void ClearWaitingPacket ();

    /**
     * \brief Is the entry STALE
     * \return true if the entry is in STALE state, false otherwise
     */
    bool IsStale () const;

    /**
     * \brief Is the entry REACHABLE
     * \return true if the entry is in REACHABLE state, false otherwise
     */
    bool IsReachable () const;

    /**
     * \brief Is the entry DELAY
     * \return true if the entry is in DELAY state, false otherwise
     */
    bool IsDelay () const;

    /**
     * \brief Is the entry INCOMPLETE
     * \return true if the entry is in INCOMPLETE state, false otherwise
     */
    bool IsIncomplete () const;

    /**
     * \brief Is the entry PROBE
     * \return true if the entry is in PROBE state, false otherwise
     */
    bool IsProbe () const;

    /**
     * \brief Is the entry PERMANENT
     * \return true if the entry is in PERMANENT state, false otherwise
     */
    bool IsPermanent () const;

    /**
     * \brief Get the MAC address of this entry.
     * \return the L2 address
     */
    Address GetMacAddress () const;

    /**
     * \brief Set the MAC address of this entry.
     * \param mac the MAC address to set
     */
    void SetMacAddress (Address mac);

    /**
     * \brief If the entry is a host or a router.
     * \return true if the node is a router, 0 if it is a host
     */
    bool IsRouter () const;

    /**
     * \brief Set the node type.
     * \param router true is a router, false means a host
     */
    void SetRouter (bool router);

    /**
     * \brief Get the time of last reachability confirmation.
     * \return time
     */
    Time GetLastReachabilityConfirmation () const;

    /**
     * \brief Start the reachable timer.
     */
    void StartReachableTimer ();

    /**
     * \brief Update the reachable timer.
     */
    void UpdateReachableTimer ();

    /**
     * \brief Start retransmit timer.
     */
    void StartRetransmitTimer ();

    /**
     * \brief Start probe timer.
     */
    void StartProbeTimer ();

    /**
     * \brief Start delay timer.
     */
    void StartDelayTimer ();

    /**
     * \brief Stop NUD timer and reset the NUD retransmission counter
     */
    void StopNudTimer ();

    /**
     * \brief Function called when reachable timer timeout.
     */
    void FunctionReachableTimeout ();

    /**
     * \brief Function called when retransmit timer timeout.
     * It verify that the NS retransmit has reached the max so discard the entry
     * otherwise it retransmit a NS.
     */
    void FunctionRetransmitTimeout ();

    /**
     * \brief Function called when probe timer timeout.
     */
    void FunctionProbeTimeout ();

    /**
     * \brief Function called when delay timer timeout.
     */
    void FunctionDelayTimeout ();

    /**
     * \brief Set the IPv6 address.
     * \param ipv6Address IPv6 address
     */
    void SetIpv6Address (Ipv6Address ipv6Address);
    
    /**
     * \brief Get the IPv6 address.
     * \returns The IPv6 address
     */
    Ipv6Address GetIpv6Address (void) const;

    /**
     * \brief Print this entry to the given output stream.
     *
     * \param os the output stream to which this Ipv6Address is printed
     */
    void Print (std::ostream &os) const;

protected:
    /**
     * \brief the NdiscCache associated.
     */
    NdiscCache* m_ndCache;

private:
    /**
     * \brief The IPv6 address.
     */
    Ipv6Address m_ipv6Address;

    /**
     * \brief The Entry state enumeration.
     */
    enum NdiscCacheEntryState_e
    {
      INCOMPLETE, /**< No mapping between IPv6 and L2 addresses */
      REACHABLE, /**< Mapping exists between IPv6 and L2 addresses */
      STALE, /**< Mapping is stale */
      DELAY, /**< Try to wait contact from remote host */
      PROBE, /**< Try to contact IPv6 address to know again its L2 address */
      PERMANENT /**< Permanent Mapping exists between IPv6 and L2 addresses */
    };

    /**
     * \brief The state of the entry.
     */
    NdiscCacheEntryState_e m_state;

    /**
     * \brief The MAC address.
     */
    Address m_macAddress;

    /**
     * \brief The list of packet waiting.
     */
    std::list<Ipv6PayloadHeaderPair> m_waiting;

    /**
     * \brief Type of node (router or host).
     */
    bool m_router;

    /**
     * \brief Timer (used for NUD).
     */
    Timer m_nudTimer;

    /**
     * \brief Last time we see a reachability confirmation.
     */
    Time m_lastReachabilityConfirmation;

    /**
     * \brief Number of NS retransmission.
     */
    uint8_t m_nsRetransmit;
  };

protected:
  /**
   * \brief Dispose this object.
   */
  void DoDispose ();

  /**
   * \brief Neighbor Discovery Cache container
   */
  typedef std::unordered_map<Ipv6Address, NdiscCache::Entry *, Ipv6AddressHash> Cache;
  /**
   * \brief Neighbor Discovery Cache container iterator
   */
  typedef std::unordered_map<Ipv6Address, NdiscCache::Entry *, Ipv6AddressHash>::iterator CacheI;

  /**
   * \brief A list of Entry.
   */
  Cache m_ndCache;

private:
  /**
   * \brief The NetDevice.
   */
  Ptr<NetDevice> m_device;

  /**
   * \brief the interface.
   */
  Ptr<Ipv6Interface> m_interface;

  /**
   * \brief the icmpv6 L4 protocol for this cache.
   */
  Ptr<Icmpv6L4Protocol> m_icmpv6;

  /**
   * \brief Max number of packet stored in m_waiting.
   */
  uint32_t m_unresQlen;
};

/**
 * \brief Stream insertion operator.
 *
 * \param os the reference to the output stream
 * \param entry the NdiscCache::Entry
 * \returns the reference to the output stream
 */
std::ostream & operator << (std::ostream& os, NdiscCache::Entry const& entry);


} /* namespace ns3 */

#endif /* NDISC_CACHE_H */

