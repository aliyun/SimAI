/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006 INRIA
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
#ifndef ARP_CACHE_H
#define ARP_CACHE_H

#include <stdint.h>
#include <list>
#include <unordered_map>
#include "ns3/simulator.h"
#include "ns3/callback.h"
#include "ns3/packet.h"
#include "ns3/nstime.h"
#include "ns3/net-device.h"
#include "ns3/ipv4-address.h"
#include "ns3/address.h"
#include "ns3/ptr.h"
#include "ns3/object.h"
#include "ns3/traced-callback.h"
#include "ns3/output-stream-wrapper.h"

namespace ns3 {

class NetDevice;
class Ipv4Interface;
class Ipv4Header;

/**
 * \ingroup arp
 * \brief An ARP cache
 *
 * A cached lookup table for translating layer 3 addresses to layer 2.
 * This implementation does lookups from IPv4 to a MAC address
 */
class ArpCache : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  class Entry;

  ArpCache ();
  ~ArpCache ();

  // Delete copy constructor and assignment operator to avoid misuse
  ArpCache (ArpCache const &) = delete;
  ArpCache & operator = (ArpCache const &) = delete;

  /**
   * \brief Set the NetDevice and Ipv4Interface associated with the ArpCache
   *
   * \param device The hardware NetDevice associated with this ARP cache
   * \param interface the Ipv4Interface associated with this ARP cache
   */
  void SetDevice (Ptr<NetDevice> device, Ptr<Ipv4Interface> interface);
  /**
   * \brief Returns the NetDevice that this ARP cache is associated with
   * \return The NetDevice that this ARP cache is associated with
   */
  Ptr<NetDevice> GetDevice (void) const;
  /**
   * \brief Returns the Ipv4Interface that this ARP cache is associated with
   * \return the Ipv4Interface that this ARP cache is associated with
   */
  Ptr<Ipv4Interface> GetInterface (void) const;

  /**
   * \brief Set the time the entry will be in ALIVE state (unless refreshed)
   * \param aliveTimeout the Alive state timeout
   */
  void SetAliveTimeout (Time aliveTimeout);
  /**
   * \brief Set the time the entry will be in DEAD state before being removed
   * \param deadTimeout the Dead state timeout
   */
  void SetDeadTimeout (Time deadTimeout);
  /**
   * \brief Set the time the entry will be in WAIT_REPLY state
   * \param waitReplyTimeout the WAIT_REPLY state timeout
   */
  void SetWaitReplyTimeout (Time waitReplyTimeout);

  /**
   * \brief Get the time the entry will be in ALIVE state (unless refreshed)
   * \returns the Alive state timeout
   */
  Time GetAliveTimeout (void) const;
  /**
   * \brief Get the time the entry will be in DEAD state before being removed
   * \returns the Dead state timeout
   */
  Time GetDeadTimeout (void) const;
  /**
   * \brief Get the time the entry will be in WAIT_REPLY state
   * \returns the WAIT_REPLY state timeout
   */
  Time GetWaitReplyTimeout (void) const;

  /**
   * This callback is set when the ArpCache is set up and allows
   * the cache to generate an Arp request when the WaitReply
   * time expires and a retransmission must be sent
   *
   * \param arpRequestCallback Callback for transmitting an Arp request.
   */
  void SetArpRequestCallback (Callback<void, Ptr<const ArpCache>, 
                                       Ipv4Address> arpRequestCallback);
  /**
   * This method will schedule a timeout at WaitReplyTimeout interval
   * in the future, unless a timer is already running for the cache,
   * in which case this method does nothing.
   */
  void StartWaitReplyTimer (void);
  /**
   * \brief Do lookup in the ARP cache against an IP address
   * \param destination The destination IPv4 address to lookup the MAC address
   * of
   * \return An ArpCache::Entry with info about layer 2
   */
  ArpCache::Entry *Lookup (Ipv4Address destination);
  /**
   * \brief Do lookup in the ARP cache against a MAC address
   * \param destination The destination MAC address to lookup
   * of
   * \return A std::list of ArpCache::Entry with info about layer 2
   */
  std::list<ArpCache::Entry *> LookupInverse (Address destination);
  /**
   * \brief Add an Ipv4Address to this ARP cache
   * \param to the destination address of the ARP entry.
   * \returns A pointer to a new ARP Entry.
   */
  ArpCache::Entry *Add (Ipv4Address to);
  /**
   * \brief Remove an entry.
   * \param entry pointer to delete it from the list
   */
  void Remove (ArpCache::Entry *entry);
  /**
   * \brief Clear the ArpCache of all entries
   */
  void Flush (void);

  /**
   * \brief Print the ARP cache entries
   *
   * \param stream the ostream the ARP cache entries is printed to
   */
  void PrintArpCache (Ptr<OutputStreamWrapper> stream);

  /**
   * \brief Pair of a packet and an Ipv4 header.
   */
  typedef std::pair<Ptr<Packet>, Ipv4Header> Ipv4PayloadHeaderPair;

  /**
   * \brief A record that that holds information about an ArpCache entry
   */
  class Entry {
public:
    /**
     * \brief Constructor
     * \param arp The ArpCache this entry belongs to
     */
    Entry (ArpCache *arp);

    /**
     * \brief Changes the state of this entry to dead
     */
    void MarkDead (void);
    /**
     * \param macAddress
     */
    void MarkAlive (Address macAddress);
    /**
     * \param waiting
     */
    void MarkWaitReply (Ipv4PayloadHeaderPair waiting);
    /**
     * \brief Changes the state of this entry to Permanent.
     *
     * The entry must have a valid MacAddress.
     */
    void MarkPermanent (void);
    /**
     * \param waiting
     * \return 
     */
    bool UpdateWaitReply (Ipv4PayloadHeaderPair waiting);
    /**
     * \return True if the state of this entry is dead; false otherwise.
     */
    bool IsDead (void);
    /**
     * \return True if the state of this entry is alive; false otherwise.
     */
    bool IsAlive (void);
    /**
     * \return True if the state of this entry is wait_reply; false otherwise.
     */
    bool IsWaitReply (void);
    /**
     * \return True if the state of this entry is permanent; false otherwise.
     */
    bool IsPermanent (void); 
    /**
     * \return The MacAddress of this entry
     */
    Address GetMacAddress (void) const;
    /**
     * \return The Ipv4Address for this entry
     */
    Ipv4Address GetIpv4Address (void) const;
    /**
     * \param macAddress The MacAddress for this entry
     */
    void SetMacAddress (Address macAddress);
    /**
     * \param destination The Ipv4Address for this entry
     */
    void SetIpv4Address (Ipv4Address destination);
    /**
     * \return True if this entry has timed out; false otherwise.
     *
     * This function returns true if the time elapsed strictly exceeds
     * the timeout value (i.e., is not less than or equal to the timeout).
     */
    bool IsExpired (void) const;
    /**
     * \returns 0 is no packet is pending, the next packet to send if 
     *            packets are pending.
     */
    Ipv4PayloadHeaderPair DequeuePending (void);
    /**
     * \brief Clear the pending packet list
     */
    void ClearPendingPacket (void);
    /**
     * \returns number of retries that have been sent for an ArpRequest
     *  in WaitReply state.
     */
    uint32_t GetRetries (void) const;
    /**
     * \brief Increment the counter of number of retries for an entry
     */
    void IncrementRetries (void);
    /**
     * \brief Zero the counter of number of retries for an entry
     */
    void ClearRetries (void);

    /**
     * \brief Update the entry when seeing a packet
     */
    void UpdateSeen (void);

private:
    /**
     * \brief ARP cache entry states
     */
    enum ArpCacheEntryState_e {
      ALIVE,
      WAIT_REPLY,
      DEAD,
      PERMANENT
    };

    /**
     * \brief Returns the entry timeout
     * \returns the entry timeout
     */
    Time GetTimeout (void) const;

    ArpCache *m_arp; //!< pointer to the ARP cache owning the entry
    ArpCacheEntryState_e m_state; //!< state of the entry
    Time m_lastSeen; //!< last moment a packet from that address has been seen
    Address m_macAddress; //!< entry's MAC address
    Ipv4Address m_ipv4Address; //!< entry's IP address
    std::list<Ipv4PayloadHeaderPair> m_pending; //!< list of pending packets for the entry's IP
    uint32_t m_retries; //!< rerty counter
  };

private:
  /**
   * \brief ARP Cache container
   */
  typedef std::unordered_map<Ipv4Address, ArpCache::Entry *, Ipv4AddressHash> Cache;
  /**
   * \brief ARP Cache container iterator
   */
  typedef std::unordered_map<Ipv4Address, ArpCache::Entry *, Ipv4AddressHash>::iterator CacheI;

  virtual void DoDispose (void);

  Ptr<NetDevice> m_device; //!< NetDevice associated with the cache
  Ptr<Ipv4Interface> m_interface; //!< Ipv4Interface associated with the cache
  Time m_aliveTimeout; //!< cache alive state timeout
  Time m_deadTimeout; //!< cache dead state timeout
  Time m_waitReplyTimeout; //!< cache reply state timeout
  EventId m_waitReplyTimer;  //!< cache alive state timer
  Callback<void, Ptr<const ArpCache>, Ipv4Address> m_arpRequestCallback;  //!< reply timeout callback
  uint32_t m_maxRetries; //!< max retries for a resolution

  /**
   * This function is an event handler for the event that the
   * ArpCache wants to check whether it must retry any Arp requests.
   * If there are no Arp requests pending, this event is not scheduled.
   */
  void HandleWaitReplyTimeout (void);
  uint32_t m_pendingQueueSize; //!< number of packets waiting for a resolution
  Cache m_arpCache; //!< the ARP cache
  TracedCallback<Ptr<const Packet> > m_dropTrace; //!< trace for packets dropped by the ARP cache queue
};


} // namespace ns3

#endif /* ARP_CACHE_H */
