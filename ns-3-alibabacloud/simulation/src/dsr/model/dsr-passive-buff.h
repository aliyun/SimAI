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

#ifndef DSR_PASSIVEBUFF_H
#define DSR_PASSIVEBUFF_H

#include <vector>
#include "ns3/ipv4-routing-protocol.h"
#include "ns3/simulator.h"

namespace ns3 {
namespace dsr {
/**
 * \ingroup dsr
 * \brief DSR Passive Buffer Entry
 */
class DsrPassiveBuffEntry
{
public:
  /**
   * Construct a DsrPassiveBuffEntry with the given parameters
   *
   * \param pa packet
   * \param d IPv4 address of the destination
   * \param s IPv4 address of the source
   * \param n IPv4 address of the next hop node
   * \param i ID
   * \param f fragment offset
   * \param seg number of segments left
   * \param exp expiration time
   * \param p protocol number
   */
  DsrPassiveBuffEntry (Ptr<const Packet> pa = 0, Ipv4Address d = Ipv4Address (), Ipv4Address s = Ipv4Address (),
                       Ipv4Address n = Ipv4Address (), uint16_t i = 0, uint16_t f = 0, uint8_t seg = 0, Time exp = Simulator::Now (),
                       uint8_t p = 0)
    : m_packet (pa),
      m_dst (d),
      m_source (s),
      m_nextHop (n),
      m_identification (i),
      m_fragmentOffset (f),
      m_segsLeft (seg),
      m_expire (exp + Simulator::Now ()),
      m_protocol (p)
  {
  }
  /**
   * Compare send buffer entries
   * \param o passive buffer entry to compare
   * \return true if equal
   */
  bool operator== (DsrPassiveBuffEntry const & o) const
  {
    return ((m_packet == o.m_packet) && (m_source == o.m_source) && (m_nextHop == o.m_nextHop) && (m_dst == o.m_dst) && (m_expire == o.m_expire));
  }

  // Fields
  /**
   * Get packet function
   * \returns the current packet
   */
  Ptr<const Packet> GetPacket () const
  {
    return m_packet;
  }
  /**
   * Set packet function
   * \param p the current packet
   */
  void SetPacket (Ptr<const Packet> p)
  {
    m_packet = p;
  }
  /**
   * Get destination address function
   * \returns the destination IP address
   */
  Ipv4Address GetDestination () const
  {
    return m_dst;
  }
  /**
   * Set destination address function
   * \param d the destination IP address
   */
  void SetDestination (Ipv4Address d)
  {
    m_dst = d;
  }
  /**
   * Get source address function
   * \returns the source IP address
   */
  Ipv4Address GetSource () const
  {
    return m_source;
  }
  /**
   * Set surce address function
   * \param s the source IP address
   */
  void SetSource (Ipv4Address s)
  {
    m_source = s;
  }
  /**
   * Get next hop address function
   * \returns the next hop IP address
   */
  Ipv4Address GetNextHop () const
  {
    return m_nextHop;
  }
  /**
   * Set next hop address function
   * \param n the next hop address
   */
  void SetNextHop (Ipv4Address n)
  {
    m_nextHop = n;
  }
  /**
   * Get identification function
   * \returns the identification
   */
  uint16_t GetIdentification () const
  {
    return m_identification;
  }
  /**
   * Set identification function
   * \param i the identification
   */
  void SetIdentification (uint16_t i)
  {
    m_identification = i;
  }
  /**
   * Get fragment offset function
   * \returns the fragment offset
   */
  uint16_t GetFragmentOffset () const
  {
    return m_fragmentOffset;
  }
  /**
   * Set fragment offset function
   * \param f the fragment offset
   */
  void SetFragmentOffset (uint16_t f)
  {
    m_fragmentOffset = f;
  }
  /**
   * Get segments left function
   * \returns the number of segments left
   */
  uint8_t GetSegsLeft () const
  {
    return m_segsLeft;
  }
  /**
   * Set segments left
   * \param seg the number of segments left
   */
  void SetSegsLeft (uint8_t seg)
  {
    m_segsLeft = seg;
  }
  /**
   * Set expire time
   * \param exp the expire time
   */
  void SetExpireTime (Time exp)
  {
    m_expire = exp + Simulator::Now ();
  }
  /**
   * Get expire time
   * \returns the expire time
   */
  Time GetExpireTime () const
  {
    return m_expire - Simulator::Now ();
  }
  /**
   * Set protocol function
   * \param p the protocol
   */
  void SetProtocol (uint8_t p)
  {
    m_protocol = p;
  }
  /**
   * Get protocol
   * \returns the protocol number
   */
  uint8_t GetProtocol () const
  {
    return m_protocol;
  }

private:
  /// Data packet
  Ptr<const Packet> m_packet;
  /// Destination address
  Ipv4Address m_dst;
  /// Source address
  Ipv4Address m_source;
  /// Nexthop address
  Ipv4Address m_nextHop;
  /// Identification
  uint16_t m_identification;
  /// Fragment offset
  uint16_t m_fragmentOffset;
  /// Segments left
  uint8_t m_segsLeft;
  /// Expire time for queue entry
  Time m_expire;
  /// The protocol number
  uint8_t m_protocol;
};

/**
 * \ingroup dsr
 * \class DsrPassiveBuffer
 * \brief DSR passive buffer
 */
/************************************************************************************************************************/
class DsrPassiveBuffer  : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();

  DsrPassiveBuffer ();
  virtual ~DsrPassiveBuffer ();

  /// Push entry in queue, if there is no entry with the same packet and destination address in queue.
  /// \param entry Buffer Entry
  /// \return true on success adding the Entry.
  bool Enqueue (DsrPassiveBuffEntry & entry);
  /// Return first found (the earliest) entry for given destination
  /// \param [in] dst Entry destination
  /// \param [out] entry The Entry found (if any).
  /// \return true on success
  bool Dequeue (Ipv4Address dst, DsrPassiveBuffEntry & entry);
  /// Finds whether a packet with destination dst exists in the queue
  /// \param dst Destination.
  /// \return true if there is a packet.
 bool Find (Ipv4Address dst);
  /// Check if all the entries in passive buffer entry is all equal or not
 /// \note For real this function checks if at most one entry is equal. If it is,
 /// that entry is removed. Further entries are NOT checked. This could be a bug.
 /// \param entry The Entry to check
 /// \return true if an Entry was found and removed.
  bool AllEqual (DsrPassiveBuffEntry & entry);
  /// Number of entries
  /// \return The number of entries.
  uint32_t GetSize ();

  // Fields
  /**
   * Get maximum queue length
   * \returns the maximum queue length
   */
  uint32_t GetMaxQueueLen () const
  {
    return m_maxLen;
  }
  /**
   * Set maximum queue length
   * \param len the maximum queue length
   */
  void SetMaxQueueLen (uint32_t len)
  {
    m_maxLen = len;
  }
  /**
   * Get passive buffer timeout
   * \returns the passive buffer timeout
   */
  Time GetPassiveBufferTimeout () const
  {
    return m_passiveBufferTimeout;
  }
  /**
   * Set passive buffer timeout
   * \param t the passive buffer timeout
   */
  void SetPassiveBufferTimeout (Time t)
  {
    m_passiveBufferTimeout = t;
  }

private:
  /// The send buffer to cache unsent packet
  std::vector<DsrPassiveBuffEntry> m_passiveBuffer;
  /// Remove all expired entries
  void Purge ();
  /// Notify that packet is dropped from queue by timeout
  /// \param en BuffEntry Buffer entry
  /// \param reason Drop reason
  void Drop (DsrPassiveBuffEntry en, std::string reason);
  /// Notify that packet is dropped from queue by timeout
  /// \param en BuffEntry Buffer entry
  /// \param reason Drop reason
  void DropLink (DsrPassiveBuffEntry en, std::string reason);
  /// The maximum number of packets that we allow a routing protocol to buffer.
  uint32_t m_maxLen;
  /// The maximum period of time that a routing protocol is allowed to buffer a packet for, seconds.
  Time m_passiveBufferTimeout;
  /// Check if the send buffer entry is the same or not
  /// \param en The Entry to check
  /// \param link The link to check
  /// \return true if an Entry source and Next hop are equal to the Link parameters
  static bool LinkEqual (DsrPassiveBuffEntry en, const std::vector<Ipv4Address> link)
  {
    return ((en.GetSource () == link[0]) && (en.GetNextHop () == link[1]));
  }
};
/*******************************************************************************************************************************/
} // namespace dsr
} // namespace ns3

#endif /* DSR_PASSIVEBUFF_H */
