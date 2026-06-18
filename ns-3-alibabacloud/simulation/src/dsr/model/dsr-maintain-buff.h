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

#ifndef DSR_MAINTAIN_BUFF_H
#define DSR_MAINTAIN_BUFF_H

#include <vector>
#include "ns3/ipv4-routing-protocol.h"
#include "ns3/simulator.h"
#include "ns3/ipv4-header.h"
#include "dsr-option-header.h"

namespace ns3 {
namespace dsr {
/**
 * The maintenance buffer is responsible for maintaining packet next hop delivery
 * The data packet is saved in maintenance buffer whenever the data packet is sent out of send buffer
 */
struct LinkKey
{
  Ipv4Address m_source; ///< source address
  Ipv4Address m_destination; ///< destination address
  Ipv4Address m_ourAdd; ///< local address
  Ipv4Address m_nextHop; ///< next hop address

  /**
   * Compare maintain Buffer entries
   * \param o object to compare
   * \return true if equal
   */
  bool operator < (const LinkKey & o) const
  {
    if (m_source < o.m_source)
      {
        return true;
      }
    if (o.m_source < m_source)
      {
        return false;
      }
    if (m_destination < o.m_destination)
      {
        return true;
      }
    if (o.m_destination < m_destination)
      {
        return false;
      }
    if (m_ourAdd < o.m_ourAdd)
      {
        return true;
      }
    if (o.m_ourAdd < m_ourAdd)
      {
        return false;
      }
    if (m_nextHop < o.m_nextHop)
      {
        return true;
      }
    if (o.m_nextHop < m_nextHop)
      {
        return false;
      }
    return false;
  }
};

/// NetworkKey structure
struct NetworkKey
{
  uint16_t m_ackId; ///< acknowledge ID
  Ipv4Address m_ourAdd; ///< local address
  Ipv4Address m_nextHop; ///< next hop
  Ipv4Address m_source; ///< source address
  Ipv4Address m_destination; ///< destination address

  /**
   * Compare maintain Buffer entries
   * \param o object to compare
   * \return true if equal
   */
  bool operator < (const NetworkKey & o) const
  {
    if (m_ackId < o.m_ackId)
      {
        return true;
      }
    if (o.m_ackId < m_ackId)
      {
        return false;
      }
    if (m_source < o.m_source)
      {
        return true;
      }
    if (o.m_source < m_source)
      {
        return false;
      }
    if (m_destination < o.m_destination)
      {
        return true;
      }
    if (o.m_destination < m_destination)
      {
        return false;
      }
    if (m_ourAdd < o.m_ourAdd)
      {
        return true;
      }
    if (o.m_ourAdd < m_ourAdd)
      {
        return false;
      }
    if (m_nextHop < o.m_nextHop)
      {
        return true;
      }
    if (o.m_nextHop < m_nextHop)
      {
        return false;
      }
    return false;
  }
};

/// PassiveKey structure
struct PassiveKey
{
  uint16_t m_ackId; ///< acknowledge ID
  Ipv4Address m_source; ///< source address
  Ipv4Address m_destination; ///< destination address
  uint8_t m_segsLeft; ///< segments left

  /**
   * Compare maintain Buffer entries
   * \param o is the object to compare
   * \return true if equal
   */
  bool operator < (const PassiveKey & o) const
  {
    if (m_ackId < o.m_ackId)
      {
        return true;
      }
    if (o.m_ackId < m_ackId)
      {
        return false;
      }
    if (m_source < o.m_source)
      {
        return true;
      }
    if (o.m_source < m_source)
      {
        return false;
      }
    if (m_destination < o.m_destination)
      {
        return true;
      }
    if (o.m_destination < m_destination)
      {
        return false;
      }
    if (m_segsLeft < o.m_segsLeft)
      {
        return true;
      }
    if (o.m_segsLeft < m_segsLeft)
      {
        return false;
      }
    return false;
  }
};

/**
 * \ingroup dsr
 * \brief DSR Maintain Buffer Entry
 */
class DsrMaintainBuffEntry
{
public:
  /**
   * Construct a DsrMaintainBuffEntry with the given parameters
   *
   * \param pa packet
   * \param us our IPv4 address
   * \param n next hop IPv4 address
   * \param s IPv4 address of the source
   * \param dst IPv4 address of the destination
   * \param ackId ACK ID
   * \param segs number of segments left
   * \param exp expiration time
   */
  DsrMaintainBuffEntry (Ptr<const Packet> pa = 0, Ipv4Address us = Ipv4Address (),
                        Ipv4Address n = Ipv4Address (), Ipv4Address s = Ipv4Address (), Ipv4Address dst = Ipv4Address (),
                        uint16_t ackId = 0, uint8_t segs = 0, Time exp = Simulator::Now ())
    : m_packet (pa),
      m_ourAdd (us),
      m_nextHop (n),
      m_src (s),
      m_dst (dst),
      m_ackId (ackId),
      m_segsLeft (segs),
      m_expire (exp + Simulator::Now ())
  {
  }

  // Fields
  /**
   * Get packet
   * \returns the current packet
   */
  Ptr<const Packet> GetPacket () const
  {
    return m_packet;
  }
  /**
   * Set packet
   * \param p the current packet
   */
  void SetPacket (Ptr<const Packet> p)
  {
    m_packet = p;
  }
  /**
   * Get local address of entry
   * \returns the local IP address
   */
  Ipv4Address GetOurAdd () const
  {
    return m_ourAdd;
  }
  /**
   * Set local address of entry
   * \param us the local IP address
   */
  void SetOurAdd (Ipv4Address us)
  {
    m_ourAdd = us;
  }
  /**
   * Get next hop of entry
   * \returns the IP address for the next hop
   */
  Ipv4Address GetNextHop () const
  {
    return m_nextHop;
  }
  /**
   * Set next hop of entry
   * \param n the next hop IP address
   */
  void SetNextHop (Ipv4Address n)
  {
    m_nextHop = n;
  }
  /**
   * Get destination address
   * \returns the destination IP address
   */
  Ipv4Address GetDst () const
  {
    return m_dst;
  }
  /**
   * Set destination address
   * \param n the destination IP address
   */
  void SetDst (Ipv4Address n)
  {
    m_dst = n;
  }
  /**
   * Get source address
   * \returns the source IP address
   */
  Ipv4Address GetSrc () const
  {
    return m_src;
  }
  /**
   * Set source address
   * \param s the source IP address
   */
  void SetSrc (Ipv4Address s)
  {
    m_src = s;
  }
  /**
   * Get acknowledge ID
   * \returns the acknowledge ID
   */
  uint16_t GetAckId () const
  {
    return m_ackId;
  }
  /**
   * Set acknowledge ID
   * \param ackId the acknowledge ID
   */
  void SetAckId (uint16_t ackId)
  {
    m_ackId = ackId;
  }
  /**
   * Get segments left
   * \returns the number of segments left
   */
  uint8_t GetSegsLeft () const
  {
    return m_segsLeft;
  }
  /**
   * Set segments left
   * \param segs the number of segments left
   */
  void SetSegsLeft (uint8_t segs)
  {
    m_segsLeft = segs;
  }
  /**
   * Set expiration time
   * \param exp the expire time
   */
  void SetExpireTime (Time exp)
  {
    m_expire = exp + Simulator::Now ();
  }
  /**
   * Get expiration time
   * \returns the expiration time
   */
  Time GetExpireTime () const
  {
    return m_expire - Simulator::Now ();
  }

private:
  /// Data packet
  Ptr<const Packet> m_packet;
  /// Our own ip address
  Ipv4Address m_ourAdd;
  /// Next hop Ip address
  Ipv4Address m_nextHop;
  /// The source address
  Ipv4Address m_src;
  /// The destination address
  Ipv4Address m_dst;
  /// The data ack id
  uint16_t m_ackId;
  /// The segments left field
  uint8_t m_segsLeft;
  /// Expire time for queue entry
  Time m_expire;
};
/**
 * \ingroup dsr
 * \brief DSR maintain buffer
 */
/************************************************************************************************************************/
class DsrMaintainBuffer
{
public:
  /**
   * Default constructor
   */
  DsrMaintainBuffer ()
  {
  }
  /// Push entry in queue, if there is no entry with the same packet and destination address in queue.
  /// \param entry Maintain Buffer Entry
  /// \return true on success adding the Entry.
  bool Enqueue (DsrMaintainBuffEntry & entry);
  /// Return first found (the earliest) entry for given destination
  /// \param [in] dst Entry destination
  /// \param [out] entry The Entry found (if any).
  /// \return true on success
  bool Dequeue (Ipv4Address dst, DsrMaintainBuffEntry & entry);
  /// Remove all packets with next hop IP address dst
  /// \param nextHop Next hop in the route.
  void DropPacketWithNextHop (Ipv4Address nextHop);
  /// Finds whether a packet with next hop dst exists in the queue
  /// \param nextHop Next hop in the route.
  /// \return true if there is a packet directed to the next hop.
  bool Find (Ipv4Address nextHop);
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
   * Get maintain buffer timeout
   * \returns the maintain buffer timeout
   */
  Time GetMaintainBufferTimeout () const
  {
    return m_maintainBufferTimeout;
  }
  /**
   * Set maintain buffer timeout
   * \param t the maintain buffer timeoout
   */
  void SetMaintainBufferTimeout (Time t)
  {
    m_maintainBufferTimeout = t;
  }
  /// Verify if all the elements in the maintenance buffer entry is the same
  /// \note For real this function checks if at most one entry is equal. If it is,
  /// that entry is removed. Further entries are NOT checked. This could be a bug.
  /// \param entry The Entry to check
  /// \return true if an Entry was found and removed.
  bool AllEqual (DsrMaintainBuffEntry & entry);
  /// Verify if the maintain buffer entry is the same in every field for link ack
  /// \param entry The Entry to check
  /// \return true if an Entry was found and removed.
  bool LinkEqual (DsrMaintainBuffEntry & entry);
  /// Verify if the maintain buffer entry is the same in every field for network ack
  /// \param entry The Entry to check
  /// \return true if an Entry was found and removed.
  bool NetworkEqual (DsrMaintainBuffEntry & entry);
  /// Verify if the maintain buffer entry is the same in every field for promiscuous ack
  /// \param entry The Entry to check
  /// \return true if an Entry was found and removed.
  bool PromiscEqual (DsrMaintainBuffEntry & entry);

private:
  /// The vector of maintain buffer entries
  std::vector<DsrMaintainBuffEntry> m_maintainBuffer;
  /// The vector of network keys
  std::vector<NetworkKey> m_allNetworkKey;
  /// Remove all expired entries
  void Purge ();
  /// The maximum number of packets that we allow a routing protocol to buffer.
  uint32_t m_maxLen;
  /// The maximum period of time that a routing protocol is allowed to buffer a packet for, seconds.
  Time m_maintainBufferTimeout;
};
/*******************************************************************************************************************************/
} // namespace dsr
} // namespace ns3
#endif /* DSR_MAINTAIN_BUFF_H */
