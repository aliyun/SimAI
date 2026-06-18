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

#ifndef DSR_NETWORK_QUEUE_H
#define DSR_NETWORK_QUEUE_H

#include <vector>
#include "ns3/ipv4-routing-protocol.h"
#include "ns3/simulator.h"
#include "ns3/ipv4-header.h"
#include "dsr-option-header.h"

namespace ns3 {
namespace dsr {

enum DsrMessageType
{
  DSR_CONTROL_PACKET = 1,
  DSR_DATA_PACKET = 2
};
/**
 * \ingroup dsr
 * \brief DSR Network Queue Entry
 */
class DsrNetworkQueueEntry
{
public:
  /**
   * Construct a DsrNetworkQueueEntry with the given parameters
   *
   * \param pa packet
   * \param s IPv4 address of the source
   * \param n IPv4 address of the next hop node
   * \param exp expiration time
   * \param r Route
   */
  DsrNetworkQueueEntry (Ptr<const Packet> pa = 0, Ipv4Address s = Ipv4Address (), Ipv4Address n = Ipv4Address (),
                        Time exp = Simulator::Now (), Ptr<Ipv4Route> r = 0)
    : m_packet (pa),
      m_srcAddr (s),
      m_nextHopAddr (n),
      tstamp (exp),
      m_ipv4Route (r)
  {
  }
  /**
   * Compare send buffer entries
   * \param o entry to compare
   * \return true if equal
   */
  bool operator== (DsrNetworkQueueEntry const & o) const
  {
    return ((m_packet == o.m_packet) && (m_srcAddr == o.m_srcAddr) && (m_nextHopAddr == o.m_nextHopAddr)
            && (tstamp == o.tstamp) && (m_ipv4Route == o.m_ipv4Route));
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
   * Get IP route function
   * \returns the IP route
   */
  Ptr<Ipv4Route> GetIpv4Route () const
  {
    return m_ipv4Route;
  }
  /**
   * Set IP route function
   * \param route
   */
  void SetIpv4Route (Ptr<Ipv4Route> route)
  {
    m_ipv4Route = route;
  }
  /**
   * Get source address function
   * \returns the source IP address
   */
  Ipv4Address GetSourceAddress () const
  {
    return m_srcAddr;
  }
  /**
   * Set source address function
   * \param addr the source IP address
   */
  void SetSourceAddress (Ipv4Address addr)
  {
    m_srcAddr = addr;
  }
  /**
   * Get next hop address function
   * \returns the next hop IP address
   */
  Ipv4Address GetNextHopAddress () const
  {
    return m_nextHopAddr;
  }
  /**
   * Set next hop address function
   * \param addr the next hop IP address
   */
  void SetNextHopAddress (Ipv4Address addr)
  {
    m_nextHopAddr = addr;
  }
  /**
   * Get inserted time stamp function
   * \returns the inserted time stamp
   */
  Time GetInsertedTimeStamp (void) const
  {
    return tstamp;
  }
  /**
   * Set inserted time stamp function
   * \param time the inserted timestamp
   */
  void SetInsertedTimeStamp (Time time)
  {
    tstamp = time;
  }

private:
  /// Data packet
  Ptr<const Packet> m_packet; ///< the packet
  Ipv4Address m_srcAddr; ///< source address
  Ipv4Address m_nextHopAddr; ///< next hop address
  Time tstamp; ///< timestamp
  /// Ipv4Route
  Ptr<Ipv4Route> m_ipv4Route;
};

class DsrNetworkQueue : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  DsrNetworkQueue ();
  /**
   * Construct a DsrNetworkQueue with the given
   * maximum length and maximum delay.
   *
   * \param maxLen Maximum queue size
   * \param maxDelay Maximum entry lifetime in the queue
   */
  DsrNetworkQueue (uint32_t maxLen, Time maxDelay);
  ~DsrNetworkQueue ();

  /**
   * Find the packet entry with a given next hop
   * \param nextHop the IP address of the next hop
   * \param entry the DSR queue entry
   * \returns true if found
   */
  bool FindPacketWithNexthop (Ipv4Address nextHop, DsrNetworkQueueEntry & entry);
  /**
   * Try to find an entry with a particular next hop, and return true if found
   * \param nextHop the next hop IP address
   * \returns true if found
   */
  bool Find (Ipv4Address nextHop);
  /**
   * Push entry in queue, if there is no entry with the same
   * packet and destination address in queue.
   *
   * \param entry packet entry
   * \return true if the given entry was put in the queue,
   *         false otherwise
   */
  bool Enqueue (DsrNetworkQueueEntry & entry);
  /**
   * Return first found (the earliest) entry for given destination
   *
   * \param entry pointer to the return entry
   * \return true if an entry is returned,
   *         false otherwise
   */
  bool Dequeue (DsrNetworkQueueEntry & entry);
  /**
   * Number of entries
   *
   * \return the current queue size/length
   */
  uint32_t GetSize ();

  /**
   * Set the maximum queue size
   *
   * \param maxSize the maximum queue size
   */
  void SetMaxNetworkSize (uint32_t maxSize);
  /**
   * Set the maximum entry lifetime in the queue
   *
   * \param delay the maximum entry lifetime
   */
  void SetMaxNetworkDelay (Time delay);
  /**
   * Return the maximum queue size
   *
   * \return the maximum queue size
   */
  uint32_t GetMaxNetworkSize (void) const;
  /**
   * Return the maximum entry lifetime for this queue
   *
   * \return the maximum entry lifetime for this queue
   */
  Time GetMaxNetworkDelay (void) const;
  /**
   * Clear the queue
   */
  void Flush (void);

  /**
   * Return the current queue entry
   *
   * \return the current queue entry
   */
  std::vector<DsrNetworkQueueEntry> & GetQueue ()
  {
    return m_dsrNetworkQueue;
  }

private:
  /**
   * Clean the queue by removing entries that exceeded lifetime.
   */
  void Cleanup (void);
  std::vector<DsrNetworkQueueEntry> m_dsrNetworkQueue; //!< Queue (vector) of entries
  uint32_t m_size; //!< Current queue size
  uint32_t m_maxSize; //!< Maximum queue size
  Time m_maxDelay; //!< Maximum entry lifetime
};

} // namespace dsr
} // namespace ns3

#endif /* DSR_NETWORK_QUEUE_H */
