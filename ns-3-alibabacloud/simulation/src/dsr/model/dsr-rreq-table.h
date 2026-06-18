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

#ifndef DSR_RREQ_TABLE_H
#define DSR_RREQ_TABLE_H

#include "ns3/simulator.h"
#include "ns3/timer.h"
#include "ns3/ipv4-address.h"
#include "ns3/callback.h"
#include <list>
#include <vector>
#include <map>

namespace ns3 {
namespace dsr {

/**
 * State of link
 */
enum LinkStates
{
  PROBABLE = 0,            //!< PROBABLE
  QUESTIONABLE = 1,        //!< QUESTIONABLE
};
/// BlackList description
struct BlackList
{
  Ipv4Address m_neighborAddress; //!< IPv4 address of the black-listed neighbor
  Time m_expireTime;             //!< Expire time in the black list
  LinkStates m_linkStates;       //!< State of the link

  /**
   * Construct a BlackList with the given parameters
   *
   * \param ip IPv4 address of the neighbor
   * \param t expire time for the black list entry
   */
  BlackList (Ipv4Address ip, Time t)
    : m_neighborAddress (ip),
      m_expireTime (t),
      m_linkStates (PROBABLE)
  {
  }
};
/**
 * The route request table entries
 */
struct RreqTableEntry
{
  uint32_t m_reqNo; //!< Route request number
  Time m_expire;    //!< Expire time
};
/**
 * The request entry for intermediate nodes to check if they have received this request or not
 * This is used to control the duplication request from being processed
 */
class DsrReceivedRreqEntry
{
public:
  /**
   * Construct a DsrReceivedRreqEntry with the given parameters
   *
   * \param d IPv4 address of the destination
   * \param i identification
   */
  DsrReceivedRreqEntry (Ipv4Address d = Ipv4Address (), uint16_t i = 0)
    : m_destination (d),
      m_identification (i)
  {
  }
  /**
   * \brief Compare send buffer entries (destination address and identification)
   * \param o another DsrReceivedRreqEntry
   * \return true if equal
   */
  bool operator== (DsrReceivedRreqEntry const & o) const
  {
    return ((m_destination == o.m_destination) && (m_identification == o.m_identification)
            );
  }

  /**
   * Return IPv4 address of the destination
   *
   * \return IPv4 address of the destination
   */
  Ipv4Address GetDestination () const
  {
    return m_destination;
  }
  /**
   * Set IPv4 address of the destination
   *
   * \param d IPv4 address of the destination
   */
  void SetDestination (Ipv4Address d)
  {
    m_destination = d;
  }
  /**
   * Return IPv4 address of the source
   *
   * \return IPv4 address of the source
   */
  Ipv4Address GetSource () const
  {
    return m_source;
  }
  /**
   * Set IPv4 address of the source
   *
   * \param s IPv4 address of the source
   */
  void SetSource (Ipv4Address s)
  {
    m_source = s;
  }
  /**
   * Return identification
   *
   * \return identification
   */
  uint16_t GetIdentification () const
  {
    return m_identification;
  }
  /**
   * Set identification
   *
   * \param i identification
   */
  void SetIdentification (uint16_t i)
  {
    m_identification = i;
  }
  /**
   * Set expire time for the RREQ entry.
   * Note that the parameter is duration but
   * the stored value is the absolute time.
   *
   * \param exp duration before expire
   */
  void SetExpireTime (Time exp)
  {
    m_expire = exp + Simulator::Now ();
  }
  /**
   * Return the remaining time before the RREQ entry expires.
   * Note that we return the remaining time but the stored
   * value is the absolute time.
   *
   * \return the remaining time before the RREQ entry expires
   */
  Time GetExpireTime () const
  {
    return m_expire - Simulator::Now ();
  }
  // \}
private:
  Ipv4Address m_destination; //!< IPv4 address of the destinaton
  Ipv4Address m_source;      //!< IPv4 address of the source
  uint16_t m_identification; //!< Route request identification
  Time m_expire;             //!< Route request expire time
};

/**
 * \ingroup dsr
 * \brief maintain list of DsrRreqTable entry
 */
class DsrRreqTable  : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();

  DsrRreqTable ();
  virtual ~DsrRreqTable ();

  /**
   * Set the initial discovert hop limit
   *
   * \param hl the initial discovert hop limit
   */
  void SetInitHopLimit (uint32_t hl)
  {
    m_initHopLimit = hl;
  }
  /**
   * Return the initial discovert hop limit
   *
   * \return the initial discovert hop limit
   */
  uint32_t GetInitHopLimit () const
  {
    return m_initHopLimit;
  }
  /**
   * Set the maximum number of request entries in
   * the request table.
   *
   * \param rt the maximum number of request entries
   */
  void SetRreqTableSize (uint32_t rt)
  {
    m_requestTableSize = rt;
  }
  /**
   * Return the maximum number of request entries in
   * the request table.
   *
   * \return the maximum number of request entries
   */
  uint32_t GetRreqTableSize () const
  {
    return m_requestTableSize;
  }
  /**
   * Set the maximum number of request source Ids in
   * the request table
   *
   * \param id the maximum number of request source Ids
   */
  void SetRreqIdSize (uint32_t id)
  {
    m_requestIdSize = id;
  }
  /**
   * Return the maximum number of request source Ids in
   * the request table
   *
   * \return the maximum number of request source Ids
   */
  uint32_t GetRreqIdSize () const
  {
    return m_requestIdSize;
  }
  /**
   * Set the maximum number of request Ids in
   * the request table for a single destination.
   *
   * \param uid the maximum number of request Ids
   */
  void SetUniqueRreqIdSize (uint32_t uid)
  {
    m_maxRreqId = uid;
  }
  /**
   * Return the maximum number of request Ids in
   * the request table for a single destination.
   *
   * \return the maximum number of request Ids
   */
  uint32_t GetUniqueRreqIdSize () const
  {
    return m_maxRreqId;
  }

  /// Remove the least used entry
  void RemoveLeastExpire ();
  /// Find the entry in the route request queue to see if already exists
  /// \param dst Destination IP
  void FindAndUpdate (Ipv4Address dst);
  /// Remove route request entry for dst
  /// \param dst Destination IP
  void RemoveRreqEntry (Ipv4Address dst);
  /// Get the request count number for one destination address
  /// \param dst Destination IP
  /// \return the route request counter
  uint32_t GetRreqCnt (Ipv4Address dst);

  /**
   * The following code generates new request id for each destination.
   * Check for duplicate ids and save new entries if the id is not present in the table.
   *
   * \param dst IPv4 address of the destination
   * \return id
   */
  uint32_t CheckUniqueRreqId (Ipv4Address dst);
  /**
   * Get the request id size
   *
   * \return the request id size
   */
  uint32_t GetRreqSize ();

  /**
   * set the unidirectional entry as QUESTIONABLE state
   */
  void Invalidate ();
  /**
   * \brief Verify if entry is unidirectional or not(e.g. add this neighbor to "blacklist" for blacklistTimeout period)
   * \param neighbor neighbor address link to which assumed to be unidirectional
   * \return true on success
   */
  BlackList* FindUnidirectional (Ipv4Address neighbor);
  /**
   * \brief Mark entry as unidirectional (e.g. add this neighbor to "blacklist" for blacklistTimeout period)
   * \param neighbor - neighbor address link to which assumed to be unidirectional
   * \param blacklistTimeout - time for which the neighboring node is put into the blacklist
   * \return true on success
   */
  bool MarkLinkAsUnidirectional (Ipv4Address neighbor, Time blacklistTimeout);
  /**
   * Remove all expired black list entries
   */
  void PurgeNeighbor ();

  /**
   * Find the source request entry in the route request queue, return false if not found
   * \param src the source address we just received the source request
   * \param dst the destination address the request is targeted at
   * \param id the identification number for this request
   * \return true if found, false otherwise
   */
  bool FindSourceEntry (Ipv4Address src, Ipv4Address dst, uint16_t id);

private:
  /// The max request period among requests
  Time  MaxRequestPeriod;
  /// The original request period
  Time  RequestPeriod;
  /// The non-propagaton request timeout
  Time  NonpropRequestTimeout;
  /// The source route entry expire time
  Time m_rreqEntryExpire;
  /// The initial hop limit
  uint32_t m_initHopLimit;
  /// The request table size
  uint32_t m_requestTableSize;
  /// The request source id size
  uint32_t m_requestIdSize;
  /// The unique request id for any destination
  uint32_t m_maxRreqId;
  /// The state of the unidirectional link
  LinkStates m_linkStates;
  /// Map of entries
  std::list<DsrReceivedRreqEntry> m_sourceRequests;
  /// The id cache to ensure all the ids are unique, it is used when sending out route request
  std::map<Ipv4Address, uint32_t> m_rreqIdCache;
  /// The cache to save route request table entries indexed with destination address
  std::map<Ipv4Address, RreqTableEntry > m_rreqDstMap;
  /// The cache to ensure all the route request from unique source
  std::map<Ipv4Address, std::list<DsrReceivedRreqEntry> > m_sourceRreqMap;

  /// The Black list
  std::vector<BlackList> m_blackList;
  /// Check if the entry is expired or not
  struct IsExpired
  {
    /**
     * Check if the entry is expired
     *
     * \param b BlackList entry
     * \return true if expired, false otherwise
     */
    bool operator() (const struct BlackList & b) const
    {
      return (b.m_expireTime < Simulator::Now ());
    }
  };
};
}  // namespace dsr
}  // namespace ns3

#endif /* DSR_RREQ_TABLE_H */
