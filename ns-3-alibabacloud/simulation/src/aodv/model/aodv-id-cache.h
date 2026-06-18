/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 IITP RAS
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
 * Based on
 *      NS-2 AODV model developed by the CMU/MONARCH group and optimized and
 *      tuned by Samir Das and Mahesh Marina, University of Cincinnati;
 *
 *      AODV-UU implementation by Erik Nordström of Uppsala University
 *      http://core.it.uu.se/core/index.php/AODV-UU
 *
 * Authors: Elena Buchatskaia <borovkovaes@iitp.ru>
 *          Pavel Boyko <boyko@iitp.ru>
 */

#ifndef AODV_ID_CACHE_H
#define AODV_ID_CACHE_H

#include "ns3/ipv4-address.h"
#include "ns3/simulator.h"
#include <vector>

namespace ns3 {
namespace aodv {
/**
 * \ingroup aodv
 *
 * \brief Unique packets identification cache used for simple duplicate detection.
 */
class IdCache
{
public:
  /**
   * constructor
   * \param lifetime the lifetime for added entries
   */
  IdCache (Time lifetime) : m_lifetime (lifetime)
  {
  }
  /**
   * Check that entry (addr, id) exists in cache. Add entry, if it doesn't exist.
   * \param addr the IP address
   * \param id the cache entry ID
   * \returns true if the pair exists
   */ 
  bool IsDuplicate (Ipv4Address addr, uint32_t id);
  /// Remove all expired entries
  void Purge ();
  /**
   * \returns number of entries in cache
   */
  uint32_t GetSize ();
  /**
   * Set lifetime for future added entries.
   * \param lifetime the lifetime for entries
   */
  void SetLifetime (Time lifetime)
  {
    m_lifetime = lifetime;
  }
  /**
   * Return lifetime for existing entries in cache
   * \returns thhe lifetime
   */
  Time GetLifeTime () const
  {
    return m_lifetime;
  }
private:
  /// Unique packet ID
  struct UniqueId
  {
    /// ID is supposed to be unique in single address context (e.g. sender address)
    Ipv4Address m_context;
    /// The id
    uint32_t m_id;
    /// When record will expire
    Time m_expire;
  };
  /**
   * \brief IsExpired structure
   */
  struct IsExpired
  {
    /**
     * \brief Check if the entry is expired
     *
     * \param u UniqueId entry
     * \return true if expired, false otherwise
     */
    bool operator() (const struct UniqueId & u) const
    {
      return (u.m_expire < Simulator::Now ());
    }
  };
  /// Already seen IDs
  std::vector<UniqueId> m_idCache;
  /// Default lifetime for ID records
  Time m_lifetime;
};

}  // namespace aodv
}  // namespace ns3

#endif /* AODV_ID_CACHE_H */
