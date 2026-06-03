/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008,2009 IITP RAS
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
 * Author: Kirill Andreev <andreev@iitp.ru>
 */

#ifndef HWMP_RTABLE_H
#define HWMP_RTABLE_H

#include <map>
#include "ns3/nstime.h"
#include "ns3/mac48-address.h"
#include "ns3/hwmp-protocol.h"
namespace ns3 {
namespace dot11s {
/**
 * \ingroup dot11s
 *
 * \brief Routing table for HWMP -- 802.11s routing protocol
 */
class HwmpRtable : public Object
{
public:
  /// Means all interfaces
  const static uint32_t INTERFACE_ANY = 0xffffffff;
  /// Maximum (the best?) path metric
  const static uint32_t MAX_METRIC = 0xffffffff;

  /// Route lookup result, return type of LookupXXX methods
  struct LookupResult
  {
    Mac48Address retransmitter; ///< retransmitter
    uint32_t ifIndex; ///< IF index
    uint32_t metric; ///< metric
    uint32_t seqnum; ///< sequence number
    Time lifetime; ///< lifetime
    /**
     * Lookup result function
     *
     * \param r the result address
     * \param i the interface
     * \param m the metric
     * \param s the sequence number
     * \param l the lifetime
     */
    LookupResult (Mac48Address r = Mac48Address::GetBroadcast (),
                  uint32_t i = INTERFACE_ANY,
                  uint32_t m = MAX_METRIC,
                  uint32_t s = 0,
                  Time l = Seconds (0.0));
    /**
     * \returns True for valid route
     */
    bool IsValid () const;
    /**
     * Compare route lookup results, used by tests
     * \param o the lookup result to compare
     * \returns true if equal
     */
    bool operator== (const LookupResult & o) const;
  };
  /// Path precursor = {MAC, interface ID}
  typedef std::vector<std::pair<uint32_t, Mac48Address> > PrecursorList;

public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();
  HwmpRtable ();
  ~HwmpRtable ();
  void DoDispose ();

  /// \name Add/delete paths
  ///@{

  /**
   * Add a reactive path
   * \param destination the destination address
   * \param retransmitter the retransmitter address
   * \param interface the interface
   * \param metric  the metric
   * \param lifetime  the lifetime
   * \param seqnum  the sequence number
   */
  void AddReactivePath (
      Mac48Address destination,
      Mac48Address retransmitter,
      uint32_t interface,
      uint32_t metric,
      Time  lifetime,
      uint32_t seqnum
  );
  /**
   * Add a proactive path
   * \param metric  the metric
   * \param root the address of the root
   * \param retransmitter the retransmitter address
   * \param interface the interface
   * \param lifetime  the lifetime
   * \param seqnum  the sequence number
   */
  void AddProactivePath (
      uint32_t metric,
      Mac48Address root,
      Mac48Address retransmitter,
      uint32_t interface,
      Time  lifetime,
      uint32_t seqnum
  );
   /**
    * Add a precursor
   * \param destination the destination address
    * \param precursorInterface the precursor interface
    * \param precursorAddress the address of the precursor
    * \param lifetime the lifetime
    */
  void AddPrecursor (Mac48Address destination, uint32_t precursorInterface, Mac48Address precursorAddress, Time lifetime);

  /**
   * Get the precursors list
   * \param destination the destination
   * \return the precursors list
   */
  PrecursorList GetPrecursors (Mac48Address destination);

  /**
   * Delete all the proactive paths
   */
  void DeleteProactivePath ();
  /**
   * Delete all the proactive paths from a given root
   * \param root the address of the root
   */
  void DeleteProactivePath (Mac48Address root);
  /**
   * Delete the reactive paths toward a destination
   * \param destination the destination
   */
  void DeleteReactivePath (Mac48Address destination);
  ///@}

  /// \name Lookup
  ///@{
  /**
   * Lookup path to destination
   * \param destination the destination
   * \return The lookup result
   */
  LookupResult LookupReactive (Mac48Address destination);
  /**
   * Return all reactive paths, including expired
   * \param destination the destination
   * \return The lookup result
   */
  LookupResult LookupReactiveExpired (Mac48Address destination);
  /**
   * Find proactive path to tree root. Note that calling this method has side effect of deleting expired proactive path
   * \return The lookup result
   */
  LookupResult LookupProactive ();
  /**
   * Return all proactive paths, including expired
   * \return The lookup result
   */
  LookupResult LookupProactiveExpired ();
  ///@}

  /**
   * When peer link with a given MAC-address fails - it returns list of unreachable destination addresses
   * \param peerAddress the peer address
   * \returns the list of unreachable destinations
   */
  std::vector<HwmpProtocol::FailedDestination> GetUnreachableDestinations (Mac48Address peerAddress);

private:
  /// Route found in reactive mode
  struct Precursor
  {
    Mac48Address address; ///< address
    uint32_t interface; ///< interface
    Time whenExpire; ///< expire time
  };
  /// Route found in reactive mode
  struct ReactiveRoute
  {
    Mac48Address retransmitter; ///< transmitter
    uint32_t interface; ///< interface
    uint32_t metric; ///< metric
    Time whenExpire; ///< expire time
    uint32_t seqnum; ///< sequence number
    std::vector<Precursor> precursors; ///< precursors
  };
  /// Route found in proactive mode
  struct ProactiveRoute
  {
    Mac48Address root; ///< root
    Mac48Address retransmitter; ///< retransmitter
    uint32_t interface; ///< interface
    uint32_t metric; ///< metric
    Time whenExpire; ///< expire time
    uint32_t seqnum; ///< sequence number
    std::vector<Precursor> precursors; ///< precursors
  };

  /// List of routes
  std::map<Mac48Address, ReactiveRoute>  m_routes;
  /// Path to proactive tree root MP
  ProactiveRoute  m_root;
};
} // namespace dot11s
} // namespace ns3
#endif
