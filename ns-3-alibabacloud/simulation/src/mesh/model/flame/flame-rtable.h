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
 * Author: Kirill Andreev <andreev@iitp.ru>
 */

#ifndef FLAME_RTABLE_H
#define FLAME_RTABLE_H

#include <map>
#include "ns3/nstime.h"
#include "ns3/object.h"
#include "ns3/mac48-address.h"

namespace ns3 {
namespace flame {
/**
 * \ingroup flame
 *
 * \brief Routing table for FLAME
 */
class FlameRtable : public Object
{
public:
  /// Means all interfaces
  const static uint32_t INTERFACE_ANY = 0xffffffff;
  /// Maximum (the best?) path cost
  const static uint32_t MAX_COST = 0xff;

  /// Route lookup result, return type of LookupXXX methods
  struct LookupResult
  {
    Mac48Address retransmitter; ///< retransmitter
    uint32_t ifIndex; ///< IF index
    uint8_t  cost; ///< cost
    uint16_t seqnum; ///< sequence number
    /**
     * Constructor
     *
     * \param r retransmitter MAC address
     * \param i interfce index
     * \param c cost
     * \param s sequence number
     */
    LookupResult (Mac48Address r = Mac48Address::GetBroadcast (),
                  uint32_t i = INTERFACE_ANY,
                  uint8_t  c = MAX_COST,
                  uint16_t s = 0)
      : retransmitter (r),
        ifIndex (i),
        cost (c),
        seqnum (s)
    {
    }
    /**
     * \returns True for valid route
     */
    bool IsValid () const;
    /**
     * Compare route lookup results, used by tests
     *
     * \param o the object to compare
     * \returns true if equal
     */
    bool operator== (const LookupResult & o) const;
  };
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();
  FlameRtable ();
  ~FlameRtable ();
  void DoDispose ();

  /**
   * Add path
   *
   * \param destination the destination address
   * \param retransmitter the retransmitter address
   * \param interface the interface
   * \param cost the cost
   * \param seqnum the sequence number
   */
  void AddPath (
    const Mac48Address destination,
    const Mac48Address retransmitter,
    const uint32_t interface,
    const uint8_t cost,
    const uint16_t seqnum
    );
  /**
   * \brief Lookup path to destination
   * \param destination
   * \return Broadcast if not found
   */
  LookupResult Lookup (Mac48Address destination);
private:
  /**
   * assignment operator
   *
   * \param table the flame table to assign
   * \returns the assigned value
   */
  FlameRtable& operator= (const FlameRtable & table);
  /// type conversion operator
  FlameRtable (const FlameRtable &);

  /// Routing table entry
  struct Route
  {
    Mac48Address retransmitter; ///< retransmitter
    uint32_t interface; ///< interface
    uint32_t cost; ///< cost
    Time whenExpire; ///< expire when?
    uint32_t seqnum; ///< sequence number
  };
  /// Lifetime parameter
  Time m_lifetime;
  /// List of routes
  std::map<Mac48Address, Route>  m_routes;
};

} // namespace flame
} // namespace ns3
#endif /* FLAME_PROTOCOL_H */
