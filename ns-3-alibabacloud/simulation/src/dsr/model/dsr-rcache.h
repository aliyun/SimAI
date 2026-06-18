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
 *              Song Luan <lsuper@mail.ustc.edu.cn> (Implemented Link Cache using dijsktra algorithm to get the best route)
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

#ifndef DSR_RCACHE_H
#define DSR_RCACHE_H

#include <map>
#include <stdint.h>
#include <cassert>
#include <sys/types.h>
#include <iostream>
#include <vector>

#include "ns3/simulator.h"
#include "ns3/timer.h"
#include "ns3/simple-ref-count.h"
#include "ns3/header.h"
#include "ns3/enum.h"
#include "ns3/ipv4-address.h"
#include "ns3/nstime.h"
#include "ns3/ipv4.h"
#include "ns3/ipv4-route.h"
#include "ns3/net-device.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/callback.h"
#include "ns3/arp-cache.h"
#include "dsr-option-header.h"

namespace ns3 {

class Time;
class WifiMacHeader;

namespace dsr {

/**
 * The route cache structure
  \verbatim
  +-+-+-+-+-+-+-+-+-+-+-+-         +-+-+-+-+-+-+-+-+-+-+-            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
  |  Destination Address |---------| Route Cache Entry  | ---------- | IP_VECTOR | dst | exp time |
  +-+-+-+-+-+-+-+-+-+-+-+-   Map   +-+-+-+-+-+-+-+-+-+-+-  Contains  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
                                   +-+-+-+-+-+-+-+-+-+-+-            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
                                   | Route Cache Entry  | ---------- | IP_VECTOR | dst | exp time |
                                   +-+-+-+-+-+-+-+-+-+-+-  Contains  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
                                             .                                       .
                                             .                                       .
                                             .                                       .
                                             .                                       .
                                   +-+-+-+-+-+-+-+-+-+-+-            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
                                   | Route Cache Entry  | ---------- | IP_VECTOR | dst | exp time |
                                   +-+-+-+-+-+-+-+-+-+-+-  Contains  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

  \endverbatim
 */
/**
 * \ingroup dsr
 * \brief DSR Route Cache Entry
 */
struct Link
{
  Ipv4Address m_low; ///< low IP address
  Ipv4Address m_high; ///< high IP address
  /**
   * Constructor
   *
   * \param ip1 first IP address
   * \param ip2 second IP address
   */
  Link (Ipv4Address ip1, Ipv4Address ip2)
  {
    if (ip1 < ip2)
      {
        m_low = ip1;
        m_high = ip2;
      }
    else
      {
        m_low = ip2;
        m_high = ip1;
      }
  }
/**
 * \brief less than comparison operator
 * \param L link to compare
 * \return true if less than
 */
  bool operator < (Link const& L) const
  {
    if (m_low < L.m_low)
      {
        return true;
      }
    else if (m_low == L.m_low)
      {
        return (m_high < L.m_high);
      }
    else
      {
        return false;
      }
  }
  /// Print function
  void Print () const;
};


/**
 * \class DsrLinkStab
 * \brief DsrLinkStab class (DSR link stability)
 */
class DsrLinkStab
{
public:
  /**
   * \brief Constructor
   * \param linkStab duration of the link stability
   */
  DsrLinkStab (Time linkStab = Simulator::Now ());
  /**
   * \brief Destructor
   */
  virtual ~DsrLinkStab ();

  /**
   * \brief set the link stability
   * \param linkStab duration of the link stability
   */
  void SetLinkStability (Time linkStab)
  {
    m_linkStability = linkStab + Simulator::Now ();
  }
  /**
   * \brief get the link stability
   * \returns remaining duration of the link stability
   */
  Time GetLinkStability () const
  {
    return m_linkStability - Simulator::Now ();
  }

  /// Print function
  void Print () const;

private:
  /**
   * The link stability lifetime expected, when the time is due, the link expires the expiration happens
   * when purge the node and link cache before update them when receiving new information
   */
  Time m_linkStability;
};

/**
 * \class DsrNodeStab
 * \brief DsrNodeStab class (DSR node stability)
 */
class DsrNodeStab
{
public:
  /**
   * Constructor
   *
   * \param nodeStab duration of stable node time
   */
  DsrNodeStab (Time nodeStab = Simulator::Now ());
  virtual ~DsrNodeStab ();

  /**
   * Set node stability
   * \param nodeStab duration of the node stability
   */
  void SetNodeStability (Time nodeStab)
  {
    m_nodeStability = nodeStab + Simulator::Now ();
  }
  /**
   * Get node stability
   * \returns the remaining time for node stability
   */
  Time GetNodeStability () const
  {
    return m_nodeStability - Simulator::Now ();
  }
private:
  Time m_nodeStability; ///< the node stability
};

/**
 * \class DsrRouteCacheEntry
 * \brief DsrRouteCacheEntry class for entries in the route cache
 */
class DsrRouteCacheEntry
{
public:
  typedef std::vector<Ipv4Address> IP_VECTOR;                ///< Define the vector to hold Ip address
  typedef std::vector<Ipv4Address>::iterator Iterator;       ///< Define the iterator

  /**
   * Constructor
   *
   * \param ip IP_VECTOR
   * \param dst destination IPv4 address
   * \param exp expiration time
   */
  DsrRouteCacheEntry (IP_VECTOR const  & ip = IP_VECTOR (), Ipv4Address dst = Ipv4Address (), Time exp = Simulator::Now ());
  virtual ~DsrRouteCacheEntry ();

  /// Mark entry as "down" (i.e. disable it)
  /// \param badLinkLifetime Time before purging the link for real.
  void Invalidate (Time badLinkLifetime);

  // Fields
  /**
   * Set unidirectional flag
   * \param u the unidirectional flag
   */
  void SetUnidirectional (bool u)
  {
    m_blackListState = u;
  }
  /**
   * Get unidirectional flag
   * \returns the unidirectional flag
   */
  bool IsUnidirectional () const
  {
    return m_blackListState;
  }
  /**
   * Set blacklist timeout
   * \param t the blacklist timeout
   */
  void SetBlacklistTimeout (Time t)
  {
    m_blackListTimeout = t;
  }
  /**
   * Get blacklist timeout
   * \returns the blacklist timeout
   */
  Time GetBlacklistTimeout () const
  {
    return m_blackListTimeout;
  }
  /**
   * Get destination address
   * \returns the destination IP address
   */
  Ipv4Address GetDestination () const
  {
    return m_dst;
  }
  /**
   * Set destination address
   * \param d the destination IP address
   */
  void SetDestination (Ipv4Address d)
  {
    m_dst = d;
  }
  /**
   * Get the IP vector
   * \returns the IP vector
   */
  IP_VECTOR GetVector () const
  {
    return m_path;
  }
  /**
   * Sets the IP vector
   * \param v the IP vector
   */
  void SetVector (IP_VECTOR v)
  {
    m_path = v;
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
   * \brief Print necessary fields
   * \param os the output stream
   */
  void Print (std::ostream & os) const;
  /**
   * \brief Compare the route cache entry
   * \param o entry to compare
   * \return true if equal
   */
  bool operator== (DsrRouteCacheEntry const & o) const
  {
    if (m_path.size () != o.m_path.size ())
      {
        NS_ASSERT (false);
        return false;
      }
    IP_VECTOR::const_iterator j = o.m_path.begin ();
    for (IP_VECTOR::const_iterator i = m_path.begin (); i
         != m_path.end (); i++, j++)
      {
        /*
         * Verify if neither the entry are not 0 and they equal to each other
         */
        if (((*i) == 0) || ((*j) == 0))
          {
            return false;
          }
        else if (!((*i) == (*j)) )
          {
            return false;
          }
        else
          {
            return true;
          }
      }
    return false;
  }
  // \}

private:
  Timer m_ackTimer;                                                     ///< RREP_ACK timer
  Ipv4Address m_dst;                                                    ///< The destination Ip address
  IP_VECTOR m_path;                                                     ///< brief The IP address constructed route
  Time m_expire;                                                        ///< Expire time for queue entry
  Ipv4InterfaceAddress m_iface;                                         ///< Output interface address
  uint8_t m_reqCount;                                                   ///< Number of route requests
  bool m_blackListState;                                                ///< Indicate if this entry is in "blacklist"
  Time m_blackListTimeout;                                              ///< Time for which the node is put into the blacklist
  Ptr<Ipv4Route> m_ipv4Route;                                           ///< The Ipv4 route
  Ptr<Ipv4> m_ipv4;                                                     ///< The Ipv4 layer 3
};
/**
 * \ingroup dsr
 * \brief DSR route request queue
 * Since DSR is an on demand routing we queue requests while looking for route.
 */
class DsrRouteCache : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();

  DsrRouteCache ();
  virtual ~DsrRouteCache ();

  // Delete assignment operator to avoid misuse
  DsrRouteCache & operator = (DsrRouteCache const &) = delete;

  /**
   * \brief Remove the aged route cache entries when the route cache is full
   * \param rtVector the route cache to scan.
   */
  void RemoveLastEntry (std::list<DsrRouteCacheEntry> & rtVector);
  /**
   * \brief Define the vector of route entries.
   */
  typedef std::list<DsrRouteCacheEntry::IP_VECTOR> routeVector;

  // Fields
  /**
   * Get subroute indicator
   * \returns true if a subroute exists
   */
  bool GetSubRoute () const
  {
    return m_subRoute;
  }
  /**
   * Set subroute indicator
   * \param subRoute the subroute indicator
   */
  void SetSubRoute (bool subRoute)
  {
    m_subRoute = subRoute;
  }
  /**
   * Get the max queue length
   * \returns the maximum queue length
   */
  uint32_t GetMaxCacheLen () const
  {
    return m_maxCacheLen;
  }
  /**
   * Set the max queue length
   * \param len the maximum queue length
   */
  void SetMaxCacheLen (uint32_t len)
  {
    m_maxCacheLen = len;
  }
  /**
   * Get cache timeout value
   * \returns the cache timeout time
   */
  Time GetCacheTimeout () const
  {
    return RouteCacheTimeout;
  }
  /**
   * Set cache timeout value
   * \param t the cache timeout time
   */
  void SetCacheTimeout (Time t)
  {
    RouteCacheTimeout = t;
  }
  /**
   * Get max entries for each destination
   * \returns the maximum entries for each destination
   */
  uint32_t GetMaxEntriesEachDst () const
  {
    return m_maxEntriesEachDst;
  }
  /**
   * Set max entries for each destination
   * \param entries the maximum entries for each destination
   */
  void SetMaxEntriesEachDst (uint32_t entries)
  {
    m_maxEntriesEachDst = entries;
  }
  /**
   * Get bad link lifetime function
   * \returns the bad link lifetime
   */
  Time GetBadLinkLifetime () const
  {
    return m_badLinkLifetime;
  }
  /**
   * Set bad link lifetime function
   * \param t the bad link lifetime
   */
  void SetBadLinkLifetime (Time t)
  {
    m_badLinkLifetime = t;
  }
  /**
   * Get stability decrease factor
   * \returns the stability decrease factor
   */
  uint64_t GetStabilityDecrFactor () const
  {
    return m_stabilityDecrFactor;
  }
  /**
   * Set stability decrease factor
   * \param decrFactor the stability decrease factor
   */
  void SetStabilityDecrFactor (uint64_t decrFactor)
  {
    m_stabilityDecrFactor = decrFactor;
  }
  /**
   * Get stability increase factor
   * \returns the stability increase factor
   */
  uint64_t GetStabilityIncrFactor () const
  {
    return m_stabilityIncrFactor;
  }
  /**
   * Set stability increase factor
   * \param incrFactor the stability increase factor
   */
  void SetStabilityIncrFactor (uint64_t incrFactor)
  {
    m_stabilityIncrFactor = incrFactor;
  }
  /**
   * Get initial stability
   * \returns the initial stability
   */
  Time GetInitStability () const
  {
    return m_initStability;
  }
  /**
   * Set initial stability
   * \param initStability the initial stability
   */
  void SetInitStability (Time initStability)
  {
    m_initStability = initStability;
  }
  /**
   * Get minimum lifetime
   * \returns the minimum lifetime
   */
  Time GetMinLifeTime () const
  {
    return m_minLifeTime;
  }
  /**
   * Set minimum lifetime
   * \param minLifeTime the minimum lifetime
   */
  void SetMinLifeTime (Time minLifeTime)
  {
    m_minLifeTime = minLifeTime;
  }
  /**
   * Get use extends
   * \returns the use extends time
   */
  Time GetUseExtends () const
  {
    return m_useExtends;
  }
  /**
   * Set use extends
   * \param useExtends the use extends time
   */
  void SetUseExtends (Time useExtends)
  {
    m_useExtends = useExtends;
  }

  /**
   * \brief Update route cache entry if it has been recently used and successfully delivered the data packet
   * \param dst destination address of the route
   * \return true in success
   */
  bool UpdateRouteEntry (Ipv4Address dst);
  /**
   * \brief Add route cache entry if it doesn't yet exist in route cache
   * \param rt route cache entry
   * \return true on success
   */
  bool AddRoute (DsrRouteCacheEntry & rt);
  /**
   * \brief Lookup route cache entry with destination address dst
   * \param id destination address
   * \param rt entry with destination address id, if exists
   * \return true on success
   */
  bool LookupRoute (Ipv4Address id, DsrRouteCacheEntry & rt);
  /**
   * \brief Print the route vector elements
   * \param vec the route vector
   */
  void PrintVector (std::vector<Ipv4Address>& vec);
  /**
   * \brief Print all the route vector elements from the route list
   * \param route the route list
   */
  void PrintRouteVector (std::list<DsrRouteCacheEntry> route);
  /**
   * \brief Find the same route in the route cache
   * \param rt entry with destination address dst, if exists
   * \param rtVector the route vector
   * \return true if same
   */
  bool FindSameRoute (DsrRouteCacheEntry & rt, std::list<DsrRouteCacheEntry> & rtVector);
  /**
   * \brief Delete the route with certain destination address
   * \param dst the destination address of the routes that should be deleted
   * \return true if the route was deleted
   */
  bool DeleteRoute (Ipv4Address dst);
  /**
   * \brief Delete all the routes which includes the link from next hop address that has just been notified
   * as unreachable
   *
   * \param errorSrc The error source address
   * \param unreachNode The unreachable node
   * \param node This node's ip address
   */
  void DeleteAllRoutesIncludeLink (Ipv4Address errorSrc, Ipv4Address unreachNode, Ipv4Address node);
  /// Delete all entries from routing table
  void Clear ()
  {
    m_routeEntryVector.erase (m_routeEntryVector.begin (), m_routeEntryVector.end ());
  }
  /// Delete all outdated entries and invalidate valid entry if Lifetime is expired
  void Purge ();
  /// Print route cache
  /// \param os the output stream
  void Print (std::ostream &os);

  //------------------------------------------------------------------------------------------
  /**
   * \brief Check for duplicate ids and save new entries if the id is not present in the table
   * \param nextHop to check for in cache
   * \return ack ID
   */
  uint16_t CheckUniqueAckId (Ipv4Address nextHop);
  /**
   * \brief Get the ack table size
   * \return ack size
   */
  uint16_t GetAckSize ();

  // --------------------------------------------------------------------------------------------
  /// Structure to manage neighbor state
  struct Neighbor
  {
    Ipv4Address m_neighborAddress; ///< neightbor address
    Mac48Address m_hardwareAddress; ///< neighbor MAC address
    Time m_expireTime; ///< route expire time
    bool close; ///< is route active

    /**
     * Constructor
     *
     * \param ip IP address of neighbor
     * \param mac MAC address of neighbor
     * \param t expiration time
     */
    Neighbor (Ipv4Address ip, Mac48Address mac, Time t)
      : m_neighborAddress (ip),
        m_hardwareAddress (mac),
        m_expireTime (t),
        close (false)
    {
    }

    Neighbor ()
    {
    }              // For Python bindings
  };
  /**
   * \brief Return expire time for neighbor node with address addr, if exists, else return 0.
   * \param addr IP address
   * \return expire time
   */
  Time GetExpireTime (Ipv4Address addr);
  /**
   * \brief Check that node with address addr is neighbor
   * \param addr IP address
   * \return true if neighbor
   */
  bool IsNeighbor (Ipv4Address addr);
  /**
   * \brief Update expire time for entry with address addr, if it exists, else add new entry
   * \param nodeList list of addresses
   * \param expire expiration time
   */
  void UpdateNeighbor (std::vector<Ipv4Address> nodeList, Time expire);
  /**
   * \brief Add to the neighbor list
   * \param nodeList neighbor list
   * \param ownAddress local address
   * \param expire expiration time
   */
  void AddNeighbor (std::vector<Ipv4Address> nodeList, Ipv4Address ownAddress, Time expire);
  /**
   * \brief Remove all expired mac entries
   */
  void PurgeMac ();
  /**
   * \brief Schedule m_ntimer.
   */
  void ScheduleTimer ();
  /**
   * \brief Remove all entries
   */
  void ClearMac ()
  {
    m_nb.clear ();
  }
  /**
   * \brief Add ARP cache to be used to allow layer 2 notifications processing
   * \param a ARP cache
   */
  void AddArpCache (Ptr<ArpCache> a);
  /**
   * \brief Don't use the provided ARP cache any more (interface is down)
   * \param a ARP cache
   */
  void DelArpCache (Ptr<ArpCache>);
  /**
   * \brief Get callback to ProcessTxError, this callback is trying to use the wifi mac tx error header to notify a link layer drop event, however,
   * it is not fully supported yet
   * \return The callback to ProcessTxError
   */
  Callback<void, WifiMacHeader const &> GetTxErrorCallback () const
  {
    return m_txErrorCallback;
  }

  /**
   * Handle link failure callback
   * \param cb the callback to be set
   */
  void SetCallback (Callback<void, Ipv4Address, uint8_t > cb)
  {
    m_handleLinkFailure = cb;
  }
  /**
   * Handle link failure callback
   * \return The callback to LinkFailure
   */
  Callback<void, Ipv4Address, uint8_t > GetCallback () const
  {
    return m_handleLinkFailure;
  }

private:
  DsrRouteCacheEntry::IP_VECTOR m_vector;               ///< The route vector to save the ip addresses for intermediate nodes.
  uint32_t m_maxCacheLen;                               ///< The maximum number of packets that we allow a routing protocol to buffer.
  Time     RouteCacheTimeout;                           ///< The maximum period of time that dsr is allowed to for an unused route.
  Time     m_badLinkLifetime;                           ///< The time for which the neighboring node is put into the blacklist.
  /*
   * Define the parameters for link cache type
   */
  uint32_t m_stabilityDecrFactor; ///< stability decrease factor
  uint32_t m_stabilityIncrFactor; ///< stability increase factor
  Time m_initStability; ///< initial stability
  Time m_minLifeTime; ///< minimum lifetime
  Time m_useExtends; ///< use extend
  /**
   * Define the route cache data structure
   */
  typedef std::list<DsrRouteCacheEntry> routeEntryVector;

  std::map<Ipv4Address, routeEntryVector> m_sortedRoutes;       ///< Map the ipv4Address to route entry vector

  routeEntryVector m_routeEntryVector;                          ///< Define the route vector

  uint32_t m_maxEntriesEachDst;                                 ///< number of entries for each destination

  std::map<Ipv4Address, uint16_t> m_ackIdCache;                 ///< The id cache to ensure all the ids are unique

  bool m_isLinkCache;                                           ///< Check if the route is using path cache or link cache

  bool m_subRoute;                                              ///< Check if save the sub route entries or not
  /**
   * The link cache to update all the link status, bi-link is two link for link is a struct
   * when the weight is calculated we normalized them: 100*weight/max of Weight
   */
  #define MAXWEIGHT 0xFFFF;
  /**
   * Current network graph state for this node, double is weight, which is calculated by the node information
   * and link information, any time some changes of link cache and node cache
   * change the weight and then recompute the best choice for each node
   */
  std::map<Ipv4Address, std::map<Ipv4Address, uint32_t> > m_netGraph;

  std::map<Ipv4Address, DsrRouteCacheEntry::IP_VECTOR> m_bestRoutesTable_link;     ///< for link route cache
  std::map<Link, DsrLinkStab> m_linkCache;                                         ///< The data structure to store link info
  std::map<Ipv4Address, DsrNodeStab> m_nodeCache;                                  ///< The data structure to store node info
  /**
   * \brief used by LookupRoute when LinkCache
   * \param id the ip address we are looking for
   * \param rt the route cache entry to store the found one
   * \return true if route route found
   */
  bool LookupRoute_Link (Ipv4Address id, DsrRouteCacheEntry & rt);
  /**
   * \brief increase the stability of the node
   * \param node the ip address of the node we want to increase stability
   * \return true if success
   */
  bool IncStability (Ipv4Address node);
  /**
   * \brief decrease the stability of the node
   * \param node the ip address of the node we want to decrease stability
   * \return true if success
   */
  bool DecStability (Ipv4Address node);

public:
  /**
   * \brief Dijsktra algorithm to get the best route from m_netGraph and update the m_bestRoutesTable_link
   * when current graph information has changed
   * \param type The type of the cache
   */
  void SetCacheType (std::string type);
  /**
   * \brief is link cached
   * \return true if the link is cached
   */
  bool IsLinkCache ();
  /**
   * \brief dd route link to cache
   * \param nodelist vector of nodes
   * \param node ip address of node to add
   * \return true if the link is cached
   */
  bool AddRoute_Link (DsrRouteCacheEntry::IP_VECTOR nodelist, Ipv4Address node);
  /**
   *  \brief Rebuild the best route table
   *  \note Use MAXWEIGHT to represeant maximum weight, use the IPv4 broadcast
   *        address of 255.255.255.255 to represent a null preceding address
   *  \param source The source address used for computing the routes
   */
  void RebuildBestRouteTable (Ipv4Address source);
  /**
   *  \brief Purge from the cache if the stability time expired
   */
  void PurgeLinkNode ();
  /**
   * When a link from the Route Cache is used in routing a packet originated or salvaged
   * by that node, the stability metric for each of the two endpoint nodes of that link is incremented by the
   * amount of time since that link was last used. When a link is used in a route chosen for a packet originated or
   * salvaged by this node, the link's lifetime is set to be at least UseExtends into the future
   * \param rt cache entry
   */
  void UseExtends (DsrRouteCacheEntry::IP_VECTOR rt);
  /**
   *  \brief Update the Net Graph for the link and node cache has changed
   */
  void UpdateNetGraph ();
  //---------------------------------------------------------------------------------------
  /**
   * The following code handles link-layer acks
   */
  Callback<void, Ipv4Address, uint8_t > m_handleLinkFailure;            ///< link failure callback

  Callback<void, WifiMacHeader const &> m_txErrorCallback;              ///< TX error callback

  Timer m_ntimer;                                                       ///< Timer for neighbor's list. Schedule Purge().

  std::vector<Neighbor> m_nb;                                           ///< vector of entries

  std::vector<Ptr<ArpCache> > m_arp;                                    ///< list of ARP cached to be used for layer 2 notifications processing

  Time m_delay;                                                         ///< This timeout deals with the passive ack

  /// Find MAC address by IP using list of ARP caches
  /// \param addr the IPv4 address to look for
  /// \return The MAC address
  Mac48Address LookupMacAddress (Ipv4Address addr);

  /// Process layer 2 TX error notification
  /// \param hdr Wi-Fi Mac Header
  void ProcessTxError (WifiMacHeader const &hdr);
};
} // namespace dsr
} // namespace ns3
#endif /* DSR_RCACHE_H */
