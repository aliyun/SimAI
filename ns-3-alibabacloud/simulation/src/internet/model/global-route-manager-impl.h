/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright 2007 University of Washington
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
 * Authors:  Craig Dowell (craigdo@ee.washington.edu)
 *           Tom Henderson (tomhend@u.washington.edu)
 */

#ifndef GLOBAL_ROUTE_MANAGER_IMPL_H
#define GLOBAL_ROUTE_MANAGER_IMPL_H

#include <stdint.h>
#include <list>
#include <queue>
#include <map>
#include <vector>
#include "ns3/object.h"
#include "ns3/ptr.h"
#include "ns3/ipv4-address.h"
#include "global-router-interface.h"

namespace ns3 {

const uint32_t SPF_INFINITY = 0xffffffff; //!< "infinite" distance between nodes

class CandidateQueue;
class Ipv4GlobalRouting;

/**
 * \ingroup globalrouting
 *
 * @brief Vertex used in shortest path first (SPF) computations. See \RFC{2328},
 * Section 16.
 *
 * Each router in the simulation is associated with an SPFVertex object.  When
 * calculating routes, each of these routers is, in turn, chosen as the "root"
 * of the calculation and routes to all of the other routers are eventually
 * saved in the routing tables of each of the chosen nodes.  Each of these 
 * routers in the calculation has an associated SPFVertex.
 *
 * The "Root" vertex is the SPFVertex representing the router that is having
 * its routing tables set.  The SPFVertex objects representing other routers
 * or networks in the simulation are arranged in the SPF tree.  It is this 
 * tree that represents the Shortest Paths to the other networks.
 *
 * Each SPFVertex has a pointer to the Global Router Link State Advertisement
 * (LSA) that its underlying router has exported.  Within these LSAs are
 * Global Router Link Records that describe the point to point links from the
 * underlying router to other nodes (represented by other SPFVertex objects)
 * in the simulation topology.  The combination of the arrangement of the 
 * SPFVertex objects in the SPF tree, along with the details of the link
 * records that connect them provide the information required to construct the
 * required routes.
 */
class SPFVertex
{
public:
/**
 * @brief Enumeration of the possible types of SPFVertex objects.
 *
 * Currently we use VertexRouter to identify objects that represent a router 
 * in the simulation topology, and VertexNetwork to identify objects that 
 * represent a network.
 */
  enum VertexType {
    VertexUnknown = 0,  /**< Uninitialized Link Record */
    VertexRouter,       /**< Vertex representing a router in the topology */
    VertexNetwork       /**< Vertex representing a network in the topology */
  };

/**
 * @brief Construct an empty ("uninitialized") SPFVertex (Shortest Path First 
 * Vertex).
 *
 * The Vertex Type is set to VertexUnknown, the Vertex ID is set to 
 * 255.255.255.255, and the distance from root is set to infinity 
 * (UINT32_MAX).  The referenced Link State Advertisement (LSA) is set to 
 * null as is the parent SPFVertex.  The outgoing interface index is set to
 * infinity, the next hop address is set to 0.0.0.0 and the list of children
 * of the SPFVertex is initialized to empty.
 *
 * @see VertexType
 */
  SPFVertex();

/**
 * @brief Construct an initialized SPFVertex (Shortest Path First Vertex).
 *
 * The Vertex Type is initialized to VertexRouter and the Vertex ID is found
 * from the Link State ID of the Link State Advertisement (LSA) passed as a
 * parameter.  The Link State ID is set to the Router ID of the advertising
 * router.  The referenced LSA (m_lsa) is set to the given LSA.  Other than 
 * these members, initialization is as in the default constructor.
 * of the SPFVertex is initialized to empty.
 *
 * @see SPFVertex::SPFVertex ()
 * @see VertexType
 * @see GlobalRoutingLSA
 * @param lsa The Link State Advertisement used for finding initial values.
 */
  SPFVertex(GlobalRoutingLSA* lsa);

/**
 * @brief Destroy an SPFVertex (Shortest Path First Vertex).
 *
 * The children vertices of the SPFVertex are recursively deleted.
 *
 * @see SPFVertex::SPFVertex ()
 */
  ~SPFVertex();

/**
 * @brief Get the Vertex Type field of a SPFVertex object.
 *
 * The Vertex Type describes the kind of simulation object a given SPFVertex
 * represents.
 *
 * @see VertexType
 * @returns The VertexType of the current SPFVertex object.
 */
  VertexType GetVertexType (void) const;

/**
 * @brief Set the Vertex Type field of a SPFVertex object.
 *
 * The Vertex Type describes the kind of simulation object a given SPFVertex
 * represents.
 *
 * @see VertexType
 * @param type The new VertexType for the current SPFVertex object.
 */
  void SetVertexType (VertexType type);

/**
 * @brief Get the Vertex ID field of a SPFVertex object.
 *
 * The Vertex ID uniquely identifies the simulation object a given SPFVertex
 * represents.  Typically, this is the Router ID for SPFVertex objects 
 * representing routers, and comes from the Link State Advertisement of a 
 * router aggregated to a node in the simulation.  These IDs are allocated
 * automatically by the routing environment and look like IP addresses 
 * beginning at 0.0.0.0 and monotonically increasing as new routers are
 * instantiated.
 *
 * @returns The Ipv4Address Vertex ID of the current SPFVertex object.
 */
  Ipv4Address GetVertexId (void) const;

/**
 * @brief Set the Vertex ID field of a SPFVertex object.
 *
 * The Vertex ID uniquely identifies the simulation object a given SPFVertex
 * represents.  Typically, this is the Router ID for SPFVertex objects 
 * representing routers, and comes from the Link State Advertisement of a 
 * router aggregated to a node in the simulation.  These IDs are allocated
 * automatically by the routing environment and look like IP addresses 
 * beginning at 0.0.0.0 and monotonically increase as new routers are
 * instantiated.  This method is an explicit override of the automatically
 * generated value.
 *
 * @param id The new Ipv4Address Vertex ID for the current SPFVertex object.
 */
  void SetVertexId (Ipv4Address id);

/**
 * @brief Get the Global Router Link State Advertisement returned by the 
 * Global Router represented by this SPFVertex during the route discovery 
 * process.
 *
 * @see GlobalRouter
 * @see GlobalRoutingLSA
 * @see GlobalRouter::DiscoverLSAs ()
 * @returns A pointer to the GlobalRoutingLSA found by the router represented
 * by this SPFVertex object.
 */
  GlobalRoutingLSA* GetLSA (void) const;

/**
 * @brief Set the Global Router Link State Advertisement returned by the 
 * Global Router represented by this SPFVertex during the route discovery 
 * process.
 *
 * @see SPFVertex::GetLSA ()
 * @see GlobalRouter
 * @see GlobalRoutingLSA
 * @see GlobalRouter::DiscoverLSAs ()
 * @warning Ownership of the LSA is transferred to the "this" SPFVertex.  You
 * must not delete the LSA after calling this method.
 * @param lsa A pointer to the GlobalRoutingLSA.
 */
  void SetLSA (GlobalRoutingLSA* lsa);

/**
 * @brief Get the distance from the root vertex to "this" SPFVertex object.
 *
 * Each router in the simulation is associated with an SPFVertex object.  When
 * calculating routes, each of these routers is, in turn, chosen as the "root"
 * of the calculation and routes to all of the other routers are eventually
 * saved in the routing tables of each of the chosen nodes.  Each of these 
 * routers in the calculation has an associated SPFVertex.
 *
 * The "Root" vertex is then the SPFVertex representing the router that is
 * having its routing tables set.  The "this" SPFVertex is the vertex to which
 * a route is being calculated from the root.  The distance from the root that
 * we're asking for is the number of hops from the root vertex to the vertex
 * in question.
 *
 * The distance is calculated during route discovery and is stored in a
 * member variable.  This method simply fetches that value.
 *
 * @returns The distance, in hops, from the root SPFVertex to "this" SPFVertex.
 */
  uint32_t GetDistanceFromRoot (void) const;

/**
 * @brief Set the distance from the root vertex to "this" SPFVertex object.
 *
 * Each router in the simulation is associated with an SPFVertex object.  When
 * calculating routes, each of these routers is, in turn, chosen as the "root"
 * of the calculation and routes to all of the other routers are eventually
 * saved in the routing tables of each of the chosen nodes.  Each of these 
 * routers in the calculation has an associated SPFVertex.
 *
 * The "Root" vertex is then the SPFVertex representing the router that is
 * having its routing tables set.  The "this" SPFVertex is the vertex to which
 * a route is being calculated from the root.  The distance from the root that
 * we're asking for is the number of hops from the root vertex to the vertex
 * in question.
 *
 * @param distance The distance, in hops, from the root SPFVertex to "this"
 * SPFVertex.
 */
  void SetDistanceFromRoot (uint32_t distance);

/**
 * @brief Set the IP address and outgoing interface index that should be used 
 * to begin forwarding packets from the root SPFVertex to "this" SPFVertex.
 *
 * Each router node in the simulation is associated with an SPFVertex object.
 * When calculating routes, each of these routers is, in turn, chosen as the 
 * "root" of the calculation and routes to all of the other routers are
 * eventually saved in the routing tables of each of the chosen nodes.
 *
 * The "Root" vertex is then the SPFVertex representing the router that is
 * having its routing tables set.  The "this" SPFVertex is the vertex that
 * represents the host or network to which a route is being calculated from 
 * the root.  The IP address that we're asking for is the address on the 
 * remote side of a link off of the root node that should be used as the
 * destination for packets along the path to "this" vertex.
 *
 * When initializing the root SPFVertex, the IP address used when forwarding
 * packets is determined by examining the Global Router Link Records of the
 * Link State Advertisement generated by the root node's GlobalRouter.  This
 * address is used to forward packets off of the root's network down those
 * links.  As other vertices / nodes are discovered which are further away
 * from the root, they will be accessible down one of the paths via a link
 * described by one of these Global Router Link Records.
 * 
 * To forward packets to these hosts or networks, the root node must begin
 * the forwarding process by sending the packets to a first hop router down
 * an interface.  This means that the first hop address and interface ID must
 * be the same for all downstream SPFVertices.  We call this "inheriting"
 * the interface and next hop.
 *
 * In this method we are telling the root node which exit direction it should send
 * should I send a packet to the network or host represented by 'this' SPFVertex.
 *
 * @see GlobalRouter
 * @see GlobalRoutingLSA
 * @see GlobalRoutingLinkRecord
 * @param nextHop The IP address to use when forwarding packets to the host
 * or network represented by "this" SPFVertex.
 * @param id The interface index to use when forwarding packets to the host or
 * network represented by "this" SPFVertex.
 */
  void SetRootExitDirection (Ipv4Address nextHop, int32_t id = SPF_INFINITY);

  typedef std::pair<Ipv4Address, int32_t> NodeExit_t; //!< IPv4 / interface container for exit nodes.

/**
 * @brief Set the IP address and outgoing interface index that should be used 
 * to begin forwarding packets from the root SPFVertex to "this" SPFVertex.
 *
 * Each router node in the simulation is associated with an SPFVertex object.
 * When calculating routes, each of these routers is, in turn, chosen as the 
 * "root" of the calculation and routes to all of the other routers are
 * eventually saved in the routing tables of each of the chosen nodes.
 *
 * The "Root" vertex is then the SPFVertex representing the router that is
 * having its routing tables set.  The "this" SPFVertex is the vertex that
 * represents the host or network to which a route is being calculated from 
 * the root.  The IP address that we're asking for is the address on the 
 * remote side of a link off of the root node that should be used as the
 * destination for packets along the path to "this" vertex.
 *
 * When initializing the root SPFVertex, the IP address used when forwarding
 * packets is determined by examining the Global Router Link Records of the
 * Link State Advertisement generated by the root node's GlobalRouter.  This
 * address is used to forward packets off of the root's network down those
 * links.  As other vertices / nodes are discovered which are further away
 * from the root, they will be accessible down one of the paths via a link
 * described by one of these Global Router Link Records.
 * 
 * To forward packets to these hosts or networks, the root node must begin
 * the forwarding process by sending the packets to a first hop router down
 * an interface.  This means that the first hop address and interface ID must
 * be the same for all downstream SPFVertices.  We call this "inheriting"
 * the interface and next hop.
 *
 * In this method we are telling the root node which exit direction it should send
 * should I send a packet to the network or host represented by 'this' SPFVertex.
 *
 * @see GlobalRouter
 * @see GlobalRoutingLSA
 * @see GlobalRoutingLinkRecord
 * @param exit The pair of next-hop-IP and outgoing-interface-index to use when 
 * forwarding packets to the host or network represented by "this" SPFVertex.
 */
  void SetRootExitDirection (SPFVertex::NodeExit_t exit);
  /**
   * \brief Obtain a pair indicating the exit direction from the root
   *
   * \param i An index to a pair
   * \return A pair of next-hop-IP and outgoing-interface-index for
   * indicating an exit direction from the root. It is 0 if the index 'i'
   * is out-of-range
   */
  NodeExit_t GetRootExitDirection (uint32_t i) const;
  /**
   * \brief Obtain a pair indicating the exit direction from the root
   *
   * This method assumes there is only a single exit direction from the root.
   * Error occur if this assumption is invalid.
   *
   * \return The pair of next-hop-IP and outgoing-interface-index for reaching
   * 'this' vertex from the root
   */
  NodeExit_t GetRootExitDirection () const;
  /**
   * \brief Merge into 'this' vertex the list of exit directions from
   * another vertex
   *
   * This merge is necessary when ECMP are found. 
   *
   * \param vertex From which the list of exit directions are obtain
   * and are merged into 'this' vertex
   */
  void MergeRootExitDirections (const SPFVertex* vertex);
  /**
   * \brief Inherit all root exit directions from a given vertex to 'this' vertex
   * \param vertex The vertex from which all root exit directions are to be inherited
   *
   * After the call of this method, the original root exit directions
   * in 'this' vertex are all lost.
   */
  void InheritAllRootExitDirections (const SPFVertex* vertex);
  /**
   * \brief Get the number of exit directions from root for reaching 'this' vertex
   * \return The number of exit directions from root
   */
  uint32_t GetNRootExitDirections () const;

/**
 * @brief Get a pointer to the SPFVector that is the parent of "this" 
 * SPFVertex.
 *
 * Each router node in the simulation is associated with an SPFVertex object.
 * When calculating routes, each of these routers is, in turn, chosen as the 
 * "root" of the calculation and routes to all of the other routers are
 * eventually saved in the routing tables of each of the chosen nodes.
 *
 * The "Root" vertex is then the SPFVertex representing the router that is
 * having its routing tables set and is the root of the SPF tree.
 *
 * This method returns a pointer to the parent node of "this" SPFVertex
 * (both of which reside in that SPF tree).
 *
 * @param i The index to one of the parents
 * @returns A pointer to the SPFVertex that is the parent of "this" SPFVertex
 * in the SPF tree.
 */
  SPFVertex* GetParent (uint32_t i = 0) const;

/**
 * @brief Set the pointer to the SPFVector that is the parent of "this" 
 * SPFVertex.
 *
 * Each router node in the simulation is associated with an SPFVertex object.
 * When calculating routes, each of these routers is, in turn, chosen as the 
 * "root" of the calculation and routes to all of the other routers are
 * eventually saved in the routing tables of each of the chosen nodes.
 *
 * The "Root" vertex is then the SPFVertex representing the router that is
 * having its routing tables set and is the root of the SPF tree.
 *
 * This method sets the parent pointer of "this" SPFVertex (both of which
 * reside in that SPF tree).
 *
 * @param parent A pointer to the SPFVertex that is the parent of "this" 
 * SPFVertex* in the SPF tree.
 */
  void SetParent (SPFVertex* parent);
  /**
   * \brief Merge the Parent list from the v into this vertex
   *
   * \param v The vertex from which its list of Parent is read
   * and then merged into the list of Parent of *this* vertex.
   * Note that the list in v remains intact
   */
  void MergeParent (const SPFVertex* v);

/**
 * @brief Get the number of children of "this" SPFVertex.
 *
 * Each router node in the simulation is associated with an SPFVertex object.
 * When calculating routes, each of these routers is, in turn, chosen as the 
 * "root" of the calculation and routes to all of the other routers are
 * eventually saved in the routing tables of each of the chosen nodes.
 *
 * The "Root" vertex is then the SPFVertex representing the router that is
 * having its routing tables set and is the root of the SPF tree.  Each vertex
 * in the SPF tree can have a number of children that represent host or 
 * network routes available via that vertex.
 *
 * This method returns the number of children of "this" SPFVertex (which 
 * reside in the SPF tree).
 *
 * @returns The number of children of "this" SPFVertex (which reside in the
 * SPF tree).
 */
  uint32_t GetNChildren (void) const;

/**
 * @brief Get a borrowed SPFVertex pointer to the specified child of "this" 
 * SPFVertex.
 *
 * Each router node in the simulation is associated with an SPFVertex object.
 * When calculating routes, each of these routers is, in turn, chosen as the 
 * "root" of the calculation and routes to all of the other routers are
 * eventually saved in the routing tables of each of the chosen nodes.
 *
 * The "Root" vertex is then the SPFVertex representing the router that is
 * having its routing tables set and is the root of the SPF tree.  Each vertex
 * in the SPF tree can have a number of children that represent host or 
 * network routes available via that vertex.
 *
 * This method the number of children of "this" SPFVertex (which reside in
 * the SPF tree.
 *
 * @see SPFVertex::GetNChildren
 * @param n The index (from 0 to the number of children minus 1) of the 
 * child SPFVertex to return.
 * @warning The pointer returned by GetChild () is a borrowed pointer.  You
 * do not have any ownership of the underlying object and must not delete
 * that object.
 * @returns A pointer to the specified child SPFVertex (which resides in the
 * SPF tree).
 */
  SPFVertex* GetChild (uint32_t n) const;

/**
 * @brief Get a borrowed SPFVertex pointer to the specified child of "this" 
 * SPFVertex.
 *
 * Each router node in the simulation is associated with an SPFVertex object.
 * When calculating routes, each of these routers is, in turn, chosen as the 
 * "root" of the calculation and routes to all of the other routers are
 * eventually saved in the routing tables of each of the chosen nodes.
 *
 * The "Root" vertex is then the SPFVertex representing the router that is
 * having its routing tables set and is the root of the SPF tree.  Each vertex
 * in the SPF tree can have a number of children that represent host or 
 * network routes available via that vertex.
 *
 * This method the number of children of "this" SPFVertex (which reside in
 * the SPF tree.
 *
 * @see SPFVertex::GetNChildren
 * @warning Ownership of the pointer added to the children of "this" 
 * SPFVertex is transferred to the "this" SPFVertex.  You must not delete the
 * (now) child SPFVertex after calling this method.
 * @param child A pointer to the SPFVertex (which resides in the SPF tree) to
 * be added to the list of children of "this" SPFVertex.
 * @returns The number of children of "this" SPFVertex after the addition of
 * the new child.
 */
  uint32_t AddChild (SPFVertex* child);

  /**
   * @brief Set the value of the VertexProcessed flag
   *
   * Flag to note whether vertex has been processed in stage two of 
   * SPF computation
   * @param value boolean value to set the flag
   */ 
  void SetVertexProcessed (bool value);

  /**
   * @brief Check the value of the VertexProcessed flag
   *
   * Flag to note whether vertex has been processed in stage two of 
   * SPF computation
   * @returns value of underlying flag
   */ 
  bool IsVertexProcessed (void) const;

  /**
   * @brief Clear the value of the VertexProcessed flag
   *
   * Flag to note whether vertex has been processed in stage two of
   * SPF computation
   */
  void ClearVertexProcessed (void);

private:
  VertexType m_vertexType; //!< Vertex type
  Ipv4Address m_vertexId; //!< Vertex ID
  GlobalRoutingLSA* m_lsa; //!< Link State Advertisement
  uint32_t m_distanceFromRoot; //!< Distance from root node
  int32_t m_rootOif; //!< root Output Interface
  Ipv4Address m_nextHop; //!< next hop
  typedef std::list< NodeExit_t > ListOfNodeExit_t; //!< container of Exit nodes
  ListOfNodeExit_t m_ecmpRootExits; //!< store the multiple root's exits for supporting ECMP
  typedef std::list<SPFVertex*> ListOfSPFVertex_t; //!< container of SPFVertexes
  ListOfSPFVertex_t m_parents; //!< parent list
  ListOfSPFVertex_t m_children; //!< Children list
  bool m_vertexProcessed; //!< Flag to note whether vertex has been processed in stage two of SPF computation

/**
 * @brief The SPFVertex copy construction is disallowed.  There's no need for
 * it and a compiler provided shallow copy would be wrong.
 * @param v object to copy from
 */
  SPFVertex (SPFVertex& v);

/**
 * @brief The SPFVertex copy assignment operator is disallowed.  There's no 
 * need for it and a compiler provided shallow copy would be wrong.
 * @param v object to copy from
 * @returns the copied object
 */
  SPFVertex& operator= (SPFVertex& v);

  /**
   * \brief Stream insertion operator.
   *
   * \param os the reference to the output stream
   * \param vs a list of SPFVertexes
   * \returns the reference to the output stream
   */
  friend std::ostream& operator<< (std::ostream& os, const SPFVertex::ListOfSPFVertex_t& vs);
};

/**
 * @brief The Link State DataBase (LSDB) of the Global Route Manager.
 *
 * Each node in the simulation participating in global routing has a
 * GlobalRouter interface.  The primary job of this interface is to export
 * Global Router Link State Advertisements (LSAs).  These advertisements in
 * turn contain a number of Global Router Link Records that describe the 
 * point to point links from the underlying node to other nodes (that will 
 * also export their own LSAs.
 *
 * This class implements a searchable database of LSAs gathered from every
 * router in the simulation.
 */
class GlobalRouteManagerLSDB
{
public:
/**
 * @brief Construct an empty Global Router Manager Link State Database.
 *
 * The database map composing the Link State Database is initialized in
 * this constructor.
 */
  GlobalRouteManagerLSDB ();

/**
 * @brief Destroy an empty Global Router Manager Link State Database.
 *
 * The database map is walked and all of the Link State Advertisements stored
 * in the database are freed; then the database map itself is clear ()ed to
 * release any remaining resources.
 */
  ~GlobalRouteManagerLSDB ();

/**
 * @brief Insert an IP address / Link State Advertisement pair into the Link
 * State Database.
 *
 * The IPV4 address and the GlobalRoutingLSA given as parameters are converted
 * to an STL pair and are inserted into the database map.
 *
 * @see GlobalRoutingLSA
 * @see Ipv4Address
 * @param addr The IP address associated with the LSA.  Typically the Router 
 * ID.
 * @param lsa A pointer to the Link State Advertisement for the router.
 */
  void Insert (Ipv4Address addr, GlobalRoutingLSA* lsa);

/**
 * @brief Look up the Link State Advertisement associated with the given
 * link state ID (address).
 *
 * The database map is searched for the given IPV4 address and corresponding
 * GlobalRoutingLSA is returned.
 *
 * @see GlobalRoutingLSA
 * @see Ipv4Address
 * @param addr The IP address associated with the LSA.  Typically the Router 
 * ID.
 * @returns A pointer to the Link State Advertisement for the router specified
 * by the IP address addr.
 */
  GlobalRoutingLSA* GetLSA (Ipv4Address addr) const;
/**
 * @brief Look up the Link State Advertisement associated with the given
 * link state ID (address).  This is a variation of the GetLSA call
 * to allow the LSA to be found by matching addr with the LinkData field
 * of the TransitNetwork link record.
 *
 * @see GetLSA
 * @param addr The IP address associated with the LSA.  Typically the Router 
 * @returns A pointer to the Link State Advertisement for the router specified
 * by the IP address addr.
 * ID.
 */
  GlobalRoutingLSA* GetLSAByLinkData (Ipv4Address addr) const;

/**
 * @brief Set all LSA flags to an initialized state, for SPF computation
 *
 * This function walks the database and resets the status flags of all of the
 * contained Link State Advertisements to LSA_SPF_NOT_EXPLORED.  This is done
 * prior to each SPF calculation to reset the state of the SPFVertex structures
 * that will reference the LSAs during the calculation.
 *
 * @see GlobalRoutingLSA
 * @see SPFVertex
 */
  void Initialize ();

  /**
   * @brief Look up the External Link State Advertisement associated with the given
   * index.
   *
   * The external database map is searched for the given index and corresponding
   * GlobalRoutingLSA is returned.
   *
   * @see GlobalRoutingLSA
   * @param index the index associated with the LSA.
   * @returns A pointer to the Link State Advertisement.
   */
  GlobalRoutingLSA* GetExtLSA (uint32_t index) const;
  /**
   * @brief Get the number of External Link State Advertisements.
   *
   * @see GlobalRoutingLSA
   * @returns the number of External Link State Advertisements.
   */
  uint32_t GetNumExtLSAs () const;


private:
  typedef std::map<Ipv4Address, GlobalRoutingLSA*> LSDBMap_t; //!< container of IPv4 addresses / Link State Advertisements
  typedef std::pair<Ipv4Address, GlobalRoutingLSA*> LSDBPair_t; //!< pair of IPv4 addresses / Link State Advertisements

  LSDBMap_t m_database; //!< database of IPv4 addresses / Link State Advertisements
  std::vector<GlobalRoutingLSA*> m_extdatabase; //!< database of External Link State Advertisements

/**
 * @brief GlobalRouteManagerLSDB copy construction is disallowed.  There's no 
 * need for it and a compiler provided shallow copy would be wrong.
 * @param lsdb object to copy from
 */
  GlobalRouteManagerLSDB (GlobalRouteManagerLSDB& lsdb);

/**
 * @brief The SPFVertex copy assignment operator is disallowed.  There's no 
 * need for it and a compiler provided shallow copy would be wrong.
 * @param lsdb object to copy from
 * @returns the copied object
 */
  GlobalRouteManagerLSDB& operator= (GlobalRouteManagerLSDB& lsdb);
};

/**
 * @brief A global router implementation.
 *
 * This singleton object can query interface each node in the system
 * for a GlobalRouter interface.  For those nodes, it fetches one or
 * more Link State Advertisements and stores them in a local database.
 * Then, it can compute shortest paths on a per-node basis to all routers, 
 * and finally configure each of the node's forwarding tables.
 *
 * The design is guided by OSPFv2 \RFC{2328} section 16.1.1 and quagga ospfd.
 */
class GlobalRouteManagerImpl
{
public:
  GlobalRouteManagerImpl ();
  virtual ~GlobalRouteManagerImpl ();
/**
 * @brief Delete all static routes on all nodes that have a
 * GlobalRouterInterface
 *
 * \todo  separate manually assigned static routes from static routes that
 * the global routing code injects, and only delete the latter
 */
  virtual void DeleteGlobalRoutes ();

/**
 * @brief Build the routing database by gathering Link State Advertisements
 * from each node exporting a GlobalRouter interface.
 */
  virtual void BuildGlobalRoutingDatabase ();

/**
 * @brief Compute routes using a Dijkstra SPF computation and populate
 * per-node forwarding tables
 */
  virtual void InitializeRoutes ();

/**
 * @brief Debugging routine; allow client code to supply a pre-built LSDB
 * @param lsdb the pre-built LSDB
 */
  void DebugUseLsdb (GlobalRouteManagerLSDB* lsdb);

/**
 * @brief Debugging routine; call the core SPF from the unit tests
 * @param root the root node to start calculations
 */
  void DebugSPFCalculate (Ipv4Address root);

private:
/**
 * @brief GlobalRouteManagerImpl copy construction is disallowed.
 * There's no  need for it and a compiler provided shallow copy would be 
 * wrong.
 *
 * @param srmi object to copy from
 */
  GlobalRouteManagerImpl (GlobalRouteManagerImpl& srmi);

/**
 * @brief Global Route Manager Implementation assignment operator is
 * disallowed.  There's no  need for it and a compiler provided shallow copy
 * would be hopelessly wrong.
 *
 * @param srmi object to copy from
 * @returns the copied object
 */
  GlobalRouteManagerImpl& operator= (GlobalRouteManagerImpl& srmi);

  SPFVertex* m_spfroot; //!< the root node
  GlobalRouteManagerLSDB* m_lsdb; //!< the Link State DataBase (LSDB) of the Global Route Manager

  /**
   * \brief Test if a node is a stub, from an OSPF sense.
   *
   * If there is only one link of type 1 or 2, then a default route
   * can safely be added to the next-hop router and SPF does not need
   * to be run
   *
   * \param root the root node
   * \returns true if the node is a stub
   */
  bool CheckForStubNode (Ipv4Address root);

  /**
   * \brief Calculate the shortest path first (SPF) tree
   *
   * Equivalent to quagga ospf_spf_calculate
   * \param root the root node
   */
  void SPFCalculate (Ipv4Address root);

  /**
   * \brief Process Stub nodes
   *
   * Processing logic from RFC 2328, page 166 and quagga ospf_spf_process_stubs ()
   * stub link records will exist for point-to-point interfaces and for
   * broadcast interfaces for which no neighboring router can be found
   *
   * \param v vertex to be processed
   */
  void SPFProcessStubs (SPFVertex* v);

  /**
   * \brief Process Autonomous Systems (AS) External LSA
   *
   * \param v vertex to be processed
   * \param extlsa external LSA
   */
  void ProcessASExternals (SPFVertex* v, GlobalRoutingLSA* extlsa);

  /**
   * \brief Examine the links in v's LSA and update the list of candidates with any
   *        vertices not already on the list
   *
   * \internal
   *
   * This method is derived from quagga ospf_spf_next ().  See RFC2328 Section
   * 16.1 (2) for further details.
   *
   * We're passed a parameter \a v that is a vertex which is already in the SPF
   * tree.  A vertex represents a router node.  We also get a reference to the
   * SPF candidate queue, which is a priority queue containing the shortest paths
   * to the networks we know about.
   *
   * We examine the links in v's LSA and update the list of candidates with any
   * vertices not already on the list.  If a lower-cost path is found to a
   * vertex already on the candidate list, store the new (lower) cost.
   *
   * \param v the vertex
   * \param candidate the SPF candidate queue
   */
  void SPFNext (SPFVertex* v, CandidateQueue& candidate);

  /**
   * \brief Calculate nexthop from root through V (parent) to vertex W (destination)
   *        with given distance from root->W.
   *
   * This method is derived from quagga ospf_nexthop_calculation() 16.1.1.
   * For now, this is greatly simplified from the quagga code
   *
   * \param v the parent
   * \param w the destination
   * \param l the link record
   * \param distance the target distance
   * \returns 1 on success
   */
  int SPFNexthopCalculation (SPFVertex* v, SPFVertex* w, 
                             GlobalRoutingLinkRecord* l, uint32_t distance);

  /**
   * \brief Adds a vertex to the list of children *in* each of its parents
   *
   * Derived from quagga ospf_vertex_add_parents ()
   *
   * This is a somewhat oddly named method (blame quagga).  Although you might
   * expect it to add a parent *to* something, it actually adds a vertex
   * to the list of children *in* each of its parents.
   *
   * Given a pointer to a vertex, it links back to the vertex's parent that it
   * already has set and adds itself to that vertex's list of children.
   *
   * \param v the vertex
   */
  void SPFVertexAddParent (SPFVertex* v);

  /**
   * \brief Search for a link between two vertices.
   *
   * This method is derived from quagga ospf_get_next_link ()
   *
   * First search the Global Router Link Records of vertex \a v for one
   * representing a point-to point link to vertex \a w.
   *
   * What is done depends on prev_link.  Contrary to appearances, prev_link just
   * acts as a flag here.  If prev_link is NULL, we return the first Global
   * Router Link Record we find that describes a point-to-point link from \a v
   * to \a w.  If prev_link is not NULL, we return a Global Router Link Record
   * representing a possible *second* link from \a v to \a w.
   *
   * \param v first vertex
   * \param w second vertex
   * \param prev_link the previous link in the list
   * \returns the link's record
   */
  GlobalRoutingLinkRecord* SPFGetNextLink (SPFVertex* v, SPFVertex* w, 
                                           GlobalRoutingLinkRecord* prev_link);

  /**
   * \brief Add a host route to the routing tables
   *
   *
   * This method is derived from quagga ospf_intra_add_router ()
   *
   * This is where we are actually going to add the host routes to the routing
   * tables of the individual nodes.
   *
   * The vertex passed as a parameter has just been added to the SPF tree.
   * This vertex must have a valid m_root_oid, corresponding to the outgoing
   * interface on the root router of the tree that is the first hop on the path
   * to the vertex.  The vertex must also have a next hop address, corresponding
   * to the next hop on the path to the vertex.  The vertex has an m_lsa field
   * that has some number of link records.  For each point to point link record,
   * the m_linkData is the local IP address of the link.  This corresponds to
   * a destination IP address, reachable from the root, to which we add a host
   * route.
   *
   * \param v the vertex
   *
   */
  void SPFIntraAddRouter (SPFVertex* v);

  /**
   * \brief Add a transit to the routing tables
   *
   * \param v the vertex
   */
  void SPFIntraAddTransit (SPFVertex* v);

  /**
   * \brief Add a stub to the routing tables
   *
   * \param l the global routing link record
   * \param v the vertex
   */
  void SPFIntraAddStub (GlobalRoutingLinkRecord *l, SPFVertex* v);

  /**
   * \brief Add an external route to the routing tables
   *
   * \param extlsa the external LSA
   * \param v the vertex
   */
  void SPFAddASExternal (GlobalRoutingLSA *extlsa, SPFVertex *v);

  /**
   * \brief Return the interface number corresponding to a given IP address and mask
   *
   * This is a wrapper around GetInterfaceForPrefix(), but we first
   * have to find the right node pointer to pass to that function.
   * If no such interface is found, return -1 (note:  unit test framework
   * for routing assumes -1 to be a legal return value)
   *
   * \param a the target IP address
   * \param amask the target subnet mask
   * \return the outgoing interface number
   */
  int32_t FindOutgoingInterfaceId (Ipv4Address a, 
                                   Ipv4Mask amask = Ipv4Mask ("255.255.255.255"));
};

} // namespace ns3

#endif /* GLOBAL_ROUTE_MANAGER_IMPL_H */
