/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright 2007 University of Washington
 * Copyright (C) 1999, 2000 Kunihiro Ishiguro, Toshiaki Takada
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
 * Authors:  Tom Henderson (tomhend@u.washington.edu)
 * 
 * Kunihiro Ishigura, Toshiaki Takada (GNU Zebra) are attributed authors
 * of the quagga 0.99.7/src/ospfd/ospf_spf.c code which was ported here
 */

#include <utility>
#include <vector>
#include <queue>
#include <algorithm>
#include <iostream>
#include "ns3/assert.h"
#include "ns3/fatal-error.h"
#include "ns3/log.h"
#include "ns3/node-list.h"
#include "ns3/ipv4.h"
#include "ns3/ipv4-routing-protocol.h"
#include "ns3/ipv4-list-routing.h"
#include "global-router-interface.h"
#include "global-route-manager-impl.h"
#include "candidate-queue.h"
#include "ipv4-global-routing.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("GlobalRouteManagerImpl");

/**
 * \brief Stream insertion operator.
 *
 * \param os the reference to the output stream
 * \param exit the exit node
 * \returns the reference to the output stream
 */
std::ostream& 
operator<< (std::ostream& os, const SPFVertex::NodeExit_t& exit)
{
  os << "(" << exit.first << " ," << exit.second << ")";
  return os;
}

std::ostream& 
operator<< (std::ostream& os, const SPFVertex::ListOfSPFVertex_t& vs)
{
  typedef SPFVertex::ListOfSPFVertex_t::const_iterator CIter_t;
  os << "{";
  for (CIter_t iter = vs.begin (); iter != vs.end ();)
    {
      os << (*iter)->m_vertexId;
      if (++iter != vs.end ()) 
        {
          os << ", ";
        }
      else 
        { 
          break;
        }
    }
  os << "}";
  return os;
}

// ---------------------------------------------------------------------------
//
// SPFVertex Implementation
//
// ---------------------------------------------------------------------------

SPFVertex::SPFVertex () : 
  m_vertexType (VertexUnknown), 
  m_vertexId ("255.255.255.255"), 
  m_lsa (0),
  m_distanceFromRoot (SPF_INFINITY), 
  m_rootOif (SPF_INFINITY),
  m_nextHop ("0.0.0.0"),
  m_parents (),
  m_children (),
  m_vertexProcessed (false)
{
  NS_LOG_FUNCTION (this);
}

SPFVertex::SPFVertex (GlobalRoutingLSA* lsa) : 
  m_vertexId (lsa->GetLinkStateId ()),
  m_lsa (lsa),
  m_distanceFromRoot (SPF_INFINITY), 
  m_rootOif (SPF_INFINITY),
  m_nextHop ("0.0.0.0"),
  m_parents (),
  m_children (),
  m_vertexProcessed (false)
{
  NS_LOG_FUNCTION (this << lsa);

  if (lsa->GetLSType () == GlobalRoutingLSA::RouterLSA) 
    {
      NS_LOG_LOGIC ("Setting m_vertexType to VertexRouter");
      m_vertexType = SPFVertex::VertexRouter;
    }
  else if (lsa->GetLSType () == GlobalRoutingLSA::NetworkLSA) 
    { 
      NS_LOG_LOGIC ("Setting m_vertexType to VertexNetwork");
      m_vertexType = SPFVertex::VertexNetwork;
    }
}

SPFVertex::~SPFVertex ()
{
  NS_LOG_FUNCTION (this);

  NS_LOG_LOGIC ("Children vertices - " << m_children);
  NS_LOG_LOGIC ("Parent verteices - " << m_parents);

  // find this node from all its parents and remove the entry of this node
  // from all its parents
  for (ListOfSPFVertex_t::iterator piter = m_parents.begin (); 
       piter != m_parents.end ();
       piter++)
    {
      // remove the current vertex from its parent's children list. Check
      // if the size of the list is reduced, or the child<->parent relation
      // is not bidirectional
      uint32_t orgCount = (*piter)->m_children.size ();
      (*piter)->m_children.remove (this);
      uint32_t newCount = (*piter)->m_children.size ();
      if (orgCount > newCount)
        {
          NS_ASSERT_MSG (orgCount > newCount, "Unable to find the current vertex from its parents --- impossible!");
        }
    }

  // delete children
  while (m_children.size () > 0)
    {
      // pop out children one by one. Some children may disappear 
      // when deleting some other children in the list. As a result,
      // it is necessary to use pop to walk through all children, instead
      // of using iterator.
      //
      // Note that m_children.pop_front () is not necessary as this
      // p is removed from the children list when p is deleted
      SPFVertex* p = m_children.front ();
      // 'p' == 0, this child is already deleted by its other parent
      if (p == 0) continue;
      NS_LOG_LOGIC ("Parent vertex-" << m_vertexId << " deleting its child vertex-" << p->GetVertexId ());
      delete p;
      p = 0;
    }
  m_children.clear ();
  // delete parents
  m_parents.clear ();
  // delete root exit direction
  m_ecmpRootExits.clear ();

  NS_LOG_LOGIC ("Vertex-" << m_vertexId << " completed deleted");
}

void
SPFVertex::SetVertexType (SPFVertex::VertexType type)
{
  NS_LOG_FUNCTION (this << type);
  m_vertexType = type;
}

SPFVertex::VertexType
SPFVertex::GetVertexType (void) const
{
  NS_LOG_FUNCTION (this);
  return m_vertexType;
}

void
SPFVertex::SetVertexId (Ipv4Address id)
{
  NS_LOG_FUNCTION (this << id);
  m_vertexId = id;
}

Ipv4Address
SPFVertex::GetVertexId (void) const
{
  NS_LOG_FUNCTION (this);
  return m_vertexId;
}

void
SPFVertex::SetLSA (GlobalRoutingLSA* lsa)
{
  NS_LOG_FUNCTION (this << lsa);
  m_lsa = lsa;
}

GlobalRoutingLSA*
SPFVertex::GetLSA (void) const
{
  NS_LOG_FUNCTION (this);
  return m_lsa;
}

void
SPFVertex::SetDistanceFromRoot (uint32_t distance)
{
  NS_LOG_FUNCTION (this << distance);
  m_distanceFromRoot = distance;
}

uint32_t
SPFVertex::GetDistanceFromRoot (void) const
{
  NS_LOG_FUNCTION (this);
  return m_distanceFromRoot;
}

void
SPFVertex::SetParent (SPFVertex* parent)
{
  NS_LOG_FUNCTION (this << parent);

  // always maintain only one parent when using setter/getter methods
  m_parents.clear ();
  m_parents.push_back (parent);
}

SPFVertex*
SPFVertex::GetParent (uint32_t i) const
{
  NS_LOG_FUNCTION (this << i);

  // If the index i is out-of-range, return 0 and do nothing
  if (m_parents.size () <= i)
    {
      NS_LOG_LOGIC ("Index to SPFVertex's parent is out-of-range.");
      return 0;
    }
  ListOfSPFVertex_t::const_iterator iter = m_parents.begin ();
  while (i-- > 0) 
    {
      iter++;
    }
  return *iter;
}

void 
SPFVertex::MergeParent (const SPFVertex* v)
{
  NS_LOG_FUNCTION (this << v);

  NS_LOG_LOGIC ("Before merge, list of parents = " << m_parents);
  // combine the two lists first, and then remove any duplicated after
  m_parents.insert (m_parents.end (), 
                    v->m_parents.begin (), v->m_parents.end ());
  // remove duplication
  m_parents.sort ();
  m_parents.unique ();
  NS_LOG_LOGIC ("After merge, list of parents = " << m_parents);
}

void 
SPFVertex::SetRootExitDirection (Ipv4Address nextHop, int32_t id)
{
  NS_LOG_FUNCTION (this << nextHop << id);

  // always maintain only one root's exit
  m_ecmpRootExits.clear ();
  m_ecmpRootExits.push_back (NodeExit_t (nextHop, id));
  // update the following in order to be backward compatitable with
  // GetNextHop and GetOutgoingInterface methods
  m_nextHop = nextHop;
  m_rootOif = id;
}

void 
SPFVertex::SetRootExitDirection (SPFVertex::NodeExit_t exit)
{
  NS_LOG_FUNCTION (this << exit);
  SetRootExitDirection (exit.first, exit.second);
}

SPFVertex::NodeExit_t
SPFVertex::GetRootExitDirection (uint32_t i) const
{
  NS_LOG_FUNCTION (this << i);
  typedef ListOfNodeExit_t::const_iterator CIter_t;

  NS_ASSERT_MSG (i < m_ecmpRootExits.size (), "Index out-of-range when accessing SPFVertex::m_ecmpRootExits!");
  CIter_t iter = m_ecmpRootExits.begin ();
  while (i-- > 0) { iter++; }

  return *iter;
}

SPFVertex::NodeExit_t 
SPFVertex::GetRootExitDirection () const
{
  NS_LOG_FUNCTION (this);

  NS_ASSERT_MSG (m_ecmpRootExits.size () <= 1, "Assumed there is at most one exit from the root to this vertex");
  return GetRootExitDirection (0);
}

void 
SPFVertex::MergeRootExitDirections (const SPFVertex* vertex)
{
  NS_LOG_FUNCTION (this << vertex);

  // obtain the external list of exit directions
  //
  // Append the external list into 'this' and remove duplication afterward
  const ListOfNodeExit_t& extList = vertex->m_ecmpRootExits;
  m_ecmpRootExits.insert (m_ecmpRootExits.end (), 
                          extList.begin (), extList.end ());
  m_ecmpRootExits.sort ();
  m_ecmpRootExits.unique ();
}

void 
SPFVertex::InheritAllRootExitDirections (const SPFVertex* vertex)
{
  NS_LOG_FUNCTION (this << vertex);

  // discard all exit direction currently associated with this vertex,
  // and copy all the exit directions from the given vertex
  if (m_ecmpRootExits.size () > 0)
    {
      NS_LOG_WARN ("x root exit directions in this vertex are going to be discarded");
    }
  m_ecmpRootExits.clear ();
  m_ecmpRootExits.insert (m_ecmpRootExits.end (), 
                          vertex->m_ecmpRootExits.begin (), vertex->m_ecmpRootExits.end ());
}

uint32_t 
SPFVertex::GetNRootExitDirections () const
{
  NS_LOG_FUNCTION (this);
  return m_ecmpRootExits.size ();
}

uint32_t 
SPFVertex::GetNChildren (void) const
{
  NS_LOG_FUNCTION (this);
  return m_children.size ();
}

SPFVertex*
SPFVertex::GetChild (uint32_t n) const
{
  NS_LOG_FUNCTION (this << n);
  uint32_t j = 0;

  for ( ListOfSPFVertex_t::const_iterator i = m_children.begin ();
        i != m_children.end ();
        i++, j++)
    {
      if (j == n)
        {
          return *i;
        }
    }
  NS_ASSERT_MSG (false, "Index <n> out of range.");
  return 0;
}

uint32_t
SPFVertex::AddChild (SPFVertex* child)
{
  NS_LOG_FUNCTION (this << child);
  m_children.push_back (child);
  return m_children.size ();
}

void 
SPFVertex::SetVertexProcessed (bool value)
{
  NS_LOG_FUNCTION (this << value);
  m_vertexProcessed = value;
}

bool 
SPFVertex::IsVertexProcessed (void) const
{
  NS_LOG_FUNCTION (this);
  return m_vertexProcessed;
}

void
SPFVertex::ClearVertexProcessed (void)
{
  NS_LOG_FUNCTION (this);
  for (uint32_t i = 0; i < this->GetNChildren (); i++)
    {
      this->GetChild (i)->ClearVertexProcessed ();
    }
  this->SetVertexProcessed (false);
}

// ---------------------------------------------------------------------------
//
// GlobalRouteManagerLSDB Implementation
//
// ---------------------------------------------------------------------------

GlobalRouteManagerLSDB::GlobalRouteManagerLSDB ()
  :
    m_database (),
    m_extdatabase ()
{
  NS_LOG_FUNCTION (this);
}

GlobalRouteManagerLSDB::~GlobalRouteManagerLSDB ()
{
  NS_LOG_FUNCTION (this);
  LSDBMap_t::iterator i;
  for (i= m_database.begin (); i!= m_database.end (); i++)
    {
      NS_LOG_LOGIC ("free LSA");
      GlobalRoutingLSA* temp = i->second;
      delete temp;
    }
  for (uint32_t j = 0; j < m_extdatabase.size (); j++)
    {
      NS_LOG_LOGIC ("free ASexternalLSA");
      GlobalRoutingLSA* temp = m_extdatabase.at (j);
      delete temp;
    }
  NS_LOG_LOGIC ("clear map");
  m_database.clear ();
}

void
GlobalRouteManagerLSDB::Initialize ()
{
  NS_LOG_FUNCTION (this);
  LSDBMap_t::iterator i;
  for (i= m_database.begin (); i!= m_database.end (); i++)
    {
      GlobalRoutingLSA* temp = i->second;
      temp->SetStatus (GlobalRoutingLSA::LSA_SPF_NOT_EXPLORED);
    }
}

void
GlobalRouteManagerLSDB::Insert (Ipv4Address addr, GlobalRoutingLSA* lsa)
{
  NS_LOG_FUNCTION (this << addr << lsa);
  if (lsa->GetLSType () == GlobalRoutingLSA::ASExternalLSAs) 
    {
      m_extdatabase.push_back (lsa);
    } 
  else
    {
      m_database.insert (LSDBPair_t (addr, lsa));
    }
}

GlobalRoutingLSA*
GlobalRouteManagerLSDB::GetExtLSA (uint32_t index) const
{
  NS_LOG_FUNCTION (this << index);
  return m_extdatabase.at (index);
}

uint32_t
GlobalRouteManagerLSDB::GetNumExtLSAs () const
{
  NS_LOG_FUNCTION (this);
  return m_extdatabase.size ();
}

GlobalRoutingLSA*
GlobalRouteManagerLSDB::GetLSA (Ipv4Address addr) const
{
  NS_LOG_FUNCTION (this << addr);
//
// Look up an LSA by its address.
//
  LSDBMap_t::const_iterator i;
  for (i= m_database.begin (); i!= m_database.end (); i++)
    {
      if (i->first == addr)
        {
          return i->second;
        }
    }
  return 0;
}

GlobalRoutingLSA*
GlobalRouteManagerLSDB::GetLSAByLinkData (Ipv4Address addr) const
{
  NS_LOG_FUNCTION (this << addr);
//
// Look up an LSA by its address.
//
  LSDBMap_t::const_iterator i;
  for (i= m_database.begin (); i!= m_database.end (); i++)
    {
      GlobalRoutingLSA* temp = i->second;
// Iterate among temp's Link Records
      for (uint32_t j = 0; j < temp->GetNLinkRecords (); j++)
        {
          GlobalRoutingLinkRecord *lr = temp->GetLinkRecord (j);
          if ( lr->GetLinkType () == GlobalRoutingLinkRecord::TransitNetwork &&
               lr->GetLinkData () == addr)
            {
              return temp;
            }
        }
    }
  return 0;
}

// ---------------------------------------------------------------------------
//
// GlobalRouteManagerImpl Implementation
//
// ---------------------------------------------------------------------------

GlobalRouteManagerImpl::GlobalRouteManagerImpl () 
  :
    m_spfroot (0)
{
  NS_LOG_FUNCTION (this);
  m_lsdb = new GlobalRouteManagerLSDB ();
}

GlobalRouteManagerImpl::~GlobalRouteManagerImpl ()
{
  NS_LOG_FUNCTION (this);
  if (m_lsdb)
    {
      delete m_lsdb;
    }
}

void
GlobalRouteManagerImpl::DebugUseLsdb (GlobalRouteManagerLSDB* lsdb)
{
  NS_LOG_FUNCTION (this << lsdb);
  if (m_lsdb)
    {
      delete m_lsdb;
    }
  m_lsdb = lsdb;
}

void
GlobalRouteManagerImpl::DeleteGlobalRoutes ()
{
  NS_LOG_FUNCTION (this);
  NodeList::Iterator listEnd = NodeList::End ();
  for (NodeList::Iterator i = NodeList::Begin (); i != listEnd; i++)
    {
      Ptr<Node> node = *i;
      Ptr<GlobalRouter> router = node->GetObject<GlobalRouter> ();
      if (router == 0)
        {
          continue;
        }
      Ptr<Ipv4GlobalRouting> gr = router->GetRoutingProtocol ();
      uint32_t j = 0;
      uint32_t nRoutes = gr->GetNRoutes ();
      NS_LOG_LOGIC ("Deleting " << gr->GetNRoutes ()<< " routes from node " << node->GetId ());
      // Each time we delete route 0, the route index shifts downward
      // We can delete all routes if we delete the route numbered 0
      // nRoutes times
      for (j = 0; j < nRoutes; j++)
        {
          NS_LOG_LOGIC ("Deleting global route " << j << " from node " << node->GetId ());
          gr->RemoveRoute (0);
        }
      NS_LOG_LOGIC ("Deleted " << j << " global routes from node "<< node->GetId ());
    }
  if (m_lsdb)
    {
      NS_LOG_LOGIC ("Deleting LSDB, creating new one");
      delete m_lsdb;
      m_lsdb = new GlobalRouteManagerLSDB ();
    }
}

//
// In order to build the routing database, we need to walk the list of nodes
// in the system and look for those that support the GlobalRouter interface.
// These routers will export a number of Link State Advertisements (LSAs)
// that describe the links and networks that are "adjacent" (i.e., that are
// on the other side of a point-to-point link).  We take these LSAs and put
// add them to the Link State DataBase (LSDB) from which the routes will 
// ultimately be computed.
//
void
GlobalRouteManagerImpl::BuildGlobalRoutingDatabase () 
{
  NS_LOG_FUNCTION (this);
//
// Walk the list of nodes looking for the GlobalRouter Interface.  Nodes with
// global router interfaces are, not too surprisingly, our routers.
//
  NodeList::Iterator listEnd = NodeList::End ();
  for (NodeList::Iterator i = NodeList::Begin (); i != listEnd; i++)
    {
      Ptr<Node> node = *i;

      Ptr<GlobalRouter> rtr = node->GetObject<GlobalRouter> ();
//
// Ignore nodes that aren't participating in routing.
//
      if (!rtr)
        {
          continue;
        }
//
// You must call DiscoverLSAs () before trying to use any routing info or to
// update LSAs.  DiscoverLSAs () drives the process of discovering routes in
// the GlobalRouter.  Afterward, you may use GetNumLSAs (), which is a very
// computationally inexpensive call.  If you call GetNumLSAs () before calling 
// DiscoverLSAs () will get zero as the number since no routes have been 
// found.
//
      Ptr<Ipv4GlobalRouting> grouting = rtr->GetRoutingProtocol ();
      uint32_t numLSAs = rtr->DiscoverLSAs ();
      NS_LOG_LOGIC ("Found " << numLSAs << " LSAs");

      for (uint32_t j = 0; j < numLSAs; ++j)
        {
          GlobalRoutingLSA* lsa = new GlobalRoutingLSA ();
//
// This is the call to actually fetch a Link State Advertisement from the 
// router.
//
          rtr->GetLSA (j, *lsa);
          NS_LOG_LOGIC (*lsa);
//
// Write the newly discovered link state advertisement to the database.
//
          m_lsdb->Insert (lsa->GetLinkStateId (), lsa); 
        }
    }
}

//
// For each node that is a global router (which is determined by the presence
// of an aggregated GlobalRouter interface), run the Dijkstra SPF calculation
// on the database rooted at that router, and populate the node forwarding
// tables.
//
// This function parallels RFC2328, Section 16.1.1, and quagga ospfd
//
// This calculation yields the set of intra-area routes associated
// with an area (called hereafter Area A).  A router calculates the
// shortest-path tree using itself as the root.  The formation
// of the shortest path tree is done here in two stages.  In the
// first stage, only links between routers and transit networks are
// considered.  Using the Dijkstra algorithm, a tree is formed from
// this subset of the link state database.  In the second stage,
// leaves are added to the tree by considering the links to stub
// networks.
//
// The area's link state database is represented as a directed graph.
// The graph's vertices are routers, transit networks and stub networks.
//
// The first stage of the procedure (i.e., the Dijkstra algorithm)
// can now be summarized as follows. At each iteration of the
// algorithm, there is a list of candidate vertices.  Paths from
// the root to these vertices have been found, but not necessarily
// the shortest ones.  However, the paths to the candidate vertex
// that is closest to the root are guaranteed to be shortest; this
// vertex is added to the shortest-path tree, removed from the
// candidate list, and its adjacent vertices are examined for
// possible addition to/modification of the candidate list.  The
// algorithm then iterates again.  It terminates when the candidate
// list becomes empty. 
//
void
GlobalRouteManagerImpl::InitializeRoutes ()
{
  NS_LOG_FUNCTION (this);
//
// Walk the list of nodes in the system.
//
  NS_LOG_INFO ("About to start SPF calculation");
  NodeList::Iterator listEnd = NodeList::End ();
  for (NodeList::Iterator i = NodeList::Begin (); i != listEnd; i++)
    {
      Ptr<Node> node = *i;
//
// Look for the GlobalRouter interface that indicates that the node is
// participating in routing.
//
      Ptr<GlobalRouter> rtr = 
        node->GetObject<GlobalRouter> ();

#ifdef NS3_MPI
      uint32_t systemId = Simulator::GetSystemId ();
      // Ignore nodes that are not assigned to our systemId (distributed sim)
      if (node->GetSystemId () != systemId) 
        {
          continue;
        }
#endif

//
// if the node has a global router interface, then run the global routing
// algorithms.
//
      if (rtr && rtr->GetNumLSAs () )
        {
          SPFCalculate (rtr->GetRouterId ());
        }
    }
  NS_LOG_INFO ("Finished SPF calculation");
}

//
// This method is derived from quagga ospf_spf_next ().  See RFC2328 Section 
// 16.1 (2) for further details.
//
// We're passed a parameter <v> that is a vertex which is already in the SPF
// tree.  A vertex represents a router node.  We also get a reference to the
// SPF candidate queue, which is a priority queue containing the shortest paths
// to the networks we know about.
//
// We examine the links in v's LSA and update the list of candidates with any
// vertices not already on the list.  If a lower-cost path is found to a
// vertex already on the candidate list, store the new (lower) cost.
//
void
GlobalRouteManagerImpl::SPFNext (SPFVertex* v, CandidateQueue& candidate)
{
  NS_LOG_FUNCTION (this << v << &candidate);

  SPFVertex* w = 0;
  GlobalRoutingLSA* w_lsa = 0;
  GlobalRoutingLinkRecord *l = 0;
  uint32_t distance = 0;
  uint32_t numRecordsInVertex = 0;
//
// V points to a Router-LSA or Network-LSA
// Loop over the links in router LSA or attached routers in Network LSA
//
  if (v->GetVertexType () == SPFVertex::VertexRouter)
    {
      numRecordsInVertex = v->GetLSA ()->GetNLinkRecords (); 
    }
  if (v->GetVertexType () == SPFVertex::VertexNetwork)
    {
      numRecordsInVertex = v->GetLSA ()->GetNAttachedRouters (); 
    }

  for (uint32_t i = 0; i < numRecordsInVertex; i++)
    {
// Get w_lsa:  In case of V is Router-LSA
      if (v->GetVertexType () == SPFVertex::VertexRouter) 
        {
          NS_LOG_LOGIC ("Examining link " << i << " of " << 
                        v->GetVertexId () << "'s " <<
                        v->GetLSA ()->GetNLinkRecords () << " link records");
//
// (a) If this is a link to a stub network, examine the next link in V's LSA.
// Links to stub networks will be considered in the second stage of the
// shortest path calculation.
//
          l = v->GetLSA ()->GetLinkRecord (i);
          NS_ASSERT (l != 0);
          if (l->GetLinkType () == GlobalRoutingLinkRecord::StubNetwork)
            {
              NS_LOG_LOGIC ("Found a Stub record to " << l->GetLinkId ());
              continue;
            }
//
// (b) Otherwise, W is a transit vertex (router or transit network).  Look up
// the vertex W's LSA (router-LSA or network-LSA) in Area A's link state
// database. 
//
          if (l->GetLinkType () == GlobalRoutingLinkRecord::PointToPoint)
            {
//
// Lookup the link state advertisement of the new link -- we call it <w> in
// the link state database.
//
              w_lsa = m_lsdb->GetLSA (l->GetLinkId ());
              NS_ASSERT (w_lsa);
              NS_LOG_LOGIC ("Found a P2P record from " << 
                            v->GetVertexId () << " to " << w_lsa->GetLinkStateId ());
            }
          else if (l->GetLinkType () == 
                   GlobalRoutingLinkRecord::TransitNetwork)
            {
              w_lsa = m_lsdb->GetLSA (l->GetLinkId ());
              NS_ASSERT (w_lsa);
              NS_LOG_LOGIC ("Found a Transit record from " << 
                            v->GetVertexId () << " to " << w_lsa->GetLinkStateId ());
            }
          else 
            {
              NS_ASSERT_MSG (0, "illegal Link Type");
            }
        }
// Get w_lsa:  In case of V is Network-LSA
      if (v->GetVertexType () == SPFVertex::VertexNetwork) 
        {
          w_lsa = m_lsdb->GetLSAByLinkData 
              (v->GetLSA ()->GetAttachedRouter (i));
          if (!w_lsa)
            {
              continue;
            }
          NS_LOG_LOGIC ("Found a Network LSA from " << 
                        v->GetVertexId () << " to " << w_lsa->GetLinkStateId ());
        }

// Note:  w_lsa at this point may be either RouterLSA or NetworkLSA
//
// (c) If vertex W is already on the shortest-path tree, examine the next
// link in the LSA.
//
// If the link is to a router that is already in the shortest path first tree
// then we have it covered -- ignore it.
//
      if (w_lsa->GetStatus () == GlobalRoutingLSA::LSA_SPF_IN_SPFTREE) 
        {
          NS_LOG_LOGIC ("Skipping ->  LSA "<< 
                        w_lsa->GetLinkStateId () << " already in SPF tree");
          continue;
        }
//
// (d) Calculate the link state cost D of the resulting path from the root to 
// vertex W.  D is equal to the sum of the link state cost of the (already 
// calculated) shortest path to vertex V and the advertised cost of the link
// between vertices V and W.
//
      if (v->GetLSA ()->GetLSType () == GlobalRoutingLSA::RouterLSA)
        {
          NS_ASSERT (l != 0);
          distance = v->GetDistanceFromRoot () + l->GetMetric ();
        }
      else
        {
          distance = v->GetDistanceFromRoot ();
        }

      NS_LOG_LOGIC ("Considering w_lsa " << w_lsa->GetLinkStateId ());

// Is there already vertex w in candidate list?
      if (w_lsa->GetStatus () == GlobalRoutingLSA::LSA_SPF_NOT_EXPLORED)
        {
// Calculate nexthop to w
// We need to figure out how to actually get to the new router represented
// by <w>.  This will (among other things) find the next hop address to send
// packets destined for this network to, and also find the outbound interface
// used to forward the packets.

// prepare vertex w
          w = new SPFVertex (w_lsa);
          if (SPFNexthopCalculation (v, w, l, distance))
            {
              w_lsa->SetStatus (GlobalRoutingLSA::LSA_SPF_CANDIDATE);
//
// Push this new vertex onto the priority queue (ordered by distance from the
// root node).
//
              candidate.Push (w);
              NS_LOG_LOGIC ("Pushing " << 
                            w->GetVertexId () << ", parent vertexId: " <<
                            v->GetVertexId () << ", distance: " <<
                            w->GetDistanceFromRoot ());
            }
          else
            NS_ASSERT_MSG (0, "SPFNexthopCalculation never " 
                           << "return false, but it does now!");
        }
      else if (w_lsa->GetStatus () == GlobalRoutingLSA::LSA_SPF_CANDIDATE)
        {
//
// We have already considered the link represented by <w>.  What wse have to
// do now is to decide if this new router represents a route with a shorter
// distance metric.
//
// So, locate the vertex in the candidate queue and take a look at the 
// distance.

/* (quagga-0.98.6) W is already on the candidate list; call it cw.
* Compare the previously calculated cost (cw->distance)
* with the cost we just determined (w->distance) to see
* if we've found a shorter path.
*/
          SPFVertex* cw;
          cw = candidate.Find (w_lsa->GetLinkStateId ());
          if (cw->GetDistanceFromRoot () < distance)
            {
//
// This is not a shorter path, so don't do anything.
//
              continue;
            }
          else if (cw->GetDistanceFromRoot () == distance)
            {
//
// This path is one with an equal cost.
//
              NS_LOG_LOGIC ("Equal cost multiple paths found.");

// At this point, there are two instances 'w' and 'cw' of the
// same vertex, the vertex that is currently being considered
// for adding into the shortest path tree. 'w' is the instance
// as seen from the root via vertex 'v', and 'cw' is the instance 
// as seen from the root via some other vertices other than 'v'.
// These two instances are being merged in the following code.
// In particular, the parent nodes, the next hops, and the root's
// output interfaces of the two instances are being merged.
// 
// Note that this is functionally equivalent to calling
// ospf_nexthop_merge (cw->nexthop, w->nexthop) in quagga-0.98.6
// (ospf_spf.c::859), although the detail implementation
// is very different from quagga (blame ns3::GlobalRouteManagerImpl)

// prepare vertex w
              w = new SPFVertex (w_lsa);
              SPFNexthopCalculation (v, w, l, distance);
              cw->MergeRootExitDirections (w);
              cw->MergeParent (w);
// SPFVertexAddParent (w) is necessary as the destructor of 
// SPFVertex checks if the vertex and its parent is linked
// bidirectionally
              SPFVertexAddParent (w);
              delete w;
            }
          else // cw->GetDistanceFromRoot () > w->GetDistanceFromRoot ()
            {
// 
// this path represents a new, lower-cost path to <w> (the vertex we found in
// the current link record of the link state advertisement of the current root
// (vertex <v>)
//
// N.B. the nexthop_calculation is conditional, if it finds a valid nexthop
// it will call spf_add_parents, which will flush the old parents
//
              if (SPFNexthopCalculation (v, cw, l, distance))
                {
//
// If we've changed the cost to get to the vertex represented by <w>, we 
// must reorder the priority queue keyed to that cost.
//
                  candidate.Reorder ();
                }
            } // new lower cost path found
        } // end W is already on the candidate list
    } // end loop over the links in V's LSA
}

//
// This method is derived from quagga ospf_nexthop_calculation() 16.1.1.
//
// Calculate nexthop from root through V (parent) to vertex W (destination)
// with given distance from root->W.
//
// As appropriate, set w's parent, distance, and nexthop information
//
// For now, this is greatly simplified from the quagga code
//
int
GlobalRouteManagerImpl::SPFNexthopCalculation (
  SPFVertex* v, 
  SPFVertex* w,
  GlobalRoutingLinkRecord* l,
  uint32_t distance)
{
  NS_LOG_FUNCTION (this << v << w << l << distance);
//
// If w is a NetworkVertex, l should be null
/*
  if (w->GetVertexType () == SPFVertex::VertexNetwork && l)
    {
        NS_ASSERT_MSG (0, "Error:  SPFNexthopCalculation parameter problem");
    }
*/

//
// The vertex m_spfroot is a distinguished vertex representing the node at
// the root of the calculations.  That is, it is the node for which we are
// calculating the routes.
//
// There are two distinct cases for calculating the next hop information.
// First, if we're considering a hop from the root to an "adjacent" network
// (one that is on the other side of a point-to-point link connected to the
// root), then we need to store the information needed to forward down that
// link.  The second case is if the network is not directly adjacent.  In that
// case we need to use the forwarding information from the vertex on the path
// to the destination that is directly adjacent [node 1] in both cases of the
// diagram below.
// 
// (1) [root] -> [point-to-point] -> [node 1]
// (2) [root] -> [point-to-point] -> [node 1] -> [point-to-point] -> [node 2]
//
// We call the propagation of next hop information down vertices of a path
// "inheriting" the next hop information.
//
// The point-to-point link information is only useful in this calculation when
// we are examining the root node. 
//
  if (v == m_spfroot)
    {
//
// In this case <v> is the root node, which means it is the starting point
// for the packets forwarded by that node.  This also means that the next hop
// address of packets headed for some arbitrary off-network destination must
// be the destination at the other end of one of the links off of the root
// node if this root node is a router.  We then need to see if this node <w>
// is a router.
//
      if (w->GetVertexType () == SPFVertex::VertexRouter) 
        {
//
// In the case of point-to-point links, the link data field (m_linkData) of a
// Global Router Link Record contains the local IP address.  If we look at the
// link record describing the link from the perspecive of <w> (the remote
// node from the viewpoint of <v>) back to the root node, we can discover the
// IP address of the router to which <v> is adjacent.  This is a distinguished
// address -- the next hop address to get from <v> to <w> and all networks 
// accessed through that path.
//
// SPFGetNextLink () is a little odd.  used in this way it is just going to
// return the link record describing the link from <w> to <v>.  Think of it as
// SPFGetLink.
//
          NS_ASSERT (l);
          GlobalRoutingLinkRecord *linkRemote = 0;
          linkRemote = SPFGetNextLink (w, v, linkRemote);
// 
// At this point, <l> is the Global Router Link Record describing the point-
// to point link from <v> to <w> from the perspective of <v>; and <linkRemote>
// is the Global Router Link Record describing that same link from the 
// perspective of <w> (back to <v>).  Now we can just copy the next hop 
// address from the m_linkData member variable.
// 
// The next hop member variable we put in <w> has the sense "in order to get
// from the root node to the host represented by vertex <w>, you have to send
// the packet to the next hop address specified in w->m_nextHop.
//
          Ipv4Address nextHop = linkRemote->GetLinkData ();
// 
// Now find the outgoing interface corresponding to the point to point link
// from the perspective of <v> -- remember that <l> is the link "from"
// <v> "to" <w>.
//
          uint32_t outIf = FindOutgoingInterfaceId (l->GetLinkData ());

          w->SetRootExitDirection (nextHop, outIf);
          w->SetDistanceFromRoot (distance);
          w->SetParent (v);
          NS_LOG_LOGIC ("Next hop from " << 
                        v->GetVertexId () << " to " << w->GetVertexId () <<
                        " goes through next hop " << nextHop <<
                        " via outgoing interface " << outIf <<
                        " with distance " << distance);
        }  // end W is a router vertes
      else 
        {
          NS_ASSERT (w->GetVertexType () == SPFVertex::VertexNetwork);
// W is a directly connected network; no next hop is required
          GlobalRoutingLSA* w_lsa = w->GetLSA ();
          NS_ASSERT (w_lsa->GetLSType () == GlobalRoutingLSA::NetworkLSA);
// Find outgoing interface ID for this network
          uint32_t outIf = FindOutgoingInterfaceId (w_lsa->GetLinkStateId (), 
                                                    w_lsa->GetNetworkLSANetworkMask () );
// Set the next hop to 0.0.0.0 meaning "not exist"
          Ipv4Address nextHop = Ipv4Address::GetZero ();
          w->SetRootExitDirection (nextHop, outIf);
          w->SetDistanceFromRoot (distance);
          w->SetParent (v);
          NS_LOG_LOGIC ("Next hop from " << 
                        v->GetVertexId () << " to network " << w->GetVertexId () <<
                        " via outgoing interface " << outIf <<
                        " with distance " << distance);
          return 1;
        }
    } // end v is the root
  else if (v->GetVertexType () == SPFVertex::VertexNetwork) 
    {
// See if any of v's parents are the root
      if (v->GetParent () == m_spfroot)
        {
// 16.1.1 para 5. ...the parent vertex is a network that
// directly connects the calculating router to the destination
// router.  The list of next hops is then determined by
// examining the destination's router-LSA...
          NS_ASSERT (w->GetVertexType () == SPFVertex::VertexRouter);
          GlobalRoutingLinkRecord *linkRemote = 0;
          while ((linkRemote = SPFGetNextLink (w, v, linkRemote)))
            {
/* ...For each link in the router-LSA that points back to the
 * parent network, the link's Link Data field provides the IP
 * address of a next hop router.  The outgoing interface to
 * use can then be derived from the next hop IP address (or 
 * it can be inherited from the parent network).
 */
              Ipv4Address nextHop = linkRemote->GetLinkData ();
              uint32_t outIf = v->GetRootExitDirection ().second;
              w->SetRootExitDirection (nextHop, outIf);
              NS_LOG_LOGIC ("Next hop from " <<
                            v->GetVertexId () << " to " << w->GetVertexId () <<
                            " goes through next hop " << nextHop <<
                            " via outgoing interface " << outIf);
            }
        }
      else 
        {
          w->SetRootExitDirection (v->GetRootExitDirection ());
        }
    }
  else 
    {
//
// If we're calculating the next hop information from a node (v) that is 
// *not* the root, then we need to "inherit" the information needed to
// forward the packet from the vertex closer to the root.  That is, we'll
// still send packets to the next hop address of the router adjacent to the
// root on the path toward <w>.
//
// Above, when we were considering the root node, we calculated the next hop
// address and outgoing interface required to get off of the root network.
// At this point, we are further away from the root network along one of the
// (shortest) paths.  So the next hop and outoing interface remain the same
// (are inherited).
//
      w->InheritAllRootExitDirections (v);
    }
//
// In all cases, we need valid values for the distance metric and a parent.
//
  w->SetDistanceFromRoot (distance);
  w->SetParent (v);

  return 1;
}

//
// This method is derived from quagga ospf_get_next_link ()
//
// First search the Global Router Link Records of vertex <v> for one
// representing a point-to point link to vertex <w>.
//
// What is done depends on prev_link.  Contrary to appearances, prev_link just
// acts as a flag here.  If prev_link is NULL, we return the first Global
// Router Link Record we find that describes a point-to-point link from <v> 
// to <w>.  If prev_link is not NULL, we return a Global Router Link Record
// representing a possible *second* link from <v> to <w>.
//
GlobalRoutingLinkRecord*
GlobalRouteManagerImpl::SPFGetNextLink (
  SPFVertex* v,
  SPFVertex* w,
  GlobalRoutingLinkRecord* prev_link) 
{
  NS_LOG_FUNCTION (this << v << w << prev_link);

  bool skip = true;
  bool found_prev_link = false;
  GlobalRoutingLinkRecord* l;
//
// If prev_link is 0, we are really looking for the first link, not the next 
// link.
//
  if (prev_link == 0)
    {
      skip = false;
      found_prev_link = true;
    }
//
// Iterate through the Global Router Link Records advertised by the vertex
// <v> looking for records representing the point-to-point links off of this
// vertex.
//
  for (uint32_t i = 0; i < v->GetLSA ()->GetNLinkRecords (); ++i)
    {
      l = v->GetLSA ()->GetLinkRecord (i);
//
// The link ID of a link record representing a point-to-point link is set to
// the router ID of the neighboring router -- the router to which the link
// connects from the perspective of <v> in this case.  The vertex ID is also
// set to the router ID (using the link state advertisement of a router node).
// We're just checking to see if the link <l> is actually the link from <v> to
// <w>.
//
      if (l->GetLinkId () == w->GetVertexId ()) 
        {
          if (!found_prev_link)
            {
              NS_LOG_LOGIC ("Skipping links before prev_link found");
              found_prev_link = true;
              continue;
            }

          NS_LOG_LOGIC ("Found matching link l:  linkId = " <<
                        l->GetLinkId () << " linkData = " << l->GetLinkData ());
//
// If skip is false, don't (not too surprisingly) skip the link found -- it's 
// the one we're interested in.  That's either because we didn't pass in a 
// previous link, and we're interested in the first one, or because we've 
// skipped a previous link and moved forward to the next (which is then the
// one we want).
//
          if (skip == false)
            {
              NS_LOG_LOGIC ("Returning the found link");
              return l;
            }
          else
            {
//
// Skip is true and we've found a link from <v> to <w>.  We want the next one.
// Setting skip to false gets us the next point-to-point global router link
// record in the LSA from <v>.
//
              NS_LOG_LOGIC ("Skipping the found link");
              skip = false;
              continue;
            }
        }
    }
  return 0;
}

//
// Used for unit tests.
//
void
GlobalRouteManagerImpl::DebugSPFCalculate (Ipv4Address root)
{
  NS_LOG_FUNCTION (this << root);
  SPFCalculate (root);
}

//
// Used to test if a node is a stub, from an OSPF sense.
// If there is only one link of type 1 or 2, then a default route
// can safely be added to the next-hop router and SPF does not need
// to be run
//
bool
GlobalRouteManagerImpl::CheckForStubNode (Ipv4Address root)
{
  NS_LOG_FUNCTION (this << root);
  GlobalRoutingLSA *rlsa = m_lsdb->GetLSA (root);
  Ipv4Address myRouterId = rlsa->GetLinkStateId ();
  int transits = 0;
  GlobalRoutingLinkRecord *transitLink = 0;
  for (uint32_t i = 0; i < rlsa->GetNLinkRecords (); i++)
    {
      GlobalRoutingLinkRecord *l = rlsa->GetLinkRecord (i);
      if (l->GetLinkType () == GlobalRoutingLinkRecord::TransitNetwork)
        {
          transits++;
          transitLink = l;
        }
      else if (l->GetLinkType () == GlobalRoutingLinkRecord::PointToPoint)
        {
          transits++;
          transitLink = l;
        }
    }
  if (transits == 0)
    {
      // This router is not connected to any router.  Probably, global
      // routing should not be called for this node, but we can just raise
      // a warning here and return true.
      NS_LOG_WARN ("all nodes should have at least one transit link:" << root );
      return true;
    }
  if (transits == 1)
    {
      if (transitLink->GetLinkType () == GlobalRoutingLinkRecord::TransitNetwork)
        {
          // Install default route to next hop router
          // What is the next hop?  We need to check all neighbors on the link.
          // If there is a single router that has two transit links, then
          // that is the default next hop.  If there are more than one
          // routers on link with multiple transit links, return false.
          // Not yet implemented, so simply return false
          NS_LOG_LOGIC ("TBD: Would have inserted default for transit");
          return false;
        }
      else if (transitLink->GetLinkType () == GlobalRoutingLinkRecord::PointToPoint)
        {
          // Install default route to next hop
          // The link record LinkID is the router ID of the peer.
          // The Link Data is the local IP interface address
          GlobalRoutingLSA *w_lsa = m_lsdb->GetLSA (transitLink->GetLinkId ());
          uint32_t nLinkRecords = w_lsa->GetNLinkRecords ();
          for (uint32_t j = 0; j < nLinkRecords; ++j)
            {
              //
              // We are only concerned about point-to-point links
              //
              GlobalRoutingLinkRecord *lr = w_lsa->GetLinkRecord (j);
              if (lr->GetLinkType () != GlobalRoutingLinkRecord::PointToPoint)
                {
                  continue;
                }
              // Find the link record that corresponds to our routerId
              if (lr->GetLinkId () == myRouterId)
                {
                  // Next hop is stored in the LinkID field of lr
                  Ptr<GlobalRouter> router = rlsa->GetNode ()->GetObject<GlobalRouter> ();
                  NS_ASSERT (router);
                  Ptr<Ipv4GlobalRouting> gr = router->GetRoutingProtocol ();
                  NS_ASSERT (gr);
                  gr->AddNetworkRouteTo (Ipv4Address ("0.0.0.0"), Ipv4Mask ("0.0.0.0"), lr->GetLinkData (), 
                                         FindOutgoingInterfaceId (transitLink->GetLinkData ()));
                  NS_LOG_LOGIC ("Inserting default route for node " << myRouterId << " to next hop " << 
                                lr->GetLinkData () << " via interface " << 
                                FindOutgoingInterfaceId (transitLink->GetLinkData ()));
                  return true;
                }
            }
        }
    }
  return false;
}

// quagga ospf_spf_calculate
void
GlobalRouteManagerImpl::SPFCalculate (Ipv4Address root)
{
  NS_LOG_FUNCTION (this << root);

  SPFVertex *v;
//
// Initialize the Link State Database.
//
  m_lsdb->Initialize ();
//
// The candidate queue is a priority queue of SPFVertex objects, with the top
// of the queue being the closest vertex in terms of distance from the root
// of the tree.  Initially, this queue is empty.
//
  CandidateQueue candidate;
  NS_ASSERT (candidate.Size () == 0);
//
// Initialize the shortest-path tree to only contain the router doing the 
// calculation.  Each router (and corresponding network) is a vertex in the
// shortest path first (SPF) tree.
//
  v = new SPFVertex (m_lsdb->GetLSA (root));
// 
// This vertex is the root of the SPF tree and it is distance 0 from the root.
// We also mark this vertex as being in the SPF tree.
//
  m_spfroot= v;
  v->SetDistanceFromRoot (0);
  v->GetLSA ()->SetStatus (GlobalRoutingLSA::LSA_SPF_IN_SPFTREE);
  NS_LOG_LOGIC ("Starting SPFCalculate for node " << root);

//
// Optimize SPF calculation, for ns-3.
// We do not need to calculate SPF for every node in the network if this
// node has only one interface through which another router can be 
// reached.  Instead, short-circuit this computation and just install
// a default route in the CheckForStubNode() method.
//
  if (NodeList::GetNNodes () > 0 && CheckForStubNode (root))
    {
      NS_LOG_LOGIC ("SPFCalculate truncated for stub node " << root);
      delete m_spfroot;
      return;
    }

  for (;;)
    {
//
// The operations we need to do are given in the OSPF RFC which we reference
// as we go along.
//
// RFC2328 16.1. (2). 
//
// We examine the Global Router Link Records in the Link State 
// Advertisements of the current vertex.  If there are any point-to-point
// links to unexplored adjacent vertices we add them to the tree and update
// the distance and next hop information on how to get there.  We also add
// the new vertices to the candidate queue (the priority queue ordered by
// shortest path).  If the new vertices represent shorter paths, we use them
// and update the path cost.
//
      SPFNext (v, candidate);
//
// RFC2328 16.1. (3). 
//
// If at this step the candidate list is empty, the shortest-path tree (of
// transit vertices) has been completely built and this stage of the
// procedure terminates. 
//
      if (candidate.Size () == 0)
        {
          break;
        }
//
// Choose the vertex belonging to the candidate list that is closest to the
// root, and add it to the shortest-path tree (removing it from the candidate
// list in the process).
//
// Recall that in the previous step, we created SPFVertex structures for each
// of the routers found in the Global Router Link Records and added tehm to 
// the candidate list.
//
      NS_LOG_LOGIC (candidate);
      v = candidate.Pop ();
      NS_LOG_LOGIC ("Popped vertex " << v->GetVertexId ());
//
// Update the status field of the vertex to indicate that it is in the SPF
// tree.
//
      v->GetLSA ()->SetStatus (GlobalRoutingLSA::LSA_SPF_IN_SPFTREE);
//
// The current vertex has a parent pointer.  By calling this rather oddly 
// named method (blame quagga) we add the current vertex to the list of 
// children of that parent vertex.  In the next hop calculation called during
// SPFNext, the parent pointer was set but the vertex has been orphaned up
// to now.
//
      SPFVertexAddParent (v);
//
// Note that when there is a choice of vertices closest to the root, network
// vertices must be chosen before router vertices in order to necessarily
// find all equal-cost paths. 
//
// RFC2328 16.1. (4). 
//
// This is the method that actually adds the routes.  It'll walk the list
// of nodes in the system, looking for the node corresponding to the router
// ID of the root of the tree -- that is the router we're building the routes
// for.  It looks for the Ipv4 interface of that node and remembers it.  So
// we are only actually adding routes to that one node at the root of the SPF 
// tree.
//
// We're going to pop of a pointer to every vertex in the tree except the 
// root in order of distance from the root.  For each of the vertices, we call
// SPFIntraAddRouter ().  Down in SPFIntraAddRouter, we look at all of the 
// point-to-point Global Router Link Records (the links to nodes adjacent to
// the node represented by the vertex).  We add a route to the IP address 
// specified by the m_linkData field of each of those link records.  This will
// be the *local* IP address associated with the interface attached to the 
// link.  We use the outbound interface and next hop information present in 
// the vertex <v> which have possibly been inherited from the root.
//
// To summarize, we're going to look at the node represented by <v> and loop
// through its point-to-point links, adding a *host* route to the local IP
// address (at the <v> side) for each of those links.
//
      if (v->GetVertexType () == SPFVertex::VertexRouter)
        {
          SPFIntraAddRouter (v);
        }
      else if (v->GetVertexType () == SPFVertex::VertexNetwork)
        {
          SPFIntraAddTransit (v);
        }
      else
        {
          NS_ASSERT_MSG (0, "illegal SPFVertex type");
        }
//
// RFC2328 16.1. (5). 
//
// Iterate the algorithm by returning to Step 2 until there are no more
// candidate vertices.

    }  // end for loop

// Second stage of SPF calculation procedure
  SPFProcessStubs (m_spfroot);
  for (uint32_t i = 0; i < m_lsdb->GetNumExtLSAs (); i++)
    {
      m_spfroot->ClearVertexProcessed ();
      GlobalRoutingLSA *extlsa = m_lsdb->GetExtLSA (i);
      NS_LOG_LOGIC ("Processing External LSA with id " << extlsa->GetLinkStateId ());
      ProcessASExternals (m_spfroot, extlsa);
    }

//
// We're all done setting the routing information for the node at the root of
// the SPF tree.  Delete all of the vertices and corresponding resources.  Go
// possibly do it again for the next router.
//
  delete m_spfroot;
  m_spfroot = 0;
}

void
GlobalRouteManagerImpl::ProcessASExternals (SPFVertex* v, GlobalRoutingLSA* extlsa)
{
  NS_LOG_FUNCTION (this << v << extlsa);
  NS_LOG_LOGIC ("Processing external for destination " << 
                extlsa->GetLinkStateId () <<
                ", for router "  << v->GetVertexId () <<
                ", advertised by " << extlsa->GetAdvertisingRouter ());
  if (v->GetVertexType () == SPFVertex::VertexRouter)
    {
      GlobalRoutingLSA *rlsa = v->GetLSA ();
      NS_LOG_LOGIC ("Processing router LSA with id " << rlsa->GetLinkStateId ());
      if ((rlsa->GetLinkStateId ()) == (extlsa->GetAdvertisingRouter ()))
        {
          NS_LOG_LOGIC ("Found advertising router to destination");
          SPFAddASExternal (extlsa,v);
        }
    }
  for (uint32_t i = 0; i < v->GetNChildren (); i++)
    {
      if (!v->GetChild (i)->IsVertexProcessed ())
        {
          NS_LOG_LOGIC ("Vertex's child " << i << " not yet processed, processing...");
          ProcessASExternals (v->GetChild (i), extlsa);
          v->GetChild (i)->SetVertexProcessed (true);
        }
    }
}

//
// Adding external routes to routing table - modeled after
// SPFAddIntraAddStub()
//

void
GlobalRouteManagerImpl::SPFAddASExternal (GlobalRoutingLSA *extlsa, SPFVertex *v)
{
  NS_LOG_FUNCTION (this << extlsa << v);

  NS_ASSERT_MSG (m_spfroot, "GlobalRouteManagerImpl::SPFAddASExternal (): Root pointer not set");
// Two cases to consider: We are advertising the external ourselves
// => No need to add anything
// OR find best path to the advertising router
  if (v->GetVertexId () == m_spfroot->GetVertexId ())
    {
      NS_LOG_LOGIC ("External is on local host: " 
                    << v->GetVertexId () << "; returning");
      return;
    }
  NS_LOG_LOGIC ("External is on remote host: " 
                << extlsa->GetAdvertisingRouter () << "; installing");

  Ipv4Address routerId = m_spfroot->GetVertexId ();

  NS_LOG_LOGIC ("Vertex ID = " << routerId);
//
// We need to walk the list of nodes looking for the one that has the router
// ID corresponding to the root vertex.  This is the one we're going to write
// the routing information to.
//
  NodeList::Iterator i = NodeList::Begin (); 
  NodeList::Iterator listEnd = NodeList::End ();
  for (; i != listEnd; i++)
    {
      Ptr<Node> node = *i;
//
// The router ID is accessible through the GlobalRouter interface, so we need
// to QI for that interface.  If there's no GlobalRouter interface, the node
// in question cannot be the router we want, so we continue.
// 
      Ptr<GlobalRouter> rtr = node->GetObject<GlobalRouter> ();

      if (rtr == 0)
        {
          NS_LOG_LOGIC ("No GlobalRouter interface on node " << node->GetId ());
          continue;
        }
//
// If the router ID of the current node is equal to the router ID of the 
// root of the SPF tree, then this node is the one for which we need to 
// write the routing tables.
//
      NS_LOG_LOGIC ("Considering router " << rtr->GetRouterId ());

      if (rtr->GetRouterId () == routerId)
        {
          NS_LOG_LOGIC ("Setting routes for node " << node->GetId ());
//
// Routing information is updated using the Ipv4 interface.  We need to QI
// for that interface.  If the node is acting as an IP version 4 router, it
// should absolutely have an Ipv4 interface.
//
          Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
          NS_ASSERT_MSG (ipv4, 
                         "GlobalRouteManagerImpl::SPFIntraAddRouter (): "
                         "QI for <Ipv4> interface failed");
//
// Get the Global Router Link State Advertisement from the vertex we're
// adding the routes to.  The LSA will have a number of attached Global Router
// Link Records corresponding to links off of that vertex / node.  We're going
// to be interested in the records corresponding to point-to-point links.
//
          NS_ASSERT_MSG (v->GetLSA (), 
                         "GlobalRouteManagerImpl::SPFIntraAddRouter (): "
                         "Expected valid LSA in SPFVertex* v");
          Ipv4Mask tempmask = extlsa->GetNetworkLSANetworkMask ();
          Ipv4Address tempip = extlsa->GetLinkStateId ();
          tempip = tempip.CombineMask (tempmask);

//
// Here's why we did all of that work.  We're going to add a host route to the
// host address found in the m_linkData field of the point-to-point link
// record.  In the case of a point-to-point link, this is the local IP address
// of the node connected to the link.  Each of these point-to-point links
// will correspond to a local interface that has an IP address to which
// the node at the root of the SPF tree can send packets.  The vertex <v> 
// (corresponding to the node that has these links and interfaces) has 
// an m_nextHop address precalculated for us that is the address to which the
// root node should send packets to be forwarded to these IP addresses.
// Similarly, the vertex <v> has an m_rootOif (outbound interface index) to
// which the packets should be send for forwarding.
//
          Ptr<GlobalRouter> router = node->GetObject<GlobalRouter> ();
          if (router == 0)
            {
              continue;
            }
          Ptr<Ipv4GlobalRouting> gr = router->GetRoutingProtocol ();
          NS_ASSERT (gr);
          // walk through all next-hop-IPs and out-going-interfaces for reaching
          // the stub network gateway 'v' from the root node
          for (uint32_t i = 0; i < v->GetNRootExitDirections (); i++)
            {
              SPFVertex::NodeExit_t exit = v->GetRootExitDirection (i);
              Ipv4Address nextHop = exit.first;
              int32_t outIf = exit.second;
              if (outIf >= 0)
                {
                  gr->AddASExternalRouteTo (tempip, tempmask, nextHop, outIf);
                  NS_LOG_LOGIC ("(Route " << i << ") Node " << node->GetId () <<
                                " add external network route to " << tempip <<
                                " using next hop " << nextHop <<
                                " via interface " << outIf);
                }
              else
                {
                  NS_LOG_LOGIC ("(Route " << i << ") Node " << node->GetId () <<
                                " NOT able to add network route to " << tempip <<
                                " using next hop " << nextHop <<
                                " since outgoing interface id is negative");
                }
            }
          return;
        } // if
    } // for
}


// Processing logic from RFC 2328, page 166 and quagga ospf_spf_process_stubs ()
// stub link records will exist for point-to-point interfaces and for
// broadcast interfaces for which no neighboring router can be found
void
GlobalRouteManagerImpl::SPFProcessStubs (SPFVertex* v)
{
  NS_LOG_FUNCTION (this << v);
  NS_LOG_LOGIC ("Processing stubs for " << v->GetVertexId ());
  if (v->GetVertexType () == SPFVertex::VertexRouter)
    {
      GlobalRoutingLSA *rlsa = v->GetLSA ();
      NS_LOG_LOGIC ("Processing router LSA with id " << rlsa->GetLinkStateId ());
      for (uint32_t i = 0; i < rlsa->GetNLinkRecords (); i++)
        {
          NS_LOG_LOGIC ("Examining link " << i << " of " << 
                        v->GetVertexId () << "'s " <<
                        v->GetLSA ()->GetNLinkRecords () << " link records");
          GlobalRoutingLinkRecord *l = v->GetLSA ()->GetLinkRecord (i);
          if (l->GetLinkType () == GlobalRoutingLinkRecord::StubNetwork)
            {
              NS_LOG_LOGIC ("Found a Stub record to " << l->GetLinkId ());
              SPFIntraAddStub (l, v);
              continue;
            }
        }
    }
  for (uint32_t i = 0; i < v->GetNChildren (); i++)
    {
      if (!v->GetChild (i)->IsVertexProcessed ())
        {
          SPFProcessStubs (v->GetChild (i));
          v->GetChild (i)->SetVertexProcessed (true);
        }
    }
}

// RFC2328 16.1. second stage. 
void
GlobalRouteManagerImpl::SPFIntraAddStub (GlobalRoutingLinkRecord *l, SPFVertex* v)
{
  NS_LOG_FUNCTION (this << l << v);

  NS_ASSERT_MSG (m_spfroot, 
                 "GlobalRouteManagerImpl::SPFIntraAddStub (): Root pointer not set");

  // XXX simplifed logic for the moment.  There are two cases to consider:
  // 1) the stub network is on this router; do nothing for now
  //    (already handled above)
  // 2) the stub network is on a remote router, so I should use the
  // same next hop that I use to get to vertex v
  if (v->GetVertexId () == m_spfroot->GetVertexId ())
    {
      NS_LOG_LOGIC ("Stub is on local host: " << v->GetVertexId () << "; returning");
      return;
    }
  NS_LOG_LOGIC ("Stub is on remote host: " << v->GetVertexId () << "; installing");
//
// The root of the Shortest Path First tree is the router to which we are 
// going to write the actual routing table entries.  The vertex corresponding
// to this router has a vertex ID which is the router ID of that node.  We're
// going to use this ID to discover which node it is that we're actually going
// to update.
//
  Ipv4Address routerId = m_spfroot->GetVertexId ();

  NS_LOG_LOGIC ("Vertex ID = " << routerId);
//
// We need to walk the list of nodes looking for the one that has the router
// ID corresponding to the root vertex.  This is the one we're going to write
// the routing information to.
//
  NodeList::Iterator i = NodeList::Begin (); 
  NodeList::Iterator listEnd = NodeList::End ();
  for (; i != listEnd; i++)
    {
      Ptr<Node> node = *i;
//
// The router ID is accessible through the GlobalRouter interface, so we need
// to QI for that interface.  If there's no GlobalRouter interface, the node
// in question cannot be the router we want, so we continue.
// 
      Ptr<GlobalRouter> rtr = 
        node->GetObject<GlobalRouter> ();

      if (rtr == 0)
        {
          NS_LOG_LOGIC ("No GlobalRouter interface on node " << 
                        node->GetId ());
          continue;
        }
//
// If the router ID of the current node is equal to the router ID of the 
// root of the SPF tree, then this node is the one for which we need to 
// write the routing tables.
//
      NS_LOG_LOGIC ("Considering router " << rtr->GetRouterId ());

      if (rtr->GetRouterId () == routerId)
        {
          NS_LOG_LOGIC ("Setting routes for node " << node->GetId ());
//
// Routing information is updated using the Ipv4 interface.  We need to QI
// for that interface.  If the node is acting as an IP version 4 router, it
// should absolutely have an Ipv4 interface.
//
          Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
          NS_ASSERT_MSG (ipv4, 
                         "GlobalRouteManagerImpl::SPFIntraAddRouter (): "
                         "QI for <Ipv4> interface failed");
//
// Get the Global Router Link State Advertisement from the vertex we're
// adding the routes to.  The LSA will have a number of attached Global Router
// Link Records corresponding to links off of that vertex / node.  We're going
// to be interested in the records corresponding to point-to-point links.
//
          NS_ASSERT_MSG (v->GetLSA (), 
                         "GlobalRouteManagerImpl::SPFIntraAddRouter (): "
                         "Expected valid LSA in SPFVertex* v");
          Ipv4Mask tempmask (l->GetLinkData ().Get ());
          Ipv4Address tempip = l->GetLinkId ();
          tempip = tempip.CombineMask (tempmask);
//
// Here's why we did all of that work.  We're going to add a host route to the
// host address found in the m_linkData field of the point-to-point link
// record.  In the case of a point-to-point link, this is the local IP address
// of the node connected to the link.  Each of these point-to-point links
// will correspond to a local interface that has an IP address to which
// the node at the root of the SPF tree can send packets.  The vertex <v> 
// (corresponding to the node that has these links and interfaces) has 
// an m_nextHop address precalculated for us that is the address to which the
// root node should send packets to be forwarded to these IP addresses.
// Similarly, the vertex <v> has an m_rootOif (outbound interface index) to
// which the packets should be send for forwarding.
//

          Ptr<GlobalRouter> router = node->GetObject<GlobalRouter> ();
          if (router == 0)
            {
              continue;
            }
          Ptr<Ipv4GlobalRouting> gr = router->GetRoutingProtocol ();
          NS_ASSERT (gr);
          // walk through all next-hop-IPs and out-going-interfaces for reaching
          // the stub network gateway 'v' from the root node
          for (uint32_t i = 0; i < v->GetNRootExitDirections (); i++)
            {
              SPFVertex::NodeExit_t exit = v->GetRootExitDirection (i);
              Ipv4Address nextHop = exit.first;
              int32_t outIf = exit.second;
              if (outIf >= 0)
                {
                  gr->AddNetworkRouteTo (tempip, tempmask, nextHop, outIf);
                  NS_LOG_LOGIC ("(Route " << i << ") Node " << node->GetId () <<
                                " add network route to " << tempip <<
                                " using next hop " << nextHop <<
                                " via interface " << outIf);
                }
              else
                {
                  NS_LOG_LOGIC ("(Route " << i << ") Node " << node->GetId () <<
                                " NOT able to add network route to " << tempip <<
                                " using next hop " << nextHop <<
                                " since outgoing interface id is negative");
                }
            }
          return;
        } // if
    } // for
}

//
// Return the interface number corresponding to a given IP address and mask
// This is a wrapper around GetInterfaceForPrefix(), but we first
// have to find the right node pointer to pass to that function.
// If no such interface is found, return -1 (note:  unit test framework
// for routing assumes -1 to be a legal return value)
//
int32_t
GlobalRouteManagerImpl::FindOutgoingInterfaceId (Ipv4Address a, Ipv4Mask amask)
{
  NS_LOG_FUNCTION (this << a << amask);
//
// We have an IP address <a> and a vertex ID of the root of the SPF tree.
// The question is what interface index does this address correspond to.
// The answer is a little complicated since we have to find a pointer to
// the node corresponding to the vertex ID, find the Ipv4 interface on that
// node in order to iterate the interfaces and find the one corresponding to
// the address in question.
//
  Ipv4Address routerId = m_spfroot->GetVertexId ();
//
// Walk the list of nodes in the system looking for the one corresponding to
// the node at the root of the SPF tree.  This is the node for which we are
// building the routing table.
//
  NodeList::Iterator i = NodeList::Begin (); 
  NodeList::Iterator listEnd = NodeList::End ();
  for (; i != listEnd; i++)
    {
      Ptr<Node> node = *i;

      Ptr<GlobalRouter> rtr = 
        node->GetObject<GlobalRouter> ();
//
// If the node doesn't have a GlobalRouter interface it can't be the one
// we're interested in.
//
      if (rtr == 0)
        {
          continue;
        }

      if (rtr->GetRouterId () == routerId)
        {
//
// This is the node we're building the routing table for.  We're going to need
// the Ipv4 interface to look for the ipv4 interface index.  Since this node
// is participating in routing IP version 4 packets, it certainly must have 
// an Ipv4 interface.
//
          Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
          NS_ASSERT_MSG (ipv4, 
                         "GlobalRouteManagerImpl::FindOutgoingInterfaceId (): "
                         "GetObject for <Ipv4> interface failed");
//
// Look through the interfaces on this node for one that has the IP address
// we're looking for.  If we find one, return the corresponding interface
// index, or -1 if not found.
//
          int32_t interface = ipv4->GetInterfaceForPrefix (a, amask);

#if 0
          if (interface < 0)
            {
              NS_FATAL_ERROR ("GlobalRouteManagerImpl::FindOutgoingInterfaceId(): "
                              "Expected an interface associated with address a:" << a);
            }
#endif 
          return interface;
        }
    }
//
// Couldn't find it.
//
  NS_LOG_LOGIC ("FindOutgoingInterfaceId():Can't find root node " << routerId);
  return -1;
}

//
// This method is derived from quagga ospf_intra_add_router ()
//
// This is where we are actually going to add the host routes to the routing
// tables of the individual nodes.
//
// The vertex passed as a parameter has just been added to the SPF tree.
// This vertex must have a valid m_root_oid, corresponding to the outgoing
// interface on the root router of the tree that is the first hop on the path
// to the vertex.  The vertex must also have a next hop address, corresponding
// to the next hop on the path to the vertex.  The vertex has an m_lsa field
// that has some number of link records.  For each point to point link record,
// the m_linkData is the local IP address of the link.  This corresponds to
// a destination IP address, reachable from the root, to which we add a host
// route.
//
void
GlobalRouteManagerImpl::SPFIntraAddRouter (SPFVertex* v)
{
  NS_LOG_FUNCTION (this << v);

  NS_ASSERT_MSG (m_spfroot, 
                 "GlobalRouteManagerImpl::SPFIntraAddRouter (): Root pointer not set");
//
// The root of the Shortest Path First tree is the router to which we are 
// going to write the actual routing table entries.  The vertex corresponding
// to this router has a vertex ID which is the router ID of that node.  We're
// going to use this ID to discover which node it is that we're actually going
// to update.
//
  Ipv4Address routerId = m_spfroot->GetVertexId ();

  NS_LOG_LOGIC ("Vertex ID = " << routerId);
//
// We need to walk the list of nodes looking for the one that has the router
// ID corresponding to the root vertex.  This is the one we're going to write
// the routing information to.
//
  NodeList::Iterator i = NodeList::Begin (); 
  NodeList::Iterator listEnd = NodeList::End ();
  for (; i != listEnd; i++)
    {
      Ptr<Node> node = *i;
//
// The router ID is accessible through the GlobalRouter interface, so we need
// to GetObject for that interface.  If there's no GlobalRouter interface, 
// the node in question cannot be the router we want, so we continue.
// 
      Ptr<GlobalRouter> rtr = 
        node->GetObject<GlobalRouter> ();

      if (rtr == 0)
        {
          NS_LOG_LOGIC ("No GlobalRouter interface on node " << 
                        node->GetId ());
          continue;
        }
//
// If the router ID of the current node is equal to the router ID of the 
// root of the SPF tree, then this node is the one for which we need to 
// write the routing tables.
//
      NS_LOG_LOGIC ("Considering router " << rtr->GetRouterId ());

      if (rtr->GetRouterId () == routerId)
        {
          NS_LOG_LOGIC ("Setting routes for node " << node->GetId ());
//
// Routing information is updated using the Ipv4 interface.  We need to 
// GetObject for that interface.  If the node is acting as an IP version 4 
// router, it should absolutely have an Ipv4 interface.
//
          Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
          NS_ASSERT_MSG (ipv4, 
                         "GlobalRouteManagerImpl::SPFIntraAddRouter (): "
                         "GetObject for <Ipv4> interface failed");
//
// Get the Global Router Link State Advertisement from the vertex we're
// adding the routes to.  The LSA will have a number of attached Global Router
// Link Records corresponding to links off of that vertex / node.  We're going
// to be interested in the records corresponding to point-to-point links.
//
          GlobalRoutingLSA *lsa = v->GetLSA ();
          NS_ASSERT_MSG (lsa, 
                         "GlobalRouteManagerImpl::SPFIntraAddRouter (): "
                         "Expected valid LSA in SPFVertex* v");

          uint32_t nLinkRecords = lsa->GetNLinkRecords ();
//
// Iterate through the link records on the vertex to which we're going to add
// routes.  To make sure we're being clear, we're going to add routing table
// entries to the tables on the node corresping to the root of the SPF tree.
// These entries will have routes to the IP addresses we find from looking at
// the local side of the point-to-point links found on the node described by
// the vertex <v>.
//
          NS_LOG_LOGIC (" Node " << node->GetId () <<
                        " found " << nLinkRecords << " link records in LSA " << lsa << "with LinkStateId "<< lsa->GetLinkStateId ());
          for (uint32_t j = 0; j < nLinkRecords; ++j)
            {
//
// We are only concerned about point-to-point links
//
              GlobalRoutingLinkRecord *lr = lsa->GetLinkRecord (j);
              if (lr->GetLinkType () != GlobalRoutingLinkRecord::PointToPoint)
                {
                  continue;
                }
//
// Here's why we did all of that work.  We're going to add a host route to the
// host address found in the m_linkData field of the point-to-point link
// record.  In the case of a point-to-point link, this is the local IP address
// of the node connected to the link.  Each of these point-to-point links
// will correspond to a local interface that has an IP address to which
// the node at the root of the SPF tree can send packets.  The vertex <v> 
// (corresponding to the node that has these links and interfaces) has 
// an m_nextHop address precalculated for us that is the address to which the
// root node should send packets to be forwarded to these IP addresses.
// Similarly, the vertex <v> has an m_rootOif (outbound interface index) to
// which the packets should be send for forwarding.
//
              Ptr<GlobalRouter> router = node->GetObject<GlobalRouter> ();
              if (router == 0)
                {
                  continue;
                }
              Ptr<Ipv4GlobalRouting> gr = router->GetRoutingProtocol ();
              NS_ASSERT (gr);
              // walk through all available exit directions due to ECMP,
              // and add host route for each of the exit direction toward
              // the vertex 'v'
              for (uint32_t i = 0; i < v->GetNRootExitDirections (); i++)
                {
                  SPFVertex::NodeExit_t exit = v->GetRootExitDirection (i);
                  Ipv4Address nextHop = exit.first;
                  int32_t outIf = exit.second;
                  if (outIf >= 0)
                    {
                      gr->AddHostRouteTo (lr->GetLinkData (), nextHop,
                                          outIf);
                      NS_LOG_LOGIC ("(Route " << i << ") Node " << node->GetId () <<
                                    " adding host route to " << lr->GetLinkData () <<
                                    " using next hop " << nextHop <<
                                    " and outgoing interface " << outIf);
                    }
                  else
                    {
                      NS_LOG_LOGIC ("(Route " << i << ") Node " << node->GetId () <<
                                    " NOT able to add host route to " << lr->GetLinkData () <<
                                    " using next hop " << nextHop <<
                                    " since outgoing interface id is negative " << outIf);
                    }
                } // for all routes from the root the vertex 'v'
            }
//
// Done adding the routes for the selected node.
//
          return;
        }
    }
}
void
GlobalRouteManagerImpl::SPFIntraAddTransit (SPFVertex* v)
{
  NS_LOG_FUNCTION (this << v);

  NS_ASSERT_MSG (m_spfroot, 
                 "GlobalRouteManagerImpl::SPFIntraAddTransit (): Root pointer not set");
//
// The root of the Shortest Path First tree is the router to which we are 
// going to write the actual routing table entries.  The vertex corresponding
// to this router has a vertex ID which is the router ID of that node.  We're
// going to use this ID to discover which node it is that we're actually going
// to update.
//
  Ipv4Address routerId = m_spfroot->GetVertexId ();

  NS_LOG_LOGIC ("Vertex ID = " << routerId);
//
// We need to walk the list of nodes looking for the one that has the router
// ID corresponding to the root vertex.  This is the one we're going to write
// the routing information to.
//
  NodeList::Iterator i = NodeList::Begin (); 
  NodeList::Iterator listEnd = NodeList::End ();
  for (; i != listEnd; i++)
    {
      Ptr<Node> node = *i;
//
// The router ID is accessible through the GlobalRouter interface, so we need
// to GetObject for that interface.  If there's no GlobalRouter interface, 
// the node in question cannot be the router we want, so we continue.
// 
      Ptr<GlobalRouter> rtr = 
        node->GetObject<GlobalRouter> ();

      if (rtr == 0)
        {
          NS_LOG_LOGIC ("No GlobalRouter interface on node " << 
                        node->GetId ());
          continue;
        }
//
// If the router ID of the current node is equal to the router ID of the 
// root of the SPF tree, then this node is the one for which we need to 
// write the routing tables.
//
      NS_LOG_LOGIC ("Considering router " << rtr->GetRouterId ());

      if (rtr->GetRouterId () == routerId)
        {
          NS_LOG_LOGIC ("setting routes for node " << node->GetId ());
//
// Routing information is updated using the Ipv4 interface.  We need to 
// GetObject for that interface.  If the node is acting as an IP version 4 
// router, it should absolutely have an Ipv4 interface.
//
          Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
          NS_ASSERT_MSG (ipv4, 
                         "GlobalRouteManagerImpl::SPFIntraAddTransit (): "
                         "GetObject for <Ipv4> interface failed");
//
// Get the Global Router Link State Advertisement from the vertex we're
// adding the routes to.  The LSA will have a number of attached Global Router
// Link Records corresponding to links off of that vertex / node.  We're going
// to be interested in the records corresponding to point-to-point links.
//
          GlobalRoutingLSA *lsa = v->GetLSA ();
          NS_ASSERT_MSG (lsa, 
                         "GlobalRouteManagerImpl::SPFIntraAddTransit (): "
                         "Expected valid LSA in SPFVertex* v");
          Ipv4Mask tempmask = lsa->GetNetworkLSANetworkMask ();
          Ipv4Address tempip = lsa->GetLinkStateId ();
          tempip = tempip.CombineMask (tempmask);
          Ptr<GlobalRouter> router = node->GetObject<GlobalRouter> ();
          if (router == 0)
            {
              continue;
            }
          Ptr<Ipv4GlobalRouting> gr = router->GetRoutingProtocol ();
          NS_ASSERT (gr);
          // walk through all available exit directions due to ECMP,
          // and add host route for each of the exit direction toward
          // the vertex 'v'
          for (uint32_t i = 0; i < v->GetNRootExitDirections (); i++)
            {
              SPFVertex::NodeExit_t exit = v->GetRootExitDirection (i);
              Ipv4Address nextHop = exit.first;
              int32_t outIf = exit.second;

              if (outIf >= 0)
                {
                  gr->AddNetworkRouteTo (tempip, tempmask, nextHop, outIf);
                  NS_LOG_LOGIC ("(Route " << i << ") Node " << node->GetId () <<
                                " add network route to " << tempip <<
                                " using next hop " << nextHop <<
                                " via interface " << outIf);
                }
              else
                {
                  NS_LOG_LOGIC ("(Route " << i << ") Node " << node->GetId () <<
                                " NOT able to add network route to " << tempip <<
                                " using next hop " << nextHop <<
                                " since outgoing interface id is negative " << outIf);
                }
            }
        }
    } 
}

// Derived from quagga ospf_vertex_add_parents ()
//
// This is a somewhat oddly named method (blame quagga).  Although you might
// expect it to add a parent *to* something, it actually adds a vertex
// to the list of children *in* each of its parents. 
//
// Given a pointer to a vertex, it links back to the vertex's parent that it
// already has set and adds itself to that vertex's list of children.
//
void
GlobalRouteManagerImpl::SPFVertexAddParent (SPFVertex* v)
{
  NS_LOG_FUNCTION (this << v);

  for (uint32_t i=0;;)
    {
      SPFVertex* parent;
      // check if all parents of vertex v
      if ((parent = v->GetParent (i++)) == 0) break;
      parent->AddChild (v);
    }
}

} // namespace ns3


