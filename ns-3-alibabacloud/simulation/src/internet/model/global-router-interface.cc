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
 * Authors:  Tom Henderson (tomhend@u.washington.edu)
 */

#include "ns3/log.h"
#include "ns3/assert.h"
#include "ns3/abort.h"
#include "ns3/channel.h"
#include "ns3/net-device.h"
#include "ns3/node.h"
#include "ns3/node-list.h"
#include "ns3/ipv4.h"
#include "ns3/bridge-net-device.h"
#include "ipv4-global-routing.h"
#include "global-router-interface.h"
#include "loopback-net-device.h"
#include <vector>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("GlobalRouter");

// ---------------------------------------------------------------------------
//
// GlobalRoutingLinkRecord Implementation
//
// ---------------------------------------------------------------------------

GlobalRoutingLinkRecord::GlobalRoutingLinkRecord ()
  :
    m_linkId ("0.0.0.0"),
    m_linkData ("0.0.0.0"),
    m_linkType (Unknown),
    m_metric (0)
{
  NS_LOG_FUNCTION (this);
}

GlobalRoutingLinkRecord::GlobalRoutingLinkRecord (
  LinkType    linkType, 
  Ipv4Address linkId, 
  Ipv4Address linkData, 
  uint16_t    metric)
  :
    m_linkId (linkId),
    m_linkData (linkData),
    m_linkType (linkType),
    m_metric (metric)
{
  NS_LOG_FUNCTION (this << linkType << linkId << linkData << metric);
}

GlobalRoutingLinkRecord::~GlobalRoutingLinkRecord ()
{
  NS_LOG_FUNCTION (this);
}

Ipv4Address
GlobalRoutingLinkRecord::GetLinkId (void) const
{
  NS_LOG_FUNCTION (this);
  return m_linkId;
}

void
GlobalRoutingLinkRecord::SetLinkId (Ipv4Address addr)
{
  NS_LOG_FUNCTION (this << addr);
  m_linkId = addr;
}

Ipv4Address
GlobalRoutingLinkRecord::GetLinkData (void) const
{
  NS_LOG_FUNCTION (this);
  return m_linkData;
}

void
GlobalRoutingLinkRecord::SetLinkData (Ipv4Address addr)
{
  NS_LOG_FUNCTION (this << addr);
  m_linkData = addr;
}

GlobalRoutingLinkRecord::LinkType
GlobalRoutingLinkRecord::GetLinkType (void) const
{
  NS_LOG_FUNCTION (this);
  return m_linkType;
}

void
GlobalRoutingLinkRecord::SetLinkType (
  GlobalRoutingLinkRecord::LinkType linkType)
{
  NS_LOG_FUNCTION (this << linkType);
  m_linkType = linkType;
}

uint16_t
GlobalRoutingLinkRecord::GetMetric (void) const
{
  NS_LOG_FUNCTION (this);
  return m_metric;
}

void
GlobalRoutingLinkRecord::SetMetric (uint16_t metric)
{
  NS_LOG_FUNCTION (this << metric);
  m_metric = metric;
}

// ---------------------------------------------------------------------------
//
// GlobalRoutingLSA Implementation
//
// ---------------------------------------------------------------------------

GlobalRoutingLSA::GlobalRoutingLSA()
  : 
    m_lsType (GlobalRoutingLSA::Unknown),
    m_linkStateId ("0.0.0.0"),
    m_advertisingRtr ("0.0.0.0"),
    m_linkRecords (),
    m_networkLSANetworkMask ("0.0.0.0"),
    m_attachedRouters (),
    m_status (GlobalRoutingLSA::LSA_SPF_NOT_EXPLORED),
    m_node_id (0)
{
  NS_LOG_FUNCTION (this);
}

GlobalRoutingLSA::GlobalRoutingLSA (
  GlobalRoutingLSA::SPFStatus status,
  Ipv4Address linkStateId, 
  Ipv4Address advertisingRtr)
  :
    m_lsType (GlobalRoutingLSA::Unknown),
    m_linkStateId (linkStateId),
    m_advertisingRtr (advertisingRtr),
    m_linkRecords (),
    m_networkLSANetworkMask ("0.0.0.0"),
    m_attachedRouters (),
    m_status (status),
    m_node_id (0)
{
  NS_LOG_FUNCTION (this << status << linkStateId << advertisingRtr);
}

GlobalRoutingLSA::GlobalRoutingLSA (GlobalRoutingLSA& lsa)
  : m_lsType (lsa.m_lsType), m_linkStateId (lsa.m_linkStateId),
    m_advertisingRtr (lsa.m_advertisingRtr),
    m_networkLSANetworkMask (lsa.m_networkLSANetworkMask),
    m_status (lsa.m_status),
    m_node_id (lsa.m_node_id)
{
  NS_LOG_FUNCTION (this << &lsa);
  NS_ASSERT_MSG (IsEmpty (),
                 "GlobalRoutingLSA::GlobalRoutingLSA (): Non-empty LSA in constructor");
  CopyLinkRecords (lsa);
}

GlobalRoutingLSA&
GlobalRoutingLSA::operator= (const GlobalRoutingLSA& lsa)
{
  NS_LOG_FUNCTION (this << &lsa);
  m_lsType = lsa.m_lsType;
  m_linkStateId = lsa.m_linkStateId;
  m_advertisingRtr = lsa.m_advertisingRtr;
  m_networkLSANetworkMask = lsa.m_networkLSANetworkMask, 
  m_status = lsa.m_status;
  m_node_id = lsa.m_node_id;

  ClearLinkRecords ();
  CopyLinkRecords (lsa);
  return *this;
}

void
GlobalRoutingLSA::CopyLinkRecords (const GlobalRoutingLSA& lsa)
{
  NS_LOG_FUNCTION (this << &lsa);
  for (ListOfLinkRecords_t::const_iterator i = lsa.m_linkRecords.begin ();
       i != lsa.m_linkRecords.end (); 
       i++)
    {
      GlobalRoutingLinkRecord *pSrc = *i;
      GlobalRoutingLinkRecord *pDst = new GlobalRoutingLinkRecord;

      pDst->SetLinkType (pSrc->GetLinkType ());
      pDst->SetLinkId (pSrc->GetLinkId ());
      pDst->SetLinkData (pSrc->GetLinkData ());
      pDst->SetMetric (pSrc->GetMetric ());

      m_linkRecords.push_back (pDst);
      pDst = 0;
    }

  m_attachedRouters = lsa.m_attachedRouters;
}

GlobalRoutingLSA::~GlobalRoutingLSA()
{
  NS_LOG_FUNCTION (this);
  ClearLinkRecords ();
}

void
GlobalRoutingLSA::ClearLinkRecords (void)
{
  NS_LOG_FUNCTION (this);
  for ( ListOfLinkRecords_t::iterator i = m_linkRecords.begin ();
        i != m_linkRecords.end (); 
        i++)
    {
      NS_LOG_LOGIC ("Free link record");

      GlobalRoutingLinkRecord *p = *i;
      delete p;
      p = 0;

      *i = 0;
    }
  NS_LOG_LOGIC ("Clear list");
  m_linkRecords.clear ();
}

uint32_t
GlobalRoutingLSA::AddLinkRecord (GlobalRoutingLinkRecord* lr)
{
  NS_LOG_FUNCTION (this << lr);
  m_linkRecords.push_back (lr);
  return m_linkRecords.size ();
}

uint32_t
GlobalRoutingLSA::GetNLinkRecords (void) const
{
  NS_LOG_FUNCTION (this);
  return m_linkRecords.size ();
}

GlobalRoutingLinkRecord *
GlobalRoutingLSA::GetLinkRecord (uint32_t n) const
{
  NS_LOG_FUNCTION (this << n);
  uint32_t j = 0;
  for ( ListOfLinkRecords_t::const_iterator i = m_linkRecords.begin ();
        i != m_linkRecords.end (); 
        i++, j++)
    {
      if (j == n) 
        {
          return *i;
        }
    }
  NS_ASSERT_MSG (false, "GlobalRoutingLSA::GetLinkRecord (): invalid index");
  return 0;
}

bool
GlobalRoutingLSA::IsEmpty (void) const
{
  NS_LOG_FUNCTION (this);
  return m_linkRecords.size () == 0;
}

GlobalRoutingLSA::LSType
GlobalRoutingLSA::GetLSType (void) const
{
  NS_LOG_FUNCTION (this);
  return m_lsType;
}

void
GlobalRoutingLSA::SetLSType (GlobalRoutingLSA::LSType typ) 
{
  NS_LOG_FUNCTION (this << typ);
  m_lsType = typ;
}

Ipv4Address
GlobalRoutingLSA::GetLinkStateId (void) const
{
  NS_LOG_FUNCTION (this);
  return m_linkStateId;
}

void
GlobalRoutingLSA::SetLinkStateId (Ipv4Address addr)
{
  NS_LOG_FUNCTION (this << addr);
  m_linkStateId = addr;
}

Ipv4Address
GlobalRoutingLSA::GetAdvertisingRouter (void) const
{
  NS_LOG_FUNCTION (this);
  return m_advertisingRtr;
}

void
GlobalRoutingLSA::SetAdvertisingRouter (Ipv4Address addr)
{
  NS_LOG_FUNCTION (this << addr);
  m_advertisingRtr = addr;
}

void
GlobalRoutingLSA::SetNetworkLSANetworkMask (Ipv4Mask mask)
{
  NS_LOG_FUNCTION (this << mask);
  m_networkLSANetworkMask = mask;
}

Ipv4Mask
GlobalRoutingLSA::GetNetworkLSANetworkMask (void) const
{
  NS_LOG_FUNCTION (this);
  return m_networkLSANetworkMask;
}

GlobalRoutingLSA::SPFStatus
GlobalRoutingLSA::GetStatus (void) const
{
  NS_LOG_FUNCTION (this);
  return m_status;
}

uint32_t
GlobalRoutingLSA::AddAttachedRouter (Ipv4Address addr)
{
  NS_LOG_FUNCTION (this << addr);
  m_attachedRouters.push_back (addr);
  return m_attachedRouters.size ();
}

uint32_t
GlobalRoutingLSA::GetNAttachedRouters (void) const
{
  NS_LOG_FUNCTION (this);
  return m_attachedRouters.size (); 
}

Ipv4Address
GlobalRoutingLSA::GetAttachedRouter (uint32_t n) const
{
  NS_LOG_FUNCTION (this << n);
  uint32_t j = 0;
  for ( ListOfAttachedRouters_t::const_iterator i = m_attachedRouters.begin ();
        i != m_attachedRouters.end (); 
        i++, j++)
    {
      if (j == n) 
        {
          return *i;
        }
    }
  NS_ASSERT_MSG (false, "GlobalRoutingLSA::GetAttachedRouter (): invalid index");
  return Ipv4Address ("0.0.0.0");
}

void
GlobalRoutingLSA::SetStatus (GlobalRoutingLSA::SPFStatus status)
{
  NS_LOG_FUNCTION (this << status);
  m_status = status;
}

Ptr<Node>
GlobalRoutingLSA::GetNode (void) const
{
  NS_LOG_FUNCTION (this);
  return NodeList::GetNode (m_node_id);
}

void
GlobalRoutingLSA::SetNode (Ptr<Node> node)
{
  NS_LOG_FUNCTION (this << node);
  m_node_id = node->GetId ();
}

void
GlobalRoutingLSA::Print (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  os << std::endl;
  os << "========== Global Routing LSA ==========" << std::endl;
  os << "m_lsType = " << m_lsType;
  if (m_lsType == GlobalRoutingLSA::RouterLSA) 
    {
      os << " (GlobalRoutingLSA::RouterLSA)";
    }
  else if (m_lsType == GlobalRoutingLSA::NetworkLSA) 
    {
      os << " (GlobalRoutingLSA::NetworkLSA)";
    }
  else if (m_lsType == GlobalRoutingLSA::ASExternalLSAs)
    {
      os << " (GlobalRoutingLSA::ASExternalLSA)";
    }
  else
    {
      os << "(Unknown LSType)";
    }
  os << std::endl;

  os << "m_linkStateId = " << m_linkStateId << " (Router ID)" << std::endl;
  os << "m_advertisingRtr = " << m_advertisingRtr << " (Router ID)" << std::endl;

  if (m_lsType == GlobalRoutingLSA::RouterLSA) 
    {
      for ( ListOfLinkRecords_t::const_iterator i = m_linkRecords.begin ();
            i != m_linkRecords.end (); 
            i++)
        {
          GlobalRoutingLinkRecord *p = *i;

          os << "---------- RouterLSA Link Record ----------" << std::endl;
          os << "m_linkType = " << p->m_linkType;
          if (p->m_linkType == GlobalRoutingLinkRecord::PointToPoint)
            {
              os << " (GlobalRoutingLinkRecord::PointToPoint)" << std::endl;
              os << "m_linkId = " << p->m_linkId << std::endl;
              os << "m_linkData = " << p->m_linkData << std::endl;
              os << "m_metric = " << p->m_metric << std::endl;
            }
          else if (p->m_linkType == GlobalRoutingLinkRecord::TransitNetwork)
            {
              os << " (GlobalRoutingLinkRecord::TransitNetwork)" << std::endl;
              os << "m_linkId = " << p->m_linkId << " (Designated router for network)" << std::endl;
              os << "m_linkData = " << p->m_linkData << " (This router's IP address)" << std::endl;
              os << "m_metric = " << p->m_metric << std::endl;
            }
          else if (p->m_linkType == GlobalRoutingLinkRecord::StubNetwork)
            {
              os << " (GlobalRoutingLinkRecord::StubNetwork)" << std::endl;
              os << "m_linkId = " << p->m_linkId << " (Network number of attached network)" << std::endl;
              os << "m_linkData = " << p->m_linkData << " (Network mask of attached network)" << std::endl;
              os << "m_metric = " << p->m_metric << std::endl;
            }
          else
            {
              os << " (Unknown LinkType)" << std::endl;
              os << "m_linkId = " << p->m_linkId << std::endl;
              os << "m_linkData = " << p->m_linkData << std::endl;
              os << "m_metric = " << p->m_metric << std::endl;
            }
          os << "---------- End RouterLSA Link Record ----------" << std::endl;
        }
    }
  else if (m_lsType == GlobalRoutingLSA::NetworkLSA) 
    {
      os << "---------- NetworkLSA Link Record ----------" << std::endl;
      os << "m_networkLSANetworkMask = " << m_networkLSANetworkMask << std::endl;
      for ( ListOfAttachedRouters_t::const_iterator i = m_attachedRouters.begin (); i != m_attachedRouters.end (); i++)
        {
          Ipv4Address p = *i;
          os << "attachedRouter = " << p << std::endl;
        }
      os << "---------- End NetworkLSA Link Record ----------" << std::endl;
    }
  else if (m_lsType == GlobalRoutingLSA::ASExternalLSAs)
    {
      os << "---------- ASExternalLSA Link Record --------" << std::endl;
      os << "m_linkStateId = " << m_linkStateId << std::endl;
      os << "m_networkLSANetworkMask = " << m_networkLSANetworkMask << std::endl;
    }
  else 
    {
      NS_ASSERT_MSG (0, "Illegal LSA LSType: " << m_lsType);
    }
  os << "========== End Global Routing LSA ==========" << std::endl;
}

std::ostream& operator<< (std::ostream& os, GlobalRoutingLSA& lsa)
{
  lsa.Print (os);
  return os;
}

// ---------------------------------------------------------------------------
//
// GlobalRouter Implementation
//
// ---------------------------------------------------------------------------

NS_OBJECT_ENSURE_REGISTERED (GlobalRouter);

TypeId 
GlobalRouter::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::GlobalRouter")
    .SetParent<Object> ()
    .SetGroupName ("Internet");
  return tid;
}

GlobalRouter::GlobalRouter ()
  : m_LSAs ()
{
  NS_LOG_FUNCTION (this);
  m_routerId.Set (GlobalRouteManager::AllocateRouterId ());
}

GlobalRouter::~GlobalRouter ()
{
  NS_LOG_FUNCTION (this);
  ClearLSAs ();
}

void 
GlobalRouter::SetRoutingProtocol (Ptr<Ipv4GlobalRouting> routing)
{
  NS_LOG_FUNCTION (this << routing);
  m_routingProtocol = routing;
}
Ptr<Ipv4GlobalRouting> 
GlobalRouter::GetRoutingProtocol (void)
{
  NS_LOG_FUNCTION (this);
  return m_routingProtocol;
}

void
GlobalRouter::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  m_routingProtocol = 0;
  for (InjectedRoutesI k = m_injectedRoutes.begin ();
       k != m_injectedRoutes.end ();
       k = m_injectedRoutes.erase (k))
    {
      delete (*k);
    }
  Object::DoDispose ();
}

void
GlobalRouter::ClearLSAs ()
{
  NS_LOG_FUNCTION (this);
  for ( ListOfLSAs_t::iterator i = m_LSAs.begin ();
        i != m_LSAs.end (); 
        i++)
    {
      NS_LOG_LOGIC ("Free LSA");

      GlobalRoutingLSA *p = *i;
      delete p;
      p = 0;

      *i = 0;
    }
  NS_LOG_LOGIC ("Clear list of LSAs");
  m_LSAs.clear ();
}

Ipv4Address
GlobalRouter::GetRouterId (void) const
{
  NS_LOG_FUNCTION (this);
  return m_routerId;
}

//
// DiscoverLSAs is called on all nodes in the system that have a GlobalRouter
// interface aggregated.  We need to go out and discover any adjacent routers 
// and build the Link State Advertisements that reflect them and their associated
// networks.
// 
uint32_t 
GlobalRouter::DiscoverLSAs ()
{
  NS_LOG_FUNCTION (this);
  Ptr<Node> node = GetObject<Node> ();
  NS_ABORT_MSG_UNLESS (node, "GlobalRouter::DiscoverLSAs (): GetObject for <Node> interface failed");
  NS_LOG_LOGIC ("For node " << node->GetId () );

  ClearLSAs ();

  //
  // While building the Router-LSA, keep a list of those NetDevices for
  // which the current node is the designated router and we will later build 
  // a NetworkLSA for.
  //
  NetDeviceContainer c;

  //
  // We're aggregated to a node.  We need to ask the node for a pointer to its
  // Ipv4 interface.  This is where the information regarding the attached 
  // interfaces lives.  If we're a router, we had better have an Ipv4 interface.
  //
  Ptr<Ipv4> ipv4Local = node->GetObject<Ipv4> ();
  NS_ABORT_MSG_UNLESS (ipv4Local, "GlobalRouter::DiscoverLSAs (): GetObject for <Ipv4> interface failed");

  //
  // Every router node originates a Router-LSA
  //
  GlobalRoutingLSA *pLSA = new GlobalRoutingLSA;
  pLSA->SetLSType (GlobalRoutingLSA::RouterLSA);
  pLSA->SetLinkStateId (m_routerId);
  pLSA->SetAdvertisingRouter (m_routerId);
  pLSA->SetStatus (GlobalRoutingLSA::LSA_SPF_NOT_EXPLORED);
  pLSA->SetNode (node);

  //
  // Ask the node for the number of net devices attached. This isn't necessarily 
  // equal to the number of links to adjacent nodes (other routers) as the number
  // of devices may include those for stub networks (e.g., ethernets, etc.) and 
  // bridge devices also take up an "extra" net device.
  //
  uint32_t numDevices = node->GetNDevices ();

  //
  // Iterate through the devices on the node and walk the channel to see what's
  // on the other side of the standalone devices..
  //
  for (uint32_t i = 0; i < numDevices; ++i)
    {
      Ptr<NetDevice> ndLocal = node->GetDevice (i);

      if (DynamicCast <LoopbackNetDevice> (ndLocal))
        {
          continue;
        }

      //
      // There is an assumption that bridge ports must never have an IP address 
      // associated with them.  This turns out to be a very convenient place to
      // check and make sure that this is the case.
      //
      if (NetDeviceIsBridged (ndLocal))
        {
          // Initialize to value out of bounds to silence compiler
          uint32_t interfaceBridge = ipv4Local->GetNInterfaces () + 1;
          bool rc = FindInterfaceForDevice (node, ndLocal, interfaceBridge);
          NS_ABORT_MSG_IF (rc, "GlobalRouter::DiscoverLSAs(): Bridge ports must not have an IPv4 interface index");
        }

      //
      // Check to see if the net device we just got has a corresponding IP 
      // interface (could be a pure L2 NetDevice) -- for example a net device
      // associated with a bridge.  We are only going to involve devices with 
      // IP addresses in routing.
      //
      bool isForwarding = false;
      for (uint32_t j = 0; j < ipv4Local->GetNInterfaces (); ++j )
        {
          if (ipv4Local->GetNetDevice (j) == ndLocal && ipv4Local->IsUp (j) &&
              ipv4Local->IsForwarding (j)) 
            {
              isForwarding = true;
              break;
            }
        }

      if (!isForwarding)
        {
          NS_LOG_LOGIC ("Net device " << ndLocal << "has no IP interface or is not enabled for forwarding, skipping");
          continue;
        }

      //
      // We have a net device that we need to check out.  If it supports 
      // broadcast and is not a point-point link, then it will be either a stub 
      // network or a transit network depending on the number of routers on
      // the segment.  We add the appropriate link record to the LSA.
      //
      // If the device is a point to point link, we treat it separately.  In
      // that case, there may be zero, one, or two link records added.
      //

      if (ndLocal->IsBroadcast () && !ndLocal->IsPointToPoint () )
        {
          NS_LOG_LOGIC ("Broadcast link");
          ProcessBroadcastLink (ndLocal, pLSA, c);
        }
      else if (ndLocal->IsPointToPoint () )
        {
          NS_LOG_LOGIC ("Point=to-point link");
          ProcessPointToPointLink (ndLocal, pLSA);
        }
      else
        {
          NS_ASSERT_MSG (0, "GlobalRouter::DiscoverLSAs (): unknown link type");
        }
    }

  NS_LOG_LOGIC ("========== LSA for node " << node->GetId () << " ==========");
  NS_LOG_LOGIC (*pLSA);
  m_LSAs.push_back (pLSA);
  pLSA = 0;

  // 
  // Now, determine whether we need to build a NetworkLSA.  This is the case if
  // we found at least one designated router.
  //
  uint32_t nDesignatedRouters = c.GetN ();
  if (nDesignatedRouters > 0)
    {
      NS_LOG_LOGIC ("Build Network LSAs");
      BuildNetworkLSAs (c);
    }

  //
  // Build injected route LSAs as external routes
  // RFC 2328, section 12.4.4
  //
  for (InjectedRoutesCI i = m_injectedRoutes.begin ();
       i != m_injectedRoutes.end ();
       i++)
    {
      GlobalRoutingLSA *pLSA = new GlobalRoutingLSA;
      pLSA->SetLSType (GlobalRoutingLSA::ASExternalLSAs);
      pLSA->SetLinkStateId ((*i)->GetDestNetwork ());
      pLSA->SetAdvertisingRouter (m_routerId);
      pLSA->SetNetworkLSANetworkMask ((*i)->GetDestNetworkMask ());
      pLSA->SetStatus (GlobalRoutingLSA::LSA_SPF_NOT_EXPLORED);
      m_LSAs.push_back (pLSA); 
    }
  return m_LSAs.size ();
}

void
GlobalRouter::ProcessBroadcastLink (Ptr<NetDevice> nd, GlobalRoutingLSA *pLSA, NetDeviceContainer &c)
{
  NS_LOG_FUNCTION (this << nd << pLSA << &c);

  if (nd->IsBridge ())
    {
      ProcessBridgedBroadcastLink (nd, pLSA, c);
    }
  else
    {
      ProcessSingleBroadcastLink (nd, pLSA, c);
    }
}

void
GlobalRouter::ProcessSingleBroadcastLink (Ptr<NetDevice> nd, GlobalRoutingLSA *pLSA, NetDeviceContainer &c)
{
  NS_LOG_FUNCTION (this << nd << pLSA << &c);

  GlobalRoutingLinkRecord *plr = new GlobalRoutingLinkRecord;
  NS_ABORT_MSG_IF (plr == 0, "GlobalRouter::ProcessSingleBroadcastLink(): Can't alloc link record");

  //
  // We have some preliminaries to do to get enough information to proceed.
  // This information we need comes from the internet stack, so notice that
  // there is an implied assumption that global routing is only going to 
  // work with devices attached to the internet stack (have an ipv4 interface
  // associated to them.
  //
  Ptr<Node> node = nd->GetNode ();

  Ptr<Ipv4> ipv4Local = node->GetObject<Ipv4> ();
  NS_ABORT_MSG_UNLESS (ipv4Local, "GlobalRouter::ProcessSingleBroadcastLink (): GetObject for <Ipv4> interface failed");

  // Initialize to value out of bounds to silence compiler
  uint32_t interfaceLocal = ipv4Local->GetNInterfaces () + 1;
  bool rc = FindInterfaceForDevice (node, nd, interfaceLocal);
  NS_ABORT_MSG_IF (rc == false, "GlobalRouter::ProcessSingleBroadcastLink(): No interface index associated with device");

  if (ipv4Local->GetNAddresses (interfaceLocal) > 1)
    {
      NS_LOG_WARN ("Warning, interface has multiple IP addresses; using only the primary one");
    }
  Ipv4Address addrLocal = ipv4Local->GetAddress (interfaceLocal, 0).GetLocal ();
  Ipv4Mask maskLocal = ipv4Local->GetAddress (interfaceLocal, 0).GetMask ();
  NS_LOG_LOGIC ("Working with local address " << addrLocal);
  uint16_t metricLocal = ipv4Local->GetMetric (interfaceLocal);

  //
  // Check to see if the net device is connected to a channel/network that has
  // another router on it.  If there is no other router on the link (but us) then
  // this is a stub network.  If we find another router, then what we have here
  // is a transit network.
  //
  ClearBridgesVisited ();
  if (AnotherRouterOnLink (nd) == false)
    {
      //
      // This is a net device connected to a stub network
      //
      NS_LOG_LOGIC ("Router-LSA Stub Network");
      plr->SetLinkType (GlobalRoutingLinkRecord::StubNetwork);

      // 
      // According to OSPF, the Link ID is the IP network number of 
      // the attached network.
      //
      plr->SetLinkId (addrLocal.CombineMask (maskLocal));

      //
      // and the Link Data is the network mask; converted to Ipv4Address
      //
      Ipv4Address maskLocalAddr;
      maskLocalAddr.Set (maskLocal.Get ());
      plr->SetLinkData (maskLocalAddr);
      plr->SetMetric (metricLocal);
      pLSA->AddLinkRecord (plr);
      plr = 0;
    }
  else
    {
      //
      // We have multiple routers on a broadcast interface, so this is
      // a transit network.
      //
      NS_LOG_LOGIC ("Router-LSA Transit Network");
      plr->SetLinkType (GlobalRoutingLinkRecord::TransitNetwork);

      // 
      // By definition, the router with the lowest IP address is the
      // designated router for the network.  OSPF says that the Link ID
      // gets the IP interface address of the designated router in this 
      // case.
      //
      ClearBridgesVisited ();
      Ipv4Address desigRtr;
      desigRtr = FindDesignatedRouterForLink (nd);

      //
      // Let's double-check that any designated router we find out on our
      // network is really on our network.
      //
      if (desigRtr != "255.255.255.255")
        {
          Ipv4Address networkHere = addrLocal.CombineMask (maskLocal);
          Ipv4Address networkThere = desigRtr.CombineMask (maskLocal);
          NS_ABORT_MSG_UNLESS (networkHere == networkThere,
                               "GlobalRouter::ProcessSingleBroadcastLink(): Network number confusion (" <<
                               addrLocal << "/" << maskLocal.GetPrefixLength () << ", " <<
                               desigRtr << "/" << maskLocal.GetPrefixLength () << ")");
        }
      if (desigRtr == addrLocal)
        {
          c.Add (nd);
          NS_LOG_LOGIC ("Node " << node->GetId () << " elected a designated router");
        }
      plr->SetLinkId (desigRtr);

      //
      // OSPF says that the Link Data is this router's own IP address.
      //
      plr->SetLinkData (addrLocal);
      plr->SetMetric (metricLocal);
      pLSA->AddLinkRecord (plr);
      plr = 0;
    }
}

void
GlobalRouter::ProcessBridgedBroadcastLink (Ptr<NetDevice> nd, GlobalRoutingLSA *pLSA, NetDeviceContainer &c)
{
  NS_LOG_FUNCTION (this << nd << pLSA << &c);
  NS_ASSERT_MSG (nd->IsBridge (), "GlobalRouter::ProcessBridgedBroadcastLink(): Called with non-bridge net device");

#if 0
  //
  // It is possible to admit the possibility that a bridge device on a node
  // can also participate in routing.  This would surprise people who don't
  // come from Microsoft-land where they do use such a construct.  Based on
  // the principle of least-surprise, we will leave the relatively simple
  // code in place to do this, but not enable it until someone really wants
  // the capability.  Even then, we will not enable this code as a default
  // but rather something you will have to go and turn on.
  //

  Ptr<BridgeNetDevice> bnd = nd->GetObject<BridgeNetDevice> ();
  NS_ABORT_MSG_UNLESS (bnd, "GlobalRouter::DiscoverLSAs (): GetObject for <BridgeNetDevice> failed");

  //
  // We have some preliminaries to do to get enough information to proceed.
  // This information we need comes from the internet stack, so notice that
  // there is an implied assumption that global routing is only going to 
  // work with devices attached to the internet stack (have an ipv4 interface
  // associated to them.
  //
  Ptr<Node> node = nd->GetNode ();
  Ptr<Ipv4> ipv4Local = node->GetObject<Ipv4> ();
  NS_ABORT_MSG_UNLESS (ipv4Local, "GlobalRouter::ProcessBridgedBroadcastLink (): GetObject for <Ipv4> interface failed");

  // Initialize to value out of bounds to silence compiler
  uint32_t interfaceLocal = ipv4Local->GetNInterfaces () + 1;
  bool rc = FindInterfaceForDevice (node, nd, interfaceLocal);
  NS_ABORT_MSG_IF (rc == false, "GlobalRouter::ProcessBridgedBroadcastLink(): No interface index associated with device");

  if (ipv4Local->GetNAddresses (interfaceLocal) > 1)
    {
      NS_LOG_WARN ("Warning, interface has multiple IP addresses; using only the primary one");
    }
  Ipv4Address addrLocal = ipv4Local->GetAddress (interfaceLocal, 0).GetLocal ();
  Ipv4Mask maskLocal = ipv4Local->GetAddress (interfaceLocal, 0).GetMask ();
  NS_LOG_LOGIC ("Working with local address " << addrLocal);
  uint16_t metricLocal = ipv4Local->GetMetric (interfaceLocal);

  //
  // We need to handle a bridge on the router.  This means that we have been 
  // given a net device that is a BridgeNetDevice.  It has an associated Ipv4
  // interface index and address.  Some number of other net devices live "under"
  // the bridge device as so-called bridge ports.  In a nutshell, what we have
  // to do is to repeat what is done for a single broadcast link on all of 
  // those net devices living under the bridge (trolls?)
  //

  bool areTransitNetwork = false;
  Ipv4Address desigRtr ("255.255.255.255");

  for (uint32_t i = 0; i < bnd->GetNBridgePorts (); ++i)
    {
      Ptr<NetDevice> ndTemp = bnd->GetBridgePort (i);

      //
      // We have to decide if we are a transit network.  This is characterized
      // by the presence of another router on the network segment.  If we find
      // another router on any of our bridged links, we are a transit network.
      //
      ClearBridgesVisited ();
      if (AnotherRouterOnLink (ndTemp))
        {
          areTransitNetwork = true;

          //
          // If we're going to be a transit network, then we have got to elect
          // a designated router for the whole bridge.  This means finding the
          // router with the lowest IP address on the whole bridge.  We ask 
          // for the lowest address on each segment and pick the lowest of them
          // all.
          //
          ClearBridgesVisited ();
          Ipv4Address desigRtrTemp = FindDesignatedRouterForLink (ndTemp);

          //
          // Let's double-check that any designated router we find out on our
          // network is really on our network.
          //
          if (desigRtrTemp != "255.255.255.255")
            {
              Ipv4Address networkHere = addrLocal.CombineMask (maskLocal);
              Ipv4Address networkThere = desigRtrTemp.CombineMask (maskLocal);
              NS_ABORT_MSG_UNLESS (networkHere == networkThere, 
                                   "GlobalRouter::ProcessSingleBroadcastLink(): Network number confusion (" <<
                                   addrLocal << "/" << maskLocal.GetPrefixLength () << ", " <<
                                   desigRtrTemp << "/" << maskLocal.GetPrefixLength () << ")");
            }
          if (desigRtrTemp < desigRtr)
            {
              desigRtr = desigRtrTemp;
            }
        }
    }
  //
  // That's all the information we need to put it all together, just like we did
  // in the case of a single broadcast link.
  //

  GlobalRoutingLinkRecord *plr = new GlobalRoutingLinkRecord;
  NS_ABORT_MSG_IF (plr == 0, "GlobalRouter::ProcessBridgedBroadcastLink(): Can't alloc link record");

  if (areTransitNetwork == false)
    {
      //
      // This is a net device connected to a bridge of stub networks
      //
      NS_LOG_LOGIC ("Router-LSA Stub Network");
      plr->SetLinkType (GlobalRoutingLinkRecord::StubNetwork);

      // 
      // According to OSPF, the Link ID is the IP network number of 
      // the attached network.
      //
      plr->SetLinkId (addrLocal.CombineMask (maskLocal));

      //
      // and the Link Data is the network mask; converted to Ipv4Address
      //
      Ipv4Address maskLocalAddr;
      maskLocalAddr.Set (maskLocal.Get ());
      plr->SetLinkData (maskLocalAddr);
      plr->SetMetric (metricLocal);
      pLSA->AddLinkRecord (plr);
      plr = 0;
    }
  else
    {
      //
      // We have multiple routers on a bridged broadcast interface, so this is
      // a transit network.
      //
      NS_LOG_LOGIC ("Router-LSA Transit Network");
      plr->SetLinkType (GlobalRoutingLinkRecord::TransitNetwork);

      // 
      // By definition, the router with the lowest IP address is the
      // designated router for the network.  OSPF says that the Link ID
      // gets the IP interface address of the designated router in this 
      // case.
      //
      if (desigRtr == addrLocal) 
        {
          c.Add (nd);
          NS_LOG_LOGIC ("Node " << node->GetId () << " elected a designated router");
        }
      plr->SetLinkId (desigRtr);

      //
      // OSPF says that the Link Data is this router's own IP address.
      //
      plr->SetLinkData (addrLocal);
      plr->SetMetric (metricLocal);
      pLSA->AddLinkRecord (plr);
      plr = 0;
    }
#endif
}

void
GlobalRouter::ProcessPointToPointLink (Ptr<NetDevice> ndLocal, GlobalRoutingLSA *pLSA)
{
  NS_LOG_FUNCTION (this << ndLocal << pLSA);

  //
  // We have some preliminaries to do to get enough information to proceed.
  // This information we need comes from the internet stack, so notice that
  // there is an implied assumption that global routing is only going to 
  // work with devices attached to the internet stack (have an ipv4 interface
  // associated to them.
  //
  Ptr<Node> nodeLocal = ndLocal->GetNode ();

  Ptr<Ipv4> ipv4Local = nodeLocal->GetObject<Ipv4> ();
  NS_ABORT_MSG_UNLESS (ipv4Local, "GlobalRouter::ProcessPointToPointLink (): GetObject for <Ipv4> interface failed");

  uint32_t interfaceLocal = ipv4Local->GetNInterfaces () + 1;
  bool rc = FindInterfaceForDevice (nodeLocal, ndLocal, interfaceLocal);
  NS_ABORT_MSG_IF (rc == false, "GlobalRouter::ProcessPointToPointLink (): No interface index associated with device");

  if (ipv4Local->GetNAddresses (interfaceLocal) > 1)
    {
      NS_LOG_WARN ("Warning, interface has multiple IP addresses; using only the primary one");
    }
  Ipv4Address addrLocal = ipv4Local->GetAddress (interfaceLocal, 0).GetLocal ();
  NS_LOG_LOGIC ("Working with local address " << addrLocal);
  uint16_t metricLocal = ipv4Local->GetMetric (interfaceLocal);

  //
  // Now, we're going to walk over to the remote net device on the other end of 
  // the point-to-point channel we know we have.  This is where our adjacent 
  // router (to use OSPF lingo) is running.
  //
  Ptr<Channel> ch = ndLocal->GetChannel ();

  //
  // Get the net device on the other side of the point-to-point channel.
  //
  Ptr<NetDevice> ndRemote = GetAdjacent (ndLocal, ch);

  //
  // The adjacent net device is aggregated to a node.  We need to ask that net 
  // device for its node, then ask that node for its Ipv4 interface.  Note a
  // requirement that nodes on either side of a point-to-point link must have 
  // internet stacks; and an assumption that point-to-point links are incompatible 
  // with bridging.
  //
  Ptr<Node> nodeRemote = ndRemote->GetNode ();
  Ptr<Ipv4> ipv4Remote = nodeRemote->GetObject<Ipv4> ();
  NS_ABORT_MSG_UNLESS (ipv4Remote, 
                       "GlobalRouter::ProcessPointToPointLink(): GetObject for remote <Ipv4> failed");

  //
  // Further note the requirement that nodes on either side of a point-to-point 
  // link must participate in global routing and therefore have a GlobalRouter
  // interface aggregated.
  //
  Ptr<GlobalRouter> rtrRemote = nodeRemote->GetObject<GlobalRouter> ();
  if (rtrRemote == 0)
    {
      // This case is possible if the remote does not participate in global routing
      return;
    }
  //
  // We're going to need the remote router ID, so we might as well get it now.
  //
  Ipv4Address rtrIdRemote = rtrRemote->GetRouterId ();
  NS_LOG_LOGIC ("Working with remote router " << rtrIdRemote);

  //
  // Now, just like we did above, we need to get the IP interface index for the 
  // net device on the other end of the point-to-point channel.
  //
  uint32_t interfaceRemote = ipv4Remote->GetNInterfaces () + 1;
  rc = FindInterfaceForDevice (nodeRemote, ndRemote, interfaceRemote);
  NS_ABORT_MSG_IF (rc == false, "GlobalRouter::ProcessPointToPointLinks(): No interface index associated with remote device");

  //
  // Now that we have the Ipv4 interface, we can get the (remote) address and
  // mask we need.
  //
  if (ipv4Remote->GetNAddresses (interfaceRemote) > 1)
    {
      NS_LOG_WARN ("Warning, interface has multiple IP addresses; using only the primary one");
    }
  Ipv4Address addrRemote = ipv4Remote->GetAddress (interfaceRemote, 0).GetLocal ();
  Ipv4Mask maskRemote = ipv4Remote->GetAddress (interfaceRemote, 0).GetMask ();
  NS_LOG_LOGIC ("Working with remote address " << addrRemote);

  //
  // Now we can fill out the link records for this link.  There are always two
  // link records; the first is a point-to-point record describing the link and
  // the second is a stub network record with the network number.
  //
  GlobalRoutingLinkRecord *plr;
  if (ipv4Remote->IsUp (interfaceRemote))
    {
      NS_LOG_LOGIC ("Remote side interface " << interfaceRemote << " is up-- add a type 1 link");
 
      plr  = new GlobalRoutingLinkRecord;
      NS_ABORT_MSG_IF (plr == 0, "GlobalRouter::ProcessPointToPointLink(): Can't alloc link record");
      plr->SetLinkType (GlobalRoutingLinkRecord::PointToPoint);
      plr->SetLinkId (rtrIdRemote);
      plr->SetLinkData (addrLocal);
      plr->SetMetric (metricLocal);
      pLSA->AddLinkRecord (plr);
      plr = 0;
    }

  // Regardless of state of peer, add a type 3 link (RFC 2328: 12.4.1.1)
  plr = new GlobalRoutingLinkRecord;
  NS_ABORT_MSG_IF (plr == 0, "GlobalRouter::ProcessPointToPointLink(): Can't alloc link record");
  plr->SetLinkType (GlobalRoutingLinkRecord::StubNetwork);
  plr->SetLinkId (addrRemote);
  plr->SetLinkData (Ipv4Address (maskRemote.Get ()));  // Frown
  plr->SetMetric (metricLocal);
  pLSA->AddLinkRecord (plr);
  plr = 0;
}

void
GlobalRouter::BuildNetworkLSAs (NetDeviceContainer c)
{
  NS_LOG_FUNCTION (this << &c);

  uint32_t nDesignatedRouters = c.GetN ();
  NS_LOG_DEBUG ("Number of designated routers: " << nDesignatedRouters);

  for (uint32_t i = 0; i < nDesignatedRouters; ++i)
    {
      //
      // Build one NetworkLSA for each net device talking to a network that we are the 
      // designated router for.  These devices are in the provided container.
      //
      Ptr<NetDevice> ndLocal = c.Get (i);
      Ptr<Node> node = ndLocal->GetNode ();

      Ptr<Ipv4> ipv4Local = node->GetObject<Ipv4> ();
      NS_ABORT_MSG_UNLESS (ipv4Local, "GlobalRouter::ProcessPointToPointLink (): GetObject for <Ipv4> interface failed");

      uint32_t interfaceLocal = ipv4Local->GetNInterfaces () + 1;
      bool rc = FindInterfaceForDevice (node, ndLocal, interfaceLocal);
      NS_ABORT_MSG_IF (rc == false, "GlobalRouter::BuildNetworkLSAs (): No interface index associated with device");

      if (ipv4Local->GetNAddresses (interfaceLocal) > 1)
        {
          NS_LOG_WARN ("Warning, interface has multiple IP addresses; using only the primary one");
        }
      Ipv4Address addrLocal = ipv4Local->GetAddress (interfaceLocal, 0).GetLocal ();
      Ipv4Mask maskLocal = ipv4Local->GetAddress (interfaceLocal, 0).GetMask ();

      GlobalRoutingLSA *pLSA = new GlobalRoutingLSA;
      NS_ABORT_MSG_IF (pLSA == 0, "GlobalRouter::BuildNetworkLSAs(): Can't alloc link record");

      pLSA->SetLSType (GlobalRoutingLSA::NetworkLSA);
      pLSA->SetLinkStateId (addrLocal);
      pLSA->SetAdvertisingRouter (m_routerId);
      pLSA->SetNetworkLSANetworkMask (maskLocal);
      pLSA->SetStatus (GlobalRoutingLSA::LSA_SPF_NOT_EXPLORED);
      pLSA->SetNode (node);

      //
      // Build a list of AttachedRouters by walking the devices in the channel
      // and, if we find a node with a GlobalRouter interface and an IPv4 
      // interface associated with that device, we call it an attached router.
      //
      ClearBridgesVisited ();
      Ptr<Channel> ch = ndLocal->GetChannel ();
      std::size_t nDevices = ch->GetNDevices ();
      NS_ASSERT (nDevices);
      NetDeviceContainer deviceList = FindAllNonBridgedDevicesOnLink (ch);
      NS_LOG_LOGIC ("Found " << deviceList.GetN () << " non-bridged devices on channel");

      for (uint32_t i = 0; i < deviceList.GetN (); i++)
        {
          Ptr<NetDevice> tempNd = deviceList.Get (i);
          NS_ASSERT (tempNd);
          if (tempNd == ndLocal)
            {
              NS_LOG_LOGIC ("Adding " << addrLocal << " to Network LSA");
              pLSA->AddAttachedRouter (addrLocal);
              continue;
            }
          Ptr<Node> tempNode = tempNd->GetNode ();

          // Does the node in question have a GlobalRouter interface?  If not it can
          // hardly be considered an attached router.
          //
          Ptr<GlobalRouter> rtr = tempNode->GetObject<GlobalRouter> ();
          if (rtr == 0)
            { 
              NS_LOG_LOGIC ("Node " << tempNode->GetId () << " does not have GlobalRouter interface--skipping");
              continue;
            }

          //
          // Does the attached node have an ipv4 interface for the device we're probing?
          // If not, it can't play router.
          //
          uint32_t tempInterface = 0;
          if (FindInterfaceForDevice (tempNode, tempNd, tempInterface))
            {
              Ptr<Ipv4> tempIpv4 = tempNode->GetObject<Ipv4> ();
              NS_ASSERT (tempIpv4);
              if (!tempIpv4->IsUp (tempInterface))
                {
                  NS_LOG_LOGIC ("Remote side interface " << tempInterface << " not up");
                }
              else 
                {
                  if (tempIpv4->GetNAddresses (tempInterface) > 1)
                    {
                      NS_LOG_WARN ("Warning, interface has multiple IP addresses; using only the primary one");
                    }
                  Ipv4Address tempAddr = tempIpv4->GetAddress (tempInterface, 0).GetLocal ();
                  NS_LOG_LOGIC ("Adding " << tempAddr << " to Network LSA");
                  pLSA->AddAttachedRouter (tempAddr);
                }
            }
          else
            {
              NS_LOG_LOGIC ("Node " << tempNode->GetId () << " device " << tempNd << " does not have IPv4 interface; skipping");
            }
        }
      m_LSAs.push_back (pLSA);
      NS_LOG_LOGIC ("========== LSA for node " << node->GetId () << " ==========");
      NS_LOG_LOGIC (*pLSA);
      pLSA = 0;
    }
}

NetDeviceContainer
GlobalRouter::FindAllNonBridgedDevicesOnLink (Ptr<Channel> ch) const
{
  NS_LOG_FUNCTION (this << ch);
  NetDeviceContainer c;

  for (std::size_t i = 0; i < ch->GetNDevices (); i++)
    {
      Ptr<NetDevice> nd = ch->GetDevice (i);
      NS_LOG_LOGIC ("checking to see if the device " << nd << " is bridged");
      Ptr<BridgeNetDevice> bnd = NetDeviceIsBridged (nd);
      if (bnd && BridgeHasAlreadyBeenVisited (bnd) == false)
        {
          NS_LOG_LOGIC ("Device is bridged by BridgeNetDevice " << bnd << " with " << bnd->GetNBridgePorts () << " ports");
          MarkBridgeAsVisited (bnd);
          // Find all channels bridged together, and recursively call
          // on all other channels
          for (uint32_t j = 0; j < bnd->GetNBridgePorts (); j++)
            {
              Ptr<NetDevice> bridgedDevice = bnd->GetBridgePort (j);
              if (bridgedDevice->GetChannel () == ch)
                {
                  NS_LOG_LOGIC ("Skipping my own device/channel");
                  continue;
                }
              NS_LOG_LOGIC ("Calling on channel " << bridgedDevice->GetChannel ());
              c.Add (FindAllNonBridgedDevicesOnLink (bridgedDevice->GetChannel ()));
            }
        }
      else
        {
          NS_LOG_LOGIC ("Device is not bridged; adding");
          c.Add (nd);
        }
    }
  NS_LOG_LOGIC ("Found " << c.GetN () << " devices");
  return c;
}

//
// Given a local net device, we need to walk the channel to which the net device is
// attached and look for nodes with GlobalRouter interfaces on them (one of them 
// will be us).  Of these, the router with the lowest IP address on the net device 
// connecting to the channel becomes the designated router for the link.
//
Ipv4Address
GlobalRouter::FindDesignatedRouterForLink (Ptr<NetDevice> ndLocal) const
{
  NS_LOG_FUNCTION (this << ndLocal);

  Ptr<Channel> ch = ndLocal->GetChannel ();
  uint32_t nDevices = ch->GetNDevices ();
  NS_ASSERT (nDevices);

  NS_LOG_LOGIC ("Looking for designated router off of net device " << ndLocal << " on node " << 
                ndLocal->GetNode ()->GetId ());

  Ipv4Address desigRtr ("255.255.255.255");

  //
  // Look through all of the devices on the channel to which the net device
  // in question is attached.
  //
  for (uint32_t i = 0; i < nDevices; i++)
    {
      Ptr<NetDevice> ndOther = ch->GetDevice (i);
      NS_ASSERT (ndOther);

      Ptr<Node> nodeOther = ndOther->GetNode ();

      NS_LOG_LOGIC ("Examine channel device " << i << " on node " << nodeOther->GetId ());

      //
      // For all other net devices, we need to check and see if a router
      // is present.  If the net device on the other side is a bridged
      // device, we need to consider all of the other devices on the 
      // bridge as well (all of the bridge ports.
      //
      NS_LOG_LOGIC ("checking to see if the device is bridged");
      Ptr<BridgeNetDevice> bnd = NetDeviceIsBridged (ndOther);
      if (bnd)
        {
          NS_LOG_LOGIC ("Device is bridged by BridgeNetDevice " << bnd);

          //
          // When enumerating a bridge, don't count the netdevice we came in on
          //
          if (ndLocal == ndOther)
            {
              NS_LOG_LOGIC ("Skip -- it is where we came from.");
              continue;
            }

          //
          // It is possible that the bridge net device is sitting under a
          // router, so we have to check for the presence of that router
          // before we run off and follow all the links
          //
          // We require a designated router to have a GlobalRouter interface and
          // an internet stack that includes the Ipv4 interface.  If it doesn't
          // it can't play router.
          //
          NS_LOG_LOGIC ("Checking for router on bridge net device " << bnd);
          Ptr<GlobalRouter> rtr = nodeOther->GetObject<GlobalRouter> ();
          Ptr<Ipv4> ipv4 = nodeOther->GetObject<Ipv4> ();
          if (rtr && ipv4)
            {
              // Initialize to value out of bounds to silence compiler
              uint32_t interfaceOther = ipv4->GetNInterfaces () + 1;
              if (FindInterfaceForDevice (nodeOther, bnd, interfaceOther))
                {
                  NS_LOG_LOGIC ("Found router on bridge net device " << bnd);
                  if (!ipv4->IsUp (interfaceOther))
                    {
                      NS_LOG_LOGIC ("Remote side interface " << interfaceOther << " not up");
                      continue;
                    }
                  if (ipv4->GetNAddresses (interfaceOther) > 1)
                    {
                      NS_LOG_WARN ("Warning, interface has multiple IP addresses; using only the primary one");
                    }
                  Ipv4Address addrOther = ipv4->GetAddress (interfaceOther, 0).GetLocal ();
                  desigRtr = addrOther < desigRtr ? addrOther : desigRtr;
                  NS_LOG_LOGIC ("designated router now " << desigRtr);
                }
            }

          // 
          // Check if we have seen this bridge net device already while
          // recursively enumerating an L2 broadcast domain. If it is new 
          // to us, go ahead and process it. If we have already processed it,
          // move to the next
          // 
          if(BridgeHasAlreadyBeenVisited(bnd))
            {
              NS_ABORT_MSG ("ERROR: L2 forwarding loop detected!");
            }

          MarkBridgeAsVisited(bnd);

          NS_LOG_LOGIC ("Looking through bridge ports of bridge net device " << bnd);
          for (uint32_t j = 0; j < bnd->GetNBridgePorts (); ++j)
            {
              Ptr<NetDevice> ndBridged = bnd->GetBridgePort (j);
              NS_LOG_LOGIC ("Examining bridge port " << j << " device " << ndBridged);
              if (ndBridged == ndOther)
                {
                  NS_LOG_LOGIC ("That bridge port is me, don't walk backward");
                  continue;
                }

              NS_LOG_LOGIC ("Recursively looking for routers down bridge port " << ndBridged);
              Ipv4Address addrOther = FindDesignatedRouterForLink (ndBridged);
              desigRtr = addrOther < desigRtr ? addrOther : desigRtr;
              NS_LOG_LOGIC ("designated router now " << desigRtr);
            }
        }
      else
        {
          NS_LOG_LOGIC ("This device is not bridged");
          Ptr<Node> nodeOther = ndOther->GetNode ();
          NS_ASSERT (nodeOther);

          //
          // We require a designated router to have a GlobalRouter interface and
          // an internet stack that includes the Ipv4 interface.  If it doesn't
          //
          Ptr<GlobalRouter> rtr = nodeOther->GetObject<GlobalRouter> ();
          Ptr<Ipv4> ipv4 = nodeOther->GetObject<Ipv4> ();
          if (rtr && ipv4)
            {
              // Initialize to value out of bounds to silence compiler
              uint32_t interfaceOther = ipv4->GetNInterfaces () + 1;
              if (FindInterfaceForDevice (nodeOther, ndOther, interfaceOther))
                {
                  if (!ipv4->IsUp (interfaceOther))
                    {
                      NS_LOG_LOGIC ("Remote side interface " << interfaceOther << " not up");
                      continue;
                    }
                  NS_LOG_LOGIC ("Found router on net device " << ndOther);
                  if (ipv4->GetNAddresses (interfaceOther) > 1)
                    {
                      NS_LOG_WARN ("Warning, interface has multiple IP addresses; using only the primary one");
                    }
                  Ipv4Address addrOther = ipv4->GetAddress (interfaceOther, 0).GetLocal ();
                  desigRtr = addrOther < desigRtr ? addrOther : desigRtr;
                  NS_LOG_LOGIC ("designated router now " << desigRtr);
                }
            }
        }
    }
  return desigRtr;
}

//
// Given a node and an attached net device, take a look off in the channel to 
// which the net device is attached and look for a node on the other side
// that has a GlobalRouter interface aggregated.  Life gets more complicated
// when there is a bridged net device on the other side.
//
bool
GlobalRouter::AnotherRouterOnLink (Ptr<NetDevice> nd) const
{
  NS_LOG_FUNCTION (this << nd);

  Ptr<Channel> ch = nd->GetChannel ();
  if (!ch)
    {
      // It may be that this net device is a stub device, without a channel
      return false;
    }
  uint32_t nDevices = ch->GetNDevices ();
  NS_ASSERT (nDevices);

  NS_LOG_LOGIC ("Looking for routers off of net device " << nd << " on node " << nd->GetNode ()->GetId ());

  //
  // Look through all of the devices on the channel to which the net device
  // in question is attached.
  //
  for (uint32_t i = 0; i < nDevices; i++)
    {
      Ptr<NetDevice> ndOther = ch->GetDevice (i);
      NS_ASSERT (ndOther);

      NS_LOG_LOGIC ("Examine channel device " << i << " on node " << ndOther->GetNode ()->GetId ());

      // 
      // Ignore the net device itself.
      //
      if (ndOther == nd)
        {
          NS_LOG_LOGIC ("Myself, skip");
          continue;
        }

      //
      // For all other net devices, we need to check and see if a router
      // is present.  If the net device on the other side is a bridged
      // device, we need to consider all of the other devices on the 
      // bridge.
      //
      NS_LOG_LOGIC ("checking to see if device is bridged");
      Ptr<BridgeNetDevice> bnd = NetDeviceIsBridged (ndOther);
      if (bnd)
        {
          NS_LOG_LOGIC ("Device is bridged by net device " << bnd);

          // 
          // Check if we have seen this bridge net device already while
          // recursively enumerating an L2 broadcast domain. If it is new 
          // to us, go ahead and process it. If we have already processed it,
          // move to the next
          // 
          if(BridgeHasAlreadyBeenVisited(bnd))
            {
              NS_ABORT_MSG ("ERROR: L2 forwarding loop detected!");
            }

          MarkBridgeAsVisited(bnd);

          NS_LOG_LOGIC ("Looking through bridge ports of bridge net device " << bnd);
          for (uint32_t j = 0; j < bnd->GetNBridgePorts (); ++j)
            {
              Ptr<NetDevice> ndBridged = bnd->GetBridgePort (j);
              NS_LOG_LOGIC ("Examining bridge port " << j << " device " << ndBridged);
              if (ndBridged == ndOther)
                {
                  NS_LOG_LOGIC ("That bridge port is me, skip");
                  continue;
                }

              NS_LOG_LOGIC ("Recursively looking for routers on bridge port " << ndBridged);
              if (AnotherRouterOnLink (ndBridged))
                {
                  NS_LOG_LOGIC ("Found routers on bridge port, return true");
                  return true;
                }
            }
          NS_LOG_LOGIC ("No routers on bridged net device, return false");
          return false;
        }

      NS_LOG_LOGIC ("This device is not bridged");
      Ptr<Node> nodeTemp = ndOther->GetNode ();
      NS_ASSERT (nodeTemp);

      Ptr<GlobalRouter> rtr = nodeTemp->GetObject<GlobalRouter> ();
      if (rtr)
        {
          NS_LOG_LOGIC ("Found GlobalRouter interface, return true");
          return true;
        }
      else 
        {
          NS_LOG_LOGIC ("No GlobalRouter interface on device, continue search");
        }
    }
  NS_LOG_LOGIC ("No routers found, return false");
  return false;
}

uint32_t
GlobalRouter::GetNumLSAs (void) const
{
  NS_LOG_FUNCTION (this);
  return m_LSAs.size ();
}

//
// Get the nth link state advertisement from this router.
//
bool
GlobalRouter::GetLSA (uint32_t n, GlobalRoutingLSA &lsa) const
{
  NS_LOG_FUNCTION (this << n << &lsa);
  NS_ASSERT_MSG (lsa.IsEmpty (), "GlobalRouter::GetLSA (): Must pass empty LSA");
//
// All of the work was done in GetNumLSAs.  All we have to do here is to
// walk the list of link state advertisements created there and return the 
// one the client is interested in.
//
  ListOfLSAs_t::const_iterator i = m_LSAs.begin ();
  uint32_t j = 0;

  for (; i != m_LSAs.end (); i++, j++)
    {
      if (j == n)
        {
          GlobalRoutingLSA *p = *i;
          lsa = *p;
          return true;
        }
    }

  return false;
}

void
GlobalRouter::InjectRoute (Ipv4Address network, Ipv4Mask networkMask)
{
  NS_LOG_FUNCTION (this << network << networkMask);
  Ipv4RoutingTableEntry *route = new Ipv4RoutingTableEntry ();
//
// Interface number does not matter here, using 1.
//
  *route = Ipv4RoutingTableEntry::CreateNetworkRouteTo (network,
                                                        networkMask,
                                                        1);
  m_injectedRoutes.push_back (route);
}

Ipv4RoutingTableEntry *
GlobalRouter::GetInjectedRoute (uint32_t index)
{
  NS_LOG_FUNCTION (this << index);
  if (index < m_injectedRoutes.size ())
    {
      uint32_t tmp = 0;
      for (InjectedRoutesCI i = m_injectedRoutes.begin ();
           i != m_injectedRoutes.end ();
           i++)
        {
          if (tmp  == index)
            {
              return *i;
            }
          tmp++;
        }
    }
  NS_ASSERT (false);
  // quiet compiler.
  return 0;
}

uint32_t
GlobalRouter::GetNInjectedRoutes ()
{
  NS_LOG_FUNCTION (this);
  return m_injectedRoutes.size ();
}

void
GlobalRouter::RemoveInjectedRoute (uint32_t index)
{
  NS_LOG_FUNCTION (this << index);
  NS_ASSERT (index < m_injectedRoutes.size ());
  uint32_t tmp = 0;
  for (InjectedRoutesI i = m_injectedRoutes.begin (); i != m_injectedRoutes.end (); i++)
    {
      if (tmp  == index)
        {
          NS_LOG_LOGIC ("Removing route " << index << "; size = " << m_injectedRoutes.size ());
          delete *i;
          m_injectedRoutes.erase (i);
          return;
        }
      tmp++;
    }
}

bool
GlobalRouter::WithdrawRoute (Ipv4Address network, Ipv4Mask networkMask)
{
  NS_LOG_FUNCTION (this << network << networkMask);
  for (InjectedRoutesI i = m_injectedRoutes.begin (); i != m_injectedRoutes.end (); i++)
    {
      if ((*i)->GetDestNetwork () == network && (*i)->GetDestNetworkMask () == networkMask)
        {
          NS_LOG_LOGIC ("Withdrawing route to network/mask " << network << "/" << networkMask);
          delete *i;
          m_injectedRoutes.erase (i);
          return true;
        }
    }
  return false;
}


//
// Link through the given channel and find the net device that's on the
// other end.  This only makes sense with a point-to-point channel.
//
Ptr<NetDevice>
GlobalRouter::GetAdjacent (Ptr<NetDevice> nd, Ptr<Channel> ch) const
{
  NS_LOG_FUNCTION (this << nd << ch);
  NS_ASSERT_MSG (ch->GetNDevices () == 2, "GlobalRouter::GetAdjacent (): Channel with other than two devices");
//
// This is a point to point channel with two endpoints.  Get both of them.
//
  Ptr<NetDevice> nd1 = ch->GetDevice (0);
  Ptr<NetDevice> nd2 = ch->GetDevice (1);
//
// One of the endpoints is going to be "us" -- that is the net device attached
// to the node on which we're running -- i.e., "nd".  The other endpoint (the
// one to which we are connected via the channel) is the adjacent router.
//
  if (nd1 == nd)
    {
      return nd2;
    }
  else if (nd2 == nd)
    {
      return nd1;
    }
  else
    {
      NS_ASSERT_MSG (false,
                     "GlobalRouter::GetAdjacent (): Wrong or confused channel?");
      return 0;
    }
}

//
// Given a node and a net device, find an IPV4 interface index that corresponds
// to that net device.  This function may fail for various reasons.  If a node
// does not have an internet stack (for example if it is a bridge) we won't have
// an IPv4 at all.  If the node does have a stack, but the net device in question
// is bridged, there will not be an interface associated directly with the device.
//
bool
GlobalRouter::FindInterfaceForDevice (Ptr<Node> node, Ptr<NetDevice> nd, uint32_t &index) const
{
  NS_LOG_FUNCTION (this << node << nd << &index);
  NS_LOG_LOGIC ("For node " << node->GetId () << " for net device " << nd );

  Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
  if (ipv4 == 0)
    {
      NS_LOG_LOGIC ("No Ipv4 interface on node " << node->GetId ());
      return false;
    }

  for (uint32_t i = 0; i < ipv4->GetNInterfaces (); ++i )
    {
      if (ipv4->GetNetDevice (i) == nd)
        {
          NS_LOG_LOGIC ("Device " << nd << " has associated ipv4 index " << i);
          index = i;
          return true;
        }
    }

  NS_LOG_LOGIC ("Device " << nd << " has no associated ipv4 index");
  return false;
}

//
// Decide whether or not a given net device is being bridged by a BridgeNetDevice.
//
Ptr<BridgeNetDevice>
GlobalRouter::NetDeviceIsBridged (Ptr<NetDevice> nd) const
{
  NS_LOG_FUNCTION (this << nd);

  Ptr<Node> node = nd->GetNode ();
  uint32_t nDevices = node->GetNDevices ();

  //
  // There is no bit on a net device that says it is being bridged, so we have
  // to look for bridges on the node to which the device is attached.  If we
  // find a bridge, we need to look through its bridge ports (the devices it
  // bridges) to see if we find the device in question.
  //
  for (uint32_t i = 0; i < nDevices; ++i)
    {
      Ptr<NetDevice> ndTest = node->GetDevice (i);
      NS_LOG_LOGIC ("Examine device " << i << " " << ndTest);

      if (ndTest->IsBridge ())
        {
          NS_LOG_LOGIC ("device " << i << " is a bridge net device");
          Ptr<BridgeNetDevice> bnd = ndTest->GetObject<BridgeNetDevice> ();
          NS_ABORT_MSG_UNLESS (bnd, "GlobalRouter::DiscoverLSAs (): GetObject for <BridgeNetDevice> failed");

          for (uint32_t j = 0; j < bnd->GetNBridgePorts (); ++j)
            {
              NS_LOG_LOGIC ("Examine bridge port " << j << " " << bnd->GetBridgePort (j));
              if (bnd->GetBridgePort (j) == nd)
                {
                  NS_LOG_LOGIC ("Net device " << nd << " is bridged by " << bnd);
                  return bnd;
                }
            }
        }
    }
  NS_LOG_LOGIC ("Net device " << nd << " is not bridged");
  return 0;
}

//
// Start a new enumeration of an L2 broadcast domain by clearing m_bridgesVisited
//
void 
GlobalRouter::ClearBridgesVisited (void) const
{
  m_bridgesVisited.clear();
}

//
// Check if we have already visited a given bridge net device by searching m_bridgesVisited
//
bool
GlobalRouter::BridgeHasAlreadyBeenVisited (Ptr<BridgeNetDevice> bridgeNetDevice) const
{
  std::vector<Ptr<BridgeNetDevice> >::iterator iter;
  for (iter = m_bridgesVisited.begin (); iter != m_bridgesVisited.end (); ++iter)
    {
      if (bridgeNetDevice == *iter)
        {
          NS_LOG_LOGIC ("Bridge " << bridgeNetDevice << " has been visited.");
          return true;
        }
    }
  return false;
}

//
// Remember that we visited a bridge net device by adding it to m_bridgesVisited
//
void 
GlobalRouter::MarkBridgeAsVisited (Ptr<BridgeNetDevice> bridgeNetDevice) const
{
  NS_LOG_FUNCTION (this << bridgeNetDevice);
  m_bridgesVisited.push_back (bridgeNetDevice);
}


} // namespace ns3
