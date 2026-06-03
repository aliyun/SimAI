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

#include "dsr-main-helper.h"
#include "ns3/dsr-helper.h"
#include "ns3/dsr-routing.h"
#include "ns3/dsr-rcache.h"
#include "ns3/dsr-rreq-table.h"
#include "ns3/node-list.h"
#include "ns3/names.h"
#include "ns3/log.h"
#include "ns3/ptr.h"
#include "ns3/node.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("DsrMainHelper");

DsrMainHelper::DsrMainHelper ()
  : m_dsrHelper (0)
{
  NS_LOG_FUNCTION (this);
}

DsrMainHelper::DsrMainHelper (const DsrMainHelper &o)
{
  NS_LOG_FUNCTION (this);
  m_dsrHelper = o.m_dsrHelper->Copy ();
}

DsrMainHelper::~DsrMainHelper ()
{
  NS_LOG_FUNCTION (this);
  delete m_dsrHelper;
}

DsrMainHelper &
DsrMainHelper::operator = (const DsrMainHelper &o)
{
  if (this == &o)
    {
      return *this;
    }
  m_dsrHelper = o.m_dsrHelper->Copy ();
  return *this;
}

void
DsrMainHelper::Install (DsrHelper &dsrHelper, NodeContainer nodes)
{
  NS_LOG_DEBUG ("Passed node container");
  delete m_dsrHelper;
  m_dsrHelper = dsrHelper.Copy ();
  for (NodeContainer::Iterator i = nodes.Begin (); i != nodes.End (); ++i)
    {
      Install (*i);
    }
}

void
DsrMainHelper::Install (Ptr<Node> node)
{
  NS_LOG_FUNCTION (node);
  Ptr<ns3::dsr::DsrRouting> dsr = m_dsrHelper->Create (node);
//  Ptr<ns3::dsr::RouteCache> routeCache = CreateObject<ns3::dsr::RouteCache> ();
//  Ptr<ns3::dsr::RreqTable> rreqTable = CreateObject<ns3::dsr::RreqTable> ();
//  dsr->SetRouteCache (routeCache);
//  dsr->SetRequestTable (rreqTable);
  dsr->SetNode (node);
//  node->AggregateObject (routeCache);
//  node->AggregateObject (rreqTable);
}

void
DsrMainHelper::SetDsrHelper (DsrHelper &dsrHelper)
{
  NS_LOG_FUNCTION (this);
  delete m_dsrHelper;
  m_dsrHelper = dsrHelper.Copy ();
}

} // namespace ns3
