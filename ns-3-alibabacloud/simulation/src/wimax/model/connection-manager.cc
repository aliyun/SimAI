/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008,2009 INRIA, UDcast
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
 * Authors: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 *          Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 *                               <amine.ismail@UDcast.com>
 */

#include "connection-manager.h"
#include "ns3/log.h"
#include "cid-factory.h"
#include "ss-record.h"
#include "mac-messages.h"
#include "ns3/pointer.h"
#include "ns3/enum.h"
#include "service-flow.h"
#include "ss-net-device.h"
#include "bs-net-device.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ConnectionManager");

NS_OBJECT_ENSURE_REGISTERED (ConnectionManager);

TypeId ConnectionManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ConnectionManager")
    .SetParent<Object> ()
    .SetGroupName("Wimax");
  return tid;
}

ConnectionManager::ConnectionManager (void)
  : m_cidFactory (0)
{
}

void
ConnectionManager::DoDispose (void)
{
}

ConnectionManager::~ConnectionManager (void)
{
}

void
ConnectionManager::SetCidFactory (CidFactory *cidFactory)
{
  m_cidFactory = cidFactory;
}

void
ConnectionManager::AllocateManagementConnections (SSRecord *ssRecord, RngRsp *rngrsp)
{
  Ptr<WimaxConnection> basicConnection = CreateConnection (Cid::BASIC);
  ssRecord->SetBasicCid (basicConnection->GetCid ());

  Ptr<WimaxConnection> primaryConnection = CreateConnection (Cid::PRIMARY);
  ssRecord->SetPrimaryCid (primaryConnection->GetCid ());

  rngrsp->SetBasicCid (basicConnection->GetCid ());
  rngrsp->SetPrimaryCid (primaryConnection->GetCid ());
}

Ptr<WimaxConnection>
ConnectionManager::CreateConnection (Cid::Type type)
{
  Cid cid;
  switch (type)
    {
    case Cid::BASIC:
    case Cid::MULTICAST:
    case Cid::PRIMARY:
      cid = m_cidFactory->Allocate (type);
      break;
    case Cid::TRANSPORT:
      cid = m_cidFactory->AllocateTransportOrSecondary ();
      break;
    default:
      NS_FATAL_ERROR ("Invalid connection type");
      break;
    }
  Ptr<WimaxConnection> connection = CreateObject<WimaxConnection> (cid, type);
  AddConnection (connection, type);
  return connection;
}

void
ConnectionManager::AddConnection (Ptr<WimaxConnection> connection, Cid::Type type)
{
  switch (type)
    {
    case Cid::BASIC:
      m_basicConnections.push_back (connection);
      break;
    case Cid::PRIMARY:
      m_primaryConnections.push_back (connection);
      break;
    case Cid::TRANSPORT:
      m_transportConnections.push_back (connection);
      break;
    case Cid::MULTICAST:
      m_multicastConnections.push_back (connection);
      break;
    default:
      NS_FATAL_ERROR ("Invalid connection type");
      break;
    }
}

Ptr<WimaxConnection>
ConnectionManager::GetConnection (Cid cid)
{
  std::vector<Ptr<WimaxConnection> >::const_iterator iter;

  for (iter = m_basicConnections.begin (); iter != m_basicConnections.end (); ++iter)
    {
      if ((*iter)->GetCid () == cid)
        {
          return *iter;
        }
    }

  for (iter = m_primaryConnections.begin (); iter != m_primaryConnections.end (); ++iter)
    {
      if ((*iter)->GetCid () == cid)
        {
          return *iter;
        }
    }

  for (iter = m_transportConnections.begin (); iter != m_transportConnections.end (); ++iter)
    {
      if ((*iter)->GetCid () == cid)
        {
          return *iter;
        }
    }

  return 0;
}

std::vector<Ptr<WimaxConnection> >
ConnectionManager::GetConnections (Cid::Type type) const
{
  std::vector<Ptr<WimaxConnection> > connections;

  switch (type)
    {
    case Cid::BASIC:
      connections = m_basicConnections;
      break;
    case Cid::PRIMARY:
      connections = m_primaryConnections;
      break;
    case Cid::TRANSPORT:
      connections = m_transportConnections;
      break;
    default:
      NS_FATAL_ERROR ("Invalid connection type");
      break;
    }

  return connections;
}

uint32_t
ConnectionManager::GetNPackets (Cid::Type type, ServiceFlow::SchedulingType schedulingType) const
{
  uint32_t nrPackets = 0;

  switch (type)
    {
    case Cid::BASIC:
      {
        for (std::vector<Ptr<WimaxConnection> >::const_iterator iter = m_basicConnections.begin (); iter
             != m_basicConnections.end (); ++iter)
          {
            nrPackets += (*iter)->GetQueue ()->GetSize ();
          }
        break;
      }
    case Cid::PRIMARY:
      {
        for (std::vector<Ptr<WimaxConnection> >::const_iterator iter = m_primaryConnections.begin (); iter
             != m_primaryConnections.end (); ++iter)
          {
            nrPackets += (*iter)->GetQueue ()->GetSize ();
          }
        break;
      }
    case Cid::TRANSPORT:
      {
        for (std::vector<Ptr<WimaxConnection> >::const_iterator iter = m_transportConnections.begin (); iter
             != m_transportConnections.end (); ++iter)
          {
            if (schedulingType == ServiceFlow::SF_TYPE_ALL || (*iter)->GetSchedulingType () == schedulingType)
              {
                nrPackets += (*iter)->GetQueue ()->GetSize ();
              }
          }
        break;
      }
    default:
      NS_FATAL_ERROR ("Invalid connection type");
      break;
    }

  return nrPackets;
}

bool
ConnectionManager::HasPackets (void) const
{
  std::vector<Ptr<WimaxConnection> >::const_iterator iter;
  for (iter = m_basicConnections.begin (); iter != m_basicConnections.end (); ++iter)
    {
      if ((*iter)->HasPackets ())
        {
          return true;
        }
    }

  for (iter = m_primaryConnections.begin (); iter != m_primaryConnections.end (); ++iter)
    {
      if ((*iter)->HasPackets ())
        {
          return true;
        }
    }

  for (iter = m_transportConnections.begin (); iter != m_transportConnections.end (); ++iter)
    {
      if ((*iter)->HasPackets ())
        {
          return true;
        }
    }

  return false;
}
} // namespace ns3


