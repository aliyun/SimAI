/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008, 2009 INRIA, UDcast
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
 */

#include <stdint.h>
#include "ns3/node.h"
#include "ns3/simulator.h"
#include "ns3/packet.h"
#include "service-flow.h"
#include "service-flow-manager.h"
#include "ns3/log.h"
#include "wimax-net-device.h"
#include "bs-net-device.h"
#include "ss-net-device.h"
#include "ss-record.h"
#include "ns3/pointer.h"
#include "ns3/enum.h"
#include "wimax-connection.h"
#include "ss-manager.h"
#include "connection-manager.h"
#include "bs-uplink-scheduler.h"
#include "ss-scheduler.h"
#include "ns3/buffer.h"
#include "service-flow-record.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ServiceFlowManager");

NS_OBJECT_ENSURE_REGISTERED (ServiceFlowManager);

TypeId ServiceFlowManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ServiceFlowManager")
    .SetParent<Object> ()
    .SetGroupName("Wimax");
  return tid;
}

ServiceFlowManager::ServiceFlowManager ()
{
  m_serviceFlows = new std::vector<ServiceFlow*>;
}

ServiceFlowManager::~ServiceFlowManager (void)
{
}

void
ServiceFlowManager::DoDispose (void)
{
  for (std::vector<ServiceFlow*>::iterator iter = m_serviceFlows->begin (); iter != m_serviceFlows->end (); ++iter)
    {
      delete (*iter);
    }
  m_serviceFlows->clear ();
  delete m_serviceFlows;
}

void
ServiceFlowManager::AddServiceFlow (ServiceFlow *serviceFlow)
{
  m_serviceFlows->push_back (serviceFlow);
}

ServiceFlow* ServiceFlowManager::DoClassify (Ipv4Address srcAddress,
                                             Ipv4Address dstAddress,
                                             uint16_t srcPort,
                                             uint16_t dstPort,
                                             uint8_t proto,
                                             ServiceFlow::Direction dir) const
{
  for (std::vector<ServiceFlow*>::iterator iter = m_serviceFlows->begin (); iter != m_serviceFlows->end (); ++iter)
    {
      if ((*iter)->GetDirection () == dir)
        {
          if ((*iter)->CheckClassifierMatch (srcAddress, dstAddress, srcPort, dstPort, proto))
            {
              return (*iter);
            }
        }
    }
  return 0;
}

ServiceFlow*
ServiceFlowManager::GetServiceFlow (uint32_t sfid) const
{
  for (std::vector<ServiceFlow*>::iterator iter = m_serviceFlows->begin (); iter != m_serviceFlows->end (); ++iter)
    {
      if ((*iter)->GetSfid () == sfid)
        {
          return (*iter);
        }
    }

  NS_LOG_DEBUG ("GetServiceFlow: service flow not found!");
  return 0;
}

ServiceFlow*
ServiceFlowManager::GetServiceFlow (Cid cid) const
{
  for (std::vector<ServiceFlow*>::iterator iter = m_serviceFlows->begin (); iter != m_serviceFlows->end (); ++iter)
    {
      if ((*iter)->GetCid () == cid.GetIdentifier ())
        {
          return (*iter);
        }
    }

  NS_LOG_DEBUG ("GetServiceFlow: service flow not found!");
  return 0;
}

std::vector<ServiceFlow*>
ServiceFlowManager::GetServiceFlows (enum ServiceFlow::SchedulingType schedulingType) const
{
  std::vector<ServiceFlow*> tmpServiceFlows;
  for (std::vector<ServiceFlow*>::iterator iter = m_serviceFlows->begin (); iter != m_serviceFlows->end (); ++iter)
    {
      if (((*iter)->GetSchedulingType () == schedulingType) || (schedulingType == ServiceFlow::SF_TYPE_ALL))
        {
          tmpServiceFlows.push_back ((*iter));
        }
    }
  return tmpServiceFlows;
}

bool
ServiceFlowManager::AreServiceFlowsAllocated ()
{
  return AreServiceFlowsAllocated (m_serviceFlows);
}

bool
ServiceFlowManager::AreServiceFlowsAllocated (std::vector<ServiceFlow*>* serviceFlowVector)
{
  return AreServiceFlowsAllocated (*serviceFlowVector);
}

bool
ServiceFlowManager::AreServiceFlowsAllocated (std::vector<ServiceFlow*> serviceFlowVector)
{
  for (std::vector<ServiceFlow*>::const_iterator iter = serviceFlowVector.begin (); iter != serviceFlowVector.end (); ++iter)
    {
      if (!(*iter)->GetIsEnabled ())
        {
          return false;
        }
    }
  return true;
}

ServiceFlow*
ServiceFlowManager::GetNextServiceFlowToAllocate ()
{
  std::vector<ServiceFlow*>::iterator iter;
  for (iter = m_serviceFlows->begin (); iter != m_serviceFlows->end (); ++iter)
    {
      if (!(*iter)->GetIsEnabled ())
        {
          return (*iter);
        }
    }
  return 0;
}

uint32_t
ServiceFlowManager::GetNrServiceFlows (void) const
{
  return m_serviceFlows->size ();
}

} // namespace ns3
