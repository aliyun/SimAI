/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 Strasbourg University
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
 * Author: Sebastien Vincent <vincent@clarinet.u-strasbg.fr>
 */

#include "radvd-prefix.h"
#include <ns3/log.h>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE ("RadvdPrefix");

RadvdPrefix::RadvdPrefix (Ipv6Address network, uint8_t prefixLength, uint32_t preferredLifeTime, uint32_t validLifeTime, bool onLinkFlag, bool autonomousFlag, bool routerAddrFlag)
  : m_network (network),
    m_prefixLength (prefixLength),
    m_preferredLifeTime (preferredLifeTime),
    m_validLifeTime (validLifeTime),
    m_onLinkFlag (onLinkFlag),
    m_autonomousFlag (autonomousFlag),
    m_routerAddrFlag (routerAddrFlag)
{
  NS_LOG_FUNCTION (this << network << prefixLength << preferredLifeTime << validLifeTime << onLinkFlag << autonomousFlag << routerAddrFlag);
}

RadvdPrefix::~RadvdPrefix ()
{
  NS_LOG_FUNCTION (this);
}

Ipv6Address RadvdPrefix::GetNetwork () const
{
  NS_LOG_FUNCTION (this);
  return m_network;
}

void RadvdPrefix::SetNetwork (Ipv6Address network)
{
  NS_LOG_FUNCTION (this << network);
  m_network = network;
}

uint8_t RadvdPrefix::GetPrefixLength () const
{
  NS_LOG_FUNCTION (this);
  return m_prefixLength;
}

void RadvdPrefix::SetPrefixLength (uint8_t prefixLength)
{
  NS_LOG_FUNCTION (this << prefixLength);
  m_prefixLength = prefixLength;
}

uint32_t RadvdPrefix::GetValidLifeTime () const
{
  NS_LOG_FUNCTION (this);
  return m_validLifeTime;
}

void RadvdPrefix::SetValidLifeTime (uint32_t validLifeTime)
{
  NS_LOG_FUNCTION (this << validLifeTime);
  m_validLifeTime = validLifeTime;
}

uint32_t RadvdPrefix::GetPreferredLifeTime () const
{
  NS_LOG_FUNCTION (this);
  return m_preferredLifeTime;
}

void RadvdPrefix::SetPreferredLifeTime (uint32_t preferredLifeTime)
{
  NS_LOG_FUNCTION (this << preferredLifeTime);
  m_preferredLifeTime = preferredLifeTime;
}

bool RadvdPrefix::IsOnLinkFlag () const
{
  NS_LOG_FUNCTION (this);
  return m_onLinkFlag;
}

void RadvdPrefix::SetOnLinkFlag (bool onLinkFlag)
{
  NS_LOG_FUNCTION (this << onLinkFlag);
  m_onLinkFlag = onLinkFlag;
}

bool RadvdPrefix::IsAutonomousFlag () const
{
  NS_LOG_FUNCTION (this);
  return m_autonomousFlag; 
}

void RadvdPrefix::SetAutonomousFlag (bool autonomousFlag)
{
  NS_LOG_FUNCTION (this << autonomousFlag);
  m_autonomousFlag = autonomousFlag;
}

bool RadvdPrefix::IsRouterAddrFlag () const
{
  NS_LOG_FUNCTION (this);
  return m_routerAddrFlag;
}

void RadvdPrefix::SetRouterAddrFlag (bool routerAddrFlag)
{
  NS_LOG_FUNCTION (this << routerAddrFlag);
  m_routerAddrFlag = routerAddrFlag;
}

} /* namespace ns3 */

