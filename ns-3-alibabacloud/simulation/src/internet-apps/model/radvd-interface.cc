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

#include "radvd-interface.h"
#include <ns3/log.h>

namespace ns3 
{

NS_LOG_COMPONENT_DEFINE ("RadvdInterface");

RadvdInterface::RadvdInterface (uint32_t interface)
  : m_interface (interface)
{
  NS_LOG_FUNCTION (this << interface);
  /* initialize default value as specified in radvd.conf manpage */
  m_sendAdvert = true;
  m_maxRtrAdvInterval = 600000;
  m_minRtrAdvInterval = (uint32_t)(double) (0.33 * m_maxRtrAdvInterval);
  m_minDelayBetweenRAs = 3000;
  m_managedFlag = false;
  m_otherConfigFlag = false;
  m_linkMtu = 0; /* 0 means not sending MTU option in RA */
  m_reachableTime = 0; /* means unspecified for the router */
  m_retransTimer = 0; /* means unspecified for the router */
  m_curHopLimit = 64;
  m_defaultLifeTime = (3 * m_maxRtrAdvInterval) / 1000;
  m_defaultPreference = 1;
  m_sourceLLAddress = true;
  m_homeAgentFlag = false;
  m_homeAgentInfo = false;
  m_homeAgentLifeTime = 0;
  m_homeAgentPreference = 0;
  m_mobRtrSupportFlag = false;
  m_intervalOpt = false;
  m_initialRtrAdvertisementsLeft = 3;
}

RadvdInterface::RadvdInterface (uint32_t interface, uint32_t maxRtrAdvInterval, uint32_t minRtrAdvInterval)
  : m_interface (interface)
{
  NS_LOG_FUNCTION (this << interface << maxRtrAdvInterval << minRtrAdvInterval);
  NS_ASSERT (maxRtrAdvInterval > minRtrAdvInterval);
  m_sendAdvert = true;
  m_maxRtrAdvInterval = maxRtrAdvInterval;
  m_minRtrAdvInterval = minRtrAdvInterval;
  m_minDelayBetweenRAs = 3000;
  m_managedFlag = false;
  m_otherConfigFlag = false;
  m_linkMtu = 0; /* 0 means not sending MTU option in RA */
  m_reachableTime = 0; /* means unspecified for the router */
  m_retransTimer = 0; /* means unspecified for the router */
  m_curHopLimit = 64;
  m_defaultLifeTime = 3 * m_maxRtrAdvInterval;
  m_defaultPreference = 1;
  m_sourceLLAddress = true;
  m_homeAgentFlag = false;
  m_homeAgentInfo = false;
  m_homeAgentLifeTime = 0;
  m_homeAgentPreference = 0;
  m_mobRtrSupportFlag = false;
  m_intervalOpt = false;
  m_initialRtrAdvertisementsLeft = 3;
}

RadvdInterface::~RadvdInterface ()
{
  NS_LOG_FUNCTION (this);
  /* clear prefixes */
  for (RadvdPrefixListI it = m_prefixes.begin (); it != m_prefixes.end (); ++it)
    {
      (*it) = 0;
    }
  m_prefixes.clear ();
}

void RadvdInterface::AddPrefix (Ptr<RadvdPrefix> routerPrefix)
{
  NS_LOG_FUNCTION (this << routerPrefix);
  m_prefixes.push_back (routerPrefix);
}


uint32_t RadvdInterface::GetInterface () const
{
  NS_LOG_FUNCTION (this);
  return m_interface;
}

std::list<Ptr<RadvdPrefix> > RadvdInterface::GetPrefixes () const
{
  NS_LOG_FUNCTION (this);
  return m_prefixes;
}

bool RadvdInterface::IsSendAdvert () const
{
  NS_LOG_FUNCTION (this);
  return m_sendAdvert;
}

void RadvdInterface::SetSendAdvert (bool sendAdvert)
{
  NS_LOG_FUNCTION (this << sendAdvert);
  m_sendAdvert = sendAdvert;
}

uint32_t RadvdInterface::GetMaxRtrAdvInterval () const
{
  NS_LOG_FUNCTION (this);
  return m_maxRtrAdvInterval;
}

void RadvdInterface::SetMaxRtrAdvInterval (uint32_t maxRtrAdvInterval)
{
  NS_LOG_FUNCTION (this << maxRtrAdvInterval);
  m_maxRtrAdvInterval = maxRtrAdvInterval;
}

uint32_t RadvdInterface::GetMinRtrAdvInterval () const
{
  NS_LOG_FUNCTION (this);
  return m_minRtrAdvInterval;
}

void RadvdInterface::SetMinRtrAdvInterval (uint32_t minRtrAdvInterval)
{
  NS_LOG_FUNCTION (this << minRtrAdvInterval);
  m_minRtrAdvInterval = minRtrAdvInterval;
}

uint32_t RadvdInterface::GetMinDelayBetweenRAs () const
{
  NS_LOG_FUNCTION (this);
  return m_minDelayBetweenRAs;
}

void RadvdInterface::SetMinDelayBetweenRAs (uint32_t minDelayBetweenRAs)
{
  NS_LOG_FUNCTION (this << minDelayBetweenRAs);
  m_minDelayBetweenRAs = minDelayBetweenRAs;
}

bool RadvdInterface::IsManagedFlag () const
{
  NS_LOG_FUNCTION (this);
  return m_managedFlag;
}

void RadvdInterface::SetManagedFlag (bool managedFlag)
{
  NS_LOG_FUNCTION (this << managedFlag);
  m_managedFlag = managedFlag;
}

bool RadvdInterface::IsOtherConfigFlag () const
{
  NS_LOG_FUNCTION (this);
  return m_otherConfigFlag;
}

void RadvdInterface::SetOtherConfigFlag (bool otherConfigFlag)
{
  NS_LOG_FUNCTION (this << otherConfigFlag);
  m_otherConfigFlag = otherConfigFlag;
}

uint32_t RadvdInterface::GetLinkMtu () const
{
  NS_LOG_FUNCTION (this);
  return m_linkMtu;
}

void RadvdInterface::SetLinkMtu (uint32_t linkMtu)
{
  NS_LOG_FUNCTION (this << linkMtu);
  m_linkMtu = linkMtu;
}

uint32_t RadvdInterface::GetReachableTime () const
{
  NS_LOG_FUNCTION (this);
  return m_reachableTime;
}

void RadvdInterface::SetReachableTime (uint32_t reachableTime)
{
  NS_LOG_FUNCTION (this << reachableTime);
  m_reachableTime = reachableTime;
}

uint32_t RadvdInterface::GetDefaultLifeTime () const
{
  NS_LOG_FUNCTION (this);
  return m_defaultLifeTime;
}

void RadvdInterface::SetDefaultLifeTime (uint32_t defaultLifeTime)
{
  NS_LOG_FUNCTION (this << defaultLifeTime);
  m_defaultLifeTime = defaultLifeTime;
}

uint32_t RadvdInterface::GetRetransTimer () const
{
  NS_LOG_FUNCTION (this);
  return m_retransTimer;
}

void RadvdInterface::SetRetransTimer (uint32_t retransTimer)
{
  NS_LOG_FUNCTION (this << retransTimer);
  m_retransTimer = retransTimer;
}

uint8_t RadvdInterface::GetCurHopLimit () const
{
  NS_LOG_FUNCTION (this);
  return m_curHopLimit;
}

void RadvdInterface::SetCurHopLimit (uint8_t curHopLimit)
{
  NS_LOG_FUNCTION (this << curHopLimit);
  m_curHopLimit = curHopLimit;
}

uint8_t RadvdInterface::GetDefaultPreference () const
{
  NS_LOG_FUNCTION (this);
  return m_defaultPreference;
}

void RadvdInterface::SetDefaultPreference (uint8_t defaultPreference)
{
  NS_LOG_FUNCTION (this << defaultPreference);
  m_defaultPreference = defaultPreference;
}

bool RadvdInterface::IsSourceLLAddress () const
{
  NS_LOG_FUNCTION (this);
  return m_sourceLLAddress;
}

void RadvdInterface::SetSourceLLAddress (bool sourceLLAddress)
{
  NS_LOG_FUNCTION (this << sourceLLAddress);
  m_sourceLLAddress = sourceLLAddress;
}

bool RadvdInterface::IsHomeAgentFlag () const
{
  NS_LOG_FUNCTION (this);
  return m_homeAgentFlag;
}

void RadvdInterface::SetHomeAgentFlag (bool homeAgentFlag)
{
  NS_LOG_FUNCTION (this << homeAgentFlag);
  m_homeAgentFlag = homeAgentFlag;
}

bool RadvdInterface::IsHomeAgentInfo () const
{
  NS_LOG_FUNCTION (this);
  return m_homeAgentInfo;
}

void RadvdInterface::SetHomeAgentInfo (bool homeAgentInfo)
{
  NS_LOG_FUNCTION (this << homeAgentInfo);
  m_homeAgentInfo = homeAgentInfo;
}

uint32_t RadvdInterface::GetHomeAgentLifeTime () const
{
  NS_LOG_FUNCTION (this);
  return m_homeAgentLifeTime;
}

void RadvdInterface::SetHomeAgentLifeTime (uint32_t homeAgentLifeTime)
{
  NS_LOG_FUNCTION (this << homeAgentLifeTime);
  m_homeAgentLifeTime = homeAgentLifeTime;
}

uint32_t RadvdInterface::GetHomeAgentPreference () const
{
  NS_LOG_FUNCTION (this);
  return m_homeAgentPreference;
}

void RadvdInterface::SetHomeAgentPreference (uint32_t homeAgentPreference)
{
  NS_LOG_FUNCTION (this << homeAgentPreference);
  m_homeAgentPreference = homeAgentPreference;
}

bool RadvdInterface::IsMobRtrSupportFlag () const
{
  NS_LOG_FUNCTION (this);
  return m_mobRtrSupportFlag;
}

void RadvdInterface::SetMobRtrSupportFlag (bool mobRtrSupportFlag)
{
  NS_LOG_FUNCTION (this << mobRtrSupportFlag);
  m_mobRtrSupportFlag = mobRtrSupportFlag;
}

bool RadvdInterface::IsIntervalOpt () const
{
  NS_LOG_FUNCTION (this);
  return m_intervalOpt;
}

void RadvdInterface::SetIntervalOpt (bool intervalOpt)
{
  NS_LOG_FUNCTION (this << intervalOpt);
  m_intervalOpt = intervalOpt;
}

Time RadvdInterface::GetLastRaTxTime ()
{
  return m_lastSendTime;
}

void RadvdInterface::SetLastRaTxTime (Time now)
{
  m_lastSendTime = now;
  if (m_initialRtrAdvertisementsLeft)
    {
      m_initialRtrAdvertisementsLeft --;
    }
}

bool RadvdInterface::IsInitialRtrAdv ()
{
  return m_initialRtrAdvertisementsLeft;
}

} /* namespace ns3 */

