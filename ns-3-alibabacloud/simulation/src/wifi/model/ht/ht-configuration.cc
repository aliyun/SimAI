/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018  Sébastien Deronne
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
 * Author: Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#include "ns3/log.h"
#include "ns3/boolean.h"
#include "ns3/uinteger.h"
#include "ht-configuration.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("HtConfiguration");

NS_OBJECT_ENSURE_REGISTERED (HtConfiguration);

HtConfiguration::HtConfiguration ()
{
  NS_LOG_FUNCTION (this);
}

HtConfiguration::~HtConfiguration ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
HtConfiguration::GetTypeId (void)
{
  static ns3::TypeId tid = ns3::TypeId ("ns3::HtConfiguration")
    .SetParent<Object> ()
    .SetGroupName ("Wifi")
    .AddConstructor<HtConfiguration> ()
    .AddAttribute ("ShortGuardIntervalSupported",
                   "Whether or not short guard interval is supported.",
                   BooleanValue (false),
                   MakeBooleanAccessor (&HtConfiguration::GetShortGuardIntervalSupported,
                                        &HtConfiguration::SetShortGuardIntervalSupported),
                   MakeBooleanChecker ())
    .AddAttribute ("LdpcSupported",
                   "Whether or not LDPC coding is supported.",
                   BooleanValue (false),
                   MakeBooleanAccessor (&HtConfiguration::GetLdpcSupported,
                                        &HtConfiguration::SetLdpcSupported),
                   MakeBooleanChecker ())
    ;
    return tid;
}

void
HtConfiguration::SetShortGuardIntervalSupported (bool enable)
{
  NS_LOG_FUNCTION (this << enable);
  m_sgiSupported = enable;
}

bool
HtConfiguration::GetShortGuardIntervalSupported (void) const
{
  return m_sgiSupported;
}

void
HtConfiguration::SetLdpcSupported (bool enable)
{
  NS_LOG_FUNCTION (this << enable);
  m_ldpcSupported = enable;
}

bool
HtConfiguration::GetLdpcSupported (void) const
{
  return m_ldpcSupported;
}

} //namespace ns3
