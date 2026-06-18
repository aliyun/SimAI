/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#include "wifi-mac-trailer.h"

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (WifiMacTrailer);

WifiMacTrailer::WifiMacTrailer ()
{
}

WifiMacTrailer::~WifiMacTrailer ()
{
}

TypeId
WifiMacTrailer::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::WifiMacTrailer")
    .SetParent<Trailer> ()
    .SetGroupName ("Wifi")
    .AddConstructor<WifiMacTrailer> ()
  ;
  return tid;
}

TypeId
WifiMacTrailer::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void
WifiMacTrailer::Print (std::ostream &os) const
{
}

uint32_t
WifiMacTrailer::GetSerializedSize (void) const
{
  return WIFI_MAC_FCS_LENGTH;
}

void
WifiMacTrailer::Serialize (Buffer::Iterator start) const
{
  start.Prev (WIFI_MAC_FCS_LENGTH);
  start.WriteU32 (0);
}

uint32_t
WifiMacTrailer::Deserialize (Buffer::Iterator start)
{
  return WIFI_MAC_FCS_LENGTH;
}

} //namespace ns3
