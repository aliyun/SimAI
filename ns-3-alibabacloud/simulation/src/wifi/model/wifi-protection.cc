/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Universita' degli Studi di Napoli Federico II
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
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#include "wifi-protection.h"
#include "wifi-utils.h"

namespace ns3 {

/*
 * WifiProtection
 */

WifiProtection::WifiProtection (Method m)
  : method (m),
    protectionTime (Time::Min ())   // uninitialized
{
}

WifiProtection::~WifiProtection ()
{
}


/*
 * WifiNoProtection
 */

WifiNoProtection::WifiNoProtection ()
  : WifiProtection (NONE)
{
  protectionTime = Seconds (0);
}

std::unique_ptr<WifiProtection>
WifiNoProtection::Copy (void) const
{
  return std::unique_ptr<WifiProtection> (new WifiNoProtection (*this));
}

void
WifiNoProtection::Print (std::ostream &os) const
{
  os << "NONE";
}


/*
 * WifiRtsCtsProtection
 */

WifiRtsCtsProtection::WifiRtsCtsProtection ()
  : WifiProtection (RTS_CTS)
{
}

std::unique_ptr<WifiProtection>
WifiRtsCtsProtection::Copy (void) const
{
  return std::unique_ptr<WifiProtection> (new WifiRtsCtsProtection (*this));
}

void
WifiRtsCtsProtection::Print (std::ostream &os) const
{
  os << "RTS_CTS";
}


/*
 * WifiCtsToSelfProtection
 */

WifiCtsToSelfProtection::WifiCtsToSelfProtection ()
  : WifiProtection (CTS_TO_SELF)
{
}

std::unique_ptr<WifiProtection>
WifiCtsToSelfProtection::Copy (void) const
{
  return std::unique_ptr<WifiProtection> (new WifiCtsToSelfProtection (*this));
}

void
WifiCtsToSelfProtection::Print (std::ostream &os) const
{
  os << "CTS_TO_SELF";
}


std::ostream & operator << (std::ostream &os, const WifiProtection* protection)
{
  protection->Print (os);
  return os;
}

} //namespace ns3
