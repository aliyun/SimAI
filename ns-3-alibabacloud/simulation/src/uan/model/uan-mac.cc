/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 University of Washington
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
 * Author: Mitch Watrous <watrous@u.washington.edu>
 */

#include "uan-mac.h"

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (UanMac);

UanMac::UanMac ()
  : m_txModeIndex (0)
{

}

TypeId UanMac::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::UanMac")
    .SetParent<Object> ()
    .SetGroupName ("Uan")
  ;
  return tid;
}

void
UanMac::SetTxModeIndex (uint32_t txModeIndex)
{
  m_txModeIndex = txModeIndex;
}

uint32_t
UanMac::GetTxModeIndex ()
{
  return m_txModeIndex;
}

Address
UanMac::GetAddress (void)
{
  return m_address;
}

void
UanMac::SetAddress (Mac8Address addr)
{
  m_address=addr;
}

Address
UanMac::GetBroadcast (void) const
{
  Mac8Address broadcast = Mac8Address (255);
  return broadcast;
}

} // namespace ns3
