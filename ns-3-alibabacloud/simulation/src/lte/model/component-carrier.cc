/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 Danilo Abrignani
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
 * Author: Danilo Abrignani <danilo.abrignani@unibo.it>
 */

#include "component-carrier.h"
#include <ns3/uinteger.h>
#include <ns3/boolean.h>
#include <ns3/log.h>
#include <ns3/abort.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ComponentCarrier");

NS_OBJECT_ENSURE_REGISTERED ( ComponentCarrier);

TypeId ComponentCarrier::GetTypeId (void)
{
  static TypeId
    tid =
    TypeId ("ns3::ComponentCarrier")
    .SetParent<Object> ()
    .AddConstructor<ComponentCarrier> ()
    .AddAttribute ("UlBandwidth",
                   "Uplink Transmission Bandwidth Configuration in number of Resource Blocks",
                   UintegerValue (25),
                   MakeUintegerAccessor (&ComponentCarrier::SetUlBandwidth,
                                         &ComponentCarrier::GetUlBandwidth),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("DlBandwidth",
                   "Downlink Transmission Bandwidth Configuration in number of Resource Blocks",
                   UintegerValue (25),
                   MakeUintegerAccessor (&ComponentCarrier::SetDlBandwidth,
                                         &ComponentCarrier::GetDlBandwidth),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("DlEarfcn",
                   "Downlink E-UTRA Absolute Radio Frequency Channel Number (EARFCN) "
                   "as per 3GPP 36.101 Section 5.7.3. ",
                   UintegerValue (100),
                   MakeUintegerAccessor (&ComponentCarrier::SetDlEarfcn,
                                         &ComponentCarrier::GetDlEarfcn),
                   MakeUintegerChecker<uint32_t> (0, 262143))
    .AddAttribute ("UlEarfcn",
                   "Uplink E-UTRA Absolute Radio Frequency Channel Number (EARFCN) "
                   "as per 3GPP 36.101 Section 5.7.3. ",
                   UintegerValue (18100),
                   MakeUintegerAccessor (&ComponentCarrier::SetUlEarfcn,
                                         &ComponentCarrier::GetUlEarfcn),
                   MakeUintegerChecker<uint32_t> (18000, 262143))
    .AddAttribute ("CsgId",
                   "The Closed Subscriber Group (CSG) identity that this eNodeB belongs to",
                   UintegerValue (0),
                   MakeUintegerAccessor (&ComponentCarrier::SetCsgId,
                                         &ComponentCarrier::GetCsgId),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("CsgIndication",
                   "If true, only UEs which are members of the CSG (i.e. same CSG ID) "
                   "can gain access to the eNodeB, therefore enforcing closed access mode. "
                   "Otherwise, the eNodeB operates as a non-CSG cell and implements open access mode.",
                   BooleanValue (false),
                   MakeBooleanAccessor (&ComponentCarrier::SetCsgIndication,
                                        &ComponentCarrier::GetCsgIndication),
                   MakeBooleanChecker ())
    .AddAttribute ("PrimaryCarrier",
                   "If true, this Carrier Component will be the Primary Carrier Component (PCC) "
                   "Only one PCC per eNodeB is (currently) allowed",
                   BooleanValue (false),
                   MakeBooleanAccessor (&ComponentCarrier::SetAsPrimary,
                                        &ComponentCarrier::IsPrimary),
                   MakeBooleanChecker ())
  ;
  return tid;
}
ComponentCarrier::ComponentCarrier () : Object ()
{
  NS_LOG_FUNCTION (this);
}

ComponentCarrier::~ComponentCarrier (void)
{
  NS_LOG_FUNCTION (this);
}

void
ComponentCarrier::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  Object::DoDispose ();
}

uint16_t
ComponentCarrier::GetUlBandwidth () const
{
  return m_ulBandwidth;
}

void
ComponentCarrier::SetUlBandwidth (uint16_t bw)
{
  NS_LOG_FUNCTION (this << bw);
  switch (bw)
    {
    case 6:
    case 15:
    case 25:
    case 50:
    case 75:
    case 100:
      m_ulBandwidth = bw;
      break;

    default:
      NS_FATAL_ERROR ("Invalid bandwidth value " << bw);
      break;
    }
}

uint16_t
ComponentCarrier::GetDlBandwidth () const
{
  return m_dlBandwidth;
}

void
ComponentCarrier::SetDlBandwidth (uint16_t bw)
{
  NS_LOG_FUNCTION (this << bw);
  switch (bw)
    {
    case 6:
    case 15:
    case 25:
    case 50:
    case 75:
    case 100:
      m_dlBandwidth = bw;
      break;

    default:
      NS_FATAL_ERROR ("Invalid bandwidth value " << bw);
      break;
    }
}

uint32_t
ComponentCarrier::GetDlEarfcn () const
{
  return m_dlEarfcn;
}

void
ComponentCarrier::SetDlEarfcn (uint32_t earfcn)
{
  NS_LOG_FUNCTION (this << earfcn);
  m_dlEarfcn = earfcn;
}

uint32_t
ComponentCarrier::GetUlEarfcn () const
{
  return m_ulEarfcn;
}

void
ComponentCarrier::SetUlEarfcn (uint32_t earfcn)
{
  NS_LOG_FUNCTION (this << earfcn);
  m_ulEarfcn = earfcn;
}

uint32_t
ComponentCarrier::GetCsgId () const
{
  return m_csgId;
}

void
ComponentCarrier::SetCsgId (uint32_t csgId)
{
  NS_LOG_FUNCTION (this << csgId);
  m_csgId = csgId;
}

bool
ComponentCarrier::GetCsgIndication () const
{
  return m_csgIndication;
}

void
ComponentCarrier::SetCsgIndication (bool csgIndication)
{
  NS_LOG_FUNCTION (this << csgIndication);
  m_csgIndication = csgIndication;
}

bool
ComponentCarrier::IsPrimary () const
{
  return m_primaryCarrier;
}

void
ComponentCarrier::SetAsPrimary (bool primaryCarrier)
{
  NS_LOG_FUNCTION (this << primaryCarrier);
  m_primaryCarrier = primaryCarrier;
}

//====================================================

NS_OBJECT_ENSURE_REGISTERED (ComponentCarrierBaseStation);

TypeId ComponentCarrierBaseStation::GetTypeId (void)
{
  static TypeId
    tid =
    TypeId ("ns3::ComponentCarrierBaseStation")
    .SetParent<ComponentCarrier> ()
    .AddConstructor<ComponentCarrierBaseStation> ()
  ;
  return tid;
}

ComponentCarrierBaseStation::ComponentCarrierBaseStation () : ComponentCarrier ()
{
  NS_LOG_FUNCTION (this);
}

ComponentCarrierBaseStation::~ComponentCarrierBaseStation (void)
{
  NS_LOG_FUNCTION (this);
}

uint16_t
ComponentCarrierBaseStation::GetCellId ()
{
  return m_cellId;
}

void
ComponentCarrierBaseStation::SetCellId (uint16_t cellId)
{
  NS_LOG_FUNCTION (this << cellId);
  m_cellId = cellId;
}

} // namespace ns3


