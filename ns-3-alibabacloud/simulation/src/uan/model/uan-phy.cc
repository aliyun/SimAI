/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 * Author: Leonard Tracy <lentracy@gmail.com>
 */

#include "uan-phy.h"

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (UanPhyCalcSinr);

TypeId UanPhyCalcSinr::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::UanPhyCalcSinr")
    .SetParent<Object> ()
    .SetGroupName ("Uan")
  ;
  return tid;
}

void
UanPhyCalcSinr::Clear ()
{
}

void
UanPhyCalcSinr::DoDispose ()
{
  Clear ();
  Object::DoDispose ();
}

NS_OBJECT_ENSURE_REGISTERED (UanPhyPer);

TypeId UanPhyPer::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::UanPhyPer")
    .SetParent<Object> ()
    .SetGroupName ("Uan")
  ;
  return tid;
}

void
UanPhyPer::Clear ()
{
}

void
UanPhyPer::DoDispose ()
{
  Clear ();
  Object::DoDispose ();
}

NS_OBJECT_ENSURE_REGISTERED (UanPhy);

TypeId UanPhy::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::UanPhy")
    .SetParent<Object> ()
    .SetGroupName ("Uan")
    .AddTraceSource ("PhyTxBegin",
                     "Trace source indicating a packet has "
                     "begun transmitting over the channel medium.",
                     MakeTraceSourceAccessor (&UanPhy::m_phyTxBeginTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("PhyTxEnd",
                     "Trace source indicating a packet has "
                     "been completely transmitted over the channel.",
                   MakeTraceSourceAccessor (&UanPhy::m_phyTxEndTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("PhyTxDrop",
                     "Trace source indicating a packet has "
                     "been dropped by the device during transmission.",
                     MakeTraceSourceAccessor (&UanPhy::m_phyTxDropTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("PhyRxBegin",
                     "Trace source indicating a packet has "
                     "begun being received from the channel medium by the device.",
                     MakeTraceSourceAccessor (&UanPhy::m_phyRxBeginTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("PhyRxEnd",
                     "Trace source indicating a packet has "
                     "been completely received from the channel medium by the device.",
                     MakeTraceSourceAccessor (&UanPhy::m_phyRxEndTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("PhyRxDrop",
                     "Trace source indicating a packet has "
                     "been dropped by the device during reception.",
                     MakeTraceSourceAccessor (&UanPhy::m_phyRxDropTrace),
                     "ns3::Packet::TracedCallback")
  ;
  return tid;
}


void
UanPhy::NotifyTxBegin (Ptr<const Packet> packet)
{
  m_phyTxBeginTrace (packet);
}

void
UanPhy::NotifyTxEnd (Ptr<const Packet> packet)
{
  m_phyTxEndTrace (packet);
}

void
UanPhy::NotifyTxDrop (Ptr<const Packet> packet)
{
  m_phyTxDropTrace (packet);
}

void
UanPhy::NotifyRxBegin (Ptr<const Packet> packet)
{
  m_phyRxBeginTrace (packet);
}

void
UanPhy::NotifyRxEnd (Ptr<const Packet> packet)
{
  m_phyRxEndTrace (packet);
}

void
UanPhy::NotifyRxDrop (Ptr<const Packet> packet)
{
  m_phyRxDropTrace (packet);
}

} // namespace ns3
