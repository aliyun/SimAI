/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 NITK Surathkal
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
 * Author: Viyom Mittal <viyommittal@gmail.com>
 *         Vivek Jain <jain.vivek.anand@gmail.com>
 *         Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 *
 */
#include "tcp-recovery-ops.h"
#include "tcp-socket-state.h"

#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TcpRecoveryOps");

NS_OBJECT_ENSURE_REGISTERED (TcpRecoveryOps);

TypeId
TcpRecoveryOps::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpRecoveryOps")
    .SetParent<Object> ()
    .SetGroupName ("Internet")
  ;
  return tid;
}

TcpRecoveryOps::TcpRecoveryOps () : Object ()
{
  NS_LOG_FUNCTION (this);
}

TcpRecoveryOps::TcpRecoveryOps (const TcpRecoveryOps &other) : Object (other)
{
  NS_LOG_FUNCTION (this);
}

TcpRecoveryOps::~TcpRecoveryOps ()
{
  NS_LOG_FUNCTION (this);
}

void
TcpRecoveryOps::UpdateBytesSent (uint32_t bytesSent)
{
  NS_LOG_FUNCTION (this << bytesSent);
}

// Classic recovery

NS_OBJECT_ENSURE_REGISTERED (TcpClassicRecovery);

TypeId
TcpClassicRecovery::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpClassicRecovery")
    .SetParent<TcpRecoveryOps> ()
    .SetGroupName ("Internet")
    .AddConstructor<TcpClassicRecovery> ()
  ;
  return tid;
}

TcpClassicRecovery::TcpClassicRecovery (void) : TcpRecoveryOps ()
{
  NS_LOG_FUNCTION (this);
}

TcpClassicRecovery::TcpClassicRecovery (const TcpClassicRecovery& sock)
  : TcpRecoveryOps (sock)
{
  NS_LOG_FUNCTION (this);
}

TcpClassicRecovery::~TcpClassicRecovery (void)
{
  NS_LOG_FUNCTION (this);
}

void
TcpClassicRecovery::EnterRecovery (Ptr<TcpSocketState> tcb, uint32_t dupAckCount,
                                   [[maybe_unused]] uint32_t unAckDataCount, 
                                   [[maybe_unused]] uint32_t deliveredBytes)
{
  NS_LOG_FUNCTION (this << tcb << dupAckCount << unAckDataCount);
  tcb->m_cWnd = tcb->m_ssThresh;
  tcb->m_cWndInfl = tcb->m_ssThresh + (dupAckCount * tcb->m_segmentSize);
}

void
TcpClassicRecovery::DoRecovery (Ptr<TcpSocketState> tcb, [[maybe_unused]] uint32_t deliveredBytes)
{
  NS_LOG_FUNCTION (this << tcb << deliveredBytes);
  tcb->m_cWndInfl += tcb->m_segmentSize;
}

void
TcpClassicRecovery::ExitRecovery (Ptr<TcpSocketState> tcb)
{
  NS_LOG_FUNCTION (this << tcb);
  // Follow NewReno procedures to exit FR if SACK is disabled
  // (RFC2582 sec.3 bullet #5 paragraph 2, option 2)
  // In this implementation, actual m_cWnd value is reset to ssThresh
  // immediately before calling ExitRecovery(), so we just need to
  // reset the inflated cWnd trace variable
  tcb->m_cWndInfl = tcb->m_ssThresh.Get ();
}

std::string
TcpClassicRecovery::GetName () const
{
  return "TcpClassicRecovery";
}

Ptr<TcpRecoveryOps>
TcpClassicRecovery::Fork ()
{
  return CopyObject<TcpClassicRecovery> (this);
}

} // namespace ns3

