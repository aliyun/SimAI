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

#include "tcp-prr-recovery.h"
#include "tcp-socket-state.h"

#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TcpPrrRecovery");
NS_OBJECT_ENSURE_REGISTERED (TcpPrrRecovery);

TypeId
TcpPrrRecovery::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpPrrRecovery")
    .SetParent<TcpClassicRecovery> ()
    .AddConstructor<TcpPrrRecovery> ()
    .SetGroupName ("Internet")
    .AddAttribute ("ReductionBound", "Type of Reduction Bound",
                   EnumValue (SSRB),
                   MakeEnumAccessor (&TcpPrrRecovery::m_reductionBoundMode),
                   MakeEnumChecker (CRB, "CRB",
                                    SSRB, "SSRB"))
  ;
  return tid;
}

TcpPrrRecovery::TcpPrrRecovery (void)
  : TcpClassicRecovery ()
{
  NS_LOG_FUNCTION (this);
}

TcpPrrRecovery::TcpPrrRecovery (const TcpPrrRecovery& recovery)
  : TcpClassicRecovery (recovery),
    m_prrDelivered (recovery.m_prrDelivered),
    m_prrOut (recovery.m_prrOut),
    m_recoveryFlightSize (recovery.m_recoveryFlightSize),
    m_reductionBoundMode (recovery.m_reductionBoundMode)
{
  NS_LOG_FUNCTION (this);
}

TcpPrrRecovery::~TcpPrrRecovery (void)
{
  NS_LOG_FUNCTION (this);
}

void
TcpPrrRecovery::EnterRecovery (Ptr<TcpSocketState> tcb, 
                               [[maybe_unused]] uint32_t dupAckCount,
                               uint32_t unAckDataCount, uint32_t deliveredBytes)
{
  NS_LOG_FUNCTION (this << tcb << dupAckCount << unAckDataCount);

  m_prrOut = 0;
  m_prrDelivered = 0;
  m_recoveryFlightSize = unAckDataCount;

  DoRecovery (tcb, deliveredBytes);
}

void
TcpPrrRecovery::DoRecovery (Ptr<TcpSocketState> tcb, uint32_t deliveredBytes)
{
  NS_LOG_FUNCTION (this << tcb << deliveredBytes);
  m_prrDelivered += deliveredBytes;

  int sendCount;
  if (tcb->m_bytesInFlight > tcb->m_ssThresh)
    {
      sendCount = std::ceil (m_prrDelivered * tcb->m_ssThresh * 1.0 / m_recoveryFlightSize) - m_prrOut;
    }
  else
    {
      int limit = static_cast<int> (tcb->m_ssThresh - tcb->m_bytesInFlight);
      if (m_reductionBoundMode == CRB)
        {
          limit = m_prrDelivered - m_prrOut;
        }
      else if (m_reductionBoundMode == SSRB)
        {
          if (tcb->m_isRetransDataAcked)
            {
              limit = std::max (m_prrDelivered - m_prrOut, deliveredBytes) + tcb->m_segmentSize;
            }
          else
            {
              limit = deliveredBytes;
            }
        }
      sendCount = std::min (limit, static_cast<int> (tcb->m_ssThresh - tcb->m_bytesInFlight));
    }

  /* Force a fast retransmit upon entering fast recovery */
  sendCount = std::max (sendCount, static_cast<int> (m_prrOut > 0 ? 0 : tcb->m_segmentSize));
  tcb->m_cWnd = tcb->m_bytesInFlight + sendCount;
  tcb->m_cWndInfl = tcb->m_cWnd;
}

void
TcpPrrRecovery::ExitRecovery (Ptr<TcpSocketState> tcb)
{
  NS_LOG_FUNCTION (this << tcb);
  tcb->m_cWndInfl = tcb->m_cWnd;
}

void
TcpPrrRecovery::UpdateBytesSent (uint32_t bytesSent)
{
  NS_LOG_FUNCTION (this << bytesSent);
  m_prrOut += bytesSent;
}

Ptr<TcpRecoveryOps>
TcpPrrRecovery::Fork (void)
{
  return CopyObject<TcpPrrRecovery> (this);
}

std::string
TcpPrrRecovery::GetName () const
{
  return "PrrRecovery";
}

} // namespace ns3
