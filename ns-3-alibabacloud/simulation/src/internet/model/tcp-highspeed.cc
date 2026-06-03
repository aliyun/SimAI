/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Natale Patriciello, <natale.patriciello@gmail.com>
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
 */

#include "tcp-highspeed.h"
#include "tcp-socket-state.h"

#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TcpHighSpeed");
NS_OBJECT_ENSURE_REGISTERED (TcpHighSpeed);

TypeId
TcpHighSpeed::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpHighSpeed")
    .SetParent<TcpNewReno> ()
    .AddConstructor<TcpHighSpeed> ()
    .SetGroupName ("Internet")
  ;
  return tid;
}

TcpHighSpeed::TcpHighSpeed (void)
  : TcpNewReno (),
    m_ackCnt (0)
{
  NS_LOG_FUNCTION (this);
}

TcpHighSpeed::TcpHighSpeed (const TcpHighSpeed& sock)
  : TcpNewReno (sock),
    m_ackCnt (sock.m_ackCnt)
{
  NS_LOG_FUNCTION (this);
}

TcpHighSpeed::~TcpHighSpeed (void)
{
  NS_LOG_FUNCTION (this);
}

Ptr<TcpCongestionOps>
TcpHighSpeed::Fork (void)
{
  return CopyObject<TcpHighSpeed> (this);
}

/**
 * \brief Congestion avoidance of TcpHighSpeed
 *
 * As implementation choice, we increment cWnd only by MSS, when the right
 * number of ACK has been received. At this point, the important question is:
 * what is the "right number of ACK" ?
 *
 * As you can recall from RFC, Highspeed works this way:
 *
 *               w = w + a(w)/w
 *
 * Let's start when a(w) is 1 (so it is classical NewReno). The formula then is
 * the classical text-book version for NewReno:
 *
 *               w = w + 1 / w
 *
 * So, for each segment acked, we increase the window by the quantity 1/w. Or,
 * instead of adding the 1/w quantity for each segment acked, we can track the
 * number of segments acked (m_ackCnt) and increment by 1 MSS when m_ackCnt
 * reaches w.
 *
 * When a(w) > 1, it means that each segment acked has a different "weight".
 * For instance, when it is equal to 2, we need to increase the window by the
 * quantity 2/w. But, this means that one segment acked is equivalent (from
 * the point of view of incrementing cWnd) to two segments acked in NewReno
 * (1/w + 1/w). That a coefficient is, in other word, the weight of each segment
 * acked. More weight, less ACK are necessary to increment cWnd, which is
 * exactly the Highspeed principle.
 *
 * \param tcb internal congestion state
 * \param segmentsAcked count of segments acked
 */
void
TcpHighSpeed::CongestionAvoidance (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
  NS_LOG_FUNCTION (this << tcb << segmentsAcked);

  uint32_t segCwnd = tcb->GetCwndInSegments ();
  uint32_t oldCwnd = segCwnd;

  if (segmentsAcked > 0)
    {
      uint32_t coeffA = TableLookupA (segCwnd);
      m_ackCnt += segmentsAcked * coeffA;
    }

  while (m_ackCnt >= segCwnd)
    {
      m_ackCnt -= segCwnd;
      segCwnd += 1;
    }

  if (segCwnd != oldCwnd)
    {
      tcb->m_cWnd = segCwnd * tcb->m_segmentSize;
      NS_LOG_INFO ("In CongAvoid, updated to cwnd " << tcb->m_cWnd <<
                   " ssthresh " << tcb->m_ssThresh);
    }
}

std::string
TcpHighSpeed::GetName () const
{
  return "TcpHighSpeed";
}

/**
 * \brief Get slow start threshold following HighSpeed principles
 *
 * \param tcb internal congestion state
 * \param bytesInFlight Bytes in flight
 *
 * \return the slow start threshold value
 */
uint32_t
TcpHighSpeed::GetSsThresh (Ptr<const TcpSocketState> tcb,
                           uint32_t bytesInFlight)
{
  NS_LOG_FUNCTION (this << tcb << bytesInFlight);

  uint32_t segCwnd = bytesInFlight / tcb->m_segmentSize;

  double b = 1.0 - TableLookupB (segCwnd);
  uint32_t ssThresh = static_cast<uint32_t> (std::max (2.0, segCwnd * b));

  NS_LOG_DEBUG ("Calculated b(w) = " << b <<
                " resulting (in segment) ssThresh=" << ssThresh);

  return ssThresh * tcb->m_segmentSize;
}

uint32_t
TcpHighSpeed::TableLookupA (uint32_t w)
{
  if (w <= 38)
    {
      return 1;
    }
  else if (w <= 118)
    {
      return 2;
    }
  else if (w <= 221)
    {
      return 3;
    }
  else if (w <= 347)
    {
      return 4;
    }
  else if (w <= 495)
    {
      return 5;
    }
  else if (w <= 663)
    {
      return 6;
    }
  else if (w <= 851)
    {
      return 7;
    }
  else if (w <= 1058)
    {
      return 8;
    }
  else if (w <= 1284)
    {
      return 9;
    }
  else if (w <= 1529)
    {
      return 10;
    }
  else if (w <= 1793)
    {
      return 11;
    }
  else if (w <= 2076)
    {
      return 12;
    }
  else if (w <= 2378)
    {
      return 13;
    }
  else if (w <= 2699)
    {
      return 14;
    }
  else if (w <= 3039)
    {
      return 15;
    }
  else if (w <= 3399)
    {
      return 16;
    }
  else if (w <= 3778)
    {
      return 17;
    }
  else if (w <= 4177)
    {
      return 18;
    }
  else if (w <= 4596)
    {
      return 19;
    }
  else if (w <= 5036)
    {
      return 20;
    }
  else if (w <= 5497)
    {
      return 21;
    }
  else if (w <= 5979)
    {
      return 22;
    }
  else if (w <= 6483)
    {
      return 23;
    }
  else if (w <= 7009)
    {
      return 24;
    }
  else if (w <= 7558)
    {
      return 25;
    }
  else if (w <= 8130)
    {
      return 26;
    }
  else if (w <= 8726)
    {
      return 27;
    }
  else if (w <= 9346)
    {
      return 28;
    }
  else if (w <= 9991)
    {
      return 29;
    }
  else if (w <= 10661)
    {
      return 30;
    }
  else if (w <= 11358)
    {
      return 31;
    }
  else if (w <= 12082)
    {
      return 32;
    }
  else if (w <= 12834)
    {
      return 33;
    }
  else if (w <= 13614)
    {
      return 34;
    }
  else if (w <= 14424)
    {
      return 35;
    }
  else if (w <= 15265)
    {
      return 36;
    }
  else if (w <= 16137)
    {
      return 37;
    }
  else if (w <= 17042)
    {
      return 38;
    }
  else if (w <= 17981)
    {
      return 39;
    }
  else if (w <= 18955)
    {
      return 40;
    }
  else if (w <= 19965)
    {
      return 41;
    }
  else if (w <= 21013)
    {
      return 42;
    }
  else if (w <= 22101)
    {
      return 43;
    }
  else if (w <= 23230)
    {
      return 44;
    }
  else if (w <= 24402)
    {
      return 45;
    }
  else if (w <= 25618)
    {
      return 46;
    }
  else if (w <= 26881)
    {
      return 47;
    }
  else if (w <= 28193)
    {
      return 48;
    }
  else if (w <= 29557)
    {
      return 49;
    }
  else if (w <= 30975)
    {
      return 50;
    }
  else if (w <= 32450)
    {
      return 51;
    }
  else if (w <= 33986)
    {
      return 52;
    }
  else if (w <= 35586)
    {
      return 53;
    }
  else if (w <= 37253)
    {
      return 54;
    }
  else if (w <= 38992)
    {
      return 55;
    }
  else if (w <= 40808)
    {
      return 56;
    }
  else if (w <= 42707)
    {
      return 57;
    }
  else if (w <= 44694)
    {
      return 58;
    }
  else if (w <= 46776)
    {
      return 59;
    }
  else if (w <= 48961)
    {
      return 60;
    }
  else if (w <= 51258)
    {
      return 61;
    }
  else if (w <= 53667)
    {
      return 62;
    }
  else if (w <= 56230)
    {
      return 63;
    }
  else if (w <= 58932)
    {
      return 64;
    }
  else if (w <= 61799)
    {
      return 65;
    }
  else if (w <= 64851)
    {
      return 66;
    }
  else if (w <= 68113)
    {
      return 67;
    }
  else if (w <= 71617)
    {
      return 68;
    }
  else if (w <= 75401)
    {
      return 69;
    }
  else if (w <= 79517)
    {
      return 70;
    }
  else if (w <= 84035)
    {
      return 71;
    }
  else if (w <= 89053)
    {
      return 72;
    }
  else if (w <= 94717)
    {
      return 73;
    }
  else
    {
      return 73;
    }
}

double
TcpHighSpeed::TableLookupB (uint32_t w)
{
  if (w <= 38)
    {
      return 0.50;
    }
  else if (w <= 118)
    {
      return 0.44;
    }
  else if (w <= 221)
    {
      return 0.41;
    }
  else if (w <= 347)
    {
      return 0.38;
    }
  else if (w <= 495)
    {
      return 0.37;
    }
  else if (w <= 663)
    {
      return 0.35;
    }
  else if (w <= 851)
    {
      return 0.34;
    }
  else if (w <= 1058)
    {
      return 0.33;
    }
  else if (w <= 1284)
    {
      return 0.32;
    }
  else if (w <= 1529)
    {
      return 0.31;
    }
  else if (w <= 1793)
    {
      return 0.30;
    }
  else if (w <= 2076)
    {
      return 0.29;
    }
  else if (w <= 2378)
    {
      return 0.28;
    }
  else if (w <= 2699)
    {
      return 0.28;
    }
  else if (w <= 3039)
    {
      return 0.27;
    }
  else if (w <= 3399)
    {
      return 0.27;
    }
  else if (w <= 3778)
    {
      return 0.26;
    }
  else if (w <= 4177)
    {
      return 0.26;
    }
  else if (w <= 4596)
    {
      return 0.25;
    }
  else if (w <= 5036)
    {
      return 0.25;
    }
  else if (w <= 5497)
    {
      return 0.24;
    }
  else if (w <= 5979)
    {
      return 0.24;
    }
  else if (w <= 6483)
    {
      return 0.23;
    }
  else if (w <= 7009)
    {
      return 0.23;
    }
  else if (w <= 7558)
    {
      return 0.22;
    }
  else if (w <= 8130)
    {
      return 0.22;
    }
  else if (w <= 8726)
    {
      return 0.22;
    }
  else if (w <= 9346)
    {
      return 0.21;
    }
  else if (w <= 9991)
    {
      return 0.21;
    }
  else if (w <= 10661)
    {
      return 0.21;
    }
  else if (w <= 11358)
    {
      return 0.20;
    }
  else if (w <= 12082)
    {
      return 0.20;
    }
  else if (w <= 12834)
    {
      return 0.20;
    }
  else if (w <= 13614)
    {
      return 0.19;
    }
  else if (w <= 14424)
    {
      return 0.19;
    }
  else if (w <= 15265)
    {
      return 0.19;
    }
  else if (w <= 16137)
    {
      return 0.19;
    }
  else if (w <= 17042)
    {
      return 0.18;
    }
  else if (w <= 17981)
    {
      return 0.18;
    }
  else if (w <= 18955)
    {
      return 0.18;
    }
  else if (w <= 19965)
    {
      return 0.17;
    }
  else if (w <= 21013)
    {
      return 0.17;
    }
  else if (w <= 22101)
    {
      return 0.17;
    }
  else if (w <= 23230)
    {
      return 0.17;
    }
  else if (w <= 24402)
    {
      return 0.16;
    }
  else if (w <= 25618)
    {
      return 0.16;
    }
  else if (w <= 26881)
    {
      return 0.16;
    }
  else if (w <= 28193)
    {
      return 0.16;
    }
  else if (w <= 29557)
    {
      return 0.15;
    }
  else if (w <= 30975)
    {
      return 0.15;
    }
  else if (w <= 32450)
    {
      return 0.15;
    }
  else if (w <= 33986)
    {
      return 0.15;
    }
  else if (w <= 35586)
    {
      return 0.14;
    }
  else if (w <= 37253)
    {
      return 0.14;
    }
  else if (w <= 38992)
    {
      return 0.14;
    }
  else if (w <= 40808)
    {
      return 0.14;
    }
  else if (w <= 42707)
    {
      return 0.13;
    }
  else if (w <= 44694)
    {
      return 0.13;
    }
  else if (w <= 46776)
    {
      return 0.13;
    }
  else if (w <= 48961)
    {
      return 0.13;
    }
  else if (w <= 51258)
    {
      return 0.13;
    }
  else if (w <= 53667)
    {
      return 0.12;
    }
  else if (w <= 56230)
    {
      return 0.12;
    }
  else if (w <= 58932)
    {
      return 0.12;
    }
  else if (w <= 61799)
    {
      return 0.12;
    }
  else if (w <= 64851)
    {
      return 0.11;
    }
  else if (w <= 68113)
    {
      return 0.11;
    }
  else if (w <= 71617)
    {
      return 0.11;
    }
  else if (w <= 75401)
    {
      return 0.10;
    }
  else if (w <= 79517)
    {
      return 0.10;
    }
  else if (w <= 84035)
    {
      return 0.10;
    }
  else if (w <= 89053)
    {
      return 0.10;
    }
  else if (w <= 94717)
    {
      return 0.09;
    }
  else
    {
      return 0.09;
    }
}

} // namespace ns3
