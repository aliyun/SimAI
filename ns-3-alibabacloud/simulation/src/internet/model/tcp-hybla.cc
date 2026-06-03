/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Natale Patriciello <natale.patriciello@gmail.com>
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

#include "tcp-hybla.h"
#include "tcp-socket-state.h"

#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TcpHybla");
NS_OBJECT_ENSURE_REGISTERED (TcpHybla);

TypeId
TcpHybla::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpHybla")
    .SetParent<TcpNewReno> ()
    .AddConstructor<TcpHybla> ()
    .SetGroupName ("Internet")
    .AddAttribute ("RRTT", "Reference RTT",
                   TimeValue (MilliSeconds (50)),
                   MakeTimeAccessor (&TcpHybla::m_rRtt),
                   MakeTimeChecker ())
    .AddTraceSource ("Rho",
                     "Rho parameter of Hybla",
                     MakeTraceSourceAccessor (&TcpHybla::m_rho),
                     "ns3::TracedValueCallback::Double")
  ;
  return tid;
}

TcpHybla::TcpHybla ()
  : TcpNewReno (),
  m_rho (1.0),
  m_cWndCnt (0)
{
  NS_LOG_FUNCTION (this);
}

TcpHybla::TcpHybla (const TcpHybla &sock)
  : TcpNewReno (sock),
  m_rho (sock.m_rho),
  m_cWndCnt (sock.m_cWndCnt)
{
  NS_LOG_FUNCTION (this);
}

TcpHybla::~TcpHybla ()
{
  NS_LOG_FUNCTION (this);
}

void
TcpHybla::RecalcParam (const Ptr<TcpSocketState> &tcb)
{
  NS_LOG_FUNCTION (this);

  m_rho = std::max ((double) tcb->m_minRtt.GetMilliSeconds () / m_rRtt.GetMilliSeconds (), 1.0);

  NS_ASSERT (m_rho > 0.0);
  NS_LOG_DEBUG ("Calculated rho=" << m_rho);
}

void
TcpHybla::PktsAcked (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked,
                     const Time &rtt)
{
  NS_LOG_FUNCTION (this << tcb << segmentsAcked << rtt);

  if (rtt == tcb->m_minRtt)
    {
      RecalcParam (tcb);
      NS_LOG_DEBUG ("min rtt seen: " << rtt);
    }
}

uint32_t
TcpHybla::SlowStart (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
  NS_LOG_FUNCTION (this << tcb << segmentsAcked);

  NS_ASSERT (tcb->m_cWnd <= tcb->m_ssThresh);

  if (segmentsAcked >= 1)
    {
      /*
       * slow start
       * INC = 2^RHO - 1
       */

      double increment = std::pow (2, m_rho) - 1.0;
      uint32_t incr = static_cast<uint32_t> (increment * tcb->m_segmentSize);
      NS_LOG_INFO ("Slow start: inc=" << increment);

      tcb->m_cWnd = std::min (tcb->m_cWnd + incr, tcb->m_ssThresh);

      NS_LOG_INFO ("In SlowStart, updated to cwnd " << tcb->m_cWnd <<
                   " ssthresh " << tcb->m_ssThresh <<
                   " with an increment of " << increment * tcb->m_segmentSize);

      return segmentsAcked - 1;
    }

  return 0;
}

void
TcpHybla::CongestionAvoidance (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
  NS_LOG_FUNCTION (this << tcb << segmentsAcked);

  uint32_t segCwnd;
  double increment;

  while (segmentsAcked > 0)
    {
      /*
       * congestion avoidance
       * INC = RHO^2 / W
       */
      segCwnd = tcb->GetCwndInSegments ();
      increment = std::pow (m_rho, 2) / static_cast<double> (segCwnd);

      m_cWndCnt += increment;
      segmentsAcked -= 1;
    }

  if (m_cWndCnt >= 1.0)
    {
      // double to int truncates every time.
      uint32_t inc = static_cast<uint32_t> (m_cWndCnt);
      m_cWndCnt -= inc;

      NS_ASSERT (m_cWndCnt >= 0.0);

      /* This leaves space for a tcp pacing implementation; it would be easy
         to setup a limit on the maximum increment of the cWnd per ACK received.
         The remaining increment is leaved for the next ACK. */

      tcb->m_cWnd += inc * tcb->m_segmentSize;


      NS_LOG_INFO ("In CongAvoid, updated to cwnd " << tcb->m_cWnd <<
                   " ssthresh " << tcb->m_ssThresh <<
                   " with an increment of " << inc * tcb->m_segmentSize);
    }
}

Ptr<TcpCongestionOps>
TcpHybla::Fork (void)
{
  return CopyObject<TcpHybla> (this);
}

std::string
TcpHybla::GetName () const
{
  return "TcpHybla";
}


} // namespace ns3
