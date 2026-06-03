/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 ResiliNets, ITTC, University of Kansas
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
 * Authors: Truc Anh N. Nguyen <annguyen@ittc.ku.edu>
 *          Keerthi Ganta <keerthig@ittc.ku.edu>
 *          Md. Moshfequr Rahman <moshfequr@ittc.ku.edu>
 *          Amir Modarresi <amodarresi@ittc.ku.edu>
 *
 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  http://wiki.ittc.ku.edu/resilinets
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 */

#include "tcp-scalable.h"
#include "tcp-socket-state.h"

#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TcpScalable");
NS_OBJECT_ENSURE_REGISTERED (TcpScalable);

TypeId
TcpScalable::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpScalable")
    .SetParent<TcpNewReno> ()
    .AddConstructor<TcpScalable> ()
    .SetGroupName ("Internet")
    .AddAttribute ("AIFactor", "Additive Increase Factor",
                   UintegerValue (50),
                   MakeUintegerAccessor (&TcpScalable::m_aiFactor),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("MDFactor", "Multiplicative Decrease Factor",
                   DoubleValue (0.125),
                   MakeDoubleAccessor (&TcpScalable::m_mdFactor),
                   MakeDoubleChecker<double> ())
  ;
  return tid;
}

TcpScalable::TcpScalable (void)
  : TcpNewReno (),
    m_ackCnt (0),
    m_aiFactor (50),
    m_mdFactor (0.125)
{
  NS_LOG_FUNCTION (this);
}

TcpScalable::TcpScalable (const TcpScalable& sock)
  : TcpNewReno (sock),
    m_ackCnt (sock.m_ackCnt),
    m_aiFactor (sock.m_aiFactor),
    m_mdFactor (sock.m_mdFactor)
{
  NS_LOG_FUNCTION (this);
}

TcpScalable::~TcpScalable (void)
{
  NS_LOG_FUNCTION (this);
}

Ptr<TcpCongestionOps>
TcpScalable::Fork (void)
{
  return CopyObject<TcpScalable> (this);
}

void
TcpScalable::CongestionAvoidance (Ptr<TcpSocketState> tcb,
                                  uint32_t segmentsAcked)
{
  NS_LOG_FUNCTION (this << tcb << segmentsAcked);

  uint32_t segCwnd = tcb->GetCwndInSegments ();
  NS_ASSERT (segCwnd >= 1);

  uint32_t oldCwnd = segCwnd;
  uint32_t w = std::min (segCwnd, m_aiFactor);

  if (m_ackCnt >= w)
    {
      m_ackCnt = 0;
      segCwnd++;
    }

  m_ackCnt += segmentsAcked;
  if (m_ackCnt >= w)
    {
      uint32_t delta = m_ackCnt / w;
      m_ackCnt = 0;
      segCwnd += delta;
    }

  if (segCwnd != oldCwnd)
    {
      tcb->m_cWnd = segCwnd * tcb->m_segmentSize;
      NS_LOG_INFO ("In CongAvoid, updated to cwnd " << tcb->m_cWnd <<
                   " ssthresh " << tcb->m_ssThresh);
    }
}

std::string
TcpScalable::GetName () const
{
  return "TcpScalable";
}

uint32_t
TcpScalable::GetSsThresh (Ptr<const TcpSocketState> tcb,
                          uint32_t bytesInFlight)
{
  NS_LOG_FUNCTION (this << tcb << bytesInFlight);

  uint32_t segCwnd = bytesInFlight / tcb->m_segmentSize;

  double b = 1.0 - m_mdFactor;
  uint32_t ssThresh = static_cast<uint32_t> (std::max (2.0, segCwnd * b));

  NS_LOG_DEBUG ("Calculated b(w) = " << b <<
                " resulting (in segment) ssThresh=" << ssThresh);

  return ssThresh * tcb->m_segmentSize;
}

} // namespace ns3
