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
 * Author: Keerthi Ganta <keerthiganta@ku.edu>
 *         Truc Anh N. Nguyen <annguyen@ittc.ku.edu>
 *
 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  http://wiki.ittc.ku.edu/resilinets
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 */


#include "tcp-illinois.h"
#include "tcp-socket-state.h"

#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TcpIllinois");
NS_OBJECT_ENSURE_REGISTERED (TcpIllinois);

TypeId
TcpIllinois::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpIllinois")
    .SetParent<TcpNewReno> ()
    .AddConstructor<TcpIllinois> ()
    .SetGroupName ("Internet")
    .AddAttribute ("AlphaMin", "Minimum alpha threshold",
                   DoubleValue (0.3),
                   MakeDoubleAccessor (&TcpIllinois::m_alphaMin),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("AlphaMax", "Maximum alpha threshold",
                   DoubleValue (10.0),
                   MakeDoubleAccessor (&TcpIllinois::m_alphaMax),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("AlphaBase", "Alpha base threshold",
                   DoubleValue (1.0),
                   MakeDoubleAccessor (&TcpIllinois::m_alphaBase),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("BetaMin", "Minimum beta threshold",
                   DoubleValue (0.125),
                   MakeDoubleAccessor (&TcpIllinois::m_betaMin),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("BetaMax", "Maximum beta threshold",
                   DoubleValue (0.5),
                   MakeDoubleAccessor (&TcpIllinois::m_betaMax),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("BetaBase", "Beta base threshold",
                   DoubleValue (0.5),
                   MakeDoubleAccessor (&TcpIllinois::m_betaBase),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("WinThresh", "Window threshold",
                   UintegerValue (15),
                   MakeUintegerAccessor (&TcpIllinois::m_winThresh),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("Theta", "Theta threshold",
                   UintegerValue (5),
                   MakeUintegerAccessor (&TcpIllinois::m_theta),
                   MakeUintegerChecker<uint32_t> ())
  ;
  return tid;
}

TcpIllinois::TcpIllinois (void)
  : TcpNewReno (),
    m_sumRtt (Time (0)),
    m_cntRtt (0),
    m_baseRtt (Time::Max ()),
    m_maxRtt (Time::Min ()),
    m_endSeq (0),
    m_rttAbove (false),
    m_rttLow (0),
    m_alphaMin (0.3),
    m_alphaMax (10.0),
    m_alphaBase (1.0),
    m_alpha (m_alphaMax),
    m_betaMin (0.125),
    m_betaMax (0.5),
    m_betaBase (0.5),
    m_beta (m_betaBase),
    m_winThresh (15),
    m_theta (5),
    m_ackCnt (0)
{
  NS_LOG_FUNCTION (this);
}

TcpIllinois::TcpIllinois (const TcpIllinois& sock)
  : TcpNewReno (sock),
    m_sumRtt (sock.m_sumRtt),
    m_cntRtt (sock.m_cntRtt),
    m_baseRtt (sock.m_baseRtt),
    m_maxRtt (sock.m_maxRtt),
    m_endSeq (sock.m_endSeq),
    m_rttAbove (sock.m_rttAbove),
    m_rttLow (sock.m_rttLow),
    m_alphaMin (sock.m_alphaMin),
    m_alphaMax (sock.m_alphaMax),
    m_alphaBase (sock.m_alphaBase),
    m_alpha (sock.m_alpha),
    m_betaMin (sock.m_betaMin),
    m_betaMax (sock.m_betaMax),
    m_betaBase (sock.m_betaBase),
    m_beta (sock.m_beta),
    m_winThresh (sock.m_winThresh),
    m_theta (sock.m_theta),
    m_ackCnt (sock.m_ackCnt)
{
  NS_LOG_FUNCTION (this);
}

TcpIllinois::~TcpIllinois (void)
{
  NS_LOG_FUNCTION (this);
}

void
TcpIllinois::RecalcParam (uint32_t cWnd)
{
  NS_LOG_FUNCTION (this << cWnd);

  if (cWnd < m_winThresh)
    {
      NS_LOG_INFO ("cWnd < winThresh, set alpha & beta to base values");

      m_alpha = m_alphaBase;
      m_beta = m_betaBase;
    }
  else if (m_cntRtt > 0)
    {
      double dm = static_cast<double> (CalculateMaxDelay ().GetMilliSeconds ());
      double da = static_cast<double> (CalculateAvgDelay ().GetMilliSeconds ());

      NS_LOG_INFO ("Updated to dm = " << dm << " da = " << da);

      CalculateAlpha (da, dm);
      CalculateBeta (da, dm);
    }
}

void
TcpIllinois::CongestionStateSet (Ptr<TcpSocketState> tcb,
                                 const TcpSocketState::TcpCongState_t newState)
{
  NS_LOG_FUNCTION (this << tcb << newState);

  if (newState == TcpSocketState::CA_LOSS)
    {
      m_alpha = m_alphaBase;
      m_beta = m_betaBase;
      m_rttLow = 0;
      m_rttAbove = false;
      Reset (tcb->m_nextTxSequence);
    }
}

void
TcpIllinois::IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
  NS_LOG_FUNCTION (this << segmentsAcked);

  if (tcb->m_lastAckedSeq >= m_endSeq)
    {
      RecalcParam (tcb->m_cWnd);
      Reset (tcb->m_nextTxSequence);
    }

  if (tcb->m_cWnd < tcb->m_ssThresh)
    {
      TcpNewReno::SlowStart (tcb, segmentsAcked);
      NS_LOG_INFO ("In SlowStart, updated to cwnd " << tcb->m_cWnd <<
                   " ssthresh " << tcb->m_ssThresh);
    }
  else
    {
      uint32_t segCwnd = tcb->GetCwndInSegments ();
      uint32_t oldCwnd = segCwnd;

      if (segmentsAcked > 0)
        {
          m_ackCnt += segmentsAcked * m_alpha;
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
}

void
TcpIllinois::PktsAcked (Ptr<TcpSocketState> tcb, uint32_t packetsAcked,
                        const Time &rtt)
{
  NS_LOG_FUNCTION (this << tcb << packetsAcked << rtt);

  if (rtt.IsZero ())
    {
      return;
    }

  // Keep track of minimum RTT
  m_baseRtt = std::min (m_baseRtt, rtt);

  // Keep track of maximum RTT
  m_maxRtt = std::max (rtt, m_maxRtt);

  ++m_cntRtt;
  m_sumRtt += rtt;

  NS_LOG_INFO ("Updated baseRtt = " << m_baseRtt << " maxRtt = " << m_maxRtt <<
               " cntRtt = " << m_cntRtt << " sumRtt = " << m_sumRtt);
}

uint32_t
TcpIllinois::GetSsThresh (Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight)
{
  NS_LOG_FUNCTION (this << tcb << bytesInFlight);

  uint32_t segBytesInFlight = bytesInFlight / tcb->m_segmentSize;
  uint32_t ssThresh = static_cast<uint32_t> (std::max (2.0, (1.0 - m_beta) * segBytesInFlight));

  NS_LOG_DEBUG ("Calculated ssThresh (in segments) = " << ssThresh);

  return ssThresh * tcb->m_segmentSize;
}

void
TcpIllinois::CalculateAlpha (double da, double dm)
{
  NS_LOG_FUNCTION (this << da << dm);

  double d1 = dm / 100;

  if (da <= d1)
    {
      NS_LOG_INFO ("da <= d1");

      if (!m_rttAbove)
        { // In case we can't get out of this low delay zone, we use alphaMax
          m_alpha = m_alphaMax;
        }
      if (++m_rttLow >= m_theta)
        {
          /*
           * da needs to stay below d1 for theta times RTT amount of time
           * before we can increase alpha to alphaMax
           */
          NS_LOG_INFO ("da stays below d1 for theta times RTT amount of time, "
                       "increase alpha to alphaMax");

          m_rttLow = 0;
          m_rttAbove = false;
          m_alpha = m_alphaMax;
        }
    }
  else
    {
      NS_LOG_INFO ("da > d1");

      m_rttAbove = true;
      /*
       * alpha = k1 / (k2 + da), where
       * k1 = ((dm - d1) * alphaMin * alphaMax) / (alphaMax - alphaMin)
       * k2 = (((dm - d1) * alphaMin) / (alphaMax - alphaMin)) - d1
       */
      dm -= d1;
      da -= d1;
      m_alpha = (dm * m_alphaMax) / (dm + (da * (m_alphaMax - m_alphaMin)) / m_alphaMin);
    }

  NS_LOG_INFO ("Updated to alpha = " << m_alpha);
}

void
TcpIllinois::CalculateBeta (double da, double dm)
{
  NS_LOG_FUNCTION (this << da << dm);

  double d2, d3;

  d2 = dm / 10;
  d3 = (8 * dm) / 10;

  if (da <= d2)
    {
      NS_LOG_INFO ("da <= d2");

      m_beta = m_betaMin;
    }

  else if (da > d2 && da < d3)
    {
      NS_LOG_INFO ("da > d2 && da < d3");

      /*
       * beta = k3 + k4 * da, where
       * k3 = (betaMin * d3 - betaMax * d2) / (d3 - d2)
       * k4 = (betaMax - betaMin) / (d3 - d2)
       */
      m_beta = (m_betaMin * d3 - m_betaMax * d2 + (m_betaMax - m_betaMin) * da) / (d3 - d2);

    }

  else if (da >= d3 || d3 <= d2)
    {
      NS_LOG_INFO ("da >= d3 || d3 <= d2");

      m_beta = m_betaMax;
    }

  NS_LOG_INFO ("Updated to beta = " << m_beta);
}

Time
TcpIllinois::CalculateAvgDelay () const
{
  NS_LOG_FUNCTION (this);

  return (m_sumRtt / m_cntRtt - m_baseRtt);
}

Time
TcpIllinois::CalculateMaxDelay () const
{
  NS_LOG_FUNCTION (this);

  return (m_maxRtt - m_baseRtt);
}

void
TcpIllinois::Reset (const SequenceNumber32 &nextTxSequence)
{
  NS_LOG_FUNCTION (this << nextTxSequence);

  m_endSeq = nextTxSequence;
  m_cntRtt = 0;
  m_sumRtt = Time (0);
}

Ptr<TcpCongestionOps>
TcpIllinois::Fork (void)
{
  NS_LOG_FUNCTION (this);

  return CopyObject<TcpIllinois> (this);
}

std::string
TcpIllinois::GetName () const
{
  NS_LOG_FUNCTION (this);

  return "TcpIllinois";
}

} // namespace ns3
